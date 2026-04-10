"""Entry point: load config, init data, start health server, wait for signal."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from elephant.brain.question_manager import QuestionManager
from elephant.config import load_config
from elephant.data.store import DataStore
from elephant.database import DatabaseInstance
from elephant.flows.anytime_log import AnytimeLogFlow
from elephant.flows.evening_checkin import EveningCheckinFlow
from elephant.flows.integrity_check import IntegrityCheckFlow
from elephant.flows.monthly_report import MonthlyReportFlow
from elephant.flows.morning_digest import MorningDigestFlow
from elephant.flows.weekly_recap import WeeklyRecapFlow
from elephant.flows.year_in_review import YearInReviewFlow
from elephant.git_ops import GitRepo
from elephant.health import create_app
from elephant.llm.client import LLMClient
from elephant.messaging.telegram import TelegramClient
from elephant.messaging.twilio import TwilioClient
# --- ADDED THIS NEW IMPORT ---
from elephant.messaging.nextcloud_talk import NextcloudTalkClient
# --- Also import MessagingClient if it's used for type hinting the messaging_client variable ---
from elephant.messaging.base import MessagingClient # Assuming base.py defines MessagingClient
from elephant.router import ChatRouter
from elephant.scheduler import Scheduler

logger = logging.getLogger("elephant")


class _TZFormatter(logging.Formatter):
    """Formatter that converts timestamps to a specific timezone."""

    def __init__(self, fmt: str, datefmt: str, tz: datetime.tzinfo) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._tz = tz

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._tz)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def _setup_logging(data_dir: str, timezone: str = "America/Chicago") -> None:
    """Configure file + console logging."""
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    tz = ZoneInfo(timezone)
    formatter = _TZFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=tz,
    )

    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


async def run(
    config_path: str | None = None,
    port: int = 8080,
) -> None:
    """Main application lifecycle."""
    # 1. Load config
    config = load_config(config_path)
    logger.info("Config loaded (LLM backend: %s)", config.llm.backend)

    # 1b. Telegram mode-dependent startup
    # --- This entire block needs to be refactored to handle different messaging providers ---
    # The existing code is specific to Telegram. You'll need to decide how to handle
    # provider-specific startup logic for Nextcloud Talk (e.g., no webhooks by default,
    # or if you implement webhooks, their specific setup).

    # The most robust approach would be to instantiate the messaging client first,
    # then handle provider-specific setup based on the client type or provider name.

    # --- Start of the refactored messaging client instantiation ---
    messaging_client: MessagingClient # Declare type for messaging_client

    if config.messaging.provider == "telegram":
        tg = config.messaging.telegram
        if tg.mode == "polling":
            # Delete any active webhook so getUpdates works
            from elephant.telegram_api import delete_webhook
            delete_webhook(tg.bot_token)
            logger.info("Telegram polling mode: webhook deleted")
        elif tg.webhook_url:
            from elephant.telegram_api import build_webhook_url, get_webhook_info
            expected = build_webhook_url(tg)
            info = get_webhook_info(tg.bot_token)
            registered = info.get("result", {}).get("url", "")
            if registered != expected:
                logger.error(
                    "Telegram webhook mismatch!\n  registered: %s\n  expected:   %s",
                    registered or "(not set)",
                    expected,
                )
                sys.exit(1)
            logger.info("Telegram webhook OK: %s", registered)
        messaging_client = TelegramClient(config.messaging.telegram) # Instantiate TelegramClient here
    elif config.messaging.provider == "twilio":
        # Assuming Twilio doesn't have specific startup modes like Telegram polling/webhook
        messaging_client = TwilioClient(config.messaging.twilio) # Instantiate TwilioClient
    elif config.messaging.provider == "nextcloud_talk": # --- ADD THIS NEW CONDITION ---
        messaging_client = NextcloudTalkClient(config.messaging.nextcloud_talk)
        # Add any Nextcloud Talk specific startup logic here if necessary.
        # For example, if you implement webhooks for Nextcloud Talk,
        # you might need to register them here, similar to Telegram's webhook check.
        # Currently, your NextcloudTalkClient implementation primarily sends messages,
        # so no complex startup logic is strictly needed beyond instantiation.
    else:
        raise ValueError(f"Unknown messaging provider: {config.messaging.provider}")
    # --- End of the refactored messaging client instantiation ---


    # 2. Setup logging (use first database's data_dir for logs)
    _setup_logging(config.databases[0].data_dir, timezone=config.schedule.timezone)

    # 3. Create HTTP session (always needed for messaging) and LLM client
    session = aiohttp.ClientSession() # This session will now be used by NextcloudTalkClient as well, if passed.

    if config.llm.backend == "agent_sdk":
        from elephant.llm.agent_sdk import AgentSDKClient
        llm: LLMClient | AgentSDKClient = AgentSDKClient(
            default_model=config.llm.morning_model,
        )
        logger.info("Using Agent SDK backend (model: %s)", config.llm.morning_model)
    else:
        llm = LLMClient(session, config.llm.base_url, config.llm.api_key)


    # 4. Build per-database object graphs
    router = ChatRouter()
    for db_cfg in config.databases:
        store = DataStore(db_cfg.data_dir)
        store.initialize()
        logger.info("Data store initialized at %s", db_cfg.data_dir)

        git = GitRepo(db_cfg.data_dir)
        git.initialize()
        git.auto_commit("schema", "Schema deployment")
        logger.info("Git repo ready at %s", db_cfg.data_dir)

        # Create messaging client
        messaging: TelegramClient | TwilioClient
        if config.messaging.provider == "telegram":
            messaging = TelegramClient(session, config.messaging.telegram, store)
        else:
            messaging = TwilioClient(session, config.messaging.twilio)

        anytime = AnytimeLogFlow(
            store=store,
            llm=llm,
            parsing_model=config.llm.parsing_model,
            messaging=messaging,
            git=git,
            history_limit=db_cfg.chat_history_limit,
            database_name=db_cfg.name,
            verify_traces=config.llm.verify_traces,
            guardrail_output=config.llm.guardrail_output,
        )
        morning = MorningDigestFlow(
            store=store,
            llm=llm,
            model=config.llm.morning_model,
            messaging=messaging,
            git=git,
        )
        evening = EveningCheckinFlow(
            store=store,
            llm=llm,
            model=config.llm.parsing_model,
            messaging=messaging,
        )
        question_mgr = QuestionManager(
            store=store,
            llm=llm,
            model=config.llm.parsing_model,
            messaging=messaging,
        )
        monthly = MonthlyReportFlow(store=store, messaging=messaging)
        weekly = WeeklyRecapFlow(
            store=store,
            llm=llm,
            model=config.llm.morning_model,
            messaging=messaging,
        )
        year_review = YearInReviewFlow(
            store=store,
            llm=llm,
            model=config.llm.morning_model,
            messaging=messaging,
        )
        integrity = IntegrityCheckFlow(
            store=store, git=git, llm=llm, model=config.llm.parsing_model,
            database_name=db_cfg.name,
        )

        db = DatabaseInstance(
            name=db_cfg.name,
            auth_secret=db_cfg.auth_secret,
            store=store,
            git=git,
            messaging=messaging,
            anytime=anytime,
            morning=morning,
            evening=evening,
            question_mgr=question_mgr,
            monthly_report=monthly,
            weekly_recap=weekly,
            year_in_review=year_review,
            integrity_check=integrity,
            schedule=db_cfg.schedule,
        )
        router.register_database(db)

    # 5. Create per-database schedulers
    schedulers: list[Scheduler] = []
    for db in router.get_all_databases():
        sched = Scheduler(db.schedule.timezone)
        await sched.start()
        sched.schedule_daily(
            db.schedule.morning_digest, db.morning.run,
            name=f"{db.name}:morning_digest",
        )
        sched.schedule_daily(
            db.schedule.evening_checkin, db.evening.run,
            name=f"{db.name}:evening_checkin",
        )
        sched.schedule_monthly(
            1, db.schedule.monthly_report, db.monthly_report.run,
            name=f"{db.name}:monthly_report",
        )
        sched.schedule_weekly(
            6, db.schedule.weekly_recap, db.weekly_recap.run,
            name=f"{db.name}:weekly_recap",
        )
        sched.schedule_periodic(
            900, db.question_mgr.process_pending,
            name=f"{db.name}:question_manager",
        )
        sched.schedule_yearly(
            12, 31, db.schedule.year_in_review, db.year_in_review.run,
            name=f"{db.name}:year_in_review",
        )
        sched.schedule_daily(
            db.schedule.integrity_check, db.integrity_check.run,
            name=f"{db.name}:integrity_check",
        )
        schedulers.append(sched)

    # 6. Build flows dict for API (namespaced by db)
    flows: dict[str, Callable[[], Awaitable[object]]] = {}
    for db in router.get_all_databases():
        flows[f"{db.name}:morning_digest"] = db.morning.run
        flows[f"{db.name}:evening_checkin"] = db.evening.run
        flows[f"{db.name}:monthly_report"] = db.monthly_report.run
        flows[f"{db.name}:weekly_recap"] = db.weekly_recap.run
        flows[f"{db.name}:question_manager"] = db.question_mgr.process_pending
        flows[f"{db.name}:year_in_review"] = db.year_in_review.run
        flows[f"{db.name}:integrity_check"] = db.integrity_check.run

    # 7. Start web server
    app = create_app(
        config=config,
        router=router,
        flows=flows,
    )
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("My Little Elephant is running (health on :%d)", port)

    if config.messaging.provider == "telegram" and tg.mode == "polling":
        from elephant.polling.telegram import TelegramPoller

        poller = TelegramPoller(session, tg, router)
        await poller.start()

    try:
        # 8. Wait for shutdown signal
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)

        await stop.wait()
    finally:
        # 9. Cleanup
        logger.info("Shutting down...")
        if poller is not None:
            await poller.stop()
        for sched in schedulers:
            await sched.stop()
        await runner.cleanup()
        await session.close()
        logger.info("Goodbye!")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="My Little Elephant")
    parser.add_argument(
        "-c", "--config",
        default=os.environ.get("CONFIG_PATH", "config.yaml"),
        help="Path to config.yaml (default: $CONFIG_PATH or config.yaml)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
        help="HTTP port (default: $PORT or 8080)",
    )
    args = parser.parse_args()
    asyncio.run(run(config_path=args.config, port=args.port))


if __name__ == "__main__":
    main()
