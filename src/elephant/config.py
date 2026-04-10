"""Load config.yaml into frozen dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    morning_model: str = "claude-sonnet-4-6"
    parsing_model: str = "gpt-4.1-mini"
    verify_traces: bool = False
    guardrail_output: bool = True
    backend: str = "http"  # "http" or "agent_sdk"


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str = ""
    auth_token: str = ""
    whatsapp_from: str = ""
    whatsapp_to: str = ""


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = ""
    webhook_secret: str = ""
    webhook_url: str = ""
    mode: str = "polling"  # "polling" or "webhook"

# --- NEW DATACLASS FOR NEXTCLOUD TALK ---
@dataclass(frozen=True)
class NextcloudTalkConfig:
    nextcloud_url: str = ""
    username: str = ""
    app_password: str = ""
    default_chat_token: str = ""
    actor_display_name: str = "My Little Elephant"

@dataclass(frozen=True)
class MessagingConfig:
    provider: str = "twilio"
    twilio: TwilioConfig = TwilioConfig()
    telegram: TelegramConfig = TelegramConfig()


@dataclass(frozen=True)
class ScheduleConfig:
    morning_digest: str = "07:00"
    evening_checkin: str = "20:00"
    monthly_report: str = "09:00"
    weekly_recap: str = "10:00"
    year_in_review: str = "10:00"
    integrity_check: str = "03:00"
    timezone: str = "America/Chicago"


@dataclass(frozen=True)
class DatabaseConfig:
    name: str
    data_dir: str
    auth_secret: str
    schedule: ScheduleConfig = ScheduleConfig()
    chat_history_limit: int = 500


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    databases: tuple[DatabaseConfig, ...]
    schedule: ScheduleConfig = ScheduleConfig()
    messaging: MessagingConfig = MessagingConfig()


def _pick(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Extract only the keys that cls accepts from data."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def load_config(path: str | None = None) -> AppConfig:
    """Load config from YAML file.

    Resolution order: explicit path -> CONFIG_PATH env -> /app/config/config.yaml
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Config file must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    llm = LLMConfig(**_pick(LLMConfig, raw.get("llm", {})))
    schedule = ScheduleConfig(**_pick(ScheduleConfig, raw.get("schedule", {})))

    messaging_data: dict[str, Any] = raw.get("messaging", {})
    twilio = TwilioConfig(**_pick(TwilioConfig, messaging_data.get("twilio", {})))
    telegram = TelegramConfig(**_pick(TelegramConfig, messaging_data.get("telegram", {})))
    messaging = MessagingConfig(
        provider=messaging_data.get("provider", "twilio"),
        twilio=twilio,
        telegram=telegram,
    )

    # Parse databases section (required)
    raw_databases: dict[str, Any] = raw.get("databases", {})
    if not raw_databases:
        msg = "Config must have a 'databases' section with at least one database"
        raise ValueError(msg)

    dbs: list[DatabaseConfig] = []
    for db_name, db_data in raw_databases.items():
        db_schedule_data = db_data.get("schedule")
        if db_schedule_data:
            db_schedule = ScheduleConfig(**_pick(ScheduleConfig, db_schedule_data))
        else:
            db_schedule = schedule  # fall back to top-level
        dbs.append(
            DatabaseConfig(
                name=db_name,
                data_dir=db_data["data_dir"],
                auth_secret=db_data["auth_secret"],
                schedule=db_schedule,
                chat_history_limit=db_data.get("chat_history_limit", 500),
            )
        )

    return AppConfig(
        llm=llm,
        schedule=schedule,
        messaging=messaging,
        databases=tuple(dbs),
    )
