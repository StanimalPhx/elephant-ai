"""Pure asyncio daily/periodic scheduler using zoneinfo."""

import asyncio
import calendar
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ScheduleCallback = Callable[[], Awaitable[object]]


class Scheduler:
    """asyncio-based scheduler with daily and periodic tasks."""

    def __init__(self, timezone: str) -> None:
        self._tz = ZoneInfo(timezone)
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    def schedule_daily(
        self,
        time_str: str,
        callback: ScheduleCallback,
        name: str = "",
    ) -> None:
        """Schedule a callback to run daily at a specific time (HH:MM)."""
        hour, minute = (int(x) for x in time_str.split(":"))

        async def _loop() -> None:
            while self._running:
                now = datetime.now(self._tz)
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                label = name or f"daily@{time_str}"
                logger.info("Scheduler: %s next run in %.0fs", label, wait_seconds)
                await asyncio.sleep(wait_seconds)
                if not self._running:
                    break
                try:
                    await callback()
                except Exception:
                    logger.exception("Scheduler: %s failed", label)

        self._tasks.append(asyncio.ensure_future(_loop()))

    def schedule_monthly(
        self,
        day: int,
        time_str: str,
        callback: ScheduleCallback,
        name: str = "",
    ) -> None:
        """Schedule a callback to run on a specific day of the month at a given time."""
        hour, minute = (int(x) for x in time_str.split(":"))

        async def _loop() -> None:
            while self._running:
                now = datetime.now(self._tz)
                # Find the next occurrence of the target day
                year, month = now.year, now.month
                target = now.replace(
                    day=min(day, calendar.monthrange(year, month)[1]),
                    hour=hour, minute=minute, second=0, microsecond=0,
                )
                if target <= now:
                    # Move to next month
                    if month == 12:
                        year += 1
                        month = 1
                    else:
                        month += 1
                    target = target.replace(
                        year=year, month=month,
                        day=min(day, calendar.monthrange(year, month)[1]),
                    )
                wait_seconds = (target - now).total_seconds()
                label = name or f"monthly@{day}/{time_str}"
                logger.info("Scheduler: %s next run in %.0fs", label, wait_seconds)
                await asyncio.sleep(wait_seconds)
                if not self._running:
                    break
                try:
                    await callback()
                except Exception:
                    logger.exception("Scheduler: %s failed", label)

        self._tasks.append(asyncio.ensure_future(_loop()))

    def schedule_weekly(
        self,
        weekday: int,
        time_str: str,
        callback: ScheduleCallback,
        name: str = "",
    ) -> None:
        """Schedule a callback to run on a specific weekday at a given time.

        weekday: 0=Monday .. 6=Sunday (same as date.weekday()).
        """
        hour, minute = (int(x) for x in time_str.split(":"))

        async def _loop() -> None:
            while self._running:
                now = datetime.now(self._tz)
                # Days until target weekday
                days_ahead = (weekday - now.weekday()) % 7
                target = (now + timedelta(days=days_ahead)).replace(
                    hour=hour, minute=minute, second=0, microsecond=0,
                )
                if target <= now:
                    target += timedelta(weeks=1)
                wait_seconds = (target - now).total_seconds()
                label = name or f"weekly@{weekday}/{time_str}"
                logger.info("Scheduler: %s next run in %.0fs", label, wait_seconds)
                await asyncio.sleep(wait_seconds)
                if not self._running:
                    break
                try:
                    await callback()
                except Exception:
                    logger.exception("Scheduler: %s failed", label)

        self._tasks.append(asyncio.ensure_future(_loop()))

    def schedule_yearly(
        self,
        month: int,
        day: int,
        time_str: str,
        callback: ScheduleCallback,
        name: str = "",
    ) -> None:
        """Schedule a callback to run once a year on a specific month/day at a given time."""
        hour, minute = (int(x) for x in time_str.split(":"))

        async def _loop() -> None:
            while self._running:
                now = datetime.now(self._tz)
                year = now.year
                target_day = min(day, calendar.monthrange(year, month)[1])
                target = now.replace(
                    month=month, day=target_day,
                    hour=hour, minute=minute, second=0, microsecond=0,
                )
                if target <= now:
                    year += 1
                    target_day = min(day, calendar.monthrange(year, month)[1])
                    target = target.replace(year=year, month=month, day=target_day)
                wait_seconds = (target - now).total_seconds()
                label = name or f"yearly@{month}/{day}/{time_str}"
                logger.info("Scheduler: %s next run in %.0fs", label, wait_seconds)
                await asyncio.sleep(wait_seconds)
                if not self._running:
                    break
                try:
                    await callback()
                except Exception:
                    logger.exception("Scheduler: %s failed", label)

        self._tasks.append(asyncio.ensure_future(_loop()))

    def schedule_periodic(
        self,
        interval_seconds: float,
        callback: ScheduleCallback,
        name: str = "",
    ) -> None:
        """Schedule a callback to run every interval_seconds."""

        async def _loop() -> None:
            while self._running:
                await asyncio.sleep(interval_seconds)
                if not self._running:
                    break
                label = name or f"periodic@{interval_seconds}s"
                try:
                    await callback()
                except Exception:
                    logger.exception("Scheduler: %s failed", label)

        self._tasks.append(asyncio.ensure_future(_loop()))

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        logger.info("Scheduler started (tz=%s)", self._tz)

    async def stop(self) -> None:
        """Stop the scheduler and cancel all tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Scheduler stopped")
