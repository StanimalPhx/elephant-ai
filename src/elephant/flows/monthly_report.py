"""Monthly report flow: summarize preferences, metrics, and activity."""

from __future__ import annotations

import calendar
import logging
from collections import Counter
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)

_WEIGHT_LABELS: dict[str, str] = {
    "milestones": "Milestones",
    "mundane_daily": "Daily moments",
    "people_focus": "People-focused stories",
    "location_focus": "Location context",
}


def _weight_description(value: float) -> str:
    if value >= 1.5:
        return f"high priority ({value:.1f})"
    if value <= 0.7:
        return f"low ({value:.1f})"
    return f"standard ({value:.1f})"


def _weight_arrow(value: float) -> str:
    if value >= 1.5:
        return "\u2b06\ufe0f "
    if value <= 0.7:
        return "\u2b07\ufe0f "
    return ""


class MonthlyReportFlow:
    """Generates and sends a monthly report on the 1st of each month."""

    def __init__(
        self,
        store: DataStore,
        messaging: MessagingClient,
    ) -> None:
        self._store = store
        self._messaging = messaging

    async def run(self) -> bool:
        """Generate and send the monthly report. Returns True if sent."""
        today = date.today()

        # Determine last month
        if today.month == 1:
            last_month = 12
            last_year = today.year - 1
        else:
            last_month = today.month - 1
            last_year = today.year

        first_of_last_month = date(last_year, last_month, 1)
        last_day = calendar.monthrange(last_year, last_month)[1]
        last_of_last_month = date(last_year, last_month, last_day)

        month_name = calendar.month_name[last_month]

        # Count memories from last month
        memories = self._store.list_memories(
            date_from=first_of_last_month,
            date_to=last_of_last_month,
            limit=None,
        )
        memory_count = len(memories)

        # Count people mentioned
        people_counter: Counter[str] = Counter()
        for m in memories:
            for person in m.people:
                people_counter[person] += 1

        # Read preferences
        prefs = self._store.read_preferences()
        weights = prefs.nostalgia_weights

        # Read metrics for last month
        metrics = self._store.read_metrics()
        month_metrics = [
            d for d in metrics.days
            if first_of_last_month <= d.date <= last_of_last_month
        ]
        total_digests = sum(d.digests_sent for d in month_metrics)
        total_replies = sum(d.digest_replies for d in month_metrics)
        total_checkins = sum(d.checkins_sent for d in month_metrics)

        # Churn: check for no-new-people signal
        from elephant.brain.engagement import compute_churn_signals, format_churn_for_monthly

        people = self._store.read_all_people()
        churn_state = self._store.read_churn_state()
        known_names = {p.display_name for p in people}
        churn_signals = compute_churn_signals(
            today, memories, month_metrics, [], known_names, churn_state,
        )
        churn_text = format_churn_for_monthly(churn_signals, len(people))

        # Coverage gaps
        from elephant.brain.coverage import find_coverage_gaps, format_gaps_for_monthly

        all_memories = self._store.list_memories(limit=None)
        if all_memories:
            first_memory_date = min(m.date for m in all_memories)
            # Only look at months up to last month
            coverage_gaps = find_coverage_gaps(all_memories, first_memory_date, last_of_last_month)
            coverage_text = format_gaps_for_monthly(coverage_gaps)
        else:
            coverage_text = None

        # People completeness
        from elephant.brain.people_completeness import format_completeness_for_monthly

        completeness_text = format_completeness_for_monthly(people)

        # Build report
        report = self._format_report(
            month_name=month_name,
            year=last_year,
            memory_count=memory_count,
            people_counter=people_counter,
            weights=weights,
            tone=prefs.tone_preference,
            total_digests=total_digests,
            total_replies=total_replies,
            total_checkins=total_checkins,
            churn_text=churn_text,
            coverage_text=coverage_text,
            completeness_text=completeness_text,
        )

        results = await self._messaging.broadcast_text(report)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send monthly report: %s", errors or "no approved chats")
            return False

        logger.info("Monthly report sent for %s %d", month_name, last_year)
        return True

    @staticmethod
    def _format_report(
        month_name: str,
        year: int,
        memory_count: int,
        people_counter: Counter[str],
        weights: object,
        tone: object,
        total_digests: int,
        total_replies: int,
        total_checkins: int,
        churn_text: str | None = None,
        coverage_text: str | None = None,
        completeness_text: str | None = None,
    ) -> str:
        from elephant.data.models import NostalgiaWeights, TonePreference

        assert isinstance(weights, NostalgiaWeights)
        assert isinstance(tone, TonePreference)

        lines = [f"\U0001f4ca Monthly Report \u2014 {month_name} {year}", ""]

        lines.append(
            f"Memories: {memory_count} captured this month"
        )

        if people_counter:
            top_people = people_counter.most_common(5)
            people_str = ", ".join(f"{name} ({count})" for name, count in top_people)
            lines.append(f"People mentioned: {people_str}")

        if total_digests or total_replies or total_checkins:
            lines.append(
                f"Engagement: {total_digests} digests sent, "
                f"{total_replies} replies, {total_checkins} check-ins"
            )

        lines.append("")
        lines.append("What I've learned from your feedback:")
        for field, label in _WEIGHT_LABELS.items():
            value = getattr(weights, field, 1.0)
            arrow = _weight_arrow(value)
            desc = _weight_description(value)
            lines.append(f"\u2022 {label}: {arrow}{desc}")

        if churn_text:
            lines.append(f"\n{churn_text}")

        if coverage_text:
            lines.append(f"\n{coverage_text}")

        if completeness_text:
            lines.append(f"\n{completeness_text}")

        lines.append(f"\nYour style: {tone.style}, {tone.length}")
        lines.append("\nKeep sharing \u2014 your elephant never forgets! \U0001f418")

        return "\n".join(lines)
