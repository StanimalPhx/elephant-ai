"""Churn detection & intervention: detect declining engagement, format for prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from elephant.data.models import ChurnStateFile, DailyMetrics, Memory, PendingQuestion


@dataclass
class ChurnSignals:
    quiet_week: bool = False
    no_digest_replies: bool = False
    question_pile_up: bool = False
    no_new_people_30d: bool = False
    negative_feedback_streak: bool = False
    weekday_silent: bool = False


def compute_churn_signals(
    today: date,
    memories_30d: list[Memory],
    metrics_30d: list[DailyMetrics],
    pending_questions: list[PendingQuestion],
    known_people_names: set[str],
    churn_state: ChurnStateFile,
) -> ChurnSignals:
    """Compute churn signals from existing data. Pure function."""
    from datetime import timedelta

    signals = ChurnSignals()

    # quiet_week: no memories in last 7 days
    week_ago = today - timedelta(days=7)
    signals.quiet_week = not any(m.date >= week_ago for m in memories_30d)

    # no_digest_replies: digests sent but no replies in last 14 days
    fourteen_ago = today - timedelta(days=14)
    recent_metrics = [d for d in metrics_30d if d.date >= fourteen_ago]
    total_digests = sum(d.digests_sent for d in recent_metrics)
    total_replies = sum(d.digest_replies for d in recent_metrics)
    signals.no_digest_replies = total_digests > 0 and total_replies == 0

    # question_pile_up: more than 5 pending/asked questions
    active_questions = [
        q for q in pending_questions if q.status in ("pending", "asked")
    ]
    signals.question_pile_up = len(active_questions) > 5

    # no_new_people_30d: all people in recent memories are already known
    people_in_memories: set[str] = set()
    for m in memories_30d:
        for p in m.people:
            people_in_memories.add(p.lower())
    known_lower = {n.lower() for n in known_people_names}
    new_people = people_in_memories - known_lower
    signals.no_new_people_30d = len(known_people_names) > 0 and len(new_people) == 0

    # negative_feedback_streak: digest paused due to consecutive negative feedback
    signals.negative_feedback_streak = (
        churn_state.digest_paused_until is not None
        and churn_state.digest_paused_until >= today
    )

    # weekday_silent: had weekday memories historically, but none in last 14 days
    weekday_memories_30d = [m for m in memories_30d if m.date.weekday() < 5]
    total_memories_30d = len(memories_30d)
    if total_memories_30d > 0:
        weekday_ratio_30d = len(weekday_memories_30d) / total_memories_30d
    else:
        weekday_ratio_30d = 0.0

    recent_memories = [m for m in memories_30d if m.date >= fourteen_ago]
    recent_weekday = [m for m in recent_memories if m.date.weekday() < 5]
    signals.weekday_silent = weekday_ratio_30d > 0.3 and len(recent_weekday) == 0

    return signals


def _highest_priority_signal(signals: ChurnSignals) -> str | None:
    """Return the name of the highest-priority active signal, or None."""
    # Priority order: most actionable first
    if signals.negative_feedback_streak:
        return "negative_feedback_streak"
    if signals.quiet_week:
        return "quiet_week"
    if signals.no_digest_replies:
        return "no_digest_replies"
    if signals.question_pile_up:
        return "question_pile_up"
    if signals.weekday_silent:
        return "weekday_silent"
    if signals.no_new_people_30d:
        return "no_new_people_30d"
    return None


_DIGEST_MESSAGES: dict[str, str] = {
    "quiet_week": (
        "It's been a quiet week — no new memories logged. "
        "Even small moments are worth capturing!"
    ),
    "no_digest_replies": (
        "I've been sending digests but haven't heard back. "
        "Would you prefer a different style or schedule?"
    ),
    "question_pile_up": (
        "There are a few unanswered questions piling up. "
        "No rush, but they'd help me tell richer stories."
    ),
    "weekday_silent": (
        "Weekdays have been quiet lately. "
        "Even a quick note about your day helps build the story."
    ),
}

_CHECKIN_MESSAGES: dict[str, str] = {
    "quiet_week": (
        "It's been a quiet week — anything happen today worth jotting down?"
    ),
    "no_digest_replies": (
        "I'd love to hear what you thought of recent digests. "
        "Any feedback helps me improve."
    ),
    "question_pile_up": (
        "I have a few questions waiting whenever you have a moment."
    ),
    "weekday_silent": (
        "Weekdays have been quiet — even small weekday moments count!"
    ),
}


def format_churn_for_digest(signals: ChurnSignals) -> str | None:
    """Format the highest-priority churn signal for the morning digest prompt."""
    signal = _highest_priority_signal(signals)
    if signal is None or signal == "negative_feedback_streak":
        return None
    return _DIGEST_MESSAGES.get(signal)


def format_churn_for_checkin(signals: ChurnSignals) -> str | None:
    """Format the highest-priority churn signal for the evening check-in prompt."""
    signal = _highest_priority_signal(signals)
    if signal is None or signal == "negative_feedback_streak":
        return None
    return _CHECKIN_MESSAGES.get(signal)


def format_churn_for_monthly(signals: ChurnSignals, people_count: int = 0) -> str | None:
    """Format churn text for the monthly report. Only surfaces no_new_people_30d."""
    if signals.no_new_people_30d and people_count > 0:
        return (
            "No new people appeared in your memories this month. "
            "Have you met anyone new worth remembering?"
        )
    return None


_PAUSE_DAYS = 3
_STREAK_THRESHOLD = 3


def update_churn_state_after_feedback(
    state: ChurnStateFile,
    sentiment: str,
    today: date,
) -> ChurnStateFile:
    """Update churn state based on feedback sentiment. Returns new state."""
    from datetime import timedelta

    if sentiment == "negative":
        new_count = state.consecutive_negative_sentiments + 1
        paused_until = state.digest_paused_until
        if new_count >= _STREAK_THRESHOLD:
            paused_until = today + timedelta(days=_PAUSE_DAYS)
        return state.model_copy(update={
            "consecutive_negative_sentiments": new_count,
            "digest_paused_until": paused_until,
        })
    else:
        # positive or neutral resets the streak
        return state.model_copy(update={
            "consecutive_negative_sentiments": 0,
            "last_negative_streak_reset": today,
        })
