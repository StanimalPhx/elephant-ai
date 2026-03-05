"""Tests for churn detection & intervention."""

from __future__ import annotations

from datetime import date, timedelta

from elephant.brain.engagement import (
    ChurnSignals,
    compute_churn_signals,
    format_churn_for_checkin,
    format_churn_for_digest,
    format_churn_for_monthly,
    update_churn_state_after_feedback,
)
from elephant.data.models import (
    ChurnStateFile,
    DailyMetrics,
    Memory,
    PendingQuestion,
)

TODAY = date(2026, 3, 4)


def _memory(days_ago: int, people: list[str] | None = None, weekday: bool = True) -> Memory:
    d = TODAY - timedelta(days=days_ago)
    # If weekday requested but date is weekend, shift to previous Friday
    if weekday and d.weekday() >= 5:
        d = d - timedelta(days=d.weekday() - 4)
    return Memory(
        id=f"{d.strftime('%Y%m%d')}_test",
        date=d,
        title="Test",
        type="daily",
        description="Test memory",
        people=people or ["Alice"],
        source="Telegram",
    )


def _metrics(days_ago: int, digests: int = 1, replies: int = 0) -> DailyMetrics:
    return DailyMetrics(
        date=TODAY - timedelta(days=days_ago),
        digests_sent=digests,
        digest_replies=replies,
    )


def _pending_q(status: str = "pending") -> PendingQuestion:
    from datetime import UTC, datetime

    return PendingQuestion(
        id="q_test",
        type="memory_enrichment",
        subject="test",
        status=status,
        created_at=datetime.now(UTC),
    )


_DEFAULT_STATE = ChurnStateFile()
_KNOWN_NAMES: set[str] = {"Alice", "Bob", "Carol"}


class TestComputeChurnSignals:
    def test_quiet_week_when_no_recent_memories(self) -> None:
        # All memories older than 7 days
        memories = [_memory(10), _memory(15)]
        signals = compute_churn_signals(TODAY, memories, [], [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.quiet_week is True

    def test_quiet_week_false_with_recent_memory(self) -> None:
        memories = [_memory(3)]
        signals = compute_churn_signals(TODAY, memories, [], [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.quiet_week is False

    def test_no_digest_replies_when_digests_sent_but_no_replies(self) -> None:
        metrics = [_metrics(3, digests=1, replies=0), _metrics(7, digests=1, replies=0)]
        signals = compute_churn_signals(TODAY, [], metrics, [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.no_digest_replies is True

    def test_no_digest_replies_false_when_reply_exists(self) -> None:
        metrics = [_metrics(3, digests=1, replies=1)]
        signals = compute_churn_signals(TODAY, [], metrics, [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.no_digest_replies is False

    def test_no_digest_replies_false_when_no_digests_sent(self) -> None:
        metrics = [_metrics(3, digests=0, replies=0)]
        signals = compute_churn_signals(TODAY, [], metrics, [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.no_digest_replies is False

    def test_question_pile_up(self) -> None:
        questions = [_pending_q("pending") for _ in range(6)]
        signals = compute_churn_signals(TODAY, [], [], questions, _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.question_pile_up is True

    def test_question_pile_up_false_under_threshold(self) -> None:
        questions = [_pending_q("pending") for _ in range(5)]
        signals = compute_churn_signals(TODAY, [], [], questions, _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.question_pile_up is False

    def test_negative_feedback_streak(self) -> None:
        state = ChurnStateFile(
            consecutive_negative_sentiments=3,
            digest_paused_until=TODAY + timedelta(days=2),
        )
        signals = compute_churn_signals(TODAY, [], [], [], _KNOWN_NAMES, state)
        assert signals.negative_feedback_streak is True

    def test_negative_feedback_streak_false_when_expired(self) -> None:
        state = ChurnStateFile(
            consecutive_negative_sentiments=3,
            digest_paused_until=TODAY - timedelta(days=1),
        )
        signals = compute_churn_signals(TODAY, [], [], [], _KNOWN_NAMES, state)
        assert signals.negative_feedback_streak is False

    def test_no_new_people_30d(self) -> None:
        # All memories mention only known people — no new faces
        memories = [_memory(5, people=["Alice"]), _memory(25, people=["Bob"])]
        signals = compute_churn_signals(TODAY, memories, [], [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.no_new_people_30d is True

    def test_no_new_people_false_with_new_person(self) -> None:
        # "Dave" is NOT in _KNOWN_NAMES → new person detected
        memories = [
            _memory(5, people=["Alice", "Dave"]),
            _memory(25, people=["Alice"]),
        ]
        signals = compute_churn_signals(TODAY, memories, [], [], _KNOWN_NAMES, _DEFAULT_STATE)
        assert signals.no_new_people_30d is False

    def test_all_clear(self) -> None:
        """No signals when everything is healthy."""
        # "Dave" is unknown → new person exists → no_new_people_30d is False
        memories = [_memory(2, people=["Alice", "Dave"]), _memory(25, people=["Alice"])]
        metrics = [_metrics(3, digests=1, replies=1)]
        questions = [_pending_q("answered")]
        signals = compute_churn_signals(
            TODAY, memories, metrics, questions, _KNOWN_NAMES, _DEFAULT_STATE,
        )
        assert signals.quiet_week is False
        assert signals.no_digest_replies is False
        assert signals.question_pile_up is False
        assert signals.negative_feedback_streak is False
        assert signals.no_new_people_30d is False


class TestFormatChurnForDigest:
    def test_returns_none_when_no_signals(self) -> None:
        assert format_churn_for_digest(ChurnSignals()) is None

    def test_returns_message_for_quiet_week(self) -> None:
        signals = ChurnSignals(quiet_week=True)
        result = format_churn_for_digest(signals)
        assert result is not None
        assert "quiet week" in result.lower()

    def test_negative_streak_returns_none(self) -> None:
        """negative_feedback_streak is handled by digest pause, not prompt text."""
        signals = ChurnSignals(negative_feedback_streak=True)
        assert format_churn_for_digest(signals) is None


class TestFormatChurnForCheckin:
    def test_returns_none_when_no_signals(self) -> None:
        assert format_churn_for_checkin(ChurnSignals()) is None

    def test_returns_message_for_no_replies(self) -> None:
        signals = ChurnSignals(no_digest_replies=True)
        result = format_churn_for_checkin(signals)
        assert result is not None
        assert "feedback" in result.lower() or "digest" in result.lower()


class TestFormatChurnForMonthly:
    def test_returns_message_when_no_new_people(self) -> None:
        signals = ChurnSignals(no_new_people_30d=True)
        result = format_churn_for_monthly(signals, people_count=5)
        assert result is not None
        assert "new people" in result.lower()

    def test_returns_none_when_has_new_people(self) -> None:
        signals = ChurnSignals(no_new_people_30d=False)
        assert format_churn_for_monthly(signals, people_count=5) is None

    def test_returns_none_when_no_people(self) -> None:
        signals = ChurnSignals(no_new_people_30d=True)
        assert format_churn_for_monthly(signals, people_count=0) is None


class TestUpdateChurnStateAfterFeedback:
    def test_negative_increments_count(self) -> None:
        state = ChurnStateFile(consecutive_negative_sentiments=1)
        new = update_churn_state_after_feedback(state, "negative", TODAY)
        assert new.consecutive_negative_sentiments == 2
        assert new.digest_paused_until is None

    def test_negative_streak_triggers_pause(self) -> None:
        state = ChurnStateFile(consecutive_negative_sentiments=2)
        new = update_churn_state_after_feedback(state, "negative", TODAY)
        assert new.consecutive_negative_sentiments == 3
        assert new.digest_paused_until == TODAY + timedelta(days=3)

    def test_positive_resets_streak(self) -> None:
        state = ChurnStateFile(consecutive_negative_sentiments=2)
        new = update_churn_state_after_feedback(state, "positive", TODAY)
        assert new.consecutive_negative_sentiments == 0
        assert new.last_negative_streak_reset == TODAY

    def test_neutral_resets_streak(self) -> None:
        state = ChurnStateFile(consecutive_negative_sentiments=1)
        new = update_churn_state_after_feedback(state, "neutral", TODAY)
        assert new.consecutive_negative_sentiments == 0
