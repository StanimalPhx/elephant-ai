"""Tests for morning digest flow: full flow with all deps mocked."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import elephant.flows.morning_digest as mod
from elephant.data.models import Memory, PendingQuestion, PendingQuestionsFile, Person
from elephant.data.store import DataStore
from elephant.flows.morning_digest import (
    MorningDigestFlow,
    find_upcoming_birthdays,
)
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMResponse
from elephant.messaging.base import SendResult


class TestMorningDigestFlow:
    async def test_sends_digest_with_memories(self, store_with_memories):
        store = store_with_memories

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="On this day last year, Lily took her first steps!",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_digest_1")]
        )

        git = MagicMock(spec=GitRepo)
        git.auto_commit = MagicMock(return_value="abc123")

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)

        mock_now = datetime(2026, 2, 24, 7, 0, 0, tzinfo=UTC)
        with patch.object(mod, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now = MagicMock(return_value=mock_now)
            result = await flow.run()

        assert result is True
        messaging.broadcast_text.assert_called_once()
        git.auto_commit.assert_called_once()

        # Check digest state was updated
        state = store.read_digest_state()
        assert state.last_digest_message_id == "msg_digest_1"
        assert len(state.last_digest_memory_ids) > 0
        assert state.last_digest_text == "On this day last year, Lily took her first steps!"

    async def test_handles_send_failure(self, store_with_memories):
        store = store_with_memories

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Digest text", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=False, error="Network error")]
        )

        git = MagicMock(spec=GitRepo)

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)

        mock_now = datetime(2026, 2, 24, 7, 0, 0, tzinfo=UTC)
        with patch.object(mod, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now = MagicMock(return_value=mock_now)
            result = await flow.run()

        assert result is False
        git.auto_commit.assert_not_called()


class TestMorningQuestionFallback:
    """Tests for _send_question_fallback when there are no memories for today."""

    async def test_sends_existing_pending_question(self, data_dir):
        """When a pending question exists, wrap it and send."""
        store = DataStore(data_dir)
        store.initialize()

        pq = PendingQuestionsFile(
            questions=[
                PendingQuestion(
                    id="q_existing",
                    type="memory_enrichment",
                    subject="20260220_park",
                    question="Who was at the park that day?",
                    status="pending",
                    created_at=datetime.now(UTC),
                )
            ]
        )
        store.write_pending_questions(pq)

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Good morning! Quick question — who was at the park that day?",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_q_1")]
        )

        git = MagicMock(spec=GitRepo)

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)
        result = await flow.run()

        assert result is True
        messaging.broadcast_text.assert_called_once()

        # Question should be marked as asked with message_id
        updated = store.read_pending_questions()
        assert updated.questions[0].status == "asked"
        assert updated.questions[0].message_id == "msg_q_1"

        # Digest state should NOT be updated (not a digest)
        state = store.read_digest_state()
        assert state.last_digest_message_id is None

    async def test_finds_thin_memory_and_generates_question(self, data_dir):
        """When no pending questions but thin memories exist, generate one."""
        store = DataStore(data_dir)
        store.initialize()

        # Create a thin memory (short description, no location, few people)
        thin_memory = Memory(
            id="20260220_something",
            date=date(2026, 2, 20),
            title="Something happened",
            type="daily",
            description="Something",
            people=["Lily"],
            source="Telegram",
        )
        store.write_memory(thin_memory)

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Good morning! What happened that day?",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_q_2")]
        )

        git = MagicMock(spec=GitRepo)

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)

        # Patch to a date with no memories
        mock_now = datetime(2026, 3, 15, 7, 0, 0, tzinfo=UTC)
        with patch.object(mod, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now = MagicMock(return_value=mock_now)
            result = await flow.run()

        assert result is True

        # A new question should have been created and sent
        updated = store.read_pending_questions()
        assert len(updated.questions) == 1
        assert updated.questions[0].status == "asked"
        assert updated.questions[0].type == "memory_enrichment"
        assert updated.questions[0].message_id == "msg_q_2"

    async def test_context_gap_fallback(self, data_dir):
        """When no pending questions and no thin memories, generate a context_gap question."""
        store = DataStore(data_dir)
        store.initialize()

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Good morning! Tell me about your family traditions.",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_q_3")]
        )

        git = MagicMock(spec=GitRepo)

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)
        result = await flow.run()

        assert result is True

        # A context_gap question should have been created
        updated = store.read_pending_questions()
        assert len(updated.questions) == 1
        assert updated.questions[0].type == "context_gap"
        assert updated.questions[0].status == "asked"
        assert updated.questions[0].message_id == "msg_q_3"

    async def test_fallback_send_failure(self, data_dir):
        """When sending the question fails, return False."""
        store = DataStore(data_dir)
        store.initialize()

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Good morning!", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=False, error="Network error")]
        )

        git = MagicMock(spec=GitRepo)

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)
        result = await flow.run()

        assert result is False


class TestBirthdayReminders:
    """Tests for find_upcoming_birthdays."""

    def test_close_friend_within_window(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 3, 10), groups=["close-friends"],
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 1
        assert reminders[0].days_until == 13
        assert reminders[0].is_close_friend is True

    def test_close_friend_outside_window(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 6, 15), groups=["close-friends"],
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 0

    def test_regular_person_day_of_only(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 2, 25),
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 1
        assert reminders[0].days_until == 0

    def test_regular_person_not_today(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 3, 10),
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 0

    def test_feb_29_birthday_non_leap_year(self):
        people = [
            Person(
                person_id="leap", display_name="Leap", relationship=["friend"],
                birthday=date(2000, 2, 29), groups=["close-friends"],
            ),
        ]
        # 2025 is not a leap year, so Feb 29 -> Mar 1
        today = date(2025, 2, 28)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 1
        assert reminders[0].days_until == 1  # Mar 1

    def test_feb_29_birthday_leap_year(self):
        people = [
            Person(
                person_id="leap", display_name="Leap", relationship=["friend"],
                birthday=date(2000, 2, 29), groups=["close-friends"],
            ),
        ]
        today = date(2028, 2, 28)  # 2028 is a leap year
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 1
        assert reminders[0].days_until == 1  # Feb 29

    def test_no_birthday_set(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 0

    def test_multiple_sorted_by_days_until(self):
        people = [
            Person(
                person_id="far", display_name="Far", relationship=["friend"],
                birthday=date(1990, 3, 15), groups=["close-friends"],
            ),
            Person(
                person_id="near", display_name="Near", relationship=["friend"],
                birthday=date(1990, 3, 1), groups=["close-friends"],
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 2
        assert reminders[0].person.display_name == "Near"
        assert reminders[1].person.display_name == "Far"

    def test_close_friend_day_of(self):
        people = [
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 2, 25), groups=["close-friends"],
            ),
        ]
        today = date(2026, 2, 25)
        reminders = find_upcoming_birthdays(people, today)
        assert len(reminders) == 1
        assert reminders[0].days_until == 0
        assert reminders[0].is_close_friend is True


class TestBirthdayDigestIntegration:
    """Integration: digest includes birthday data when no memories but birthdays exist."""

    async def test_birthday_only_digest(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()

        # Add a close friend with a birthday coming up
        store.write_person(
            Person(
                person_id="theo", display_name="Theo", relationship=["friend"],
                birthday=date(1990, 3, 10), groups=["close-friends"],
            ),
        )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Good morning! Just a heads up — Theo's birthday is coming up!",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_bday_1")]
        )

        git = MagicMock(spec=GitRepo)
        git.auto_commit = MagicMock(return_value="abc123")

        flow = MorningDigestFlow(store, llm, "test-model", messaging, git)

        mock_now = datetime(2026, 2, 25, 7, 0, 0, tzinfo=UTC)
        with patch.object(mod, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now = MagicMock(return_value=mock_now)
            result = await flow.run()

        assert result is True
        messaging.broadcast_text.assert_called_once()

        # Verify LLM was called with birthday data in the prompt
        call_args = llm.chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "birthday" in system_msg.lower()
