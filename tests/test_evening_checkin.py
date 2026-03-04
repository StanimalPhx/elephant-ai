"""Tests for evening check-in flow: full flow with all deps mocked."""

from datetime import date
from unittest.mock import AsyncMock, patch

from elephant.data.models import Memory
from elephant.data.store import DataStore
from elephant.flows.evening_checkin import EveningCheckinFlow
from elephant.llm.client import LLMResponse
from elephant.messaging.base import SendResult


class TestEveningCheckinFlow:
    async def test_sends_checkin(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Hey! How was your day? Anything worth remembering?",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_checkin_1")]
        )

        flow = EveningCheckinFlow(store, llm, "test-model", messaging)
        result = await flow.run()

        assert result is True
        messaging.broadcast_text.assert_called_once()
        call_text = messaging.broadcast_text.call_args[0][0]
        assert len(call_text) > 0

    async def test_handles_send_failure(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Check-in text", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=False, error="Network error")]
        )

        flow = EveningCheckinFlow(store, llm, "test-model", messaging)
        result = await flow.run()

        assert result is False

    async def test_includes_memory_count_in_prompt(self, data_dir):
        """When memories exist for today, the LLM prompt includes the count."""
        store = DataStore(data_dir)
        store.initialize()

        today = date(2026, 3, 4)
        for i in range(2):
            store.write_memory(
                Memory(
                    id=f"20260304_event_{i}",
                    date=today,
                    title=f"Event {i}",
                    type="daily",
                    description=f"Something happened {i}",
                    people=["Lily"],
                    source="Telegram",
                )
            )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Great day! Anything else to remember?",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_checkin_2")]
        )

        flow = EveningCheckinFlow(store, llm, "test-model", messaging)
        with patch("elephant.flows.evening_checkin.date") as mock_date:
            mock_date.today.return_value = today
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = await flow.run()

        assert result is True
        call_args = llm.chat.call_args
        system_msg = call_args[0][0][0]["content"]
        assert "2 memories logged so far" in system_msg
