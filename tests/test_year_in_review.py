"""Tests for year-in-review flow."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

from elephant.data.models import Memory
from elephant.data.store import DataStore
from elephant.flows.year_in_review import YearInReviewFlow
from elephant.llm.client import LLMResponse
from elephant.messaging.base import SendResult


class TestYearInReviewFlow:
    async def test_sends_review_with_memories(self, data_dir: str) -> None:
        store = DataStore(data_dir)
        store.initialize()

        for i in range(5):
            store.write_memory(
                Memory(
                    id=f"20250{i + 1}15_event_{i}",
                    date=date(2025, i + 1, 15),
                    title=f"Event {i}",
                    type="milestone" if i == 0 else "daily",
                    description=f"Something happened {i}",
                    people=["Lily", "Dad"] if i < 3 else ["Mom"],
                    source="Telegram",
                    nostalgia_score=1.0 + i * 0.3,
                )
            )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="What an amazing year for the family!",
                model="test",
                usage={},
            )
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_yr_1")]
        )

        flow = YearInReviewFlow(store, llm, "test-model", messaging)
        result = await flow.run(year=2025)

        assert result is True
        messaging.broadcast_text.assert_called_once()
        llm.chat.assert_called_once()

        # Check LLM prompt includes stats
        call_args = llm.chat.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "2025" in user_msg
        assert "5" in user_msg  # total memories

    async def test_skips_when_no_memories(self, data_dir: str) -> None:
        store = DataStore(data_dir)
        store.initialize()

        llm = AsyncMock()
        messaging = AsyncMock()

        flow = YearInReviewFlow(store, llm, "test-model", messaging)
        result = await flow.run(year=2025)

        assert result is False
        messaging.broadcast_text.assert_not_called()
        llm.chat.assert_not_called()

    async def test_send_failure_returns_false(self, data_dir: str) -> None:
        store = DataStore(data_dir)
        store.initialize()

        store.write_memory(
            Memory(
                id="20250115_test",
                date=date(2025, 1, 15),
                title="Test event",
                type="daily",
                description="Something",
                people=["Dad"],
                source="Telegram",
            )
        )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Review text", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=False, error="Network error")]
        )

        flow = YearInReviewFlow(store, llm, "test-model", messaging)
        result = await flow.run(year=2025)

        assert result is False

    async def test_metric_incremented_on_success(self, data_dir: str) -> None:
        store = DataStore(data_dir)
        store.initialize()

        store.write_memory(
            Memory(
                id="20250115_test",
                date=date(2025, 1, 15),
                title="Test event",
                type="daily",
                description="Something",
                people=["Dad"],
                source="Telegram",
            )
        )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Review", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_yr_2")]
        )

        flow = YearInReviewFlow(store, llm, "test-model", messaging)
        await flow.run(year=2025)

        metrics = store.read_metrics()
        today_metrics = [d for d in metrics.days if d.date == date.today()]
        assert len(today_metrics) == 1
        assert today_metrics[0].year_reviews_sent == 1

    async def test_defaults_to_last_year_in_january(self, data_dir: str) -> None:
        """When called in January with no year arg, reviews last year."""
        store = DataStore(data_dir)
        store.initialize()

        store.write_memory(
            Memory(
                id="20250615_summer",
                date=date(2025, 6, 15),
                title="Summer fun",
                type="daily",
                description="Pool day",
                people=["Lily"],
                source="Telegram",
            )
        )

        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=LLMResponse(content="Review", model="test", usage={})
        )

        messaging = AsyncMock()
        messaging.broadcast_text = AsyncMock(
            return_value=[SendResult(success=True, message_id="msg_yr_3")]
        )

        flow = YearInReviewFlow(store, llm, "test-model", messaging)
        with patch("elephant.flows.year_in_review.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = await flow.run()

        assert result is True
        # Verify prompt mentions 2025
        call_args = llm.chat.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "2025" in user_msg
