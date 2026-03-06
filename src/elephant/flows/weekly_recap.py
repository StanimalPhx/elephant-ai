"""Weekly recap flow: summarize the past week's memories with an LLM narrative."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from elephant.llm.prompts import weekly_recap

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.llm.backend import LLMBackend
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)


class WeeklyRecapFlow:
    """Orchestrates the Sunday weekly memory recap."""

    def __init__(
        self,
        store: DataStore,
        llm: LLMBackend,
        model: str,
        messaging: MessagingClient,
    ) -> None:
        self._store = store
        self._llm = llm
        self._model = model
        self._messaging = messaging

    async def run(self) -> bool:
        """Generate and send the weekly recap. Returns True if sent."""
        today = date.today()
        week_ago = today - timedelta(days=7)

        memories = self._store.list_memories(
            date_from=week_ago, date_to=today, limit=None,
        )
        memory_count = len(memories)

        # Count unique people
        people_counter: Counter[str] = Counter()
        for m in memories:
            for person in m.people:
                people_counter[person] += 1
        unique_people = len(people_counter)

        # Build highlights (top 10 by nostalgia score)
        sorted_memories = sorted(memories, key=lambda m: m.nostalgia_score, reverse=True)
        highlights: list[dict[str, Any]] = [
            {
                "title": m.title,
                "description": m.description,
                "people": ", ".join(m.people),
                "type": m.type,
                "date": str(m.date),
            }
            for m in sorted_memories[:10]
        ]

        people = self._store.read_all_people()
        prefs = self._store.read_preferences()

        messages = weekly_recap(
            memory_count=memory_count,
            unique_people=unique_people,
            highlights=highlights,
            people=people,
            prefs=prefs,
        )
        response = await self._llm.chat(messages, model=self._model)
        recap_text = (response.content or "").strip()

        results = await self._messaging.broadcast_text(recap_text)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send weekly recap: %s", errors or "no approved chats")
            return False

        self._store.increment_metric("weekly_recaps_sent")
        logger.info("Weekly recap sent (%d memories, %d people)", memory_count, unique_people)
        return True
