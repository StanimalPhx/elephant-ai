"""Year-in-review flow: generate annual summary with statistics and narrative."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import TYPE_CHECKING, Any

from elephant.llm.prompts import year_in_review

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.llm.backend import LLMBackend
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)


class YearInReviewFlow:
    """Generates and sends an annual year-in-review summary."""

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

    async def run(self, year: int | None = None) -> bool:
        """Generate and send the year-in-review. Returns True if sent.

        If year is None, reviews the year that just ended (last year if Jan,
        otherwise current year when called on Dec 31).
        """
        today = date.today()
        if year is None:
            # When triggered Dec 31, review current year; Jan 1+, review last year
            year = today.year if today.month == 12 else today.year - 1

        date_from = date(year, 1, 1)
        date_to = date(year, 12, 31)

        memories = self._store.list_memories(
            date_from=date_from, date_to=date_to, limit=None,
        )
        total_memories = len(memories)

        if total_memories == 0:
            logger.info("No memories for %d, skipping year-in-review", year)
            return False

        # Statistics
        people_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        for m in memories:
            type_counter[m.type] += 1
            for person in m.people:
                people_counter[person] += 1

        unique_people = len(people_counter)
        top_people = people_counter.most_common(10)
        memories_by_type = dict(type_counter.most_common())
        milestones_count = type_counter.get("milestone", 0)

        # Top memories by nostalgia score
        sorted_memories = sorted(memories, key=lambda m: m.nostalgia_score, reverse=True)
        top_memories: list[dict[str, Any]] = [
            {
                "date": str(m.date),
                "title": m.title,
                "description": m.description,
                "people": ", ".join(m.people),
                "type": m.type,
            }
            for m in sorted_memories[:5]
        ]

        people = self._store.read_all_people()
        prefs = self._store.read_preferences()

        messages = year_in_review(
            year=year,
            total_memories=total_memories,
            unique_people=unique_people,
            memories_by_type=memories_by_type,
            top_memories=top_memories,
            top_people=top_people,
            milestones_count=milestones_count,
            people=people,
            prefs=prefs,
        )
        response = await self._llm.chat(messages, model=self._model)
        review_text = (response.content or "").strip()

        results = await self._messaging.broadcast_text(review_text)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send year-in-review: %s", errors or "no approved chats")
            return False

        self._store.increment_metric("year_reviews_sent")
        logger.info("Year-in-review sent for %d (%d memories)", year, total_memories)
        return True
