"""Evening check-in flow: generate prompt, send."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from elephant.flows.contact_nudges import (
    find_overdue_contacts,
    format_nudges_for_prompt,
    record_nudge,
)
from elephant.llm.prompts import evening_checkin

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.llm.client import LLMClient
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)


class EveningCheckinFlow:
    """Orchestrates the evening check-in message."""

    def __init__(
        self,
        store: DataStore,
        llm: LLMClient,
        model: str,
        messaging: MessagingClient,
    ) -> None:
        self._store = store
        self._llm = llm
        self._model = model
        self._messaging = messaging

    async def run(self) -> bool:
        """Send an evening check-in message. Returns True if sent."""
        today = date.today()
        todays_memories = self._store.list_memories(
            date_from=today, date_to=today, limit=None,
        )
        people = self._store.read_all_people()
        prefs = self._store.read_preferences()

        # Contact nudges
        names_with_target = [
            p.display_name for p in people if p.interaction_frequency_target is not None
        ]
        last_contacts = self._store.get_latest_memory_dates_for_people(names_with_target)
        nudge_state = self._store.read_nudge_state()
        nudges = find_overdue_contacts(
            people, last_contacts, nudge_state.records, today, max_nudges=1,
        )
        nudges_text = format_nudges_for_prompt(nudges) or None

        messages = evening_checkin(
            people, prefs, memory_count_today=len(todays_memories), nudges=nudges_text,
        )
        response = await self._llm.chat(messages, model=self._model)
        checkin_text = (response.content or "").strip()

        results = await self._messaging.broadcast_text(checkin_text)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send evening checkin: %s", errors or "no approved chats")
            return False

        # Record nudges sent
        if nudges:
            for nudge in nudges:
                record_nudge(nudge_state, nudge.person.person_id, today, "evening_checkin")
            self._store.write_nudge_state(nudge_state)
            self._store.increment_metric("nudges_sent", count=len(nudges))

        self._store.increment_metric("checkins_sent")
        logger.info("Evening checkin sent")
        return True
