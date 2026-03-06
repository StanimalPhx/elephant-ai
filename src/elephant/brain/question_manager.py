"""Periodic sender for pending questions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from elephant.llm.prompts import generate_question_text

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.llm.backend import LLMBackend
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)


class QuestionManager:
    """Manages sending pending questions to the user."""

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

    async def process_pending(self) -> int:
        """Check for pending questions and send them. Returns count sent."""
        pq = self._store.read_pending_questions()
        people = self._store.read_all_people()
        prefs = self._store.read_preferences()
        sent = 0

        for question in pq.questions:
            if question.status != "pending":
                continue

            # Generate question text if not already set
            if not question.question:
                try:
                    messages = generate_question_text(
                        question.type, question.subject, people, prefs,
                    )
                    response = await self._llm.chat(
                        messages, model=self._model, temperature=0.7
                    )
                    question.question = (response.content or "").strip()
                except Exception:
                    logger.exception("Failed to generate question text for %s", question.id)
                    continue

            # Send the question (broadcast to all approved chats)
            results = await self._messaging.broadcast_text(question.question)
            if any(r.success for r in results):
                question.status = "asked"
                first_ok = next(r for r in results if r.success)
                question.message_id = first_ok.message_id
                sent += 1
                logger.info("Sent question %s: %s", question.id, question.question[:50])
            else:
                errors = ", ".join(r.error or "unknown" for r in results)
                logger.warning("Failed to send question %s: %s", question.id, errors)

        if sent > 0:
            self._store.write_pending_questions(pq)

        return sent
