"""Determine message intent from metadata and timing."""

from __future__ import annotations

import logging
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING

from elephant.llm.prompts import classify_intent

if TYPE_CHECKING:
    from elephant.data.models import DigestState, PendingQuestionsFile
    from elephant.llm.backend import LLMBackend
    from elephant.messaging.base import IncomingMessage

logger = logging.getLogger(__name__)

# Window after digest send during which replies are treated as feedback
DIGEST_FEEDBACK_WINDOW = timedelta(minutes=30)


class Intent(Enum):
    NEW_MEMORY = "new_memory"
    DIGEST_FEEDBACK = "digest_feedback"
    ANSWER_TO_QUESTION = "answer_to_question"
    CONTEXT_UPDATE = "context_update"



async def resolve_intent(
    message: IncomingMessage,
    digest_state: DigestState,
    pending_questions: PendingQuestionsFile,
    llm: LLMBackend | None = None,
    model: str = "gpt-4.1-mini",
) -> Intent:
    """Resolve the intent of an incoming message.

    Resolution order:
    1. reply_to_id matches digest message → DIGEST_FEEDBACK
    2. reply_to_id matches an asked question → ANSWER_TO_QUESTION
    3. Within 30 min of digest send → DIGEST_FEEDBACK
    4. LLM classifies as context update → CONTEXT_UPDATE
    5. Default → NEW_MEMORY
    """
    # 1. Reply to digest
    if (
        message.reply_to_id
        and digest_state.last_digest_message_id
        and message.reply_to_id == digest_state.last_digest_message_id
    ):
        return Intent.DIGEST_FEEDBACK

    # 2. Reply to a pending question
    if message.reply_to_id:
        for q in pending_questions.questions:
            if q.status == "asked" and q.message_id and message.reply_to_id == q.message_id:
                return Intent.ANSWER_TO_QUESTION

    # 3. Within digest feedback window
    if digest_state.last_digest_sent_at:
        elapsed = message.timestamp - digest_state.last_digest_sent_at
        if elapsed < DIGEST_FEEDBACK_WINDOW:
            return Intent.DIGEST_FEEDBACK

    # 4. LLM classification (if available)
    if llm is not None:
        try:
            has_recent_digest = (
                digest_state.last_digest_sent_at is not None
                and (message.timestamp - digest_state.last_digest_sent_at) < timedelta(hours=2)
            )
            messages = classify_intent(message.text, has_recent_digest)
            response = await llm.chat(messages, model=model, temperature=0.1)
            label = (response.content or "").strip().lower()
            if label == "context_update":
                return Intent.CONTEXT_UPDATE
            if label == "digest_feedback":
                return Intent.DIGEST_FEEDBACK
            if label == "answer_to_question" and message.reply_to_id:
                return Intent.ANSWER_TO_QUESTION
        except Exception:
            logger.warning("LLM intent classification failed, defaulting to NEW_MEMORY")

    # 5. Default
    return Intent.NEW_MEMORY
