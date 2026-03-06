"""Generate follow-up questions for thin memories and process answers."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from elephant.data.models import Memory, PendingQuestion, Person, PreferencesFile
from elephant.data.store import DataStore
from elephant.llm.backend import LLMBackend
from elephant.llm.prompts import enrich_memory, generate_clarification

logger = logging.getLogger(__name__)

MAX_PENDING_QUESTIONS = 1
THIN_MEMORY_THRESHOLD = 50  # chars in description

CANONICAL_PERSON_FIELDS = {"birthday", "relationship", "display_name"}


def detect_person_conflicts(person: Person, updates: dict[str, Any]) -> list[dict[str, Any]]:
    """Return conflicts where existing non-None value differs from new value."""
    conflicts: list[dict[str, Any]] = []
    for field in CANONICAL_PERSON_FIELDS:
        if field not in updates:
            continue
        existing = getattr(person, field, None)
        new_val = updates[field]
        if existing is not None and str(existing) != str(new_val):
            conflicts.append({
                "field": field,
                "existing_value": str(existing),
                "new_value": str(new_val),
            })
    return conflicts


def is_thin_memory(memory: Memory) -> bool:
    """Check if a memory is too thin and needs enrichment."""
    short_desc = len(memory.description) < THIN_MEMORY_THRESHOLD
    few_people = len(memory.people) <= 1
    no_location = memory.location is None
    return short_desc and (few_people or no_location)




async def generate_question_for_memory(
    memory: Memory,
    llm: LLMBackend,
    model: str,
    people: list[Person],
    prefs: PreferencesFile,
    store: DataStore,
) -> PendingQuestion | None:
    """Generate a follow-up question for a thin memory, if under rate limit."""
    pq = store.read_pending_questions()
    pending_count = sum(1 for q in pq.questions if q.status in ("pending", "asked"))
    if pending_count >= MAX_PENDING_QUESTIONS:
        logger.debug("Rate limit: %d pending questions, skipping", pending_count)
        return None

    messages = generate_clarification(memory.title, memory.description, people, prefs)
    response = await llm.chat(messages, model=model, temperature=0.7)

    question = PendingQuestion(
        id=f"q_{uuid.uuid4().hex[:8]}",
        type="memory_enrichment",
        subject=memory.id,
        question=(response.content or "").strip(),
        status="pending",
        created_at=datetime.now(UTC),
    )

    pq.questions.append(question)
    store.write_pending_questions(pq)
    logger.info("Generated question %s for memory %s", question.id, memory.id)
    return question




async def process_answer(
    question_id: str,
    answer_text: str,
    llm: LLMBackend,
    model: str,
    store: DataStore,
) -> bool:
    """Process user's answer to a clarification question."""
    pq = store.read_pending_questions()
    question = None
    for q in pq.questions:
        if q.id == question_id:
            question = q
            break

    if question is None:
        logger.warning("Question %s not found", question_id)
        return False

    if question.type != "memory_enrichment":
        logger.info("Question %s is type %s, not memory_enrichment", question_id, question.type)
        question.status = "answered"
        question.answer = answer_text
        question.answered_at = datetime.now(UTC)
        store.write_pending_questions(pq)
        return True

    # Find the memory
    memory_id = question.subject
    if len(memory_id) >= 8 and memory_id[:8].isdigit():
        from datetime import date as _date

        y, m, d = int(memory_id[:4]), int(memory_id[4:6]), int(memory_id[6:8])
        slug = memory_id[9:] if len(memory_id) > 9 else memory_id
        path = store._memory_path(_date(y, m, d), slug)
        try:
            memory = store.read_memory(path)
        except FileNotFoundError:
            logger.warning("Memory %s not found for question %s", memory_id, question_id)
            return False

        # Enrich the memory description via LLM
        messages = enrich_memory(
            memory.title,
            memory.description,
            question.question or "",
            answer_text,
        )
        response = await llm.chat(messages, model=model, temperature=0.3)
        enriched_desc = (response.content or "").strip()

        # Update memory
        updated = memory.model_copy(update={"description": enriched_desc})
        store.write_memory(updated)
        logger.info("Enriched memory %s with answer to %s", memory_id, question_id)

    # Mark question as answered
    question.status = "answered"
    question.answer = answer_text
    question.answered_at = datetime.now(UTC)
    store.write_pending_questions(pq)
    return True
