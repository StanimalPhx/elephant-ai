"""Morning digest flow: query memories, score, LLM story, send, git commit."""

from __future__ import annotations

import calendar
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from elephant.brain.clarification import is_thin_memory
from elephant.data.models import PendingQuestion, Person
from elephant.flows.contact_nudges import (
    find_overdue_contacts,
    format_nudges_for_prompt,
    record_nudge,
)
from elephant.llm.prompts import generate_question_text, morning_digest, morning_question
from elephant.memory_scorer import score_memory

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.git_ops import GitRepo
    from elephant.llm.client import LLMClient
    from elephant.messaging.base import MessagingClient

logger = logging.getLogger(__name__)


@dataclass
class BirthdayReminder:
    person: Person
    birthday: date
    days_until: int
    is_close_friend: bool  # True when person is in "close-friends" group


def find_upcoming_birthdays(
    people: list[Person],
    today: date,
    close_friend_window_days: int = 21,
) -> list[BirthdayReminder]:
    """Find upcoming birthdays. Close friends: within window. Others: day-of only."""
    reminders: list[BirthdayReminder] = []
    is_leap = calendar.isleap(today.year)

    for person in people:
        if not person.birthday:
            continue

        bday_month = person.birthday.month
        bday_day = person.birthday.day

        # Feb 29 birthdays: treat as Mar 1 in non-leap years
        if bday_month == 2 and bday_day == 29 and not is_leap:
            bday_month = 3
            bday_day = 1

        try:
            this_year_bday = date(today.year, bday_month, bday_day)
        except ValueError:
            continue

        # If birthday already passed this year, check next year
        if this_year_bday < today:
            next_year = today.year + 1
            next_leap = calendar.isleap(next_year)
            if person.birthday.month == 2 and person.birthday.day == 29 and not next_leap:
                this_year_bday = date(next_year, 3, 1)
            else:
                try:
                    this_year_bday = date(next_year, bday_month, bday_day)
                except ValueError:
                    continue

        days_until = (this_year_bday - today).days

        if "close-friends" in person.groups and days_until <= close_friend_window_days:
            reminders.append(BirthdayReminder(
                person=person, birthday=this_year_bday,
                days_until=days_until, is_close_friend=True,
            ))
        elif days_until == 0:
            reminders.append(BirthdayReminder(
                person=person, birthday=this_year_bday,
                days_until=0, is_close_friend=False,
            ))

    reminders.sort(key=lambda r: r.days_until)
    return reminders


def _format_birthday_reminders(reminders: list[BirthdayReminder]) -> str:
    """Format birthday reminders into a string for the LLM prompt."""
    lines: list[str] = []
    for r in reminders:
        name = r.person.display_name
        if r.days_until == 0:
            lines.append(f"- TODAY is {name}'s birthday!")
        elif r.days_until <= 7:
            lines.append(
                f"- {name}'s birthday is in {r.days_until} days — finalize gift plans!"
            )
        elif r.days_until <= 14:
            lines.append(
                f"- {name}'s birthday is in {r.days_until} days — start thinking about a gift"
            )
        else:
            lines.append(f"- {name}'s birthday is coming up in {r.days_until} days")
    return "\n".join(lines)


class MorningDigestFlow:
    """Orchestrates the daily morning digest."""

    def __init__(
        self,
        store: DataStore,
        llm: LLMClient,
        model: str,
        messaging: MessagingClient,
        git: GitRepo,
    ) -> None:
        self._store = store
        self._llm = llm
        self._model = model
        self._messaging = messaging
        self._git = git

    async def run(self) -> bool:
        """Execute the morning digest flow. Returns True if digest was sent."""
        now = datetime.now(UTC)
        today = now.date()

        # 1. Query memories for today's month/day across all years
        memories = self._store.query_memories_by_month_day(now.month, now.day)

        # 2. Check for upcoming birthdays
        people = self._store.read_all_people()
        birthdays = find_upcoming_birthdays(people, today)

        if not memories and not birthdays:
            logger.info("No memories for %02d/%02d, trying question fallback", now.month, now.day)
            return await self._send_question_fallback()

        # 3. Score and rank memories
        prefs = self._store.read_preferences()
        top_memories: list[Any] = []
        if memories:
            scored = [(m, score_memory(m, prefs.nostalgia_weights)) for m in memories]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_memories = [m for m, _ in scored[:5]]

        # 4. Generate digest via LLM
        memories_data = [
            {
                "date": m.date.isoformat(),
                "title": m.resolved_value("title"),
                "description": m.resolved_value("description"),
                "people": m.resolved_value("people"),
                "location": m.resolved_value("location"),
                "attributes": m.attributes,
            }
            for m in top_memories
        ]
        birthday_data: list[dict[str, str | int | bool]] | None = None
        if birthdays:
            birthday_data = [
                {
                    "name": r.person.display_name,
                    "days_until": r.days_until,
                    "is_close_friend": r.is_close_friend,
                }
                for r in birthdays
            ]

        # 4a. Contact nudges
        names_with_target = [
            p.display_name for p in people if p.interaction_frequency_target is not None
        ]
        last_contacts = self._store.get_latest_memory_dates_for_people(names_with_target)
        nudge_state = self._store.read_nudge_state()
        nudges = find_overdue_contacts(
            people, last_contacts, nudge_state.records, today, max_nudges=2,
        )
        nudges_text = format_nudges_for_prompt(nudges) or None

        messages = morning_digest(
            memories_data,
            people,
            prefs,
            tone_style=prefs.tone_preference.style,
            tone_length=prefs.tone_preference.length,
            birthdays=birthday_data,
            nudges=nudges_text,
        )
        response = await self._llm.chat(messages, model=self._model)
        digest_text = (response.content or "").strip()

        # 4. Send via messaging (broadcast to all approved chats)
        results = await self._messaging.broadcast_text(digest_text)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send digest: %s", errors or "no approved chats")
            return False
        first_success = next(r for r in results if r.success)

        # 5. Update digest state
        state = self._store.read_digest_state()
        state = state.model_copy(update={
            "last_digest_sent_at": now,
            "last_digest_memory_ids": [m.id for m in top_memories],
            "last_digest_message_id": first_success.message_id,
            "last_digest_text": digest_text,
        })
        self._store.write_digest_state(state)

        # 6. Record nudges sent
        if nudges:
            for nudge in nudges:
                record_nudge(nudge_state, nudge.person.person_id, today, "morning_digest")
            self._store.write_nudge_state(nudge_state)
            self._store.increment_metric("nudges_sent", count=len(nudges))

        # 6b. Track metrics
        self._store.increment_metric("digests_sent")

        # 7. Git commit
        self._git.auto_commit(
            "morning",
            f"Digest sent ({len(top_memories)} memories)",
            timestamp=now.date(),
        )

        logger.info("Morning digest sent with %d memories", len(top_memories))
        return True

    async def _send_question_fallback(self) -> bool:
        """Send a question instead of a digest when there are no memories for today."""
        pq = self._store.read_pending_questions()
        people = self._store.read_all_people()
        prefs = self._store.read_preferences()
        question: PendingQuestion | None = None

        # 1. Pick first pending question
        for q in pq.questions:
            if q.status == "pending":
                question = q
                break

        # 2. No pending questions — scan memories for thin ones
        if question is None:
            all_memories = self._store.list_memories(limit=100)
            for memory in all_memories:
                if is_thin_memory(memory):
                    messages = generate_question_text(
                        "memory_enrichment", memory.id, people, prefs,
                    )
                    response = await self._llm.chat(
                        messages, model=self._model, temperature=0.7,
                    )
                    question = PendingQuestion(
                        id=f"q_{uuid.uuid4().hex[:8]}",
                        type="memory_enrichment",
                        subject=memory.id,
                        question=(response.content or "").strip(),
                        status="pending",
                        created_at=datetime.now(UTC),
                    )
                    pq.questions.append(question)
                    break

        # 3. No thin memories — generate a context_gap question
        if question is None:
            messages = generate_question_text("context_gap", "family", people, prefs)
            response = await self._llm.chat(
                messages, model=self._model, temperature=0.7,
            )
            question = PendingQuestion(
                id=f"q_{uuid.uuid4().hex[:8]}",
                type="context_gap",
                subject="family",
                question=(response.content or "").strip(),
                status="pending",
                created_at=datetime.now(UTC),
            )
            pq.questions.append(question)

        # Generate question text if missing
        if not question.question:
            messages = generate_question_text(question.type, question.subject, people, prefs)
            response = await self._llm.chat(
                messages, model=self._model, temperature=0.7,
            )
            question.question = (response.content or "").strip()

        # 4. Wrap in a morning greeting via LLM
        messages = morning_question(question.question, people, prefs)
        response = await self._llm.chat(messages, model=self._model, temperature=0.7)
        greeting_text = (response.content or "").strip()

        # 5. Send and track
        results = await self._messaging.broadcast_text(greeting_text)
        if not results or not any(r.success for r in results):
            errors = ", ".join(r.error or "unknown" for r in results)
            logger.error("Failed to send question fallback: %s", errors or "no approved chats")
            return False

        first_ok = next(r for r in results if r.success)
        question.status = "asked"
        question.message_id = first_ok.message_id
        self._store.write_pending_questions(pq)

        logger.info("Morning question fallback sent: %s", question.id)
        return True
