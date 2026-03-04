"""Contact awareness nudges: detect overdue contacts, format for prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from elephant.data.models import NudgeRecord, NudgeStateFile

if TYPE_CHECKING:
    from datetime import date

    from elephant.data.models import CurrentThread, Person


@dataclass
class ContactNudge:
    person: Person
    days_since_contact: int | None  # None means never contacted
    days_overdue: int | None  # None means never contacted (infinitely overdue)
    last_contact: date | None
    threads: list[CurrentThread] = field(default_factory=list)


def find_overdue_contacts(
    people: list[Person],
    last_contacts: dict[str, date | None],
    nudge_records: list[NudgeRecord],
    today: date,
    *,
    max_nudges: int = 2,
    cooldown_days: int = 30,
) -> list[ContactNudge]:
    """Find people who are overdue for contact, respecting cooldown.

    - Only considers people with interaction_frequency_target set.
    - Skips anyone nudged within cooldown_days.
    - Treats missing last_contact as infinitely overdue.
    - Returns up to max_nudges, sorted by most overdue first.
    """
    # Build cooldown lookup: person_id -> last_nudged_at
    cooldown_map: dict[str, date] = {}
    for rec in nudge_records:
        existing = cooldown_map.get(rec.person_id)
        if existing is None or rec.last_nudged_at > existing:
            cooldown_map[rec.person_id] = rec.last_nudged_at

    nudges: list[ContactNudge] = []
    for person in people:
        if person.interaction_frequency_target is None:
            continue

        # Check cooldown
        last_nudged = cooldown_map.get(person.person_id)
        if last_nudged is not None and (today - last_nudged).days < cooldown_days:
            continue

        last_contact = last_contacts.get(person.display_name)
        if last_contact is None:
            # Never contacted — infinitely overdue
            nudges.append(ContactNudge(
                person=person,
                days_since_contact=None,
                days_overdue=None,
                last_contact=None,
                threads=list(person.current_threads),
            ))
        else:
            days_since = (today - last_contact).days
            target = person.interaction_frequency_target
            if days_since > target:
                nudges.append(ContactNudge(
                    person=person,
                    days_since_contact=days_since,
                    days_overdue=days_since - target,
                    last_contact=last_contact,
                    threads=list(person.current_threads),
                ))

    # Sort: never-contacted first, then most overdue
    def _sort_key(n: ContactNudge) -> tuple[int, int]:
        if n.days_overdue is None:
            return (0, 0)  # never contacted → highest priority
        return (1, -n.days_overdue)

    nudges.sort(key=_sort_key)
    return nudges[:max_nudges]


def format_nudges_for_prompt(nudges: list[ContactNudge]) -> str:
    """Build a text block describing overdue contacts for LLM prompts."""
    if not nudges:
        return ""
    lines: list[str] = []
    for nudge in nudges:
        name = nudge.person.display_name
        if nudge.last_contact is None:
            line = f"- {name}: no recorded contact yet"
        else:
            line = f"- {name}: last contact {nudge.days_since_contact} days ago"
            if nudge.days_overdue is not None:
                line += f" ({nudge.days_overdue} days overdue)"
        if nudge.threads:
            topics = ", ".join(t.topic for t in nudge.threads)
            line += f" — active threads: {topics}"
        lines.append(line)
    return "People you haven't reached out to in a while:\n" + "\n".join(lines)


def record_nudge(
    nudge_state: NudgeStateFile,
    person_id: str,
    today: date,
    context: str | None = None,
) -> None:
    """Update or append a nudge record in-place."""
    for rec in nudge_state.records:
        if rec.person_id == person_id:
            rec.last_nudged_at = today
            rec.context = context
            return
    nudge_state.records.append(NudgeRecord(
        person_id=person_id,
        last_nudged_at=today,
        context=context,
    ))
