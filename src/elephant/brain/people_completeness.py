"""People completeness scoring: rate profile fill-level."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elephant.data.models import Person


def score_person(person: Person) -> int:
    """Score a person's profile completeness (0-100).

    Scoring:
      birthday set: 25 points
      relationship not ["unknown"]: 15 points
      groups non-empty: 10 points
      life_events non-empty: 15 points
      current_threads non-empty: 10 points
      notes set: 10 points
      interaction_frequency_target set: 10 points
      attributes non-empty: 5 points
    Total: 100 points
    """
    score = 0
    if person.birthday is not None:
        score += 25
    if person.relationship != ["unknown"]:
        score += 15
    if person.groups:
        score += 10
    if person.life_events:
        score += 15
    if person.current_threads:
        score += 10
    if person.notes:
        score += 10
    if person.interaction_frequency_target is not None:
        score += 10
    if person.attributes:
        score += 5
    return score


def format_completeness_for_monthly(people: list[Person]) -> str | None:
    """Format completeness info for the monthly report. Returns None if no people."""
    if not people:
        return None

    scores = [(p, score_person(p)) for p in people]

    # Find people missing birthdays
    missing_birthday = [p for p in people if p.birthday is None]

    # Find the least complete person
    scores.sort(key=lambda x: x[1])
    least_complete = scores[0]

    # Find the most complete person
    most_complete = scores[-1]

    parts: list[str] = []

    if missing_birthday:
        count = len(missing_birthday)
        noun = "person is" if count == 1 else "people are"
        parts.append(f"{count} {noun} missing a birthday.")

    parts.append(
        f"{least_complete[0].display_name}'s profile is {least_complete[1]}% complete "
        f"(least complete)."
    )

    if most_complete[1] > least_complete[1]:
        parts.append(
            f"{most_complete[0].display_name}'s profile is {most_complete[1]}% complete "
            f"(most complete)."
        )

    return " ".join(parts)
