"""Coverage gap detection: find months with zero memories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date


def find_coverage_gaps(
    memories: Sequence[Any],
    start_date: date,
    end_date: date,
) -> list[tuple[int, int]]:
    """Find (year, month) pairs with zero memories between start_date and end_date.

    Only considers months that have fully passed (end_date should be in the past).
    """
    # Build set of (year, month) that have at least one memory
    covered: set[tuple[int, int]] = set()
    for m in memories:
        d = getattr(m, "date", None)
        if d is not None:
            covered.add((d.year, d.month))

    # Enumerate all months in range
    gaps: list[tuple[int, int]] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        if (year, month) not in covered:
            gaps.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return gaps


_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def format_gaps_for_monthly(gaps: list[tuple[int, int]], max_gaps: int = 3) -> str | None:
    """Format coverage gaps for the monthly report. Returns None if no gaps."""
    if not gaps:
        return None

    shown = gaps[:max_gaps]
    parts = [f"{_MONTH_NAMES[m]} {y}" for y, m in shown]
    text = f"You have no memories from {', '.join(parts)}."
    if len(gaps) > max_gaps:
        text += f" ({len(gaps) - max_gaps} more gaps not shown.)"
    text += " Want to fill in those gaps?"
    return text


def generate_backfill_prompt(year: int, month: int) -> str:
    """Generate a prompt to ask the user about a gap month."""
    month_name = _MONTH_NAMES[month]
    return (
        f"I noticed you have no memories from {month_name} {year}. "
        f"Did anything happen that month worth remembering?"
    )
