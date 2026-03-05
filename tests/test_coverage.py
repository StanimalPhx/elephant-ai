"""Tests for coverage gap detection."""

from __future__ import annotations

from datetime import date

from elephant.brain.coverage import (
    find_coverage_gaps,
    format_gaps_for_monthly,
    generate_backfill_prompt,
)


class _FakeMemory:
    def __init__(self, d: date) -> None:
        self.date = d


class TestFindCoverageGaps:
    def test_no_gaps(self) -> None:
        memories = [_FakeMemory(date(2025, m, 15)) for m in range(1, 7)]
        gaps = find_coverage_gaps(memories, date(2025, 1, 1), date(2025, 6, 30))
        assert gaps == []

    def test_single_gap(self) -> None:
        # March is missing
        memories = [
            _FakeMemory(date(2025, 1, 10)),
            _FakeMemory(date(2025, 2, 10)),
            _FakeMemory(date(2025, 4, 10)),
        ]
        gaps = find_coverage_gaps(memories, date(2025, 1, 1), date(2025, 4, 30))
        assert gaps == [(2025, 3)]

    def test_multiple_gaps(self) -> None:
        memories = [_FakeMemory(date(2025, 1, 10))]
        gaps = find_coverage_gaps(memories, date(2025, 1, 1), date(2025, 4, 30))
        assert gaps == [(2025, 2), (2025, 3), (2025, 4)]

    def test_empty_memories(self) -> None:
        gaps = find_coverage_gaps([], date(2025, 1, 1), date(2025, 3, 31))
        assert gaps == [(2025, 1), (2025, 2), (2025, 3)]

    def test_cross_year_boundary(self) -> None:
        memories = [_FakeMemory(date(2024, 12, 10))]
        gaps = find_coverage_gaps(memories, date(2024, 11, 1), date(2025, 2, 28))
        assert (2024, 11) in gaps
        assert (2024, 12) not in gaps
        assert (2025, 1) in gaps
        assert (2025, 2) in gaps


class TestFormatGapsForMonthly:
    def test_no_gaps(self) -> None:
        assert format_gaps_for_monthly([]) is None

    def test_single_gap(self) -> None:
        result = format_gaps_for_monthly([(2025, 8)])
        assert result is not None
        assert "August 2025" in result
        assert "Want to fill" in result

    def test_max_gaps(self) -> None:
        gaps = [(2025, 1), (2025, 3), (2025, 5), (2025, 7)]
        result = format_gaps_for_monthly(gaps, max_gaps=2)
        assert result is not None
        assert "January 2025" in result
        assert "March 2025" in result
        assert "2 more gaps" in result


class TestGenerateBackfillPrompt:
    def test_prompt_text(self) -> None:
        prompt = generate_backfill_prompt(2025, 8)
        assert "August 2025" in prompt
        assert "no memories" in prompt
