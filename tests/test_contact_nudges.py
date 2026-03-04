"""Tests for contact awareness nudges."""

from __future__ import annotations

from datetime import date, timedelta

from elephant.data.models import CurrentThread, NudgeRecord, NudgeStateFile, Person
from elephant.flows.contact_nudges import (
    ContactNudge,
    find_overdue_contacts,
    format_nudges_for_prompt,
    record_nudge,
)


def _person(
    pid: str,
    name: str,
    target: int | None = None,
    threads: list[CurrentThread] | None = None,
) -> Person:
    return Person(
        person_id=pid,
        display_name=name,
        relationship=["friend"],
        interaction_frequency_target=target,
        current_threads=threads or [],
    )


TODAY = date(2026, 3, 4)


class TestFindOverdueContacts:
    def test_no_people_with_target(self) -> None:
        people = [_person("a", "Alice")]
        result = find_overdue_contacts(people, {}, [], TODAY)
        assert result == []

    def test_person_with_target_no_contact(self) -> None:
        """Person with target but no contact history is infinitely overdue."""
        people = [_person("a", "Alice", target=14)]
        result = find_overdue_contacts(people, {"Alice": None}, [], TODAY)
        assert len(result) == 1
        assert result[0].person.person_id == "a"
        assert result[0].days_since_contact is None
        assert result[0].days_overdue is None
        assert result[0].last_contact is None

    def test_person_not_overdue(self) -> None:
        """Person contacted within target is not returned."""
        people = [_person("a", "Alice", target=14)]
        last = {"Alice": TODAY - timedelta(days=10)}
        result = find_overdue_contacts(people, last, [], TODAY)
        assert result == []

    def test_person_overdue(self) -> None:
        people = [_person("a", "Alice", target=14)]
        last = {"Alice": TODAY - timedelta(days=20)}
        result = find_overdue_contacts(people, last, [], TODAY)
        assert len(result) == 1
        assert result[0].days_since_contact == 20
        assert result[0].days_overdue == 6

    def test_cooldown_skips_recently_nudged(self) -> None:
        people = [_person("a", "Alice", target=14)]
        last = {"Alice": TODAY - timedelta(days=20)}
        records = [NudgeRecord(person_id="a", last_nudged_at=TODAY - timedelta(days=5))]
        result = find_overdue_contacts(people, last, records, TODAY, cooldown_days=30)
        assert result == []

    def test_cooldown_expired(self) -> None:
        people = [_person("a", "Alice", target=14)]
        last = {"Alice": TODAY - timedelta(days=60)}
        records = [NudgeRecord(person_id="a", last_nudged_at=TODAY - timedelta(days=31))]
        result = find_overdue_contacts(people, last, records, TODAY, cooldown_days=30)
        assert len(result) == 1

    def test_max_nudges_limit(self) -> None:
        people = [
            _person("a", "Alice", target=7),
            _person("b", "Bob", target=7),
            _person("c", "Carol", target=7),
        ]
        last = {
            "Alice": TODAY - timedelta(days=20),
            "Bob": TODAY - timedelta(days=30),
            "Carol": TODAY - timedelta(days=10),
        }
        result = find_overdue_contacts(people, last, [], TODAY, max_nudges=2)
        assert len(result) == 2
        # Most overdue first
        assert result[0].person.person_id == "b"
        assert result[1].person.person_id == "a"

    def test_never_contacted_sorted_first(self) -> None:
        people = [
            _person("a", "Alice", target=7),
            _person("b", "Bob", target=7),
        ]
        last = {"Alice": TODAY - timedelta(days=30), "Bob": None}
        result = find_overdue_contacts(people, last, [], TODAY, max_nudges=2)
        assert result[0].person.person_id == "b"  # never contacted → first
        assert result[1].person.person_id == "a"

    def test_threads_included_in_nudge(self) -> None:
        threads = [
            CurrentThread(topic="Job search", latest_update="Applied", last_mentioned_date=TODAY)
        ]
        people = [_person("a", "Alice", target=7, threads=threads)]
        last = {"Alice": TODAY - timedelta(days=20)}
        result = find_overdue_contacts(people, last, [], TODAY)
        assert len(result[0].threads) == 1
        assert result[0].threads[0].topic == "Job search"


class TestFormatNudgesForPrompt:
    def test_empty(self) -> None:
        assert format_nudges_for_prompt([]) == ""

    def test_single_overdue(self) -> None:
        nudge = ContactNudge(
            person=_person("a", "Alice", target=14),
            days_since_contact=20,
            days_overdue=6,
            last_contact=TODAY - timedelta(days=20),
        )
        text = format_nudges_for_prompt([nudge])
        assert "Alice" in text
        assert "20 days ago" in text
        assert "6 days overdue" in text

    def test_never_contacted(self) -> None:
        nudge = ContactNudge(
            person=_person("a", "Alice", target=14),
            days_since_contact=None,
            days_overdue=None,
            last_contact=None,
        )
        text = format_nudges_for_prompt([nudge])
        assert "no recorded contact" in text

    def test_includes_threads(self) -> None:
        threads = [
            CurrentThread(topic="Wedding", latest_update="Booked venue", last_mentioned_date=TODAY)
        ]
        nudge = ContactNudge(
            person=_person("a", "Alice", target=14, threads=threads),
            days_since_contact=20,
            days_overdue=6,
            last_contact=TODAY - timedelta(days=20),
            threads=threads,
        )
        text = format_nudges_for_prompt([nudge])
        assert "Wedding" in text


class TestRecordNudge:
    def test_new_record(self) -> None:
        state = NudgeStateFile(records=[])
        record_nudge(state, "a", TODAY, "morning_digest")
        assert len(state.records) == 1
        assert state.records[0].person_id == "a"
        assert state.records[0].last_nudged_at == TODAY
        assert state.records[0].context == "morning_digest"

    def test_update_existing(self) -> None:
        old_date = TODAY - timedelta(days=45)
        state = NudgeStateFile(records=[
            NudgeRecord(person_id="a", last_nudged_at=old_date, context="evening_checkin"),
        ])
        record_nudge(state, "a", TODAY, "morning_digest")
        assert len(state.records) == 1  # no duplicate
        assert state.records[0].last_nudged_at == TODAY
        assert state.records[0].context == "morning_digest"

    def test_multiple_people(self) -> None:
        state = NudgeStateFile(records=[])
        record_nudge(state, "a", TODAY, "morning_digest")
        record_nudge(state, "b", TODAY, "morning_digest")
        assert len(state.records) == 2
