"""Tests for people completeness scoring."""

from __future__ import annotations

from datetime import date

from elephant.brain.people_completeness import (
    format_completeness_for_monthly,
    score_person,
)
from elephant.data.models import CurrentThread, LifeEvent, Person


class TestScorePerson:
    def test_empty_person(self) -> None:
        person = Person(person_id="test", display_name="Test")
        assert score_person(person) == 0

    def test_full_person(self) -> None:
        person = Person(
            person_id="test",
            display_name="Test",
            relationship=["child"],
            birthday=date(2020, 5, 15),
            groups=["close-friends"],
            life_events=[LifeEvent(date=date(2020, 5, 15), description="Born")],
            current_threads=[
                CurrentThread(
                    topic="School", latest_update="Started 1st grade",
                    last_mentioned_date=date(2026, 1, 1),
                )
            ],
            notes="Loves dinosaurs",
            interaction_frequency_target=7,
            attributes={"hobby": "drawing"},
        )
        assert score_person(person) == 100

    def test_partial_person(self) -> None:
        person = Person(
            person_id="test",
            display_name="Test",
            relationship=["child"],  # 15 points
            birthday=date(2020, 5, 15),  # 25 points
        )
        assert score_person(person) == 40

    def test_unknown_relationship_gets_zero(self) -> None:
        person = Person(
            person_id="test",
            display_name="Test",
            relationship=["unknown"],
        )
        assert score_person(person) == 0


class TestFormatCompletenessForMonthly:
    def test_no_people(self) -> None:
        assert format_completeness_for_monthly([]) is None

    def test_with_people(self) -> None:
        people = [
            Person(
                person_id="a", display_name="Alice",
                birthday=date(1990, 1, 1), relationship=["friend"],
            ),
            Person(
                person_id="b", display_name="Bob",
                relationship=["unknown"],
            ),
        ]
        result = format_completeness_for_monthly(people)
        assert result is not None
        assert "1 person is missing a birthday" in result
        assert "Bob" in result
        assert "0% complete" in result

    def test_all_have_birthdays(self) -> None:
        people = [
            Person(
                person_id="a", display_name="Alice",
                birthday=date(1990, 1, 1), relationship=["friend"],
            ),
        ]
        result = format_completeness_for_monthly(people)
        assert result is not None
        assert "missing a birthday" not in result
