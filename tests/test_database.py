"""Tests for DatabaseInstance."""

from unittest.mock import MagicMock

from elephant.config import ScheduleConfig
from elephant.database import DatabaseInstance


class TestDatabaseInstance:
    def test_create_instance(self):
        """DatabaseInstance bundles all per-database objects."""
        db = DatabaseInstance(
            name="test_db",
            auth_secret="secret123",
            store=MagicMock(),
            git=MagicMock(),
            messaging=MagicMock(),
            anytime=MagicMock(),
            morning=MagicMock(),
            evening=MagicMock(),
            question_mgr=MagicMock(),
            monthly_report=MagicMock(),
            weekly_recap=MagicMock(),
            year_in_review=MagicMock(),
            schedule=ScheduleConfig(),
        )
        assert db.name == "test_db"
        assert db.auth_secret == "secret123"
        assert db.schedule.timezone == "America/Chicago"

    def test_instances_are_independent(self):
        """Two instances should have separate stores."""
        store_a = MagicMock()
        store_b = MagicMock()
        db_a = DatabaseInstance(
            name="a", auth_secret="sec_a",
            store=store_a, git=MagicMock(), messaging=MagicMock(),
            anytime=MagicMock(), morning=MagicMock(), evening=MagicMock(),
            question_mgr=MagicMock(), monthly_report=MagicMock(),
            weekly_recap=MagicMock(), year_in_review=MagicMock(),
            schedule=ScheduleConfig(),
        )
        db_b = DatabaseInstance(
            name="b", auth_secret="sec_b",
            store=store_b, git=MagicMock(), messaging=MagicMock(),
            anytime=MagicMock(), morning=MagicMock(), evening=MagicMock(),
            question_mgr=MagicMock(), monthly_report=MagicMock(),
            weekly_recap=MagicMock(), year_in_review=MagicMock(),
            schedule=ScheduleConfig(),
        )
        assert db_a.store is not db_b.store
        assert db_a.name != db_b.name
