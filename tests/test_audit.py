"""Tests for audit consistency checks."""

from datetime import date, timedelta

from elephant.audit import (
    AuditIssue,
    run_audit,
    run_full_audit,
)
from elephant.data.models import (
    CurrentThread,
    Group,
    Memory,
    Person,
)
from elephant.data.store import DataStore


class TestAuditIssueAutoFixable:
    def test_default_not_auto_fixable(self):
        issue = AuditIssue(category="test", severity="warning", message="msg")
        assert issue.auto_fixable is False

    def test_auto_fixable_flag(self):
        issue = AuditIssue(
            category="stale_thread", severity="warning", message="msg",
            auto_fixable=True,
        )
        assert issue.auto_fixable is True


class TestCheckEmptyPersonId:
    def test_detects_empty_person_id(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="", display_name="Ghost", relationship=["unknown"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "empty_person_id" in cats

    def test_empty_person_id_is_auto_fixable(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="", display_name="Ghost", relationship=["unknown"],
        ))
        report = run_full_audit(store)
        issues = [i for i in report.issues if i.category == "empty_person_id"]
        assert len(issues) == 1
        assert issues[0].auto_fixable is True

    def test_no_issue_for_valid_person(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="alice", display_name="Alice", relationship=["sister"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "empty_person_id" not in cats


class TestCheckInvalidGroupRefs:
    def test_detects_invalid_group(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="bob", display_name="Bob",
            relationship=["brother"], groups=["nonexistent_group"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "invalid_group_ref" in cats

    def test_no_issue_for_valid_group(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_group(Group(group_id="family", display_name="Family"))
        store.write_person(Person(
            person_id="bob", display_name="Bob",
            relationship=["brother"], groups=["family"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "invalid_group_ref" not in cats


class TestCheckDuplicatePersonNames:
    def test_detects_duplicate_names(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="alice1", display_name="Alice", relationship=["sister"],
        ))
        store.write_person(Person(
            person_id="alice2", display_name="Alice", relationship=["friend"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "duplicate_person_name" in cats

    def test_no_issue_for_unique_names(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        store.write_person(Person(
            person_id="alice", display_name="Alice", relationship=["sister"],
        ))
        store.write_person(Person(
            person_id="bob", display_name="Bob", relationship=["brother"],
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "duplicate_person_name" not in cats


class TestStaleThreadAutoFixable:
    def test_stale_thread_is_auto_fixable(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        old_date = date.today() - timedelta(days=90)
        store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["friend"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="...",
                last_mentioned_date=old_date,
            )],
        ))
        report = run_audit(store)
        stale = [i for i in report.issues if i.category == "stale_thread"]
        assert len(stale) == 1
        assert stale[0].auto_fixable is True


class TestRunFullAudit:
    def test_includes_all_checks(self, data_dir):
        """run_full_audit should not crash on an empty store."""
        store = DataStore(data_dir)
        store.initialize()
        report = run_full_audit(store)
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_full_audit_catches_existing_issues(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        # Create a memory with empty title (malformed check)
        store.write_memory(Memory(
            id="20250101_test",
            date=date(2025, 1, 1),
            title="",
            type="daily",
            description="A memory with no title",
            people=["Alice"],
            source="manual",
        ))
        report = run_full_audit(store)
        cats = [i.category for i in report.issues]
        assert "malformed" in cats
