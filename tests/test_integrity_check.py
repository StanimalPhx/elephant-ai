"""Tests for IntegrityCheckFlow."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from elephant.data.models import (
    CurrentThread,
    Memory,
    PendingQuestion,
    PendingQuestionsFile,
    Person,
)
from elephant.data.store import DataStore
from elephant.flows.integrity_check import IntegrityCheckFlow
from elephant.llm.client import LLMResponse


@pytest.fixture
def integrity_store(data_dir):
    store = DataStore(data_dir)
    store.initialize()
    return store


@pytest.fixture
def mock_git():
    git = MagicMock()
    git.auto_commit = MagicMock()
    return git


class TestIntegrityCheckFlow:
    async def test_clean_run(self, integrity_store, mock_git):
        """No issues should produce a clean run record."""
        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        result = await flow.run()
        assert result is True

        runs, total = integrity_store.read_integrity_runs()
        assert total == 1
        record = runs[0]
        assert record.issues_found == 0
        assert record.auto_fixed == 0
        assert record.questions_created == 0
        assert record.error is None
        assert record.finished_at is not None

    async def test_auto_fix_stale_thread(self, integrity_store, mock_git):
        """Stale threads should be auto-archived."""
        old_date = date.today() - timedelta(days=90)
        integrity_store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["friend"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="planning trip",
                last_mentioned_date=old_date,
            )],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        result = await flow.run()
        assert result is True

        # Check that thread was archived
        person = integrity_store.read_person("charlie")
        assert person is not None
        assert len(person.current_threads) == 0
        assert len(person.archived_threads) == 1
        assert person.archived_threads[0].topic == "vacation"

        # Check git commit was called
        mock_git.auto_commit.assert_called_once()

        # Check run record
        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        assert record.auto_fixed >= 1
        auto_fixed = [f for f in record.findings if f.action == "auto_fixed"]
        assert len(auto_fixed) >= 1

    async def test_question_created_for_unknown_relationship(
        self, integrity_store, mock_git,
    ):
        """Unknown relationships should create pending questions."""
        integrity_store.write_person(Person(
            person_id="dana", display_name="Dana",
            relationship=["unknown"],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        pq = integrity_store.read_pending_questions()
        matching = [q for q in pq.questions if q.type == "unknown_relationship"]
        assert len(matching) == 1
        assert "Dana" in matching[0].question

    async def test_question_created_for_orphan_person(
        self, integrity_store, mock_git,
    ):
        """Orphan people (in memories but no profile) should create questions."""
        integrity_store.write_memory(Memory(
            id="20250101_test",
            date=date(2025, 1, 1),
            title="Lunch with Zoe",
            type="daily",
            description="Had lunch",
            people=["Zoe"],
            source="manual",
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        pq = integrity_store.read_pending_questions()
        matching = [q for q in pq.questions if q.type == "orphan_person"]
        assert len(matching) == 1
        assert "Zoe" in matching[0].question

    async def test_question_deduplication(self, integrity_store, mock_git):
        """Should not create duplicate questions."""
        integrity_store.write_person(Person(
            person_id="dana", display_name="Dana",
            relationship=["unknown"],
        ))

        # Pre-existing question
        from datetime import UTC, datetime

        pq = PendingQuestionsFile(questions=[
            PendingQuestion(
                id="q_existing",
                type="unknown_relationship",
                subject="dana",
                question="How is Dana related?",
                status="pending",
                created_at=datetime.now(UTC),
            ),
        ])
        integrity_store.write_pending_questions(pq)

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        pq_after = integrity_store.read_pending_questions()
        matching = [q for q in pq_after.questions if q.type == "unknown_relationship"]
        assert len(matching) == 1  # Still just the original

    async def test_question_created_for_duplicate_person_name(
        self, integrity_store, mock_git,
    ):
        """Duplicate person names should create pending questions."""
        integrity_store.write_person(Person(
            person_id="alice1", display_name="Alice",
            relationship=["sister"],
        ))
        integrity_store.write_person(Person(
            person_id="alice2", display_name="Alice",
            relationship=["friend"],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        pq = integrity_store.read_pending_questions()
        matching = [q for q in pq.questions if q.type == "duplicate_person_name"]
        assert len(matching) == 1
        assert "Alice" in matching[0].question.lower() or "alice" in matching[0].question.lower()

    async def test_run_record_persisted(self, integrity_store, mock_git):
        """Run record should be readable by ID."""
        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        runs, _ = integrity_store.read_integrity_runs()
        run_id = runs[0].run_id

        record = integrity_store.read_integrity_run_by_id(run_id)
        assert record is not None
        assert record.run_id == run_id

    async def test_auto_fix_empty_person_id(self, integrity_store, mock_git):
        """Empty person_id should be auto-fixed by deriving from display_name."""
        integrity_store.write_person(Person(
            person_id="", display_name="Test Person", relationship=["unknown"],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        result = await flow.run()
        assert result is True

        # Old .yaml file should be gone
        import os

        old_path = os.path.join(integrity_store.data_dir, "people", ".yaml")
        assert not os.path.exists(old_path)

        # New file should exist with correct person_id
        person = integrity_store.read_person("test_person")
        assert person is not None
        assert person.person_id == "test_person"
        assert person.display_name == "Test Person"

        # Check run record
        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        assert record.auto_fixed >= 1
        auto_fixed = [f for f in record.findings if f.action == "auto_fixed"]
        assert any(f.category == "empty_person_id" for f in auto_fixed)

        # Git commit should have been called
        mock_git.auto_commit.assert_called_once()

    async def test_no_git_commit_without_fixes(self, integrity_store, mock_git):
        """No git commit should happen if nothing was auto-fixed."""
        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()
        mock_git.auto_commit.assert_not_called()


class TestDryRun:
    async def test_dry_run_does_not_persist_record(self, integrity_store, mock_git):
        """Dry run should not write anything to the store."""
        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        result = await flow.run(dry_run=True)
        assert result is True

        runs, total = integrity_store.read_integrity_runs()
        assert total == 0

    async def test_dry_run_does_not_auto_fix(self, integrity_store, mock_git):
        """Dry run should not archive stale threads."""
        old_date = date.today() - timedelta(days=90)
        integrity_store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["friend"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="planning trip",
                last_mentioned_date=old_date,
            )],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run(dry_run=True)

        # Thread should NOT have been archived
        person = integrity_store.read_person("charlie")
        assert person is not None
        assert len(person.current_threads) == 1
        assert len(person.archived_threads) == 0
        mock_git.auto_commit.assert_not_called()

    async def test_dry_run_does_not_create_questions(
        self, integrity_store, mock_git,
    ):
        """Dry run should not create pending questions."""
        integrity_store.write_person(Person(
            person_id="dana", display_name="Dana",
            relationship=["unknown"],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run(dry_run=True)

        pq = integrity_store.read_pending_questions()
        assert len(pq.questions) == 0

    async def test_dry_run_labels_would_actions(self, integrity_store, mock_git):
        """Dry run findings should have 'would_auto_fix' / 'would_create_question'."""
        old_date = date.today() - timedelta(days=90)
        integrity_store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["unknown"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="...",
                last_mentioned_date=old_date,
            )],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        record = await flow.run_dry()

        actions = {f.action for f in record.findings}
        assert "would_auto_fix" in actions
        assert "would_create_question" in actions
        assert "auto_fixed" not in actions
        assert "question_created" not in actions

    async def test_run_dry_returns_record(self, integrity_store, mock_git):
        """run_dry() should return a complete IntegrityRunRecord."""
        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        record = await flow.run_dry()

        assert record.run_id.startswith("integrity_")
        assert record.started_at is not None
        assert record.finished_at is not None
        assert record.error is None


class TestIntegrityStoreOps:
    def test_append_and_read_integrity_runs(self, data_dir):
        from datetime import UTC, datetime

        from elephant.data.models import IntegrityRunRecord

        store = DataStore(data_dir)
        store.initialize()

        r1 = IntegrityRunRecord(
            run_id="run-1", started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC), issues_found=3,
        )
        r2 = IntegrityRunRecord(
            run_id="run-2", started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC), issues_found=0,
        )
        store.append_integrity_run(r1)
        store.append_integrity_run(r2)

        runs, total = store.read_integrity_runs()
        assert total == 2
        assert runs[0].run_id == "run-2"  # Newest first
        assert runs[1].run_id == "run-1"

    def test_read_integrity_runs_empty(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        runs, total = store.read_integrity_runs()
        assert total == 0
        assert runs == []

    def test_read_integrity_run_by_id(self, data_dir):
        from datetime import UTC, datetime

        from elephant.data.models import IntegrityFinding, IntegrityRunRecord

        store = DataStore(data_dir)
        store.initialize()

        record = IntegrityRunRecord(
            run_id="run-find-me",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            issues_found=1,
            findings=[IntegrityFinding(
                category="stale_thread",
                severity="warning",
                message="test finding",
                action="auto_fixed",
            )],
        )
        store.append_integrity_run(record)

        found = store.read_integrity_run_by_id("run-find-me")
        assert found is not None
        assert found.run_id == "run-find-me"
        assert len(found.findings) == 1

    def test_read_integrity_run_by_id_not_found(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        assert store.read_integrity_run_by_id("nonexistent") is None

    def test_read_integrity_runs_pagination(self, data_dir):
        from datetime import UTC, datetime

        from elephant.data.models import IntegrityRunRecord

        store = DataStore(data_dir)
        store.initialize()

        for i in range(10):
            store.append_integrity_run(IntegrityRunRecord(
                run_id=f"run-{i}",
                started_at=datetime.now(UTC),
            ))

        runs, total = store.read_integrity_runs(limit=3, offset=0)
        assert total == 10
        assert len(runs) == 3
        assert runs[0].run_id == "run-9"


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="test", usage={"total_tokens": 50})


def _make_mock_llm(content: str = "duplicates: []\ncontradictions: []") -> AsyncMock:
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=_llm_response(content))
    return llm


class TestLLMDuplicateCheck:
    async def test_finds_semantic_duplicates(self, integrity_store, mock_git):
        """LLM-detected semantic duplicates should appear as findings."""
        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_park\n"
            "  id_b: 20250301_stroll\n"
            "  reason: Both describe walking in the park with Lily\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        dup_findings = [f for f in record.findings if f.category == "semantic_duplicate"]
        assert len(dup_findings) == 1
        assert "20250301_park" in dup_findings[0].message
        assert dup_findings[0].action == "would_create_question"

    async def test_no_duplicates_found(self, integrity_store, mock_git):
        """Empty LLM response should produce no duplicate findings."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Cooking dinner", type="daily",
            description="Made pasta", people=["Mom"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park", people=["Dad"], source="manual",
        ))

        llm = _make_mock_llm("[]")
        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        dup_findings = [f for f in record.findings if f.category == "semantic_duplicate"]
        assert len(dup_findings) == 0


class TestLLMContradictionCheck:
    async def test_finds_contradictions(self, integrity_store, mock_git):
        """LLM-detected contradictions should appear as findings."""
        integrity_store.write_memory(Memory(
            id="20250301_home", date=date(2025, 3, 1),
            title="Stayed home", type="daily",
            description="Stayed home all day watching movies",
            people=["Family"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_beach", date=date(2025, 3, 1),
            title="Beach day", type="outing",
            description="Spent the whole day at the beach",
            people=["Family"], source="manual",
        ))

        yaml_resp = (
            "duplicates: []\n"
            "contradictions:\n"
            "- id_a: 20250301_home\n"
            "  id_b: 20250301_beach\n"
            "  contradiction: One says stayed home all day, other says at beach all day\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        contra_findings = [f for f in record.findings if f.category == "contradiction"]
        assert len(contra_findings) == 1
        assert "20250301_home" in contra_findings[0].message
        assert contra_findings[0].action == "would_create_question"

    async def test_no_contradictions_found(self, integrity_store, mock_git):
        """Empty LLM response should produce no contradiction findings."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Morning jog", type="daily",
            description="Jogged in the morning", people=["Dad"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Afternoon nap", type="daily",
            description="Napped in the afternoon", people=["Dad"], source="manual",
        ))

        llm = _make_mock_llm("[]")
        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        contra_findings = [f for f in record.findings if f.category == "contradiction"]
        assert len(contra_findings) == 0


class TestLLMErrorHandling:
    async def test_llm_error_does_not_crash(self, integrity_store, mock_git):
        """LLM errors should be logged but not crash the run."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Event A", type="daily",
            description="Something happened", people=["Dad"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Event B", type="daily",
            description="Something else", people=["Mom"], source="manual",
        ))

        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        # Should complete without error (LLM errors are caught per-date)
        assert record.error is None

    async def test_llm_invalid_yaml_graceful(self, integrity_store, mock_git):
        """Invalid YAML from LLM should not crash the run."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Event A", type="daily",
            description="Something", people=["Dad"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Event B", type="daily",
            description="Something else", people=["Mom"], source="manual",
        ))

        llm = _make_mock_llm("not: valid: yaml: [[[")
        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        # Should handle gracefully (invalid yaml → not a list → skip)
        assert record.error is None

    async def test_skips_dates_with_single_memory(self, integrity_store, mock_git):
        """Dates with only one memory should not trigger LLM calls."""
        integrity_store.write_memory(Memory(
            id="20250301_only", date=date(2025, 3, 1),
            title="Only event", type="daily",
            description="Just one thing", people=["Dad"], source="manual",
        ))

        llm = _make_mock_llm("[]")
        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        await flow.run_dry()

        # LLM should never be called for single-memory dates
        llm.chat.assert_not_called()

    async def test_no_llm_when_not_configured(self, integrity_store, mock_git):
        """Flow without LLM should skip LLM checks entirely."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Event A", type="daily",
            description="Something", people=["Dad"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Event B", type="daily",
            description="Something else", people=["Mom"], source="manual",
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        record = await flow.run_dry()

        # No LLM findings should be present
        llm_findings = [
            f for f in record.findings
            if f.category in ("semantic_duplicate", "contradiction")
        ]
        assert len(llm_findings) == 0


class TestLLMIssuesCount:
    async def test_issues_found_includes_llm_findings(self, integrity_store, mock_git):
        """issues_found should count LLM findings too."""
        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_park\n"
            "  id_b: 20250301_stroll\n"
            "  reason: Both describe walking in the park with Lily\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        # issues_found should include the 1 LLM duplicate finding
        llm_count = len([
            f for f in record.findings
            if f.category in ("semantic_duplicate", "contradiction")
        ])
        assert llm_count > 0
        assert record.issues_found >= llm_count


class TestLLMTracing:
    async def test_llm_calls_produce_trace_steps(self, integrity_store, mock_git):
        """LLM calls should record LLMCallStep trace steps."""
        integrity_store.write_memory(Memory(
            id="20250301_a", date=date(2025, 3, 1),
            title="Event A", type="daily",
            description="Went to the park", people=["Dad"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_b", date=date(2025, 3, 1),
            title="Event B", type="daily",
            description="Stayed at home", people=["Mom"], source="manual",
        ))

        llm = _make_mock_llm("[]")
        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        # run_dry starts and finishes its own trace, so the context is cleared.
        # Single combined call per date group (duplicates + contradictions together)
        assert llm.chat.call_count == 1
        assert record.error is None


class TestExplanations:
    async def test_auto_fix_stale_thread_has_explanation(self, integrity_store, mock_git):
        """Auto-fixed stale thread findings should have an explanation."""
        old_date = date.today() - timedelta(days=90)
        integrity_store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["friend"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="planning trip",
                last_mentioned_date=old_date,
            )],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        stale_findings = [
            f for f in record.findings
            if f.category == "stale_thread" and f.action == "auto_fixed"
        ]
        assert len(stale_findings) >= 1
        finding = stale_findings[0]
        assert finding.explanation is not None
        assert "vacation" in finding.explanation
        assert "Charlie" in finding.explanation
        assert "days ago" in finding.explanation

    async def test_auto_fix_empty_person_id_has_explanation(self, integrity_store, mock_git):
        """Auto-fixed empty person_id findings should have an explanation."""
        integrity_store.write_person(Person(
            person_id="", display_name="Test Person", relationship=["unknown"],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        await flow.run()

        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        fixed = [
            f for f in record.findings
            if f.category == "empty_person_id" and f.action == "auto_fixed"
        ]
        assert len(fixed) >= 1
        assert fixed[0].explanation is not None
        assert "test_person" in fixed[0].explanation
        assert "Test Person" in fixed[0].explanation

    async def test_dry_run_stale_thread_has_explanation(self, integrity_store, mock_git):
        """Dry-run stale thread findings should describe what would happen."""
        old_date = date.today() - timedelta(days=90)
        integrity_store.write_person(Person(
            person_id="charlie", display_name="Charlie",
            relationship=["friend"],
            current_threads=[CurrentThread(
                topic="vacation", latest_update="planning trip",
                last_mentioned_date=old_date,
            )],
        ))

        flow = IntegrityCheckFlow(store=integrity_store, git=mock_git)
        record = await flow.run_dry()

        stale = [
            f for f in record.findings
            if f.category == "stale_thread" and f.action == "would_auto_fix"
        ]
        assert len(stale) >= 1
        assert stale[0].explanation is not None
        assert "Would archive" in stale[0].explanation

    async def test_backward_compat_no_explanation(self):
        """IntegrityFinding without explanation should default to None."""
        from elephant.data.models import IntegrityFinding

        finding = IntegrityFinding(
            category="stale_thread",
            severity="warning",
            message="test",
            action="logged",
        )
        assert finding.explanation is None

        # Also test deserialization from dict without explanation
        data = {
            "category": "stale_thread",
            "severity": "warning",
            "message": "test",
            "action": "logged",
            "details": {},
        }
        finding2 = IntegrityFinding(**data)
        assert finding2.explanation is None


class TestLLMQuestionCreation:
    async def test_semantic_duplicate_creates_question_on_run(
        self, integrity_store, mock_git,
    ):
        """Real run should create a pending question for semantic_duplicate."""
        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_park\n"
            "  id_b: 20250301_stroll\n"
            "  reason: Both describe walking in the park with Lily\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        await flow.run()

        pq = integrity_store.read_pending_questions()
        matching = [q for q in pq.questions if q.type == "semantic_duplicate"]
        assert len(matching) == 1
        assert "20250301_park" in matching[0].question
        assert "20250301_stroll" in matching[0].question

        # Check finding action
        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        dup_findings = [
            f for f in record.findings if f.category == "semantic_duplicate"
        ]
        assert len(dup_findings) == 1
        assert dup_findings[0].action == "question_created"

    async def test_contradiction_creates_question_on_run(
        self, integrity_store, mock_git,
    ):
        """Real run should create a pending question for contradiction."""
        integrity_store.write_memory(Memory(
            id="20250301_home", date=date(2025, 3, 1),
            title="Stayed home", type="daily",
            description="Stayed home all day",
            people=["Family"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_beach", date=date(2025, 3, 1),
            title="Beach day", type="outing",
            description="Spent the day at the beach",
            people=["Family"], source="manual",
        ))

        yaml_resp = (
            "duplicates: []\n"
            "contradictions:\n"
            "- id_a: 20250301_home\n"
            "  id_b: 20250301_beach\n"
            "  contradiction: One says home, other says beach\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        await flow.run()

        pq = integrity_store.read_pending_questions()
        matching = [q for q in pq.questions if q.type == "contradiction"]
        assert len(matching) == 1
        assert "20250301_home" in matching[0].question

        # Check finding action
        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        contra = [f for f in record.findings if f.category == "contradiction"]
        assert len(contra) == 1
        assert contra[0].action == "question_created"

    async def test_dry_run_llm_findings_would_create_question(
        self, integrity_store, mock_git,
    ):
        """Dry-run LLM findings should show would_create_question action."""
        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_park\n"
            "  id_b: 20250301_stroll\n"
            "  reason: Both describe walking in the park with Lily\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        record = await flow.run_dry()

        dup_findings = [
            f for f in record.findings if f.category == "semantic_duplicate"
        ]
        assert len(dup_findings) == 1
        assert dup_findings[0].action == "would_create_question"

        # No questions should be created in dry-run
        pq = integrity_store.read_pending_questions()
        assert len(pq.questions) == 0

    async def test_llm_question_dedup_sorted_key(
        self, integrity_store, mock_git,
    ):
        """LLM question dedup should use sorted id_a|id_b as subject key."""
        from datetime import UTC, datetime

        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        # Pre-existing question with sorted subject key
        pq = PendingQuestionsFile(questions=[
            PendingQuestion(
                id="q_existing",
                type="semantic_duplicate",
                subject="20250301_park|20250301_stroll",
                question="Already asked",
                status="pending",
                created_at=datetime.now(UTC),
            ),
        ])
        integrity_store.write_pending_questions(pq)

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_stroll\n"
            "  id_b: 20250301_park\n"
            "  reason: Same event\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        await flow.run()

        pq_after = integrity_store.read_pending_questions()
        matching = [
            q for q in pq_after.questions if q.type == "semantic_duplicate"
        ]
        assert len(matching) == 1  # No duplicate created

    async def test_questions_created_count_includes_llm(
        self, integrity_store, mock_git,
    ):
        """questions_created on the record should include LLM question counts."""
        integrity_store.write_memory(Memory(
            id="20250301_park", date=date(2025, 3, 1),
            title="Park walk", type="daily",
            description="Walked in the park with Lily",
            people=["Lily"], source="manual",
        ))
        integrity_store.write_memory(Memory(
            id="20250301_stroll", date=date(2025, 3, 1),
            title="Stroll outside", type="daily",
            description="Took Lily for a stroll in the park",
            people=["Lily"], source="manual",
        ))

        yaml_resp = (
            "duplicates:\n"
            "- id_a: 20250301_park\n"
            "  id_b: 20250301_stroll\n"
            "  reason: Both describe walking in the park with Lily\n"
            "contradictions: []\n"
        )
        llm = _make_mock_llm(yaml_resp)

        flow = IntegrityCheckFlow(
            store=integrity_store, git=mock_git, llm=llm, model="test-model",
        )
        await flow.run()

        runs, _ = integrity_store.read_integrity_runs()
        record = runs[0]
        assert record.questions_created >= 1
