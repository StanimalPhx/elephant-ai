"""Integrity check flow: auto-fix safe issues, queue questions for the rest."""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import yaml

from elephant.audit import run_full_audit
from elephant.data.models import IntegrityFinding, IntegrityRunRecord, PendingQuestion
from elephant.llm.prompts import check_memory_issues
from elephant.tracing import LLMCallStep, finish_trace, record_step, start_trace

if TYPE_CHECKING:
    from elephant.data.store import DataStore
    from elephant.git_ops import GitRepo
    from elephant.llm.backend import LLMBackend

logger = logging.getLogger(__name__)

# Question templates keyed by audit category
_QUESTION_TEMPLATES: dict[str, str] = {
    "unknown_relationship": (
        "I noticed {name} has an unknown relationship."
        " How is {name} related to your family?"
    ),
    "orphan_person": (
        "I found '{name}' in memories but there's no profile."
        " Should I create one? Who are they?"
    ),
    "duplicate_memory": (
        "These memories look similar: {ids}. Should I merge them?"
    ),
    "duplicate_person_name": (
        "I found multiple profiles for '{name}'. Are these the same person?"
    ),
    "semantic_duplicate": (
        "Two memories on {date} look like they describe the same event:"
        " '{id_a}' and '{id_b}'. {reason} Should I merge them into one?"
    ),
    "contradiction": (
        "Two memories on {date} seem to contradict each other:"
        " '{id_a}' and '{id_b}'. {contradiction} Which one is correct?"
    ),
}

# Categories that should generate pending questions
_QUESTION_CATEGORIES = set(_QUESTION_TEMPLATES.keys())


class IntegrityCheckFlow:
    """Run full audit, auto-fix safe issues, create questions for the rest."""

    def __init__(
        self,
        store: DataStore,
        git: GitRepo,
        llm: LLMBackend | None = None,
        model: str = "",
        database_name: str = "",
    ) -> None:
        self._store = store
        self._git = git
        self._llm = llm
        self._model = model
        self._database_name = database_name

    async def run(self, *, dry_run: bool = False) -> bool:
        """Execute the integrity check. Returns True on success.

        When *dry_run* is True the audit runs read-only: no auto-fixes, no
        questions created, no git commit, but findings are still recorded so
        the result can be inspected.
        """
        run_id = f"integrity_{uuid.uuid4().hex[:12]}"
        record = IntegrityRunRecord(
            run_id=run_id,
            started_at=datetime.now(UTC),
        )

        mode_label = "dry-run" if dry_run else "run"
        start_trace(
            database_name=self._database_name,
            message_id=run_id,
            sender="system",
            message_text=f"integrity-check {mode_label}",
        )

        try:
            report = run_full_audit(self._store)
            any_auto_fixed = False

            for issue in report.issues:
                finding = IntegrityFinding(
                    category=issue.category,
                    severity=issue.severity,
                    message=issue.message,
                    details=issue.details,
                    action="logged",
                )

                if dry_run:
                    # In dry-run mode, label what *would* happen
                    if issue.auto_fixable:
                        finding.action = "would_auto_fix"
                        finding.explanation = self._describe_fix(issue)
                    elif issue.category in _QUESTION_CATEGORIES:
                        finding.action = "would_create_question"
                elif issue.auto_fixable:
                    explanation = self._auto_fix(issue)
                    if explanation:
                        finding.action = "auto_fixed"
                        finding.explanation = explanation
                        record.auto_fixed += 1
                        any_auto_fixed = True
                    # If fix failed, fall through to logged
                elif issue.category in _QUESTION_CATEGORIES:
                    created = self._maybe_create_question(issue)
                    if created:
                        finding.action = "question_created"
                        record.questions_created += 1

                record.findings.append(finding)

            record.issues_found = len(report.issues)

            # LLM-powered checks
            if self._llm and self._model:
                llm_findings = await self._llm_check_memory_issues(
                    dry_run=dry_run,
                )
                record.findings.extend(llm_findings)
                record.issues_found += len(llm_findings)
                if not dry_run:
                    record.questions_created += sum(
                        1 for f in llm_findings if f.action == "question_created"
                    )

            if any_auto_fixed and not dry_run:
                self._git.auto_commit("integrity", "Auto-fix integrity issues")

        except Exception as exc:
            logger.exception("Integrity check failed")
            record.error = str(exc)

        record.finished_at = datetime.now(UTC)

        summary = (
            f"{record.issues_found} issues, "
            f"{record.auto_fixed} auto-fixed, "
            f"{record.questions_created} questions"
        )
        finished = finish_trace(
            intent="integrity_check",
            final_response=summary,
            error=record.error,
        )
        if finished is not None:
            record.trace_id = finished.trace_id
            if self._database_name:
                self._store.append_trace(finished)

        if not dry_run:
            self._store.append_integrity_run(record)

        mode = " (dry-run)" if dry_run else ""
        logger.info(
            "Integrity check %s%s: %s", run_id, mode, summary,
        )
        return record.error is None

    async def run_dry(self) -> IntegrityRunRecord:
        """Run audit in dry-run mode and return the record without persisting."""
        run_id = f"integrity_{uuid.uuid4().hex[:12]}"
        record = IntegrityRunRecord(
            run_id=run_id,
            started_at=datetime.now(UTC),
        )

        start_trace(
            database_name=self._database_name,
            message_id=run_id,
            sender="system",
            message_text="integrity-check dry-run",
        )

        try:
            report = run_full_audit(self._store)
            for issue in report.issues:
                action = "logged"
                explanation: str | None = None
                if issue.auto_fixable:
                    action = "would_auto_fix"
                    explanation = self._describe_fix(issue)
                elif issue.category in _QUESTION_CATEGORIES:
                    action = "would_create_question"
                record.findings.append(IntegrityFinding(
                    category=issue.category,
                    severity=issue.severity,
                    message=issue.message,
                    details=issue.details,
                    action=action,
                    explanation=explanation,
                ))
            record.issues_found = len(report.issues)

            # LLM-powered checks (dry-run)
            if self._llm and self._model:
                llm_findings = await self._llm_check_memory_issues(
                    dry_run=True,
                )
                record.findings.extend(llm_findings)
                record.issues_found += len(llm_findings)

        except Exception as exc:
            logger.exception("Integrity check dry-run failed")
            record.error = str(exc)

        record.finished_at = datetime.now(UTC)

        summary = f"{record.issues_found} issues (dry-run)"
        finished = finish_trace(
            intent="integrity_check_dry",
            final_response=summary,
            error=record.error,
        )
        if finished is not None:
            record.trace_id = finished.trace_id
            if self._database_name:
                self._store.append_trace(finished)

        return record

    def _group_memories_by_date(self) -> dict[str, list[dict[str, Any]]]:
        """Group all memories by date, returning only dates with 2+ memories."""
        all_memories = self._store.list_memories(limit=None)
        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in all_memories:
            by_date[str(m.date)].append({
                "id": m.id,
                "title": m.title,
                "description": m.description,
                "people": m.people,
            })
        return {d: mems for d, mems in by_date.items() if len(mems) >= 2}

    def _format_memories_block(self, date_str: str, memories: list[dict[str, Any]]) -> str:
        """Format a group of memories for LLM consumption."""
        lines = [f"Date: {date_str}", ""]
        for m in memories:
            lines.append(f"- id: {m['id']}")
            lines.append(f"  title: {m['title']}")
            lines.append(f"  description: {m['description']}")
            lines.append(f"  people: {m['people']}")
            lines.append("")
        return "\n".join(lines)

    async def _llm_check_memory_issues(
        self, *, dry_run: bool = False,
    ) -> list[IntegrityFinding]:
        """Use LLM to find duplicate and contradictory memories (one call per date group)."""
        assert self._llm is not None
        findings: list[IntegrityFinding] = []
        grouped = self._group_memories_by_date()

        for date_str, memories in grouped.items():
            block = self._format_memories_block(date_str, memories)
            messages = check_memory_issues(block)
            try:
                resp = await self._llm.chat(
                    messages,
                    model=self._model,
                    temperature=0.1,
                )
                record_step(LLMCallStep(
                    method="chat",
                    model=self._model,
                    temperature=0.1,
                    messages=messages,
                    response_content=resp.content,
                    usage=resp.usage,
                ))
                parsed = yaml.safe_load(resp.content or "{}") or {}
                if not isinstance(parsed, dict):
                    continue

                # Process duplicates
                duplicates = parsed.get("duplicates") or []
                if isinstance(duplicates, list):
                    for item in duplicates:
                        if not isinstance(item, dict):
                            continue
                        finding = IntegrityFinding(
                            category="semantic_duplicate",
                            severity="warning",
                            message=(
                                f"Possible duplicate on {date_str}: "
                                f"{item.get('id_a', '?')} and "
                                f"{item.get('id_b', '?')} — "
                                f"{item.get('reason', 'similar content')}"
                            ),
                            action="logged",
                            details={
                                "id_a": str(item.get("id_a", "")),
                                "id_b": str(item.get("id_b", "")),
                                "date": date_str,
                                "reason": str(item.get("reason", "")),
                            },
                        )
                        if dry_run:
                            finding.action = "would_create_question"
                        else:
                            created = self._maybe_create_question_from_finding(finding)
                            if created:
                                finding.action = "question_created"
                        findings.append(finding)

                # Process contradictions
                contradictions = parsed.get("contradictions") or []
                if isinstance(contradictions, list):
                    for item in contradictions:
                        if not isinstance(item, dict):
                            continue
                        finding = IntegrityFinding(
                            category="contradiction",
                            severity="warning",
                            message=(
                                f"Contradiction on {date_str}: "
                                f"{item.get('id_a', '?')} vs "
                                f"{item.get('id_b', '?')} — "
                                f"{item.get('contradiction', 'conflicting info')}"
                            ),
                            action="logged",
                            details={
                                "id_a": str(item.get("id_a", "")),
                                "id_b": str(item.get("id_b", "")),
                                "date": date_str,
                                "contradiction": str(
                                    item.get("contradiction", "")
                                ),
                            },
                        )
                        if dry_run:
                            finding.action = "would_create_question"
                        else:
                            created = self._maybe_create_question_from_finding(finding)
                            if created:
                                finding.action = "question_created"
                        findings.append(finding)
            except Exception:
                logger.exception("LLM memory issues check failed for %s", date_str)

        return findings

    def _auto_fix(self, issue: object) -> str | None:
        """Auto-fix safe issues. Returns explanation string if fixed, None otherwise."""
        from elephant.audit import AuditIssue

        assert isinstance(issue, AuditIssue)

        if issue.category == "stale_thread":
            return self._fix_stale_thread(issue)
        elif issue.category == "empty_person_id":
            return self._fix_empty_person_id(issue)
        return None

    def _fix_stale_thread(self, issue: object) -> str | None:
        """Archive a stale thread. Returns explanation if fixed."""
        from elephant.audit import AuditIssue

        assert isinstance(issue, AuditIssue)
        person_id = issue.details.get("person_id", "")
        topic = issue.details.get("topic", "")
        if not person_id or not topic:
            return None

        person = self._store.read_person(person_id)
        if person is None:
            return None

        # Move stale thread from current to archived
        thread_to_archive = None
        remaining: list[object] = []
        for t in person.current_threads:
            if t.topic == topic:
                thread_to_archive = t
            else:
                remaining.append(t)

        if thread_to_archive is None:
            return None

        from elephant.data.models import CurrentThread

        days_ago = (date.today() - thread_to_archive.last_mentioned_date).days
        person.current_threads = [
            t for t in person.current_threads if isinstance(t, CurrentThread) and t.topic != topic
        ]
        person.archived_threads.append(thread_to_archive)
        self._store.write_person(person)
        return (
            f"Archived thread '{topic}' for {person.display_name}"
            f" — last mentioned {days_ago} days ago"
        )

    def _fix_empty_person_id(self, issue: object) -> str | None:
        """Derive person_id from display_name, rename the file. Returns explanation if fixed."""
        from elephant.audit import AuditIssue

        assert isinstance(issue, AuditIssue)
        display_name = issue.details.get("display_name", "")
        if not display_name:
            return None

        # Derive person_id from display_name (lowercase, spaces → underscores)
        new_id = display_name.lower().replace(" ", "_")
        new_id = "".join(c for c in new_id if c.isalnum() or c == "_")
        if not new_id:
            return None

        # Check for conflict
        if self._store.read_person(new_id) is not None:
            return None

        # Read the broken person (empty person_id → file is ".yaml")
        person = self._store.read_person("")
        if person is None:
            return None

        old_path = os.path.join(self._store.data_dir, "people", ".yaml")
        person.person_id = new_id
        self._store.write_person(person)
        if os.path.exists(old_path):
            os.remove(old_path)
        return f"Derived person_id '{new_id}' from display_name '{display_name}'"

    def _describe_fix(self, issue: object) -> str | None:
        """Describe what an auto-fix *would* do, without applying it."""
        from elephant.audit import AuditIssue

        assert isinstance(issue, AuditIssue)
        if issue.category == "stale_thread":
            person_id = issue.details.get("person_id", "")
            topic = issue.details.get("topic", "")
            person = self._store.read_person(person_id) if person_id else None
            if person and topic:
                thread = next(
                    (t for t in person.current_threads if t.topic == topic), None,
                )
                if thread:
                    days_ago = (date.today() - thread.last_mentioned_date).days
                    return (
                        f"Would archive thread '{topic}' for {person.display_name}"
                        f" — last mentioned {days_ago} days ago"
                    )
        elif issue.category == "empty_person_id":
            display_name = issue.details.get("display_name", "")
            if display_name:
                new_id = display_name.lower().replace(" ", "_")
                new_id = "".join(c for c in new_id if c.isalnum() or c == "_")
                if new_id:
                    return (
                        f"Would derive person_id '{new_id}'"
                        f" from display_name '{display_name}'"
                    )
        return None

    def _maybe_create_question_from_finding(self, finding: IntegrityFinding) -> bool:
        """Create a pending question from an LLM finding if one doesn't already exist."""
        template = _QUESTION_TEMPLATES.get(finding.category)
        if template is None:
            return False

        # Build sorted subject key from id_a|id_b for dedup
        id_a = finding.details.get("id_a", "")
        id_b = finding.details.get("id_b", "")
        subject = "|".join(sorted([id_a, id_b]))
        if not subject or subject == "|":
            return False

        # De-duplicate: check existing pending/asked questions
        pq_file = self._store.read_pending_questions()
        for q in pq_file.questions:
            if (
                q.subject == subject
                and q.type == finding.category
                and q.status in ("pending", "asked")
            ):
                return False

        fmt_vars: dict[str, str] = {
            "date": finding.details.get("date", "?"),
            "id_a": id_a,
            "id_b": id_b,
            "reason": finding.details.get("reason", ""),
            "contradiction": finding.details.get("contradiction", ""),
        }
        question_text = template.format(**fmt_vars)
        new_q = PendingQuestion(
            id=f"q_{uuid.uuid4().hex[:8]}",
            type=finding.category,
            subject=subject,
            question=question_text,
            status="pending",
            created_at=datetime.now(UTC),
        )
        pq_file.questions.append(new_q)
        self._store.write_pending_questions(pq_file)
        return True

    def _maybe_create_question(self, issue: object) -> bool:
        """Create a pending question if one doesn't already exist."""
        from elephant.audit import AuditIssue

        assert isinstance(issue, AuditIssue)
        template = _QUESTION_TEMPLATES.get(issue.category)
        if template is None:
            return False

        # Determine subject and format vars
        subject = ""
        fmt_vars: dict[str, str] = {}
        if issue.category == "unknown_relationship":
            subject = issue.details.get("person_id", "")
            name = issue.message.split(" has ")[0] if " has " in issue.message else subject
            fmt_vars["name"] = name
        elif issue.category == "orphan_person":
            subject = issue.details.get("name", "")
            fmt_vars["name"] = subject
        elif issue.category == "duplicate_memory":
            subject = issue.details.get("memory_ids", "")
            fmt_vars["ids"] = subject
        elif issue.category == "duplicate_person_name":
            subject = issue.details.get("display_name", "")
            fmt_vars["name"] = subject

        if not subject:
            return False

        # De-duplicate: check existing pending/asked questions
        pq_file = self._store.read_pending_questions()
        for q in pq_file.questions:
            if (
                q.subject == subject
                and q.type == issue.category
                and q.status in ("pending", "asked")
            ):
                return False

        question_text = template.format(**fmt_vars)
        new_q = PendingQuestion(
            id=f"q_{uuid.uuid4().hex[:8]}",
            type=issue.category,
            subject=subject,
            question=question_text,
            status="pending",
            created_at=datetime.now(UTC),
        )
        pq_file.questions.append(new_q)
        self._store.write_pending_questions(pq_file)
        return True
