"""integrity-check subcommand: run database integrity check from the CLI."""

from __future__ import annotations

import asyncio
import sys

import aiohttp

from elephant.config import load_config
from elephant.data.models import IntegrityRunRecord
from elephant.data.store import DataStore
from elephant.flows.integrity_check import IntegrityCheckFlow
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMClient


def run_integrity_cli(
    config_path: str,
    database: str | None,
    *,
    dry_run: bool = False,
) -> None:
    """Load config, init store, run integrity check, print results."""
    config = load_config(config_path)

    if database:
        db_cfg = None
        for db in config.databases:
            if db.name == database:
                db_cfg = db
                break
        if db_cfg is None:
            names = ", ".join(db.name for db in config.databases)
            print(
                f"Error: database '{database}' not found. Available: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        db_cfg = config.databases[0]

    store = DataStore(db_cfg.data_dir)
    git = GitRepo(db_cfg.data_dir)

    async def _run() -> IntegrityRunRecord | bool:
        async with aiohttp.ClientSession() as session:
            llm = LLMClient(session, config.llm.base_url, config.llm.api_key)
            flow = IntegrityCheckFlow(
                store=store, git=git, llm=llm, model=config.llm.parsing_model,
                database_name=db_cfg.name,
            )
            if dry_run:
                return await flow.run_dry()
            await flow.run()
            return True

    record: IntegrityRunRecord | None
    result = asyncio.run(_run())
    if dry_run:
        assert isinstance(result, IntegrityRunRecord)
        record = result
    else:
        runs, _ = store.read_integrity_runs(limit=1)
        record = runs[0] if runs else None

    if record is None:
        print("Error: could not retrieve run record.", file=sys.stderr)
        sys.exit(1)

    mode = " (DRY RUN)" if dry_run else ""
    print(f"Integrity check{mode}: {record.run_id}")
    print(f"  Issues found: {record.issues_found}")
    if not dry_run:
        print(f"  Auto-fixed:   {record.auto_fixed}")
        print(f"  Questions:    {record.questions_created}")
    if record.error:
        print(f"  Error: {record.error}")

    if record.findings:
        print()
        for f in record.findings:
            severity = f.severity.upper()
            action = f.action.replace("_", " ")
            print(f"  [{severity}] {f.category} ({action}): {f.message}")

    if not record.findings:
        print("\n  No issues found.")

    sys.exit(1 if record.error else 0)
