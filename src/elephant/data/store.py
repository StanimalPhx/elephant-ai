"""DataStore: directory init, YAML read/write, schema deployment."""

from __future__ import annotations

import os
from datetime import date
from datetime import date as _date
from typing import Any

import yaml

from elephant.atomic import atomic_write
from elephant.data.models import (
    AuthorizedChatsFile,
    ChatHistoryEntry,
    ChatHistoryFile,
    DailyMetrics,
    DigestState,
    Group,
    Memory,
    MetricsFile,
    NudgeStateFile,
    PendingQuestionsFile,
    Person,
    PhotoEntry,
    PreferencesFile,
    RawMessage,
    VideoEntry,
)
from elephant.data.schemas import DIR_SCHEMAS, SINGLE_FILE_SCHEMAS
from elephant.tracing import Trace

# Directories to create under data_dir
_DIRS = [
    "memories",
    "photo_index",
    "video_index",
    "people",
    "groups",
    "faces",
    "logs",
    "media",
]


class DataStore:
    """Centralized YAML I/O with path helpers and schema deployment."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = os.path.abspath(data_dir)

    def initialize(self) -> None:
        """Create directory structure and deploy schemas. Idempotent."""
        for d in _DIRS:
            os.makedirs(os.path.join(self.data_dir, d), exist_ok=True)

        # Deploy directory schemas (e.g. memories/_schema.yaml)
        for rel_path, content in DIR_SCHEMAS.items():
            full_path = os.path.join(self.data_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            atomic_write(full_path, content)

        # Deploy single-file schemas (only if file doesn't exist yet)
        for rel_path, content in SINGLE_FILE_SCHEMAS.items():
            full_path = os.path.join(self.data_dir, rel_path)
            if not os.path.exists(full_path):
                atomic_write(full_path, content)

        # Deploy .gitignore for the data repo
        gitignore_path = os.path.join(self.data_dir, ".gitignore")
        if not os.path.exists(gitignore_path):
            atomic_write(
                gitignore_path,
                "faces/\nlogs/\nchat_history.yaml\n*.jpg\n*.jpeg\n*.png\n*.mp4\n*.mov\n",
            )


    def media_dir(self) -> str:
        """Return the path to the media directory."""
        return os.path.join(self.data_dir, "media")

    # --- Path helpers ---

    def _memory_path(self, memory_date: date, slug: str) -> str:
        return os.path.join(
            self.data_dir,
            "memories",
            memory_date.strftime("%Y"),
            memory_date.strftime("%m"),
            f"{memory_date.strftime('%Y%m%d')}_{slug}.yaml",
        )

    def _photo_index_path(self, index_date: date) -> str:
        return os.path.join(
            self.data_dir,
            "photo_index",
            index_date.strftime("%Y"),
            index_date.strftime("%m"),
            f"{index_date.isoformat()}.yaml",
        )

    def _video_index_path(self, index_date: date) -> str:
        return os.path.join(
            self.data_dir,
            "video_index",
            index_date.strftime("%Y"),
            index_date.strftime("%m"),
            f"{index_date.isoformat()}.yaml",
        )

    def _single_file_path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    # --- YAML I/O helpers ---

    @staticmethod
    def _read_yaml(path: str) -> Any:
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _write_yaml(path: str, data: Any) -> None:
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        atomic_write(path, content)

    # --- Memories ---

    def write_memory(self, memory: Memory) -> str:
        """Write a memory to its date-based path. Returns the file path."""
        slug = memory.id.split("_", 1)[1] if "_" in memory.id else memory.id
        path = self._memory_path(memory.date, slug)
        self._write_yaml(path, memory.model_dump(mode="json", exclude_none=True))
        return path

    def read_memory(self, path: str) -> Memory:
        """Read a memory from a YAML file."""
        data = self._read_yaml(path)
        return Memory.model_validate(data)

    # --- Photo Index ---

    def write_photo_index(self, index_date: date, entries: list[PhotoEntry]) -> str:
        """Write photo index entries for a date. Returns the file path."""
        path = self._photo_index_path(index_date)
        data = [e.model_dump(mode="json", exclude_none=True) for e in entries]
        self._write_yaml(path, data)
        return path

    def read_photo_index(self, index_date: date) -> list[PhotoEntry]:
        """Read photo index entries for a date."""
        path = self._photo_index_path(index_date)
        if not os.path.exists(path):
            return []
        data = self._read_yaml(path)
        if not data:
            return []
        return [PhotoEntry.model_validate(item) for item in data]

    # --- Video Index ---

    def write_video_index(self, index_date: date, entries: list[VideoEntry]) -> str:
        """Write video index entries for a date. Returns the file path."""
        path = self._video_index_path(index_date)
        data = [e.model_dump(mode="json", exclude_none=True) for e in entries]
        self._write_yaml(path, data)
        return path

    def read_video_index(self, index_date: date) -> list[VideoEntry]:
        """Read video index entries for a date."""
        path = self._video_index_path(index_date)
        if not os.path.exists(path):
            return []
        data = self._read_yaml(path)
        if not data:
            return []
        return [VideoEntry.model_validate(item) for item in data]

    # --- Single-file stores (preserve _schema block) ---

    def _read_single_file(self, name: str) -> dict[str, Any]:
        path = self._single_file_path(name)
        if not os.path.exists(path):
            return {}
        data = self._read_yaml(path)
        return data if isinstance(data, dict) else {}

    def _write_single_file(self, name: str, data: dict[str, Any]) -> None:
        """Write a single-file store, preserving the _schema block."""
        path = self._single_file_path(name)
        existing = self._read_single_file(name)
        schema = existing.get("_schema")
        if schema is not None:
            data = {"_schema": schema, **{k: v for k, v in data.items() if k != "_schema"}}
        self._write_yaml(path, data)

    # --- People (directory-based) ---

    def _person_path(self, person_id: str) -> str:
        return os.path.join(self.data_dir, "people", f"{person_id}.yaml")

    def read_person(self, person_id: str) -> Person | None:
        """Read a single person by ID. Returns None if not found."""
        path = self._person_path(person_id)
        if not os.path.exists(path):
            return None
        data = self._read_yaml(path)
        if not data:
            return None
        return Person.model_validate(data)

    def write_person(self, person: Person) -> str:
        """Write a person to their file. Returns the file path."""
        path = self._person_path(person.person_id)
        self._write_yaml(path, person.model_dump(mode="json", exclude_none=True))
        return path

    def read_all_people(self) -> list[Person]:
        """Read all people from the people directory."""
        people_dir = os.path.join(self.data_dir, "people")
        if not os.path.isdir(people_dir):
            return []
        results: list[Person] = []
        for fname in sorted(os.listdir(people_dir)):
            if not fname.endswith(".yaml") or fname.startswith("_"):
                continue
            path = os.path.join(people_dir, fname)
            try:
                data = self._read_yaml(path)
                if data:
                    results.append(Person.model_validate(data))
            except Exception:
                continue
        return results

    def delete_person(self, person_id: str) -> bool:
        """Delete a person by ID. Returns True if deleted."""
        path = self._person_path(person_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    # --- Groups (directory-based) ---

    def _group_path(self, group_id: str) -> str:
        return os.path.join(self.data_dir, "groups", f"{group_id}.yaml")

    def read_group(self, group_id: str) -> Group | None:
        """Read a single group by ID. Returns None if not found."""
        path = self._group_path(group_id)
        if not os.path.exists(path):
            return None
        data = self._read_yaml(path)
        if not data:
            return None
        return Group.model_validate(data)

    def write_group(self, group: Group) -> str:
        """Write a group to its file. Returns the file path."""
        path = self._group_path(group.group_id)
        self._write_yaml(path, group.model_dump(mode="json", exclude_none=True))
        return path

    def read_all_groups(self) -> list[Group]:
        """Read all groups from the groups directory."""
        groups_dir = os.path.join(self.data_dir, "groups")
        if not os.path.isdir(groups_dir):
            return []
        results: list[Group] = []
        for fname in sorted(os.listdir(groups_dir)):
            if not fname.endswith(".yaml") or fname.startswith("_"):
                continue
            path = os.path.join(groups_dir, fname)
            try:
                data = self._read_yaml(path)
                if data:
                    results.append(Group.model_validate(data))
            except Exception:
                continue
        return results

    def delete_group(self, group_id: str) -> bool:
        """Delete a group by ID. Returns True if deleted."""
        path = self._group_path(group_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    # --- Preferences ---

    def read_preferences(self) -> PreferencesFile:
        raw = self._read_single_file("preferences.yaml")
        return PreferencesFile.model_validate(
            {
                "nostalgia_weights": raw.get("nostalgia_weights", {}),
                "tone_preference": raw.get("tone_preference", {}),
                "locations": raw.get("locations", {}),
                "notes": raw.get("notes", []),
            }
        )

    def write_preferences(self, prefs: PreferencesFile) -> None:
        self._write_single_file(
            "preferences.yaml",
            prefs.model_dump(mode="json"),
        )

    # --- Pending Questions ---

    def read_pending_questions(self) -> PendingQuestionsFile:
        raw = self._read_single_file("pending_questions.yaml")
        return PendingQuestionsFile.model_validate({"questions": raw.get("questions", [])})

    def write_pending_questions(self, pq: PendingQuestionsFile) -> None:
        self._write_single_file(
            "pending_questions.yaml",
            pq.model_dump(mode="json", exclude_none=True),
        )

    # --- Digest State ---

    def read_digest_state(self) -> DigestState:
        raw = self._read_single_file("digest_state.yaml")
        return DigestState.model_validate(
            {
                "last_digest_sent_at": raw.get("last_digest_sent_at"),
                "last_digest_memory_ids": raw.get("last_digest_memory_ids", []),
                "last_digest_message_id": raw.get("last_digest_message_id"),
                "last_digest_text": raw.get("last_digest_text"),
            }
        )

    def write_digest_state(self, state: DigestState) -> None:
        self._write_single_file(
            "digest_state.yaml",
            state.model_dump(mode="json", exclude_none=True),
        )

    # --- Nudge State ---

    def read_nudge_state(self) -> NudgeStateFile:
        raw = self._read_single_file("nudge_state.yaml")
        return NudgeStateFile.model_validate({"records": raw.get("records", [])})

    def write_nudge_state(self, state: NudgeStateFile) -> None:
        self._write_single_file(
            "nudge_state.yaml",
            state.model_dump(mode="json"),
        )

    # --- Metrics ---

    def read_metrics(self) -> MetricsFile:
        raw = self._read_single_file("metrics.yaml")
        return MetricsFile.model_validate({"days": raw.get("days", [])})

    def write_metrics(self, metrics: MetricsFile) -> None:
        self._write_single_file(
            "metrics.yaml",
            metrics.model_dump(mode="json"),
        )

    def increment_metric(self, metric_name: str, count: int = 1) -> None:
        """Find or create today's DailyMetrics entry and increment the named field."""
        today = _date.today()
        metrics = self.read_metrics()
        entry: DailyMetrics | None = None
        for d in metrics.days:
            if d.date == today:
                entry = d
                break
        if entry is None:
            entry = DailyMetrics(date=today)
            metrics.days.append(entry)
        current = getattr(entry, metric_name, 0)
        setattr(entry, metric_name, current + count)
        self.write_metrics(metrics)

    # --- Authorized Chats ---

    def read_authorized_chats(self) -> AuthorizedChatsFile:
        raw = self._read_single_file("authorized_chats.yaml")
        return AuthorizedChatsFile.model_validate({"chats": raw.get("chats", [])})

    def write_authorized_chats(self, ac: AuthorizedChatsFile) -> None:
        self._write_single_file(
            "authorized_chats.yaml",
            ac.model_dump(mode="json", exclude_none=True),
        )

    # --- Chat History ---

    def read_chat_history(self) -> ChatHistoryFile:
        raw = self._read_single_file("chat_history.yaml")
        return ChatHistoryFile.model_validate({"entries": raw.get("entries", [])})

    def write_chat_history(self, history: ChatHistoryFile) -> None:
        self._write_single_file(
            "chat_history.yaml",
            history.model_dump(mode="json", exclude_none=True),
        )

    def append_chat_history(
        self,
        user_content: str,
        assistant_content: str,
        max_entries: int = 1000,
    ) -> None:
        """Append a user+assistant exchange to history, trimming old entries."""
        from datetime import UTC, datetime

        history = self.read_chat_history()
        now = datetime.now(UTC)
        history.entries.append(
            ChatHistoryEntry(role="user", content=user_content, timestamp=now)
        )
        history.entries.append(
            ChatHistoryEntry(role="assistant", content=assistant_content, timestamp=now)
        )
        if len(history.entries) > max_entries:
            history.entries = history.entries[-max_entries:]
        self.write_chat_history(history)

    # --- Raw Messages (JSONL) ---

    def _raw_messages_jsonl_path(self) -> str:
        return os.path.join(self.data_dir, "raw_messages.jsonl")

    def _raw_messages_yaml_path(self) -> str:
        return os.path.join(self.data_dir, "raw_messages.yaml")

    def read_raw_messages(self) -> list[RawMessage]:
        """Read all raw messages from the JSONL file, skipping malformed lines."""
        path = self._raw_messages_jsonl_path()
        self._migrate_raw_messages_yaml_to_jsonl()
        if not os.path.exists(path):
            return []
        import json as _json

        results: list[RawMessage] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    results.append(RawMessage.model_validate(data))
                except Exception:
                    continue
        return results

    def write_raw_messages(self, messages: list[RawMessage]) -> None:
        """Full rewrite of the JSONL file (for migration / bulk operations)."""
        path = self._raw_messages_jsonl_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")

    def append_raw_message(self, message: RawMessage) -> None:
        """Append a single raw message — O(1) append-only."""
        self._migrate_raw_messages_yaml_to_jsonl()
        path = self._raw_messages_jsonl_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(message.model_dump_json() + "\n")

    def _migrate_raw_messages_yaml_to_jsonl(self) -> None:
        """One-time migration from raw_messages.yaml to raw_messages.jsonl."""
        yaml_path = self._raw_messages_yaml_path()
        jsonl_path = self._raw_messages_jsonl_path()
        if not os.path.exists(yaml_path) or os.path.exists(jsonl_path):
            return
        try:
            data = self._read_yaml(yaml_path)
            if not isinstance(data, dict):
                return
            raw_list = data.get("messages", [])
            if not raw_list:
                # Empty YAML file — just rename to .bak
                os.rename(yaml_path, yaml_path + ".bak")
                return
            messages = [RawMessage.model_validate(m) for m in raw_list]
            self.write_raw_messages(messages)
            os.rename(yaml_path, yaml_path + ".bak")
        except Exception:
            pass

    # --- Memory querying ---

    def query_memories_by_month_day(self, month: int, day: int) -> list[Memory]:
        """Find memories matching a given month and day across all years."""
        memories_dir = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(memories_dir):
            return []

        month_str = f"{month:02d}"
        day_str = f"{day:02d}"
        results: list[Memory] = []

        for year_name in sorted(os.listdir(memories_dir)):
            year_dir = os.path.join(memories_dir, year_name)
            if not os.path.isdir(year_dir) or year_name.startswith("_"):
                continue
            month_dir = os.path.join(year_dir, month_str)
            if not os.path.isdir(month_dir):
                continue
            for fname in sorted(os.listdir(month_dir)):
                if not fname.endswith(".yaml") or fname.startswith("_"):
                    continue
                # Filename format: YYYYMMDD_slug.yaml — check day digits [6:8]
                if len(fname) >= 8 and fname[6:8] == day_str:
                    path = os.path.join(month_dir, fname)
                    results.append(self.read_memory(path))

        return results

    # --- Memory CRUD ---

    def list_memories(
        self,
        date_from: date | None = None,
        date_to: date | None = None,
        people: list[str] | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        query: str | None = None,
        limit: int | None = 20,
    ) -> list[Memory]:
        """List memories with optional filters. Returns newest first."""
        memories_dir = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(memories_dir):
            return []

        results: list[Memory] = []

        for year_name in sorted(os.listdir(memories_dir)):
            year_dir = os.path.join(memories_dir, year_name)
            if not os.path.isdir(year_dir) or year_name.startswith("_"):
                continue
            # Skip irrelevant years
            try:
                year_int = int(year_name)
            except ValueError:
                continue
            if date_from and year_int < date_from.year:
                continue
            if date_to and year_int > date_to.year:
                continue

            for month_name in sorted(os.listdir(year_dir)):
                month_dir = os.path.join(year_dir, month_name)
                if not os.path.isdir(month_dir) or month_name.startswith("_"):
                    continue

                for fname in sorted(os.listdir(month_dir)):
                    if not fname.endswith(".yaml") or fname.startswith("_"):
                        continue
                    path = os.path.join(month_dir, fname)
                    try:
                        memory = self.read_memory(path)
                    except Exception:
                        continue

                    # Apply filters
                    if date_from and memory.date < date_from:
                        continue
                    if date_to and memory.date > date_to:
                        continue
                    if memory_type and memory.type != memory_type:
                        continue
                    if people:
                        lower_people = [p.lower() for p in people]
                        if not any(p.lower() in lower_people for p in memory.people):
                            continue
                    if tags:
                        lower_tags = [t.lower() for t in tags]
                        if not any(t.lower() in lower_tags for t in memory.tags):
                            continue
                    if query:
                        q = query.lower()
                        if (
                            q not in memory.title.lower()
                            and q not in memory.description.lower()
                        ):
                            continue

                    results.append(memory)

        # Sort newest first, apply limit
        results.sort(key=lambda e: e.date, reverse=True)
        return results[:limit] if limit is not None else results

    # --- Last-contact computation from memories ---

    def get_latest_memory_date_for_person(self, person_name: str) -> date | None:
        """Compute last contact date for a person by scanning memories."""
        memories = self.list_memories(people=[person_name], limit=1)
        return memories[0].date if memories else None

    def get_latest_memory_dates_for_people(
        self, names: list[str],
    ) -> dict[str, date | None]:
        """Compute last contact dates for multiple people in a single pass."""
        if not names:
            return {}
        all_memories = self.list_memories(limit=None)
        name_set = {n.lower() for n in names}
        result: dict[str, date | None] = {n: None for n in names}
        for memory in all_memories:
            for person_name in memory.people:
                key = person_name.lower()
                if key in name_set:
                    # Find the original name from the names list
                    for n in names:
                        if n.lower() == key:
                            existing = result[n]
                            if existing is None or memory.date > existing:
                                result[n] = memory.date
                            break
        return result

    def find_memory_by_id(self, memory_id: str) -> Memory | None:
        """Find a memory by its ID. Falls back to fuzzy slug match within the date dir."""
        # Parse date from ID (first 8 chars: YYYYMMDD)
        if len(memory_id) < 9 or memory_id[8] != "_":
            return None
        date_str = memory_id[:8]
        slug = memory_id[9:]
        try:
            memory_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except ValueError:
            return None

        # Exact match
        path = self._memory_path(memory_date, slug)
        if os.path.exists(path):
            return self.read_memory(path)

        # Fuzzy fallback: search the date directory for the best match
        date_dir = os.path.join(
            self.data_dir, "memories",
            memory_date.strftime("%Y"), memory_date.strftime("%m"),
        )
        if not os.path.isdir(date_dir):
            return None

        from difflib import SequenceMatcher

        prefix = date_str + "_"
        slug_lower = slug.lower()
        best_path: str | None = None
        best_score = 0.0
        for fname in os.listdir(date_dir):
            if not fname.startswith(prefix) or not fname.endswith(".yaml"):
                continue
            file_slug = fname[len(prefix):-5]  # strip prefix and .yaml
            # Substring check
            if slug_lower in file_slug.lower():
                return self.read_memory(os.path.join(date_dir, fname))
            score = SequenceMatcher(None, slug_lower, file_slug.lower()).ratio()
            if score > best_score:
                best_score = score
                best_path = os.path.join(date_dir, fname)

        if best_path and best_score >= 0.5:
            return self.read_memory(best_path)
        return None

    def update_memory(self, memory_id: str, updates: dict[str, Any]) -> Memory | None:
        """Update fields on an existing memory. Returns updated memory or None."""
        memory = self.find_memory_by_id(memory_id)
        if memory is None:
            return None
        updated = memory.model_copy(update=updates)
        self.write_memory(updated)
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        if len(memory_id) < 9 or memory_id[8] != "_":
            return False
        date_str = memory_id[:8]
        slug = memory_id[9:]
        try:
            memory_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except ValueError:
            return False

        path = self._memory_path(memory_date, slug)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    # --- Traces (JSONL) ---

    def _traces_jsonl_path(self) -> str:
        return os.path.join(self.data_dir, "logs", "traces.jsonl")

    def append_trace(self, trace: Trace) -> None:
        """Append a finished trace as a single JSON line."""
        path = self._traces_jsonl_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(trace.model_dump_json() + "\n")

    def read_traces(self, limit: int = 30, offset: int = 0) -> tuple[list[Trace], int]:
        """Read traces newest-first with pagination. Returns (traces, total)."""
        import json as _json

        path = self._traces_jsonl_path()
        if not os.path.exists(path):
            return [], 0

        all_lines: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_lines.append(line)

        total = len(all_lines)
        # Newest first — reverse, then slice
        all_lines.reverse()
        page = all_lines[offset : offset + limit]

        traces: list[Trace] = []
        for line in page:
            try:
                data = _json.loads(line)
                traces.append(Trace.model_validate(data))
            except Exception:
                continue
        return traces, total

    def read_trace_by_id(self, trace_id: str) -> Trace | None:
        """Find a single trace by its trace_id."""
        import json as _json

        path = self._traces_jsonl_path()
        if not os.path.exists(path):
            return None

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    if data.get("trace_id") == trace_id:
                        return Trace.model_validate(data)
                except Exception:
                    continue
        return None

