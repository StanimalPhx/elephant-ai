"""Tests for tool executor: each handler with mocked DataStore."""

import json
import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from elephant.data.models import Memory, Person
from elephant.data.store import DataStore
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMResponse, ToolCall
from elephant.tools.executor import ToolExecutor


@pytest.fixture
def executor(data_dir):
    store = DataStore(data_dir)
    store.initialize()
    os.makedirs(store.media_dir(), exist_ok=True)
    git = MagicMock(spec=GitRepo)
    git.auto_commit = MagicMock(return_value="abc123")
    llm = AsyncMock()
    return ToolExecutor(store, git, llm, "test-model"), store, git, llm


class TestListMemories:
    async def test_empty_store(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(id="1", function_name="list_memories", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert result["count"] == 0
        assert result["memories"] == []

    async def test_with_memories(self, executor):
        ex, store, git, llm = executor
        memory = Memory(
            id="20260224_park_day",
            date=date(2026, 2, 24),
            title="Park day",
            type="daily",
            description="Went to the park",
            people=["Lily"],
            source="agent",
        )
        store.write_memory(memory)

        tc = ToolCall(id="1", function_name="list_memories", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert result["count"] == 1
        assert result["memories"][0]["title"] == "Park day"

    async def test_filter_by_people(self, executor):
        ex, store, git, llm = executor
        store.write_memory(Memory(
            id="20260224_park", date=date(2026, 2, 24), title="Park",
            type="daily", description="Park", people=["Lily"], source="agent",
        ))
        store.write_memory(Memory(
            id="20260224_work", date=date(2026, 2, 24), title="Work",
            type="daily", description="Work", people=["Dad"], source="agent",
        ))

        tc = ToolCall(
            id="1", function_name="list_memories",
            arguments=json.dumps({"people": ["Lily"]}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["count"] == 1
        assert result["memories"][0]["title"] == "Park"


class TestGetMemory:
    async def test_found(self, executor):
        ex, store, git, llm = executor
        store.write_memory(Memory(
            id="20260224_park_day", date=date(2026, 2, 24), title="Park day",
            type="daily", description="Fun", people=["Lily"], source="agent",
        ))

        tc = ToolCall(
            id="1", function_name="get_memory",
            arguments=json.dumps({"memory_id": "20260224_park_day"}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["title"] == "Park day"

    async def test_not_found(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="get_memory",
            arguments=json.dumps({"memory_id": "20260224_nope"}),
        )
        result = json.loads(await ex.execute(tc))
        assert "error" in result


class TestCreateMemory:
    async def test_creates_and_commits(self, executor):
        ex, store, git, llm = executor
        # Pre-create person so disambiguation doesn't block
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Park day",
                "date": "2026-02-24",
                "type": "daily",
                "description": "Went to the park with Lily",
                "people": ["Lily"],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert "created" in result
        assert result["title"] == "Park day"
        git.auto_commit.assert_called()

        # Verify stored
        memory = store.find_memory_by_id(result["created"])
        assert memory is not None
        assert memory.title == "Park day"


class TestCreateMemoryAttributes:
    async def test_create_with_attributes(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="nick", display_name="Nicholas", relationship=["son"]),
        )

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "First bike ride",
                "date": "2026-03-01",
                "type": "milestone",
                "description": "Nicholas rode his bike for the first time in the rain",
                "people": ["Nicholas"],
                "attributes": {"mood": "excited", "weather": "rainy", "first_time": "true"},
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert "created" in result

        memory = store.find_memory_by_id(result["created"])
        assert memory is not None
        assert memory.attributes["mood"] == "excited"
        assert memory.attributes["weather"] == "rainy"

    async def test_create_without_attributes(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Park day",
                "date": "2026-03-01",
                "type": "daily",
                "description": "Went to the park",
                "people": ["Lily"],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert "created" in result

        memory = store.find_memory_by_id(result["created"])
        assert memory is not None
        assert memory.attributes == {}


class TestUpdateMemory:
    async def test_updates_same_day_memory(self, executor):
        from datetime import date as _date

        ex, store, git, llm = executor
        today = _date.today()
        mem_id = f"{today.strftime('%Y%m%d')}_park_day"
        store.write_memory(Memory(
            id=mem_id, date=today, title="Park day",
            type="daily", description="Fun", people=["Lily"], source="agent",
        ))

        tc = ToolCall(
            id="1", function_name="update_memory",
            arguments=json.dumps({
                "memory_id": mem_id,
                "description": "Had a great time at the park!",
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] == mem_id

        memory = store.find_memory_by_id(mem_id)
        assert memory is not None
        assert memory.description == "Had a great time at the park!"

    async def test_corrects_past_memory(self, executor):
        ex, store, git, llm = executor
        store.write_memory(Memory(
            id="20260224_park_day", date=date(2026, 2, 24), title="Park day",
            type="daily", description="Fun", people=["Lily"], source="agent",
        ))

        tc = ToolCall(
            id="1", function_name="update_memory",
            arguments=json.dumps({
                "memory_id": "20260224_park_day",
                "description": "Had a great time at the park!",
                "reason": "More detail recalled",
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["corrected"] == "20260224_park_day"
        assert "description" in result["fields"]

        memory = store.find_memory_by_id("20260224_park_day")
        assert memory is not None
        # Original preserved
        assert memory.description == "Fun"
        # Correction appended
        assert len(memory.corrections) == 1
        assert memory.corrections[0].field == "description"
        assert memory.corrections[0].new_value == "Had a great time at the park!"
        assert memory.corrections[0].reason == "More detail recalled"
        # resolved_value returns corrected value
        assert memory.resolved_value("description") == "Had a great time at the park!"


class TestUpdateMemoryAttributes:
    async def test_merge_attributes_same_day(self, executor):
        from datetime import date as _date

        ex, store, git, llm = executor
        today = _date.today()
        mem_id = f"{today.strftime('%Y%m%d')}_bike_ride"
        store.write_memory(Memory(
            id=mem_id, date=today, title="Bike ride",
            type="daily", description="Rode bikes", people=["Nicholas"],
            source="agent", attributes={"mood": "happy", "weather": "sunny"},
        ))

        tc = ToolCall(
            id="1", function_name="update_memory",
            arguments=json.dumps({
                "memory_id": mem_id,
                "attributes": {"weather": "rainy", "season": "spring"},
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] == mem_id

        memory = store.find_memory_by_id(mem_id)
        assert memory is not None
        # Existing key overwritten
        assert memory.attributes["weather"] == "rainy"
        # New key added
        assert memory.attributes["season"] == "spring"
        # Missing key preserved
        assert memory.attributes["mood"] == "happy"

    async def test_merge_attributes_past_memory(self, executor):
        """Attributes on past memories are merged directly (not corrections)."""
        ex, store, git, llm = executor
        store.write_memory(Memory(
            id="20260224_bike", date=date(2026, 2, 24), title="Bike ride",
            type="daily", description="Rode bikes", people=["Nicholas"],
            source="agent", attributes={"mood": "happy"},
        ))

        tc = ToolCall(
            id="1", function_name="update_memory",
            arguments=json.dumps({
                "memory_id": "20260224_bike",
                "attributes": {"weather": "rainy"},
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["corrected"] == "20260224_bike"

        memory = store.find_memory_by_id("20260224_bike")
        assert memory is not None
        # Attributes merged directly, not via corrections
        assert memory.attributes["mood"] == "happy"
        assert memory.attributes["weather"] == "rainy"
        # No corrections for attributes
        assert len(memory.corrections) == 0


class TestDeleteMemory:
    async def test_deletes_existing(self, executor):
        ex, store, git, llm = executor
        store.write_memory(Memory(
            id="20260224_park_day", date=date(2026, 2, 24), title="Park day",
            type="daily", description="Fun", people=["Lily"], source="agent",
        ))

        tc = ToolCall(
            id="1", function_name="delete_memory",
            arguments=json.dumps({"memory_id": "20260224_park_day", "confirm": True}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["deleted"] == "20260224_park_day"
        assert store.find_memory_by_id("20260224_park_day") is None

    async def test_not_found(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="delete_memory",
            arguments=json.dumps({"memory_id": "20260224_nope"}),
        )
        result = json.loads(await ex.execute(tc))
        assert "error" in result


class TestSearchPeople:
    async def test_search_by_name(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )
        store.write_person(
            Person(person_id="theo", display_name="Theo", relationship=["friend"]),
        )

        tc = ToolCall(
            id="1", function_name="search_people",
            arguments=json.dumps({"name": "Lily"}),
        )
        result = json.loads(await ex.execute(tc))
        assert len(result["people"]) == 1
        assert result["people"][0]["display_name"] == "Lily"

    async def test_partial_match(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="search_people",
            arguments=json.dumps({"name": "li"}),
        )
        result = json.loads(await ex.execute(tc))
        assert len(result["people"]) == 1

    async def test_no_matches(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="search_people",
            arguments=json.dumps({"name": "nobody"}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["people"] == []


class TestGetPerson:
    async def test_found(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="get_person",
            arguments=json.dumps({"person_id": "lily"}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["display_name"] == "Lily"
        assert result["relationship"] == ["daughter"]

    async def test_not_found(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="get_person",
            arguments=json.dumps({"person_id": "nope"}),
        )
        result = json.loads(await ex.execute(tc))
        assert "error" in result


class TestUpdateLocations:
    async def test_updates_locations(self, executor):
        ex, store, git, llm = executor

        tc = ToolCall(
            id="1", function_name="update_locations",
            arguments=json.dumps({"locations": {"Home": "123 Main St"}}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] is True

        prefs = store.read_preferences()
        assert prefs.locations["Home"] == "123 Main St"


class TestAddNote:
    async def test_adds_note(self, executor):
        ex, store, git, llm = executor

        tc = ToolCall(
            id="1", function_name="add_note",
            arguments=json.dumps({"note": "Dad loves coffee"}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] is True

        prefs = store.read_preferences()
        assert "Dad loves coffee" in prefs.notes


class TestListPeople:
    async def test_empty(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(id="1", function_name="list_people", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert result["people"] == []

    async def test_with_people(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(id="1", function_name="list_people", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert len(result["people"]) == 1
        assert result["people"][0]["display_name"] == "Lily"
        assert result["people"][0]["groups"] == []


class TestUpdatePerson:
    async def test_updates_existing(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="update_person",
            arguments=json.dumps({
                "person_id": "lily",
                "groups": ["close-friends"],
                "birthday": "2023-01-10",
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] == "lily"

        person = store.read_person("lily")
        assert person is not None
        assert person.groups == ["close-friends"]
        assert person.birthday is not None

    async def test_updates_current_threads(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="theo", display_name="Theo", relationship=["friend"]),
        )

        tc = ToolCall(
            id="1", function_name="update_person",
            arguments=json.dumps({
                "person_id": "theo",
                "current_threads": [
                    {
                        "topic": "Job search",
                        "latest_update": "Applied to Google",
                        "last_mentioned_date": "2026-02-20",
                    },
                ],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] == "theo"

        person = store.read_person("theo")
        assert person is not None
        assert len(person.current_threads) == 1
        assert person.current_threads[0].topic == "Job search"

    async def test_merge_attributes(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(
                person_id="nick", display_name="Nicholas", relationship=["son"],
                attributes={"hobby": "soccer", "school": "Lincoln Elementary"},
            ),
        )

        tc = ToolCall(
            id="1", function_name="update_person",
            arguments=json.dumps({
                "person_id": "nick",
                "attributes": {"hobby": "basketball", "allergy": "peanuts"},
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["updated"] == "nick"

        person = store.read_person("nick")
        assert person is not None
        # Existing key overwritten
        assert person.attributes["hobby"] == "basketball"
        # New key added
        assert person.attributes["allergy"] == "peanuts"
        # Missing key preserved
        assert person.attributes["school"] == "Lincoln Elementary"

    async def test_not_found(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="update_person",
            arguments=json.dumps({"person_id": "nope"}),
        )
        result = json.loads(await ex.execute(tc))
        assert "error" in result


class TestComputedLastContact:
    async def test_last_contact_computed_from_memories(self, executor):
        """last_contact is derived from memories, not stored on Person."""
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Park day",
                "date": "2026-02-24",
                "type": "daily",
                "description": "Park with Lily",
                "people": ["Lily"],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert "created" in result

        # Verify computed via store
        last = store.get_latest_memory_date_for_person("Lily")
        assert last is not None
        assert str(last) == "2026-02-24"

    async def test_list_people_includes_computed_last_contact(self, executor):
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )
        from elephant.data.models import Memory
        store.write_memory(Memory(
            id="20260224_park", date=date(2026, 2, 24), title="Park",
            type="daily", description="Park", people=["Lily"], source="agent",
        ))

        tc = ToolCall(id="1", function_name="list_people", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert result["people"][0]["last_contact"] == "2026-02-24"


class TestEntityDisambiguation:
    async def test_unknown_people_returns_warning(self, executor):
        """Unknown person without auto_create_people returns a warning."""
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Dinner with Yan",
                "date": "2026-03-01",
                "type": "daily",
                "description": "Had dinner with Yan",
                "people": ["Yan"],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["warning"] == "unknown_people"
        assert "Yan" in result["unknown_names"]

    async def test_near_match_suggestions(self, executor):
        """Unknown name similar to existing person returns suggestions."""
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )
        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Park day",
                "date": "2026-03-01",
                "type": "daily",
                "description": "Park with Lil",
                "people": ["Lil"],
            }),
        )
        result = json.loads(await ex.execute(tc))
        assert result["warning"] == "unknown_people"
        assert "Lily" in result["suggestions"]["Lil"]


class TestAutoCreatePerson:
    async def test_auto_creates_person_on_memory(self, executor):
        """New person mentioned in memory gets a Person file created when confirmed."""
        ex, store, git, llm = executor
        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Dinner with Yan",
                "date": "2026-03-01",
                "type": "daily",
                "description": "Had dinner with Yan",
                "people": ["Yan"],
                "auto_create_people": True,
            }),
        )
        await ex.execute(tc)

        person = store.read_person("yan")
        assert person is not None
        assert person.display_name == "Yan"
        assert person.relationship == ["unknown"]

    async def test_no_duplicate_when_person_exists(self, executor):
        """Existing person is not duplicated."""
        ex, store, git, llm = executor
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Park day",
                "date": "2026-03-04",
                "type": "daily",
                "description": "Park",
                "people": ["Lily"],
            }),
        )
        await ex.execute(tc)

        people = store.read_all_people()
        assert len(people) == 1
        assert people[0].person_id == "lily"

    async def test_auto_create_multiple_people(self, executor):
        """Multiple new people in one memory all get created."""
        ex, store, git, llm = executor

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Party",
                "date": "2026-03-05",
                "type": "celebration",
                "description": "Birthday party",
                "people": ["Ana", "Bruno"],
                "auto_create_people": True,
            }),
        )
        await ex.execute(tc)

        ana = store.read_person("ana")
        bruno = store.read_person("bruno")
        assert ana is not None
        assert bruno is not None

    async def test_auto_create_commits_for_new_person(self, executor):
        """Auto-created person triggers a git commit."""
        ex, store, git, llm = executor

        tc = ToolCall(
            id="1", function_name="create_memory",
            arguments=json.dumps({
                "title": "Coffee with Marcos",
                "date": "2026-03-06",
                "type": "daily",
                "description": "Coffee",
                "people": ["Marcos"],
                "auto_create_people": True,
            }),
        )
        await ex.execute(tc)

        # First call is for the memory, second for the auto-created person
        assert git.auto_commit.call_count == 2
        people_call = git.auto_commit.call_args_list[1]
        assert people_call.args[0] == "people"


class TestDescribeAttachment:
    async def test_image_returns_description(self, executor):
        ex, store, git, llm = executor
        # Create a fake image file inside the media directory
        media = store.media_dir()
        img = os.path.join(media, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0fake-jpeg-data")

        llm.chat = AsyncMock(return_value=LLMResponse(
            content="A family photo at the park.", model="m", usage={},
        ))

        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": img}),
        )
        result = json.loads(await ex.execute(tc))
        assert "description" in result
        assert "park" in result["description"].lower()
        llm.chat.assert_called_once()

    async def test_text_file_returns_contents(self, executor):
        ex, store, git, llm = executor
        media = store.media_dir()
        doc = os.path.join(media, "notes.txt")
        with open(doc, "w") as f:
            f.write("Hello, this is a test document.")

        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": doc}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["file_type"] == "txt"
        assert "test document" in result["contents"]

    async def test_missing_file_returns_error(self, executor):
        ex, store, git, llm = executor
        media = store.media_dir()
        missing = os.path.join(media, "nonexistent.jpg")
        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": missing}),
        )
        result = json.loads(await ex.execute(tc))
        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_large_file_truncated(self, executor):
        ex, store, git, llm = executor
        media = store.media_dir()
        doc = os.path.join(media, "big.csv")
        with open(doc, "w") as f:
            f.write("x" * 200_000)

        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": doc}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["file_type"] == "csv"
        assert len(result["contents"]) == 100_000

    async def test_json_file_returns_contents(self, executor):
        ex, store, git, llm = executor
        media = store.media_dir()
        doc = os.path.join(media, "data.json")
        with open(doc, "w") as f:
            f.write('{"key": "value"}')

        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": doc}),
        )
        result = json.loads(await ex.execute(tc))
        assert result["file_type"] == "json"
        assert '"key"' in result["contents"]

    async def test_png_uses_vision(self, executor):
        ex, store, git, llm = executor
        media = store.media_dir()
        img = os.path.join(media, "screenshot.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake-png-data")

        llm.chat = AsyncMock(return_value=LLMResponse(
            content="A screenshot of a chat.", model="m", usage={},
        ))

        tc = ToolCall(
            id="1", function_name="describe_attachment",
            arguments=json.dumps({"file_path": img}),
        )
        result = json.loads(await ex.execute(tc))
        assert "description" in result
        llm.chat.assert_called_once()


class TestUnknownTool:
    async def test_returns_error(self, executor):
        ex, store, git, llm = executor
        tc = ToolCall(id="1", function_name="does_not_exist", arguments="{}")
        result = json.loads(await ex.execute(tc))
        assert "error" in result
        assert "Unknown tool" in result["error"]
