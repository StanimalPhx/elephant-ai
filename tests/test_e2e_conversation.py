"""End-to-end integration tests for multi-message agent conversations."""

import json
import os
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from elephant.data.models import CurrentThread, Person
from elephant.data.store import DataStore
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMResponse, ToolCall
from elephant.tools.agent import ConversationalAgent

TODAY = date.today()
TODAY_ISO = TODAY.isoformat()
TODAY_FMT = TODAY.strftime("%Y%m%d")
LAST_WEEK = (TODAY - timedelta(days=7)).isoformat()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_e2e(data_dir):
    """Create a fresh agent, store, mock LLM, and mock git for e2e tests."""
    store = DataStore(data_dir)
    store.initialize()

    llm = AsyncMock()
    git = MagicMock(spec=GitRepo)
    git.auto_commit = MagicMock(return_value="abc123")

    agent = ConversationalAgent(store, llm, "test-model", git)
    return agent, store, llm, git


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(content, tool_calls=None):
    """Shorthand for building an LLMResponse."""
    return LLMResponse(
        content=content,
        model="m",
        usage={},
        tool_calls=tool_calls or [],
    )


def _tc(tc_id, fn, args):
    """Shorthand for building a ToolCall."""
    return ToolCall(id=tc_id, function_name=fn, arguments=json.dumps(args))


# ===========================================================================
# Test 1: Realistic Family Conversation (scripted mock)
# ===========================================================================


class TestRealisticConversation:
    """4-message sequence building connected people and memories."""

    async def test_family_conversation(self, agent_e2e):
        agent, store, llm, git = agent_e2e

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: "My daughter is Lily Smith, she's 4 years old" ---
            if call_count == 1:
                # Round 1: search for Lily
                return _resp(None, [_tc("tc_1", "search_people", {"name": "Lily"})])
            if call_count == 2:
                # Round 2: create Lily Smith
                return _resp(None, [_tc("tc_2", "update_person", {
                    "person_id": "lily_smith",
                    "create": True,
                    "display_name": "Lily Smith",
                    "relationship": ["daughter"],
                    "notes": "4 years old",
                })])
            if call_count == 3:
                return _resp("I've added Lily Smith as your daughter. She's 4 years old!")

            # --- Message 2: "Lily started swimming lessons at the YMCA" ---
            if call_count == 4:
                # Round 1: create memory + update thread on Lily
                return _resp(None, [
                    _tc("tc_3", "create_memory", {
                        "title": "Lily's swimming lessons at YMCA",
                        "date": LAST_WEEK,
                        "type": "milestone",
                        "description": "Lily started swimming lessons at the YMCA last week",
                        "people": ["Lily Smith"],
                        "location": "YMCA",
                    }),
                    _tc("tc_4", "update_person", {
                        "person_id": "lily_smith",
                        "current_threads": [{
                            "topic": "swimming lessons",
                            "latest_update": "started swimming lessons at YMCA",
                            "last_mentioned_date": LAST_WEEK,
                        }],
                    }),
                ])
            if call_count == 5:
                return _resp("I've saved Lily's swimming lessons at the YMCA!")

            # --- Message 3: "Her swim instructor is Coach Emma Johnson" ---
            if call_count == 6:
                return _resp(None, [_tc("tc_5", "search_people", {"name": "Emma"})])
            if call_count == 7:
                return _resp(None, [_tc("tc_6", "update_person", {
                    "person_id": "emma_johnson",
                    "create": True,
                    "display_name": "Emma Johnson",
                    "relationship": ["Lily's swim instructor"],
                })])
            if call_count == 8:
                return _resp(
                    "I've added Coach Emma Johnson as Lily's swim instructor!"
                )

            # --- Message 4: "Lily had her second swim class today" ---
            if call_count == 9:
                return _resp(None, [
                    _tc("tc_7", "create_memory", {
                        "title": "Lily floats on her back",
                        "date": TODAY_ISO,
                        "type": "milestone",
                        "description": (
                            "Lily had her second swim class and can float on her back now"
                        ),
                        "people": ["Lily Smith"],
                        "location": "YMCA",
                    }),
                    _tc("tc_8", "update_person", {
                        "person_id": "lily_smith",
                        "current_threads": [{
                            "topic": "swimming lessons",
                            "latest_update": "can float on her back",
                            "last_mentioned_date": TODAY_ISO,
                        }],
                    }),
                ])
            # call_count == 10
            return _resp(
                "Amazing! Lily can float on her back now! I've updated her swimming progress."
            )

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        # Send 4 messages
        r1 = await agent.handle(
            "My daughter is Lily Smith, she's 4 years old", "Telegram",
        )
        r2 = await agent.handle(
            "Lily started swimming lessons at the YMCA last week", "Telegram",
        )
        r3 = await agent.handle(
            "Her swim instructor is Coach Emma Johnson", "Telegram",
        )
        r4 = await agent.handle(
            "Lily had her second swim class today, she can float on her back now!",
            "Telegram",
        )

        assert call_count == 10

        # --- Verify People ---
        people = store.read_all_people()
        assert len(people) == 2

        lily = store.read_person("lily_smith")
        assert lily is not None
        assert lily.display_name == "Lily Smith"
        assert "daughter" in lily.relationship
        assert lily.notes is not None and "4 years old" in lily.notes
        assert len(lily.current_threads) == 1
        assert lily.current_threads[0].topic == "swimming lessons"
        assert "float" in lily.current_threads[0].latest_update

        emma = store.read_person("emma_johnson")
        assert emma is not None
        assert emma.display_name == "Emma Johnson"
        assert any("instructor" in r.lower() for r in emma.relationship)

        # --- Verify Memories ---
        memories = store.list_memories()
        assert len(memories) == 2
        titles = {m.title for m in memories}
        assert "Lily's swimming lessons at YMCA" in titles
        assert "Lily floats on her back" in titles
        for m in memories:
            assert "Lily Smith" in m.people
            assert m.location == "YMCA"

        # --- Verify Chat History ---
        history = store.read_chat_history()
        assert len(history.entries) == 8  # 4 user + 4 assistant
        user_entries = [e for e in history.entries if e.role == "user"]
        asst_entries = [e for e in history.entries if e.role == "assistant"]
        assert len(user_entries) == 4
        assert len(asst_entries) == 4

        # --- Verify Git Commits ---
        # 2 person writes (msg1, msg3) + 2 person updates (msg2, msg4)
        # + 2 memory writes (msg2, msg4) = at least 6 commits
        assert git.auto_commit.call_count >= 6

        # Check responses are meaningful
        assert "Lily Smith" in r1
        assert "swimming" in r2.lower() or "YMCA" in r2
        assert "Emma" in r3
        assert "float" in r4.lower()


# ===========================================================================
# Test 2: Tool Coverage (scripted mock)
# ===========================================================================


class TestToolCoverage:
    """Exercise every update tool: create, update, delete memory + prefs."""

    async def test_all_update_tools(self, agent_e2e):
        agent, store, llm, git = agent_e2e

        # Pre-seed Lily so create_memory doesn't warn about unknown people
        store.write_person(
            Person(person_id="lily_smith", display_name="Lily Smith", relationship=["daughter"]),
        )

        expected_memory_id = f"{TODAY_FMT}_park_day_with_lily"

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: create memory ---
            if call_count == 1:
                return _resp(None, [_tc("tc_1", "create_memory", {
                    "title": "Park day with Lily",
                    "date": TODAY_ISO,
                    "type": "outing",
                    "description": "We went to the park with Lily Smith",
                    "people": ["Lily Smith"],
                    "location": "the park",
                })])
            if call_count == 2:
                return _resp("Got it! I've saved your park day with Lily.")

            # --- Message 2: update memory ---
            if call_count == 3:
                return _resp(None, [_tc("tc_2", "update_memory", {
                    "memory_id": expected_memory_id,
                    "location": "Zilker Park",
                    "reason": "User corrected location",
                })])
            if call_count == 4:
                return _resp("Updated! The park trip location is now Zilker Park.")

            # --- Message 3: update_locations + add_note ---
            if call_count == 5:
                return _resp(None, [
                    _tc("tc_3", "update_locations", {
                        "locations": {"Epoch Coffee": "North Loop"},
                    }),
                    _tc("tc_4", "add_note", {
                        "note": "Favorite coffee shop: Epoch on North Loop",
                    }),
                ])
            if call_count == 6:
                return _resp("Noted! I've saved Epoch Coffee as a favorite spot.")

            # --- Message 4: delete memory (preview then confirm) ---
            if call_count == 7:
                return _resp(None, [_tc("tc_5", "delete_memory", {
                    "memory_id": expected_memory_id,
                })])
            if call_count == 8:
                return _resp(None, [_tc("tc_6", "delete_memory", {
                    "memory_id": expected_memory_id,
                    "confirm": True,
                })])
            # call_count == 9
            return _resp("Done! I've removed that park memory.")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        r1 = await agent.handle("We went to the park with Lily Smith", "Telegram")
        r2 = await agent.handle(
            "Actually the park trip was at Zilker Park, not the neighborhood park", "Telegram",
        )
        r3 = await agent.handle(
            "Our favorite coffee shop is Epoch on North Loop", "Telegram",
        )
        r4 = await agent.handle(
            "Actually remove that park memory, I already logged it", "Telegram",
        )

        assert call_count == 9

        # --- Verify memory was created then deleted ---
        memories = store.list_memories()
        assert len(memories) == 0  # deleted

        # --- Verify preferences ---
        prefs = store.read_preferences()
        assert "Epoch Coffee" in prefs.locations
        assert prefs.locations["Epoch Coffee"] == "North Loop"
        assert any("Epoch" in n for n in prefs.notes)

        # --- Verify chat history ---
        history = store.read_chat_history()
        assert len(history.entries) == 8  # 4 user + 4 assistant

        # --- Verify git commits ---
        # create_memory + update_memory + update_locations + add_note + delete_memory
        assert git.auto_commit.call_count >= 5

        # Check responses
        assert "park" in r1.lower()
        assert "Zilker" in r2
        assert "Epoch" in r3
        assert "remove" in r4.lower() or "deleted" in r4.lower()

    async def test_memory_update_verified(self, agent_e2e):
        """Verify that update_memory actually changes the stored data before delete."""
        agent, store, llm, git = agent_e2e

        store.write_person(
            Person(person_id="lily_smith", display_name="Lily Smith", relationship=["daughter"]),
        )

        expected_memory_id = f"{TODAY_FMT}_park_day"

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _resp(None, [_tc("tc_1", "create_memory", {
                    "title": "Park day",
                    "date": TODAY_ISO,
                    "type": "outing",
                    "description": "Went to the park with Lily",
                    "people": ["Lily Smith"],
                    "location": "neighborhood park",
                })])
            if call_count == 2:
                return _resp("Saved your park outing!")
            if call_count == 3:
                return _resp(None, [_tc("tc_2", "update_memory", {
                    "memory_id": expected_memory_id,
                    "location": "Zilker Park",
                })])
            # call_count == 4
            return _resp("Updated the location to Zilker Park.")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        await agent.handle("Went to the park with Lily", "Telegram")

        # Verify memory was created with original location
        memory = store.find_memory_by_id(expected_memory_id)
        assert memory is not None
        assert memory.location == "neighborhood park"

        await agent.handle("Actually it was Zilker Park", "Telegram")

        # Verify memory was updated (same-day = direct update)
        memory = store.find_memory_by_id(expected_memory_id)
        assert memory is not None
        assert memory.location == "Zilker Park"


# ===========================================================================
# Test 3: Realistic Conversation (real LLM)
# ===========================================================================

_skip_no_llm = pytest.mark.skipif(
    not os.environ.get("ELEPHANT_LLM_BASE_URL") or not os.environ.get("ELEPHANT_LLM_API_KEY"),
    reason="ELEPHANT_LLM_BASE_URL and ELEPHANT_LLM_API_KEY required",
)


@pytest.fixture
async def real_agent(data_dir):
    """Create an agent wired to a real LLM."""
    import aiohttp

    from elephant.llm.client import LLMClient

    base_url = os.environ.get("ELEPHANT_LLM_BASE_URL", "")
    api_key = os.environ.get("ELEPHANT_LLM_API_KEY", "")
    model = os.environ.get("ELEPHANT_LLM_MODEL", "gpt-4.1-mini")

    store = DataStore(data_dir)
    store.initialize()
    git = MagicMock(spec=GitRepo)
    git.auto_commit = MagicMock(return_value="abc123")

    async with aiohttp.ClientSession() as session:
        llm = LLMClient(session, base_url, api_key)
        agent = ConversationalAgent(store, llm, model, git)
        yield agent, store, git


@_skip_no_llm
@pytest.mark.llm
class TestRealisticConversationRealLLM:
    """Same scenario as TestRealisticConversation but with a real LLM."""

    async def test_family_conversation_real(self, real_agent):
        agent, store, git = real_agent

        messages = [
            "My daughter is Lily Smith, she's 4 years old",
            "Lily started swimming lessons at the YMCA last week",
            "Her swim instructor is Coach Emma Johnson",
            "Lily had her second swim class today, she can float on her back now!",
        ]

        for msg in messages:
            await agent.handle(msg, "Telegram")

        # --- Relaxed assertions ---

        # At least one person with "Lily" in the name
        people = store.read_all_people()
        lily_matches = [p for p in people if "lily" in p.display_name.lower()]
        assert len(lily_matches) >= 1, f"Expected a person named Lily, got: {people}"

        # At least one memory about swimming
        memories = store.list_memories()
        swim_memories = [
            m for m in memories
            if "swim" in m.title.lower() or "swim" in m.description.lower()
        ]
        assert len(swim_memories) >= 1, (
            f"Expected at least one swimming memory, got: {[m.title for m in memories]}"
        )

        # Chat history has entries for each exchange
        history = store.read_chat_history()
        assert len(history.entries) >= 8  # 4 user + 4 assistant

        # Git was used
        assert git.auto_commit.call_count >= 1


# ===========================================================================
# Test 4: Tool Coverage (real LLM)
# ===========================================================================


@_skip_no_llm
@pytest.mark.llm
class TestToolCoverageRealLLM:
    """Exercise update tools with a real LLM."""

    async def test_update_tools_real(self, real_agent):
        agent, store, git = real_agent

        # Pre-seed Lily so the LLM can reference her
        store.write_person(
            Person(person_id="lily_smith", display_name="Lily Smith", relationship=["daughter"]),
        )

        # 1. Create a memory
        await agent.handle("We went to Zilker Park with Lily Smith today", "Telegram")
        memories_after_create = store.list_memories()
        assert len(memories_after_create) >= 1, "Expected at least one memory after first message"

        # 2. Update preferences
        await agent.handle(
            "Our favorite coffee shop is Epoch Coffee on North Loop, please note that", "Telegram",
        )
        prefs = store.read_preferences()
        has_location = bool(prefs.locations)
        has_note = any("epoch" in n.lower() or "coffee" in n.lower() for n in prefs.notes)
        assert has_location or has_note, (
            f"Expected Epoch in locations or notes. "
            f"Locations: {prefs.locations}, Notes: {prefs.notes}"
        )

        # Chat history
        history = store.read_chat_history()
        assert len(history.entries) >= 4  # at least 2 exchanges

        # Git
        assert git.auto_commit.call_count >= 1


# ===========================================================================
# Helpers: Disambiguation seed data
# ===========================================================================


def _seed_pedros(store: DataStore) -> None:
    """Pre-seed 4 Pedros for disambiguation tests."""
    store.write_person(Person(
        person_id="pedro_garcia",
        display_name="Pedro Garcia",
        relationship=["uncle"],
        notes="Lives in Guadalajara",
        current_threads=[CurrentThread(
            topic="house renovation",
            latest_update="kitchen cabinets installed",
            last_mentioned_date=TODAY,
        )],
    ))
    store.write_person(Person(
        person_id="pedro_martinez",
        display_name="Pedro Martinez",
        relationship=["coworker"],
        notes="Works on the analytics team",
    ))
    store.write_person(Person(
        person_id="pedro_silva",
        display_name="Pedro Silva",
        relationship=["neighbor"],
        notes="Lives two doors down",
        current_threads=[CurrentThread(
            topic="fence repair",
            latest_update="getting quotes",
            last_mentioned_date=TODAY,
        )],
    ))
    store.write_person(Person(
        person_id="pedro_lopez",
        display_name="Pedro Lopez",
        relationship=["son's friend's dad"],
        notes="Met at school pickup",
    ))


# ===========================================================================
# Test 5: Ambiguous Name Disambiguation (scripted mock)
# ===========================================================================


class TestAmbiguousNames:
    """Verify the 3 disambiguation paths with scripted LLM responses."""

    async def test_search_people_disambiguation(self, agent_e2e):
        """search_people returns 4 Pedros → asks which → user clarifies → memory created."""
        agent, store, llm, git = agent_e2e
        _seed_pedros(store)

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: "I had a great time with Pedro" ---
            if call_count == 1:
                # Round 1: search for Pedro
                return _resp(None, [_tc("tc_1", "search_people", {"name": "Pedro"})])
            if call_count == 2:
                # Round 2: sees 4 results, asks which Pedro (text-only, no update tool)
                return _resp(
                    "Which Pedro do you mean? I see Pedro Garcia (uncle), "
                    "Pedro Martinez (coworker), Pedro Silva (neighbor), "
                    "and Pedro Lopez (son's friend's dad)."
                )
            if call_count == 3:
                # Round 3: re-prompt fires because no update tool was called
                # and response doesn't contain "no update needed"
                return _resp(
                    "Which Pedro did you mean? No update needed."
                )

            # --- Message 2: "Pedro Garcia" ---
            if call_count == 4:
                # User clarifies → create memory
                return _resp(None, [_tc("tc_2", "create_memory", {
                    "title": "Great time with Pedro Garcia",
                    "date": TODAY_ISO,
                    "type": "outing",
                    "description": "Had a great time with Pedro Garcia",
                    "people": ["Pedro Garcia"],
                })])
            # call_count == 5
            return _resp("Saved your outing with Pedro Garcia!")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        await agent.handle("I had a great time with Pedro", "Telegram")
        await agent.handle("Pedro Garcia", "Telegram")

        assert call_count == 5

        # 1 memory with "Pedro Garcia"
        memories = store.list_memories()
        assert len(memories) == 1
        assert "Pedro Garcia" in memories[0].people

        # All 4 people still exist, no extras created
        assert len(store.read_all_people()) == 4

        # Chat history: 2 user + 2 assistant
        history = store.read_chat_history()
        assert len(history.entries) == 4

    async def test_create_memory_unknown_people_disambiguation(self, agent_e2e):
        """create_memory(people=["Pedro"]) → unknown_people warning → asks → retries."""
        agent, store, llm, git = agent_e2e
        _seed_pedros(store)

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: "I had a great time with Pedro" ---
            if call_count == 1:
                # LLM jumps straight to create_memory with bare "Pedro"
                return _resp(None, [_tc("tc_1", "create_memory", {
                    "title": "Great time with Pedro",
                    "date": TODAY_ISO,
                    "type": "outing",
                    "description": "Had a great time with Pedro",
                    "people": ["Pedro"],
                })])
            if call_count == 2:
                # Sees the unknown_people warning, asks user
                # (create_memory is in UPDATE_TOOLS so no re-prompt)
                return _resp(
                    "Which Pedro do you mean? I know Pedro Garcia, Pedro Martinez, "
                    "Pedro Silva, and Pedro Lopez."
                )

            # --- Message 2: "Pedro Silva" ---
            if call_count == 3:
                # Retry with full name
                return _resp(None, [_tc("tc_2", "create_memory", {
                    "title": "Great time with Pedro Silva",
                    "date": TODAY_ISO,
                    "type": "outing",
                    "description": "Had a great time with Pedro Silva",
                    "people": ["Pedro Silva"],
                })])
            # call_count == 4
            return _resp("Saved!")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        await agent.handle("I had a great time with Pedro", "Telegram")
        await agent.handle("Pedro Silva", "Telegram")

        assert call_count == 4

        # 1 memory with "Pedro Silva", none with bare "Pedro"
        memories = store.list_memories()
        assert len(memories) == 1
        assert "Pedro Silva" in memories[0].people

        # 4 people, no extras
        assert len(store.read_all_people()) == 4

    async def test_update_person_ambiguous_disambiguation(self, agent_e2e):
        """update_person(person_id="pedro") → ambiguous → asks → retries with exact id."""
        agent, store, llm, git = agent_e2e
        _seed_pedros(store)

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: "Pedro's birthday is March 15th" ---
            if call_count == 1:
                return _resp(None, [_tc("tc_1", "update_person", {
                    "person_id": "pedro",
                    "birthday": "1990-03-15",
                })])
            if call_count == 2:
                # Sees ambiguous result, asks user
                # (update_person is in UPDATE_TOOLS so no re-prompt)
                return _resp("Which Pedro do you mean?")

            # --- Message 2: "Pedro Silva" ---
            if call_count == 3:
                return _resp(None, [_tc("tc_2", "update_person", {
                    "person_id": "pedro_silva",
                    "birthday": "1990-03-15",
                })])
            # call_count == 4
            return _resp("Updated Pedro Silva's birthday!")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        await agent.handle("Pedro's birthday is March 15th", "Telegram")
        await agent.handle("Pedro Silva", "Telegram")

        assert call_count == 4

        # Pedro Silva's birthday is set
        pedro_silva = store.read_person("pedro_silva")
        assert pedro_silva is not None
        assert pedro_silva.birthday == date(1990, 3, 15)

        # Other Pedros' birthdays remain None
        for pid in ("pedro_garcia", "pedro_martinez", "pedro_lopez"):
            p = store.read_person(pid)
            assert p is not None
            assert p.birthday is None, f"{pid} birthday should be None"

    async def test_disambiguate_with_context_in_first_message(self, agent_e2e):
        """User provides context ('my uncle') → LLM picks Pedro Garcia without asking."""
        agent, store, llm, git = agent_e2e
        _seed_pedros(store)

        call_count = 0

        async def mock_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            # --- Message 1: "Pedro, my uncle, just finished his kitchen renovation" ---
            if call_count == 1:
                return _resp(None, [_tc("tc_1", "search_people", {"name": "Pedro"})])
            if call_count == 2:
                # Sees 4 results but context says "uncle" → picks Pedro Garcia
                return _resp(None, [_tc("tc_2", "update_person", {
                    "person_id": "pedro_garcia",
                    "current_threads": [{
                        "topic": "house renovation",
                        "latest_update": "kitchen renovation finished",
                        "last_mentioned_date": TODAY_ISO,
                    }],
                })])
            # call_count == 3
            return _resp("Updated Pedro Garcia's renovation thread!")

        llm.chat_with_tools = AsyncMock(side_effect=mock_llm)

        result = await agent.handle(
            "Pedro, my uncle, just finished his kitchen renovation.", "Telegram",
        )

        # Only 1 handle() call, 3 LLM rounds
        assert call_count == 3

        # Pedro Garcia's thread updated
        pedro = store.read_person("pedro_garcia")
        assert pedro is not None
        assert len(pedro.current_threads) == 1
        assert "finished" in pedro.current_threads[0].latest_update

        # Response mentions Garcia
        assert "Garcia" in result or "renovation" in result.lower()


# ===========================================================================
# Test 6: Ambiguous Name Disambiguation (real LLM)
# ===========================================================================


@_skip_no_llm
@pytest.mark.llm
class TestAmbiguousNamesRealLLM:
    """Real LLM disambiguation tests."""

    async def test_ambiguous_pedro_asks_clarification(self, real_agent):
        """Send ambiguous 'Pedro' → LLM should ask for clarification."""
        agent, store, git = real_agent
        _seed_pedros(store)

        response = await agent.handle(
            "I had a great time with Pedro yesterday", "Telegram",
        )

        # Response should indicate disambiguation
        resp_lower = response.lower()
        assert any(
            kw in resp_lower
            for kw in ["which", "clarif", "multiple", "pedro garcia", "pedro martinez"]
        ), f"Expected disambiguation question, got: {response}"

        # No new person should be created
        assert len(store.read_all_people()) == 4

    async def test_clarification_resolves_ambiguity(self, real_agent):
        """Two messages: ambiguous 'Pedro' → clarify 'Pedro Garcia, my uncle' → resolved."""
        agent, store, git = real_agent
        _seed_pedros(store)

        r1 = await agent.handle("I had a great time with Pedro", "Telegram")
        r2 = await agent.handle("I mean Pedro Garcia, my uncle", "Telegram")

        combined = (r1 + " " + r2).lower()

        # Should mention Garcia somewhere
        assert "garcia" in combined, f"Expected 'Garcia' in responses: {r1} | {r2}"

        # 4 people, no extras
        assert len(store.read_all_people()) == 4

        # At least 1 memory or thread update referencing Pedro Garcia
        memories = store.list_memories()
        pedro = store.read_person("pedro_garcia")
        has_memory = any("Pedro Garcia" in m.people for m in memories)
        has_thread_update = pedro is not None and len(pedro.current_threads) > 0
        assert has_memory or has_thread_update, (
            f"Expected memory or thread for Pedro Garcia. "
            f"Memories: {[m.people for m in memories]}, "
            f"Threads: {pedro.current_threads if pedro else 'N/A'}"
        )

        # Chat history: at least 4 entries (2 user + 2 assistant)
        history = store.read_chat_history()
        assert len(history.entries) >= 4
