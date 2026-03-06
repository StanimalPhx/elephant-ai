"""Tests for the conversational agent tool-calling loop."""

import json
import sys
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elephant.data.models import Memory, Person
from elephant.data.store import DataStore
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMResponse, ToolCall
from elephant.tools.agent import ConversationalAgent, _needs_reprompt


@pytest.fixture
def agent_deps(data_dir):
    store = DataStore(data_dir)
    store.initialize()

    llm = AsyncMock()
    git = MagicMock(spec=GitRepo)
    git.auto_commit = MagicMock(return_value="abc123")

    agent = ConversationalAgent(store, llm, "test-model", git)
    return agent, store, llm, git


@pytest.mark.llm_api
class TestConversationalAgent:
    async def test_direct_text_response(self, agent_deps):
        """LLM responds with text containing opt-out, no tool calls."""
        agent, store, llm, git = agent_deps

        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content="Hello! How can I help you? No update needed.",
            model="m",
            usage={},
            tool_calls=[],
        ))

        result = await agent.handle("Hello!", "Telegram")
        assert result == "Hello! How can I help you? No update needed."
        llm.chat_with_tools.assert_called_once()

    async def test_tool_call_then_text(self, agent_deps):
        """LLM makes a tool call, gets result, then responds with text."""
        agent, store, llm, git = agent_deps

        # Seed a memory
        store.write_memory(Memory(
            id="20260224_park_day", date=date(2026, 2, 24), title="Park day",
            type="daily", description="Went to the park with Lily",
            people=["Lily"], source="agent",
        ))

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: LLM wants to search memories
                return LLMResponse(
                    content=None,
                    model="m",
                    usage={},
                    tool_calls=[ToolCall(
                        id="tc_1",
                        function_name="list_memories",
                        arguments=json.dumps({"date_from": "2026-02-24", "date_to": "2026-02-24"}),
                    )],
                )
            else:
                # Second call: LLM responds with summary (query-only, so opt-out)
                return LLMResponse(
                    content="Today you went to the park with Lily! No update needed.",
                    model="m",
                    usage={},
                    tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("What happened today?", "Telegram")
        assert "park" in result.lower() or "Lily" in result
        assert call_count == 2

    async def test_create_memory_tool_call(self, agent_deps):
        """LLM calls create_memory, then confirms."""
        agent, store, llm, git = agent_deps
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None,
                    model="m",
                    usage={},
                    tool_calls=[ToolCall(
                        id="tc_1",
                        function_name="create_memory",
                        arguments=json.dumps({
                            "title": "Park day",
                            "date": "2026-02-24",
                            "type": "outing",
                            "description": "We went to the park with Lily",
                            "people": ["Lily"],
                            "location": "Central Park",
                        }),
                    )],
                )
            else:
                return LLMResponse(
                    content="Got it! I've saved your park outing with Lily.",
                    model="m",
                    usage={},
                    tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("We went to the park with Lily", "Telegram")
        assert "park" in result.lower() or "Lily" in result
        git.auto_commit.assert_called()  # Memory was committed

    async def test_multiple_tool_calls(self, agent_deps):
        """LLM makes multiple tool calls in sequence."""
        agent, store, llm, git = agent_deps

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, model="m", usage={},
                    tool_calls=[
                        ToolCall(id="tc_1", function_name="list_people", arguments="{}"),
                        ToolCall(id="tc_2", function_name="list_memories", arguments="{}"),
                    ],
                )
            else:
                return LLMResponse(
                    content="Here's what I know about your family... No update needed.",
                    model="m", usage={}, tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("Tell me about my family", "Telegram")
        assert "family" in result.lower()

    async def test_tool_error_retries_then_succeeds(self, agent_deps):
        """LLM gets a tool error, retries with corrected args, and succeeds."""
        agent, store, llm, git = agent_deps

        store.write_memory(Memory(
            id="20260224_park_day", date=date(2026, 2, 24), title="Park day",
            type="daily", description="Went to the park",
            people=["Lily"], source="agent",
        ))

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: LLM uses wrong memory_id
                return LLMResponse(
                    content=None, model="m", usage={},
                    tool_calls=[ToolCall(
                        id="tc_1",
                        function_name="get_memory",
                        arguments=json.dumps({"memory_id": "bad_id"}),
                    )],
                )
            elif call_count == 2:
                # Second call: LLM retries with correct id
                return LLMResponse(
                    content=None, model="m", usage={},
                    tool_calls=[ToolCall(
                        id="tc_2",
                        function_name="get_memory",
                        arguments=json.dumps({"memory_id": "20260224_park_day"}),
                    )],
                )
            else:
                # Third call: LLM responds with the memory info
                return LLMResponse(
                    content="Found it! You went to the park with Lily. No update needed.",
                    model="m", usage={}, tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("Tell me about the park day", "Telegram")
        assert "park" in result.lower()
        assert call_count == 3

    async def test_tool_error_max_retries_exceeded(self, agent_deps):
        """Agent stops after MAX_ERROR_RETRIES consecutive all-error rounds."""
        agent, store, llm, git = agent_deps

        # LLM always calls get_memory with a bad id
        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content=None, model="m", usage={},
            tool_calls=[ToolCall(
                id="tc_1",
                function_name="get_memory",
                arguments=json.dumps({"memory_id": "nonexistent"}),
            )],
        ))

        result = await agent.handle("Find that memory", "Telegram")
        assert "I tried several times" in result
        assert "Memory not found: nonexistent" in result
        # Should stop after MAX_ERROR_RETRIES rounds, not MAX_TOOL_ROUNDS
        assert llm.chat_with_tools.call_count == agent.MAX_ERROR_RETRIES

    async def test_max_rounds_safety(self, agent_deps):
        """Agent stops after MAX_TOOL_ROUNDS and falls back to plain chat."""
        agent, store, llm, git = agent_deps

        # Always return tool calls (infinite loop scenario)
        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content=None, model="m", usage={},
            tool_calls=[ToolCall(
                id="tc_1", function_name="list_memories", arguments="{}",
            )],
        ))
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="Sorry, I had trouble processing that.",
            model="m", usage={},
        ))

        result = await agent.handle("Do something", "Telegram")
        assert result == "Sorry, I had trouble processing that."
        assert llm.chat_with_tools.call_count == agent.MAX_TOOL_ROUNDS
        # llm.chat is called multiple times: injection check, final fallback, output sanitizer
        assert llm.chat.call_count >= 1


@pytest.mark.llm_api
class TestNeedsReprompt:
    """Unit tests for _needs_reprompt."""

    def test_returns_true_when_no_update_tool_and_no_opt_out(self):
        assert _needs_reprompt("I'll update his profile!", set()) is True

    def test_returns_false_when_update_tool_called(self):
        assert _needs_reprompt("Done!", {"update_person"}) is False

    def test_returns_false_when_text_contains_opt_out(self):
        assert _needs_reprompt("Hello! No update needed.", set()) is False

    def test_case_insensitive_opt_out(self):
        assert _needs_reprompt("NO UPDATE NEEDED.", set()) is False
        assert _needs_reprompt("no update needed.", set()) is False

    def test_returns_false_with_update_tool_even_without_opt_out(self):
        assert _needs_reprompt("Saved!", {"create_memory", "list_people"}) is False

    def test_returns_true_with_only_query_tools(self):
        assert _needs_reprompt("Here's what I found.", {"list_memories"}) is True


@pytest.mark.llm_api
class TestRepromptBehavior:
    """Integration tests for the re-prompt logic in handle()."""

    async def test_reprompt_then_tool_call(self, agent_deps):
        """LLM responds without tool or opt-out → re-prompted → calls update_person."""
        agent, store, llm, git = agent_deps
        store.write_person(
            Person(person_id="nicholas", display_name="Nicholas", relationship=["son"]),
        )

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First: LLM promises but doesn't call tool
                return LLMResponse(
                    content="I'll update Nicholas's profile!",
                    model="m", usage={}, tool_calls=[],
                )
            elif call_count == 2:
                # Re-prompted: now calls update_person
                return LLMResponse(
                    content=None, model="m", usage={},
                    tool_calls=[ToolCall(
                        id="tc_1",
                        function_name="update_person",
                        arguments=json.dumps({
                            "person_id": "nicholas",
                            "notes": "Does swimming, KidStrong, ice skating",
                        }),
                    )],
                )
            else:
                # Final confirmation
                return LLMResponse(
                    content="I've updated Nicholas's activities.",
                    model="m", usage={}, tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("Nicholas does swimming and KidStrong", "Telegram")
        assert call_count == 3
        assert "Nicholas" in result

    async def test_no_reprompt_with_opt_out(self, agent_deps):
        """LLM responds with 'No update needed.' → no re-prompt."""
        agent, store, llm, git = agent_deps

        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content="That's interesting! No update needed.",
            model="m", usage={}, tool_calls=[],
        ))

        result = await agent.handle("Hello there", "Telegram")
        assert result == "That's interesting! No update needed."
        assert llm.chat_with_tools.call_count == 1

    async def test_no_reprompt_after_update_tool(self, agent_deps):
        """LLM calls update_person then responds → no re-prompt."""
        agent, store, llm, git = agent_deps
        store.write_person(
            Person(person_id="lily", display_name="Lily", relationship=["daughter"]),
        )

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, model="m", usage={},
                    tool_calls=[ToolCall(
                        id="tc_1",
                        function_name="update_person",
                        arguments=json.dumps({
                            "person_id": "lily",
                            "notes": "Started kindergarten",
                        }),
                    )],
                )
            else:
                return LLMResponse(
                    content="I've noted that Lily started kindergarten!",
                    model="m", usage={}, tool_calls=[],
                )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        await agent.handle("Lily started kindergarten", "Telegram")
        assert call_count == 2  # tool call + final response, no re-prompt

    async def test_reprompt_only_once(self, agent_deps):
        """LLM keeps omitting after re-prompt → accepts on second try."""
        agent, store, llm, git = agent_deps

        call_count = 0

        async def mock_chat_with_tools(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always responds without tool call or opt-out
            return LLMResponse(
                content=f"Response {call_count}",
                model="m", usage={}, tool_calls=[],
            )

        llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

        result = await agent.handle("Some info to store", "Telegram")
        # First attempt triggers re-prompt, second attempt is accepted (no infinite loop)
        assert call_count == 2
        assert result == "Response 2"


@pytest.mark.llm_agent
class TestAgentSDKPath:
    """Tests for the Agent SDK code path (_handle_agent_sdk)."""

    @pytest.fixture
    def agent_sdk_deps(self, data_dir):
        """Create agent with a mock LLM that looks like AgentSDKClient."""
        store = DataStore(data_dir)
        store.initialize()

        # Create a real class named "AgentSDKClient" so type().__name__ check works
        class AgentSDKClient:
            _default_model = "test-model"
            chat = AsyncMock()
            chat_with_tools = AsyncMock()

        llm = AgentSDKClient()

        git = MagicMock(spec=GitRepo)
        git.auto_commit = MagicMock(return_value="abc123")

        # The new client that __init__ will create (also named AgentSDKClient)
        new_client = AgentSDKClient()

        # Mock the modules that __init__ imports locally
        mock_mcp_mod = MagicMock()
        mock_mcp_mod.build_elephant_mcp_server.return_value = MagicMock()

        mock_sdk_mod = MagicMock()
        mock_sdk_mod.AgentSDKClient.return_value = new_client

        with patch.dict(sys.modules, {
            "elephant.mcp.tools": mock_mcp_mod,
            "elephant.llm.agent_sdk": mock_sdk_mod,
        }):
            agent = ConversationalAgent(store, llm, "test-model", git)

        return agent, store, new_client, git

    async def test_is_agent_sdk_detected(self, agent_sdk_deps):
        """Verify _is_agent_sdk returns True for AgentSDKClient."""
        agent, store, llm, git = agent_sdk_deps
        assert agent._is_agent_sdk is True

    async def test_handle_agent_sdk_returns_response(self, agent_sdk_deps):
        """Agent SDK path calls chat_with_tools and returns sanitized text."""
        agent, store, llm, git = agent_sdk_deps

        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content="I've saved the memory about the park trip!",
            model="m",
            usage={},
            tool_calls=[],
        ))

        result = await agent.handle("We went to the park", "Telegram")
        assert "park" in result.lower()
        llm.chat_with_tools.assert_called_once()

    async def test_handle_agent_sdk_saves_chat_history(self, agent_sdk_deps):
        """Agent SDK path saves chat history after responding."""
        agent, store, llm, git = agent_sdk_deps

        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content="Got it! No update needed.",
            model="m",
            usage={},
            tool_calls=[],
        ))

        await agent.handle("Hello there", "Telegram")
        history = store.read_chat_history()
        assert len(history.entries) == 2  # user + assistant
        assert history.entries[0].role == "user"
        assert history.entries[1].role == "assistant"

    async def test_handle_agent_sdk_sanitizes_output(self, agent_sdk_deps):
        """Agent SDK path redacts sensitive patterns from output."""
        agent, store, llm, git = agent_sdk_deps

        llm.chat_with_tools = AsyncMock(return_value=LLMResponse(
            content="Here's a key: sk-abcdefghijklmnopqrstuvwxyz No update needed.",
            model="m",
            usage={},
            tool_calls=[],
        ))

        result = await agent.handle("Show me secrets", "Telegram")
        assert "sk-" not in result
        assert "[REDACTED]" in result
