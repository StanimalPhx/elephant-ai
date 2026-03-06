"""Unit tests for AgentSDKClient — mocks claude_agent_sdk entirely."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Build a fake claude_agent_sdk module so imports succeed even when the
# optional dependency is not installed.
# ---------------------------------------------------------------------------
def _install_fake_sdk():
    """Create and register a mock claude_agent_sdk module in sys.modules."""
    mod = ModuleType("claude_agent_sdk")
    mod.AssistantMessage = type("AssistantMessage", (), {})  # type: ignore[attr-defined]
    mod.TextBlock = type("TextBlock", (), {})  # type: ignore[attr-defined]
    mod.ClaudeAgentOptions = MagicMock()  # type: ignore[attr-defined]
    mod.query = MagicMock()  # type: ignore[attr-defined]
    sys.modules["claude_agent_sdk"] = mod
    return mod


_sdk = _install_fake_sdk()

from elephant.llm.agent_sdk import AgentSDKClient, _format_messages_as_prompt  # noqa: E402
from elephant.llm.client import LLMResponse  # noqa: E402

# Module path for patching the imported names inside agent_sdk
_MOD = "elephant.llm.agent_sdk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assistant_message(*texts: str):
    """Return a fake AssistantMessage containing TextBlock(s)."""
    blocks = []
    for t in texts:
        b = _sdk.TextBlock.__new__(_sdk.TextBlock)
        b.text = t
        blocks.append(b)
    msg = _sdk.AssistantMessage.__new__(_sdk.AssistantMessage)
    msg.content = blocks
    return msg


async def _async_iter(*items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# _format_messages_as_prompt
# ---------------------------------------------------------------------------

@pytest.mark.llm_agent
class TestFormatMessagesAsPrompt:
    def test_extracts_system_prompt(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, prompt = _format_messages_as_prompt(msgs)
        assert system == "You are helpful."
        assert prompt == "Hi"

    def test_joins_user_messages(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        system, prompt = _format_messages_as_prompt(msgs)
        assert system is None
        assert prompt == "First\n\nSecond"

    def test_prefixes_assistant_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _, prompt = _format_messages_as_prompt(msgs)
        assert "[Previous assistant response]: Hi there" in prompt

    def test_prefixes_tool_messages(self):
        msgs = [
            {"role": "tool", "content": "result data", "tool_call_id": "tc_42"},
        ]
        _, prompt = _format_messages_as_prompt(msgs)
        assert "[Tool result for tc_42]: result data" in prompt

    def test_converts_non_string_content(self):
        msgs = [{"role": "user", "content": 12345}]
        _, prompt = _format_messages_as_prompt(msgs)
        assert prompt == "12345"

    def test_empty_messages(self):
        system, prompt = _format_messages_as_prompt([])
        assert system is None
        assert prompt == ""


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

@pytest.mark.llm_agent
class TestAgentSDKChat:
    async def test_returns_text_from_query(self):
        msg = _make_assistant_message("Hello!")
        with patch(f"{_MOD}.query", return_value=_async_iter(msg)):
            client = AgentSDKClient(default_model="test-model")
            result = await client.chat(
                [{"role": "user", "content": "Hi"}], model="test-model",
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"

    async def test_empty_response_returns_none(self):
        with patch(f"{_MOD}.query", return_value=_async_iter()):
            client = AgentSDKClient(default_model="test-model")
            result = await client.chat(
                [{"role": "user", "content": "Hi"}], model="test-model",
            )

        assert result.content is None

    async def test_multiple_text_blocks_joined(self):
        msg = _make_assistant_message("Part 1", "Part 2")
        with patch(f"{_MOD}.query", return_value=_async_iter(msg)):
            client = AgentSDKClient(default_model="test-model")
            result = await client.chat(
                [{"role": "user", "content": "Hi"}], model="test-model",
            )

        assert result.content == "Part 1\nPart 2"

    async def test_passes_correct_options(self):
        mock_opts = MagicMock()
        with (
            patch(f"{_MOD}.query", return_value=_async_iter()),
            patch(f"{_MOD}.ClaudeAgentOptions", return_value=mock_opts) as mock_cls,
        ):
            client = AgentSDKClient(default_model="fallback-model")
            await client.chat(
                [
                    {"role": "system", "content": "Be nice"},
                    {"role": "user", "content": "Hi"},
                ],
                model="specific-model",
            )

            mock_cls.assert_called_once_with(
                model="specific-model",
                system_prompt="Be nice",
                max_turns=1,
                permission_mode="bypassPermissions",
                allowed_tools=[],
            )

    async def test_falls_back_to_default_model(self):
        mock_opts = MagicMock()
        with (
            patch(f"{_MOD}.query", return_value=_async_iter()),
            patch(f"{_MOD}.ClaudeAgentOptions", return_value=mock_opts) as mock_cls,
        ):
            client = AgentSDKClient(default_model="fallback-model")
            await client.chat(
                [{"role": "user", "content": "Hi"}], model="",
            )

            assert mock_cls.call_args[1]["model"] == "fallback-model"


# ---------------------------------------------------------------------------
# chat_with_tools()
# ---------------------------------------------------------------------------

@pytest.mark.llm_agent
class TestAgentSDKChatWithTools:
    async def test_no_mcp_server_falls_back_to_chat(self):
        msg = _make_assistant_message("Fallback response")
        with patch(f"{_MOD}.query", return_value=_async_iter(msg)):
            client = AgentSDKClient(mcp_server=None, default_model="m")
            result = await client.chat_with_tools(
                [{"role": "user", "content": "Hi"}],
                model="m",
                tools=[{"function": {"name": "list_memories"}}],
            )

        assert result.content == "Fallback response"

    async def test_with_mcp_server_calls_query(self):
        msg = _make_assistant_message("Tool result")
        with patch(f"{_MOD}.query", return_value=_async_iter(msg)) as mock_query:
            mcp_server = MagicMock()
            client = AgentSDKClient(mcp_server=mcp_server, default_model="m")
            result = await client.chat_with_tools(
                [{"role": "user", "content": "What happened?"}],
                model="m",
                tools=[{"function": {"name": "list_memories"}}],
            )

        assert result.content == "Tool result"
        mock_query.assert_called_once()

    async def test_builds_allowed_tools_format(self):
        mock_opts = MagicMock()
        with (
            patch(f"{_MOD}.query", return_value=_async_iter()),
            patch(f"{_MOD}.ClaudeAgentOptions", return_value=mock_opts) as mock_cls,
        ):
            mcp_server = MagicMock()
            client = AgentSDKClient(mcp_server=mcp_server, default_model="m")
            await client.chat_with_tools(
                [{"role": "user", "content": "Hi"}],
                model="m",
                tools=[
                    {"function": {"name": "list_memories"}},
                    {"function": {"name": "create_memory"}},
                ],
            )

            call_kwargs = mock_cls.call_args[1]
            assert "mcp__elephant__list_memories" in call_kwargs["allowed_tools"]
            assert "mcp__elephant__create_memory" in call_kwargs["allowed_tools"]

    async def test_always_returns_empty_tool_calls(self):
        msg = _make_assistant_message("Done")
        with patch(f"{_MOD}.query", return_value=_async_iter(msg)):
            mcp_server = MagicMock()
            client = AgentSDKClient(mcp_server=mcp_server, default_model="m")
            result = await client.chat_with_tools(
                [{"role": "user", "content": "Hi"}],
                model="m",
                tools=[{"function": {"name": "list_memories"}}],
            )

        assert result.tool_calls == []

    async def test_uses_max_turns_10(self):
        mock_opts = MagicMock()
        with (
            patch(f"{_MOD}.query", return_value=_async_iter()),
            patch(f"{_MOD}.ClaudeAgentOptions", return_value=mock_opts) as mock_cls,
        ):
            mcp_server = MagicMock()
            client = AgentSDKClient(mcp_server=mcp_server, default_model="m")
            await client.chat_with_tools(
                [{"role": "user", "content": "Hi"}],
                model="m",
                tools=[{"function": {"name": "list_memories"}}],
            )

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["max_turns"] == 10
