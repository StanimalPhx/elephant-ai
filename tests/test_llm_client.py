"""Tests for LLM client: retry logic, error classes, response parsing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elephant.llm.client import LLMAuthError, LLMClient, LLMError, LLMResponse, ToolCall


def _make_response(status, body):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=json.dumps(body) if isinstance(body, dict) else body)
    resp.json = AsyncMock(return_value=body if isinstance(body, dict) else {})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@pytest.mark.llm_api
class TestLLMClient:
    def _make_client(self, session=None):
        session = session or AsyncMock()
        return LLMClient(session, "https://api.example.com/v1", "test-key")

    async def test_successful_chat(self):
        body = {
            "choices": [{"message": {"content": "Hello!"}}],
            "model": "test-model",
            "usage": {"total_tokens": 50},
        }
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(200, body))
        client = self._make_client(session)

        result = await client.chat(
            [{"role": "user", "content": "Hi"}],
            model="test-model",
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "test-model"
        assert result.usage["total_tokens"] == 50

    async def test_auth_error_raises_immediately(self):
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(401, "Unauthorized"))
        client = self._make_client(session)

        with pytest.raises(LLMAuthError, match="Auth failed"):
            await client.chat([{"role": "user", "content": "Hi"}], model="m")

    async def test_403_raises_auth_error(self):
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(403, "Forbidden"))
        client = self._make_client(session)

        with pytest.raises(LLMAuthError):
            await client.chat([{"role": "user", "content": "Hi"}], model="m")

    @patch("elephant.llm.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_429(self, mock_sleep):
        success_body = {
            "choices": [{"message": {"content": "OK"}}],
            "model": "m",
            "usage": {},
        }
        session = AsyncMock()
        session.post = MagicMock(
            side_effect=[
                _make_response(429, {"error": "rate limited"}),
                _make_response(200, success_body),
            ]
        )
        client = self._make_client(session)

        result = await client.chat([{"role": "user", "content": "Hi"}], model="m")
        assert result.content == "OK"
        assert mock_sleep.call_count == 1

    @patch("elephant.llm.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_500(self, mock_sleep):
        success_body = {
            "choices": [{"message": {"content": "OK"}}],
            "model": "m",
            "usage": {},
        }
        session = AsyncMock()
        session.post = MagicMock(
            side_effect=[
                _make_response(500, {"error": "server error"}),
                _make_response(200, success_body),
            ]
        )
        client = self._make_client(session)

        result = await client.chat([{"role": "user", "content": "Hi"}], model="m")
        assert result.content == "OK"

    @patch("elephant.llm.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted(self, mock_sleep):
        session = AsyncMock()
        session.post = MagicMock(
            return_value=_make_response(503, {"error": "unavailable"})
        )
        client = self._make_client(session)

        with pytest.raises(LLMError, match="Transient error"):
            await client.chat([{"role": "user", "content": "Hi"}], model="m")
        # 1 initial + 3 retries = 4 attempts, 3 sleeps
        assert mock_sleep.call_count == 3

    async def test_unknown_error_raises(self):
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(400, "Bad request"))
        client = self._make_client(session)

        with pytest.raises(LLMError, match="400"):
            await client.chat([{"role": "user", "content": "Hi"}], model="m")

    async def test_chat_with_tools_parses_tool_calls(self):
        body = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "list_memories",
                                "arguments": '{"date_from": "2026-02-24"}',
                            },
                        }
                    ],
                }
            }],
            "model": "test-model",
            "usage": {"total_tokens": 80},
        }
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(200, body))
        client = self._make_client(session)

        result = await client.chat_with_tools(
            [{"role": "user", "content": "What happened today?"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "list_memories"}}],
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_abc123"
        assert tc.function_name == "list_memories"
        assert '"date_from"' in tc.arguments

    async def test_chat_with_tools_text_response(self):
        body = {
            "choices": [{"message": {"content": "Hello!", "tool_calls": None}}],
            "model": "m",
            "usage": {},
        }
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(200, body))
        client = self._make_client(session)

        result = await client.chat_with_tools(
            [{"role": "user", "content": "Hi"}],
            model="m",
            tools=[],
        )

        assert result.content == "Hello!"
        assert result.tool_calls == []

    async def test_chat_with_tools_multiple_tool_calls(self):
        body = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_context", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "list_people", "arguments": "{}"},
                        },
                    ],
                }
            }],
            "model": "m",
            "usage": {},
        }
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(200, body))
        client = self._make_client(session)

        result = await client.chat_with_tools(
            [{"role": "user", "content": "Info"}], model="m", tools=[],
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function_name == "get_context"
        assert result.tool_calls[1].function_name == "list_people"

    async def test_backward_compat_chat_still_works(self):
        """Existing chat() method still works with the updated LLMResponse."""
        body = {
            "choices": [{"message": {"content": "Hi there!"}}],
            "model": "m",
            "usage": {"total_tokens": 10},
        }
        session = AsyncMock()
        session.post = MagicMock(return_value=_make_response(200, body))
        client = self._make_client(session)

        result = await client.chat(
            [{"role": "user", "content": "Hello"}], model="m",
        )

        assert result.content == "Hi there!"
        assert result.tool_calls == []
