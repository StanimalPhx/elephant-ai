"""Claude Agent SDK backend — uses the claude CLI for LLM calls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from elephant.llm.client import LLMResponse

if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig

logger = logging.getLogger(__name__)


def _format_messages_as_prompt(messages: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Extract system prompt and flatten conversation messages into a single prompt string.

    Returns (system_prompt, user_prompt).
    """
    system_prompt: str | None = None
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            system_prompt = content
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Previous assistant response]: {content}")
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "")
            parts.append(f"[Tool result for {tool_id}]: {content}")

    return system_prompt, "\n\n".join(parts)


class AgentSDKClient:
    """LLM backend that uses the Claude Agent SDK (claude CLI subprocess).

    Satisfies the LLMBackend protocol.
    """

    def __init__(
        self,
        *,
        mcp_server: McpSdkServerConfig | None = None,
        default_model: str = "claude-sonnet-4-6",
    ) -> None:
        self._mcp_server = mcp_server
        self._default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Simple text-in/text-out call (no tools). Temperature is ignored."""
        system_prompt, user_prompt = _format_messages_as_prompt(messages)

        options = ClaudeAgentOptions(
            model=model or self._default_model,
            system_prompt=system_prompt,
            max_turns=1,
            permission_mode="bypassPermissions",
            allowed_tools=[],
        )

        text_parts: list[str] = []
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

        content = "\n".join(text_parts) or None
        return LLMResponse(
            content=content,
            model=model or self._default_model,
            usage={},
        )

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Call with MCP tools. Claude handles the full tool loop internally.

        Returns the final text response with tool_calls=[] (already executed).
        Temperature is ignored (Agent SDK does not support it).
        """
        if self._mcp_server is None:
            # Fall back to simple chat if no MCP server configured
            return await self.chat(messages, model, temperature, max_tokens)

        system_prompt, user_prompt = _format_messages_as_prompt(messages)

        # Build allowed_tools list from tool definitions
        allowed_mcp_tools = [
            f"mcp__elephant__{t['function']['name']}" for t in tools
        ]

        options = ClaudeAgentOptions(
            model=model or self._default_model,
            system_prompt=system_prompt,
            max_turns=10,
            permission_mode="bypassPermissions",
            mcp_servers={"elephant": self._mcp_server},
            allowed_tools=allowed_mcp_tools,
        )

        text_parts: list[str] = []
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

        content = "\n".join(text_parts) or None
        return LLMResponse(
            content=content,
            model=model or self._default_model,
            usage={},
            tool_calls=[],  # Tools already executed via MCP
        )
