"""LLM backend protocol — defines the interface both HTTP and Agent SDK clients satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elephant.llm.client import LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends (HTTP API or Agent SDK)."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
