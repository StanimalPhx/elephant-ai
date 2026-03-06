"""debug-message subcommand: run a message through the full flow with instrumentation."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from elephant.config import load_config
from elephant.data.store import DataStore
from elephant.flows.anytime_log import AnytimeLogFlow
from elephant.git_ops import GitRepo
from elephant.llm.client import LLMClient, LLMResponse
from elephant.messaging.base import IncomingMessage, SendResult

if TYPE_CHECKING:
    from elephant.data.models import RawMessage


# ---------------------------------------------------------------------------
# Stderr helpers
# ---------------------------------------------------------------------------

def _err(text: str = "") -> None:
    print(text, file=sys.stderr)


def _divider(title: str) -> None:
    _err(f"\n\u2550\u2550\u2550 LLM Call: {title} \u2550\u2550\u2550")


def _thin_divider(title: str) -> None:
    _err(f"\n\u2500\u2500\u2500 {title} \u2500\u2500\u2500")


def _format_message(msg: dict[str, Any]) -> str:
    role = msg.get("role", "?")
    content = msg.get("content", "")

    # Tool-call assistant messages
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        parts: list[str] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            raw_args = fn.get("arguments", "")
            try:
                parsed = json.loads(raw_args)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pretty = raw_args
            parts.append(f"{name}({pretty})")
        body = "<tool_calls: " + ", ".join(parts) + ">"
        if content:
            body = f"{content}\n{body}"
        return f"[{role}] {body}"

    # Tool result messages
    if role == "tool":
        tid = msg.get("tool_call_id", "")
        try:
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass
        return f"[tool:{tid}] {content}"

    # Truncate very long content for readability
    text = str(content)
    if len(text) > 2000:
        text = text[:2000] + f"... ({len(text)} chars total)"
    return f"[{role}] {text}"


def _print_llm_call(
    method: str,
    messages: list[dict[str, Any]],
    response: LLMResponse,
    *,
    model: str = "",
    temperature: float = 0.7,
) -> None:
    _divider(method)
    _err(f"Model: {model or response.model} | Temp: {temperature}")
    _thin_divider(f"Prompt ({len(messages)} messages)")
    for msg in messages:
        _err(_format_message(msg))
    _thin_divider("Response")
    if response.content:
        _err(f"Content: {response.content}")
    else:
        _err("Content: (none)")
    if response.tool_calls:
        for tc in response.tool_calls:
            try:
                pretty = json.dumps(json.loads(tc.arguments), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pretty = tc.arguments
            _err(f"  Tool: {tc.function_name}({pretty})")
    else:
        _err("Tool calls: (none)")
    usage = response.usage
    prompt_tok = usage.get("prompt_tokens", "?")
    completion_tok = usage.get("completion_tokens", "?")
    _err(f"Usage: {prompt_tok} prompt + {completion_tok} completion")
    _err("\u2550" * 40)


# ---------------------------------------------------------------------------
# Instrumented LLM client
# ---------------------------------------------------------------------------

class InstrumentedLLMClient(LLMClient):
    """Wraps LLMClient to print every prompt/response to stderr."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = await super().chat(messages, model, temperature, max_tokens)
        _print_llm_call("chat", messages, response, model=model, temperature=temperature)
        return response

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = await super().chat_with_tools(messages, model, tools, temperature, max_tokens)
        _print_llm_call(
            "chat_with_tools", messages, response, model=model, temperature=temperature,
        )
        return response


# ---------------------------------------------------------------------------
# Read-only store (reads normally, skips all writes)
# ---------------------------------------------------------------------------

class ReadOnlyStore(DataStore):
    """DataStore that reads normally but intercepts all writes."""

    def append_raw_message(self, message: RawMessage) -> None:
        _err(f"[ReadOnlyStore] skip append_raw_message (sender={message.sender})")

    def append_chat_history(
        self,
        user_content: str,
        assistant_content: str,
        max_entries: int = 1000,
    ) -> None:
        _err("[ReadOnlyStore] skip append_chat_history")

    def write_memory(self, memory: Any) -> str:
        _err(f"[ReadOnlyStore] skip write_memory (id={memory.id})")
        return f"(dry-run) memories/{memory.id}.yaml"

    def update_memory(self, memory_id: str, updates: dict[str, Any]) -> Any:
        memory = self.find_memory_by_id(memory_id)
        if memory is None:
            return None
        updated = memory.model_copy(update=updates)
        _err(f"[ReadOnlyStore] skip update_memory (id={memory_id})")
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        _err(f"[ReadOnlyStore] skip delete_memory (id={memory_id})")
        return True

    def write_person(self, person: Any) -> str:
        _err(f"[ReadOnlyStore] skip write_person (id={person.person_id})")
        return f"(dry-run) people/{person.person_id}.yaml"

    def write_pending_questions(self, pq: Any) -> None:
        _err("[ReadOnlyStore] skip write_pending_questions")

    def write_digest_state(self, state: Any) -> None:
        _err("[ReadOnlyStore] skip write_digest_state")

    def write_chat_history(self, history: Any) -> None:
        _err("[ReadOnlyStore] skip write_chat_history")

    def write_raw_messages(self, messages: Any) -> None:
        _err("[ReadOnlyStore] skip write_raw_messages")

    def write_preferences(self, prefs: Any) -> None:
        _err("[ReadOnlyStore] skip write_preferences")

    def write_authorized_chats(self, ac: Any) -> None:
        _err("[ReadOnlyStore] skip write_authorized_chats")


# ---------------------------------------------------------------------------
# No-op git repo
# ---------------------------------------------------------------------------

class NoOpGitRepo(GitRepo):
    """GitRepo that does nothing."""

    def __init__(self) -> None:
        # Don't call super().__init__ — we never touch the filesystem
        self.repo_dir = "/dev/null"

    def initialize(self) -> None:
        pass

    def auto_commit(
        self,
        tag: str,
        message: str,
        timestamp: Any = None,
        paths: list[str] | None = None,
    ) -> str | None:
        _err(f"[NoOpGit] skip commit [{tag}] {message}")
        return None


# ---------------------------------------------------------------------------
# No-op messaging client
# ---------------------------------------------------------------------------

class CapturingMessagingClient:
    """Captures replies instead of sending via Telegram."""

    def __init__(self) -> None:
        self.replies: list[str] = []

    async def send_text(self, text: str) -> SendResult:
        self.replies.append(text)
        _err(f"\n>>> Reply to user:\n{text}\n")
        return SendResult(success=True, message_id="debug-0")

    async def send_text_with_media(self, text: str, media_url: str) -> SendResult:
        self.replies.append(f"{text}\n[media: {media_url}]")
        _err(f"\n>>> Reply to user (with media):\n{text}\n[media: {media_url}]\n")
        return SendResult(success=True, message_id="debug-0")

    async def send_chat_action(self, action: str = "typing") -> None:
        pass

    async def broadcast_text(self, text: str) -> list[SendResult]:
        return [await self.send_text(text)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _run(config_path: str, message: str, database: str | None) -> None:
    config = load_config(config_path)

    # Select database
    if database:
        db_cfg = None
        for db in config.databases:
            if db.name == database:
                db_cfg = db
                break
        if db_cfg is None:
            names = ", ".join(db.name for db in config.databases)
            _err(f"Error: database '{database}' not found. Available: {names}")
            sys.exit(1)
    else:
        db_cfg = config.databases[0]

    _err(f"Config: {config_path}")
    _err(f"Database: {db_cfg.name} ({db_cfg.data_dir})")
    _err(f"Message: {message!r}")
    _err(f"LLM: backend={config.llm.backend} | model: {config.llm.parsing_model}")
    _err("")

    store = ReadOnlyStore(db_cfg.data_dir)
    git = NoOpGitRepo()
    messaging = CapturingMessagingClient()

    async with aiohttp.ClientSession() as session:
        if config.llm.backend == "agent_sdk":
            from elephant.llm.agent_sdk import AgentSDKClient

            llm: LLMClient | AgentSDKClient = AgentSDKClient(
                default_model=config.llm.parsing_model,
            )
        else:
            llm = InstrumentedLLMClient(session, config.llm.base_url, config.llm.api_key)

        flow = AnytimeLogFlow(
            store=store,
            llm=llm,
            parsing_model=config.llm.parsing_model,
            messaging=messaging,
            git=git,
            history_limit=db_cfg.chat_history_limit,
        )

        incoming = IncomingMessage(
            text=message,
            sender="debug-cli",
            message_id="debug-msg-0",
            timestamp=datetime.now(UTC),
        )

        await flow.handle_message(incoming)

    # Print final replies to stdout
    if messaging.replies:
        for reply in messaging.replies:
            print(reply)


def run_debug_message(config_path: str, message: str, database: str | None) -> None:
    """Sync wrapper to run the async debug flow."""
    asyncio.run(_run(config_path, message, database))
