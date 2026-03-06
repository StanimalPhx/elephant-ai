"""Build an in-process MCP server from Elephant's tool definitions."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from elephant.tools.definitions import TOOL_DEFINITIONS
from elephant.tracing import ToolExecStep, record_step

if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig

    from elephant.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class _FakeToolCall:
    """Minimal object matching the ToolCall interface for ToolExecutor.execute()."""

    def __init__(self, name: str, arguments: str) -> None:
        self.id = f"mcp_{name}_{uuid.uuid4().hex[:8]}"
        self.function_name = name
        self.arguments = arguments


def build_elephant_mcp_server(executor: ToolExecutor) -> McpSdkServerConfig:
    """Create an in-process MCP server with all Elephant tools.

    Each tool delegates to the corresponding ToolExecutor handler,
    reusing existing validation and business logic.
    """
    sdk_tools = []

    for tool_def in TOOL_DEFINITIONS:
        func_def = tool_def["function"]
        name: str = func_def["name"]
        description: str = func_def["description"]
        params: dict[str, Any] = func_def["parameters"]

        # Build the handler closure — capture name/executor by default arg
        async def _handler(
            args: dict[str, Any],
            _name: str = name,
            _executor: ToolExecutor = executor,
        ) -> dict[str, Any]:
            # Record tracing step
            args_json = json.dumps(args, default=str)
            call_id = f"mcp_{_name}_{uuid.uuid4().hex[:8]}"
            step = ToolExecStep(
                tool_call_id=call_id,
                function_name=_name,
                arguments=args_json,
            )
            record_step(step)

            # Delegate to the existing handler
            fake_call = _FakeToolCall(_name, args_json)
            result_str = await _executor.execute(fake_call)  # type: ignore[arg-type]
            step.result = result_str

            logger.info("MCP tool %s executed", _name)
            return {"content": [{"type": "text", "text": result_str}]}

        # Create the @tool-decorated function
        decorated = tool(name, description, params)(_handler)
        sdk_tools.append(decorated)

    return create_sdk_mcp_server(
        name="elephant",
        version="1.0.0",
        tools=sdk_tools,
    )
