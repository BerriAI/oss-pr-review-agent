"""Tool-filter plugin protocol.

Each plugin inspects the per-call `ChatDeps` (caller identity) and the current
tool list, and returns a (possibly narrower) tool list. Plugins are chained
in order, so each one sees the prior plugin's output. Deny rules compose
naturally — if any plugin drops a tool, later plugins can't bring it back.

This mirrors the LiteLLM proxy custom-callback pattern: a thin Protocol you
implement in a separate file, registered in a list, invoked at one well-known
hook site (`_filter_chat_tools` in `app.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic_ai.tools import ToolDefinition

if TYPE_CHECKING:
    from app import ChatDeps


class ToolFilterPlugin(Protocol):
    """Implement this to gate which tools the chat_agent LLM sees per call.

    `name` is used in logs and (eventually) config files so a plugin can be
    enabled/disabled without code edits. Keep it short and stable.
    """

    name: str

    async def filter(
        self,
        deps: ChatDeps,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]: ...
