from __future__ import annotations

from typing import Protocol

from pydantic_ai.tools import ToolDefinition

from .deps import ChatDeps


class ToolFilterPlugin(Protocol):
    name: str

    async def filter(
        self,
        deps: ChatDeps,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]: ...
