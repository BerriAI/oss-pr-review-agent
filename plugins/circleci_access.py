from __future__ import annotations

import logging
import os

from pydantic_ai.tools import ToolDefinition

from .deps import ChatDeps

log = logging.getLogger("litellm-bot.plugins.circleci_access")

CIRCLECI_TOOL_PREFIX = "circleci_"


def _csv_env(name: str) -> set[str]:
    return {v.strip() for v in os.getenv(name, "").split(",") if v.strip()}


class CircleCIAccessPlugin:
    name = "circleci_access"

    async def filter(
        self,
        deps: ChatDeps,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        print(f"deps: {deps}")
        allowed_users = _csv_env("CIRCLECI_ALLOWED_USER_IDS")
        allowed_channels = _csv_env("CIRCLECI_ALLOWED_CHANNEL_IDS")

        if deps.user_id and deps.user_id in allowed_users:
            return tools
        if deps.channel_id and deps.channel_id in allowed_channels:
            return tools

        # Surface silent-deny when caller has no identity at all — most
        # likely a misconfigured call site, not a real outsider.
        dropped = [t.name for t in tools if t.name.startswith(CIRCLECI_TOOL_PREFIX)]
        if dropped and not deps.user_id and not deps.channel_id:
            log.warning(
                "circleci_access: hiding %s — caller has no user_id/channel_id",
                dropped,
            )

        return [t for t in tools if not t.name.startswith(CIRCLECI_TOOL_PREFIX)]
