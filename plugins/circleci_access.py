"""Restrict CircleCI tools to allowed Slack users / channels.

Tools whose name starts with `CIRCLECI_TOOL_PREFIX` are hidden from the
chat_agent LLM unless the caller is on the user allowlist OR is calling from
the allowed Slack channel. The check runs in `_filter_chat_tools` before
every LLM step.

Allowlist sourcing:
  - `CIRCLECI_ALLOWED_USER_IDS` env var, comma-separated Slack user ids.
  - `ALLOWED_CHANNEL_IDS` is hardcoded; move to env if more channels are added.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai.tools import ToolDefinition

if TYPE_CHECKING:
    from app import ChatDeps


CIRCLECI_TOOL_PREFIX = "circleci_"
ALLOWED_CHANNEL_IDS = {"C0AE9HJQUHG"}


def _load_allowed_user_ids() -> set[str]:
    raw = os.getenv("CIRCLECI_ALLOWED_USER_IDS", "")
    return {uid.strip() for uid in raw.split(",") if uid.strip()}


ALLOWED_USER_IDS = _load_allowed_user_ids()


class CircleCIAccessPlugin:
    name = "circleci_access"

    async def filter(
        self,
        deps: ChatDeps,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if deps.user_id and deps.user_id in ALLOWED_USER_IDS:
            return tools
        if deps.channel_id and deps.channel_id in ALLOWED_CHANNEL_IDS:
            return tools
        return [t for t in tools if not t.name.startswith(CIRCLECI_TOOL_PREFIX)]
