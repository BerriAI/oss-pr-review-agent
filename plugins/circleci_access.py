"""Restrict CircleCI tools to allowed Slack users / channels.

Tools whose name starts with `CIRCLECI_TOOL_PREFIX` are hidden from the
chat_agent LLM unless the caller is on the user allowlist OR is calling from
the allowed Slack channel. The check runs in `_filter_chat_tools` before
every LLM step.

Allowlist sourcing (read on each call so env edits don't need a process
restart):
  - `CIRCLECI_ALLOWED_USER_IDS`    — comma-separated user ids (Slack U...,
    or the web `ADMIN_USERNAME` for the dev chat path).
  - `CIRCLECI_ALLOWED_CHANNEL_IDS` — comma-separated Slack channel ids.
"""

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
        allowed_users = _csv_env("CIRCLECI_ALLOWED_USER_IDS")
        allowed_channels = _csv_env("CIRCLECI_ALLOWED_CHANNEL_IDS")

        if deps.user_id and deps.user_id in allowed_users:
            return tools
        if deps.channel_id and deps.channel_id in allowed_channels:
            return tools

        # Surface the silent-deny case so a misconfigured caller (no
        # user_id/channel_id at all) is debuggable without spelunking the
        # transcript. Only log when we actually drop something — no-op calls
        # stay quiet.
        dropped = [t.name for t in tools if t.name.startswith(CIRCLECI_TOOL_PREFIX)]
        if dropped and not deps.user_id and not deps.channel_id:
            log.warning(
                "circleci_access: hiding %s — caller has no user_id/channel_id "
                "(check the call site populates ChatDeps)",
                dropped,
            )

        return [t for t in tools if not t.name.startswith(CIRCLECI_TOOL_PREFIX)]
