"""Per-call context passed to chat_agent and read by tool-filter plugins.

Lives in `plugins/` (not `app.py`) so plugin modules can import it without
creating a circular dependency through the top-level app module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatDeps:
    """Caller identity for one chat_agent run. All fields optional so callers
    fill what they have. Plugins must tolerate None on any field."""

    user_id: str | None = None
    workspace_id: str | None = None
    channel_id: str | None = None
