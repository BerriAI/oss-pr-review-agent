from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatDeps:
    user_id: str | None = None
    workspace_id: str | None = None
    channel_id: str | None = None
