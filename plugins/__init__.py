"""Tool-filter plugin registry.

Plugins run in order, chained — each receives the previous plugin's output.
Append to `TOOL_FILTER_PLUGINS` to enable a new gate. Order should put the
most-restrictive / cheapest checks first so denied tools fall out early.
"""

from .base import ToolFilterPlugin
from .circleci_access import CircleCIAccessPlugin
from .deps import ChatDeps

TOOL_FILTER_PLUGINS: list[ToolFilterPlugin] = [
    CircleCIAccessPlugin(),
]

__all__ = ["ChatDeps", "TOOL_FILTER_PLUGINS", "ToolFilterPlugin"]
