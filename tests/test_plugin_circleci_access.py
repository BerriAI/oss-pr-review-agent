import asyncio

import pytest
from pydantic_ai.tools import ToolDefinition

from plugins.circleci_access import CIRCLECI_TOOL_PREFIX, CircleCIAccessPlugin
from plugins.deps import ChatDeps


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="",
        parameters_json_schema={"type": "object", "properties": {}, "required": []},
    )


@pytest.fixture
def tools() -> list[ToolDefinition]:
    return [
        _tool(f"{CIRCLECI_TOOL_PREFIX}rerun_failed"),
        _tool(f"{CIRCLECI_TOOL_PREFIX}fetch_logs"),
        _tool("run_pr_review"),
        _tool("add_memory"),
    ]


@pytest.fixture
def plugin() -> CircleCIAccessPlugin:
    return CircleCIAccessPlugin()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CIRCLECI_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("CIRCLECI_ALLOWED_CHANNEL_IDS", raising=False)


def test_allowed_user_keeps_circleci_tools(plugin, tools, monkeypatch):
    monkeypatch.setenv("CIRCLECI_ALLOWED_USER_IDS", "U123,U456")
    deps = ChatDeps(user_id="U123")

    out = asyncio.run(plugin.filter(deps, tools))

    assert {t.name for t in out} == {t.name for t in tools}


def test_allowed_channel_keeps_circleci_tools(plugin, tools, monkeypatch):
    monkeypatch.setenv("CIRCLECI_ALLOWED_CHANNEL_IDS", "C0AE9HJQUHG")
    deps = ChatDeps(user_id="U_unknown", channel_id="C0AE9HJQUHG")

    out = asyncio.run(plugin.filter(deps, tools))

    assert {t.name for t in out} == {t.name for t in tools}


def test_blocked_caller_loses_circleci_tools_only(plugin, tools):
    deps = ChatDeps(user_id="U_outsider", channel_id="C_random")

    out = asyncio.run(plugin.filter(deps, tools))

    names = {t.name for t in out}
    assert names == {"run_pr_review", "add_memory"}


def test_no_caller_id_blocks_circleci_and_logs(plugin, tools, caplog):
    deps = ChatDeps()

    with caplog.at_level("WARNING", logger="litellm-bot.plugins.circleci_access"):
        out = asyncio.run(plugin.filter(deps, tools))

    assert all(not t.name.startswith(CIRCLECI_TOOL_PREFIX) for t in out)
    assert any("no user_id/channel_id" in rec.message for rec in caplog.records)


def test_no_circleci_tools_registered_is_passthrough(plugin):
    deps = ChatDeps()
    safe = [_tool("run_pr_review"), _tool("add_memory")]

    out = asyncio.run(plugin.filter(deps, safe))

    assert [t.name for t in out] == ["run_pr_review", "add_memory"]


def test_env_change_takes_effect_without_restart(plugin, tools, monkeypatch):
    deps = ChatDeps(user_id="U999")

    out_before = asyncio.run(plugin.filter(deps, tools))
    assert all(not t.name.startswith(CIRCLECI_TOOL_PREFIX) for t in out_before)

    monkeypatch.setenv("CIRCLECI_ALLOWED_USER_IDS", "U999")
    out_after = asyncio.run(plugin.filter(deps, tools))
    assert {t.name for t in out_after} == {t.name for t in tools}
