"""Smoke test for the auto-derived capability prose fed to the memory-merger.

The merger LLM gates add_memory writes by checking the proposed fact against
this prose. If a TriageReport / PatternReport field is renamed or removed,
or if the CAN/CANNOT separator drifts, this test catches it before the gate
silently starts misjudging instructions.

Not asserting the *full* prose because the renderer changes a few times per
quarter as we add fields. Instead, pin the load-bearing fragments:
- the schema-walk emits each model name
- every TriageReport field shows up by name (and its _FIELD_NOTES note)
- the hand-maintained CANNOT list is present and the no-longer-true
  merge-conflicts bullet is gone (it was hand-maintained and went stale once
  has_merge_conflicts became a real field; this test stops that regressing)
"""

import os

os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")
os.environ.setdefault("LITELLM_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test")

from app import (  # noqa: E402
    _AGENT_CAPABILITIES,
    _AGENT_LIMITATIONS,
    _FIELD_NOTES,
    PatternReport,
    TriageReport,
)


def test_every_triage_field_is_described():
    """Every TriageReport field name appears in the rendered prose. If you add
    a field without a _FIELD_NOTES entry, _describe_model raises at import
    time — but if you slip past that and the field is e.g. only in
    _FIELD_NOTES with a typo'd name, this catches it from the other side."""
    for field_name in TriageReport.model_fields:
        assert f"- {field_name}:" in _AGENT_CAPABILITIES, (
            f"TriageReport.{field_name} missing from capability prose"
        )


def test_every_pattern_field_is_described():
    for field_name in PatternReport.model_fields:
        assert f"- {field_name}:" in _AGENT_CAPABILITIES


def test_field_notes_have_no_orphans():
    """Every _FIELD_NOTES entry should be a real field on one of the walked
    models. A typo'd key would silently never get rendered, so catch it."""
    walked: set[str] = set()
    from app import PatternFinding, PatternReport, TechDebtItem, TriageCard, TriageReport
    for cls in (TriageReport, PatternReport, PatternFinding, TechDebtItem, TriageCard):
        walked.update(cls.model_fields.keys())
    orphans = set(_FIELD_NOTES) - walked
    assert not orphans, f"_FIELD_NOTES entries with no matching field: {orphans}"


def test_cannot_section_present():
    """The CANNOT prose is the part the schemas can't auto-generate. Pin the
    section header so a future edit doesn't accidentally drop it."""
    assert "The agent CANNOT" in _AGENT_CAPABILITIES
    assert _AGENT_LIMITATIONS.strip() in _AGENT_CAPABILITIES


def test_merge_conflicts_not_in_cannot_list():
    """Regression guard: the hand-maintained CANNOT list used to claim merge
    conflicts weren't fetched. They are — has_merge_conflicts is a real
    TriageReport field with a rubric blocker. The bullet must stay deleted."""
    assert "merge conflicts" not in _AGENT_LIMITATIONS.lower()
    assert "mergeable" not in _AGENT_LIMITATIONS.lower()


def test_memory_tools_excluded_from_tool_listing():
    """add_memory / reset_memory describing themselves to the gate is noise.
    They must be filtered from the tools section."""
    tools_section = _AGENT_CAPABILITIES.split("Tools available")[1]
    assert "add_memory" not in tools_section
    assert "reset_memory" not in tools_section
    assert "run_pr_review" in tools_section
