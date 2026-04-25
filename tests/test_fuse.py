"""Unit tests for the deterministic part of the review pipeline.

fuse() is a pure function: (TriageReport, PatternReport) -> TriageCard.
Every rubric branch should be pinned here so the score → verdict mapping
can't drift silently when someone edits _RUBRIC.

These tests don't hit any LLM, network, or filesystem.
"""

import os

os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")
os.environ.setdefault("LITELLM_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test")

from app import (  # noqa: E402
    PatternFinding,
    PatternReport,
    TriageReport,
    fuse,
    render_card,
    render_drilldown,
    render_fallback_card,
)


def _triage(**overrides) -> TriageReport:
    """Default = a clean PR. Override any field per test via model_copy."""
    base = TriageReport(
        pr_number=1,
        pr_title="test pr",
        pr_author="alice",
        pr_summary="Adds a thing.",
        pr_related_failures=[],
        unrelated_failures=[],
        running_checks=[],
        greptile_score=5,
        has_circleci_checks=True,
    )
    return base.model_copy(update=overrides) if overrides else base


def _pattern(findings=None, tech_debt=None) -> PatternReport:
    return PatternReport(findings=findings or [], tech_debt=tech_debt or [])


def _finding(severity="blocker", file="litellm/main.py") -> PatternFinding:
    return PatternFinding(
        file=file,
        severity=severity,
        source="docs" if severity == "blocker" else "code",
        citation="docs/foo.md#bar" if severity == "blocker" else "litellm/sibling.py",
        rationale="rationale",
    )


# --- Score / verdict matrix ---------------------------------------------------


def test_clean_pr_scores_5_and_is_ready():
    card = fuse(_triage(), _pattern())
    assert card.score == 5
    assert card.verdict == "READY"
    assert card.emoji == "✅"
    assert card.verdict_one_liner == "Ready to ship."


def test_pr_related_failure_docks_2():
    card = fuse(_triage(pr_related_failures=["test-foo"]), _pattern())
    assert card.score == 3
    assert card.verdict == "BLOCKED"
    assert "PR-related" in card.verdict_one_liner


def test_blocker_finding_docks_2():
    card = fuse(_triage(), _pattern(findings=[_finding("blocker")]))
    assert card.score == 3
    assert card.verdict == "BLOCKED"
    assert "pattern blocker" in card.verdict_one_liner


def test_low_greptile_docks_1():
    card = fuse(_triage(greptile_score=3), _pattern())
    assert card.score == 4
    assert card.verdict == "BLOCKED"


def test_null_greptile_docks_1():
    card = fuse(_triage(greptile_score=None), _pattern())
    assert card.score == 4
    assert "Greptile has not reviewed" in card.justification


def test_missing_circleci_docks_1():
    """The PR #26506 case from the design discussion: code is fine, only
    penalty is that CircleCI didn't run."""
    card = fuse(_triage(has_circleci_checks=False), _pattern())
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "CircleCI" in card.justification


def test_suggestions_and_nits_do_not_dock():
    card = fuse(
        _triage(),
        _pattern(findings=[_finding("suggestion"), _finding("nit")]),
    )
    assert card.score == 5
    assert card.verdict == "READY"


def test_running_checks_force_waiting_regardless_of_score():
    card = fuse(_triage(running_checks=["test-bar"]), _pattern())
    # WAITING wins over READY even when score would otherwise be 5.
    assert card.verdict == "WAITING"
    assert card.emoji == "⏳"
    assert "still running" in card.verdict_one_liner


def test_running_checks_overrides_blocked_too():
    card = fuse(
        _triage(running_checks=["test-bar"], pr_related_failures=["test-foo"]),
        _pattern(),
    )
    assert card.verdict == "WAITING"


def test_score_floors_at_zero():
    """Worst case: every penalty fires, score should clip at 0 not go negative."""
    card = fuse(
        _triage(
            pr_related_failures=["a"],
            greptile_score=2,
            has_circleci_checks=False,
        ),
        _pattern(findings=[_finding("blocker"), _finding("suggestion")]),
    )
    # 5 - 2 (pr_related) - 2 (blocker) - 1 (greptile<4) - 1 (no circleci) = -1, clipped to 0.
    # Suggestion has weight 0 in current rubric.
    assert card.score == 0
    assert card.verdict == "BLOCKED"


def test_unrelated_failures_do_not_dock_but_appear_in_justification():
    card = fuse(
        _triage(unrelated_failures=["Verify PR source branch"]),
        _pattern(),
    )
    # No score penalty for unrelated failures themselves; only missing CI etc.
    # Here everything else is clean so score stays 5 → READY.
    assert card.score == 5
    assert card.verdict == "READY"


# --- Rendering ----------------------------------------------------------------


def test_render_card_has_required_sections_in_order():
    card = fuse(_triage(), _pattern())
    text = render_card(card)
    s = text.find("*Triage Summary*")
    m = text.find("*Merge Confidence:")
    j = text.rfind("All checks green")
    assert 0 <= s < m < j, f"sections out of order: {text!r}"


def test_render_card_score_format():
    card = fuse(_triage(has_circleci_checks=False), _pattern())
    text = render_card(card)
    assert "Merge Confidence: 4/5" in text
    assert "❌ BLOCKED" in text


def test_render_drilldown_lists_each_failure_class():
    triage = _triage(
        pr_related_failures=["test-a"],
        unrelated_failures=["policy-b"],
        running_checks=["test-c"],
    )
    pattern = _pattern(
        findings=[_finding("blocker", "litellm/x.py")],
        tech_debt=[],
    )
    text = render_drilldown(triage, pattern)
    assert "PR-related failures" in text and "test-a" in text
    assert "Unrelated failures" in text and "policy-b" in text
    assert "Still running" in text and "test-c" in text
    assert "Pattern findings" in text and "litellm/x.py" in text


def test_render_drilldown_says_nothing_when_clean():
    text = render_drilldown(_triage(), _pattern())
    assert "Nothing to drill into" in text


def test_fallback_card_keeps_card_shape():
    text = render_fallback_card("https://github.com/org/repo/pull/1", "boom")
    assert "*Triage Summary*" in text
    assert "*Merge Confidence: ?/5*" in text
    assert "boom" in text


# --- Schema validation --------------------------------------------------------


def test_pr_summary_rejects_markdown_bold():
    """The validator should reject ** so the model gets retry feedback instead
    of silently shipping bold prose to Slack."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TriageReport(
            pr_number=1,
            pr_title="x",
            pr_author="a",
            pr_summary="this is **bold** which is banned",
            has_circleci_checks=True,
        )
