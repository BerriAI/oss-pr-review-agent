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


def _finding(
    severity="blocker", file="litellm/main.py", risk="low"
) -> PatternFinding:
    return PatternFinding(
        file=file,
        severity=severity,
        risk=risk,
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


def test_missing_circleci_does_not_dock_oss_pr():
    """OSS PRs from external contributors often can't run CircleCI (gated on
    repo secrets). Don't penalize them for it — the skill makes the same call.
    PR #26506 from the design discussion is the canonical example."""
    card = fuse(_triage(has_circleci_checks=False), _pattern())
    assert card.score == 5
    assert card.verdict == "READY"
    # CI status still surfaces in the justification so reviewers see it.
    assert "no CircleCI" in card.justification


def test_suggestions_and_nits_do_not_dock():
    card = fuse(
        _triage(),
        _pattern(findings=[_finding("suggestion"), _finding("nit")]),
    )
    assert card.score == 5
    assert card.verdict == "READY"


# --- Risk-based docks (orthogonal to severity) --------------------------------
# Bug fixed here from the 2026-04-25 v2 eval: PRs #26294 (gated import that
# silently 404s existing routes) and #26074 (error_msg = None that masks
# failures) both got nit-severity findings and the old rubric ignored them.
# Risk lets fuse dock on impact regardless of evidence strength.


def test_high_risk_nit_finding_docks_2_and_blocks():
    """A nit-severity finding flagged risk=high (e.g. lazy import gating a
    public route, single sibling diverges) must dock 2 → BLOCKED. The whole
    point of decoupling risk from severity is that thin evidence + bad shape
    still has to block."""
    card = fuse(
        _triage(),
        _pattern(findings=[_finding(severity="nit", risk="high")]),
    )
    assert card.score == 3
    assert card.verdict == "BLOCKED"
    assert "high-risk pattern finding" in card.justification


def test_medium_risk_finding_docks_1_and_blocks():
    """Medium risk = internal-contract change (helper-signature drift,
    silent return-None instead of raise). Soft penalty, but enough to flip
    READY → BLOCKED so the reviewer sees it."""
    card = fuse(
        _triage(),
        _pattern(findings=[_finding(severity="nit", risk="medium")]),
    )
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "medium-risk pattern finding" in card.justification


def test_low_risk_findings_do_not_dock():
    """The default — risk=low matches pre-risk behavior. Cosmetic findings
    (naming, formatting, layout) shouldn't move the score regardless of
    severity."""
    card = fuse(
        _triage(),
        _pattern(
            findings=[
                _finding(severity="suggestion", risk="low"),
                _finding(severity="nit", risk="low"),
            ]
        ),
    )
    assert card.score == 5
    assert card.verdict == "READY"


def test_high_risk_one_liner_takes_priority_over_soft_penalties():
    """When the only blockers left are pattern-risk + soft-failures, the
    one-liner should call out the high-risk pattern finding by name — that's
    the most actionable signal for the reviewer."""
    card = fuse(
        _triage(greptile_score=3),  # also docks 1, but lower priority
        _pattern(findings=[_finding(severity="nit", risk="high")]),
    )
    assert card.verdict == "BLOCKED"
    assert "high-risk pattern finding" in card.verdict_one_liner


def test_high_and_medium_risk_findings_stack():
    """One high (-2) + one medium (-1) → score 2/5. Independent rubric rows,
    so both fire on the same report."""
    card = fuse(
        _triage(),
        _pattern(
            findings=[
                _finding(severity="nit", risk="high", file="litellm/a.py"),
                _finding(severity="nit", risk="medium", file="litellm/b.py"),
            ]
        ),
    )
    assert card.score == 2
    assert card.verdict == "BLOCKED"


def test_render_drilldown_sorts_high_risk_first_and_tags_risk():
    """Reviewer's eye should land on the dangerous findings before the naming
    nits, regardless of list order. Low-risk findings don't show the risk tag
    (it's the default — printing it on every line is noise)."""
    pattern = _pattern(
        findings=[
            _finding(severity="nit", risk="low", file="litellm/cosmetic.py"),
            _finding(severity="nit", risk="high", file="litellm/dangerous.py"),
        ]
    )
    text = render_drilldown(_triage(), pattern)
    high_pos = text.find("dangerous.py")
    low_pos = text.find("cosmetic.py")
    assert 0 <= high_pos < low_pos, f"high-risk not first: {text!r}"
    assert "risk=high" in text
    assert "risk=low" not in text


# --- Wide low-density fan-out -------------------------------------------------
# Bug from BerriAI/litellm PR #26284: 66-file diff applying urllib.parse.quote()
# inline at every callsite. Grader called it brittle because the next contributor
# adding a similar callsite without remembering the helper silently regresses
# the fix. Triage rubric now flags this shape directly from diff size metadata.


def test_wide_low_density_fanout_docks_1():
    """30+ files, <5 lines/file avg → fan-out dock. Picks up the inline-helper
    -everywhere shape that's easy to silently regress."""
    triage = _triage(files_changed=66, additions=120, deletions=10)
    card = fuse(triage, _pattern())
    # 130 lines / 66 files = 1.97 → triggers
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "wide low-density fan-out" in card.justification


def test_wide_fanout_does_not_fire_below_file_threshold():
    """A focused 25-file change with the same per-file density should NOT
    trigger — small diffs that touch a few callsites are normal refactors,
    not the brittle-fanout shape."""
    triage = _triage(files_changed=25, additions=40, deletions=10)
    card = fuse(triage, _pattern())
    assert card.score == 5
    assert card.verdict == "READY"


def test_wide_fanout_does_not_fire_when_density_high():
    """A 50-file diff that adds 40+ lines per file is a real refactor (logic,
    not find-replace) — don't penalize it. Threshold is <5 lines/file."""
    triage = _triage(files_changed=50, additions=2000, deletions=200)
    card = fuse(triage, _pattern())
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
    """Worst case: every score-affecting penalty fires, score clips at 0."""
    card = fuse(
        _triage(
            pr_related_failures=["a"],
            unrelated_failures=["b"],
            greptile_score=2,
            has_circleci_checks=False,
        ),
        _pattern(findings=[_finding("blocker"), _finding("suggestion")]),
    )
    # 5 - 2 (pr_related) - 2 (blocker) - 1 (greptile<4) - 1 (unrelated) = -1 → 0.
    # Suggestion and missing-CircleCI both have weight 0 in current rubric.
    assert card.score == 0
    assert card.verdict == "BLOCKED"


def test_merge_conflicts_force_blocked_even_on_otherwise_clean_pr():
    """Stored policy: a PR with merge conflicts cannot ship regardless of
    green CI / Greptile / patterns. Conflict rubric row weight is 5 so an
    otherwise-perfect PR clips to score 0 → BLOCKED."""
    card = fuse(_triage(has_merge_conflicts=True), _pattern())
    assert card.score == 0
    assert card.verdict == "BLOCKED"
    assert "Merge conflicts" in card.verdict_one_liner
    assert "merge conflicts" in card.failing_line


def test_merge_conflicts_unknown_does_not_block():
    """has_merge_conflicts=None means GitHub is still computing mergeability.
    We must NOT punish a PR for a slow background job — null is unknown, not
    'has conflicts'. Score should match the clean baseline (5/READY)."""
    card = fuse(_triage(has_merge_conflicts=None), _pattern())
    assert card.score == 5
    assert card.verdict == "READY"


def test_merge_conflicts_false_does_not_block():
    """Explicit clean-merge signal behaves identically to a clean PR — no
    penalty, READY verdict."""
    card = fuse(_triage(has_merge_conflicts=False), _pattern())
    assert card.score == 5
    assert card.verdict == "READY"
    assert "merge conflicts" not in card.failing_line


def test_merge_conflicts_dominate_one_liner_over_other_blockers():
    """Conflicts come first in the one-liner priority chain — a reviewer
    can't act on CI failures or pattern findings until the branch is mergeable
    again, so the rebase ask must be the headline."""
    card = fuse(
        _triage(
            has_merge_conflicts=True,
            pr_related_failures=["test-foo"],
        ),
        _pattern(findings=[_finding("blocker")]),
    )
    assert card.verdict == "BLOCKED"
    assert card.verdict_one_liner.startswith("Merge conflicts")


def test_render_drilldown_lists_merge_conflicts():
    triage = _triage(has_merge_conflicts=True)
    text = render_drilldown(triage, _pattern())
    assert "Merge state" in text
    assert "merge conflicts" in text


def test_failing_line_combines_conflicts_and_failing_checks():
    """Both signals appear on a single line, separated by ' · ', so the user
    sees both blockers on the card without scrolling to the drill-down."""
    triage = _triage(has_merge_conflicts=True, pr_related_failures=["test-foo"])
    card = fuse(triage, _pattern())
    assert "merge conflicts" in card.failing_line
    assert "test-foo" in card.failing_line
    assert " · " in card.failing_line


def test_unrelated_failures_now_dock_one_and_flip_to_blocked():
    """Bug from BerriAI/litellm PR #26451: a single failing `code-quality`
    check got bucketed as unrelated and the card silently shipped 5/5 READY.
    Fix: any unrelated CI failure docks 1 → 4/5 → BLOCKED, and the failure
    name lands in both the justification and the failing_line on the card."""
    card = fuse(
        _triage(unrelated_failures=["code-quality"]),
        _pattern(),
    )
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "code-quality" in card.justification
    assert "code-quality" in card.failing_line
    assert card.failing_line.startswith("⚠️")


# --- Unique-vs-elsewhere split for unrelated failures -------------------------
# Regression cases from the 2026-04-25 eval: PRs #26385 / #26011 / #26122 of
# BerriAI/litellm got 4/5 BLOCKED for `lint` / `codecov` failures that were
# also red on neighboring open PRs (clearly broken-for-everyone infra). The
# rubric now docks ONLY for unrelated failures unique to this PR, leaving the
# silent-pass canary armed for #26451-style cases.


def test_unrelated_failure_also_failing_elsewhere_does_not_dock():
    """Same `lint` failure red here AND on neighboring PRs → infra noise,
    not docking. Score stays 5, verdict READY. The check still surfaces in
    failing_line and justification so the reviewer sees we noticed."""
    card = fuse(
        _triage(
            unrelated_failures=["lint"],
            unrelated_failures_also_failing_elsewhere=["lint"],
        ),
        _pattern(),
    )
    assert card.score == 5
    assert card.verdict == "READY"
    assert "lint" in card.failing_line  # still shown, just not docked
    # Justification on a READY card with all-elsewhere failures should
    # honestly call out the also-red-on-other-PRs framing.
    assert (
        "neighboring" in card.justification.lower()
        or "broken-for-everyone" in card.justification.lower()
    )


def test_unrelated_failure_unique_to_pr_still_docks():
    """`code-quality` red here but NOT on neighbors → unique unrelated →
    docks 1 → 4/5 → BLOCKED. Pins the silent-pass canary in place after the
    split."""
    card = fuse(
        _triage(
            unrelated_failures=["code-quality"],
            unrelated_failures_also_failing_elsewhere=[],
        ),
        _pattern(),
    )
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "code-quality" in card.justification
    assert "unique to this PR" in card.justification


def test_mixed_unrelated_buckets_only_docks_for_unique_subset():
    """Two unrelated failures, one cross-repo and one unique. Should dock
    1 (for the unique one) and mention both in the justification — the
    cross-repo one labeled as no-penalty so the reviewer sees the
    distinction."""
    card = fuse(
        _triage(
            unrelated_failures=["lint", "weird-suite"],
            unrelated_failures_also_failing_elsewhere=["lint"],
        ),
        _pattern(),
    )
    assert card.score == 4
    assert card.verdict == "BLOCKED"
    assert "weird-suite" in card.justification
    assert "lint" in card.justification
    assert "no penalty" in card.justification.lower()


def test_ready_card_with_all_elsewhere_failures_keeps_failing_line():
    """READY + all unrelated failures are also-elsewhere: card must still
    list the check names in failing_line so the user sees red checks even
    on a READY verdict (the silent-pass guard from PR #26451)."""
    card = fuse(
        _triage(
            unrelated_failures=["lint", "codecov"],
            unrelated_failures_also_failing_elsewhere=["lint", "codecov"],
        ),
        _pattern(),
    )
    assert card.verdict == "READY"
    assert "lint" in card.failing_line
    assert "codecov" in card.failing_line


# --- Policy/meta failures (zero-penalty bucket) ------------------------------
# Regression case from the 2026-04-25 eval: PR #26419 of BerriAI/litellm got
# 4/5 BLOCKED because `Verify PR source branch` failed. That check is a repo
# policy ("PR not from main"), nothing to do with the diff. policy_meta_failures
# is its own zero-penalty bucket so the rubric ignores it but the card still
# surfaces the ask.


def test_policy_meta_failure_does_not_dock():
    """`Verify PR source branch` etc. should NOT dock the score. Card stays
    READY and the meta failure surfaces in failing_line under its own glyph."""
    card = fuse(
        _triage(policy_meta_failures=["Verify PR source branch"]),
        _pattern(),
    )
    assert card.score == 5
    assert card.verdict == "READY"
    assert "Verify PR source branch" in card.failing_line
    # Different glyph than the warning glyph used for real failures so the
    # eye doesn't read it as an error of equal weight.
    assert "ℹ️" in card.failing_line


def test_policy_meta_does_not_double_count_with_real_failures():
    """A real PR-related failure docks 2; a co-occurring policy-meta failure
    must NOT add an extra penalty. Score should match the no-policy baseline."""
    triage_with_policy = _triage(
        pr_related_failures=["test-foo"],
        policy_meta_failures=["DCO"],
    )
    triage_no_policy = _triage(pr_related_failures=["test-foo"])
    assert (
        fuse(triage_with_policy, _pattern()).score
        == fuse(triage_no_policy, _pattern()).score
    )


def test_render_drilldown_lists_policy_meta_failures():
    triage = _triage(policy_meta_failures=["Verify PR source branch", "DCO"])
    text = render_drilldown(triage, _pattern())
    assert "Policy / meta failures" in text
    assert "Verify PR source branch" in text
    assert "DCO" in text


# --- Rendering ----------------------------------------------------------------


def test_render_card_has_required_sections_in_order():
    card = fuse(_triage(), _pattern())
    text = render_card(card)
    s = text.find("*Triage Summary*")
    m = text.find("*Merge Confidence:")
    j = text.rfind("All checks green")
    assert 0 <= s < m < j, f"sections out of order: {text!r}"


def test_render_card_score_format():
    card = fuse(_triage(greptile_score=3), _pattern())
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


# --- Diff size on the card ----------------------------------------------------


def test_render_card_includes_size_line_when_diff_known():
    """Reviewers should see PR scale (lines + files) before deciding whether
    to dig in. The size line sits between the summary and the merge confidence."""
    triage = _triage(files_changed=3, additions=42, deletions=8)
    text = render_card(fuse(triage, _pattern()))
    assert "50 lines across 3 files (+42 / -8)" in text
    s = text.find("*Triage Summary*")
    sz = text.find("50 lines")
    m = text.find("*Merge Confidence:")
    assert 0 <= s < sz < m, f"size line out of order: {text!r}"


def test_render_card_singularizes_one_file_one_line():
    """Plural helper should say "1 line across 1 file", not "1 lines across 1 files"."""
    triage = _triage(files_changed=1, additions=1, deletions=0)
    text = render_card(fuse(triage, _pattern()))
    assert "1 line across 1 file (+1 / -0)" in text


def test_render_card_omits_size_line_when_unknown():
    """Default TriageReport has size fields = 0 (e.g. older callers, fallback
    path). Don't print a misleading "0 lines" — just skip the line."""
    text = render_card(fuse(_triage(), _pattern()))
    assert "lines across" not in text
    assert "files (+" not in text


# --- Failing-checks call-out --------------------------------------------------
# The card must NEVER silently swallow a failing check, even one the model
# bucketed as "unrelated". Regression bar from BerriAI/litellm PR #26451.


def test_failing_line_lists_pr_related_checks():
    triage = _triage(pr_related_failures=["test-foo", "lint"])
    card = fuse(triage, _pattern())
    assert "test-foo" in card.failing_line
    assert "lint" in card.failing_line
    assert "2 checks failing" in card.failing_line


def test_failing_line_lists_unrelated_checks():
    triage = _triage(unrelated_failures=["code-quality"])
    card = fuse(triage, _pattern())
    assert "code-quality" in card.failing_line
    assert "1 check failing" in card.failing_line


def test_failing_line_combines_both_buckets():
    triage = _triage(
        pr_related_failures=["test-foo"],
        unrelated_failures=["code-quality"],
    )
    card = fuse(triage, _pattern())
    assert "test-foo" in card.failing_line
    assert "code-quality" in card.failing_line
    assert "2 checks failing" in card.failing_line


def test_failing_line_empty_when_clean():
    card = fuse(_triage(), _pattern())
    assert card.failing_line == ""


def test_render_card_includes_failing_line_above_one_liner():
    """Failing-line must sit between the verdict and the verdict_one_liner so
    the reader sees check names immediately after the score."""
    triage = _triage(unrelated_failures=["code-quality"])
    text = render_card(fuse(triage, _pattern()))
    m = text.find("*Merge Confidence:")
    f = text.find("⚠️ 1 check failing: code-quality")
    o = text.find("Unrelated CI failure", f) if f >= 0 else -1
    assert 0 <= m < f, f"failing line missing or out of order: {text!r}"
    # one-liner appears after failing_line
    assert f < o or o == -1


def test_render_card_omits_failing_line_when_clean():
    text = render_card(fuse(_triage(), _pattern()))
    assert "checks failing" not in text
    assert "check failing" not in text


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
