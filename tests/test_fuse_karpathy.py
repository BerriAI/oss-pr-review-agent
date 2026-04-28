"""Unit tests for the karpathy-extended `fuse()` signature.

These tests pin the contract for the upcoming `fuse(triage, pattern,
karpathy=None)` extension. The parallel agent is adding:
  - a third positional/keyword arg `karpathy: KarpathyReview | None = None`
  - new rubric rows that dock the score when `karpathy.merge_gate
    .safe_for_high_rps_gateway` is "no" (block) or "conditional" (soft dock)

Until that change lands, every test in this module is xfail-or-skip — the
file is a SPEC the agent will satisfy. We assert on RELATIVE behavior
(verdict + score deltas vs. the no-karpathy baseline) rather than absolute
weights wherever the spec didn't pin a number, so the tests survive small
weight tuning.

The module skips cleanly at collection time if `karpathy_check` isn't
importable yet (parallel agent hasn't committed). Same pattern as
test_karpathy_check.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")
os.environ.setdefault("LITELLM_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test")

import pytest  # noqa: E402

karpathy_check = pytest.importorskip("karpathy_check")

from karpathy_check import (  # noqa: E402
    KarpathyFinding,
    KarpathyMergeGate,
    KarpathyReview,
)

from app import (  # noqa: E402
    PatternReport,
    TriageReport,
    fuse,
)


# --- Fixtures ----------------------------------------------------------------
# Mirror tests/test_fuse.py's helpers (_triage / _pattern). Kept in this file
# rather than importing from test_fuse.py so the two files don't grow a
# fixtures-import chain that pytest's collection ordering could trip over.


def _triage(**overrides) -> TriageReport:
    """Default = a clean PR that produces score=5 READY in fuse(). Same
    shape as tests/test_fuse.py::_triage so the karpathy-extension tests
    behave exactly like the existing ones when no karpathy is passed."""
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


def _pattern() -> PatternReport:
    return PatternReport(findings=[], tech_debt=[])


def _merge_gate(
    safe: str = "yes",
    one_liner: str = "Merge — no production risk found.",
    consequences: list[str] | None = None,
    what_would_make_yes: str = "",
) -> KarpathyMergeGate:
    return KarpathyMergeGate(
        safe_for_high_rps_gateway=safe,
        one_liner=one_liner,
        unintended_consequences=consequences or [],
        hot_path_notes=[],
        what_would_make_yes=what_would_make_yes,
    )


def _narrow_missed_finding(
    bug_class: str = "image_generation reads access groups; other modes don't",
    fix_locus: str = "litellm/proxy/auth/model_checks.py:get_complete_model_list",
) -> KarpathyFinding:
    return KarpathyFinding(
        regression_archetype="narrow_fix_missed_class",
        bug_class=bug_class,
        fix_locus=fix_locus,
        sibling_loci=[],
        evidence=[
            "litellm/proxy/auth/model_checks.py:67-92 — only image_generation guarded"
        ],
        breadth="narrow_missed_class",
        recommended_fix=(
            "Widen guard to all non-chat dispatch entries; add regression "
            "test exercising at least two endpoints."
        ),
    )


def _supports_karpathy_arg() -> bool:
    """Probe whether `fuse()` already accepts a karpathy arg. Until the
    parallel agent lands the signature change, the karpathy tests xfail
    instead of erroring — so a plain `pytest` run on a half-merged tree
    doesn't go red just because the dependent edit isn't in yet.

    Cached per process via a lru_cache wrapper would be tidier, but a
    bare module-level call works fine for ~5 tests."""
    import inspect

    try:
        sig = inspect.signature(fuse)
    except (TypeError, ValueError):
        return False
    return "karpathy" in sig.parameters


_KARPATHY_NOT_WIRED = pytest.mark.xfail(
    not _supports_karpathy_arg(),
    reason=(
        "fuse() does not yet accept a karpathy arg. The parallel agent's "
        "edit to app.py hasn't landed; this test pins the planned API and "
        "will pass once fuse(triage, pattern, karpathy=...) is wired up."
    ),
    strict=False,
)


# --- (a) Default behavior preserved -----------------------------------------
# The whole point of `karpathy: KarpathyReview | None = None` is that
# existing callers must not have to change. Three spellings — no arg,
# explicit None, and an empty review — all have to land on the same card
# the no-karpathy baseline produces. Otherwise we've silently broken every
# fuse() callsite that doesn't pass karpathy.


@_KARPATHY_NOT_WIRED
def test_fuse_no_karpathy_arg_matches_explicit_none_and_empty_review():
    """Three spellings, one card. `fuse(t, p)` == `fuse(t, p, None)` ==
    `fuse(t, p, KarpathyReview())`. Pins back-compat for every callsite
    that doesn't pass karpathy (i.e. all current callsites)."""
    triage, pattern = _triage(), _pattern()

    baseline = fuse(triage, pattern)
    explicit_none = fuse(triage, pattern, None)
    empty_review = fuse(triage, pattern, KarpathyReview())

    # Sanity check: the no-karpathy baseline is the clean 5/READY case
    # documented in tests/test_fuse.py::test_clean_pr_scores_5_and_is_ready.
    assert baseline.score == 5
    assert baseline.verdict == "READY"

    assert explicit_none == baseline, (
        f"explicit None diverges from no-arg:\n"
        f"  baseline: {baseline!r}\n  explicit: {explicit_none!r}"
    )
    assert empty_review == baseline, (
        f"empty KarpathyReview diverges from no-arg:\n"
        f"  baseline: {baseline!r}\n  empty:    {empty_review!r}"
    )


# --- (b) safe_for_high_rps_gateway="no" blocks -------------------------------
# A "no" verdict from the karpathy stage is the strongest signal we have:
# a staff-engineer look at the diff against the real tree said "do not
# merge." It must flip the card to BLOCKED and dock the score below 5,
# regardless of how clean the rest of the pipeline came back.


@_KARPATHY_NOT_WIRED
def test_safe_no_blocks_even_on_clean_pipeline():
    """karpathy says 'no' → BLOCKED, score < 5. Pins the strongest gate
    signal: even if triage + pattern came back clean, a karpathy 'no'
    must flip the verdict. Asserts on RELATIVE score (< baseline, not =
    a specific number) so small weight tuning doesn't break the test —
    we just need to know the dock fired."""
    triage, pattern = _triage(), _pattern()
    baseline = fuse(triage, pattern)
    assert baseline.score == 5  # sanity

    review = KarpathyReview(
        merge_gate=_merge_gate(
            safe="no",
            one_liner="Hold: narrow guard misses non-chat dispatch.",
        ),
        findings=[_narrow_missed_finding()],
    )
    card = fuse(triage, pattern, review)

    assert card.verdict == "BLOCKED", (
        f"karpathy 'no' must flip READY → BLOCKED, got {card.verdict}"
    )
    assert card.score < baseline.score, (
        f"karpathy 'no' must dock score below baseline {baseline.score}, "
        f"got {card.score}"
    )


# --- (c) safe_for_high_rps_gateway="conditional" docks but doesn't kill -----
# "conditional" means "merge after X" — it should dock the score (so the
# reviewer sees the open ask) but the rubric doesn't have to clip to 0.
# The exact weight is what the rubric in app.py decides; we assert on
# the SHAPE (verdict=BLOCKED, score strictly between 0 and baseline)
# rather than a specific weight so weight tuning doesn't break the test.


@_KARPATHY_NOT_WIRED
def test_safe_conditional_docks_but_does_not_clip_to_zero():
    """karpathy says 'conditional' → BLOCKED with a soft dock. Score must
    be strictly between 0 and baseline so the reviewer sees both the
    open ask AND that the rest of the diff is healthy.

    NOTE: the exact penalty weight is set by the new rubric rows in
    app.py and may be 1, 2, or some other value. This test asserts on
    relative position (0 < score < baseline) instead of a fixed number
    so weight tuning doesn't break it; if you want to pin a specific
    weight, add a separate test once the value stabilizes."""
    triage, pattern = _triage(), _pattern()
    baseline = fuse(triage, pattern)

    review = KarpathyReview(
        merge_gate=_merge_gate(
            safe="conditional",
            one_liner="Merge after a regression test for embeddings is added.",
            what_would_make_yes=(
                "Add a test exercising at least one non-chat endpoint that "
                "uses access groups."
            ),
        ),
        findings=[_narrow_missed_finding()],
    )
    card = fuse(triage, pattern, review)

    assert card.verdict == "BLOCKED", (
        f"karpathy 'conditional' must flip READY → BLOCKED, got {card.verdict}"
    )
    assert 0 < card.score < baseline.score, (
        f"karpathy 'conditional' should soft-dock (0 < score < {baseline.score}), "
        f"got {card.score}"
    )


# --- (d) safe_for_high_rps_gateway="yes" with no findings is a no-op --------
# A clean karpathy verdict (yes + empty findings) is the no-signal case —
# it must not change the card vs. not passing karpathy at all. This is the
# common path for boring PRs (typo fixes, doc updates, dependency bumps)
# where karpathy ran but found nothing to flag.


@_KARPATHY_NOT_WIRED
def test_safe_yes_with_empty_findings_is_no_op():
    """yes + no findings = same card as no karpathy at all. Pins that the
    rubric doesn't accidentally penalize the act of *running* karpathy
    on a clean PR (would be silly, but easy to introduce by mis-coding
    the predicate as `karpathy is not None` instead of `karpathy.merge_gate
    .safe_for_high_rps_gateway in ('no', 'conditional')`)."""
    triage, pattern = _triage(), _pattern()
    baseline = fuse(triage, pattern)

    review = KarpathyReview(
        merge_gate=_merge_gate(safe="yes"),
        findings=[],
    )
    card = fuse(triage, pattern, review)

    assert card == baseline, (
        f"karpathy 'yes' + empty findings must match no-karpathy baseline:\n"
        f"  baseline: {baseline!r}\n  yes-empty: {card!r}"
    )


# --- (e) Justification carries karpathy reason when docked ------------------
# When the rubric docks for karpathy, the user must see WHY in the card
# prose — either the literal "karpathy" word or the merge_gate one_liner
# verbatim. Otherwise a reviewer reads "BLOCKED 3/5" with no context for
# what the karpathy stage said, and has to click through to the run JSON
# in the UI to find the gate verdict.


@_KARPATHY_NOT_WIRED
def test_justification_or_one_liner_carries_karpathy_signal_when_docked():
    """When karpathy docks, the merge_gate.one_liner (or the word
    'karpathy') must surface in either the verdict_one_liner or the
    justification — otherwise the reviewer sees a docked score with no
    explanation. We accept either field because the exact wiring depends
    on the rubric's _compose_one_liner / _compose_justification edits the
    parallel agent makes."""
    triage, pattern = _triage(), _pattern()
    one_liner = "Hold: narrow guard misses non-chat dispatch at high QPS."
    review = KarpathyReview(
        merge_gate=_merge_gate(safe="no", one_liner=one_liner),
        findings=[_narrow_missed_finding()],
    )
    card = fuse(triage, pattern, review)

    # Reviewer must see SOME karpathy context, in EITHER the headline or
    # the longer justification. We accept three substrings to give the
    # rubric impl flexibility on phrasing:
    #   - the literal merge_gate one_liner (if the rubric inlines it),
    #   - the first few words of the one_liner (if it's truncated),
    #   - the word "karpathy" (if the rubric labels the row by source).
    haystack = (card.verdict_one_liner + "\n" + card.justification).lower()
    one_liner_head = one_liner.split(":")[0].lower()  # "hold"
    one_liner_substr = "narrow guard"
    assert (
        one_liner.lower() in haystack
        or one_liner_substr in haystack
        or "karpathy" in haystack
        or one_liner_head in haystack
    ), (
        f"karpathy dock left no trace in card prose:\n"
        f"  verdict_one_liner: {card.verdict_one_liner!r}\n"
        f"  justification:     {card.justification!r}"
    )
