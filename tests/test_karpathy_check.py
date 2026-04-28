"""Unit tests for the new `karpathy_check` module.

The karpathy-check stage is a Claude-Code-driven senior-engineer pre-merge
review (skill at skills/pr-review-agent-skills/litellm-karpathy-check/SKILL.md).
It runs after the rest of the pipeline already leans READY and second-guesses
that verdict by asking "is this safe at 10k+ RPS?" via a `claude` subprocess.

What this file pins:
  - the pure helpers (`_extract_json_line`, `_linked_issue_numbers`) so the
    last-line-JSON parsing and `Fixes #N` extraction can't silently drift,
  - the two short-circuit paths that must NEVER spawn a claude subprocess
    (`KARPATHY_CHECK_ENABLED=false` and missing `claude` on PATH),
  - the Pydantic round-trip + breadth-enum validation so a schema typo on
    the model output is caught at parse time, not at fuse() time.

Everything here is pure: no LLM call, no network, no real subprocess. The
two short-circuit tests assert on a mocked `asyncio.create_subprocess_exec`
to prove the early-out actually short-circuits before reaching subprocess.

`pytest.importorskip("karpathy_check")` is at module top so this file
collects cleanly on a checkout where the parallel agent hasn't committed
`karpathy_check.py` yet — the tests skip rather than error.
"""

from __future__ import annotations

import os

os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")
os.environ.setdefault("LITELLM_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test")

import asyncio  # noqa: E402
import logging  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

karpathy_check = pytest.importorskip("karpathy_check")

from karpathy_check import (  # noqa: E402
    KarpathyFinding,
    KarpathyMergeGate,
    KarpathyReview,
    run_karpathy_check,
)


# --- _extract_json_line ------------------------------------------------------
# Helper that scans a Claude reply (which may contain prose, multiple JSON
# candidates, or no JSON at all) and returns the LAST valid JSON object
# encoded as a single line. Modeled on
# scripts/run_karpathy_check_eval.py::_extract_json_line — when the parallel
# agent ports it into the module, the contract pinned here is the contract
# the eval script already relies on.


def _extract():
    """Resolve the JSON-line helper by name, falling back to a sibling name
    if the parallel agent renamed it. Keeps the test from being brittle to
    a cosmetic rename — the contract pinned here is the BEHAVIOR, not the
    function name."""
    for name in ("_extract_json_line", "_extract_last_json_line", "extract_json_line"):
        fn = getattr(karpathy_check, name, None)
        if callable(fn):
            return fn
    pytest.skip("karpathy_check has no _extract_json_line-like helper yet")


def test_extract_json_line_plain_json_at_end_of_multiline_reply():
    """The simplest happy path: prose followed by a single JSON line on its
    own. Helper returns the parsed dict."""
    fn = _extract()
    reply = (
        "Sure, here is my analysis.\n"
        "I read three files.\n"
        '{"merge_gate": {"safe_for_high_rps_gateway": "yes"}, "findings": []}'
    )
    out = fn(reply)
    assert isinstance(out, dict)
    assert out["merge_gate"]["safe_for_high_rps_gateway"] == "yes"


def test_extract_json_line_finds_json_inside_surrounding_prose():
    """Trailing prose AFTER the JSON line shouldn't hide it — the helper
    walks lines from the end and returns the first parseable one. Catches
    chatty Claude replies where it appends a closing remark."""
    fn = _extract()
    reply = (
        "OK I'm done.\n"
        '{"merge_gate": {"safe_for_high_rps_gateway": "no"}, "findings": []}\n'
        "Hope that helps!\n"
    )
    out = fn(reply)
    assert out is not None, "trailing prose hid the JSON line"
    assert out["merge_gate"]["safe_for_high_rps_gateway"] == "no"


def test_extract_json_line_returns_last_valid_when_multiple_candidates():
    """When the reply contains multiple JSON candidates (e.g. an example
    schema earlier in the prose, then the real result at the end), the
    helper must return the LAST one — that's the one the skill instructs
    the model to print as the final terminal line."""
    fn = _extract()
    reply = (
        '{"example": "from earlier in the prose", "findings": []}\n'
        "Now the real output:\n"
        '{"merge_gate": {"safe_for_high_rps_gateway": "yes"}, "findings": []}'
    )
    out = fn(reply)
    assert out is not None
    # The skill schema wraps merge_gate; the example object didn't. If the
    # helper returned the earlier candidate this assertion catches it.
    assert "merge_gate" in out, f"got the wrong JSON candidate: {out!r}"


def test_extract_json_line_returns_none_when_no_json():
    """Pure prose reply (e.g. claude refused, or returned an apology) —
    helper returns None so the caller can short-circuit cleanly instead of
    raising. Important for robustness: a None karpathy result must NOT
    block the pipeline."""
    fn = _extract()
    reply = "I cannot help with that request.\nPlease try again later."
    assert fn(reply) is None


def test_extract_json_line_skips_malformed_returns_next_valid():
    """A malformed JSON candidate (truncated, missing brace) is skipped
    silently — the helper keeps walking back and returns the next valid
    line. Catches the case where Claude's output got cut off mid-token
    and the salvageable JSON sits one line above the broken one."""
    fn = _extract()
    reply = (
        '{"findings": [{"breadth": "narrow_correct"}]}\n'
        '{"this is broken: "no closing'
    )
    out = fn(reply)
    assert out is not None, "malformed-then-valid case returned None"
    assert out["findings"][0]["breadth"] == "narrow_correct"


# --- _linked_issue_numbers ---------------------------------------------------
# Extracts the integer issue numbers from a PR body's `Fixes #N` /
# `Closes https://.../issues/N` / `Resolves #N` markers. Used to fetch the
# linked issue text so the karpathy-check skill can compare diff-vs-ticket.
# Mirrors scripts/run_karpathy_check_eval.py::_linked_issue_numbers — the
# tests here pin the contract that script already relies on.


def _linked():
    for name in (
        "_linked_issue_numbers",
        "linked_issue_numbers",
        "_extract_linked_issues",
    ):
        fn = getattr(karpathy_check, name, None)
        if callable(fn):
            return fn
    pytest.skip("karpathy_check has no _linked_issue_numbers-like helper yet")


def test_linked_issue_short_form():
    """Bare `Fixes #123` form — the most common in PR bodies. Returns the
    integer (not the string), matching the eval-script contract."""
    fn = _linked()
    assert fn("Fixes #123") == [123]


def test_linked_issue_full_url_form():
    """GitHub auto-linking only fires on the bare form when the repo is
    the same; cross-repo or copy-pasted-from-issue links use the full URL.
    Helper must catch both shapes."""
    fn = _linked()
    out = fn("Closes https://github.com/BerriAI/litellm/issues/26586")
    assert out == [26586]


def test_linked_issue_mixed_and_deduplicated():
    """A PR body that links the same issue twice (once short-form, once
    URL) must NOT yield duplicates — the helper de-dupes so downstream
    `gh issue view` is called once per unique issue, not per mention."""
    fn = _linked()
    body = (
        "Fixes #100\n"
        "Also closes https://github.com/BerriAI/litellm/issues/200\n"
        "Resolves #100 (mentioned again above)\n"
    )
    out = fn(body)
    assert sorted(out) == [100, 200], f"expected dedup [100, 200], got {out!r}"


def test_linked_issue_empty_body_returns_empty_list():
    """No body / no markers = no linked issues. Helper returns `[]`, not
    None — caller code can iterate without a None-guard."""
    fn = _linked()
    assert fn("") == []
    assert fn("Some PR description with no issue references at all.") == []


def test_linked_issue_case_insensitive_keywords():
    """`FIXES`, `Closes`, `resolves`, mixed case — all valid GitHub
    closing keywords. The helper matches case-insensitively so a body
    written in shouty caps still links its issues."""
    fn = _linked()
    body = (
        "FIXES #1\n"
        "Closes #2\n"
        "resolves #3\n"
        "Resolves https://github.com/BerriAI/litellm/issues/4\n"
    )
    out = sorted(fn(body))
    assert out == [1, 2, 3, 4], f"case-insensitive match failed: {out!r}"


# --- Disabled flag short-circuit ---------------------------------------------
# KARPATHY_CHECK_ENABLED=false is the kill-switch for production: when set,
# `run_karpathy_check` returns None immediately WITHOUT spawning a claude
# subprocess. We assert on a mocked subprocess to prove the short-circuit
# fires before the spawn (a flaky / mis-implemented kill-switch that still
# spawned the subprocess would burn the eval budget for nothing).


def test_disabled_flag_returns_none_and_does_not_spawn_subprocess(monkeypatch):
    """KARPATHY_CHECK_ENABLED=false is the cost kill-switch. Must short-
    circuit before any subprocess is spawned, otherwise toggling it off
    in production wouldn't actually stop the spend. Patch
    `asyncio.create_subprocess_exec` so a mis-implemented kill-switch
    that still spawned would be caught by the assert_not_called below.

    Sync test calling asyncio.run() to match the project's test
    convention (see tests/test_mention.py); avoids pulling in a
    pytest-asyncio dep just for two tests."""
    monkeypatch.setenv("KARPATHY_CHECK_ENABLED", "false")
    spawn = AsyncMock()
    with patch("asyncio.create_subprocess_exec", spawn):
        result = asyncio.run(
            run_karpathy_check("https://github.com/BerriAI/litellm/pull/1")
        )
    assert result is None, f"disabled flag should return None, got {result!r}"
    spawn.assert_not_called()


# --- Missing `claude` CLI short-circuit --------------------------------------
# Even with the flag enabled, a checkout without the `claude` binary on PATH
# (e.g. a CI runner that doesn't install Claude Code) must not crash. We
# return None and emit a structured-log line so an operator can see WHY the
# stage was skipped without scrolling through traceback noise.


def test_missing_claude_cli_returns_none_and_logs(monkeypatch, caplog):
    """No `claude` on PATH → return None, log `karpathy_skipped
    reason=no_claude`. Operators see the reason in structured logs and the
    pipeline continues without the karpathy stage. Subprocess must NOT be
    spawned (we'd just get a FileNotFoundError if it did).

    Sync test calling asyncio.run() to match the project's test
    convention (see tests/test_mention.py); avoids pulling in a
    pytest-asyncio dep just for two tests."""
    monkeypatch.setenv("KARPATHY_CHECK_ENABLED", "true")
    # Patch which on the karpathy module so the test isolates "is there
    # a claude binary?" from anything else the module imports. Patching
    # `shutil.which` directly works too but is a wider blast radius.
    monkeypatch.setattr("karpathy_check.shutil.which", lambda name: None)

    spawn = AsyncMock()
    with caplog.at_level(logging.WARNING, logger="litellm-bot.karpathy_check"), patch(
        "asyncio.create_subprocess_exec", spawn
    ):
        result = asyncio.run(
            run_karpathy_check("https://github.com/BerriAI/litellm/pull/1")
        )

    assert result is None, "missing claude should short-circuit to None"
    spawn.assert_not_called()
    log_text = caplog.text
    assert "karpathy_skipped" in log_text, (
        f"expected karpathy_skipped log line, got:\n{log_text}"
    )
    assert "no_claude" in log_text, (
        f"expected reason=no_claude in log line, got:\n{log_text}"
    )


# --- Pydantic round-trip -----------------------------------------------------
# The KarpathyReview / KarpathyFinding / KarpathyMergeGate triple is
# persisted as JSONB in `runs.karpathy_check` (see migrations/0002). The
# round-trip test pins that model_dump(mode="json") output is stable and
# re-parseable, so a schema edit that breaks JSONB-level back-compat is
# caught here instead of at the next DB read.


def test_karpathy_review_pydantic_round_trip():
    """Build → dump → re-parse → equality. Pins JSON serialization stability
    for the runs.karpathy_check JSONB column. If a future field gets a
    non-JSON-default (e.g. datetime, enum without `use_enum_values`) this
    test catches the round-trip break."""
    finding = KarpathyFinding(
        regression_archetype="narrow_fix_missed_class",
        bug_class="image_generation reads access groups but other modes don't",
        fix_locus="litellm/proxy/auth/model_checks.py:get_complete_model_list",
        sibling_loci=[],
        evidence=[
            "litellm/proxy/auth/model_checks.py:67-92 — only image_generation guarded"
        ],
        breadth="narrow_missed_class",
        recommended_fix=(
            "Widen the access-group check to all non-chat dispatch entries; "
            "split fine-tune resolution into its own PR."
        ),
    )
    review = KarpathyReview(
        linked_issue="BerriAI/litellm#26586",
        fix_shapes=["param_filter_or_allowlist"],
        merge_gate=KarpathyMergeGate(
            safe_for_high_rps_gateway="no",
            one_liner="Hold: narrow guard misses non-chat dispatch.",
            unintended_consequences=[
                "embeddings + image_generation still leak access groups",
            ],
            hot_path_notes=[],
            what_would_make_yes=(
                "Widen the guard to every non-chat entry in DISPATCH; add "
                "regression test exercising at least two endpoints."
            ),
        ),
        findings=[finding],
    )

    dumped = review.model_dump(mode="json")
    rehydrated = KarpathyReview.model_validate(dumped)
    assert rehydrated == review, (
        f"round-trip lost data:\nbefore: {review!r}\nafter:  {rehydrated!r}"
    )


# --- Schema validation: breadth enum -----------------------------------------
# breadth is a closed enum (10 values from the SKILL.md table). A typo or
# made-up value coming back from Claude must be rejected at parse time so
# fuse() can rely on the value being one of the documented set.


def test_karpathy_finding_rejects_unknown_breadth():
    """`breadth` is a closed Literal of the 10 SKILL.md values. A bogus
    value (typo / hallucinated category) must raise ValidationError at
    parse time. The downstream rubric in fuse() trusts the breadth string,
    so silent acceptance of a bogus value would mis-score the card."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KarpathyFinding(
            regression_archetype="narrow_fix_missed_class",
            bug_class="example",
            fix_locus="litellm/x.py:foo",
            sibling_loci=[],
            evidence=["litellm/x.py:1-2 — note"],
            breadth="bogus",  # not in the closed enum
            recommended_fix="example",
        )
