"""Focused triage-classifier regression tests.

Each test in this module pins a specific real-world misclassification we
have seen in production, using a captured `gather_pr_data` payload as the
input. The agent is run live (real LLM, real prompt, real SKILL.md) but
the gather subprocess is monkeypatched to return the fixture, so the test
is deterministic on the input side and runs in ~10s for ~$0.02.

Why not use the live eval suite for this:
  - The eval suite (`tests/eval/`) runs the agent against ~20 real PRs
    and grades the *aggregate* behavior. It does NOT pin specific
    failure shapes — if one PR drifts from READY to BLOCKED, the human
    grader notices in `summary.md`, but nothing is asserted in code.
  - These tests assert ONE specific classification on ONE specific
    payload shape. When the underlying bug recurs (someone tweaks the
    SKILL prompt and the model goes back to over-trusting annotations),
    the test fails and names the regression.
  - Captured fixtures don't go stale here the way they do in the live
    eval — we WANT the input frozen. The test exists to lock in the
    classifier's response to a specific input shape.

Gated on `@pytest.mark.triage_regression` so a plain `pytest` doesn't
fire it (live LLM, costs money, needs creds). Opt in with
`pytest -m triage_regression`.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same env-shim as `tests/test_fuse.py` — lets `import app` work without
# a real .env, then we re-load .env below for the live-LLM path. Defaults
# only apply if not already set, so a real .env-loaded creds dict isn't
# clobbered.
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-signing-secret")
os.environ.setdefault("LITELLM_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import app as app_mod  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _is_real_token(name: str) -> bool:
    """Mirror of tests/eval/run_eval.py — distinguishes real creds from
    the test placeholders set by os.environ.setdefault above."""
    v = os.environ.get(name) or ""
    if not v:
        return False
    placeholders = {"sk-test", "ghp-test", "xoxb-test", "test-signing-secret"}
    return v not in placeholders


def _has_real_credentials() -> bool:
    return _is_real_token("LITELLM_API_KEY") and _is_real_token("GITHUB_TOKEN")


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / name
    assert path.exists(), (
        f"missing fixture {path} — capture it with "
        f"`python skills/pr-review-agent-skills/litellm-pr-reviewer/"
        f"scripts/gather_pr_triage_data.py <pr_url> > {path}`"
    )
    return json.loads(path.read_text())


@pytest.fixture
def patch_gather_to_fixture(monkeypatch):
    """Returns an installer: `install(fixture_dict)` swaps
    `app._run_gather_script` so any subsequent gather_pr_data tool call
    inside the agent returns the fixture verbatim. Mirrors the real
    function's signature so the agent.tool_plain wrapper is unchanged.
    """

    def install(fixture: dict[str, Any]) -> None:
        def fake_run_gather_script(script: Path, pr_ref: str) -> dict[str, Any]:
            # Sanity-bound: the agent should only call gather once per
            # run; if it calls multiple times something else is wrong
            # (loop-back, retry-storm). Returning the fixture every time
            # keeps the test deterministic instead of failing weirdly on
            # the second call.
            return fixture

        monkeypatch.setattr(app_mod, "_run_gather_script", fake_run_gather_script)

    return install


@pytest.mark.triage_regression
@pytest.mark.skipif(
    not _has_real_credentials(),
    reason=(
        "Live triage regression tests need real LITELLM_API_KEY + GITHUB_TOKEN. "
        "Skip by default; opt in with `pytest -m triage_regression` after "
        "sourcing .env."
    ),
)
def test_pr_26460_documentation_check_classified_as_pr_related(
    patch_gather_to_fixture,
):
    """BerriAI/litellm PR #26460 added three new env-var constants
    (`LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_*`) without updating the
    `environment settings - Reference` doc. The `documentation` GitHub
    Actions check fails with the test-emitted line:

        Exception: Keys not documented in 'environment settings -
        Reference': {'LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_*'}

    Pre-fix the gather script tail-truncated the 248KB log to its
    last 3KB, which contained only post-job git-cleanup output and
    none of the diagnostic. The triage agent saw uninformative inputs
    + `.github`-pointing annotations and classified `documentation` as
    `unrelated_failures`, marking the PR ready to merge.

    Post-fix (failure-aware truncation in `_extract_failure_window`)
    the diagnostic appears in the log tail. The agent must now classify
    `documentation` as `pr_related_failures` because:
      - The log names symbols (`LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_*`)
        that are added by `litellm/constants.py` (in this PR's diff).
      - The check is passing on the sampled neighbor PRs.

    This is THE test that locks the fix in. If someone reverts the
    truncation strategy, the broken-tail fixture would regenerate and
    this assertion would fail.
    """
    fixture = _load_fixture("pr_26460_gather_fixed_tail.json")

    # Sanity-check the fixture itself before paying for an LLM call —
    # if the fixture lost its load-bearing line we'd be testing the
    # wrong thing. Catches "someone re-captured the fixture against a
    # reverted gather script" without burning a triage round-trip.
    doc_ctx = next(
        c
        for c in fixture["failing_check_contexts"]
        if c["check_name"] == "documentation"
    )
    excerpt = doc_ctx.get("failure_excerpt") or ""
    assert "LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP" in excerpt, (
        "fixed-tail fixture is missing the env-var diagnostic — re-capture "
        "with the post-fix gather script (see _extract_failure_window)"
    )

    patch_gather_to_fixture(fixture)

    triage, _, err = asyncio.run(
        app_mod._run_triage(
            "Triage this PR: https://github.com/BerriAI/litellm/pull/26460"
        )
    )

    assert err is None, f"triage agent crashed: {err}"
    assert triage is not None, "triage returned no report"

    # Core assertion: documentation MUST be in the PR-related bucket.
    # The eval-grade nuance ("the PR isn't ready") is a downstream
    # consequence handled by fuse() and pinned in test_fuse.py; this
    # test only locks in the upstream classification.
    assert "documentation" in triage.pr_related_failures, (
        f"BUG: `documentation` was classified as unrelated despite the "
        f"failure log naming env-var constants this PR adds. "
        f"pr_related_failures={triage.pr_related_failures} "
        f"unrelated_failures={triage.unrelated_failures} "
        f"rationale={triage.failure_rationales.get('documentation', '<missing>')!r}"
    )
    assert "documentation" not in triage.unrelated_failures, (
        f"`documentation` ended up in BOTH buckets — schema invariant broken. "
        f"pr_related_failures={triage.pr_related_failures} "
        f"unrelated_failures={triage.unrelated_failures}"
    )

    # Soft check: the model should also produce a rationale citing the
    # diagnostic. We don't fail the test on this (rationale wording
    # varies run-to-run), but if it's empty we want the failure
    # message to flag it for the next reviewer.
    rationale = triage.failure_rationales.get("documentation", "")
    assert rationale.strip(), (
        "triage agent classified `documentation` as PR-related but emitted "
        "no rationale — the per-failure-rationale prompt block is silently "
        "skipping this entry. Expected something like 'log names env vars "
        "added in litellm/constants.py'."
    )


@pytest.mark.triage_regression
@pytest.mark.skipif(
    not _has_real_credentials(),
    reason=(
        "Live triage regression tests need real LITELLM_API_KEY + GITHUB_TOKEN. "
        "Skip by default; opt in with `pytest -m triage_regression` after "
        "sourcing .env."
    ),
)
def test_pr_26460_documentation_classified_pr_related_even_when_log_uninformative(
    patch_gather_to_fixture,
):
    """Defense-in-depth companion to
    `test_pr_26460_documentation_check_classified_as_pr_related`. Uses
    the PRE-fix gather output: the `failure_excerpt` is 3KB of post-job
    `git config --unset-all` lines (the failure-aware truncation
    helper hadn't been added yet) and `annotations` only point at
    `.github` Node.js 20 deprecation noise. NOTHING in the agent's
    inputs references this PR's diff.

    The classifier should STILL bucket `documentation` as PR-related
    because SKILL.md Step 2 line 63 says: *"True if `failure_excerpt`
    AND `annotations` give no actionable hint (both null/empty, or
    annotations only point at paths outside `diff_files` like
    `.github`), AND the same check is passing on every entry in
    `other_prs`. A check that fails only on this PR is the PR's fault
    until proven otherwise — uninformative logs don't earn a free
    pass."*

    This test locks in that rule. If a future SKILL edit drops the
    "uninformative logs don't earn a free pass" branch, the model
    would default to unrelated and this test fires. The gather-script
    fix in `_extract_failure_window` reduces how often we have to lean
    on this rule, but it doesn't replace it — Actions logs without a
    structured failure marker (e.g. opaque infra 500s) will still hit
    this codepath.
    """
    fixture = _load_fixture("pr_26460_gather_broken_tail.json")

    # Sanity-check that the fixture is in fact the bug shape — if
    # someone re-captures this fixture against the post-fix gather
    # script the file would gain the diagnostic and we'd be testing a
    # different path than the one we mean to.
    doc_ctx = next(
        c
        for c in fixture["failing_check_contexts"]
        if c["check_name"] == "documentation"
    )
    excerpt = doc_ctx.get("failure_excerpt") or ""
    assert "LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP" not in excerpt, (
        "broken-tail fixture has gained the env-var diagnostic — this test "
        "needs the PRE-fix shape (uninformative tail). Re-capture against a "
        "build that does NOT have _extract_failure_window applied, or "
        "synthesize the broken-tail shape by hand."
    )
    annotations = doc_ctx.get("annotations") or []
    assert all(".github" in a for a in annotations), (
        "broken-tail fixture's annotations should ONLY point at .github "
        "paths (the Node 20 deprecation noise). Other-path annotations "
        "would mean this isn't the uninformative-log codepath we mean to "
        "test."
    )

    patch_gather_to_fixture(fixture)

    triage, _, err = asyncio.run(
        app_mod._run_triage(
            "Triage this PR: https://github.com/BerriAI/litellm/pull/26460"
        )
    )

    assert err is None, f"triage agent crashed: {err}"
    assert triage is not None, "triage returned no report"

    assert "documentation" in triage.pr_related_failures, (
        "BUG: with no actionable log hint and `.github`-only annotations, "
        "the SKILL Step 2 line 63 rule should classify `documentation` as "
        "PR-related (failing only on this PR, uninformative log). The "
        "model treated it as unrelated, which is the regression. "
        f"pr_related_failures={triage.pr_related_failures} "
        f"unrelated_failures={triage.unrelated_failures} "
        f"rationale={triage.failure_rationales.get('documentation', '<missing>')!r}"
    )
    assert "documentation" not in triage.unrelated_failures, (
        f"`documentation` ended up in BOTH buckets — schema invariant broken. "
        f"pr_related_failures={triage.pr_related_failures} "
        f"unrelated_failures={triage.unrelated_failures}"
    )
