"""Pytest entry point for the live PR-review eval.

This module exists so the eval can be triggered from the same `pytest`
command that runs everything else, *opt-in* via the `eval` marker:

    pytest -m eval tests/eval/test_eval.py

Skipped by default in regular test runs because it (a) takes minutes,
(b) costs LLM money, and (c) needs real LITELLM_API_KEY + GITHUB_TOKEN.

The harness itself lives in run_eval.py — this file is a thin shim that
runs it and asserts only on structural invariants (record-shape, no
top-level crash). Subjective grading happens in summary.md, by a human.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.eval.run_eval import _is_real_token, run_eval


def _has_real_credentials() -> bool:
    return _is_real_token("LITELLM_API_KEY") and _is_real_token("GITHUB_TOKEN")


@pytest.mark.eval
@pytest.mark.skipif(
    not _has_real_credentials(),
    reason=(
        "Live eval needs real LITELLM_API_KEY + GITHUB_TOKEN (not test placeholders). "
        "Skip by default; opt in with `pytest -m eval` after sourcing .env."
    ),
)
def test_run_eval_against_curated_pr_set():
    """Smoke-assert the eval runs end-to-end and produces both artifacts.

    We deliberately do NOT assert on per-PR scores or verdicts — those are
    subjective and live in summary.md for human review. The pass bar here
    is "the harness didn't crash and at least one PR produced a valid
    card". Anything stricter would either flake on real-world PR drift
    (PRs change state mid-eval) or need a fixture-mode rewrite.
    """
    out_dir = asyncio.run(run_eval())
    results_json = out_dir / "results.json"
    summary_md = out_dir / "summary.md"
    assert results_json.exists(), "eval did not write results.json"
    assert summary_md.exists(), "eval did not write summary.md"

    import json

    payload = json.loads(results_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["repo"] == "BerriAI/litellm"
    assert len(payload["results"]) == 20, "expected the curated 20-PR set"

    # At least one PR must have produced a real card. If every single PR
    # errored, something is fundamentally wrong (e.g. the LiteLLM proxy
    # is down) and the eval result file is useless on its own — fail the
    # test so the human sees a red signal, not just a quiet 100%-error md.
    cards = [r for r in payload["results"] if r.get("card") is not None]
    assert cards, (
        "no PR produced a valid card — either every single agent run "
        "failed or the harness is broken. See results.json for per-PR "
        f"errors. Output dir: {out_dir}"
    )

    # Card-shape invariants. fuse() guarantees these but the eval is the
    # one place we exercise the agent's *real* outputs through fuse() at
    # scale, so any drift in the agent's TriageReport schema would show
    # up here as a Pydantic ValidationError before this point — keep the
    # asserts simple and grounded in the published contract.
    for r in cards:
        card = r["card"]
        assert 0 <= card["score"] <= 5, f"score out of range: {card}"
        assert card["verdict"] in {"READY", "BLOCKED", "WAITING"}
        assert card["emoji"] in {"✅", "❌", "⏳"}
