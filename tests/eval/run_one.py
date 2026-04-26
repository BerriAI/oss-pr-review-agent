"""One-off harness: run the PR-review agent against a single PR URL.

Mirrors `_eval_one_pr` from `run_eval.py` but for ad-hoc "what does the
agent say about this one PR right now?" checks. Prints the rendered Slack
card + drilldown to stdout and writes the full structured record to
`tests/eval/results/<timestamp>-one-<pr_num>/record.json` so we can
inspect TriageReport / PatternReport fields after the fact.

Usage:
    python -m tests.eval.run_one https://github.com/BerriAI/litellm/pull/25763
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import app as app_mod  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = EVAL_DIR / "results"


async def review_one(url: str) -> dict[str, Any]:
    """Run triage + pattern in parallel against `url`. Returns the same
    record shape `_eval_one_pr` produces, minus the human-grading fields.
    """
    record: dict[str, Any] = {"url": url}
    t0 = time.perf_counter()
    triage_prompt = f"Triage this PR: {url}"
    pattern_prompt = f"Review this PR for pattern conformance: {url}"
    try:
        (triage, _, triage_err), (pattern, _, pattern_err) = await asyncio.gather(
            app_mod._run_triage(triage_prompt),
            app_mod._run_pattern(pattern_prompt),
        )
    except Exception as e:
        record["error"] = f"top-level: {type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        record["latency_s"] = round(time.perf_counter() - t0, 2)
        return record
    record["latency_s"] = round(time.perf_counter() - t0, 2)
    record["triage_error"] = triage_err
    record["pattern_error"] = pattern_err
    record["triage"] = triage.model_dump(mode="json") if triage else None
    record["pattern"] = pattern.model_dump(mode="json") if pattern else None
    if triage and pattern:
        card = app_mod.fuse(triage, pattern)
        record["card"] = card.model_dump(mode="json")
        record["rendered_card"] = app_mod.render_card(card)
        record["rendered_drilldown"] = app_mod.render_drilldown(triage, pattern)
    else:
        record["card"] = None
        record["rendered_card"] = app_mod.render_fallback_card(
            url, triage_err or pattern_err or "unknown agent failure"
        )
        record["rendered_drilldown"] = ""
    return record


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m tests.eval.run_one <pr_url>", file=sys.stderr)
        return 2
    url = sys.argv[1]
    pr_num = url.rstrip("/").split("/")[-1]
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = RESULTS_ROOT / f"{started}-one-{pr_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_one] reviewing {url}", flush=True)
    print(f"[run_one] writing to {out_dir}", flush=True)

    record = asyncio.run(review_one(url))

    (out_dir / "record.json").write_text(json.dumps(record, indent=2))

    print()
    print("=" * 80)
    print("RENDERED CARD")
    print("=" * 80)
    print(record.get("rendered_card", "(none)"))
    print()
    print("=" * 80)
    print("DRILLDOWN")
    print("=" * 80)
    print(record.get("rendered_drilldown", "(none)"))
    print()
    print(f"[run_one] latency: {record.get('latency_s')}s")
    if record.get("error"):
        print(f"[run_one] ERROR: {record['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
