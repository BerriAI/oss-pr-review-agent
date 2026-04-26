"""Live eval harness for the litellm-bot PR-review agent.

Runs the full triage + pattern pipeline against a curated set of real
BerriAI/litellm PRs and dumps the results to two artifacts under
`tests/eval/results/<timestamp>/`:

  - results.json: full machine-readable record per PR (TriageReport,
    PatternReport, fused TriageCard, latency, errors). The schema is
    versioned by `schema_version` at the top so downstream tooling can
    detect format changes.
  - summary.md: human-graded markdown table — one row per PR with score,
    verdict, failing-check counts, latency, and a snippet of the verdict
    one-liner. Sorted by category so related PRs sit next to each other,
    making "did the agent treat both deps-bumps the same?" a one-glance
    check.

This is a LIVE eval. It hits real LiteLLM (LLM calls) and real GitHub
(via the gather scripts) on every run. Costs money + 1-3 minutes per PR
of wall time. Not run automatically. Two entry points:

  - python -m tests.eval.run_eval               # ad-hoc CLI
  - pytest -m eval tests/eval/test_eval.py      # gated pytest

The pytest entry point is gated on a real LITELLM_API_KEY+GITHUB_TOKEN
being present so it never accidentally fires in the smoke-test CI run.

Why we eval the live pipeline (vs fixtures):
  - The thing that breaks in production is the LLM, not fuse(). fuse()
    is already exhaustively unit-tested in tests/test_fuse.py.
  - Snapshotted gather output goes stale within hours (PR state changes,
    new checks come in). A fixture eval would tell us about a frozen
    moment that's rarely the moment we care about.
  - The cost of one eval run (~$0.50, ~5min wall) is small enough to do
    on every meaningful prompt/skill/rubric change.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure we can `import app` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import app as app_mod  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_PR_SET_PATH = EVAL_DIR / "pr_set.json"
RESULTS_ROOT = EVAL_DIR / "results"

# Bounded concurrency to avoid hammering LiteLLM / GitHub. 4 in flight is
# a balance: high enough to finish a 20-PR run in ~5min, low enough that
# a single rate-limit hit only kills one PR's run instead of cascading
# across the whole set. Tune via the EVAL_CONCURRENCY env var.
DEFAULT_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))

# Schema version on the results dump. Bump this any time the per-PR
# record shape changes so downstream graders can detect drift.
RESULTS_SCHEMA_VERSION = 1


def _load_pr_set(path: Path = DEFAULT_PR_SET_PATH) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


async def _eval_one_pr(
    pr: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run triage + pattern in parallel against one PR, return a record dict.

    Mirrors the in-flight shape of app.review_pr / app.run_pr_review so the
    eval result matches what a real Slack or chat user would see. Captures
    everything we'd need later to grade the run, including per-side errors
    so a single agent crash doesn't lose the other side's output.
    """
    url = pr["url"]
    record: dict[str, Any] = {
        "url": url,
        "category": pr.get("category"),
        "notes": pr.get("notes"),
        # Human ground-truth labels travel into every record so summary.md
        # and results.json both show "what a human said" alongside "what
        # the agent said". null when not yet hand-graded.
        "human_label": pr.get("human_label"),
        "human_notes": pr.get("human_notes"),
    }
    async with sem:
        t0 = time.perf_counter()
        # Mirror the prod path in app.run_pr_review / slack_handler: the
        # memory doc is fetched and prepended as a "User context" block to
        # both prompts so triage + pattern see the same stable repo
        # conventions a real Slack/chat run would. Without this the eval
        # tests an artificial "stock-skill" path that doesn't match what
        # actual users see, and any memory-driven calibration (e.g. the
        # repo conventions seeded under the bot's AGENT_ID) is invisible
        # to the eval — which is exactly the gap that hid the missing
        # convention-block from earlier eval rounds.
        ctx = await app_mod._memory_context()
        triage_prompt = f"{ctx}Triage this PR: {url}"
        pattern_prompt = f"{ctx}Review this PR for pattern conformance: {url}"
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


# Map agent verdict → coarse "ready vs not_ready" bucket so we can compare
# against human labels which only have those two values. WAITING is bucketed
# as "not_ready" because the human-grading rubric treats "checks still
# running" as a not-yet-mergeable state too — same UX bucket as BLOCKED.
_AGENT_TO_LABEL = {"READY": "ready", "BLOCKED": "not_ready", "WAITING": "not_ready"}


def _agreement(human: str | None, verdict: str | None) -> str:
    """Return one of: 'agree', 'disagree', 'ungraded', 'error'.

    Used both for per-row coloring in the table and for the aggregate matrix.
    Kept tiny + explicit so the comparison rule is obvious from the renderer.
    """
    if verdict is None:
        return "error"
    if human is None:
        return "ungraded"
    return "agree" if _AGENT_TO_LABEL.get(verdict) == human else "disagree"


def _summary_markdown(
    pr_set: dict[str, Any],
    results: list[dict[str, Any]],
    started_at: str,
    duration_s: float,
) -> str:
    """Human-graded markdown report. Layout:

      1. Header (when, how long, model)
      2. Aggregate stats (verdict counts, error rate, p50/p95 latency)
      3. Per-PR table sorted by category, then by PR number — so related
         PRs are adjacent and the eye can scan for inconsistencies.
      4. A 'Failures' section listing every record with a non-null error,
         with the first ~200 chars of the traceback for quick triage.
    """
    lines: list[str] = []
    lines.append(f"# PR-review eval — {started_at}")
    lines.append("")
    lines.append(
        f"- repo: `{pr_set['repo']}`  ·  PRs: {len(results)}  ·  "
        f"duration: {duration_s:.1f}s  ·  "
        f"model: `{os.environ.get('LITELLM_MODEL', 'claude-sonnet-4-6')}`"
    )
    lines.append("")
    verdict_counts: dict[str, int] = {}
    err_count = 0
    latencies: list[float] = []
    for r in results:
        if r.get("error") or not r.get("card"):
            err_count += 1
            continue
        v = r["card"]["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        latencies.append(r["latency_s"])
    latencies.sort()

    def _pct(p: float) -> float:
        if not latencies:
            return 0.0
        i = max(0, min(len(latencies) - 1, int(round(p * (len(latencies) - 1)))))
        return latencies[i]

    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- verdicts: {verdict_counts or '(none — all errored)'}")
    lines.append(f"- errors: {err_count} / {len(results)}")
    lines.append(
        f"- latency p50/p95/max: "
        f"{_pct(0.5):.1f}s / {_pct(0.95):.1f}s / "
        f"{(latencies[-1] if latencies else 0):.1f}s"
    )
    lines.append("")

    # Agreement matrix vs the human grader. Only counts records that have
    # both a human label AND a successful agent run; ungraded/errored PRs
    # are listed separately so the headline accuracy isn't quietly inflated.
    matrix: dict[str, int] = {
        "agree_ready": 0,
        "agree_not_ready": 0,
        "agent_too_lenient": 0,  # agent READY, human not_ready
        "agent_too_strict": 0,   # agent not_ready (BLOCKED/WAITING), human ready
    }
    ungraded = 0
    for r in results:
        card = r.get("card")
        verdict = card["verdict"] if card else None
        agree = _agreement(r.get("human_label"), verdict)
        if agree == "ungraded":
            ungraded += 1
        elif agree == "error":
            pass  # already counted in err_count
        elif agree == "agree":
            matrix[
                "agree_ready" if r["human_label"] == "ready" else "agree_not_ready"
            ] += 1
        else:  # disagree
            matrix[
                "agent_too_lenient"
                if r["human_label"] == "not_ready"
                else "agent_too_strict"
            ] += 1
    graded_total = sum(matrix.values())
    agreed = matrix["agree_ready"] + matrix["agree_not_ready"]
    lines.append("## Agreement vs human grader")
    lines.append("")
    if graded_total == 0:
        lines.append("- No graded PRs in this run (all entries have `human_label: null`).")
    else:
        pct = 100.0 * agreed / graded_total
        lines.append(
            f"- agreement: **{agreed} / {graded_total} = {pct:.0f}%**  "
            f"(ungraded: {ungraded}, errors: {err_count})"
        )
        lines.append(
            f"- agree (human=ready, agent=READY):       {matrix['agree_ready']}"
        )
        lines.append(
            f"- agree (human=not_ready, agent=BLOCKED/WAITING): {matrix['agree_not_ready']}"
        )
        lines.append(
            f"- disagree — agent too lenient (human=not_ready, agent=READY):     {matrix['agent_too_lenient']}"
        )
        lines.append(
            f"- disagree — agent too strict  (human=ready, agent=BLOCKED/WAITING): {matrix['agent_too_strict']}"
        )
    lines.append("")

    # Stable sort: category first, then PR number ascending so related PRs
    # are adjacent and a human grader can spot inconsistencies cheaply.
    def _sort_key(r: dict[str, Any]) -> tuple[str, int]:
        try:
            num = int(r["url"].rstrip("/").split("/")[-1])
        except Exception:
            num = 0
        return (r.get("category") or "zz_unknown", num)

    sorted_results = sorted(results, key=_sort_key)

    lines.append("## Per-PR results")
    lines.append("")
    lines.append(
        "Agree column legend: ✅ = agent matches human, ❌ = disagrees, "
        "— = no human label yet, ⚠️ = agent errored."
    )
    lines.append("")
    lines.append(
        "| Category | PR | Human | Agent | Score | Agree | Failing | Latency | Verdict one-liner |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for r in sorted_results:
        pr_num = r["url"].rstrip("/").split("/")[-1]
        cat = r.get("category", "?")
        human = r.get("human_label") or "—"
        if r.get("error") or not r.get("card"):
            lines.append(
                f"| {cat} | [#{pr_num}]({r['url']}) | {human} | ⚠️ ERROR | — | ⚠️ "
                f"| — | {r.get('latency_s', '?')}s "
                f"| {(r.get('error') or '')[:80]} |"
            )
            continue
        card = r["card"]
        triage = r["triage"]
        failing = (
            len(triage.get("pr_related_failures", []))
            + len(triage.get("unrelated_failures", []))
        )
        agree = _agreement(r.get("human_label"), card["verdict"])
        agree_glyph = {
            "agree": "✅",
            "disagree": "❌",
            "ungraded": "—",
            "error": "⚠️",
        }[agree]
        snippet = card["verdict_one_liner"].replace("|", "\\|")[:100]
        lines.append(
            f"| {cat} "
            f"| [#{pr_num}]({r['url']}) "
            f"| {human} "
            f"| {card['emoji']} {card['verdict']} "
            f"| {card['score']}/5 "
            f"| {agree_glyph} "
            f"| {failing} "
            f"| {r['latency_s']}s "
            f"| {snippet} |"
        )
    lines.append("")

    # Per-disagreement deep-dive so the grader can see WHY the agent and the
    # human diverged on each row that flipped. The full agent justification
    # + the human's notes sit side-by-side here so you don't have to cross-
    # reference results.json + pr_set.json by hand.
    disagreements = [
        r
        for r in sorted_results
        if r.get("card")
        and r.get("human_label")
        and _agreement(r["human_label"], r["card"]["verdict"]) == "disagree"
    ]
    if disagreements:
        lines.append("## Disagreements")
        lines.append("")
        for r in disagreements:
            pr_num = r["url"].rstrip("/").split("/")[-1]
            card = r["card"]
            lines.append(
                f"### #{pr_num} — human said `{r['human_label']}`, "
                f"agent said `{card['verdict']}` ({card['score']}/5)"
            )
            lines.append("")
            lines.append(f"- url: {r['url']}")
            lines.append(f"- category: `{r.get('category')}`")
            lines.append(f"- human notes: {r.get('human_notes') or '(none)'}")
            lines.append(f"- agent one-liner: {card['verdict_one_liner']}")
            lines.append(f"- agent justification: {card['justification']}")
            lines.append("")

    failures = [r for r in results if r.get("error")]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            pr_num = r["url"].rstrip("/").split("/")[-1]
            lines.append(f"### #{pr_num} — {r['url']}")
            lines.append("")
            lines.append(f"- error: `{r['error']}`")
            tb = r.get("traceback", "")
            if tb:
                lines.append("```")
                lines.append(tb[:1200])
                lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n"


async def run_eval(
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    pr_set_path: Path = DEFAULT_PR_SET_PATH,
) -> Path:
    """Run the eval end-to-end. Returns the results directory path.

    `limit` (when set) trims the PR set to the first N entries — only
    used by smoke runs from the CLI to validate wiring without paying
    for the full set. The dump still labels itself with the curated
    set's repo so the artifact shape matches a full run.

    `pr_set_path` lets callers swap in an alternate curated set
    (e.g. pr_set_v2.json) without editing the default. The set's
    filename stem is included in the results directory name so
    multiple sets don't collide on the same UTC second.
    """
    pr_set = _load_pr_set(pr_set_path)
    if limit is not None:
        pr_set = {**pr_set, "prs": pr_set["prs"][:limit]}
    sem = asyncio.Semaphore(concurrency)
    started = datetime.now(timezone.utc)
    started_iso = started.strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = RESULTS_ROOT / f"{started_iso}-{pr_set_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval] {len(pr_set['prs'])} PRs, concurrency={concurrency}, "
        f"out={out_dir}",
        flush=True,
    )
    t0 = time.perf_counter()

    # Stream progress as each PR finishes so a multi-minute run isn't a
    # silent staring contest. Order of completion will be roughly the
    # order PRs unblock from the semaphore, not the order in pr_set.
    tasks = [
        asyncio.create_task(_eval_one_pr(pr, sem), name=pr["url"])
        for pr in pr_set["prs"]
    ]
    results: list[dict[str, Any]] = []
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        record = await fut
        url = record["url"]
        if record.get("error") or not record.get("card"):
            tag = "ERR  "
        else:
            tag = f"{record['card']['emoji']} {record['card']['verdict']:8s} {record['card']['score']}/5"
        print(
            f"[eval] {i:>2}/{len(tasks)}  {tag}  "
            f"{record.get('latency_s', '?'):>5}s  {url}",
            flush=True,
        )
        results.append(record)
    duration = time.perf_counter() - t0

    # Re-order results to match pr_set order so the JSON dump is
    # deterministic across runs (regardless of completion order).
    by_url = {r["url"]: r for r in results}
    ordered = [by_url[pr["url"]] for pr in pr_set["prs"] if pr["url"] in by_url]

    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "started_at": started.isoformat(),
        "duration_s": round(duration, 2),
        "model": os.environ.get("LITELLM_MODEL", "claude-sonnet-4-6"),
        "concurrency": concurrency,
        "repo": pr_set["repo"],
        "results": ordered,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.md").write_text(
        _summary_markdown(pr_set, ordered, started.isoformat(), duration)
    )
    print(
        f"[eval] done in {duration:.1f}s. wrote {out_dir}/{{results.json,summary.md}}",
        flush=True,
    )
    return out_dir


def _is_real_token(name: str) -> bool:
    """Tokens used by tests/test_fuse.py etc. are placeholder strings like
    'sk-test'. We must NOT fire a live eval against those — it would either
    immediately 401 or, worse, point at the wrong proxy. Treat anything
    starting with 'sk-test' / 'ghp-test' / 'xoxb-test' / 'test-' as fake.
    """
    v = os.environ.get(name, "")
    if not v:
        return False
    fake_prefixes = ("sk-test", "ghp-test", "xoxb-test", "test-")
    return not v.startswith(fake_prefixes)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N PRs (smoke-test the harness without paying for all 20).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"In-flight PR cap (default: {DEFAULT_CONCURRENCY}; also reads EVAL_CONCURRENCY).",
    )
    parser.add_argument(
        "--pr-set",
        type=Path,
        default=DEFAULT_PR_SET_PATH,
        help=(
            "Path to the PR set JSON to evaluate "
            f"(default: {DEFAULT_PR_SET_PATH.name}). "
            "Use this to run alternate curated sets like pr_set_v2.json."
        ),
    )
    args = parser.parse_args()
    pr_set_path: Path = args.pr_set
    if not pr_set_path.is_absolute():
        # Resolve relative to repo root for usability — `--pr-set tests/eval/pr_set_v2.json`
        # from the repo root should Just Work without the user thinking about cwd.
        candidate = (REPO_ROOT / pr_set_path).resolve()
        if candidate.exists():
            pr_set_path = candidate
        else:
            pr_set_path = pr_set_path.resolve()
    if not pr_set_path.exists():
        print(f"[eval] PR set file not found: {pr_set_path}", file=sys.stderr)
        return 2

    if not _is_real_token("LITELLM_API_KEY"):
        print(
            "[eval] LITELLM_API_KEY missing or looks like a test placeholder. "
            "Set a real key (e.g. via .env) and re-run.",
            file=sys.stderr,
        )
        return 2
    if not _is_real_token("GITHUB_TOKEN"):
        print(
            "[eval] GITHUB_TOKEN missing or looks like a test placeholder. "
            "Set a real PAT (public_repo scope is enough) and re-run.",
            file=sys.stderr,
        )
        return 2
    asyncio.run(
        run_eval(
            concurrency=args.concurrency,
            limit=args.limit,
            pr_set_path=pr_set_path,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
