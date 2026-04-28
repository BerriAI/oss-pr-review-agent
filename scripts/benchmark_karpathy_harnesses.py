#!/usr/bin/env python3
"""
Benchmark: litellm-karpathy-check on Claude Code (`claude -p`) vs Pi (`pi -p`).

For each PR in `tests/eval/pr_set_shin_regressions.json` (or `--pr-set`):
  1. Run `gather_pr_triage_data.py` to get `head_sha`, `pr_title`, `pr_body`, `diff_files`, `linked_issues`.
  2. Ensure a local `BerriAI/litellm` clone, fetch `pull/{N}/head`, `git checkout FETCH_HEAD`
     (verifies at `head_sha` when the commit is the PR tip; otherwise fetches the SHA).
  3. Build the stdin JSON payload the skill describes.
  4. Run Claude and/or Pi (see flags) and record wall time, token/cost from JSON, max RSS (best-effort).

Required for gather + checkout: `GITHUB_TOKEN` (recommended), network.
Required for LLM: Anthropic (or your configured provider) auth for `claude` and/or `pi`.

Usage (from repo root, with .env or exported keys):
  uv run python scripts/benchmark_karpathy_harnesses.py
  uv run python scripts/benchmark_karpathy_harnesses.py --pr 26417 --skip-pi
  KARPATHY_BENCH_LITELLM_CLONE=~/dev/litellm uv run python scripts/benchmark_karpathy_harnesses.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
GATHER = (
    REPO_ROOT
    / "skills/pr-review-agent-skills/litellm-pr-reviewer/scripts/gather_pr_triage_data.py"
)
SKILL_PATH = (
    REPO_ROOT / "skills/pr-review-agent-skills/litellm-karpathy-check/SKILL.md"
)
DEFAULT_CLONE = REPO_ROOT / ".cache" / "benchmark-litellm"
DEFAULT_SET = REPO_ROOT / "tests/eval/pr_set_shin_regressions.json"
DEFAULT_CLAUDE_MODEL = "sonnet"
# Scope runs can be many tool turns; allow opt-in long wall times.
_DEFAULT_TIMEOUT = int(os.environ.get("KARPATHY_BENCH_TIMEOUT_S", "1800"))


@dataclass
class HarnessResult:
    harness: str
    ok: bool
    error: str | None
    wall_s: float
    max_rss_kb: int | None
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    extra: dict[str, Any] = field(default_factory=dict)


def _pr_number_from_url(url: str) -> int:
    m = re.search(r"/pull/(\d+)", url.rstrip("/"))
    if not m:
        raise ValueError(f"not a PR url: {url!r}")
    return int(m.group(1))


def _run_gather(pr_url: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(GATHER), pr_url],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
    )
    gather_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"gather failed {proc.returncode}: {proc.stderr[:2000] or proc.stdout[:2000]}"
        )
    return {"payload": json.loads(proc.stdout), "gather_s": gather_s}


def _norm_diff_files(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in rows:
        if not isinstance(f, dict):
            continue
        p = f.get("path") or f.get("filename")
        if not p:
            continue
        out.append(
            {
                "path": p,
                "additions": int(f.get("additions") or 0),
                "deletions": int(f.get("deletions") or 0),
            }
        )
    return out


def _github_pr_body(owner: str, repo: str, pr_number: int) -> str:
    """gather_pr_triage_data does not include pr.body in its JSON; fetch it."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token and not any(
        token.startswith(p) for p in ("ghp-test", "sk-test", "test-")
    ):
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=60.0) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        j = r.json()
    body = j.get("body")
    return body if isinstance(body, str) else ""


def _build_karpathy_input(payload: dict[str, Any], pr_body: str, repo_path: Path) -> dict[str, Any]:
    return {
        "pr_url": f"https://github.com/{payload['owner']}/{payload['repo']}/pull/{payload['pr_number']}",
        "head_sha": payload.get("head_sha", "") or "",
        "pr_title": (payload.get("pr_title") or "") or "",
        "pr_body": pr_body,
        "diff_files": _norm_diff_files(payload.get("diff_files") or []),
        "linked_issues": payload.get("linked_issues") or [],
        "repo_path": str(repo_path.resolve()),
    }


def _ensure_fetched_head(clone: Path, pr_number: int, head_sha: str) -> None:
    if not (clone / ".git").is_dir():
        raise FileNotFoundError(
            f"missing git repo at {clone} — clone with:\n"
            f"  git clone https://github.com/BerriAI/litellm.git {clone}\n"
            f"  or re-run with default cache (set by script)."
        )
    # Fetch PR head and any missing objects for head_sha
    ref = f"refs/pull/{pr_number}/head"
    subprocess.run(
        ["git", "fetch", "-q", "origin", f"pull/{pr_number}/head:refs/bench/pr_{pr_number}"],
        cwd=clone,
        check=False,
    )
    r = subprocess.run(
        ["git", "cat-file", "-t", head_sha],
        cwd=clone,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or "commit" not in (r.stdout or ""):
        subprocess.run(
            ["git", "fetch", "-q", "origin", head_sha],
            cwd=clone,
            check=False,
        )
    c = subprocess.run(
        ["git", "checkout", "-q", head_sha], cwd=clone, capture_output=True, text=True
    )
    if c.returncode != 0:
        subprocess.run(
            ["git", "checkout", "-q", f"refs/bench/pr_{pr_number}"],
            cwd=clone,
            check=True,
        )


def _max_rss_sampler(
    pid: int, out: list[int | None], stop: threading.Event
) -> None:
    peak = 0
    while not stop.wait(0.2):
        try:
            p = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
            )
            if p.returncode != 0:
                continue
            v = int(p.stdout.split()[0].strip() or 0)
            peak = max(peak, v)
        except (ValueError, FileNotFoundError, IndexError):
            pass
    out[0] = peak or None


def _run_claude_bench(
    user_prompt: str, skill_path: Path, litellm_path: Path, timeout_s: int, model: str
) -> HarnessResult:
    cmd: list[str] = [
        "claude",
        user_prompt,
        "-p",
        "--bare",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        f"--add-dir={litellm_path.resolve()}",
        "--model",
        model,
        f"--append-system-prompt-file={skill_path.resolve()}",
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak: list[int | None] = [None]
    stop = threading.Event()
    sampler: threading.Thread | None = None
    if proc.pid:
        sampler = threading.Thread(
            target=_max_rss_sampler, args=(proc.pid, peak, stop), daemon=True
        )
        sampler.start()
    out_t: str | None
    err_t: str | None
    err: str | None = None
    try:
        out_t, err_t = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out_t, err_t = proc.communicate()
        err = f"timeout {timeout_s}s"
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=1.0)
    wall = time.perf_counter() - t0
    text_out = (out_t or "")
    if proc.returncode != 0 and err is None:
        err = f"exit {proc.returncode}: {(err_t or '')[:2000]}"
    cost, tin, tout = None, None, None
    extra: dict[str, Any] = {"stderr_tail": (err_t or "")[-1500:]}
    try:
        o = None
        txt = text_out.strip()
        if txt.startswith("{"):
            o = json.loads(txt)
        if o is None:
            for line in reversed(txt.splitlines()):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                o = json.loads(line)
                break
        if o and o.get("type") == "result" and o.get("subtype") == "success":
            cost = o.get("total_cost_usd")
            u = o.get("usage") or {}
            tin, tout = u.get("input_tokens"), u.get("output_tokens")
            extra["duration_api_ms"] = o.get("duration_api_ms")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return HarnessResult(
        harness="claude_code",
        ok=err is None and proc.returncode == 0,
        error=err,
        wall_s=round(wall, 3),
        max_rss_kb=peak[0],
        cost_usd=float(cost) if cost is not None else None,
        input_tokens=tin,
        output_tokens=tout,
        extra=extra,
    )


def _parse_pi_jsonl_for_cost(stdout: str) -> tuple[float | None, int | None, int | None]:
    cost, tin, tout = None, None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Try common shapes; pi versions differ.
        if o.get("type") == "usage" and isinstance(o.get("data"), dict):
            d = o["data"]
            cost = d.get("costUSD") or d.get("totalCostUSD")
            tin = d.get("inputTokens", tin)
            tout = d.get("outputTokens", tout)
        t = o.get("total")
        if isinstance(t, dict):
            if "cost" in t:
                try:
                    cost = float(t["cost"])
                except (TypeError, ValueError):
                    pass
        c = o.get("cost")
        if isinstance(c, (int, float)):
            cost = float(c)
    return cost, tin, tout


def _run_pi_bench(
    user_prompt: str,
    skill_path: Path,
    litellm_path: Path,
    timeout_s: int,
    model: str,
) -> HarnessResult:
    cmd: list[str] = [
        "pi",
        "-p",
        "--no-session",
        "--no-context-files",
        f"--model={model}",
        f"--skill={skill_path}",
        "--tools=read,bash,edit,write",
        user_prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=litellm_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak: list[int | None] = [None]
    stop = threading.Event()
    sampler: threading.Thread | None = None
    if proc.pid:
        sampler = threading.Thread(
            target=_max_rss_sampler, args=(proc.pid, peak, stop), daemon=True
        )
        sampler.start()
    err: str | None = None
    out_t: str | None
    err_t: str | None
    try:
        out_t, err_t = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out_t, err_t = proc.communicate()
        err = f"timeout {timeout_s}s"
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=1.0)
    wall = time.perf_counter() - t0
    text_out = out_t or ""
    if proc.returncode != 0 and err is None:
        err = f"exit {proc.returncode}"
    if err is None and not text_out.strip():
        err = "empty stdout"
    cost, tin, tout = _parse_pi_jsonl_for_cost(text_out)
    return HarnessResult(
        harness="pi_coding_agent",
        ok=err is None and proc.returncode == 0,
        error=err,
        wall_s=round(wall, 3),
        max_rss_kb=peak[0],
        cost_usd=cost,
        input_tokens=tin,
        output_tokens=tout,
        extra={"stderr_tail": (err_t or "")[-2000:]},
    )


def _load_pr_set(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data.get("prs") or []


def _clone_litellm(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").is_dir():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/BerriAI/litellm.git", str(dest)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-set", type=Path, default=DEFAULT_SET)
    ap.add_argument(
        "--litellm-clone", type=Path, default=Path(os.environ.get("KARPATHY_BENCH_LITELLM_CLONE", DEFAULT_CLONE))
    )
    ap.add_argument("--claude-model", default=os.environ.get("KARPATHY_BENCH_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL))
    ap.add_argument(
        "--pi-model",
        default=os.environ.get("KARPATHY_BENCH_PI_MODEL", "anthropic/claude-sonnet-4-6"),
    )
    ap.add_argument("--pr", type=int, help="Run a single PR number (must be in the set)")
    ap.add_argument("--skip-pi", action="store_true")
    ap.add_argument("--skip-claude", action="store_true")
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run gather + git only; no LLM (no cost). Writes payload preview to the JSON output.",
    )
    ap.add_argument("--no-clone", action="store_true", help="Fail if litellm clone is missing")
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    args = ap.parse_args()

    if not GATHER.is_file():
        print(f"Missing gather script: {GATHER}", file=sys.stderr)
        return 2
    if not SKILL_PATH.is_file():
        print(f"Missing karpathy skill: {SKILL_PATH}", file=sys.stderr)
        return 2

    prs = _load_pr_set(args.pr_set)
    if args.pr is not None:
        prs = [p for p in prs if str(args.pr) in p.get("url", "")]
        if not prs:
            print(f"PR {args.pr} not in {args.pr_set}", file=sys.stderr)
            return 2

    if not prs:
        print("no PRs to run", file=sys.stderr)
        return 2

    if not args.no_clone and not (args.litellm_clone / ".git").is_dir():
        print(f"[bench] Cloning BerriAI/litellm into {args.litellm_clone} (first run)…")
        _clone_litellm(args.litellm_clone)
    else:
        _clone_litellm(args.litellm_clone)  # no-op if exists

    results: list[dict[str, Any]] = []
    for pr in prs:
        url = pr["url"]
        pr_num = _pr_number_from_url(url)
        print(f"\n[bench] === PR #{pr_num} {url} ===", flush=True)
        try:
            g = _run_gather(url)
        except Exception as e:
            results.append(
                {
                    "pr": pr_num,
                    "url": url,
                    "error": f"gather: {e!s}",
                }
            )
            continue

        pl = g["payload"]
        pr_body = _github_pr_body(pl["owner"], pl["repo"], pl["pr_number"])
        payload = _build_karpathy_input(pl, pr_body, args.litellm_clone)
        user_prompt = (
            "The following JSON is the only task input. Follow the loaded skill. "
            "The field repo_path is an on-disk litellm checkout; use Read / Grep / "
            "Bash (e.g. rg) against that tree only.\n"
            "Emit exactly one line of JSON (no markdown fence).\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            _ensure_fetched_head(args.litellm_clone, pr_num, payload["head_sha"])
        except Exception as e:
            results.append(
                {
                    "pr": pr_num,
                    "url": url,
                    "gather_s": g["gather_s"],
                    "error": f"git: {e!s}",
                }
            )
            continue

        rec: dict[str, Any] = {
            "pr": pr_num,
            "url": url,
            "head_sha": payload["head_sha"],
            "category": pr.get("category"),
            "gather_s": g["gather_s"],
            "harnesses": {},
            "karpathy_input_preview": {
                "pr_url": payload["pr_url"],
                "head_sha": payload["head_sha"],
                "pr_title": payload["pr_title"],
                "pr_body_chars": len(payload.get("pr_body") or ""),
                "diff_file_count": len(payload.get("diff_files") or []),
                "linked_issue_count": len(payload.get("linked_issues") or []),
                "repo_path": payload["repo_path"],
            },
        }
        if args.prepare_only:
            results.append(rec)
            continue
        if not args.skip_claude:
            print("[bench] running claude (claude -p)…", flush=True)
            c = _run_claude_bench(
                user_prompt, SKILL_PATH, args.litellm_clone, args.timeout, args.claude_model
            )
            rec["harnesses"]["claude_code"] = {
                "ok": c.ok,
                "error": c.error,
                "wall_s": c.wall_s,
                "max_rss_kb": c.max_rss_kb,
                "cost_usd": c.cost_usd,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
            }
        if not args.skip_pi:
            print("[bench] running pi (pi -p)…", flush=True)
            p = _run_pi_bench(
                user_prompt, SKILL_PATH, args.litellm_clone, args.timeout, args.pi_model
            )
            rec["harnesses"]["pi_coding_agent"] = {
                "ok": p.ok,
                "error": p.error,
                "wall_s": p.wall_s,
                "max_rss_kb": p.max_rss_kb,
                "cost_usd": p.cost_usd,
                "input_tokens": p.input_tokens,
                "output_tokens": p.output_tokens,
            }
        results.append(rec)

    out = {
        "pr_set": str(args.pr_set),
        "litellm_clone": str(args.litellm_clone.resolve()),
        "claude_model": args.claude_model,
        "pi_model": args.pi_model,
        "timeout_s": args.timeout,
        "results": results,
    }
    out_path = REPO_ROOT / "runs" / f"karpathy_bench_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[bench] Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
