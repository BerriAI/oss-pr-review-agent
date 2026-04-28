#!/usr/bin/env python3
"""Gather everything needed to triage a single GitHub PR.

Standalone port of the gather_pr_triage_data tool from BerriAI's PR review
agent. Prints a single JSON object to stdout describing the PR's checks,
diff files, Greptile score, and (where applicable) CircleCI failure logs.

Dependencies: only stdlib + httpx. Install with: pip install httpx

Required env:
    GITHUB_TOKEN   - PAT with public_repo (or repo) scope. Optional for
                     public repos but strongly recommended (60 req/hr without).

Optional env:
    LITELLM_API_BASE + LITELLM_API_KEY - When BOTH are set, raw CircleCI
                     failure log tails are spliced in for failing CircleCI
                     jobs by calling the `circle_ci_mcp-get_build_failure_logs`
                     tool on the LiteLLM proxy's MCP endpoint
                     (`{LITELLM_API_BASE}/mcp/`). The proxy holds the
                     CircleCI credential, so this script never sees a
                     CircleCI token. Without these vars, the script falls
                     back to GitHub's check-run summary alone.

Usage:
    python gather_pr_triage_data.py https://github.com/owner/repo/pull/123
    python gather_pr_triage_data.py owner/repo#123

The output JSON shape is described in references/verdict-schema.md and
matches what the litellm-pr-reviewer SKILL.md expects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"
OTHER_PRS_SAMPLE_SIZE = 3
MAX_PATCH_CHARS = 2000
MAX_LOG_CHARS = 3000

# GitHub Actions check-run html_url shape:
#   https://github.com/{owner}/{repo}/actions/runs/{run_id}/job/{job_id}
# Anchored on `/actions/runs/.../job/...` so we don't false-positive on
# CircleCI URLs or other check html_urls (Greptile, GitGuardian, etc.).
_GH_ACTIONS_JOB_URL_RE = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/actions/runs/\d+/job/(?P<job_id>\d+)"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Failure-marker patterns used by `_extract_failure_window` to centre the
# truncated log on the actual error rather than blind-tailing into post-job
# runner cleanup. Order = priority (high signal first); the helper takes
# the LAST match of the highest-priority pattern that hits. Tuned for the
# two log shapes we ingest: GitHub Actions logs (each line prefixed with a
# `2026-04-25T00:02:42.123Z ` timestamp) and CircleCI v1.1 step output
# (no timestamp prefix). Hence we use a leading whitespace boundary
# (\b on the keyword) instead of `^...` line anchors — the latter would
# miss every match in Actions logs because the timestamp eats the start.
#
# `Exception:` and `Traceback` are restricted to the Python form (capital
# E, exact phrase) so we don't match arbitrary log lines that mention the
# word "exception" in prose. `FAILED ` requires a trailing space to stay
# anchored on the pytest `FAILED tests/foo.py::test_bar` summary shape.
# `##[error]` is GitHub Actions specific. `Error:` / `error:` are generic
# fallbacks for lint/build tools that don't raise Python exceptions.
_FAILURE_MARKERS = (
    re.compile(r"\bTraceback \(most recent call last\):"),
    re.compile(r"(?<!\S)Exception:"),
    re.compile(r"(?<!\S)FAILED "),
    re.compile(r"##\[error\]"),
    re.compile(r"(?<!\S)Error:"),
    re.compile(r"(?<!\S)error:"),
)
_GREPTILE_LOGIN_RE = re.compile(r"greptile", re.IGNORECASE)
_GREPTILE_SCORE_RE = re.compile(
    r"confidence\s*score[^0-9]{0,10}([1-5])\s*/\s*5", re.IGNORECASE
)
_GREPTILE_SCORE_FALLBACK_RE = re.compile(r"\b([1-5])\s*/\s*5\b")
_CIRCLECI_NAME_RE = re.compile(r"(^|/)circleci(\s*[:/]|\b)", re.IGNORECASE)

# Policy/meta checks that operate on PR shape (branch source, signed commits,
# CLA, etc.) instead of code. They CAN block merge but their failure tells
# the reviewer nothing about whether the diff is sound, so the triage
# agent should always bucket them as `unrelated_failures`. Pre-flagging
# them in gather output (rather than relying on the model's classifier)
# kills the false-positive #26419 surfaced: a UI-dropdown PR was marked
# pr_related because "Verify PR source branch" failed.
#
# Match is case-insensitive substring against the check name. Add new
# entries here when a repo's policy infra introduces another such check.
_POLICY_META_CHECK_SUBSTRINGS = (
    "verify pr source branch",
    "dco",
    "cla/cla-bot",
    "cla-assistant",
    "license/cla",
    "signed-off-by",
    "semantic-pull-request",
    "semantic pull request",
)


def _is_policy_meta_check(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in _POLICY_META_CHECK_SUBSTRINGS)


def _extract_failure_window(text: str, max_chars: int = MAX_LOG_CHARS) -> str:
    """Truncate `text` to `max_chars` centred on the LAST high-signal failure
    marker, falling back to plain tail-truncation when no marker is found.

    Why this exists: GitHub Actions and CircleCI logs have ~100KB+ of post-job
    runner cleanup (git config --unset, artifact upload, orphan-process
    sweep) that runs AFTER the actual test/build failure. A naive
    `text[-max_chars:]` slice on a 250KB pytest log discards the
    `Exception: Keys not documented in ...` diagnostic and hands the
    classifier 3KB of `git submodule foreach --recursive ...` lines, which
    look exactly like noise unrelated to the diff. Concrete repro on
    BerriAI/litellm PR #26460: the `documentation` check's failure was at
    byte 49,740 of a 248,741-byte log; the last 3,000 chars contained zero
    mention of the test that actually failed, so the triage agent
    classified it as unrelated.

    Strategy: walk the markers in priority order (Exception > Traceback >
    FAILED > ##[error] > Error: > error:); for the first pattern that has
    any match, slice a `max_chars`-wide window starting ~200 chars before
    the LAST match. We use the last match because pytest's Exception lines
    sit AFTER the rest of the test output (so we want context above), and
    in multi-step jobs the last failure is typically the one that took the
    job down before cleanup. The 200-char lookback gives the model a
    little context above the marker (e.g. the `File "...py", line N` line
    above a `Traceback`).

    No marker hit → fall back to `text[-max_chars:]` so we degrade
    identically to the previous behavior on logs that genuinely have no
    structured failure signal.
    """
    if len(text) <= max_chars:
        return text
    for marker in _FAILURE_MARKERS:
        matches = list(marker.finditer(text))
        if not matches:
            continue
        last = matches[-1]
        # 200-char lookback: enough to catch the `File "...", line N, in <module>`
        # frame above a `Traceback`/`Exception`, small enough to leave room
        # for the failure body + a chunk of stack/runner output below.
        start = max(0, last.start() - 200)
        end = min(len(text), start + max_chars)
        prefix = "...[truncated]\n" if start > 0 else ""
        suffix = "\n...[truncated]" if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"
    # No structured failure marker — fall back to plain tail truncation so
    # callers see the same shape as before for marker-less logs (e.g. infra
    # 500 errors that just log a stack-less HTTP failure).
    return "...[truncated]\n" + text[-max_chars:]

PR_URL_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<num>\d+)"
)
PR_SHORT_RE = re.compile(r"^(?P<owner>[^/\s]+)/(?P<repo>[^#\s]+)#(?P<num>\d+)$")


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(url) or PR_SHORT_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a recognised PR reference: {url}")
    return m["owner"], m["repo"], int(m["num"])


# --------------------------------------------------------------------------- #
# GitHub HTTP helpers                                                          #
# --------------------------------------------------------------------------- #


async def _gh(
    client: httpx.AsyncClient,
    token: str | None,
    path: str,
    **params: Any,
) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = await client.get(
        f"{GITHUB_API}{path}",
        params=params or None,
        headers=headers,
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


async def _gh_list(
    client: httpx.AsyncClient,
    token: str | None,
    path: str,
    *,
    per_page: int = 100,
    list_key: str | None = None,
    **params: Any,
) -> list[dict]:
    """Page through a GitHub list endpoint until exhausted.

    GitHub list endpoints silently truncate at the per_page boundary. Any
    caller that wants "all of them" must loop -- this wraps that.
    list_key is for envelope responses (e.g. /status -> 'statuses').
    """
    items: list[dict] = []
    page = 1
    while True:
        data = await _gh(
            client, token, path, per_page=per_page, page=page, **params
        )
        batch = data.get(list_key, []) if list_key else data
        if not isinstance(batch, list):
            break
        items.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return items


# --------------------------------------------------------------------------- #
# Check enumeration                                                            #
# --------------------------------------------------------------------------- #


async def _list_check_runs(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    sha: str,
) -> list[dict]:
    """Page /commits/{sha}/check-runs and dedupe to latest-per-name."""
    runs = await _gh_list(
        client,
        token,
        f"/repos/{owner}/{repo}/commits/{sha}/check-runs",
        list_key="check_runs",
    )
    latest: dict[str, dict] = {}
    for r in runs:
        latest[r["name"]] = r
    return list(latest.values())


async def _list_classic_statuses(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    sha: str,
) -> list[dict]:
    """Return classic commit statuses, shaped to look like check-runs."""
    statuses = await _gh_list(
        client,
        token,
        f"/repos/{owner}/{repo}/commits/{sha}/status",
        list_key="statuses",
    )
    out: list[dict] = []
    for s in statuses:
        state = s.get("state")  # success | failure | error | pending
        conclusion = {
            "success": "success",
            "failure": "failure",
            "error": "failure",
            "pending": None,
        }.get(state)
        out.append(
            {
                "id": None,
                "name": s["context"],
                "conclusion": conclusion,
                "status": "completed" if conclusion else "in_progress",
                "html_url": s.get("target_url"),
                "output": {"summary": s.get("description"), "text": None},
            }
        )
    return out


async def _all_checks(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    sha: str,
) -> list[dict]:
    """Combined view: check-runs win on collisions with classic statuses."""
    runs, statuses = await asyncio.gather(
        _list_check_runs(client, token, owner, repo, sha),
        _list_classic_statuses(client, token, owner, repo, sha),
    )
    by_name: dict[str, dict] = {s["name"]: s for s in statuses}
    for r in runs:
        by_name[r["name"]] = r
    return list(by_name.values())


def _has_circleci_checks(checks: list[dict]) -> bool:
    """True iff any check-run/status is from CircleCI."""
    for c in checks or []:
        name = c.get("name") or ""
        if _CIRCLECI_NAME_RE.search(name):
            return True
        app = c.get("app") or {}
        slug = (app.get("slug") or "").lower()
        if "circleci" in slug:
            return True
        html_url = c.get("html_url") or ""
        if "circleci.com" in html_url:
            return True
    return False


# --------------------------------------------------------------------------- #
# Per-failure enrichment                                                       #
# --------------------------------------------------------------------------- #


async def _fetch_annotations(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    run_id: int | None,
) -> list[str]:
    if run_id is None:
        return []
    try:
        ann = await _gh(
            client,
            token,
            f"/repos/{owner}/{repo}/check-runs/{run_id}/annotations",
            per_page=20,
        )
    except httpx.HTTPStatusError:
        return []
    out: list[str] = []
    for a in ann or []:
        msg = (a.get("message") or "").strip()
        path = a.get("path") or ""
        line = a.get("start_line")
        out.append(f"{path}:{line}: {msg}"[:300])
    return out


class _LiteLLMMcp:
    """Minimal MCP (Streamable HTTP) client for the LiteLLM proxy.

    Why a hand-rolled client instead of `mcp.client`: we need exactly one
    tool (`circle_ci_mcp-get_build_failure_logs`) per failing CircleCI
    check-run, the proxy is stateless (no session id required), and the
    full SDK would pull in extra runtime deps for a 30-line JSON-RPC POST.
    The proxy returns SSE-framed responses (one `data: {...}` line per
    JSON-RPC reply) — we parse those by hand.

    Endpoint URL trailing slash: `/mcp` 307-redirects to `/mcp/`. We pin
    the trailing slash so we save a redirect on every call.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.url = base_url.rstrip("/") + "/mcp/"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-litellm-api-key": f"Bearer {api_key}",
        }

    async def call_tool(
        self,
        client: httpx.AsyncClient,
        name: str,
        arguments: dict,
    ) -> dict | None:
        """POST a `tools/call` and return the parsed result, or None on error.

        Returns None — never raises — so callers degrade identically to
        the pre-MCP path (no log tail spliced) if the proxy is down,
        the tool 500s, or the response shape is unexpected.
        """
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        try:
            r = await client.post(
                self.url, headers=self.headers, json=body, timeout=60.0
            )
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        for line in r.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                obj = json.loads(line[len("data: "):])
            except ValueError:
                continue
            if "error" in obj:
                return None
            res = obj.get("result")
            if isinstance(res, dict):
                return res
        return None


# Truncation-warning prelude that the upstream CircleCI MCP server prepends
# when its log slice exceeds an internal cap. We strip it before passing the
# body to `_extract_failure_window` so the marker centring isn't fooled by
# the warning text. See:
# https://github.com/CircleCI-Public/mcp-server-circleci
_MCP_TRUNCATION_PRELUDE_RE = re.compile(
    r"^\s*<MCPTruncationWarning>.*?</MCPTruncationWarning>\s*",
    re.DOTALL,
)


async def _fetch_circleci_failure_log(
    client: httpx.AsyncClient,
    mcp: _LiteLLMMcp | None,
    html_url: str | None,
) -> str | None:
    """Fetch the failing-step log tail for a CircleCI build via the LiteLLM
    MCP proxy.

    Hands the GitHub status `target_url` straight to the upstream
    `circle_ci_mcp-get_build_failure_logs` tool — that tool already
    handles all five CircleCI URL shapes (project / pipeline / workflow
    / job / legacy) so we don't parse them ourselves. Returns the
    failure-window-extracted log tail (same shape as the GitHub Actions
    fetcher) or None if `mcp` is unset, the URL isn't a CircleCI URL,
    or the call fails.
    """
    if mcp is None or not html_url:
        return None
    if "circleci.com" not in html_url:
        return None
    res = await mcp.call_tool(
        client,
        "circle_ci_mcp-get_build_failure_logs",
        {"params": {"projectURL": html_url}},
    )
    if not res or res.get("isError"):
        return None
    parts = []
    for c in res.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text") or "")
    text = "\n".join(parts).strip()
    if not text:
        return None
    text = _MCP_TRUNCATION_PRELUDE_RE.sub("", text)
    text = _ANSI_ESCAPE_RE.sub("", text)
    if not text.strip():
        return None
    return _extract_failure_window(text)


async def _fetch_actions_job_log(
    client: httpx.AsyncClient,
    github_token: str | None,
    html_url: str | None,
) -> str | None:
    """Fetch the tail of a GitHub Actions job log given its check-run html_url.

    GitHub's `output.text` field on a check-run is almost always empty for
    Actions jobs (the runner doesn't populate it — annotations are the only
    structured surface, and annotations only fire on explicit `::error::`
    workflow commands). Tools like `black`, `ruff`, `pytest` exit non-zero
    via stdout instead, so the actual diagnostic ("would reformat
    .../foo.py", "1 failed in 0.5s", etc.) only lives in the rendered job
    logs. Mirror of `_fetch_circleci_failure_log` but for Actions.

    Endpoint: `GET /repos/{o}/{r}/actions/jobs/{job_id}/logs`
        - 302-redirects to a presigned blob URL on pipelines.actions.
          githubusercontent.com — we follow the redirect and read it as
          plain text.
        - Docs claim "anyone with read access" suffices, but in practice
          GitHub returns 403 ("Must have admin rights to Repository") for
          non-admin tokens on many public repos. That's a long-standing
          GitHub quirk, not a bug here. Returns None on any non-200 so the
          gather pipeline degrades gracefully — caller behavior stays
          identical to the old (annotations-only) path. To unlock this for
          a given repo at deploy time: install a GitHub App with
          `actions:read` on the target org, or use a PAT belonging to a
          repo admin.

    Returns a MAX_LOG_CHARS-wide window centred on the last high-signal
    failure marker (see `_extract_failure_window`) on success, None
    otherwise. The window strategy matches the CircleCI fetcher so the
    model sees the same shape from both sources.
    """
    if not html_url:
        return None
    m = _GH_ACTIONS_JOB_URL_RE.search(html_url)
    if not m:
        return None
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # Auth is optional per GitHub docs for public repos, but we always send
    # GITHUB_TOKEN when available so private repos work + we get the higher
    # 5000/hr rate-limit bucket instead of the 60/hr unauth quota.
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    url = (
        f"{GITHUB_API}/repos/{m['owner']}/{m['repo']}"
        f"/actions/jobs/{m['job_id']}/logs"
    )
    try:
        # follow_redirects=True is critical: the API returns 302 to a
        # presigned blob URL. Without it httpx returns the 302 and we get
        # nothing useful.
        r = await client.get(
            url,
            headers=headers,
            follow_redirects=True,
            timeout=30.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    text = r.text
    # Strip ANSI color codes the same way the CircleCI fetcher does — the
    # model wastes context budget reading escape sequences otherwise.
    text = _ANSI_ESCAPE_RE.sub("", text)
    if not text.strip():
        return None
    return _extract_failure_window(text)


# --------------------------------------------------------------------------- #
# PR-level fetches                                                             #
# --------------------------------------------------------------------------- #


async def _fetch_diff(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    files = await _gh_list(
        client, token, f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
    )
    out: list[dict] = []
    for f in files:
        patch = f.get("patch")
        if patch and len(patch) > MAX_PATCH_CHARS:
            patch = patch[:MAX_PATCH_CHARS] + "\n...[truncated]"
        out.append(
            {
                "filename": f["filename"],
                "status": f.get("status", "modified"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": patch,
            }
        )
    return out


async def _fetch_other_open_prs(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    exclude_pr: int,
    n: int,
) -> list[dict]:
    pulls = await _gh(
        client,
        token,
        f"/repos/{owner}/{repo}/pulls",
        state="open",
        sort="updated",
        direction="desc",
        per_page=n + 5,
    )
    return [p for p in pulls if p["number"] != exclude_pr][:n]


async def _fetch_greptile_score(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    pr_number: int,
) -> int | None:
    """Latest Greptile confidence score (1-5), or None. Best-effort."""
    try:
        reviews, comments = await asyncio.gather(
            _gh_list(
                client, token, f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
            ),
            _gh_list(
                client,
                token,
                f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            ),
        )
    except httpx.HTTPStatusError:
        return None

    candidates: list[tuple[str, str]] = []
    for r in reviews or []:
        login = (r.get("user") or {}).get("login") or ""
        if _GREPTILE_LOGIN_RE.search(login):
            candidates.append((r.get("submitted_at") or "", r.get("body") or ""))
    for c in comments or []:
        login = (c.get("user") or {}).get("login") or ""
        if _GREPTILE_LOGIN_RE.search(login):
            candidates.append((c.get("created_at") or "", c.get("body") or ""))

    candidates.sort(reverse=True)
    for _, body in candidates:
        m = _GREPTILE_SCORE_RE.search(body) or _GREPTILE_SCORE_FALLBACK_RE.search(
            body
        )
        if m:
            return int(m.group(1))
    return None


async def _noop_none() -> None:
    return None


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


async def _fetch_pr_with_mergeable(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    pr_number: int,
) -> dict:
    """Fetch a PR, retrying once if `mergeable` is null.

    GitHub computes `mergeable` lazily on the first PR fetch — the initial
    response is often `null` while a background job runs the merge test. A
    single retry after a short pause is the documented pattern. If it stays
    null after the retry we give up and let downstream treat it as unknown
    (better than blocking the whole triage on a slow background job).
    """
    pr = await _gh(client, token, f"/repos/{owner}/{repo}/pulls/{pr_number}")
    if pr.get("mergeable") is None:
        await asyncio.sleep(1.5)
        pr = await _gh(client, token, f"/repos/{owner}/{repo}/pulls/{pr_number}")
    return pr


async def gather(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    github_token: str | None,
    mcp: _LiteLLMMcp | None,
) -> dict:
    async with httpx.AsyncClient() as client:
        pr, diff_files, other_prs, greptile_score = await asyncio.gather(
            _fetch_pr_with_mergeable(
                client, github_token, owner, repo, pr_number
            ),
            _fetch_diff(client, github_token, owner, repo, pr_number),
            _fetch_other_open_prs(
                client,
                github_token,
                owner,
                repo,
                pr_number,
                OTHER_PRS_SAMPLE_SIZE,
            ),
            _fetch_greptile_score(client, github_token, owner, repo, pr_number),
        )
        head_sha = pr["head"]["sha"]
        # `mergeable`: true | false | null (still computing). `mergeable_state`
        # adds nuance: "dirty" = conflicts, "blocked" = required review/check
        # missing, "behind" = base moved, "clean"/"unstable" = mergeable. We
        # surface both so downstream can treat null as "unknown" (don't block)
        # while false/dirty is a hard merge-conflict signal.
        #
        # ALREADY-MERGED override: GitHub keeps recomputing `mergeable` against
        # current main even after a PR is merged, so a months-old merged PR
        # whose head no longer fast-forwards into HEAD comes back as
        # mergeable=false / state=dirty. That is meaningless — the PR shipped
        # cleanly at merge time. Without this override the conflict-blocker
        # rubric (weight 5) flipped already-merged PRs to BLOCKED 0/5 (eval
        # @ 22:30 UTC: PR #26467 of BerriAI/litellm). For closed-merged PRs
        # we force the clean-merge signal so the rubric ignores the ghost.
        mergeable = pr.get("mergeable")
        mergeable_state = pr.get("mergeable_state")
        if pr.get("state") == "closed" and pr.get("merged_at"):
            mergeable = True
            mergeable_state = "clean"

        own_checks_task = _all_checks(client, github_token, owner, repo, head_sha)
        other_checks_tasks = [
            _all_checks(client, github_token, owner, repo, p["head"]["sha"])
            for p in other_prs
        ]
        own_checks, *other_checks = await asyncio.gather(
            own_checks_task, *other_checks_tasks
        )

        passing: list[str] = []
        in_progress: list[str] = []
        failing_runs: list[dict] = []
        for r in own_checks:
            concl = r.get("conclusion")
            if concl in ("success", "neutral", "skipped"):
                passing.append(r["name"])
            elif concl in ("failure", "timed_out", "cancelled"):
                failing_runs.append(r)
            else:
                in_progress.append(r["name"])

        if failing_runs:
            (
                annotations_per,
                circleci_logs_per,
                actions_logs_per,
            ) = await asyncio.gather(
                asyncio.gather(
                    *[
                        _fetch_annotations(
                            client, github_token, owner, repo, r.get("id")
                        )
                        for r in failing_runs
                    ]
                ),
                asyncio.gather(
                    *[
                        (
                            _fetch_circleci_failure_log(
                                client, mcp, r.get("html_url")
                            )
                            if mcp is not None
                            else _noop_none()
                        )
                        for r in failing_runs
                    ]
                ),
                # GitHub Actions log tails. Same shape as the CircleCI
                # branch (returns None when the URL isn't an Actions one,
                # or when the API returns 403 because the token lacks
                # admin rights — see _fetch_actions_job_log docstring).
                # The regex inside the fetcher silently no-ops on
                # non-Actions URLs (Greptile, GitGuardian, etc.) so we
                # can hand it every failing check unconditionally.
                asyncio.gather(
                    *[
                        _fetch_actions_job_log(
                            client, github_token, r.get("html_url")
                        )
                        for r in failing_runs
                    ]
                ),
            )
        else:
            annotations_per = []
            circleci_logs_per = []
            actions_logs_per = []

        failure_contexts: list[dict] = []
        for r, ann_list, cci_log, gha_log in zip(
            failing_runs, annotations_per, circleci_logs_per, actions_logs_per
        ):
            name = r["name"]
            output = r.get("output") or {}
            text = output.get("text") or ""
            if len(text) > MAX_LOG_CHARS:
                text = text[:MAX_LOG_CHARS] + "\n...[truncated]"
            if cci_log:
                text = (
                    f"{text}\n\n--- CircleCI raw log tail ---\n{cci_log}"
                    if text
                    else f"--- CircleCI raw log tail ---\n{cci_log}"
                )
            # Splice GitHub Actions log tail under its own header so the
            # model can tell sources apart at a glance — matches the
            # CircleCI splicing pattern verbatim. Only one of cci_log /
            # gha_log fires per check by construction (the URL regexes
            # are mutually exclusive), so there's no double-tail risk.
            if gha_log:
                text = (
                    f"{text}\n\n--- GitHub Actions raw log tail ---\n{gha_log}"
                    if text
                    else f"--- GitHub Actions raw log tail ---\n{gha_log}"
                )
            other_status: list[dict] = []
            for p, p_checks in zip(other_prs, other_checks):
                match = next((c for c in p_checks if c["name"] == name), None)
                other_status.append(
                    {
                        "pr_number": p["number"],
                        "pr_title": p.get("title", ""),
                        "found": match is not None,
                        "conclusion": (match or {}).get("conclusion"),
                    }
                )
            # Pre-derived "also red on a neighbor PR" boolean. The model
            # was unreliable at computing this from other_prs at output
            # time (eval @ 22:26 UTC: only 1/7 cases populated correctly),
            # so do it here in pure Python and let the model just copy
            # the answer through. Treats only conclusive failures as
            # signal — `null` (didn't run / pending) is NOT counted as
            # "failing" because we can't tell which.
            also_failing_elsewhere = any(
                p.get("conclusion") in ("failure", "timed_out", "cancelled")
                for p in other_status
            )
            failure_contexts.append(
                {
                    "check_name": name,
                    "conclusion": r.get("conclusion"),
                    "summary": output.get("summary"),
                    "failure_excerpt": text or None,
                    "annotations": ann_list,
                    "html_url": r.get("html_url"),
                    "other_prs": other_status,
                    # Pre-classified meta/policy bucket. SKILL.md Step 2
                    # tells the model: is_policy_meta=true → ALWAYS
                    # related_to_pr_diff=false. Removes the model's
                    # ability to false-positive on PR-shape checks.
                    "is_policy_meta": _is_policy_meta_check(name),
                    # Pre-derived for the rubric's unique-vs-elsewhere
                    # split. True iff this same check is conclusively
                    # failing on at least one sampled neighbor PR.
                    "also_failing_on_other_prs": also_failing_elsewhere,
                }
            )

        return {
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "pr_title": pr.get("title", ""),
            "pr_author": (pr.get("user") or {}).get("login") or "",
            "head_sha": head_sha,
            "passing_checks": passing,
            "in_progress_checks": in_progress,
            "failing_check_contexts": failure_contexts,
            "diff_files": diff_files,
            "other_pr_numbers": [p["number"] for p in other_prs],
            "greptile_score": greptile_score,
            "has_circleci_checks": _has_circleci_checks(own_checks),
            "mergeable": mergeable,
            "mergeable_state": mergeable_state,
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Print a JSON triage report for a single GitHub PR. Consumed by "
            "the litellm-pr-reviewer SKILL.md."
        )
    )
    ap.add_argument(
        "pr",
        help=(
            "PR reference. Either a full URL "
            "(https://github.com/owner/repo/pull/N) or owner/repo#N."
        ),
    )
    args = ap.parse_args()

    try:
        owner, repo, pr_number = parse_pr_url(args.pr)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    gh_token = os.environ.get("GITHUB_TOKEN") or None
    if not gh_token:
        print(
            "warning: GITHUB_TOKEN not set; using unauthenticated GitHub API "
            "(60 req/hr limit; expect 403 on busy repos).",
            file=sys.stderr,
        )

    litellm_base = os.environ.get("LITELLM_API_BASE")
    litellm_key = os.environ.get("LITELLM_API_KEY")
    mcp = (
        _LiteLLMMcp(litellm_base, litellm_key)
        if litellm_base and litellm_key
        else None
    )
    if mcp is None:
        print(
            "warning: LITELLM_API_BASE/LITELLM_API_KEY not set; CircleCI "
            "failure log tails will be omitted (GitHub check-run summaries "
            "still included).",
            file=sys.stderr,
        )

    try:
        report = asyncio.run(
            gather(
                owner,
                repo,
                pr_number,
                github_token=gh_token,
                mcp=mcp,
            )
        )
    except httpx.HTTPStatusError as exc:
        print(
            f"error: GitHub returned {exc.response.status_code} for "
            f"{exc.request.url}",
            file=sys.stderr,
        )
        sys.exit(1)

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
