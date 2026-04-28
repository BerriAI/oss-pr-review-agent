"""Karpathy-style senior-engineer pre-merge check via Claude Code.

A second opinion the bot only consults when the rest of the pipeline
(triage + pattern fuse) is about to emit `READY`. The skill is the
gate's *tool*, not the gate itself: this module shells out to the
`claude` CLI, points it at the SKILL.md and a fresh worktree of the
litellm clone at the PR's `head_sha`, and parses the senior-eng JSON
verdict back into pydantic.

When it runs
------------
Only when fused verdict would otherwise be READY. The point is to
catch things triage + pattern can't see: scope drift, hot-path perf
regressions, dead code, sink-not-source fixes. We never gate-block
on it during shadow rollout; the caller decides whether to demote
READY based on the returned `merge_gate.safe_for_high_rps_gateway`.

Fail-open contract
------------------
`run_karpathy_check` returns `None` on any failure — missing CLI,
missing skill, subprocess timeout, parse error, validation error.
The caller MUST treat `None` as "no check ran" and NOT as "check
passed". The whole point of fail-open is that an outage in the
karpathy harness (CC down, network, budget exhaustion) cannot
silently turn `READY` into `BLOCKED`.

Env knobs (all read at call time, not import)
---------------------------------------------
- `KARPATHY_CHECK_ENABLED`     — `"true"` (default). Set to anything
                                 else to short-circuit to None.
- `KARPATHY_CHECK_SKILL`       — path to SKILL.md. Default:
                                 `<repo_root>/skills/pr-review-agent-skills/litellm-karpathy-check/SKILL.md`.
- `LITELLM_CLONE`              — local litellm checkout. Default tries
                                 `/Users/krrishdholakia/Documents/litellm`
                                 (laptop), falls back to `/opt/litellm`
                                 (typical Render layout).
- `KARPATHY_CHECK_MAX_USD`     — `claude --max-budget-usd`. Default `"2.00"`.
- `KARPATHY_CHECK_TIMEOUT_SEC` — wall-clock cap on the CC subprocess.
                                 Default `"600"`.
- `KARPATHY_CHECK_CONCURRENCY` — global asyncio semaphore size so a
                                 batch review doesn't fork N CC
                                 subprocesses at once. Default `"2"`.
- `KARPATHY_CHECK_OUT`         — optional dir for transcripts and the
                                 worktree. Default: a fresh tempdir
                                 per call (cleaned up in `finally`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger("litellm-bot.karpathy_check")

# --- Constants ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SKILL = (
    REPO_ROOT / "skills/pr-review-agent-skills/litellm-karpathy-check/SKILL.md"
)
# Two reasonable defaults: the maintainer's laptop layout and the typical
# Render disk layout. Env override (`LITELLM_CLONE`) wins over both.
_LOCAL_LITELLM = Path("/Users/krrishdholakia/Documents/litellm")
_RENDER_LITELLM = Path("/opt/litellm")

# litellm is the only repo this skill knows how to read.
_LITELLM_REPO = "BerriAI/litellm"

PR_URL_RE = re.compile(r"github\.com/[\w.-]+/[\w.-]+/pull/(\d+)")


# --- Schema (public) ---------------------------------------------------------


class KarpathyFinding(BaseModel):
    regression_archetype: str = ""
    bug_class: str = ""
    fix_locus: str = ""
    sibling_loci: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    breadth: Literal[
        "narrow_correct",
        "narrow_missed_class",
        "scope_expansion",
        "scope_drift",
        "wrong_fix_layer",
        "performance_regression_hot_path",
        "dead_code_unreachable",
        "production_behavior_mismatch",
        "maintainability_risk",
        "behavior_change_high_blast_radius",
    ]
    recommended_fix: str = ""


class KarpathyMergeGate(BaseModel):
    safe_for_high_rps_gateway: Literal["yes", "no", "conditional"] = "yes"
    one_liner: str = ""
    unintended_consequences: list[str] = Field(default_factory=list)
    hot_path_notes: list[str] = Field(default_factory=list)
    what_would_make_yes: str = ""


class KarpathyReview(BaseModel):
    # extra="allow" so we can stash CC-side telemetry (cost, duration, raw
    # envelope) on the same object without polluting the on-the-wire schema
    # the skill returns. The spec said "private extra"; this is that.
    model_config = ConfigDict(extra="allow")

    linked_issue: str | None = None
    fix_shapes: list[str] = Field(default_factory=list)
    merge_gate: KarpathyMergeGate = Field(default_factory=KarpathyMergeGate)
    findings: list[KarpathyFinding] = Field(default_factory=list)


# --- Concurrency (lazily constructed) ---------------------------------------

# Built on first call inside the running event loop. asyncio.Semaphore
# binds to whatever loop is current at construction, so creating it at
# import time would (a) require a loop to exist, and (b) silently bind
# to the wrong loop under pytest-asyncio's per-test loops.
_sema: asyncio.Semaphore | None = None
_sema_size: int = 0


def _get_sema() -> asyncio.Semaphore:
    global _sema, _sema_size
    desired = _read_int_env("KARPATHY_CHECK_CONCURRENCY", 2, minimum=1)
    if _sema is None or desired != _sema_size:
        # Re-create if the env var changed between calls (mostly relevant
        # for tests that monkeypatch). Plain new() is fine — no in-flight
        # acquires get cancelled because every caller holds its own ref
        # via the `async with` block.
        _sema = asyncio.Semaphore(desired)
        _sema_size = desired
    return _sema


# --- Small helpers (extracted from scripts/run_karpathy_check_eval.py) ------


def _pr_number_from_url(url: str) -> int:
    m = PR_URL_RE.search(url)
    if not m:
        raise ValueError(f"not a github PR url: {url!r}")
    return int(m.group(1))


def _linked_issue_numbers(body: str) -> list[int]:
    """Issue numbers referenced via fixes/closes/resolves keywords.

    Both `#NN` and full GitHub issue URLs are recognized. Same-repo
    only — cross-repo `org/repo#NN` is intentionally not supported
    because the skill is litellm-specific and we don't want to fetch
    issues from other repos by accident.
    """
    out: list[int] = []
    for m in re.finditer(r"(?:fixes|closes|resolves)\s+#(\d+)", body, flags=re.I):
        out.append(int(m.group(1)))
    for m in re.finditer(
        r"(?:fixes|closes|resolves)\s+https?://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)",
        body,
        flags=re.I,
    ):
        out.append(int(m.group(1)))
    return sorted(set(out))


def _gh_json(args: list[str]) -> dict:
    """Sync `gh ... --json ...` invocation; raises on non-zero exit."""
    r = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def _extract_json_line(text: str) -> dict | None:
    """Last line of `text` that parses as a JSON object.

    The skill is told to print its verdict as the LAST line of stdout,
    but CC sometimes sticks a trailing log line after it; scanning
    bottom-up tolerates that without loosening the contract.
    """
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# --- Env helpers -------------------------------------------------------------


def _read_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        log.warning("karpathy_env_invalid name=%s value=%r", name, raw)
        return default
    return max(minimum, v)


def _resolve_litellm_clone() -> Path | None:
    """Pick a litellm checkout: env override → laptop → Render → None."""
    env = os.environ.get("LITELLM_CLONE", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    for candidate in (_LOCAL_LITELLM, _RENDER_LITELLM):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_skill_path() -> Path:
    return Path(os.environ.get("KARPATHY_CHECK_SKILL") or DEFAULT_SKILL)


# --- gh wrappers (off the loop) ---------------------------------------------


async def _gather_pr_data(pr_url: str) -> dict[str, Any]:
    """Build the JSON payload the skill expects.

    All `gh` calls are sync; we offload them to a thread so we don't
    block the event loop while shelling out for ~hundreds of ms. The
    eval script (run_karpathy_check_eval.py) does the same dance
    inline; this is the cleaned-up async-friendly version.
    """
    pr = _pr_number_from_url(pr_url)
    pr_data = await asyncio.to_thread(
        _gh_json,
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            _LITELLM_REPO,
            "--json",
            "headRefOid,title,body,files",
        ],
    )
    body = pr_data.get("body") or ""

    linked: list[dict[str, Any]] = []
    for num in _linked_issue_numbers(body):
        try:
            issue = await asyncio.to_thread(
                _gh_json,
                [
                    "issue",
                    "view",
                    str(num),
                    "--repo",
                    _LITELLM_REPO,
                    "--json",
                    "number,title,body",
                ],
            )
        except subprocess.CalledProcessError as e:
            # A missing / private issue link shouldn't fail the whole
            # check — just drop it and keep going.
            log.warning("karpathy_issue_fetch_failed pr=%s issue=%s err=%s", pr, num, e)
            continue
        linked.append(
            {
                "number": issue["number"],
                "title": issue["title"],
                # 4kB cap: matches the eval script and keeps the prompt
                # well under CC's input budget even with a chatty issue.
                "body": (issue.get("body") or "")[:4000],
            }
        )

    return {
        "pr_number": pr,
        "pr_url": pr_url,
        "head_sha": pr_data["headRefOid"],
        "pr_title": pr_data.get("title") or "",
        "pr_body": body[:4000],
        "diff_files": [
            {
                "path": f["path"],
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            }
            for f in (pr_data.get("files") or [])
        ],
        "linked_issues": linked,
    }


# --- Worktree management ----------------------------------------------------


def _git_run(litellm: Path, args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(litellm), *args],
        capture_output=True,
        text=True,
        check=check,
    )


async def _make_worktree(litellm: Path, pr: int, head_sha: str, dest: Path) -> str:
    """Fetch the PR head and add a detached worktree at it.

    Returns the SHA actually checked out (which may differ from
    `head_sha` if the PR head moved between `gh pr view` and now).
    The caller cleans up via `_remove_worktree`.
    """

    def _do() -> str:
        # Fetching `pull/N/head` is the cheap way to get the PR tip
        # without configuring a remote per PR.
        _git_run(litellm, ["fetch", "origin", f"pull/{pr}/head"], check=True)
        fetch_head = _git_run(litellm, ["rev-parse", "FETCH_HEAD"], check=True).stdout.strip()
        # Race: PR head can move between `gh pr view` and `git fetch`.
        # Log it and keep `head_sha` as the source of truth (gh value
        # is what we already built the prompt around — switching here
        # would desync evidence from the diff_files list).
        if fetch_head != head_sha:
            log.warning(
                "karpathy_head_drift pr=%s gh=%s fetch=%s",
                pr,
                head_sha[:12],
                fetch_head[:12],
            )
        _git_run(
            litellm,
            ["worktree", "add", "--detach", str(dest), head_sha],
            check=True,
        )
        return head_sha

    return await asyncio.to_thread(_do)


async def _remove_worktree(litellm: Path, dest: Path) -> None:
    """Best-effort worktree teardown. Never raises."""

    def _do() -> None:
        try:
            _git_run(
                litellm,
                ["worktree", "remove", "--force", str(dest)],
                check=False,
            )
        except Exception as e:
            log.warning("karpathy_worktree_remove_failed wt=%s err=%s", dest, e)

    await asyncio.to_thread(_do)


# --- Prompt assembly --------------------------------------------------------


def _build_prompt(skill_body: str, payload: dict[str, Any], wt: Path) -> str:
    """Same shape as the eval script: skill body, then JSON payload.

    `repo_path` is added at prompt-build time (not gather time) because
    the worktree lives next to this call, not next to the gh data.
    """
    payload = {**payload, "repo_path": str(wt.resolve())}
    input_json = json.dumps(payload, indent=2)
    return f"""You are a litellm-karpathy-check reviewer. Follow the skill below verbatim.
The "Inputs" JSON is provided directly in this prompt (no stdin).
Read files only via Read/Grep/Glob/Bash within the repo_path.
Print your final JSON report as the LAST line of your reply (single line JSON matching the skill schema).

=== SKILL BEGIN ===
{skill_body}
=== SKILL END ===

=== INPUT JSON BEGIN ===
{input_json}
=== INPUT JSON END ===
"""


# --- Subprocess invocation --------------------------------------------------


async def _invoke_claude(
    prompt: str, wt: Path, *, timeout_s: int, max_usd: str
) -> tuple[int, bytes, bytes] | None:
    """Run `claude -p` with the karpathy prompt on stdin.

    Returns `(returncode, stdout, stderr)`, or None on timeout (and
    kills the child). Wider exceptions bubble up to the caller's
    catch-all so logging stays in one place.
    """
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--output-format",
        "json",
        "--add-dir",
        str(wt.resolve()),
        "--allowedTools",
        "Read,Grep,Glob,Bash",
        "--permission-mode",
        "bypassPermissions",
        "--max-budget-usd",
        max_usd,
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode()), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        # Hard-kill so the CC subprocess doesn't keep burning budget
        # after we've given up waiting for it.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return None
    return proc.returncode or 0, stdout or b"", stderr or b""


def _parse_envelope(stdout: bytes) -> tuple[dict | None, str, float | None]:
    """Pull (envelope, result_text, total_cost_usd) out of CC's JSON.

    `claude -p --output-format json` wraps the model's reply in an
    envelope dict; the actual skill output sits in `envelope.result`.
    Cost surfaces at `envelope.total_cost_usd`. We tolerate the
    envelope being missing (older CC versions, or stderr-only failures).
    """
    raw = stdout.decode("utf-8", errors="replace")
    envelope: dict | None = None
    if raw.strip():
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            envelope = None
    result_text = ""
    cost: float | None = None
    if isinstance(envelope, dict):
        result_text = str(envelope.get("result") or "")
        c = envelope.get("total_cost_usd")
        if isinstance(c, (int, float)):
            cost = float(c)
    return envelope, result_text, cost


# --- Optional logfire span (no-op if logfire not installed) ----------------


class _NoopSpan:
    """Minimal span shim: same surface (`set_attribute`, context manager)
    as `logfire.span`, no-op everywhere. Used when logfire isn't installed
    or hasn't been configured yet."""

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def set_attribute(self, _key: str, _value: Any) -> None:
        return None


def _open_span(pr_url: str) -> Any:
    try:
        import logfire  # local import: dep is optional from this module's POV
    except Exception:
        return _NoopSpan()
    try:
        return logfire.span("karpathy_check", pr_url=pr_url)
    except Exception:
        # logfire not configured (no token, etc.) — degrade silently.
        return _NoopSpan()


# --- Public entrypoint ------------------------------------------------------


async def run_karpathy_check(pr_url: str) -> KarpathyReview | None:
    """Run the karpathy senior-eng pre-merge check on a PR.

    Returns None on any failure (subprocess error, parse error, missing
    `claude` CLI, missing skill, missing litellm clone). Caller must
    treat None as 'no check ran' — never as 'check passed'.
    """
    if os.environ.get("KARPATHY_CHECK_ENABLED", "true").strip().lower() != "true":
        log.info("karpathy_skipped reason=disabled pr_url=%s", pr_url)
        return None

    if shutil.which("claude") is None:
        log.warning("karpathy_skipped reason=no_claude pr_url=%s", pr_url)
        return None

    skill_path = _resolve_skill_path()
    if not skill_path.is_file():
        log.warning(
            "karpathy_skipped reason=no_skill pr_url=%s skill=%s", pr_url, skill_path
        )
        return None

    litellm = _resolve_litellm_clone()
    if litellm is None:
        log.warning("karpathy_skipped reason=no_clone pr_url=%s", pr_url)
        return None

    timeout_s = _read_int_env("KARPATHY_CHECK_TIMEOUT_SEC", 600, minimum=30)
    max_usd = os.environ.get("KARPATHY_CHECK_MAX_USD", "2.00")

    out_root_env = os.environ.get("KARPATHY_CHECK_OUT", "").strip()
    out_root: Path
    cleanup_out_root = False
    if out_root_env:
        out_root = Path(out_root_env).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        # Per-call tempdir keeps parallel checks from colliding on
        # `wt-pr-NNN`. Removed in `finally`.
        out_root = Path(tempfile.mkdtemp(prefix="karpathy-pr-"))
        cleanup_out_root = True

    sema = _get_sema()
    # `pr_num` starts unknown so the bare-bones exception handler can still
    # log a readable message even if `_gather_pr_data` blows up before we
    # know the number. Narrowed to int via assertion once payload lands.
    pr_num: int | None = None
    wt: Path | None = None
    t_start = time.perf_counter()

    async with sema:
        try:
            with _open_span(pr_url) as span:
                payload = await _gather_pr_data(pr_url)
                pr_num_resolved: int = int(payload["pr_number"])
                pr_num = pr_num_resolved
                head_sha: str = str(payload["head_sha"])

                wt = out_root / f"wt-pr-{pr_num_resolved}"
                if wt.exists():
                    # Stale leftover from a prior crashed run: nuke it before
                    # `git worktree add` rejects the path.
                    await _remove_worktree(litellm, wt)

                await _make_worktree(litellm, pr_num_resolved, head_sha, wt)
                skill_body = await asyncio.to_thread(skill_path.read_text)
                prompt = _build_prompt(skill_body, payload, wt)

                cc_result = await _invoke_claude(
                    prompt, wt, timeout_s=timeout_s, max_usd=max_usd
                )
                if cc_result is None:
                    log.warning(
                        "karpathy_failed pr=%s err=timeout timeout_s=%s",
                        pr_num,
                        timeout_s,
                    )
                    span.set_attribute("karpathy.timeout", True)
                    return None

                rc, stdout, stderr = cc_result
                envelope, result_text, cost_usd = _parse_envelope(stdout)

                if envelope is not None and out_root_env:
                    # Persist transcript only when caller asked for a
                    # durable out dir; tempdirs get nuked anyway.
                    transcript = out_root / f"pr_{pr_num}_transcript.json"
                    try:
                        transcript.write_bytes(stdout)
                    except OSError as e:
                        log.warning(
                            "karpathy_transcript_write_failed pr=%s err=%s",
                            pr_num,
                            e,
                        )

                # The skill prints its verdict as the last JSON line of
                # `result_text`; fall back to the combined raw stream so
                # we still try to recover something if CC dropped its
                # envelope (older CLI versions, hard errors, etc.).
                review_obj = _extract_json_line(result_text) or _extract_json_line(
                    stdout.decode("utf-8", errors="replace")
                    + "\n"
                    + stderr.decode("utf-8", errors="replace")
                )
                if review_obj is None:
                    log.warning(
                        "karpathy_failed pr=%s err=no_json rc=%s stderr_tail=%s",
                        pr_num,
                        rc,
                        stderr[-400:].decode("utf-8", errors="replace"),
                    )
                    return None

                try:
                    review = KarpathyReview.model_validate(review_obj)
                except ValidationError as e:
                    # Don't dump the whole error (can be many KB on
                    # nested schemas); first error is enough to debug.
                    log.warning(
                        "karpathy_failed pr=%s err=validation msg=%s",
                        pr_num,
                        str(e).splitlines()[0] if str(e) else "?",
                    )
                    return None

                duration_s = round(time.perf_counter() - t_start, 3)
                # Stash CC telemetry on the model via the `extra="allow"`
                # config — keeps the on-the-wire schema clean while still
                # surfacing cost/duration to the caller.
                if cost_usd is not None:
                    review.total_cost_usd = cost_usd  # type: ignore[attr-defined]
                review.duration_s = duration_s  # type: ignore[attr-defined]

                gate = review.merge_gate.safe_for_high_rps_gateway
                span.set_attribute("karpathy.gate", gate)
                span.set_attribute("karpathy.duration_s", duration_s)
                if cost_usd is not None:
                    span.set_attribute("karpathy.cost_usd", cost_usd)

                log.info(
                    "karpathy_done pr=%s gate=%s duration_s=%.2f cost_usd=%s",
                    pr_num,
                    gate,
                    duration_s,
                    f"{cost_usd:.4f}" if cost_usd is not None else "?",
                )
                return review
        except Exception:
            log.exception("karpathy_failed pr=%s err=unhandled", pr_num)
            return None
        finally:
            # Worktree teardown MUST happen even on timeout/exception or
            # the next run trips on a stale `wt-pr-N` directory.
            if wt is not None:
                await _remove_worktree(litellm, wt)
            if cleanup_out_root:
                try:
                    shutil.rmtree(out_root, ignore_errors=True)
                except Exception as e:
                    log.warning("karpathy_tempdir_cleanup_failed err=%s", e)
