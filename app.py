import asyncio
import json
import logging
import os
import re
import secrets
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Literal

import httpx
import logfire
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.sessions import SessionMiddleware
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

import slack_handler

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("litellm-bot")

# Logfire: no-op locally without LOGFIRE_TOKEN, ships traces in prod when set.
# instrument_pydantic_ai() must run before any Agent() is constructed below.
logfire.configure(
    send_to_logfire="if-token-present",
    service_name="litellm-bot",
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)
# Bridge existing stdlib `log.*` calls into Logfire so they show up alongside spans.
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

REQUIRED = ("LITELLM_API_KEY", "GITHUB_TOKEN")
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    raise RuntimeError(f"missing required env vars: {', '.join(missing)}")

SKILLS_ROOT = Path(__file__).parent / "skills/pr-review-agent-skills"
SKILL_DIR = SKILLS_ROOT / "litellm-pr-reviewer"
GATHER_SCRIPT = SKILL_DIR / "scripts/gather_pr_triage_data.py"
PATTERN_SKILL_DIR = SKILLS_ROOT / "litellm-pattern-conformance-reviewer"
PATTERN_GATHER_SCRIPT = PATTERN_SKILL_DIR / "scripts/gather_pattern_data.py"


# Each SKILL.md tells the model to shell out to its gather script. We expose
# those scripts as typed Pydantic AI tools instead so the agent never gets a
# generic Bash. This prefix overrides the "Step 1" bash instruction in each
# SKILL.md without having to fork the upstream skill files.
def _redirect(script_name: str, tool_name: str) -> str:
    return (
        f"TOOL USE: Wherever the instructions below say to run "
        f"`python ${{CLAUDE_SKILL_DIR}}/scripts/{script_name} <ref>`, "
        f"instead call the `{tool_name}` tool with the PR reference. "
        f"It returns the same JSON shape the script would have printed.\n\n"
    )


# Override the "emit prose" steps in each upstream SKILL. The skills are a git
# submodule pointing at BerriAI/pr-review-agent-skills, so we can't edit them
# directly without forking. These prefixes are appended *after* the SKILL text
# so the model reads them as the most recent (and overriding) instruction
# before producing its final structured output.
#
# Single source of truth for the prose-formatting rule. Both override blocks
# reference this so the wording can't drift between blocks. The validator
# `_no_markdown_bold` enforces every marker named here; keep them in sync —
# adding a marker (e.g. backtick-italics) means extending the validator too.
_PROSE_RULE = (
    "plain prose, no markdown bold (`**` / `__`) or italics (`*x*` / `_x_`)"
)

# Stable grounding rule reused across both overrides. Consolidates what was
# previously three scattered phrasings ("never guess" / "never speculate" /
# "DEFAULT IS EMPTY") into one block the model integrates once. Also closes
# the "what if the gather data is malformed?" gap — used to be undefined
# behavior, now it's "return the schema default, never invent."
_GROUNDING_RULE = """
GROUNDING (applies to every field below):
- State only what the gathered data shows. Never guess, never speculate.
- If a field the spec references is missing or null in the gather output,
  return the empty/null default for that schema field — do NOT invent one.
- Lists default to []. Optional scalars default to null.
"""

TRIAGE_OUTPUT_OVERRIDE = f"""
OUTPUT OVERRIDE (supersedes the "Step 6: write the verdict" section above):

Ignore the "write the verdict" instructions in that section. Do not emit
prose with overview / summary / details / file_callouts. Instead, return
the TriageReport schema with these fields:
{_GROUNDING_RULE}
- pr_number, pr_title, pr_author: from the gathered data.
- pr_summary: ONE paragraph (max 600 chars), {_PROSE_RULE}.
  Describe what the PR changes (infer from pr_title + diff_files).
  Target voice — concrete, load-bearing, no marketing tone. Example:
    "Adds a `--retries` flag to the CLI so transient 5xx responses retry
     up to N times with exponential backoff; touches cli.py and
     http_client.py; default behavior unchanged when the flag is omitted."
- files_changed, additions, deletions: leave at the schema default (0).
  Python recomputes these deterministically from the gather output post-run
  — anything you put here is overwritten, so don't waste reasoning on sums.
- pr_related_failures: list of check names from failing_check_contexts where
  the "Step 2: classify each failing check" rules above set
  related_to_pr_diff=True AND is_policy_meta is False.
- unrelated_failures: list of check names where related_to_pr_diff is False
  AND is_policy_meta is False.
- unrelated_failures_also_failing_elsewhere: SUBSET of unrelated_failures.
  Algorithm:
    for name in unrelated_failures:
      include name iff failing_check_contexts[name].also_failing_on_other_prs
      is True (already pre-derived by the gather script — copy it, do NOT
      recompute from other_prs[*].conclusion).
  Result MUST be a subset of unrelated_failures; the rubric depends on it.
- policy_meta_failures: list of check names where is_policy_meta is True.
  These are NEVER in pr_related_failures or unrelated_failures — separate
  bucket. The rubric ignores them; surface them so the contributor knows
  to fix (rebase, sign CLA, etc.) but not as a merge-confidence penalty.
- running_checks: in_progress_checks verbatim.
- greptile_score: the int from the gathered data, or null.
- has_circleci_checks: bool from the gathered data.
- has_merge_conflicts: tri-state bool. Set to true iff the gathered data has
  `mergeable` == false OR `mergeable_state` == "dirty". Set to false iff
  `mergeable` == true AND `mergeable_state` is one of "clean"/"unstable"/
  "has_hooks". Set to null iff `mergeable` is null (GitHub still computing) or
  `mergeable_state` is "unknown" — never guess.

Worked example (illustrative shape only — your values come from the gather
output, not from this template):
{{
  "pr_number": 26451,
  "pr_title": "fix: handle null tool_calls in streaming response",
  "pr_author": "ishaan-jaff",
  "pr_summary": "Guards the streaming response handler against a null tool_calls field that newer OpenAI responses can return; without the guard the handler raises AttributeError and the stream silently drops. No behavior change when tool_calls is present.",
  "files_changed": 0,
  "additions": 0,
  "deletions": 0,
  "pr_related_failures": ["code-quality"],
  "unrelated_failures": ["lint", "codecov/patch"],
  "unrelated_failures_also_failing_elsewhere": ["lint"],
  "policy_meta_failures": [],
  "running_checks": [],
  "greptile_score": 4,
  "has_circleci_checks": true,
  "has_merge_conflicts": false
}}

Do not include any prose justification, summary, or details — Python composes
those from these structured fields downstream. Your job ends at field-filling.
"""

# Pattern override is split into THREE blocks (schema / risk rubric /
# rejection checklist) and composed into PATTERN_OUTPUT_OVERRIDE. Lets us
# tune any one independently — e.g. updating the risk taxonomy doesn't
# force a re-read of the schema or the rejection rules.
_PATTERN_OUTPUT_SCHEMA = f"""
OUTPUT OVERRIDE (supersedes the "Step 4" emit-prose section above):

Ignore the "emit overview / summary" instructions in that section. Do not
write prose. Return the PatternReport schema with these fields:
{_GROUNDING_RULE}
- findings: list of {{file, severity, risk, source, citation, rationale}}
  per the "Step 3: classify" rules above. Use severity blocker/suggestion/nit
  exactly as defined there. rationale max 200 chars, {_PROSE_RULE}.
- tech_debt: list of {{doc_path, code_path, note}} per the existing rule.
  note max 200 chars.

If there are no findings, return findings: []. If no tech_debt, return [].
Do not include overview or summary — Python composes the user-facing card
from your findings list downstream.
"""

_PATTERN_RISK_RUBRIC = """
RISK FIELD — for every finding, also set `risk` to one of high/medium/low
per the "Step 3.5" risk-assignment section of the SKILL above. Severity is
evidence strength; risk is BLAST RADIUS if you're right. They are
independent — a nit-severity finding can be high-risk and vice versa.

Assign risk by answering two questions about the worst-case behavior if
the finding is correct:

  1. Who is affected? users / operators / developers / nobody
  2. How does the bad state recover? unrecoverable / manual /
     self-healing / not-yet-deployed

Then look up the cell in this matrix:

| recovery \\ affected | users  | operators | developers | nobody |
|---------------------|--------|-----------|------------|--------|
| unrecoverable       | high   | high      | medium     | low    |
| manual              | high   | medium    | medium     | low    |
| self-healing        | medium | low       | low        | low    |
| not-yet-deployed    | low    | low       | low        | low    |

State the (affected, recovery) pair in the rationale so a reviewer can
audit the call — e.g. "(users, self-healing) → medium" for a cache
format change, or "(users, manual) → high" for a removed import still
referenced in a handler.

When in doubt between two adjacent cells, pick the higher risk. A
false-positive costs the reviewer 30s; a false-negative ships a bug.
"""

_PATTERN_REJECTION_RULES = """
DEFAULT IS EMPTY. Most small focused PRs should produce findings: []. Only
emit a finding when the patch text shows a concrete deviation from a cited
doc or sibling — never to look thorough, never on truncated patches you
can't read.

REJECTION CHECKLIST — before emitting any finding, verify ALL of these
or drop the finding silently. The cost of one false positive is the
reviewer learning to ignore the agent on the next PR.

1. Rationale describes what the patch DOES (visible in patch text), not
   what it MIGHT do. Reject finding if rationale contains: "may", "might",
   "could", "risks", "if never populated", "potentially", "unverifiable",
   "cannot be verified", "if X happens".
2. Rationale does NOT mention truncation or unreadable patches. Reject
   if it contains: "patch is truncated", "truncated patch", "cannot
   verify", "can't verify", "not visible in this patch". If you can't
   read the change, you cannot make a finding about it.
3. Conforms files emit nothing. Files classified `conforms` or
   `no_pattern_found` in the "Step 3" classification produce zero findings.

Must-flag triggers (the "Step 3.5" / SKILL hard-rules section) are NOT
speculative — they describe shapes visible in the patch text. Apply them
when the patch literally contains them: gated public-route imports,
ERROR-METADATA fields (error_message / error_msg / error_information /
exception_str / failure_reason — NOT general response/content fields) set
to None/empty in non-test code, removal of an import still referenced in
the diff, removal of public config defaults. These emit risk=high.
"""

PATTERN_OUTPUT_OVERRIDE = (
    _PATTERN_OUTPUT_SCHEMA + _PATTERN_RISK_RUBRIC + _PATTERN_REJECTION_RULES
)


SYSTEM_PROMPT = (
    _redirect("gather_pr_triage_data.py", "gather_pr_data")
    + (SKILL_DIR / "SKILL.md").read_text()
    + TRIAGE_OUTPUT_OVERRIDE
)
PATTERN_SYSTEM_PROMPT = (
    _redirect("gather_pattern_data.py", "gather_pattern_data")
    + (PATTERN_SKILL_DIR / "SKILL.md").read_text()
    + PATTERN_OUTPUT_OVERRIDE
)

# --- Typed agent outputs + card rendering -------------------------------------
# Both agents return Pydantic models (not free-form prose). Python composes the
# final Slack card from those typed fields via fuse() + render_card(), so the
# layout is guaranteed identical across runs and the only thing the model
# controls is the *content* of each slot.

# Length caps below are deliberate. They're enforced by Pydantic AI: if the
# model exceeds them, the run retries with the validation error as feedback.
# Pick numbers that fit comfortably in a Slack message without truncation.


# Single-char emphasis (italics) detector. Matches `*x*` and `_x_` while
# explicitly NOT matching:
#   - `**bold**` / `__bold__`  (the bold case is checked separately so the
#     error message stays specific)
#   - snake_case identifiers  (boundary `\w` rejects `_` adjacent to a word
#     char on either side)
#   - lone `*` or `_` characters (must close on a matching delimiter)
#   - delimiters with whitespace immediately inside (`* foo *`) — markdown
#     parsers ignore those, so neither do we
# Body capped at 200 chars to keep pathological inputs O(n).
_ITALICS_RE = re.compile(
    r"(?<![*_\w])([*_])(?=\S)(?:(?!\1).){1,200}?(?<=\S)\1(?![*_\w])"
)


def _no_markdown_bold(v: str) -> str:
    """Hard-fail any agent prose that contains markdown emphasis; the
    deterministic Slack rendering owns formatting end-to-end. Pydantic AI
    catches the ValueError and retries the model with this message as
    feedback. Name kept (`_no_markdown_bold`) for back-compat with the
    existing field_validator bindings; the check now covers both bold AND
    italics so the prompt's "no markdown bold or italics" rule is actually
    enforced — used to be a silent gap where italic-formatted summaries
    shipped through.
    """
    if "**" in v or "__" in v:
        raise ValueError("text fields must not contain markdown bold (** or __)")
    if _ITALICS_RE.search(v):
        raise ValueError(
            "text fields must not contain markdown italics (*word* or _word_)"
        )
    return v


class TriageReport(BaseModel):
    """Structured CI/policy signals for one PR. Filled by the triage agent."""

    pr_number: int
    pr_title: str
    pr_author: str
    # 1 paragraph: what the PR changes. Drives the *Triage Summary* line.
    pr_summary: str = Field(..., max_length=600)
    # Diff size, surfaced on the card so reviewers can eyeball whether a deeper
    # pass is needed. Computed in Python from the gather payload via
    # _overlay_diff_size after the agent run — the model is told (in
    # TRIAGE_OUTPUT_OVERRIDE) to leave these at 0 because anything it emits
    # gets overwritten. Defaulted to 0 so older callers / the fallback card
    # path don't have to construct them, and so a missing gather payload
    # degrades gracefully (size_line just renders empty).
    files_changed: int = Field(default=0, ge=0)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    # Check names only. Classification logic lives in the SKILL.
    pr_related_failures: list[str] = Field(default_factory=list)
    unrelated_failures: list[str] = Field(default_factory=list)
    # Subset of unrelated_failures that were ALSO failing on at least one of
    # the sampled `other_prs` (gather script's `failing_check_contexts[*].
    # other_prs` field). Distinguishes "infra flake breaking everyone" from
    # "infra-shaped failure unique to this PR". Drives the rubric so the
    # latter still docks 1 (silent-pass canary stays armed) while the former
    # only docks 0 — the bug we hit on PRs #26385 / #26011 / #26122 where
    # the agent flipped to BLOCKED on infra noise the reviewer shrugged off.
    # Names here MUST also appear in unrelated_failures (subset relationship);
    # the rubric assumes that and won't fire double-penalties.
    unrelated_failures_also_failing_elsewhere: list[str] = Field(default_factory=list)
    # Policy/meta checks (e.g. `Verify PR source branch`, `DCO`, `cla-bot`)
    # that operate on PR shape, not code. Pre-flagged by the gather script
    # via `failing_check_contexts[*].is_policy_meta`. NEVER appear in
    # pr_related_failures or unrelated_failures — they live in their own
    # zero-penalty bucket because their failure tells the reviewer nothing
    # about the diff. Surface them on the card so the contributor still
    # knows to fix the policy issue (rebase from upstream main, sign the
    # CLA, etc.); just don't dock the merge-confidence score for them.
    # Recovered #26419 from the 2026-04-25 eval where a UI dropdown PR got
    # 4/5 BLOCKED because `Verify PR source branch` failed.
    policy_meta_failures: list[str] = Field(default_factory=list)
    running_checks: list[str] = Field(default_factory=list)
    greptile_score: int | None = None
    has_circleci_checks: bool
    # Tri-state: True = confirmed conflicts (mergeable=false or state="dirty"),
    # False = confirmed clean merge, None = GitHub still computing or unknown.
    # Treated as a hard blocker only when True; None never blocks (we don't
    # punish the PR for GitHub's lazy compute). Defaults to None so the
    # fallback card path and older callers don't have to populate it.
    has_merge_conflicts: bool | None = None

    _strip_bold = field_validator("pr_summary")(_no_markdown_bold)


class PatternFinding(BaseModel):
    file: str
    # `severity` = evidence strength (how confident the finding is true).
    # `risk` = impact strength (how bad if true). The two are orthogonal.
    # Eval lesson from BerriAI/litellm PRs #26294 and #26074: a pattern
    # reviewer that only sees evidence-count buckets a "lazy import gating
    # a public route" as `nit` (only one sibling diverges) and an
    # "error_msg = None" as `nit` (single-file, no doc), even though both
    # are production-down-class risks. Splitting risk out lets fuse() dock
    # on impact regardless of how thin the evidence is, and lets the human
    # reading the card see "yes I'm only 60% sure, but if I'm right this
    # breaks prod" as a first-class signal.
    severity: Literal["blocker", "suggestion", "nit"]
    # Default `low` for back-compat: any older agent output (or test fixture)
    # that omits `risk` parses as low-risk and behaves like the pre-risk
    # rubric. New runs are expected to populate it per Step 3.5 of the
    # pattern SKILL.
    risk: Literal["high", "medium", "low"] = "low"
    source: Literal["docs", "code"]
    citation: str
    rationale: str = Field(..., max_length=200)

    _strip_emphasis = field_validator("rationale")(_no_markdown_bold)


class TechDebtItem(BaseModel):
    doc_path: str
    code_path: str
    note: str = Field(..., max_length=200)

    _strip_emphasis = field_validator("note")(_no_markdown_bold)


class PatternReport(BaseModel):
    """Pattern-conformance findings + ambient tech debt. Filled by pattern agent."""

    findings: list[PatternFinding] = Field(default_factory=list)
    tech_debt: list[TechDebtItem] = Field(default_factory=list)


class TriageCard(BaseModel):
    """Final card. Built by fuse() from TriageReport + PatternReport, never by
    the model directly."""

    summary: str
    # Pre-rendered "N lines across M files (+a / -d)" or "size unknown".
    # Composed in fuse() so render_card stays a dumb formatter.
    size_line: str
    # Pre-rendered "⚠️ N check(s) failing: foo, bar" or "" when none. Composed
    # in fuse() from triage.pr_related_failures + triage.unrelated_failures so
    # the card *always* names failing checks regardless of verdict. Closes the
    # silent-pass bug where unrelated_failures were never surfaced on READY
    # cards (PR #26451 of BerriAI/litellm: code-quality failed, card said 5/5).
    failing_line: str
    score: int = Field(..., ge=0, le=5)
    verdict: Literal["READY", "BLOCKED", "WAITING"]
    emoji: str
    verdict_one_liner: str
    justification: str


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _join(items: list[str], cap: int = 3) -> str:
    head = items[:cap]
    tail = "" if len(items) <= cap else f" (+{len(items) - cap} more)"
    return ", ".join(head) + tail


def _count(p: PatternReport, sev: str) -> int:
    return sum(1 for f in p.findings if f.severity == sev)


def _count_risk(p: PatternReport, risk: str) -> int:
    return sum(1 for f in p.findings if f.risk == risk)


# Wide-fan-out heuristic. A patch that touches lots of files but adds very
# few lines per file is structurally suspect: it tends to be a "find-replace
# or apply-this-helper-everywhere" change where the next contributor who
# adds a similar callsite without remembering the helper silently regresses
# the fix. Eval lesson from BerriAI/litellm PR #26284 (66 files, inline
# urllib.parse.quote() at every callsite). Thresholds picked to NOT trip
# legitimate large refactors (which are usually deletion-heavy or have
# meaningful per-file logic) — see test_fuse.py for the boundary cases.
_WIDE_FANOUT_FILE_THRESHOLD = 30
_WIDE_FANOUT_LINES_PER_FILE = 5


def _is_wide_low_density_fanout(t: TriageReport) -> bool:
    if t.files_changed < _WIDE_FANOUT_FILE_THRESHOLD:
        return False
    total = t.additions + t.deletions
    return (total / t.files_changed) < _WIDE_FANOUT_LINES_PER_FILE


def _unrelated_unique_to_pr(t: TriageReport) -> list[str]:
    """Unrelated failures that are NOT also red on the sampled other_prs.

    These are the only unrelated failures that should dock the score:
    a check failing here AND on neighboring PRs is infra/repo-wide noise
    the contributor can't fix; docking for it is just punishing them for
    being downstream of someone else's outage. A check failing only here
    is suspicious — bucketed unrelated by the model but maybe wrongly,
    so the silent-pass canary still needs to fire.

    Order-preserving so the user-facing list keeps the model's ordering.
    """
    elsewhere = set(t.unrelated_failures_also_failing_elsewhere)
    return [n for n in t.unrelated_failures if n not in elsewhere]


# Rubric: one row per penalty. Each row is (weight, predicate, label_fn).
# label_fn returns the human-readable reason that lands in the justification.
# Edit this table to retune scoring; everything else flows from it.
_RubricRow = tuple[
    int,
    Callable[[TriageReport, PatternReport], bool],
    Callable[[TriageReport, PatternReport], str],
]
_RUBRIC: list[_RubricRow] = [
    # Merge conflicts are an unconditional blocker per stored policy: a PR
    # with conflicts cannot merge regardless of green CI. Weight 5 forces
    # score to 0 so the verdict is BLOCKED even on otherwise-clean PRs. Only
    # fires when has_merge_conflicts is exactly True; None (GitHub still
    # computing) never blocks so a slow background job doesn't false-positive.
    (
        5,
        lambda t, p: t.has_merge_conflicts is True,
        lambda t, p: "merge conflicts (rebase against base branch)",
    ),
    (
        2,
        lambda t, p: bool(t.pr_related_failures),
        lambda t, p: f"{_plural(len(t.pr_related_failures), 'PR-related CI failure')} "
        f"({_join(t.pr_related_failures)})",
    ),
    (
        2,
        lambda t, p: any(f.severity == "blocker" for f in p.findings),
        lambda t, p: f"{_plural(_count(p, 'blocker'), 'doc violation')} "
        f"({_join([f.file for f in p.findings if f.severity == 'blocker'])})",
    ),
    # Risk-based docks. Severity is "how confident am I"; risk is "how bad
    # if I'm right". Decoupling them lets a low-evidence finding still block
    # when the underlying change pattern is production-down-class. See the
    # PatternFinding docstring + the 2026-04-25 v2-eval misses on PRs #26294
    # (gated import → silent route 404) and #26074 (error_msg = None →
    # masked failure surface) — both got nit findings the old rubric ignored.
    (
        2,
        lambda t, p: any(f.risk == "high" for f in p.findings),
        lambda t, p: f"{_plural(_count_risk(p, 'high'), 'high-risk pattern finding')} "
        f"({_join([f.file for f in p.findings if f.risk == 'high'])})",
    ),
    (
        1,
        lambda t, p: any(f.risk == "medium" for f in p.findings),
        lambda t, p: f"{_plural(_count_risk(p, 'medium'), 'medium-risk pattern finding')} "
        f"({_join([f.file for f in p.findings if f.risk == 'medium'])})",
    ),
    # Wide low-density fan-out. Catches the brittle-helper-everywhere shape
    # where a patch touches many files but barely changes any of them — the
    # exact pattern the human grader called out as "if someone adds a new
    # file later and forgets the helper, we're back to square one." Lives
    # on triage (not pattern) because the file count + line count come from
    # the diff metadata, not from sibling/doc analysis. Soft penalty so
    # legitimate wide refactors aren't unfairly buried — they'll typically
    # also fail one of the heavier rows (tests, doc-violation, CI).
    (
        1,
        lambda t, p: _is_wide_low_density_fanout(t),
        lambda t, p: f"wide low-density fan-out "
        f"({t.files_changed} files, +{t.additions}/-{t.deletions}) "
        f"— inline change duplicated across many sites is brittle; "
        f"prefer a single-source helper that future callsites can't miss",
    ),
    (
        1,
        lambda t, p: t.greptile_score is not None and t.greptile_score < 4,
        lambda t, p: f"Greptile {t.greptile_score}/5",
    ),
    (
        1,
        lambda t, p: t.greptile_score is None,
        lambda t, p: "Greptile has not reviewed this PR yet",
    ),
    # Soft-dock for unrelated failures UNIQUE TO THIS PR so a READY card
    # never silently swallows a red check the model can't explain by
    # pointing at neighboring PRs. PR #26451 of BerriAI/litellm shipped
    # a 5/5 READY card while `code-quality` was failing because the model
    # bucketed it as unrelated (null log + annotations only mentioned
    # `.github`) and the rubric had no row for that bucket.
    #
    # Weight 1 because the model might still be right (real infra flake),
    # so the verdict drops to BLOCKED but not as hard as a confirmed
    # PR-related failure. Either way the failure name lands in the
    # justification + the card's failing_line.
    #
    # Filtered through `_unrelated_unique_to_pr` so a failure ALSO red on
    # sampled other_prs (clearly broken-for-everyone infra) does NOT dock
    # — fixed the false-BLOCKED on PRs #26385 / #26011 / #26122 from the
    # 2026-04-25 eval, where one-line bumps got 4/5 BLOCKED for cross-repo
    # `lint` / `codecov` flakes the human grader correctly read as noise.
    (
        1,
        lambda t, p: bool(_unrelated_unique_to_pr(t)),
        lambda t, p: f"{_plural(len(_unrelated_unique_to_pr(t)), 'unrelated CI failure')} "
        f"unique to this PR ({_join(_unrelated_unique_to_pr(t))})",
    ),
    # Note: missing CircleCI is intentionally NOT a penalty. OSS PRs from
    # external contributors often can't trigger CircleCI (it's gated on repo
    # secrets), so docking them would punish the contributor for a config
    # constraint they have no control over. The skill makes the same call —
    # see litellm-pr-reviewer/SKILL.md Step 4. has_circleci_checks still flows
    # through to the drill-down for reviewer awareness.
]


def _compose_one_liner(
    verdict: str,
    penalties: list[str],
    triage: TriageReport,
    pattern: PatternReport,
) -> str:
    if verdict == "WAITING":
        n = len(triage.running_checks)
        return f"{_plural(n, 'check')} still running: {_join(triage.running_checks)}."
    if verdict == "READY":
        return "Ready to ship."
    # BLOCKED: name the most-blocking thing first, in priority order.
    # Merge conflicts come first — no amount of green CI matters if the PR
    # can't be merged at all.
    if triage.has_merge_conflicts is True:
        return "Merge conflicts — rebase against base branch first."
    if triage.pr_related_failures:
        return f"{_plural(len(triage.pr_related_failures), 'PR-related CI failure')} need fixes first."
    blocker_n = _count(pattern, "blocker")
    if blocker_n:
        return f"{_plural(blocker_n, 'pattern blocker')} need fixes first."
    # High-risk findings come next in the priority chain because they signal
    # production-impact pattern smells (gated public-route imports, silent
    # error suppression, etc.) — louder than a vanilla soft penalty even
    # when evidence is thin.
    high_risk_n = _count_risk(pattern, "high")
    if high_risk_n:
        return f"{_plural(high_risk_n, 'high-risk pattern finding')} need a closer look first."
    # Only soft penalties left → name the top one verbatim.
    return penalties[0][:1].upper() + penalties[0][1:] + "."


def _compose_justification(
    verdict: str,
    score: int,
    penalties: list[str],
    triage: TriageReport,
    pattern: PatternReport,
) -> str:
    if verdict == "WAITING":
        bits = [
            f"Greptile {'pending' if triage.greptile_score is None else f'{triage.greptile_score}/5'}",
            f"{_plural(len(pattern.findings), 'pattern finding')}",
            f"CircleCI {'present' if triage.has_circleci_checks else 'absent'}",
        ]
        return (
            "Verdict provisional. Current signals: "
            + ", ".join(bits)
            + ". Score will update once checks complete."
        )
    if verdict == "READY":
        ci_note = (
            "CircleCI passed"
            if triage.has_circleci_checks
            else "no CircleCI runs (OSS-typical)"
        )
        # Even a READY card must name any failing checks the model bucketed as
        # unrelated, otherwise the user reads "All checks green" while the PR
        # has a red ❌ in GitHub. After the unique-vs-elsewhere split, READY
        # implies every unrelated failure was ALSO red on neighboring PRs
        # (otherwise the unique-only rubric row would have docked → BLOCKED),
        # so the prose can call that out honestly instead of just "verify".
        if triage.unrelated_failures:
            return (
                f"Greptile {triage.greptile_score}/5, "
                f"no blocking pattern findings, {ci_note}. "
                f"{_plural(len(triage.unrelated_failures), 'check')} failing "
                f"but also red on neighboring PRs (broken-for-everyone infra): "
                f"{_join(triage.unrelated_failures)}."
            )
        return (
            f"All checks green. Greptile {triage.greptile_score}/5, "
            f"no blocking pattern findings, {ci_note}."
        )
    sentences: list[str] = []
    if penalties:
        sentences.append("Score docked for: " + "; ".join(penalties) + ".")
    # Two separate sentences for the two unrelated buckets. The unique-to-PR
    # subset is what the rubric just docked (already in penalties); call it
    # out by name. The "also failing elsewhere" subset is non-docking — say
    # so explicitly so the reviewer knows we saw it and chose not to penalize.
    unique = _unrelated_unique_to_pr(triage)
    if unique:
        sentences.append(
            f"{_plural(len(unique), 'unrelated CI failure')} unique to this PR "
            f"({_join(unique)}) — not related to this diff but worth a glance."
        )
    if triage.unrelated_failures_also_failing_elsewhere:
        sentences.append(
            f"{_plural(len(triage.unrelated_failures_also_failing_elsewhere), 'check')} "
            f"also red on neighboring PRs "
            f"({_join(triage.unrelated_failures_also_failing_elsewhere)}) "
            f"— infra-wide noise, no penalty."
        )
    nits = _count(pattern, "nit")
    sugg = _count(pattern, "suggestion")
    extras = []
    if sugg:
        extras.append(_plural(sugg, "suggestion"))
    if nits:
        extras.append(_plural(nits, "nit"))
    if extras:
        sentences.append("Also " + " and ".join(extras) + " — see thread.")
    return " ".join(sentences) or "No specific signal — see thread for detail."


def _format_size_line(triage: TriageReport) -> str:
    """One-line diff-size signal so reviewers can eyeball PR scale at a glance.

    Returns "" when no size info was reported (e.g. fallback card path) so
    render_card can skip the line cleanly without printing a misleading "0 lines".
    """
    total = triage.additions + triage.deletions
    if triage.files_changed == 0 and total == 0:
        return ""
    return (
        f"{_plural(total, 'line')} across "
        f"{_plural(triage.files_changed, 'file')} "
        f"(+{triage.additions} / -{triage.deletions})"
    )


def _format_failing_line(triage: TriageReport) -> str:
    """One-line "failing checks" call-out for the card.

    Pulls from BOTH pr_related_failures and unrelated_failures so the user
    always sees check names regardless of how the model bucketed them. Also
    prepends a merge-conflict notice so a conflict-only blocker (no failing
    CI) still surfaces visibly on the card. Empty string when there's
    nothing to report — render_card skips the line.

    Why two buckets surface the same way: see the design-doc bug from PR
    #26451 of BerriAI/litellm. The model is allowed to be wrong about which
    bucket a check belongs in; the card must not be silent either way.
    """
    parts: list[str] = []
    if triage.has_merge_conflicts is True:
        parts.append("⚠️ merge conflicts")
    all_failing = list(triage.pr_related_failures) + list(triage.unrelated_failures)
    if all_failing:
        parts.append(
            f"⚠️ {_plural(len(all_failing), 'check')} failing: {_join(all_failing)}"
        )
    # Policy/meta gets its own segment so a contributor sees the ask
    # (rebase / sign CLA) without the score-docking warning glyph.
    # Different separator glyph (ℹ️) signals zero-penalty / informational.
    if triage.policy_meta_failures:
        parts.append(
            f"ℹ️ {_plural(len(triage.policy_meta_failures), 'policy check')} "
            f"failing: {_join(triage.policy_meta_failures)}"
        )
    return " · ".join(parts)


def fuse(triage: TriageReport, pattern: PatternReport) -> TriageCard:
    """Combine the two agent outputs into the single card the user sees.

    Pure function. No I/O. Deterministic given (triage, pattern). Tested in
    tests/test_fuse.py.
    """
    score = 5
    penalties: list[str] = []
    for weight, predicate, label in _RUBRIC:
        if predicate(triage, pattern):
            score -= weight
            penalties.append(label(triage, pattern))
    score = max(score, 0)

    # WAITING wins over score-derived states because the score is provisional
    # while checks are still running.
    if triage.running_checks:
        verdict, emoji = "WAITING", "⏳"
    elif score == 5:
        verdict, emoji = "READY", "✅"
    else:
        verdict, emoji = "BLOCKED", "❌"

    return TriageCard(
        summary=triage.pr_summary,
        size_line=_format_size_line(triage),
        failing_line=_format_failing_line(triage),
        score=score,
        verdict=verdict,
        emoji=emoji,
        verdict_one_liner=_compose_one_liner(verdict, penalties, triage, pattern),
        justification=_compose_justification(
            verdict, score, penalties, triage, pattern
        ),
    )


def render_card(card: TriageCard) -> str:
    """Slack mrkdwn for the top-level message. Exact shape locked here."""
    size = f"_{card.size_line}_\n\n" if card.size_line else ""
    # failing_line lives BELOW the verdict line so the eye lands on the
    # confidence score first, then immediately sees the check names if any
    # are red. Empty string skips the line cleanly (same pattern as size_line).
    failing = f"{card.failing_line}\n" if card.failing_line else ""
    return (
        f"*Triage Summary*\n"
        f"{card.summary}\n\n"
        f"{size}"
        f"*Merge Confidence: {card.score}/5*  {card.emoji} {card.verdict}\n"
        f"{failing}"
        f"{card.verdict_one_liner}\n\n"
        f"{card.justification}"
    )


def render_drilldown(triage: TriageReport, pattern: PatternReport) -> str:
    """Slack mrkdwn for the threaded follow-up: full failure list + findings.

    Posted as a reply so it doesn't compete with the card visually but is one
    click away for anyone who wants the detail.
    """
    lines: list[str] = ["*Drill-down*"]

    if triage.has_merge_conflicts is True:
        lines.append("\n_Merge state_")
        lines.append("  • merge conflicts — branch must be rebased before it can merge")

    if triage.pr_related_failures:
        lines.append("\n_PR-related failures_")
        for c in triage.pr_related_failures:
            lines.append(f"  • {c}")
    if triage.unrelated_failures:
        lines.append("\n_Unrelated failures_")
        for c in triage.unrelated_failures:
            lines.append(f"  • {c}")
    if triage.policy_meta_failures:
        lines.append("\n_Policy / meta failures (zero-penalty)_")
        for c in triage.policy_meta_failures:
            lines.append(
                f"  • {c} — operates on PR shape, not code; fix per repo policy"
            )
    if triage.running_checks:
        lines.append("\n_Still running_")
        for c in triage.running_checks:
            lines.append(f"  • {c}")

    if pattern.findings:
        lines.append("\n_Pattern findings_")
        # Sort high-risk first so the reviewer's eye lands on the dangerous
        # findings before the naming nits, regardless of severity ordering.
        risk_order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(pattern.findings, key=lambda x: risk_order.get(x.risk, 3)):
            risk_tag = f" risk={f.risk}" if f.risk != "low" else ""
            lines.append(
                f"  • [{f.severity}{risk_tag}] `{f.file}` — {f.rationale} "
                f"(source: {f.source}, {f.citation})"
            )
    if pattern.tech_debt:
        lines.append("\n_Tech debt (FYI, not blocking)_")
        for td in pattern.tech_debt:
            lines.append(f"  • `{td.code_path}` vs `{td.doc_path}` — {td.note}")

    if len(lines) == 1:
        lines.append("Nothing to drill into. Card has the full story.")
    return "\n".join(lines)


def render_fallback_card(pr_url: str, error: str) -> str:
    """Same shape as render_card() so the reader's eye lands in the same slots
    even when the agent crashed. Triggered from the exception path in the
    runner functions below.
    """
    return (
        f"*Triage Summary*\n"
        f"Could not analyze {pr_url} automatically.\n\n"
        f"*Merge Confidence: ?/5*  ⚠️ ERROR\n"
        f"Manual review required.\n\n"
        f"Reason: {error[:300]}"
    )


model = OpenAIChatModel(
    os.environ.get("LITELLM_MODEL", "claude-sonnet-4-6"),
    provider=LiteLLMProvider(
        api_base=os.environ.get("LITELLM_API_BASE", "http://0.0.0.0:4000"),
        api_key=os.environ["LITELLM_API_KEY"],
    ),
)


# --- LiteLLM memory client ----------------------------------------------------
# Memory is a single freeform document stored under one key (AGENT_ID) in the
# LiteLLM proxy at /v1/memory. The chat_agent owns it via two tools:
#   - add_memory(text): smart-merge a new fact into the doc (LLM decides
#     whether to replace an existing line or append).
#   - reset_memory(): wipe the doc.
# Slack-driven runs (review_pr → triage + pattern agents) never see these
# tools, so memory writes can't happen via Slack by construction. Both Slack
# and chat *read* the doc via _memory_context() so triage/pattern see the
# user's stable prefs.

LITELLM_MEMORY_BASE = os.environ.get("LITELLM_API_BASE", "http://0.0.0.0:4000").rstrip(
    "/"
)
_MEMORY_HEADERS = {"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"}

# Single key per bot. Multiple bots sharing a LiteLLM key keep separate docs
# because each uses its own AGENT_ID as the row key. Empty AGENT_ID falls
# back to "agent-memory" so the bot still works in dev without env config.
AGENT_ID = os.environ.get("AGENT_ID", "agent-memory")
log.info("memory key: %s", AGENT_ID)


async def _memory_get_doc() -> str:
    """Read the current memory doc, or "" if not yet created."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{LITELLM_MEMORY_BASE}/v1/memory/{AGENT_ID}",
            headers=_MEMORY_HEADERS,
        )
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        return r.json().get("value", "") or ""


async def _memory_put_doc(value: str) -> None:
    """Replace the entire memory doc."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.put(
            f"{LITELLM_MEMORY_BASE}/v1/memory/{AGENT_ID}",
            headers=_MEMORY_HEADERS,
            json={"value": value},
        )
        r.raise_for_status()


class _MemoryUpdate(BaseModel):
    """Merger output: capability verdict + (when honored) the new doc."""

    can_honor: bool
    reason: str = Field(..., max_length=200)
    merged: str = ""


# _merger_agent is built at the bottom of this module (see _build_merger_agent
# below). It needs to introspect chat_agent's registered tools, which don't
# exist until *after* their @chat_agent.tool_plain decorators run. Forward-
# declared as None here so _memory_check_and_merge can reference it at call
# time; the real Agent is assigned post-toolset construction.
_merger_agent: Agent | None = None


async def _memory_check_and_merge(existing: str, new_fact: str) -> _MemoryUpdate:
    """Single LLM call: capability-gate the new fact, and if honored, return
    the merged doc. Best-effort on errors: returns can_honor=true with the
    existing doc untouched + new fact appended, so a flaky LLM never silently
    eats a save."""
    if _merger_agent is None:
        # Programmer error — the bottom-of-module build never ran. Fail loud
        # instead of silently appending unvalidated memory.
        raise RuntimeError("_merger_agent not initialized; module load order broken")
    new_fact = new_fact.strip()
    prompt = (
        f"Existing memory:\n---\n{existing}\n---\n\n"
        f"Proposed new fact:\n{new_fact}\n\n"
        "Decide can_honor and (if honoring) return the merged doc."
    )
    try:
        result = await _merger_agent.run(prompt)
    except Exception as e:
        log.warning("memory_merge_failed err=%s", e)
        fallback = (
            new_fact if not existing.strip() else f"{existing.rstrip()}\n{new_fact}"
        )
        return _MemoryUpdate(
            can_honor=True,
            reason=f"merger unavailable ({e}); appended without check",
            merged=fallback,
        )
    return result.output


async def _memory_context() -> str:
    """Format the memory doc as a 'User context' block to prepend to PR-review
    prompts. Read-only — both Slack and chat call this so triage/pattern see
    the user's stable prefs without needing a tool call. Best-effort: any
    fetch error logs a warning and returns "" so PR review is never blocked
    by memory infra problems.
    """
    try:
        doc = await _memory_get_doc()
    except Exception as e:
        log.warning("memory_context_fetch_failed err=%s", e)
        return ""
    doc = doc.strip()
    if not doc:
        return ""
    return "User context (stable prefs/defaults; honor when relevant):\n" f"{doc}\n\n"


# retries=3: Pydantic AI default is 1, which gave up on PR #26415 of
# BerriAI/litellm with "Exceeded maximum retries (1) for output validation".
# Triage/pattern outputs have hard length caps (max_length=200/600) the
# model occasionally overruns on multi-file diffs; an extra 1-2 retries lets
# it self-correct with the validator's error message instead of producing
# a fallback card. Cost upper bound: 2 extra LLM calls per failed PR, only
# on the unhappy path. Worth the spend.
agent = Agent(model, system_prompt=SYSTEM_PROMPT, output_type=TriageReport, retries=3)
pattern_agent = Agent(
    model, system_prompt=PATTERN_SYSTEM_PROMPT, output_type=PatternReport, retries=3
)


def _run_gather_script(script: Path, pr_ref: str) -> dict:
    with logfire.span(
        "gather_script {script_name}",
        script_name=script.name,
        pr_ref=pr_ref,
    ) as span:
        proc = subprocess.run(
            ["python", str(script), pr_ref],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        span.set_attribute("exit_code", proc.returncode)
        span.set_attribute("stdout_bytes", len(proc.stdout))
        if proc.returncode != 0:
            raise ModelRetry(
                f"gather script failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ModelRetry(
                f"gather script returned non-JSON: {e}; stdout head: {proc.stdout[:200]}"
            )


@agent.tool_plain
def gather_pr_data(pr_ref: str) -> dict:
    """Gather PR checks, diff files, Greptile score, and CircleCI logs as a JSON object.

    Args:
        pr_ref: A full GitHub PR URL (https://github.com/owner/repo/pull/N)
                or short ref (owner/repo#N). If only "#N" is given,
                BerriAI/litellm is assumed.
    """
    return _run_gather_script(GATHER_SCRIPT, pr_ref)


@pattern_agent.tool_plain
def gather_pattern_data(pr_ref: str) -> dict:
    """Gather PR diff, doc excerpts, and sibling-file excerpts as a JSON object.

    Args:
        pr_ref: A full GitHub PR URL (https://github.com/owner/repo/pull/N)
                or short ref (owner/repo#N). If only "#N" is given,
                BerriAI/litellm is assumed.
    """
    return _run_gather_script(PATTERN_GATHER_SCRIPT, pr_ref)


def _last_gather_payload(messages: list, tool_name: str) -> dict | None:
    """Return the payload from the most recent successful call to `tool_name`
    in `messages`, or None if none was called.

    Pydantic AI sometimes hands tool returns back as the raw dict, sometimes
    as a JSON string (depends on serialization path); accept both. Iterates
    forward and keeps the last hit so retries pick up the freshest data.
    """
    last: dict | None = None
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name == tool_name
            ):
                payload = part.content
                if isinstance(payload, dict):
                    last = payload
                elif isinstance(payload, str):
                    try:
                        parsed = json.loads(payload)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(parsed, dict):
                        last = parsed
    return last


def _overlay_diff_size(report: TriageReport, gather: dict | None) -> TriageReport:
    """Recompute files_changed / additions / deletions from the gather output
    and overlay them onto `report`, replacing whatever the model emitted.

    Why post-process instead of asking the model: every sum is a chance for
    the model to miscount on a 40-file PR, and the agent burns reasoning
    tokens on arithmetic Python can do for free. The override prompt now
    tells the model to leave these at 0 and trust this fixup. If the gather
    payload is missing or shaped unexpectedly we leave the model's values
    in place (defaults to 0), which preserves the fallback-card behavior
    `_format_size_line` already handles.
    """
    if not gather:
        return report
    files = gather.get("diff_files")
    if not isinstance(files, list):
        return report
    files_changed = len(files)
    additions = sum(int(f.get("additions") or 0) for f in files if isinstance(f, dict))
    deletions = sum(int(f.get("deletions") or 0) for f in files if isinstance(f, dict))
    return report.model_copy(
        update={
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
        }
    )


async def _run_triage(
    prompt: str,
    history: list | None = None,
) -> tuple[TriageReport | None, list, str | None]:
    """Run the triage agent, returning (report, new_history, error_message).

    On success: report is a TriageReport, error_message is None.
    On failure (model crashed, validation never converged, etc.): report is
    None and error_message holds the exception text. The caller renders a
    fallback card so the user always sees a valid card shape.
    """
    try:
        result = await agent.run(prompt, message_history=history or [])
    except Exception as e:
        log.exception("triage_agent_failed prompt_head=%s", prompt[:80])
        return (None, history or [], str(e))
    messages = list(result.all_messages())
    # Diff sizes are computed from the gather payload deterministically
    # rather than trusted from the model — see _overlay_diff_size.
    report = _overlay_diff_size(
        result.output, _last_gather_payload(messages, "gather_pr_data")
    )
    return (report, messages, None)


async def _run_pattern(
    prompt: str,
    history: list | None = None,
) -> tuple[PatternReport | None, list, str | None]:
    """Mirror of _run_triage for the pattern-conformance agent."""
    try:
        result = await pattern_agent.run(prompt, message_history=history or [])
    except Exception as e:
        log.exception("pattern_agent_failed prompt_head=%s", prompt[:80])
        return (None, history or [], str(e))
    return (result.output, list(result.all_messages()), None)


# --- Chat agent: the front door for the dev chat UI ---------------------------
# Two skills, exposed as tools:
#   - run_pr_review(pr_url): runs triage + pattern in parallel, returns the
#     same rendered card Slack would post.
#   - add_memory(text) / reset_memory(): writes to the single memory doc.
#     Only this agent has them, so they're unreachable from the Slack path
#     by construction.
#
# Memory is auto-injected into the system prompt on every turn (dynamic=True)
# so the model never needs a "read_memory" tool — it just sees the current
# doc as ambient context. That keeps the tool surface to two action verbs
# (add, reset) instead of read+write+delete.
#
# String output (no schema) so it can answer free-form or pass through a card.
# The Slack path does NOT go through this agent — it calls _run_triage /
# _run_pattern directly so the triage+pattern pair stays parallel and the
# existing card-rendering contract is preserved.

CHAT_SYSTEM_PROMPT = (
    "You are the litellm-bot dev assistant.\n\n"
    "You have two skills, exposed as tools:\n"
    "1. run_pr_review(pr_url) — full PR triage + pattern review. Use this "
    "whenever the user mentions a GitHub PR (URL or `owner/repo#N`). Return "
    "its output verbatim; do not paraphrase the card.\n"
    "2. add_memory(text) / reset_memory() — long-term memory. The current "
    "memory doc is shown to you below; you don't need a 'read' tool. Call "
    "add_memory whenever the user asks you to remember a stable fact "
    "(preferences, defaults, policies). It auto-merges: a similar fact "
    "replaces the existing one, otherwise it's appended. add_memory runs a "
    "capability check first; if it returns {refused: true, reason: ...}, "
    "the save did NOT happen — relay the reason to the user verbatim and "
    "ask them how they'd like to rephrase. Call reset_memory only when the "
    "user explicitly asks you to wipe/forget everything.\n\n"
    "If the user's request doesn't fit either skill, just answer normally."
)

chat_agent = Agent(model, system_prompt=CHAT_SYSTEM_PROMPT, output_type=str)


@chat_agent.system_prompt(dynamic=True)
async def _inject_memory() -> str:
    """Dynamic system prompt: re-fetch memory on every turn so adds within a
    thread show up immediately. Best-effort — failure returns empty string
    rather than blocking the chat."""
    try:
        doc = (await _memory_get_doc()).strip()
    except Exception as e:
        log.warning("chat_memory_inject_failed err=%s", e)
        return ""
    if not doc:
        return "Current memory: (empty)"
    return f"Current memory:\n---\n{doc}\n---"


@chat_agent.tool_plain
async def add_memory(text: str) -> dict:
    """Remember a fact in long-term memory. Smart-merges with existing facts:
    if `text` restates or updates an existing line, that line is replaced;
    otherwise `text` is appended. Use natural-language sentences.

    Before saving, the proposed text is run through a capability check: if it
    asks the PR-review agent to do something it can't observe or act on
    (e.g. "reject PRs with merge conflicts" — mergeable state isn't fetched),
    the save is refused and the rejection reason is returned. Relay the
    reason verbatim to the user so they can rephrase or drop the request.

    Examples:
        add_memory("Prefer concise PR reviews.")
        add_memory("Reject PRs that add new dependencies.")

    Returns either {"memory": <updated doc>} on success or
    {"refused": true, "reason": <why>} when the capability check fails.
    """
    existing = await _memory_get_doc()
    update = await _memory_check_and_merge(existing, text)
    if not update.can_honor:
        log.info("memory_refused text=%s reason=%s", text[:80], update.reason)
        return {"refused": True, "reason": update.reason}
    await _memory_put_doc(update.merged)
    return {"memory": update.merged}


@chat_agent.tool_plain
async def reset_memory() -> dict:
    """Wipe the entire memory doc. Cannot be undone. Only use when the user
    explicitly asks to forget everything."""
    await _memory_put_doc("")
    return {"memory": ""}


@chat_agent.tool_plain
async def run_pr_review(pr_url: str) -> str:
    """Run full triage + pattern review on a PR and return the rendered card
    (plus drill-down). Use this for any PR triage / merge-readiness request.

    Args:
        pr_url: A GitHub PR URL or short ref (owner/repo#N).
    """
    # Mirror review_pr's observability so dev-chat-driven runs also emit the
    # silent-failure canary. Same attribute names so a single Logfire query
    # covers both paths.
    with logfire.span("run_pr_review", pr_url=pr_url) as span:
        ctx = await _memory_context()
        triage_prompt = f"{ctx}Triage this PR: {pr_url}"
        pattern_prompt = f"{ctx}Review this PR for pattern conformance: {pr_url}"
        (triage, _, triage_err), (pattern, _, pattern_err) = await asyncio.gather(
            _run_triage(triage_prompt),
            _run_pattern(pattern_prompt),
        )
        if triage is None or pattern is None:
            err = triage_err or pattern_err or "unknown"
            span.set_attribute("review_pr.outcome", "fallback")
            span.set_attribute("review_pr.error", str(err)[:300])
            return render_fallback_card(pr_url, err)
        card = fuse(triage, pattern)
        pr_related_n = len(triage.pr_related_failures)
        unrelated_n = len(triage.unrelated_failures)
        failing_total = pr_related_n + unrelated_n
        # Conflict-aware silent failure: READY card with either red CI OR
        # confirmed merge conflicts both count as silent passes. Conflict-only
        # READY cards shouldn't be possible after the rubric change, but keep
        # the canary anyway in case a future edit decouples the two.
        silent_failure = card.verdict == "READY" and (
            failing_total > 0 or triage.has_merge_conflicts is True
        )
        span.set_attribute("review_pr.outcome", "card")
        span.set_attribute("review_pr.score", card.score)
        span.set_attribute("review_pr.verdict", card.verdict)
        span.set_attribute("review_pr.failing_total", failing_total)
        span.set_attribute("review_pr.pr_related_failures", pr_related_n)
        span.set_attribute("review_pr.unrelated_failures", unrelated_n)
        span.set_attribute(
            "review_pr.has_merge_conflicts",
            (
                ""
                if triage.has_merge_conflicts is None
                else str(triage.has_merge_conflicts)
            ),
        )
        span.set_attribute("review_pr.silent_failure", silent_failure)
        if silent_failure:
            log.error(
                "silent_failure_detected url=%s verdict=%s failing=%s conflicts=%s names=%s",
                pr_url,
                card.verdict,
                failing_total,
                triage.has_merge_conflicts,
                triage.pr_related_failures + triage.unrelated_failures,
            )
        return render_card(card) + "\n\n---\n\n" + render_drilldown(triage, pattern)


# --- Memory-merger capability description (auto-derived) ----------------------
# The merger LLM gates add_memory writes by checking whether the proposed fact
# is honorable using the agent's actual capabilities. Hand-maintaining that
# list got it wrong inside two turns of editing (the previous version claimed
# merge conflicts weren't fetched, even though has_merge_conflicts has been a
# TriageReport field with a rubric blocker for a while). So now we derive the
# CAN side from the schemas + tools and only the negative-space CANNOT list
# stays hand-maintained.
#
# The walk needs chat_agent.toolset to be populated, which means it has to run
# AFTER all @chat_agent.tool_plain decorators above. That's why this block
# lives below run_pr_review and not next to the memory I/O helpers.

# Per-field one-liners shown to the merger. The keys MUST match field names in
# the schemas walked below; the asserts in _describe_capabilities will fail
# loudly at module load if a field is added without an entry, so adding a new
# observable forces you to think about how to describe it to the gate.
_FIELD_NOTES: dict[str, str] = {
    # TriageReport
    "pr_number": "PR number",
    "pr_title": "PR title (raw GitHub title)",
    "pr_author": "PR author's GitHub login",
    "pr_summary": "one-paragraph (≤600 chars) summary the agent infers from title+diff",
    "files_changed": "count of files in the diff",
    "additions": "total added lines across the diff",
    "deletions": "total deleted lines across the diff",
    "pr_related_failures": "list of failing check names classified as caused by this PR",
    "unrelated_failures": "list of failing check names classified as infra/unrelated",
    "unrelated_failures_also_failing_elsewhere": "subset of unrelated_failures whose same check is also failing on at least one of the sampled other_prs (broken-for-everyone infra; rubric does NOT dock these)",
    "policy_meta_failures": "policy/meta checks (Verify PR source branch, DCO, cla-bot) failing on this PR; zero-penalty bucket, surfaced for contributor awareness only",
    "running_checks": "list of check names still in progress",
    "greptile_score": "Greptile bot's confidence score 1-5, or null if Greptile hasn't reviewed",
    "has_circleci_checks": "true iff any CircleCI check ran on this PR",
    "has_merge_conflicts": "tri-state: true=confirmed conflicts, false=clean, null=GitHub still computing",
    # PatternReport
    "findings": "pattern-conformance findings vs the repo's docs and sibling files",
    "tech_debt": "ambient tech-debt items spotted (FYI, never blocks)",
    # PatternFinding
    "file": "path of the file the finding is about",
    "severity": "blocker / suggestion / nit (evidence strength)",
    "risk": "high / medium / low (impact if true; orthogonal to severity)",
    "source": "docs (cited from repo docs) or code (cited from sibling files)",
    "citation": "doc path or code path the finding is grounded in",
    "rationale": "≤200-char prose explaining the finding",
    # TechDebtItem
    "doc_path": "path to the doc that documents the pattern",
    "code_path": "path to the file that drifted from it",
    "note": "≤200-char prose on the drift",
    # TriageCard (actions / outputs the agent controls end-to-end)
    "summary": "the headline summary line shown to the user",
    "size_line": "pre-rendered diff-size signal for the card",
    "failing_line": "pre-rendered failing-checks call-out for the card",
    "score": "0-5 confidence score the agent emits",
    "verdict": "READY / BLOCKED / WAITING — the agent's final call",
    "emoji": "card emoji matching the verdict",
    "verdict_one_liner": "one-sentence verdict shown under the score",
    "justification": "longer prose explaining the score breakdown",
}


def _format_type(annotation: object) -> str:
    """Render a Pydantic field annotation as readable prose: 'int', 'list[str]',
    'int | None', 'Literal[\"a\", \"b\"]'. Strips noisy `<class 'X'>` wrappers
    and module prefixes (typing., app.) so the merger LLM gets clean text."""
    t = str(annotation)
    # Bare classes render as "<class 'int'>" via str() — pull the name out.
    if t.startswith("<class '") and t.endswith("'>"):
        t = t[len("<class '") : -len("'>")]
    return t.replace("typing.", "").replace("app.", "")


def _describe_model(name: str, model_cls: type[BaseModel]) -> list[str]:
    """Render one BaseModel as a bulleted block: 'field: type — note' lines.

    Asserts every field has a _FIELD_NOTES entry; missing notes fail loudly at
    module load so you can't ship a new observable without describing it to
    the merger gate.
    """
    doc = (model_cls.__doc__ or "no docstring").strip().splitlines()[0]
    lines: list[str] = [f"{name} ({doc}):"]
    for field_name, info in model_cls.model_fields.items():
        if field_name not in _FIELD_NOTES:
            raise RuntimeError(
                f"_FIELD_NOTES missing entry for {model_cls.__name__}.{field_name}; "
                f"add a one-line description so the memory-gate LLM knows what it does"
            )
        type_str = _format_type(info.annotation)
        lines.append(f"  - {field_name}: {type_str} — {_FIELD_NOTES[field_name]}")
    return lines


def _describe_chat_tools() -> list[str]:
    """Render chat_agent's tools (excluding memory-management ones, which are
    self-referential here) as 'tool(args): docstring-first-line' bullets."""
    # ._function_toolset is pydantic-ai's own attribute name for the toolset
    # built up by @agent.tool_plain decorators. Public-ish: stable across the
    # versions we use, and a missing attr would throw AttributeError loudly.
    toolset = getattr(chat_agent, "_function_toolset", None)
    if toolset is None or not hasattr(toolset, "tools"):
        raise RuntimeError(
            "chat_agent._function_toolset.tools missing — pydantic-ai API "
            "changed; update _describe_chat_tools to match"
        )
    skip = {"add_memory", "reset_memory"}  # describing the gate to itself is noise
    lines: list[str] = []
    for name, tool in toolset.tools.items():
        if name in skip:
            continue
        first_line = (
            (tool.description or "").strip().splitlines()[0]
            if tool.description
            else "(no docstring)"
        )
        lines.append(f"  - {name}: {first_line}")
    return lines


# Hand-maintained negative space: things the agent CAN'T do, which can't be
# auto-derived (you can't enumerate the absence of a field). Edit this when
# the agent gains or loses a capability that doesn't show up as a schema field
# — e.g. "we now post comments to GitHub" would mean dropping the last bullet.
_AGENT_LIMITATIONS = """\
The agent CANNOT (non-exhaustive list of common asks that fail this check):
- look at PR labels, milestones, projects, assignees, or human reviewers
- look at human review comments, approvals, or change-requests (only the
  Greptile bot's score is parsed; other reviewers are invisible)
- read full file contents -- only truncated diff patches (≤2000 chars/file)
- read CI logs beyond the failure_excerpt already aggregated into the
  pr_related vs unrelated bucket classification
- see branch names, base branch, draft status, PR age, or commit history
- identify whether the author is an external contributor vs maintainer
- run code, tests, linters, or any tool against the PR
- post comments on GitHub, request changes, merge, or close PRs
- look at any repo other than the one the PR is in
"""


def _describe_capabilities() -> str:
    """Build the full CAN+CANNOT capability prose for the merger system prompt.

    Pure function called once at module load. Output shape is stable so the
    merger prompt and the smoke test in tests/test_capabilities.py agree.
    """
    blocks: list[str] = [
        "The PR-review agent has exactly the following observations and actions",
        "available. It cannot do anything else.",
        "",
        "Observable per PR (from the gather_pr_data + gather_pattern_data tools):",
    ]
    blocks += _describe_model("TriageReport", TriageReport)
    blocks += _describe_model("PatternReport", PatternReport)
    blocks += _describe_model(
        "PatternFinding (one entry of PatternReport.findings)", PatternFinding
    )
    blocks += _describe_model(
        "TechDebtItem (one entry of PatternReport.tech_debt)", TechDebtItem
    )
    blocks += [
        "",
        "Final card slots the agent controls (score, verdict, prose):",
    ]
    blocks += _describe_model("TriageCard", TriageCard)
    blocks += [
        "",
        "Tools available to the chat agent (in addition to the schema fields above):",
    ]
    blocks += _describe_chat_tools()
    blocks += [
        "",
        "Actions the agent can take:",
        "- score the PR 0-5 and emit verdict READY / BLOCKED / WAITING",
        "- name specific failing checks, files, or pattern findings in its summary",
        "- adjust how it weights any of the signals above (be stricter, be looser,",
        "  add or remove rubric penalties)",
        "",
        _AGENT_LIMITATIONS.rstrip(),
    ]
    return "\n".join(blocks)


_AGENT_CAPABILITIES = _describe_capabilities()


_MERGER_SYSTEM_PROMPT = (
    "You maintain a long-term memory document for a PR-review agent.\n\n"
    "On each call you receive (a) the existing memory doc and (b) a proposed "
    "new fact from the user. You do TWO things:\n\n"
    "STEP 1 — capability check. Decide whether the new fact is something the "
    "PR-review agent can actually honor given its capabilities below.\n\n"
    f"{_AGENT_CAPABILITIES}\n\n"
    "Bias: if the instruction is vague but plausibly satisfiable with the "
    "listed signals (e.g. 'prefer concise reviews', 'be stricter on tests'), "
    "set can_honor=true. Only reject when the instruction clearly needs a "
    "signal or action not in the lists above.\n\n"
    "STEP 2 — merge (only if can_honor=true). Produce the updated memory:\n"
    "  - If the new fact restates, contradicts, or updates an existing line, "
    "REPLACE that line in-place (preserve order).\n"
    "  - Otherwise APPEND the new fact as a new line at the end.\n"
    "  - Do NOT rewrite, reorder, summarize, or 'improve' unrelated lines.\n"
    "  - Do NOT add commentary, headings, bullets, or meta-text.\n"
    "  - If the existing doc is empty, the merged doc is just the new fact.\n\n"
    "Output fields:\n"
    "  - can_honor: bool\n"
    "  - reason: ≤200 chars. When false, name the specific missing capability. "
    "When true, briefly cite which signal(s) the agent would use.\n"
    "  - merged: the full updated doc as plain text. Required when "
    "can_honor=true; ignored (use empty string) when false."
)


_merger_agent = Agent(
    model, system_prompt=_MERGER_SYSTEM_PROMPT, output_type=_MemoryUpdate
)


app = FastAPI()
logfire.instrument_fastapi(app, capture_headers=True)

# --- Login gate for the dev chat UI -------------------------------------------
# The Slack handler (/slack/events) and /healthz stay public — Slack needs to
# POST without credentials and uptime probes don't carry cookies. Only the
# /chat UI and its /chat/api/* JSON endpoints require login.
#
# Auth is opt-in: if ADMIN_USERNAME/ADMIN_PASSWORD are unset (e.g. local dev)
# the gate is disabled and the chat UI is wide open, same as before. Setting
# both turns the gate on.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
AUTH_ENABLED = bool(ADMIN_USERNAME and ADMIN_PASSWORD)

# SESSION_SECRET signs the session cookie. If unset we generate one per
# process — fine for single-instance dev, but it means every restart logs
# everyone out. Set it explicitly in prod to keep sessions across restarts.
SESSION_SECRET = os.environ.get("SESSION_SECRET")
if AUTH_ENABLED and not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(32)
    log.warning(
        "SESSION_SECRET unset; generated ephemeral one. Sessions will not survive restarts."
    )
if not AUTH_ENABLED:
    log.warning("ADMIN_USERNAME/ADMIN_PASSWORD unset; /chat is unauthenticated")

if AUTH_ENABLED:
    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET,
        session_cookie="litellm_bot_session",
        same_site="lax",
        https_only=False,  # set True behind HTTPS-terminating proxy in prod
    )


def require_login(request: Request) -> None:
    """FastAPI dependency: 401 if no session, 302 to /login for HTML requests.

    No-op when AUTH_ENABLED is False so local dev without env vars Just Works.
    """
    if not AUTH_ENABLED:
        return
    if request.session.get("user") == ADMIN_USERNAME:
        return
    # Browser navigations (Accept: text/html) get redirected so the user lands
    # on the login form. XHR/fetch from the chat UI gets a JSON 401 instead so
    # the frontend can surface it cleanly.
    accept = request.headers.get("accept", "")
    if "text/html" in accept and request.method == "GET":
        raise HTTPException(
            status_code=303,
            detail="login required",
            headers={"Location": "/login"},
        )
    raise HTTPException(status_code=401, detail="login required")


def _extract_tool_trace(messages) -> list[dict]:
    """Pull tool-call/return pairs out of an agent run's message history."""
    trace: list[dict] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                trace.append(
                    {"kind": "call", "tool": part.tool_name, "args": part.args}
                )
            elif isinstance(part, ToolReturnPart):
                preview = str(part.content)
                if len(preview) > 500:
                    preview = preview[:500] + "…"
                trace.append(
                    {"kind": "return", "tool": part.tool_name, "preview": preview}
                )
    return trace


# Tools whose return value IS the final user-facing reply. When the chat agent
# calls one of these, we discard the model's free-form post-tool prose and
# pass the tool's verbatim output through. Keeps deterministic-card contracts
# (render_card layout, etc.) intact even though chat_agent is a string-output
# agent that would otherwise paraphrase.
_PASSTHROUGH_TOOLS = {"run_pr_review"}


def _last_passthrough_return(messages) -> str | None:
    """Return verbatim string from the most recent _PASSTHROUGH_TOOLS call in
    `messages`, or None if none called this turn."""
    last: str | None = None
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name in _PASSTHROUGH_TOOLS
            ):
                last = str(part.content)
    return last


async def review_pr(pr_url: str, channel: str, thread_ts: str) -> None:
    """Run both agents in parallel, fuse their typed outputs into one card,
    and post it (plus a threaded drill-down) to Slack.

    Card shape is deterministic — the model fills schema slots, Python composes
    the prose. See render_card() / fuse() for the contract.
    """
    with logfire.span(
        "review_pr",
        pr_url=pr_url,
        channel=channel,
        thread_ts=thread_ts,
    ) as span:
        if not slack_handler.is_enabled():
            log.error("review_pr called without Slack configured url=%s", pr_url)
            return

        ctx = await _memory_context()
        triage_prompt = f"{ctx}Triage this PR: {pr_url}"
        pattern_prompt = f"{ctx}Review this PR for pattern conformance: {pr_url}"
        (triage, _, triage_err), (pattern, _, pattern_err) = await asyncio.gather(
            _run_triage(triage_prompt),
            _run_pattern(pattern_prompt),
        )

        # Either agent failing means we can't build a real card. Show the
        # fallback (same shape, marked as ⚠️ ERROR) so the user still sees the
        # familiar layout instead of a raw exception trace.
        if triage is None or pattern is None:
            err = triage_err or pattern_err or "unknown agent failure"
            span.set_attribute("review_pr.outcome", "fallback")
            span.set_attribute("review_pr.error", str(err)[:300])
            await slack_handler.bolt.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=render_fallback_card(pr_url, err),
            )
            log.warning("posted_fallback_card url=%s err=%s", pr_url, err)
            return

        card = fuse(triage, pattern)
        card_text = render_card(card)
        drilldown = render_drilldown(triage, pattern)

        # Observability: emit failure counts + verdict on the span so we can
        # query "PRs where the bot returned READY but failures > 0" in Logfire.
        # That's the canary for the silent-pass bug PR #26451 surfaced — even
        # with the rubric/render fixes, this dashboard catches future drift
        # (e.g. someone adds another verdict path that swallows failures).
        pr_related_n = len(triage.pr_related_failures)
        unrelated_n = len(triage.unrelated_failures)
        failing_total = pr_related_n + unrelated_n
        span.set_attribute("review_pr.outcome", "card")
        span.set_attribute("review_pr.score", card.score)
        span.set_attribute("review_pr.verdict", card.verdict)
        span.set_attribute("review_pr.failing_total", failing_total)
        span.set_attribute("review_pr.pr_related_failures", pr_related_n)
        span.set_attribute("review_pr.unrelated_failures", unrelated_n)
        span.set_attribute("review_pr.running_checks", len(triage.running_checks))
        span.set_attribute(
            "review_pr.greptile_score",
            triage.greptile_score if triage.greptile_score is not None else -1,
        )
        span.set_attribute(
            "review_pr.has_merge_conflicts",
            (
                ""
                if triage.has_merge_conflicts is None
                else str(triage.has_merge_conflicts)
            ),
        )
        # The canary attribute itself: explicit boolean so the Logfire query is
        # `attributes.review_pr.silent_failure = true` instead of computing it.
        # Now also fires when a READY card slips through with merge conflicts —
        # belt-and-suspenders for the conflict-blocker rubric row.
        silent_failure = card.verdict == "READY" and (
            failing_total > 0 or triage.has_merge_conflicts is True
        )
        span.set_attribute("review_pr.silent_failure", silent_failure)
        if silent_failure:
            log.error(
                "silent_failure_detected url=%s verdict=%s failing=%s conflicts=%s names=%s",
                pr_url,
                card.verdict,
                failing_total,
                triage.has_merge_conflicts,
                triage.pr_related_failures + triage.unrelated_failures,
            )

        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=card_text
        )
        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=drilldown
        )
        log.info(
            "posted_review url=%s score=%s verdict=%s failing=%s",
            pr_url,
            card.score,
            card.verdict,
            failing_total,
        )


slack_handler.mount(app, on_pr_review=review_pr)


# --- Local dev chat UI ---------------------------------------------------------
# Single-page chat for sanity-checking the agent without going through Slack.
# Hits the same agent.run(...) the Slack handler uses.


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    # Client-suggested title, used iff this is the first turn of a new thread.
    # Lets the sidebar render the user's input as the thread title immediately
    # on submit instead of waiting for the agent run to finish (the previous
    # design set the title server-side post-response, which left the sidebar
    # showing "New chat" for 30-60s while the gather script ran).
    title: str | None = None


class ChatResponse(BaseModel):
    output: str
    tool_trace: list[dict]
    thread_id: str


class ThreadSummary(BaseModel):
    id: str
    title: str
    updated_at: float


class ThreadDetail(BaseModel):
    id: str
    title: str
    updated_at: float
    # Flat transcript the UI replays when you switch threads. Each turn is
    # {role: "user"|"assistant", content: str, tool_trace?: list[dict]}.
    turns: list[dict]


# In-memory chat threads for the dev UI. Each thread keeps:
#   - chat_agent ModelMessage history (so tool returns stay paired with their
#     tool definitions on subsequent turns)
#   - a flat UI transcript so the frontend can rehydrate when the user
#     switches between threads
#   - a title (first user message, truncated) and updated_at for the sidebar
THREADS: dict[str, dict] = {}


def _new_thread(thread_id: str | None = None, title: str | None = None) -> dict:
    return {
        "id": thread_id or uuid.uuid4().hex,
        "title": title or "New chat",
        "updated_at": time.time(),
        "history": [],
        "turns": [],
    }


def _summarize(thread: dict) -> ThreadSummary:
    return ThreadSummary(
        id=thread["id"], title=thread["title"], updated_at=thread["updated_at"]
    )


@app.get(
    "/chat/api/threads",
    response_model=list[ThreadSummary],
    dependencies=[Depends(require_login)],
)
async def list_threads() -> list[ThreadSummary]:
    return [
        _summarize(t)
        for t in sorted(THREADS.values(), key=lambda t: t["updated_at"], reverse=True)
    ]


@app.post(
    "/chat/api/threads",
    response_model=ThreadSummary,
    dependencies=[Depends(require_login)],
)
async def create_thread() -> ThreadSummary:
    thread = _new_thread()
    THREADS[thread["id"]] = thread
    return _summarize(thread)


@app.get(
    "/chat/api/threads/{thread_id}",
    response_model=ThreadDetail,
    dependencies=[Depends(require_login)],
)
async def get_thread(thread_id: str) -> ThreadDetail:
    thread = THREADS.get(thread_id)
    if not thread:
        raise HTTPException(404, "thread not found")
    return ThreadDetail(
        id=thread["id"],
        title=thread["title"],
        updated_at=thread["updated_at"],
        turns=thread["turns"],
    )


@app.delete(
    "/chat/api/threads/{thread_id}",
    dependencies=[Depends(require_login)],
)
async def delete_thread(thread_id: str) -> dict:
    if thread_id not in THREADS:
        raise HTTPException(404, "thread not found")
    del THREADS[thread_id]
    return {"ok": True}


@app.post(
    "/chat/api",
    response_model=ChatResponse,
    dependencies=[Depends(require_login)],
)
async def chat_api(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "message is empty")

    # Upsert by client-provided id so the browser can mint the thread locally
    # and have the sidebar row appear instantly. The id is treated as opaque
    # (any string the client picks); collisions across users aren't a concern
    # because THREADS is process-local single-tenant in this dev UI.
    thread = THREADS.get(req.thread_id) if req.thread_id else None
    if thread is None:
        thread = _new_thread(thread_id=req.thread_id, title=req.title)
        THREADS[thread["id"]] = thread
    elif req.title and thread["title"] == "New chat":
        # Pre-existing thread (e.g. spawned by `+ New` before any send) gets
        # its placeholder title upgraded to the user's first message. Only on
        # the placeholder so we never clobber a real title on a follow-up turn.
        thread["title"] = req.title

    history = thread["history"]
    try:
        result = await chat_agent.run(req.message, message_history=history)
        output = result.output
        new_msgs = list(result.all_messages())
    except Exception as e:
        log.exception("chat_agent_failed prompt_head=%s", req.message[:80])
        output = f"⚠️ chat agent failed: {e}"
        new_msgs = history

    new_turn_msgs = new_msgs[len(history) :]
    trace = _extract_tool_trace(new_turn_msgs)
    # Passthrough tools (e.g. run_pr_review) return a fully-rendered card whose
    # layout is contractually fixed by render_card(). chat_agent's final-step
    # inference rewrites that card in its own style (## headers, **bold**,
    # editorial sections), breaking the contract. Override with the tool's
    # verbatim return so the chat UI sees the same bytes Slack does.
    passthrough = _last_passthrough_return(new_turn_msgs)
    if passthrough is not None:
        output = passthrough
    thread["history"] = new_msgs

    thread["turns"].append({"role": "user", "content": req.message})
    thread["turns"].append(
        {"role": "assistant", "content": output, "tool_trace": trace}
    )
    thread["updated_at"] = time.time()
    # Title is seeded at upsert time above (from req.title), so this endpoint
    # stays a pure "append turns" op — no post-response title mutation.

    return ChatResponse(output=output, tool_trace=trace, thread_id=thread["id"])


CHAT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>litellm-bot dev chat</title>
<style>
  *{box-sizing:border-box}
  body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;color:#222;height:100vh;display:flex}
  #sidebar{width:240px;border-right:1px solid #ddd;background:#f5f5f7;display:flex;flex-direction:column;height:100vh}
  #sidebar-head{padding:12px;border-bottom:1px solid #e0e0e3;display:flex;gap:8px;align-items:center}
  #sidebar-head h2{font-size:13px;margin:0;flex:1;color:#333}
  #new-btn{padding:6px 10px;border:0;border-radius:4px;background:#222;color:#fff;cursor:pointer;font:inherit;font-size:12px}
  #threads{flex:1;overflow-y:auto;padding:6px}
  .thread{padding:8px 10px;border-radius:5px;cursor:pointer;display:flex;align-items:center;gap:6px;margin-bottom:2px;color:#444}
  .thread:hover{background:#e8e8ec}
  .thread.active{background:#222;color:#fff}
  .thread.active .del{color:#bbb}
  .thread .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
  .thread .del{border:0;background:transparent;color:#999;cursor:pointer;font-size:14px;padding:0 4px;visibility:hidden}
  .thread:hover .del{visibility:visible}
  .thread.active .del:hover{color:#fff}
  #main{flex:1;display:flex;flex-direction:column;height:100vh;max-width:900px;margin:0 auto;padding:24px;overflow:hidden}
  h1{font-size:18px;margin:0 0 4px}
  .sub{color:#666;font-size:12px;margin-bottom:16px}
  #log{flex:1;border:1px solid #ddd;border-radius:6px;padding:12px;overflow-y:auto;background:#fafafa}
  .msg{margin:0 0 14px;white-space:pre-wrap;word-wrap:break-word}
  .user{color:#0a4}
  .bot{color:#222}
  .tool{color:#888;font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#eef;border-left:3px solid #88a;padding:6px 8px;margin:4px 0;border-radius:3px}
  .err{color:#c00}
  form{display:flex;gap:8px;margin-top:12px}
  input[type=text]{flex:1;padding:10px;border:1px solid #ccc;border-radius:6px;font:inherit}
  button{padding:10px 16px;border:0;border-radius:6px;background:#222;color:#fff;cursor:pointer;font:inherit}
  button:disabled{opacity:.5;cursor:wait}
  .hint{color:#888;font-size:12px;margin-top:8px}
  .empty{color:#999;font-size:12px;text-align:center;padding:16px}
  #sidebar-foot{padding:10px 12px;border-top:1px solid #e0e0e3}
  #logout-form{margin:0}
  #logout-btn{width:100%;padding:7px 10px;border:1px solid #ccc;border-radius:4px;
              background:#fff;color:#444;cursor:pointer;font:inherit;font-size:12px}
  #logout-btn:hover{background:#eee}
</style></head>
<body>
<aside id="sidebar">
  <div id="sidebar-head">
    <h2>Chats</h2>
    <button id="new-btn" type="button">+ New</button>
  </div>
  <div id="threads"></div>
  <div id="sidebar-foot">
    <form id="logout-form" method="post" action="/logout">
      <button id="logout-btn" type="submit">Sign out</button>
    </form>
  </div>
</aside>
<main id="main">
  <h1>litellm-bot dev chat</h1>
  <div class="sub">Sends to <code>POST /chat/api</code> → same agent the Slack handler uses.</div>
  <div id="log"></div>
  <form id="f">
    <input id="m" type="text" placeholder="Triage this PR: https://github.com/BerriAI/litellm/pull/123" autofocus>
    <button id="b" type="submit">Send</button>
  </form>
  <div class="hint">Tool calls show inline. First request can take 30–60s while the gather script hits GitHub. You can click + New or switch threads while a request is in flight — the response will land in its origin thread.</div>
</main>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('m');
const btn = document.getElementById('b');
const threadsEl = document.getElementById('threads');
const newBtn = document.getElementById('new-btn');

let threadId = localStorage.getItem('threadId');
// Per-thread in-flight state. If pending.has(id), there is a request out for
// that thread and we should show the user message + a thinking placeholder
// when that thread is the active view. Cleared when the response (or error)
// lands. This is what lets the user spawn / switch to other threads without
// the in-flight request blocking the UI.
const pending = new Map(); // threadId -> {userMessage: string}

// Client-minted threads that the server hasn't acknowledged yet. The submit
// handler registers an entry here *before* sending so the sidebar can render
// the row immediately (with the user's input as the title). On the next
// refreshThreads() the server-returned thread for this id replaces the
// optimistic stub. Cleared when the chat response for the thread lands.
const optimistic = new Map(); // threadId -> {title, updated_at}

function mintThreadId(){
  // crypto.randomUUID is available in all evergreen browsers; the dashes are
  // stripped to keep ids visually consistent with the server's uuid4().hex.
  return crypto.randomUUID().replace(/-/g, '');
}

function titleFromMessage(msg){
  // Mirror the server-side truncation in chat_api so the sidebar title
  // doesn't shift when the response lands and the server reconciles.
  return msg.length > 60 ? msg.slice(0, 60) + '…' : msg;
}

async function authFetch(url, opts){
  const r = await fetch(url, opts);
  if (r.status === 401) {
    window.location.href = '/login';
    throw new Error('redirecting to login');
  }
  return r;
}

function add(cls, text){
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function addTrace(trace){
  for (const t of trace) {
    if (t.kind === 'call') {
      add('tool', '→ ' + t.tool + '(' + JSON.stringify(t.args) + ')');
    } else if (t.kind === 'return') {
      add('tool', '← ' + t.tool + ' returned: ' + t.preview);
    } else if (t.kind === 'label') {
      add('tool', '— ' + t.tool + ' —');
    }
  }
}

// Disable Send only for the active thread when its own request is in flight.
// Other threads (and brand-new ones) stay interactive.
function syncSendButton(){
  btn.disabled = !!(threadId && pending.has(threadId));
}

function setActiveThread(id){
  threadId = id;
  if (id) localStorage.setItem('threadId', id);
  else localStorage.removeItem('threadId');
  for (const el of threadsEl.querySelectorAll('.thread')) {
    el.classList.toggle('active', el.dataset.id === id);
  }
  syncSendButton();
}

// Cached last server-known thread list. renderThreads() uses this as its
// baseline so a synchronous repaint (e.g. after registering an optimistic
// thread) doesn't have to wait on a fetch.
let lastServerThreads = [];

function renderThreads(threads){
  if (threads !== undefined) lastServerThreads = threads;
  // Merge the cached server list with any client-minted threads the server
  // hasn't seen yet. Server entries win on duplicate id (their title is
  // canonical once they exist), so when the chat response lands and
  // refreshThreads() re-runs the optimistic stub gets replaced cleanly.
  const merged = new Map();
  for (const [id, stub] of optimistic) {
    merged.set(id, {id, title: stub.title, updated_at: stub.updated_at});
  }
  for (const t of lastServerThreads) {
    merged.set(t.id, t);
  }
  const rows = [...merged.values()].sort((a, b) => b.updated_at - a.updated_at);

  threadsEl.innerHTML = '';
  if (!rows.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'No chats yet. Click + New to start.';
    threadsEl.appendChild(e);
    return;
  }
  for (const t of rows) {
    const row = document.createElement('div');
    row.className = 'thread' + (t.id === threadId ? ' active' : '');
    row.dataset.id = t.id;
    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = t.title + (pending.has(t.id) ? ' …' : '');
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '×';
    del.title = 'Delete chat';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Delete this chat?')) return;
      // 404 is fine here — the thread may be optimistic-only (server never
      // saw it because the user deleted before submit completed).
      await authFetch('/chat/api/threads/' + t.id, {method: 'DELETE'}).catch(() => {});
      optimistic.delete(t.id);
      pending.delete(t.id);
      if (t.id === threadId) {
        log.innerHTML = '';
        setActiveThread(null);
      }
      await refreshThreads();
    });
    row.appendChild(title);
    row.appendChild(del);
    row.addEventListener('click', () => loadThread(t.id));
    threadsEl.appendChild(row);
  }
}

async function refreshThreads(){
  const r = await authFetch('/chat/api/threads');
  const threads = await r.json();
  renderThreads(threads);
}

async function loadThread(id){
  setActiveThread(id);
  log.innerHTML = '';
  const r = await authFetch('/chat/api/threads/' + id);
  if (!r.ok) {
    // 404 is expected for an optimistic thread whose first chat request is
    // still in flight — the server only creates the row when /chat/api hits
    // its upsert. Keep the thread active and replay the optimistic state
    // below; the post-response refresh will surface the real turns.
    if (r.status !== 404 || !optimistic.has(id)) {
      setActiveThread(null);
      await refreshThreads();
      return;
    }
  } else {
    const data = await r.json();
    for (const turn of data.turns) {
      if (turn.role === 'user') {
        add('user', '> ' + turn.content);
      } else {
        if (turn.tool_trace) addTrace(turn.tool_trace);
        add('bot', turn.content);
      }
    }
  }
  // If this thread has a request in flight, replay the optimistic user
  // message + thinking placeholder so switching back mid-request looks right.
  const p = pending.get(id);
  if (p) {
    add('user', '> ' + p.userMessage);
    add('bot', 'thinking...');
  }
  refreshThreads();
}

async function createThread(){
  const r = await authFetch('/chat/api/threads', {method: 'POST'});
  if (!r.ok) throw new Error('failed to create thread');
  const t = await r.json();
  return t.id;
}

newBtn.addEventListener('click', async () => {
  try {
    const id = await createThread();
    log.innerHTML = '';
    setActiveThread(id);
    await refreshThreads();
    input.focus();
  } catch (err) {
    add('err', String(err));
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;
  // Lock in the originating thread *now*. The user may switch away while the
  // request is pending; the response must land in (and only in) this thread.
  // If there's no active thread we mint one client-side (id + title) so the
  // sidebar row appears immediately, with no round-trip to POST /chat/api/threads.
  // The server upserts the thread on first sight inside /chat/api.
  let originId = threadId;
  const sendTitle = titleFromMessage(msg);
  let isNewOrigin = false;
  if (!originId) {
    originId = mintThreadId();
    optimistic.set(originId, {title: sendTitle, updated_at: Date.now() / 1000});
    setActiveThread(originId);
    isNewOrigin = true;
  }

  pending.set(originId, {userMessage: msg});
  input.value = '';
  // Render optimistically only if we're still looking at the origin thread.
  if (originId === threadId) {
    if (isNewOrigin) log.innerHTML = '';
    add('user', '> ' + msg);
    add('bot', 'thinking...');
  }
  syncSendButton();
  // Synchronous repaint using the cached server list overlaid with the
  // optimistic stub — the new row appears instantly. The background
  // refreshThreads() then reconciles with the actual server state once the
  // chat response (or any other change) has had a chance to land.
  renderThreads();
  refreshThreads();

  try {
    const r = await authFetch('/chat/api', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({message: msg, thread_id: originId, title: sendTitle}),
    });
    pending.delete(originId);
    // Server has now upserted the thread under originId, so the optimistic
    // stub is redundant. Drop it on both success and failure paths so the
    // sidebar shows a single canonical row.
    optimistic.delete(originId);
    if (!r.ok) {
      const t = await r.text();
      if (originId === threadId) {
        // Drop the placeholder before showing the error.
        if (log.lastChild) log.lastChild.remove();
        add('err', 'HTTP ' + r.status + ': ' + t);
      }
      syncSendButton();
      refreshThreads();
      return;
    }
    const data = await r.json();
    if (originId === threadId) {
      // Drop the optimistic placeholder, then render the real response.
      if (log.lastChild) log.lastChild.remove();
      addTrace(data.tool_trace);
      add('bot', data.output);
    }
    syncSendButton();
    refreshThreads();
  } catch (err) {
    pending.delete(originId);
    // Network/abort error: server may or may not have created the thread.
    // Drop the optimistic stub either way — refreshThreads() below will
    // re-introduce the row from the server list if it actually exists.
    optimistic.delete(originId);
    if (originId === threadId) {
      if (log.lastChild) log.lastChild.remove();
      add('err', String(err));
    }
    syncSendButton();
    refreshThreads();
  } finally {
    if (originId === threadId) input.focus();
  }
});

(async () => {
  await refreshThreads();
  if (threadId) {
    const r = await authFetch('/chat/api/threads/' + threadId);
    if (r.ok) loadThread(threadId);
    else setActiveThread(null);
  }
})();
</script>
</body></html>"""


@app.get(
    "/chat",
    response_class=HTMLResponse,
    dependencies=[Depends(require_login)],
)
async def chat_ui() -> str:
    return CHAT_HTML


# --- Login endpoints -----------------------------------------------------------
# Kept tiny on purpose: single user, single password, single env var pair.
# The HTML matches the chat UI's neutral aesthetic so it doesn't feel bolted on.

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Sign in — litellm-bot</title>
<style>
  *{box-sizing:border-box}
  body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;color:#222;
       min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f5f5f7}
  .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:28px 32px;
        width:320px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  h1{font-size:16px;margin:0 0 4px}
  .sub{color:#666;font-size:12px;margin-bottom:18px}
  label{display:block;font-size:12px;color:#444;margin:10px 0 4px}
  input{width:100%;padding:9px 10px;border:1px solid #ccc;border-radius:5px;font:inherit}
  button{margin-top:16px;width:100%;padding:10px;border:0;border-radius:5px;
         background:#222;color:#fff;cursor:pointer;font:inherit}
  .err{color:#c00;font-size:12px;margin-top:10px;min-height:1em}
</style></head>
<body>
<form class="card" method="post" action="/login">
  <h1>litellm-bot dev chat</h1>
  <div class="sub">Sign in to continue.</div>
  <label for="u">Username</label>
  <input id="u" name="username" autocomplete="username" autofocus required>
  <label for="p">Password</label>
  <input id="p" name="password" type="password" autocomplete="current-password" required>
  <button type="submit">Sign in</button>
  <div class="err">__ERROR__</div>
</form>
</body></html>"""


def _render_login(error: str = "") -> str:
    return LOGIN_HTML.replace("__ERROR__", error)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> str:
    if AUTH_ENABLED and request.session.get("user") == ADMIN_USERNAME:
        # Already signed in — bounce back to /chat instead of showing the form.
        raise HTTPException(303, headers={"Location": "/chat"})
    return _render_login()


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if not AUTH_ENABLED:
        # Auth turned off entirely — nothing to validate, just send them in.
        return RedirectResponse("/chat", status_code=303)
    # compare_digest avoids a timing oracle on the username/password compare.
    ok = secrets.compare_digest(
        username, ADMIN_USERNAME or ""
    ) and secrets.compare_digest(password, ADMIN_PASSWORD or "")
    if not ok:
        return HTMLResponse(
            _render_login("Invalid username or password."), status_code=401
        )
    request.session["user"] = ADMIN_USERNAME
    return RedirectResponse("/chat", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    if AUTH_ENABLED:
        request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
