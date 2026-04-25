import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Literal

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
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
TRIAGE_OUTPUT_OVERRIDE = """
OUTPUT OVERRIDE (supersedes Step 6 above):

Ignore the "write the verdict" instructions in Step 6. Do not emit prose with
overview / summary / details / file_callouts. Instead, return the
TriageReport schema with these fields:

- pr_number, pr_title, pr_author: from the gathered data
- pr_summary: ONE paragraph (max 600 chars), plain prose, no markdown bold or
  italics. Describe what the PR changes (infer from pr_title + diff_files).
- pr_related_failures: list of check names from failing_check_contexts where
  related_to_pr_diff is True (per Step 2 classification rules).
- unrelated_failures: list of check names where related_to_pr_diff is False.
- running_checks: in_progress_checks verbatim.
- greptile_score: the int from the gathered data, or null.
- has_circleci_checks: bool from the gathered data.

Do not include any prose justification, summary, or details — Python composes
those from these structured fields downstream. Your job ends at field-filling.
"""

PATTERN_OUTPUT_OVERRIDE = """
OUTPUT OVERRIDE (supersedes Step 4 above):

Ignore the "emit overview / summary" instructions in Step 4. Do not write
prose. Return the PatternReport schema with these fields:

- findings: list of {file, severity, source, citation, rationale} per Step 3
  classification. Use severity blocker/suggestion/nit exactly as defined.
  rationale max 200 chars, plain prose, no markdown bold or italics.
- tech_debt: list of {doc_path, code_path, note} per the existing rule. note
  max 200 chars.

If there are no findings, return findings: []. If no tech_debt, return [].
Do not include overview or summary — Python composes the user-facing card
from your findings list downstream.
"""


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


def _no_markdown_bold(v: str) -> str:
    """Hard-fail any agent prose that contains markdown bold; the deterministic
    Slack rendering owns formatting end-to-end. Pydantic AI catches the
    ValueError and retries the model with this message as feedback."""
    if "**" in v or "__" in v:
        raise ValueError("text fields must not contain markdown bold (** or __)")
    return v


class TriageReport(BaseModel):
    """Structured CI/policy signals for one PR. Filled by the triage agent."""

    pr_number: int
    pr_title: str
    pr_author: str
    # 1 paragraph: what the PR changes. Drives the *Triage Summary* line.
    pr_summary: str = Field(..., max_length=600)
    # Check names only. Classification logic lives in the SKILL.
    pr_related_failures: list[str] = Field(default_factory=list)
    unrelated_failures: list[str] = Field(default_factory=list)
    running_checks: list[str] = Field(default_factory=list)
    greptile_score: int | None = None
    has_circleci_checks: bool

    _strip_bold = field_validator("pr_summary")(_no_markdown_bold)


class PatternFinding(BaseModel):
    file: str
    severity: Literal["blocker", "suggestion", "nit"]
    source: Literal["docs", "code"]
    citation: str
    rationale: str = Field(..., max_length=200)


class TechDebtItem(BaseModel):
    doc_path: str
    code_path: str
    note: str = Field(..., max_length=200)


class PatternReport(BaseModel):
    """Pattern-conformance findings + ambient tech debt. Filled by pattern agent."""

    findings: list[PatternFinding] = Field(default_factory=list)
    tech_debt: list[TechDebtItem] = Field(default_factory=list)


class TriageCard(BaseModel):
    """Final card. Built by fuse() from TriageReport + PatternReport, never by
    the model directly."""

    summary: str
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


# Rubric: one row per penalty. Each row is (weight, predicate, label_fn).
# label_fn returns the human-readable reason that lands in the justification.
# Edit this table to retune scoring; everything else flows from it.
_RubricRow = tuple[
    int,
    Callable[[TriageReport, PatternReport], bool],
    Callable[[TriageReport, PatternReport], str],
]
_RUBRIC: list[_RubricRow] = [
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
    (
        1,
        lambda t, p: not t.has_circleci_checks,
        lambda t, p: "no CircleCI checks ran on this PR",
    ),
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
    if triage.pr_related_failures:
        return f"{_plural(len(triage.pr_related_failures), 'PR-related CI failure')} need fixes first."
    blocker_n = _count(pattern, "blocker")
    if blocker_n:
        return f"{_plural(blocker_n, 'pattern blocker')} need fixes first."
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
        return (
            f"All checks green. Greptile {triage.greptile_score}/5, "
            f"no blocking pattern findings, CircleCI passed."
        )
    sentences: list[str] = []
    if penalties:
        sentences.append("Score docked for: " + "; ".join(penalties) + ".")
    if triage.unrelated_failures:
        sentences.append(
            f"{_plural(len(triage.unrelated_failures), 'unrelated CI failure')} "
            f"({_join(triage.unrelated_failures)}) — not related to this diff."
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
    return (
        f"*Triage Summary*\n"
        f"{card.summary}\n\n"
        f"*Merge Confidence: {card.score}/5*  {card.emoji} {card.verdict}\n"
        f"{card.verdict_one_liner}\n\n"
        f"{card.justification}"
    )


def render_drilldown(triage: TriageReport, pattern: PatternReport) -> str:
    """Slack mrkdwn for the threaded follow-up: full failure list + findings.

    Posted as a reply so it doesn't compete with the card visually but is one
    click away for anyone who wants the detail.
    """
    lines: list[str] = ["*Drill-down*"]

    if triage.pr_related_failures:
        lines.append("\n_PR-related failures_")
        for c in triage.pr_related_failures:
            lines.append(f"  • {c}")
    if triage.unrelated_failures:
        lines.append("\n_Unrelated failures_")
        for c in triage.unrelated_failures:
            lines.append(f"  • {c}")
    if triage.running_checks:
        lines.append("\n_Still running_")
        for c in triage.running_checks:
            lines.append(f"  • {c}")

    if pattern.findings:
        lines.append("\n_Pattern findings_")
        for f in pattern.findings:
            lines.append(
                f"  • [{f.severity}] `{f.file}` — {f.rationale} "
                f"(source: {f.source}, {f.citation})"
            )
    if pattern.tech_debt:
        lines.append("\n_Tech debt (FYI, not blocking)_")
        for td in pattern.tech_debt:
            lines.append(
                f"  • `{td.code_path}` vs `{td.doc_path}` — {td.note}"
            )

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
agent = Agent(model, system_prompt=SYSTEM_PROMPT, output_type=TriageReport)
pattern_agent = Agent(
    model, system_prompt=PATTERN_SYSTEM_PROMPT, output_type=PatternReport
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
    return (result.output, list(result.all_messages()), None)


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


app = FastAPI()
logfire.instrument_fastapi(app, capture_headers=True)


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
    ):
        if not slack_handler.is_enabled():
            log.error("review_pr called without Slack configured url=%s", pr_url)
            return

        triage_prompt = f"Triage this PR: {pr_url}"
        pattern_prompt = f"Review this PR for pattern conformance: {pr_url}"
        (triage, _, triage_err), (pattern, _, pattern_err) = await asyncio.gather(
            _run_triage(triage_prompt),
            _run_pattern(pattern_prompt),
        )

        # Either agent failing means we can't build a real card. Show the
        # fallback (same shape, marked as ⚠️ ERROR) so the user still sees the
        # familiar layout instead of a raw exception trace.
        if triage is None or pattern is None:
            err = triage_err or pattern_err or "unknown agent failure"
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

        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=card_text
        )
        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=drilldown
        )
        log.info(
            "posted_review url=%s score=%s verdict=%s",
            pr_url,
            card.score,
            card.verdict,
        )


slack_handler.mount(app, on_pr_review=review_pr)


# --- Local dev chat UI ---------------------------------------------------------
# Single-page chat for sanity-checking the agent without going through Slack.
# Hits the same agent.run(...) the Slack handler uses.

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


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
#   - per-agent ModelMessage history (so tool returns stay paired with the
#     right agent's tool definitions on subsequent turns)
#   - a flat UI transcript so the frontend can rehydrate when the user
#     switches between threads
#   - a title (first user message, truncated) and updated_at for the sidebar
THREADS: dict[str, dict] = {}


def _new_thread() -> dict:
    return {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "updated_at": time.time(),
        "histories": {"triage": [], "pattern": []},
        "turns": [],
    }


def _summarize(thread: dict) -> ThreadSummary:
    return ThreadSummary(
        id=thread["id"], title=thread["title"], updated_at=thread["updated_at"]
    )


@app.get("/chat/api/threads", response_model=list[ThreadSummary])
async def list_threads() -> list[ThreadSummary]:
    return [
        _summarize(t)
        for t in sorted(THREADS.values(), key=lambda t: t["updated_at"], reverse=True)
    ]


@app.post("/chat/api/threads", response_model=ThreadSummary)
async def create_thread() -> ThreadSummary:
    thread = _new_thread()
    THREADS[thread["id"]] = thread
    return _summarize(thread)


@app.get("/chat/api/threads/{thread_id}", response_model=ThreadDetail)
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


@app.delete("/chat/api/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    if thread_id not in THREADS:
        raise HTTPException(404, "thread not found")
    del THREADS[thread_id]
    return {"ok": True}


@app.post("/chat/api", response_model=ChatResponse)
async def chat_api(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "message is empty")

    thread = THREADS.get(req.thread_id) if req.thread_id else None
    if thread is None:
        thread = _new_thread()
        THREADS[thread["id"]] = thread

    histories = thread["histories"]
    (triage, triage_msgs, triage_err), (pattern, pattern_msgs, pattern_err) = await asyncio.gather(
        _run_triage(req.message, histories["triage"]),
        _run_pattern(req.message, histories["pattern"]),
    )
    thread["histories"] = {"triage": triage_msgs, "pattern": pattern_msgs}

    # Mirror Slack: render the same card here so the dev UI is a real preview
    # of what users will see, not a debugging dump. Drill-down appended below
    # the card with a separator instead of being a separate message.
    if triage is None or pattern is None:
        combined = render_fallback_card(
            req.message, triage_err or pattern_err or "unknown agent failure"
        )
    else:
        card = fuse(triage, pattern)
        combined = render_card(card) + "\n\n---\n\n" + render_drilldown(triage, pattern)

    trace = (
        [{"kind": "label", "tool": "triage", "preview": ""}]
        + _extract_tool_trace(triage_msgs[len(histories["triage"]):])
        + [{"kind": "label", "tool": "pattern", "preview": ""}]
        + _extract_tool_trace(pattern_msgs[len(histories["pattern"]):])
    )

    thread["turns"].append({"role": "user", "content": req.message})
    thread["turns"].append(
        {"role": "assistant", "content": combined, "tool_trace": trace}
    )
    thread["updated_at"] = time.time()
    # First user turn doubles as the thread title in the sidebar.
    if thread["title"] == "New chat":
        thread["title"] = (req.message[:60] + "…") if len(req.message) > 60 else req.message

    return ChatResponse(output=combined, tool_trace=trace, thread_id=thread["id"])


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
</style></head>
<body>
<aside id="sidebar">
  <div id="sidebar-head">
    <h2>Chats</h2>
    <button id="new-btn" type="button">+ New</button>
  </div>
  <div id="threads"></div>
</aside>
<main id="main">
  <h1>litellm-bot dev chat</h1>
  <div class="sub">Sends to <code>POST /chat/api</code> → same agent the Slack handler uses.</div>
  <div id="log"></div>
  <form id="f">
    <input id="m" type="text" placeholder="Triage this PR: https://github.com/BerriAI/litellm/pull/123" autofocus>
    <button id="b" type="submit">Send</button>
  </form>
  <div class="hint">Tool calls show inline. First request can take 30–60s while the gather script hits GitHub.</div>
</main>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('m');
const btn = document.getElementById('b');
const threadsEl = document.getElementById('threads');
const newBtn = document.getElementById('new-btn');

let threadId = localStorage.getItem('threadId');

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

function setActiveThread(id){
  threadId = id;
  if (id) localStorage.setItem('threadId', id);
  else localStorage.removeItem('threadId');
  for (const el of threadsEl.querySelectorAll('.thread')) {
    el.classList.toggle('active', el.dataset.id === id);
  }
}

async function refreshThreads(){
  const r = await fetch('/chat/api/threads');
  const threads = await r.json();
  threadsEl.innerHTML = '';
  if (!threads.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'No chats yet. Send a message to start.';
    threadsEl.appendChild(e);
    return;
  }
  for (const t of threads) {
    const row = document.createElement('div');
    row.className = 'thread' + (t.id === threadId ? ' active' : '');
    row.dataset.id = t.id;
    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = t.title;
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '×';
    del.title = 'Delete chat';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Delete this chat?')) return;
      await fetch('/chat/api/threads/' + t.id, {method: 'DELETE'});
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

async function loadThread(id){
  setActiveThread(id);
  log.innerHTML = '';
  const r = await fetch('/chat/api/threads/' + id);
  if (!r.ok) {
    setActiveThread(null);
    await refreshThreads();
    return;
  }
  const data = await r.json();
  for (const turn of data.turns) {
    if (turn.role === 'user') {
      add('user', '> ' + turn.content);
    } else {
      if (turn.tool_trace) addTrace(turn.tool_trace);
      add('bot', turn.content);
    }
  }
  refreshThreads();
}

newBtn.addEventListener('click', () => {
  log.innerHTML = '';
  setActiveThread(null);
  refreshThreads();
  input.focus();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;
  add('user', '> ' + msg);
  input.value = '';
  btn.disabled = true;
  add('bot', 'thinking...');
  const placeholder = log.lastChild;
  try {
    const r = await fetch('/chat/api', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({message: msg, thread_id: threadId}),
    });
    placeholder.remove();
    if (!r.ok) {
      const t = await r.text();
      add('err', 'HTTP ' + r.status + ': ' + t);
      return;
    }
    const data = await r.json();
    setActiveThread(data.thread_id);
    addTrace(data.tool_trace);
    add('bot', data.output);
    refreshThreads();
  } catch (err) {
    placeholder.remove();
    add('err', String(err));
  } finally {
    btn.disabled = false;
    input.focus();
  }
});

(async () => {
  await refreshThreads();
  if (threadId) {
    const r = await fetch('/chat/api/threads/' + threadId);
    if (r.ok) loadThread(threadId);
    else setActiveThread(null);
  }
})();
</script>
</body></html>"""


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui() -> str:
    return CHAT_HTML


@app.get("/healthz")
async def healthz():
    return {"ok": True}
