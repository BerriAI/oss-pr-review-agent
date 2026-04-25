import asyncio
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
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

SKILL_DIR = (
    Path(__file__).parent / "skills/pr-review-agent-skills/litellm-pr-reviewer"
)
GATHER_SCRIPT = SKILL_DIR / "scripts/gather_pr_triage_data.py"
PATTERN_SKILL_DIR = (
    Path(__file__).parent
    / "skills-local/litellm-pattern-conformance-reviewer"
)
PATTERN_GATHER_SCRIPT = PATTERN_SKILL_DIR / "scripts/gather_pattern_data.py"

# SKILL.md tells the model to shell out to the gather script. We expose that as a
# typed tool instead so the agent never gets a generic Bash. This prefix overrides
# the "Step 1" bash instruction without touching the skill file.
TOOL_REDIRECT = (
    "TOOL USE: Wherever the instructions below say to run "
    "`python ${CLAUDE_SKILL_DIR}/scripts/gather_pr_triage_data.py <ref>`, "
    "instead call the `gather_pr_data` tool with the PR reference. "
    "It returns the same JSON shape the script would have printed.\n\n"
)
SYSTEM_PROMPT = TOOL_REDIRECT + (SKILL_DIR / "SKILL.md").read_text()

PATTERN_SYSTEM_PROMPT = (PATTERN_SKILL_DIR / "SKILL.md").read_text()

model = OpenAIChatModel(
    os.environ.get("LITELLM_MODEL", "claude-sonnet-4-6"),
    provider=LiteLLMProvider(
        api_base=os.environ.get("LITELLM_API_BASE", "http://0.0.0.0:4000"),
        api_key=os.environ["LITELLM_API_KEY"],
    ),
)
agent = Agent(model, system_prompt=SYSTEM_PROMPT)
pattern_agent = Agent(model, system_prompt=PATTERN_SYSTEM_PROMPT)


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


async def _run_one(
    selected_agent: Agent,
    prompt: str,
    history: list | None = None,
) -> tuple[str, list]:
    """Run a single agent and return (output_text, new_messages_for_history).
    Catches exceptions so a failure in one agent doesn't kill the other in
    asyncio.gather().
    """
    try:
        result = await selected_agent.run(prompt, message_history=history or [])
    except Exception as e:
        log.exception("agent_failed prompt_head=%s", prompt[:80])
        return (f":warning: agent failed: {e}", history or [])
    output = (result.output or "").strip() or "_(empty)_"
    return (output, list(result.all_messages()))


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
    """Run both the CI triage agent and the pattern conformance agent in
    parallel and post a single two-section reply in the thread.
    """
    # Background task spawned via asyncio.create_task; opening a root span here
    # so the two parallel agent runs and the Slack post share one trace.
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
        (triage_text, _), (pattern_text, _) = await asyncio.gather(
            _run_one(agent, triage_prompt),
            _run_one(pattern_agent, pattern_prompt),
        )

        body = (
            f"*Review:* {pr_url}\n\n"
            f"*CI Triage*\n{triage_text}\n\n"
            f"*Pattern Conformance*\n{pattern_text}"
        )
        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=body
        )
        log.info(
            "posted_review url=%s triage_chars=%s pattern_chars=%s",
            pr_url,
            len(triage_text),
            len(pattern_text),
        )


slack_handler.mount(app, on_pr_review=review_pr)


# --- Local dev chat UI ---------------------------------------------------------
# Single-page chat for sanity-checking the agent without going through Slack.
# Hits the same agent.run(...) the Slack handler uses.

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    output: str
    tool_trace: list[dict]
    session_id: str


# In-memory chat sessions for the dev UI: session_id -> {agent_name: history}.
# Each agent gets its own message history so tool returns stay paired with the
# correct agent's tool definitions.
SESSIONS: dict[str, dict[str, list[ModelMessage]]] = {}


@app.post("/chat/api", response_model=ChatResponse)
async def chat_api(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "message is empty")

    session_id = req.session_id or uuid.uuid4().hex
    histories = SESSIONS.get(session_id, {"triage": [], "pattern": []})

    (triage_text, triage_msgs), (pattern_text, pattern_msgs) = await asyncio.gather(
        _run_one(agent, req.message, histories["triage"]),
        _run_one(pattern_agent, req.message, histories["pattern"]),
    )
    SESSIONS[session_id] = {"triage": triage_msgs, "pattern": pattern_msgs}

    combined = (
        f"**CI Triage**\n{triage_text}\n\n"
        f"**Pattern Conformance**\n{pattern_text}"
    )
    trace = (
        [{"kind": "label", "tool": "triage", "preview": ""}]
        + _extract_tool_trace(triage_msgs[len(histories["triage"]):])
        + [{"kind": "label", "tool": "pattern", "preview": ""}]
        + _extract_tool_trace(pattern_msgs[len(histories["pattern"]):])
    )
    return ChatResponse(output=combined, tool_trace=trace, session_id=session_id)


CHAT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>litellm-bot dev chat</title>
<style>
  body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:780px;margin:24px auto;padding:0 16px;color:#222}
  h1{font-size:18px;margin:0 0 4px}
  .sub{color:#666;font-size:12px;margin-bottom:16px}
  #log{border:1px solid #ddd;border-radius:6px;padding:12px;height:60vh;overflow-y:auto;background:#fafafa}
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
</style></head>
<body>
<h1>litellm-bot dev chat</h1>
<div class="sub">Sends to <code>POST /chat/api</code> → same agent the Slack handler uses.</div>
<div id="log"></div>
<form id="f">
  <input id="m" type="text" placeholder="Triage this PR: https://github.com/BerriAI/litellm/pull/123" autofocus>
  <button id="b" type="submit">Send</button>
</form>
<div class="hint">Tool calls show inline. First request can take 30–60s while the gather script hits GitHub.</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('m');
const btn = document.getElementById('b');
let sessionId = null;

function add(cls, text){
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

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
      body: JSON.stringify({message: msg, session_id: sessionId}),
    });
    placeholder.remove();
    if (!r.ok) {
      const t = await r.text();
      add('err', 'HTTP ' + r.status + ': ' + t);
      return;
    }
    const data = await r.json();
    sessionId = data.session_id;
    for (const t of data.tool_trace) {
      if (t.kind === 'call') {
        add('tool', '→ ' + t.tool + '(' + JSON.stringify(t.args) + ')');
      } else {
        add('tool', '← ' + t.tool + ' returned: ' + t.preview);
      }
    }
    add('bot', data.output);
  } catch (err) {
    placeholder.remove();
    add('err', String(err));
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
</script>
</body></html>"""


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui() -> str:
    return CHAT_HTML


@app.get("/healthz")
async def healthz():
    return {"ok": True}
