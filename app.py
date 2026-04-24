import asyncio
import hashlib
import hmac
import logging
import os

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from slack_sdk import WebClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("litellm-bot")

REQUIRED = ("GITHUB_WEBHOOK_SECRET", "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID", "ANTHROPIC_API_KEY", "GITHUB_TOKEN")
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    raise RuntimeError(f"missing required env vars: {', '.join(missing)}")

SLACK = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
CHANNEL = os.environ["SLACK_CHANNEL_ID"]
SECRET = os.environ["GITHUB_WEBHOOK_SECRET"].encode()
PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "skills/pr-review-agent-skills/litellm-pr-reviewer")

app = FastAPI()


def verify(body: bytes, sig: str | None) -> bool:
    if not sig:
        return False
    expected = "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


async def review_pr(pr: dict) -> None:
    options = ClaudeAgentOptions(
        plugins=[{"type": "local", "path": PLUGIN_PATH}],
        system_prompt="You are a PR reviewer for the LiteLLM OSS repo.",
        allowed_tools=["Read", "Bash", "Grep", "WebFetch"],
    )
    prompt = f"Use the litellm-pr-reviewer skill to triage {pr['html_url']}"
    chunks: list[str] = []
    try:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
    except Exception:
        log.exception("agent_failed pr=%s", pr["number"])
        return
    review = "\n".join(c for c in chunks if c).strip() or "_(empty review)_"
    SLACK.chat_postMessage(
        channel=CHANNEL,
        text=f"*PR #{pr['number']}:* <{pr['html_url']}|{pr['title']}>\n\n{review}",
    )
    log.info("posted_review pr=%s chars=%s", pr["number"], len(review))


@app.post("/webhooks/github")
async def github(
    request: Request,
    x_hub_signature_256: str | None = Header(None),
    x_github_event: str | None = Header(None),
):
    body = await request.body()
    if not verify(body, x_hub_signature_256):
        raise HTTPException(401, "bad signature")
    if x_github_event != "pull_request":
        return {"ok": True, "skipped": "not_pr"}
    payload = await request.json()
    if payload["action"] not in ("opened", "synchronize", "reopened"):
        return {"ok": True, "skipped": payload["action"]}
    if payload["pull_request"].get("draft"):
        return {"ok": True, "skipped": "draft"}
    asyncio.create_task(review_pr(payload["pull_request"]))
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
