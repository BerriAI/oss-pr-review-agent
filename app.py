import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from httpx import HTTPStatusError
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

import slack_handler

load_dotenv()
log = logging.getLogger(__name__)

SHIN_URL = os.environ.get("SHIN_AGENT_URL", "https://shin-pr-review-agent.onrender.com")
SHIN_API_KEY = os.environ.get("SHIN_AGENT_API_KEY", "key1")


async def review_pr(
    pr_url: str,
    channel: str,
    thread_ts: str,
    message_text: str | None = None,
) -> None:
    if not slack_handler.is_enabled():
        log.error("review_pr called without Slack configured url=%s", pr_url)
        return
    message = message_text or f"review this PR: {pr_url}"
    output = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{SHIN_URL}/chat/api",
                    json={"message": message},
                    headers={"Authorization": f"Bearer {SHIN_API_KEY}"},
                )
                resp.raise_for_status()
                output = resp.json().get("output", str(resp.json()))
                break
        except (HTTPStatusError, httpx.TransportError) as e:
            status = getattr(e, "response", None)
            status_code = status.status_code if status else 0
            if attempt < 2 and status_code in (0, 502, 503, 504):
                wait = 10 * (attempt + 1)
                log.warning("shin_retry attempt=%d url=%s err=%s waiting=%ds", attempt + 1, pr_url, e, wait)
                await asyncio.sleep(wait)
            else:
                log.error("shin_agent_error url=%s err=%s", pr_url, e)
                output = f":x: Review failed after {attempt + 1} attempt(s): {e}"
                break
        except Exception as e:
            log.error("shin_agent_error url=%s err=%s", pr_url, e)
            output = f":x: Review failed: {e}"
            break
    if not output:
        log.error("shin_empty_output url=%s retrying with explicit message", pr_url)
        output = None
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{SHIN_URL}/chat/api",
                    json={"message": f"review this PR: {pr_url}"},
                    headers={"Authorization": f"Bearer {SHIN_API_KEY}"},
                )
                resp.raise_for_status()
                output = resp.json().get("output") or f":x: shin returned empty output for {pr_url}"
        except Exception as e:
            output = f":x: Review failed: {e}"
    try:
        await slack_handler.bolt.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=output
        )
    except Exception as e:
        log.error("slack_post_error url=%s err=%s", pr_url, e)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if slack_handler.is_enabled():
        asyncio.create_task(slack_handler.startup_scan(review_pr))
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "dev"),
)

slack_handler.mount(app, on_pr_review=review_pr)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
async def login_get():
    return (
        "<html><body>"
        "<form method='post'>"
        "<input name='username' placeholder='username'> "
        "<input name='password' type='password' placeholder='password'> "
        "<button>Login</button>"
        "</form></body></html>"
    )


@app.post("/login")
async def login_post(_request: Request):
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
