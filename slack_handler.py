"""Slack-specific wiring: Bolt app, event handlers, FastAPI route.

Kept separate from `app.py` so the agent code doesn't have to know anything
about Slack channels, threads, or message subtypes.

Public surface:
- `is_enabled()` - True iff SLACK_BOT_TOKEN + SLACK_SIGNING_SECRET are set
- `mount(fastapi_app, on_pr_review)` - registers /slack/events and event handlers.
  `on_pr_review(pr_url, channel, thread_ts, message_text)` is invoked (via asyncio.create_task)
  whenever a user sends us a PR URL via @-mention, DM, or channel message.
- `startup_scan(on_pr_review)` - scans last 20 messages in all bot channels for
  unreviewed PRs and triggers reviews. Call from app lifespan.
- `bolt` / `request_handler` - the underlying Bolt app + FastAPI adapter, or
  None if Slack creds aren't configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request

log = logging.getLogger("litellm-bot.slack")

PR_URL_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+/pull/\d+")
BOT_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")

THREAD_LOOKBACK_LIMIT = 50

ReviewCallback = Callable[[str, str, str, Optional[str]], Awaitable[None]]

bolt = None
request_handler = None

if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_SIGNING_SECRET"):
    from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
    from slack_bolt.async_app import AsyncApp

    bolt = AsyncApp(
        token=os.environ["SLACK_BOT_TOKEN"],
        signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    )
    request_handler = AsyncSlackRequestHandler(bolt)
else:
    log.warning("SLACK_BOT_TOKEN/SLACK_SIGNING_SECRET unset; /slack/events disabled")


def is_enabled() -> bool:
    return bolt is not None


async def _find_pr_url_in_thread(channel: str, thread_ts: str) -> Optional[str]:
    if bolt is None:
        return None
    try:
        resp = await bolt.client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=THREAD_LOOKBACK_LIMIT,
        )
    except Exception as e:
        log.warning(
            "conversations_replies failed channel=%s ts=%s err=%s",
            channel,
            thread_ts,
            e,
        )
        return None

    for msg in resp.get("messages", []) or []:
        match = PR_URL_RE.search(msg.get("text", "") or "")
        if match:
            return match.group(0)
    return None


async def _bot_already_replied(channel: str, thread_ts: str, bot_id: str) -> bool:
    if bolt is None:
        return False
    try:
        resp = await bolt.client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=50,
        )
        return any(m.get("bot_id") == bot_id for m in resp.get("messages", []))
    except Exception:
        return False


async def startup_scan(on_pr_review: ReviewCallback) -> None:
    if bolt is None:
        return
    try:
        auth = await bolt.client.auth_test()
        bot_id = auth.get("bot_id", "")

        channels_resp = await bolt.client.conversations_list(
            types="public_channel,private_channel",
            exclude_archived=True,
            limit=200,
        )
        channels = [c for c in channels_resp.get("channels", []) if c.get("is_member")]
        log.info("startup_scan checking %d channels", len(channels))

        for channel in channels:
            channel_id = channel["id"]
            try:
                history = await bolt.client.conversations_history(
                    channel=channel_id,
                    limit=20,
                )
            except Exception as e:
                log.warning("startup_scan history failed channel=%s err=%s", channel_id, e)
                continue

            for msg in history.get("messages", []):
                if msg.get("bot_id") or msg.get("subtype"):
                    continue
                match = PR_URL_RE.search(msg.get("text", "") or "")
                if not match:
                    continue
                pr_url = match.group(0)
                msg_ts = msg["ts"]
                if await _bot_already_replied(channel_id, msg_ts, bot_id):
                    log.info("startup_scan skip already_reviewed url=%s", pr_url)
                    continue
                log.info("startup_scan trigger url=%s channel=%s", pr_url, channel_id)
                asyncio.create_task(on_pr_review(pr_url, channel_id, msg_ts, None))
    except Exception as e:
        log.error("startup_scan failed err=%s", e)


def _mount_handlers(on_pr_review: ReviewCallback) -> None:
    assert bolt is not None

    async def handle_mention(event, say) -> None:
        channel = event["channel"]
        parent_ts = event.get("thread_ts")
        in_thread = parent_ts is not None
        reply_thread_ts = parent_ts or event["ts"]

        if in_thread:
            pr_url = await _find_pr_url_in_thread(channel, parent_ts)
        else:
            match = PR_URL_RE.search(event.get("text", "") or "")
            pr_url = match.group(0) if match else None

        if pr_url is None:
            await say(
                text="Give me a GitHub PR URL, e.g. `@bot https://github.com/BerriAI/litellm/pull/123`",
                thread_ts=reply_thread_ts,
            )
            return

        await say(
            text=f":eyes: reviewing {pr_url}...",
            thread_ts=reply_thread_ts,
        )
        cleaned_text = BOT_MENTION_RE.sub("", event.get("text", "") or "").strip()
        if cleaned_text and pr_url in cleaned_text:
            message_text = cleaned_text
        elif cleaned_text:
            message_text = f"{cleaned_text} {pr_url}"
        else:
            message_text = None
        asyncio.create_task(on_pr_review(pr_url, channel, reply_thread_ts, message_text))

    async def handle_message(event, say) -> None:
        if event.get("bot_id") or event.get("subtype"):
            return
        channel_type = event.get("channel_type", "")
        if channel_type == "im":
            await handle_mention(event, say)
        elif channel_type in ("channel", "group", "mpim"):
            match = PR_URL_RE.search(event.get("text", "") or "")
            if not match:
                return
            pr_url = match.group(0)
            channel = event["channel"]
            msg_ts = event["ts"]
            await say(text=f":eyes: reviewing {pr_url}...", thread_ts=msg_ts)
            cleaned_text = BOT_MENTION_RE.sub("", event.get("text", "") or "").strip()
            asyncio.create_task(
                on_pr_review(pr_url, channel, msg_ts, cleaned_text or None)
            )

    bolt.event("app_mention")(handle_mention)
    bolt.event("message")(handle_message)

    global handle_mention_fn, handle_message_fn  # noqa: PLW0603
    handle_mention_fn = handle_mention
    handle_message_fn = handle_message


handle_mention_fn: Optional[Callable[..., Awaitable[None]]] = None
handle_message_fn: Optional[Callable[..., Awaitable[None]]] = None


def mount(fastapi_app: FastAPI, on_pr_review: ReviewCallback) -> None:
    if not is_enabled():
        return

    _mount_handlers(on_pr_review)

    @fastapi_app.post("/slack/events")
    async def slack_events(req: Request):
        return await request_handler.handle(req)
