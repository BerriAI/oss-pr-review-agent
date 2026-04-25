"""Slack-specific wiring: Bolt app, event handlers, FastAPI route.

Kept separate from `app.py` so the agent code doesn't have to know anything
about Slack channels, threads, or message subtypes.

Public surface:
- `is_enabled()` - True iff SLACK_BOT_TOKEN + SLACK_SIGNING_SECRET are set
- `mount(fastapi_app, on_pr_review)` - registers /slack/events and event handlers.
  `on_pr_review(pr_url, channel, thread_ts)` is invoked (via asyncio.create_task)
  whenever a user sends us a PR URL via @-mention or DM.
- `bolt` / `request_handler` - the underlying Bolt app + FastAPI adapter, or
  None if Slack creds aren't configured. Exposed mainly so tests can patch them.
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

# How many messages back in a thread we'll scan for a PR URL. Slack's default
# page size is 28; 50 covers any reasonable "PR in the OP, @-mention much
# later" case without a second pagination round-trip.
THREAD_LOOKBACK_LIMIT = 50

ReviewCallback = Callable[[str, str, str], Awaitable[None]]

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
    """Scan a Slack thread (parent + replies) for a GitHub PR URL.

    Called for every in-thread @-mention so we pick up the PR URL whether it
    was in the OP, an earlier reply, or the mention itself. Typical flow:
    user pastes a PR link in the OP, then later replies "@bot review this".

    Returns the first PR URL found, or None. Silently returns None on API
    errors (e.g. missing channels:history scope) so the caller can still
    fall back to the "give me a URL" prompt.
    """
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
            "conversations_replies failed channel=%s ts=%s err=%s "
            "(bot may need channels:history / groups:history / im:history scope)",
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


def _mount_handlers(on_pr_review: ReviewCallback) -> None:
    """Attach `app_mention` + DM `message` handlers to the Bolt app.

    Split out so the registration is testable without spinning up FastAPI.
    """
    assert bolt is not None  # caller checked is_enabled()

    async def handle_mention(event, say) -> None:
        # Two cases we handle differently:
        #   1. Mention in a thread → scan the whole thread for a PR URL.
        #      (The user usually pastes the URL in the OP and then @-mentions
        #      us with "please review this".)
        #   2. Mention at top level in a channel / DM → only look at the
        #      mention text itself. Grabbing channel history would be both
        #      noisy (random old PR URLs) and a bigger scope ask.
        channel = event["channel"]
        parent_ts = event.get("thread_ts")
        in_thread = parent_ts is not None
        # Where to post the reply: existing thread if any, else start one off
        # the mention itself.
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
            text=f":eyes: reviewing {pr_url} (CI triage + pattern conformance)...",
            thread_ts=reply_thread_ts,
        )
        asyncio.create_task(on_pr_review(pr_url, channel, reply_thread_ts))

    async def handle_dm(event, say) -> None:
        # Slack also delivers the bot's own messages and message_changed/deleted
        # subtypes as message events; ignore those so we don't loop on ourselves.
        if event.get("bot_id") or event.get("subtype"):
            return
        if event.get("channel_type") != "im":
            return
        await handle_mention(event, say)

    bolt.event("app_mention")(handle_mention)
    bolt.event("message")(handle_dm)

    # Expose at module scope so tests (and app.py, if it ever needs to) can
    # call them directly without faking a Slack event through Bolt.
    global handle_mention_fn, handle_dm_fn  # noqa: PLW0603
    handle_mention_fn = handle_mention
    handle_dm_fn = handle_dm


# Set by `_mount_handlers`. None until `mount()` runs (i.e. when Slack creds
# are missing, or before app startup). Tests call `_mount_handlers` directly to
# populate these without needing FastAPI.
handle_mention_fn: Optional[Callable[..., Awaitable[None]]] = None
handle_dm_fn: Optional[Callable[..., Awaitable[None]]] = None


def mount(fastapi_app: FastAPI, on_pr_review: ReviewCallback) -> None:
    """Register Slack event handlers + the /slack/events HTTP route.

    No-op if Slack creds aren't configured (so local /chat dev still works).
    """
    if not is_enabled():
        return

    _mount_handlers(on_pr_review)

    @fastapi_app.post("/slack/events")
    async def slack_events(req: Request):
        return await request_handler.handle(req)
