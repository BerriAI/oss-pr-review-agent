import asyncio
import os

import pytest

os.environ["SLACK_BOT_TOKEN"] = "xoxb-test"
os.environ["SLACK_SIGNING_SECRET"] = "test-signing-secret"
os.environ["LITELLM_API_KEY"] = "sk-test"
os.environ["GITHUB_TOKEN"] = "ghp-test"

import app as app_mod  # noqa: E402
import slack_handler as slack_mod  # noqa: E402


class FakeSay:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, text: str, thread_ts: str | None = None):
        self.calls.append({"text": text, "thread_ts": thread_ts})


class FakeSlackClient:
    """Mimics enough of `bolt.client` for the handler tests.

    `replies_by_thread` is keyed by (channel, thread_ts) and returns the
    `messages` list `conversations.replies` would have returned.
    """

    def __init__(
        self,
        replies_by_thread: dict[tuple[str, str], list[dict]] | None = None,
    ):
        self.replies_by_thread = replies_by_thread or {}
        self.posts: list[dict] = []
        self.replies_calls: list[dict] = []

    async def conversations_replies(
        self, *, channel: str, ts: str, limit: int | None = None
    ):
        self.replies_calls.append({"channel": channel, "ts": ts, "limit": limit})
        return {"messages": self.replies_by_thread.get((channel, ts), [])}

    async def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)


class FakeBolt:
    """Stand-in for slack_bolt.async_app.AsyncApp.

    Real AsyncApp exposes `client` as a read-only property, so tests that need
    to inject a fake client swap out `slack_mod.bolt` itself with one of these.
    `event(...)` is a no-op decorator factory so `_mount_handlers` can still
    re-register against this fake without bolt internals.
    """

    def __init__(self, client: FakeSlackClient | None = None):
        self.client = client or FakeSlackClient()

    def event(self, _name: str):
        def _decorator(fn):
            return fn

        return _decorator


@pytest.fixture
def env(monkeypatch):
    """Patch out the agent run so we only exercise Slack handler logic.

    Returns a list that captures (pr_url, channel, thread_ts) for each call
    the handler makes into the review pipeline.
    """
    review_calls: list[dict] = []

    async def fake_on_pr_review(pr_url, channel, thread_ts):
        review_calls.append(
            {"pr_url": pr_url, "channel": channel, "thread_ts": thread_ts}
        )

    # Re-mount slack handlers against our fake callback. This rebinds
    # slack_mod.handle_mention / handle_dm to closures that call fake_on_pr_review,
    # so the existing bolt.event registrations from app.py import don't matter.
    slack_mod._mount_handlers(fake_on_pr_review)
    return review_calls


def mention_event(
    text: str,
    ts: str = "1700000000.000100",
    thread_ts: str | None = None,
    channel: str = "C456",
):
    event = {
        "type": "app_mention",
        "user": "U123",
        "channel": channel,
        "text": text,
        "ts": ts,
    }
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


async def invoke(event: dict):
    """Call the app_mention handler directly with a fake `say`."""
    say = FakeSay()
    assert slack_mod.handle_mention_fn is not None, (
        "env fixture should have called _mount_handlers"
    )
    await slack_mod.handle_mention_fn(event, say)
    await asyncio.sleep(0)
    return say


# --- top-level (channel) mentions: only consider the mention text -----------

def test_mention_with_pr_url_triggers_review(env):
    review_calls = env
    event = mention_event("<@U999BOT> please review https://github.com/BerriAI/litellm/pull/42 thanks")
    say = asyncio.run(invoke(event))

    assert len(review_calls) == 1
    call = review_calls[0]
    assert call["pr_url"] == "https://github.com/BerriAI/litellm/pull/42"
    assert call["channel"] == "C456"
    assert call["thread_ts"] == "1700000000.000100"
    assert any(":eyes:" in c["text"] for c in say.calls)
    assert any("CI triage + pattern conformance" in c["text"] for c in say.calls)


def test_mention_without_url_prompts_for_one(env):
    review_calls = env
    event = mention_event("<@U999BOT> hello")
    say = asyncio.run(invoke(event))

    assert review_calls == []
    assert len(say.calls) == 1
    assert "GitHub PR URL" in say.calls[0]["text"]


def test_mention_top_level_uses_message_ts_as_thread(env):
    review_calls = env
    event = mention_event("<@U999BOT> https://github.com/BerriAI/litellm/pull/7")
    asyncio.run(invoke(event))

    assert review_calls[0]["thread_ts"] == "1700000000.000100"


def test_pr_url_regex_rejects_non_pr_links(env):
    review_calls = env
    event = mention_event("<@U999BOT> https://github.com/BerriAI/litellm/issues/42")
    asyncio.run(invoke(event))

    assert review_calls == []


def test_top_level_mention_does_not_fetch_thread_history(env, monkeypatch):
    """Channel mentions should never call conversations.replies — that would
    pull in unrelated old PR URLs from the channel's threads.
    """
    fake_bolt = FakeBolt()
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)

    event = mention_event("<@U999BOT> hello, no url here")
    asyncio.run(invoke(event))

    assert fake_bolt.client.replies_calls == []


# --- in-thread mentions: scan the whole thread for a PR URL -----------------

def test_mention_in_thread_replies_in_same_thread(env, monkeypatch):
    """The mention itself contains the URL, but we're in a thread, so we
    still scan the thread (which will also contain that URL) and post the
    reply back into the parent thread."""
    fake_bolt = FakeBolt(
        FakeSlackClient(
            replies_by_thread={
                ("C456", "1700000000.000100"): [
                    {"text": "parent message"},
                    {"text": "<@U999BOT> https://github.com/BerriAI/litellm/pull/7"},
                ]
            }
        )
    )
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)
    review_calls = env

    event = mention_event(
        "<@U999BOT> https://github.com/BerriAI/litellm/pull/7",
        ts="1700000000.000200",
        thread_ts="1700000000.000100",
    )
    asyncio.run(invoke(event))

    assert len(review_calls) == 1
    assert review_calls[0]["thread_ts"] == "1700000000.000100"
    assert review_calls[0]["pr_url"] == "https://github.com/BerriAI/litellm/pull/7"


def test_mention_in_thread_picks_up_pr_url_from_parent_message(env, monkeypatch):
    """The reproducer from the screenshot: parent message has the PR URL,
    the @-mention reply just says "please review this PR". We should still
    kick off a review using the URL from the OP.
    """
    fake_bolt = FakeBolt(
        FakeSlackClient(
            replies_by_thread={
                ("C456", "1700000000.000100"): [
                    {
                        "text": "Hi, I added a recording: "
                        "https://github.com/BerriAI/litellm/pull/26011"
                    },
                    {"text": "<@U999BOT> please review this PR"},
                ]
            }
        )
    )
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)
    review_calls = env

    event = mention_event(
        "<@U999BOT> please review this PR",
        ts="1700000000.000300",
        thread_ts="1700000000.000100",
    )
    say = asyncio.run(invoke(event))

    assert len(review_calls) == 1
    assert review_calls[0]["pr_url"] == "https://github.com/BerriAI/litellm/pull/26011"
    assert review_calls[0]["thread_ts"] == "1700000000.000100"
    assert any(":eyes:" in c["text"] for c in say.calls)
    assert fake_bolt.client.replies_calls == [
        {
            "channel": "C456",
            "ts": "1700000000.000100",
            "limit": slack_mod.THREAD_LOOKBACK_LIMIT,
        }
    ]


def test_mention_in_thread_with_no_pr_url_anywhere_prompts_for_one(env, monkeypatch):
    fake_bolt = FakeBolt(
        FakeSlackClient(
            replies_by_thread={
                ("C456", "1700000000.000100"): [
                    {"text": "just chatting, no link"},
                    {"text": "<@U999BOT> review please"},
                ]
            }
        )
    )
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)
    review_calls = env

    event = mention_event(
        "<@U999BOT> review please",
        ts="1700000000.000300",
        thread_ts="1700000000.000100",
    )
    say = asyncio.run(invoke(event))

    assert review_calls == []
    assert len(say.calls) == 1
    assert "GitHub PR URL" in say.calls[0]["text"]
    assert say.calls[0]["thread_ts"] == "1700000000.000100"


def test_thread_history_fetch_failure_falls_back_to_prompt(env, monkeypatch):
    """If conversations.replies blows up (e.g. missing scope) we should
    still respond gracefully instead of 500ing on the user."""

    class ExplodingClient(FakeSlackClient):
        async def conversations_replies(self, **kwargs):
            raise RuntimeError("missing_scope: channels:history")

    monkeypatch.setattr(slack_mod, "bolt", FakeBolt(ExplodingClient()))
    review_calls = env

    event = mention_event(
        "<@U999BOT> review please",
        ts="1700000000.000300",
        thread_ts="1700000000.000100",
    )
    say = asyncio.run(invoke(event))

    assert review_calls == []
    assert len(say.calls) == 1
    assert "GitHub PR URL" in say.calls[0]["text"]


# --- review_pr (the agent pipeline) -----------------------------------------

def test_review_pr_runs_both_agents_and_posts_card_plus_drilldown(monkeypatch):
    """review_pr should run both typed agents in parallel and post:
      1. The fused card (built by fuse() + render_card())
      2. A threaded drill-down reply
    """
    run_calls: list[str] = []

    fake_triage = app_mod.TriageReport(
        pr_number=77,
        pr_title="test pr",
        pr_author="alice",
        pr_summary="Adds a feature.",
        has_circleci_checks=True,
        greptile_score=5,
    )
    fake_pattern = app_mod.PatternReport(findings=[], tech_debt=[])

    async def fake_run_triage(prompt, history=None):
        run_calls.append("triage")
        return (fake_triage, [], None)

    async def fake_run_pattern(prompt, history=None):
        run_calls.append("pattern")
        return (fake_pattern, [], None)

    fake_bolt = FakeBolt()

    monkeypatch.setattr(app_mod, "_run_triage", fake_run_triage)
    monkeypatch.setattr(app_mod, "_run_pattern", fake_run_pattern)
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)

    asyncio.run(
        app_mod.review_pr(
            "https://github.com/BerriAI/litellm/pull/77", "C1", "1700.000200"
        )
    )

    assert sorted(run_calls) == ["pattern", "triage"]
    # One card message + one drill-down reply, both threaded.
    assert len(fake_bolt.client.posts) == 2
    card_body = fake_bolt.client.posts[0]["text"]
    drilldown_body = fake_bolt.client.posts[1]["text"]
    assert "*Triage Summary*" in card_body
    assert "*Merge Confidence: 5/5*" in card_body
    assert "✅ READY" in card_body
    assert "*Drill-down*" in drilldown_body


def test_review_pr_posts_fallback_card_when_agent_fails(monkeypatch):
    """If either agent crashes, user should still see a card-shaped message
    rather than a raw exception trace."""

    async def fake_run_triage(prompt, history=None):
        return (None, [], "model timed out")

    async def fake_run_pattern(prompt, history=None):
        return (app_mod.PatternReport(), [], None)

    fake_bolt = FakeBolt()
    monkeypatch.setattr(app_mod, "_run_triage", fake_run_triage)
    monkeypatch.setattr(app_mod, "_run_pattern", fake_run_pattern)
    monkeypatch.setattr(slack_mod, "bolt", fake_bolt)

    asyncio.run(
        app_mod.review_pr(
            "https://github.com/BerriAI/litellm/pull/77", "C1", "1700.000200"
        )
    )

    assert len(fake_bolt.client.posts) == 1
    body = fake_bolt.client.posts[0]["text"]
    assert "*Triage Summary*" in body
    assert "⚠️ ERROR" in body
    assert "model timed out" in body


def test_healthz():
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
