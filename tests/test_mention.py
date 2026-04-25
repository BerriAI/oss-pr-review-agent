import asyncio
import os

import pytest

os.environ["SLACK_BOT_TOKEN"] = "xoxb-test"
os.environ["SLACK_SIGNING_SECRET"] = "test-signing-secret"
os.environ["LITELLM_API_KEY"] = "sk-test"
os.environ["GITHUB_TOKEN"] = "ghp-test"

import app as app_mod  # noqa: E402


class FakeSay:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, text: str, thread_ts: str | None = None):
        self.calls.append({"text": text, "thread_ts": thread_ts})


@pytest.fixture
def env(monkeypatch):
    review_calls: list[dict] = []

    async def fake_review(pr_url, channel, thread_ts):
        review_calls.append(
            {
                "pr_url": pr_url,
                "channel": channel,
                "thread_ts": thread_ts,
            }
        )

    monkeypatch.setattr(app_mod, "review_pr", fake_review)
    return review_calls


def mention_event(text: str, ts: str = "1700000000.000100", thread_ts: str | None = None):
    event = {
        "type": "app_mention",
        "user": "U123",
        "channel": "C456",
        "text": text,
        "ts": ts,
    }
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


async def invoke(event: dict):
    """Call the app_mention handler directly with a fake `say`."""
    say = FakeSay()
    await app_mod.handle_mention(event=event, say=say)
    await asyncio.sleep(0)
    return say


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


def test_review_pr_runs_both_agents_and_posts_one_combined_message(monkeypatch):
    """review_pr should run both agents in parallel and post a single Slack
    message with both 'CI Triage' and 'Pattern Conformance' sections.
    """
    run_calls: list[tuple[str, str]] = []

    async def fake_run_one(selected_agent, prompt, history=None):
        name = "triage" if selected_agent is app_mod.agent else "pattern"
        run_calls.append((name, prompt))
        return (f"{name} verdict for {prompt}", [])

    posts: list[dict] = []

    class FakeSlackClient:
        async def chat_postMessage(self, **kwargs):
            posts.append(kwargs)

    class FakeBolt:
        client = FakeSlackClient()

    monkeypatch.setattr(app_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(app_mod, "bolt", FakeBolt())

    asyncio.run(
        app_mod.review_pr(
            "https://github.com/BerriAI/litellm/pull/77", "C1", "1700.000200"
        )
    )

    names = sorted(name for name, _ in run_calls)
    assert names == ["pattern", "triage"]
    assert len(posts) == 1
    body = posts[0]["text"]
    assert "*CI Triage*" in body
    assert "*Pattern Conformance*" in body
    assert "https://github.com/BerriAI/litellm/pull/77" in body


def test_mention_without_url_prompts_for_one(env):
    review_calls = env
    event = mention_event("<@U999BOT> hello")
    say = asyncio.run(invoke(event))

    assert review_calls == []
    assert len(say.calls) == 1
    assert "GitHub PR URL" in say.calls[0]["text"]


def test_mention_in_thread_replies_in_same_thread(env):
    review_calls = env
    event = mention_event(
        "<@U999BOT> https://github.com/BerriAI/litellm/pull/7",
        ts="1700000000.000200",
        thread_ts="1700000000.000100",
    )
    asyncio.run(invoke(event))

    assert len(review_calls) == 1
    assert review_calls[0]["thread_ts"] == "1700000000.000100"


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


def test_healthz():
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
