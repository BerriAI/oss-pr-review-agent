import hashlib
import hmac
import json
import os

import pytest

os.environ["GITHUB_WEBHOOK_SECRET"] = "test-secret"
os.environ["SLACK_BOT_TOKEN"] = "xoxb-test"
os.environ["SLACK_CHANNEL_ID"] = "C123"
os.environ["ANTHROPIC_API_KEY"] = "sk-test"
os.environ["GITHUB_TOKEN"] = "ghp-test"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_mod  # noqa: E402

SECRET = b"test-secret"


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()


def post(client, body: dict, event: str = "pull_request", sig: str | None = None):
    raw = json.dumps(body).encode()
    headers = {"X-GitHub-Event": event}
    if sig is not None:
        headers["X-Hub-Signature-256"] = sig
    else:
        headers["X-Hub-Signature-256"] = sign(raw)
    return client.post("/webhooks/github", content=raw, headers=headers)


@pytest.fixture
def env(monkeypatch):
    calls: list[dict] = []

    async def fake_review(pr):
        calls.append(pr)

    monkeypatch.setattr(app_mod, "review_pr", fake_review)
    return TestClient(app_mod.app), calls


def pr_payload(action="opened", draft=False):
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "title": "Fix bug",
            "html_url": "https://github.com/BerriAI/litellm/pull/42",
            "diff_url": "https://github.com/BerriAI/litellm/pull/42.diff",
            "user": {"login": "alice"},
            "draft": draft,
        },
    }


def test_opened_non_draft_triggers_review(env):
    client, calls = env
    r = post(client, pr_payload())
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["number"] == 42


def test_draft_pr_skipped(env):
    client, calls = env
    r = post(client, pr_payload(draft=True))
    assert r.status_code == 200
    assert r.json() == {"ok": True, "skipped": "draft"}
    assert calls == []


def test_labeled_action_skipped(env):
    client, calls = env
    r = post(client, pr_payload(action="labeled"))
    assert r.status_code == 200
    assert r.json() == {"ok": True, "skipped": "labeled"}
    assert calls == []


def test_bad_signature_401(env):
    client, calls = env
    r = post(client, pr_payload(), sig="sha256=deadbeef")
    assert r.status_code == 401
    assert calls == []


def test_non_pr_event_skipped(env):
    client, calls = env
    r = post(client, {"action": "opened"}, event="issues")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "skipped": "not_pr"}
    assert calls == []
