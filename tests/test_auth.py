"""Auth surface tests for /chat/api: bearer-token + session-cookie matrix.

The endpoint accepts either credential type independently. Either auth path
must work standalone, both can be configured at once, and unsetting both
disables the gate (local dev mode).

We re-import `app` per test with a fresh env so `AUTH_ENABLED` /
`SESSION_AUTH_ENABLED` / `BOT_API_KEYS` are recomputed at module load (they
read os.environ at import time on purpose — no hot-reload needed in prod).
"""
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient


# Required env for app.py to import at all.
BASE_ENV = {
    "LITELLM_API_KEY": "sk-test",
    "GITHUB_TOKEN": "ghp-test",
}


def _fresh_app(monkeypatch, **extra_env):
    """Reload app.py with a controlled env so module-level auth flags are
    recomputed. Tests use TestClient against the returned app.

    `app.py` calls `load_dotenv()` at import — that would re-populate values
    from the developer's local .env (e.g. ADMIN_USERNAME) and defeat the
    monkeypatch. We stub it to a no-op so tests get only what they pass via
    `extra_env`.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    for k, v in {**BASE_ENV, **extra_env}.items():
        monkeypatch.setenv(k, v)
    # Drop any stale env this test isn't setting so a previous test's value
    # doesn't bleed in (e.g. test A sets BOT_API_KEYS, test B doesn't want it).
    for k in ("ADMIN_USERNAME", "ADMIN_PASSWORD", "BOT_API_KEYS", "SESSION_SECRET"):
        if k not in extra_env:
            monkeypatch.delenv(k, raising=False)
    # Pop the cached module so the import re-reads os.environ.
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def _stub_chat_agent(app_mod, monkeypatch):
    """Prevent the real chat_agent.run from being called. Keeps these tests
    pure auth-surface tests with no LLM dependency."""

    class FakeResult:
        output = "ok"

        def all_messages(self):
            return []

    async def fake_run(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(app_mod.chat_agent, "run", fake_run)


# --- AUTH OFF: no creds configured ------------------------------------------

def test_no_auth_configured_chat_api_is_open(monkeypatch):
    """Local dev: neither password nor bearer keys set → /chat/api accepts
    requests with no credentials, same as before this change."""
    app_mod = _fresh_app(monkeypatch)
    _stub_chat_agent(app_mod, monkeypatch)

    assert app_mod.AUTH_ENABLED is False
    assert app_mod.SESSION_AUTH_ENABLED is False
    assert app_mod.BOT_API_KEYS == frozenset()

    client = TestClient(app_mod.app)
    r = client.post("/chat/api", json={"message": "hello"})
    assert r.status_code == 200, r.text


# --- AUTH ON via bearer only ------------------------------------------------

def test_bearer_only_valid_key_accepted(monkeypatch):
    app_mod = _fresh_app(monkeypatch, BOT_API_KEYS="key-alpha,key-beta")
    _stub_chat_agent(app_mod, monkeypatch)

    assert app_mod.AUTH_ENABLED is True
    assert app_mod.SESSION_AUTH_ENABLED is False  # no admin password
    assert app_mod.BOT_API_KEYS == {"key-alpha", "key-beta"}

    client = TestClient(app_mod.app)
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer key-alpha"},
    )
    assert r.status_code == 200, r.text

    # Second key in the CSV also works (rotation use-case).
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer key-beta"},
    )
    assert r.status_code == 200, r.text


def test_bearer_only_missing_or_invalid_rejected(monkeypatch):
    app_mod = _fresh_app(monkeypatch, BOT_API_KEYS="key-alpha")
    _stub_chat_agent(app_mod, monkeypatch)

    client = TestClient(app_mod.app)

    # No header at all.
    r = client.post("/chat/api", json={"message": "hi"})
    assert r.status_code == 401

    # Wrong key.
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401

    # Right key but wrong scheme — explicit Bearer parser only accepts Bearer.
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Basic key-alpha"},
    )
    assert r.status_code == 401

    # Empty bearer value.
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer "},
    )
    assert r.status_code == 401


def test_bearer_only_csv_handles_whitespace_and_empties(monkeypatch):
    """`BOT_API_KEYS=" k1 ,, k2 "` should parse as {"k1", "k2"} — operators
    edit env files by hand and shouldn't get bitten by stray spaces."""
    app_mod = _fresh_app(monkeypatch, BOT_API_KEYS=" k1 ,, k2 ")
    assert app_mod.BOT_API_KEYS == {"k1", "k2"}

    _stub_chat_agent(app_mod, monkeypatch)
    client = TestClient(app_mod.app)
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer k2"},
    )
    assert r.status_code == 200


def test_bearer_only_login_form_tells_user_to_use_bearer(monkeypatch):
    """Bearer-only deployment → /login POST should explain that password
    auth isn't configured rather than 500ing on missing session middleware."""
    app_mod = _fresh_app(monkeypatch, BOT_API_KEYS="k1")

    client = TestClient(app_mod.app)
    r = client.post("/login", data={"username": "x", "password": "y"})
    assert r.status_code == 400
    assert "bearer token" in r.text.lower()


# --- AUTH ON via session only -----------------------------------------------

def test_session_only_unauthenticated_request_rejected(monkeypatch):
    """Existing behavior preserved: session auth without bearer still 401s
    requests that don't carry a session cookie."""
    app_mod = _fresh_app(
        monkeypatch,
        ADMIN_USERNAME="admin",
        ADMIN_PASSWORD="hunter2",
        SESSION_SECRET="test-secret",
    )
    _stub_chat_agent(app_mod, monkeypatch)

    assert app_mod.SESSION_AUTH_ENABLED is True
    assert app_mod.BOT_API_KEYS == frozenset()

    client = TestClient(app_mod.app)
    r = client.post("/chat/api", json={"message": "hi"})
    assert r.status_code == 401


def test_session_only_bearer_header_does_not_grant_access(monkeypatch):
    """If BOT_API_KEYS is unset, a bearer header must not authenticate even
    if the value looks key-shaped — avoids accidental open-by-default."""
    app_mod = _fresh_app(
        monkeypatch,
        ADMIN_USERNAME="admin",
        ADMIN_PASSWORD="hunter2",
        SESSION_SECRET="test-secret",
    )
    _stub_chat_agent(app_mod, monkeypatch)

    client = TestClient(app_mod.app)
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer some-random-token"},
    )
    assert r.status_code == 401


# --- AUTH ON via both -------------------------------------------------------

def test_both_paths_configured_either_credential_works(monkeypatch):
    """The common prod shape: humans use the password form, bots use bearer.
    Both paths must work independently against the same endpoint."""
    app_mod = _fresh_app(
        monkeypatch,
        ADMIN_USERNAME="admin",
        ADMIN_PASSWORD="hunter2",
        SESSION_SECRET="test-secret",
        BOT_API_KEYS="bot-key",
    )
    _stub_chat_agent(app_mod, monkeypatch)

    assert app_mod.SESSION_AUTH_ENABLED is True
    assert app_mod.BOT_API_KEYS == {"bot-key"}

    client = TestClient(app_mod.app)

    # Bearer path.
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": "Bearer bot-key"},
    )
    assert r.status_code == 200

    # Session path: log in via /login, then re-use the cookie jar.
    r = client.post("/login", data={"username": "admin", "password": "hunter2"})
    assert r.status_code in (200, 303)
    r = client.post("/chat/api", json={"message": "hi"})
    assert r.status_code == 200


# --- regression: bearer requires "Bearer " (case-insensitive) -----------------

@pytest.mark.parametrize(
    "header",
    [
        "Bearer key-alpha",
        "bearer key-alpha",
        "BEARER key-alpha",
        "Bearer\tkey-alpha",  # tab not space — should fail (strict prefix)
    ],
)
def test_bearer_scheme_case_insensitive_but_strict_separator(monkeypatch, header):
    app_mod = _fresh_app(monkeypatch, BOT_API_KEYS="key-alpha")
    _stub_chat_agent(app_mod, monkeypatch)

    client = TestClient(app_mod.app)
    r = client.post(
        "/chat/api",
        json={"message": "hi"},
        headers={"Authorization": header},
    )
    if header.startswith(("Bearer ", "bearer ", "BEARER ")):
        assert r.status_code == 200
    else:
        assert r.status_code == 401
