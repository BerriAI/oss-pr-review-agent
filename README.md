# litellm-bot

Slack bot for `BerriAI/litellm`. Mention the bot in any channel it's in with a GitHub PR URL and it runs the [litellm-pr-reviewer](https://github.com/BerriAI/pr-review-agent-skills) skill via [Pydantic AI](https://ai.pydantic.dev/) (routed through a [LiteLLM proxy](https://docs.litellm.ai/)), posting the review back in-thread.

```
@litellm-bot https://github.com/BerriAI/litellm/pull/123
```

## Architecture

`app.py` is a FastAPI app that wires a Slack front-end to two parallel Pydantic AI agents, both routed through a LiteLLM proxy.

```
                     ┌──────────────────────────┐
 Slack @-mention ──► │ slack_handler.py         │
 (or DM)             │  • verifies signing key  │
                     │  • finds PR URL in       │
                     │    mention or thread     │
                     └────────────┬─────────────┘
                                  │ on_pr_review(pr_url, channel, thread_ts)
                                  ▼
                     ┌──────────────────────────┐
                     │ app.review_pr            │
                     │  asyncio.gather(         │
                     │    triage_agent,         │  ─┐
                     │    pattern_agent,        │  ─┤  both share one
                     │  )                       │   │  Logfire root span
                     └────────────┬─────────────┘   │
                                  │                 │
              ┌───────────────────┴────────────────┐│
              ▼                                    ▼│
   ┌────────────────────┐             ┌────────────────────────┐
   │ agent              │             │ pattern_agent          │
   │  SKILL.md          │             │  SKILL.md              │
   │  + gather_pr_data  │             │  + gather_pattern_data │
   └─────────┬──────────┘             └────────────┬───────────┘
             │ tool call                           │ tool call
             ▼                                     ▼
    subprocess: scripts/             subprocess: scripts/
    gather_pr_triage_data.py         gather_pattern_data.py
    (GitHub + Greptile + CircleCI)   (GitHub diff + sibling files)
             │                                     │
             └────────────────┬────────────────────┘
                              ▼
                  LiteLLM proxy (OpenAI-compatible)
                              │
                              ▼
                       upstream LLM
                              │
                              ▼
              slack_handler.bolt.client.chat_postMessage
              (single two-section reply in the thread)
```

Key pieces:

- **`slack_handler.py`** owns everything Slack-specific (Bolt app, signing-secret verification, `app_mention` + DM `message` handlers, thread-scanning for a PR URL). It exposes `mount(app, on_pr_review=...)` so `app.py` never has to know about channels or thread timestamps. If `SLACK_BOT_TOKEN` / `SLACK_SIGNING_SECRET` aren't set, the module is a no-op and `/slack/events` simply isn't registered — the `/chat` dev UI still works.
- **Two agents, one model.** `agent` (CI triage) and `pattern_agent` (pattern conformance) are both Pydantic AI `Agent`s pointed at the same `OpenAIChatModel` configured with `LiteLLMProvider`. Each has its own system prompt loaded from a `SKILL.md` file under `skills/` and `skills-local/`.
- **Skills become typed tools.** Each SKILL.md tells the model to shell out to a `gather_*.py` script. Instead of giving the agent a generic `Bash` tool, `app.py` registers `gather_pr_data` / `gather_pattern_data` via `@agent.tool_plain`, runs the script in a subprocess, and returns parsed JSON. A `TOOL_REDIRECT` prefix is prepended to the triage SKILL.md so the model calls the tool instead of trying to run bash. Subprocess failures and non-JSON output raise `ModelRetry` so Pydantic AI re-prompts the agent.
- **`review_pr`** is the Slack callback. It runs both agents in parallel with `asyncio.gather`, wraps the whole thing in a single `logfire.span("review_pr", ...)` so both runs + the Slack post share one trace, and posts one combined reply (`*CI Triage*` + `*Pattern Conformance*`) in-thread. `_run_one` catches per-agent exceptions so one failing agent doesn't take the other down.
- **`/chat` + `/chat/api`** are a local dev UI that calls the same two agents via `asyncio.gather`, keyed by an in-memory `SESSIONS` dict so each browser session keeps separate `triage` / `pattern` message histories. `_extract_tool_trace` pulls `ToolCallPart` / `ToolReturnPart` pairs out of the run so the UI can render tool calls inline.
- **Observability.** `logfire.configure(send_to_logfire="if-token-present")` makes Logfire a no-op locally. With `LOGFIRE_TOKEN` set, `instrument_pydantic_ai`, `instrument_httpx`, and `instrument_fastapi` ship traces for agent runs, all HTTP calls (including to the LiteLLM proxy and GitHub), FastAPI requests, and the gather subprocesses. Stdlib `log.*` calls are bridged into Logfire via `LogfireLoggingHandler`.

## Run locally

```bash
git clone --recursive <this-repo>
cp .env.example .env  # fill in values
uv sync
uv run uvicorn app:app --reload
```

Expose your local port to Slack with `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`) and point your Slack app's **Event Subscriptions → Request URL** at `<tunnel-url>/slack/events`.

If you cloned without `--recursive`: `git submodule update --init --recursive`.

## Test it without Slack

Slack creds are optional. With just `LITELLM_API_KEY` + `GITHUB_TOKEN` set, run `uv run uvicorn app:app --reload` and open <http://localhost:8000/chat>. It's a single-page UI that hits the same agent the Slack handler uses, with tool calls printed inline so you can see the agent run the gather script. Try:

```
Triage this PR: https://github.com/BerriAI/litellm/pull/123
```

There's also a JSON endpoint if you'd rather curl it:

```bash
curl -s localhost:8000/chat/api -H 'content-type: application/json' \
  -d '{"message": "Triage this PR: https://github.com/BerriAI/litellm/pull/123"}'
```

## Slack app setup

In [api.slack.com/apps](https://api.slack.com/apps):

1. **OAuth & Permissions → Bot Token Scopes**: add
   - `chat:write` — post the review back
   - `app_mentions:read` — receive `@bot` events
   - `channels:history`, `groups:history`, `im:history`, `mpim:history` — read the parent message of a thread when someone @-mentions the bot in a thread but the PR URL is in the OP (e.g. `@bot please review this PR`)
2. **Event Subscriptions**: enable, set Request URL to `<your-host>/slack/events`, subscribe to bot event `app_mention`.
3. **Install to Workspace** (or reinstall after scope changes). Copy the Bot Token (`xoxb-…`) into `SLACK_BOT_TOKEN`.
4. **Basic Information → App Credentials**: copy the Signing Secret into `SLACK_SIGNING_SECRET`.
5. Invite the bot to whichever channels people will mention it from.

### How the bot finds the PR URL

- **Top-level mention in a channel** (`@bot https://...pull/123`): only the mention text is parsed. Channel history is *not* searched, to avoid grabbing unrelated old PR links.
- **Mention inside a thread** (`@bot please review this PR`): the bot fetches the whole thread via `conversations.replies` and uses the first PR URL it finds in any message (OP or earlier reply). This handles the common pattern of "user pastes a PR in the OP, then later @-mentions the bot to review it."

If the `*:history` scopes aren't granted, in-thread mentions silently fall back to scanning only the mention text and the bot will reply asking for a URL.

## Deploy

Run `fly launch` (Fly.io reads `Procfile`) or create a new Python service on Render/Railway and point it at this repo. Set the env vars below in the platform's secret store. `Procfile` declares the start command.

## Env vars

| Var | Required | Notes |
|---|---|---|
| `SLACK_BOT_TOKEN` | for Slack | `xoxb-…`, scopes `chat:write` + `app_mentions:read`. Skip to use only the `/chat` UI |
| `SLACK_SIGNING_SECRET` | for Slack | Verifies events came from Slack. Skip to use only the `/chat` UI |
| `LITELLM_API_KEY` | yes | LiteLLM proxy virtual key (or upstream provider key if not using the proxy) |
| `LITELLM_API_BASE` | no | LiteLLM proxy URL. Defaults to `http://0.0.0.0:4000` |
| `LITELLM_MODEL` | no | Model alias the proxy routes. Defaults to `claude-sonnet-4-6` |
| `GITHUB_TOKEN` | yes | PAT with `public_repo` — required by the skill |
| `PORT` | no | Defaults to 8000 |
| `LOGFIRE_TOKEN` | no | Pydantic Logfire write token. When set, agent runs, HTTPX calls, FastAPI requests, and the gather subprocess are traced and shipped to Logfire. No-op without it. |

## Tests

```bash
uv run pytest
```
