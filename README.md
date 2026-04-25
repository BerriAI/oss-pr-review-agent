# litellm-bot

Slack bot for `BerriAI/litellm`. Mention the bot in any channel it's in with a GitHub PR URL and it runs the [litellm-pr-reviewer](https://github.com/BerriAI/pr-review-agent-skills) skill via [Pydantic AI](https://ai.pydantic.dev/) (routed through a [LiteLLM proxy](https://docs.litellm.ai/)), posting the review back in-thread.

```
@litellm-bot https://github.com/BerriAI/litellm/pull/123
```

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

1. **OAuth & Permissions → Bot Token Scopes**: add `chat:write` and `app_mentions:read`.
2. **Event Subscriptions**: enable, set Request URL to `<your-host>/slack/events`, subscribe to bot event `app_mention`.
3. **Install to Workspace** (or reinstall after scope changes). Copy the Bot Token (`xoxb-…`) into `SLACK_BOT_TOKEN`.
4. **Basic Information → App Credentials**: copy the Signing Secret into `SLACK_SIGNING_SECRET`.
5. Invite the bot to whichever channels people will mention it from.

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
| `CIRCLECI_TOKEN` | no | Enables CircleCI log splicing in the review |
| `PORT` | no | Defaults to 8000 |

## Tests

```bash
uv run pytest
```
