# litellm-bot

Receives GitHub PR webhooks for `BerriAI/litellm`, runs the [litellm-pr-reviewer](https://github.com/BerriAI/pr-review-agent-skills) skill via the Claude Agent SDK, and posts the verdict to a Slack channel.

## Run locally

```bash
git clone --recursive <this-repo>
cp .env.example .env  # fill in values
uv sync
uv run uvicorn app:app --reload
```

Expose your local port to GitHub with `ngrok http 8000` (or `cloudflared tunnel --url http://localhost:8000`) and point your GitHub App's webhook URL at `<tunnel-url>/webhooks/github`.

If you cloned without `--recursive`: `git submodule update --init --recursive`.

## Deploy

Run `fly launch` (Fly.io reads `Procfile`) or create a new Python service on Render/Railway and point it at this repo. Set the env vars below in the platform's secret store. `Procfile` declares the start command.

## Env vars

| Var | Required | Notes |
|---|---|---|
| `GITHUB_WEBHOOK_SECRET` | yes | Secret you set on the GitHub App's webhook |
| `SLACK_BOT_TOKEN` | yes | `xoxb-…`, scope `chat:write` |
| `SLACK_CHANNEL_ID` | yes | Channel ID, bot must be invited |
| `ANTHROPIC_API_KEY` | yes | Read by the Claude Agent SDK |
| `GITHUB_TOKEN` | yes | PAT with `public_repo` — required by the skill |
| `CIRCLECI_TOKEN` | no | Enables CircleCI log splicing in the review |
| `PORT` | no | Defaults to 8000 |

## Tests

```bash
uv run pytest
```
