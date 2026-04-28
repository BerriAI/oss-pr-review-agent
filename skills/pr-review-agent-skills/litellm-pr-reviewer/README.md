# litellm-pr-reviewer

> Triage a `BerriAI/litellm` pull request and decide whether it's ready for human review.

A self-contained Claude Code skill that classifies each failing check as PR-related vs infra/pre-existing, factors in the Greptile confidence score and CircleCI presence, and emits a thumbs-up/thumbs-down verdict with a 5-item checklist.

## What it does

Given a single PR URL, the skill:

1. Pulls all check-runs and classic statuses on the PR HEAD (paginated — busy repos with 40+ checks were silently truncating before).
2. Bundles the diff (truncated patches), the latest Greptile score, and the same checks' status on the 3 most recently updated other open PRs (for cross-comparison).
3. For each failing check, fetches GitHub annotations and — if `LITELLM_API_BASE` + `LITELLM_API_KEY` are set — splices in the raw CircleCI failure log tail (fetched via the LiteLLM proxy's `circle_ci_mcp-get_build_failure_logs` MCP tool, so no CircleCI token is needed locally).
4. Hands the bundle to the LLM, which decides per-check whether the failure is the PR's fault or pre-existing/infra, and emits a structured verdict.

## Install

### As a Claude Code plugin

Once published to a plugin marketplace (e.g. via the [LiteLLM Skills Gateway](https://docs.litellm.ai/docs/skills_gateway)):

```text
/plugin marketplace add <your-marketplace>
/plugin install litellm-pr-reviewer
```

### Manually as a project skill

```bash
mkdir -p .claude/skills
git clone --depth 1 https://github.com/BerriAI/pr-review-agent-skills /tmp/pras
cp -R /tmp/pras/litellm-pr-reviewer .claude/skills/
pip install httpx
```

## Use

```text
/litellm-review-pr https://github.com/BerriAI/litellm/pull/26455
```

Or just describe what you want — Claude will load the skill automatically when you mention triaging a litellm PR or paste a `github.com/BerriAI/litellm/pull/<N>` URL.

## Required environment

| Variable | Required | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | yes | PAT with `public_repo` scope (or `repo` for private repos). Without it, GitHub's 60 req/hr limit will 403 partway through. |
| `LITELLM_API_BASE` + `LITELLM_API_KEY` | no | LiteLLM proxy URL + virtual key. When BOTH are set, the script calls the proxy's `circle_ci_mcp-get_build_failure_logs` MCP tool to splice raw CircleCI failure log tails into the LLM's view — the proxy holds the CircleCI credential, so no CircleCI token lives in your env. Without these, only GitHub's check-run summaries are used. |

## Layout

```text
litellm-pr-reviewer/
├── SKILL.md                          ← agent instructions (the prompt)
├── README.md                         ← this file
├── .claude-plugin/
│   └── plugin.json                   ← plugin manifest
├── commands/
│   └── litellm-review-pr.md          ← /litellm-review-pr <pr-url>
├── agents/
│   └── litellm-pr-reviewer.md        ← sub-agent definition for delegation
└── scripts/
    └── gather_pr_triage_data.py      ← stdlib + httpx; prints JSON to stdout
```

## How the verdict is computed

`ready` is `true` only when **all** of:

- No PR-related failures (the LLM classifies each failing check as PR-related or unrelated).
- No checks still running.
- Greptile confidence score is `null` (not yet reviewed) **or** ≥ 4/5.
- At least one CircleCI check exists on the PR HEAD.

The 5-item checklist surfaces each gate explicitly so the reviewer can see exactly which one is failing.

## Origin

Extracted from BerriAI's internal PR review agent ([`BerriAI/internal-pr-review-agent`](https://github.com/BerriAI/internal-pr-review-agent)). The `gather_pr_triage_data` data layer was the most reusable piece — same shape works for any agent host (Claude Code, pydantic-ai, raw LLM API), so it lives here as a standalone script.

## License

MIT.
