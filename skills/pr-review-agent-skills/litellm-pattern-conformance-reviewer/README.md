# litellm-pattern-conformance-reviewer

> Decide whether a `BerriAI/litellm` pull request follows the repo's documented and de-facto code patterns.

A self-contained Claude Code skill that pulls patterns from `docs/my-website/docs/` first and from sibling files in the diff's directories second. On conflict, **docs win** — contradicting code goes into `tech_debt[]` so it's noticed without being treated as precedent for accepting a non-conforming diff.

Sibling skill to [`litellm-pr-reviewer`](../litellm-pr-reviewer/) (CI triage). Use this one for "does this look idiomatic", that one for "is this ready to merge".

## What it does

Given a single PR URL, the skill:

1. Pulls the PR's diff (truncated patches per file).
2. Heuristically picks doc excerpts under `docs/my-website/docs/` whose headings reference the diff's paths/topics, and yanks the relevant heading sections.
3. For each changed file, fetches up to 3 sibling files in the same directory (head-of-file excerpt only) as de-facto evidence.
4. Flags conflict candidates where a doc rule and a sibling file appear to disagree.
5. Hands the bundle to the LLM, which classifies each changed file as `conforms` / `violates_docs` / `violates_code_only` / `no_pattern_found` and emits a structured verdict.

## Install

### As a Claude Code plugin

Once published to a plugin marketplace (e.g. via the [LiteLLM Skills Gateway](https://docs.litellm.ai/docs/skills_gateway)):

```text
/plugin marketplace add <your-marketplace>
/plugin install litellm-pattern-conformance-reviewer
```

### Manually as a project skill

```bash
mkdir -p .claude/skills
git clone --depth 1 https://github.com/BerriAI/pr-review-agent-skills /tmp/pras
cp -R /tmp/pras/litellm-pattern-conformance-reviewer .claude/skills/
pip install httpx
```

## Use

Just describe what you want — Claude will load the skill automatically when you ask "does this PR follow our patterns", say "is this idiomatic litellm", or paste a `github.com/BerriAI/litellm/pull/<N>` URL with a pattern/convention question.

```text
Does https://github.com/BerriAI/litellm/pull/26455 follow our patterns?
```

## Required environment

| Variable | Required | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | yes | PAT with `public_repo` scope (or `repo` for private repos). Without it, GitHub's 60 req/hr anonymous quota will 403 partway through fetching docs + siblings. |

## Layout

```text
litellm-pattern-conformance-reviewer/
├── SKILL.md                          ← agent instructions (the prompt)
├── README.md                         ← this file
└── scripts/
    └── gather_pattern_data.py        ← stdlib + httpx; prints JSON to stdout
```

## How the verdict is structured

Each finding cites a source — `docs` (heading under `docs/my-website/docs/`) or `code` (sibling file path) — and is classified `blocker`, `suggestion`, or `nit`:

- `blocker` — the diff contradicts an authoritative doc excerpt.
- `suggestion` — the diff contradicts a de-facto pattern that ≥2 siblings agree on.
- `nit` — only one sibling supports the pattern, or other low-confidence cases.

The verdict has four top-level fields:

- `overview` — two or three short sentences on what the PR does and the overall conformance state.
- `summary` — one-sentence recommendation, picked from a fixed template based on whether any blocker exists.
- `findings` — list of `{file, severity, source, citation, rationale}`. Empty if nothing to flag.
- `tech_debt` — list of `{doc_path, code_path, note}` for places where existing code already contradicts a doc. Informational, not blocking.

See [`SKILL.md`](./SKILL.md) for the full classification rules and output contract.

## Origin

Extracted from BerriAI's internal PR review agent ([`BerriAI/internal-pr-review-agent`](https://github.com/BerriAI/internal-pr-review-agent)), same way as the `litellm-pr-reviewer` skill. The data layer (`gather_pattern_data.py`) is the most reusable piece — same JSON shape works for any agent host (Claude Code, pydantic-ai, raw LLM API).

## License

MIT.
