# PR-review eval

Live end-to-end eval for the litellm-bot triage + pattern agents against
a curated set of 20 real BerriAI/litellm PRs.

## What this is (and isn't)

- **Is**: a way to spot-check whether the agents still produce sensible
  cards after a prompt, skill, rubric, or model change. Outputs land in
  a markdown table you grade by eye.
- **Isn't**: a pass/fail regression suite. Subjective output (verdict
  one-liners, justifications, severity calls) doesn't have a correct
  answer the harness can check. Pin invariants (score range, verdict
  enum, schema shape) in `tests/test_fuse.py` and `test_capabilities.py`,
  not here.

## Quick start

```bash
# 1. Make sure .env has real LITELLM_API_KEY + GITHUB_TOKEN
cp .env.example .env  # if you haven't already
# edit .env

# 2. Run the eval (5–10 min wall, ~$0.50 in LLM calls)
uv run python -m tests.eval.run_eval

# Or smoke-test the harness against just the first PR first:
uv run python -m tests.eval.run_eval --limit 1

# 3. Read the report
open tests/eval/results/<timestamp>/summary.md
```

Or via pytest, gated on the `eval` marker (deselected by default in
`pyproject.toml`'s `addopts` so it never fires from a plain `pytest`):

```bash
uv run pytest -m eval -s
```

## What gets written

Each run creates `tests/eval/results/<UTC-timestamp>/`:

- `results.json` — full per-PR record: TriageReport, PatternReport,
  fused TriageCard, latency, and any errors. Schema-versioned at the
  top so downstream tooling can detect drift.
- `summary.md` — markdown report with aggregate stats and a per-PR
  table sorted by category, plus a Failures section. This is what you
  read; `results.json` is what you query if you want to dig in.

The `results/` directory is gitignored — eval runs are personal artifacts,
not source-controlled.

## The PR set

Lives in `pr_set.json`. 20 PRs spanning:

- merge-conflict cases (the canonical ones cited in `tests/test_fuse.py`)
- `mergeable=None` (GitHub still computing) — must NOT block
- clean / blocked / unstable open PRs — exercises every `mergeable_state`
- net-new provider integrations (Mavvrik, Cycraft) — pattern-reviewer
  cold-start, no obvious sibling files
- multi-component features (per-team budgets, MCP self-service)
- merged baseline cases (small fixes, UI-only, security hardening)
- two dependabot bumps in a row — built-in consistency check

To change the set, edit `pr_set.json`. Each entry has `url`, `category`,
and `notes` (the notes show up in the summary table so the grader sees
why the PR was picked).

## Tuning

- `EVAL_CONCURRENCY=4` — how many PRs run in parallel. Higher is faster
  but more likely to trip GitHub or LLM rate limits.
- `LITELLM_MODEL` — overrides the default model for the run. Set to
  compare two models side-by-side: run twice, diff the two summary.md.

## Interpreting results

Things to scan for in `summary.md`:

1. **Errors**: any non-zero error count is a real signal — the harness
   only errors when the agent itself crashes or returns invalid output.
2. **Conflict PRs (#26451, #25609)**: should both be `BLOCKED 0/5`. If
   one is `READY` something regressed in the conflict rubric row.
3. **Deps bumps (#25662, #26188)**: should look near-identical. Big
   divergence between two near-identical PRs usually means the agent
   is leaning on irrelevant context.
4. **`mergeable=None` PR (#26505)**: should NOT be docked for conflicts.
   `has_merge_conflicts` should be `null` in `results.json`.
5. **Latency p95**: if it climbs past ~90s/PR, the model is probably
   thrashing tool calls — worth a Logfire dive.
