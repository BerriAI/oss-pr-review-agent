---
name: litellm-review-pr
description: Triage a BerriAI/litellm PR. Pass the PR URL or owner/repo#N as the argument.
argument-hint: [pr-url-or-ref]
---

Triage the litellm PR `$ARGUMENTS` using the `litellm-pr-reviewer` skill. Follow the skill's 6-step workflow exactly: gather data via the bundled script (one call), classify each failing check, pick the overall status, set the ready flag, emit the 5-item checklist, and write the verdict (overview / summary / details / file_callouts).

If `$ARGUMENTS` is empty, ask the user for a PR URL or `owner/repo#N` reference and stop.
