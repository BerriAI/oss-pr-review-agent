#!/usr/bin/env python3
"""Run Claude Code + litellm-karpathy-check against PRs in an eval JSON.

Requires: `claude` on PATH, `gh` with GITHUB_TOKEN, LITELLM_CLONE (default
~/Documents/litellm), ANTHROPIC_* or ~/.claude/settings.json for CC auth.

Usage:
  uv run python scripts/run_karpathy_check_eval.py tests/eval/pr_set_shin_regressions.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILL = REPO_ROOT / "skills/pr-review-agent-skills/litellm-karpathy-check/SKILL.md"
PR_URL_RE = re.compile(r"github\.com/[\w.-]+/[\w.-]+/pull/(\d+)")


def _pr_number(url: str) -> int:
    m = PR_URL_RE.search(url)
    if not m:
        raise ValueError(f"not a github PR url: {url}")
    return int(m.group(1))


def _linked_issue_numbers(body: str) -> list[int]:
    out: list[int] = []
    for m in re.finditer(
        r"(?:fixes|closes|resolves)\s+#(\d+)", body, flags=re.I
    ):
        out.append(int(m.group(1)))
    for m in re.finditer(
        r"(?:fixes|closes|resolves)\s+https?://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)",
        body,
        flags=re.I,
    ):
        out.append(int(m.group(1)))
    return sorted(set(out))


def _gh_json(args: list[str]) -> dict:
    r = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def _extract_json_line(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main() -> int:
    eval_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO_ROOT / "tests/eval/pr_set_shin_regressions.json"
    )
    skill_path = Path(
        os.environ.get("KARPATHY_CHECK_SKILL", str(DEFAULT_SKILL))
    )
    _lc = os.environ.get("LITELLM_CLONE", "").strip()
    litellm = (
        Path(_lc).expanduser()
        if _lc
        else Path("/Users/krrishdholakia/Documents/litellm")
    )
    if not litellm.is_dir():
        print(f"LITELLM_CLONE not a directory: {litellm}", file=sys.stderr)
        return 2

    data = json.loads(eval_path.read_text())
    prs = data.get("prs") or []
    skill_body = skill_path.read_text()
    out_dir = Path(
        os.environ.get(
            "KARPATHY_CHECK_OUT",
            str(REPO_ROOT / "runs" / "karpathy_check"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in prs:
        url = entry["url"]
        pr = _pr_number(url)
        print(f"\n=== PR {pr} ===", flush=True)

        pr_data = _gh_json(
            [
                "pr",
                "view",
                str(pr),
                "--repo",
                "BerriAI/litellm",
                "--json",
                "headRefOid,title,body,files",
            ]
        )
        body = pr_data.get("body") or ""
        linked: list[dict] = []
        for num in _linked_issue_numbers(body):
            try:
                issue = _gh_json(
                    [
                        "issue",
                        "view",
                        str(num),
                        "--repo",
                        "BerriAI/litellm",
                        "--json",
                        "number,title,body",
                    ]
                )
                linked.append(
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "body": (issue.get("body") or "")[:4000],
                    }
                )
            except subprocess.CalledProcessError as e:
                print(f"  warn: could not fetch issue #{num}: {e}", flush=True)

        head_sha = pr_data["headRefOid"]
        wt = out_dir / f"wt-pr-{pr}"
        if wt.exists():
            subprocess.run(
                ["git", "-C", str(litellm), "worktree", "remove", "--force", str(wt)],
                capture_output=True,
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(litellm),
                "fetch",
                "origin",
                f"pull/{pr}/head",
            ],
            check=True,
            capture_output=True,
        )
        fetch_head = subprocess.run(
            ["git", "-C", str(litellm), "rev-parse", "FETCH_HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if fetch_head != head_sha:
            print(
                f"  warn: FETCH_HEAD {fetch_head[:12]} != gh head {head_sha[:12]}",
                flush=True,
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(litellm),
                "worktree",
                "add",
                "--detach",
                str(wt),
                head_sha,
            ],
            check=True,
            capture_output=True,
        )

        payload = {
            "pr_url": url,
            "head_sha": head_sha,
            "pr_title": pr_data.get("title") or "",
            "pr_body": body[:4000],
            "diff_files": [
                {
                    "path": f["path"],
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                }
                for f in pr_data.get("files") or []
            ],
            "linked_issues": linked,
            "repo_path": str(wt.resolve()),
        }
        input_json = json.dumps(payload, indent=2)
        prompt = f"""You are a litellm-karpathy-check. Follow the skill below verbatim.
The "Inputs" JSON is provided directly in this prompt (no stdin).
Read files only via Read/Grep/Glob/Bash within the repo_path.
Print your final JSON report as the LAST line of your reply (single line JSON matching the skill schema).

=== SKILL BEGIN ===
{skill_body}
=== SKILL END ===

=== INPUT JSON BEGIN ===
{input_json}
=== INPUT JSON END ===
"""

        proc = subprocess.run(
            [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--add-dir",
                str(wt.resolve()),
                "--allowedTools",
                "Read,Grep,Glob,Bash",
                "--permission-mode",
                "bypassPermissions",
                "--max-budget-usd",
                os.environ.get("KARPATHY_CHECK_MAX_USD", "2.00"),
                "-",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("KARPATHY_CHECK_TIMEOUT_SEC", "600")),
        )

        raw = (proc.stdout or "") + (proc.stderr or "")
        transcript_path = out_dir / f"pr_{pr}_transcript.json"
        transcript_path.write_text(proc.stdout or "")

        parsed_envelope: dict | None = None
        if proc.stdout:
            try:
                parsed_envelope = json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass

        result_text = ""
        if isinstance(parsed_envelope, dict):
            result_text = str(parsed_envelope.get("result") or "")
        report_obj = _extract_json_line(result_text) or _extract_json_line(raw)

        summary_path = out_dir / f"pr_{pr}_karpathy.json"
        if report_obj is not None:
            summary_path.write_text(json.dumps(report_obj, indent=2))
            mg = report_obj.get("merge_gate") or {}
            print(
                f"  merge_gate: {mg.get('safe_for_high_rps_gateway')} — {mg.get('one_liner', '')[:120]}",
                flush=True,
            )
            for f in report_obj.get("findings") or []:
                print(f"  finding: {f.get('breadth')} — {f.get('bug_class', '')[:80]}", flush=True)
        else:
            print(f"  FAILED parse (exit {proc.returncode})", flush=True)
            print(raw[:2000], flush=True)

        subprocess.run(
            ["git", "-C", str(litellm), "worktree", "remove", "--force", str(wt)],
            capture_output=True,
        )

    print(f"\nArtifacts under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
