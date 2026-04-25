#!/usr/bin/env python3
"""Gather everything needed to review a litellm PR for pattern conformance.

Standalone script. Prints a single JSON object to stdout describing the PR's
diff, candidate documentation excerpts (docs/my-website/docs/), and sibling-
file excerpts from the directories the diff touches.

Dependencies: stdlib + httpx. Install with: pip install httpx

Required env:
    GITHUB_TOKEN   - PAT with public_repo scope. Without it, GitHub anonymous
                     quotas (60 req/hr) will exhaust before this finishes for
                     non-trivial PRs.

Usage:
    python gather_pattern_data.py https://github.com/BerriAI/litellm/pull/123
    python gather_pattern_data.py BerriAI/litellm#123

Output JSON shape (stable contract with SKILL.md):

    {
        "owner": str, "repo": str, "pr_number": int,
        "pr_title": str, "head_sha": str,
        "diff_files": [
            {"filename": str, "status": str,
             "additions": int, "deletions": int,
             "patch": str (truncated)}
        ],
        "doc_excerpts": [
            {"path": str, "heading": str, "excerpt": str,
             "matched_files": [str, ...]}
        ],
        "sibling_excerpts": [
            {"diff_file": str,
             "siblings": [{"path": str, "head_excerpt": str}, ...]}
        ],
        "conflict_hints": [
            {"topic": str, "doc_path": str,
             "sibling_path": str, "note": str}
        ]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"
DOCS_ROOT = "docs/my-website/docs"
MAX_PATCH_CHARS = 2000
MAX_DOC_EXCERPT_CHARS = 1500
MAX_SIBLING_HEAD_CHARS = 1200
MAX_SIBLINGS_PER_FILE = 3
MAX_DOC_EXCERPTS_PER_FILE = 3
MAX_DOCS_FETCHED = 30

PR_URL_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<num>\d+)"
)
PR_SHORT_RE = re.compile(r"^(?P<owner>[^/\s]+)/(?P<repo>[^#\s]+)#(?P<num>\d+)$")

# Conflict heuristics: topic name -> regex applied to both doc and sibling text.
# A topic fires when the same regex matches inconsistent values across sources.
# Kept intentionally small; the agent confirms before treating these as real.
_CONFLICT_PATTERNS = {
    "logger_import": re.compile(
        r"(from\s+litellm[._\w]*\s+import\s+verbose_logger|"
        r"import\s+logging\s*$|"
        r"logger\s*=\s*logging\.getLogger)",
        re.MULTILINE,
    ),
    "async_client": re.compile(
        r"(httpx\.AsyncClient|aiohttp\.ClientSession|"
        r"litellm\.module_level_aclient)"
    ),
}


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(url) or PR_SHORT_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a recognised PR reference: {url}")
    return m["owner"], m["repo"], int(m["num"])


# --------------------------------------------------------------------------- #
# GitHub HTTP helpers                                                          #
# --------------------------------------------------------------------------- #


async def _gh(
    client: httpx.AsyncClient,
    token: str | None,
    path: str,
    *,
    accept: str = "application/vnd.github+json",
    **params: Any,
) -> Any:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = await client.get(
        f"{GITHUB_API}{path}",
        params=params or None,
        headers=headers,
        timeout=30.0,
    )
    r.raise_for_status()
    if accept == "application/vnd.github.raw":
        return r.text
    return r.json()


async def _gh_list(
    client: httpx.AsyncClient,
    token: str | None,
    path: str,
    *,
    per_page: int = 100,
    **params: Any,
) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        data = await _gh(
            client, token, path, per_page=per_page, page=page, **params
        )
        if not isinstance(data, list):
            break
        items.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return items


# --------------------------------------------------------------------------- #
# PR + diff                                                                    #
# --------------------------------------------------------------------------- #


async def _fetch_pr(
    client: httpx.AsyncClient, token: str | None, owner: str, repo: str, num: int
) -> dict:
    return await _gh(client, token, f"/repos/{owner}/{repo}/pulls/{num}")


async def _fetch_diff_files(
    client: httpx.AsyncClient, token: str | None, owner: str, repo: str, num: int
) -> list[dict]:
    raw = await _gh_list(client, token, f"/repos/{owner}/{repo}/pulls/{num}/files")
    out: list[dict] = []
    for f in raw:
        patch = f.get("patch") or ""
        if len(patch) > MAX_PATCH_CHARS:
            patch = patch[:MAX_PATCH_CHARS] + "\n... [truncated]"
        out.append(
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": patch,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Docs: search docs/my-website/docs for filenames and module symbols           #
# --------------------------------------------------------------------------- #


def _keywords_for_file(filename: str) -> list[str]:
    """Pull plausible search keywords from a changed file's path.

    Strategy: stem of the filename + each non-trivial path segment. Skips
    generic dirs ('src', 'lib', 'tests') and stopword stems ('utils', 'init').
    """
    skip_dirs = {"src", "lib", "tests", "test", "litellm", "__init__"}
    skip_stems = {"utils", "init", "main", "base", "types", "constants"}
    parts = [p for p in re.split(r"[\\/]", filename) if p]
    if not parts:
        return []
    stem = parts[-1].rsplit(".", 1)[0]
    keywords: list[str] = []
    if stem and stem not in skip_stems:
        keywords.append(stem)
    for p in parts[:-1]:
        if p and p not in skip_dirs and p not in keywords:
            keywords.append(p)
    return keywords[:4]


async def _search_docs_for_keyword(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    keyword: str,
    head_sha: str,
) -> list[str]:
    """Use GitHub's code search to find docs mentioning `keyword`.

    Limited to docs/my-website/docs/ and markdown files. Returns a list of
    unique paths (capped). Code search is best-effort: errors -> empty list.
    """
    q = (
        f'"{keyword}" repo:{owner}/{repo} '
        f'path:{DOCS_ROOT} extension:md'
    )
    try:
        data = await _gh(
            client,
            token,
            "/search/code",
            q=q,
            per_page=5,
        )
    except httpx.HTTPStatusError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    paths: list[str] = []
    for it in items:
        p = it.get("path")
        if p and p not in paths:
            paths.append(p)
    return paths


async def _fetch_doc_file(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> str | None:
    try:
        return await _gh(
            client,
            token,
            f"/repos/{owner}/{repo}/contents/{path}",
            accept="application/vnd.github.raw",
            ref=ref,
        )
    except httpx.HTTPStatusError:
        return None


def _excerpt_around(text: str, keyword: str) -> tuple[str, str]:
    """Return (heading, excerpt) for the section in `text` that mentions
    `keyword`. Heading is the nearest preceding markdown heading. Excerpt is
    bounded to MAX_DOC_EXCERPT_CHARS centred on the first match.
    """
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return ("", text[:MAX_DOC_EXCERPT_CHARS])
    half = MAX_DOC_EXCERPT_CHARS // 2
    start = max(0, idx - half)
    end = min(len(text), idx + half)
    excerpt = text[start:end]
    heading = ""
    for line in text[:idx].splitlines()[::-1]:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            heading = m.group(2).strip()
            break
    return (heading, excerpt)


async def _gather_doc_excerpts(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    head_sha: str,
    diff_files: list[dict],
) -> list[dict]:
    """For each diff file, look up keywords in the docs and return excerpts.

    Caches doc fetches across diff files so a doc that matches multiple
    changed files is only downloaded once.
    """
    doc_cache: dict[str, str] = {}
    fetched_count = 0
    excerpts: list[dict] = []

    for f in diff_files:
        filename = f["filename"]
        if not filename:
            continue
        keywords = _keywords_for_file(filename)
        if not keywords:
            continue
        per_file: list[dict] = []
        for kw in keywords:
            paths = await _search_docs_for_keyword(
                client, token, owner, repo, kw, head_sha
            )
            for path in paths:
                if fetched_count >= MAX_DOCS_FETCHED and path not in doc_cache:
                    continue
                if path not in doc_cache:
                    body = await _fetch_doc_file(
                        client, token, owner, repo, path, head_sha
                    )
                    if body is None:
                        continue
                    doc_cache[path] = body
                    fetched_count += 1
                heading, excerpt = _excerpt_around(doc_cache[path], kw)
                per_file.append(
                    {
                        "path": path,
                        "heading": heading,
                        "excerpt": excerpt,
                        "matched_files": [filename],
                    }
                )
                if len(per_file) >= MAX_DOC_EXCERPTS_PER_FILE:
                    break
            if len(per_file) >= MAX_DOC_EXCERPTS_PER_FILE:
                break
        excerpts.extend(per_file)

    merged: dict[tuple[str, str], dict] = {}
    for e in excerpts:
        key = (e["path"], e["heading"])
        if key in merged:
            for mf in e["matched_files"]:
                if mf not in merged[key]["matched_files"]:
                    merged[key]["matched_files"].append(mf)
        else:
            merged[key] = e
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Siblings: list directory contents per changed file                           #
# --------------------------------------------------------------------------- #


async def _list_dir(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    dir_path: str,
    ref: str,
) -> list[dict]:
    contents_path = f"/repos/{owner}/{repo}/contents"
    if dir_path:
        contents_path = f"{contents_path}/{dir_path}"
    try:
        data = await _gh(client, token, contents_path, ref=ref)
    except httpx.HTTPStatusError:
        return []
    return data if isinstance(data, list) else []


async def _gather_sibling_excerpts(
    client: httpx.AsyncClient,
    token: str | None,
    owner: str,
    repo: str,
    head_sha: str,
    diff_files: list[dict],
) -> list[dict]:
    """For each changed file, fetch up to MAX_SIBLINGS_PER_FILE siblings."""
    seen_files: dict[str, str] = {}
    out: list[dict] = []

    for f in diff_files:
        filename = f["filename"]
        if not filename:
            continue
        parts = filename.rsplit("/", 1)
        dir_path = parts[0] if len(parts) == 2 else ""
        listing = await _list_dir(client, token, owner, repo, dir_path, head_sha)
        siblings: list[dict] = []
        for entry in listing:
            if entry.get("type") != "file":
                continue
            sib_path = entry.get("path")
            if not sib_path or sib_path == filename:
                continue
            if not re.search(r"\.(py|ts|tsx|js|jsx|go|rs|java)$", sib_path):
                continue
            if sib_path not in seen_files:
                body = await _fetch_doc_file(
                    client, token, owner, repo, sib_path, head_sha
                )
                if body is None:
                    continue
                seen_files[sib_path] = body[:MAX_SIBLING_HEAD_CHARS]
            siblings.append(
                {"path": sib_path, "head_excerpt": seen_files[sib_path]}
            )
            if len(siblings) >= MAX_SIBLINGS_PER_FILE:
                break
        if siblings:
            out.append({"diff_file": filename, "siblings": siblings})
    return out


# --------------------------------------------------------------------------- #
# Conflict hints                                                               #
# --------------------------------------------------------------------------- #


def _find_conflict_hints(
    doc_excerpts: list[dict], sibling_excerpts: list[dict]
) -> list[dict]:
    """Surface candidate doc-vs-code disagreements via shared-topic regexes.

    A hint fires when the same topic regex matches different concrete values
    in a doc excerpt and at least one sibling excerpt. The agent decides
    whether the hint is real.
    """
    hints: list[dict] = []
    for topic, pat in _CONFLICT_PATTERNS.items():
        doc_hits: list[tuple[str, str]] = []
        for d in doc_excerpts:
            for m in pat.findall(d["excerpt"]):
                val = m if isinstance(m, str) else next((x for x in m if x), "")
                if val:
                    doc_hits.append((d["path"], val))
        if not doc_hits:
            continue
        for sg in sibling_excerpts:
            for sib in sg["siblings"]:
                for m in pat.findall(sib["head_excerpt"]):
                    val = m if isinstance(m, str) else next((x for x in m if x), "")
                    if not val:
                        continue
                    for doc_path, doc_val in doc_hits:
                        d = doc_val.strip()
                        s = val.strip()
                        if d == s:
                            continue
                        hints.append(
                            {
                                "topic": topic,
                                "doc_path": doc_path,
                                "sibling_path": sib["path"],
                                "note": f"doc shows `{d}`, sibling uses `{s}`",
                            }
                        )
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for h in hints:
        k = (h["topic"], h["doc_path"], h["sibling_path"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(h)
    return deduped


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


async def gather(pr_ref: str) -> dict:
    owner, repo, num = parse_pr_url(pr_ref)
    token = os.environ.get("GITHUB_TOKEN") or None

    async with httpx.AsyncClient() as client:
        pr = await _fetch_pr(client, token, owner, repo, num)
        head_sha = pr["head"]["sha"]
        diff_files = await _fetch_diff_files(client, token, owner, repo, num)
        doc_excerpts, sibling_excerpts = await asyncio.gather(
            _gather_doc_excerpts(
                client, token, owner, repo, head_sha, diff_files
            ),
            _gather_sibling_excerpts(
                client, token, owner, repo, head_sha, diff_files
            ),
        )
        conflict_hints = _find_conflict_hints(doc_excerpts, sibling_excerpts)

    return {
        "owner": owner,
        "repo": repo,
        "pr_number": num,
        "pr_title": pr.get("title", ""),
        "head_sha": head_sha,
        "diff_files": diff_files,
        "doc_excerpts": doc_excerpts,
        "sibling_excerpts": sibling_excerpts,
        "conflict_hints": conflict_hints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pr_ref",
        help=(
            "PR URL (https://github.com/owner/repo/pull/N) "
            "or short ref (owner/repo#N)"
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "warning: GITHUB_TOKEN not set; GitHub anonymous quotas will likely "
            "exhaust mid-run.",
            file=sys.stderr,
        )

    try:
        result = asyncio.run(gather(args.pr_ref))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    except httpx.HTTPStatusError as e:
        print(
            f"GitHub HTTP {e.response.status_code} for {e.request.url}: "
            f"{e.response.text[:300]}",
            file=sys.stderr,
        )
        return 1

    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
