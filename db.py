"""Async Postgres access for litellm-bot.

Owns the asyncpg connection pool, the schema bootstrap (`init_db()`), and
the typed read/write helpers every other module uses. Two design choices
worth flagging:

1. JSONB round-trips. We register asyncpg JSONB codecs at pool acquire
   time so `triage`, `pattern`, `card`, `messages`, `tool_trace` are
   plain Python `dict` / `list` on read and write — no manual
   `json.dumps` / `json.loads` sprinkled at every callsite.

2. Fail-fast on missing DATABASE_URL. The earlier JSONL design had a
   "fall back to local files" branch; we removed it so there is exactly
   one source of truth in prod and dev. A contributor without the Neon
   URL set will see a clear error at startup, not silently get a
   degraded UI half a screen later.

Schema lives in migrations/. `init_db()` runs the bootstrap script
unconditionally on startup; it's all `IF NOT EXISTS` so re-running
against an already-set-up DB is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import asyncpg

log = logging.getLogger("litellm-bot.db")

_pool: asyncpg.Pool | None = None
SCHEMA_PATH = Path(__file__).parent / "migrations" / "0001_init.sql"


# --- Connection lifecycle -----------------------------------------------------


async def init_db() -> asyncpg.Pool:
    """Create the pool, register JSONB codecs, run the schema bootstrap.

    Idempotent at the schema level (every CREATE in the bootstrap is
    IF NOT EXISTS). Safe to call from FastAPI startup; subsequent calls
    return the existing pool.
    """
    global _pool
    if _pool is not None:
        return _pool
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is required. Set it in .env (Neon / local Postgres) "
            "or unset the import path that pulls in db.py."
        )
    # Neon-friendly defaults: small min_size so we don't hold idle conns
    # against their connection cap; statement_cache_size=0 because Neon's
    # pooler (PgBouncer transaction mode) doesn't preserve prepared
    # statements across reconnects.
    _pool = await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=10,
        command_timeout=30,
        statement_cache_size=0,
        init=_register_codecs,
    )
    await _bootstrap_schema(_pool)
    log.info("db_pool_ready dsn_host=%s", _redact_host(dsn))
    return _pool


async def _register_codecs(conn: asyncpg.Connection) -> None:
    """Tell asyncpg to give us Python dict/list for JSONB columns instead
    of raw JSON strings. Saves a `json.loads` at every read site."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def _bootstrap_schema(pool: asyncpg.Pool) -> None:
    # Run every *.sql under migrations/ in lexical order. Filenames are
    # numbered (0001_init.sql, 0002_*.sql, ...) so sorted() == apply
    # order. Each script is independently idempotent (IF NOT EXISTS on
    # every CREATE / ALTER ... ADD COLUMN), so re-running on an already-
    # bootstrapped DB is a no-op.
    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))
    async with pool.acquire() as conn:
        for path in migration_files:
            await conn.execute(path.read_text())


def _to_dt(ts: float | datetime | None) -> datetime | None:
    """Normalize epoch floats to UTC datetimes for asyncpg's timestamptz
    codec. None / datetime pass through unchanged.

    Why this lives here vs at every callsite: the JSONL records, the
    Logfire traces, and review_pr() all carry timestamps as epoch floats,
    so converting at the DB boundary keeps the rest of the code
    epoch-native. asyncpg won't accept a float for timestamptz, so we'd
    otherwise need this exact dance at every insert.
    """
    if ts is None or isinstance(ts, datetime):
        return ts
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _redact_host(dsn: str) -> str:
    # Quick non-secret-leaking host extraction for logs. We don't try to
    # be a full URL parser — just enough to confirm "yes we're hitting
    # Neon" vs "wait, why did this point at localhost".
    try:
        after_at = dsn.split("@", 1)[1]
        host = after_at.split("/", 1)[0].split("?", 1)[0]
        return host
    except Exception:
        return "?"


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    """Synchronous accessor for the pool. Raises if init_db() hasn't run.

    Most callers should just `from db import fetch_runs` etc. — this is
    the escape hatch for bespoke queries (e.g. tests / migration scripts)
    that need a raw connection.
    """
    if _pool is None:
        raise RuntimeError("db.init_db() has not been called yet")
    return _pool


# --- Run helpers --------------------------------------------------------------
# Every helper takes the pool implicitly (via pool()) and returns plain
# dicts so callers can hand them straight to Pydantic / FastAPI without
# adapter code in between.


async def insert_run(record: dict) -> None:
    """Persist one PR-review run. Caller is responsible for the run_id
    (typically the Logfire trace id so the two systems stay aligned).

    Re-inserting the same run_id is an UPSERT — overwrites the agent
    payloads but preserves any human_label/human_notes already set, so a
    re-run from /chat doesn't blow away earlier human grading.
    """
    sql = """
    INSERT INTO runs (
        run_id, ts, pr_url, pr_number, pr_title, pr_author,
        source, channel, thread_ts, duration_s, logfire_trace_id,
        model_name, tokens_in, tokens_out, cost_usd,
        triage, pattern, card, tool_trace, messages,
        human_label, human_notes, karpathy_check
    ) VALUES (
        $1, COALESCE($2, NOW()), $3, $4, $5, $6,
        $7, $8, $9, $10, $11,
        $12, $13, $14, $15,
        $16, $17, $18, $19, $20,
        $21, $22, $23
    )
    ON CONFLICT (run_id) DO UPDATE SET
        ts = EXCLUDED.ts,
        pr_url = EXCLUDED.pr_url,
        pr_number = EXCLUDED.pr_number,
        pr_title = EXCLUDED.pr_title,
        pr_author = EXCLUDED.pr_author,
        source = EXCLUDED.source,
        channel = EXCLUDED.channel,
        thread_ts = EXCLUDED.thread_ts,
        duration_s = EXCLUDED.duration_s,
        logfire_trace_id = EXCLUDED.logfire_trace_id,
        model_name = EXCLUDED.model_name,
        tokens_in = EXCLUDED.tokens_in,
        tokens_out = EXCLUDED.tokens_out,
        cost_usd = EXCLUDED.cost_usd,
        triage = EXCLUDED.triage,
        pattern = EXCLUDED.pattern,
        card = EXCLUDED.card,
        tool_trace = EXCLUDED.tool_trace,
        messages = EXCLUDED.messages,
        karpathy_check = EXCLUDED.karpathy_check
        -- intentionally NOT touching human_label / human_notes here; those
        -- are the human grader's territory and live through agent re-runs.
    """
    async with pool().acquire() as conn:
        await conn.execute(
            sql,
            record["run_id"],
            _to_dt(record.get("ts")),  # epoch float / None → tz-aware datetime / None
            record["pr_url"],
            record.get("pr_number"),
            record.get("pr_title"),
            record.get("pr_author"),
            record.get("source", "slack"),
            record.get("channel"),
            record.get("thread_ts"),
            record.get("duration_s"),
            record.get("logfire_trace_id"),
            record.get("model_name"),
            record.get("tokens_in"),
            record.get("tokens_out"),
            record.get("cost_usd"),
            record.get("triage") or {},
            record.get("pattern") or {},
            record.get("card") or {},
            record.get("tool_trace") or [],
            record.get("messages") or {"triage": [], "pattern": []},
            record.get("human_label"),
            record.get("human_notes"),
            record.get("karpathy_check") or {},
        )


async def list_runs_summary(
    *,
    source: str | None = None,
    label_state: str | None = None,
    since_epoch: float | None = None,
    limit: int = 200,
) -> list[dict]:
    """Sidebar list. Returns trimmed records — no JSONB blobs — so the
    page loads fast even with thousands of runs in the table.

    Filters:
      - source: 'slack' | 'chat' | 'eval' | 'mock' (exact match)
      - label_state: 'labeled' | 'unlabeled' | 'ready' | 'not_ready'
      - since_epoch: only runs newer than this UNIX timestamp
    """
    where: list[str] = []
    args: list[Any] = []

    def add(clause: str, value: Any) -> None:
        args.append(value)
        where.append(clause.format(n=len(args)))

    if source:
        add("source = ${n}", source)
    if since_epoch is not None:
        add("ts >= to_timestamp(${n})", since_epoch)
    if label_state == "labeled":
        where.append("human_label IS NOT NULL")
    elif label_state == "unlabeled":
        where.append("human_label IS NULL")
    elif label_state in ("ready", "not_ready"):
        add("human_label = ${n}", label_state)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    args.append(limit)
    sql = f"""
    SELECT
        run_id,
        EXTRACT(EPOCH FROM ts)::float8 AS ts_epoch,
        pr_url, pr_number, pr_title, pr_author,
        source, duration_s, cost_usd, human_label,
        card->>'score' AS score_str,
        card->>'verdict' AS verdict,
        card->>'emoji' AS emoji
    FROM runs
    {where_sql}
    ORDER BY ts DESC
    LIMIT ${len(args)}
    """
    async with pool().acquire() as conn:
        rows = await conn.fetch(sql, *args)
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # `card->>'score'` returns text; coerce to int with a safe
        # fallback so a malformed payload doesn't 500 the list endpoint.
        try:
            d["score"] = int(d.pop("score_str") or 0)
        except (ValueError, TypeError):
            d["score"] = 0
        d["ts"] = d.pop("ts_epoch")
        out.append(d)
    return out


async def get_run(run_id: str) -> dict | None:
    sql = """
    SELECT
        run_id,
        EXTRACT(EPOCH FROM ts)::float8 AS ts,
        pr_url, pr_number, pr_title, pr_author,
        source, channel, thread_ts, duration_s, logfire_trace_id,
        model_name, tokens_in, tokens_out, cost_usd,
        triage, pattern, card, tool_trace, messages,
        human_label, human_notes, karpathy_check
    FROM runs
    WHERE run_id = $1
    """
    async with pool().acquire() as conn:
        row = await conn.fetchrow(sql, run_id)
    return dict(row) if row else None


async def add_annotation(run_id: str, label: str | None, notes: str | None) -> dict:
    """Append one annotation row + update the denormalized run row.

    Both writes happen in a single transaction so a crash between them
    can't leave the list view (which reads runs.human_label) out of sync
    with the audit trail (annotations.*).
    """
    async with pool().acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO annotations (run_id, human_label, human_notes)
                VALUES ($1, $2, $3)
                RETURNING id, EXTRACT(EPOCH FROM created_at)::float8 AS created_at
                """,
                run_id,
                label,
                notes,
            )
            updated = await conn.fetchval(
                """
                UPDATE runs
                SET human_label = $2, human_notes = $3
                WHERE run_id = $1
                RETURNING run_id
                """,
                run_id,
                label,
                notes,
            )
            if not updated:
                raise LookupError(f"run not found: {run_id}")
    return {
        "id": row["id"],
        "run_id": run_id,
        "human_label": label,
        "human_notes": notes,
        "created_at": row["created_at"],
    }


async def list_annotations(run_id: str) -> list[dict]:
    sql = """
    SELECT id, run_id, human_label, human_notes,
           EXTRACT(EPOCH FROM created_at)::float8 AS created_at
    FROM annotations
    WHERE run_id = $1
    ORDER BY created_at DESC
    """
    async with pool().acquire() as conn:
        rows = await conn.fetch(sql, run_id)
    return [dict(r) for r in rows]


async def stream_runs_for_export(
    *,
    label_state: str | None = None,
    source: str | None = None,
    since_epoch: float | None = None,
) -> AsyncIterator[dict]:
    """Yield full run records (with messages) for the bulk-export route.

    Used by /api/v1/runs/export. Returns an async generator instead of a
    list so a 10k-row export doesn't materialize in memory; FastAPI
    streams each yielded JSONL line straight to the client.
    """
    where: list[str] = []
    args: list[Any] = []

    def add(clause: str, value: Any) -> None:
        args.append(value)
        where.append(clause.format(n=len(args)))

    if source:
        add("source = ${n}", source)
    if since_epoch is not None:
        add("ts >= to_timestamp(${n})", since_epoch)
    if label_state == "labeled":
        where.append("human_label IS NOT NULL")
    elif label_state == "unlabeled":
        where.append("human_label IS NULL")
    elif label_state in ("ready", "not_ready"):
        add("human_label = ${n}", label_state)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    SELECT
        run_id,
        EXTRACT(EPOCH FROM ts)::float8 AS ts,
        pr_url, pr_number, pr_title, pr_author,
        source, channel, thread_ts, duration_s, logfire_trace_id,
        model_name, tokens_in, tokens_out, cost_usd,
        triage, pattern, card, tool_trace, messages,
        human_label, human_notes, karpathy_check
    FROM runs
    {where_sql}
    ORDER BY ts DESC
    """
    async with pool().acquire() as conn:
        async with conn.transaction():
            async for row in conn.cursor(sql, *args):
                yield dict(row)


# --- Eval-set helpers ---------------------------------------------------------


async def list_eval_sets() -> list[dict]:
    """All distinct set_names with their row counts. Drives the
    set-picker dropdown in the future eval UI / `run_eval.py --set ...`.
    """
    sql = """
    SELECT
        set_name,
        COUNT(*) AS pr_count,
        SUM(CASE WHEN human_label IS NOT NULL THEN 1 ELSE 0 END) AS labeled_count,
        MAX(updated_at) AS updated_at
    FROM eval_prs
    GROUP BY set_name
    ORDER BY set_name
    """
    async with pool().acquire() as conn:
        rows = await conn.fetch(sql)
    return [
        {
            "set_name": r["set_name"],
            "pr_count": int(r["pr_count"]),
            "labeled_count": int(r["labeled_count"] or 0),
            "updated_at": r["updated_at"].timestamp() if r["updated_at"] else None,
        }
        for r in rows
    ]


async def list_eval_prs(set_name: str) -> list[dict]:
    sql = """
    SELECT id, url, repo, set_name, category, notes,
           human_label, human_notes, source_run_id,
           EXTRACT(EPOCH FROM created_at)::float8 AS created_at,
           EXTRACT(EPOCH FROM updated_at)::float8 AS updated_at
    FROM eval_prs
    WHERE set_name = $1
    ORDER BY id
    """
    async with pool().acquire() as conn:
        rows = await conn.fetch(sql, set_name)
    return [dict(r) for r in rows]


async def upsert_eval_pr(
    *,
    url: str,
    set_name: str,
    repo: str = "BerriAI/litellm",
    category: str | None = None,
    notes: str | None = None,
    human_label: str | None = None,
    human_notes: str | None = None,
    source_run_id: str | None = None,
) -> dict:
    """Insert or update one PR in an eval set, keyed on (url, set_name).

    Used by the migration script (idempotent re-bootstrap), the graduate
    flow, and the eval-set CRUD API. Returns the resulting row.
    """
    sql = """
    INSERT INTO eval_prs (
        url, repo, set_name, category, notes,
        human_label, human_notes, source_run_id, updated_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
    ON CONFLICT (url, set_name) DO UPDATE SET
        repo = EXCLUDED.repo,
        category = COALESCE(EXCLUDED.category, eval_prs.category),
        notes = COALESCE(EXCLUDED.notes, eval_prs.notes),
        human_label = COALESCE(EXCLUDED.human_label, eval_prs.human_label),
        human_notes = COALESCE(EXCLUDED.human_notes, eval_prs.human_notes),
        source_run_id = COALESCE(EXCLUDED.source_run_id, eval_prs.source_run_id),
        updated_at = NOW()
    RETURNING id, url, repo, set_name, category, notes,
              human_label, human_notes, source_run_id,
              EXTRACT(EPOCH FROM created_at)::float8 AS created_at,
              EXTRACT(EPOCH FROM updated_at)::float8 AS updated_at
    """
    async with pool().acquire() as conn:
        row = await conn.fetchrow(
            sql,
            url,
            repo,
            set_name,
            category,
            notes,
            human_label,
            human_notes,
            source_run_id,
        )
    return dict(row)


async def delete_eval_pr(set_name: str, pr_id: int) -> bool:
    async with pool().acquire() as conn:
        result = await conn.execute(
            "DELETE FROM eval_prs WHERE set_name = $1 AND id = $2",
            set_name,
            pr_id,
        )
    # asyncpg returns "DELETE 1" / "DELETE 0"
    return result.endswith("1")


async def update_eval_pr(
    set_name: str,
    pr_id: int,
    *,
    category: str | None = None,
    notes: str | None = None,
    human_label: str | None = None,
    human_notes: str | None = None,
) -> dict | None:
    """Partial update. Anything passed as None is left untouched (i.e.
    NULL in the request body means "don't change", not "set to NULL").
    Use the dedicated unlabel flow if you want to clear a label.
    """
    sets: list[str] = []
    args: list[Any] = []

    def add(field: str, value: Any) -> None:
        args.append(value)
        sets.append(f"{field} = ${len(args)}")

    if category is not None:
        add("category", category)
    if notes is not None:
        add("notes", notes)
    if human_label is not None:
        add("human_label", human_label)
    if human_notes is not None:
        add("human_notes", human_notes)
    if not sets:
        # Nothing to update — just read the current row back.
        async with pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, url, repo, set_name, category, notes,
                       human_label, human_notes, source_run_id,
                       EXTRACT(EPOCH FROM created_at)::float8 AS created_at,
                       EXTRACT(EPOCH FROM updated_at)::float8 AS updated_at
                FROM eval_prs
                WHERE set_name = $1 AND id = $2
                """,
                set_name,
                pr_id,
            )
        return dict(row) if row else None

    sets.append(f"updated_at = NOW()")
    args.append(set_name)
    args.append(pr_id)
    sql = f"""
    UPDATE eval_prs
    SET {', '.join(sets)}
    WHERE set_name = ${len(args) - 1} AND id = ${len(args)}
    RETURNING id, url, repo, set_name, category, notes,
              human_label, human_notes, source_run_id,
              EXTRACT(EPOCH FROM created_at)::float8 AS created_at,
              EXTRACT(EPOCH FROM updated_at)::float8 AS updated_at
    """
    async with pool().acquire() as conn:
        row = await conn.fetchrow(sql, *args)
    return dict(row) if row else None
