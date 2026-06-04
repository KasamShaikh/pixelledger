"""SQLite-backed run history.

Stores one row per analysis run (`runs`) plus one row per pipeline result
(`run_results`). The full `PipelineResult` is serialized to JSON so a past run
can be re-rendered without spending tokens again.

The DB path defaults to ``data/runs.db`` and can be overridden with the
``RUNS_DB_PATH`` env var. Note: on Azure Container Apps the filesystem is
ephemeral, so for durable history point ``RUNS_DB_PATH`` at a mounted volume.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..pipelines.base import PipelineResult

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# Fields on PipelineResult that are not JSON-serializable / not needed for replay.
_DROP_FIELDS = {"raw_response"}
_RESULT_FIELDS = {f.name for f in fields(PipelineResult)}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    created_utc    TEXT NOT NULL,
    username       TEXT NOT NULL,
    filename       TEXT NOT NULL,
    mime_type      TEXT,
    pages          INTEGER,
    repeat_runs    INTEGER,
    pipeline_count INTEGER,
    redacted       INTEGER NOT NULL DEFAULT 0,
    gt_present     INTEGER NOT NULL DEFAULT 0,
    judge_present  INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    options_json   TEXT
);

CREATE TABLE IF NOT EXISTS run_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    pipeline_id  TEXT NOT NULL,
    display_name TEXT,
    run_index    INTEGER,
    result_json  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_user_time ON runs(username, created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id);
"""


def _db_path() -> Path:
    override = os.getenv("RUNS_DB_PATH")
    return Path(override) if override else DATA_DIR / "runs.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    return conn


def _result_to_json(result: PipelineResult) -> str:
    data = {k: v for k, v in asdict(result).items() if k not in _DROP_FIELDS}
    return json.dumps(data, default=str)


def _result_from_json(payload: str) -> PipelineResult:
    data = json.loads(payload)
    kwargs = {k: v for k, v in data.items() if k in _RESULT_FIELDS}
    return PipelineResult(**kwargs)


def save_run(run_meta: dict[str, Any], results: list[PipelineResult]) -> None:
    """Persist a single run and its pipeline results in one transaction."""
    created = run_meta.get("created_utc") or datetime.now(timezone.utc).isoformat()
    total_cost = run_meta.get("total_cost_usd")
    if total_cost is None:
        total_cost = sum(float(getattr(r, "cost_usd", 0.0) or 0.0) for r in results)

    options = run_meta.get("options")
    options_json = json.dumps(options, default=str) if options is not None else None

    conn = _connect()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_utc, username, filename, mime_type, pages,
                    repeat_runs, pipeline_count, redacted, gt_present,
                    judge_present, total_cost_usd, options_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_meta["run_id"],
                    created,
                    run_meta.get("username", ""),
                    run_meta.get("filename", ""),
                    run_meta.get("mime_type"),
                    run_meta.get("pages"),
                    run_meta.get("repeat_runs"),
                    run_meta.get("pipeline_count"),
                    1 if run_meta.get("redacted") else 0,
                    1 if run_meta.get("gt_present") else 0,
                    1 if run_meta.get("judge_present") else 0,
                    float(total_cost),
                    options_json,
                ),
            )
            # Replace any prior results for this run_id (idempotent re-save).
            conn.execute(
                "DELETE FROM run_results WHERE run_id = ?", (run_meta["run_id"],)
            )
            conn.executemany(
                """
                INSERT INTO run_results (
                    run_id, pipeline_id, display_name, run_index, result_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_meta["run_id"],
                        r.pipeline_id,
                        r.display_name,
                        r.run_index,
                        _result_to_json(r),
                    )
                    for r in results
                ],
            )
    finally:
        conn.close()


def list_runs(username: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Return run metadata rows (no heavy payloads), newest first."""
    conn = _connect()
    try:
        if username:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE username = ?
                ORDER BY created_utc DESC
                LIMIT ?
                """,
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_utc DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_run(run_id: str) -> tuple[dict[str, Any] | None, list[PipelineResult]]:
    """Reconstruct run metadata and PipelineResult objects for replay."""
    conn = _connect()
    try:
        meta_row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if meta_row is None:
            return None, []
        result_rows = conn.execute(
            """
            SELECT result_json FROM run_results
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        results = [_result_from_json(row["result_json"]) for row in result_rows]
        meta = dict(meta_row)
        if meta.get("options_json"):
            try:
                meta["options"] = json.loads(meta["options_json"])
            except (TypeError, json.JSONDecodeError):
                meta["options"] = None
        return meta, results
    finally:
        conn.close()


def delete_run(run_id: str) -> None:
    """Delete a run and its results (cascade)."""
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    finally:
        conn.close()
