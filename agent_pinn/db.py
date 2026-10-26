"""SQLite scientific state store (the SRHarness 'persistent state' idea).

One file, WAL mode, single writer at a time.  Every candidate, run, audit,
budget charge and provenance link lives here so that an MCP process restart
loses nothing.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from . import paths

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates(
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  episode_id TEXT,
  round INTEGER,
  structure_json TEXT NOT NULL,
  structure_hash TEXT NOT NULL,
  hyper_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(structure_hash, hyper_json)
);

CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_name TEXT NOT NULL UNIQUE,
  candidate_id TEXT,
  status TEXT NOT NULL,
  run_dir TEXT NOT NULL,
  config_json TEXT,
  summary_json TEXT,
  metrics_json TEXT,
  log_train TEXT,
  log_post TEXT,
  exit_code INTEGER,
  duration_s REAL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS audits(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  code TEXT NOT NULL,
  kind TEXT NOT NULL,
  passed INTEGER NOT NULL,
  value REAL,
  threshold REAL,
  detail_json TEXT,
  code_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, code, code_version)
);

CREATE TABLE IF NOT EXISTS pareto(
  run_id INTEGER PRIMARY KEY,
  accuracy REAL, complexity REAL, compliance REAL, score REAL,
  front_rank INTEGER, eligible INTEGER,
  duplicate_of INTEGER, computed_at TEXT
);

CREATE TABLE IF NOT EXISTS literature_refs(
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT, title TEXT, source_path TEXT,
  page INTEGER, section TEXT, text TEXT, lang TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS proposal_refs(
  candidate_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  quote TEXT,
  rank INTEGER,
  PRIMARY KEY(candidate_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  run_name TEXT,
  candidate_id TEXT,
  state TEXT NOT NULL,
  pid INTEGER,
  started_at TEXT,
  finished_at TEXT,
  timeout_s INTEGER,
  error TEXT,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS budgets(
  episode_id TEXT NOT NULL,
  key TEXT NOT NULL,
  used REAL NOT NULL DEFAULT 0,
  cap REAL NOT NULL,
  PRIMARY KEY(episode_id, key)
);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  entity TEXT,
  entity_id TEXT,
  kind TEXT,
  payload_json TEXT
);

CREATE INDEX IF NOT EXISTS ix_runs_candidate ON runs(candidate_id);
CREATE INDEX IF NOT EXISTS ix_audits_run ON audits(run_id);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def connect(db_path: str | Path | None = None,
            create: bool = True) -> sqlite3.Connection:
    p = Path(db_path) if db_path else paths.AGENT_DB
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    if create:
        conn.executescript(DDL)
        conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
                     ("schema_version", str(SCHEMA_VERSION)))
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Immediate write transaction.

    The body must be committed *or* rolled back whatever it raises.  An earlier
    version only caught ``OperationalError``, so a domain exception (for
    example ``BudgetExceeded``) escaped with the transaction still open and the
    next ``BEGIN IMMEDIATE`` failed with "cannot start a transaction within a
    transaction".  Lock contention is retried; everything else propagates after
    a rollback.
    """
    for attempt in range(3):
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < 2:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise
    try:
        yield conn
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    try:
        conn.execute("COMMIT")
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def table_counts(conn: sqlite3.Connection) -> dict:
    out = {}
    for t in ("candidates", "runs", "audits", "pareto", "jobs", "events",
              "literature_refs", "proposal_refs"):
        out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return out
