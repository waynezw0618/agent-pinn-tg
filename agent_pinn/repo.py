"""CRUD over the SQLite state store.

Everything here is deliberately boring: no LLM, no torch, no subprocess.
That way the audit trail can be rebuilt and unit-tested without any of the
heavy dependencies.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from .db import now, transaction


# ---------------------------------------------------------------------------
def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _hash8(obj) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()[:8]


def make_candidate_id(structure: dict, hyper: dict,
                      structure_hash: str | None = None) -> str:
    sh = structure_hash or hashlib.sha256(
        canon(structure).encode("utf-8")).hexdigest()[:12]
    return f"c_{sh}_{_hash8(hyper)}"


# ---------------------------------------------------------------------------
def upsert_candidate(conn, *, structure: dict, hyper: dict, provenance: dict,
                     structure_hash: str, episode_id: str | None = None,
                     round_: int | None = None, parent_id: str | None = None,
                     status: str = "proposed") -> str:
    cid = make_candidate_id(structure, hyper, structure_hash)
    with transaction(conn):
        conn.execute(
            """INSERT INTO candidates(id,parent_id,episode_id,round,
                   structure_json,structure_hash,hyper_json,provenance_json,
                   status,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   parent_id=COALESCE(excluded.parent_id,parent_id),
                   episode_id=COALESCE(excluded.episode_id,episode_id),
                   provenance_json=excluded.provenance_json,
                   status=CASE WHEN status IN ('vetoed','archived')
                               THEN status ELSE excluded.status END""",
            (cid, parent_id, episode_id, round_, canon(structure),
             structure_hash, canon(hyper), canon(provenance), status, now()))
        conn.execute(
            "INSERT INTO events(ts,entity,entity_id,kind,payload_json) "
            "VALUES(?,?,?,?,?)",
            (now(), "candidate", cid, "upsert", canon(
                {"status": status, "source": provenance.get("source")})))
    return cid


def get_candidate(conn, cid: str) -> dict | None:
    row = conn.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
    return _candidate_row(row) if row else None


def list_candidates(conn, episode_id: str | None = None,
                    status: str | None = None, limit: int = 200) -> list[dict]:
    q = "SELECT * FROM candidates"
    where, args = [], []
    if episode_id:
        where.append("episode_id=?")
        args.append(episode_id)
    if status:
        where.append("status=?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [_candidate_row(r) for r in conn.execute(q, args)]


def _candidate_row(row) -> dict:
    d = dict(row)
    for k in ("structure_json", "hyper_json", "provenance_json"):
        d[k[:-5]] = json.loads(d.pop(k))
    return d


def set_candidate_status(conn, cid: str, status: str) -> None:
    with transaction(conn):
        conn.execute("UPDATE candidates SET status=? WHERE id=?", (status, cid))
        conn.execute("INSERT INTO events(ts,entity,entity_id,kind,payload_json)"
                     " VALUES(?,?,?,?,?)",
                     (now(), "candidate", cid, "status", canon({"status": status})))


# ---------------------------------------------------------------------------
RUN_FIELDS = ("run_name", "candidate_id", "status", "run_dir", "config_json",
              "summary_json", "metrics_json", "log_train", "log_post",
              "exit_code", "duration_s", "started_at", "finished_at")


def upsert_run(conn, run_name: str, **kw) -> int:
    row = conn.execute("SELECT id FROM runs WHERE run_name=?",
                       (run_name,)).fetchone()
    vals = {k: v for k, v in kw.items() if k in RUN_FIELDS}
    if row is None:
        vals.setdefault("status", "queued")
        vals.setdefault("run_dir", "")
        cols = ["run_name"] + list(vals)
        with transaction(conn):
            cur = conn.execute(
                f"INSERT INTO runs({','.join(cols)}) "
                f"VALUES({','.join('?' * len(cols))})",
                [run_name] + [vals[c] for c in vals])
            rid = int(cur.lastrowid)
    else:
        rid = int(row["id"])
        if vals:
            sets = ", ".join(f"{k}=?" for k in vals)
            with transaction(conn):
                conn.execute(f"UPDATE runs SET {sets} WHERE id=?",
                             list(vals.values()) + [rid])
    if "status" in vals:
        add_event(conn, "run", run_name, "status",
                  {"status": vals["status"]})
    return rid


def get_run(conn, run_name: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE run_name=?",
                       (run_name,)).fetchone()
    return _run_row(row) if row else None


def list_runs(conn, status: str | None = None, limit: int = 200) -> list[dict]:
    q = "SELECT * FROM runs"
    args = []
    if status:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return [_run_row(r) for r in conn.execute(q, args)]


def _run_row(row) -> dict:
    d = dict(row)
    for k in ("config_json", "summary_json", "metrics_json"):
        raw = d.pop(k)
        d[k[:-5]] = json.loads(raw) if raw else None
    return d


# ---------------------------------------------------------------------------
def add_audit(conn, run_id: int, *, code: str, kind: str, passed: bool,
              value: float | None, threshold: float | None,
              detail: dict | None = None, code_version: str = "1") -> None:
    with transaction(conn):
        conn.execute(
            """INSERT INTO audits(run_id,code,kind,passed,value,threshold,
                   detail_json,code_version,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id,code,code_version) DO UPDATE SET
                   passed=excluded.passed, value=excluded.value,
                   threshold=excluded.threshold, detail_json=excluded.detail_json
            """,
            (run_id, code, kind, int(bool(passed)), value, threshold,
             canon(detail or {}), code_version, now()))


def audits_for(conn, run_id: int, code_version: str | None = None) -> list[dict]:
    q = "SELECT * FROM audits WHERE run_id=?"
    args: list = [run_id]
    if code_version:
        q += " AND code_version=?"
        args.append(code_version)
    q += " ORDER BY code"
    out = []
    for r in conn.execute(q, args):
        d = dict(r)
        d["detail"] = json.loads(d.pop("detail_json") or "{}")
        d["passed"] = bool(d["passed"])
        out.append(d)
    return out


def compliance(conn, run_id: int, weights: dict | None = None,
               code_version: str | None = None) -> tuple[float, bool]:
    """Return ``(P, eligible)`` where eligible = all hard constraints pass."""
    weights = weights or {"hard": 3.0, "soft": 2.0}
    rows = audits_for(conn, run_id, code_version)
    scored = [r for r in rows if r["kind"] in weights]
    if not scored:
        return 0.0, False
    num = sum(weights[r["kind"]] * (1.0 if r["passed"] else 0.0)
              for r in scored)
    den = sum(weights[r["kind"]] for r in scored)
    eligible = all(r["passed"] for r in scored if r["kind"] == "hard")
    return num / den, eligible


# ---------------------------------------------------------------------------
def set_pareto(conn, rows: Iterable[dict]) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM pareto")
        for r in rows:
            conn.execute(
                """INSERT INTO pareto(run_id,accuracy,complexity,compliance,
                       score,front_rank,eligible,duplicate_of,computed_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (r["run_id"], r.get("accuracy"), r.get("complexity"),
                 r.get("compliance"), r.get("score"), r.get("front_rank"),
                 int(bool(r.get("eligible"))), r.get("duplicate_of"), now()))


def get_pareto(conn, top: int = 20, include_rejected: bool = False) -> list[dict]:
    q = """SELECT p.*, r.run_name, r.candidate_id, r.metrics_json,
                  c.structure_json, c.structure_hash, c.provenance_json
           FROM pareto p
           JOIN runs r ON r.id = p.run_id
           LEFT JOIN candidates c ON c.id = r.candidate_id"""
    if not include_rejected:
        q += " WHERE p.eligible=1 AND p.duplicate_of IS NULL"
    q += " ORDER BY p.front_rank ASC, p.score DESC LIMIT ?"
    out = []
    for r in conn.execute(q, (top,)):
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        sj = d.pop("structure_json")
        d["structure"] = json.loads(sj) if sj else None
        pj = d.pop("provenance_json")
        d["provenance"] = json.loads(pj) if pj else None
        d["eligible"] = bool(d["eligible"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
def add_event(conn, entity: str, entity_id: str, kind: str,
              payload: dict | None = None) -> None:
    conn.execute("INSERT INTO events(ts,entity,entity_id,kind,payload_json)"
                 " VALUES(?,?,?,?,?)",
                 (now(), entity, entity_id, kind, canon(payload or {})))


def events_for(conn, entity_id: str, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM events WHERE entity_id=? ORDER BY id DESC LIMIT ?",
        (entity_id, limit))
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
def add_ref(conn, candidate_id: str, chunk_id: str, quote: str,
            rank: int = 0) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO proposal_refs(candidate_id,chunk_id,quote,rank) "
            "VALUES(?,?,?,?) ON CONFLICT(candidate_id,chunk_id) DO UPDATE "
            "SET quote=excluded.quote, rank=excluded.rank",
            (candidate_id, chunk_id, quote, rank))


def refs_for(conn, candidate_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT p.chunk_id, p.quote, p.rank, l.title, l.source_path, l.page
           FROM proposal_refs p LEFT JOIN literature_refs l
                ON l.chunk_id = p.chunk_id
           WHERE p.candidate_id=? ORDER BY p.rank""", (candidate_id,))
    return [dict(r) for r in rows]


def add_literature(conn, chunks: Iterable[dict]) -> int:
    n = 0
    with transaction(conn):
        for c in chunks:
            conn.execute(
                """INSERT INTO literature_refs(chunk_id,doc_id,title,
                       source_path,page,section,text,lang,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET text=excluded.text""",
                (c["chunk_id"], c.get("doc_id"), c.get("title"),
                 c.get("source_path"), c.get("page"), c.get("section"),
                 c.get("text"), c.get("lang", "en"), now()))
            n += 1
    return n


# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    def __init__(self, key: str, used: float, cap: float):
        super().__init__(f"budget {key} exhausted ({used:.0f}/{cap:.0f})")
        self.key, self.used, self.cap = key, used, cap


def budget_init(conn, episode_id: str, caps: dict) -> None:
    with transaction(conn):
        for k, cap in caps.items():
            conn.execute(
                "INSERT INTO budgets(episode_id,key,used,cap) VALUES(?,?,0,?) "
                "ON CONFLICT(episode_id,key) DO UPDATE SET cap=excluded.cap",
                (episode_id, k, float(cap)))


def budget_take(conn, episode_id: str, key: str, cap: float,
                amount: float = 1.0) -> float:
    with transaction(conn):
        row = conn.execute(
            "SELECT used FROM budgets WHERE episode_id=? AND key=?",
            (episode_id, key)).fetchone()
        used = float(row["used"]) if row else 0.0
        if used + amount > cap:
            raise BudgetExceeded(key, used, cap)
        conn.execute(
            "INSERT INTO budgets(episode_id,key,used,cap) VALUES(?,?,?,?) "
            "ON CONFLICT(episode_id,key) DO UPDATE SET used=excluded.used",
            (episode_id, key, used + amount, float(cap)))
        return used + amount


def budget_used(conn, episode_id: str) -> dict:
    rows = conn.execute("SELECT key,used,cap FROM budgets WHERE episode_id=?",
                        (episode_id,))
    return {r["key"]: {"used": r["used"], "cap": r["cap"]} for r in rows}


# ---------------------------------------------------------------------------
def job_enqueue(conn, *, job_id: str, kind: str, run_name: str | None,
                candidate_id: str | None, timeout_s: int,
                payload: dict | None = None) -> str:
    with transaction(conn):
        conn.execute(
            """INSERT INTO jobs(id,kind,run_name,candidate_id,state,
                   timeout_s,payload_json) VALUES(?,?,?,?,'queued',?,?)""",
            (job_id, kind, run_name, candidate_id, timeout_s,
             canon(payload or {})))
    return job_id


def job_claim(conn) -> dict | None:
    """Atomically take one queued job (single worker semantics)."""
    with transaction(conn):
        row = conn.execute(
            "SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE jobs SET state='running', started_at=? "
                     "WHERE id=?", (now(), row["id"]))
        d = dict(row)
        d["state"] = "running"
        d["payload"] = json.loads(d.pop("payload_json") or "{}")
        return d


def job_update(conn, job_id: str, **kw) -> None:
    allowed = {"state", "pid", "error", "finished_at"}
    vals = {k: v for k, v in kw.items() if k in allowed}
    if not vals:
        return
    sets = ", ".join(f"{k}=?" for k in vals)
    with transaction(conn):
        conn.execute(f"UPDATE jobs SET {sets} WHERE id=?",
                     list(vals.values()) + [job_id])


def job_get(conn, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json") or "{}")
    return d


def list_jobs(conn, states: tuple[str, ...] | None = None,
              limit: int = 50) -> list[dict]:
    q = "SELECT * FROM jobs"
    args: list = []
    if states:
        q += " WHERE state IN (" + ",".join("?" * len(states)) + ")"
        args.extend(states)
    q += " ORDER BY rowid DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(q, args):
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json") or "{}")
        out.append(d)
    return out
