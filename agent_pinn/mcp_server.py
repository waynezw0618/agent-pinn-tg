"""MCP (stdio) tool surface for Agent-PINN.

Design rules
------------
* Every tool returns an envelope ``{"ok": bool, "data"|"error", "trace"}``
  instead of raising, so an LLM client can self-recover.
* Training is *queued*, never executed inline: ``train_candidate`` returns a
  ``job_id`` immediately.  A background worker thread (started by
  ``main``) drains the queue.
* Nothing is printed to stdout: stdout belongs to the JSON-RPC transport.
  Use :func:`_log` (file logging) instead.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import config, db, jobs, paths, repo, runner, scoring
from pinn_tg.structure import (legacy_structure, normalize_structure,
                               sample_valid, structure_hash,
                               validate_structure)

mcp = FastMCP("agent-pinn")

LOG = logging.getLogger("agent_pinn.mcp")


def _setup_logging() -> None:
    paths.ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=str(paths.AGENT_LOGS / "mcp.log"),
        filemode="a")


def _log(msg: str) -> None:
    LOG.info(msg)


def _ok(data, tool: str, t0: float) -> dict:
    return {"ok": True, "data": data,
            "trace": {"tool": tool, "elapsed_s": round(time.time() - t0, 3)}}


def _err(code: str, message: str, tool: str, t0: float,
         hint: str | None = None) -> dict:
    return {"ok": False,
            "error": {"code": code, "message": message, "hint": hint},
            "trace": {"tool": tool, "elapsed_s": round(time.time() - t0, 3)}}


def _conn():
    return db.connect(paths.AGENT_DB)


# ---------------------------------------------------------------------------
@mcp.tool()
def search_literature(query: str, k: int = 5) -> dict:
    """Search the project literature (BM25 + dense + rerank).

    Returns the top-k chunks with their ids, so a later proposal can cite
    them via ``proposal_refs``.
    """
    t0 = time.time()
    try:
        from .rag.retrieve import search
        res = search(query, k=k)
        return _ok(res, "search_literature", t0)
    except FileNotFoundError as exc:
        return _err("KB_MISSING", str(exc), "search_literature", t0,
                    hint="run `python -m agent_pinn.rag.ingest` first")
    except Exception as exc:  # noqa: BLE001
        return _err("RAG_FAILED", f"{type(exc).__name__}: {exc}",
                    "search_literature", t0)


@mcp.tool()
def propose_structures(episode_id: str, n: int = 3, strategy: str = "auto",
                       base_run: str = "lam2.0",
                       violation_codes: list[str] | None = None) -> dict:
    """Propose up to ``n`` structurally valid candidate configurations.

    Without an LLM this degrades to deterministic mutations of a base
    structure (rule provider), so the pipeline stays reproducible.
    """
    t0 = time.time()
    if n < 1 or n > 4:
        return _err("BAD_ARG", "n must be in 1..4", "propose_structures", t0)
    try:
        from pinn_tg.checkpoint import load_run_net
        rd = paths.RUNS / base_run
        if not rd.is_dir():
            return _err("NO_BASE_RUN", f"{base_run} not found",
                        "propose_structures", t0)
        _net, _cfg, base = load_run_net(str(rd))
        from .llm import get_provider
        provider = get_provider()
        hyper = dict(config.load_budgets()["train"])
        out = []
        conn = _conn()
        for s in provider.propose(base, n=n, strategy=strategy,
                                  violation_codes=violation_codes or []):
            errs = validate_structure(s)
            if errs:
                continue
            sh = structure_hash(s)
            cid = repo.upsert_candidate(
                conn, structure=s, hyper=hyper, structure_hash=sh,
                provenance={"source": provider.name, "strategy": strategy,
                            "episode_id": episode_id},
                episode_id=episode_id, status="proposed")
            out.append({"candidate_id": cid, "structure_hash": sh,
                        "structure": s})
        conn.close()
        return _ok({"proposed": out, "provider": provider.name},
                   "propose_structures", t0)
    except Exception as exc:  # noqa: BLE001
        return _err("PROPOSE_FAILED", f"{type(exc).__name__}: {exc}",
                    "propose_structures", t0)


@mcp.tool()
def train_candidate(candidate_id: str, wait_seconds: int = 0,
                    post: bool = True, smoke: bool = False) -> dict:
    """Queue training for a candidate; returns immediately with a job id."""
    t0 = time.time()
    conn = _conn()
    cand = repo.get_candidate(conn, candidate_id)
    if cand is None:
        conn.close()
        return _err("NO_CANDIDATE", candidate_id, "train_candidate", t0)
    existing = conn.execute(
        "SELECT run_name,status FROM runs WHERE candidate_id=? "
        "ORDER BY id DESC LIMIT 1", (candidate_id,)).fetchone()
    if existing and existing["status"] in ("complete", "trained",
                                           "postprocessing", "training",
                                           "queued"):
        conn.close()
        return _ok({"job_id": None, "run": existing["run_name"],
                    "status": existing["status"], "cached": True},
                   "train_candidate", t0)

    n = conn.execute("SELECT COUNT(*) FROM runs WHERE candidate_id=?",
                     (candidate_id,)).fetchone()[0]
    run_name = f"agent_{cand['structure_hash']}" + (f"_r{n + 1}" if n else "")
    hyper = dict(cand["hyper"])
    post_cfg = dict(config.load_budgets()["post"])
    if smoke:
        sb = config.load_budgets()["episode"]["smoke"]
        hyper.update({"adam": sb["adam"], "lbfgs": sb["lbfgs"],
                      "ny": sb["ny"], "nz": sb["nz"]})
        post_cfg = {"ny": 65, "nz": 32, "n_lam": 2}
    jid = jobs.enqueue(conn, kind="train", run_name=run_name,
                       candidate_id=candidate_id,
                       timeout_s=int(config.load_budgets()["episode"]
                                     ["train_timeout_s"]),
                       payload={"structure": cand["structure"], "hyper": hyper,
                                "post": post, "post_cfg": post_cfg})
    conn.close()
    data = {"job_id": jid, "run": run_name, "status": "queued"}
    if wait_seconds > 0:
        c2 = _conn()
        j = jobs.wait_job(c2, jid, timeout_s=min(wait_seconds, 30))
        c2.close()
        data["job"] = j
    return _ok(data, "train_candidate", t0)


@mcp.tool()
def postprocess_candidate(run_name: str, wait_seconds: int = 0,
                          ny: int = 289, nz: int = 256,
                          n_lam: int = 2) -> dict:
    """Queue post-processing (summary.json + fields.npz + figures)."""
    t0 = time.time()
    conn = _conn()
    run = repo.get_run(conn, run_name)
    if run is None:
        conn.close()
        return _err("NO_RUN", run_name, "postprocess_candidate", t0)
    jid = jobs.enqueue(conn, kind="post", run_name=run_name,
                       candidate_id=run.get("candidate_id"),
                       timeout_s=int(config.load_budgets()["episode"]
                                     ["post_timeout_s"]),
                       payload={"post_cfg": {"ny": ny, "nz": nz,
                                             "n_lam": n_lam}})
    conn.close()
    data = {"job_id": jid, "run": run_name, "status": "queued"}
    if wait_seconds > 0:
        c2 = _conn()
        data["job"] = jobs.wait_job(c2, jid, timeout_s=min(wait_seconds, 30))
        c2.close()
    return _ok(data, "postprocess_candidate", t0)


@mcp.tool()
def audit_candidate(run_name: str, strict: bool = False) -> dict:
    """Run the deterministic physics-constraint Inspector on a run."""
    t0 = time.time()
    try:
        from .audit import audit_run
        conn = _conn()
        res = audit_run(conn, run_name)
        conn.close()
        return _ok(res, "audit_candidate", t0)
    except Exception as exc:  # noqa: BLE001
        return _err("AUDIT_FAILED", f"{type(exc).__name__}: {exc}",
                    "audit_candidate", t0)


@mcp.tool()
def mine_structure(teacher_run: str, k_min: int = 2, k_max: int = 4,
                   prune_frac: float = 0.25) -> dict:
    """Psi-NN mining: distill, cluster the head weights, emit a structure."""
    t0 = time.time()
    try:
        from .miner import mine
        out = str(paths.AGENT_MINER / teacher_run)
        res = mine(str(paths.RUNS / teacher_run), out, k_min=k_min,
                   k_max=k_max, prune_frac=prune_frac)
        return _ok(res, "mine_structure", t0)
    except Exception as exc:  # noqa: BLE001
        return _err("MINE_FAILED", f"{type(exc).__name__}: {exc}",
                    "mine_structure", t0)


@mcp.tool()
def get_pareto(top: int = 20, include_rejected: bool = False) -> dict:
    """Return the current Pareto front (rank, structure, metrics, audits)."""
    t0 = time.time()
    conn = _conn()
    rows = repo.get_pareto(conn, top=top,
                           include_rejected=include_rejected)
    conn.close()
    return _ok({"rows": rows, "n": len(rows)}, "get_pareto", t0)


@mcp.tool()
def get_run_history(candidate_id: str = "", run_name: str = "") -> dict:
    """Return a run row plus its training history and audit results."""
    t0 = time.time()
    conn = _conn()
    if run_name:
        run = repo.get_run(conn, run_name)
    elif candidate_id:
        row = conn.execute(
            "SELECT run_name FROM runs WHERE candidate_id=? "
            "ORDER BY id DESC LIMIT 1", (candidate_id,)).fetchone()
        run = repo.get_run(conn, row["run_name"]) if row else None
    else:
        conn.close()
        return _err("BAD_ARG", "give run_name or candidate_id",
                    "get_run_history", t0)
    if run is None:
        conn.close()
        return _err("NO_RUN", run_name or candidate_id, "get_run_history", t0)
    hist_path = Path(paths.ROOT) / run["run_dir"] / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.is_file() else []
    audits = repo.audits_for(conn, run["id"])
    conn.close()
    return _ok({"run": run, "history": history, "audits": audits},
               "get_run_history", t0)


@mcp.tool()
def compare_runs(run_names: list[str],
                 metrics: list[str] | None = None) -> dict:
    """Compare several runs on selected metrics (default: accuracy + cost)."""
    t0 = time.time()
    keys = metrics or ["L_pde", "L_energy", "param_count", "rms_R"]
    conn = _conn()
    out = []
    for name in run_names:
        run = repo.get_run(conn, name)
        if not run:
            out.append({"run": name, "missing": True})
            continue
        m = run.get("metrics") or {}
        out.append({"run": name,
                    **{k: m.get(k) for k in keys},
                    "status": run["status"],
                    "candidate_id": run.get("candidate_id")})
    conn.close()
    return _ok({"rows": out, "metrics": keys}, "compare_runs", t0)


@mcp.tool()
def get_job(job_id: str) -> dict:
    """Poll a queued/running job."""
    t0 = time.time()
    conn = _conn()
    job = repo.job_get(conn, job_id)
    conn.close()
    if job is None:
        return _err("NO_JOB", job_id, "get_job", t0)
    return _ok(job, "get_job", t0)


@mcp.tool()
def episode_status(episode_id: str) -> dict:
    """Summarise an episode: candidate/run states and budget usage."""
    t0 = time.time()
    conn = _conn()
    cand = repo.list_candidates(conn, episode_id=episode_id)
    by_status: dict[str, int] = {}
    for c in cand:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
    runs = conn.execute(
        """SELECT r.status, COUNT(*) n FROM runs r
           JOIN candidates c ON c.id = r.candidate_id
           WHERE c.episode_id=? GROUP BY r.status""",
        (episode_id,)).fetchall()
    budgets = repo.budget_used(conn, episode_id)
    conn.close()
    return _ok({"episode": episode_id, "candidates": by_status,
                "runs": {r["status"]: r["n"] for r in runs},
                "budgets": budgets}, "episode_status", t0)


# ---------------------------------------------------------------------------
def worker_loop(stop: threading.Event, poll_s: float = 2.0) -> None:
    conn = db.connect(paths.AGENT_DB)
    _log("worker started")
    while not stop.is_set():
        try:
            done = jobs.work(conn, max_jobs=1)
            if not done:
                time.sleep(poll_s)
            else:
                _log(f"job done: {done}")
        except Exception as exc:  # noqa: BLE001
            _log(f"worker error: {exc}")
            time.sleep(5)
    conn.close()
    _log("worker stopped")


def main() -> None:
    _setup_logging()
    paths.ensure_dirs()
    stop = threading.Event()
    th = threading.Thread(target=worker_loop, args=(stop,), daemon=True,
                          name="agent-pinn-worker")
    th.start()
    try:
        mcp.run()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
