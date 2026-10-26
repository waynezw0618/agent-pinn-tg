"""Minimal job queue on top of SQLite (SRHarness 'managed trajectories').

Only one training worker may run at a time (CPU-bound, 7-11 min per run), so
the queue is deliberately single-consumer and guarded by a lock file that
also works across processes.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import bootstrap, config, db, paths, repo, runner

STALE_LOCK_S = 6 * 3600


def new_job_id(kind: str = "job") -> str:
    return f"{kind}_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


@contextmanager
def train_lock(timeout_s: float = 5.0):
    """Cross-process exclusive lock; the holder PID is written inside."""
    paths.TRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fh = None
    while True:
        try:
            fh = os.open(paths.TRAIN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fh, str(os.getpid()).encode())
            break
        except FileExistsError:
            try:
                age = time.time() - paths.TRAIN_LOCK.stat().st_mtime
                if age > STALE_LOCK_S:
                    paths.TRAIN_LOCK.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout_s:
                raise TimeoutError("training lock is held by another worker")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            os.close(fh)
        except OSError:
            pass
        # Lock release must never mask a successful training: some
        # sandboxes block unlink via a recycle-bin wrapper (SHFileOperationW
        # fails, the wrapper raises).  Fall back to truncating the file so
        # stale-lock detection still works, then retry unlink later.
        try:
            paths.TRAIN_LOCK.unlink(missing_ok=True)
        except OSError:
            try:
                with open(paths.TRAIN_LOCK, "w") as f:
                    f.write("")
            except OSError:
                pass


# ---------------------------------------------------------------------------
def enqueue(conn, *, kind: str, run_name: str | None = None,
            candidate_id: str | None = None, timeout_s: int = 1800,
            payload: dict | None = None) -> str:
    jid = new_job_id(kind)
    repo.job_enqueue(conn, job_id=jid, kind=kind, run_name=run_name,
                     candidate_id=candidate_id, timeout_s=timeout_s,
                     payload=payload or {})
    if run_name:
        repo.upsert_run(conn, run_name, status="queued",
                        candidate_id=candidate_id,
                        run_dir=paths.rel(paths.RUNS / run_name))
    return jid


def execute(conn, job: dict) -> dict:
    kind = job["kind"]
    try:
        if kind == "train":
            return _do_train(conn, job)
        if kind == "post":
            return _do_post(conn, job)
        if kind == "audit":
            from .audit import run_audits  # local import: heavy deps
            return run_audits(conn, job["run_name"])
        raise ValueError(f"unknown job kind {kind!r}")
    except Exception as exc:  # noqa: BLE001 - job failures must not kill worker
        repo.job_update(conn, job["id"], state="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        finished_at=db.now())
        if job.get("run_name"):
            repo.upsert_run(conn, job["run_name"], status="failed")
        return {"ok": False, "error": str(exc)}


def _do_train(conn, job: dict) -> dict:
    p = job["payload"]
    run_name = job["run_name"]
    structure = p["structure"]
    hyper = p.get("hyper") or {}
    do_post = p.get("post", True)
    budget = config.load_budgets()
    with train_lock():
        repo.upsert_run(conn, run_name, status="training",
                        candidate_id=job.get("candidate_id"),
                        run_dir=paths.rel(paths.RUNS / run_name),
                        started_at=db.now())
        repo.job_update(conn, job["id"], pid=os.getpid())
        cmd = runner.train_cmd(run_name, structure, hyper)
        res = runner.run(cmd, runner.train_log(run_name),
                         timeout_s=job["timeout_s"] or
                         int(budget["episode"]["train_timeout_s"]))
        if not res["ok"]:
            status = "timeout" if res["timed_out"] else "failed"
            repo.upsert_run(conn, run_name, status=status,
                            exit_code=res["code"],
                            duration_s=res["duration_s"],
                            finished_at=db.now())
            repo.job_update(conn, job["id"], state=status,
                            error=res["tail"][-500:], finished_at=db.now())
            return {"ok": False, "run": run_name, "status": status,
                    "result": res}

        repo.upsert_run(conn, run_name, status="trained",
                        exit_code=res["code"],
                        duration_s=res["duration_s"])
        out = {"ok": True, "run": run_name, "train": res}
        if do_post:
            out["post"] = _run_post(conn, run_name, p.get("post_cfg"),
                                    budget)
        repo.job_update(conn, job["id"], state="done", finished_at=db.now())
        return out


def _run_post(conn, run_name: str, post_cfg: dict | None, budget: dict) -> dict:
    repo.upsert_run(conn, run_name, status="postprocessing")
    res = runner.run(runner.post_cmd(run_name, post_cfg),
                     runner.post_log(run_name),
                     timeout_s=int(budget["episode"]["post_timeout_s"]))
    if res["ok"]:
        bootstrap.refresh_run_metrics(conn, run_name)
        repo.upsert_run(conn, run_name, status="complete",
                        finished_at=db.now())
        if repo.get_run(conn, run_name):
            cid = repo.get_run(conn, run_name).get("candidate_id")
            if cid:
                repo.set_candidate_status(conn, cid, "trained")
    else:
        repo.upsert_run(conn, run_name, status="failed", exit_code=res["code"])
    return res


def _do_post(conn, job: dict) -> dict:
    budget = config.load_budgets()
    res = _run_post(conn, job["run_name"], job["payload"].get("post_cfg"),
                    budget)
    state = "done" if res["ok"] else "failed"
    repo.job_update(conn, job["id"], state=state,
                    error=None if res["ok"] else res["tail"][-500:],
                    finished_at=db.now())
    return {"ok": res["ok"], "run": job["run_name"], "post": res}


# ---------------------------------------------------------------------------
def work(conn, max_jobs: int = 1) -> list[dict]:
    """Drain up to ``max_jobs`` queued jobs in this process."""
    done = []
    for _ in range(max_jobs):
        job = repo.job_claim(conn)
        if job is None:
            break
        done.append(execute(conn, job))
    return done


def run_now(conn, job_id: str) -> dict | None:
    """Execute one specific queued job in this process (CLI use)."""
    job = repo.job_get(conn, job_id)
    if job is None:
        return None
    if job["state"] != "queued":
        return job
    repo.job_update(conn, job_id, state="running", started_at=db.now())
    job["state"] = "running"
    return execute(conn, job)


def wait_job(conn, job_id: str, timeout_s: float = 30.0,
             poll_s: float = 0.5) -> dict | None:
    """Block until the job leaves the queued/running state (bounded)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        j = repo.job_get(conn, job_id)
        if j and j["state"] not in ("queued", "running"):
            return j
        time.sleep(poll_s)
    return repo.job_get(conn, job_id)
