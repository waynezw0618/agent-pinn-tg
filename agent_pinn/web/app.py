"""FastAPI dashboard for the Agent-PINN loop.

Read-mostly: everything it shows is derivable from ``agent_pinn.db`` plus the
run directories, so the UI can be started at any time and never becomes the
source of truth.  The three write endpoints are deliberate and narrow:

``POST /api/run/{name}/audit``   re-run the Inspector (cheap, idempotent)
``POST /api/run/{name}/veto``    human override -- a hard veto that the
                                 orchestrator cannot undo
``POST /api/run/{name}/retrain`` enqueue a training job for the same structure

No CDN, no build step: one HTML file with inline CSS/JS and a hand-drawn
canvas chart.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import config, db, jobs, paths, repo, scoring

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Agent-PINN", version="0.1.0")


def _conn():
    return db.connect(paths.AGENT_DB)


# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    p = STATIC / "index.html"
    if not p.is_file():
        raise HTTPException(500, "index.html missing")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/summary")
def summary():
    conn = _conn()
    runs = repo.list_runs(conn, limit=1000)
    eps = sorted([d.name for d in paths.AGENT_EPISODES.iterdir()
                  if d.is_dir()]) if paths.AGENT_EPISODES.is_dir() else []
    by_status: dict[str, int] = {}
    for r in runs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "db": str(paths.AGENT_DB),
        "tables": db.table_counts(conn),
        "runs": len(runs),
        "runs_by_status": by_status,
        "candidates": len(repo.list_candidates(conn, limit=5000)),
        "episodes": eps[-12:],
        "budgets": config.load_budgets()["episode"],
    }


@app.get("/api/pareto")
def pareto(top: int = 30):
    conn = _conn()
    rows = scoring.refresh(conn)
    rows.sort(key=lambda r: (r["front_rank"], -r["score"]))
    out = []
    for r in rows[:top]:
        out.append({k: r.get(k) for k in
                    ("run_name", "candidate_id", "param_count", "accuracy",
                     "complexity", "compliance", "score", "front_rank",
                     "eligible", "L_pde", "L_energy", "provisional",
                     "duplicate_of")})
    return {"rows": out, "total": len(rows)}


@app.get("/api/runs")
def runs(limit: int = 60):
    conn = _conn()
    rows = repo.list_runs(conn, limit=limit)
    slim = []
    for r in rows:
        m = r.get("metrics") or {}
        slim.append({"run_name": r["run_name"], "status": r["status"],
                     "candidate_id": r.get("candidate_id"),
                     "L_pde": m.get("L_pde"), "L_energy": m.get("L_energy"),
                     "param_count": m.get("param_count"),
                     "duration_s": r.get("duration_s")})
    return {"rows": slim}


@app.get("/api/run/{name}")
def run_detail(name: str):
    conn = _conn()
    run = repo.get_run(conn, name)
    if not run:
        raise HTTPException(404, "no such run")
    rd = paths.RUNS / name
    audits = repo.audits_for(conn, run["id"])
    hist = []
    hp = rd / "history.json"
    if hp.is_file():
        try:
            hist = json.loads(hp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hist = []
    cfg = {}
    cp = rd / "config.json"
    if cp.is_file():
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}
    prov, refs, cand = None, [], None
    if run.get("candidate_id"):
        cand = repo.get_candidate(conn, run["candidate_id"])
        if cand:
            prov = cand.get("provenance")
            refs = repo.refs_for(conn, run["candidate_id"])
    structure = cfg.get("structure")
    if isinstance(structure, str):
        try:
            structure = json.loads(structure)
        except json.JSONDecodeError:
            structure = None
    if not structure and cand:
        structure = cand.get("structure")
    if not structure and "lam" in cfg:      # pre-v2 run: reconstruct
        from pinn_tg.structure import legacy_structure
        structure = legacy_structure(float(cfg["lam"]), int(cfg.get("m", 8)),
                                     int(cfg.get("width", 64)),
                                     int(cfg.get("depth", 3)),
                                     float(cfg.get("out_scale", 0.05)))
    log = ""
    lp = paths.RUNS / name / "train.log"
    for cand_log in (lp, paths.AGENT_LOGS / f"{name}.train.log"):
        if Path(cand_log).is_file():
            log = Path(cand_log).read_text(encoding="utf-8",
                                           errors="replace")[-4000:]
            break
    figs = sorted([p.name for p in (rd / "figures").glob("*.png")]) \
        if (rd / "figures").is_dir() else []
    return {"run": run, "audits": audits, "history": hist, "config": cfg,
            "provenance": prov, "refs": refs, "log_tail": log,
            "figures": figs, "structure": structure}


@app.get("/figures/{name}/{fname}")
def figure(name: str, fname: str):
    if "/" in fname or "\\" in fname or not fname.endswith(".png"):
        raise HTTPException(400, "bad figure name")
    p = paths.RUNS / name / "figures" / fname
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/png")


@app.post("/api/run/{name}/audit")
def re_audit(name: str):
    from ..audit import run_audits
    conn = _conn()
    if not repo.get_run(conn, name):
        raise HTTPException(404, "no such run")
    res = run_audits(conn, name)
    scoring.refresh(conn)
    return res


@app.post("/api/run/{name}/veto")
def veto(name: str):
    conn = _conn()
    run = repo.get_run(conn, name)
    if not run:
        raise HTTPException(404, "no such run")
    if run.get("candidate_id"):
        repo.set_candidate_status(conn, run["candidate_id"], "vetoed")
    repo.upsert_run(conn, name, status="vetoed")
    repo.add_event(conn, "run", name, "veto", {"by": "ui"})
    scoring.refresh(conn)
    return {"ok": True, "run": name, "status": "vetoed"}


@app.post("/api/run/{name}/retrain")
def retrain(name: str, suffix: str = "_retrain"):
    conn = _conn()
    run = repo.get_run(conn, name)
    if not run:
        raise HTTPException(404, "no such run")
    cfgp = paths.RUNS / name / "config.json"
    if not cfgp.is_file():
        raise HTTPException(400, "run has no config.json")
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    new_name = f"{name}{suffix}"
    structure = cfg["structure"]
    if isinstance(structure, str):
        structure = json.loads(structure)
    hyper = {k: cfg[k] for k in ("beta", "lam", "m", "ny", "nz", "adam",
                                 "lbfgs", "seed") if k in cfg}
    jid = jobs.enqueue(conn, kind="train", run_name=new_name,
                       candidate_id=run.get("candidate_id"),
                       payload={"structure": structure, "hyper": hyper,
                                "post": True})
    return {"ok": True, "job": jid, "run": new_name,
            "note": "queued; run `python -m agent_pinn.cli work` to execute"}


@app.get("/api/episodes")
def episodes():
    out = []
    if paths.AGENT_EPISODES.is_dir():
        for d in sorted(paths.AGENT_EPISODES.iterdir()):
            if not d.is_dir():
                continue
            ep = d / "episode.json"
            if ep.is_file():
                try:
                    s = json.loads(ep.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                out.append({"episode": s.get("episode"),
                            "state": s.get("state"),
                            "rounds": s.get("rounds"),
                            "trains": s.get("trains"),
                            "compliance_rate": s.get("compliance_rate"),
                            "best": s.get("best"),
                            "rejection_codes": s.get("rejection_codes")})
            else:
                out.append({"episode": d.name, "state": "running"})
    return {"rows": out[-20:]}


@app.get("/api/episode/{epid}")
def episode_detail(epid: str):
    d = paths.AGENT_EPISODES / epid
    ep = d / "episode.json"
    pool = d / "pool.json"
    if not d.is_dir():
        raise HTTPException(404, "no such episode")
    return {
        "summary": json.loads(ep.read_text(encoding="utf-8"))
        if ep.is_file() else None,
        "pool": json.loads(pool.read_text(encoding="utf-8"))
        if pool.is_file() else None,
    }


@app.get("/api/jobs")
def job_list(top: int = 20):
    conn = _conn()
    return {"rows": repo.list_jobs(conn, limit=top)}


@app.exception_handler(Exception)
def _boom(_request, exc):  # pragma: no cover - defensive
    return JSONResponse({"ok": False, "error": str(exc)[:400]}, status_code=500)
