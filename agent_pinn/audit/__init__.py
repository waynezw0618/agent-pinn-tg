"""Inspector entry points: audit one run, many runs, or a whole DB."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .. import db, paths, repo, scoring
from .base import CODE_VERSION, AuditContext, AuditResult
from .constraints import CONSTRAINTS, SCORED_KINDS
from .probes import build_context

WEIGHTS = {"hard": 3.0, "soft": 2.0}


def _run_constraints(ctx: AuditContext) -> list[AuditResult]:
    out: list[AuditResult] = []
    for fn in CONSTRAINTS:
        try:
            out.append(fn(ctx))
        except Exception as exc:  # noqa: BLE001
            code = getattr(fn, "__name__", "unknown").upper()
            out.append(AuditResult(
                code=f"E-{code}", kind="hard", passed=False, value=None,
                threshold=None, detail={"error": f"{type(exc).__name__}: {exc}"}))
    return out


def _score(results: list[AuditResult]) -> tuple[float, bool]:
    scored = [r for r in results if r.kind in WEIGHTS]
    if not scored:
        return 0.0, False
    num = sum(WEIGHTS[r.kind] * (1.0 if r.passed else 0.0) for r in scored)
    den = sum(WEIGHTS[r.kind] for r in scored)
    eligible = all(r.passed for r in scored if r.kind == "hard")
    return num / den, eligible


def run_audits(conn, run_name: str, probe: tuple[int, int] = (321, 192),
               persist: bool = True) -> dict:
    run = repo.get_run(conn, run_name)
    if run is None:
        return {"ok": False, "error": f"unknown run {run_name}"}
    try:
        ctx = build_context(run_name, ny=probe[0], nz=probe[1])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    results = _run_constraints(ctx)
    p, eligible = _score(results)
    if persist:
        for r in results:
            repo.add_audit(conn, run["id"], code=r.code, kind=r.kind,
                           passed=r.passed, value=r.value,
                           threshold=r.threshold, detail=r.detail,
                           code_version=CODE_VERSION)
        if run.get("candidate_id"):
            repo.set_candidate_status(conn, run["candidate_id"], "audited")
        repo.add_event(conn, "run", run_name, "audit",
                       {"P": p, "eligible": eligible,
                        "failed": [r.code for r in results if not r.passed]})
    return {"ok": True, "run": run_name, "compliance": p,
            "eligible": eligible, "code_version": CODE_VERSION,
            "results": [r.as_dict() for r in results]}


def audit_run(conn, run_name: str, db_path=None) -> dict:
    return run_audits(conn, run_name)


def audit_all(conn, csv_path: str | Path | None = None,
              limit: int | None = None) -> dict:
    runs = [r for r in repo.list_runs(conn, limit=1000)
            if r["status"] in ("complete", "complete_no_figures", "trained")]
    if limit:
        runs = runs[:limit]
    rows, failed = [], []
    for run in runs:
        res = run_audits(conn, run["run_name"])
        if not res.get("ok"):
            failed.append({"run": run["run_name"], "error": res.get("error")})
            continue
        for r in res["results"]:
            rows.append({"run": run["run_name"], "code": r["code"],
                         "kind": r["kind"], "passed": int(r["passed"]),
                         "value": r["value"], "threshold": r["threshold"],
                         "code_version": CODE_VERSION})
    out_csv = Path(csv_path) if csv_path else paths.AGENT / "audit_baseline.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "code", "kind", "passed",
                                           "value", "threshold",
                                           "code_version"])
        w.writeheader()
        w.writerows(rows)
    scores = scoring.refresh(conn, code_version=CODE_VERSION)
    hard = [r for r in rows if r["kind"] == "hard"]
    summary = {
        "runs": len(runs), "rows": len(rows), "errors": failed,
        "hard_pass_rate": (sum(r["passed"] for r in hard) / len(hard)
                           if hard else None),
        "eligible_share": (sum(1 for r in scores if r["eligible"]) /
                           max(1, len(scores))),
        "csv": str(out_csv.relative_to(paths.ROOT))
        if str(out_csv).startswith(str(paths.ROOT)) else str(out_csv),
    }
    return summary
