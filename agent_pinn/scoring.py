"""Accuracy / complexity / compliance scoring and Pareto ranking.

Anti-gaming notes
-----------------
* ``C`` is anchored on the *measured* legacy parameter count (5388) and a
  lower reference (256), so a model cannot gain by reporting a smaller number
  without actually shrinking.
* ``P`` only counts the deterministic Inspector constraints; when a run has
  not been audited yet the score is marked *provisional* and ``P`` is left
  out of the composite (weights renormalised) instead of being assumed 1.0.
"""
from __future__ import annotations

import math
from typing import Iterable

from . import db, repo

N_LEGACY = 5388
N_FLOOR = 256
# Re-anchored 2026-10-19: the old PDE_GOOD=0.40 was saturated by the incumbent
# best (L_pde 0.2947 and 0.3841 both scored a_pde=1.0), which left accuracy to
# be decided solely by L_energy and produced two co-ranked front-1 entries.
# The anchors now straddle the *observed* range so a_pde stays monotone.
PDE_GOOD = 0.25
PDE_BAD = 0.55

# Runs trained on the reduced smoke grid (61x32 instead of 101x48) are plumbing
# tests: their L_pde is computed on a coarser residual stencil and is not
# comparable with the full-budget numbers, so they are recorded and audited but
# kept out of the Pareto front.  The prefix is set by the orchestrator.
SMOKE_PREFIX = "smoke_"

# Auxiliary runs share the *same* structure as an eligible candidate but vary
# only the seed (replicates) or the operating condition (cross-Ro transfers).
# They are provenance, not competitors: a cheaper seed or an easier Ro must not
# displace the search candidates on the front.
AUX_PREFIXES = ("smoke_", "agent_repl_", "agent_xro_")

# The canonical task the episode searches over.  Runs trained against a
# different closure (asm_npz), a different rotation number or a different lam
# sweep point are off-task for this front and are recorded but excluded.
TASK_LAM = 2.0
TASK_M = 6
TASK_RO = 30.0


def is_smoke(run_name: str) -> bool:
    return str(run_name).startswith(SMOKE_PREFIX)


def is_auxiliary(run_name: str) -> bool:
    return str(run_name).startswith(AUX_PREFIXES)


def off_task_reason(config: dict | None) -> str | None:
    """Return why a run's training target differs from the canonical task."""
    if not config:
        return None
    if config.get("asm_npz"):
        return "asm"
    try:
        if abs(float(config.get("lam", TASK_LAM)) - TASK_LAM) > 1e-9:
            return "lam"
        if int(config.get("m", TASK_M)) != TASK_M:
            return "m"
        if "ro" in config and \
                abs(float(config["ro"]) - TASK_RO) > 1e-9:
            return "ro"
    except (TypeError, ValueError):
        return "config"
    return None


def accuracy(metrics: dict) -> tuple[float, float, float]:
    lp = metrics.get("L_pde")
    le = metrics.get("L_energy")
    if lp is None or not math.isfinite(lp) or lp <= 0:
        a_pde = 0.0
    else:
        a_pde = (math.log10(PDE_BAD) - math.log10(max(lp, 1e-9))) / \
                (math.log10(PDE_BAD) - math.log10(PDE_GOOD))
        a_pde = max(0.0, min(1.0, a_pde))
    if le is None or not math.isfinite(le):
        a_e = 0.0
    else:
        a_e = max(0.0, min(1.0, (0.05 - le) / 0.045))
    return 0.75 * a_pde + 0.25 * a_e, a_pde, a_e


def complexity(metrics: dict) -> float:
    n = metrics.get("param_count")
    if not n:
        return 0.0
    return max(0.0, min(1.0, (N_LEGACY - n) / (N_LEGACY - N_FLOOR)))


def composite(a: float, c: float, p: float | None) -> float:
    if p is None:
        return (0.50 * a + 0.20 * c) / 0.70
    return 0.50 * a + 0.20 * c + 0.30 * p


def _v(row: dict, key: str) -> float:
    val = row.get(key)
    return 0.0 if val is None else float(val)


def dominates(x: dict, y: dict) -> bool:
    keys = ("accuracy", "complexity", "compliance")
    ge = all(_v(x, k) >= _v(y, k) - 1e-12 for k in keys)
    gt = any(_v(x, k) > _v(y, k) + 1e-12 for k in keys)
    return ge and gt


def front_ranks(rows: list[dict]) -> None:
    """Assign ``front_rank`` in place via successive peeling."""
    remaining = [r for r in rows if r.get("eligible")]
    rank = 1
    while remaining:
        layer = [r for r in remaining
                 if not any(dominates(o, r) for o in remaining if o is not r)]
        if not layer:  # numerical ties
            layer = list(remaining)
        for r in layer:
            r["front_rank"] = rank
        ids = {id(r) for r in layer}
        remaining = [r for r in remaining if id(r) not in ids]
        rank += 1


def current_version() -> str:
    """The audit definition rows should be scored against.

    ``repo.compliance(code_version=None)`` aggregates *every* audit generation,
    so a run re-audited under v1.1 still carries its v1.0 failures and gets
    marked ineligible. Any caller that omits the version silently mixes
    generations -- default to the code's own version instead.
    """
    from .audit.base import CODE_VERSION
    return CODE_VERSION


def build_rows(conn, code_version: str | None = None) -> list[dict]:
    code_version = code_version or current_version()
    rows: list[dict] = []
    seen: dict[tuple, dict] = {}
    for run in repo.list_runs(conn, limit=1000):
        if run["status"] not in ("complete", "complete_no_figures", "trained"):
            continue
        metrics = run.get("metrics") or {}
        a, a_pde, a_e = accuracy(metrics)
        c = complexity(metrics)
        p, eligible = repo.compliance(conn, run["id"], code_version=code_version)
        if not repo.audits_for(conn, run["id"], code_version):
            p = None
            eligible = run["status"] in ("complete", "complete_no_figures")
        row = {
            "run_id": run["id"], "run_name": run["run_name"],
            "candidate_id": run.get("candidate_id"),
            "accuracy": a, "a_pde": a_pde, "a_energy": a_e,
            "complexity": c, "compliance": p,
            "score": composite(a, c, p),
            "eligible": bool(eligible), "front_rank": 99,
            "duplicate_of": None,
            "param_count": metrics.get("param_count"),
            "L_pde": metrics.get("L_pde"),
            "L_energy": metrics.get("L_energy"),
            "provisional": p is None,
        }
        aux = None
        if is_smoke(run["run_name"]):
            aux = "smoke"
        elif is_auxiliary(run["run_name"]):
            aux = run["run_name"].split("_", 2)[1] if \
                run["run_name"].startswith("agent_") else "aux"
        else:
            aux = off_task_reason(run.get("config"))
        if aux:
            row["eligible"] = False
            row["aux"] = aux
        # exact-duplicate candidate -> keep the best accuracy representative;
        # eligible and auxiliary rows dedup separately so a replicate can never
        # evict the search candidate it reproduces.
        key = (run.get("candidate_id") or run["run_name"], bool(row["eligible"]))
        prev = seen.get(key)
        if prev is None:
            seen[key] = row
            rows.append(row)
        else:
            keep, drop = (row, prev) if row["score"] > prev["score"] else \
                (prev, row)
            drop["duplicate_of"] = keep["run_id"]
            if keep is row:
                rows[rows.index(prev)] = row
                seen[key] = row
    front_ranks(rows)
    return rows


def refresh(conn, code_version: str | None = None) -> list[dict]:
    rows = build_rows(conn, code_version or current_version())
    repo.set_pareto(conn, rows)
    return rows


def bootstrap_metrics(rows: Iterable[dict]) -> dict:
    vals = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
    return {"n": len(vals),
            "median_A": sorted(vals)[len(vals) // 2] if vals else None,
            "best_A": max(vals) if vals else None}
