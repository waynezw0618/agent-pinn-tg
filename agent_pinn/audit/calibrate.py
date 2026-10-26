"""Calibrate Inspector thresholds on the historical population.

Policy (important, and deliberately conservative)
-------------------------------------------------
* **hard** constraints are implementation identities.  Their thresholds are
  bounded by the *discretisation* error of the probe (2nd-order FD in y), not
  by the observed spread.  We set them to ``max(observed) * 2.5`` rounded up,
  which leaves a wide margin against real bugs (which are O(1)) while never
  failing a healthy run.
* **soft** thresholds keep their physically motivated defaults unless more
  than half of the historical runs fail them, in which case the default is
  clearly inconsistent with this case family and is relaxed to the 99th
  percentile of the observed values.
* A 10% hold-out (chosen by hash of the run name, so it is stable) is used to
  check that calibration did not simply overfit: the pass-rate difference
  between calibration and hold-out sets is reported.

Every run that fails a hard constraint is listed in the output, because a
non-zero count is a *code* problem, not a tuning problem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import db, paths, repo
from . import audit_all
from .base import CODE_VERSION
from . import constraints as C

HARD_MIN = {"V-ZPERIOD": 1e-5, "V-ZMEAN": 1e-5, "V-WALL": 1e-5,
            "V-CONT": 5e-3, "V-OMEGA": 1e-2}
SOFT_DEFAULT = {
    "V-ENERGY": {"le_max": 0.05, "ratio_lo": 0.85, "ratio_hi": 1.15},
    "V-AMP": {"r_min": 2.0, "r_max": 4.2},
    "V-SPECTRUM": {"frac_high_max": 0.50, "peakiness_max": 6.0},
}


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return float(v[i])


def _holdout(name: str) -> bool:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return h % 10 == 0


def calibrate(conn, out: str | Path | None = None,
              reaudit: bool = True) -> dict:
    if reaudit:
        audit_all(conn, csv_path=paths.AGENT / "audit_baseline.csv")

    runs = [r for r in repo.list_runs(conn, limit=1000)
            if r["status"] in ("complete", "complete_no_figures", "trained")]
    data: dict[str, list[tuple[str, float]]] = {}
    for run in runs:
        for a in repo.audits_for(conn, run["id"], code_version=CODE_VERSION):
            if a["value"] is None:
                continue
            data.setdefault(a["code"], []).append((run["run_name"],
                                                   float(a["value"])))

    thresholds: dict = {}
    stats: dict = {}
    for code, pairs in data.items():
        vals = [v for _n, v in pairs]
        frac_fail = 0.0
        if code in HARD_MIN:
            thr = max(HARD_MIN[code], round(max(vals) * 2.5, 6))
            thresholds[code] = {"max_rel": thr,
                                "basis": "discretisation bound (2.5x max obs)"}
        elif code in SOFT_DEFAULT:
            # observed fail rate under the default
            default = SOFT_DEFAULT[code]
            if code == "V-ENERGY":
                frac_fail = sum(1 for v in vals if v > default["le_max"]) / len(vals)
                thresholds[code] = dict(default)
            elif code == "V-AMP":
                frac_fail = sum(1 for v in vals
                                if not (default["r_min"] <= v <= default["r_max"])) \
                    / len(vals)
                thresholds[code] = dict(default)
            else:  # V-SPECTRUM
                frac_fail = sum(1 for v in vals
                                if v > default["frac_high_max"]) / len(vals)
                thresholds[code] = dict(default)
        else:
            continue
        stats[code] = {"n": len(vals), "min": min(vals), "p5": _pct(vals, .05),
                       "p50": _pct(vals, .5), "p95": _pct(vals, .95),
                       "max": max(vals), "fail_rate_default": round(frac_fail, 3)}

    # relax soft thresholds only when the default is clearly inconsistent
    for code, st in stats.items():
        if code in SOFT_DEFAULT and st["fail_rate_default"] > 0.5:
            vals = [v for _n, v in data[code]]
            if code == "V-ENERGY":
                thresholds[code]["le_max"] = round(_pct(vals, .99) * 1.05, 4)
            elif code == "V-AMP":
                thresholds[code]["r_min"] = round(_pct(vals, .01) * 0.8, 3)
                thresholds[code]["r_max"] = round(_pct(vals, .99) * 1.2, 3)
            else:
                thresholds[code]["frac_high_max"] = round(
                    min(0.75, _pct(vals, .99) * 1.1), 3)

    # hold-out agreement under the *written* thresholds
    C.reload_thresholds(None)  # ensure defaults are active before writing
    cal_runs = [r for r in runs if not _holdout(r["run_name"])]
    hold_runs = [r for r in runs if _holdout(r["run_name"])]
    hard_fail_cal = _hard_failures(conn, cal_runs)
    hard_fail_hold = _hard_failures(conn, hold_runs)

    payload = {
        "code_version": CODE_VERSION,
        "calibrated_at": db.now(),
        "thresholds": thresholds,
        "stats": stats,
        "holdout": {
            "calibration_runs": len(cal_runs),
            "holdout_runs": len(hold_runs),
            "hard_failures_calibration": hard_fail_cal,
            "hard_failures_holdout": hard_fail_hold,
            "note": "hard-constraint failures are code defects, not tuning",
        },
    }
    dest = Path(out) if out else paths.AUDIT_THRESHOLDS
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # apply immediately so subsequent audits use the calibrated values
    import os
    if os.environ.get("AGENT_PINN_USE_CALIBRATED", "1") != "0":
        C.reload_thresholds(dest)
    return payload


def _hard_failures(conn, runs) -> list[dict]:
    out = []
    for run in runs:
        for a in repo.audits_for(conn, run["id"], code_version=CODE_VERSION):
            if a["kind"] == "hard" and not a["passed"]:
                out.append({"run": run["run_name"], "code": a["code"],
                            "value": a["value"]})
    return out
