"""One-shot, idempotent import of ``outputs/runs/*`` into the state store.

Classification is by *file presence*, never by a hard-coded count:

``complete``            config + model + history + summary(2D) (+ figures)
``complete_no_figures`` same but the figure directory is missing
``partial``            config and/or model but no usable 2D summary
``skipped_schema``     1D RANS-ASM run (different network and summary schema)
"""
from __future__ import annotations

import json
from pathlib import Path

from . import db, paths, repo
from pinn_tg.structure import (legacy_structure, normalize_structure,
                               param_count as struct_param_count,
                               structure_hash)

HYPER_KEYS = ("beta", "lam", "m", "width", "depth", "ny", "nz", "adam",
              "lbfgs", "lr", "w_pde", "w_energy", "seed", "data", "asm_npz",
              "ro", "no_nut", "no_aniso")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _structure_of(run_dir: Path, cfg: dict) -> dict:
    ckpt_path = run_dir / "model.pt"
    if ckpt_path.is_file():
        try:
            import torch
            ckpt = torch.load(ckpt_path, weights_only=False,
                              map_location="cpu")
            s = ckpt.get("structure")
            if s:
                return normalize_structure(s)
        except Exception:  # noqa: BLE001 - fall back to cfg below
            pass
    return legacy_structure(cfg["lam"], cfg["m"], cfg["width"],
                            cfg["depth"])


def _hyper_of(cfg: dict) -> dict:
    return {k: cfg[k] for k in HYPER_KEYS if k in cfg}


def _steps_to_95(history: list[dict] | None) -> int | None:
    if not history:
        return None
    rows = [h for h in history if "L_pde" in h]
    if not rows:
        return None
    best = min(h["L_pde"] for h in rows)
    for h in rows:
        if h["L_pde"] <= 1.05 * best:
            return int(h["step"])
    return None


def _metrics(cfg: dict, summary: dict, history: list[dict] | None,
             n_params: int | None, duration_s: float | None) -> dict:
    d = summary.get("diagnostics", {})
    v = summary.get("vortex_scales", {})
    return {
        "L_pde": d.get("L_pde"),
        "L_energy": d.get("L_energy"),
        "rms_R": d.get("rms_R"),
        "max_V": d.get("max_V"),
        "max_Wt": d.get("max_Wt"),
        "max_abs_V_utau": v.get("max_abs_V_utau"),
        "max_abs_Omega_fluct": v.get("max_abs_Omega_fluct"),
        "vortex_pairs_per_lambda": v.get("vortex_pairs_per_lambda"),
        "y_of_strongest_vorticity": v.get("y_of_strongest_vorticity"),
        "energy_integral_ratio": v.get("energy_integral_ratio"),
        "param_count": n_params,
        "steps_to_95pct": _steps_to_95(history),
        "wall_time_s": duration_s,
        "U_s2": cfg.get("U_s2"),
        "seed": cfg.get("seed"),
    }


def classify(run_dir: Path) -> tuple[str, dict | None, dict | None]:
    cfg = _read_json(run_dir / "config.json")
    if cfg is None:
        return "skipped_no_config", None, None
    if "lam" not in cfg or "m" not in cfg:
        return "skipped_schema", cfg, None
    summary = _read_json(run_dir / "summary.json")
    if summary is None or "diagnostics" not in summary:
        return "partial", cfg, None
    if (run_dir / "figures").is_dir():
        return "complete", cfg, summary
    return "complete_no_figures", cfg, summary


def refresh_run_metrics(conn, run_name: str) -> dict | None:
    """Re-read a run directory and update its row (used after post-processing)."""
    run = repo.get_run(conn, run_name)
    if run is None:
        return None
    run_dir = paths.ROOT / run["run_dir"] if run.get("run_dir") else \
        paths.RUNS / run_name
    status, cfg, summary = classify(run_dir)
    if cfg is None:
        return None
    history = _read_json(run_dir / "history.json")
    structure = _structure_of(run_dir, cfg)
    ckpt_params = None
    try:
        import torch
        ck = torch.load(run_dir / "model.pt", weights_only=False,
                        map_location="cpu")
        ckpt_params = ck.get("param_count")
        duration = ck.get("wall_time_s")
    except Exception:  # noqa: BLE001
        duration = None
    n_params = ckpt_params or struct_param_count(structure)
    metrics = _metrics(cfg, summary or {}, history, n_params, duration)
    repo.upsert_run(conn, run_name, status=status,
                    run_dir=paths.rel(run_dir),
                    config_json=repo.canon(cfg),
                    summary_json=repo.canon(summary) if summary else None,
                    metrics_json=repo.canon(metrics))
    return metrics


def bootstrap(runs_dir: str | Path | None = None,
              db_path: str | Path | None = None) -> dict:
    runs_dir = Path(runs_dir) if runs_dir else paths.RUNS
    conn = db.connect(db_path)
    counts: dict[str, int] = {}
    new_candidates = 0
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        status, cfg, summary = classify(run_dir)
        counts[status] = counts.get(status, 0) + 1
        if cfg is None or status.startswith("skipped"):
            repo.upsert_run(conn, run_dir.name, status=status,
                            run_dir=paths.rel(run_dir),
                            config_json=None if cfg is None else repo.canon(cfg))
            continue

        structure = _structure_of(run_dir, cfg)
        hyper = _hyper_of(cfg)
        n_params = summary.get("config", {}).get("param_count") if summary else None
        if n_params is None:
            try:
                n_params = struct_param_count(structure)
            except Exception:  # noqa: BLE001
                n_params = None

        before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        cid = repo.upsert_candidate(
            conn, structure=structure, hyper=hyper,
            structure_hash=structure_hash(structure),
            provenance={"source": "bootstrap", "run_dir": paths.rel(run_dir)},
            status="trained" if summary else "proposed")
        after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        new_candidates += int(after > before)

        history = _read_json(run_dir / "history.json")
        metrics = (_metrics(cfg, summary, history, n_params, None)
                   if summary else {"param_count": n_params})
        repo.upsert_run(
            conn, run_dir.name, candidate_id=cid, status=status,
            run_dir=paths.rel(run_dir),
            config_json=repo.canon(cfg),
            summary_json=repo.canon(summary) if summary else None,
            metrics_json=repo.canon(metrics),
            log_train=str(paths.rel(paths.RUNS / f"{run_dir.name}_train.log")),
            log_post=str(paths.rel(paths.RUNS / f"{run_dir.name}_post.log")))

    conn.execute("INSERT INTO meta(key,value) VALUES('bootstrapped_at',?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (db.now(),))
    out = {"counts": counts, "new_candidates": new_candidates,
           "tables": db.table_counts(conn)}
    conn.close()
    return out
