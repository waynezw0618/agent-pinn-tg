"""Psi-NN stages 2-3 for this problem: cluster the head weights, decide which
structure is *expressible*, prune units, and distil a student.

What is actually mined
----------------------
The last linear layer maps the shared trunk features ``h(y)`` to the ``2m``
coefficient functions ``[a_1..a_m, b_1..b_m]``.  Each mode therefore owns two
rows of that matrix.  Modes whose rows point in similar directions can share a
head; modes that do not must keep their own.  We cluster those row pairs and
then measure whether a *shared low-rank basis* can reproduce the full head
matrix -- that is the honest feasibility test for ``tied_grouped``.

Parameter accounting (be explicit in the paper)
-----------------------------------------------
``shared`` / ``grouped`` / ``per_mode`` have the *same* parameter count; only
``tied_grouped`` (low-rank basis) and trunk width reduction actually compress.
In this network the head is ~11% of the parameters, so real compression has to
come from pruning the trunk -- which is why ``prune_frac`` defaults to 0.25.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import paths
from .distill import distill, fidelity
from pinn_tg.checkpoint import load_run_net, save_checkpoint
from pinn_tg.pinn import TGProblem
from pinn_tg.structure import (head_rank, normalize_structure, param_count,
                               validate_structure)

GRID_MARGIN = 0.05


@dataclass
class MineResult:
    teacher_run: str
    teacher_params: int
    student_params: int
    param_reduction: float
    k_selected: int
    silhouette: float
    groups: list[list[int]]
    tied_residual: float
    rank_used: int
    prune_frac: float
    teacher_width: int
    student_width: int
    structure: dict
    criteria: dict = field(default_factory=dict)
    distill_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def _head_matrix(net) -> np.ndarray:
    """(2m, width) final linear map for a legacy/shared teacher."""
    if getattr(net, "is_legacy", False):
        return net.mlp[-1].weight.detach().cpu().numpy()
    import torch.nn as nn
    if isinstance(net.heads, nn.Linear):
        return net.heads.weight.detach().cpu().numpy()
    raise ValueError("teacher already has grouped/tied heads; nothing to mine")


def _features(net, y: torch.Tensor) -> torch.Tensor:
    if getattr(net, "is_legacy", False):
        h = y
        for mod in list(net.mlp)[:-1]:
            h = mod(h)
        return h
    return net.trunk(y)


def _cluster_modes(W: np.ndarray, m: int, k_min: int, k_max: int):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    vec = np.stack([np.concatenate([W[2 * i], W[2 * i + 1]]) for i in range(m)])
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    best = (None, -1.0, 1)
    hi = min(k_max, m - 1)
    for k in range(max(2, k_min), hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vec)
        if len(set(km.labels_)) < 2:
            continue
        try:
            s = float(silhouette_score(vec, km.labels_))
        except ValueError:
            continue
        if s > best[1]:
            best = (km.labels_, s, k)
    if best[0] is None:
        return None, -1.0, 1
    labels = best[0]
    groups = [[int(i) + 1 for i in range(m) if labels[i] == c]
              for c in sorted(set(labels))]
    return groups, best[1], best[2]


def _low_rank_residual(W: np.ndarray, r_max: int = 16) -> tuple[float, int]:
    """Relative Frobenius error of the best rank-r approximation of the head."""
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    e = s ** 2
    cum = np.cumsum(e) / max(e.sum(), 1e-30)
    r = int(np.searchsorted(cum, 0.95) + 1)
    r = max(1, min(r, r_max))
    err = float(np.sqrt(max(0.0, 1.0 - cum[r - 1])))
    return err, r


def _unit_saliency(net, y: torch.Tensor, W_out: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        h = _features(net, y).detach().cpu().numpy()
    rms = np.sqrt((h ** 2).mean(axis=0)) + 1e-30
    return np.linalg.norm(W_out, axis=0) * rms


def _teacher_prob(teacher_run: str, ny: int | None = None,
                  nz: int | None = None):
    net, cfg, st = load_run_net(str(paths.RUNS / teacher_run))
    from .audit.probes import _load_profiles
    prof = _load_profiles(cfg)
    prob = TGProblem(prof, lam=cfg["lam"], ny=int(ny or cfg.get("ny", 101)),
                     nz=int(nz or cfg.get("nz", 48)), beta=cfg["beta"],
                     use_nut=not cfg.get("no_nut", False),
                     use_aniso=not cfg.get("no_aniso", False))
    return net, cfg, st, prob


def mine(teacher_run: str, out_dir: str | Path,
         k_min: int = 2, k_max: int = 4, prune_frac: float = 0.25,
         distill_steps: int = 600, seed: int = 0,
         min_width: int = 8,
         distill_ny: int | None = None, distill_nz: int | None = None) -> dict:
    """Mine a student structure from a trained teacher and distil it.

    ``distill_ny`` / ``distill_nz`` set the *distillation* grid, which is
    independent of the grid the teacher was trained on.  All acceptance
    numbers (L_pde, L_energy, fidelity) are always re-measured on the
    teacher's own grid so the comparison stays honest; the coarse grid only
    makes the KD optimisation cheaper.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, cfg, teacher_struct, prob = _teacher_prob(teacher_run)
    if distill_ny or distill_nz:
        dprob = _teacher_prob(teacher_run, distill_ny, distill_nz)[3]
    else:
        dprob = prob
    m = int(cfg["m"])
    W = _head_matrix(net)
    y = prob.base_grid[0]

    groups, sil, k_sel = _cluster_modes(W, m, k_min, k_max)
    residual, r_used = _low_rank_residual(W, r_max=max(4, head_rank(teacher_struct)))

    sal = _unit_saliency(net, y, W)
    width = int(teacher_struct["trunk"]["width"])
    keep = max(min_width, int(round(width * (1.0 - prune_frac))))
    order = np.argsort(-sal)
    kept_idx = np.sort(order[:keep])

    student = normalize_structure(teacher_struct)
    student["trunk"]["width"] = int(keep)
    if groups and len(groups) >= 2:
        equal = len({len(g) for g in groups}) == 1
        if equal and residual <= 0.05:
            student["heads"]["mode"] = "tied_grouped"
            student["heads"]["groups"] = groups
            student["heads"]["group_size"] = len(groups[0])
            student["heads"]["rank"] = int(r_used)
        else:
            student["heads"]["mode"] = "grouped"
            student["heads"]["groups"] = groups
            student["heads"]["group_size"] = len(groups[0]) if equal else 1
    student = normalize_structure(student)
    errs = validate_structure(student)
    if errs:
        raise ValueError(f"mined structure invalid: {errs}")

    teacher_params = int(net.param_count)
    student_params = int(param_count(student))
    (out / "student_structure.json").write_text(
        json.dumps(student, indent=2), encoding="utf-8")
    (out / "clusters.json").write_text(json.dumps({
        "k": k_sel, "silhouette": sil, "groups": groups,
        "tied_residual": residual, "rank_used": r_used,
        "kept_units": kept_idx.tolist(), "prune_frac": prune_frac,
    }, indent=2), encoding="utf-8")

    stud, hist = distill(net, dprob, student, steps=distill_steps, seed=seed)
    # ``distill`` freezes the teacher's parameters for its own loop; the flat
    # evaluation below is forward-only, so a pristine reload keeps the two
    # evaluations independent of anything distill did.
    net, cfg, teacher_struct, prob = _teacher_prob(teacher_run)
    grid, shape = prob.base_grid, (prob.ny, prob.nz)
    with torch.enable_grad():
        lp_t, le_t, _, _ = prob.losses(net, grid=grid, shape=shape)
        lp_s, le_s, diag_s, _ = prob.losses(stud, grid=grid, shape=shape)
    fid = fidelity(net, stud, prob)

    criteria = {
        "J1_teacher_fidelity_le_0.05": fid["max"] <= 0.05,
        "J2a_Lpde_ratio_le_1.15": float(lp_s.detach()) <= 1.15 * float(lp_t.detach()),
        "J2b_Lenergy_le_0.05": float(le_s.detach()) <= 0.05,
        "J5_param_reduction_ge_0.25":
            1.0 - student_params / max(1, teacher_params) >= 0.25,
        "J6_lowrank_residual_le_0.05": residual <= 0.05,
    }
    result = MineResult(
        teacher_run=teacher_run, teacher_params=teacher_params,
        student_params=student_params,
        param_reduction=1.0 - student_params / max(1, teacher_params),
        k_selected=k_sel, silhouette=sil, groups=groups or [],
        tied_residual=residual, rank_used=r_used, prune_frac=prune_frac,
        teacher_width=width, student_width=keep, structure=student,
        criteria={
            **criteria,
            "fidelity": fid,
            "L_pde_teacher": float(lp_t.detach()),
            "L_pde_student": float(lp_s.detach()),
            "L_energy_teacher": float(le_t.detach()),
            "L_energy_student": float(le_s.detach()),
        },
        distill_history=hist[-5:],
    )
    (out / "report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    save_checkpoint(str(out / "student.pt"), stud, {**cfg, **{
        "structure": student}}, mined_from=teacher_run)
    return result.to_dict()
