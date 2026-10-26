"""The twelve Inspector constraints (9 scored + 3 diagnostic).

Levels
------
``hard``        construction identities.  A failure means the *implementation*
                is wrong (wrong wall factor, wrong derivative assembly, broken
                checkpoint) and must block the candidate.
``soft``        physical plausibility of the solution (amplitude, energy
                calibration, vortex topology, spectrum).
``advisory``    should hold but is reported separately (realizability).
``diagnostic``  reported only, never scored -- e.g. the solution branch, which
                the existing DNS-driven solutions do not satisfy symmetrically.

Thresholds live in ``configs/audit_thresholds.json`` (versioned) and fall back
to the defaults below.
"""
from __future__ import annotations

import numpy as np

from .base import (AuditContext, AuditResult, grad_y, grad_z_periodic,
                   load_thresholds, rel_l2)

_THR = load_thresholds()


def reload_thresholds(path=None) -> dict:
    """Re-read ``configs/audit_thresholds.json`` (after calibration)."""
    global _THR
    _THR = load_thresholds(path)
    return _THR


def _t(code: str, key: str, default):
    return _THR.get(code, {}).get(key, default)


# ---------------------------------------------------------------------------
def z_periodicity(ctx: AuditContext) -> AuditResult:
    """phi(y, z+lambda) == phi(y, z) for arbitrary points."""
    from .probes import eval_phi
    rng = np.random.default_rng(0)
    y = rng.uniform(-0.995, 0.995, 512)
    z = rng.uniform(0.0, ctx.lam, 512)
    a = eval_phi(ctx, y, z)
    b = eval_phi(ctx, y, z + ctx.lam)
    val = rel_l2(a, b)
    thr = float(_t("V-ZPERIOD", "max_rel", 1e-5))
    return AuditResult("V-ZPERIOD", "hard", val < thr, val, thr,
                       {"n_points": 512})


def z_mean_zero(ctx: AuditContext) -> AuditResult:
    """Spanwise mean of phi, v~, w~ must vanish identically (k >= 1 ansatz)."""
    vals = {}
    for key, arr in (("phi", ctx.phi), ("V", ctx.V), ("Wt", ctx.Wt)):
        m = np.abs(arr.mean(axis=1)).max()
        s = float(arr.std()) + 1e-30
        vals[key] = float(m / s)
    val = max(vals.values())
    thr = float(_t("V-ZMEAN", "max_rel", 1e-5))
    return AuditResult("V-ZMEAN", "hard", val < thr, val, thr, vals)


def wall_no_slip(ctx: AuditContext) -> AuditResult:
    """phi = dphi/dy = 0 at y = +-1 (hence v~ = w~ = 0)."""
    from .probes import eval_phi
    z = ctx.z
    y = np.concatenate([np.full_like(z, -1.0), np.full_like(z, 1.0)])
    zz = np.concatenate([z, z])
    phi, dphi = eval_phi(ctx, y, zz, with_grad=True)
    scale = float(ctx.phi.std()) + 1e-30
    val = max(float(np.abs(phi).max()), float(np.abs(dphi).max())) / scale
    thr = float(_t("V-WALL", "max_rel", 1e-5))
    return AuditResult("V-WALL", "hard", val < thr, val, thr,
                       {"scale": scale})


def continuity(ctx: AuditContext) -> AuditResult:
    """div(u_perp) = d v~/dy + d w~/dz = 0.

    Two *independent* discretisations are compared (2nd-order FD in y,
    spectral in z), so this is an implementation check on the wall factor and
    derivative assembly -- it is exact only in the continuum limit.
    """
    dvdy = grad_y(ctx.V, ctx.dy)
    dwdz = grad_z_periodic(ctx.Wt, ctx.dz)
    res = dvdy + dwdz
    val = float(np.sqrt(np.mean(res ** 2)) /
                (np.sqrt(np.mean(dvdy ** 2) + np.mean(dwdz ** 2)) + 1e-30))
    thr = float(_t("V-CONT", "max_rel", 5e-3))
    return AuditResult("V-CONT", "hard", val < thr, val, thr,
                       {"note": "FD-y vs spectral-z"})


def vorticity_consistency(ctx: AuditContext) -> AuditResult:
    """Omega_x - dWbar/dy must equal d w~/dy - d v~/dz."""
    dvdz = grad_z_periodic(ctx.V, ctx.dz)
    dwdy = grad_y(ctx.Wt, ctx.dy)
    rhs = dwdy - dvdz
    mask = np.abs(ctx.y) < 0.95
    val = rel_l2(ctx.Omf[mask], rhs[mask])
    thr = float(_t("V-OMEGA", "max_rel", 1e-2))
    return AuditResult("V-OMEGA", "hard", val < thr, val, thr,
                       {"rows": int(mask.sum())})


# ---------------------------------------------------------------------------
def energy_calibration(ctx: AuditContext) -> AuditResult:
    """L_energy and the integral energy ratio must both be acceptable."""
    le = ctx.metric("L_energy")
    lo = float(_t("V-ENERGY", "ratio_lo", 0.85))
    hi = float(_t("V-ENERGY", "ratio_hi", 1.15))
    le_max = float(_t("V-ENERGY", "le_max", 0.05))
    ratio = ctx.energy_ratio()
    ok_le = (le is not None) and (le <= le_max)
    ok_ratio = np.isfinite(ratio) and lo <= ratio <= hi
    return AuditResult("V-ENERGY", "soft", bool(ok_le and ok_ratio),
                       None if le is None else float(le), le_max,
                       {"integral_ratio": ratio, "le": le,
                        "ratio_band": [lo, hi]})


def amplitude(ctx: AuditContext) -> AuditResult:
    """Non-collapsed, physically scaled vortex amplitude."""
    vv_max = ctx.cfg.get("vv_max") or 1.0
    ref = float(np.sqrt(ctx.beta * vv_max)) + 1e-30
    rv = float(np.abs(ctx.V).max()) / ref
    lo = float(_t("V-AMP", "r_min", 2.0))
    hi = float(_t("V-AMP", "r_max", 4.2))
    return AuditResult("V-AMP", "soft", lo <= rv <= hi, rv, None,
                       {"r_min": lo, "r_max": hi,
                        "max_abs_V": float(np.abs(ctx.V).max()),
                        "sqrt_beta_vvmax": ref})


def vortex_topology(ctx: AuditContext) -> AuditResult:
    """Counter-rotating roll pairs, located away from the walls."""
    pairs = ctx.vscale("vortex_pairs_per_lambda")
    ystar = ctx.vscale("y_of_strongest_vorticity")
    mx = float(np.abs(ctx.Omf).max()) + 1e-30
    both_signs = (float(ctx.Omf.max()) > 0.10 * mx
                  and float(ctx.Omf.min()) < -0.10 * mx)
    ok = (pairs is not None and 2 <= int(pairs) <= 2 * ctx.m
          and ystar is not None and abs(float(ystar)) <= 0.85 and both_signs)
    return AuditResult("V-VORTEX", "soft", bool(ok),
                       None if pairs is None else float(pairs),
                       float(2 * ctx.m),
                       {"y_of_strongest": ystar, "both_signs": both_signs})


def spanwise_spectrum(ctx: AuditContext) -> AuditResult:
    """The solution must be resolved by the retained harmonics.

    We deliberately do *not* require low-k dominance: at Ro = 75-150 the rolls
    are thinner and carrying energy at k = 3-6 is physical.  What must not
    happen is energy piling up against the truncation edge (k > m/2), which
    means the answer is set by the cut-off rather than by the physics.
    """
    n_z = ctx.z.size
    spec = np.abs(np.fft.rfft(ctx.V, axis=1)) ** 2
    spec = spec[:, 1:]                      # drop the (zero) z-mean
    e = spec.sum(axis=0)
    tot = float(e.sum()) + 1e-30
    frac_low = float(e[:2].sum() / tot)
    hi_lo = ctx.m // 2
    frac_high = float(e[hi_lo:].sum() / tot) if hi_lo < e.size else 0.0
    peakiness = float(np.abs(ctx.phi).max() / (ctx.phi.std() + 1e-30))
    hi = float(_t("V-SPECTRUM", "frac_high_max", 0.50))
    pk = float(_t("V-SPECTRUM", "peakiness_max", 6.0))
    ok = frac_high <= hi and peakiness <= pk
    return AuditResult("V-SPECTRUM", "soft", bool(ok), frac_high, hi,
                       {"frac_low": frac_low, "frac_high": frac_high,
                        "peakiness": peakiness, "pk_max": pk,
                        "n_harmonics": int(e.size)})


# ---------------------------------------------------------------------------
def realizability(ctx: AuditContext) -> AuditResult:
    """Cauchy-Schwarz on the modelled vortex stresses (should be automatic)."""
    v2 = float(np.mean(ctx.V ** 2)) + 1e-30
    w2 = float(np.mean(ctx.Wt ** 2)) + 1e-30
    vw = float(np.mean(ctx.V * ctx.Wt))
    corr = abs(vw) / float(np.sqrt(v2 * w2))
    ok = corr <= 1.0 + 1e-9
    return AuditResult("D-REALIZ", "advisory", bool(ok), corr, 1.0,
                       {"vw": vw})


def ro_limit(ctx: AuditContext) -> AuditResult:
    """Cross-Ro degradation is evaluated by the orchestrator, not here."""
    ro = ctx.cfg.get("ro")
    return AuditResult("D-RO-LIMIT", "advisory", True, None, None,
                       {"ro": ro, "status": "not_evaluated",
                        "note": "requires a cross-case retrain"})


def branch_symmetry(ctx: AuditContext) -> AuditResult:
    """Solution branch diagnostic: phi(-y, z+lam/2) = +-phi(y, z)?

    The DNS-driven solutions of this project are *not* symmetric under this
    transformation, so this is reported, never enforced.
    """
    nz = ctx.z.size
    shift = nz // 2
    flipped = ctx.phi[::-1, :]
    flipped = np.roll(flipped, shift, axis=1)
    val = rel_l2(flipped, -ctx.phi)
    label = "antisymmetric" if val < 0.35 else "asymmetric"
    return AuditResult("D-BRANCH-SYM", "diagnostic", True, val, 0.35,
                       {"branch": label,
                        "definition": "||phi(-y,z+lam/2)+phi(y,z)||/||phi||"})


CONSTRAINTS = [
    z_periodicity, z_mean_zero, wall_no_slip, continuity,
    vorticity_consistency, energy_calibration, amplitude, vortex_topology,
    spanwise_spectrum, realizability, ro_limit, branch_symmetry,
]

SCORED_KINDS = ("hard", "soft")
