"""Build an :class:`AuditContext` from a run directory (deterministic probes)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .. import paths
from .base import AuditContext

_PROFILE_CACHE: dict[tuple, dict] = {}


def _load_profiles(cfg: dict) -> dict:
    data = cfg.get("data", "Retau180Ro30.txt")
    asm = cfg.get("asm_npz", "") or ""
    key = (data, asm)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    if asm:
        from pinn_tg.data import load_asm_profiles
        p = asm if os.path.isabs(asm) else str(paths.ROOT / asm)
        prof = load_asm_profiles(p)
        prof["U"] = None
    else:
        from pinn_tg.data import build_profiles, load_dns
        dns, _ = load_dns(str(paths.ROOT / "data" / data))
        prof = build_profiles(dns)
    _PROFILE_CACHE[key] = prof
    return prof


def build_context(run_name: str, run_dir: str | Path | None = None,
                  ny: int = 321, nz: int = 192,
                  device: str = "cpu") -> AuditContext:
    from pinn_tg.checkpoint import load_run_net
    from pinn_tg.pinn import TGProblem

    rd = Path(run_dir) if run_dir else paths.RUNS / run_name
    if not (rd / "model.pt").is_file():
        raise FileNotFoundError(f"no model.pt in {rd}")
    net, cfg, _structure = load_run_net(str(rd), device=device)
    profiles = _load_profiles(cfg)
    prob = TGProblem(profiles, lam=cfg["lam"], ny=cfg.get("ny", 101),
                     nz=cfg.get("nz", 48), beta=cfg["beta"],
                     use_nut=not cfg.get("no_nut", False),
                     use_aniso=not cfg.get("no_aniso", False))

    lam = float(cfg["lam"])
    grid, shape = prob.make_grid(ny, nz, zmax=lam)
    with torch.enable_grad():
        f = prob.fields(net, grid=grid, shape=shape)

    def np2(key: str) -> np.ndarray:
        return f[key].detach().view(ny, nz).cpu().numpy()

    y = grid[0].detach().view(ny, nz)[:, 0].cpu().numpy().astype(np.float64)
    z = np.arange(nz) * (lam / nz)
    Om = np2("Om")
    # fluctuation vorticity = total - dWbar/dy (the mean sheet has no z-structure)
    Wp = profiles["W"].deriv(1).np(y)
    Omf = Om - Wp[:, None]

    vv_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    vv_target = (cfg["beta"] * profiles["vv"](vv_t)).detach().numpy()[:, 0]

    summary_path = rd / "summary.json"
    summary = {}
    if summary_path.is_file():
        import json
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return AuditContext(
        run_name=run_name, cfg=cfg, summary=summary, lam=lam, m=int(cfg["m"]),
        beta=float(cfg["beta"]), y=y, z=z,
        V=np2("V"), Wt=np2("Wt"), Wtot=np2("W"), Om=Om, Omf=Omf,
        phi=np2("phi"), R=np2("R"), vv_target=vv_target.astype(np.float64),
        net=net, prob=prob,
        extra={"run_dir": str(rd), "ny": ny, "nz": nz,
               "profiles": profiles})


def eval_phi(ctx: AuditContext, y_np: np.ndarray, z_np: np.ndarray,
             with_grad: bool = False):
    """Evaluate phi (and optionally dphi/dy) at arbitrary points."""
    y = torch.tensor(np.asarray(y_np, dtype=np.float32)).reshape(-1, 1)
    z = torch.tensor(np.asarray(z_np, dtype=np.float32)).reshape(-1, 1)
    if with_grad:
        y.requires_grad_(True)
    phi = ctx.net(y, z)
    if not with_grad:
        return phi.detach().numpy()[:, 0]
    dphi = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi),
                               create_graph=False)[0]
    return phi.detach().numpy()[:, 0], dphi.detach().numpy()[:, 0]
