"""Teacher -> student knowledge distillation for the TG-vortex PINN.

Psi-NN stage 1 ("physics-informed distillation") for this problem:

    L = L_pde + w_e L_energy
        + w_kd       * ||phi_s - phi_t||^2 / ||phi_t||^2
        + w_kd_deriv * (||V_s - V_t||^2 + ||W_s - W_t||^2)
                       / (||V_t||^2 + ||W_t||^2)

The teacher is only ever run forward (``no_grad``); the student keeps the full
PDE/energy graph so the physics losses still shape it.  This avoids the
brittle "weight surgery" that direct head splicing would require when the
student has a different shape.
"""
from __future__ import annotations

import time

import torch

from pinn_tg.pinn import TGNet, TGProblem


def distill(teacher: TGNet, prob: TGProblem, structure: dict,
            steps: int = 600, lr: float = 1e-3,
            w_pde: float = 1.0, w_energy: float = 10.0,
            w_kd: float = 1.0, w_kd_deriv: float = 0.1,
            seed: int = 0, log_every: int = 100) -> tuple[TGNet, list[dict]]:
    torch.manual_seed(seed)
    student = TGNet(structure=structure)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    grid, shape = prob.base_grid, (prob.ny, prob.nz)
    # fields() differentiates through y, so the graph must be enabled even
    # though the teacher's parameters are frozen.
    with torch.enable_grad():
        ft = prob.fields(teacher, grid=grid, shape=shape)
    phi_t = ft["phi"].detach()
    V_t = ft["V"].detach()
    W_t = ft["Wt"].detach()
    n_phi = float((phi_t ** 2).mean()) + 1e-30
    n_vel = float((V_t ** 2).mean() + (W_t ** 2).mean()) + 1e-30

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist: list[dict] = []
    t0 = time.time()
    best = float("inf")
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr if step <= 0.7 * steps else lr * 0.2
        L_pde, L_e, diag, fs = prob.losses(student, grid=grid, shape=shape)
        kd_phi = ((fs["phi"] - phi_t) ** 2).mean() / n_phi
        kd_vel = (((fs["V"] - V_t) ** 2).mean()
                  + ((fs["Wt"] - W_t) ** 2).mean()) / n_vel
        loss = (w_pde * L_pde + w_energy * L_e
                + w_kd * kd_phi + w_kd_deriv * kd_vel)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        cur = float(loss.detach())
        if not torch.isfinite(loss):
            break
        best = min(best, cur)
        if step == 1 or step % log_every == 0 or step == steps:
            hist.append({"step": step, "loss": cur,
                         "L_pde": diag["L_pde"], "L_energy": diag["L_energy"],
                         "kd_phi": float(kd_phi.detach()),
                         "kd_vel": float(kd_vel.detach()),
                         "elapsed_s": round(time.time() - t0, 1)})
    student.eval()
    return student, hist


def fidelity(teacher: TGNet, student: TGNet, prob: TGProblem) -> dict:
    """Relative L2 difference of phi / V / Wt on the training grid."""
    grid, shape = prob.base_grid, (prob.ny, prob.nz)
    with torch.enable_grad():
        ft = prob.fields(teacher, grid=grid, shape=shape)
        fs = prob.fields(student, grid=grid, shape=shape)
    ft = {k: v.detach() for k, v in ft.items() if torch.is_tensor(v)}
    fs = {k: v.detach() for k, v in fs.items() if torch.is_tensor(v)}
    out = {}
    for key in ("phi", "V", "Wt"):
        num = float(((fs[key] - ft[key]) ** 2).mean().sqrt())
        den = float((ft[key] ** 2).mean().sqrt()) + 1e-30
        out[key] = num / den
    out["max"] = max(out["phi"], out["V"], out["Wt"])
    return out
