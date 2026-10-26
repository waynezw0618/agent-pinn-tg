"""Audit context, result types and the Constraint protocol.

The Inspector is the "two agents are better than one" idea applied to PINNs:
the Creator (LLM or rule) proposes, this package *numerically* decides whether
the proposal respects the physics.  No LLM ever votes on pass/fail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

CODE_VERSION = "1.1"


@dataclass
class AuditResult:
    code: str
    kind: str                 # hard | soft | advisory | diagnostic
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"code": self.code, "kind": self.kind, "passed": self.passed,
                "value": self.value, "threshold": self.threshold,
                "detail": self.detail}


@dataclass
class AuditContext:
    run_name: str
    cfg: dict
    summary: dict
    lam: float
    m: int
    beta: float
    y: np.ndarray              # (ny,) y/h
    z: np.ndarray              # (nz,) z/h over exactly one wavelength
    V: np.ndarray              # (ny,nz) wall-normal fluctuation velocity
    Wt: np.ndarray             # (ny,nz) spanwise fluctuation velocity
    Wtot: np.ndarray           # (ny,nz) Wbar + Wt
    Om: np.ndarray             # (ny,nz) Omega_x
    Omf: np.ndarray            # (ny,nz) fluctuation vorticity
    phi: np.ndarray            # (ny,nz) stream function
    R: np.ndarray              # (ny,nz) PDE residual
    vv_target: np.ndarray      # (ny,) beta*<vv>(y)
    net: object = None
    prob: object = None
    extra: dict = field(default_factory=dict)

    # -- convenience ---------------------------------------------------
    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    @property
    def dz(self) -> float:
        return float(self.lam / self.z.size)

    def metric(self, key: str, default=None):
        return (self.summary.get("diagnostics", {}) or {}).get(key, default)

    def vscale(self, key: str, default=None):
        return (self.summary.get("vortex_scales", {}) or {}).get(key, default)

    def energy_ratio(self) -> float:
        trapz = getattr(np, "trapezoid", None) or np.trapz
        v2 = (self.V ** 2).mean(axis=1)
        num = float(trapz(v2, self.y))
        den = float(trapz(self.vv_target, self.y))
        return num / den if den > 0 else float("nan")


class Constraint(Protocol):
    code: str
    kind: str

    def check(self, ctx: AuditContext) -> AuditResult: ...


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b ** 2))) + eps
    return num / den


def grad_y(f: np.ndarray, dy: float) -> np.ndarray:
    """Second-order central difference along axis 0 (one-sided at walls)."""
    g = np.gradient(f, dy, axis=0, edge_order=2)
    return g


def grad_z_periodic(f: np.ndarray, dz: float) -> np.ndarray:
    """Spectral z-derivative (the field is periodic in z by construction)."""
    n = f.shape[1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dz)
    return np.real(np.fft.ifft(1j * k[None, :] * np.fft.fft(f, axis=1),
                               axis=1))


def load_thresholds(path=None) -> dict:
    from .. import paths
    p = path or paths.AUDIT_THRESHOLDS
    try:
        return json.loads(open(p, encoding="utf-8").read())
    except (OSError, ValueError):
        return {}
