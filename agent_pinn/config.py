"""Budgets, thresholds and provider configuration (env-var driven)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import paths

DEFAULT_BUDGETS: dict = {
    "episode": {
        "max_rounds": 4,
        "proposals_per_round": 3,
        "max_trains": 8,
        "max_llm_calls": 24,
        "max_wall_minutes": 120,
        "train_timeout_s": 1500,
        "post_timeout_s": 300,
        "smoke": {"adam": 200, "lbfgs": 20, "ny": 61, "nz": 32},
    },
    "early_stop": {
        "no_hypervolume_gain_rounds": 2,
        "score_plateau": 1.0e-3,
        "max_consecutive_hard_failures": 4,
    },
    "train": {
        "beta": 0.05,
        "lam": 2.0,
        "m": 6,
        "width": 48,
        "depth": 3,
        "ny": 101,
        "nz": 48,
        "adam": 1200,
        "lbfgs": 80,
        "w_pde": 1.0,
        "w_energy": 10.0,
        "seed": 0,
    },
    "post": {"ny": 289, "nz": 256, "n_lam": 2},
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_budgets(path: str | Path | None = None) -> dict:
    p = Path(path) if path else paths.BUDGETS
    if not p.is_file():
        return json.loads(json.dumps(DEFAULT_BUDGETS))
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _deep_merge(DEFAULT_BUDGETS, data)


# ---------------------------------------------------------------------------
def llm_settings() -> dict:
    return {
        "provider": os.environ.get("AGENT_PINN_LLM_PROVIDER", "auto"),
        "base_url": os.environ.get("AGENT_PINN_LLM_BASE_URL",
                                   "https://open.bigmodel.cn/api/paas/v4"),
        "api_key": os.environ.get("AGENT_PINN_LLM_API_KEY", ""),
        "model": os.environ.get("AGENT_PINN_LLM_MODEL", "glm-5.3"),
        "timeout_s": float(os.environ.get("AGENT_PINN_LLM_TIMEOUT", "60")),
        "temperature": float(os.environ.get("AGENT_PINN_LLM_TEMPERATURE",
                                            "0.2")),
    }


def embed_settings() -> dict:
    return {
        "model": os.environ.get("AGENT_PINN_EMBED_MODEL", "BAAI/bge-m3"),
        "fallback": os.environ.get("AGENT_PINN_EMBED_FALLBACK",
                                   "BAAI/bge-base-zh-v1.5"),
        "rerank": os.environ.get("AGENT_PINN_RERANK", "1") not in ("0", "false",
                                                                  "False"),
    }


def audit_thresholds() -> dict:
    if paths.AUDIT_THRESHOLDS.is_file():
        return json.loads(paths.AUDIT_THRESHOLDS.read_text(encoding="utf-8"))
    return {}
