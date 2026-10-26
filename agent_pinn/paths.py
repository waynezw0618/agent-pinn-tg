"""Single source of truth for every filesystem path used by agent_pinn.

No module outside this file may hard-code an absolute path.  In particular the
external LGF repository is *optional*: it is read from the environment and a
missing directory disables the feature instead of crashing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RUNS = OUTPUTS / "runs"
REPORTS = OUTPUTS / "report"
EXTRACTED = ROOT / "extracted"
CONFIGS = ROOT / "configs"
STRUCTURES = CONFIGS / "structures"

AGENT = OUTPUTS / "agent"
AGENT_EPISODES = AGENT / "episodes"
AGENT_MINER = AGENT / "miner"
AGENT_LLM = AGENT / "llm"
AGENT_DB = AGENT / "state.db"
AGENT_MANIFEST = AGENT / "baseline_manifest.json"
AGENT_LOGS = AGENT / "logs"

KB = ROOT / "agent_pinn" / "kb"
KB_CHUNKS = KB / "chunks.jsonl"
KB_INDEX = KB / "index.npz"
KB_QUERIES = KB / "eval_queries.jsonl"

BUDGETS = CONFIGS / "budgets.yaml"
AUDIT_THRESHOLDS = CONFIGS / "audit_thresholds.json"

TRAIN_LOCK = AGENT / "train.lock"

#: Interpreter used for training subprocesses.  The agent may itself run under
#: a different interpreter, so this is resolved explicitly.
PYTHON = os.environ.get("AGENT_PINN_PYTHON") or sys.executable


def lgf_root() -> Path | None:
    """Optional external LGF repository root (``AGENT_PINN_LGF_ROOT``)."""
    raw = os.environ.get("AGENT_PINN_LGF_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


def ensure_dirs() -> None:
    for d in (AGENT, AGENT_EPISODES, AGENT_MINER, AGENT_LLM, AGENT_LOGS,
              CONFIGS, STRUCTURES, KB):
        d.mkdir(parents=True, exist_ok=True)


def run_dir(name: str) -> Path:
    return RUNS / name


def rel(p: str | Path) -> str:
    """Path relative to the project root when possible (portable provenance)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
