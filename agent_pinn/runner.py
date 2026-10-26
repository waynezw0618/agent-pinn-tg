"""Subprocess execution of training / post-processing.

Training takes 7-11 min on this machine, so every caller must treat it as a
job: this module only knows how to *run one command with a timeout and a log
file*, never how to schedule.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import paths


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def tail(path: str | Path, n: int = 40) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, check=False)
    else:  # pragma: no cover - this project targets Windows
        try:
            os.killpg(os.getpgid(pid), 9)
        except OSError:
            pass


def run(cmd: list[str], log_path: str | Path, timeout_s: int = 1800,
        cwd: str | Path | None = None) -> dict:
    """Run ``cmd`` logging stdout+stderr to ``log_path``; never raises."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timed_out = False
    code = None
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd or paths.ROOT),
                                env=child_env(), stdout=fh,
                                stderr=subprocess.STDOUT,
                                start_new_session=(os.name != "nt"))
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc.pid)
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                code = -9
    dur = time.time() - t0
    return {"ok": (code == 0 and not timed_out), "code": code,
            "timed_out": timed_out, "duration_s": round(dur, 1),
            "log": str(log_path), "tail": tail(log_path, 20)}


# ---------------------------------------------------------------------------
def train_cmd(run_name: str, structure: dict, hyper: dict,
              extra: list[str] | None = None) -> list[str]:
    h = dict(hyper)
    cmd = [paths.PYTHON, "-m", "pinn_tg.train",
           "--name", run_name,
           "--structure", json.dumps(structure, separators=(",", ":")),
           "--beta", str(h.get("beta", 0.05)),
           "--ny", str(h.get("ny", 101)),
           "--nz", str(h.get("nz", 48)),
           "--adam", str(h.get("adam", 1200)),
           "--lbfgs", str(h.get("lbfgs", 80)),
           "--lr", str(h.get("lr", 1e-3)),
           "--w-pde", str(h.get("w_pde", 1.0)),
           "--w-energy", str(h.get("w_energy", 10.0)),
           "--seed", str(h.get("seed", 0))]
    if h.get("data"):
        cmd += ["--data", str(h["data"])]
    if h.get("asm_npz"):
        cmd += ["--asm-npz", str(h["asm_npz"])]
    if h.get("ro") is not None:
        cmd += ["--ro", str(h["ro"])]
    if h.get("no_nut"):
        cmd.append("--no-nut")
    if h.get("no_aniso"):
        cmd.append("--no-aniso")
    if h.get("time_budget"):
        cmd += ["--time-budget", str(h["time_budget"])]
    if h.get("early_stop"):
        cmd += ["--early-stop", str(h["early_stop"])]
    if h.get("init_from"):
        cmd += ["--init-from", str(h["init_from"])]
    if extra:
        cmd += list(extra)
    return cmd


def post_cmd(run_name: str, post: dict | None = None) -> list[str]:
    p = post or {}
    return [paths.PYTHON, "-m", "pinn_tg.postprocess", "--run", run_name,
            "--ny", str(p.get("ny", 289)),
            "--nz", str(p.get("nz", 256)),
            "--n-lam", str(p.get("n_lam", 2))]


def train_log(run_name: str) -> Path:
    return paths.RUNS / f"{run_name}_train.log"


def post_log(run_name: str) -> Path:
    return paths.RUNS / f"{run_name}_post.log"
