"""Episode-level ledger of proposals (SRHarness' persistent scientific state).

Every structure the Creator emits is written here, accepted or rejected, with
the round, the strategy, the rationale and -- once trained -- the run name and
metrics.  Consequences:

* duplicates are cheap to detect across rounds (``seen()``);
* the rejection ledger is the compliance metric;
* the Web UI and the paper figures can be regenerated from the file alone;
* a crashed episode can be resumed without re-asking the LLM.

The DB holds the same information in normalised form; this file is the
human-readable, diffable record of *how the search actually went*.
"""
from __future__ import annotations

import json
from pathlib import Path

from .db import now


class Pool:
    def __init__(self, path: str | Path, *, episode: str, task: dict):
        self.path = Path(path)
        self.episode = episode
        self.task = task
        self.entries: list[dict] = []
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", [])
            self.episode = data.get("episode", episode)
            self.task = data.get("task", task) or task

    # -- persistence ----------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "episode": self.episode, "task": self.task,
            "updated_at": now(), "entries": self.entries,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # -- queries --------------------------------------------------------
    def seen(self) -> set[str]:
        return {e["hash"] for e in self.entries}

    def get(self, h: str) -> dict | None:
        for e in self.entries:
            if e["hash"] == h:
                return e
        return None

    def by_run(self, run_name: str) -> dict | None:
        for e in self.entries:
            if e.get("run_name") == run_name:
                return e
        return None

    def with_status(self, *status: str) -> list[dict]:
        want = set(status)
        return [e for e in self.entries if e.get("status") in want]

    def pending(self) -> list[dict]:
        return self.with_status("pending")

    def trained(self) -> list[dict]:
        return self.with_status("trained", "audited", "vetoed")

    def best(self, n: int = 2, require_audited: bool = False) -> list[dict]:
        rows = self.trained()
        if require_audited:
            rows = [r for r in rows if r.get("status") == "audited"] or rows
        rows.sort(key=lambda r: (
            -(r.get("score") if r.get("score") is not None else -1.0),
            r.get("L_pde") if r.get("L_pde") is not None else 9e9))
        return rows[:n]

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.get("status", "?")] = out.get(e.get("status", "?"), 0) + 1
        out["total"] = len(self.entries)
        return out

    def rejection_codes(self) -> dict:
        out: dict[str, int] = {}
        for e in self.entries:
            code = e.get("reject_code")
            if code:
                key = str(code).split(":")[0]
                out[key] = out.get(key, 0) + 1
        return out

    # -- mutation -------------------------------------------------------
    def add(self, prop: dict, *, status: str, round_: int, strategy: str,
            parent: str | None = None, **extra) -> dict:
        entry = {
            "hash": prop["hash"],
            "structure": prop["structure"],
            "param_count": prop.get("param_count"),
            "rationale": prop.get("rationale", ""),
            "expected_effect": prop.get("expected_effect", ""),
            "confidence": prop.get("confidence"),
            "status": status,
            "round": round_,
            "strategy": strategy,
            "parent": parent,
            "created_at": now(),
        }
        entry.update(extra)
        existing = self.get(prop["hash"])
        if existing is not None:
            return existing
        self.entries.append(entry)
        return entry

    def mark(self, h: str, **fields) -> dict | None:
        e = self.get(h)
        if e is not None:
            e.update(fields)
            e["updated_at"] = now()
        return e
