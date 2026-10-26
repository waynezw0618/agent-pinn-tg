"""Creator-Inspector closed loop for structure discovery.

Fused design
------------
The three sources contribute at different levels, and none of them is copied
verbatim:

``SRHarness`` (Tsinghua)
    the *runtime shape*: a persistent scientific state (here ``pool.json`` plus
    the SQLite tables), composable tools (search / propose / train / audit),
    and a budgeted lifecycle.  SRHarness' own point is that a structured runtime
    buys more than a bigger model.

``Two agents are better than one`` (Tacke et al. 2026)
    the *separation of powers*: the Creator never scores its own output.  A
    numerical Inspector -- 12 constraints, no LLM -- is the only thing that can
    veto, and a hard-identity failure vetoes regardless of how good the loss is.
    Rejections are counted, so structural compliance is a reported metric.

``LLM4eD`` (Du et al. 2024)
    the *strategy schedule*: alternate self-improvement (the LLM as a
    black-box optimiser of one candidate) with evolution (the LLM as a
    recombination operator over two parents).  Round 1 is a literature seed so
    the first proposal is grounded rather than random.

State machine::

    INIT -> PROPOSE -> VALIDATE -> TRAIN -> POST -> AUDIT -> SCORE -> DECIDE
                                                             ^         |
                                                             +-- next round
    DECIDE -> DONE

Budget guardrails (rounds, proposals/round, trains, LLM calls, wall clock) are
checked *before* every expensive step, so an episode can always be interrupted
without losing the ledger.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from . import config, db, jobs, paths, prompts, repo, runner, schema, scoring
from .llm import RuleBasedProvider
from .structure_pool import Pool
from pinn_tg.structure import legacy_structure, normalize_structure, param_count

STRATEGY_ROTATION = ("literature_seed", "self_improve", "evolution",
                     "self_improve")

KB_QUERIES = {
    "literature_seed": [
        "旋转槽道湍流 泰勒-哥特勒涡 二次流 神经网络 结构",
        "physics-informed neural network spectral ansatz wall function "
        "boundary layer",
        "Reynolds stress anisotropy low-rank tensor basis constitutive model",
    ],
    "self_improve": [
        "near-wall energy calibration turbulent channel PINN amplitude",
        "vorticity identity Taylor-Gortler roll pair wavelength",
    ],
    "evolution": [
        "weight sharing multi-head neural network mode decomposition",
        "activation function spectral bias sin tanh fitting oscillatory",
    ],
}


class Orchestrator:
    def __init__(self, episode: str, *, db_path: str | Path | None = None,
                 provider=None, budget: dict | None = None,
                 smoke: bool = False, conn=None, seed: int = 0,
                 task: dict | None = None, use_kb: bool = True):
        self.episode = str(episode)
        self.db_path = Path(db_path) if db_path else paths.AGENT_DB
        self.budget = budget or config.load_budgets()
        self.smoke = bool(smoke)
        self.seed = int(seed)
        self.use_kb = use_kb
        self.provider = provider or RuleBasedProvider()
        self._owns_conn = conn is None
        self.conn = conn or db.connect(self.db_path)
        self.dir = paths.AGENT_EPISODES / self.episode
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task = task or self._default_task()
        self.pool = Pool(self.dir / "pool.json", episode=self.episode,
                         task=self.task)
        self.t0 = time.time()
        self.log: list[dict] = []
        self.state = "INIT"

    # -- task / budget ---------------------------------------------------
    def _default_task(self) -> dict:
        b = self.budget["train"]
        return {
            "lam": float(b.get("lam", 2.0)),
            "m": int(b.get("m", 6)),
            "ro": float(b.get("ro", 30.0)),
            "beta": float(b.get("beta", 0.05)),
            "max_params": 12000,
            "base": legacy_structure(float(b.get("lam", 2.0)),
                                     int(b.get("m", 6)),
                                     int(b.get("width", 48)),
                                     int(b.get("depth", 3))),
        }

    def _ep_budget(self) -> dict:
        return self.budget["episode"]

    def _elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60.0

    def _can_afford_train(self) -> bool:
        """Wall-clock guard only; the train *count* is capped in the DB."""
        return self._elapsed_min() < float(self._ep_budget()["max_wall_minutes"])

    def _take(self, key: str, cap_key: str) -> bool:
        try:
            repo.budget_take(self.conn, self.episode, key,
                             float(self._ep_budget()[cap_key]), 1)
            return True
        except repo.BudgetExceeded:
            return False

    # -- event log -------------------------------------------------------
    def _event(self, kind: str, payload: dict) -> dict:
        row = {"t": round(time.time() - self.t0, 1), "state": self.state,
               "kind": kind, **payload}
        self.log.append(row)
        try:
            repo.add_event(self.conn, "episode", self.episode, kind, payload)
        except Exception:  # noqa: BLE001 - logging must never break the loop
            pass
        return row

    # -- main ------------------------------------------------------------
    def run(self) -> dict:
        self.state = "INIT"
        self.n_trains = 0
        self.rejections: list[dict] = []
        self.rounds_done = 0
        repo.budget_init(self.conn, self.episode, {
            "trains": self._ep_budget()["max_trains"],
            "llm_calls": self._ep_budget()["max_llm_calls"],
            "rounds": self._ep_budget()["max_rounds"],
        })
        self._event("init", {"task": {k: v for k, v in self.task.items()
                                     if k != "base"},
                             "smoke": self.smoke,
                             "provider": self.provider.name})
        self._save_meta()

        best_before, no_gain = -1.0, 0
        tries = 0
        while True:
            tries += 1
            if not self._take("rounds", "max_rounds"):
                self._event("stop", {"reason": "max_rounds"})
                break
            if self._elapsed_min() >= float(self._ep_budget()["max_wall_minutes"]):
                self._event("stop", {"reason": "max_wall_minutes"})
                break
            if tries > int(self._ep_budget()["max_rounds"]) + 2:
                self._event("stop", {"reason": "round_guard"})
                break

            self.rounds_done += 1
            rnd = self.rounds_done
            strategy = STRATEGY_ROTATION[(rnd - 1) % len(STRATEGY_ROTATION)]
            if strategy == "self_improve" and not self.pool.best(1):
                strategy = "literature_seed"
            if strategy == "evolution" and len(self.pool.best(2)) < 2:
                strategy = "literature_seed"

            accepted = self._propose(rnd, strategy)
            trained = self._train(accepted, rnd)
            self._audit(trained, rnd)
            scoring.refresh(self.conn)

            best_now = self._best_score()
            if best_now > best_before + self.budget["early_stop"][
                    "score_plateau"]:
                best_before, no_gain = best_now, 0
            else:
                no_gain += 1
            self._event("round_done", {"round": rnd, "strategy": strategy,
                                       "accepted": len(accepted),
                                       "trained": len(trained),
                                       "best_score": round(best_now, 4),
                                       "no_gain_rounds": no_gain})
            if no_gain >= int(self.budget["early_stop"][
                    "no_hypervolume_gain_rounds"]):
                self._event("stop", {"reason": "no_improvement"})
                break
            if not accepted and not trained:
                self._event("stop", {"reason": "nothing_accepted"})
                break

        self.state = "DONE"
        return self._finish()

    # -- PROPOSE + VALIDATE ----------------------------------------------
    def _retrieve(self, strategy: str) -> list[dict]:
        if not self.use_kb:
            return []
        try:
            from .rag.retrieve import get_kb
            kb = get_kb()
            seen, out = set(), []
            for q in KB_QUERIES.get(strategy, []):
                for r in kb.search(q, k=2).get("results", []):
                    if r["chunk_id"] in seen:
                        continue
                    seen.add(r["chunk_id"])
                    out.append(r)
            return out[:4]
        except Exception as exc:  # noqa: BLE001 - RAG is an aid, not a gate
            self._event("kb_error", {"error": f"{type(exc).__name__}: {exc}"})
            return []

    def _front_rows(self) -> list[dict]:
        try:
            rows = repo.get_pareto(self.conn, top=8)
        except Exception:  # noqa: BLE001
            return []
        from pinn_tg.structure import describe
        for r in rows:
            m = r.get("metrics") or {}
            for k in ("L_pde", "L_energy", "param_count"):
                if r.get(k) is None:
                    r[k] = m.get(k)
            if r.get("description") is None and r.get("structure"):
                try:
                    r["description"] = describe(r["structure"])
                except Exception:  # noqa: BLE001
                    r["description"] = ""
        return rows

    def _propose(self, rnd: int, strategy: str) -> list[dict]:
        self.state = "PROPOSE"
        n_want = int(self._ep_budget()["proposals_per_round"])
        if not self._take("llm_calls", "max_llm_calls"):
            self._event("llm_budget_exhausted", {})
            return []

        parents = self.pool.best(2)
        parent = parents[0]["structure"] if parents else None
        parent2 = parents[1]["structure"] if len(parents) > 1 else None
        audit = None
        if parent and strategy == "self_improve":
            audit = self._audit_summary(parents[0].get("run_name"))

        passages = self._retrieve(strategy)
        messages = prompts.build_messages(
            strategy, task=self.task, front=self._front_rows(),
            passages=passages, parent=parent, second_parent=parent2,
            audit=audit, n=n_want,
            feedback=self._feedback())

        try:
            raw = self.provider.chat(messages, context={
                "strategy": strategy, "parents": [p for p in (parent, parent2)
                                                  if p],
                "base": self.task["base"], "seed": self.seed, "index": rnd,
                "n": n_want,
            })
        except Exception as exc:  # noqa: BLE001
            self._event("llm_error", {"error": f"{type(exc).__name__}: {exc}"})
            if not isinstance(self.provider, RuleBasedProvider):
                self.provider = RuleBasedProvider()
                self._event("provider_fallback", {"to": "rule"})
            try:
                raw = self.provider.chat([], context={
                    "strategy": strategy,
                    "parents": [p for p in (parent, parent2) if p],
                    "base": self.task["base"], "seed": self.seed,
                    "index": rnd, "n": n_want})
            except Exception as exc2:  # noqa: BLE001
                self._event("propose_failed", {"error": str(exc2)[:200]})
                return []

        payloads = prompts.parse_multi(raw)
        accepted: list[dict] = []
        for i, payload in enumerate(payloads[:n_want]):
            self.state = "VALIDATE"
            try:
                prop = schema.validate_proposal(
                    payload, task=self.task, seen=self.pool.seen())
            except schema.Rejection as rej:
                self.rejections.append({"round": rnd, "code": rej.code,
                                        "detail": rej.detail})
                self._event("reject", {"round": rnd, "code": rej.code,
                                       "detail": rej.detail[:200]})
                continue
            entry = self.pool.add(prop, status="pending", round_=rnd,
                                  strategy=strategy,
                                  parent=parents[0]["hash"] if parents else None)
            self._event("accept", {"round": rnd, "hash": prop["hash"],
                                   "params": prop["param_count"],
                                   "confidence": prop["confidence"]})
            for p in passages[:2]:
                try:
                    repo.add_ref(self.conn, prop["hash"], p["chunk_id"],
                                 str(p.get("text", ""))[:400],
                                 source=str(p.get("doc_id", "")))
                except Exception:  # noqa: BLE001
                    pass
            accepted.append({**prop, "strategy": strategy, "round": rnd})
        self.pool.save()
        return accepted

    def _feedback(self) -> str:
        codes = self.pool.rejection_codes()
        if not codes:
            return ""
        parts = [f"{k} x{v}" for k, v in sorted(codes.items())]
        return ("Previously rejected proposals: " + ", ".join(parts) +
                ". Avoid repeating those exact mistakes.")

    # -- TRAIN -----------------------------------------------------------
    def _hyper(self) -> dict:
        h = {k: v for k, v in self.budget["train"].items()}
        if self.smoke:
            sb = self._ep_budget()["smoke"]
            h.update({"adam": sb["adam"], "lbfgs": sb["lbfgs"],
                      "ny": sb["ny"], "nz": sb["nz"]})
        return h

    def _post_cfg(self) -> dict:
        if self.smoke:
            return {"ny": 65, "nz": 32, "n_lam": 2}
        return dict(self.budget["post"])

    def _train(self, accepted: list[dict], rnd: int) -> list[dict]:
        self.state = "TRAIN"
        out: list[dict] = []
        for i, prop in enumerate(accepted, start=1):
            if not self._can_afford_train():
                self._event("train_skipped", {"hash": prop["hash"],
                                              "reason": "wall_clock"})
                break
            if not self._take("trains", "max_trains"):
                self._event("train_skipped", {"hash": prop["hash"],
                                              "reason": "train_budget"})
                break
            run_name = f"agent_{self.episode}_r{rnd}_{i}"
            if self.smoke:
                # scoring.is_smoke() keeps the reduced-grid numbers off the
                # Pareto front; they are not comparable with ny=101 x nz=48.
                run_name = "smoke_" + run_name
            hyper = self._hyper()
            conn = self.conn
            cid = repo.upsert_candidate(
                conn, structure=prop["structure"], hyper=hyper,
                structure_hash=prop["hash"], episode_id=self.episode,
                round_=rnd, status="queued",
                provenance={"source": "creator", "provider": self.provider.name,
                            "model": self.provider.model,
                            "strategy": prop["strategy"],
                            "rationale": prop["rationale"][:600],
                            "confidence": prop["confidence"]})
            self.pool.mark(prop["hash"], status="training",
                           candidate_id=cid, run_name=run_name)
            jid = jobs.enqueue(conn, kind="train", run_name=run_name,
                               candidate_id=cid,
                               timeout_s=int(self._ep_budget()[
                                   "train_timeout_s"]),
                               payload={"structure": prop["structure"],
                                        "hyper": hyper, "post": True,
                                        "post_cfg": self._post_cfg()})
            self.n_trains += 1
            self._event("train_start", {"round": rnd, "run": run_name,
                                        "job": jid, "params":
                                        prop["param_count"]})
            res = jobs.run_now(self.conn, jid) or {}
            ok = bool(res.get("ok"))
            self._event("train_done", {"run": run_name, "ok": ok,
                                       "status": (res.get("post") or {}
                                                  ).get("code")})
            if not ok:
                self.pool.mark(prop["hash"], status="failed",
                               error=str(res)[:300])
                continue
            out.append({**prop, "run_name": run_name, "candidate_id": cid})
        self.pool.save()
        return out

    # -- AUDIT -----------------------------------------------------------
    def _audit_summary(self, run_name: str | None) -> dict | None:
        if not run_name:
            return None
        run = repo.get_run(self.conn, run_name)
        if not run:
            return None
        rows = repo.audits_for(self.conn, run["id"])
        if not rows:
            return None
        return {
            "hard_failures": [r["code"] for r in rows
                              if r["kind"] == "hard" and not r["passed"]],
            "soft_failures": [r["code"] for r in rows
                              if r["kind"] == "soft" and not r["passed"]],
            "values": {r["code"]: r["value"] for r in rows},
        }

    def _audit(self, trained: list[dict], rnd: int) -> None:
        self.state = "AUDIT"
        from .audit import run_audits
        for t in trained:
            # 161x96 halves the y-stencil error 4x poorly: a sharper but still
            # valid field tripped V-CONT purely by discretisation (E2 r1_3).
            probe = (81, 48) if self.smoke else (321, 192)
            res = run_audits(self.conn, t["run_name"], probe=probe)
            if not res.get("ok"):
                self.pool.mark(t["hash"], status="audit_error",
                               error=res.get("error", "")[:200])
                self._event("audit_error", {"run": t["run_name"],
                                            "error": res.get("error")})
                continue
            results = res["results"]
            hard_fail = [r["code"] for r in results
                         if r["kind"] == "hard" and not r["passed"]]
            run = repo.get_run(self.conn, t["run_name"]) or {}
            metrics = self._metrics(t["run_name"])
            status = "vetoed" if hard_fail else "audited"
            score = None
            try:
                rows = scoring.refresh(self.conn)
                hit = [r for r in rows if r["run_name"] == t["run_name"]]
                score = hit[0]["score"] if hit else None
            except Exception:  # noqa: BLE001
                pass
            self.pool.mark(t["hash"], status=status,
                           run_name=t["run_name"],
                           L_pde=metrics.get("L_pde"),
                           L_energy=metrics.get("L_energy"),
                           hard_pass=not hard_fail,
                           compliance=res.get("compliance"),
                           score=score,
                           veto=hard_fail or None)
            self._event("audit", {"run": t["run_name"], "status": status,
                                  "P": res.get("compliance"),
                                  "hard_fail": hard_fail})
        self.pool.save()

    def _metrics(self, run_name: str) -> dict:
        p = paths.RUNS / run_name / "summary.json"
        if not p.is_file():
            return {}
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        diag = s.get("diagnostics") or {}
        return {"L_pde": diag.get("L_pde"), "L_energy": diag.get("L_energy")}

    # -- SCORE / DECIDE --------------------------------------------------
    def _best_score(self) -> float:
        rows = self.pool.trained()
        vals = [r["score"] for r in rows if r.get("score") is not None]
        return max(vals) if vals else -1.0

    def _finish(self) -> dict:
        scoring.refresh(self.conn)
        best = self.pool.best(1)
        # Rejections that never became ledger entries (bad JSON, duplicate,
        # locked m/lam) are still *proposals*, so they belong in the
        # denominator -- otherwise the compliance rate is always 100%.
        n_proposed = len(self.pool.entries) + len(self.rejections)
        summary = {
            "state": self.state,
            "episode": self.episode,
            "rounds": self.rounds_done,
            "counts": self.pool.counts(),
            "rejection_codes": self.pool.rejection_codes(),
            "n_rejected": len(self.rejections),
            "n_proposed": n_proposed,
            "compliance_rate": round(len(self.pool.entries) /
                                     max(1, n_proposed), 3),
            "trains": self.n_trains,
            "elapsed_min": round(self._elapsed_min(), 1),
            "provider": self.provider.stats(),
            "best": ({"hash": best[0]["hash"], "run": best[0].get("run_name"),
                      "score": best[0].get("score"),
                      "L_pde": best[0].get("L_pde"),
                      "params": best[0].get("param_count"),
                      "status": best[0]["status"]} if best else None),
            "log": self.log,
        }
        (self.dir / "episode.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.dir / "REPORT.md").write_text(self._report(summary),
                                           encoding="utf-8")
        return summary

    def _save_meta(self) -> None:
        (self.dir / "task.json").write_text(
            json.dumps(self.task, indent=2), encoding="utf-8")

    def _report(self, s: dict) -> str:
        lines = [f"# Episode `{s['episode']}`", "",
                 f"- state: **{s['state']}**, rounds: {s['rounds']}, "
                 f"trains: {s['trains']}, wall: {s['elapsed_min']} min",
                 f"- provider: {s['provider']['provider']} "
                 f"({s['provider']['model']}), calls: "
                 f"{s['provider']['calls']}, errors: {s['provider']['errors']}",
                 f"- compliance rate (accepted/proposed): "
                 f"{s['compliance_rate']:.0%}",
                 f"- rejection codes: {s['rejection_codes'] or 'none'}", ""]
        if s["best"]:
            b = s["best"]
            lines += [f"**Best**: `{b['run']}` score={b['score']} "
                      f"L_pde={b['L_pde']} params={b['params']} "
                      f"status={b['status']}", ""]
        lines += ["| round | strategy | hash | params | status | L_pde | "
                  "L_e | score | veto |", "|---|---|---|---|---|---|---|---|---|"]
        for e in self.pool.entries:
            lines.append(
                f"| {e.get('round')} | {e.get('strategy')} | "
                f"`{e['hash'][:8]}` | {e.get('param_count')} | "
                f"{e.get('status')} | {_f(e.get('L_pde'))} | "
                f"{_f(e.get('L_energy'))} | {_f(e.get('score'))} | "
                f"{','.join(e.get('veto') or []) or '-'} |")
        return "\n".join(lines) + "\n"


def _f(v, nd: int = 4) -> str:
    return "-" if v is None else f"{float(v):.{nd}g}"


def run_episode(episode: str, **kw) -> dict:
    return Orchestrator(episode, **kw).run()
