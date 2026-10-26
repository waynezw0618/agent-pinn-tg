"""Prompts for the Creator role.

Design notes
------------
* The system prompt carries the *whole* contract (schema, ranges, the physics
  meaning of each knob, the output format).  Anything left implicit is a
  free rejection, and the rejection rate is a metric we report, so the prompt
  is treated as part of the method rather than boilerplate.
* Three strategies, mirroring LLM4eD's alternation between *self-improvement*
  (refine a known-good candidate) and *evolution* (recombine two parents):
  ``literature_seed``, ``self_improve``, ``evolution``.  The orchestrator
  rotates them so a round never repeats the previous mode.
* Retrieved passages are quoted with their ``chunk_id`` so the episode log can
  point at the exact paragraph that motivated a proposal (provenance).
"""
from __future__ import annotations

import json

STRUCTURE_CONTRACT = """\
You output ONE JSON object and nothing else (no prose, no markdown fence):

{
  "rationale": ">=40 chars, cite the mechanism you expect to matter",
  "structure": {
    "schema_version": 2,
    "lam": <float, MUST equal __LAM__>,
    "m": <int, MUST equal __M__>,
    "trunk": {
      "width": <int 8..128>, "depth": <int 1..6>,
      "activation": "tanh" | "sin" | "gelu",
      "skip": "none" | "residual",
      "norm": "none" | "layernorm"
    },
    "heads": {
      "mode": "shared" | "grouped" | "per_mode" | "tied_grouped",
      "group_size": <int, must divide __M__ when mode="grouped">,
      "groups": [[<mode ids 1..__M__>], ...]  (partition of 1..__M__; required
                 unless mode="shared"),
      "tie_weights": <bool>,
      "rank": <int 1..__M__; only used by "tied_grouped">,
      "out_scale": <float 0.001..0.2>
    },
    "wall": {
      "type": "power2" | "power2_plus_lag" | "exp_blend",
      "params": {} | {"a": <float>}      // power2_plus_lag: a in [-0.5,0.5];
                                          // exp_blend: a in [-1,1]
    },
    "mask": {"active_modes": [<mode ids 1..__M__>, ...]},
    "init": {"scheme": "xavier_normal", "seed": 0},
    "freeze": {"trunk": false, "heads": false}
  },
  "expected_effect": "one sentence on which diagnostic should improve",
  "confidence": <float 0..1>
}

Hard rules:
- `lam` and `m` are FIXED by the problem. Any other value is rejected.
- `groups` must partition {1..__M__} exactly, with no empty group.
- Parameter count must stay <= __MAXPARAMS__. Budgets below are approximate:
  the head is only ~11% of the parameters, so `width` dominates the count.
- Do NOT reuse a structure that already exists in the pool; it is rejected as a
  duplicate.

What the knobs mean physically (streamwise-rotating channel, Taylor-Gortler
rolls, spectral-in-z stream function):
- `width`/`depth`: capacity of the coefficient functions a_k(y), b_k(y).
- `activation`: tanh is smooth and trains well for boundary layers; sin gives
  spectral-like fits of oscillatory wall layers; gelu sits in between.
- `heads.mode`: `shared` = one linear map to all 2m coefficients (the current
  baseline); `grouped`/`per_mode` = dedicated maps per mode or per mode group;
  `tied_grouped` = a shared low-rank basis of rank r, the only mode that
  actually *reduces* head parameters.
- `wall.type`: the hard no-slip factor. `power2` = (1-y^2)^2 (baseline,
  vanishes quadratically); `power2_plus_lag` adds a lag term for a thinner
  viscous sublayer; `exp_blend` blends an exponential decay for a sharper
  near-wall peak.
- `mask.active_modes`: dropping high-k modes regularises when the DNS shows no
  energy there, but at high rotation number the rolls thin and carry energy at
  k=3..6, so dropping them is not automatically safe.
"""


SYSTEM_CREATOR = """\
You are the Creator agent in a two-agent structure-search loop for a
physics-informed neural network that solves the Taylor-Gortler secondary flow
in a streamwise-rotating turbulent channel (Re_tau = 180).

A separate, non-LLM Inspector agent numerically audits every trained network
against 12 physical constraints (z-periodicity, zero z-mean, no-slip, continuity,
vorticity identity, energy calibration, amplitude, vortex-pair count, spectral
concentration). You never see the DNS data and you cannot run training. Your job
is to propose network *structures* -- not weights -- whose inductive bias makes
those constraints easy to satisfy and the residual small.

You will be shown: (a) retrieved paragraphs from the project's literature
corpus, (b) the current Pareto front of trained candidates with their metrics,
(c) for a revision round, the audit failures of one specific candidate.

{contract}

Prefer structural changes you can justify with a mechanism. Do not propose a
random search; the Inspector will catch nonsense but it will also be wasted
compute.
"""


def _fmt_front(rows: list[dict], limit: int = 6) -> str:
    if not rows:
        return "(no trained candidate yet -- this is the first round)"
    lines = []
    for r in rows[:limit]:
        p = r.get("compliance")
        lines.append(
            "- {run}: L_pde={lp} L_energy={le} params={n} "
            "compliance={p} score={sc} | {desc}".format(
                run=r.get("run_name") or r.get("candidate_id"),
                lp=_num(r.get("L_pde"), 3),
                le=_num(r.get("L_energy"), 4),
                n=r.get("param_count"),
                p="n/a" if p is None else f"{float(p):.2f}",
                sc=_num(r.get("score"), 3),
                desc=(r.get("description") or "")[:150]))
    return "\n".join(lines)


def _num(v, nd: int) -> str:
    return "n/a" if v is None else f"{float(v):.{nd}f}"


def _fmt_literature(passages: list[dict], limit: int = 4) -> str:
    if not passages:
        return "(no passage retrieved)"
    out = []
    for p in passages[:limit]:
        text = " ".join(str(p.get("text", "")).split())[:600]
        out.append(f"[{p.get('chunk_id')}] ({p.get('doc_id')}) {text}")
    return "\n\n".join(out)


def _fmt_audit(audit: dict | None) -> str:
    if not audit:
        return "(no audit available)"
    hard = audit.get("failures") or []
    soft = audit.get("soft_failures") or []
    return ("hard failures: " + (", ".join(hard) if hard else "none") +
            "\nsoft failures: " + (", ".join(soft) if soft else "none"))


def build_messages(strategy: str, *, task: dict, front: list[dict],
                   passages: list[dict], parent: dict | None = None,
                   second_parent: dict | None = None,
                   audit: dict | None = None, n: int = 1,
                   feedback: str = "") -> list[dict]:
    contract = (STRUCTURE_CONTRACT
                .replace("__LAM__", str(task["lam"]))
                .replace("__M__", str(task["m"]))
                .replace("__MAXPARAMS__", str(task.get("max_params", 12000))))
    system = SYSTEM_CREATOR.format(contract=contract)
    head = [
        f"Problem: Re_tau=180, Ro_tau={task.get('ro', 30)}, "
        f"lam={task['lam']} (rotation number of the case), "
        f"z-spectrum truncation m={task['m']}.",
        f"Current Pareto front:\n{_fmt_front(front)}",
        f"Retrieved literature:\n{_fmt_literature(passages)}",
    ]
    if feedback:
        head.append(f"Notes from the previous round:\n{feedback}")

    if strategy == "self_improve" and parent:
        body = (
            "TASK: revise exactly one existing candidate (self-improvement).\n"
            f"It has this structure:\n{json.dumps(parent, ensure_ascii=False)}\n"
            f"Its audit result:\n{_fmt_audit(audit)}\n\n"
            "Change at most three fields, chosen because they attack the "
            "reported failure. Keep everything else identical. Explain in "
            "`rationale` which failure each change addresses.")
    elif strategy == "evolution" and parent and second_parent:
        body = (
            "TASK: recombine two parents (evolution).\n"
            f"Parent A:\n{json.dumps(parent, ensure_ascii=False)}\n"
            f"Parent B:\n{json.dumps(second_parent, ensure_ascii=False)}\n\n"
            "Take the trunk from one and the head/wall organisation from the "
            "other, or merge them in a way you can justify. The child must be "
            "valid on its own; do not just average numbers.")
    else:
        body = (
            "TASK: propose a structure grounded in the retrieved literature "
            "(literature seed). Name the specific mechanism in `rationale` "
            "and say which of the 12 constraints it should help.")
    body += f"\n\nReturn exactly {n} JSON object{'s' if n > 1 else ''}."
    if n > 1:
        body += (" If you return several, wrap them as "
                 '{"proposals": [ {...}, {...} ]}.')
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(head + [body])}]


def parse_multi(payload: str) -> list:
    """Accept either a single proposal object or a ``{"proposals": [...]}``."""
    from .llm import extract_json
    obj = extract_json(payload)
    if obj is None:
        return [payload]
    if isinstance(obj.get("proposals"), list):
        return obj["proposals"] or [payload]
    return [obj]
