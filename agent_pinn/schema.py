"""Validation gate for LLM proposals.

The Inspector of Tacke et al. (2026) is *numerical* -- it probes the trained
network.  This module is the cheaper first gate that runs before any compute is
spent: it turns a free-form completion into a structure dict or a coded
rejection.  Keeping the two gates separate is the whole point -- a model that
hallucinates ``"activation": "relu"`` should cost us one string comparison, not
a 10-minute training.

Rejection codes are stable and counted per episode, so the *structural
compliance rate* (accepted / proposed) is a reportable number rather than an
anecdote.  Tacke et al. report 37% -> 56% compliance from adding this split.
"""
from __future__ import annotations

import json

from pinn_tg.structure import (MAX_PARAMS, normalize_structure, param_count,
                               structure_hash, validate_structure)

MIN_PARAMS = 256
MIN_RATIONALE = 20


class Rejection(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _as_obj(payload) -> dict:
    if isinstance(payload, str):
        from .llm import extract_json
        obj = extract_json(payload)
        if obj is None:
            raise Rejection("R_BAD_JSON", payload[:120])
        return obj
    if isinstance(payload, dict):
        return payload
    raise Rejection("R_NOT_OBJECT", type(payload).__name__)


def validate_proposal(payload, *, task: dict, seen: set[str] | None = None,
                      require_rationale: bool = True) -> dict:
    """Return ``{"structure", "hash", "rationale", "expected_effect",
    "confidence", "param_count"}`` or raise :class:`Rejection`."""
    obj = _as_obj(payload)

    struct = obj.get("structure") or obj.get("ansatz") or obj.get("config")
    if not isinstance(struct, dict):
        raise Rejection("R_NO_STRUCTURE", "missing 'structure' object")

    rationale = str(obj.get("rationale") or "").strip()
    if require_rationale and len(rationale) < MIN_RATIONALE:
        raise Rejection("R_NO_RATIONALE",
                        f"{len(rationale)} chars < {MIN_RATIONALE}")

    try:
        cand = normalize_structure(struct)
    except Exception as exc:  # noqa: BLE001 - malformed JSON shapes land here
        raise Rejection("R_NORMALIZE", f"{type(exc).__name__}: {exc}"[:160])
    # m and lam define the *problem discretisation*, not the network, so they
    # are locked to the episode task; changing them would make L_pde values
    # incomparable across candidates.
    if int(cand["m"]) != int(task["m"]):
        raise Rejection("R_M_LOCKED",
                        f"m={cand['m']} != task m={task['m']}")
    if abs(float(cand["lam"]) - float(task["lam"])) > 1e-9:
        raise Rejection("R_LAM_LOCKED",
                        f"lam={cand['lam']} != task lam={task['lam']}")

    # init.seed never reaches the initialiser (train.py seeds torch/numpy from
    # --seed) but it *does* enter structure_hash, so a proposer could mint
    # "new" candidates that train identically.  freeze likewise leaves frozen
    # parameters counted in param_count without them being searchable.  Both
    # are anti-gaming guards, not style rules.
    if int(cand.get("init", {}).get("seed", 0)) != 0:
        raise Rejection("R_INIT_SEED",
                        "init.seed is inert (use --seed); nonzero only "
                        "forges a new hash")
    fr = cand.get("freeze", {})
    if fr.get("trunk") or fr.get("heads"):
        raise Rejection("R_FREEZE",
                        "frozen params still count toward param_count")

    errs = validate_structure(cand)
    if errs:
        raise Rejection("R_SCHEMA", ";".join(errs))

    n = param_count(cand)
    if n > MAX_PARAMS:
        raise Rejection("R_PARAM_BUDGET", f"{n} > {MAX_PARAMS}")
    if n < MIN_PARAMS:
        raise Rejection("R_PARAM_FLOOR", f"{n} < {MIN_PARAMS}")

    h = structure_hash(cand)
    if seen and h in seen:
        raise Rejection("R_DUPLICATE", h)
    if cand["heads"]["mode"] == "tied_grouped" and \
            int(cand["heads"]["rank"]) > int(task["m"]):
        raise Rejection("R_RANK", f"rank={cand['heads']['rank']} > m")

    conf = obj.get("confidence", 0.5)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5

    return {
        "structure": cand,
        "hash": h,
        "rationale": rationale,
        "expected_effect": str(obj.get("expected_effect") or "")[:400],
        "confidence": conf,
        "param_count": n,
        "raw": json.dumps(obj, ensure_ascii=False)[:4000],
    }


def summarize_rejections(rejections: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rejections:
        counts[str(r["code"]).split(":")[0]] = \
            counts.get(str(r["code"]).split(":")[0], 0) + 1
    total = len(rejections)
    return {"n_rejected": total, "by_code": counts}
