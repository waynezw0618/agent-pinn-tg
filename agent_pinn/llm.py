"""LLM backends for the Creator role.

Two providers, one interface:

``OpenAICompatProvider``
    Any OpenAI-compatible ``/chat/completions`` endpoint.  Default is Zhipu
    GLM-5.3 (``https://open.bigmodel.cn/api/paas/v4``), the same backend the
    LGF symbolic-regression stage already uses, so the two halves of the
    pipeline share one model.

``RuleBasedProvider``
    A deterministic, offline "LLM".  It receives the same ``context`` dict as
    the real model and emits a structurally valid proposal by mutating a
    parent or sampling the structure space.  It is *not* a stand-in for
    reasoning -- it exists so the closed loop, the audit gate and the Pareto
    bookkeeping can be exercised and unit-tested with the network unplugged,
    and so an episode never dead-ends on a missing key.

Key resolution order (first hit wins, the value is never logged)::

    AGENT_PINN_LLM_API_KEY > LGF_LLM_API_KEY > ZHIPU_API_KEY > OPENAI_API_KEY
    > configs/llm_key.txt > ~/.config/evoscientist/config.yaml:zhipu_api_key
"""
from __future__ import annotations

import json
import os
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from . import config, paths

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_KEY_FILES = ("configs/llm_key.txt",)


# ---------------------------------------------------------------------------
def _from_yaml(path: Path, fields: tuple[str, ...]) -> str:
    """Return the first value matching *fields*, **field order wins over
    line order** so ``zhipu_api_key`` is preferred over an ``openai_api_key``
    that merely appears earlier in the file."""
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    for field in fields:
        for line in lines:
            m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*[\"']?([^\"'\s#]+)", line)
            if m and m.group(1) == field:
                return m.group(2)
    return ""


def find_api_key() -> tuple[str, str]:
    """Return ``(key, source)``; ``("", "none")`` when nothing is configured."""
    for env in ("AGENT_PINN_LLM_API_KEY", "LGF_LLM_API_KEY", "ZHIPU_API_KEY",
                "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val, f"env:{env}"
    for rel in _KEY_FILES:
        p = paths.ROOT / rel
        if p.is_file():
            val = p.read_text(encoding="utf-8", errors="replace").strip()
            if val:
                return val.splitlines()[0].strip(), f"file:{rel}"
    home = Path(os.path.expanduser("~")) / ".config" / "evoscientist" \
        / "config.yaml"
    val = _from_yaml(home, ("zhipu_api_key", "openai_api_key"))
    if val:
        return val, f"file:{home}"
    return "", "none"


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a chatty completion."""
    if not text:
        return None
    raw = strip_reasoning(text)
    for m in _FENCE.finditer(raw):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(obj, dict):
                    return obj
                start = None
    return None


# ---------------------------------------------------------------------------
@dataclass
class BaseProvider:
    name: str = "base"
    model: str = ""
    calls: int = 0
    errors: int = 0
    last_error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def chat(self, messages: list[dict], *, context: dict | None = None,
             temperature: float | None = None,
             max_tokens: int = 3000) -> str:
        raise NotImplementedError

    def stats(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "calls": self.calls, "errors": self.errors,
                "last_error": self.last_error}


class RuleBasedProvider(BaseProvider):
    """Deterministic proposal generator (no network, no randomness leak).

    ``context`` must carry ``strategy`` and, for the revision/evolution
    branches, ``parents`` (list of structure dicts).  The output mimics the
    real provider's JSON so downstream validation is identical.
    """

    name = "rule"

    def __init__(self) -> None:
        super().__init__(name="rule", model="deterministic")

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages: list[dict], *, context: dict | None = None,
             temperature: float | None = None, max_tokens: int = 3000) -> str:
        ctx = context or {}
        self.calls += 1
        strategy = ctx.get("strategy", "self_improve")
        n = max(1, int(ctx.get("n", 1)))
        outs = [json.loads(self._one(strategy, ctx, i)) for i in range(n)]
        return json.dumps(outs[0] if n == 1 else {"proposals": outs},
                          ensure_ascii=False)

    def _one(self, strategy: str, ctx: dict, i: int) -> str:
        import random

        from pinn_tg.structure import (MUTATIONS, crossover, sample_valid)

        parents = [p for p in (ctx.get("parents") or []) if p]
        seed = int(ctx.get("seed", 0))
        idx = int(ctx.get("index", 0))
        # ``hash(str)`` is salted per process, so it must not enter the seed.
        salt = zlib.crc32(strategy.encode("utf-8")) % 997
        rng = random.Random(1000 * seed + 17 * idx + salt * (1 + i))
        base = parents[0] if parents else ctx.get("base")

        if strategy == "evolution" and len(parents) >= 2:
            cand = crossover(parents[0], parents[1], rng)
            op = "crossover"
        elif strategy == "literature_seed" or not base:
            cand = sample_valid(base or ctx["base"], rng, n_ops=2)
            op = "sample"
        else:
            op = MUTATIONS[rng.randrange(len(MUTATIONS))]
            cand = _guarded_mutate(base, op, rng)
            op = f"mutate:{op}"
        return json.dumps({
            "rationale": f"rule-based {strategy} via {op}",
            "structure": cand,
            "expected_effect": "exploratory",
            "confidence": 0.2,
        }, ensure_ascii=False)


def _guarded_mutate(cfg: dict, op: str, rng) -> dict:
    from pinn_tg.structure import mutate, normalize_structure, validate_structure
    for _ in range(6):
        try:
            cand = normalize_structure(mutate(cfg, op, rng))
        except Exception:  # noqa: BLE001
            continue
        if not validate_structure(cand):
            return cand
    return normalize_structure(cfg)


class OpenAICompatProvider(BaseProvider):
    """OpenAI-compatible chat endpoint (GLM-5.3 by default)."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout_s: float = 90.0, temperature: float = 0.4,
                 max_tokens: int = 3000, extra_body: dict | None = None):
        super().__init__(name="openai-compat", model=model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        # glm-5.3 always reasons; ``reasoning_effort=max`` burns the whole
        # completion budget on hidden tokens and returns empty content (same
        # lesson as LGF's OpenAICompatLLM).  Ask for a low effort explicitly.
        self.extra_body = extra_body if extra_body is not None else (
            {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
            if "bigmodel" in self.base_url else None)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def url(self) -> str:
        return self.base_url + "/chat/completions"

    def chat(self, messages: list[dict], *, context: dict | None = None,
             temperature: float | None = None,
             max_tokens: int = 3000) -> str:
        import requests

        self.calls += 1
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature if temperature is None
                else temperature,
                "max_tokens": int(max_tokens), "stream": False}
        if self.extra_body:
            body.update(self.extra_body)
        t0 = time.time()
        try:
            r = requests.post(self.url,
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Content-Type": "application/json"},
                              json=body, timeout=self.timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = strip_reasoning(msg.get("content") or "")
            if not content:
                raise RuntimeError(
                    f"empty content (finish={choice.get('finish_reason')})")
            self.extra["last_latency_s"] = round(time.time() - t0, 2)
            self.extra["last_usage"] = data.get("usage") or {}
            return content
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            raise


# ---------------------------------------------------------------------------
def get_provider(spec: str | None = None, **kw) -> BaseProvider:
    """Build a provider from a short spec: ``auto``/``glm``/``rule``/URL."""
    spec = (spec or config.llm_settings()["provider"] or "auto").strip().lower()
    cfg = config.llm_settings()
    if spec in ("rule", "rules", "offline", "none"):
        return RuleBasedProvider()
    key, source = find_api_key()
    if spec in ("auto", "glm", "zhipu", "glm-5.3") and not key:
        if spec == "auto":
            p = RuleBasedProvider()
            p.last_error = "no api key found; using rule-based provider"
            return p
        raise RuntimeError("no LLM api key found (see agent_pinn.llm docstring)")
    if spec in ("auto", "glm", "zhipu"):
        p = OpenAICompatProvider(cfg["base_url"], cfg["model"], key,
                                 timeout_s=cfg["timeout_s"],
                                 temperature=cfg["temperature"], **kw)
        p.extra["key_source"] = source
        return p
    if spec.startswith("http"):
        return OpenAICompatProvider(spec, cfg["model"], key or "EMPTY",
                                    timeout_s=cfg["timeout_s"], **kw)
    raise ValueError(f"unknown provider spec: {spec}")
