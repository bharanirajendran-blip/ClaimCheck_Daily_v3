"""
observability.py — Lightweight LLM call tracing for ClaimCheck Daily v3.

Zero external dependencies. Wraps every LLM call to capture:
  - model name
  - prompt + completion token counts
  - wall-clock latency (ms)
  - estimated cost (USD) using current published API pricing
  - which pipeline node made the call
  - whether the call succeeded or raised an exception

Trace events are appended to outputs/traces.jsonl (one JSON object per line)
so they accumulate across runs and can be queried with any JSON tooling.

A run-level summary (total tokens, total cost, slowest node) is returned
from Tracer.summary() and stored in the daily JSON report under "trace_summary".

Usage (in any module that makes LLM calls):

    from agent.observability import get_tracer

    tracer = get_tracer()   # returns the process-level singleton

    # Wrap an OpenAI call:
    with tracer.span("verdict_node", model="gpt-4o") as span:
        response = client.chat.completions.create(...)
        span.record_openai(response)

    # Wrap an Anthropic call:
    with tracer.span("research_node", model="claude-opus-4-5") as span:
        response = client.messages.create(...)
        span.record_anthropic(response)

The context manager handles timing and exception recording automatically.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Cost table (USD per 1M tokens, verified 2026-04-13) ──────────────────────
# Sources:
# - https://platform.openai.com/docs/models/gpt-4o
# - https://www.anthropic.com/news/claude-opus-4-5
_COST_PER_M: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
    "chatgpt-4o-latest":        {"input": 5.00,  "output": 15.00},
    "gpt-4-turbo":              {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-opus-4-5":          {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5-20251101": {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":          {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5":        {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":         {"input": 0.80,  "output": 4.00},
    # Legacy aliases still used in some configs
    "claude-3-opus-20240229":   {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00,  "output": 15.00},
    "claude-3-haiku-20240307":  {"input": 0.25,  "output": 1.25},
}
_DEFAULT_COST = {"input": 5.00, "output": 15.00}  # safe fallback


def _canonical_model_name(model: str) -> str:
    """Map dated/aliased model strings onto a stable pricing key."""
    raw = (model or "").split(":")[0].strip()
    if raw in _COST_PER_M:
        return raw

    # Trim trailing date/version suffixes while preserving family names.
    simplified = re.sub(r"-20\d{6,8}$", "", raw)
    if simplified in _COST_PER_M:
        return simplified

    for prefix in (
        "gpt-4o-mini",
        "gpt-4o",
        "chatgpt-4o-latest",
        "gpt-4-turbo",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        if raw.startswith(prefix):
            return prefix

    return raw


def _provider_for_model(model: str) -> str:
    canonical = _canonical_model_name(model)
    if canonical.startswith("gpt-") or canonical.startswith("chatgpt-"):
        return "openai"
    if canonical.startswith("claude-"):
        return "anthropic"
    return "other"


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for one LLM call."""
    rates = _COST_PER_M.get(_canonical_model_name(model), _DEFAULT_COST)
    return round(
        (input_tokens  / 1_000_000) * rates["input"]
        + (output_tokens / 1_000_000) * rates["output"],
        6,
    )


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    node:             str
    model:            str
    started_at:       float          # unix timestamp
    latency_ms:       float  = 0.0
    input_tokens:     int    = 0
    output_tokens:    int    = 0
    total_tokens:     int    = 0
    cost_usd:         float  = 0.0
    success:          bool   = True
    error:            str    = ""
    extra:            dict   = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["started_at_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)
        )
        return d


# ── Span (context manager) ────────────────────────────────────────────────────

class Span:
    """Context manager for a single LLM call. Record tokens after the call."""

    def __init__(self, tracer: "Tracer", node: str, model: str) -> None:
        self._tracer = tracer
        self._event  = TraceEvent(node=node, model=model, started_at=time.time())

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._event.latency_ms = round((time.time() - self._event.started_at) * 1000, 1)
        if exc_type is not None:
            self._event.success = False
            self._event.error   = str(exc_val)[:200]
        self._tracer._record(self._event)
        return False  # don't suppress exceptions

    def record_openai(self, response: Any) -> None:
        """Extract token usage from an OpenAI ChatCompletion response."""
        try:
            usage = response.usage
            self._event.input_tokens  = usage.prompt_tokens
            self._event.output_tokens = usage.completion_tokens
            self._event.total_tokens  = usage.total_tokens
            self._event.cost_usd      = _estimate_cost(
                self._event.model, usage.prompt_tokens, usage.completion_tokens
            )
        except Exception as exc:
            logger.debug("[obs] could not parse OpenAI usage: %s", exc)

    def record_anthropic(self, response: Any) -> None:
        """Extract token usage from an Anthropic Messages response."""
        try:
            usage = response.usage
            inp  = getattr(usage, "input_tokens",  0)
            out  = getattr(usage, "output_tokens", 0)
            self._event.input_tokens  = inp
            self._event.output_tokens = out
            self._event.total_tokens  = inp + out
            self._event.cost_usd      = _estimate_cost(self._event.model, inp, out)
        except Exception as exc:
            logger.debug("[obs] could not parse Anthropic usage: %s", exc)

    def set_extra(self, **kwargs: Any) -> None:
        """Attach arbitrary metadata to the span (e.g. claim_id, retry_count)."""
        self._event.extra.update(kwargs)


# ── Tracer (run-level accumulator) ────────────────────────────────────────────

class Tracer:
    """
    Accumulates TraceEvents for one pipeline run.
    Call summary() at the end of the run to get aggregate stats.
    Pass outputs_dir to flush() to persist events to traces.jsonl.
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def span(self, node: str, model: str) -> Span:
        return Span(self, node, model)

    def _record(self, event: TraceEvent) -> None:
        self._events.append(event)
        status = "✓" if event.success else "✗"
        logger.info(
            "[obs] %s  %s  %s  tokens=%d  cost=$%.4f  latency=%dms",
            status, event.node, event.model,
            event.total_tokens, event.cost_usd, event.latency_ms,
        )

    def summary(self) -> dict:
        """Return aggregate stats across all recorded spans."""
        if not self._events:
            return {"spans": 0}

        total_input   = sum(e.input_tokens  for e in self._events)
        total_output  = sum(e.output_tokens for e in self._events)
        total_tokens  = sum(e.total_tokens  for e in self._events)
        total_cost    = sum(e.cost_usd      for e in self._events)
        total_latency = sum(e.latency_ms    for e in self._events)
        failures      = [e for e in self._events if not e.success]

        # per-node breakdown
        by_node: dict[str, dict] = {}
        by_provider: dict[str, dict] = {}
        by_model: dict[str, dict] = {}
        for e in self._events:
            if e.node not in by_node:
                by_node[e.node] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                }
            by_node[e.node]["calls"]      += 1
            by_node[e.node]["tokens"]     += e.total_tokens
            by_node[e.node]["cost_usd"]   += e.cost_usd
            by_node[e.node]["latency_ms"] += e.latency_ms

            provider = _provider_for_model(e.model)
            if provider not in by_provider:
                by_provider[provider] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_provider[provider]["calls"] += 1
            by_provider[provider]["input_tokens"] += e.input_tokens
            by_provider[provider]["output_tokens"] += e.output_tokens
            by_provider[provider]["total_tokens"] += e.total_tokens
            by_provider[provider]["cost_usd"] += e.cost_usd

            canonical_model = _canonical_model_name(e.model)
            if canonical_model not in by_model:
                by_model[canonical_model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_model[canonical_model]["calls"] += 1
            by_model[canonical_model]["input_tokens"] += e.input_tokens
            by_model[canonical_model]["output_tokens"] += e.output_tokens
            by_model[canonical_model]["total_tokens"] += e.total_tokens
            by_model[canonical_model]["cost_usd"] += e.cost_usd

        # round node stats
        for nd in by_node.values():
            nd["cost_usd"]   = round(nd["cost_usd"],   5)
            nd["latency_ms"] = round(nd["latency_ms"],  1)
        for stats in by_provider.values():
            stats["cost_usd"] = round(stats["cost_usd"], 5)
        for stats in by_model.values():
            stats["cost_usd"] = round(stats["cost_usd"], 5)

        slowest = max(self._events, key=lambda e: e.latency_ms)

        return {
            "spans":         len(self._events),
            "failures":      len(failures),
            "total_input_tokens":  total_input,
            "total_output_tokens": total_output,
            "total_tokens":  total_tokens,
            "total_cost_usd": round(total_cost, 5),
            "total_latency_ms": round(total_latency, 1),
            "slowest_node":  slowest.node,
            "slowest_ms":    slowest.latency_ms,
            "by_node":       by_node,
            "by_provider":   by_provider,
            "by_model":      by_model,
        }

    def flush(self, outputs_dir: str | Path) -> Path:
        """Append all recorded events to outputs_dir/traces.jsonl."""
        path = Path(outputs_dir) / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        logger.info("[obs] Flushed %d trace events → %s", len(self._events), path)
        return path


# ── Process-level singleton ───────────────────────────────────────────────────
# A new Tracer is created per pipeline run via reset_tracer().
# get_tracer() returns the current run's tracer so modules don't need
# to pass it as an argument.

_current_tracer: Tracer = Tracer()


def get_tracer() -> Tracer:
    """Return the current run-level tracer."""
    return _current_tracer


def reset_tracer() -> Tracer:
    """Create a fresh tracer for a new pipeline run. Call at run start."""
    global _current_tracer
    _current_tracer = Tracer()
    return _current_tracer
