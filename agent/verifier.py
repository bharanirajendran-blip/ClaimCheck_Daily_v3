"""
verifier.py — LLM-as-a-Judge 

Runs a second GPT-4o call that independently evaluates the Director's
Verdict against the retrieved evidence chunks. This is the "LLM-as-a-Judge"
pattern required by a separate model instance acting as an
objective evaluator of another model's output.

Why a second model call instead of self-evaluation?
  Self-evaluation is known to be overconfident — a model scores its own
  output higher than warranted. Using a separate call (even the same
  model, different context) breaks the coherence bias and produces more
  calibrated scores.

Rubric dimensions (maps to RAGAS framework):
  groundedness     → faithfulness:    every factual claim backed by context
  citation_score   → answer_correct:  cited evidence actually supports claims
  no_contradiction → consistency:     no internal contradictions
  no_assumption    → hallucination:   no unsupported assumptions beyond evidence

When should_retry is True, pipeline.py routes back to retrieve_evidence_node
with a refined query built from missing_citations. This is the self-correcting
generation loop .
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from .models import RetrievalHit, Verdict, VerifierReport
from .observability import get_tracer

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """
You are an LLM-as-a-Judge evaluating an automated fact-checking verdict.

You will receive:
  1. The original claim
  2. The verdict produced by the fact-checking system
  3. The retrieved evidence chunks the system had access to

Score each dimension 0.0–1.0:

  groundedness     How well every factual statement in the summary is
                   directly supported by the retrieved evidence.
                   1.0 = every statement has a clear evidence source.

  citation_score   Whether the key_evidence bullets correctly cite and
                   represent content from the retrieved chunks.
                   1.0 = all citations accurate and present.

  no_contradiction Any contradictions between the verdict summary and
                   the retrieved evidence, or within the summary itself.
                   1.0 = no contradictions found.

  no_assumption    Unsupported assumptions or leaps beyond what the
                   evidence shows.
                   1.0 = verdict stays within what the evidence supports.

Set should_retry to true if groundedness < 0.70 OR citation_score < 0.70.

Return STRICT JSON only — no markdown:
{
  "groundedness_score": <float>,
  "citation_score": <float>,
  "no_contradiction_score": <float>,
  "no_assumption_score": <float>,
  "overall_score": <float>,
  "unsupported_statements": ["...", ...],
  "contradictions": ["...", ...],
  "missing_citations": ["...", ...],
  "rewrite_suggestion": "<string or empty>",
  "should_retry": <bool>
}
""".strip()


def verify_verdict(
    claim_text: str,
    verdict: Verdict,
    hits: list[RetrievalHit],
) -> VerifierReport:
    """
    Run the LLM-as-a-Judge pass. Falls back to heuristic scoring
    if OPENAI_API_KEY is not available (for local smoke tests).
    """
    if os.getenv("OPENAI_API_KEY"):
        return _verify_llm(claim_text, verdict, hits)
    return _verify_fallback(verdict, hits)


# ── LLM path ───────────────────────────────────────────────────────────────────

def _verify_llm(
    claim_text: str,
    verdict: Verdict,
    hits: list[RetrievalHit],
) -> VerifierReport:
    client = OpenAI()

    context_blocks = "\n\n---\n\n".join(
        f"[{h.chunk.chunk_id}] {h.chunk.section} (from: {h.chunk.source_url})\n{h.chunk.text}"
        for h in hits
    )
    verdict_text = json.dumps({
        "verdict":      verdict.verdict,
        "confidence":   verdict.confidence,
        "summary":      verdict.summary,
        "key_evidence": verdict.key_evidence,
    }, indent=2)

    _model = os.getenv("OPENAI_MODEL", "gpt-4o")
    with get_tracer().span("verify_node", model=_model) as span:
        response = client.chat.completions.create(
            model=_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{claim_text}\n\n"
                        f"Verdict to evaluate:\n{verdict_text}\n\n"
                        f"Retrieved evidence:\n{context_blocks}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0.0,
        )
        span.record_openai(response)
        span.set_extra(claim_id=verdict.claim_id)

    payload = _parse_json(response.choices[0].message.content or "{}")

    # Compute overall as weighted average if not supplied by model
    if "overall_score" not in payload or not isinstance(payload.get("overall_score"), float):
        payload["overall_score"] = round(
            0.35 * payload.get("groundedness_score", 0)
            + 0.35 * payload.get("citation_score", 0)
            + 0.15 * payload.get("no_contradiction_score", 0)
            + 0.15 * payload.get("no_assumption_score", 0),
            3,
        )

    logger.info(
        "[verifier] claim=%s overall=%.2f retry=%s",
        verdict.claim_id,
        payload.get("overall_score", 0),
        payload.get("should_retry", False),
    )
    return VerifierReport.model_validate(payload)


# ── Fallback (no API key) ──────────────────────────────────────────────────────

def _verify_fallback(verdict: Verdict, hits: list[RetrievalHit]) -> VerifierReport:
    """
    Score-based heuristic so the pipeline runs end-to-end without an API key.
    Mirrors the rubric logic so the retry loop still exercises.
    """
    n_hits  = len(hits)
    n_evid  = len(verdict.key_evidence)

    groundedness = min(1.0, 0.4 + 0.12 * n_evid)
    citation     = 0.85 if n_evid > 0 else 0.25
    no_contra    = 0.90
    no_assume    = 0.80 if n_hits >= 3 else 0.55

    overall = round(
        0.35 * groundedness + 0.35 * citation
        + 0.15 * no_contra + 0.15 * no_assume, 3
    )

    unsupported: list[str] = []
    if n_evid == 0:
        unsupported.append("No key_evidence bullets — verdict cannot be verified against context.")
    if n_hits < 2:
        unsupported.append("Fewer than 2 retrieved chunks; verdict may extend beyond evidence.")

    should_retry = groundedness < 0.70 or citation < 0.70
    rewrite = (
        "Re-retrieve with the specific entities and sources mentioned in the verdict, "
        "then rewrite key_evidence using only cited chunk content."
        if should_retry else ""
    )

    return VerifierReport(
        groundedness_score=round(groundedness, 2),
        citation_score=round(citation, 2),
        no_contradiction_score=round(no_contra, 2),
        no_assumption_score=round(no_assume, 2),
        overall_score=overall,
        unsupported_statements=unsupported,
        contradictions=[],
        missing_citations=verdict.key_evidence[:1] if not n_evid else [],
        rewrite_suggestion=rewrite,
        should_retry=should_retry,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON in verifier response: {text[:200]}")
    return json.loads(text[s: e + 1])
