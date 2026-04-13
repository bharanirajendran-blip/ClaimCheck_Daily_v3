"""
Director — GPT-4o orchestration layer
--------------------------------------
Responsibilities:
  1. Score & rank candidate claims by newsworthiness / checkability
  2. Dispatch selected claims to the Claude Researcher
  3. Synthesise research results into validated Pydantic Verdict objects
  4. Assemble the final DailyReport
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from .models import (
    Claim,
    DailyReport,
    ResearchResult,
    RetrievalHit,
    ReviewQueueItem,
    Verdict,
    VerdictLabel,
    VerifierReport,
)
from .observability import get_tracer
from .utils import retry

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Director of ClaimCheck Daily, a rigorous automated fact-checking service.
Your job is to:
1. Select the most impactful, verifiable claims from the candidate list.
2. After a researcher provides evidence, produce a clear verdict with a confidence score.

When selecting claims, follow these rules:

DIVERSITY RULES (most important):
- Select at most 1 claim per source. Never pick 2 claims from the same outlet.
- Pick claims from at least 2 different topics (e.g. one political, one science/health, one technology)

PREFER claims where:
- A public figure or viral post said something that might be false or misleading (PolitiFact, FactCheck.org)
- A specific statistic, number, or named fact can be verified (AP News, Science Daily)
- A health or technology claim makes a concrete, checkable assertion

AVOID:
- Straightforward announcements from official bodies (e.g. "WHO releases report")
- Claims where the source URL is behind a paywall (Reuters.com, Nature.com)
- Picking more than 1 claim from the same news outlet

The goal is a balanced, varied daily report — not a single-topic feed.

For verdict generation:
- Treat retrieved evidence chunks as the primary source of truth.
- Prefer raw source chunks over researcher summaries when there is tension.
- Do not make claims that are not supported by the retrieved evidence.
- Write key_evidence bullets so they reflect specific retrieved support, not vague summaries.

Always respond with valid JSON matching the schema provided in each user message."""

SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of claim IDs chosen for fact-checking (max 3)",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["selected", "reasoning"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [v.value for v in VerdictLabel],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "summary", "key_evidence"],
}


class Director:
    """GPT-4o powered Director agent."""

    def __init__(
        self,
        model: str | None = None,
        max_claims_per_day: int | None = None,
    ):
        # Both params can be overridden via env vars so the same codebase runs
        # across dev / CI / different GPT tiers without touching source files.
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_claims_per_day = max_claims_per_day or int(
            os.getenv("MAX_CLAIMS_PER_DAY", "3")
        )
        self._client = OpenAI()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_claims(self, candidates: list[Claim]) -> list[Claim]:
        """Ask GPT to rank and pick the best claims for today."""
        logger.info("Director selecting from %d candidates…", len(candidates))
        payload = [{"id": c.id, "text": c.text, "source": c.source} for c in candidates]

        response = self._chat(
            f"Select up to {self.max_claims_per_day} claims worth fact-checking today.\n\n"
            f"Candidates:\n{json.dumps(payload, indent=2)}\n\n"
            f"Respond with JSON matching schema:\n{json.dumps(SELECTION_SCHEMA)}"
        )

        selected_ids = set(response.get("selected", []))
        chosen = [c for c in candidates if c.id in selected_ids]
        logger.info("Director selected %d claims.", len(chosen))
        return chosen

    def synthesize_verdict(
        self,
        claim: Claim,
        research: ResearchResult,
        hits: list[RetrievalHit] | None = None,
    ) -> Verdict:
        """
        Turn raw research into a Pydantic-validated Verdict.
        When retrieval hits are available (RAG path), the top evidence chunks
        are injected into the prompt so the Director can ground its verdict
        directly in the retrieved passages rather than relying solely on the
        researcher's free-form findings text.
        """
        logger.info("Director synthesising verdict for claim %s…", claim.id)

        rag_section = ""
        if hits:
            raw_hits = [h for h in hits if h.chunk.chunk_kind == "raw_source"][:4]
            support_hits = [h for h in hits if h.chunk.chunk_kind != "raw_source"][:2]
            chosen_hits = raw_hits + [h for h in support_hits if h.chunk.chunk_id not in {r.chunk.chunk_id for r in raw_hits}]
            chunk_lines = "\n\n".join(
                f"[{h.chunk.chunk_id}] kind={h.chunk.chunk_kind} section={h.chunk.section} "
                f"source={h.chunk.source_url or 'unknown'} score={h.hybrid_score:.3f}\n{h.chunk.text}"
                for h in chosen_hits[:6]
            )
            rag_section = (
                "\n\nTop retrieved evidence chunks (use these as your main grounding context; "
                "raw_source chunks are strongest):\n"
                f"{chunk_lines}"
            )

        response = self._chat(
            f"Claim: {claim.text}\n\n"
            f"Research findings (secondary context only):\n{research.findings}\n\n"
            f"Sources consulted:\n{json.dumps(research.sources, indent=2)}"
            f"{rag_section}\n\n"
            f"Respond with JSON matching schema:\n{json.dumps(VERDICT_SCHEMA)}"
        )

        # Parse verdict enum — fall back to UNVERIFIABLE rather than crashing
        # if GPT returns a value that isn't in the enum (e.g. a typo or a new
        # label the model invented).  The bad value is logged so it's auditable.
        raw_verdict = response.get("verdict", "")
        try:
            verdict_label = VerdictLabel(raw_verdict)
        except ValueError:
            logger.warning(
                "Director returned unknown verdict label %r for claim %s; "
                "defaulting to UNVERIFIABLE.",
                raw_verdict, claim.id,
            )
            verdict_label = VerdictLabel("UNVERIFIABLE")

        return Verdict(
            claim_id=claim.id,
            verdict=verdict_label,
            confidence=float(response.get("confidence", 0.0)),
            summary=response.get("summary", "Verdict could not be determined."),
            key_evidence=response.get("key_evidence", []),
        )

    def build_report(
        self,
        verdicts: list[Verdict],
        claims: list[Claim],
        retrieval_hits: dict[str, list[RetrievalHit]] | None = None,
        verifier_reports: dict[str, VerifierReport] | None = None,
        review_queue: dict[str, ReviewQueueItem] | None = None,
    ) -> DailyReport:
        """Assemble the final DailyReport Pydantic model."""
        claim_map = {c.id: c for c in claims}
        return DailyReport(
            verdicts=verdicts,
            claims=[claim_map[v.claim_id] for v in verdicts if v.claim_id in claim_map],
            retrieval_hits=retrieval_hits or {},
            verifier_reports=verifier_reports or {},
            review_queue=review_queue or {},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(times=3, delay=2)
    def _chat(self, user_content: str) -> dict[str, Any]:
        with get_tracer().span("director", model=self.model) as span:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            span.record_openai(response)
        return json.loads(response.choices[0].message.content)
