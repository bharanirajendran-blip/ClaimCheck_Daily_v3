"""
Shared data models for ClaimCheck Daily.
Uses Pydantic BaseModel for validation, serialization, and schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Verdict enum ──────────────────────────────────────────────────────────────

class VerdictLabel(str, Enum):
    TRUE         = "TRUE"
    MOSTLY_TRUE  = "MOSTLY_TRUE"
    MIXED        = "MIXED"
    MOSTLY_FALSE = "MOSTLY_FALSE"
    FALSE        = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ── Core models ───────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single checkable claim harvested from a feed."""
    id:           str           = Field(default_factory=_new_id)
    text:         str
    source:       str
    published_at: Optional[str] = None
    feed_name:    Optional[str] = None
    url:          Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text must not be empty.")
        return v.strip()


class ResearchResult(BaseModel):
    """Raw findings returned by the Claude Researcher."""
    claim_id: str
    findings: str
    sources:  list[dict] = Field(default_factory=list)
    fetched_pages: list[dict] = Field(default_factory=list)


class Verdict(BaseModel):
    """Structured verdict produced by the GPT Director."""
    claim_id:     str
    verdict:      VerdictLabel
    confidence:   float      = Field(..., ge=0.0, le=1.0)
    summary:      str
    key_evidence: list[str]  = Field(default_factory=list)


# ── RAG + Verification models (additions) ────────────────────────

class EvidenceChunk(BaseModel):
    """
    One chunk of evidence stored from a researched claim.
    Chunks are persisted across runs so the retriever can draw on
    evidence accumulated over multiple daily cycles (— Knowledge
    Representation / persistent context management).
    """
    chunk_id:   str
    claim_id:   str           # which claim produced this chunk
    claim_text: str           # the original claim headline (for graph edges)
    source_url: str           # URL the content came from
    section:    str           # logical section label (e.g. "Evidence Assessment")
    text:       str           # chunk body (≤ 700 chars)
    date_slug:  str           # YYYY-MM-DD run date
    chunk_kind: str = "summary"  # raw_source | summary | source_metadata


class RetrievalHit(BaseModel):
    """
    One ranked result from the hybrid retriever.
    Carries both raw scores and the originating chunk so the Director
    can cite the exact section that supports its verdict .
    """
    chunk:         EvidenceChunk
    vector_score:  float   # TF-IDF cosine similarity (normalised 0-1)
    keyword_score: float   # BM25-style term overlap  (normalised 0-1)
    hybrid_score:  float   # 0.6 * vector + 0.4 * keyword


class VerifierReport(BaseModel):
    """
    Output of the LLM-as-a-Judge pass .
    Four rubric dimensions map directly to the RAGAS framework:
      groundedness     → faithfulness
      citation_score   → answer correctness
      no_contradiction → consistency
      no_assumption    → hallucination safety
    """
    groundedness_score:     float = Field(..., ge=0.0, le=1.0)
    citation_score:         float = Field(..., ge=0.0, le=1.0)
    no_contradiction_score: float = Field(..., ge=0.0, le=1.0)
    no_assumption_score:    float = Field(..., ge=0.0, le=1.0)
    overall_score:          float = Field(..., ge=0.0, le=1.0)

    unsupported_statements: list[str] = Field(default_factory=list)
    contradictions:         list[str] = Field(default_factory=list)
    missing_citations:      list[str] = Field(default_factory=list)
    rewrite_suggestion:     str       = ""
    should_retry:           bool      = False


class ReviewQueueItem(BaseModel):
    """Persisted human-review task created for low-confidence claims."""
    review_id:       str
    claim_id:        str
    claim_text:      str
    source:          str = ""
    date_slug:       str
    verdict:         VerdictLabel
    confidence:      float = Field(..., ge=0.0, le=1.0)
    verifier_score:  Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason:          str
    status:          ReviewStatus = ReviewStatus.PENDING_REVIEW
    created_at:      str = Field(default_factory=_utcnow)


class DailyReport(BaseModel):
    """Complete report for one publishing cycle."""
    claims:           list[Claim]                     = Field(default_factory=list)
    verdicts:         list[Verdict]                   = Field(default_factory=list)
    # RAG evidence and verifier scores
    retrieval_hits:   dict[str, list[RetrievalHit]]   = Field(default_factory=dict)
    verifier_reports: dict[str, VerifierReport]        = Field(default_factory=dict)
    review_queue:     dict[str, ReviewQueueItem]       = Field(default_factory=dict)
    generated_at:     str                             = Field(default_factory=_utcnow)
    date_slug:        str                             = Field(default_factory=_today_slug)

    def get_verdict(self, claim_id: str) -> Optional[Verdict]:
        return next((v for v in self.verdicts if v.claim_id == claim_id), None)

    def get_review(self, claim_id: str) -> Optional[ReviewQueueItem]:
        return self.review_queue.get(claim_id)


# ── LangGraph pipeline state ──────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Typed state object passed between every LangGraph node.
    Each node receives the full state, updates its own fields, and returns it.
    Pydantic ensures every mutation is validated at runtime.
    """
    # Runtime config (injected once at graph invocation)
    feeds_path:  str = "feeds.yaml"
    docs_dir:    str = "docs"
    outputs_dir: str = "outputs"
    max_workers: int = 1  # sequential to stay within 30k tokens/min rate limit

    # Data fields populated progressively by each node
    candidates:       list[Claim]               = Field(default_factory=list)
    selected:         list[Claim]               = Field(default_factory=list)
    research_results: dict[str, ResearchResult] = Field(default_factory=dict)
    verdicts:         list[Verdict]             = Field(default_factory=list)
    report:           Optional[DailyReport]     = None

    # RAG + verification state
    evidence_chunks:  dict[str, list[EvidenceChunk]]  = Field(default_factory=dict)
    retrieval_hits:   dict[str, list[RetrievalHit]]   = Field(default_factory=dict)
    graph_context:    dict[str, str]                  = Field(default_factory=dict)
    verifier_reports: dict[str, VerifierReport]        = Field(default_factory=dict)
    retry_counts:     dict[str, int]                  = Field(default_factory=dict)
    review_queue:     dict[str, ReviewQueueItem]       = Field(default_factory=dict)
