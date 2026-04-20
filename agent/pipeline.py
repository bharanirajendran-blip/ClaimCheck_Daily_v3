"""
Pipeline — LangGraph StateGraph orchestration
----------------------------------------------
Graph nodes (extended topology):

  harvest_node       → parse feeds.yaml → state.candidates
  select_node        → Director (GPT) picks top claims → state.selected
  research_node      → Researcher (Claude) investigates each claim
  store_evidence_node → chunk & persist ResearchResults → evidence store
  graph_context_node  → query knowledge graph → cross-run context strings
  retrieve_node       → HybridRetriever (TF-IDF + BM25) → state.retrieval_hits
  verdict_node        → Director synthesises Verdict grounded in retrieved hits
  verify_node         → LLM-as-a-Judge evaluates verdict quality (RAGAS rubric)
  revise_query_node   → refine retrieval query from verifier's missing_citations
  publish_node        → Publisher writes docs/ HTML + outputs/ JSON

Retry loop (self-correcting generation):
  verify_node → should_retry_verdict → revise_query_node → retrieve_node
  Loop exits after 2 retries or when verifier scores pass threshold.

State is a Pydantic PipelineState model — every transition is validated.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .director import Director
from .feeds import harvest_claims
from .graph import get_related_context, update_graph
from .models import Claim, DailyReport, PipelineState, ResearchResult, ReviewQueueItem
from .observability import get_tracer, reset_tracer
from .publisher import Publisher
from .researcher import Researcher
from .review_queue import upsert_review_queue
from .retriever import HybridRetriever, build_retrieval_query, classify_query
from .store import load_all_chunks, store_research
import re as _re
from .utils import setup_logging
from .verifier import verify_verdict

logger = logging.getLogger(__name__)

FALLBACK_RETRIEVAL_SCORE_FLOOR = 0.2
_ERROR_FETCH_PREFIX = "Error fetching page"
REVIEW_CONFIDENCE_THRESHOLD = 0.60
REVIEW_VERIFIER_SCORE_THRESHOLD = 0.75

# ── Retriever cache ───────────────────────────────────────────────────────────
# HybridRetriever.fit_transform is expensive (~30-60s on large stores).
# We cache the fitted retriever keyed by a fingerprint of the chunk list so
# the same chunk set is never re-fit within a single Python process.
# The cache is intentionally small (max 4 entries) to bound memory usage.
_retriever_cache: dict[str, HybridRetriever] = {}
_RETRIEVER_CACHE_MAX = 4


def _chunk_fingerprint(chunks: list) -> str:
    """Fast fingerprint of a chunk list — chunk count + XOR of first/last IDs."""
    if not chunks:
        return "empty"
    n = len(chunks)
    first_id = getattr(chunks[0], "chunk_id", str(chunks[0]))
    last_id  = getattr(chunks[-1], "chunk_id", str(chunks[-1]))
    raw = f"{n}:{first_id}:{last_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _get_retriever(chunks: list) -> HybridRetriever:
    """Return a cached HybridRetriever for this chunk set, building if needed."""
    if not chunks:
        return HybridRetriever([])
    key = _chunk_fingerprint(chunks)
    if key not in _retriever_cache:
        if len(_retriever_cache) >= _RETRIEVER_CACHE_MAX:
            # Evict the oldest entry (insertion-order dict, Python 3.7+)
            _retriever_cache.pop(next(iter(_retriever_cache)))
        logger.info("[retriever_cache] Building new retriever for %d chunks (key=%s)", len(chunks), key)
        _retriever_cache[key] = HybridRetriever(chunks)
    else:
        logger.debug("[retriever_cache] Cache hit for %d chunks (key=%s)", len(chunks), key)
    return _retriever_cache[key]

# Lazily-initialised singletons — not created until first node call,
# so API keys are guaranteed to be loaded from .env before instantiation.
_director:   Director   | None = None
_researcher: Researcher | None = None


def _get_director() -> Director:
    global _director
    if _director is None:
        _director = Director()
    return _director


def _get_researcher() -> Researcher:
    global _researcher
    if _researcher is None:
        _researcher = Researcher()
    return _researcher


# ── Node functions ────────────────────────────────────────────────────────────

def harvest_node(state: PipelineState) -> dict[str, Any]:
    """Node 1 — ingest RSS/Atom feeds and populate candidate claims."""
    if state.selected:
        logger.info("[harvest] Manual claim mode detected — skipping feed harvest.")
        return {"candidates": state.selected}
    logger.info("[harvest] Parsing feeds from %s…", state.feeds_path)
    candidates = harvest_claims(state.feeds_path)
    logger.info("[harvest] %d candidate claims harvested.", len(candidates))
    return {"candidates": candidates}


def select_node(state: PipelineState) -> dict[str, Any]:
    """Node 2 — Director (GPT) scores and selects the day's best claims."""
    if state.selected:
        logger.info("[select] Manual claim already selected — skipping Director selection.")
        return {"selected": state.selected}
    if not state.candidates:
        logger.warning("[select] No candidates to select from.")
        return {"selected": []}
    selected = _get_director().select_claims(state.candidates)
    logger.info("[select] Director chose %d claims.", len(selected))
    return {"selected": selected}


def research_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 3 — Researcher (Claude) investigates each selected claim in parallel.
    Uses ThreadPoolExecutor for fan-out; results collected into research_results dict.
    """
    if not state.selected:
        logger.warning("[research] No claims to research.")
        return {"research_results": {}}

    results: dict[str, ResearchResult] = {}
    with ThreadPoolExecutor(max_workers=state.max_workers) as pool:
        futures = {pool.submit(_get_researcher().research, claim): claim for claim in state.selected}
        for future in as_completed(futures):
            claim = futures[future]
            try:
                results[claim.id] = future.result()
                logger.info("[research] Completed claim %s.", claim.id)
            except Exception as exc:
                logger.error("[research] Failed claim %s: %s", claim.id, exc)

    return {"research_results": results}


def store_evidence_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 4 — Chunk each ResearchResult and persist to evidence store.
    Writes outputs/evidence_YYYY-MM-DD.json (daily) and
    outputs/evidence_store.json (cumulative, deduplicated).
    """
    evidence_chunks: dict[str, list] = {}
    for claim in state.selected:
        research = state.research_results.get(claim.id)
        if not research:
            continue
        try:
            chunks = store_research(research, claim.text, state.outputs_dir)
            evidence_chunks[claim.id] = chunks
            logger.info("[store_evidence] %d chunks stored for claim %s", len(chunks), claim.id)
        except Exception as exc:
            logger.error("[store_evidence] Failed for claim %s: %s", claim.id, exc)
    return {"evidence_chunks": evidence_chunks}


def graph_context_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 5 — Query the persistent knowledge graph for cross-run context.
    For each claim, does a depth-2 traversal to find related past claims
    that share source domains. Context is injected into retrieval queries.
    """
    # Update graph with today's newly stored chunks
    all_new_chunks = [c for chunks in state.evidence_chunks.values() for c in chunks]
    if all_new_chunks:
        try:
            update_graph(all_new_chunks, state.outputs_dir)
        except Exception as exc:
            logger.warning("[graph_context] Graph update failed: %s", exc)

    graph_context: dict[str, str] = {}
    for claim in state.selected:
        try:
            ctx = get_related_context(claim.text, claim.id, state.outputs_dir)
            if ctx:
                graph_context[claim.id] = ctx
                logger.info("[graph_context] Context found for claim %s (%d chars)", claim.id, len(ctx))
        except Exception as exc:
            logger.warning("[graph_context] Failed for claim %s: %s", claim.id, exc)
    return {"graph_context": graph_context}


def retrieve_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 6 — HybridRetriever ranks claim-local evidence first, then falls back
    to the cumulative evidence store only when strong cross-claim matches exist.

    Adaptive routing: classify_query() inspects the raw claim text (before
    graph-context enrichment so the signal is clean) and selects a BM25-heavy
    blend for entity-rich claims (numbers, dates, names) or a TF-IDF-heavy
    blend for conceptual/abstract claims. The routing hint is passed to every
    search() call so both the local and fallback passes use the same weights.
    """
    all_chunks = load_all_chunks(state.outputs_dir)
    has_local_chunks = any(state.evidence_chunks.get(claim.id) for claim in state.selected)
    if not all_chunks and not has_local_chunks:
        logger.warning("[retrieve] Evidence store empty — skipping retrieval.")
        return {"retrieval_hits": {}}

    retrieval_hits: dict[str, list] = {}

    for claim in state.selected:
        graph_ctx = state.graph_context.get(claim.id, "")
        # On retry, the revised query is stored in graph_context with a special key
        revised_query = state.graph_context.get(f"{claim.id}::revised_query", "")
        query_text = revised_query if revised_query else claim.text
        query = build_retrieval_query(query_text, graph_ctx)

        # Classify on the raw claim text (before context enrichment) so
        # the classification reflects the claim's inherent entity density,
        # not the length-inflated graph context string.
        query_type = classify_query(claim.text)
        logger.info(
            "[retrieve] claim=%s query_type=%s (%s%s)",
            claim.id, query_type,
            "BM25-dominant" if query_type == "entity" else "TF-IDF-dominant",
            " [retry]" if revised_query else "",
        )

        try:
            local_chunks = state.evidence_chunks.get(claim.id, [])
            local_hits = _search_chunks(local_chunks, query, claim.id, top_k=6, query_type=query_type)
            fallback_hits = _search_chunks(all_chunks, query, claim.id, top_k=12, query_type=query_type)
            hits = _merge_retrieval_hits(local_hits, fallback_hits, claim.id, top_k=6)
            retrieval_hits[claim.id] = hits
            logger.info(
                "[retrieve] %d hits for claim %s (local=%d fallback_kept=%d top score: %.3f)",
                len(hits),
                claim.id,
                len(local_hits),
                sum(1 for hit in hits if hit.chunk.claim_id != claim.id),
                hits[0].hybrid_score if hits else 0.0,
            )
        except Exception as exc:
            logger.error("[retrieve] Failed for claim %s: %s", claim.id, exc)

    return {"retrieval_hits": retrieval_hits}


def _search_chunks(
    chunks: list,
    query: str,
    claim_id: str,
    top_k: int,
    query_type: str = "default",
) -> list:
    if not chunks:
        return []
    return _get_retriever(chunks).search(query, top_k=top_k, claim_id=claim_id, query_type=query_type)


def _merge_retrieval_hits(
    local_hits: list,
    fallback_hits: list,
    claim_id: str,
    top_k: int,
) -> list:
    merged: list = []
    seen_ids: set[str] = set()

    for hit in local_hits:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in seen_ids:
            continue
        # Drop chunks that are stored fetch-error messages (retroactive guard)
        if hit.chunk.text.startswith(_ERROR_FETCH_PREFIX):
            continue
        merged.append(hit)
        seen_ids.add(chunk_id)
        if len(merged) >= top_k:
            return merged

    for hit in fallback_hits:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in seen_ids:
            continue
        if hit.chunk.claim_id == claim_id:
            continue
        if hit.hybrid_score < FALLBACK_RETRIEVAL_SCORE_FLOOR:
            continue
        merged.append(hit)
        seen_ids.add(chunk_id)
        if len(merged) >= top_k:
            break

    return merged


def verdict_node(state: PipelineState) -> dict[str, Any]:
    """Node 7 — Director (GPT) synthesises a structured Verdict per claim.
    Passes retrieval_hits so the Director can ground its verdict in
    the top-ranked evidence chunks (RAG-augmented generation).
    """
    verdicts = []
    for claim in state.selected:
        research = state.research_results.get(claim.id)
        if not research:
            logger.warning("[verdict] No research for claim %s — skipping.", claim.id)
            continue
        hits = state.retrieval_hits.get(claim.id, [])
        logger.info("[verdict] Findings preview for %s: %s", claim.id, research.findings[:300])
        try:
            verdict = _get_director().synthesize_verdict(claim, research, hits)
            verdicts.append(verdict)
            logger.info("[verdict] %s → %s (%.0f%%)", claim.id, verdict.verdict, verdict.confidence * 100)
        except Exception as exc:
            logger.error("[verdict] Failed for claim %s: %s", claim.id, exc)
    return {"verdicts": verdicts}


def verify_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 8 — LLM-as-a-Judge: a separate GPT-4o call evaluates each verdict
    against the retrieved evidence using a RAGAS-style rubric.
    Sets should_retry=True on any verdict scoring below threshold.
    """
    verifier_reports: dict[str, Any] = dict(state.verifier_reports)
    for verdict in state.verdicts:
        claim_id = verdict.claim_id
        hits = state.retrieval_hits.get(claim_id, [])
        claim_text = next((c.text for c in state.selected if c.id == claim_id), "")
        try:
            report = verify_verdict(claim_text, verdict, hits)
            verifier_reports[claim_id] = report
            logger.info(
                "[verify] claim=%s overall=%.2f retry=%s",
                claim_id, report.overall_score, report.should_retry,
            )
        except Exception as exc:
            logger.error("[verify] Failed for claim %s: %s", claim_id, exc)
    return {"verifier_reports": verifier_reports}


def _extract_domain(url: str) -> str:
    """Return bare domain from a URL string (e.g. 'factcheck.org')."""
    url = url or ""
    url = _re.sub(r"^https?://", "", url).split("/")[0]
    return _re.sub(r"^www\.", "", url).lower()


def _review_reason(verdict, verifier_report) -> str:
    reasons: list[str] = []
    if verdict.confidence < REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Model confidence {verdict.confidence:.2f} is below the review threshold "
            f"of {REVIEW_CONFIDENCE_THRESHOLD:.2f}."
        )
    if verifier_report and verifier_report.overall_score < REVIEW_VERIFIER_SCORE_THRESHOLD:
        reasons.append(
            f"Verifier overall score {verifier_report.overall_score:.2f} is below the "
            f"review threshold of {REVIEW_VERIFIER_SCORE_THRESHOLD:.2f}."
        )
    return " ".join(reasons)


def revise_query_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 9 — Build refined retrieval queries from verifier missing_citations.
    Also attempts to fetch one corroborating source from a new domain when the
    verifier flags weak grounding, so the next retrieve pass has richer evidence.
    Increments retry_counts.
    """
    graph_context  = dict(state.graph_context)
    retry_counts   = dict(state.retry_counts)
    evidence_chunks = dict(state.evidence_chunks)

    for report_id, report in state.verifier_reports.items():
        if not report.should_retry:
            continue

        retry_counts[report_id] = retry_counts.get(report_id, 0) + 1
        claim_text = next(
            (c.text for c in state.selected if c.id == report_id), ""
        )

        # Refine the retrieval query with verifier's missing-citation hints
        hint_text = " ".join(report.missing_citations[:3])
        refined = f"{claim_text} {hint_text}".strip()
        graph_context[f"{report_id}::revised_query"] = refined
        logger.info(
            "[revise_query] Refined query for claim %s (retry #%d): %s",
            report_id, retry_counts[report_id], refined[:120],
        )

        # ── Corroboration fetch ──────────────────────────────────────────
        # Collect domains already seen for this claim so corroborate()
        # avoids re-fetching the same sources.
        existing_chunks = evidence_chunks.get(report_id, [])
        existing_domains = list({
            _extract_domain(c.source_url)
            for c in existing_chunks
            if getattr(c, "source_url", "")
        })
        logger.info(
            "[revise_query] Requesting corroboration for %s (existing domains: %s)",
            report_id, existing_domains,
        )
        try:
            extra = _get_researcher().corroborate(
                claim_id=report_id,
                claim_text=claim_text,
                existing_domains=existing_domains,
            )
            if extra:
                new_chunks = store_research(extra, claim_text, state.outputs_dir)
                evidence_chunks[report_id] = existing_chunks + new_chunks
                logger.info(
                    "[revise_query] Corroboration added %d new chunks for claim %s",
                    len(new_chunks), report_id,
                )
        except Exception as exc:
            logger.warning(
                "[revise_query] Corroboration failed for claim %s: %s", report_id, exc
            )

    return {
        "graph_context": graph_context,
        "retry_counts": retry_counts,
        "evidence_chunks": evidence_chunks,
    }


def review_gate_node(state: PipelineState) -> dict[str, Any]:
    """
    Week 11 slice — queue low-confidence claims for human review.
    This does not pause execution yet; it persists a review queue entry and
    marks the claim as pending review in the published artifacts.
    """
    review_queue = dict(state.review_queue)
    pending_items: list[ReviewQueueItem] = []
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for verdict in state.verdicts:
        verifier_report = state.verifier_reports.get(verdict.claim_id)
        reason = _review_reason(verdict, verifier_report)
        if not reason:
            continue

        claim = next((c for c in state.selected if c.id == verdict.claim_id), None)
        item = ReviewQueueItem(
            review_id=f"{date_slug}:{verdict.claim_id}",
            claim_id=verdict.claim_id,
            claim_text=claim.text if claim else "",
            source=claim.source if claim else "",
            date_slug=date_slug,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            verifier_score=verifier_report.overall_score if verifier_report else None,
            reason=reason,
        )
        review_queue[verdict.claim_id] = item
        pending_items.append(item)

    if pending_items:
        upsert_review_queue(state.outputs_dir, pending_items)
        logger.info("[review_gate] Queued %d claim(s) for manual review.", len(pending_items))
    else:
        logger.info("[review_gate] No claims required manual review.")

    return {"review_queue": review_queue}


def publish_node(state: PipelineState) -> dict[str, Any]:
    """Node 10 — Publisher renders HTML to docs/ and JSON to outputs/.
    Passes verifier_reports and retrieval_hits into DailyReport so the
    HTML template can surface quality scores alongside each verdict.
    """
    report = _get_director().build_report(
        state.verdicts,
        state.selected,
        retrieval_hits=state.retrieval_hits,
        verifier_reports=state.verifier_reports,
        review_queue=state.review_queue,
    )
    # Attach trace summary to report before publishing
    tracer = get_tracer()
    report.trace_summary = tracer.summary()
    Publisher(docs_dir=state.docs_dir, outputs_dir=state.outputs_dir).publish(report)
    # Flush raw trace events to outputs/traces.jsonl
    tracer.flush(state.outputs_dir)
    logger.info("[publish] Report written → %s/%s.html", state.docs_dir, report.date_slug)
    return {"report": report}


# ── Conditional edges ─────────────────────────────────────────────────────────

MAX_RETRIES = int(os.getenv("CLAIMCHECK_MAX_RETRIES", "2"))


def should_continue(state: PipelineState) -> str:
    return "end" if not state.candidates else "select"


def should_research(state: PipelineState) -> str:
    return "end" if not state.selected else "research"


def should_retry_verdict(state: PipelineState) -> str:
    """
    After verify_node: route to revise_query if any verdict needs a retry
    AND we haven't exhausted MAX_RETRIES. Otherwise route to publish.
    """
    for report_id, report in state.verifier_reports.items():
        if report.should_retry:
            retries_so_far = state.retry_counts.get(report_id, 0)
            if retries_so_far < MAX_RETRIES:
                logger.info(
                    "[router] Retry triggered for claim %s (attempt %d/%d)",
                    report_id, retries_so_far + 1, MAX_RETRIES,
                )
                return "revise"
    return "publish"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the ClaimCheck Daily LangGraph."""
    graph = StateGraph(PipelineState)

    # Original nodes (untouched logic)
    graph.add_node("harvest",        harvest_node)
    graph.add_node("select",         select_node)
    graph.add_node("research",       research_node)

    # New RAG + verification nodes
    graph.add_node("store_evidence", store_evidence_node)
    graph.add_node("graph_context",  graph_context_node)
    graph.add_node("retrieve",       retrieve_node)
    graph.add_node("verdict",        verdict_node)
    graph.add_node("verify",         verify_node)
    graph.add_node("revise_query",   revise_query_node)
    graph.add_node("review_gate",    review_gate_node)
    graph.add_node("publish",        publish_node)

    # Linear spine
    graph.add_edge(START, "harvest")
    graph.add_conditional_edges(
        "harvest", should_continue, {"select": "select", "end": END}
    )
    graph.add_conditional_edges(
        "select", should_research, {"research": "research", "end": END}
    )
    graph.add_edge("research",       "store_evidence")
    graph.add_edge("store_evidence", "graph_context")
    graph.add_edge("graph_context",  "retrieve")
    graph.add_edge("retrieve",       "verdict")
    graph.add_edge("verdict",        "verify")

    # Self-correcting loop: verify → (revise → retrieve → verdict → verify) or publish
    graph.add_conditional_edges(
        "verify",
        should_retry_verdict,
        {"revise": "revise_query", "publish": "review_gate"},
    )
    graph.add_edge("revise_query",   "retrieve")
    graph.add_edge("review_gate",    "publish")

    graph.add_edge("publish",        END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    feeds_path:  str | Path = "feeds.yaml",
    docs_dir:    str | Path = "docs",
    outputs_dir: str | Path = "outputs",
    max_workers: int = 3,
    log_level:   str = "INFO",
    manual_claim: str | None = None,
) -> DailyReport:
    """Compile and invoke the LangGraph; return the final DailyReport."""
    setup_logging(log_level)
    reset_tracer()   # fresh tracer for this run
    logger.info("=== ClaimCheck Daily pipeline starting (LangGraph) ===")

    selected: list[Claim] = []
    candidates: list[Claim] = []
    if manual_claim:
        claim = Claim(
            id=hashlib.md5(manual_claim.encode()).hexdigest()[:8],
            text=manual_claim,
            source="Manual Input",
            feed_name="Manual Input",
        )
        selected = [claim]
        candidates = [claim]

    initial_state = PipelineState(
        feeds_path=str(feeds_path),
        docs_dir=str(docs_dir),
        outputs_dir=str(outputs_dir),
        max_workers=max_workers,
        candidates=candidates,
        selected=selected,
    )

    final_state = build_graph().invoke(initial_state)
    # LangGraph returns a dict; extract the report safely
    report = (final_state.get("report") if isinstance(final_state, dict) else final_state.report) or DailyReport()
    logger.info("=== Pipeline complete — %d verdicts ===", len(report.verdicts))
    return report
