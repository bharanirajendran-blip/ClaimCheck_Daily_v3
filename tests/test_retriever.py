from agent.models import EvidenceChunk, RetrievalHit
from agent.retriever import HybridRetriever


def _make_hit(chunk_id: str, domain: str, score: float, chunk_kind: str = "raw_source") -> RetrievalHit:
    return RetrievalHit(
        chunk=EvidenceChunk(
            chunk_id=chunk_id,
            claim_id="claim-1",
            claim_text="Vaccines cause autism.",
            source_url=f"https://{domain}/article/{chunk_id}",
            section="Evidence",
            text=f"Evidence from {domain} ({chunk_id})",
            date_slug="2026-04-19",
            chunk_kind=chunk_kind,
        ),
        vector_score=score,
        keyword_score=score,
        hybrid_score=score,
    )


def test_diversify_hits_limits_single_domain_when_alternatives_exist():
    retriever = HybridRetriever([])
    hits = [
        _make_hit("a1", "example-a.org", 0.99),
        _make_hit("a2", "example-a.org", 0.98),
        _make_hit("a3", "example-a.org", 0.97),
        _make_hit("a4", "example-a.org", 0.96),
        _make_hit("b1", "example-b.org", 0.95),
        _make_hit("c1", "example-c.org", 0.94),
        _make_hit("d1", "example-d.org", 0.93, chunk_kind="summary"),
    ]

    selected = retriever._diversify_hits(hits, top_k=5)
    domains = [hit.chunk.source_url.split("/")[2] for hit in selected]

    assert domains.count("example-a.org") <= retriever.MAX_HITS_PER_DOMAIN
    assert "example-b.org" in domains
    assert "example-c.org" in domains
    assert "example-d.org" in domains


def test_diversify_hits_can_repeat_domain_when_no_alternatives_exist():
    retriever = HybridRetriever([])
    hits = [
        _make_hit("a1", "only-source.org", 0.99),
        _make_hit("a2", "only-source.org", 0.98),
        _make_hit("a3", "only-source.org", 0.97),
    ]

    selected = retriever._diversify_hits(hits, top_k=3)

    assert len(selected) == 3
