"""
tests/test_evals.py — Pytest harness for ClaimCheck Daily v3 eval suite.

Run with:
    pytest tests/test_evals.py -v

What these tests do:
  1. Golden dataset structure — validates all 20 golden claims are well-formed
     (no API calls, always passes)

  2. RAGAS score gates — if pipeline outputs exist, computes RAGAS metrics and
     asserts each metric meets its minimum threshold.
     Skipped gracefully if no outputs have been generated yet.

  3. Golden verdict match — if golden claims have been run through the pipeline,
     checks pass rate is ≥ 70% (one-step adjacency allowed on truth scale).
     Skipped if fewer than 5 golden claims have been run.

  4. Verifier rubric alignment — validates the fallback verifier produces
     scores in valid range and correct structure (no API call needed).

  5. Evidence recall minimum — for any claim with a saved verdict, at least
     1 key_evidence bullet must be present (basic grounding check).

Usage in CI/CD:
  - Add `pytest tests/test_evals.py` as a step after `python run.py`
  - Pipeline should fail the merge if RAGAS composite drops below MIN_RAGAS_COMPOSITE

Environment variable CLAIMCHECK_OUTPUTS_DIR overrides the default outputs dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.evals import (
    GOLDEN_DATASET,
    MIN_ANSWER_RELEVANCY,
    MIN_CONTEXT_PRECISION,
    MIN_CONTEXT_RECALL,
    MIN_FAITHFULNESS,
    MIN_RAGAS_COMPOSITE,
    RAGASScores,
    compute_ragas_scores,
    golden_pass_rate,
    score_golden_dataset,
)
from agent.models import Verdict, VerdictLabel
from agent.verifier import _verify_fallback

# ── helpers ───────────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path(os.getenv("CLAIMCHECK_OUTPUTS_DIR", "outputs"))
VALID_VERDICTS = {v.value for v in VerdictLabel}


def _has_outputs() -> bool:
    """Return True if at least one daily report JSON exists."""
    if not OUTPUTS_DIR.exists():
        return False
    return any(
        p for p in OUTPUTS_DIR.glob("20*.json")
        if p.name not in {"evidence_store.json", "knowledge_graph.json", "review_queue.json"}
    )


# ── 1. Golden dataset structure ───────────────────────────────────────────────

class TestGoldenDatasetStructure:
    """Validates the golden dataset itself — no API calls, always runs."""

    def test_golden_dataset_not_empty(self):
        assert len(GOLDEN_DATASET) >= 10, "Golden dataset must have at least 10 claims"

    def test_all_claims_have_required_fields(self):
        for item in GOLDEN_DATASET:
            assert "claim" in item,            f"Missing 'claim' field: {item}"
            assert "expected_verdict" in item, f"Missing 'expected_verdict': {item}"
            assert "label" in item,            f"Missing 'label': {item}"

    def test_all_expected_verdicts_are_valid(self):
        for item in GOLDEN_DATASET:
            assert item["expected_verdict"] in VALID_VERDICTS, (
                f"Invalid verdict '{item['expected_verdict']}' for claim: {item['claim'][:60]}"
            )

    def test_claims_are_non_empty_strings(self):
        for item in GOLDEN_DATASET:
            assert isinstance(item["claim"], str), "Claim must be a string"
            assert len(item["claim"].strip()) > 10, f"Claim too short: {item['claim']}"

    def test_verdict_distribution_is_diverse(self):
        """Ensures the golden set tests all verdict labels, not just TRUE/FALSE."""
        found = {item["expected_verdict"] for item in GOLDEN_DATASET}
        assert len(found) >= 4, (
            f"Golden dataset should cover ≥ 4 verdict types; found: {found}"
        )

    def test_domain_diversity(self):
        """Ensures at least 4 distinct topic labels in the golden set."""
        labels = {item["label"] for item in GOLDEN_DATASET}
        assert len(labels) >= 4, f"Need ≥ 4 distinct labels; found: {labels}"


# ── 2. RAGAS score gates ──────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_outputs(), reason="No pipeline outputs found — run the pipeline first")
class TestRAGASGates:
    """Asserts each RAGAS metric meets minimum threshold on saved outputs."""

    @pytest.fixture(scope="class")
    def scores(self) -> RAGASScores:
        return compute_ragas_scores(OUTPUTS_DIR)

    def test_has_claims(self, scores):
        assert scores.n_claims > 0, "No claims found in outputs"

    def test_faithfulness_gate(self, scores):
        assert scores.faithfulness >= MIN_FAITHFULNESS, (
            f"Faithfulness {scores.faithfulness:.3f} < min {MIN_FAITHFULNESS}. "
            "Verdicts are not sufficiently grounded in retrieved evidence."
        )

    def test_answer_relevancy_gate(self, scores):
        assert scores.answer_relevancy >= MIN_ANSWER_RELEVANCY, (
            f"Answer relevancy {scores.answer_relevancy:.3f} < min {MIN_ANSWER_RELEVANCY}. "
            "Verdicts are not addressing claims with sufficient confidence."
        )

    def test_context_precision_gate(self, scores):
        assert scores.context_precision >= MIN_CONTEXT_PRECISION, (
            f"Context precision {scores.context_precision:.3f} < min {MIN_CONTEXT_PRECISION}. "
            "Retrieved evidence chunks are not precise enough for the claims."
        )

    def test_context_recall_gate(self, scores):
        assert scores.context_recall >= MIN_CONTEXT_RECALL, (
            f"Context recall {scores.context_recall:.3f} < min {MIN_CONTEXT_RECALL}. "
            "Not enough evidence bullets per verdict (expect ≥ 3 per claim)."
        )

    def test_composite_gate(self, scores):
        assert scores.composite >= MIN_RAGAS_COMPOSITE, (
            f"Composite RAGAS score {scores.composite:.3f} < min {MIN_RAGAS_COMPOSITE}. "
            "Overall system quality below acceptable threshold."
        )

    def test_retry_rate_reasonable(self, scores):
        """Retry rate > 50% suggests retrieval or prompt quality issues."""
        if scores.n_claims > 0:
            retry_rate = scores.n_retried / scores.n_claims
            assert retry_rate <= 0.50, (
                f"Retry rate {retry_rate:.1%} is too high — retrieval or verification "
                "needs improvement."
            )


# ── 3. Golden verdict match ───────────────────────────────────────────────────

@pytest.mark.skipif(not _has_outputs(), reason="No pipeline outputs found")
class TestGoldenVerdictMatch:
    """Checks pipeline verdicts against known-answer golden dataset."""

    @pytest.fixture(scope="class")
    def golden_results(self):
        return score_golden_dataset(OUTPUTS_DIR, run_missing=False)

    def test_some_golden_claims_have_been_run(self, golden_results):
        covered = [r for r in golden_results if r.actual_verdict is not None]
        if len(covered) < 5:
            pytest.skip(f"Only {len(covered)} golden claims have been run (need ≥ 5 to gate)")

    def test_golden_pass_rate(self, golden_results):
        covered = [r for r in golden_results if r.actual_verdict is not None]
        if len(covered) < 5:
            pytest.skip("Not enough golden claims run yet")
        rate = golden_pass_rate(golden_results)
        assert rate >= 0.70, (
            f"Golden pass rate {rate:.1%} < 70%. "
            "Too many verdicts don't match expected labels. "
            "Check which claims are failing and review prompt or retrieval."
        )

    def test_obvious_false_claims_not_marked_true(self, golden_results):
        """Misinformation claims (vaccines/autism, 5G/COVID) must never be TRUE/MOSTLY_TRUE."""
        misinformation_claims = [
            r for r in golden_results
            if r.actual_verdict is not None
            and r.expected_verdict == "FALSE"
            and r.score_match is False
            and r.actual_verdict in ("TRUE", "MOSTLY_TRUE")
        ]
        assert len(misinformation_claims) == 0, (
            "Pipeline marked known-false misinformation as TRUE/MOSTLY_TRUE:\n"
            + "\n".join(f"  - {r.claim[:80]}" for r in misinformation_claims)
        )


# ── 4. Verifier rubric alignment ──────────────────────────────────────────────

class TestVerifierRubric:
    """Validates verifier output structure and score ranges — no API calls."""

    def _make_verdict(self, n_evidence: int = 3) -> Verdict:
        return Verdict(
            claim_id="test-claim-001",
            verdict=VerdictLabel.MOSTLY_TRUE,
            confidence=0.75,
            summary="The claim is mostly supported by the available evidence.",
            key_evidence=[f"Source {i}: supporting statement." for i in range(n_evidence)],
        )

    def test_fallback_verifier_produces_valid_scores(self):
        v = self._make_verdict(3)
        report = _verify_fallback(v, [])
        assert 0.0 <= report.groundedness_score <= 1.0
        assert 0.0 <= report.citation_score <= 1.0
        assert 0.0 <= report.no_contradiction_score <= 1.0
        assert 0.0 <= report.no_assumption_score <= 1.0
        assert 0.0 <= report.overall_score <= 1.0

    def test_fallback_verifier_overall_is_weighted_average(self):
        v = self._make_verdict(3)
        report = _verify_fallback(v, [])
        expected = round(
            0.35 * report.groundedness_score
            + 0.35 * report.citation_score
            + 0.15 * report.no_contradiction_score
            + 0.15 * report.no_assumption_score,
            3,
        )
        assert abs(report.overall_score - expected) < 0.01, (
            f"Overall score {report.overall_score} doesn't match weighted average {expected}"
        )

    def test_fallback_retry_triggered_on_no_evidence(self):
        """A verdict with no key_evidence should trigger a retry."""
        v = self._make_verdict(0)
        report = _verify_fallback(v, [])
        assert report.should_retry, "Empty key_evidence should trigger retry"

    def test_fallback_no_retry_on_good_evidence(self):
        """A verdict with 4+ evidence bullets and sufficient hits should not retry."""
        from agent.models import EvidenceChunk, RetrievalHit
        hits = [
            RetrievalHit(
                chunk=EvidenceChunk(
                    chunk_id=f"c{i}",
                    claim_id="test-claim-001",
                    # claim_text is required (added when knowledge graph was introduced)
                    claim_text="The claim being tested.",
                    source_url=f"https://example.com/{i}",
                    section="body",
                    text=f"Evidence statement {i} supporting the claim.",
                    chunk_kind="raw_source",
                    date_slug="2026-04-12",
                ),
                # RetrievalHit has three score fields, not a single score=
                vector_score=0.85,
                keyword_score=0.80,
                hybrid_score=0.83,
            )
            for i in range(4)
        ]
        v = self._make_verdict(4)
        report = _verify_fallback(v, hits)
        assert not report.should_retry, (
            f"Good verdict with 4 hits should not trigger retry (scores: "
            f"ground={report.groundedness_score}, cit={report.citation_score})"
        )

    def test_verifier_report_has_all_fields(self):
        v = self._make_verdict(2)
        report = _verify_fallback(v, [])
        assert hasattr(report, "unsupported_statements")
        assert hasattr(report, "contradictions")
        assert hasattr(report, "missing_citations")
        assert hasattr(report, "rewrite_suggestion")
        assert hasattr(report, "should_retry")


# ── 5. Evidence recall minimum ────────────────────────────────────────────────

@pytest.mark.skipif(not _has_outputs(), reason="No pipeline outputs found")
class TestEvidenceRecall:
    """Checks that saved verdicts include at least 1 key_evidence bullet."""

    def test_all_verdicts_have_at_least_one_evidence_bullet(self):
        missing: list[str] = []
        for rp in OUTPUTS_DIR.glob("20*.json"):
            if rp.name in {"evidence_store.json", "knowledge_graph.json", "review_queue.json"}:
                continue
            data = json.loads(rp.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                ev = r.get("key_evidence", [])
                if not ev:
                    missing.append(r.get("claim", r.get("id", "?"))[:60])

        assert len(missing) == 0, (
            f"{len(missing)} verdict(s) have no key_evidence bullets:\n"
            + "\n".join(f"  - {c}" for c in missing[:10])
        )
