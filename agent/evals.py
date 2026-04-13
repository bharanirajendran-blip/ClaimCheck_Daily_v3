"""
evals.py — RAGAS-aligned evaluation harness for ClaimCheck Daily v3.

Implements four core RAGAS metrics against the live pipeline outputs:

  faithfulness        (groundedness)   — every verdict statement backed by evidence
  answer_relevancy                     — verdict actually addresses the claim
  context_precision                    — retrieved chunks are relevant to the claim
  context_recall                       — enough evidence chunks to support the verdict

Also provides aggregate report stats and a golden-dataset scorer that
runs a set of known-answer claims through the verifier rubric.

Usage (standalone):
    python -m agent.evals --outputs-dir outputs
    python -m agent.evals --golden       # runs golden dataset only, no API calls needed

Used by tests/test_evals.py as a pytest harness with a score gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Score gate used by CI / pytest ────────────────────────────────────────────
MIN_FAITHFULNESS      = 0.65
MIN_ANSWER_RELEVANCY  = 0.60
MIN_CONTEXT_PRECISION = 0.55
MIN_CONTEXT_RECALL    = 0.50
MIN_RAGAS_COMPOSITE   = 0.60


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RAGASScores:
    faithfulness:       float = 0.0   # every statement grounded in evidence
    answer_relevancy:   float = 0.0   # verdict addresses the claim
    context_precision:  float = 0.0   # retrieved chunks are relevant
    context_recall:     float = 0.0   # evidence coverage sufficient
    composite:          float = 0.0   # weighted average
    n_claims:           int   = 0
    n_retried:          int   = 0
    n_review_queued:    int   = 0
    details:            list  = field(default_factory=list)

    def passes_gate(self) -> bool:
        return (
            self.faithfulness      >= MIN_FAITHFULNESS
            and self.answer_relevancy  >= MIN_ANSWER_RELEVANCY
            and self.context_precision >= MIN_CONTEXT_PRECISION
            and self.context_recall    >= MIN_CONTEXT_RECALL
            and self.composite         >= MIN_RAGAS_COMPOSITE
        )

    def as_dict(self) -> dict:
        return {
            "faithfulness":       round(self.faithfulness, 3),
            "answer_relevancy":   round(self.answer_relevancy, 3),
            "context_precision":  round(self.context_precision, 3),
            "context_recall":     round(self.context_recall, 3),
            "composite":          round(self.composite, 3),
            "n_claims":           self.n_claims,
            "n_retried":          self.n_retried,
            "n_review_queued":    self.n_review_queued,
            "passes_gate":        self.passes_gate(),
            "thresholds": {
                "min_faithfulness":      MIN_FAITHFULNESS,
                "min_answer_relevancy":  MIN_ANSWER_RELEVANCY,
                "min_context_precision": MIN_CONTEXT_PRECISION,
                "min_context_recall":    MIN_CONTEXT_RECALL,
                "min_composite":         MIN_RAGAS_COMPOSITE,
            },
        }


@dataclass
class GoldenResult:
    claim:            str
    expected_verdict: str          # TRUE / MOSTLY_TRUE / MIXED / MOSTLY_FALSE / FALSE
    actual_verdict:   Optional[str]
    expected_label:   str          # short human label for the expected answer
    score_match:      bool         # actual verdict matches expected verdict
    verifier_scores:  Optional[dict] = None
    notes:            str = ""


# ── GOLDEN DATASET ─────────────────────────────────────────────────────────────
# 20 diverse claims spanning science, health, history, tech, economics.
# expected_verdict is the ground-truth label that the system should produce.
# These are validated facts (TRUE/FALSE) or genuinely mixed claims (MIXED).
# The set is intentionally broad so a single-domain bias would fail the suite.

GOLDEN_DATASET: list[dict] = [
    # ── SCIENCE ──────────────────────────────────────────────────────────────
    {
        "claim": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "expected_verdict": "TRUE",
        "label": "physics constant",
        "notes": "Well-established physical constant — system should retrieve authoritative source and mark TRUE.",
    },
    {
        "claim": "DNA is a double-helix structure made of four nucleotide bases.",
        "expected_verdict": "TRUE",
        "label": "biology fact",
        "notes": "Watson-Crick model — unambiguously true.",
    },
    {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "expected_verdict": "FALSE",
        "label": "popular myth",
        "notes": "Widely debunked — astronauts have confirmed it is not visible from low Earth orbit without aid.",
    },
    {
        "claim": "Humans share approximately 98% of their DNA with chimpanzees.",
        "expected_verdict": "MOSTLY_TRUE",
        "label": "genetics approximation",
        "notes": "Commonly cited as 98-99%; method of comparison matters — MOSTLY_TRUE is appropriate.",
    },
    {
        "claim": "Scientists have successfully teleported matter across a room.",
        "expected_verdict": "FALSE",
        "label": "quantum teleportation myth",
        "notes": "Quantum teleportation transfers quantum states, not matter — FALSE.",
    },

    # ── HEALTH & MEDICINE ────────────────────────────────────────────────────
    {
        "claim": "Vaccines cause autism.",
        "expected_verdict": "FALSE",
        "label": "health misinformation",
        "notes": "Thoroughly debunked; original Wakefield paper retracted. Critical test for misinformation detection.",
    },
    {
        "claim": "Regular physical exercise reduces the risk of cardiovascular disease.",
        "expected_verdict": "TRUE",
        "label": "health consensus",
        "notes": "Consistent across decades of research — should be TRUE with high confidence.",
    },
    {
        "claim": "Drinking eight glasses of water a day is a scientifically proven requirement.",
        "expected_verdict": "MOSTLY_FALSE",
        "label": "health myth",
        "notes": "The 8x8 rule lacks strong scientific support; individual needs vary — MOSTLY_FALSE.",
    },
    {
        "claim": "Antibiotics are effective against viral infections like the common cold.",
        "expected_verdict": "FALSE",
        "label": "antibiotic misuse",
        "notes": "Antibiotics only work against bacteria — a clear FALSE.",
    },

    # ── HISTORY ──────────────────────────────────────────────────────────────
    {
        "claim": "World War II ended in 1945.",
        "expected_verdict": "TRUE",
        "label": "history date",
        "notes": "V-E Day May 1945, V-J Day September 1945 — TRUE.",
    },
    {
        "claim": "Christopher Columbus was the first European to reach the Americas.",
        "expected_verdict": "MOSTLY_FALSE",
        "label": "history revision",
        "notes": "Norse explorers (Leif Erikson) reached North America ~500 years earlier — MOSTLY_FALSE.",
    },
    {
        "claim": "The Apollo 11 mission successfully landed astronauts on the Moon in 1969.",
        "expected_verdict": "TRUE",
        "label": "space history",
        "notes": "Well-documented historical fact — TRUE.",
    },

    # ── TECHNOLOGY ───────────────────────────────────────────────────────────
    {
        "claim": "The first iPhone was released in 2007.",
        "expected_verdict": "TRUE",
        "label": "tech history date",
        "notes": "Steve Jobs announced it January 2007, released June 2007 — TRUE.",
    },
    {
        "claim": "Bitcoin was created by Satoshi Nakamoto.",
        "expected_verdict": "TRUE",
        "label": "tech attribution",
        "notes": "Pseudonymous creator — TRUE (identity unknown but attributed to Nakamoto).",
    },
    {
        "claim": "5G wireless technology causes COVID-19.",
        "expected_verdict": "FALSE",
        "label": "tech misinformation",
        "notes": "Debunked conspiracy theory — FALSE. Critical misinformation detection test.",
    },
    {
        "claim": "Large language models like GPT-4 understand language the same way humans do.",
        "expected_verdict": "MIXED",
        "label": "AI capabilities",
        "notes": "Active scientific debate — pattern matching vs. understanding. MIXED is appropriate.",
    },

    # ── ECONOMICS & SOCIETY ──────────────────────────────────────────────────
    {
        "claim": "The United States has the world's largest economy by nominal GDP.",
        "expected_verdict": "TRUE",
        "label": "economics fact",
        "notes": "As of 2024, the US remains #1 by nominal GDP (China #1 by PPP) — TRUE.",
    },
    {
        "claim": "Raising the minimum wage always leads to higher unemployment.",
        "expected_verdict": "MIXED",
        "label": "economics debate",
        "notes": "Evidence is mixed — some studies show small employment effects, others show minimal impact. MIXED.",
    },
    {
        "claim": "Solar energy is now cheaper than coal power in most of the world.",
        "expected_verdict": "MOSTLY_TRUE",
        "label": "energy economics",
        "notes": "IRENA/Lazard data confirms solar LCOE has fallen below coal in most regions — MOSTLY_TRUE.",
    },
    {
        "claim": "Immigration increases crime rates in receiving countries.",
        "expected_verdict": "MOSTLY_FALSE",
        "label": "immigration claim",
        "notes": "Majority of research finds immigrants commit crimes at lower rates than native-born — MOSTLY_FALSE.",
    },
]


# ── RAGAS METRICS (computed from pipeline output JSON) ────────────────────────

def compute_ragas_scores(outputs_dir: str | Path) -> RAGASScores:
    """
    Compute RAGAS-aligned metrics from saved pipeline output JSON reports.

    Faithfulness      — derived from avg verifier groundedness_score
    Answer Relevancy  — proxy: confidence × (1 if verdict != UNKNOWN else 0)
    Context Precision — proxy: fraction of claims where verifier score ≥ 0.70
    Context Recall    — proxy: avg number of key_evidence bullets / expected (3)
    Composite         — weighted average (0.35 faith + 0.25 relevancy +
                        0.20 precision + 0.20 recall)
    """
    outputs_path = Path(outputs_dir)
    report_files = sorted(
        p for p in outputs_path.glob("20*.json")
        if p.name not in {"evidence_store.json", "knowledge_graph.json", "review_queue.json"}
    )
    if not report_files:
        logger.warning("[evals] No report files found in %s", outputs_dir)
        return RAGASScores()

    all_results: list[dict] = []
    for rf in report_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        all_results.extend(data.get("results", []))

    if not all_results:
        return RAGASScores()

    faithfulness_scores:      list[float] = []
    answer_relevancy_scores:  list[float] = []
    context_precision_hits:   list[bool]  = []
    context_recall_scores:    list[float] = []
    n_retried      = 0
    n_review_queued = 0
    details: list[dict] = []

    for r in all_results:
        claim_id = r.get("claim_id", r.get("id", "?"))
        verdict  = r.get("verdict", "UNKNOWN")
        conf     = float(r.get("confidence", 0.0))
        ev       = r.get("key_evidence", [])
        # publisher.py writes the key as "verifier_report" (a full model_dump dict)
        vs       = r.get("verifier_report", None)
        retried  = r.get("retried", False)
        review_req = r.get("review_required", False)

        if retried:
            n_retried += 1
        if review_req:
            n_review_queued += 1

        # Faithfulness: use saved verifier groundedness if present, else conf proxy
        if isinstance(vs, dict):
            faith = float(vs.get("groundedness_score", conf))
            citn  = float(vs.get("citation_score", 0.7))
        elif isinstance(vs, (int, float)):
            faith = float(vs)
            citn  = float(vs)
        else:
            faith = conf * 0.9  # proxy when verifier not saved
            citn  = conf * 0.85

        faithfulness_scores.append(faith)

        # Answer Relevancy: verdict is relevant if it's not UNKNOWN and confidence ≥ 0.4
        relevancy = conf if verdict not in ("UNKNOWN", "") and conf >= 0.4 else conf * 0.5
        answer_relevancy_scores.append(relevancy)

        # Context Precision: verifier overall score ≥ 0.70 means retrieved context was precise
        precision_pass = (citn >= 0.70)
        context_precision_hits.append(precision_pass)

        # Context Recall: sufficient evidence bullets? Expect ≥ 3 for good recall
        recall = min(1.0, len(ev) / 3.0) if ev else 0.0
        context_recall_scores.append(recall)

        details.append({
            "claim_id":         claim_id,
            "verdict":          verdict,
            "confidence":       round(conf, 3),
            "faithfulness":     round(faith, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": precision_pass,
            "context_recall":   round(recall, 3),
            "retried":          retried,
            "review_required":  review_req,
        })

    n = len(all_results)
    faith      = sum(faithfulness_scores)   / n
    relevancy  = sum(answer_relevancy_scores) / n
    precision  = sum(context_precision_hits) / n
    recall     = sum(context_recall_scores) / n
    composite  = (
        0.35 * faith
        + 0.25 * relevancy
        + 0.20 * precision
        + 0.20 * recall
    )

    return RAGASScores(
        faithfulness=round(faith, 3),
        answer_relevancy=round(relevancy, 3),
        context_precision=round(precision, 3),
        context_recall=round(recall, 3),
        composite=round(composite, 3),
        n_claims=n,
        n_retried=n_retried,
        n_review_queued=n_review_queued,
        details=details,
    )


# ── GOLDEN DATASET SCORER ─────────────────────────────────────────────────────

def score_golden_dataset(
    outputs_dir: str | Path,
    run_missing: bool = False,
) -> list[GoldenResult]:
    """
    Compare golden dataset expected verdicts against saved pipeline outputs.

    If run_missing=True (requires API keys), runs the pipeline for any
    golden claims not already in the outputs store.
    Returns a list of GoldenResult with match/mismatch details.
    """
    outputs_path = Path(outputs_dir)
    saved_verdicts = _load_all_verdicts(outputs_path)

    results: list[GoldenResult] = []
    for item in GOLDEN_DATASET:
        claim_text = item["claim"]
        expected   = item["expected_verdict"]
        label      = item["label"]
        notes      = item.get("notes", "")

        actual_verdict = saved_verdicts.get(_normalize(claim_text))
        verifier_scores = None

        if actual_verdict is None and run_missing:
            actual_verdict, verifier_scores = _run_single_claim(claim_text, outputs_path)

        match = (
            actual_verdict is not None
            and _verdicts_compatible(actual_verdict, expected)
        )

        results.append(GoldenResult(
            claim=claim_text,
            expected_verdict=expected,
            actual_verdict=actual_verdict,
            expected_label=label,
            score_match=match,
            verifier_scores=verifier_scores,
            notes=notes,
        ))

    return results


def golden_pass_rate(results: list[GoldenResult]) -> float:
    """Fraction of golden claims where actual verdict matches expected."""
    if not results:
        return 0.0
    matched = [r for r in results if r.actual_verdict is not None]
    if not matched:
        return 0.0
    return sum(1 for r in matched if r.score_match) / len(matched)


# ── REPORT SUMMARY (legacy, kept for backwards compatibility) ─────────────────

def summarize_latest_report(outputs_dir: str | Path) -> dict:
    """Return summary stats for the most recent daily report."""
    outputs_path = Path(outputs_dir)
    daily_reports = sorted(
        p for p in outputs_path.glob("20*.json")
        if p.name not in {"evidence_store.json", "knowledge_graph.json", "review_queue.json"}
    )
    if not daily_reports:
        return {"reports": 0, "claims": 0}

    latest  = json.loads(daily_reports[-1].read_text(encoding="utf-8"))
    results = latest.get("results", [])
    verdict_counts: dict[str, int] = {}
    for r in results:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    ragas = compute_ragas_scores(outputs_path)
    return {
        "reports":       len(daily_reports),
        "claims":        len(results),
        "verdict_counts": verdict_counts,
        "ragas":         ragas.as_dict(),
    }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_all_verdicts(outputs_path: Path) -> dict[str, str]:
    """Build a normalized-claim → verdict dict from all saved JSON reports."""
    verdicts: dict[str, str] = {}
    for rp in outputs_path.glob("20*.json"):
        if rp.name in {"evidence_store.json", "knowledge_graph.json", "review_queue.json"}:
            continue
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                claim_text = r.get("claim", r.get("text", ""))
                verdict    = r.get("verdict", "")
                if claim_text and verdict:
                    verdicts[_normalize(claim_text)] = verdict
        except Exception:
            pass
    return verdicts


def _normalize(text: str) -> str:
    """Normalize claim text for fuzzy dict lookup."""
    return " ".join(text.lower().split())


def _verdicts_compatible(actual: str, expected: str) -> bool:
    """
    Allow one-step adjacency in the truth scale so minor scoring differences
    don't fail the whole test.
    Scale order: TRUE > MOSTLY_TRUE > MIXED > MOSTLY_FALSE > FALSE
    """
    scale = ["TRUE", "MOSTLY_TRUE", "MIXED", "MOSTLY_FALSE", "FALSE"]
    try:
        ai = scale.index(actual.upper())
        ei = scale.index(expected.upper())
        return abs(ai - ei) <= 1
    except ValueError:
        return actual.upper() == expected.upper()


def _run_single_claim(claim_text: str, outputs_path: Path):
    """
    Run the full pipeline for one claim (requires API keys).

    DailyReport stores results in two parallel lists:
      report.claims   — list[Claim] with .id and .text
      report.verdicts — list[Verdict] with .claim_id, .verdict, .confidence

    Verifier details live in report.verifier_reports[claim_id] (a VerifierReport).
    This function matches the produced verdict back to the requested claim_text
    by normalising both strings, then returns (verdict_label, verifier_scores_dict).
    """
    try:
        from agent.pipeline import run_pipeline
        report = run_pipeline(
            docs_dir=outputs_path.parent / "docs_manual",
            outputs_dir=outputs_path,
            manual_claim=claim_text,
            log_level="WARNING",
        )
        # Build a claim_id → claim_text lookup from report.claims
        claim_id_by_text = {
            _normalize(c.text): c.id for c in report.claims
        }
        target_id = claim_id_by_text.get(_normalize(claim_text))

        for verdict in report.verdicts:
            if target_id and verdict.claim_id != target_id:
                continue
            vr = report.verifier_reports.get(verdict.claim_id)
            vs = vr.model_dump() if vr is not None else None
            return verdict.verdict, vs
    except Exception as exc:
        logger.warning("[evals] single-claim run failed: %s", exc)
    return None, None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="ClaimCheck v3 RAGAS eval harness")
    parser.add_argument("--outputs-dir", default="outputs", help="Path to outputs dir")
    parser.add_argument("--golden", action="store_true", help="Score golden dataset only")
    parser.add_argument("--run-missing", action="store_true",
                        help="Run pipeline for golden claims not in outputs (requires API keys)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.golden or args.run_missing:
        results = score_golden_dataset(args.outputs_dir, run_missing=args.run_missing)
        pass_rate = golden_pass_rate(results)
        covered   = [r for r in results if r.actual_verdict is not None]
        not_run   = [r for r in results if r.actual_verdict is None]

        if args.json:
            print(json.dumps({
                "pass_rate": round(pass_rate, 3),
                "covered":   len(covered),
                "not_run":   len(not_run),
                "results": [
                    {
                        "claim":            r.claim[:80],
                        "label":            r.expected_label,
                        "expected":         r.expected_verdict,
                        "actual":           r.actual_verdict,
                        "match":            r.score_match,
                    }
                    for r in results
                ],
            }, indent=2))
        else:
            print(f"\n{'─'*60}")
            print(f"  GOLDEN DATASET  ({len(GOLDEN_DATASET)} claims)")
            print(f"{'─'*60}")
            print(f"  Covered  : {len(covered)} / {len(GOLDEN_DATASET)}")
            print(f"  Not run  : {len(not_run)}")
            if covered:
                print(f"  Pass rate: {pass_rate:.1%}  (≥1-step match on verdict scale)")
            print(f"{'─'*60}")
            for r in results:
                status = "✓" if r.score_match else ("–" if r.actual_verdict is None else "✗")
                print(f"  {status}  [{r.expected_label:25s}]  expected={r.expected_verdict:12s}  actual={r.actual_verdict or 'NOT RUN'}")
            print(f"{'─'*60}\n")
        return

    # RAGAS aggregate metrics
    scores = compute_ragas_scores(args.outputs_dir)
    if args.json:
        print(json.dumps(scores.as_dict(), indent=2))
    else:
        d = scores.as_dict()
        print(f"\n{'─'*60}")
        print(f"  ClaimCheck v3 — RAGAS Metrics  ({d['n_claims']} claims)")
        print(f"{'─'*60}")
        print(f"  Faithfulness      : {d['faithfulness']:.3f}  (min {MIN_FAITHFULNESS})")
        print(f"  Answer Relevancy  : {d['answer_relevancy']:.3f}  (min {MIN_ANSWER_RELEVANCY})")
        print(f"  Context Precision : {d['context_precision']:.3f}  (min {MIN_CONTEXT_PRECISION})")
        print(f"  Context Recall    : {d['context_recall']:.3f}  (min {MIN_CONTEXT_RECALL})")
        print(f"  ─────────────────────────────────")
        print(f"  Composite Score   : {d['composite']:.3f}  (min {MIN_RAGAS_COMPOSITE})")
        print(f"  Gate              : {'PASS ✓' if d['passes_gate'] else 'FAIL ✗'}")
        print(f"  Retried           : {d['n_retried']}")
        print(f"  Review queued     : {d['n_review_queued']}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    _cli()
