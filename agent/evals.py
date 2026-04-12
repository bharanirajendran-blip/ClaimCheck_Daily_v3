"""
evals.py — Minimal evaluation harness for ClaimCheck Daily v3.

Runs simple aggregate metrics over generated reports so the project has
an explicit evaluation artifact beyond single-run judge scores.
"""

from __future__ import annotations

import json
from pathlib import Path


def summarize_latest_report(outputs_dir: str | Path) -> dict:
    outputs_path = Path(outputs_dir)
    daily_reports = sorted(
        [
            path for path in outputs_path.glob("20*.json")
            if path.name != "evidence_store.json" and path.name != "knowledge_graph.json"
        ]
    )
    if not daily_reports:
        return {"reports": 0, "claims": 0}

    latest = json.loads(daily_reports[-1].read_text(encoding="utf-8"))
    results = latest.get("results", [])
    verdict_counts: dict[str, int] = {}
    for result in results:
        verdict = result.get("verdict", "UNKNOWN")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    return {
        "reports": len(daily_reports),
        "claims": len(results),
        "verdict_counts": verdict_counts,
    }
