"""
mcp_server.py — ClaimCheck MCP Server

Exposes the ClaimCheck pipeline as an MCP-compatible tool provider.
Any MCP host (Claude Desktop, custom app) can connect and call:

  Tools
  ─────
  check_claim(text)            — run the full pipeline on a single claim
  search_evidence(query, k?)   — semantic search over the evidence store
  get_verdict_history(date?)   — retrieve past verdicts from the JSON archive

Usage
─────
  # stdio transport (Claude Desktop / local MCP host)
  python -m agent.mcp_server

  # or directly
  python agent/mcp_server.py

Then add to Claude Desktop's mcp_servers config:
  {
    "claimcheck": {
      "command": "python",
      "args": ["-m", "agent.mcp_server"],
      "cwd": "/path/to/ClaimCheck_Daily_v3"
    }
  }

Design notes (Week 9)
─────────────────────
- The MCP SDK handles JSON-RPC transport and schema routing.
  We only implement tool handlers — the business logic.
- Tool descriptions are prompts: precise wording tells the LLM
  exactly when and how to invoke each tool.
- All tool outputs are plain text / JSON strings so any MCP host
  can render them without custom parsing.
- Composability: this server can be combined with other MCP servers
  in the same host session (e.g., a Slack server, a SQL server).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ── Load env early so pipeline can find API keys ──────────────────────────────
load_dotenv()

# ── Add project root to path so agent.* imports work ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from agent.pipeline import run_pipeline
from agent.retriever import HybridRetriever
from agent.store import load_all_chunks

logger = logging.getLogger(__name__)

# ── Server configuration ──────────────────────────────────────────────────────
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))

# ── Create MCP server ─────────────────────────────────────────────────────────
mcp = FastMCP(
    name="claimcheck",
    instructions=(
        "ClaimCheck is a fact-checking tool. Use check_claim to verify whether "
        "a specific statement is true, false, or mixed. Use search_evidence to "
        "find relevant past research from the evidence store. Use get_verdict_history "
        "to retrieve previously fact-checked claims and their verdicts."
    ),
)


# ── Tool 1: check_claim ───────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Fact-check a single claim using live web research and RAG-grounded verdict synthesis. "
        "Fetches real sources, stores evidence, retrieves relevant chunks, produces a structured "
        "verdict (TRUE / MOSTLY_TRUE / MIXED / MOSTLY_FALSE / FALSE / UNVERIFIABLE), and scores "
        "it with an independent LLM-as-a-Judge verifier. "
        "Use this when the user asks you to verify, fact-check, or assess the accuracy of a claim. "
        "Returns a JSON string with: claim, verdict, confidence, summary, key_evidence, "
        "verifier_score, and retrieved_evidence."
    )
)
def check_claim(text: str) -> str:
    """Run the full ClaimCheck pipeline on a single claim.

    Args:
        text: The claim to fact-check (a single declarative statement).

    Returns:
        JSON string with verdict, confidence, summary, evidence, and quality scores.
    """
    logger.info("[mcp] check_claim called: %s", text[:80])

    try:
        report = run_pipeline(
            docs_dir=DOCS_DIR / "manual",
            outputs_dir=OUTPUTS_DIR / "manual",
            manual_claim=text,
            log_level="WARNING",   # quieter in MCP context
        )
    except Exception as exc:
        logger.exception("[mcp] Pipeline error for claim: %s", text[:80])
        return json.dumps({"error": str(exc), "claim": text})

    if not report.verdicts:
        return json.dumps({"error": "Pipeline produced no verdict", "claim": text})

    verdict = report.verdicts[0]
    verifier = report.verifier_reports.get(verdict.claim_id)
    hits = report.retrieval_hits.get(verdict.claim_id, [])

    result = {
        "claim": text,
        "verdict": verdict.verdict.value if hasattr(verdict.verdict, "value") else str(verdict.verdict),
        "confidence": verdict.confidence,
        "summary": verdict.summary,
        "key_evidence": verdict.key_evidence,
        "verifier_score": verifier.overall_score if verifier else None,
        "should_retry": verifier.should_retry if verifier else None,
        "retrieved_evidence": [
            {
                "source_url": h.chunk.source_url,
                "section": h.chunk.section,
                "hybrid_score": round(h.hybrid_score, 3),
                "chunk_kind": h.chunk.chunk_kind,
                "text": h.chunk.text[:300],
            }
            for h in hits[:5]
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


# ── Tool 2: search_evidence ───────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Search the ClaimCheck evidence store for chunks relevant to a query. "
        "The store accumulates raw article text, researcher summaries, and source citations "
        "across all past pipeline runs. "
        "Use this to find existing evidence about a topic before running a full check, "
        "or to explore what sources have been consulted about a subject. "
        "Returns the top-k matching evidence chunks ranked by hybrid TF-IDF + BM25 score."
    )
)
def search_evidence(query: str, k: int = 5) -> str:
    """Search the cumulative evidence store for relevant chunks.

    Args:
        query: The search query — a question or topic string.
        k:     Number of results to return (default 5, max 20).

    Returns:
        JSON string with a list of matching evidence chunks and their scores.
    """
    logger.info("[mcp] search_evidence called: %s (k=%d)", query[:80], k)
    k = min(max(1, k), 20)   # clamp to [1, 20]

    chunks = load_all_chunks(OUTPUTS_DIR)
    if not chunks:
        return json.dumps({"query": query, "results": [], "message": "Evidence store is empty — run the pipeline first."})

    hits = HybridRetriever(chunks).search(query, top_k=k)

    results = [
        {
            "chunk_id": h.chunk.chunk_id,
            "claim_id": h.chunk.claim_id,
            "claim_text": h.chunk.claim_text[:120],
            "source_url": h.chunk.source_url,
            "section": h.chunk.section,
            "chunk_kind": h.chunk.chunk_kind,
            "hybrid_score": round(h.hybrid_score, 3),
            "text": h.chunk.text[:400],
            "date_slug": h.chunk.date_slug,
        }
        for h in hits
    ]

    return json.dumps({"query": query, "total_chunks_searched": len(chunks), "results": results}, indent=2, ensure_ascii=False)


# ── Tool 3: get_verdict_history ───────────────────────────────────────────────

@mcp.tool(
    description=(
        "Retrieve past ClaimCheck verdicts from the JSON archive. "
        "Returns all verdicts from a specific date, or the most recent run if no date is given. "
        "Each verdict includes the claim text, verdict label, confidence, summary, and verifier score. "
        "Use this when the user asks what was fact-checked recently, or wants to look up a past verdict."
    )
)
def get_verdict_history(date: str | None = None) -> str:
    """Retrieve past verdicts from the JSON archive.

    Args:
        date: Date string in YYYY-MM-DD format, or None for the most recent run.

    Returns:
        JSON string with verdict records for the requested date.
    """
    logger.info("[mcp] get_verdict_history called: date=%s", date)

    if date:
        # Validate format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return json.dumps({"error": f"Invalid date format '{date}'. Use YYYY-MM-DD."})
        target_files = list(OUTPUTS_DIR.glob(f"{date}.json"))
    else:
        # Most recent run
        target_files = sorted(OUTPUTS_DIR.glob("20*.json"), reverse=True)

    if not target_files:
        msg = f"No verdicts found for {date}." if date else "No verdict history found — run the pipeline first."
        return json.dumps({"message": msg, "results": []})

    path = target_files[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return json.dumps({"error": f"Could not read {path.name}: {exc}"})

    results = data.get("results", [])
    summary = [
        {
            "claim": r.get("claim", ""),
            "verdict": r.get("verdict", ""),
            "confidence": r.get("confidence"),
            "summary": r.get("summary", ""),
            "verifier_score": r.get("verifier_report", {}).get("overall_score"),
            "source": r.get("source", ""),
        }
        for r in results
    ]

    return json.dumps({
        "date": data.get("date", path.stem),
        "generated_at": data.get("generated_at"),
        "verdict_count": len(summary),
        "verdicts": summary,
    }, indent=2, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ClaimCheck MCP server (stdio transport)...")
    mcp.run(transport="stdio")
