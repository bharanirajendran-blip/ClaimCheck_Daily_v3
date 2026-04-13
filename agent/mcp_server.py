"""
mcp_server.py — ClaimCheck MCP Server

Exposes the ClaimCheck pipeline as an MCP-compatible tool provider.
Any MCP host (Claude Desktop, custom app) can connect and call:

  Tools
  ─────
  check_claim(text)            — run the full pipeline on a single claim
  search_evidence(query, k?)   — hybrid search over the evidence store
  get_verdict_history(date?)   — retrieve past verdicts from the daily/manual archive

Usage
─────
  # recommended stdio transport entrypoint (Claude Desktop / local MCP host)
  python run.py --serve-mcp

  # direct module entrypoint still works in some environments
  python -m agent.mcp_server
  python agent/mcp_server.py

Then add to Claude Desktop's mcp_servers config:
  {
    "claimcheck": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/ClaimCheck_Daily_v3/run.py", "--serve-mcp"],
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

_MCP_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in dependency-light envs
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


class _MissingFastMCPServer:
    """Fallback object so the module remains importable without the mcp SDK."""

    def tool(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def run(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Missing optional dependency 'mcp'. Install project dependencies with "
            "`pip install -r requirements.txt` before starting the MCP server."
        )


def _dependency_error_message(exc: ModuleNotFoundError) -> str:
    """Convert missing Python dependency errors into actionable text."""
    package = exc.name or "a required package"
    return (
        f"Missing Python dependency '{package}'. Install project dependencies with "
        "`pip install -r requirements.txt` before using this MCP tool."
    )

# ── Server configuration ──────────────────────────────────────────────────────
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))


def _default_manual_dir(base_dir: Path) -> Path:
    """Mirror the CLI's *_manual convention for single-claim runs."""
    if base_dir.name.endswith("_manual"):
        return base_dir
    return base_dir.with_name(f"{base_dir.name}_manual")


MANUAL_OUTPUTS_DIR = Path(
    os.getenv("MANUAL_OUTPUTS_DIR", str(_default_manual_dir(OUTPUTS_DIR)))
)
MANUAL_DOCS_DIR = Path(
    os.getenv("MANUAL_DOCS_DIR", str(_default_manual_dir(DOCS_DIR)))
)


def _artifact_output_dirs() -> list[Path]:
    """Return all output directories that may hold ClaimCheck artifacts."""
    dirs: list[Path] = []
    for path in (OUTPUTS_DIR, MANUAL_OUTPUTS_DIR):
        if path not in dirs:
            dirs.append(path)
    return dirs


def _load_all_evidence_chunks():
    """Load evidence from both daily and manual stores, deduplicated by chunk_id."""
    try:
        from agent.store import load_all_chunks
    except ModuleNotFoundError as exc:
        raise RuntimeError(_dependency_error_message(exc)) from exc

    merged: dict[str, object] = {}
    for outputs_dir in _artifact_output_dirs():
        for chunk in load_all_chunks(outputs_dir):
            merged[chunk.chunk_id] = chunk
    return list(merged.values())


def _report_files_for_date(date: str | None = None) -> list[Path]:
    """Find report JSON files across daily and manual output directories."""
    pattern = f"{date}.json" if date else "20*.json"
    paths: list[Path] = []
    for outputs_dir in _artifact_output_dirs():
        if not outputs_dir.exists():
            continue
        paths.extend(outputs_dir.glob(pattern))
    return sorted(set(paths), key=lambda path: (path.stem, path.stat().st_mtime), reverse=True)


def _display_path(path: Path) -> str:
    """Return a stable, readable artifact path for API responses."""
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _require_mcp_dependency() -> None:
    """Raise a user-friendly error if the optional MCP dependency is unavailable."""
    if _MCP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing optional dependency 'mcp'. Install project dependencies with "
            "`pip install -r requirements.txt` before starting the MCP server."
        ) from _MCP_IMPORT_ERROR

# ── Create MCP server ─────────────────────────────────────────────────────────
mcp = (
    FastMCP(
        name="claimcheck",
        instructions=(
            "ClaimCheck is a fact-checking tool. Use check_claim to verify whether "
            "a specific statement is true, false, or mixed. Use search_evidence to "
            "find relevant stored evidence with ClaimCheck's hybrid retriever. Use "
            "get_verdict_history to retrieve previously fact-checked claims and their verdicts."
        ),
    )
    if FastMCP is not None
    else _MissingFastMCPServer()
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
        "verifier_score, cost_summary, and retrieved_evidence."
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
        from agent.pipeline import run_pipeline
    except ModuleNotFoundError as exc:
        return json.dumps({"error": _dependency_error_message(exc), "claim": text})

    try:
        report = run_pipeline(
            docs_dir=MANUAL_DOCS_DIR,
            outputs_dir=MANUAL_OUTPUTS_DIR,
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
    review = report.review_queue.get(verdict.claim_id)
    trace_summary = report.trace_summary or {}

    result = {
        "claim": text,
        "verdict": verdict.verdict.value if hasattr(verdict.verdict, "value") else str(verdict.verdict),
        "confidence": verdict.confidence,
        "summary": verdict.summary,
        "key_evidence": verdict.key_evidence,
        "verifier_score": verifier.overall_score if verifier else None,
        "should_retry": verifier.should_retry if verifier else None,
        "review_required": review is not None,
        "review_status": review.status if review else None,
        "review_reason": review.reason if review else None,
        "cost_summary": {
            "total_cost_usd": trace_summary.get("total_cost_usd", 0.0),
            "total_tokens": trace_summary.get("total_tokens", 0),
            "total_input_tokens": trace_summary.get("total_input_tokens", 0),
            "total_output_tokens": trace_summary.get("total_output_tokens", 0),
            "by_provider": trace_summary.get("by_provider", {}),
        },
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
        "Returns the top-k matching evidence chunks ranked by ClaimCheck's hybrid TF-IDF + BM25 retriever."
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

    try:
        from agent.retriever import HybridRetriever
        chunks = _load_all_evidence_chunks()
    except (ModuleNotFoundError, RuntimeError) as exc:
        return json.dumps({"error": str(exc), "query": query})

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
        "Returns verdicts from a specific date, or the most recent run if no date is given. "
        "Searches both daily and manual-run archives. "
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
        target_files = _report_files_for_date(date)
    else:
        target_files = _report_files_for_date()

    if not target_files:
        msg = f"No verdicts found for {date}." if date else "No verdict history found — run the pipeline first."
        return json.dumps({"message": msg, "results": []})

    # Without a date, show the newest run date across both daily/manual archives.
    if not date:
        latest_date = target_files[0].stem
        target_files = [path for path in target_files if path.stem == latest_date]

    summary = []
    generated_at_values: list[str] = []

    for path in target_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return json.dumps({"error": f"Could not read {path.name}: {exc}"})

        generated_at = data.get("generated_at")
        if generated_at:
            generated_at_values.append(generated_at)

        for result in data.get("results", []):
            summary.append(
                {
                    "claim": result.get("claim", ""),
                    "verdict": result.get("verdict", ""),
                    "confidence": result.get("confidence"),
                    "summary": result.get("summary", ""),
                    "verifier_score": result.get("verifier_report", {}).get("overall_score"),
                    "review_status": (result.get("review") or {}).get("status"),
                    "review_required": result.get("review_required", False),
                    "source": result.get("source", ""),
                    "artifact": _display_path(path),
                }
            )

    return json.dumps({
        "date": date or target_files[0].stem,
        "generated_at": max(generated_at_values) if generated_at_values else None,
        "verdict_count": len(summary),
        "verdicts": summary,
    }, indent=2, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    """Run the ClaimCheck MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    try:
        _require_mcp_dependency()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1
    logger.info("Starting ClaimCheck MCP server (stdio transport)...")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
