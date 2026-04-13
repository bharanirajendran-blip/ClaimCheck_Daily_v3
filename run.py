#!/usr/bin/env python3
"""
ClaimCheck Daily — CLI entry point
------------------------------------
Usage:
    python run.py                          # daily feed run → docs/ + outputs/
    python run.py --claim "some claim"     # single-claim run → docs_manual/ + outputs_manual/
    python run.py --serve-mcp              # start the ClaimCheck MCP server over stdio
    python run.py --feeds my_feeds.yaml    # custom feed file
    python run.py --log-level DEBUG        # verbose output
    python run.py --dry-run                # harvest + select, skip research
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing agents (so API keys are available)
load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="claimcheck",
        description="ClaimCheck Daily — automated fact-checking pipeline",
    )
    p.add_argument(
        "--feeds",
        default="feeds.yaml",
        help="Path to feeds.yaml (default: feeds.yaml)",
    )
    p.add_argument(
        "--docs-dir",
        default=None,
        help="Output directory for GitHub Pages HTML (default: docs, or docs_manual with --claim)",
    )
    p.add_argument(
        "--outputs-dir",
        default=None,
        help="Output directory for JSON results (default: outputs, or outputs_manual with --claim)",
    )
    p.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("RESEARCH_WORKERS", "3")),
        help="Parallel research threads (default: 3)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Harvest + Director selection only; skip Claude research & publishing",
    )
    p.add_argument(
        "--claim",
        help="Investigate a single claim text instead of using daily feeds",
    )
    p.add_argument(
        "--serve-mcp",
        action="store_true",
        help="Start the ClaimCheck MCP server (stdio transport) instead of running the pipeline",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --serve-mcp is a standalone mode; pipeline-only flags are meaningless and
    # actively dangerous (--claim rewires output dirs to docs_manual/outputs_manual,
    # which collapses the daily and manual stores so the MCP server can no longer
    # see the daily archive via search_evidence / get_verdict_history).
    if args.serve_mcp:
        pipeline_only = {
            "--claim": args.claim,
            "--feeds": args.feeds if args.feeds != "feeds.yaml" else None,
            "--workers": args.workers if args.workers != int(os.getenv("RESEARCH_WORKERS", "3")) else None,
            "--dry-run": args.dry_run or None,
        }
        offenders = [flag for flag, val in pipeline_only.items() if val]
        if offenders:
            print(
                f"[ERROR] --serve-mcp cannot be combined with pipeline-only "
                f"flag(s): {', '.join(offenders)}\n"
                f"  Server mode has its own output paths derived from OUTPUTS_DIR / DOCS_DIR env vars.",
                file=sys.stderr,
            )
            return 2

    # Auto-separate output paths for manual claim runs so daily outputs are never overwritten
    if args.claim:
        if args.docs_dir is None:
            args.docs_dir = "docs_manual"
        if args.outputs_dir is None:
            args.outputs_dir = "outputs_manual"
    else:
        if args.docs_dir is None:
            args.docs_dir = "docs"
        if args.outputs_dir is None:
            args.outputs_dir = "outputs"

    if args.serve_mcp:
        os.environ["DOCS_DIR"] = args.docs_dir
        os.environ["OUTPUTS_DIR"] = args.outputs_dir
        try:
            from agent.mcp_server import main as run_mcp_server
        except ModuleNotFoundError as exc:
            if exc.name == "mcp":
                print(
                    "[ERROR] Missing optional dependency 'mcp'. "
                    "Install project dependencies with `pip install -r requirements.txt` "
                    "before starting the MCP server.",
                    file=sys.stderr,
                )
                return 1
            raise
        return run_mcp_server()

    if args.dry_run:
        # Lightweight check — just show what would be selected
        from agent.feeds import harvest_claims
        from agent.director import Director
        from agent.models import Claim
        from agent.utils import require_env
        from agent.utils import setup_logging

        setup_logging(args.log_level)
        if args.claim:
            claim = Claim(
                id=hashlib.md5(args.claim.encode()).hexdigest()[:8],
                text=args.claim,
                source="Manual Input",
                feed_name="Manual Input",
            )
            print("\nDry-run: manual claim selected:\n")
            print(f"  [{claim.id}] {claim.text[:200]}")
            return 0
        try:
            require_env("OPENAI_API_KEY")
        except EnvironmentError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1
        candidates = harvest_claims(args.feeds)
        director = Director()
        selected = director.select_claims(candidates)
        print(f"\nDry-run: {len(selected)} claims selected:\n")
        for c in selected:
            print(f"  [{c.id}] {c.text[:100]}")
        return 0

    # Validate required env vars before full pipeline work
    try:
        from agent.utils import require_env

        require_env("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    except EnvironmentError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    from agent.pipeline import run_pipeline

    report = run_pipeline(
        feeds_path=args.feeds,
        docs_dir=args.docs_dir,
        outputs_dir=args.outputs_dir,
        max_workers=args.workers,
        log_level=args.log_level,
        manual_claim=args.claim,
    )

    trace = report.trace_summary or {}
    provider_stats = trace.get("by_provider", {})
    provider_parts = []
    for provider in ("openai", "anthropic", "other"):
        stats = provider_stats.get(provider)
        if not stats:
            continue
        provider_parts.append(f"{provider} ${stats.get('cost_usd', 0.0):.5f}")

    print(
        f"\n✅  Published {len(report.verdicts)} verdicts → "
        f"{args.docs_dir}/{report.date_slug}.html"
    )
    if trace.get("spans"):
        provider_suffix = f" ({', '.join(provider_parts)})" if provider_parts else ""
        print(
            f"💸  Estimated LLM cost: ${trace.get('total_cost_usd', 0.0):.5f}"
            f"{provider_suffix}"
        )
        print(
            f"🧮  Tokens: {trace.get('total_tokens', 0):,} total "
            f"({trace.get('total_input_tokens', 0):,} in / "
            f"{trace.get('total_output_tokens', 0):,} out) across "
            f"{trace.get('spans', 0)} LLM call(s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
