"""
Publisher — renders DailyReport → GitHub Pages (docs/)
--------------------------------------------------------
Outputs:
  docs/YYYY-MM-DD.html   — per-day report page
  docs/index.html        — landing page (latest + archive list)
  outputs/YYYY-MM-DD.json — raw machine-readable output
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .models import DailyReport


def _bare_domain(url: str) -> str:
    """Return a short readable domain from a full URL (e.g. 'factcheck.org')."""
    url = url or ""
    url = re.sub(r"^https?://", "", url).split("/")[0]
    return re.sub(r"^www\.", "", url).lower() or "unknown source"

logger = logging.getLogger(__name__)

VERDICT_COLORS = {
    "TRUE": "#22c55e",
    "MOSTLY_TRUE": "#84cc16",
    "MIXED": "#f59e0b",
    "MOSTLY_FALSE": "#f97316",
    "FALSE": "#ef4444",
    "UNVERIFIABLE": "#94a3b8",
}

VERDICT_LABELS = {
    "TRUE": "✅ True",
    "MOSTLY_TRUE": "🟢 Mostly True",
    "MIXED": "🟡 Mixed",
    "MOSTLY_FALSE": "🟠 Mostly False",
    "FALSE": "❌ False",
    "UNVERIFIABLE": "❔ Unverifiable",
}


def _truncate(text: str, limit: int = 90) -> str:
    """Shorten long archive previews without breaking the page layout."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_cost(cost: float | int | None) -> str:
    """Render a small USD value consistently."""
    if cost is None:
        return "$0.00000"
    return f"${float(cost):.5f}"


class Publisher:
    def __init__(
        self,
        docs_dir: str | Path = "docs",
        outputs_dir: str | Path = "outputs",
    ):
        self.docs_dir = Path(docs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish(self, report: DailyReport) -> None:
        slug = report.date_slug
        logger.info("Publishing report for %s…", slug)
        self._write_json(report, slug)
        self._write_daily_page(report, slug)
        self._write_index(slug)
        logger.info("Published: docs/%s.html", slug)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_json(self, report: DailyReport, slug: str) -> None:
        path = self.outputs_dir / f"{slug}.json"
        data = {
            "date": slug,
            "generated_at": report.generated_at,
            "trace_summary": report.trace_summary,
            "results": [
                {
                    "claim": next(
                        (c.text for c in report.claims if c.id == v.claim_id), ""
                    ),
                    "source": next(
                        (c.source for c in report.claims if c.id == v.claim_id), ""
                    ),
                    "verdict": v.verdict,
                    "confidence": round(v.confidence, 3),
                    "summary": v.summary,
                    "key_evidence": v.key_evidence,
                    "retrieved_evidence": [
                        {
                            "chunk_id": hit.chunk.chunk_id,
                            "chunk_kind": hit.chunk.chunk_kind,
                            "section": hit.chunk.section,
                            "source_url": hit.chunk.source_url,
                            "hybrid_score": round(hit.hybrid_score, 3),
                            "text": hit.chunk.text,
                        }
                        for hit in report.retrieval_hits.get(v.claim_id, [])[:5]
                    ],
                    "verifier_report": (
                        report.verifier_reports[v.claim_id].model_dump()
                        if v.claim_id in report.verifier_reports
                        else None
                    ),
                    "review_required": v.claim_id in report.review_queue,
                    "review": (
                        report.review_queue[v.claim_id].model_dump()
                        if v.claim_id in report.review_queue
                        else None
                    ),
                }
                for v in report.verdicts
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _write_daily_page(self, report: DailyReport, slug: str) -> None:
        path = self.docs_dir / f"{slug}.html"
        cards = ""
        trace_summary = report.trace_summary or {}
        cost_summary_html = self._cost_summary_html(trace_summary)
        for verdict in report.verdicts:
            claim = next((c for c in report.claims if c.id == verdict.claim_id), None)
            if not claim:
                continue
            color = VERDICT_COLORS.get(verdict.verdict, "#94a3b8")
            label = VERDICT_LABELS.get(verdict.verdict, verdict.verdict)
            evidence_items = "".join(
                f"<li>{e}</li>" for e in verdict.key_evidence
            )
            retrieved_hits = report.retrieval_hits.get(verdict.claim_id, [])
            review_item = report.review_queue.get(verdict.claim_id)

            # Collect distinct domains for the sources-diversity summary
            seen_domains: list[str] = []
            for hit in retrieved_hits[:5]:
                d = _bare_domain(hit.chunk.source_url)
                if d not in seen_domains:
                    seen_domains.append(d)
            domain_tags = "".join(
                f'<span class="domain-tag">{d}</span>' for d in seen_domains
            )
            sources_summary = (
                f'<p class="sources-summary">'
                f'Evidence sources ({len(seen_domains)}): {domain_tags}'
                f'</p>'
            )

            retrieved_items = "".join(
                (
                    f"<li>"
                    f"<span class=\"chunk-domain\">{_bare_domain(hit.chunk.source_url)}</span> · "
                    f"<strong>{hit.chunk.chunk_kind}</strong> · "
                    f"{hit.chunk.section} · "
                    f"score {hit.hybrid_score:.2f}"
                    f"<div class=\"retrieved-text\">{hit.chunk.text}</div></li>"
                )
                for hit in retrieved_hits[:4]
            )

            # ── Verifier scores block ──────────────────────
            vr = report.verifier_reports.get(verdict.claim_id)
            if vr:
                def _score_bar(label: str, score: float) -> str:
                    pct  = int(score * 100)
                    hue  = int(score * 120)   # 0 = red, 120 = green
                    fill = f"hsl({hue},70%,45%)"
                    return (
                        f'<div class="score-row">'
                        f'<span class="score-label">{label}</span>'
                        f'<div class="score-track">'
                        f'<div class="score-fill" style="width:{pct}%;background:{fill}"></div>'
                        f'</div>'
                        f'<span class="score-pct">{pct}%</span>'
                        f'</div>'
                    )
                retry_note = (
                    '<p class="retry-note">⚠ Verifier requested re-retrieval</p>'
                    if vr.should_retry else ""
                )
                verifier_html = (
                    f'<details class="verifier-details">'
                    f'<summary>Quality Scores (LLM-as-a-Judge)</summary>'
                    f'<div class="scores">'
                    f'{_score_bar("Groundedness",     vr.groundedness_score)}'
                    f'{_score_bar("Citation",         vr.citation_score)}'
                    f'{_score_bar("No Contradiction", vr.no_contradiction_score)}'
                    f'{_score_bar("No Assumption",    vr.no_assumption_score)}'
                    f'{_score_bar("Overall",          vr.overall_score)}'
                    f'</div>'
                    f'{retry_note}'
                    f'</details>'
                )
            else:
                verifier_html = ""

            review_html = ""
            if review_item:
                review_html = (
                    f'<div class="review-banner">'
                    f'<strong>Pending human review.</strong> {review_item.reason}'
                    f'</div>'
                )

            cards += f"""
            <div class="card">
              <div class="verdict-badge" style="background:{color}">{label}</div>
              {review_html}
              <h2 class="claim-text">"{claim.text}"</h2>
              <p class="source">Source: <a href="{claim.url or '#'}" target="_blank">{claim.source}</a></p>
              <p class="confidence">Confidence: {int(verdict.confidence * 100)}%</p>
              {sources_summary}
              <p class="summary">{verdict.summary}</p>
              <details>
                <summary>Key Evidence</summary>
                <ul>{evidence_items}</ul>
              </details>
              <details>
                <summary>Retrieved Evidence Chunks</summary>
                <ul>{retrieved_items}</ul>
              </details>
              {verifier_html}
            </div>"""

        html = _page_template(
            title=f"ClaimCheck Daily — {slug}",
            body=f"""
            <header>
              <h1>ClaimCheck Daily</h1>
              <p class="date">{slug}</p>
              <a href="index.html">← All reports</a>
            </header>
            <main>{cost_summary_html}{cards}</main>
            <footer>
              Generated {report.generated_at} · Powered by Claude + GPT-4o
            </footer>""",
        )
        path.write_text(html, encoding="utf-8")

    def _write_index(self, latest_slug: str) -> None:
        """Regenerate index.html with links to all available report pages."""
        slugs = sorted(
            [p.stem for p in self.docs_dir.glob("20*.html")],
            reverse=True,
        )
        links = "".join(self._archive_link_html(s) for s in slugs)
        sibling_link = self._sibling_archive_link()
        sibling_html = (
            f'<p class="archive-nav"><a href="{sibling_link["href"]}">{sibling_link["label"]}</a></p>'
            if sibling_link
            else ""
        )
        html = _page_template(
            title="ClaimCheck Daily — Archive",
            body=f"""
            <header><h1>ClaimCheck Daily</h1><p>Automated fact-checking, every day.</p></header>
            <main>
              <h2>Latest: <a href="{latest_slug}.html">{latest_slug}</a></h2>
              {sibling_html}
              <h3>Archive</h3>
              <ul class="archive">{links}</ul>
            </main>
            <footer>Powered by Claude + GPT-4o · <a href="https://github.com">Source</a></footer>""",
        )
        (self.docs_dir / "index.html").write_text(html, encoding="utf-8")

    def _archive_link_html(self, slug: str) -> str:
        """Render one archive entry with a short preview from the JSON artifact."""
        artifact = self.outputs_dir / f"{slug}.json"
        preview = "Report available"
        count_text = ""

        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            results = data.get("results", [])
        except Exception:
            results = []

        if results:
            if len(results) == 1:
                preview = _truncate(results[0].get("claim", "") or "Manual claim report")
                count_text = "1 claim"
            else:
                preview = _truncate(results[0].get("claim", "") or "Daily fact-check report")
                count_text = f"{len(results)} claims"

        meta_html = (
            f'<div class="archive-meta">{count_text}</div>' if count_text else ""
        )
        return (
            f'<li class="archive-item">'
            f'<a href="{slug}.html">{slug}</a>'
            f'<div class="archive-preview">{preview}</div>'
            f'{meta_html}'
            f'</li>'
        )

    def _sibling_archive_link(self) -> dict[str, str] | None:
        """Return a cross-link between daily and manual archives when available."""
        name = self.docs_dir.name
        if name.endswith("_manual"):
            sibling = self.docs_dir.with_name(name.removesuffix("_manual"))
            label = "View Daily Archive"
        else:
            sibling = self.docs_dir.with_name(f"{name}_manual")
            label = "View Manual Claims Archive"

        if sibling == self.docs_dir or not sibling.exists():
            return None

        href = f"../{sibling.name}/index.html"
        return {"href": href, "label": label}

    def _cost_summary_html(self, trace_summary: dict) -> str:
        """Render a compact per-run cost/tokens summary above the verdict cards."""
        if not trace_summary or not trace_summary.get("spans"):
            return ""

        provider_stats = trace_summary.get("by_provider", {})
        provider_lines = "".join(
            (
                f'<li><strong>{provider.title()}</strong>: '
                f'{_format_cost(stats.get("cost_usd"))} · '
                f'{stats.get("total_tokens", 0):,} tokens '
                f'({stats.get("calls", 0)} call{"s" if stats.get("calls", 0) != 1 else ""})'
                f'</li>'
            )
            for provider, stats in sorted(provider_stats.items())
        )

        return (
            f'<section class="run-summary">'
            f'<h2>Run Cost & Usage</h2>'
            f'<div class="run-metrics">'
            f'<div class="run-metric"><span>Total Cost</span><strong>{_format_cost(trace_summary.get("total_cost_usd"))}</strong></div>'
            f'<div class="run-metric"><span>Total Tokens</span><strong>{trace_summary.get("total_tokens", 0):,}</strong></div>'
            f'<div class="run-metric"><span>LLM Calls</span><strong>{trace_summary.get("spans", 0)}</strong></div>'
            f'<div class="run-metric"><span>Slowest Node</span><strong>{trace_summary.get("slowest_node", "n/a")}</strong></div>'
            f'</div>'
            f'<ul class="provider-costs">{provider_lines}</ul>'
            f'</section>'
        )


# ------------------------------------------------------------------
# HTML helpers
# ------------------------------------------------------------------

def _page_template(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --text: #f1f5f9;
      --muted: #94a3b8; --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
            max-width: 860px; margin: 0 auto; padding: 2rem 1rem; }}
    a {{ color: var(--accent); }}
    header {{ margin-bottom: 2rem; border-bottom: 1px solid #334155; padding-bottom: 1rem; }}
    h1 {{ font-size: 2rem; }}
    .date {{ color: var(--muted); margin-top: .25rem; }}
    .card {{ background: var(--surface); border-radius: .75rem; padding: 1.5rem;
             margin-bottom: 1.5rem; }}
    .verdict-badge {{ display: inline-block; padding: .25rem .75rem; border-radius: 999px;
                      font-weight: 700; font-size: .85rem; margin-bottom: .75rem;
                      color: #000; }}
    .claim-text {{ font-size: 1.1rem; margin-bottom: .5rem; }}
    .source, .confidence {{ color: var(--muted); font-size: .85rem; margin-bottom: .4rem; }}
    .summary {{ margin-top: .75rem; line-height: 1.6; }}
    details {{ margin-top: .75rem; }}
    details summary {{ cursor: pointer; color: var(--accent); }}
    ul {{ margin: .5rem 0 0 1.25rem; line-height: 1.8; }}
    .archive {{ list-style: none; padding: 0; }}
    .archive-item {{ margin: .7rem 0; padding: .8rem 1rem; background: var(--surface);
                     border-radius: .6rem; }}
    .archive-preview {{ margin-top: .3rem; color: var(--text); line-height: 1.5; }}
    .archive-meta {{ margin-top: .2rem; color: var(--muted); font-size: .82rem; }}
    .archive-nav {{ margin: .6rem 0 1.2rem; }}
    .run-summary {{ background: var(--surface); border-radius: .75rem; padding: 1.25rem 1.5rem;
                    margin-bottom: 1.5rem; }}
    .run-summary h2 {{ font-size: 1.15rem; margin-bottom: .9rem; }}
    .run-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: .75rem; }}
    .run-metric {{ background: rgba(15, 23, 42, 0.55); border-radius: .55rem; padding: .8rem .9rem; }}
    .run-metric span {{ display: block; color: var(--muted); font-size: .8rem; margin-bottom: .2rem; }}
    .run-metric strong {{ font-size: 1rem; }}
    .provider-costs {{ margin: 1rem 0 0 1.1rem; color: var(--muted); }}
    .provider-costs li {{ margin: .35rem 0; }}
    footer {{ margin-top: 3rem; color: var(--muted); font-size: .85rem;
              border-top: 1px solid #334155; padding-top: 1rem; }}
    /* LLM-as-a-Judge score bars */
    .verifier-details {{ margin-top: .75rem; }}
    .scores {{ margin-top: .5rem; display: flex; flex-direction: column; gap: .35rem; }}
    .score-row {{ display: flex; align-items: center; gap: .5rem; font-size: .82rem; }}
    .score-label {{ width: 9rem; color: var(--muted); flex-shrink: 0; }}
    .score-track {{ flex: 1; background: #334155; border-radius: 999px; height: .55rem; overflow: hidden; }}
    .score-fill  {{ height: 100%; border-radius: 999px; transition: width .3s; }}
    .score-pct   {{ width: 2.5rem; text-align: right; color: var(--text); }}
    .retry-note  {{ margin-top: .4rem; color: #f97316; font-size: .82rem; }}
    .retrieved-text {{ margin-top: .35rem; color: var(--muted); font-size: .8rem; line-height: 1.5; }}
    /* Source diversity */
    .sources-summary {{ margin-top: .4rem; font-size: .82rem; color: var(--muted); }}
    .domain-tag {{ display: inline-block; background: #0f3460; color: var(--accent);
                   border-radius: .3rem; padding: .1rem .45rem; margin: .1rem .2rem 0 0;
                   font-size: .78rem; font-family: monospace; }}
    .chunk-domain {{ color: var(--accent); font-family: monospace; font-size: .8rem; }}
    .review-banner {{ margin: .35rem 0 .85rem; padding: .7rem .85rem; border-left: 4px solid #f59e0b;
                      background: rgba(245, 158, 11, 0.12); color: #fde68a; border-radius: .35rem; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
