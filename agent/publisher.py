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
from dataclasses import asdict
from pathlib import Path

from .models import DailyReport, Claim, Verdict


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
                }
                for v in report.verdicts
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _write_daily_page(self, report: DailyReport, slug: str) -> None:
        path = self.docs_dir / f"{slug}.html"
        cards = ""
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

            cards += f"""
            <div class="card">
              <div class="verdict-badge" style="background:{color}">{label}</div>
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
            <main>{cards}</main>
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
        links = "".join(
            f'<li><a href="{s}.html">{s}</a></li>' for s in slugs
        )
        html = _page_template(
            title="ClaimCheck Daily — Archive",
            body=f"""
            <header><h1>ClaimCheck Daily</h1><p>Automated fact-checking, every day.</p></header>
            <main>
              <h2>Latest: <a href="{latest_slug}.html">{latest_slug}</a></h2>
              <h3>Archive</h3>
              <ul class="archive">{links}</ul>
            </main>
            <footer>Powered by Claude + GPT-4o · <a href="https://github.com">Source</a></footer>""",
        )
        (self.docs_dir / "index.html").write_text(html, encoding="utf-8")


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
    .archive li {{ margin: .4rem 0; }}
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
  </style>
</head>
<body>
{body}
</body>
</html>"""
