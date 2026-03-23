# ClaimCheck Daily v2 — Project Specification

**Course:** Grad5900 — Agentic AI
**Author:** Bharani Rajendran
**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v2

---

## 1. Project Overview

ClaimCheck Daily v2 is a daily fact-checking pipeline that extends the original ClaimCheck Daily with retrieval-augmented generation, a persistent knowledge graph, and a LLM-as-a-Judge evaluation layer.

The original pipeline centered on `select → research → verdict → publish`. This version centers on `research → store → retrieve → verdict → verify → revise`, making retrieval and verification first-class parts of the workflow rather than afterthoughts.

GPT-4o acts as a high-level Director and independent Verifier. Claude acts as a deep Researcher that fetches and reads live articles. LangGraph wires all components into a stateful, validated pipeline with a self-correcting retry loop. Pydantic enforces data integrity at every step.

---

## 2. Architecture

### 2.1 Agent and Component Roles

**Director (GPT-4o)**
- Reads all harvested claim candidates and selects the top 3 most impactful and verifiable ones for the day
- Enforces source diversity: no two claims from the same outlet, covering at least 2 different topics
- Synthesises verdicts grounded in retrieved evidence chunks, treating raw source chunks as the primary source of truth and researcher findings as secondary context
- Uses OpenAI JSON mode to guarantee parseable structured output

**Researcher (Claude)**
- Receives one claim at a time from the pipeline
- Runs a ReAct tool-use loop: calls `fetch_url` to read live article content, then reasons over it
- Uses extended thinking (3,000 budget tokens, 10,000 max tokens) for deep reasoning
- Preserves all raw fetched page content in `fetched_pages` for downstream storage as `raw_source` chunks
- Fetches one corroborating source from a different domain when the claim contains a date/statistic/named event, the primary source is indirect/summary-like, or the claim is contested — skipped when the primary source is itself authoritative (capped at 2 fetches total)
- Source type hierarchy guides corroboration choice: science claims → journal + science outlet; policy claims → .gov doc + fact-checker; historical claims → official archive + reference; general claims → authoritative outlet + independent verification
- `corroborate(claim_id, claim_text, existing_domains)` method: lightweight 2-round targeted fetch called by the retry loop when the verifier flags weak grounding — adds a new domain's evidence before the next retrieval pass

**fetch_url tool**
- Defined in `agent/tools.py` using the Anthropic tool-use API schema
- Fetches a URL using `httpx`, strips HTML with BeautifulSoup, returns clean plain text
- Truncates to 4,000 characters to fit within context limits
- Claude decides autonomously when and what to fetch — not hardcoded
- Max 5 tool-use rounds per claim to prevent infinite loops

**Evidence Store (`agent/store.py`)**
- Stores three kinds of chunks per research result:
  - `raw_source` — actual text fetched from live article pages (highest retrieval value)
  - `summary` — Claude's structured analysis sections (supporting context)
  - `source_metadata` — source titles, URLs, and reliability notes (for citations and graph edges)
- Persists to `outputs/evidence_YYYY-MM-DD.json` (daily) and `outputs/evidence_store.json` (cumulative, deduplicated by `chunk_id`)
- Accumulates across runs so retrieval improves over time

**Knowledge Graph (`agent/graph.py`)**
- Directed graph over `claim` nodes and `source` (domain) nodes using networkx
- Edge types: `CITES` (claim → source), `SUPPORTS` (source → claim), `RELATED_TO` (claim → claim via shared source)
- Updated after every run; serialised to `outputs/knowledge_graph.json`
- `get_related_context()` does a depth-2 traversal to find past claims sharing the same source domains — context is prepended to retrieval queries (GraphRAG)

**HybridRetriever (`agent/retriever.py`)**
- Two-channel retrieval over all cumulative evidence chunks:
  - **Vector channel (60%)** — TF-IDF cosine similarity (scikit-learn, bigram tokenisation)
  - **Keyword channel (40%)** — BM25 term-frequency scoring (k₁ = 1.5, b = 0.75)
- Both scores min-max normalised before combining
- Chunk-type boost: `raw_source` × 1.15, `summary` × 0.95, `source_metadata` × 0.75
- `_diversify_hits()` applies two diversity axes: kind-bias (up to 4 raw source chunks) and domain-diversity (prefer chunks from distinct source domains so verdicts rest on multiple independent sources, not one article)

**Verifier (`agent/verifier.py`)**
- A separate GPT-4o call evaluating each verdict against retrieved evidence on four dimensions:

  | Dimension | Weight | Retry threshold |
  |---|---|---|
  | Groundedness | 35% | < 0.70 |
  | Citation score | 35% | < 0.70 |
  | No contradiction | 15% | — |
  | No assumption | 15% | — |

- Returns a `VerifierReport` with scores, `unsupported_statements`, `contradictions`, `missing_citations`, and `should_retry`

**Publisher (`agent/publisher.py`)**
- Renders HTML verdict cards with verdict badge, confidence, evidence source diversity summary (distinct domain tags), summary, key evidence, retrieved evidence chunks (domain / kind / section / score), and collapsible quality score bars
- Each retrieved chunk leads with its bare source domain in monospace accent colour for quick readability
- Writes `docs/YYYY-MM-DD.html`, `docs/index.html`, and `outputs/YYYY-MM-DD.json`
- JSON output includes `retrieved_evidence` and `verifier_report` per claim

**Eval Harness (`agent/evals.py`)**
- Aggregates verdict counts and claim totals across all daily report JSON files
- Provides a lightweight programmatic summary of pipeline output over time

---

### 2.2 Pipeline Flow (LangGraph StateGraph)

```
START
  │
  ▼
harvest_node          ← parse RSS/Atom feeds → candidate Claims
  │
  ▼ (conditional: abort if no candidates)
select_node           ← Director (GPT-4o) picks top 3 claims
  │
  ▼ (conditional: abort if nothing selected)
research_node         ← Researcher (Claude + fetch_url) investigates each claim
  │
  ▼
store_evidence_node   ← chunk raw fetched pages + summaries → evidence_store.json
  │
  ▼
graph_context_node    ← update knowledge_graph.json; query for related past claims
  │
  ▼
retrieve_node  ◄──────────────────────────────────────────────┐
  │                                                           │
  ▼                                                           │
verdict_node          ← Director synthesises verdict          │
                         grounded in retrieved chunks         │
  │                                                           │
  ▼                                                           │
verify_node           ← LLM-as-a-Judge evaluates verdict      │
  │                                                           │
  ├─── retry? ──► revise_query_node ── refined query + corroboration fetch ───┘
  │               (max 2 retries; fetches one new-domain source via corroborate())
  │
  └─── passed ──► publish_node → HTML + JSON output
                      │
                      ▼
                     END
```

### 2.3 State Management

All state flows through a single `PipelineState` Pydantic model. Each node receives the full state, updates only its own fields, and returns a partial dict that LangGraph merges back. The full state is inspectable and validated at every transition.

---

## 3. Data Models (Pydantic)

All models are in `agent/models.py`.

| Model | Purpose | Key Fields |
|---|---|---|
| `Claim` | One harvested or manually provided claim | `id`, `text`, `source`, `url`, `published_at` |
| `ResearchResult` | Claude's research output | `claim_id`, `findings`, `sources`, `fetched_pages` |
| `EvidenceChunk` | A single stored chunk | `chunk_id`, `claim_id`, `source_url`, `section`, `text`, `date_slug`, `chunk_kind` |
| `RetrievalHit` | A ranked chunk from the retriever | `chunk`, `vector_score`, `keyword_score`, `hybrid_score` |
| `VerifierReport` | Judge's evaluation of one verdict | `groundedness_score`, `citation_score`, `no_contradiction_score`, `no_assumption_score`, `overall_score`, `missing_citations`, `should_retry` |
| `Verdict` | GPT's final judgement | `claim_id`, `verdict` (enum), `confidence` (0–1), `summary`, `key_evidence` |
| `DailyReport` | Full day's output | `claims`, `verdicts`, `retrieval_hits`, `verifier_reports`, `date_slug`, `generated_at` |
| `PipelineState` | LangGraph shared state | all of the above + `evidence_chunks`, `graph_context`, `retry_counts`, config fields |

---

## 4. Repository Structure

```
ClaimCheck_Daily_v2/
├── agent/
│   ├── __init__.py        package exports
│   ├── models.py          Pydantic data models + PipelineState
│   ├── director.py        GPT-4o Director agent
│   ├── researcher.py      Claude Researcher agent (ReAct tool-use loop)
│   ├── tools.py           fetch_url tool — live web article fetcher
│   ├── feeds.py           RSS/Atom feed parser
│   ├── store.py           Persistent evidence store (raw_source + summary + source_metadata)
│   ├── retriever.py       HybridRetriever (TF-IDF + BM25, raw_source biased)
│   ├── graph.py           Knowledge graph (GraphRAG, networkx)
│   ├── verifier.py        LLM-as-a-Judge evaluator
│   ├── evals.py           Evaluation harness (aggregate metrics)
│   ├── pipeline.py        LangGraph StateGraph orchestration
│   ├── publisher.py       HTML + JSON output renderer
│   └── utils.py           retry decorator, logging, env helpers
│
├── docs/                  GitHub Pages output (auto-generated)
│   ├── _config.yml
│   ├── index.html
│   └── YYYY-MM-DD.html
│
├── outputs/               JSON archive (auto-generated; persists across runs)
│   ├── YYYY-MM-DD.json
│   ├── evidence_YYYY-MM-DD.json
│   ├── evidence_store.json
│   └── knowledge_graph.json
│
├── .github/workflows/
│   └── daily.yml          GitHub Actions cron job (08:00 UTC daily)
│
├── feeds.yaml
├── .env.example
├── requirements.txt
├── run.py
└── SPEC.md
```

---

## 5. Feed Sources

Configured in `feeds.yaml`. Seven sources across four categories:

| Source | Category |
|---|---|
| AP News — Top Headlines | news |
| Reuters — Top News | news |
| PolitiFact — Latest | politics |
| FactCheck.org | politics |
| Science Daily — Top Science | science |
| WHO — News | health |
| MIT Technology Review | technology |

Each run harvests up to 10 entries per feed (70 candidates max), narrowed to 3 by the Director.

---

## 6. Output Format

### HTML Report (`docs/YYYY-MM-DD.html`)
Each verdict card shows: verdict badge, claim text, source link, confidence %, evidence source diversity summary (N distinct domain tags), summary, collapsible key evidence, collapsible retrieved evidence chunks (domain / kind / section / hybrid score / text), and collapsible quality score panel with colour-coded bars.

### JSON Archive (`outputs/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-23",
  "generated_at": "2026-03-23T03:35:35Z",
  "results": [
    {
      "claim": "...",
      "source": "...",
      "verdict": "MOSTLY_TRUE",
      "confidence": 0.9,
      "summary": "...",
      "key_evidence": ["..."],
      "retrieved_evidence": [
        {
          "chunk_id": "...",
          "chunk_kind": "raw_source",
          "section": "Fetched Source 1.1",
          "source_url": "https://...",
          "hybrid_score": 1.0,
          "text": "..."
        }
      ],
      "verifier_report": {
        "groundedness_score": 0.9,
        "citation_score": 0.8,
        "overall_score": 0.9,
        "should_retry": false
      }
    }
  ]
}
```

---

## 7. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | ≥ 0.40.0 | Claude Researcher API (extended thinking + tool use) |
| `openai` | ≥ 1.50.0 | GPT-4o Director and Verifier APIs (JSON mode) |
| `langgraph` | ≥ 0.2.0 | StateGraph pipeline orchestration |
| `langchain-core` | ≥ 0.3.0 | Required by LangGraph |
| `pydantic` | ≥ 2.7.0 | Data validation and state modelling |
| `scikit-learn` | ≥ 1.5.0 | TF-IDF vectoriser (vector channel of HybridRetriever) |
| `numpy` | ≥ 1.26.0 | Array ops for cosine similarity and min-max normalisation |
| `networkx` | ≥ 3.3 | Knowledge graph (GraphRAG) |
| `feedparser` | ≥ 6.0.11 | RSS/Atom feed ingestion |
| `httpx` | ≥ 0.27.0 | HTTP client for fetch_url tool |
| `beautifulsoup4` | ≥ 4.12.0 | HTML stripping for fetch_url tool |
| `python-dotenv` | ≥ 1.0.0 | `.env` loading |
| `pyyaml` | ≥ 6.0.2 | `feeds.yaml` parsing |
| `python-dateutil` | ≥ 2.9.0 | Robust date parsing for feed entries |

---

## 8. Running Locally

```bash
# 1. Clone and install
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v2.git
cd ClaimCheck_Daily_v2
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Dry run — feed harvest + GPT selection only (no Claude research, no cost)
python run.py --dry-run

# 4. Full pipeline run
python run.py

# 5. Test a specific claim directly
python run.py --claim "SpaceX launched 50 satellites last week"

# 6. Optional flags
python run.py --log-level DEBUG
python run.py --workers 2
python run.py --outputs-dir my_outputs

# 7. View today's report (macOS)
open docs/$(date +%Y-%m-%d).html
```

**Note:** The evidence store and knowledge graph in `outputs/` accumulate across runs. Retrieval quality improves with each subsequent run.

---

## 9. Known Limitations

**Cold-start retrieval:** On the very first run there are no stored chunks, so retrieval is skipped and the verdict is generated from researcher findings alone. The evidence store builds from the second run onwards.

**Paywalled articles:** `fetch_url` can only read publicly accessible pages. Claude falls back to training knowledge for paywalled sources.

**Retry loop re-runs all claims:** When a retry is triggered, `verdict_node` and `verify_node` re-run for all claims in the batch, not just the one that failed. This is a known inefficiency that inflates token usage on retry runs.

**Claim extraction from headlines:** The feed parser uses the article headline as the claim text. Headlines are not always precise factual claims.

**GitHub Actions workflow (currently disabled):** To re-enable, make the repo private, add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` as GitHub Secrets under Settings → Secrets and variables → Actions, then re-enable the workflow from the Actions tab.

---

## 10. Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-03-02 | Initial ClaimCheck Daily — LangGraph pipeline, Claude Researcher, GPT Director, GitHub Pages publishing |
| v1.1 | 2026-03-02 | Added `fetch_url` tool-use loop — Claude reads live articles |
| v1.2 | 2026-03-07 | Tuned for reliability — 3k thinking budget, 4k content cap, Director diversity rules |
| v2.0 | 2026-03-22 | Added persistent evidence store, hybrid retrieval, GraphRAG, LLM-as-a-Judge, self-correcting retry loop, quality score bars in HTML |
| v2.1 | 2026-03-23 | RAG-centric shift — raw fetched pages stored as `raw_source` chunks; retriever biased toward raw evidence; `--claim` CLI flag; `evals.py` harness; verifier report in JSON output |
| v2.2 | 2026-03-23 | Evidence diversification — conditional corroborating fetch with source-type hierarchy; domain-aware retrieval ranking; verifier-triggered corroboration in retry loop; source domain tags in HTML report |
