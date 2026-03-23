# ClaimCheck Daily v2

An automated fact-checking pipeline that runs daily, researches claims from real news feeds, stores evidence as a reusable knowledge base, retrieves the most relevant chunks, produces grounded verdicts, and verifies them with an independent LLM judge.

**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v2

**Core idea:** not just "research and answer," but `research → store → retrieve → verify → revise`.

---

## What It Does

Every run:

1. Pulls headlines from 7 RSS feeds (AP, Reuters, PolitiFact, FactCheck.org, WHO, Science Daily, MIT Tech Review)
2. GPT-4o selects the 3 most fact-checkable claims from different sources and topics
3. Claude fetches the primary source article, then conditionally fetches one corroborating source from a different domain (triggered when the claim contains a date/stat/named event, or the primary source is indirect)
4. Raw fetched article text is stored as reusable `raw_source` evidence chunks
5. Researcher summaries and source citations are stored alongside as `summary` and `source_metadata` chunks
6. A persistent evidence store accumulates chunks across runs
7. A lightweight knowledge graph links claims to source domains for cross-run context
8. Hybrid retrieval (TF-IDF + BM25) surfaces the top evidence chunks, biased toward raw source content and spread across distinct source domains
9. GPT-4o synthesises a verdict grounded in the retrieved evidence chunks
10. A second GPT-4o call acts as an independent LLM-as-a-Judge verifier
11. If grounding or citation scores fall below threshold: the query is refined, a corroborating source from a new domain is fetched, and the verdict is regenerated (up to 2 retries)
12. HTML and JSON reports are published with verdicts, evidence source diversity summary, retrieved evidence chunks, and quality scores

---

## Architecture

```
feeds.yaml
    │
    ▼
harvest_node ──► select_node ──► research_node ──► store_evidence_node
                  (GPT-4o)        (Claude +              │
                                  extended thinking +     │  raw_source +
                                  fetch_url tool)    summary + source_metadata
                                                          ▼
                                               graph_context_node
                                                    │
                                                    ▼
                                               retrieve_node  ◄──────────┐
                                              (TF-IDF + BM25,            │
                                               raw_source biased)        │
                                                    │                    │
                                                    ▼                    │
                                               verdict_node              │
                                                (GPT-4o +                │
                                               RAG chunks)               │
                                                    │                    │
                                                    ▼                    │
                                               verify_node               │
                                           (LLM-as-a-Judge)             │
                                                    │                    │
                              ┌─────── retry? ──────┘                   │
                              │   (revise_query_node) ──────────────────┘
                              │
                              └─────── passed ──► publish_node ──► END
```

Built with **LangGraph** (StateGraph), **Pydantic v2**, **scikit-learn** (TF-IDF), and **networkx** (knowledge graph).

### Components

| Component | Role | Model |
|---|---|---|
| Director | Claim selection + RAG-grounded verdict synthesis | GPT-4o |
| Researcher | Live web research with ReAct tool-use loop | Claude + extended thinking |
| fetch_url tool | Fetches and strips article HTML to plain text | httpx + BeautifulSoup |
| Evidence Store | Chunks raw fetched pages + summaries; persists across runs | — |
| Knowledge Graph | Links claims → sources across runs; injects cross-run context | networkx |
| HybridRetriever | TF-IDF cosine (60%) + BM25 keyword (40%); biased toward raw source chunks, domain-diversified | scikit-learn |
| Verifier | Independent LLM-as-a-Judge scoring on 4-dimension rubric | GPT-4o |
| Publisher | HTML + JSON output with source diversity summary, retrieved evidence, and quality scores | — |

### Evidence chunk types

| Kind | Source | Retrieval weight |
|---|---|---|
| `raw_source` | Actual text fetched from live article pages | 1.15× (highest) |
| `summary` | Claude's structured analysis sections | 0.95× |
| `source_metadata` | Source titles, URLs, reliability notes | 0.75× |

Prioritising raw source chunks means retrieval is grounded in actual article text, not only in the model's paraphrase of it.

---

## Verdict Labels

| Label | Meaning |
|---|---|
| ✅ TRUE | Claim is accurate |
| 🟢 MOSTLY TRUE | Accurate with minor caveats |
| 🟡 MIXED | Partially true, partially false |
| 🟠 MOSTLY FALSE | Misleading or largely inaccurate |
| ❌ FALSE | Claim is inaccurate |
| ❔ UNVERIFIABLE | Insufficient evidence to judge |

---

## Project Structure

```
ClaimCheck_Daily_v2/
├── agent/
│   ├── models.py       Pydantic data models + LangGraph PipelineState
│   ├── director.py     GPT-4o Director agent
│   ├── researcher.py   Claude Researcher agent (ReAct tool-use loop)
│   ├── tools.py        fetch_url tool — live web article fetcher
│   ├── feeds.py        RSS/Atom feed parser
│   ├── store.py        Persistent evidence store (raw_source + summary + source_metadata chunks)
│   ├── retriever.py    HybridRetriever (TF-IDF + BM25, raw_source biased)
│   ├── graph.py        Knowledge graph (claims ↔ sources, networkx)
│   ├── verifier.py     LLM-as-a-Judge (4-dimension rubric)
│   ├── evals.py        Evaluation harness (aggregate metrics across runs)
│   ├── pipeline.py     LangGraph StateGraph orchestration
│   ├── publisher.py    HTML + JSON renderer
│   └── utils.py        retry, logging, env helpers
├── docs/               GitHub Pages output (auto-generated)
├── outputs/            JSON archive + evidence store (auto-generated)
│   ├── YYYY-MM-DD.json              Daily verdicts
│   ├── evidence_YYYY-MM-DD.json     Daily evidence chunks
│   ├── evidence_store.json          Cumulative deduplicated chunk store
│   └── knowledge_graph.json         Persistent claim-source graph
├── feeds.yaml          RSS feed configuration
├── run.py              CLI entry point
└── SPEC.md             Full technical specification
```

---

## Setup

### Requirements

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

### Install & Run

```bash
# 1. Clone the repo
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v2.git
cd ClaimCheck_Daily_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 4. Dry run — feed harvest + GPT claim selection only (no Claude research, no cost)
python run.py --dry-run

# 5. Full pipeline run (~2–3 minutes)
python run.py

# 6. Test a specific claim without feed harvesting
# Automatically writes to docs_manual/ + outputs_manual/ — daily docs/ is never touched
python run.py --claim "Apollo 11 landed on the Moon in 1969."
```

### Optional flags

```bash
python run.py --log-level DEBUG          # verbose logging
python run.py --workers 2               # parallel research threads (default: 3)
python run.py --feeds my_feeds.yaml     # custom feed file
python run.py --outputs-dir my_outputs  # custom output directory
```

### Output files

| File | Description |
|---|---|
| `docs/YYYY-MM-DD.html` | Dark-themed report with verdicts, evidence source diversity tags, retrieved evidence, and quality scores |
| `docs/index.html` | Landing page with links to all past reports |
| `outputs/YYYY-MM-DD.json` | Machine-readable verdicts including retrieved evidence and verifier report |
| `outputs/evidence_store.json` | Cumulative chunked evidence (grows across runs) |
| `outputs/knowledge_graph.json` | Persistent claim-source graph (grows across runs) |

```bash
open docs/$(date +%Y-%m-%d).html      # macOS — open today's report
```

---

## Troubleshooting

**`ModuleNotFoundError: sklearn` or `networkx`**
Run `pip install -r requirements.txt`.

**`max_tokens must be greater than thinking.budget_tokens`**
Make sure you have the latest `researcher.py` — `max_tokens=10000`, `thinking_budget=3000`.

**Paywalled articles return login page content**
`fetch_url` can only read public pages. Claude notes this and falls back to training knowledge for that source.

**Cold-start retrieval**
On the first run there are no stored chunks yet, so retrieval is skipped and verdict is generated from researcher findings alone. The evidence store builds up from the second run onwards.

---

## Tech Stack

- [Anthropic Claude](https://www.anthropic.com) — deep claim research with extended thinking + ReAct tool-use loop
- [OpenAI GPT-4o](https://openai.com) — claim selection, RAG-grounded verdict synthesis, LLM-as-a-Judge verification
- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful multi-agent pipeline with conditional retry loop
- [Pydantic v2](https://docs.pydantic.dev) — data validation and state modelling
- [scikit-learn](https://scikit-learn.org) — TF-IDF vectoriser for hybrid retrieval
- [networkx](https://networkx.org) — knowledge graph over claims and sources
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — HTML stripping for web fetch tool
- [GitHub Pages](https://pages.github.com) — automated publishing

---

## Course

Grad5900 — Agentic AI
