# ClaimCheck Daily v3

An automated fact-checking pipeline extending v2 with multi-agent workflows, an MCP server, human-in-the-loop confidence routing, and vector-based persistent memory.

**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3

**Built on:** ClaimCheck Daily v2 — which introduced RAG, hybrid retrieval, LLM-as-a-Judge, and a self-correcting retry loop.

**v3 adds:** MCP server (Week 9), Supervisor + specialist researchers + Debate pattern (Week 10), HITL confidence routing + vector memory + checkpointing (Week 11).

**Core idea:** `research → store → retrieve → verify → revise` extended with `supervise → debate → reflect → interrupt → remember`.

---

## What's New in v3

### Week 9 — MCP Server
ClaimCheck is now also an MCP server (`agent/mcp_server.py`). Any MCP-compatible host (Claude Desktop, custom app) can connect and call:
- `check_claim(text)` — run the full pipeline on a single claim
- `search_evidence(query)` — semantic search over the evidence store
- `get_verdict_history(date?)` — retrieve past verdicts from the archive

### Week 10 — Multi-Agent: Supervisor + Specialists + Debate
- **Supervisor pattern** routes each claim to a domain-specialist researcher (science, politics, health, tech) with tuned prompts and preferred sources
- **Debate pattern** triggers for MIXED/low-confidence verdicts: Advocate defends the verdict, Devil's Advocate challenges it, Judge synthesises
- **Self-reflection** pass: a Critic agent reviews each verdict before publish, flagging unsupported claims and logical gaps

### Week 11 — HITL + Vector Memory
- **Confidence-based routing**: score ≥ 0.85 → auto-publish; 0.60–0.85 → suggest + confirm; < 0.60 → `interrupt()` and pause for human review
- **LangGraph `interrupt()`** breakpoints at verdict_node for low-confidence claims
- **ChromaDB vector memory** replaces TF-IDF/BM25 for semantic cross-run retrieval, entity memory, and episodic recall
- **SQLite checkpointing** — runs survive crashes and can be resumed from any node

---

## Full Pipeline

```
feeds.yaml
    │
    ▼
harvest_node ──► select_node ──► supervisor_node ──► specialist_research_node
                  (GPT-4o)        (routes by domain:    (Science / Politics /
                                   science, politics,    Health / Tech researcher)
                                   health, tech)              │
                                                        store_evidence_node
                                                              │
                                                        graph_context_node
                                                              │
                                                        retrieve_node
                                                        (ChromaDB vector +
                                                         claim-local-first)
                                                              │
                                                        verdict_node
                                                        (GPT-4o RAG)
                                                              │
                                                        reflect_node ◄──────────────┐
                                                        (Critic pass)               │
                                                              │                     │
                                                        verify_node                 │
                                                        (LLM-as-a-Judge)           │
                                                              │                     │
                              ┌─────── low confidence? ──────┤                     │
                              │   interrupt() → human review │                     │
                              │                              │                     │
                              ├─────── retry? ───────────────┘                     │
                              │   (revise_query + debate_node) ────────────────────┘
                              │
                              └─────── passed ──► publish_node ──► END
```

---

## Components

| Component | Role | Model |
|---|---|---|
| Director | Claim selection | GPT-4o |
| Supervisor | Routes claims to domain-specialist researcher | GPT-4o |
| Specialist Researcher | Domain-tuned ReAct research loop (×4 specialists) | Claude + extended thinking |
| Debate agents | Advocate + Devil's Advocate + Judge for low-confidence verdicts | GPT-4o |
| Critic | Self-reflection pass on each verdict before publish | GPT-4o |
| fetch_url tool | Fetches and strips article HTML to plain text | httpx + BeautifulSoup |
| Evidence Store | Chunks raw fetched pages + summaries; persists across runs | — |
| ChromaDB | Vector memory for semantic retrieval + entity/episodic memory | text-embedding-3-small |
| Knowledge Graph | Links claims → sources across runs; injects cross-run context | networkx |
| Verifier | Independent LLM-as-a-Judge scoring on 4-dimension rubric | GPT-4o |
| MCP Server | Exposes check_claim, search_evidence, get_verdict_history tools | — |
| Publisher | HTML + JSON output with source diversity, retrieved evidence, quality scores | — |

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
ClaimCheck_Daily_v3/
├── agent/
│   ├── models.py         Pydantic data models + LangGraph PipelineState
│   ├── director.py       GPT-4o Director — claim selection
│   ├── supervisor.py     Supervisor agent — domain routing to specialists
│   ├── researcher.py     Specialist Researcher agents (science/politics/health/tech)
│   ├── debate.py         Debate pattern — Advocate + Devil's Advocate + Judge
│   ├── tools.py          fetch_url tool — live web article fetcher
│   ├── feeds.py          RSS/Atom feed parser
│   ├── store.py          Evidence store (raw_source + summary + source_metadata)
│   ├── retriever.py      ChromaDB vector retriever + claim-local-first logic
│   ├── graph.py          Knowledge graph (GraphRAG, networkx)
│   ├── verifier.py       LLM-as-a-Judge (4-dimension rubric)
│   ├── memory.py         Vector memory — entity, episodic, semantic recall
│   ├── evals.py          Evaluation harness
│   ├── pipeline.py       LangGraph StateGraph orchestration + HITL interrupt
│   ├── publisher.py      HTML + JSON output renderer
│   ├── mcp_server.py     MCP server — check_claim, search_evidence, get_verdict_history
│   └── utils.py          retry, logging, env helpers
├── docs/                 GitHub Pages output (auto-generated)
├── outputs/              JSON archive + evidence store (auto-generated)
│   ├── YYYY-MM-DD.json
│   ├── evidence_YYYY-MM-DD.json
│   ├── evidence_store.json
│   ├── knowledge_graph.json
│   └── chroma_db/        ChromaDB vector store (auto-generated)
├── feeds.yaml
├── run.py                CLI entry point
└── SPEC.md               Full technical specification
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
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3.git
cd ClaimCheck_Daily_v3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 4. Dry run — feed harvest + claim selection only (no research, no cost)
python run.py --dry-run

# 5. Full pipeline run (~3–5 minutes)
python run.py

# 6. Test a specific claim
python run.py --claim "Apollo 11 landed on the Moon in 1969."

# 7. Start the MCP server
python -m agent.mcp_server
```

### Optional flags

```bash
python run.py --log-level DEBUG
python run.py --workers 2
python run.py --feeds my_feeds.yaml
python run.py --outputs-dir my_outputs
```

### Output files

| File | Description |
|---|---|
| `docs/YYYY-MM-DD.html` | Report with verdicts, debate results, HITL flags, evidence diversity, quality scores |
| `docs/index.html` | Landing page with links to all past reports |
| `outputs/YYYY-MM-DD.json` | Machine-readable verdicts with retrieved evidence and verifier report |
| `outputs/evidence_store.json` | Cumulative chunked evidence store |
| `outputs/chroma_db/` | Persistent ChromaDB vector store |

Manual `--claim` runs write to `docs_manual/` and `outputs_manual/` — daily `docs/` is never touched.

---

## Automated Daily Run (GitHub Actions)

Configured to run at 08:00 UTC via `.github/workflows/daily.yml`. Currently **disabled** to avoid unintended API spend.

**To enable:** Make repo private → add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` as GitHub Secrets → enable workflow from Actions tab.

**To disable:** Actions tab → select workflow → Disable workflow.

---

## Tech Stack

- [Anthropic Claude](https://www.anthropic.com) — specialist researchers with extended thinking + ReAct tool-use
- [OpenAI GPT-4o](https://openai.com) — Director, Supervisor, Debate agents, Verifier
- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful multi-agent pipeline with HITL interrupt + checkpointing
- [ChromaDB](https://docs.trychroma.com/) — vector memory store for semantic retrieval
- [Pydantic v2](https://docs.pydantic.dev) — data validation and state modelling
- [MCP SDK](https://modelcontextprotocol.io) — MCP server exposing ClaimCheck as a tool provider
- [networkx](https://networkx.org) — knowledge graph (GraphRAG)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — HTML stripping for fetch_url tool
- [GitHub Pages](https://pages.github.com) — automated publishing

---

## Course

Grad5900 — Agentic AI
