# ClaimCheck Daily v3

ClaimCheck Daily v3 is the current development branch of the ClaimCheck project. At the moment, the implemented system is the proven `v2` fact-checking pipeline plus an MCP server interface for running manual checks and exploring stored artifacts.

**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3

**Current implemented scope**
- Core `v2` runtime: LangGraph pipeline, hybrid retrieval, knowledge graph, LLM-as-a-Judge verification, retry loop, HTML/JSON publishing
- Week 9 addition: MCP server in [agent/mcp_server.py](/Users/Bharani/Desktop/DS/Grad5900App_AgenticAI/ClaimCheck_Daily_v3/agent/mcp_server.py:1)
- Week 11 first slice: low-confidence review gate with persisted `review_queue.json`

**Planned, not yet integrated into the runtime graph**
- Week 10: supervisor/specialist routing, debate, reflection
- Remaining Week 11 work: real interrupts, richer human review flow, persistent memory, checkpointing

**Core idea today:** `research → store → retrieve → verify → revise`, with an MCP entry point for single-claim checks and archive lookup.

## What's Implemented

### Fact-checking pipeline
- Pulls claims from RSS feeds or accepts a manual `--claim`
- Researches each claim with Claude using a ReAct-style `fetch_url` loop
- Stores reusable evidence chunks across runs
- Retrieves evidence with hybrid TF-IDF + BM25 ranking
- Injects lightweight graph context from prior claims sharing source domains
- Generates grounded verdicts with GPT-4o
- Verifies verdict quality with an independent GPT-4o judge
- Retries weak verdicts with refined retrieval queries and corroboration fetches
- Queues low-confidence claims for human review and persists them to `review_queue.json`
- Publishes HTML and JSON reports

### MCP server
ClaimCheck can also run as an MCP server. Any MCP-compatible host can call:
- `check_claim(text)` — run the full pipeline on one claim
- `search_evidence(query, k=5)` — search the stored evidence store with the same hybrid retriever used by the pipeline
- `get_verdict_history(date?)` — fetch archived verdicts from daily and manual runs

Manual MCP checks use the same `docs_manual` / `outputs_manual` convention as the CLI, and the read tools search across both daily and manual artifact stores.

## Current Pipeline

```text
START
  │
  ▼
harvest_node          ← parse RSS/Atom feeds → candidate claims
  │
  ▼
select_node           ← Director picks top claims
  │
  ▼
research_node         ← Researcher investigates each claim
  │
  ▼
store_evidence_node   ← chunk + persist evidence
  │
  ▼
graph_context_node    ← update knowledge graph + retrieve related context
  │
  ▼
retrieve_node         ← hybrid TF-IDF + BM25 retrieval
  │
  ▼
verdict_node          ← grounded verdict synthesis
  │
  ▼
verify_node           ← LLM-as-a-Judge
  │
  ├── retry? ──► revise_query_node ──► retrieve_node
  │
  └── pass ──► review_gate_node ──► publish_node ──► END
```

The MCP server is an interface layer on top of this runtime. It does not add extra LangGraph nodes today.

## Components

| Component | Role | Model / Library |
|---|---|---|
| Director | Claim selection and verdict synthesis | GPT-4o |
| Researcher | Live web research with ReAct tool use | Claude |
| `fetch_url` tool | Fetches and strips article HTML to plain text | `httpx` + BeautifulSoup |
| Evidence Store | Persists raw source, summary, and source-metadata chunks | JSON artifacts |
| Knowledge Graph | Links claims to source domains for cross-run context | `networkx` |
| HybridRetriever | TF-IDF + BM25 retrieval with claim-local-first merge | `scikit-learn` + custom BM25 |
| Verifier | Independent LLM-as-a-Judge rubric scoring | GPT-4o |
| Review Gate | Queues low-confidence claims for human review | local persistence |
| MCP Server | Exposes ClaimCheck as MCP tools | `mcp` SDK |
| Publisher | Writes HTML and JSON reports | local renderer |

## Project Structure

```text
ClaimCheck_Daily_v3/
├── agent/
│   ├── __init__.py
│   ├── director.py
│   ├── evals.py
│   ├── feeds.py
│   ├── graph.py
│   ├── mcp_server.py
│   ├── models.py
│   ├── pipeline.py
│   ├── publisher.py
│   ├── researcher.py
│   ├── retriever.py
│   ├── review_queue.py
│   ├── store.py
│   ├── tools.py
│   ├── utils.py
│   └── verifier.py
├── docs/
├── docs_manual/
├── outputs/
├── outputs_manual/
├── feeds.yaml
├── requirements.txt
├── run.py
├── README.md
└── SPEC.md
```

## Setup

### Requirements

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

### Install and run

```bash
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3.git
cd ClaimCheck_Daily_v3
pip install -r requirements.txt

cp .env.example .env
# add ANTHROPIC_API_KEY and OPENAI_API_KEY

python run.py --dry-run
python run.py
python run.py --claim "Apollo 11 landed on the Moon in 1969."
python -m agent.mcp_server
```

### Optional flags

```bash
python run.py --log-level DEBUG
python run.py --workers 2
python run.py --feeds my_feeds.yaml
python run.py --outputs-dir my_outputs
```

## Output Files

| File | Description |
|---|---|
| `docs/YYYY-MM-DD.html` | Daily HTML report |
| `docs/index.html` | Daily report index |
| `docs_manual/YYYY-MM-DD.html` | Manual-claim HTML report |
| `docs_manual/index.html` | Manual report index |
| `outputs/YYYY-MM-DD.json` | Daily JSON report |
| `outputs/evidence_store.json` | Daily cumulative evidence store |
| `outputs/knowledge_graph.json` | Daily cumulative knowledge graph |
| `outputs/review_queue.json` | Daily persisted human review queue |
| `outputs_manual/YYYY-MM-DD.json` | Manual-claim JSON report |
| `outputs_manual/evidence_store.json` | Manual-run cumulative evidence store |
| `outputs_manual/knowledge_graph.json` | Manual-run cumulative knowledge graph |
| `outputs_manual/review_queue.json` | Manual-run persisted human review queue |

## Tech Stack

- [Anthropic Claude](https://www.anthropic.com) for deep research
- [OpenAI GPT-4o](https://openai.com) for claim selection, verdict synthesis, and verification
- [LangGraph](https://langchain-ai.github.io/langgraph/) for pipeline orchestration
- [MCP SDK](https://modelcontextprotocol.io) for the MCP server
- [Pydantic v2](https://docs.pydantic.dev) for typed models and state
- [scikit-learn](https://scikit-learn.org) and `numpy` for hybrid retrieval
- [networkx](https://networkx.org) for graph context
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) for HTML stripping

## Course Alignment

This branch currently has a real implementation through the Week 9 MCP milestone and an initial Week 11 review-queue slice. Week 10 orchestration and the rest of Week 11 memory/interrupt behavior are still planned work.
