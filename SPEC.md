# ClaimCheck Daily v3 — Current Specification

**Course:** Grad5900 — Agentic AI  
**Author:** Bharani Rajendran  
**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3

## 1. Overview

ClaimCheck Daily v3 is the current development branch of the ClaimCheck project. The code currently implements the `v2` fact-checking runtime and adds an MCP server interface for manual claim checks and archive lookup.

Implemented now:
- LangGraph fact-checking pipeline
- persistent evidence store
- hybrid retrieval with TF-IDF + BM25
- lightweight graph context using source-domain links
- GPT-4o verdict generation
- GPT-4o LLM-as-a-Judge verification
- self-correcting retry loop
- low-confidence review gate with persisted review queue
- MCP server with `check_claim`, `search_evidence`, and `get_verdict_history`

Planned but not yet integrated into the runtime graph:
- supervisor/specialist multi-agent routing
- debate or reflection nodes
- richer human-in-the-loop interrupts and approval flow
- persistent vector memory
- checkpointing

## 2. Implemented Architecture

### Director (`agent/director.py`)
- Selects the top claims from harvested candidates
- Synthesizes verdicts grounded in retrieved evidence
- Builds the final `DailyReport`

### Researcher (`agent/researcher.py`)
- Investigates one claim at a time
- Uses a ReAct-style tool loop with `fetch_url`
- Stores fetched page content for downstream chunking
- Supports targeted corroboration during retry

### fetch_url tool (`agent/tools.py`)
- Fetches a URL with `httpx`
- Strips HTML to plain text with BeautifulSoup
- Returns bounded page text for LLM consumption

### Evidence Store (`agent/store.py`)
- Persists `raw_source`, `summary`, and `source_metadata` chunks
- Writes daily and cumulative JSON artifacts
- Deduplicates chunks by `chunk_id`

### Knowledge Graph (`agent/graph.py`)
- Tracks relationships between claims and source domains
- Adds lightweight cross-run context to retrieval queries

### HybridRetriever (`agent/retriever.py`)
- Uses two channels:
  - TF-IDF cosine similarity
  - BM25 keyword scoring
- Applies chunk-type weighting and claim-local boosts
- Merges claim-local hits first, then cross-run fallback hits

### Verifier (`agent/verifier.py`)
- Scores verdicts on groundedness, citation quality, contradiction, and assumption
- Triggers retries when groundedness or citation quality is weak

### Review Queue (`agent/review_queue.py`)
- Persists pending human-review tasks for low-confidence claims
- Writes `review_queue.json` into the current outputs directory
- Lets published artifacts mark claims as `PENDING_REVIEW`

### Publisher (`agent/publisher.py`)
- Writes HTML reports to `docs/` or `docs_manual/`
- Writes JSON reports to `outputs/` or `outputs_manual/`

### MCP Server (`agent/mcp_server.py`)
- Exposes the pipeline through MCP
- Supports:
  - `check_claim(text)`
  - `search_evidence(query, k=5)`
  - `get_verdict_history(date=None)`
- Manual MCP checks use `docs_manual/` and `outputs_manual/`
- Read tools aggregate both daily and manual artifacts

## 3. Runtime Flow

The compiled LangGraph currently uses this sequence:

```text
START
  │
  ▼
harvest
  │
  ▼
select
  │
  ▼
research
  │
  ▼
store_evidence
  │
  ▼
graph_context
  │
  ▼
retrieve
  │
  ▼
verdict
  │
  ▼
verify
  │
  ├── retry? ──► revise_query ──► retrieve
  │
  └── pass ──► review_gate ──► publish ──► END
```

This is still the `v2` runtime. The MCP server invokes this flow for manual single-claim checks.

## 4. Data Models

All shared models are defined in `agent/models.py`.

| Model | Purpose |
|---|---|
| `Claim` | One harvested or manual claim |
| `ResearchResult` | Research findings plus source metadata and fetched pages |
| `EvidenceChunk` | One persisted chunk of evidence |
| `RetrievalHit` | One ranked retrieval result |
| `Verdict` | Final verdict label, confidence, summary, and key evidence |
| `VerifierReport` | LLM judge rubric output |
| `DailyReport` | Final report for one run |
| `PipelineState` | LangGraph shared state |

## 5. Repository Structure

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
├── outputs/review_queue.json
├── outputs_manual/
├── outputs_manual/review_queue.json
├── feeds.yaml
├── requirements.txt
├── run.py
├── README.md
└── SPEC.md
```

## 6. Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude research agent |
| `openai` | GPT-4o director and verifier |
| `mcp` | MCP server runtime |
| `langgraph` | Workflow orchestration |
| `pydantic` | Data validation |
| `scikit-learn` + `numpy` | Hybrid retrieval |
| `networkx` | Graph context |
| `feedparser` | RSS/Atom ingestion |
| `httpx` + `beautifulsoup4` | Fetch and strip article content |
| `python-dotenv` + `pyyaml` | Configuration loading |

## 7. Running Locally

```bash
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily_v3.git
cd ClaimCheck_Daily_v3
pip install -r requirements.txt

cp .env.example .env
# add ANTHROPIC_API_KEY and OPENAI_API_KEY

python run.py --dry-run
python run.py
python run.py --claim "Apollo 11 landed on the Moon in 1969."
python run.py --serve-mcp
```

The `run.py --serve-mcp` path is the recommended local MCP startup path for demos and Claude Desktop integration. It avoids the package-resolution issues that can occur with `python -m agent.mcp_server`.

MCP usage notes:
- `--serve-mcp` is a standalone mode and should not be combined with `--claim`, `--feeds`, `--workers`, or `--dry-run`.
- The server is a stdio MCP server, so it should be launched by an MCP host instead of being used directly in a plain terminal session.
- In Claude Desktop, the most reliable configuration is to pass `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and absolute `OUTPUTS_DIR` / `DOCS_DIR` paths in the MCP server `env` block.

Example Claude Desktop server entry:

```json
{
  "claimcheck": {
    "command": "/absolute/path/to/ClaimCheck_Daily_v3/.venv/bin/python",
    "args": ["/absolute/path/to/ClaimCheck_Daily_v3/run.py", "--serve-mcp"],
    "cwd": "/absolute/path/to/ClaimCheck_Daily_v3",
    "env": {
      "OPENAI_API_KEY": "your-openai-key",
      "ANTHROPIC_API_KEY": "your-anthropic-key",
      "OUTPUTS_DIR": "/absolute/path/to/ClaimCheck_Daily_v3/outputs",
      "DOCS_DIR": "/absolute/path/to/ClaimCheck_Daily_v3/docs",
      "MANUAL_OUTPUTS_DIR": "/absolute/path/to/ClaimCheck_Daily_v3/outputs_manual",
      "MANUAL_DOCS_DIR": "/absolute/path/to/ClaimCheck_Daily_v3/docs_manual"
    }
  }
}
```

## 8. Known Limitations

- The runtime graph does not yet include week 10 multi-agent nodes or the full week 11 HITL/memory design.
- The runtime graph now includes a simple review gate, but it does not yet pause execution or support reviewer approvals inside the graph.
- Retrieval is still the existing TF-IDF + BM25 implementation, not ChromaDB or embedding-based vector memory.
- The graph is still source-domain-centric rather than a richer entity/event GraphRAG design.
- Retry loops re-run downstream verdict/verify work for the whole selected batch rather than only the failing claim.
- Feed headlines are used as claim text and may be imprecise.

## 9. Cost Notes

As of April 12, 2026, the code defaults are:
- Claude research uses `claude-opus-4-5-20251101`
- OpenAI selection, verdict synthesis, and verification use `gpt-4o`

Official API pricing at that date:
- Claude Opus 4.5: $5 / 1M input tokens, $25 / 1M output tokens
- Claude Sonnet 4.5: $3 / 1M input tokens, $15 / 1M output tokens
- GPT-4o: $2.50 / 1M input tokens, $10 / 1M output tokens

Estimated program cost:
- Single manual claim run, no retry: about `$0.10–$0.22` total with Claude Opus 4.5, or about `$0.07–$0.15` with Claude Sonnet 4.5
- Daily run with 3 selected claims, no retry: about `$0.31–$0.69` total with Claude Opus 4.5, or about `$0.22–$0.48` with Claude Sonnet 4.5
- Retries raise cost because the pipeline may add another Claude corroboration step and extra GPT verdict / verifier passes

These figures are estimates from the current code path, not metered billing logs. Exact cost depends on article length, number of tool rounds, verifier retries, and which Claude model is configured.

Pricing sources:
- [OpenAI GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)
- [Anthropic Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)

Current runtime behavior:
- CLI runs print total estimated cost plus provider breakdown
- JSON reports persist `trace_summary`
- HTML reports render a `Run Cost & Usage` section
- MCP `check_claim` responses include `cost_summary`

## 10. Current Roadmap

The next major implementation milestones for this branch are:

1. Add a real week 10 orchestration layer to the LangGraph runtime.
2. Add week 11 confidence routing and human review checkpoints.
3. Add a real memory layer only after the runtime flow is updated to use it.
