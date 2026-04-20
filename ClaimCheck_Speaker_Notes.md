# ClaimCheck Daily — Speaker Notes
Bharani Rajendran · Grad5900 Agentic AI

---

## Slide 1 — Title
Three generations of the same project — v1 is a basic fetch agent, v2 adds RAG and verification, v3 adds MCP and a full LangGraph pipeline. The progression is the story.

---

## Slide 2 — The Challenge
News publishes faster than humans can check it. LLMs hallucinate without evidence grounding. We need a system that collects evidence, verifies verdicts independently, and is accessible as a tool — not just a script.

---

## Slide 3 — Three Generations at a Glance
v1: single Claude agent, no persistence, no verification. v2: added persistent store, hybrid retrieval, LLM judge. v3: MCP interface, web search, review gate, full LangGraph. Each version fixes what the previous couldn't do.

---

## Slide 4 — How v1 Works
Five steps: parse RSS feeds → GPT-4o picks the best claims → Claude fetches and reads the source → GPT-4o writes a verdict → publish HTML and JSON. Simple, linear, no feedback loop.

---

## Slide 5 — v1 Components & What Was Missing
v1 worked, but every run started from zero — no evidence reuse. Claude guessed URLs from training knowledge (no live search). One LLM both researched and judged — no independence. Verdicts had no quality gate. These four gaps drove v2.

---

## Slide 6 — What v2 Added
Five additions: persistent evidence store, hybrid TF-IDF + BM25 retrieval, a NetworkX knowledge graph for cross-run context, an independent GPT-4o verifier scoring on 4 RAGAS dimensions, and a self-correcting retry loop when verdicts were weak.

---

## Slide 7 — v2 Components in Detail
Six components across two rows. Director selects and synthesizes. Researcher fetches and stores. Evidence Store chunks by type (raw source, summary, metadata). HybridRetriever merges TF-IDF and BM25 with claim-local boosts. Knowledge Graph adds domain context. Verifier judges on 4 dimensions.

---

## Slide 8 — v2 → v3: Gaps & Solutions
v2 still required the CLI — no external access. Low-confidence verdicts published without a safety net. Claude still guessed URLs without live search. No single-claim interactive path. v3 fixes all four: MCP server, review gate, Serper API web search, and `--claim` flag.

---

## Slide 9 — 11-Node LangGraph Pipeline
Two rows of nodes connected in sequence, with a dashed retry branch from verify back to retrieve. Row 1: harvest → select → research → store_evidence → graph_context. Row 2: retrieve → verdict → verify → review_gate → publish. The purple dashed loop is the self-correction path — max 2 retries.

---

## Slide 10 — What Each Node Does
Each of the 11 nodes does one thing. Harvest parses feeds. Select picks claims. Research runs the tool loop. Store chunks pages. Graph updates domain context. Retrieve scores chunks. Verdict writes the label. Verify judges it. Revise_query targets the retry. Review_gate handles low confidence. Publish writes HTML and JSON.

---

## Slide 11 — Research Engine
Two tools working together: web_search calls the Serper API to find authoritative URLs, fetch_url reads them. Claude runs a ReAct loop — search, reason, fetch, reason, fetch again — up to 8 rounds for CLI or 5 for MCP. Extended thinking guides the strategy. Mandatory: at least 2 different domains before writing findings.

---

## Slide 12 — Evidence Store & Hybrid Retrieval
Three chunk types with different weights: raw_source ×1.15, summary ×1.0, source_metadata ×0.75. Retrieval combines TF-IDF cosine (60%) and BM25 keyword (40%). Claim-local hits come first with a ×1.25 boost. Domain cap: max 2 chunks per domain to prevent one source from flooding the evidence window.

---

## Slide 13 — Why Persistence Matters
Articles get edited, deleted, or taken down. ChatGPT starts from zero every session and has no record of what a page said yesterday. ClaimCheck captures source text at research time with URL and date provenance — the same principle as the Wayback Machine. Day 30 has access to all evidence from days 1-29.

---

## Slide 14 — Self-Correcting Verification Loop
The verifier scores verdicts on 4 RAGAS dimensions: groundedness, citation quality, contradiction, and assumptions. If groundedness or citation quality fail, the pipeline triggers a targeted retry — fetches one more source from a different domain, re-retrieves, rewrites the verdict. Max 2 retries. The synthesis deadline fires 2 rounds before the cap to guarantee Claude writes findings before time runs out.

---

## Slide 15 — MCP Server
Three tools at different levels of the stack. check_claim runs the full pipeline on a single claim and returns structured JSON. search_evidence queries the existing store without new research. get_verdict_history is pure archive lookup. MCP timeout fix: max_tool_rounds=5 and max_retries=0 for the MCP path keeps response time under 60 seconds.

---

## Slide 16 — v3 Runtime Optimizations
Four targeted fixes: retriever cache avoids rebuilding TF-IDF on every call. Synthesis deadline prevents Claude from exhausting all rounds without writing findings. MCP timeout tuning caps rounds and disables retries for interactive use. Director claim-framing guard detects evaluative meta-headlines and keeps GPT-4o judging the full claim.

---

## Slide 17 — What Was Built
Three versions, two LLM providers, 11 LangGraph nodes, 3 MCP tools, hybrid TF-IDF+BM25 retrieval, 4 RAGAS dimensions. v1 fetches and publishes. v2 adds persistence and verification. v3 adds MCP access and a full production pipeline. All open source.

---

## Slide 18 — Closing
ChatGPT gives you an answer. ClaimCheck gives you a system — with persistent memory, cross-model verification, self-correction, and daily automation. That's the difference between a chatbot and a pipeline.

---

*github.com/bharanirajendran-blip/ClaimCheck_Daily_v3*
