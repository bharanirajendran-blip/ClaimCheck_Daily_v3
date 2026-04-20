"""
Researcher — Claude-powered deep research agent with web fetch tool
--------------------------------------------------------------------
Responsibilities:
  1. Accept a single Claim from the Director
  2. Use Claude (claude-opus-4-5) with:
       - Extended thinking (10k budget tokens) for deep reasoning
       - web_search tool to find relevant URLs via Serper (Google results)
       - fetch_url tool to read full article text from those URLs
  3. Run an agentic tool-use loop:
       - Claude reasons, calls web_search to find sources
       - Claude calls fetch_url to read the best ones
       - Each result is returned as a tool_result
       - Loop ends when Claude returns a final text response
  4. Return a ResearchResult (findings + extracted sources)

The tool-use loop means Claude is no longer limited to training knowledge —
it can read the actual source articles published in 2025/2026.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from dataclasses import dataclass, field

import anthropic

from .models import Claim, ResearchResult
from .observability import get_tracer
from .tools import TOOL_DEFINITIONS, execute_tool
from .utils import retry

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a meticulous research analyst for ClaimCheck Daily.
    You have two tools: web_search and fetch_url.

    Your task is to evaluate a claim by finding and reading authoritative sources.

    ════════════════════════════════════════════════════════════
    MANDATORY SOURCE DIVERSITY REQUIREMENT
    ════════════════════════════════════════════════════════════
    You MUST fetch articles from AT LEAST 2 DIFFERENT domains before writing
    your final report. A single-domain report is INCOMPLETE and unacceptable.

    Step-by-step process:
      1. Fetch the primary source URL (if provided).
      2. Use web_search to find a SECOND source from a DIFFERENT domain.
      3. fetch_url that second source.
      4. ONLY THEN write your final findings — after reading BOTH sources.

    Do NOT skip step 2–3. Even if the primary source seems comprehensive,
    independent corroboration is required for every claim.
    ════════════════════════════════════════════════════════════

    TOOL USAGE STRATEGY:
    - If a specific source URL is provided: fetch it first as your primary source.
    - Always use web_search to find a second authoritative source from a DIFFERENT
      domain, then fetch_url that result. This is not optional.
    - If no source URL is provided:
        * Use web_search to find 2 authoritative sources, then fetch both.
        * If web_search is unavailable: fetch 2 known authoritative URLs directly
          (e.g. .gov, .edu, Reuters, AP, Nature, PolitiFact, FactCheck.org).

    SOURCE PAIRING GUIDE (match to claim type):
      Science / health  → primary institution or journal
                          + science news outlet (ScienceDaily, Nature News, STAT)
      Policy / politics → official .gov document or primary statement
                          + independent fact-checker (FactCheck.org, PolitiFact)
      Technology        → original announcement or company source
                          + tech journalist (Wired, MIT Tech Review, Ars Technica)
      General factual   → authoritative news outlet (AP, Reuters, BBC)
                          + independent verification from a different outlet

    OTHER RULES:
    - Use at most 3 fetch_url calls total (primary + corroboration + optional 3rd).
    - Do NOT fetch the same domain twice.
    - Do NOT chase dead links; if a fetch fails, search for an alternative and fetch that.
    - Base your verdict only on what you successfully read.

    Structure your final response as follows:

    ## Sub-questions
    [numbered list]

    ## Evidence Assessment
    [detailed prose, balanced, based on what you read from BOTH sources]

    ## Supporting evidence
    [bullet points with sources — must reference at least 2 different domains]

    ## Contradicting evidence
    [bullet points with sources]

    ## Caveats & Missing Context
    [prose]

    ## Key Sources
    [list of {"title": "...", "url": "...", "reliability": "high|medium|low"}]
""")

MAX_TOOL_ROUNDS = 8  # web_search + fetch_url (×2 sources) + synthesis + slack for retries
# At this round, inject a hard deadline message so Claude writes findings
# instead of continuing to search, preventing "Research incomplete" fallback.
SYNTHESIS_DEADLINE_ROUND = MAX_TOOL_ROUNDS - 2  # round 6 of 8

# Corroborate loop uses a fixed thinking budget independent of self.thinking_budget.
# Anthropic requires budget_tokens >= 1024; we use the minimum here since
# corroboration is a single-source fetch — not a deep reasoning task.
CORROBORATE_MAX_TOKENS    = 4000
CORROBORATE_THINKING_BUDGET = 1024  # must be < CORROBORATE_MAX_TOKENS

CORROBORATE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a fact-checking researcher tasked with finding ONE corroborating source.

    You will receive a claim and a list of domains already consulted.
    Find and read EXACTLY ONE authoritative source from a different domain that can
    independently support or refute the claim.

    TOOL USAGE:
    - Use web_search first to find a relevant URL from an authoritative source
      that is NOT in the list of already-consulted domains.
    - Then use fetch_url to read that URL.
    - Do NOT guess URLs — search first, then fetch the best result.

    RULES:
    - Fetch exactly ONE URL — choose the most authoritative result from your search.
    - Do NOT fetch from any domain listed as already consulted.
    - Prefer results from: .gov, .edu, established journals, recognized references.
    - If the fetch fails, say so and do not retry.

    Return your findings as:

    ## Corroborating Evidence
    [prose summary of what the source says about the claim]

    ## Key Sources
    [{"title": "...", "url": "...", "reliability": "high|medium|low"}]
""")


def _block_to_dict(block) -> dict | None:
    """
    Convert an Anthropic SDK content block to a plain dict for the messages API.
    Avoids the Pydantic v2 'by_alias' serialization conflict when passing
    SDK objects directly back into the messages list.
    """
    # Prefer Anthropic SDK serialization so block shape stays API-compatible
    # (important for thinking blocks + signatures).
    try:
        dumped = block.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass

    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if t == "thinking":
        signature = getattr(block, "signature", None)
        thinking = getattr(block, "thinking", None)
        if signature and thinking:
            return {"type": "thinking", "thinking": thinking, "signature": signature}
        logger.warning("Skipping malformed thinking block while replaying assistant turn.")
        return None
    if t:
        return {"type": t}
    return None


@dataclass
class Researcher:
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"))
    # max_tokens must exceed thinking_budget; override via CLAUDE_MAX_TOKENS.
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS", "10000")))
    use_extended_thinking: bool = True
    # Controls how deeply Claude reasons before committing to a finding.
    # Override via CLAUDE_THINKING_BUDGET (must stay below max_tokens).
    thinking_budget: int = field(default_factory=lambda: int(os.getenv("CLAUDE_THINKING_BUDGET", "3000")))
    # Max tool-call rounds before forcing synthesis.  MCP server sets
    # CLAIMCHECK_MAX_TOOL_ROUNDS=5 to finish well within client timeout.
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("CLAIMCHECK_MAX_TOOL_ROUNDS", str(MAX_TOOL_ROUNDS))))
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Synthesis deadline = 2 rounds before the hard cap.
        self.synthesis_deadline_round: int = max(1, self.max_tool_rounds - 2)
        # Fail fast if the token budget config is invalid rather than letting
        # every research call blow up with an opaque Anthropic API error.
        # We clamp rather than hard-error so a partial misconfiguration
        # (e.g. someone sets CLAUDE_THINKING_BUDGET too high) still runs.
        min_gap = 1024  # Anthropic requires max_tokens > thinking_budget
        if self.thinking_budget >= self.max_tokens:
            clamped = max(self.max_tokens - min_gap, 1)
            logger.warning(
                "CLAUDE_THINKING_BUDGET (%d) must be less than CLAUDE_MAX_TOKENS (%d). "
                "Clamping thinking_budget to %d.",
                self.thinking_budget, self.max_tokens, clamped,
            )
            self.thinking_budget = clamped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, claim: Claim) -> ResearchResult:
        """Deep-research a single claim using tool-use loop, return findings."""
        logger.info("Researcher investigating claim %s: %s", claim.id, claim.text[:80])

        raw_text, fetched_pages = self._run_tool_loop(claim)
        sources = self._extract_sources(raw_text)

        return ResearchResult(
            claim_id=claim.id,
            findings=raw_text,
            sources=sources,
            fetched_pages=fetched_pages,
        )

    def corroborate(
        self,
        claim_id: str,
        claim_text: str,
        existing_domains: list[str],
    ) -> ResearchResult | None:
        """
        Fetch exactly one corroborating source from a domain not already consulted.
        Called by the retry loop when the verifier flags weak grounding or missing
        citations. Returns a ResearchResult whose fetched_pages and findings will
        be chunked into the evidence store before the next retrieval pass.
        """
        logger.info(
            "[corroborate] Seeking extra source for claim %s (avoiding: %s)",
            claim_id, existing_domains,
        )
        domain_list = ", ".join(existing_domains) if existing_domains else "none"
        user_msg = (
            f"Claim:\n\"{claim_text}\"\n\n"
            f"Domains already consulted: {domain_list}\n\n"
            "Please fetch ONE corroborating source from a different authoritative domain "
            "and summarize what it says about this claim."
        )
        messages = [{"role": "user", "content": user_msg}]
        fetched_pages: list[dict] = []

        # Lightweight loop — max 4 rounds (web_search + fetch_url + synthesis + slack)
        for round_num in range(1, 5):
            kwargs: dict = dict(
                model=self.model,
                max_tokens=CORROBORATE_MAX_TOKENS,
                system=CORROBORATE_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            if self.use_extended_thinking:
                # Use a fixed budget constant that is guaranteed to satisfy
                # Anthropic's constraint: 1024 <= budget_tokens < max_tokens.
                # Never derive this from self.thinking_budget which may have
                # been clamped to 1 by __post_init__ on misconfigured envs.
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": CORROBORATE_THINKING_BUDGET,
                }
                kwargs["temperature"] = 1
                logger.debug(
                    "[corroborate] round %d — max_tokens=%d budget_tokens=%d",
                    round_num, CORROBORATE_MAX_TOKENS, CORROBORATE_THINKING_BUDGET,
                )

            try:
                with get_tracer().span("corroborate_node", model=self.model) as span:
                    response = self._client.messages.create(**kwargs)
                    span.record_anthropic(response)
                    span.set_extra(round=round_num, claim_id=claim_id)
            except Exception as exc:
                logger.error(
                    "[corroborate] API error on round %d (model=%s, max_tokens=%d, "
                    "budget_tokens=%d): %s",
                    round_num, self.model, CORROBORATE_MAX_TOKENS,
                    CORROBORATE_THINKING_BUDGET, exc,
                )
                return None

            if response.stop_reason == "end_turn":
                parts = [b.text for b in response.content if hasattr(b, "text")]
                raw_text = "\n".join(parts).strip()
                sources = self._extract_sources(raw_text)
                return ResearchResult(
                    claim_id=claim_id,
                    findings=raw_text,
                    sources=sources,
                    fetched_pages=fetched_pages,
                )

            if response.stop_reason == "tool_use":
                assistant_content = [
                    d for d in (_block_to_dict(b) for b in response.content)
                    if d is not None
                ]
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_text = execute_tool(block.name, block.input)
                        if block.name == "fetch_url":
                            fetched_pages.append({
                                "url": block.input.get("url", ""),
                                "content": result_text,
                            })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            logger.warning("[corroborate] Unexpected stop_reason: %s", response.stop_reason)
            return None

        logger.warning("[corroborate] Max rounds reached for claim %s", claim_id)
        return None

    # ------------------------------------------------------------------
    # Agentic tool-use loop
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_of(url: str) -> str:
        """Extract the bare hostname (e.g. 'bbc.com') from a URL."""
        import urllib.parse
        try:
            return urllib.parse.urlparse(url).netloc.lstrip("www.")
        except Exception:
            return url

    @retry(times=3, delay=3)
    def _run_tool_loop(self, claim: Claim) -> tuple[str, list[dict]]:  # noqa: C901
        """
        Agentic loop:
          1. Send claim + tools to Claude
          2. If Claude calls fetch_url → execute it → append tool_result → repeat
          3. When Claude returns a final text response → extract and return it

        Diversity enforcement: if Claude reaches round 4 having only fetched one
        domain, a reminder is injected asking it to search for a second source
        before writing its findings.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Claim to investigate:\n\n\"{claim.text}\"\n\n"
                    f"Original source: {claim.source}\n"
                    f"Source URL: {claim.url or 'not available'}\n"
                    f"Published: {claim.published_at or 'unknown'}\n\n"
                    "Remember: you MUST fetch articles from at least 2 DIFFERENT domains "
                    "before writing your final report. Fetch the source article first, "
                    "then search for and fetch one more source from a different domain."
                ),
            }
        ]
        fetched_pages: list[dict] = []
        fetched_domains: set[str] = set()
        _diversity_reminder_sent = False

        _synthesis_deadline_sent = False

        for round_num in range(1, self.max_tool_rounds + 1):
            # ── Diversity nudge ───────────────────────────────────────────────
            # If we're at round 4+, Claude has done at least 3 tool exchanges but
            # has only fetched from one domain — remind it before it writes up.
            if (
                not _diversity_reminder_sent
                and round_num >= 4
                and len(fetched_domains) < 2
                and fetched_domains  # at least one fetch happened
            ):
                domain_seen = next(iter(fetched_domains))
                reminder = (
                    f"⚠️ DIVERSITY REMINDER: You have only fetched content from "
                    f"'{domain_seen}' so far. You MUST fetch at least one more source "
                    f"from a DIFFERENT domain before writing your final report. "
                    f"Please use web_search to find another authoritative source, "
                    f"then fetch_url it now."
                )
                messages.append({"role": "user", "content": reminder})
                _diversity_reminder_sent = True
                logger.info(
                    "[research] Diversity reminder injected at round %d "
                    "(only domain so far: %s)", round_num, domain_seen,
                )

            # ── Synthesis deadline ────────────────────────────────────────────
            # Two rounds before the hard cap, stop all tool use and demand the
            # final structured report. This prevents the "Research incomplete:
            # maximum tool rounds reached" fallback when Serper is down and
            # Claude keeps guessing URLs that 404/403.
            if not _synthesis_deadline_sent and round_num >= self.synthesis_deadline_round:
                sources_read = len(fetched_pages)
                domains_read = len(fetched_domains)
                deadline_msg = (
                    f"⏱️ SYNTHESIS DEADLINE: You have {self.max_tool_rounds - round_num + 1} "
                    f"round(s) remaining. You have successfully read {sources_read} "
                    f"page(s) from {domains_read} domain(s). "
                    f"STOP fetching now and write your complete final report using "
                    f"the evidence you already have. Do NOT call any more tools."
                )
                messages.append({"role": "user", "content": deadline_msg})
                _synthesis_deadline_sent = True
                logger.info(
                    "[research] Synthesis deadline injected at round %d "
                    "(fetched %d pages from %d domains)", round_num, sources_read, domains_read,
                )

            kwargs: dict = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                system=RESEARCH_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if self.use_extended_thinking:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
                kwargs["temperature"] = 1  # required for extended thinking

            with get_tracer().span("research_node", model=self.model) as span:
                response = self._client.messages.create(**kwargs)
                span.record_anthropic(response)
                span.set_extra(round=round_num, claim_id=getattr(claim, "id", "?"))
            logger.debug("Round %d stop_reason=%s", round_num, response.stop_reason)

            # ── Claude finished — collect all text blocks ─────────────
            if response.stop_reason == "end_turn":
                parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                if len(fetched_domains) < 2:
                    logger.warning(
                        "[research] Claim %s finished with only %d distinct domain(s): %s. "
                        "Evidence may lack diversity.",
                        getattr(claim, "id", "?"), len(fetched_domains), fetched_domains,
                    )
                else:
                    logger.info(
                        "[research] Claim %s — fetched %d distinct domain(s): %s",
                        getattr(claim, "id", "?"), len(fetched_domains), fetched_domains,
                    )
                return "\n".join(parts).strip(), fetched_pages

            # ── Claude wants to use a tool ────────────────────────────
            if response.stop_reason == "tool_use":
                # If the synthesis deadline has already been sent and Claude
                # is still trying to call tools, refuse the tool calls and
                # force it to write its report with what it has.
                if _synthesis_deadline_sent:
                    logger.warning(
                        "[research] Claude called tool(s) after synthesis deadline "
                        "at round %d — refusing and forcing final synthesis.", round_num,
                    )
                    assistant_content = [
                        block_dict
                        for block_dict in (_block_to_dict(b) for b in response.content)
                        if block_dict is not None
                    ]
                    messages.append({"role": "assistant", "content": assistant_content})
                    # Return stub tool results so the messages list stays valid,
                    # then demand the final report.
                    stub_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Tool call refused: synthesis deadline reached. Write your final report now.",
                        }
                        for block in response.content
                        if block.type == "tool_use"
                    ]
                    messages.append({"role": "user", "content": stub_results + [{
                        "type": "text",
                        "text": "Write your complete final report now using the evidence already gathered. No more tool calls.",
                    }]})
                    continue

                # Serialize SDK content blocks to plain dicts to avoid
                # Pydantic v2 / Anthropic SDK 'by_alias' serialization conflict
                assistant_content = [
                    block_dict
                    for block_dict in (_block_to_dict(b) for b in response.content)
                    if block_dict is not None
                ]
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # Execute every tool call Claude requested
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(
                            "[tool] Claude calling %s with %s", block.name, block.input
                        )
                        result_text = execute_tool(block.name, block.input)
                        if block.name == "fetch_url":
                            url = block.input.get("url", "")
                            fetched_pages.append({
                                "url": url,
                                "content": result_text,
                            })
                            domain = self._domain_of(url)
                            fetched_domains.add(domain)
                            logger.info(
                                "[research] Fetched domain '%s' (total distinct domains: %d)",
                                domain, len(fetched_domains),
                            )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                # Return all tool results in one user turn
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — break and return whatever text we have
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            parts = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            return "\n".join(parts).strip(), fetched_pages

        logger.warning("Tool loop hit max rounds (%d) for claim %s", self.max_tool_rounds, claim.id)
        return "Research incomplete: maximum tool rounds reached.", fetched_pages

    # ------------------------------------------------------------------
    # Source extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(raw: str) -> list[dict]:
        """Best-effort extraction of sources from JSON snippets or markdown tables."""
        import json

        sources: list[dict] = []
        seen: set[tuple[str, str]] = set()

        json_pattern = r'\{[^{}]*"title"[^{}]*\}'
        for match in re.finditer(json_pattern, raw, re.DOTALL):
            try:
                item = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            if title or url:
                key = (title, url)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "title": title,
                        "url": url,
                        "reliability": str(item.get("reliability", "")).strip(),
                    })

        in_key_sources = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("## key sources"):
                in_key_sources = True
                continue
            if in_key_sources and stripped.startswith("## "):
                break
            if not in_key_sources or "|" not in stripped:
                continue
            if stripped.startswith("| Title") or stripped.startswith("|-------"):
                continue

            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) < 3:
                continue
            title, url, reliability = cells[:3]
            reliability = reliability.replace("**", "").strip()
            key = (title, url)
            if title and key not in seen:
                seen.add(key)
                sources.append({
                    "title": title,
                    "url": "" if url.lower() == "referenced in article" else url,
                    "reliability": reliability,
                })

        return sources
