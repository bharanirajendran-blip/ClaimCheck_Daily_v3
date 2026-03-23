"""
Researcher — Claude-powered deep research agent with web fetch tool
--------------------------------------------------------------------
Responsibilities:
  1. Accept a single Claim from the Director
  2. Use Claude (claude-opus-4-5) with:
       - Extended thinking (10k budget tokens) for deep reasoning
       - fetch_url tool so Claude can read live article content
  3. Run an agentic tool-use loop:
       - Claude reasons, decides to fetch a URL
       - We fetch it, return the text as a tool_result
       - Claude continues reasoning with real content
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
from .tools import TOOL_DEFINITIONS, execute_tool
from .utils import retry

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a meticulous research analyst for ClaimCheck Daily.
    You have access to a fetch_url tool to read live web articles.

    Your task is to evaluate a claim by fetching and reading its source article.

    IMPORTANT RULES:
    - Fetch the source URL provided first. That is your primary evidence.
    - Fetch ONE corroborating source from a DIFFERENT domain when any of these apply:
        * the claim contains a specific date, statistic, or named event
        * the primary source is a news summary, blog, or indirect reference
          (not the original study, official document, or primary source)
        * the claim is contested or involves numbers that could be misrepresented
      If none of the above apply and the primary source is itself authoritative
      (e.g. a .gov site, original study, official archive), you may skip the second fetch.
    - Do NOT fetch more than 2 sources total.
    - Do NOT chase dead links or keep fetching if URLs return errors; move on immediately.
    - Match your corroborating source to the claim type:
        * Science / health claim  → primary institution or journal
                                    + science news outlet (ScienceDaily, Nature News)
        * Policy / government     → official document or .gov source
                                    + independent fact-checker (FactCheck.org, PolitiFact)
        * Historical claim        → official archive (NASA, Smithsonian, national records)
                                    + established reference (Britannica, major encyclopedia)
        * General factual claim   → authoritative news outlet + independent verification
    - Base your verdict on what you successfully read.

    Structure your final response as follows:

    ## Sub-questions
    [numbered list]

    ## Evidence Assessment
    [detailed prose, balanced, based on what you read]

    ## Supporting evidence
    [bullet points with sources]

    ## Contradicting evidence
    [bullet points with sources]

    ## Caveats & Missing Context
    [prose]

    ## Key Sources
    [list of {"title": "...", "url": "...", "reliability": "high|medium|low"}]
""")

MAX_TOOL_ROUNDS = 5  # enough rounds to fetch + synthesize findings

CORROBORATE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a fact-checking researcher tasked with finding ONE corroborating source.

    You will receive a claim and a list of domains already consulted.
    Fetch EXACTLY ONE URL from a different, authoritative domain that can independently
    support or refute the claim. Then write a brief summary of what you found.

    RULES:
    - Fetch exactly ONE URL — choose the most authoritative source available.
    - Do NOT fetch from any domain listed as already consulted.
    - Prefer: .gov, .edu, established journals, recognized reference sources.
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
    max_tokens: int = 10000        # must be greater than thinking_budget
    use_extended_thinking: bool = True
    thinking_budget: int = 3000    # enough to process fetched article content
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, init=False, repr=False
    )

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

        # Lightweight loop — max 2 rounds (one fetch + one synthesis)
        for round_num in range(1, 3):
            kwargs: dict = dict(
                model=self.model,
                max_tokens=4000,
                system=CORROBORATE_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            if self.use_extended_thinking:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1000}
                kwargs["temperature"] = 1

            try:
                response = self._client.messages.create(**kwargs)
            except Exception as exc:
                logger.warning("[corroborate] API error on round %d: %s", round_num, exc)
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

    @retry(times=3, delay=3)
    def _run_tool_loop(self, claim: Claim) -> tuple[str, list[dict]]:  # noqa: C901
        """
        Agentic loop:
          1. Send claim + tools to Claude
          2. If Claude calls fetch_url → execute it → append tool_result → repeat
          3. When Claude returns a final text response → extract and return it
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Claim to investigate:\n\n\"{claim.text}\"\n\n"
                    f"Original source: {claim.source}\n"
                    f"Source URL: {claim.url or 'not available'}\n"
                    f"Published: {claim.published_at or 'unknown'}\n\n"
                    "Please fetch the source article first, then research this claim thoroughly."
                ),
            }
        ]
        fetched_pages: list[dict] = []

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
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

            response = self._client.messages.create(**kwargs)
            logger.debug("Round %d stop_reason=%s", round_num, response.stop_reason)

            # ── Claude finished — collect all text blocks ─────────────
            if response.stop_reason == "end_turn":
                parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(parts).strip(), fetched_pages

            # ── Claude wants to use a tool ────────────────────────────
            if response.stop_reason == "tool_use":
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
                            fetched_pages.append({
                                "url": block.input.get("url", ""),
                                "content": result_text,
                            })
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

        logger.warning("Tool loop hit max rounds (%d) for claim %s", MAX_TOOL_ROUNDS, claim.id)
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
