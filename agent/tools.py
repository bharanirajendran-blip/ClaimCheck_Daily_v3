"""
tools.py — Web fetch and search tools for the Claude Researcher
---------------------------------------------------------------
Defines the tools that Claude can call during research:

  fetch_url   — reads the full text of a specific URL (httpx + BeautifulSoup)
  web_search  — queries the Serper API (Google results) and returns titles, URLs, snippets

Typical ReAct flow with both tools available:
  1. Claude calls web_search("claim keywords") to find relevant articles
  2. Claude reads the search results (titles + snippets + URLs)
  3. Claude calls fetch_url(best_url) to read the full article text
  4. Claude reasons over the content and produces findings

Set SERPER_API_KEY env var to enable web_search.
Get a free key (2,500 queries, no credit card) at:
  https://serper.dev
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Maximum characters of article text returned to Claude per fetch.
# Increase via MAX_CONTENT_CHARS env var when using a larger context window.
MAX_CONTENT_CHARS = int(os.getenv("MAX_CONTENT_CHARS", "4000"))

# Per-request HTTP timeout in seconds.  Increase for slow networks.
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15"))

# Serper API key — set via SERPER_API_KEY env var.
# If not set, web_search returns a helpful error and Claude falls back to
# fetch_url with guessed URLs (same behaviour as before this change).
# Get a free key (2,500 queries, no credit card) at https://serper.dev
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Number of search results returned per web_search call.
# 5 gives Claude enough to choose from without burning context.
SERPER_SEARCH_COUNT = int(os.getenv("SERPER_SEARCH_COUNT", "5"))

# Serper API endpoint (Google Search results)
SERPER_SEARCH_URL = "https://google.serper.dev/search"

# Anthropic tool definitions — passed directly to messages.create(tools=[...])
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for articles and sources relevant to a claim. "
            "Returns titles, URLs, and snippets for the top results. "
            "Use this FIRST to discover relevant URLs, then use fetch_url to read them. "
            "This lets you find authoritative sources instead of guessing URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Be specific — include key names, "
                        "dates, statistics, or quoted phrases from the claim."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the full text content of a specific web page or article. "
            "Use this after web_search to read the full article at a URL you found. "
            f"Returns the cleaned plain text of the page up to {MAX_CONTENT_CHARS} characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch (must start with http:// or https://)",
                }
            },
            "required": ["url"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call from Claude and return the result as a string."""
    if tool_name == "fetch_url":
        return fetch_url(tool_input["url"])
    if tool_name == "web_search":
        return web_search(tool_input["query"])
    return f"Unknown tool: {tool_name}"


def web_search(query: str) -> str:
    """
    Query the Serper API (Google results) and return a formatted list of results.

    Returns titles, URLs, and snippets for the top SERPER_SEARCH_COUNT organic
    results. Claude uses this to discover relevant URLs before calling fetch_url.

    Requires SERPER_API_KEY to be set. Returns a clear error message if the key
    is missing so Claude degrades gracefully (falls back to guessing URLs).

    Free tier: 2,500 queries, no credit card. Sign up at https://serper.dev
    """
    if not SERPER_API_KEY:
        logger.warning("[tools] web_search called but SERPER_API_KEY is not set")
        return (
            "web_search is not available (SERPER_API_KEY not configured).\n"
            "FALLBACK INSTRUCTION: infer the most likely authoritative URLs for this "
            "claim from your knowledge and use fetch_url directly. "
            "Prefer well-known outlets such as Reuters, AP News, PolitiFact, "
            "FactCheck.org, Nature, ScienceDaily, or relevant .gov / .edu sources."
        )

    logger.info("[tools] web_search query: %r", query)

    try:
        response = httpx.post(
            SERPER_SEARCH_URL,
            json={"q": query, "num": SERPER_SEARCH_COUNT},
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] Serper HTTP error: %s", e)
        return f"web_search failed: HTTP {e.response.status_code}. Try fetch_url with a known URL instead."
    except httpx.RequestError as e:
        logger.warning("[tools] Serper request error: %s", e)
        return f"web_search failed: {e}. Try fetch_url with a known URL instead."

    try:
        data = response.json()
        results = data.get("organic", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("[tools] Serper response parse error: %s", e)
        return "web_search returned an unreadable response. Try fetch_url with a known URL instead."

    if not results:
        logger.info("[tools] Serper returned no results for %r", query)
        return f"No search results found for: {query!r}. Try a different query or use fetch_url directly."

    lines = [f'Search results for "{query}":\n']
    for i, result in enumerate(results, start=1):
        title = result.get("title", "No title").strip()
        url = result.get("link", "").strip()
        snippet = result.get("snippet", "").strip()
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        lines.append("")

    formatted = "\n".join(lines).strip()
    logger.info("[tools] Serper returned %d results for %r", len(results), query)
    return formatted


def fetch_url(url: str) -> str:
    """
    Fetch a URL and return cleaned plain text.
    Strips HTML tags, scripts, nav, headers, and footers.
    Truncates to MAX_CONTENT_CHARS so we don't blow the context window.
    """
    logger.info("[tools] Fetching URL: %s", url)

    try:
        response = httpx.get(
            url,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ClaimCheckDaily/1.0; "
                    "+https://github.com/bharanirajendran-blip/ClaimCheck_Daily)"
                )
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] HTTP error fetching %s: %s", url, e)
        return f"Error fetching page: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        logger.warning("[tools] Request error fetching %s: %s", url, e)
        return f"Error fetching page: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # Extract text and clean whitespace
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

    logger.info("[tools] Fetched %d chars from %s", len(text), url)
    return text
