"""
tools.py — Web fetch tool for the Claude Researcher
-----------------------------------------------------
Defines the `fetch_url` tool that Claude can call during research
to read live article content instead of relying on training knowledge.

Flow:
  1. Researcher passes TOOL_DEFINITIONS to Claude via the API
  2. Claude returns a tool_use block when it wants to fetch a URL
  3. execute_tool() is called with the tool name + inputs
  4. The fetched text is returned to Claude as a tool_result
  5. Claude continues reasoning with the real article content
"""

from __future__ import annotations

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

# Anthropic tool definition — passed directly to messages.create(tools=[...])
TOOL_DEFINITIONS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a web page or article. "
            "Use this to read the full text of a source article when researching a claim. "
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
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call from Claude and return the result as a string."""
    if tool_name == "fetch_url":
        return fetch_url(tool_input["url"])
    return f"Unknown tool: {tool_name}"


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
