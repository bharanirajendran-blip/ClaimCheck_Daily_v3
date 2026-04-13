"""
Feed parser — ingests RSS/Atom feeds defined in feeds.yaml
and extracts candidate Claim objects.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Iterator

import feedparser
import httpx
import yaml

from .models import Claim

logger = logging.getLogger(__name__)

# How long to wait for a single feed before giving up (seconds).
# Override with FEED_TIMEOUT env var.
FEED_TIMEOUT = int(os.getenv("FEED_TIMEOUT", "15"))

# Heuristics: headline patterns that often contain checkable claims
CLAIM_SIGNALS = re.compile(
    r"\b(says|claims|reports|announces|confirms|denies|alleges|warns|"
    r"according to|study finds|data shows|experts say)\b",
    re.IGNORECASE,
)


def load_feeds(feeds_path: str | Path = "feeds.yaml") -> list[dict]:
    with open(feeds_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("feeds", [])


def harvest_claims(
    feeds_path: str | Path = "feeds.yaml",
    max_per_feed: int = 10,
) -> list[Claim]:
    """Parse all feeds and return candidate claims."""
    feeds = load_feeds(feeds_path)
    claims: list[Claim] = []

    for feed_cfg in feeds:
        name = feed_cfg.get("name", feed_cfg["url"])
        url = feed_cfg["url"]
        logger.info("Parsing feed: %s", name)

        try:
            # Fetch with an explicit timeout so a slow/dead feed can't hang
            # the whole pipeline.  feedparser.parse() has no built-in timeout,
            # so we pre-fetch with httpx and hand the content to feedparser.
            resp = httpx.get(
                url,
                timeout=FEED_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "ClaimCheckDaily/3.0 (+https://github.com/bharanirajendran-blip/ClaimCheck_Daily)"},
            )
            resp.raise_for_status()
            parsed = feedparser.parse(resp.text)
        except httpx.TimeoutException:
            logger.warning("Feed timed out after %ds, skipping: %s", FEED_TIMEOUT, name)
            continue
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch feed %s: %s", name, exc)
            continue
        except Exception as exc:
            logger.warning("Failed to parse feed %s: %s", name, exc)
            continue

        for entry in parsed.entries[:max_per_feed]:
            for claim in _extract_claims(entry, feed_name=name):
                claims.append(claim)

    logger.info("Harvested %d candidate claims from %d feeds.", len(claims), len(feeds))
    return claims


def _extract_claims(entry, feed_name: str) -> Iterator[Claim]:
    """Yield 0-or-more Claim objects from a single feed entry."""
    title: str = getattr(entry, "title", "").strip()
    link: str = getattr(entry, "link", "")
    published: str = getattr(entry, "published", None)
    summary: str = getattr(entry, "summary", "").strip()

    # Use title as the primary claim text if it looks checkable
    text = title or summary
    if not text:
        return

    # Stable, collision-resistant ID: SHA-256 over (text + url), 16 hex chars.
    # Using the URL breaks ties when two headlines are identical across outlets.
    # 16 chars (8 bytes) gives a birthday-collision threshold of ~2^32 — safe
    # for tens of thousands of claims.  The old 8-char MD5 had a 2^16 threshold.
    hash_input = f"{text}\0{link}".encode()
    claim_id = hashlib.sha256(hash_input).hexdigest()[:16]

    yield Claim(
        id=claim_id,
        text=text,
        source=feed_name,
        url=link,
        published_at=published,
        feed_name=feed_name,
    )
