"""
review_queue.py — Persist pending human-review tasks for low-confidence claims.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import ReviewQueueItem

logger = logging.getLogger(__name__)

REVIEW_QUEUE_FILE = "review_queue.json"


def load_review_queue(outputs_dir: str | Path) -> list[ReviewQueueItem]:
    """Load persisted review items from disk."""
    path = Path(outputs_dir) / REVIEW_QUEUE_FILE
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ReviewQueueItem.model_validate(item) for item in raw]


def upsert_review_queue(
    outputs_dir: str | Path,
    new_items: list[ReviewQueueItem],
) -> list[ReviewQueueItem]:
    """Merge new review items into the persisted review queue."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    path = outputs_path / REVIEW_QUEUE_FILE

    existing: dict[str, ReviewQueueItem] = {}
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            review = ReviewQueueItem.model_validate(item)
            existing[review.review_id] = review

    for item in new_items:
        existing[item.review_id] = item

    merged = sorted(
        existing.values(),
        key=lambda item: (item.date_slug, item.created_at, item.review_id),
        reverse=True,
    )
    path.write_text(
        json.dumps([item.model_dump() for item in merged], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[review_queue] Upserted %d review items → %s", len(new_items), path)
    return merged
