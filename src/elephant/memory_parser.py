"""Parse free-text messages into Memory models via LLM."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import yaml

from elephant.data.models import MediaLinks, Memory, Person, PreferencesFile
from elephant.llm.prompts import parse_memories_batch, parse_memory

if TYPE_CHECKING:
    from elephant.llm.backend import LLMBackend
    from elephant.messaging.base import Attachment

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ParseResult:
    """Result of parsing a message into a memory, with a confidence score."""

    memory: Memory
    confidence: float


async def parse_memory_from_text(
    text: str,
    llm: LLMBackend,
    model: str,
    people: list[Person],
    prefs: PreferencesFile,
    source: str = "WhatsApp",
    memory_date: date | None = None,
    attachments: list[Attachment] | None = None,
) -> ParseResult:
    """Use LLM to parse free text into a structured Memory with confidence."""
    messages = parse_memory(text, people, prefs)
    response = await llm.chat(messages, model=model, temperature=0.3)

    parsed = yaml.safe_load(response.content or "")
    if not isinstance(parsed, dict):
        msg = f"LLM returned non-dict: {type(parsed).__name__}"
        raise ValueError(msg)

    today = memory_date or date.today()
    slug = _slugify(parsed.get("title", "memory"))
    memory_id = f"{today.strftime('%Y%m%d')}_{slug}"

    # Build media from attachments
    media = None
    if attachments:
        photos = [a.file_path for a in attachments if a.media_type == "photo"]
        videos = [a.file_path for a in attachments if a.media_type == "video"]
        if photos or videos:
            media = MediaLinks(photos=photos, videos=videos)

    raw_time = parsed.get("time")
    confidence = float(parsed.get("confidence", 1.0))

    memory = Memory(
        id=memory_id,
        date=today,
        time=str(raw_time) if raw_time is not None else None,
        title=parsed.get("title", text[:50]),
        type=parsed.get("type", "other"),
        description=parsed.get("description", text),
        people=parsed.get("people", []),
        location=parsed.get("location"),
        media=media,
        source=source,
        nostalgia_score=float(parsed.get("nostalgia_score", 1.0)),
        tags=parsed.get("tags", []),
    )
    return ParseResult(memory=memory, confidence=confidence)




async def parse_memories_from_document(
    caption: str,
    document_content: str,
    llm: LLMBackend,
    model: str,
    people: list[Person],
    prefs: PreferencesFile,
    source: str = "Telegram",
    attachments: list[Attachment] | None = None,
) -> list[Memory]:
    """Parse a document's contents into multiple Memory objects via LLM."""
    messages = parse_memories_batch(caption, document_content, people, prefs)
    response = await llm.chat(messages, model=model, temperature=0.3)

    parsed = yaml.safe_load(response.content or "")

    # If LLM returned a single dict, wrap it in a list
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        msg = f"LLM returned unexpected type: {type(parsed).__name__}"
        raise ValueError(msg)

    # Build media from attachments (shared across all memories from this doc)
    media = None
    if attachments:
        photos = [a.file_path for a in attachments if a.media_type == "photo"]
        videos = [a.file_path for a in attachments if a.media_type == "video"]
        if photos or videos:
            media = MediaLinks(photos=photos, videos=videos)

    memories: list[Memory] = []
    for item in parsed:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item in batch parse: %s", type(item).__name__)
            continue

        # Parse date from the item, falling back to today
        memory_date = date.today()
        raw_date = item.get("date")
        if raw_date:
            if isinstance(raw_date, date):
                memory_date = raw_date
            elif isinstance(raw_date, str):
                try:
                    memory_date = date.fromisoformat(raw_date)
                except ValueError:
                    logger.warning("Unparseable date '%s', using today", raw_date)

        slug = _slugify(item.get("title", "memory"))
        memory_id = f"{memory_date.strftime('%Y%m%d')}_{slug}"

        raw_time = item.get("time")
        memories.append(
            Memory(
                id=memory_id,
                date=memory_date,
                time=str(raw_time) if raw_time is not None else None,
                title=item.get("title", "Untitled"),
                type=item.get("type", "other"),
                description=item.get("description", ""),
                people=item.get("people", []),
                location=item.get("location"),
                media=media,
                source=source,
                nostalgia_score=float(item.get("nostalgia_score", 1.0)),
                tags=item.get("tags", []),
            )
        )

    return memories




def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    slug = text.lower().strip()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "_".join(slug.split())
    if not slug:
        slug = f"memory_{uuid.uuid4().hex[:6]}"
    return slug[:40]
