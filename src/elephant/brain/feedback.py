"""Classify feedback sentiment and adjust preference weights."""

import logging
from typing import Any

from elephant.data.models import Memory, NostalgiaWeights, PreferencesFile
from elephant.data.store import DataStore
from elephant.git_ops import GitRepo
from elephant.llm.backend import LLMBackend
from elephant.llm.prompts import classify_sentiment

logger = logging.getLogger(__name__)

WEIGHT_STEP = 0.1
WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.0


def _clamp(value: float) -> float:
    return max(WEIGHT_MIN, min(WEIGHT_MAX, value))


async def classify_feedback_sentiment(
    text: str,
    llm: LLMBackend,
    model: str,
) -> str:
    """Classify feedback text as positive/neutral/negative."""
    messages = classify_sentiment(text)
    response = await llm.chat(messages, model=model, temperature=0.1)
    label = (response.content or "").strip().lower()
    if label in ("positive", "neutral", "negative"):
        return label
    return "neutral"


def extract_memory_features(memories: list[Memory]) -> dict[str, Any]:
    """Extract features from digest memories for weight adjustment."""
    has_milestone = any(m.type in ("milestone", "celebration") for m in memories)
    has_mundane = any(m.type not in ("milestone", "celebration") for m in memories)
    avg_people = sum(len(m.people) for m in memories) / max(len(memories), 1)
    has_location = any(m.location for m in memories)
    return {
        "has_milestone": has_milestone,
        "has_mundane": has_mundane,
        "avg_people": avg_people,
        "has_location": has_location,
    }




def adjust_weights(
    prefs: PreferencesFile,
    sentiment: str,
    features: dict[str, Any],
) -> PreferencesFile:
    """Adjust nostalgia weights based on sentiment and memory features.

    Positive feedback: boost matching weights by +WEIGHT_STEP
    Negative feedback: reduce matching weights by -WEIGHT_STEP
    Neutral: no change
    """
    if sentiment == "neutral":
        return prefs

    direction = WEIGHT_STEP if sentiment == "positive" else -WEIGHT_STEP
    w = prefs.nostalgia_weights

    new_milestones = w.milestones
    new_mundane = w.mundane_daily
    new_people = w.people_focus
    new_location = w.location_focus

    if features.get("has_milestone"):
        new_milestones = _clamp(w.milestones + direction)
    if features.get("has_mundane"):
        new_mundane = _clamp(w.mundane_daily + direction)
    if features.get("avg_people", 0) > 1:
        new_people = _clamp(w.people_focus + direction)
    if features.get("has_location"):
        new_location = _clamp(w.location_focus + direction)

    return PreferencesFile(
        nostalgia_weights=NostalgiaWeights(
            milestones=new_milestones,
            mundane_daily=new_mundane,
            people_focus=new_people,
            location_focus=new_location,
        ),
        tone_preference=prefs.tone_preference,
        locations=prefs.locations,
        notes=prefs.notes,
    )


async def process_feedback(
    text: str,
    digest_memory_ids: list[str],
    llm: LLMBackend,
    model: str,
    store: DataStore,
    git: GitRepo,
) -> str:
    """Full feedback processing: classify, adjust weights, commit."""
    sentiment = await classify_feedback_sentiment(text, llm, model)

    # Load memories from digest
    memories: list[Memory] = []
    for mid in digest_memory_ids:
        if len(mid) >= 8 and mid[:8].isdigit():
            from datetime import date as _date

            y, m, d = int(mid[:4]), int(mid[4:6]), int(mid[6:8])
            slug = mid[9:] if len(mid) > 9 else mid
            path = store._memory_path(_date(y, m, d), slug)
            import contextlib

            with contextlib.suppress(FileNotFoundError):
                memories.append(store.read_memory(path))

    features = extract_memory_features(memories)
    prefs = store.read_preferences()
    new_prefs = adjust_weights(prefs, sentiment, features)
    store.write_preferences(new_prefs)

    # Update churn state based on sentiment
    from datetime import date as _date_cls

    from elephant.brain.engagement import update_churn_state_after_feedback

    churn_state = store.read_churn_state()
    new_churn = update_churn_state_after_feedback(churn_state, sentiment, _date_cls.today())
    store.write_churn_state(new_churn)

    first_mid = digest_memory_ids[0] if digest_memory_ids else "no memories"
    git.auto_commit("feedback", f"{sentiment.capitalize()} — {first_mid}")
    logger.info("Feedback processed: %s (adjusted %d features)", sentiment, len(features))
    return sentiment
