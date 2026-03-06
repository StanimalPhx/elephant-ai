"""Process context updates by routing to Person files and preferences.

This module replaces the old context.yaml-based enrichment.
Updates are now routed to individual Person files and preferences.yaml.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from elephant.data.models import Person, PreferencesFile
    from elephant.data.store import DataStore
    from elephant.git_ops import GitRepo
    from elephant.llm.backend import LLMBackend

logger = logging.getLogger(__name__)

# Accept both singular and plural key variants from LLM responses.
_LOCATION_KEYS = {"location", "locations"}
_NOTE_KEYS = {"note", "notes"}
_PERSON_UPDATE_KEYS = {"person_update", "person_updates"}


def _as_list(value: Any) -> list[Any]:
    """Normalise a value that may be a single item or a list into a list."""
    if isinstance(value, list):
        return value
    return [value]


def _get_values(updates: dict[str, Any], keys: set[str]) -> list[Any]:
    """Collect values from *updates* for any matching key, returned as a flat list."""
    items: list[Any] = []
    for key in keys:
        if key in updates:
            items.extend(_as_list(updates[key]))
    return items


async def process_context_update(
    text: str,
    llm: LLMBackend,
    model: str,
    store: DataStore,
    git: GitRepo,
) -> bool:
    """Process a context update message from the user.

    Routes updates to:
    - Person files (for people/family/friend updates)
    - preferences.yaml locations and notes (for location/note updates)
    """
    people = store.read_all_people()
    prefs = store.read_preferences()

    # Use LLM to extract structured updates from the text
    context_str = _build_update_context(people, prefs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a family context assistant. Extract structured updates from "
                "the user's message. Respond ONLY with valid YAML (no markdown fencing).\n\n"
                "Possible update types:\n"
                "  location: {name: str, description: str}\n"
                "  note: string\n"
                "  person_update: {name: str, field: str, value: any}\n\n"
                f"Current context:\n{context_str}"
            ),
        },
        {"role": "user", "content": text},
    ]
    response = await llm.chat(messages, model=model, temperature=0.3)

    try:
        updates = yaml.safe_load(response.content or "")
    except yaml.YAMLError:
        logger.warning("Failed to parse context update YAML from LLM")
        return False

    if not isinstance(updates, dict):
        logger.warning("LLM returned non-dict context update: %s", type(updates).__name__)
        return False

    changed = False

    # Handle location updates → preferences.yaml
    for loc in _get_values(updates, _LOCATION_KEYS):
        if isinstance(loc, dict) and "name" in loc:
            prefs.locations[loc["name"]] = loc.get("description", "")
            changed = True

    # Handle note updates → preferences.yaml
    for note in _get_values(updates, _NOTE_KEYS):
        if note:
            prefs.notes.append(str(note))
            changed = True

    if changed:
        store.write_preferences(prefs)

    # Handle person updates → individual Person files
    for pu in _get_values(updates, _PERSON_UPDATE_KEYS):
        if isinstance(pu, dict) and "name" in pu:
            changed |= _update_person(store, pu)

    if changed:
        git.auto_commit("context", "Context update from user")
        logger.info("Context updated from user message")
    else:
        logger.warning("No recognised updates in LLM response: %s", list(updates.keys()))

    return changed


def _build_update_context(people: list[Person], prefs: PreferencesFile) -> str:
    """Build context string for the update extraction prompt."""
    parts: list[str] = []
    if people:
        names = [f"{p.display_name} ({', '.join(p.relationship)})" for p in people]
        parts.append(f"People: {', '.join(names)}")
    if prefs.locations:
        locs = [f"{k}: {v}" for k, v in prefs.locations.items()]
        parts.append(f"Locations: {', '.join(locs)}")
    if prefs.notes:
        parts.append(f"Notes: {'; '.join(prefs.notes)}")
    return "\n".join(parts) if parts else "(no context available)"


def _update_person(store: DataStore, update: dict[str, Any]) -> bool:
    """Update a person in the people directory."""
    name = update["name"]
    field = update.get("field")
    value = update.get("value")
    if not field or value is None:
        return False

    for person in store.read_all_people():
        if person.display_name == name and hasattr(person, field):
            updated = person.model_copy(update={field: value})
            store.write_person(updated)
            return True
    return False
