"""Pydantic models for all YAML document types."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# --- Shared sub-models ---


class MediaLinks(BaseModel):
    photos: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)


class GPSCoords(BaseModel):
    lat: float
    lon: float


class CameraInfo(BaseModel):
    make: str = ""
    model: str = ""


# --- Memory sub-models ---


class MemoryMetadata(BaseModel):
    who: list[str] = Field(default_factory=list)
    what: str | None = None
    where: str | None = None
    when: str | None = None
    why: str | None = None
    how: str | None = None


class InteractionRecord(BaseModel):
    initial_log: str | None = None
    follow_up_q: str | None = None
    user_answer: str | None = None
    sentiment: str | None = None


# --- Correction sub-model ---


class Correction(BaseModel):
    timestamp: datetime
    field: str
    old_value: str | None
    new_value: str | None
    reason: str | None = None


# --- Memory (formerly Event) ---


class Memory(BaseModel):
    id: str
    date: date
    time: str | None = None
    title: str
    type: str  # milestone, daily, outing, celebration, health, travel, mundane, other
    description: str
    people: list[str]
    location: str | None = None
    media: MediaLinks | None = None
    source: str  # WhatsApp, Telegram, evening_checkin, morning_digest, manual, photo_ingest
    source_message_ids: list[str] = Field(default_factory=list)
    nostalgia_score: float = 1.0
    tags: list[str] = Field(default_factory=list)
    content: str | None = None
    participants: list[str] = Field(default_factory=list)
    metadata: MemoryMetadata | None = None
    interaction: InteractionRecord | None = None
    media_refs: list[str] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)

    def resolved_value(self, field_name: str) -> Any:
        """Return the latest corrected value for a field, or the original."""
        for correction in reversed(self.corrections):
            if correction.field == field_name:
                return correction.new_value
        return getattr(self, field_name, None)



# --- Photo Index ---


class PhotoEntry(BaseModel):
    photo_id: str
    sha256: str
    taken_at: datetime
    source: str  # google_photos, apple_photos, local_folder, manual
    gps: GPSCoords | None = None
    place: str | None = None
    people_detected: list[str] = Field(default_factory=list)
    camera: CameraInfo | None = None
    memory_id: str | None = None


# --- Video Index ---


class VideoEntry(BaseModel):
    video_id: str
    sha256: str
    taken_at: datetime
    duration_seconds: float | None = None
    source: str  # google_photos, apple_photos, local_folder, manual
    gps: GPSCoords | None = None
    place: str | None = None
    people_detected: list[str] = Field(default_factory=list)
    camera: CameraInfo | None = None
    thumbnail: str | None = None
    memory_id: str | None = None


# --- Groups ---


class Group(BaseModel):
    group_id: str
    display_name: str
    color: str | None = None


# --- People ---


class PersonRelationship(BaseModel):
    person_id: str
    label: str  # e.g. "brother", "spouse", "parent"


class LifeEvent(BaseModel):
    date: date
    description: str  # e.g. "got engaged", "wedding", "moved to Austin"


class CurrentThread(BaseModel):
    topic: str
    latest_update: str
    last_mentioned_date: date


class PersonConnection(BaseModel):
    person_id: str
    type: str  # "sibling", "spouse", "coworker"
    note: str | None = None


class PersonPreferences(BaseModel):
    remind_birthday_weeks_ahead: int = 2
    tone_preference: str | None = None


class Person(BaseModel):
    person_id: str
    display_name: str
    relationship: list[str] = Field(default_factory=lambda: ["unknown"])

    @field_validator("relationship", mode="before")
    @classmethod
    def _coerce_relationship(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v  # type: ignore[return-value]
    other_names: list[str] = Field(default_factory=list)
    birthday: date | None = None
    groups: list[str] = Field(default_factory=list)
    relationships: list[PersonRelationship] = Field(default_factory=list)
    life_events: list[LifeEvent] = Field(default_factory=list)
    face_clusters: list[str] = Field(default_factory=list)
    notes: str | None = None
    current_threads: list[CurrentThread] = Field(default_factory=list)
    archived_threads: list[CurrentThread] = Field(default_factory=list)
    interaction_frequency_target: int | None = None
    preferences: PersonPreferences | None = None
    attributes: dict[str, str] = Field(default_factory=dict)


# --- Preferences ---


class NostalgiaWeights(BaseModel):
    milestones: float = 1.0
    mundane_daily: float = 1.0
    people_focus: float = 1.0
    location_focus: float = 1.0


class TonePreference(BaseModel):
    style: str = "heartfelt"  # heartfelt, playful, factual
    length: str = "short"  # short, medium, long


class PreferencesFile(BaseModel):
    nostalgia_weights: NostalgiaWeights = NostalgiaWeights()
    tone_preference: TonePreference = TonePreference()
    locations: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# --- Pending Questions ---


class PendingQuestion(BaseModel):
    id: str
    type: str  # person_identification, memory_enrichment, context_gap, media_linking
    subject: str
    question: str | None = None
    status: str = "pending"  # pending, asked, answered, dismissed
    created_at: datetime
    answered_at: datetime | None = None
    answer: str | None = None
    message_id: str | None = None


class PendingQuestionsFile(BaseModel):
    questions: list[PendingQuestion] = Field(default_factory=list)


# --- Metrics ---


class DailyMetrics(BaseModel):
    date: date
    memories_created: int = 0
    digests_sent: int = 0
    digest_replies: int = 0
    questions_asked: int = 0
    questions_answered: int = 0
    checkins_sent: int = 0
    weekly_recaps_sent: int = 0
    nudges_sent: int = 0


class MetricsFile(BaseModel):
    days: list[DailyMetrics] = Field(default_factory=list)


# --- Digest State ---


class DigestState(BaseModel):
    last_digest_sent_at: datetime | None = None
    last_digest_memory_ids: list[str] = Field(default_factory=list)
    last_digest_message_id: str | None = None
    last_digest_text: str | None = None


# --- Nudge State ---


class NudgeRecord(BaseModel):
    person_id: str
    last_nudged_at: date
    context: str | None = None


class NudgeStateFile(BaseModel):
    records: list[NudgeRecord] = Field(default_factory=list)


# --- Authorized Chats ---


class AuthorizedChat(BaseModel):
    chat_id: str
    status: str = "approved"  # "approved" | "pending"
    added_at: datetime | None = None
    display_name: str | None = None


class AuthorizedChatsFile(BaseModel):
    chats: list[AuthorizedChat] = Field(default_factory=list)


# --- Chat History ---


class ChatHistoryEntry(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ChatHistoryFile(BaseModel):
    entries: list[ChatHistoryEntry] = Field(default_factory=list)


# --- Raw Messages ---


class RawMessageAttachment(BaseModel):
    file_path: str
    media_type: str  # "photo", "video", "document"


class RawMessage(BaseModel):
    text: str
    sender: str
    message_id: str
    timestamp: datetime
    reply_to_id: str | None = None
    attachments: list[RawMessageAttachment] = Field(default_factory=list)


