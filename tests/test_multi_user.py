"""Tests for multi-user contribution support."""

from __future__ import annotations

from datetime import date

from elephant.data.models import AuthorizedChat, AuthorizedChatsFile, Memory
from elephant.data.store import DataStore


class TestSourceUserField:
    def test_memory_source_user_roundtrip(self, data_dir: str) -> None:
        """source_user field is persisted and read back."""
        store = DataStore(data_dir)
        store.initialize()

        memory = Memory(
            id="20260301_test",
            date=date(2026, 3, 1),
            title="Test event",
            type="daily",
            description="Something happened",
            people=["Dad"],
            source="Telegram",
            source_user="Alice",
        )
        store.write_memory(memory)

        read_back = store.find_memory_by_id("20260301_test")
        assert read_back is not None
        assert read_back.source_user == "Alice"

    def test_memory_source_user_defaults_to_none(self, data_dir: str) -> None:
        """source_user defaults to None when not set."""
        store = DataStore(data_dir)
        store.initialize()

        memory = Memory(
            id="20260301_test2",
            date=date(2026, 3, 1),
            title="Test event 2",
            type="daily",
            description="Something happened",
            people=["Dad"],
            source="Telegram",
        )
        store.write_memory(memory)

        read_back = store.find_memory_by_id("20260301_test2")
        assert read_back is not None
        assert read_back.source_user is None


class TestResolveSenderDisplayName:
    def test_resolves_known_sender(self, data_dir: str) -> None:
        """AnytimeLogFlow can resolve a sender to a display name."""
        from unittest.mock import AsyncMock, MagicMock

        from elephant.flows.anytime_log import AnytimeLogFlow

        store = DataStore(data_dir)
        store.initialize()

        store.write_authorized_chats(AuthorizedChatsFile(chats=[
            AuthorizedChat(chat_id="123", status="approved", display_name="Alice"),
            AuthorizedChat(chat_id="456", status="approved", display_name="Bob"),
        ]))

        flow = AnytimeLogFlow(
            store=store,
            llm=AsyncMock(),
            parsing_model="test",
            messaging=AsyncMock(),
            git=MagicMock(),
        )

        assert flow._resolve_sender_display_name("123") == "Alice"
        assert flow._resolve_sender_display_name("456") == "Bob"
        assert flow._resolve_sender_display_name("999") is None
