"""Tests for ChatRouter."""

from unittest.mock import MagicMock

from elephant.config import ScheduleConfig
from elephant.data.models import AuthorizedChat, AuthorizedChatsFile
from elephant.database import DatabaseInstance
from elephant.router import ChatRouter


def _make_db(
    name: str = "test",
    auth_secret: str = "secret",
    approved_chat_ids: list[str] | None = None,
) -> DatabaseInstance:
    """Create a DatabaseInstance with a mock store."""
    if approved_chat_ids is None:
        approved_chat_ids = []
    store = MagicMock()
    chats = [AuthorizedChat(chat_id=cid, status="approved") for cid in approved_chat_ids]
    store.read_authorized_chats.return_value = AuthorizedChatsFile(chats=chats)
    return DatabaseInstance(
        name=name,
        auth_secret=auth_secret,
        store=store,
        git=MagicMock(),
        messaging=MagicMock(),
        anytime=MagicMock(),
        morning=MagicMock(),
        evening=MagicMock(),
        question_mgr=MagicMock(),
        monthly_report=MagicMock(),
        weekly_recap=MagicMock(),
        year_in_review=MagicMock(),
        integrity_check=MagicMock(),
        schedule=ScheduleConfig(),
    )


class TestChatRouter:
    def test_register_and_resolve_by_chat(self):
        router = ChatRouter()
        db = _make_db(name="brogli", approved_chat_ids=["100", "200"])
        router.register_database(db)

        assert router.resolve_by_chat("100") is db
        assert router.resolve_by_chat("200") is db
        assert router.resolve_by_chat("999") is None

    def test_resolve_by_auth_secret(self):
        router = ChatRouter()
        db = _make_db(name="brogli", auth_secret="abc-123")
        router.register_database(db)

        assert router.resolve_by_auth_secret("abc-123") is db
        assert router.resolve_by_auth_secret("wrong") is None

    def test_assign_chat(self):
        router = ChatRouter()
        db = _make_db(name="brogli")
        router.register_database(db)

        assert router.resolve_by_chat("new_chat") is None
        router.assign_chat("new_chat", db)
        assert router.resolve_by_chat("new_chat") is db

    def test_get_all_databases(self):
        router = ChatRouter()
        db1 = _make_db(name="a", auth_secret="s1")
        db2 = _make_db(name="b", auth_secret="s2")
        router.register_database(db1)
        router.register_database(db2)

        all_dbs = router.get_all_databases()
        assert len(all_dbs) == 2
        assert db1 in all_dbs
        assert db2 in all_dbs

    def test_multi_db_isolation(self):
        """Each database's chats resolve to the correct database."""
        router = ChatRouter()
        db1 = _make_db(name="brogli", auth_secret="s1", approved_chat_ids=["100"])
        db2 = _make_db(name="smith", auth_secret="s2", approved_chat_ids=["200"])
        router.register_database(db1)
        router.register_database(db2)

        assert router.resolve_by_chat("100") is db1
        assert router.resolve_by_chat("200") is db2
        assert router.resolve_by_auth_secret("s1") is db1
        assert router.resolve_by_auth_secret("s2") is db2

    def test_empty_router(self):
        router = ChatRouter()
        assert router.resolve_by_chat("any") is None
        assert router.resolve_by_auth_secret("any") is None
        assert router.get_all_databases() == []
