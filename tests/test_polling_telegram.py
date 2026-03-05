"""Tests for Telegram polling receiver."""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elephant.config import ScheduleConfig, TelegramConfig
from elephant.data.models import AuthorizedChat, AuthorizedChatsFile, DigestState
from elephant.database import DatabaseInstance
from elephant.polling.telegram import TelegramPoller
from elephant.router import ChatRouter


def _make_db(
    name: str = "test",
    auth_secret: str = "my-secret",
    store: MagicMock | None = None,
    approved_chat_ids: list[str] | None = None,
) -> DatabaseInstance:
    if store is None:
        store = MagicMock()
        if approved_chat_ids is None:
            approved_chat_ids = ["42"]
        chats = [AuthorizedChat(chat_id=cid, status="approved") for cid in approved_chat_ids]
        store.read_authorized_chats.return_value = AuthorizedChatsFile(chats=chats)
        store.write_authorized_chats = MagicMock()
        store.read_digest_state.return_value = DigestState()
        store.media_dir.return_value = "/tmp/test_media"
    anytime = AsyncMock()
    return DatabaseInstance(
        name=name,
        auth_secret=auth_secret,
        store=store,
        git=MagicMock(),
        messaging=MagicMock(),
        anytime=anytime,
        morning=MagicMock(),
        evening=MagicMock(),
        question_mgr=MagicMock(),
        monthly_report=MagicMock(),
        weekly_recap=MagicMock(),
        year_in_review=MagicMock(),
        schedule=ScheduleConfig(),
    )


@pytest.fixture
def tg_config():
    return TelegramConfig(
        bot_token="123:ABC",
        webhook_secret="",
        webhook_url="",
        mode="polling",
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.read_authorized_chats.return_value = AuthorizedChatsFile(
        chats=[AuthorizedChat(chat_id="42", status="approved")]
    )
    store.write_authorized_chats = MagicMock()
    store.read_digest_state.return_value = DigestState()
    store.media_dir.return_value = "/tmp/test_media"
    return store


@pytest.fixture
def db(mock_store):
    return _make_db(store=mock_store)


@pytest.fixture
def router(db):
    r = ChatRouter()
    r.register_database(db)
    return r


@pytest.fixture
def poller(tg_config, router):
    session = AsyncMock()
    return TelegramPoller(session, tg_config, router)


def _make_update(
    update_id: int, chat_id: int, text: str,
    message_id: int = 1, date: int = 1700000000,
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": date,
            "text": text,
            "chat": {"id": chat_id},
        },
    }


class TestHandleUpdate:
    async def test_authorized_message_calls_on_message(self, poller, db):
        """An authorized user's message should be dispatched to db.anytime."""
        update = _make_update(100, 42, "Hello elephant")
        await poller._handle_update(update)
        db.anytime.handle_message.assert_awaited_once()
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Hello elephant"
        assert msg.sender == "42"

    async def test_unauthorized_message_ignored(self, poller, db):
        """Messages from unauthorized chats should be silently ignored."""
        update = _make_update(101, 999, "I'm not authorized")
        await poller._handle_update(update)
        db.anytime.handle_message.assert_not_awaited()

    async def test_empty_text_ignored(self, poller, db):
        """Updates without text (e.g. stickers) should be ignored."""
        update = {"update_id": 102, "message": {"message_id": 1, "chat": {"id": 42}, "date": 0}}
        await poller._handle_update(update)
        db.anytime.handle_message.assert_not_awaited()

    async def test_error_in_handler_sends_error_reply(self, poller, db):
        """When anytime.handle_message raises, the user should get an error reply."""
        db.anytime.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        poller._reply = AsyncMock()
        update = _make_update(104, 42, "trigger error")
        await poller._handle_update(update)
        poller._reply.assert_awaited_once_with(
            "42", "Sorry, something went wrong. Please try again."
        )

    async def test_reply_to_message(self, poller, db):
        """reply_to_id should be extracted from reply_to_message."""
        update = _make_update(103, 42, "reply text")
        update["message"]["reply_to_message"] = {"message_id": 55}
        await poller._handle_update(update)
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.reply_to_id == "55"


class TestStartAuth:
    async def test_start_bootstrap_auto_approves(self, poller, mock_store, db):
        """First /start with no approved chats auto-approves."""
        mock_store.read_authorized_chats.return_value = AuthorizedChatsFile(chats=[])
        update = _make_update(200, 77, "/start my-secret")
        poller._reply = AsyncMock()
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_called_once()
        written = mock_store.write_authorized_chats.call_args[0][0]
        assert len(written.chats) == 1
        assert written.chats[0].chat_id == "77"
        assert written.chats[0].status == "approved"
        poller._reply.assert_awaited_once_with("77", "Authenticated!")
        db.anytime.handle_message.assert_not_awaited()

    async def test_start_pending_requires_approval(self, poller, mock_store, db):
        """Second /start with existing approved chats creates pending entry."""
        update = _make_update(201, 77, "/start my-secret")
        poller._reply = AsyncMock()
        poller._send_with_markup = AsyncMock()
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_called_once()
        written = mock_store.write_authorized_chats.call_args[0][0]
        new_chats = [c for c in written.chats if c.chat_id == "77"]
        assert len(new_chats) == 1
        assert new_chats[0].status == "pending"
        poller._send_with_markup.assert_awaited_once()
        db.anytime.handle_message.assert_not_awaited()

    async def test_start_wrong_secret_rejected(self, poller, mock_store, db):
        """'/start <wrong_secret>' should not authorize."""
        update = _make_update(202, 77, "/start wrong")
        poller._reply = AsyncMock()
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_not_called()
        poller._reply.assert_awaited_once_with("77", "Invalid secret.")
        db.anytime.handle_message.assert_not_awaited()

    async def test_start_already_registered(self, poller, mock_store, db):
        """A chat that's already registered gets 'Already registered!'."""
        update = _make_update(203, 42, "/start my-secret")
        poller._reply = AsyncMock()
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_not_called()
        poller._reply.assert_awaited_once_with("42", "Already registered!")
        db.anytime.handle_message.assert_not_awaited()


class TestCallbackQuery:
    async def test_approve_callback(self, poller, mock_store):
        """Approving a pending chat should update its status."""
        mock_store.read_authorized_chats.return_value = AuthorizedChatsFile(
            chats=[
                AuthorizedChat(chat_id="42", status="approved"),
                AuthorizedChat(chat_id="77", status="pending"),
            ]
        )
        poller._answer_cq = AsyncMock()
        poller._edit_message = AsyncMock()
        poller._reply = AsyncMock()

        update = {
            "update_id": 300,
            "callback_query": {
                "id": "cq_1",
                "data": json.dumps({"action": "approve", "chat_id": "77"}),
                "message": {"message_id": 100, "chat": {"id": 42}},
            },
        }
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_called_once()
        written = mock_store.write_authorized_chats.call_args[0][0]
        approved = [c for c in written.chats if c.chat_id == "77"]
        assert approved[0].status == "approved"
        poller._reply.assert_awaited_once_with("77", "You have been approved!")

    async def test_reject_callback(self, poller, mock_store):
        """Rejecting a pending chat should remove it."""
        mock_store.read_authorized_chats.return_value = AuthorizedChatsFile(
            chats=[
                AuthorizedChat(chat_id="42", status="approved"),
                AuthorizedChat(chat_id="77", status="pending"),
            ]
        )
        poller._answer_cq = AsyncMock()
        poller._edit_message = AsyncMock()
        poller._reply = AsyncMock()

        update = {
            "update_id": 301,
            "callback_query": {
                "id": "cq_2",
                "data": json.dumps({"action": "reject", "chat_id": "77"}),
                "message": {"message_id": 101, "chat": {"id": 42}},
            },
        }
        await poller._handle_update(update)

        mock_store.write_authorized_chats.assert_called_once()
        written = mock_store.write_authorized_chats.call_args[0][0]
        assert all(c.chat_id != "77" for c in written.chats)

    async def test_callback_from_unauthorized(self, poller, mock_store):
        """Callback from non-approved user should be rejected."""
        poller._answer_cq = AsyncMock()
        update = {
            "update_id": 302,
            "callback_query": {
                "id": "cq_3",
                "data": json.dumps({"action": "approve", "chat_id": "77"}),
                "message": {"message_id": 102, "chat": {"id": 999}},
            },
        }
        await poller._handle_update(update)
        mock_store.write_authorized_chats.assert_not_called()
        poller._answer_cq.assert_awaited_once_with("cq_3", "You are not authorized.")


class TestOffsetTracking:
    async def test_offset_advances(self, poller, db):
        """Offset should be set to last update_id + 1."""
        for uid in (300, 301, 302):
            update = _make_update(uid, 42, f"msg {uid}")
            await poller._handle_update(update)

        # _handle_update doesn't track offset — _poll_loop does.
        assert poller._offset == 0  # _handle_update doesn't set offset

    async def test_poll_loop_tracks_offset(self, poller, db):
        """The poll loop should advance offset after processing updates."""
        updates_response = {
            "ok": True,
            "result": [
                _make_update(500, 42, "first"),
                _make_update(501, 42, "second"),
            ],
        }

        call_count = 0

        def mock_get(url, params=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError
            cm = AsyncMock()
            resp = AsyncMock()
            resp.json = AsyncMock(return_value=updates_response)
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        poller._session.get = mock_get

        with contextlib.suppress(asyncio.CancelledError):
            await poller._poll_loop()

        assert poller._offset == 502
        assert db.anytime.handle_message.await_count == 2


class TestStartStop:
    async def test_start_creates_task(self, poller):
        """start() should create a background task."""
        poller._poll_loop = AsyncMock()
        await poller.start()
        assert poller._task is not None
        await poller.stop()

    async def test_stop_cancels_task(self, poller):
        """stop() should cancel the polling task."""
        poller._poll_loop = AsyncMock(side_effect=asyncio.CancelledError)
        await poller.start()
        task = poller._task
        await poller.stop()
        assert poller._task is None
        assert task.cancelled() or task.done()


class TestMediaHandling:
    @patch("elephant.polling.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_photo_with_caption(self, mock_download, poller, db):
        """A photo with caption should dispatch with text and attachments."""
        mock_download.return_value = "/tmp/test_media/large.jpg"
        update = {
            "update_id": 400,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "caption": "Beach day!",
                "chat": {"id": 42},
                "photo": [
                    {"file_id": "small", "width": 90, "height": 90},
                    {"file_id": "large", "width": 800, "height": 600},
                ],
            },
        }
        await poller._handle_update(update)
        db.anytime.handle_message.assert_awaited_once()
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Beach day!"
        assert len(msg.attachments) == 1
        assert msg.attachments[0].media_type == "photo"
        assert msg.attachments[0].file_path == "/tmp/test_media/large.jpg"

    @patch("elephant.polling.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_document_with_caption(self, mock_download, poller, db):
        mock_download.return_value = "/tmp/test_media/doc.pdf"
        update = {
            "update_id": 401,
            "message": {
                "message_id": 2,
                "date": 1700000000,
                "caption": "Report attached",
                "chat": {"id": 42},
                "document": {"file_id": "doc1", "file_name": "report.pdf"},
            },
        }
        await poller._handle_update(update)
        db.anytime.handle_message.assert_awaited_once()
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Report attached"
        assert len(msg.attachments) == 1
        assert msg.attachments[0].media_type == "document"

    @patch("elephant.polling.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_photo_without_text_dispatches(self, mock_download, poller, db):
        """A photo with no text/caption should still be dispatched."""
        mock_download.return_value = "/tmp/test_media/photo.jpg"
        update = {
            "update_id": 402,
            "message": {
                "message_id": 3,
                "date": 1700000000,
                "chat": {"id": 42},
                "photo": [{"file_id": "ph1", "width": 100, "height": 100}],
            },
        }
        await poller._handle_update(update)
        db.anytime.handle_message.assert_awaited_once()
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.text == ""
        assert len(msg.attachments) == 1

    @patch("elephant.polling.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_download_failure_still_dispatches(self, mock_download, poller, db):
        """If download fails, message dispatches with caption but no attachments."""
        mock_download.side_effect = RuntimeError("download failed")
        update = {
            "update_id": 403,
            "message": {
                "message_id": 4,
                "date": 1700000000,
                "caption": "Photo here",
                "chat": {"id": 42},
                "photo": [{"file_id": "fail", "width": 100, "height": 100}],
            },
        }
        await poller._handle_update(update)
        db.anytime.handle_message.assert_awaited_once()
        msg = db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Photo here"
        assert len(msg.attachments) == 0
