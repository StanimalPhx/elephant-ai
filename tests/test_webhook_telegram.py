"""Tests for Telegram webhook: secret path validation, message parsing, /start auth."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from elephant.config import ScheduleConfig
from elephant.data.models import AuthorizedChat, AuthorizedChatsFile, DigestState
from elephant.database import DatabaseInstance
from elephant.router import ChatRouter
from elephant.webhooks.telegram import (
    bot_session_key,
    bot_token_key,
    create_telegram_webhook,
    media_dir_key,
)


def _mock_store(approved_chat_ids: list[str] | None = None):
    if approved_chat_ids is None:
        approved_chat_ids = ["456"]
    store = MagicMock()
    chats = [AuthorizedChat(chat_id=cid, status="approved") for cid in approved_chat_ids]
    store.read_authorized_chats.return_value = AuthorizedChatsFile(chats=chats)
    store.write_authorized_chats = MagicMock()
    store.read_digest_state.return_value = DigestState()
    store.write_digest_state = MagicMock()
    return store


def _make_db(
    name: str = "test",
    auth_secret: str = "my_auth_secret",
    store: MagicMock | None = None,
    approved_chat_ids: list[str] | None = None,
) -> DatabaseInstance:
    if store is None:
        store = _mock_store(approved_chat_ids)
    return DatabaseInstance(
        name=name,
        auth_secret=auth_secret,
        store=store,
        git=MagicMock(),
        messaging=MagicMock(),
        anytime=AsyncMock(),
        morning=MagicMock(),
        evening=MagicMock(),
        question_mgr=MagicMock(),
        monthly_report=MagicMock(),
        weekly_recap=MagicMock(),
        year_in_review=MagicMock(),
        integrity_check=MagicMock(),
        schedule=ScheduleConfig(),
    )


def _make_router(db: DatabaseInstance) -> ChatRouter:
    router = ChatRouter()
    router.register_database(db)
    return router


def _mock_bot_session():
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"ok": True, "result": {}})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    return mock_session


class TestTelegramWebhook(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self.webhook_secret = "test_secret_123"
        self.auth_secret = "my_auth_secret"
        self.store = _mock_store()
        self.db = _make_db(store=self.store, auth_secret=self.auth_secret)
        self.router = _make_router(self.db)

        app = web.Application()
        route = create_telegram_webhook(self.webhook_secret, self.router)
        app.router.add_route(route.method, route.path, route.handler)
        app[bot_token_key] = "123:ABC"
        app[bot_session_key] = _mock_bot_session()
        return app

    async def test_valid_message(self):
        update = {
            "message": {
                "message_id": 42,
                "chat": {"id": 456},
                "text": "Hello from Telegram",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        self.db.anytime.handle_message.assert_awaited_once()
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Hello from Telegram"
        assert msg.sender == "456"
        assert msg.message_id == "42"

    async def test_wrong_secret_returns_403(self):
        update = {"message": {"message_id": 1, "chat": {"id": 1}, "text": "Hi", "date": 0}}
        resp = await self.client.post(
            "/webhook/telegram/wrong_secret",
            json=update,
        )
        assert resp.status == 403

    async def test_message_with_reply(self):
        update = {
            "message": {
                "message_id": 43,
                "chat": {"id": 456},
                "text": "Great digest!",
                "date": 1709251200,
                "reply_to_message": {"message_id": 10},
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.reply_to_id == "10"

    async def test_empty_text_is_ignored(self):
        update = {
            "message": {
                "message_id": 44,
                "chat": {"id": 456},
                "text": "",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_not_awaited()

    async def test_invalid_json(self):
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_no_message_field(self):
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json={"update_id": 123},
        )
        assert resp.status == 200  # Should still return ok

    async def test_start_bootstrap_auto_approves(self):
        """First /start with no approved chats auto-approves."""
        store = _mock_store(approved_chat_ids=[])
        db = _make_db(store=store, auth_secret=self.auth_secret)
        router = _make_router(db)

        app = web.Application()
        route = create_telegram_webhook(self.webhook_secret, router)
        app.router.add_route(route.method, route.path, route.handler)
        app[bot_token_key] = "123:ABC"
        app[bot_session_key] = _mock_bot_session()

        async with TestClient(TestServer(app)) as client:
            update = {
                "message": {
                    "message_id": 50,
                    "chat": {"id": 789, "first_name": "Alice"},
                    "text": f"/start {self.auth_secret}",
                    "date": 1709251200,
                }
            }
            resp = await client.post(
                f"/webhook/telegram/{self.webhook_secret}",
                json=update,
            )
            assert resp.status == 200
            store.write_authorized_chats.assert_called_once()
            written_ac = store.write_authorized_chats.call_args[0][0]
            assert len(written_ac.chats) == 1
            assert written_ac.chats[0].chat_id == "789"
            assert written_ac.chats[0].status == "approved"

    async def test_start_pending_requires_approval(self):
        """Second /start with existing approved chats creates pending entry."""
        update = {
            "message": {
                "message_id": 50,
                "chat": {"id": 789, "first_name": "Bob"},
                "text": f"/start {self.auth_secret}",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_called_once()
        written_ac = self.store.write_authorized_chats.call_args[0][0]
        new_chats = [c for c in written_ac.chats if c.chat_id == "789"]
        assert len(new_chats) == 1
        assert new_chats[0].status == "pending"
        self.db.anytime.handle_message.assert_not_awaited()

    async def test_start_with_invalid_secret_rejected(self):
        update = {
            "message": {
                "message_id": 51,
                "chat": {"id": 789},
                "text": "/start wrong_secret",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_not_called()
        self.db.anytime.handle_message.assert_not_awaited()

    async def test_start_already_registered(self):
        """A chat that's already registered gets 'Already registered!'."""
        update = {
            "message": {
                "message_id": 50,
                "chat": {"id": 456},
                "text": f"/start {self.auth_secret}",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_not_called()

    async def test_unauthorized_sender_ignored(self):
        """Messages from non-authorized chats should be silently ignored."""
        update = {
            "message": {
                "message_id": 52,
                "chat": {"id": 999},
                "text": "Hello!",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_not_awaited()

    async def test_error_in_handler_sends_error_reply(self):
        """When anytime.handle_message raises, the user should get an error reply."""
        store = _mock_store()
        db = _make_db(store=store, auth_secret=self.auth_secret)
        db.anytime.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        router = _make_router(db)

        app = web.Application()
        route = create_telegram_webhook(self.webhook_secret, router)
        app.router.add_route(route.method, route.path, route.handler)
        app[bot_token_key] = "123:ABC"
        mock_session = _mock_bot_session()
        app[bot_session_key] = mock_session

        async with TestClient(TestServer(app)) as client:
            update = {
                "message": {
                    "message_id": 60,
                    "chat": {"id": 456},
                    "text": "trigger error",
                    "date": 1709251200,
                }
            }
            resp = await client.post(
                f"/webhook/telegram/{self.webhook_secret}",
                json=update,
            )
            assert resp.status == 200
            mock_session.post.assert_called()
            call_args = mock_session.post.call_args
            assert "sendMessage" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["text"] == "Sorry, something went wrong. Please try again."

    async def test_no_authorized_chat_ignores_messages(self):
        """When no chat is authorized, all non-start messages are ignored."""
        self.store.read_authorized_chats.return_value = AuthorizedChatsFile(chats=[])
        update = {
            "message": {
                "message_id": 53,
                "chat": {"id": 456},
                "text": "Hello!",
                "date": 1709251200,
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_not_awaited()


class TestTelegramWebhookCallback(AioHTTPTestCase):
    """Tests for callback query handling (approve/reject flow)."""

    async def get_application(self) -> web.Application:
        self.webhook_secret = "test_secret_123"
        self.auth_secret = "my_auth_secret"
        self.store = MagicMock()
        self.store.read_authorized_chats.return_value = AuthorizedChatsFile(
            chats=[
                AuthorizedChat(chat_id="456", status="approved"),
                AuthorizedChat(chat_id="789", status="pending"),
            ]
        )
        self.store.write_authorized_chats = MagicMock()
        self.db = _make_db(store=self.store, auth_secret=self.auth_secret)
        self.router = _make_router(self.db)

        app = web.Application()
        route = create_telegram_webhook(self.webhook_secret, self.router)
        app.router.add_route(route.method, route.path, route.handler)
        app[bot_token_key] = "123:ABC"
        app[bot_session_key] = _mock_bot_session()
        return app

    async def test_approve_callback(self):
        update = {
            "callback_query": {
                "id": "cq_1",
                "data": json.dumps({"action": "approve", "chat_id": "789"}),
                "message": {
                    "message_id": 100,
                    "chat": {"id": 456},
                },
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_called_once()
        written = self.store.write_authorized_chats.call_args[0][0]
        approved = [c for c in written.chats if c.chat_id == "789"]
        assert approved[0].status == "approved"

    async def test_reject_callback(self):
        update = {
            "callback_query": {
                "id": "cq_2",
                "data": json.dumps({"action": "reject", "chat_id": "789"}),
                "message": {
                    "message_id": 101,
                    "chat": {"id": 456},
                },
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_called_once()
        written = self.store.write_authorized_chats.call_args[0][0]
        # 789 should be removed
        assert all(c.chat_id != "789" for c in written.chats)

    async def test_callback_from_unauthorized_user(self):
        """Callback from non-approved user should be rejected."""
        update = {
            "callback_query": {
                "id": "cq_3",
                "data": json.dumps({"action": "approve", "chat_id": "789"}),
                "message": {
                    "message_id": 102,
                    "chat": {"id": 999},  # not approved
                },
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}",
            json=update,
        )
        assert resp.status == 200
        self.store.write_authorized_chats.assert_not_called()


class TestTelegramWebhookMedia(AioHTTPTestCase):
    """Tests for media attachment handling in webhook."""

    async def get_application(self) -> web.Application:
        self.webhook_secret = "test_secret_123"
        self.auth_secret = "my_auth_secret"
        self.store = _mock_store()
        self.store.media_dir.return_value = "/tmp/test_media"
        self.db = _make_db(store=self.store, auth_secret=self.auth_secret)
        self.router = _make_router(self.db)

        app = web.Application()
        route = create_telegram_webhook(self.webhook_secret, self.router)
        app.router.add_route(route.method, route.path, route.handler)
        app[bot_token_key] = "123:ABC"
        app[media_dir_key] = "/tmp/test_media"
        app[bot_session_key] = _mock_bot_session()
        return app

    @patch("elephant.webhooks.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_photo_with_caption(self, mock_download):
        mock_download.return_value = "/tmp/test_media/file_large.jpg"
        update = {
            "message": {
                "message_id": 70,
                "chat": {"id": 456},
                "caption": "Look at this sunset!",
                "date": 1709251200,
                "photo": [
                    {"file_id": "small", "width": 90, "height": 90},
                    {"file_id": "large", "width": 800, "height": 600},
                ],
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}", json=update
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_awaited_once()
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Look at this sunset!"
        assert len(msg.attachments) == 1
        assert msg.attachments[0].media_type == "photo"
        assert msg.attachments[0].file_path == "/tmp/test_media/file_large.jpg"
        mock_download.assert_awaited_once()

    @patch("elephant.webhooks.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_document_with_caption(self, mock_download):
        mock_download.return_value = "/tmp/test_media/doc123.pdf"
        update = {
            "message": {
                "message_id": 71,
                "chat": {"id": 456},
                "caption": "Here's the report",
                "date": 1709251200,
                "document": {"file_id": "doc123", "file_name": "report.pdf"},
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}", json=update
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_awaited_once()
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Here's the report"
        assert len(msg.attachments) == 1
        assert msg.attachments[0].media_type == "document"

    @patch("elephant.webhooks.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_photo_without_text_dispatches(self, mock_download):
        """A photo with no text/caption should still be dispatched (not ignored)."""
        mock_download.return_value = "/tmp/test_media/photo.jpg"
        update = {
            "message": {
                "message_id": 72,
                "chat": {"id": 456},
                "date": 1709251200,
                "photo": [
                    {"file_id": "ph1", "width": 100, "height": 100},
                ],
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}", json=update
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_awaited_once()
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.text == ""
        assert len(msg.attachments) == 1
        assert msg.attachments[0].media_type == "photo"

    @patch("elephant.webhooks.telegram.download_telegram_file", new_callable=AsyncMock)
    async def test_download_failure_still_dispatches(self, mock_download):
        """If download fails, the message should still be dispatched with the caption."""
        mock_download.side_effect = RuntimeError("download failed")
        update = {
            "message": {
                "message_id": 73,
                "chat": {"id": 456},
                "caption": "Photo here",
                "date": 1709251200,
                "photo": [{"file_id": "fail", "width": 100, "height": 100}],
            }
        }
        resp = await self.client.post(
            f"/webhook/telegram/{self.webhook_secret}", json=update
        )
        assert resp.status == 200
        self.db.anytime.handle_message.assert_awaited_once()
        msg = self.db.anytime.handle_message.call_args[0][0]
        assert msg.text == "Photo here"
        assert len(msg.attachments) == 0  # download failed, no attachment
