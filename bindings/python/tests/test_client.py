"""Unit tests for SmritiClient using mocked HTTP responses."""

from unittest.mock import MagicMock, patch

import pytest

from smriti.client import SmritiClient, SmritiError


@pytest.fixture
def client():
    return SmritiClient(base_url="http://localhost:4000")


class TestRemember:
    def test_remember_returns_uuid(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc-123-def"}

        with patch.object(client._session, "request", return_value=mock_resp):
            result = client.remember("test memory", tags=["test"])
            assert result == "abc-123-def"

    def test_remember_with_all_options(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "xyz"}

        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.remember(
                "auth uses JWT",
                tags=["auth"],
                kind="decision",
                importance=0.9,
            )
            call_args = mock_req.call_args
            body = call_args[1]["json"]
            assert body["text"] == "auth uses JWT"
            assert body["kind"] == "decision"
            assert body["importance"] == 0.9
            assert body["tags"] == ["auth"]

    def test_remember_normalizes_scope_payload(self):
        client = SmritiClient(
            base_url="http://localhost:4000",
            agent="edge",
            user="session-123",
            session="turn-1",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "xyz"}

        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.remember("auth uses JWT")
            body = mock_req.call_args[1]["json"]
            assert body["scope"] == {
                "agent_id": "edge",
                "user_id": "session-123",
                "session_id": "turn-1",
            }


class TestRecall:
    def test_recall_returns_result(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "hits": [
                {
                    "node": {
                        "id": "abc",
                        "text": "JWT RS256",
                        "tags": ["auth"],
                        "kind": "fact",
                        "importance": 0.5,
                        "access_count": 0,
                        "token_count": 3,
                    },
                    "final_score": 0.85,
                    "fingerprint_sim": 0.7,
                    "ppr_score": 0.15,
                    "from_hippocampus": True,
                }
            ],
            "tokens_used": 3,
            "tokens_budget": 2000,
            "candidates_considered": 10,
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            result = client.recall("authentication", budget=2000)
            assert len(result.hits) == 1
            assert result.hits[0].node.text == "JWT RS256"
            assert result.hits[0].from_hippocampus is True
            assert result.tokens_used == 3

    def test_recall_empty(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "hits": [],
            "tokens_used": 0,
            "tokens_budget": 2000,
            "candidates_considered": 0,
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            result = client.recall("nonexistent")
            assert result.hits == []


class TestErrorHandling:
    def test_api_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal error"}

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(SmritiError) as exc_info:
                client.remember("test")
            assert exc_info.value.status_code == 500


class TestHealth:
    def test_health_ok(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(client._session, "request", return_value=mock_resp):
            assert client.health() is True

    def test_health_down(self, client):
        with patch.object(
            client._session, "request", side_effect=ConnectionError
        ):
            assert client.health() is False


class TestLink:
    def test_link_temporal_edge(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.link("abc", "def", edge="before")
            body = mock_req.call_args[1]["json"]
            assert body["edge"] == "before"


class TestForget:
    def test_forget_uses_post_route(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.forget("abc")
            call_args = mock_req.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "http://localhost:4000/api/forget"
            assert call_args[1]["json"] == {"id": "abc"}


class TestContextManager:
    def test_context_manager(self):
        with SmritiClient() as client:
            assert client is not None
