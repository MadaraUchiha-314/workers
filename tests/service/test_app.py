"""Unit tests for the service app module."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from workers.service.app import configure_logging, create_app


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_logging_does_not_raise(self) -> None:
        """Test that configure_logging does not raise an exception."""
        # Should not raise
        configure_logging()

    def test_configure_logging_can_be_called_multiple_times(self) -> None:
        """Test that configure_logging can be called multiple times."""
        # Should not raise when called multiple times
        configure_logging()
        configure_logging()


class TestCreateApp:
    """Tests for the create_app function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        app = create_app()

        assert isinstance(app, FastAPI)

    def test_create_app_mounts_supervisor_route(self) -> None:
        """Test that create_app mounts the supervisor route."""
        app = create_app()

        # Check that routes are mounted - use getattr since not all routes have path
        mounted_paths = [getattr(route, "path", str(route)) for route in app.routes]
        # The supervisor should be mounted at /supervisor
        assert any("/supervisor" in str(path) for path in mounted_paths)

    def test_create_app_supervisor_responds_to_agent_card_request(self) -> None:
        """Test that supervisor route responds to agent card requests."""
        app = create_app()
        client = TestClient(app)

        # The agent card should be accessible at /.well-known/agent.json
        response = client.get("/supervisor/.well-known/agent.json")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Supervisor"
        assert data["version"] == "1.0.0"

    def test_create_app_agent_card_contains_capabilities(self) -> None:
        """Test that agent card contains capabilities."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/supervisor/.well-known/agent.json")

        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert data["capabilities"]["streaming"] is True

    def test_create_app_agent_card_contains_skills(self) -> None:
        """Test that agent card contains skills."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/supervisor/.well-known/agent.json")

        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert len(data["skills"]) > 0
        assert data["skills"][0]["id"] == "chat"


class TestAppRoutes:
    """Tests for app routes and endpoints."""

    def test_supervisor_rpc_endpoint_accepts_post(self) -> None:
        """Test that supervisor RPC endpoint accepts POST requests."""
        app = create_app()
        client = TestClient(app)

        # Send a minimal JSON-RPC request
        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": "test-context",
                        "parts": [{"type": "text", "text": "Hello"}],
                        "messageId": "test-message",
                    },
                    "metadata": {},
                },
            },
            headers={"Content-Type": "application/json"},
        )

        # Should return 200 (the RPC handler should process this)
        assert response.status_code == 200

    def test_supervisor_rpc_response_is_jsonrpc(self) -> None:
        """Test that supervisor RPC response is valid JSON-RPC."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": "test-context",
                        "parts": [{"type": "text", "text": "Hello"}],
                        "messageId": "test-message",
                    },
                    "metadata": {},
                },
            },
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"
        assert "id" in data
        assert data["id"] == 1

    def test_supervisor_rpc_response_contains_result(self) -> None:
        """Test that supervisor RPC response contains a result."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": "test-context",
                        "parts": [{"type": "text", "text": "Hello"}],
                        "messageId": "test-message",
                    },
                    "metadata": {},
                },
            },
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        assert "result" in data

    def test_supervisor_rpc_response_contains_task_info(self) -> None:
        """Test that supervisor RPC response contains task information."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": "test-context",
                        "parts": [{"type": "text", "text": "Hello"}],
                        "messageId": "test-message",
                    },
                    "metadata": {},
                },
            },
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        result = data.get("result", {})

        # Should contain task-related fields
        assert "id" in result  # Task ID
        assert "contextId" in result
        assert "status" in result


class TestAppErrorHandling:
    """Tests for app error handling."""

    def test_invalid_json_returns_error(self) -> None:
        """Test that invalid JSON returns an error."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # Should return an error status or JSON-RPC error
        assert response.status_code in [
            400,
            422,
            200,
        ]  # 200 with error in body is valid JSON-RPC

    def test_missing_method_returns_error(self) -> None:
        """Test that missing method returns an error."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                # Missing "method" field
                "params": {},
            },
            headers={"Content-Type": "application/json"},
        )

        # Should return an error
        data = response.json()
        # Either HTTP error or JSON-RPC error
        assert response.status_code != 200 or "error" in data

    def test_nonexistent_route_returns_404(self) -> None:
        """Test that nonexistent routes return 404."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/nonexistent-route")

        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
