"""Integration tests for the service - end-to-end API tests.

These tests validate the complete request/response flow through the service,
matching the examples provided in the task specification.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from a2a.types import Artifact, DataPart, Part, TaskState, TaskStatus
from a2a.utils.message import new_agent_text_message
from fastapi.testclient import TestClient

from workers.framework.agent.agent import TaskResponse
from workers.service.app import create_app


@pytest.fixture(autouse=True)
def mock_agent_run():
    """Mock the _run_agent method to avoid needing OpenAI API key in tests."""
    with patch(
        "workers.service.agents.supervisor.supervisor.Supervisor._run_agent",
        new_callable=AsyncMock,
    ) as mock:
        # Return a TaskResponse since _run_agent now always returns TaskResponse
        mock.return_value = TaskResponse(
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message("Hello! I'm a helpful AI assistant."),
            ),
            artifacts=[
                Artifact(
                    artifact_id="test-artifact-id",
                    name="Agent State",
                    description="Test agent state",
                    parts=[Part(root=DataPart(data={"messages": [], "data": {}}))],
                )
            ],
        )
        yield mock


class TestMessageSendEndToEnd:
    """End-to-end tests for the message/send JSON-RPC method."""

    def test_message_send_returns_valid_jsonrpc_response(self) -> None:
        """Test that message/send returns a valid JSON-RPC 2.0 response."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "hi. what's up ?"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200

        data = response.json()

        # Validate JSON-RPC structure
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data

    def test_message_send_returns_task_result(self) -> None:
        """Test that message/send returns a task result."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Hello, agent!"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        result = data["result"]

        # Validate task structure
        assert "id" in result  # Task ID
        assert "contextId" in result
        assert result["contextId"] == context_id
        assert "kind" in result
        assert result["kind"] == "task"

    def test_message_send_returns_completed_status(self) -> None:
        """Test that message/send returns a completed task status."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Test message"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        result = data["result"]

        # Validate status
        assert "status" in result
        assert result["status"]["state"] == "completed"
        assert "timestamp" in result["status"]

    def test_message_send_returns_agent_response_message(self) -> None:
        """Test that message/send returns an agent response message in status."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Hello!"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        result = data["result"]

        # Validate agent message in status
        status_message = result["status"]["message"]
        assert status_message["role"] == "agent"
        assert status_message["contextId"] == context_id
        assert "parts" in status_message
        assert len(status_message["parts"]) > 0

        # The supervisor should respond with a greeting
        first_part = status_message["parts"][0]
        assert "text" in first_part or first_part.get("kind") == "text"

    def test_message_send_returns_history(self) -> None:
        """Test that message/send returns message history."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Test history"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        result = data["result"]

        # Validate history
        assert "history" in result
        assert len(result["history"]) > 0

        # First history entry should be the user message
        first_history = result["history"][0]
        assert first_history["role"] == "user"
        assert first_history["contextId"] == context_id

    def test_message_send_preserves_context_id(self) -> None:
        """Test that the context ID is preserved throughout the request."""
        app = create_app()
        client = TestClient(app)

        context_id = "c2920705-f3eb-4698-b617-cbf4b62de6e5"
        message_id = str(uuid.uuid4())

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [
                            {"type": "text", "text": "Testing context preservation"}
                        ],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        result = data["result"]

        # Context ID should be preserved everywhere
        assert result["contextId"] == context_id
        assert result["status"]["message"]["contextId"] == context_id
        if result.get("history"):
            for item in result["history"]:
                assert item["contextId"] == context_id

    def test_message_send_with_custom_rpc_id(self) -> None:
        """Test that custom RPC IDs are preserved in the response."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        custom_rpc_id = 42

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": custom_rpc_id,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Test"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        assert data["id"] == custom_rpc_id

    def test_message_send_with_string_rpc_id(self) -> None:
        """Test that string RPC IDs are preserved in the response."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        string_rpc_id = "my-custom-id-123"

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": string_rpc_id,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Test"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        data = response.json()
        assert data["id"] == string_rpc_id


class TestMultipleMessagesIntegration:
    """Integration tests for multiple message interactions."""

    def test_multiple_messages_same_context(self) -> None:
        """Test sending multiple messages in the same context."""
        app = create_app()
        client = TestClient(app)

        context_id = str(uuid.uuid4())

        # First message
        response1 = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "First message"}],
                        "messageId": str(uuid.uuid4()),
                    },
                    "metadata": {},
                },
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["result"]["contextId"] == context_id

        # Second message (same context)
        response2 = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "Second message"}],
                        "messageId": str(uuid.uuid4()),
                    },
                    "metadata": {},
                },
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["result"]["contextId"] == context_id

    def test_messages_with_different_contexts(self) -> None:
        """Test sending messages with different context IDs."""
        app = create_app()
        client = TestClient(app)

        context_id_1 = str(uuid.uuid4())
        context_id_2 = str(uuid.uuid4())

        # Message in first context
        response1 = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id_1,
                        "parts": [{"type": "text", "text": "Context 1 message"}],
                        "messageId": str(uuid.uuid4()),
                    },
                    "metadata": {},
                },
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["result"]["contextId"] == context_id_1

        # Message in second context
        response2 = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id_2,
                        "parts": [{"type": "text", "text": "Context 2 message"}],
                        "messageId": str(uuid.uuid4()),
                    },
                    "metadata": {},
                },
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["result"]["contextId"] == context_id_2


class TestAgentCardIntegration:
    """Integration tests for agent card endpoint."""

    def test_agent_card_endpoint_returns_valid_card(self) -> None:
        """Test that the agent card endpoint returns a valid agent card."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/supervisor/.well-known/agent.json")

        assert response.status_code == 200
        data = response.json()

        # Validate required fields
        assert "name" in data
        assert "description" in data
        assert "version" in data
        assert "url" in data
        assert "capabilities" in data
        assert "skills" in data

    def test_agent_card_matches_supervisor_config(self) -> None:
        """Test that the agent card matches the Supervisor configuration."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/supervisor/.well-known/agent.json")

        data = response.json()

        assert data["name"] == "Supervisor"
        assert data["version"] == "1.0.0"
        assert data["capabilities"]["streaming"] is True
        assert data["capabilities"]["pushNotifications"] is False


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_invalid_method_returns_error(self) -> None:
        """Test that an invalid method returns an error."""
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "invalid/method",
                "params": {},
            },
        )

        data = response.json()

        # Should return a JSON-RPC error
        assert "error" in data or response.status_code != 200

    def test_malformed_message_returns_error(self) -> None:
        """Test that a malformed message returns an error."""
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
                        # Missing required fields
                        "role": "user",
                    },
                    "metadata": {},
                },
            },
        )

        # Should return an error (either HTTP or JSON-RPC)
        data = response.json()
        assert "error" in data or response.status_code != 200


class TestResponseFormatMatchesSpec:
    """Tests to verify response format matches the specification from the task."""

    def test_response_matches_spec_structure(self) -> None:
        """Test that the response structure matches the specification.

        This test validates against the example response from the task:
        - Response should have jsonrpc, id, and result fields
        - Result should have contextId, history, id, kind, and status
        - Status should have message and state
        """
        app = create_app()
        client = TestClient(app)

        context_id = "c2920705-f3eb-4698-b617-cbf4b62de6e5"
        message_id = "53a733ae-9248-4b05-a823-77bb9f208d8f"

        response = client.post(
            "/supervisor/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "contextId": context_id,
                        "parts": [{"type": "text", "text": "hi. what's up ?"}],
                        "messageId": message_id,
                    },
                    "metadata": {},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Validate top-level structure
        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"
        assert "id" in data
        assert data["id"] == 1
        assert "result" in data

        result = data["result"]

        # Validate result structure
        assert "contextId" in result
        assert result["contextId"] == context_id
        assert "history" in result
        assert "id" in result  # Task ID
        assert "kind" in result
        assert result["kind"] == "task"
        assert "status" in result

        # Validate status structure
        status = result["status"]
        assert "state" in status
        assert status["state"] == "completed"
        assert "timestamp" in status
        assert "message" in status

        # Validate status message structure
        status_message = status["message"]
        assert "contextId" in status_message
        assert status_message["contextId"] == context_id
        assert "kind" in status_message
        assert status_message["kind"] == "message"
        assert "messageId" in status_message
        assert "parts" in status_message
        assert "role" in status_message
        assert status_message["role"] == "agent"
        assert "taskId" in status_message

        # Validate parts structure
        parts = status_message["parts"]
        assert len(parts) > 0
        first_part = parts[0]
        assert "kind" in first_part
        assert first_part["kind"] == "text"
        assert "text" in first_part

        # Validate history structure
        history = result["history"]
        assert len(history) > 0
        first_history = history[0]
        assert "contextId" in first_history
        assert "kind" in first_history
        assert "messageId" in first_history
        assert "parts" in first_history
        assert "role" in first_history
        assert "taskId" in first_history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
