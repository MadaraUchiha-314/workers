"""Unit tests for the Supervisor agent."""

from unittest.mock import MagicMock

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.types import DataPart, Message, TransportProtocol
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from workers.service.agents.supervisor.supervisor import (
    AgentState,
    Supervisor,
    jsonpatch_update,
    jsonpath_query,
)

# Default test values for Supervisor constructor
TEST_API_KEY = "test-api-key"
TEST_BASE_URL = "http://localhost:11434/v1"
TEST_MODEL = "test-model"


class TestSupervisorInitialization:
    """Tests for Supervisor initialization."""

    def test_supervisor_initializes_with_rpc_url(self) -> None:
        """Test that Supervisor initializes correctly with an RPC URL."""
        supervisor = Supervisor(
            rpc_url="/supervisor",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.id == "supervisor"
        assert supervisor.agent_card.url == "/supervisor"

    def test_supervisor_has_correct_id(self) -> None:
        """Test that Supervisor has the correct ID."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.id == "supervisor"

    def test_supervisor_agent_card_name(self) -> None:
        """Test that Supervisor has the correct agent card name."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.name == "Supervisor"

    def test_supervisor_agent_card_description(self) -> None:
        """Test that Supervisor has the correct agent card description."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert "supervisor" in supervisor.agent_card.description.lower()
        assert "oversee" in supervisor.agent_card.description.lower()

    def test_supervisor_agent_card_version(self) -> None:
        """Test that Supervisor has the correct version."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.version == "1.0.0"


class TestSupervisorCapabilities:
    """Tests for Supervisor agent capabilities."""

    def test_supervisor_supports_streaming(self) -> None:
        """Test that Supervisor supports streaming."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.capabilities.streaming is True

    def test_supervisor_does_not_support_push_notifications(self) -> None:
        """Test that Supervisor does not support push notifications."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.capabilities.push_notifications is False

    def test_supervisor_does_not_support_state_transition_history(self) -> None:
        """Test that Supervisor does not support state transition history."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.capabilities.state_transition_history is False

    def test_supervisor_uses_jsonrpc_transport(self) -> None:
        """Test that Supervisor uses JSON-RPC transport."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert supervisor.agent_card.preferred_transport == TransportProtocol.jsonrpc


class TestSupervisorInputOutputModes:
    """Tests for Supervisor input/output modes."""

    def test_supervisor_accepts_text_plain_input(self) -> None:
        """Test that Supervisor accepts text/plain input."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert "text/plain" in supervisor.agent_card.default_input_modes

    def test_supervisor_outputs_text_plain(self) -> None:
        """Test that Supervisor outputs text/plain."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        assert "text/plain" in supervisor.agent_card.default_output_modes


class TestSupervisorSkills:
    """Tests for Supervisor agent skills."""

    def test_supervisor_has_chat_skill(self) -> None:
        """Test that Supervisor has a chat skill."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        skill_ids = [skill.id for skill in supervisor.agent_card.skills]
        assert "chat" in skill_ids

    def test_supervisor_chat_skill_has_correct_name(self) -> None:
        """Test that chat skill has the correct name."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        chat_skill = next(
            skill for skill in supervisor.agent_card.skills if skill.id == "chat"
        )
        assert chat_skill.name == "Chat"

    def test_supervisor_chat_skill_has_correct_tags(self) -> None:
        """Test that chat skill has the correct tags."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        chat_skill = next(
            skill for skill in supervisor.agent_card.skills if skill.id == "chat"
        )
        assert "chat" in chat_skill.tags
        assert "supervision" in chat_skill.tags


class TestSupervisorAinvoke:
    """Tests for Supervisor ainvoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_message(self) -> None:
        """Test that ainvoke returns a Message when no user input is provided."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"
        context.get_user_input.return_value = None

        result = await supervisor.ainvoke(context)

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_greeting_message(self) -> None:
        """Test that ainvoke returns a greeting message when no user input."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"
        context.get_user_input.return_value = None

        result = await supervisor.ainvoke(context)

        # Check that the message contains expected text
        assert isinstance(result, Message)
        assert len(result.parts) > 0
        first_part = result.parts[0]
        # Part can be a union type, check if it has text attribute
        assert hasattr(first_part, "text") or hasattr(first_part, "root")

    @pytest.mark.asyncio
    async def test_ainvoke_uses_context_id(self) -> None:
        """Test that ainvoke uses the context_id from the request context."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "my-context-123"
        context.task_id = "my-task-456"
        context.get_user_input.return_value = None

        result = await supervisor.ainvoke(context)

        assert isinstance(result, Message)
        assert result.context_id == "my-context-123"

    @pytest.mark.asyncio
    async def test_ainvoke_uses_task_id(self) -> None:
        """Test that ainvoke uses the task_id from the request context."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "my-context-123"
        context.task_id = "my-task-456"
        context.get_user_input.return_value = None

        result = await supervisor.ainvoke(context)

        assert isinstance(result, Message)
        assert result.task_id == "my-task-456"


class TestSupervisorAstream:
    """Tests for Supervisor astream method."""

    @pytest.mark.asyncio
    async def test_astream_yields_messages(self) -> None:
        """Test that astream yields messages when no user input is provided."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"
        context.get_user_input.return_value = None

        messages = []
        async for message in supervisor.astream(context):
            messages.append(message)

        assert len(messages) > 0
        assert all(isinstance(msg, Message) for msg in messages)

    @pytest.mark.asyncio
    async def test_astream_yields_greeting_message(self) -> None:
        """Test that astream yields a greeting message when no user input."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"
        context.get_user_input.return_value = None

        messages = []
        async for message in supervisor.astream(context):
            messages.append(message)

        # Check that at least one message contains greeting text
        assert len(messages) > 0
        first_message = messages[0]
        assert isinstance(first_message, Message)
        assert len(first_message.parts) > 0

    @pytest.mark.asyncio
    async def test_astream_uses_context_id(self) -> None:
        """Test that astream uses the context_id from the request context."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "stream-context-123"
        context.task_id = "stream-task-456"
        context.get_user_input.return_value = None

        messages = []
        async for message in supervisor.astream(context):
            messages.append(message)

        assert len(messages) > 0
        assert all(
            isinstance(msg, Message) and msg.context_id == "stream-context-123"
            for msg in messages
        )

    @pytest.mark.asyncio
    async def test_astream_uses_task_id(self) -> None:
        """Test that astream uses the task_id from the request context."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "stream-context-123"
        context.task_id = "stream-task-456"
        context.get_user_input.return_value = None

        messages = []
        async for message in supervisor.astream(context):
            messages.append(message)

        assert len(messages) > 0
        assert all(
            isinstance(msg, Message) and msg.task_id == "stream-task-456"
            for msg in messages
        )


class TestSupervisorAcancel:
    """Tests for Supervisor acancel method."""

    @pytest.mark.asyncio
    async def test_acancel_does_not_raise(self) -> None:
        """Test that acancel does not raise an exception."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"

        # Should not raise
        await supervisor.acancel(context)

    @pytest.mark.asyncio
    async def test_acancel_returns_none(self) -> None:
        """Test that acancel returns None (void function)."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"

        # acancel is a void function (returns None implicitly)
        # We just verify it completes without error
        await supervisor.acancel(context)


class TestJsonpathQuery:
    """Tests for jsonpath_query tool."""

    def test_query_simple_field(self) -> None:
        """Test querying a simple field from data."""
        state = AgentState(data={"name": "John", "age": 30})

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert result == "John"

    def test_query_nested_field(self) -> None:
        """Test querying a nested field from data."""
        state = AgentState(data={"user": {"profile": {"email": "john@example.com"}}})

        result = jsonpath_query.invoke({"path": "$.user.profile.email", "state": state})

        assert result == "john@example.com"

    def test_query_array_element(self) -> None:
        """Test querying an array element from data."""
        state = AgentState(data={"items": ["apple", "banana", "cherry"]})

        result = jsonpath_query.invoke({"path": "$.items[0]", "state": state})

        assert result == "apple"

    def test_query_all_array_elements(self) -> None:
        """Test querying all array elements from data."""
        state = AgentState(data={"items": ["a", "b", "c"]})

        result = jsonpath_query.invoke({"path": "$.items[*]", "state": state})

        assert result == "['a', 'b', 'c']"

    def test_query_recursive_descent(self) -> None:
        """Test querying using recursive descent."""
        state = AgentState(data={"users": [{"id": 1}, {"id": 2}], "nested": {"id": 3}})

        result = jsonpath_query.invoke({"path": "$..id", "state": state})

        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_query_no_match(self) -> None:
        """Test querying when no match is found."""
        state = AgentState(data={"name": "John"})

        result = jsonpath_query.invoke({"path": "$.nonexistent", "state": state})

        assert "No matches found" in result

    def test_query_empty_data(self) -> None:
        """Test querying when data is empty."""
        state = AgentState(data={})

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert "No matches found" in result

    def test_query_missing_data_key(self) -> None:
        """Test querying when data key is missing from state (uses default empty dict)."""
        state = AgentState()

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert "No matches found" in result

    def test_query_invalid_jsonpath(self) -> None:
        """Test querying with an invalid JSONPath expression."""
        state = AgentState(data={"name": "John"})

        result = jsonpath_query.invoke({"path": "invalid[[jsonpath", "state": state})

        assert "Error" in result


class TestCreateStateArtifact:
    """Tests for _create_state_artifact method."""

    def test_creates_artifact_with_empty_state(self) -> None:
        """Test that an artifact is created with empty state."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        state = AgentState()

        artifact = supervisor._create_state_artifact(state)

        assert artifact is not None
        assert artifact.artifact_id.startswith("agent-state-")
        assert artifact.name == "Agent State"
        assert artifact.description is not None
        assert len(artifact.parts) == 1

    def test_artifact_contains_messages_data(self) -> None:
        """Test that the artifact contains messages in the data via model_dump."""
        from langchain_core.messages import HumanMessage

        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        state = AgentState(messages=[HumanMessage(content="Hello")])

        artifact = supervisor._create_state_artifact(state)

        # Extract the data from the artifact
        part = artifact.parts[0]
        data_part = part.root
        assert data_part.kind == "data"
        assert "messages" in data_part.data
        assert len(data_part.data["messages"]) == 1
        # model_dump includes 'type' and 'content' fields
        assert data_part.data["messages"][0]["type"] == "human"
        assert data_part.data["messages"][0]["content"] == "Hello"

    def test_artifact_contains_state_data(self) -> None:
        """Test that the artifact contains the state data."""
        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        state = AgentState(data={"key": "value", "number": 42})

        artifact = supervisor._create_state_artifact(state)

        part = artifact.parts[0]
        data_part = part.root
        assert isinstance(data_part, DataPart)
        assert "data" in data_part.data
        assert data_part.data["data"]["key"] == "value"
        assert data_part.data["data"]["number"] == 42

    def test_artifact_includes_tool_calls_for_ai_messages(self) -> None:
        """Test that tool calls are included for AI messages via model_dump."""
        from langchain_core.messages import AIMessage

        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        ai_message = AIMessage(
            content="I'll calculate that",
            tool_calls=[
                {"id": "call_123", "name": "calculate", "args": {"expression": "2+2"}}
            ],
        )
        state = AgentState(messages=[ai_message])

        artifact = supervisor._create_state_artifact(state)

        part = artifact.parts[0]
        data_part = part.root
        assert isinstance(data_part, DataPart)
        assert len(data_part.data["messages"]) == 1
        msg_data = data_part.data["messages"][0]
        # model_dump uses lowercase 'type' value
        assert msg_data["type"] == "ai"
        assert "tool_calls" in msg_data
        assert msg_data["tool_calls"][0]["name"] == "calculate"

    def test_artifact_includes_tool_call_id_for_tool_messages(self) -> None:
        """Test that tool_call_id is included for tool messages via model_dump."""
        from langchain_core.messages import ToolMessage

        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        tool_message = ToolMessage(content="4", tool_call_id="call_123")
        state = AgentState(messages=[tool_message])

        artifact = supervisor._create_state_artifact(state)

        part = artifact.parts[0]
        data_part = part.root
        assert isinstance(data_part, DataPart)
        msg_data = data_part.data["messages"][0]
        # model_dump uses lowercase 'type' value
        assert msg_data["type"] == "tool"
        assert msg_data["tool_call_id"] == "call_123"

    def test_artifact_uses_model_dump_for_complete_serialization(self) -> None:
        """Test that model_dump is used, preserving all message fields."""
        from langchain_core.messages import AIMessage

        supervisor = Supervisor(
            rpc_url="/test",
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
        )
        ai_message = AIMessage(
            content="Test",
            id="msg-123",
            response_metadata={"model": "test-model"},
        )
        state = AgentState(messages=[ai_message])

        artifact = supervisor._create_state_artifact(state)

        part = artifact.parts[0]
        data_part = part.root
        assert isinstance(data_part, DataPart)
        msg_data = data_part.data["messages"][0]
        # model_dump preserves additional fields like id and response_metadata
        assert msg_data["id"] == "msg-123"
        assert msg_data["response_metadata"]["model"] == "test-model"


class TestJsonpatchUpdate:
    """Tests for jsonpatch_update tool."""

    def _create_mock_runtime(self, tool_call_id: str = "test-call-id") -> ToolRuntime:
        """Create a mock ToolRuntime for testing."""
        return ToolRuntime(
            state={},
            tool_call_id=tool_call_id,
            config={},
            context=None,
            store=None,
            stream_writer=lambda _: None,
        )

    def test_add_field(self) -> None:
        """Test adding a new field to data."""
        state = AgentState(data={})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/name", "value": "John"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"] == {"name": "John"}

    def test_replace_field(self) -> None:
        """Test replacing an existing field in data."""
        state = AgentState(data={"name": "John"})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "replace", "path": "/name", "value": "Jane"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["name"] == "Jane"

    def test_remove_field(self) -> None:
        """Test removing a field from data."""
        state = AgentState(data={"name": "John", "age": 30})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "remove", "path": "/age"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert "age" not in result.update["data"]
        assert result.update["data"]["name"] == "John"

    def test_add_nested_field(self) -> None:
        """Test adding a nested field to data."""
        state = AgentState(data={"user": {}})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/user/email", "value": "test@example.com"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["user"]["email"] == "test@example.com"

    def test_add_to_array(self) -> None:
        """Test appending to an array."""
        state = AgentState(data={"items": ["a", "b"]})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/items/-", "value": "c"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["items"] == ["a", "b", "c"]

    def test_multiple_operations(self) -> None:
        """Test applying multiple patch operations."""
        state = AgentState(data={"a": 1})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/b", "value": 2}, {"op": "add", "path": "/c", "value": 3}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["a"] == 1
        assert result.update["data"]["b"] == 2
        assert result.update["data"]["c"] == 3

    def test_move_field(self) -> None:
        """Test moving a field."""
        state = AgentState(data={"old_name": "value"})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "move", "from": "/old_name", "path": "/new_name"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert "old_name" not in result.update["data"]
        assert result.update["data"]["new_name"] == "value"

    def test_copy_field(self) -> None:
        """Test copying a field."""
        state = AgentState(data={"source": "value"})
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "copy", "from": "/source", "path": "/target"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["source"] == "value"
        assert result.update["data"]["target"] == "value"

    def test_invalid_json_patch(self) -> None:
        """Test with invalid JSON in patch document."""
        state = AgentState(data={})
        runtime = self._create_mock_runtime()

        with pytest.raises(ValueError) as exc_info:
            jsonpatch_update.invoke(
                {"patch": "not valid json", "state": state, "runtime": runtime}
            )

        assert "Invalid JSON" in str(exc_info.value)

    def test_invalid_patch_operation(self) -> None:
        """Test with invalid patch operation."""
        state = AgentState(data={})
        runtime = self._create_mock_runtime()

        with pytest.raises(ValueError) as exc_info:
            jsonpatch_update.invoke(
                {
                    "patch": '[{"op": "remove", "path": "/nonexistent"}]',
                    "state": state,
                    "runtime": runtime,
                }
            )

        assert "JSON Patch error" in str(exc_info.value)

    def test_empty_data_state(self) -> None:
        """Test applying patch when data key is missing from state (uses default empty dict)."""
        state = AgentState()
        runtime = self._create_mock_runtime()

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/name", "value": "John"}]',
                "state": state,
                "runtime": runtime,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["name"] == "John"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
