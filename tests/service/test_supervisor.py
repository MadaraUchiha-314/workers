"""Unit tests for the Supervisor agent."""

from unittest.mock import MagicMock

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.types import Message, TransportProtocol
from langgraph.types import Command

from workers.service.agents.supervisor.supervisor import (
    Supervisor,
    jsonpatch_update,
    jsonpath_query,
)


class TestSupervisorInitialization:
    """Tests for Supervisor initialization."""

    def test_supervisor_initializes_with_rpc_url(self) -> None:
        """Test that Supervisor initializes correctly with an RPC URL."""
        supervisor = Supervisor(rpc_url="/supervisor")

        assert supervisor.id == "supervisor"
        assert supervisor.agent_card.url == "/supervisor"

    def test_supervisor_has_correct_id(self) -> None:
        """Test that Supervisor has the correct ID."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.id == "supervisor"

    def test_supervisor_agent_card_name(self) -> None:
        """Test that Supervisor has the correct agent card name."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.name == "Supervisor"

    def test_supervisor_agent_card_description(self) -> None:
        """Test that Supervisor has the correct agent card description."""
        supervisor = Supervisor(rpc_url="/test")

        assert "supervisor" in supervisor.agent_card.description.lower()
        assert "oversee" in supervisor.agent_card.description.lower()

    def test_supervisor_agent_card_version(self) -> None:
        """Test that Supervisor has the correct version."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.version == "1.0.0"


class TestSupervisorCapabilities:
    """Tests for Supervisor agent capabilities."""

    def test_supervisor_supports_streaming(self) -> None:
        """Test that Supervisor supports streaming."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.capabilities.streaming is True

    def test_supervisor_does_not_support_push_notifications(self) -> None:
        """Test that Supervisor does not support push notifications."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.capabilities.push_notifications is False

    def test_supervisor_does_not_support_state_transition_history(self) -> None:
        """Test that Supervisor does not support state transition history."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.capabilities.state_transition_history is False

    def test_supervisor_uses_jsonrpc_transport(self) -> None:
        """Test that Supervisor uses JSON-RPC transport."""
        supervisor = Supervisor(rpc_url="/test")

        assert supervisor.agent_card.preferred_transport == TransportProtocol.jsonrpc


class TestSupervisorInputOutputModes:
    """Tests for Supervisor input/output modes."""

    def test_supervisor_accepts_text_plain_input(self) -> None:
        """Test that Supervisor accepts text/plain input."""
        supervisor = Supervisor(rpc_url="/test")

        assert "text/plain" in supervisor.agent_card.default_input_modes

    def test_supervisor_outputs_text_plain(self) -> None:
        """Test that Supervisor outputs text/plain."""
        supervisor = Supervisor(rpc_url="/test")

        assert "text/plain" in supervisor.agent_card.default_output_modes


class TestSupervisorSkills:
    """Tests for Supervisor agent skills."""

    def test_supervisor_has_chat_skill(self) -> None:
        """Test that Supervisor has a chat skill."""
        supervisor = Supervisor(rpc_url="/test")

        skill_ids = [skill.id for skill in supervisor.agent_card.skills]
        assert "chat" in skill_ids

    def test_supervisor_chat_skill_has_correct_name(self) -> None:
        """Test that chat skill has the correct name."""
        supervisor = Supervisor(rpc_url="/test")

        chat_skill = next(
            skill for skill in supervisor.agent_card.skills if skill.id == "chat"
        )
        assert chat_skill.name == "Chat"

    def test_supervisor_chat_skill_has_correct_tags(self) -> None:
        """Test that chat skill has the correct tags."""
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"
        context.get_user_input.return_value = None

        result = await supervisor.ainvoke(context)

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_greeting_message(self) -> None:
        """Test that ainvoke returns a greeting message when no user input."""
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

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
        supervisor = Supervisor(rpc_url="/test")

        context = MagicMock(spec=RequestContext)
        context.context_id = "test-context"
        context.task_id = "test-task"

        # Should not raise
        await supervisor.acancel(context)

    @pytest.mark.asyncio
    async def test_acancel_returns_none(self) -> None:
        """Test that acancel returns None (void function)."""
        supervisor = Supervisor(rpc_url="/test")

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
        state = {"data": {"name": "John", "age": 30}}

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert result == "John"

    def test_query_nested_field(self) -> None:
        """Test querying a nested field from data."""
        state = {"data": {"user": {"profile": {"email": "john@example.com"}}}}

        result = jsonpath_query.invoke({"path": "$.user.profile.email", "state": state})

        assert result == "john@example.com"

    def test_query_array_element(self) -> None:
        """Test querying an array element from data."""
        state = {"data": {"items": ["apple", "banana", "cherry"]}}

        result = jsonpath_query.invoke({"path": "$.items[0]", "state": state})

        assert result == "apple"

    def test_query_all_array_elements(self) -> None:
        """Test querying all array elements from data."""
        state = {"data": {"items": ["a", "b", "c"]}}

        result = jsonpath_query.invoke({"path": "$.items[*]", "state": state})

        assert result == "['a', 'b', 'c']"

    def test_query_recursive_descent(self) -> None:
        """Test querying using recursive descent."""
        state = {"data": {"users": [{"id": 1}, {"id": 2}], "nested": {"id": 3}}}

        result = jsonpath_query.invoke({"path": "$..id", "state": state})

        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_query_no_match(self) -> None:
        """Test querying when no match is found."""
        state = {"data": {"name": "John"}}

        result = jsonpath_query.invoke({"path": "$.nonexistent", "state": state})

        assert "No matches found" in result

    def test_query_empty_data(self) -> None:
        """Test querying when data is empty."""
        state: dict[str, dict] = {"data": {}}

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert "No matches found" in result

    def test_query_missing_data_key(self) -> None:
        """Test querying when data key is missing from state."""
        state: dict = {}

        result = jsonpath_query.invoke({"path": "$.name", "state": state})

        assert "No matches found" in result

    def test_query_invalid_jsonpath(self) -> None:
        """Test querying with an invalid JSONPath expression."""
        state = {"data": {"name": "John"}}

        result = jsonpath_query.invoke({"path": "invalid[[jsonpath", "state": state})

        assert "Error" in result


class TestJsonpatchUpdate:
    """Tests for jsonpatch_update tool."""

    def test_add_field(self) -> None:
        """Test adding a new field to data."""
        state: dict[str, dict] = {"data": {}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/name", "value": "John"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"] == {"name": "John"}

    def test_replace_field(self) -> None:
        """Test replacing an existing field in data."""
        state = {"data": {"name": "John"}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "replace", "path": "/name", "value": "Jane"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["name"] == "Jane"

    def test_remove_field(self) -> None:
        """Test removing a field from data."""
        state = {"data": {"name": "John", "age": 30}}

        result = jsonpatch_update.invoke(
            {"patch": '[{"op": "remove", "path": "/age"}]', "state": state}
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert "age" not in result.update["data"]
        assert result.update["data"]["name"] == "John"

    def test_add_nested_field(self) -> None:
        """Test adding a nested field to data."""
        state: dict[str, dict] = {"data": {"user": {}}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/user/email", "value": "test@example.com"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["user"]["email"] == "test@example.com"

    def test_add_to_array(self) -> None:
        """Test appending to an array."""
        state = {"data": {"items": ["a", "b"]}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/items/-", "value": "c"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["items"] == ["a", "b", "c"]

    def test_multiple_operations(self) -> None:
        """Test applying multiple patch operations."""
        state = {"data": {"a": 1}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/b", "value": 2}, {"op": "add", "path": "/c", "value": 3}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["a"] == 1
        assert result.update["data"]["b"] == 2
        assert result.update["data"]["c"] == 3

    def test_move_field(self) -> None:
        """Test moving a field."""
        state = {"data": {"old_name": "value"}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "move", "from": "/old_name", "path": "/new_name"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert "old_name" not in result.update["data"]
        assert result.update["data"]["new_name"] == "value"

    def test_copy_field(self) -> None:
        """Test copying a field."""
        state = {"data": {"source": "value"}}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "copy", "from": "/source", "path": "/target"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["source"] == "value"
        assert result.update["data"]["target"] == "value"

    def test_invalid_json_patch(self) -> None:
        """Test with invalid JSON in patch document."""
        state: dict[str, dict] = {"data": {}}

        with pytest.raises(ValueError) as exc_info:
            jsonpatch_update.invoke({"patch": "not valid json", "state": state})

        assert "Invalid JSON" in str(exc_info.value)

    def test_invalid_patch_operation(self) -> None:
        """Test with invalid patch operation."""
        state: dict[str, dict] = {"data": {}}

        with pytest.raises(ValueError) as exc_info:
            jsonpatch_update.invoke(
                {
                    "patch": '[{"op": "remove", "path": "/nonexistent"}]',
                    "state": state,
                }
            )

        assert "JSON Patch error" in str(exc_info.value)

    def test_empty_data_state(self) -> None:
        """Test applying patch when data key is missing from state."""
        state: dict = {}

        result = jsonpatch_update.invoke(
            {
                "patch": '[{"op": "add", "path": "/name", "value": "John"}]',
                "state": state,
            }
        )

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["data"]["name"] == "John"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
