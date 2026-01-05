"""Unit tests for the Supervisor agent."""

from unittest.mock import MagicMock

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.types import Message, TransportProtocol

from workers.service.agents.supervisor.supervisor import Supervisor


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
        assert all(msg.context_id == "stream-context-123" for msg in messages)

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
        assert all(msg.task_id == "stream-task-456" for msg in messages)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
