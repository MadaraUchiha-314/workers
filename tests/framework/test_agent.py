"""Unit tests for the Agent abstract base class."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TransportProtocol,
)

from workers.framework.agent.agent import Agent


class ConcreteAgent(Agent):
    """A concrete implementation of Agent for testing purposes."""

    def __init__(self, id: str, agent_card: AgentCard):
        super().__init__(id=id, agent_card=agent_card)
        self._invoke_response: Message | Task | None = None
        self._stream_events: list = []
        self._cancel_called = False

    def set_invoke_response(self, response: Message | Task) -> None:
        """Set the response to return from ainvoke."""
        self._invoke_response = response

    def set_stream_events(self, events: list) -> None:
        """Set the events to yield from astream."""
        self._stream_events = events

    async def ainvoke(self, context: RequestContext) -> Message | Task:
        """Return the configured response."""
        if self._invoke_response is None:
            raise ValueError("No invoke response configured")
        return self._invoke_response

    async def astream(
        self, context: RequestContext
    ) -> AsyncGenerator[
        Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None
    ]:
        """Yield the configured events."""
        for event in self._stream_events:
            yield event

    async def acancel(self, context: RequestContext) -> None:
        """Mark that cancel was called."""
        self._cancel_called = True


def create_test_agent_card(
    name: str = "TestAgent",
    url: str = "/test",
) -> AgentCard:
    """Create an AgentCard for testing."""
    return AgentCard(
        name=name,
        description=f"A test agent named {name}",
        version="1.0.0",
        url=url,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
            )
        ],
        preferred_transport=TransportProtocol.jsonrpc,
    )


class TestAgentInitialization:
    """Tests for Agent initialization."""

    def test_agent_initialization_with_valid_params(self) -> None:
        """Test that an agent can be initialized with valid parameters."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        assert agent.id == "test-agent"
        assert agent.agent_card == agent_card
        assert agent.agent_card.name == "TestAgent"
        assert agent.agent_card.url == "/test"

    def test_agent_stores_id_correctly(self) -> None:
        """Test that the agent stores the ID correctly."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="my-unique-id", agent_card=agent_card)

        assert agent.id == "my-unique-id"

    def test_agent_stores_agent_card_correctly(self) -> None:
        """Test that the agent stores the AgentCard correctly."""
        agent_card = create_test_agent_card(name="CustomAgent", url="/custom")
        agent = ConcreteAgent(id="custom-agent", agent_card=agent_card)

        assert agent.agent_card.name == "CustomAgent"
        assert agent.agent_card.url == "/custom"
        assert agent.agent_card.version == "1.0.0"


class TestAgentAbstractMethods:
    """Tests for Agent abstract method implementations."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_message(self) -> None:
        """Test that ainvoke can return a Message."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        # Create a mock message
        mock_message = MagicMock(spec=Message)
        agent.set_invoke_response(mock_message)

        # Create a mock context
        mock_context = MagicMock(spec=RequestContext)

        result = await agent.ainvoke(mock_context)

        assert result is mock_message

    @pytest.mark.asyncio
    async def test_ainvoke_returns_task(self) -> None:
        """Test that ainvoke can return a Task."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        # Create a mock task
        mock_task = MagicMock(spec=Task)
        agent.set_invoke_response(mock_task)

        mock_context = MagicMock(spec=RequestContext)

        result = await agent.ainvoke(mock_context)

        assert result is mock_task

    @pytest.mark.asyncio
    async def test_ainvoke_raises_when_no_response_configured(self) -> None:
        """Test that ainvoke raises when no response is configured."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        mock_context = MagicMock(spec=RequestContext)

        with pytest.raises(ValueError, match="No invoke response configured"):
            await agent.ainvoke(mock_context)

    @pytest.mark.asyncio
    async def test_astream_yields_events(self) -> None:
        """Test that astream yields configured events."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        # Create mock events
        mock_message = MagicMock(spec=Message)
        mock_task = MagicMock(spec=Task)
        agent.set_stream_events([mock_message, mock_task])

        mock_context = MagicMock(spec=RequestContext)

        events = []
        async for event in agent.astream(mock_context):
            events.append(event)

        assert len(events) == 2
        assert events[0] is mock_message
        assert events[1] is mock_task

    @pytest.mark.asyncio
    async def test_astream_yields_empty_when_no_events(self) -> None:
        """Test that astream yields nothing when no events configured."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        mock_context = MagicMock(spec=RequestContext)

        events = []
        async for event in agent.astream(mock_context):
            events.append(event)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_acancel_can_be_called(self) -> None:
        """Test that acancel can be called."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        mock_context = MagicMock(spec=RequestContext)

        assert not agent._cancel_called
        await agent.acancel(mock_context)
        assert agent._cancel_called


class TestAgentCardCapabilities:
    """Tests for AgentCard capabilities."""

    def test_agent_card_streaming_capability(self) -> None:
        """Test that streaming capability is stored correctly."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        assert agent.agent_card.capabilities.streaming is True

    def test_agent_card_push_notifications_capability(self) -> None:
        """Test that push_notifications capability is stored correctly."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        assert agent.agent_card.capabilities.push_notifications is False

    def test_agent_card_skills(self) -> None:
        """Test that skills are stored correctly."""
        agent_card = create_test_agent_card()
        agent = ConcreteAgent(id="test-agent", agent_card=agent_card)

        assert len(agent.agent_card.skills) == 1
        assert agent.agent_card.skills[0].id == "test-skill"
        assert agent.agent_card.skills[0].name == "Test Skill"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
