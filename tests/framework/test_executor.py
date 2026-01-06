"""Unit tests for the A2AExecutor class."""

from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    TransportProtocol,
)

from workers.framework.a2a.executor import A2AExecutor
from workers.framework.agent.agent import Agent, TaskResponse


class MockAgent(Agent):
    """Mock agent for testing A2AExecutor."""

    def __init__(self):
        agent_card = AgentCard(
            name="MockAgent",
            description="A mock agent for testing",
            version="1.0.0",
            url="/mock",
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
                state_transition_history=False,
            ),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[
                AgentSkill(
                    id="mock-skill",
                    name="Mock Skill",
                    description="A mock skill",
                    tags=["mock"],
                )
            ],
            preferred_transport=TransportProtocol.jsonrpc,
        )
        super().__init__(id="mock-agent", agent_card=agent_card)
        self._invoke_response: Message | TaskResponse | None = None
        self._stream_events: list = []
        self._should_raise: Exception | None = None

    def set_invoke_response(self, response: Message | TaskResponse) -> None:
        self._invoke_response = response

    def set_stream_events(self, events: list) -> None:
        self._stream_events = events

    def set_should_raise(self, error: Exception) -> None:
        self._should_raise = error

    async def ainvoke(self, context: RequestContext) -> Message | TaskResponse:
        if self._should_raise:
            raise self._should_raise
        if self._invoke_response is None:
            raise ValueError("No invoke response configured")
        return self._invoke_response

    async def astream(
        self, context: RequestContext
    ) -> AsyncGenerator[
        Message | TaskResponse | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None
    ]:
        if self._should_raise:
            raise self._should_raise
        for event in self._stream_events:
            yield event

    async def acancel(self, context: RequestContext) -> None:
        pass


def create_mock_message(
    context_id: str = "test-context",
    task_id: str = "test-task",
    message_id: str = "test-message",
) -> Message:
    """Create a mock Message for testing."""
    return Message(
        context_id=context_id,
        task_id=task_id,
        message_id=message_id,
        role=Role.agent,
        parts=[cast(Part, TextPart(text="Hello, this is a test response!"))],
    )


def create_mock_task(
    task_id: str = "test-task",
    context_id: str = "test-context",
    state: TaskState = TaskState.completed,
) -> Task:
    """Create a mock Task for testing (used by executor for task creation)."""
    message = create_mock_message(context_id=context_id, task_id=task_id)
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state),
        history=[message],
    )


def create_mock_task_response(
    state: TaskState = TaskState.completed,
    message: Message | None = None,
    artifacts: list[Artifact] | None = None,
) -> TaskResponse:
    """Create a mock TaskResponse for testing."""
    return TaskResponse(
        status=TaskStatus(state=state, message=message),
        artifacts=artifacts,
    )


def create_mock_context(
    context_id: str = "test-context",
    task_id: str = "test-task",
    has_task: bool = False,
    blocking: bool = True,
) -> MagicMock:
    """Create a mock RequestContext for testing."""
    context = MagicMock(spec=RequestContext)
    context.context_id = context_id
    context.task_id = task_id

    # Create a mock message
    mock_message = MagicMock()
    mock_message.context_id = context_id
    mock_message.message_id = "test-message"
    mock_message.role = "user"
    mock_message.parts = [TextPart(text="Test user message")]
    context.message = mock_message

    if has_task:
        context.current_task = create_mock_task(task_id=task_id, context_id=context_id)
    else:
        context.current_task = None

    # Setup configuration
    mock_config = MagicMock()
    mock_config.blocking = blocking
    context.configuration = mock_config

    return context


class TestA2AExecutorInitialization:
    """Tests for A2AExecutor initialization."""

    def test_executor_stores_agent(self) -> None:
        """Test that the executor stores the agent correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        assert executor.agent is agent


class TestA2AExecutorBlockingMode:
    """Tests for A2AExecutor in blocking mode."""

    @pytest.mark.asyncio
    async def test_execute_with_message_response_completes(self) -> None:
        """Test that execute completes successfully with a Message response."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to return a message
        response_message = create_mock_message()
        agent.set_invoke_response(response_message)

        # Create mock context and event queue
        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        # Mock the new_task function
        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        # Verify event was enqueued (task completion)
        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_with_completed_task_response(self) -> None:
        """Test that execute handles a completed TaskResponse correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to return a completed task response
        response = create_mock_task_response(state=TaskState.completed)
        agent.set_invoke_response(response)

        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_with_failed_task_response(self) -> None:
        """Test that execute handles a failed TaskResponse correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to return a failed task response
        response = create_mock_task_response(state=TaskState.failed)
        agent.set_invoke_response(response)

        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_with_canceled_task_response(self) -> None:
        """Test that execute handles a canceled TaskResponse correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to return a canceled task response
        response = create_mock_task_response(state=TaskState.canceled)
        agent.set_invoke_response(response)

        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_with_rejected_task_response(self) -> None:
        """Test that execute handles a rejected TaskResponse correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to return a rejected task response
        response = create_mock_task_response(state=TaskState.rejected)
        agent.set_invoke_response(response)

        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_handles_exception_in_blocking_mode(self) -> None:
        """Test that execute handles exceptions in blocking mode."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to raise an exception
        agent.set_should_raise(ValueError("Test error"))

        context = create_mock_context(blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        # Verify that failed event was enqueued
        assert event_queue.enqueue_event.called


class TestA2AExecutorStreamingMode:
    """Tests for A2AExecutor in streaming mode."""

    @pytest.mark.asyncio
    async def test_execute_streams_message_events(self) -> None:
        """Test that execute streams Message events correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to stream messages
        stream_message = create_mock_message()
        agent.set_stream_events([stream_message])

        context = create_mock_context(blocking=False)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_streams_task_response_events(self) -> None:
        """Test that execute streams TaskResponse events correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to stream task responses
        stream_response = create_mock_task_response(state=TaskState.completed)
        agent.set_stream_events([stream_response])

        context = create_mock_context(blocking=False)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_streams_status_update_events(self) -> None:
        """Test that execute streams TaskStatusUpdateEvent correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Create a status update event
        status_update = TaskStatusUpdateEvent(
            task_id="test-task",
            context_id="test-context",
            status=TaskStatus(state=TaskState.working),
            final=False,
        )
        agent.set_stream_events([status_update])

        context = create_mock_context(blocking=False)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_streams_artifact_update_events(self) -> None:
        """Test that execute streams TaskArtifactUpdateEvent correctly."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Create an artifact update event
        artifact = Artifact(
            artifact_id="test-artifact",
            parts=[cast(Part, TextPart(text="Test artifact content"))],
            name="test-artifact",
        )
        artifact_update = TaskArtifactUpdateEvent(
            task_id="test-task",
            context_id="test-context",
            artifact=artifact,
            append=False,
            last_chunk=True,
        )
        agent.set_stream_events([artifact_update])

        context = create_mock_context(blocking=False)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        assert event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_handles_exception_in_streaming_mode(self) -> None:
        """Test that execute handles exceptions in streaming mode."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        # Setup agent to raise an exception during streaming
        agent.set_should_raise(RuntimeError("Stream error"))

        context = create_mock_context(blocking=False)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

        # Verify that failed event was enqueued
        assert event_queue.enqueue_event.called


class TestA2AExecutorTaskCreation:
    """Tests for A2AExecutor task creation."""

    @pytest.mark.asyncio
    async def test_execute_creates_new_task_when_none_exists(self) -> None:
        """Test that execute creates a new task when none exists in context."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        response_message = create_mock_message()
        agent.set_invoke_response(response_message)

        context = create_mock_context(has_task=False, blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            mock_task = create_mock_task()
            mock_new_task.return_value = mock_task

            await executor.execute(context, event_queue)

            # Verify new_task was called
            mock_new_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_uses_existing_task_when_present(self) -> None:
        """Test that execute uses existing task from context."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        response_message = create_mock_message()
        agent.set_invoke_response(response_message)

        context = create_mock_context(has_task=True, blocking=True)
        event_queue = MagicMock(spec=EventQueue)
        event_queue.enqueue_event = AsyncMock()

        with patch("workers.framework.a2a.executor.new_task") as mock_new_task:
            await executor.execute(context, event_queue)

            # Verify new_task was NOT called (existing task used)
            mock_new_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_raises_when_no_message_and_no_task(self) -> None:
        """Test that execute raises when no message and no task in context."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        context = create_mock_context(has_task=False, blocking=True)
        context.message = None  # No message

        event_queue = MagicMock(spec=EventQueue)

        with pytest.raises(
            ValueError, match="Cannot create task: no message in context"
        ):
            await executor.execute(context, event_queue)


class TestA2AExecutorCancel:
    """Tests for A2AExecutor cancel method."""

    @pytest.mark.asyncio
    async def test_cancel_raises_not_implemented(self) -> None:
        """Test that cancel raises NotImplementedError."""
        agent = MockAgent()
        executor = A2AExecutor(agent=agent)

        context = MagicMock(spec=RequestContext)
        event_queue = MagicMock(spec=EventQueue)

        with pytest.raises(NotImplementedError):
            await executor.cancel(context, event_queue)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
