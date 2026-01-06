from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from a2a.server.agent_execution.context import RequestContext
from a2a.types import (
    AgentCard,
    Artifact,
    Message,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from pydantic import BaseModel


class TaskResponse(BaseModel):
    """Response from an agent containing task status and optional artifacts.

    This is returned by agents instead of Task, as Task has immutable
    attributes like history, context_id, id that shouldn't be modified
    by the agent.
    """

    artifacts: list[Artifact] | None = None
    status: TaskStatus


class Agent(ABC):
    id: str
    agent_card: AgentCard

    def __init__(self, id: str, agent_card: AgentCard):
        self.id = id
        self.agent_card = agent_card

    @abstractmethod
    async def ainvoke(self, context: RequestContext) -> Message | TaskResponse:
        raise NotImplementedError()

    @abstractmethod
    async def astream(
        self, context: RequestContext
    ) -> AsyncGenerator[
        Message | TaskResponse | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None
    ]:
        # Abstract async generator - implementations should yield events
        raise NotImplementedError()
        # The yield statement makes this an async generator for proper type checking
        yield

    @abstractmethod
    async def acancel(self, context: RequestContext) -> None:
        raise NotImplementedError()
