from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from a2a.server.agent_execution.context import RequestContext
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)


class Agent(ABC):
    id: str
    agent_card: AgentCard

    def __init__(self, id: str, agent_card: AgentCard):
        self.id = id
        self.agent_card = agent_card

    @abstractmethod
    async def ainvoke(self, context: RequestContext) -> Message | Task:
        raise NotImplementedError()

    @abstractmethod
    async def astream(
        self, context: RequestContext
    ) -> AsyncGenerator[
        Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None
    ]:
        # Abstract async generator - implementations should yield events
        raise NotImplementedError()
        # The yield statement makes this an async generator for proper type checking
        yield

    @abstractmethod
    async def acancel(self, context: RequestContext) -> None:
        raise NotImplementedError()
