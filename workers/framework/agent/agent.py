from abc import ABC, abstractmethod
from typing import AsyncGenerator
from a2a.types import Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent
from a2a.server.agent_execution.context import RequestContext

class Agent(ABC):    
    @abstractmethod
    async def ainvoke(self, context: RequestContext) ->  Message | Task:
        raise NotImplementedError()

    @abstractmethod
    async def astream(self, context: RequestContext) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None]:
        raise NotImplementedError()