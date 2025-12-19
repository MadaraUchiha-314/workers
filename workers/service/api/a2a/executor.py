from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from workers.framework.agent.agent import Agent
from a2a.utils.task import new_task
from a2a.types import Message, Task, TaskState, TaskStatusUpdateEvent, TaskArtifactUpdateEvent
from a2a.server.tasks import TaskUpdater

class A2AExecutor(AgentExecutor):
    def __init__(self, agent: Agent):
        self.agent = agent
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract the current task from the context
        # This is the current task being processed
        task: Task | None = context.current_task
        # If no task is found, create a new task
        # This is the first message from the user
        if task is None:
            # Create a new task for the first message from the user
            task = new_task(context.message)
        # Initialize the task updater
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        # Set the current task in the context
        context.current_task = task
        if context.configuration and context.configuration.blocking:
            try:
                # If the configuration is blocking, wait for the agent to complete
                response = await self.agent.ainvoke(context)
                if isinstance(response, Task):
                    if task.status.state == TaskState.completed:
                        await task_updater.complete(response)
                    elif task.status.state == TaskState.failed:
                        await task_updater.failed(response)
                    elif task.status.state == TaskState.canceled:
                        await task_updater.cancel(response)
                    elif task.status.state == TaskState.rejected:
                        await task_updater.reject(response)
                elif isinstance(response, Message):
                    await task_updater.complete(response)
            except Exception as e:
                await task_updater.failed(Message(content=str(e)))
        else:
            # If the configuration is not blocking, stream the agent's response
            async for event in self.agent.astream(context):
                if isinstance(event, Task):
                    if task.status.state == TaskState.completed:
                        await task_updater.complete(response)
                    elif task.status.state == TaskState.failed:
                        await task_updater.failed(response)
                    elif task.status.state == TaskState.canceled:
                        await task_updater.cancel(response)
                    elif task.status.state == TaskState.rejected:
                        await task_updater.reject(response)
                elif isinstance(event, Message):
                    await task_updater.complete(event)
                elif isinstance(event, TaskStatusUpdateEvent):
                    await task_updater.update_status(event.state, event.message)
                elif isinstance(event, TaskArtifactUpdateEvent):
                    await task_updater.add_artifact(event.parts, event.artifact_id, event.name, event.metadata, event.append, event.last_chunk, event.extensions)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError()