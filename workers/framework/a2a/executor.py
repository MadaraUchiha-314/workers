from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.message import new_agent_text_message
from a2a.utils.task import new_task

from workers.framework.agent.agent import Agent, TaskResponse


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
            # Check that we have a message to create a task from
            if context.message is None:
                raise ValueError("Cannot create task: no message in context")
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
                if isinstance(response, TaskResponse):
                    # For TaskResponse, add artifacts and update status
                    if response.artifacts:
                        for artifact in response.artifacts:
                            await task_updater.add_artifact(
                                parts=artifact.parts,
                                artifact_id=artifact.artifact_id,
                                name=artifact.name,
                                metadata=artifact.metadata,
                                extensions=artifact.extensions,
                            )
                    # Use the task status to determine outcome
                    if response.status.state == TaskState.completed:
                        await task_updater.complete(response.status.message)
                    elif response.status.state == TaskState.failed:
                        await task_updater.failed(response.status.message)
                    elif response.status.state == TaskState.canceled:
                        await task_updater.cancel(response.status.message)
                    elif response.status.state == TaskState.rejected:
                        await task_updater.reject(response.status.message)
                elif isinstance(response, Message):
                    await task_updater.complete(response)
            except Exception as e:
                error_message = new_agent_text_message(
                    str(e), context_id=task.context_id, task_id=task.id
                )
                await task_updater.failed(error_message)
        else:
            try:
                # If the configuration is not blocking, stream the agent's response
                async for event in self.agent.astream(context):
                    if isinstance(event, TaskResponse):
                        # For TaskResponse events, add artifacts and update status
                        if event.artifacts:
                            for artifact in event.artifacts:
                                await task_updater.add_artifact(
                                    parts=artifact.parts,
                                    artifact_id=artifact.artifact_id,
                                    name=artifact.name,
                                    metadata=artifact.metadata,
                                    extensions=artifact.extensions,
                                )
                        # Use the task status to determine outcome
                        if event.status.state == TaskState.completed:
                            await task_updater.complete(event.status.message)
                        elif event.status.state == TaskState.failed:
                            await task_updater.failed(event.status.message)
                        elif event.status.state == TaskState.canceled:
                            await task_updater.cancel(event.status.message)
                        elif event.status.state == TaskState.rejected:
                            await task_updater.reject(event.status.message)
                    elif isinstance(event, Message):
                        await task_updater.complete(event)
                    elif isinstance(event, TaskStatusUpdateEvent):
                        # TaskStatusUpdateEvent has a status field with state and message
                        await task_updater.update_status(
                            event.status.state, event.status.message
                        )
                    elif isinstance(event, TaskArtifactUpdateEvent):
                        # TaskArtifactUpdateEvent has an artifact field
                        artifact = event.artifact
                        await task_updater.add_artifact(
                            parts=artifact.parts,
                            artifact_id=artifact.artifact_id,
                            name=artifact.name,
                            metadata=artifact.metadata,
                            append=event.append,
                            last_chunk=event.last_chunk,
                            extensions=artifact.extensions,
                        )
            except Exception as e:
                error_message = new_agent_text_message(
                    str(e), context_id=task.context_id, task_id=task.id
                )
                await task_updater.failed(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError()
