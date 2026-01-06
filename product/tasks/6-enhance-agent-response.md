- Agent interface for invoke and stream can return a Task
- This is not correct as Task entity has attributes like `history`, `context_id`, `id` etc which are really immutable
- Make the return value of invoke function only be `TaskResponse`

```py
class TaskResponse(Basemodel):
    artifacts: list[Artifact] | None = None
    status: TaskStatus
```

- The astream can return a Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent or TaskResponse

- The Agent Executor layer understands `TaskResponse` and uses the `TaskUpdater` to process `TaskResponse` and adds artifacts and updates the task using the methods of `TaskUpdater`

- This means `if isinstance(event, Task):` will be replaced by `if isinstance(event, TaskResponse):`