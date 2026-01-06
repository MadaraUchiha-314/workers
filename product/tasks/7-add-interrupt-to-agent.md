- Instead of going to the END state of the graph, create another node `wait_for_further_input`
- This node uses LangGraph's interrupt mechanism and waits for an input to continue the execution
- This is used to model a "chatty" agent
- Add an edge from `wait_for_further_input` back to the plan node
- If the agent interrupts, the TaskResponse status should reflect that
    - Introspect the properties of `status` in `TaskResponse`
- While calling interrupt pass only a `Message` object.
    - interrupt(Message(content="blah")) <-- something like this
- While inspecting the graph state after an interrupt:
    - make sure that multitple interrupts are not present
        - if multiple interrupts are present, raise an error
    - make sure that the object passed is of type `AIMessage` else raise an error

Updates:
- use the `task_id` as the `thread_id` instead of `context_id`