import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from string import Template
from typing import Annotated, Any

import jsonpatch
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
    TaskStatus,
    TransportProtocol,
)
from a2a.utils.message import new_agent_text_message
from jsonpath_ng import parse as jsonpath_parse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command, Interrupt, interrupt
from pydantic import BaseModel, ConfigDict, SecretStr

from workers.framework.agent.agent import Agent, Message, RequestContext, TaskResponse


def load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text()


# Define example tools that the agent can use
@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime

    return datetime.now().isoformat()


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(parse_docstring=True)
def jsonpath_query(path: str, state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Query the agent's data store using JSONPath expressions.

    Use this tool to retrieve specific values from the agent's structured data.
    JSONPath is a query language for JSON, similar to XPath for XML.

    Args:
        path: A JSONPath expression to query the data (e.g., "$.users[0].name",
              "$..price", "$.store.book[?@.price<10]")

    Examples:
        - "$.name" - Get the 'name' field from root
        - "$.users[*].email" - Get all user emails
        - "$.items[0]" - Get the first item
        - "$..id" - Get all 'id' fields recursively
    """
    try:
        data = state.get("data", {})
        jsonpath_expr = jsonpath_parse(path)
        matches = jsonpath_expr.find(data)

        if not matches:
            return "No matches found for the given JSONPath expression."

        # Extract values from matches
        results = [match.value for match in matches]

        # Return single value if only one match, otherwise return list
        if len(results) == 1:
            return str(results[0])
        return str(results)
    except Exception as e:
        return f"Error querying data: {e}"


@tool(parse_docstring=True)
def jsonpatch_update(
    patch: str, state: Annotated[dict[str, Any], InjectedState]
) -> Command:
    """Modify the agent's data store using JSON Patch operations.

    Use this tool to add, remove, replace, move, copy, or test values in the
    agent's structured data. The patch must be a valid JSON Patch document
    (RFC 6902). Each operation in the patch has an "op" field (add, remove,
    replace, move, copy, test), a "path" field (JSON Pointer to target),
    and optionally "value" (for add/replace/test) or "from" (for move/copy).

    Example: '[{"op": "add", "path": "/name", "value": "John"}]'

    Args:
        patch: A JSON Patch document as a JSON string array of operations.
    """
    import json

    try:
        # Parse the patch document
        patch_doc = json.loads(patch)

        # Get current data from state
        current_data = state.get("data", {})

        # Apply the patch
        new_data = jsonpatch.apply_patch(current_data, patch_doc)

        # Return a Command to update the state
        return Command(update={"data": new_data})
    except json.JSONDecodeError as e:
        # Return command with unchanged data and error message will be in tool output
        raise ValueError(f"Invalid JSON in patch document: {e}") from e
    except jsonpatch.JsonPatchException as e:
        raise ValueError(f"JSON Patch error: {e}") from e
    except Exception as e:
        raise ValueError(f"Error applying patch: {e}") from e


# State definition for the graph
class AgentState(BaseModel):
    """Pydantic model for the agent state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[AnyMessage], add_messages] = []
    data: dict[str, Any] = {}


class Supervisor(Agent):
    def __init__(self, rpc_url: str):
        self.id = "supervisor"
        self.agent_card = AgentCard(
            name="Supervisor",
            description="A supervisor agent that can oversee the execution of other agents",
            version="1.0.0",
            url=rpc_url,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
                state_transition_history=False,
            ),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[
                AgentSkill(
                    id="chat",
                    name="Chat",
                    description="General conversation and task supervision",
                    tags=["chat", "supervision"],
                )
            ],
            preferred_transport=TransportProtocol.jsonrpc,
        )
        super().__init__(id="supervisor", agent_card=self.agent_card)

        # Initialize the LangGraph agent configuration (lazy graph building)
        self._tools = [get_current_time, calculate, jsonpath_query, jsonpatch_update]
        self._system_prompt: str | None = None
        self._graph: Any | None = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt, loading it lazily."""
        if self._system_prompt is None:
            self._system_prompt = load_system_prompt()
        return self._system_prompt

    def _get_graph(self) -> Any:
        """Get the graph, building it lazily."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        """Build the LangGraph ReAct agent graph."""
        # Initialize the LLM with tools bound
        llm = ChatOpenAI(
            model="gpt-oss:20b",
            base_url="http://localhost:11434/v1",
            api_key=SecretStr("ollama"),  # required but ignored
        )
        llm_with_tools = llm.bind_tools(self._tools)

        # Capture system prompt template for use in closure
        system_prompt_template = Template(self._get_system_prompt())

        # Define the plan node
        def plan(state: AgentState) -> dict[str, list]:
            """Plan and reflect node - makes LLM call."""
            messages = list(state.messages)

            # Substitute $agent_state with the current state data
            # Create a minimal state dict with just the data for the prompt
            state_dict = {"data": state.data}
            # Serialize using Pydantic's JSON encoder via model_dump
            state_json = json.dumps(state_dict, indent=2, default=str)
            system_prompt = system_prompt_template.safe_substitute(
                agent_state=state_json
            )

            system_message = SystemMessage(content=system_prompt)
            # Prepend system prompt if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [system_message, *messages]
            else:
                # Update the system message with current state
                messages = [system_message, *messages[1:]]

            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Define the condition to check for tool calls
        def should_continue(state: AgentState) -> str:
            """Determine whether to continue to tools or wait for input."""
            messages = state.messages
            last_message = messages[-1]

            # Check if the last message is an AIMessage with tool_calls
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return "wait_for_further_input"

        # Define the wait_for_further_input node using LangGraph interrupt
        def wait_for_further_input(state: AgentState) -> dict[str, list]:
            """Wait for further input from user using interrupt mechanism."""
            # Get the last AI message to include in the interrupt
            messages = state.messages
            last_message = messages[-1] if messages else None

            # Create an AIMessage to pass to interrupt (keeping LangGraph types internal)
            content = ""
            if isinstance(last_message, AIMessage) and last_message.content:
                content = str(last_message.content)

            # Interrupt with an AIMessage as specified in requirements
            interrupt_message = AIMessage(content=content)
            user_input = interrupt(interrupt_message)

            # When resumed, add the user input as a HumanMessage
            return {"messages": [HumanMessage(content=str(user_input))]}

        # Build the graph
        graph_builder = StateGraph(AgentState)

        # Add nodes
        graph_builder.add_node("plan", plan)
        graph_builder.add_node("tools", ToolNode(self._tools))
        graph_builder.add_node("wait_for_further_input", wait_for_further_input)

        # Add edges
        graph_builder.add_edge(START, "plan")
        graph_builder.add_conditional_edges(
            "plan", should_continue, ["tools", "wait_for_further_input"]
        )
        graph_builder.add_edge("tools", "plan")
        graph_builder.add_edge("wait_for_further_input", "plan")

        # Compile with checkpointer to support interrupts
        checkpointer = MemorySaver()
        return graph_builder.compile(checkpointer=checkpointer)

    def _validate_interrupt_state(self, interrupts: tuple[Interrupt, ...]) -> AIMessage:
        """Validate interrupt state and return the interrupt message.

        Args:
            interrupts: Tuple of interrupts from the graph state.

        Returns:
            The AIMessage object from the interrupt.

        Raises:
            ValueError: If multiple interrupts are present or value is not an AIMessage.
        """
        if len(interrupts) > 1:
            raise ValueError(
                f"Multiple interrupts detected ({len(interrupts)}). "
                "Only a single interrupt is allowed."
            )

        if len(interrupts) == 0:
            raise ValueError("No interrupts found in state.")

        interrupt_value = interrupts[0].value
        if not isinstance(interrupt_value, AIMessage):
            raise ValueError(
                f"Interrupt value must be an AIMessage object, "
                f"got {type(interrupt_value).__name__} instead."
            )

        return interrupt_value

    async def _run_agent(
        self, user_input: str | Command, thread_id: str
    ) -> Message | TaskResponse:
        """Run the agent with user input and return the response.

        Args:
            user_input: Either a string for new input or a Command to resume.
            thread_id: The thread ID for maintaining session state.

        Returns:
            Either a Message (when interrupted) or TaskResponse (completed/interrupted).
        """
        # Run the graph (lazy initialization)
        graph = self._get_graph()

        # Config with thread_id for checkpointing
        config = {"configurable": {"thread_id": thread_id}}

        # Determine input based on whether this is a new message or resumption
        graph_input: Command[Any] | dict[str, list[HumanMessage]]
        if isinstance(user_input, Command):
            graph_input = user_input
        else:
            graph_input = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph
        result = await graph.ainvoke(graph_input, config)

        # Check for interrupts in the state
        state = await graph.aget_state(config)
        if state.tasks and any(task.interrupts for task in state.tasks):
            # Collect all interrupts from tasks
            all_interrupts: list[Interrupt] = []
            for task in state.tasks:
                all_interrupts.extend(task.interrupts)

            # Validate interrupts (single interrupt, AIMessage type)
            ai_message = self._validate_interrupt_state(tuple(all_interrupts))

            # Convert AIMessage to A2A Message for the response
            interrupt_message = new_agent_text_message(str(ai_message.content))

            # Return TaskResponse with input_required status
            return TaskResponse(
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=interrupt_message,
                )
            )

        # Extract the final AI message
        messages = result["messages"]
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return new_agent_text_message(str(message.content))

        raise RuntimeError("Graph execution failed: no valid AI response generated")

    async def ainvoke(self, context: RequestContext) -> Message | TaskResponse:
        # Extract user input from context
        user_input = context.get_user_input()

        if not user_input:
            return new_agent_text_message(
                "Hello! How can I help you today?",
                context_id=context.context_id,
                task_id=context.task_id,
            )

        # Use task_id as thread_id for session management
        # If no task_id, generate a new one
        thread_id = context.task_id or str(uuid.uuid4())

        # Run the agent and get the response
        response = await self._run_agent(user_input, thread_id)

        if isinstance(response, TaskResponse):
            # Interrupted - add context/task IDs to the message if present
            if response.status.message:
                response.status.message.context_id = context.context_id
                response.status.message.task_id = context.task_id
            return response

        # Regular message response
        response.context_id = context.context_id
        response.task_id = context.task_id
        return response

    async def astream(
        self, context: RequestContext
    ) -> AsyncGenerator[Message | TaskResponse, None]:
        # Extract user input from context
        user_input = context.get_user_input()

        if not user_input:
            yield new_agent_text_message(
                "Hello! How can I help you today?",
                context_id=context.context_id,
                task_id=context.task_id,
            )
            return

        # Use task_id as thread_id for session management
        thread_id = context.task_id or str(uuid.uuid4())

        # For streaming, we run the graph and yield the final response
        # (Full streaming of intermediate steps could be implemented with stream mode)
        response = await self._run_agent(user_input, thread_id)

        if isinstance(response, TaskResponse):
            # Interrupted - add context/task IDs to the message if present
            if response.status.message:
                response.status.message.context_id = context.context_id
                response.status.message.task_id = context.task_id
            yield response
            return

        # Regular message response
        response.context_id = context.context_id
        response.task_id = context.task_id
        yield response

    async def acancel(self, context: RequestContext) -> None:
        # Cancel is a no-op for now
        pass
