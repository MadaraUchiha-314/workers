from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any

import jsonpatch
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
from a2a.utils.message import new_agent_text_message
from jsonpath_ng import parse as jsonpath_parse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, SecretStr

from workers.framework.agent.agent import Agent, Message, RequestContext


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


@tool
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


@tool
def jsonpatch_update(
    patch: str, state: Annotated[dict[str, Any], InjectedState]
) -> Command:
    """Modify the agent's data store using JSON Patch operations.

    Use this tool to add, remove, replace, move, copy, or test values in the
    agent's structured data. The patch must be a valid JSON Patch document
    (RFC 6902).

    Args:
        patch: A JSON Patch document as a JSON string. Each operation has:
               - "op": The operation ("add", "remove", "replace", "move", "copy", "test")
               - "path": JSON Pointer to the target location
               - "value": The value (required for add, replace, test)
               - "from": Source path (required for move, copy)

    Examples:
        - '[{"op": "add", "path": "/name", "value": "John"}]' - Add a name field
        - '[{"op": "replace", "path": "/age", "value": 30}]' - Update age
        - '[{"op": "remove", "path": "/temp"}]' - Remove temp field
        - '[{"op": "add", "path": "/items/-", "value": "new item"}]' - Append to array
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

        # Capture system prompt for use in closure
        system_prompt = self._get_system_prompt()

        # Define the plan node
        def plan(state: AgentState) -> dict[str, list]:
            """Plan and reflect node - makes LLM call."""
            messages = list(state.messages)

            # Prepend system prompt if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                system_message = SystemMessage(content=system_prompt)
                messages = [system_message, *messages]

            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Define the condition to check for tool calls
        def should_continue(state: AgentState) -> str:
            """Determine whether to continue to tools or end."""
            messages = state.messages
            last_message = messages[-1]

            # Check if the last message is an AIMessage with tool_calls
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return str(END)

        # Build the graph
        graph_builder = StateGraph(AgentState)

        # Add nodes
        graph_builder.add_node("plan", plan)
        graph_builder.add_node("tools", ToolNode(self._tools))

        # Add edges
        graph_builder.add_edge(START, "plan")
        graph_builder.add_conditional_edges("plan", should_continue, ["tools", END])
        graph_builder.add_edge("tools", "plan")

        return graph_builder.compile()

    async def _run_agent(self, user_input: str) -> str:
        """Run the agent with user input and return the final response."""
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (lazy initialization)
        graph = self._get_graph()
        result = await graph.ainvoke(initial_state)

        # Extract the final AI message
        messages = result["messages"]
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)

        raise RuntimeError("Graph execution failed: no valid AI response generated")

    async def ainvoke(self, context: RequestContext) -> Message:
        # Extract user input from context
        user_input = context.get_user_input()

        if not user_input:
            return new_agent_text_message(
                "Hello! How can I help you today?",
                context_id=context.context_id,
                task_id=context.task_id,
            )

        # Run the agent and get the response
        response = await self._run_agent(user_input)

        return new_agent_text_message(
            response,
            context_id=context.context_id,
            task_id=context.task_id,
        )

    async def astream(self, context: RequestContext) -> AsyncGenerator[Message, None]:
        # Extract user input from context
        user_input = context.get_user_input()

        if not user_input:
            yield new_agent_text_message(
                "Hello! How can I help you today?",
                context_id=context.context_id,
                task_id=context.task_id,
            )
            return

        # For streaming, we run the graph and yield the final response
        # (Full streaming of intermediate steps could be implemented with stream mode)
        response = await self._run_agent(user_input)

        yield new_agent_text_message(
            response,
            context_id=context.context_id,
            task_id=context.task_id,
        )

    async def acancel(self, context: RequestContext) -> None:
        # Cancel is a no-op for now
        pass
