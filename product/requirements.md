# Agent
A chat agent which chats with you and provides solutions to your queries or does your tasks.

## Architecture

- ReACT Agent
- Takes a system prompt which has description of the agent's roles and responsibilities
- In the system prompt, there's a section which describes the internal state of the agent

### State Management & Query
- The agent can query it's state using jsonpath
    - This is exposed as a tool
    - Implement this tool
- The agent can modify it's state using jsonpatch (different from jsonpath)
    - This is exposed as a tool
    - Implement this tool
- messages: AnyMessage is also part of the state

### Agent Graph

Classic ReACT style agent.

- Plan node calls an LLM with system_prompt and current set of messages
    - Tools are bound when the LLM is called.
- If the LLM suggests any tool calls, then proceed to the ToolNode to execute tools
- If there are no futher tools to execute then the last AIMessage produced is the final output of the agent

Notes:
- All prompts are stored in the `prompts/` folder in markdown format
    - system prompt should be in `system.md`

### Agent Interface

```py
class Agent(ABC):
    @abstractmethod
    async def graph(self) -> CompiledStateGraph:
        raise NotImplementedError()
    
    @absrtractmethod
    async def invoke(self) -> Task | Message:
        raise NotImplementedError()
```


## Tech Stack

- Python
    - Poetry for dependency management
- FastAPI
- A2A is protocol through with Agent is exposed to external systems like Streamlit UI
    - A2A Link: https://a2a-protocol.org/latest/specification/
- LangGraph
- MCP
    - All tools should be MCP tools exposed as part of the same service on different URL paths
    - All tools use `streamable_http` as the transport
    - Use LangGraph MCP Adapater
        - https://docs.langchain.com/oss/python/langchain/mcp
- Streamlit for UI
    - One chat window
    - One panel showing agent state
    - One panel showing memories that agent is capturing
        - For future. 