# Workers Agent

A ReACT-style AI agent with state management, A2A protocol support, and MCP tools.

## Features

- **ReACT Agent**: Classic ReACT (Reasoning + Acting) pattern using LangGraph
- **State Management**: Query and modify agent state using JSONPath and JSON Patch
- **A2A Protocol**: Implements the [A2A protocol](https://a2a-protocol.org/) for agent-to-agent communication
- **MCP Tools**: Tools exposed via Model Context Protocol with streamable HTTP transport
- **Configurable LLM**: Support for OpenAI and Anthropic models (configurable in code)
- **Streamlit UI**: Modern chat interface with state visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chat Window │  │ State Panel │  │ Memory Panel (Future)  │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐   │
│  │ A2A REST Server │  │ MCP Server (streamable_http)        │   │
│  │ - message.send  │  │ - state_query                       │   │
│  │ - task.get      │  │ - state_modify                      │   │
│  └────────┬────────┘  └──────────────────┬──────────────────┘   │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ReACT Loop                            │    │
│  │  ┌──────────┐    ┌────────────┐    ┌─────────────────┐  │    │
│  │  │ Plan Node│───▶│ Tool Node  │───▶│ State Manager   │  │    │
│  │  │ (LLM)    │◀───│ (Execute)  │    │ (JSONPath/Patch)│  │    │
│  │  └──────────┘    └────────────┘    └─────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/MadaraUchiha-314/workers.git
cd workers

# Install dependencies using Poetry
poetry install

# Or using pip
pip install -e .
```

## Configuration

The LLM configuration is done in code. Edit `src/workers/config.py`:

```python
DEFAULT_CONFIG = Config(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,  # or LLMProvider.ANTHROPIC
        model="gpt-4o",               # or "claude-3-5-sonnet-20241022"
        temperature=0.7,
    ),
    server=ServerConfig(
        host="0.0.0.0",
        port=8000,
    ),
    agent=AgentConfig(
        name="Workers Agent",
        description="A ReACT agent that manages its own state and performs tasks",
    ),
)
```

## Environment Variables

Set your API keys:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Running the Application

### Start the Backend Server

```bash
# Using the module
python -m workers.server.main

# Or using uvicorn directly
uvicorn workers.server.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Streamlit UI

```bash
streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`.

## API Endpoints

### A2A Protocol (`/a2a`)

- `POST /a2a` - Handle A2A JSON-RPC requests
  - `message.send` - Send a message to the agent
  - `task.get` - Get task status

### REST Endpoints

- `GET /health` - Health check
- `GET /state` - Get current agent state summary
- `GET /state/query?jsonpath=<expr>` - Query state using JSONPath
- `GET /contexts` - List all conversation contexts
- `GET /contexts/{context_id}` - Get a specific context

### MCP Endpoints (`/mcp`)

- `GET /mcp/sse` - SSE endpoint for MCP connection
- `POST /mcp/messages/` - POST messages for MCP

## MCP Tools

### state_query

Query the agent's internal state using JSONPath expressions.

```python
# Examples
"$.messages"              # Get all messages
"$.messages[-1].content"  # Get last message content
"$.custom_data.context"   # Get context data
```

### state_modify

Modify the agent's internal state using JSON Patch operations (RFC 6902).

```python
# Examples
[{"op": "add", "path": "/custom_data/context/user_name", "value": "Alice"}]
[{"op": "replace", "path": "/custom_data/context/theme", "value": "dark"}]
[{"op": "remove", "path": "/custom_data/working_memory/temp"}]
```

## Project Structure

```
workers/
├── pyproject.toml           # Dependencies and project config
├── src/
│   └── workers/
│       ├── __init__.py
│       ├── config.py        # LLM and server configuration
│       ├── state.py         # State management with JSONPath/JSONPatch
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── state_query.py
│       │   └── state_modify.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py     # LangGraph ReACT agent
│       │   └── executor.py  # A2A AgentExecutor
│       ├── server/
│       │   ├── __init__.py
│       │   ├── main.py      # FastAPI app
│       │   ├── a2a.py       # A2A REST server
│       │   └── mcp.py       # MCP tools server
│       └── prompts/
│           └── system.md    # System prompt
├── ui/
│   └── app.py               # Streamlit UI
└── README.md
```

## License

MIT
