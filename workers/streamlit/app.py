"""Streamlit Chat UI for the A2A-compliant Agent Service."""

import json
import uuid
from typing import Any

import streamlit as st

# Configuration
A2A_ENDPOINT = "http://localhost:8000/supervisor/"
JSONRPC_VERSION = "2.0"


def generate_id() -> str:
    """Generate a unique ID for messages and requests."""
    return str(uuid.uuid4())


def create_jsonrpc_request(
    message_text: str,
    context_id: str,
    task_id: str | None = None,
    message_id: str | None = None,
    request_id: int = 1,
) -> dict:
    """Create a JSONRPC request for the A2A protocol.

    Args:
        message_text: The user's message text.
        context_id: The conversation context ID.
        task_id: Optional task ID. Should be None for first message,
                 then reused for input-required status.
        message_id: Optional message ID (generated if not provided).
        request_id: The JSONRPC request ID.

    Returns:
        A properly formatted JSONRPC request dict.
    """
    message = {
        "role": "user",
        "contextId": context_id,
        "parts": [{"type": "text", "text": message_text}],
        "messageId": message_id or generate_id(),
    }

    # Only include taskId if we have one (not for first message)
    if task_id:
        message["taskId"] = task_id

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": message,
            "configuration": {"blocking": True},
        },
        "metadata": {},
    }


def send_message(
    message_text: str, context_id: str, task_id: str | None = None
) -> dict | None:
    """Send a message to the A2A agent service.

    Args:
        message_text: The user's message.
        context_id: The conversation context ID.
        task_id: Optional task ID (None for first message, reused for input-required).

    Returns:
        The JSONRPC response dict, or None if an error occurred.
    """
    import urllib.error
    import urllib.request

    request_data = create_jsonrpc_request(message_text, context_id, task_id=task_id)

    try:
        # Use urllib to avoid httpx SSL certificate issues on localhost
        req = urllib.request.Request(
            A2A_ENDPOINT,
            data=json.dumps(request_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result
    except urllib.error.HTTPError as e:
        st.error(f"HTTP error: {e.code} - {e.reason}")
        return None
    except urllib.error.URLError as e:
        st.error(f"Connection error: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse response: {e}")
        return None
    except TimeoutError:
        st.error("Request timed out")
        return None


def extract_agent_response(response: dict[str, Any]) -> str | None:
    """Extract the agent's text response from the JSONRPC response.

    Args:
        response: The JSONRPC response dict.

    Returns:
        The agent's text response, or None if not found.
    """
    try:
        result = response.get("result", {})
        status = result.get("status", {})
        message = status.get("message", {})
        parts = message.get("parts", [])

        for part in parts:
            if part.get("kind") == "text":
                text = part.get("text")
                return str(text) if text is not None else None

        return None
    except (KeyError, TypeError, AttributeError):
        return None


def extract_agent_state(response: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the Agent State from artifacts in the response.

    Args:
        response: The JSONRPC response dict.

    Returns:
        The agent state data dict, or None if not found.
    """
    try:
        result = response.get("result", {})
        artifacts = result.get("artifacts", [])

        for artifact in reversed(artifacts):
            if artifact.get("name") == "Agent State":
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("kind") == "data":
                        data = part.get("data")
                        if isinstance(data, dict):
                            return data

        return None
    except (KeyError, TypeError, AttributeError):
        return None


# Terminal task states that require a new task_id for next message
TERMINAL_TASK_STATES = {"completed", "failed", "canceled"}


def initialize_session_state():
    """Initialize the Streamlit session state."""
    if "context_id" not in st.session_state:
        st.session_state.context_id = generate_id()

    if "task_id" not in st.session_state:
        st.session_state.task_id = None  # None means first message (no task yet)

    if "task_status" not in st.session_state:
        st.session_state.task_status = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None

    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 1


def reset_conversation():
    """Reset the conversation to start fresh."""
    st.session_state.context_id = generate_id()
    st.session_state.task_id = None
    st.session_state.task_status = None
    st.session_state.messages = []
    st.session_state.agent_state = None
    st.session_state.request_counter = 1


def get_custom_css() -> str:
    """Return custom CSS that works well in both light and dark modes."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    /* ===== CSS Variables for Light/Dark Mode ===== */
    :root {
        /* Light mode defaults */
        --text-primary: #1a1a2e;
        --text-secondary: #4a4a6a;
        --text-muted: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --border-color: #e2e8f0;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-error: #ef4444;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
    }

    /* Dark mode overrides using Streamlit's theme detection */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --border-color: #475569;
            --accent-primary: #818cf8;
            --accent-secondary: #a78bfa;
            --accent-success: #34d399;
            --accent-warning: #fbbf24;
            --accent-error: #f87171;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.4);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.5);
        }
    }

    /* ===== Global Styles ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== Header ===== */
    .main-header {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.25rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        color: var(--text-muted);
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* ===== Status Badge ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-connected {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* ===== Section Titles ===== */
    .section-title {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ===== State Panel ===== */
    .state-section-title {
        color: var(--accent-primary);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ===== Empty State ===== */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: var(--text-muted);
    }

    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        opacity: 0.7;
    }

    .empty-state p {
        margin: 0.25rem 0;
    }

    .empty-state small {
        font-size: 0.85rem;
        opacity: 0.8;
    }

    /* ===== Message Content ===== */
    .message-content {
        color: var(--text-primary);
        line-height: 1.6;
    }

    .message-truncated {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.5;
    }

    /* ===== Code Styling ===== */
    code {
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.85em;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        background: var(--bg-tertiary);
        color: var(--accent-primary);
    }

    /* ===== Dividers ===== */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1rem 0;
    }

    /* ===== Fix Streamlit's default dark mode text visibility ===== */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: var(--text-primary);
    }

    /* Sidebar text fix */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }

    /* Fix metric labels */
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    /* Fix expander text */
    .streamlit-expanderHeader {
        color: var(--text-primary) !important;
    }

    .streamlit-expanderContent {
        color: var(--text-primary);
    }

    /* ===== Chat Messages ===== */
    [data-testid="stChatMessage"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatMessage"] p {
        color: var(--text-primary) !important;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* ===== Input ===== */
    .stChatInput > div {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        background: var(--bg-secondary) !important;
    }

    .stChatInput input {
        color: var(--text-primary) !important;
    }

    .stChatInput input::placeholder {
        color: var(--text-muted) !important;
    }

    /* ===== JSON display ===== */
    .stJson {
        background: var(--bg-tertiary) !important;
        border-radius: 8px;
    }

    /* ===== Code blocks ===== */
    .stCodeBlock {
        background: var(--bg-tertiary) !important;
    }

    .stCodeBlock code {
        color: var(--text-primary) !important;
    }
    </style>
    """


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Workers Agent Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🤖 Workers Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">A2A Protocol-compliant AI Assistant</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Session Info")
        st.markdown(
            '<div class="status-badge status-connected">● Connected</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("**Context ID:**")
        st.code(st.session_state.context_id[:8] + "...", language=None)

        st.markdown("**Task ID:**")
        if st.session_state.task_id:
            st.code(st.session_state.task_id[:8] + "...", language=None)
            if st.session_state.task_status:
                status_color = (
                    "🟢"
                    if st.session_state.task_status == "completed"
                    else "🟡"
                    if st.session_state.task_status == "input-required"
                    else "🔴"
                    if st.session_state.task_status in ("failed", "canceled")
                    else "⚪"
                )
                st.markdown(f"Status: {status_color} `{st.session_state.task_status}`")
        else:
            st.markdown("*None (new task)*")

        st.markdown("---")

        if st.button("🔄 New Conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Requests", st.session_state.request_counter - 1)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            This chat interface communicates with an
            [A2A Protocol](https://github.com/google/A2A) compliant agent
            service via JSONRPC.
            """
        )

    # Main layout with two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Chat")

        # Chat messages display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your message...", key="chat_input"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message immediately
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            # Determine task_id to use:
            # - If previous task status was terminal, start a new task (task_id=None)
            # - If previous task status was input-required, reuse the same task_id
            # - If no previous task, this is the first message (task_id=None)
            current_task_id = st.session_state.task_id
            if st.session_state.task_status in TERMINAL_TASK_STATES:
                current_task_id = None  # Start a new task

            # Send to agent and get response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = send_message(
                        prompt,
                        st.session_state.context_id,
                        task_id=current_task_id,
                    )
                    st.session_state.request_counter += 1

                    if response:
                        # Check for errors
                        if "error" in response:
                            error_msg = response["error"].get(
                                "message", "Unknown error"
                            )
                            st.error(f"Agent error: {error_msg}")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"❌ Error: {error_msg}",
                                }
                            )
                        else:
                            # Extract task_id from response and store it
                            result = response.get("result", {})
                            response_task_id = result.get("id")
                            if response_task_id:
                                st.session_state.task_id = response_task_id

                            # Extract and store task status
                            status = result.get("status", {})
                            state = status.get("state", "")
                            st.session_state.task_status = state

                            # Extract agent response
                            agent_text = extract_agent_response(response)
                            if agent_text:
                                st.markdown(agent_text)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": agent_text}
                                )

                            # Extract and store agent state
                            extracted_state = extract_agent_state(response)
                            if extracted_state:
                                st.session_state.agent_state = extracted_state

                            # Show status message
                            if state == "input-required":
                                st.info("💡 The agent is waiting for your input...")
                            elif state == "completed":
                                st.success("✅ Task completed!")
                            elif state in ("failed", "canceled"):
                                st.warning(f"⚠️ Task {state}")
                    else:
                        st.error("Failed to get response from agent")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "❌ Failed to communicate with the agent.",
                            }
                        )

            st.rerun()

    with col2:
        st.markdown("### 🔮 Agent State")

        agent_state: dict[str, Any] | None = st.session_state.agent_state
        if agent_state is not None:
            # Data section
            data = agent_state.get("data", {})
            st.markdown(
                '<div class="state-section-title">📦 Data Store</div>',
                unsafe_allow_html=True,
            )
            if data:
                st.json(data)
            else:
                st.markdown("*No data stored yet*")

            st.markdown("---")

            # Messages section (collapsible)
            messages = agent_state.get("messages", [])
            st.markdown(
                '<div class="state-section-title">📜 Message History</div>',
                unsafe_allow_html=True,
            )

            if messages:
                with st.expander(f"View {len(messages)} messages", expanded=False):
                    for i, msg in enumerate(messages):
                        msg_type = msg.get("type", "unknown")
                        content = msg.get("content", "")

                        # Color-code by message type
                        if msg_type == "human":
                            icon = "🧑‍💻"
                            label = "User"
                        elif msg_type == "ai":
                            icon = "🤖"
                            label = "Agent"
                        elif msg_type == "tool":
                            icon = "🔧"
                            label = msg.get("name", "Tool")
                        else:
                            icon = "📝"
                            label = msg_type.capitalize()

                        st.markdown(f"**{icon} {label}**")

                        # Show content or tool calls
                        if content:
                            truncated = content[:200]
                            if len(content) > 200:
                                truncated += "..."
                            st.markdown(
                                f'<div class="message-truncated">{truncated}</div>',
                                unsafe_allow_html=True,
                            )

                        # Show tool calls for AI messages
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                st.code(
                                    f"{tc.get('name', 'unknown')}({tc.get('args', {})})",
                                    language=None,
                                )

                        if i < len(messages) - 1:
                            st.markdown("---")
            else:
                st.markdown("*No messages yet*")

            st.markdown("---")

            # Raw state (for debugging)
            with st.expander("🔍 Raw State JSON", expanded=False):
                st.json(agent_state)
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🌙</div>
                    <p>No agent state yet.</p>
                    <p><small>Send a message to see the agent's internal state.</small></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
