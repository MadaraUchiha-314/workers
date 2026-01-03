from collections.abc import AsyncGenerator

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
from a2a.utils.message import new_agent_text_message

from workers.framework.agent.agent import Agent, Message, RequestContext


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

    async def ainvoke(self, context: RequestContext) -> Message:
        # Implement the logic to handle the invoke request
        # For example, you can use the context to determine the appropriate action to take
        # and then call the corresponding method or function to handle the request
        return new_agent_text_message(
            "Hello, this is the supervisor agent!",
            context_id=context.context_id,
            task_id=context.task_id,
        )

    async def astream(self, context: RequestContext) -> AsyncGenerator[Message, None]:
        # Implement the logic to handle the stream request
        # For example, you can use the context to determine the appropriate action to take
        # and then call the corresponding method or function to handle the request
        yield new_agent_text_message(
            "Hello, this is the supervisor agent!",
            context_id=context.context_id,
            task_id=context.task_id,
        )

    async def acancel(self, context: RequestContext) -> None:
        # Implement the logic to handle the cancel request
        # For example, you can use the context to determine the appropriate action to take
        # and then call the corresponding method or function to handle the request
        pass
