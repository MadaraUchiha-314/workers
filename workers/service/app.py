import logging
import sys

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from fastapi import FastAPI

from workers.framework.a2a.executor import A2AExecutor
from workers.framework.agent.agent import Agent
from workers.service.agents.supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_app() -> FastAPI:
    configure_logging()
    logger.info("Creating A2A REST FastAPI application")
    # Initialize main FastAPI app
    main_app = FastAPI()
    # Initialize agents
    agents: list[Agent] = [
        Supervisor(rpc_url="/"),
    ]
    for agent in agents:
        # Create an instance of A2ARESTFastAPIApplication
        # Build the FastAPI application using the A2ARESTFastAPIApplication
        # Note: http_handler should be a RequestHandler, NOT a RESTHandler
        # The RESTAdapter internally wraps it in a RESTHandler
        sub_app = A2AFastAPIApplication(
            agent_card=agent.agent_card,
            http_handler=DefaultRequestHandler(
                agent_executor=A2AExecutor(
                    agent=agent,
                ),
                task_store=InMemoryTaskStore(),
            ),
        ).build(
            rpc_url=agent.agent_card.url,
        )
        # Mount the sub-app under the agent's ID
        main_app.mount("/" + agent.id, sub_app)
    return main_app
