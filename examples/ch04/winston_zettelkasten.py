"""Winston with enhanced Zettelkasten-style memory capabilities."""

import sys
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from winston.core.agent import AgentConfig, BaseAgent
from winston.core.memory.coordinator import (
  MemoryCoordinator,
)
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import Agent, System
from winston.ui.chainlit_app import AgentChat

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="TRACE", backtrace=True)


class ZettelkastenWinston(BaseAgent):
  """Winston with enhanced memory capabilities through core memory agents."""

  async def process(
    self, message: Message
  ) -> AsyncIterator[Response]:
    """Process message using memory coordinator."""
    print(
      f"Processing with memory system: {message.content}"
    )

    # Create message for memory coordinator
    coordinator_message = Message(
      content=message.content,
      metadata={
        **message.metadata,
        "shared_workspace": self.workspace_path,
      },
    )

    # Let memory coordinator handle the processing
    async for (
      response
    ) in self.system.invoke_conversation(
      "memory_coordinator",
      coordinator_message.content,
      context=coordinator_message.metadata,
    ):
      yield response


class ZettelkastenWinstonChat(AgentChat):
  """Chat interface for Zettelkasten Winston."""

  def __init__(self) -> None:
    # Set up paths with both application and system roots
    example_root = Path(__file__).parent
    project_root = (
      example_root.parent.parent
    )  # Navigate up to project root
    self.paths = AgentPaths(
      root=example_root,  # For application-specific config
      system_root=project_root,  # For system-wide config
    )
    super().__init__()

  def create_agent(self, system: System) -> Agent:
    """Create Zettelkasten Winston instance and register memory agents."""
    # Create and register memory coordinator
    coordinator_config = AgentConfig.from_yaml(
      self.paths.system_agents_config
      / "memory_coordinator.yaml"
    )
    system.register_agent(
      MemoryCoordinator(
        system=system,
        config=coordinator_config,
        paths=self.paths,
      )
    )

    # Create Winston agent
    config = AgentConfig.from_yaml(
      self.paths.config
      / "agents/winston_zettelkasten.yaml"
    )
    return ZettelkastenWinston(
      system=system,
      config=config,
      paths=self.paths,
    )


# Create the application
app = ZettelkastenWinstonChat()
