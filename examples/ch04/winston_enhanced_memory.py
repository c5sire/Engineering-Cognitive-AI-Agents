"""Winston with enhanced memory capabilities."""

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


class EnhancedMemoryWinston(BaseAgent):
  """Winston with enhanced memory capabilities."""

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Process message using memory coordinator.

    Parameters
    ----------
    message : Message
        The incoming message to process

    Yields
    ------
    Response
        Responses from processing the message
    """
    logger.info(
      f"Processing with memory system: {message.content}"
    )

    # First, check memory for relevant context
    context_message = Message(
      content=(
        "Before responding, please check your knowledge "
        f"store for anything relevant to: {message.content}"
      ),
      metadata={
        **message.metadata,
        "shared_workspace": self.workspace_path,
      },
    )

    # Let memory coordinator retrieve context
    context_responses = []
    async for (
      response
    ) in self.system.invoke_conversation(
      "memory_coordinator",
      context_message.content,
      context=context_message.metadata,
    ):
      if not response.metadata.get("streaming", False):
        context_responses.append(response.content)

    # Generate response using retrieved context
    context_str = "\n".join(context_responses)
    response_prompt = f"""
        Given this message:
        {message.content}

        And this relevant context from memory:
        {context_str}

        Generate a response that:
        1. Incorporates relevant remembered information
        2. Maintains consistency with past interactions
        3. Shows understanding of user context
        """

    # Generate and stream response
    async for (
      response
    ) in self.generate_streaming_response(
      Message(content=response_prompt)
    ):
      yield response

    # After responding, have memory coordinator evaluate for storage
    storage_message = Message(
      content=(
        "Please evaluate this interaction for important "
        f"information to store: {message.content}"
      ),
      metadata={
        **message.metadata,
        "shared_workspace": self.workspace_path,
      },
    )

    async for _ in self.system.invoke_conversation(
      "memory_coordinator",
      storage_message.content,
      context=storage_message.metadata,
    ):
      pass  # Let coordinator store silently


class EnhancedMemoryWinstonChat(AgentChat):
  """Chat interface for memory-enabled Winston."""

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
    """Create Winston instance with memory capabilities.

    Parameters
    ----------
    system : System
        The system instance

    Returns
    -------
    Agent
        The configured Winston agent
    """
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
      / "agents/winston_enhanced_memory.yaml"
    )
    return EnhancedMemoryWinston(
      system=system,
      config=config,
      paths=self.paths,
    )


# Create the application
app = EnhancedMemoryWinstonChat()
