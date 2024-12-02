"""Semantic memory agency implementation."""

from typing import AsyncIterator

from loguru import logger

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.semantic.retrieval import (
  RetrievalSpecialist,
)
from winston.core.memory.semantic.storage import (
  StorageSpecialist,
)
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


class SemanticMemoryCoordinator(BaseAgent):
  """Coordinates semantic memory operations."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    # Initialize specialists
    logger.info(
      "Initializing Retrieval and Storage Specialists."
    )
    self.retrieval_specialist = RetrievalSpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "memory"
        / "semantic"
        / "retrieval.yaml"
      ),
      paths,
    )

    self.storage_specialist = StorageSpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "memory"
        / "semantic"
        / "storage.yaml"
      ),
      paths,
    )

  async def process(
    self, message: Message
  ) -> AsyncIterator[Response]:
    """
    Process semantic memory operations.

    Parameters
    ----------
    message : Message
        The observation to process

    Yields
    ------
    Response
        Retrieved context and any storage results
    """
    logger.debug(
      f"Processing message: {message.content}"
    )

    # 1. Find relevant knowledge
    retrieval_message = Message(
      content=message.content,
      metadata=message.metadata,  # Pass through any filters/context
    )

    logger.trace(
      f"Retrieval message created: {retrieval_message}"
    )

    retrieved_content = None
    async for (
      response
    ) in self.retrieval_specialist.process(
      retrieval_message
    ):
      if not response.metadata.get("streaming"):
        retrieved_content = response.content
        logger.debug(
          f"Retrieved content: {retrieved_content}"
        )
        # Pass retrieved context back to Memory Coordinator
        yield Response(
          content=retrieved_content,
          metadata={"type": "retrieved_context"},
        )

    # 2. Let Storage Specialist analyze and handle storage needs
    storage_message = Message(
      content=message.content,
      metadata={
        "retrieved_content": retrieved_content
      },
    )

    logger.debug(
      f"Storage message created: {storage_message}"
    )

    async for (
      response
    ) in self.storage_specialist.process(
      storage_message
    ):
      if not response.metadata.get("streaming"):
        # Pass storage results back to Memory Coordinator
        logger.debug(
          f"Received response from storage: {response.content}"
        )
        yield Response(
          content=response.content,
          metadata={"type": "storage_result"},
        )
