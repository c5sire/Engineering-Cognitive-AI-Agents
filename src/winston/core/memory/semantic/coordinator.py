"""Semantic memory agency implementation."""

from typing import AsyncIterator

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import System


class SemanticMemoryCoordinator(BaseAgent):
  """Coordinates semantic memory operations."""

  def __init__(
    self,
    system: System,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    # Initialize specialists
    self.retrieval_specialist = RetrievalSpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "semantic_retrieval.yaml"
      ),
      paths,
    )

    self.storage_specialist = StorageSpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "semantic_storage.yaml"
      ),
      paths,
    )

    self.integration_specialist = (
      IntegrationSpecialist(
        system,
        AgentConfig.from_yaml(
          paths.system_agents_config
          / "semantic_integration.yaml"
        ),
        paths,
      )
    )

  async def process(
    self, message: Message
  ) -> AsyncIterator[Response]:
    """Coordinate semantic memory operations."""
    # Determine operation type from message
    if self._is_retrieval_request(message):
      async for (
        response
      ) in self.retrieval_specialist.process(message):
        yield response

    elif self._is_storage_request(message):
      async for (
        response
      ) in self.storage_specialist.process(message):
        yield response

    # After storage/retrieval, always run integration
    async for (
      response
    ) in self.integration_specialist.process(message):
      yield response

  def _is_retrieval_request(
    self, message: Message
  ) -> bool:
    """Check if message requires knowledge retrieval."""
    return any(
      trigger in message.content.lower()
      for trigger in [
        "what do you know",
        "remember",
        "recall",
        "find",
      ]
    )

  def _is_storage_request(
    self, message: Message
  ) -> bool:
    """Check if message contains information to store."""
    return (
      "store"
      in message.metadata.get("operation", "").lower()
    )
