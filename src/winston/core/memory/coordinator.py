import json
from typing import AsyncIterator

from loguru import logger

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.episode_analyst import (
  EpisodeAnalyst,
)
from winston.core.memory.semantic_memory import (
  SemanticMemorySpecialist,
)
from winston.core.memory.working_memory import (
  WorkingMemorySpecialist,
)
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.workspace import WorkspaceManager


class MemoryCoordinator(BaseAgent):
  """Orchestrates memory operations between specialist agents."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)
    logger.info(
      "MemoryCoordinator initialized with system, config, and paths."
    )

    # Initialize specialist agents with their own configs
    self.episode_analyst = EpisodeAnalyst(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "episode_analyst.yaml"
      ),
      paths,
    )

    self.semantic_memory = SemanticMemorySpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "semantic_memory.yaml"
      ),
      paths,
    )

    self.working_memory = WorkingMemorySpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "working_memory.yaml"
      ),
      paths,
    )

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Orchestrate memory operations sequence."""
    logger.debug(f"Processing message: {message}")

    # Validate shared workspace requirement
    if "shared_workspace" not in message.metadata:
      logger.error(
        "Shared workspace not found in message metadata."
      )
      raise ValueError(
        "Shared workspace required for memory operations"
      )

    logger.trace(
      "Evaluating context shift with episode analyst."
    )
    # Let episode analyst evaluate the context shift
    episode_analysis = {}
    async for response in self.episode_analyst.process(
      message
    ):
      if response.metadata.get("streaming"):
        continue
      episode_analysis = json.loads(response.content)

    logger.info(
      f"Episode analysis completed: {episode_analysis}"
    )

    logger.trace(
      "Handling knowledge operations with semantic memory specialist."
    )

    # Combine episode analysis metadata with message metadata
    logger.trace(
      f"Original message metadata: {message.metadata}"
    )
    logger.trace(
      f"Episode analysis metadata: {episode_analysis['metadata']}"
    )
    message.metadata.update(
      episode_analysis["metadata"]
    )
    logger.trace(
      f"Combined metadata: {message.metadata}"
    )

    # Let semantic memory specialist handle knowledge operations
    semantic_results = {}
    async for response in self.semantic_memory.process(
      message
    ):
      if response.metadata.get("streaming"):
        continue
      logger.trace(
        f"Raw semantic results obtained: {response}"
      )
      semantic_results = json.loads(response.content)

    logger.info(
      f"Semantic results obtained: {semantic_results}"
    )

    # Combine semantic results metadata with message metadata
    message.metadata.update(
      semantic_results.get("metadata", {})
    )

    workspace_manager = WorkspaceManager()
    working_memory_results = {}
    async for response in self.working_memory.process(
      message
    ):
      if response.metadata.get("streaming"):
        yield response
        continue
      working_memory_results = json.loads(
        response.content
      )

    logger.debug(
      f"Response data parsed successfully: {working_memory_results}"
    )

    # Update the actual workspace file
    content: str = working_memory_results["content"]
    logger.info("Saving updated content to workspace.")
    workspace_manager.save_workspace(
      message.metadata["shared_workspace"],
      working_memory_results["content"]
      .strip("```markdown\n")
      .strip("```"),
    )

    yield Response(
      content=content,
      metadata=message.metadata,
    )
