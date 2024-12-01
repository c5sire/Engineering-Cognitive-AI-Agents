from typing import AsyncIterator

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


class MemoryCoordinator(BaseAgent):
  """Orchestrates memory operations between specialist agents."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

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

    # Validate shared workspace requirement
    if "shared_workspace" not in message.metadata:
      raise ValueError(
        "Shared workspace required for memory operations"
      )

    # Let episode analyst evaluate the context shift
    async for response in self.episode_analyst.process(
      message
    ):
      episode_analysis = response
      break

    # Let semantic memory specialist handle knowledge operations
    async for response in self.semantic_memory.process(
      message
    ):
      semantic_results = response
      break

    # Let working memory specialist update workspace with all context
    workspace_message = Message(
      content=message.content,
      metadata={
        "shared_workspace": message.metadata[
          "shared_workspace"
        ],
        "episode_analysis": episode_analysis.content,
        "semantic_context": semantic_results.content,
      },
    )

    async for response in self.working_memory.process(
      workspace_message
    ):
      yield response
