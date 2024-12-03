"""Memory Coordinator: Central orchestrator for Winston's cognitive memory system.

The Memory Coordinator implements a Society of Mind approach to memory management,
coordinating specialized agents that handle different aspects of memory operations.
Rather than treating memory as simple storage, this system mirrors human cognitive
processes by maintaining working context while building long-term knowledge.

Architecture Overview:
```mermaid
graph TD
    MC[Memory Coordinator] -->|Request Analysis| EA[Episode Analyst]
    EA -->|Episode Status| MC
    MC -->|Request Operations| SMC[Semantic Memory Coordinator]
    SMC -->|Retrieval| RS[Retrieval Specialist]
    SMC -->|Storage| SS[Storage Specialist]
    RS -->|Context| SMC
    SS -->|Updates| SMC
    SMC -->|Retrieved Context + New Facts| MC
    MC -->|Update Request| WMS[Working Memory Specialist]
    WMS -->|Updated State| MC
```

Design Philosophy:
The memory system addresses a fundamental challenge in LLM-based cognitive
architectures: while language models provide sophisticated reasoning, they cannot
inherently learn from interactions or maintain context over time. The Memory
Coordinator bridges this gap by:

1. Managing Working Memory
   - Maintains immediate cognitive context
   - Coordinates shared understanding between agents
   - Preserves relevant context across interactions

2. Building Long-term Knowledge
   - Detects and processes new information
   - Maintains semantic connections between facts
   - Updates existing knowledge when understanding changes

3. Processing Experiences
   - Identifies cognitive episode boundaries
   - Extracts key facts and relationships
   - Compresses experiences into semantic knowledge

Example Flow:
When Winston learns "I usually drink coffee in the morning, like my father used to",
the system:
1. Episode Analyst determines this reveals preferences and relationships
2. Semantic Memory retrieves any related knowledge about routines/preferences
3. Storage Specialist captures both the preference and family connection
4. Working Memory Specialist updates current context while maintaining relationships

Key Architectural Principles:
- Coordinators handle flow control (pure Python)
- Specialists make cognitive decisions (LLM-based)
- Tools execute concrete actions
- Clear separation of concerns throughout

This design enables Winston to build and maintain a rich web of knowledge while
keeping each component focused and maintainable. The coordinator ensures these
pieces work together seamlessly, providing other agents with a simple interface
to sophisticated memory operations.
"""

import json
from typing import AsyncIterator

from loguru import logger

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.episode_analyst import (
  EpisodeAnalyst,
  EpisodeBoundaryResult,
)
from winston.core.memory.semantic.coordinator import (
  SemanticMemoryCoordinator,
)
from winston.core.memory.semantic.retrieval import (
  RetrieveKnowledgeResult,
)
from winston.core.memory.semantic.storage import (
  StoreKnowledgeResult,
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
        / "memory"
        / "episode_analyst.yaml"
      ),
      paths,
    )

    self.semantic_memory = SemanticMemoryCoordinator(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "memory"
        / "semantic"
        / "coordinator.yaml"
      ),
      paths,
    )

    self.working_memory = WorkingMemorySpecialist(
      system,
      AgentConfig.from_yaml(
        paths.system_agents_config
        / "memory"
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
        yield response
        continue
      logger.trace(
        f"Raw episode analysis obtained: {response}"
      )
      response_content = json.loads(response.content)
      response_metadata = response_content.get(
        "metadata"
      )
      if not response_metadata:
        logger.error("Response metadata not found")
        raise ValueError("Response metadata required")
      episode_analysis = (
        EpisodeBoundaryResult.model_validate(
          response_metadata
        )
      )

    logger.info(
      f"Episode analysis completed: {episode_analysis}"
    )

    logger.trace(
      "Handling knowledge operations with semantic memory specialist."
    )

    # Let semantic memory coordinator handle knowledge operations
    retrieval_result = None
    storage_result = None
    async for response in self.semantic_memory.process(
      message
    ):
      if response.metadata.get("streaming"):
        yield response
        continue
      logger.trace(
        f"Raw semantic results obtained: {response}"
      )

      response_type = response.metadata.get("type")
      if not response_type:
        logger.error(
          "Response type not found in metadata"
        )
        raise ValueError("Response type required")

      if (
        response_type
        == RetrieveKnowledgeResult.__name__
      ):
        logger.trace(
          f"Retrieval result obtained: {response}"
        )
        retrieval_result = (
          RetrieveKnowledgeResult.model_validate_json(
            response.content
          )
        )
      elif (
        response_type == StoreKnowledgeResult.__name__
      ):
        logger.trace(
          f"Storage result obtained: {response}"
        )
        storage_result = (
          StoreKnowledgeResult.model_validate_json(
            response.content
          )
        )
      else:
        logger.error(
          f"Unknown response type: {response_type}"
        )
        raise ValueError(
          f"Unknown response type: {response_type}"
        )

    # Finally, let working memory specialist update workspace
    workspace_manager = WorkspaceManager()

    working_memory_results = {}
    async for response in self.working_memory.process(
      message
    ):
      if response.metadata.get("streaming"):
        yield response
        continue
      try:
        working_memory_results = json.loads(
          response.content
        )
      except json.JSONDecodeError as e:
        logger.error(
          f"Failed to decode working memory response: {e}"
        )
        raise

    logger.debug(
      f"Response data parsed successfully: {working_memory_results}"
    )

    # Update the actual workspace file
    content: str = working_memory_results["content"]
    logger.info("Saving updated content to workspace.")
    try:
      workspace_manager.save_workspace(
        message.metadata["shared_workspace"],
        working_memory_results["content"]
        .strip("```markdown\n")
        .strip("```"),
      )
    except Exception as e:
      logger.error(
        f"Failed to save workspace content: {e}"
      )
      raise

    yield Response(
      content=content,
      metadata=message.metadata,
    )
