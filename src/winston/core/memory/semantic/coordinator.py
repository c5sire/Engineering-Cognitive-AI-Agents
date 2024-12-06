"""Semantic Memory Coordinator: Manages long-term knowledge storage and retrieval.

The Semantic Memory Coordinator orchestrates the storage, retrieval, and maintenance
of Winston's long-term knowledge. Rather than using explicit relationship graphs,
this system achieves connected knowledge through semantic embeddings - allowing
natural associations to emerge through meaning rather than rigid structure.

Architecture Overview:
```mermaid
graph TD
    SMC[Semantic Memory Coordinator] -->|Query| RS[Retrieval Specialist]
    SMC -->|Store/Update| SS[Storage Specialist]

    RS -->|Search| ES[Embedding Store]
    RS -->|Load| KS[Knowledge Store]

    SS -->|Check Existing| ES
    SS -->|Store New| KS
    SS -->|Update| ES

    subgraph "Storage Components"
        ES[Embedding Store<br/>ChromaDB]
        KS[Knowledge Store<br/>File-based]
    end

    RS -->|"Relevant Knowledge"| SMC
    SS -->|"Storage Results"| SMC
```

Design Philosophy:
The semantic memory system addresses several key challenges in cognitive architectures:

1. Knowledge Connections
   - Uses embedding space for natural semantic relationships
   - Avoids complexity of explicit graph management
   - Enables fuzzy matching and graceful degradation
   - Mirrors human associative memory patterns

2. Knowledge Evolution
   - Handles updates while preserving connections
   - Maintains temporal progression of understanding
   - Resolves conflicts between old and new information
   - Tracks context and metadata for knowledge pieces

3. Retrieval Patterns
   - Finds both exact and semantically related matches
   - Uses metadata for filtering when needed
   - Ranks results by semantic relevance
   - Returns multiple relevance levels for context

Example Flow:
When processing "I've switched to tea", the system:
1. Retrieval Specialist finds existing beverage preferences
2. Storage Specialist identifies this as a temporal change
3. Updates knowledge while preserving morning routine context
4. Maintains semantic connections to family patterns
5. Records change metadata for future reference

Key Design Decisions:
- Focus on semantic similarity over explicit relationships
- Simple metadata over complex categorization
- Preserve knowledge history during updates
- Handle conflicts through specialist reasoning
- Maintain context through embedding space

Implementation Notes:
- Uses ChromaDB for embedding storage/search
- File-based knowledge store for raw content
- Pydantic models for knowledge structure
- Async operations throughout
- Clear separation between storage and retrieval

This design enables sophisticated knowledge management while maintaining
architectural simplicity. The coordinator ensures proper sequencing of
operations while letting specialists handle cognitive decisions about
storage and retrieval.
"""

from typing import AsyncIterator

from loguru import logger
from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.semantic.retrieval import (
  KnowledgeItem,
  RetrievalSpecialist,
  RetrieveKnowledgeResult,
)
from winston.core.memory.semantic.storage import (
  KnowledgeActionType,
  StorageSpecialist,
  StoreKnowledgeResult,
)
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.steps import ProcessingStep
from winston.core.system import AgentSystem


class SemanticMemoryResult(BaseModel):
  """Consolidated result from semantic memory operations."""

  # From RetrieveKnowledgeResult
  content: str | None = Field(
    default=None,
    description="The retrieved knowledge content",
  )
  relevance: float | None = Field(
    default=None,
    description="Relevance score for retrieved items",
  )
  lower_relevance_results: (
    list[KnowledgeItem] | None
  ) = Field(
    default=None,
    description="List of lower relevance knowledge items",
  )

  # From StoreKnowledgeResult
  id: str | None = Field(
    default=None,
    description="ID of stored/updated knowledge",
  )
  action: KnowledgeActionType | None = Field(
    default=None, description="Type of action taken"
  )
  reason: str | None = Field(
    default=None,
    description="Explanation for the action taken",
  )


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
    self,
    message: Message,
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
    retrieval_result = None
    async with ProcessingStep(
      name="Semantic Retrieval agent",
      step_type="run",
    ) as step:
      async for (
        response
      ) in self.retrieval_specialist.process(message):
        if not response.metadata.get("streaming"):
          logger.debug(
            f"Retrieval response: {response}"
          )
          retrieval_result = RetrieveKnowledgeResult.model_validate_json(
            response.content
          )

          await step.show_response(
            Response(
              content=f"```json\n{retrieval_result.model_dump_json()}\n```",
              metadata=message.metadata,
            )
          )

    # 2. Let Storage Specialist analyze and handle storage needs
    storage_message = Message(
      content=message.content,
      metadata={
        **message.metadata,
        "content": message.content,
        "retrieved_content": retrieval_result.model_dump_json()
        if retrieval_result
        else None,
      },
    )

    logger.debug(
      f"Storage message created: {storage_message}"
    )

    storage_result = None
    async with ProcessingStep(
      name="Semantic Storage agent",
      step_type="run",
    ) as step:
      async for (
        response
      ) in self.storage_specialist.process(
        storage_message
      ):
        if not response.metadata.get("streaming"):
          logger.debug(f"Storage response: {response}")
          storage_result = (
            StoreKnowledgeResult.model_validate_json(
              response.content
            )
          )

          await step.show_response(
            Response(
              content=f"```json\n{storage_result.model_dump_json()}\n```",
              metadata=message.metadata,
            )
          )

    # 3. Yield the consolidated result
    semantic_result = SemanticMemoryResult(
      content=retrieval_result.content
      if retrieval_result
      else None,
      relevance=retrieval_result.relevance
      if retrieval_result
      else None,
      lower_relevance_results=(
        retrieval_result.lower_relevance_results
        if retrieval_result
        else None
      ),
      id=storage_result.id if storage_result else None,
      action=storage_result.action
      if storage_result
      else None,
      reason=storage_result.reason
      if storage_result
      else "No storage operation performed",
    )

    yield Response(
      content=semantic_result.model_dump_json(),
    )
