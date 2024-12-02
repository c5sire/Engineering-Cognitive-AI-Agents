"""Retrieval specialist for semantic memory."""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from winston.core.agent import AgentConfig, BaseAgent
from winston.core.memory.embeddings import (
  EmbeddingStore,
)
from winston.core.memory.storage import (
  KnowledgeStorage,
)
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class RetrieveKnowledgeRequest(BaseModel):
  """Parameters for knowledge retrieval."""

  query: str = Field(
    description="The search query to find relevant knowledge"
  )
  max_results: int = Field(
    description="Maximum number of results to return"
  )


class KnowledgeItem(BaseModel):
  """Structured knowledge item."""

  content: str | None = Field(
    default=None,
    description="The retrieved/stored knowledge",
  )
  relevance: float | None = Field(
    default=None,
    description="Relevance score for retrieved items",
  )
  metadata: dict[str, Any] = Field(
    default_factory=dict,
    description="Associated metadata",
  )


class RetrieveKnowledgeResponse(KnowledgeItem):
  """Structured knowledge response with additional lower relevance results."""

  lower_relevance_results: list[KnowledgeItem] = Field(
    default_factory=list,
    description="List of lower relevance knowledge items",
  )


class RetrievalSpecialist(BaseAgent):
  """Specialist for formulating and executing knowledge retrieval."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)
    storage_path = paths.workspaces / "knowledge"
    embedding_path = paths.workspaces / "embeddings"
    self._storage = KnowledgeStorage(storage_path)
    self._embeddings = EmbeddingStore(embedding_path)

    # Register the retrieval tool
    self.system.register_tool(
      Tool(
        name="retrieve_knowledge",
        description="Find relevant knowledge using semantic search",
        handler=self._handle_retrieve_knowledge,
        input_model=RetrieveKnowledgeRequest,
        output_model=RetrieveKnowledgeResponse,
      )
    )

    # Grant self access
    self.system.grant_tool_access(
      self.id, ["retrieve_knowledge"]
    )

  async def _handle_retrieve_knowledge(
    self,
    request: RetrieveKnowledgeRequest,
  ) -> RetrieveKnowledgeResponse:
    """Handle knowledge retrieval requests."""
    logger.debug(
      f"Handling retrieval request: {request}"
    )

    matches = await self._embeddings.find_similar(
      query=request.query,
      limit=request.max_results,
    )
    logger.debug(f"Found {len(matches)} matches")

    if not matches:
      return RetrieveKnowledgeResponse(
        content=None,
        relevance=None,
        metadata={},
      )

    # Process matches into KnowledgeItems
    lower_relevance_results: list[KnowledgeItem] = []
    best_match: KnowledgeItem | None = None

    for match in matches:
      knowledge = await self._storage.load(match.id)
      item = KnowledgeItem(
        content=knowledge.content,
        relevance=match.score,
        metadata={"id": match.id, **knowledge.context},
      )

      if best_match is None:
        best_match = item
      else:
        lower_relevance_results.append(item)

    return RetrieveKnowledgeResponse(
      content=best_match.content
      if best_match
      else None,
      relevance=best_match.relevance
      if best_match
      else None,
      metadata=best_match.metadata
      if best_match
      else {},
      lower_relevance_results=lower_relevance_results,
    )
