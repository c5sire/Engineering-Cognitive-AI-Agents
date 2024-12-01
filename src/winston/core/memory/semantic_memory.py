"""Semantic memory specialist agent."""

from typing import Any

from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.embeddings import (
  EmbeddingStore,
)
from winston.core.memory.storage import (
  KnowledgeStorage,
)
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class StoreKnowledgeRequest(BaseModel):
  """Parameters for knowledge storage."""

  content: str = Field(
    description="The core information to store"
  )
  context: str = Field(
    description="Current context and relevance"
  )
  importance: str = Field(
    description="Why this knowledge is significant"
  )


class RetrieveKnowledgeRequest(BaseModel):
  """Parameters for knowledge retrieval."""

  query: str = Field(
    description="What knowledge to find"
  )
  context: str = Field(
    description="Current context for relevance"
  )
  max_results: int = Field(
    description="Maximum number of results"
  )


class KnowledgeItem(BaseModel):
  """Structured knowledge item."""

  content: str = Field(
    description="The retrieved/stored knowledge"
  )
  relevance: float | None = Field(
    default=None,
    description="Relevance score for retrieved items",
  )
  metadata: dict[str, Any] = Field(
    default_factory=dict,
    description="Associated metadata",
  )


class KnowledgeResponse(KnowledgeItem):
  """Structured knowledge response with additional lower relevance results."""

  lower_relevance_results: list[KnowledgeItem] = Field(
    default_factory=list,
    description="List of lower relevance knowledge items",
  )


class SemanticMemorySpecialist(BaseAgent):
  """Specialist agent for semantic memory operations."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    # Initialize storage components
    storage_path = paths.workspaces / "knowledge"
    embedding_path = paths.workspaces / "embeddings"
    self._storage = KnowledgeStorage(storage_path)
    self._embeddings = EmbeddingStore(embedding_path)

    # Register tools
    self.system.register_tool(
      Tool(
        name="store_knowledge",
        description="Store important information in long-term memory",
        handler=self._handle_store_knowledge,
        input_model=StoreKnowledgeRequest,
        output_model=KnowledgeResponse,
        format_response=self._format_storage_response,
      )
    )

    self.system.register_tool(
      Tool(
        name="retrieve_knowledge",
        description="Find relevant knowledge from memory",
        handler=self._handle_retrieve_knowledge,
        input_model=RetrieveKnowledgeRequest,
        output_model=KnowledgeResponse,
        format_response=self._format_retrieval_response,
      )
    )

    # Grant self access
    self.system.grant_tool_access(
      self.id,
      ["store_knowledge", "retrieve_knowledge"],
    )

  async def _handle_store_knowledge(
    self,
    request: StoreKnowledgeRequest,
  ) -> KnowledgeResponse:
    """Handle knowledge storage requests."""
    # Store in knowledge base
    knowledge_id = await self._storage.store(
      content=request.content,
      context={
        "context": request.context,
        "importance": request.importance,
      },
    )

    # Load stored knowledge
    knowledge = await self._storage.load(knowledge_id)

    # Add embedding
    await self._embeddings.add_embedding(knowledge)

    return KnowledgeResponse(
      content=request.content,
      metadata={
        "id": knowledge_id,
        "context": request.context,
        "importance": request.importance,
      },
    )

  async def _handle_retrieve_knowledge(
    self,
    request: RetrieveKnowledgeRequest,
  ) -> KnowledgeResponse:
    """Handle knowledge retrieval requests."""
    # Find similar knowledge
    matches = await self._embeddings.find_similar(
      query=request.query,
      limit=request.max_results,
    )

    # Load full knowledge entries
    lower_relevance_results: list[KnowledgeItem] = []
    best_result: KnowledgeItem | None = None
    for match in matches:
      knowledge = await self._storage.load(match.id)
      knowledge_item = KnowledgeItem(
        content=knowledge.content,
        relevance=match.score,
        metadata={
          "id": match.id,  # Include the ID
          **knowledge.context,  # Spread the rest of the context
        },
      )
      if (
        best_result is None
        or match.score > best_result.relevance
      ):
        best_result = knowledge_item
      else:
        lower_relevance_results.append(knowledge_item)

    # Format combined results
    if best_result is None:
      return KnowledgeResponse(
        content="No relevant knowledge found",
        relevance=None,
        metadata={"context": request.context},
        lower_relevance_results=[],
      )

    return KnowledgeResponse(
      content=best_result.content,
      relevance=best_result.relevance,
      metadata=best_result.metadata,
      lower_relevance_results=lower_relevance_results,
    )

  def _format_storage_response(
    self, response: KnowledgeResponse
  ) -> str:
    """Format storage result for user display."""
    return (
      f"I've stored that information about {response.metadata['context']}. "
      f"I'll remember it's important because {response.metadata['importance']}"
    )

  def _format_retrieval_response(
    self, response: KnowledgeResponse
  ) -> str:
    """Format retrieval results for user display."""
    if response.relevance is None:
      return "No relevant knowledge found"
    return f"Here's what I know about that:\n{response.content}"
