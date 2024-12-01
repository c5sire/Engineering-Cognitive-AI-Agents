"""Memory Coordinator agent for managing memory operations."""

from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.memory.embeddings import (
  EmbeddingStore,
)
from winston.core.memory.storage import (
  KnowledgeStorage,
)
from winston.core.messages import Response
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class StoreKnowledgeRequest(BaseModel):
  """Parameters for knowledge storage."""

  content: str = Field(
    description="Content to be stored"
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
    description="Maximum number of results to return",
  )


class MemoryCoordinator(BaseAgent):
  """Coordinates memory operations between specialists."""

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

    self.storage = KnowledgeStorage(storage_path)
    self.embeddings = EmbeddingStore(embedding_path)

    # Register memory tools
    self.system.register_tool(
      Tool(
        name="store_knowledge",
        description="Store important information in long-term memory",
        handler=self._handle_store_knowledge,
        input_model=StoreKnowledgeRequest,
        output_model=Response,
      )
    )

    self.system.register_tool(
      Tool(
        name="retrieve_knowledge",
        description="Find relevant knowledge from memory",
        handler=self._handle_retrieve_knowledge,
        input_model=RetrieveKnowledgeRequest,
        output_model=Response,
      )
    )

    # Grant self access to tools
    self.system.grant_tool_access(
      self.id,
      ["store_knowledge", "retrieve_knowledge"],
    )

  async def _handle_store_knowledge(
    self,
    request: StoreKnowledgeRequest,
  ) -> Response:
    """Handle knowledge storage requests."""
    # Store in knowledge base
    knowledge_id = await self.storage.store(
      content=request.content,
      context={
        "context": request.context,
        "importance": request.importance,
      },
    )

    # Load stored knowledge
    knowledge = await self.storage.load(knowledge_id)

    # Add embedding
    await self.embeddings.add_embedding(knowledge)

    return Response(
      content=f"Stored knowledge: {request.content[:100]}...",
      metadata={
        "knowledge_id": knowledge_id,
        "formatted_response": (
          f"I've stored that information about {request.context}. "
          f"I'll remember it's important because {request.importance}"
        ),
      },
    )

  async def _handle_retrieve_knowledge(
    self,
    request: RetrieveKnowledgeRequest,
  ) -> Response:
    """Handle knowledge retrieval requests."""
    print(f"\nSearching for: {request.query}")

    # Find similar knowledge
    matches = await self.embeddings.find_similar(
      query=request.query, limit=request.max_results
    )
    print(f"Found {len(matches)} matches")

    # Load full knowledge entries
    results = []
    for match in matches:
      print(f"\nMatch ID: {match.id}")
      print(f"Score: {match.score}")
      knowledge = await self.storage.load(match.id)
      print(f"Content: {knowledge.content}")
      results.append(
        f"- {knowledge.content} "
        f"(relevance: {match.score:.2f})"
      )

    if not results:
      return Response(
        content="No relevant knowledge found",
        metadata={
          "formatted_response": (
            "I don't have any relevant information "
            f"about {request.query} in that context."
          )
        },
      )

    response_text = "\n".join(results)
    return Response(
      content=response_text,
      metadata={
        "formatted_response": (
          f"Here's what I know about {request.query} "
          f"in the context of {request.context}:\n"
          + response_text
        )
      },
    )
