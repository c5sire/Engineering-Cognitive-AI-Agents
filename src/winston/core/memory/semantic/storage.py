"""Storage specialist for semantic memory."""

import json

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


class StoreKnowledgeRequest(BaseModel):
  """Parameters for knowledge storage."""

  content: str = Field(
    description="The core information to store"
  )
  semantic_metadata: str = Field(
    description="JSON-encoded list of key/value pairs for embedding metadata"
  )


class StoreKnowledgeResponse(BaseModel):
  """Result of knowledge storage operation."""

  id: str = Field(description="ID of stored knowledge")
  content: str = Field(description="Stored content")
  metadata: dict[str, str] = Field(
    description="Associated metadata"
  )


class StorageSpecialist(BaseAgent):
  """Specialist for evaluating and storing new knowledge."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    """Initialize storage specialist."""
    super().__init__(system, config, paths)
    logger.debug(
      "Initializing StorageSpecialist with system, config, and paths."
    )

    # Initialize storage components
    storage_path = paths.workspaces / "knowledge"
    embedding_path = paths.workspaces / "embeddings"
    self._storage = KnowledgeStorage(storage_path)
    self._embeddings = EmbeddingStore(embedding_path)
    logger.debug(
      f"Storage path: {storage_path}, Embedding path: {embedding_path}"
    )

    # Register storage tool
    self.system.register_tool(
      Tool(
        name="store_knowledge",
        description="Store important information in long-term memory",
        handler=self._handle_store_knowledge,
        input_model=StoreKnowledgeRequest,
        output_model=StoreKnowledgeResponse,
      )
    )
    logger.debug("Registered store_knowledge tool.")

    # Grant self access
    self.system.grant_tool_access(
      self.id, ["store_knowledge"]
    )
    logger.debug("Granted tool access to self.")

  async def _handle_store_knowledge(
    self,
    request: StoreKnowledgeRequest,
  ) -> StoreKnowledgeResponse:
    """Handle knowledge storage requests."""
    logger.debug(
      f"Handling store knowledge request: {request}"
    )
    try:
      # Deserialize metadata from JSON string
      metadata = json.loads(request.semantic_metadata)
      logger.debug(
        f"Deserialized metadata: {metadata}"
      )

      # Convert list of k/v pairs to dict
      metadata_dict = {
        item["key"]: item["value"] for item in metadata
      }
      logger.debug(
        f"Converted metadata to dict: {metadata_dict}"
      )

      # Store in knowledge base
      knowledge_id = await self._storage.store(
        content=request.content,
        context=metadata_dict,
      )
      logger.debug(
        f"Stored knowledge with ID: {knowledge_id}"
      )

      # Load stored knowledge
      knowledge = await self._storage.load(
        knowledge_id
      )
      logger.debug(f"Loaded knowledge: {knowledge}")

      # Add embedding
      await self._embeddings.add_embedding(knowledge)
      logger.debug(
        "Added embedding for the stored knowledge."
      )

      return StoreKnowledgeResponse(
        id=knowledge_id,
        content=request.content,
        metadata=metadata_dict,
      )
    except Exception as e:
      logger.error(f"Storage error: {e}")
      raise
