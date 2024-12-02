"""Storage specialist for semantic memory."""

import json
from enum import StrEnum, auto

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


class KnowledgeActionType(StrEnum):
  """Types of knowledge storage actions."""

  NO_STORAGE_NEEDED = auto()
  CREATED = auto()
  TEMPORAL_CHANGE = auto()
  CORRECTION = auto()
  CONFLICT_RESOLUTION = auto()


class StoreKnowledgeResponse(BaseModel):
  """Result of knowledge storage/update operation."""

  id: str | None = Field(
    default=None,
    description="ID of stored/updated knowledge",
  )
  content: str | None = Field(
    default=None, description="The knowledge content"
  )
  metadata: dict[str, str] | None = Field(
    default=None, description="Associated metadata"
  )
  action: KnowledgeActionType = Field(
    description="Type of action taken"
  )
  reason: str = Field(
    description="Explanation for the action taken"
  )


class NoStorageNeededRequest(BaseModel):
  """Request to indicate no storage is needed."""

  reason: str = Field(
    description="Explanation for why no storage is needed"
  )


class StoreKnowledgeRequest(BaseModel):
  """Parameters for storing new knowledge."""

  content: str = Field(description="Content to store")
  semantic_metadata: str = Field(
    description="JSON-encoded dictionary of key:value pairs for filtering"
  )
  action: KnowledgeActionType = Field(
    default=KnowledgeActionType.CREATED,
    description="Type of storage action",
  )
  reason: str = Field(
    description="Explanation for why this needs to be stored"
  )


class UpdateKnowledgeRequest(BaseModel):
  """Parameters for updating existing knowledge."""

  knowledge_id: str = Field(
    description="ID of knowledge to update"
  )
  new_content: str = Field(
    description="Updated content"
  )
  semantic_metadata: str = Field(
    description="JSON-encoded list of key/value pairs for filtering"
  )
  preserve_history: bool = Field(
    description="Whether to preserve old version"
  )
  action: KnowledgeActionType = Field(
    description="Type of update being performed",
    # Must be one of the update types:
    # TEMPORAL_CHANGE, CORRECTION, CONFLICT_RESOLUTION
  )
  reason: str = Field(
    description="Explanation for why this update is needed"
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

    storage_tools = [
      Tool(
        name="no_storage_needed",
        description="Use this tool when there is no need to store new knowledge",
        handler=self._handle_no_storage_needed,
        input_model=NoStorageNeededRequest,
        output_model=StoreKnowledgeResponse,
      ),
      Tool(
        name="store_knowledge",
        description="Store new knowledge with metadata",
        handler=self._handle_store_knowledge,
        input_model=StoreKnowledgeRequest,
        output_model=StoreKnowledgeResponse,
      ),
      Tool(
        name="update_knowledge",
        description="Update existing knowledge",
        handler=self._handle_update_knowledge,
        input_model=UpdateKnowledgeRequest,
        output_model=StoreKnowledgeResponse,  # Same response model for both
      ),
    ]
    for tool in storage_tools:
      self.system.register_tool(tool)
      logger.debug(f"Registered {tool.name} tool.")

    # Grant self access
    self.system.grant_tool_access(
      self.id, [x.name for x in storage_tools]
    )
    logger.debug("Granted tool access to self.")

  async def _handle_no_storage_needed(
    self,
    request: NoStorageNeededRequest,
  ) -> StoreKnowledgeResponse:
    """Handle no storage needed requests."""
    logger.debug(
      f"Handling no storage needed request: {request}"
    )
    return StoreKnowledgeResponse(
      action=KnowledgeActionType.NO_STORAGE_NEEDED,
      reason=request.reason,
    )

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
      if not isinstance(metadata, dict):
        raise ValueError(
          "Expected metadata to be a dictionary."
        )

      # Store in knowledge base
      knowledge_id = await self._storage.store(
        content=request.content,
        context=metadata,
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
        metadata=metadata,
        action=request.action,
        reason=request.reason,
      )
    except Exception as e:
      logger.error(f"Storage error: {e}")
      raise

  async def _handle_update_knowledge(
    self,
    request: UpdateKnowledgeRequest,
  ) -> StoreKnowledgeResponse:
    """Handle knowledge update requests."""
    logger.debug(
      f"Handling update knowledge request: {request}"
    )
    try:
      if request.preserve_history:
        # Load and archive existing version
        existing = await self._storage.load(
          request.knowledge_id
        )
        archived_id = await self._storage.store(
          content=existing.content,
          context={
            **existing.context,
            "archived": "true",
          },
        )
        logger.debug(
          f"Archived existing version as: {archived_id}"
        )

      # Deserialize metadata
      metadata = json.loads(request.semantic_metadata)
      logger.debug(
        f"Deserialized metadata: {metadata}"
      )
      if not isinstance(metadata, dict):
        raise ValueError(
          "Expected metadata to be a dictionary."
        )

      # Update the knowledge
      updated = await self._storage.update(
        request.knowledge_id,
        content=request.new_content,
        context=metadata,
      )
      logger.debug(f"Updated knowledge: {updated}")

      # Update embedding
      await self._embeddings.update_embedding(updated)
      logger.debug("Updated embedding")

      return StoreKnowledgeResponse(
        id=request.knowledge_id,
        content=request.new_content,
        metadata=metadata,
        action=request.action,
        reason=request.reason,
      )
    except Exception as e:
      logger.error(f"Update error: {e}")
      raise
