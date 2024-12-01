"""Basic knowledge storage implementation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class Knowledge(BaseModel):
  """Single knowledge entry with metadata."""

  id: str = Field(description="Unique identifier")
  content: str = Field(description="Knowledge content")
  context: dict[str, Any] = Field(
    default_factory=dict,
    description="Associated context/metadata",
  )
  created_at: datetime = Field(
    description="Creation timestamp"
  )
  updated_at: datetime = Field(
    description="Last update timestamp"
  )


class KnowledgeStorage:
  """Simple file-based knowledge storage."""

  def __init__(self, storage_path: Path):
    """Initialize storage in specified directory."""
    self.storage_path = storage_path
    self.storage_path.mkdir(
      parents=True, exist_ok=True
    )
    logger.info(
      f"Knowledge storage initialized at {self.storage_path}"
    )

  def _generate_id(self) -> str:
    """Generate unique knowledge ID."""
    knowledge_id = str(uuid.uuid4())
    logger.debug(
      f"Generated new knowledge ID: {knowledge_id}"
    )
    return knowledge_id

  def _get_path(self, knowledge_id: str) -> Path:
    """Get file path for knowledge ID."""
    return self.storage_path / f"{knowledge_id}.json"

  async def store(
    self, content: str, context: dict[str, Any]
  ) -> str:
    """Store new knowledge entry."""
    knowledge_id = self._generate_id()
    knowledge = Knowledge(
      id=knowledge_id,
      content=content,
      context=context,
      created_at=datetime.now(),
      updated_at=datetime.now(),
    )
    logger.info(
      f"Storing knowledge entry with ID: {knowledge_id}"
    )

    # Write to file
    path = self._get_path(knowledge_id)
    try:
      path.write_text(knowledge.model_dump_json())
      logger.info(
        f"Knowledge entry stored successfully at {path}"
      )
    except Exception as e:
      logger.error(
        f"Failed to store knowledge entry: {e}"
      )
      raise

    return knowledge_id

  async def load(self, knowledge_id: str) -> Knowledge:
    """Load knowledge by ID."""
    logger.info(
      f"Loading knowledge entry with ID: {knowledge_id}"
    )
    path = self._get_path(knowledge_id)
    if not path.exists():
      logger.warning(
        f"Knowledge {knowledge_id} not found"
      )
      raise FileNotFoundError(
        f"Knowledge {knowledge_id} not found"
      )

    data = json.loads(path.read_text())
    logger.debug(f"Knowledge entry loaded: {data}")
    return Knowledge.model_validate(data)

  async def update(
    self,
    knowledge_id: str,
    content: str | None = None,
    context: dict[str, Any] | None = None,
  ) -> Knowledge:
    """Update existing knowledge."""
    logger.info(
      f"Updating knowledge entry with ID: {knowledge_id}"
    )
    knowledge = await self.load(knowledge_id)

    # Update fields
    if content is not None:
      logger.debug(
        f"Updating content for ID {knowledge_id}"
      )
      knowledge.content = content
    if context is not None:
      logger.debug(
        f"Updating context for ID {knowledge_id}"
      )
      knowledge.context.update(context)
    knowledge.updated_at = datetime.now()

    # Save changes
    path = self._get_path(knowledge_id)
    try:
      path.write_text(knowledge.model_dump_json())
      logger.info(
        f"Knowledge entry updated successfully at {path}"
      )
    except Exception as e:
      logger.error(
        f"Failed to update knowledge entry: {e}"
      )
      raise

    return knowledge

  async def list_all(self) -> list[Knowledge]:
    """List all stored knowledge entries."""
    logger.info("Listing all stored knowledge entries")
    entries = []
    for path in self.storage_path.glob("*.json"):
      data = json.loads(path.read_text())
      entries.append(Knowledge.model_validate(data))
      logger.debug(
        f"Loaded knowledge entry from {path}"
      )

    logger.info(
      f"Total entries listed: {len(entries)}"
    )
    return entries
