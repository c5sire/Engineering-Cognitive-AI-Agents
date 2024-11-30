"""Basic knowledge storage implementation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

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
    """Initialize storage in specified directory.

    Parameters
    ----------
    storage_path : Path
        Directory for storing knowledge files
    """
    self.storage_path = storage_path
    self.storage_path.mkdir(
      parents=True, exist_ok=True
    )

  def _generate_id(self) -> str:
    """Generate unique knowledge ID."""
    return str(uuid.uuid4())

  def _get_path(self, knowledge_id: str) -> Path:
    """Get file path for knowledge ID."""
    return self.storage_path / f"{knowledge_id}.json"

  async def store(
    self, content: str, context: dict[str, Any]
  ) -> str:
    """Store new knowledge entry.

    Parameters
    ----------
    content : str
        Knowledge content to store
    context : dict[str, Any]
        Associated context/metadata

    Returns
    -------
    str
        ID of stored knowledge
    """
    knowledge_id = self._generate_id()
    knowledge = Knowledge(
      id=knowledge_id,
      content=content,
      context=context,
      created_at=datetime.now(),
      updated_at=datetime.now(),
    )

    # Write to file
    path = self._get_path(knowledge_id)
    path.write_text(knowledge.model_dump_json())

    return knowledge_id

  async def load(self, knowledge_id: str) -> Knowledge:
    """Load knowledge by ID.

    Parameters
    ----------
    knowledge_id : str
        ID of knowledge to load

    Returns
    -------
    Knowledge
        Loaded knowledge entry

    Raises
    ------
    FileNotFoundError
        If knowledge ID doesn't exist
    """
    path = self._get_path(knowledge_id)
    if not path.exists():
      raise FileNotFoundError(
        f"Knowledge {knowledge_id} not found"
      )

    data = json.loads(path.read_text())
    return Knowledge.model_validate(data)

  async def update(
    self,
    knowledge_id: str,
    content: str | None = None,
    context: dict[str, Any] | None = None,
  ) -> Knowledge:
    """Update existing knowledge.

    Parameters
    ----------
    knowledge_id : str
        ID of knowledge to update
    content : str | None
        New content (if None, keep existing)
    context : dict[str, Any] | None
        New context (if None, keep existing)

    Returns
    -------
    Knowledge
        Updated knowledge entry

    Raises
    ------
    FileNotFoundError
        If knowledge ID doesn't exist
    """
    # Load existing
    knowledge = await self.load(knowledge_id)

    # Update fields
    if content is not None:
      knowledge.content = content
    if context is not None:
      knowledge.context.update(context)
    knowledge.updated_at = datetime.now()

    # Save changes
    path = self._get_path(knowledge_id)
    path.write_text(knowledge.model_dump_json())

    return knowledge

  async def list_all(self) -> list[Knowledge]:
    """List all stored knowledge entries.

    Returns
    -------
    list[Knowledge]
        All stored knowledge entries
    """
    entries = []
    for path in self.storage_path.glob("*.json"):
      data = json.loads(path.read_text())
      entries.append(Knowledge.model_validate(data))
    return entries
