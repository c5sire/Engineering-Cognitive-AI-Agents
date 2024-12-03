"""Basic knowledge storage implementation.

Winston's semantic memory system achieves connected knowledge through meaning rather
than explicit relationships. Using vector embeddings through ChromaDB, knowledge
naturally clusters by semantic similarity - "morning coffee" associates with both
"afternoon tea" and "father's habits" through shared meaning rather than explicit
links.

Architecture Overview:
```mermaid
graph TD
    K[Knowledge Entry] -->|Store| KS[Knowledge Store]
    K -->|Extract| M[Metadata]
    K -->|Track| T[Temporal Info]

    subgraph "Knowledge Store"
        KS -->|JSON| FS[File System]
        KS -->|Load| KO[Knowledge Objects]
        KS -->|List| KL[Knowledge List]
    end

    subgraph "Knowledge Management"
        C[Create Knowledge]
        U[Update Knowledge]
        D[Delete Knowledge]
        C --> K
        U --> K
        D --> K
    end

    subgraph "Temporal Layer"
        T -->|Created| CD[Creation Date]
        T -->|Modified| MD[Modified Date]
        T -->|History| H[Version History]
    end

    subgraph "Metadata Layer"
        M -->|Context| CT[Context Tags]
        M -->|Type| TY[Knowledge Type]
        M -->|Custom| CM[Custom Metadata]
    end
```

Design Philosophy:
The storage system provides the foundation for Winston's semantic memory,
implementing a simple but flexible knowledge persistence layer. Rather than using
a complex database system, it uses a straightforward file-based approach that:

1. Knowledge Structure
   - Unique identification for each piece of knowledge
   - Rich context through flexible metadata
   - Temporal tracking for knowledge evolution
   - Simple, human-readable storage format

2. Version Management
   - Creation and modification timestamps
   - Optional version history
   - Change tracking capabilities
   - Temporal context preservation

3. Flexible Organization
   - Context-aware storage
   - Metadata-based organization
   - Custom classification support
   - Natural knowledge grouping

Example Flow:
When Winston learns about a user's preference change:
1. Creates new knowledge entry with unique ID
2. Stores content and contextual metadata
3. Tracks temporal information
4. Enables future retrieval and updates

Key Architectural Principles:
- Simple, reliable storage mechanisms
- Focus on knowledge integrity
- Clear temporal tracking
- Flexible metadata support
- Human-readable formats

Implementation Note:
While the storage system supports rich metadata and versioning capabilities,
it maintains simplicity by using basic file system operations and JSON
serialization. This approach provides reliability and transparency while
enabling more sophisticated knowledge management through higher-level systems
like the semantic memory coordinator.

The storage system serves as the foundation for Winston's memory capabilities,
providing persistent storage that higher-level memory systems can build upon
while maintaining architectural clarity and operational reliability.
"""

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

  async def delete(self, knowledge_id: str) -> None:
    """Delete knowledge entry by ID.

    Parameters
    ----------
    knowledge_id : str
        ID of the knowledge entry to delete

    Raises
    ------
    FileNotFoundError
        If knowledge entry doesn't exist
    """
    logger.info(
      f"Deleting knowledge entry with ID: {knowledge_id}"
    )
    path = self._get_path(knowledge_id)

    if not path.exists():
      logger.warning(
        f"Knowledge {knowledge_id} not found"
      )
      raise FileNotFoundError(
        f"Knowledge {knowledge_id} not found"
      )

    try:
      path.unlink()
      logger.info(
        f"Knowledge entry deleted successfully: {knowledge_id}"
      )
    except Exception as e:
      logger.error(
        f"Failed to delete knowledge entry: {e}"
      )
      raise
