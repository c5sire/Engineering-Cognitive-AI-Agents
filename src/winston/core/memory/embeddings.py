"""Embedding storage and retrieval using ChromaDB."""

from pathlib import Path
from typing import Any, NamedTuple

import chromadb
from loguru import logger

from winston.core.memory.storage import Knowledge


class SimilarityMatch(NamedTuple):
  """Result from similarity search."""

  id: str
  score: float
  metadata: dict[str, Any]


class EmbeddingStore:
  """Manages embeddings using ChromaDB."""

  def __init__(self, store_path: Path) -> None:
    """Initialize ChromaDB client and collection."""
    logger.info(
      f"Initializing EmbeddingStore with store path: {store_path}"
    )
    # Initialize persistent client
    self.client = chromadb.PersistentClient(
      path=str(store_path)
    )
    logger.debug("ChromaDB client initialized.")

    # Get or create collection
    self.collection = (
      self.client.get_or_create_collection(
        name="knowledge_embeddings",
        metadata={
          "hnsw:space": "cosine"
        },  # Use cosine similarity
      )
    )
    logger.info(
      "ChromaDB collection 'knowledge_embeddings' created or retrieved."
    )

  async def add_embedding(
    self,
    knowledge: Knowledge,
  ) -> None:
    """Add knowledge embedding to store."""
    logger.info(
      f"Adding embedding for knowledge ID: {knowledge.id}"
    )
    # Add to collection
    self.collection.add(
      documents=[knowledge.content],
      metadatas=[knowledge.context],
      ids=[knowledge.id],
    )
    logger.debug(
      f"Embedding added for knowledge ID: {knowledge.id}"
    )

  async def find_similar(
    self,
    query: str,
    limit: int = 5,
    filters: dict[str, Any] | None = None,
  ) -> list[SimilarityMatch]:
    """Find similar knowledge entries."""
    logger.info(
      f"Finding similar entries for query: {query} with limit: {limit}"
    )
    # Query collection
    results = self.collection.query(
      query_texts=[query],
      n_results=limit,
      where=filters,
    )
    logger.debug("Query executed.")

    # Format results
    matches = []
    if not results["ids"]:
      logger.warning("No matching entries found.")
      return matches

    for idx, (id_, distance, metadata) in enumerate(
      zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
      )
    ):
      matches.append(
        SimilarityMatch(
          id=id_,
          score=1.0
          - distance,  # Convert distance to similarity
          metadata=metadata,
        )
      )
      logger.debug(
        f"Match found: ID={id_}, Score={1.0 - distance}"
      )

    logger.info(f"Total matches found: {len(matches)}")
    return matches

  async def update_embedding(
    self,
    knowledge: Knowledge,
  ) -> None:
    """Update embedding for modified knowledge."""
    logger.info(
      f"Updating embedding for knowledge ID: {knowledge.id}"
    )
    # Update in collection
    self.collection.update(
      documents=[knowledge.content],
      metadatas=[knowledge.context],
      ids=[knowledge.id],
    )
    logger.debug(
      f"Embedding updated for knowledge ID: {knowledge.id}"
    )

  async def delete_embedding(
    self,
    knowledge_id: str,
  ) -> None:
    """Delete embedding for knowledge entry."""
    logger.info(
      f"Deleting embedding for knowledge ID: {knowledge_id}"
    )
    self.collection.delete(ids=[knowledge_id])
    logger.debug(
      f"Embedding deleted for knowledge ID: {knowledge_id}"
    )
