"""Embedding storage and retrieval using ChromaDB."""

from pathlib import Path
from typing import Any, NamedTuple

import chromadb

from winston.core.memory.storage import Knowledge


class SimilarityMatch(NamedTuple):
  """Result from similarity search."""

  id: str
  score: float
  metadata: dict[str, Any]


class EmbeddingStore:
  """Manages embeddings using ChromaDB."""

  def __init__(self, store_path: Path):
    """Initialize ChromaDB client and collection.

    Parameters
    ----------
    store_path : Path
        Directory for persistent storage
    """
    # Initialize persistent client
    self.client = chromadb.PersistentClient(
      path=str(store_path)
    )

    # Get or create collection
    self.collection = (
      self.client.get_or_create_collection(
        name="knowledge_embeddings",
        metadata={
          "hnsw:space": "cosine"
        },  # Use cosine similarity
      )
    )

  async def add_embedding(
    self,
    knowledge: Knowledge,
  ) -> None:
    """Add knowledge embedding to store.

    Parameters
    ----------
    knowledge : Knowledge
        Knowledge entry to embed and store
    """
    # Add to collection
    self.collection.add(
      documents=[knowledge.content],
      metadatas=[knowledge.context],
      ids=[knowledge.id],
    )

  async def find_similar(
    self,
    query: str,
    limit: int = 5,
    filters: dict[str, Any] | None = None,
  ) -> list[SimilarityMatch]:
    """Find similar knowledge entries.

    Parameters
    ----------
    query : str
        Query text to match against
    limit : int, optional
        Maximum number of results, by default 5
    filters : dict[str, Any] | None, optional
        Metadata filters to apply, by default None

    Returns
    -------
    list[SimilarityMatch]
        Matching entries with similarity scores
    """
    # Query collection
    results = self.collection.query(
      query_texts=[query],
      n_results=limit,
      where=filters,
    )

    # Format results
    matches = []
    if not results["ids"]:
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

    return matches

  async def update_embedding(
    self,
    knowledge: Knowledge,
  ) -> None:
    """Update embedding for modified knowledge.

    Parameters
    ----------
    knowledge : Knowledge
        Updated knowledge entry
    """
    # Update in collection
    self.collection.update(
      documents=[knowledge.content],
      metadatas=[knowledge.context],
      ids=[knowledge.id],
    )

  async def delete_embedding(
    self,
    knowledge_id: str,
  ) -> None:
    """Delete embedding for knowledge entry.

    Parameters
    ----------
    knowledge_id : str
        ID of knowledge to remove
    """
    self.collection.delete(ids=[knowledge_id])
