"""Embedding-based semantic knowledge storage and retrieval.

Winston's semantic memory system achieves connected knowledge through meaning rather
than explicit relationships. Using vector embeddings through ChromaDB, knowledge
naturally clusters by semantic similarity - "morning coffee" associates with both
"afternoon tea" and "father's habits" through shared meaning rather than explicit
links.

Architecture Overview:
```mermaid
graph TD
    K[Knowledge Entry] -->|Embedding| VS[Vector Space]
    K -->|Extract| M[Metadata]

    VS -->|Clusters| SC[Semantic Connections]
    VS -->|Distance| SR[Similarity Ranking]

    Q[Query] -->|Embedding| VS
    Q -->|Optional Filters| M

    VS -->|Nearest| R[Raw Matches]
    M -->|Filter| R
    R -->|Ranked| FM[Final Matches]

    subgraph "Vector Space"
        SC
        SR
    end

    subgraph "Knowledge Management"
        A[Add Knowledge]
        U[Update Knowledge]
        D[Delete Knowledge]
        A --> K
        U --> K
        D --> K
    end

    subgraph "Metadata Layer"
        M
        F[Filter Rules]
        M --> F
    end
```

Design Philosophy:
Rather than maintaining explicit relationships between knowledge pieces, the system
relies on the natural semantic clustering that emerges in the embedding space.
This approach offers several key advantages:

1. Natural Associations
   - Knowledge connects through meaning
   - No explicit relationship management
   - Flexible, intuitive retrieval

2. Graceful Degradation
   - No exact matches needed
   - Returns semantically similar content
   - Ranks by relevance

3. Simple Evolution
   - Updates preserve semantic connections
   - No relationship maintenance
   - Natural knowledge organization

Metadata Strategy:
While semantic similarity drives our primary retrieval mechanism, metadata
filtering provides an optional secondary layer of organization:

1. Complementary Role
   - Metadata filters refine semantic matches
   - Binary exclusion of non-matching results
   - Useful for categorical constraints

2. Careful Application
   - Avoid over-reliance on strict filtering
   - Preserve semantic flexibility
   - Use only when categorically necessary

3. Implementation Balance
   - Primary: Semantic similarity
   - Secondary: Metadata filtering
   - Maintain retrieval flexibility

Example Flow:
When Winston learns "I usually drink coffee in the morning, like my father used to",
the embedding system:
1. Creates vector representation capturing semantic meaning
2. Places it in embedding space near related concepts
3. Enables retrieval through meaning-based similarity
4. Maintains connections as knowledge updates

Key Architectural Principles:
- Focus on semantic similarity over explicit relationships
- Simple metadata for filtering when needed
- Clean separation between storage and retrieval
- Emphasis on natural knowledge organization

This design enables sophisticated knowledge management while maintaining
architectural simplicity. The embedding space naturally captures relationships
that would be complex to maintain explicitly, while providing flexible,
meaning-based retrieval that mirrors human cognitive association.

Implementation Note:
While ChromaDB supports metadata filtering, we primarily rely on semantic
similarity for knowledge retrieval. This allows for fuzzy matching and graceful
degradation - less relevant results are still returned but with lower similarity
scores. This approach is more robust and cognitively plausible, as human memory
similarly operates through strength of association rather than strict categorical
boundaries.
"""

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
