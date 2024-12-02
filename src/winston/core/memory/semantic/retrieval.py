# winston/core/memory/semantic/retrieval.py
"""Retrieval specialist for semantic memory."""

import json
from typing import AsyncIterator

from loguru import logger

from winston.core.agent import BaseAgent
from winston.core.memory.embeddings import (
  EmbeddingStore,
)
from winston.core.memory.storage import (
  KnowledgeStorage,
)
from winston.core.messages import Message, Response


class RetrievalSpecialist(BaseAgent):
  """Specialist for formulating and executing knowledge retrieval."""

  def __init__(self, system, config, paths) -> None:
    super().__init__(system, config, paths)
    storage_path = paths.workspaces / "knowledge"
    embedding_path = paths.workspaces / "embeddings"
    self._storage = KnowledgeStorage(storage_path)
    self._embeddings = EmbeddingStore(embedding_path)

  async def process(
    self, message: Message
  ) -> AsyncIterator[Response]:
    """Process retrieval requests."""
    logger.info("Processing retrieval request")

    # Generate retrieval query
    query_prompt = f"""
        Analyze this retrieval request:
        {message.content}

        Formulate an effective search query by:
        1. Identifying key concepts to search for
        2. Including relevant context terms
        3. Considering alternative phrasings
        4. Specifying any filters needed

        Format your response as JSON with:
        - query: The main search terms
        - context: Additional context to consider
        - max_results: Number of results to return (1-5)
        """

    query_analysis = await self.generate_response(
      Message(
        content=query_prompt,
        metadata={"type": "Query Analysis"},
      )
    )

    try:
      query_data = json.loads(query_analysis.content)

      # Find similar knowledge
      matches = await self._embeddings.find_similar(
        query=query_data["query"],
        limit=query_data["max_results"],
      )

      # Load full knowledge entries
      results = []
      best_match = None
      for match in matches:
        knowledge = await self._storage.load(match.id)
        result = {
          "content": knowledge.content,
          "relevance": match.score,
          "metadata": {
            "id": match.id,
            **knowledge.context,
          },
        }

        if (
          best_match is None
          or match.score > best_match["relevance"]
        ):
          if best_match:
            results.append(best_match)
          best_match = result
        else:
          results.append(result)

      yield Response(
        content=json.dumps(
          {
            "content": best_match["content"],
            "relevance": best_match["relevance"],
            "metadata": best_match["metadata"],
            "lower_relevance_results": results,
          }
        ),
        metadata={"type": "Retrieval Complete"},
      )

    except Exception as e:
      logger.error(f"Retrieval error: {e}")
      yield Response(
        content=f"Error retrieving knowledge: {str(e)}",
        metadata={"error": True},
      )
