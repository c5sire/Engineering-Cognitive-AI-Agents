"""Retrieval Specialist: Expert agent for semantic knowledge query formulation.

The Retrieval Specialist transforms information needs into effective semantic queries,
finding relevant knowledge through meaning rather than exact matches. This specialist
focuses purely on the cognitive aspects of retrieval - analyzing content to identify
key concepts and relationships that will surface relevant knowledge.

Architecture Overview:
```mermaid
graph TD
    RS[Retrieval Specialist]

    subgraph "Query Analysis"
        QF[Query Formulation]
        ST[Search Terms]
        RC[Related Concepts]
    end

    subgraph "Search Execution"
        ES[Embedding Search]
        MF[Metadata Filtering]
        RR[Result Ranking]
    end

    RS -->|Analyzes| QF
    QF -->|Identifies| ST
    QF -->|Expands to| RC

    ST --> ES
    RC --> ES
    ES -->|Optional| MF
    ES --> RR

    RR -->|Ranked Results| RS
```

Design Philosophy:
The retrieval specialist embodies several key principles of cognitive search:

1. Semantic Understanding
   - Identifies core concepts in queries
   - Expands to related terms and ideas
   - Considers multiple phrasings
   - Maintains query intent

2. Relevance Assessment
   - Primary: Semantic similarity
   - Secondary: Metadata matching
   - Considers context importance
   - Ranks multiple relevance levels

3. Result Organization
   - Best match first
   - Related matches ranked
   - Relevance scores included
   - Context preserved

Example Flows:

1. Direct Query:
   Input: "What do I drink in the morning?"
   Analysis:
   - Core concepts: beverage, morning routine
   - Related terms: breakfast, preferences
   - Temporal context: morning, daily habits

2. Contextual Query:
   Input: "Would that work with my schedule?"
   Analysis:
   - Reference resolution needed
   - Time management context
   - Pattern matching required
   - Preference consideration

3. Implicit Query:
   Input: "I'm thinking of changing my routine"
   Analysis:
   - Current routine retrieval
   - Pattern identification
   - Historical changes
   - Related preferences

Key Design Decisions:
- Focus on query formulation over execution
- Semantic matching prioritized over filters
- Multiple relevance levels returned
- Context preserved in results
- Clear relevance explanations

The specialist's system prompt guides the LLM to:
1. Analyze the retrieval need
2. Identify key concepts and relationships
3. Consider alternative phrasings
4. Formulate effective queries
5. Explain retrieval strategy

This design enables sophisticated information retrieval while maintaining:
- Clean separation of concerns (cognitive vs mechanical)
- Focus on semantic understanding
- Natural query expansion
- Clear relevance assessment

"""

from typing import Any

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


class RetrieveKnowledgeRequest(BaseModel):
  """Parameters for knowledge retrieval."""

  query: str = Field(
    description="The search query to find relevant knowledge"
  )
  rationale: str = Field(
    description="Explanation for the retrieval query"
  )
  max_results: int = Field(
    description="Maximum number of results to return"
  )


class KnowledgeItem(BaseModel):
  """Structured knowledge item."""

  content: str | None = Field(
    default=None,
    description="The retrieved/stored knowledge",
  )
  relevance: float | None = Field(
    default=None,
    description="Relevance score for retrieved items",
  )
  metadata: dict[str, Any] = Field(
    default_factory=dict,
    description="Associated metadata",
  )


class RetrieveKnowledgeResult(KnowledgeItem):
  """Structured knowledge response with additional lower relevance results."""

  lower_relevance_results: list[KnowledgeItem] = Field(
    default_factory=list,
    description="List of lower relevance knowledge items",
  )


class RetrievalSpecialist(BaseAgent):
  """Specialist for formulating and executing knowledge retrieval."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)
    logger.info(
      f"Initializing RetrievalSpecialist {self.id}"
    )

    storage_path = paths.workspaces / "knowledge"
    embedding_path = paths.workspaces / "embeddings"
    self._storage = KnowledgeStorage(storage_path)
    self._embeddings = EmbeddingStore(embedding_path)

    # Register the retrieval tool
    tool = Tool(
      name="retrieve_knowledge",
      description="Find relevant knowledge using semantic search",
      handler=self._handle_retrieve_knowledge,
      input_model=RetrieveKnowledgeRequest,
      output_model=RetrieveKnowledgeResult,
    )
    self.system.register_tool(tool)

    # Grant self access
    self.system.grant_tool_access(self.id, [tool.name])

    logger.info(
      "RetrievalSpecialist initialization complete"
    )

  async def _handle_retrieve_knowledge(
    self,
    request: RetrieveKnowledgeRequest,
  ) -> RetrieveKnowledgeResult:
    """Handle knowledge retrieval requests."""
    logger.debug(
      f"Handling retrieval request: {request}"
    )

    matches = await self._embeddings.find_similar(
      query=request.query,
      limit=request.max_results,
    )
    logger.debug(f"Found {len(matches)} matches")

    if not matches:
      return RetrieveKnowledgeResult(
        content=None,
        relevance=None,
        metadata={},
      )

    # Process matches into KnowledgeItems
    lower_relevance_results: list[KnowledgeItem] = []
    best_match: KnowledgeItem | None = None

    for match in matches:
      knowledge = await self._storage.load(match.id)
      item = KnowledgeItem(
        content=knowledge.content,
        relevance=match.score,
        metadata={"id": match.id, **knowledge.context},
      )

      if best_match is None:
        best_match = item
      else:
        lower_relevance_results.append(item)

    return RetrieveKnowledgeResult(
      content=best_match.content
      if best_match
      else None,
      relevance=best_match.relevance
      if best_match
      else None,
      metadata=best_match.metadata
      if best_match
      else {},
      lower_relevance_results=lower_relevance_results,
    )
