"""Test retrieval specialist functionality."""

import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.semantic.retrieval import (
  RetrievalSpecialist,
  RetrieveKnowledgeResponse,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_retrieval_operations():
  """Test knowledge retrieval operations."""
  logger.info("Starting test_retrieval_operations")

  # Setup
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    project_root = Path(__file__).parent.parent.parent
    paths = AgentPaths(
      root=temp_root,
      system_root=project_root,
    )

    system = AgentSystem()
    config = AgentConfig.from_yaml(
      paths.system_agents_config
      / "memory"
      / "semantic"
      / "retrieval.yaml"
    )
    specialist = RetrievalSpecialist(
      system, config, paths
    )

    # Test 1: Retrieve with Empty Store
    empty_msg = Message(
      content="What do I usually drink in the morning?",
    )
    logger.info("Test 1: Retrieving from empty store")

    empty_response: (
      RetrieveKnowledgeResponse | None
    ) = None
    async for response in specialist.process(
      empty_msg
    ):
      if response.metadata.get("streaming"):
        continue
      empty_response = (
        RetrieveKnowledgeResponse.model_validate_json(
          response.content
        )
      )

    logger.debug(
      f"Empty store response: {empty_response}"
    )
    assert empty_response is not None
    assert empty_response.content is None

    # Test 2: Retrieve After Storage
    # First store some test knowledge using the storage specialist
    knowledge1 = await specialist._storage.store(
      content="I usually drink coffee in the morning, like my father used to",
      context={
        "type": "preference",
        "category": "beverage",
        "temporal": "morning",
      },
    )
    # Add embedding for first knowledge entry
    await specialist._embeddings.add_embedding(
      await specialist._storage.load(knowledge1)
    )

    knowledge2 = await specialist._storage.store(
      content="I enjoy green tea after meals",
      context={
        "type": "preference",
        "category": "beverage",
        "temporal": "after meals",
      },
    )
    # Add embedding for second knowledge entry
    await specialist._embeddings.add_embedding(
      await specialist._storage.load(knowledge2)
    )

    # Now try retrieval
    retrieve_msg = Message(
      content="What do I drink in the morning?",
    )
    logger.info("Test 2: Retrieving stored knowledge")

    retrieval_response: (
      RetrieveKnowledgeResponse | None
    ) = None
    async for response in specialist.process(
      retrieve_msg
    ):
      if response.metadata.get("streaming"):
        continue
      retrieval_response = (
        RetrieveKnowledgeResponse.model_validate_json(
          response.content
        )
      )

    logger.debug(
      f"Retrieval response: {retrieval_response}"
    )

    # Verify retrieval results
    assert retrieval_response is not None
    assert (
      "coffee" in retrieval_response.content.lower()
    )
    assert (
      "morning" in retrieval_response.content.lower()
    )

    # Test 3: Multiple Results
    multi_msg = Message(
      content="Tell me about my beverage preferences",
    )
    logger.info("Test 3: Retrieving multiple results")

    multi_response: (
      RetrieveKnowledgeResponse | None
    ) = None
    async for response in specialist.process(
      multi_msg
    ):
      if response.metadata.get("streaming"):
        continue
      multi_response = (
        RetrieveKnowledgeResponse.model_validate_json(
          response.content
        )
      )

    logger.debug(
      f"Multiple results response: {multi_response}"
    )

    # Verify multiple results
    assert multi_response is not None
    assert any(
      term in multi_response.content.lower()
      for term in ["coffee", "tea"]
    )
    assert any(
      term
      in multi_response.lower_relevance_results[
        i
      ].content.lower()
      for i in range(
        len(multi_response.lower_relevance_results)
      )
      for term in ["tea", "coffee"]
    )

    logger.info("Completed test_retrieval_operations")
