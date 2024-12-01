"""Test semantic memory specialist functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from winston.core.agent import AgentConfig
from winston.core.memory.semantic_memory import (
  SemanticMemorySpecialist,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_semantic_memory_operations():
  """Test semantic memory storage and retrieval."""
  print("Starting test_semantic_memory_operations")

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
      / "semantic_memory.yaml"
    )
    specialist = SemanticMemorySpecialist(
      system, config, paths
    )

    # Test 1: Store Initial Knowledge
    store_msg = Message(
      content="I usually drink coffee in the morning, like my father used to",
      metadata={
        "type": "conversation",
        "relevance_query": "beverage preferences and routines",
      },
    )
    print("Test 1: Storing initial knowledge")

    storage_response = {}
    async for response in specialist.process(
      store_msg
    ):
      if response.metadata.get("streaming"):
        continue
      storage_response = json.loads(response.content)

    print(
      f"Storage response received: {storage_response}"
    )

    # Validate storage response structure
    assert "content" in storage_response
    assert "metadata" in storage_response
    assert "id" in storage_response["metadata"]

    # Store ID for later retrieval
    knowledge_id = storage_response["metadata"]["id"]

    # Validate content reflects input
    assert (
      "coffee" in storage_response["content"].lower()
    )
    assert (
      "morning" in storage_response["content"].lower()
    )

    # Validate metadata includes context
    assert storage_response["metadata"]["context"]
    assert storage_response["metadata"]["importance"]

    # Test 2: Retrieve Knowledge
    retrieve_msg = Message(
      content="What do I usually drink in the morning?",
      metadata={
        "type": "conversation",
        "relevance_query": "morning beverage preferences",
      },
    )
    print("Test 2: Retrieving knowledge")

    retrieval_response = {}
    async for response in specialist.process(
      retrieve_msg
    ):
      if response.metadata.get("streaming"):
        continue
      retrieval_response = json.loads(response.content)

    print(
      f"Retrieval response received: {retrieval_response}"
    )

    # Validate retrieval response
    assert "content" in retrieval_response
    assert retrieval_response["relevance"] is not None
    assert (
      retrieval_response["metadata"]["id"]
      == knowledge_id
    )

    # Validate lower relevance results
    assert (
      "lower_relevance_results" in retrieval_response
    )
    for item in retrieval_response[
      "lower_relevance_results"
    ]:
      assert "content" in item
      assert "metadata" in item

    # Test 3: Update Knowledge
    update_msg = Message(
      content="Actually, I've switched to tea",
      metadata={
        "type": "conversation",
        "relevance_query": "beverage preference update",
      },
    )
    print("Test 3: Updating knowledge")

    update_response = {}
    async for response in specialist.process(
      update_msg
    ):
      if response.metadata.get("streaming"):
        continue
      update_response = json.loads(response.content)

    print(
      f"Update response received: {update_response}"
    )

    # Validate update response
    assert "content" in update_response
    assert "metadata" in update_response
    assert (
      update_response["metadata"]["id"] != knowledge_id
    )  # Should be new entry
    assert "tea" in update_response["content"].lower()

    # Test 4: Retrieve Updated Knowledge
    final_msg = Message(
      content="Tell me about my beverage preferences over time",
      metadata={
        "type": "conversation",
        "relevance_query": "beverage preference history",
      },
    )
    print("Test 4: Retrieving updated knowledge")

    final_response = None
    async for response in specialist.process(
      final_msg
    ):
      if response.metadata.get("streaming"):
        continue
      final_response = json.loads(response.content)

    print("Final response received:", final_response)

    # Should find both old and new preferences
    assert final_response is not None
    assert "content" in final_response
    assert "metadata" in final_response
    assert "lower_relevance_results" in final_response

    found_coffee = any(
      "coffee" in item["content"].lower()
      for item in final_response[
        "lower_relevance_results"
      ]
    )
    found_tea = any(
      "tea" in item["content"].lower()
      for item in final_response[
        "lower_relevance_results"
      ]
    )

    assert (
      found_coffee
      or "coffee" in final_response["content"].lower()
    )
    assert (
      found_tea
      or "tea" in final_response["content"].lower()
    )
