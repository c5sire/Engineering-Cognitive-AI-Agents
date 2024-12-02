"""Test semantic memory coordinator functionality."""

import json
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.semantic.coordinator import (
  SemanticMemoryCoordinator,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.fixture
def paths():
  """Create temporary paths for testing."""
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    project_root = Path(__file__).parent.parent.parent
    paths = AgentPaths(
      root=temp_root,
      system_root=project_root,
    )
    yield paths


@pytest.mark.asyncio
async def test_semantic_memory_operations(paths):
  """Test complete semantic memory flow through coordinator."""
  logger.info(
    "Starting semantic memory coordinator test"
  )

  # Initialize system and coordinator
  system = AgentSystem()
  config = AgentConfig.from_yaml(
    paths.system_agents_config
    / "memory"
    / "semantic"
    / "coordinator.yaml"
  )
  coordinator = SemanticMemoryCoordinator(
    system, config, paths
  )

  # Test Cases:

  # 1. New Information (Nothing Found)
  logger.info("Test Case 1: New Information")
  new_info_msg = Message(
    content="I usually drink coffee in the morning, like my father used to",
    metadata={"type": "observation"},
  )

  # Collect results by type
  retrieval_results = []
  storage_results = []
  async for response in coordinator.process(
    new_info_msg
  ):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        storage_results.append(result)

  # Now we can verify each type of operation
  assert (
    len(retrieval_results) > 0
  )  # Verify retrieval happened
  assert (
    retrieval_results[0]["content"] is None
  )  # Should be empty for new info

  assert (
    len(storage_results) > 0
  )  # Verify storage happened
  stored_result = storage_results[0]
  assert (
    "coffee" in stored_result["content"].lower()
  )  # Verify content was stored

  # 2. Update Existing (Exact Match)
  logger.info("Test Case 2: Update Existing")
  update_msg = Message(
    content="Actually, I've switched to tea",
    metadata={"type": "observation"},
  )

  # Collect results by type
  retrieval_results = []
  update_results = []
  async for response in coordinator.process(
    update_msg
  ):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        update_results.append(result)

  # Verify update handling
  assert len(update_results) > 0
  update_result = update_results[0]
  assert "tea" in update_result["content"].lower()

  # 3. Related Information
  logger.info("Test Case 3: Related Information")
  related_msg = Message(
    content="I enjoy green tea after meals",
    metadata={"type": "observation"},
  )

  # Collect results by type
  retrieval_results = []
  related_results = []
  async for response in coordinator.process(
    related_msg
  ):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        related_results.append(result)

  # Verify handling of related information
  assert len(related_results) > 0
  related_result = related_results[0]
  assert (
    "green tea" in related_result["content"].lower()
  )
  assert (
    "after meals" in related_result["content"].lower()
  )

  # 4. Command (Should Not Store)
  logger.info("Test Case 4: Command")
  command_msg = Message(
    content="Open the bag of chips",
    metadata={"type": "command"},
  )

  # Collect results by type
  retrieval_results = []
  command_results = []
  async for response in coordinator.process(
    command_msg
  ):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        command_results.append(result)

  # Verify command is not stored
  assert len(command_results) > 0
  command_result = command_results[0]
  assert (
    command_result["action"] == "no_storage_needed"
  )

  # 5. Question (Context Retrieval)
  logger.info("Test Case 5: Question")
  question_msg = Message(
    content="What do I usually drink?",
    metadata={"type": "question"},
  )

  # Collect results by type
  retrieval_results = []
  question_results = []
  async for response in coordinator.process(
    question_msg
  ):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        question_results.append(result)

  # Verify context retrieval
  assert len(question_results) > 0
  question_result = question_results[0]
  assert (
    "tea" in question_result["content"].lower()
  )  # Should find latest preference
  assert (
    question_result["relevance"] > 0.5
  )  # High relevance match

  # 6. Autonomous Observation
  logger.info("Test Case 6: Autonomous Observation")
  auto_msg = Message(
    content="Research indicates green tea has health benefits",
    metadata={
      "type": "research_finding",
      "source": "autonomous_research",
    },
  )

  # Collect results by type
  retrieval_results = []
  auto_results = []
  async for response in coordinator.process(auto_msg):
    if not response.metadata.get("streaming"):
      result = json.loads(response.content)
      if (
        response.metadata.get("type")
        == "retrieved_context"
      ):
        logger.debug(f"Retrieval result: {result}")
        retrieval_results.append(result)
      elif (
        response.metadata.get("type")
        == "storage_result"
      ):
        logger.debug(f"Storage result: {result}")
        auto_results.append(result)

  # Verify research storage
  assert len(auto_results) > 0
  auto_result = auto_results[0]
  assert (
    "health benefits" in auto_result["content"].lower()
  )
  assert (
    auto_result["metadata"].get("source")
    == "autonomous_research"
  )

  logger.info(
    "Completed semantic memory coordinator test"
  )
