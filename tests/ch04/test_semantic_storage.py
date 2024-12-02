"""Test semantic storage specialist functionality."""

import json
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.semantic.storage import (
  KnowledgeActionType,
  StorageSpecialist,
  StoreKnowledgeResponse,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_storage_operations():
  """Test knowledge storage analysis and operations."""
  logger.info("Starting test_storage_operations")

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
      / "storage.yaml"
    )
    specialist = StorageSpecialist(
      system, config, paths
    )

    # Test 1: Store New Knowledge (No Existing)
    store_msg = Message(
      content="I usually drink coffee in the morning, like my father used to",
      metadata={
        "retrieved_content": json.dumps(
          []
        )  # No existing knowledge
      },
    )
    logger.info("Test 1: Storing new knowledge")

    storage_response = None
    async for response in specialist.process(
      store_msg
    ):
      if response.metadata.get("streaming"):
        continue
      storage_response = (
        StoreKnowledgeResponse.model_validate_json(
          response.content
        )
      )
      logger.debug(
        f"Storage response: {storage_response}"
      )

    # Validate storage response
    assert storage_response is not None
    assert (
      storage_response.action
      == KnowledgeActionType.CREATED
    )
    assert storage_response.id is not None
    assert storage_response.content is not None
    assert storage_response.metadata is not None
    knowledge_id: str = storage_response.id

    # Test 2: Update Existing Knowledge
    update_msg = Message(
      content="Actually, I've switched to tea",
      metadata={
        "retrieved_content": json.dumps(
          [
            {
              "id": knowledge_id,
              "content": "I usually drink coffee in the morning",
              "metadata": {
                "type": "preference",
                "category": "beverage",
              },
            }
          ]
        )
      },
    )
    logger.info("Test 2: Updating existing knowledge")

    update_response = None
    async for response in specialist.process(
      update_msg
    ):
      if response.metadata.get("streaming"):
        continue
      update_response = (
        StoreKnowledgeResponse.model_validate_json(
          response.content
        )
      )
      logger.debug(
        f"Update response: {update_response}"
      )

    # Validate update response
    assert update_response is not None
    assert (
      update_response.action
      == KnowledgeActionType.TEMPORAL_CHANGE
    )

    # Test 3: Command (Should Not Store)
    command_msg = Message(
      content="Open the bag of chips",
      metadata={"retrieved_content": json.dumps([])},
    )
    logger.info("Test 3: Processing command")

    command_response = None
    async for response in specialist.process(
      command_msg
    ):
      if response.metadata.get("streaming"):
        continue
      command_response = (
        StoreKnowledgeResponse.model_validate_json(
          response.content
        )
      )
      logger.debug(
        f"Command response: {command_response}"
      )

    # Validate command handling
    assert command_response is not None
    assert (
      command_response.action
      == KnowledgeActionType.NO_STORAGE_NEEDED
    )
    assert command_response.reason is not None

    # Test 4: Related Knowledge
    related_msg = Message(
      content="I enjoy green tea after meals",
      metadata={
        "retrieved_content": json.dumps(
          [
            {
              "id": knowledge_id,
              "content": "I've switched to tea",
              "metadata": {
                "type": "preference",
                "category": "beverage",
              },
            }
          ]
        )
      },
    )
    logger.info("Test 4: Processing related knowledge")

    related_response = None
    async for response in specialist.process(
      related_msg
    ):
      if response.metadata.get("streaming"):
        continue
      related_response = (
        StoreKnowledgeResponse.model_validate_json(
          response.content
        )
      )
      logger.debug(
        f"Related knowledge response: {related_response}"
      )

    # Validate related knowledge handling
    assert related_response is not None
    assert (
      related_response.action
      == KnowledgeActionType.CREATED
    )
    logger.info("Completed storage specialist tests")
