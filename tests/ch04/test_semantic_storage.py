"""Test semantic storage specialist functionality."""

import json
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.semantic.storage import (
  StorageSpecialist,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_storage_operations():
  """Test semantic storage operations."""
  logger.info("Starting test_storage_operations")

  # Setup
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    project_root = Path(__file__).parent.parent.parent
    paths = AgentPaths(
      root=temp_root, system_root=project_root
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

    # Test 1: Store Initial Knowledge
    store_msg = Message(
      content="I usually drink coffee in the morning, like my father used to",
      metadata={
        "semantic_metadata": json.dumps(
          [
            {"key": "type", "value": "preference"},
            {"key": "category", "value": "beverage"},
            {"key": "temporal", "value": "morning"},
          ]
        )
      },  # Updated metadata structure
    )
    logger.info("Test 1: Storing initial knowledge")

    storage_response = {}
    async for response in specialist.process(
      store_msg
    ):
      if response.metadata.get("streaming"):
        continue
      storage_response = json.loads(response.content)

    logger.debug(
      f"Storage response received: {storage_response}"
    )

    # Validate storage response structure
    assert "content" in storage_response
    assert "metadata" in storage_response
    assert "id" in storage_response

    # Store ID for verification
    knowledge_id = storage_response["id"]

    # Test 2: Store Update
    update_msg = Message(
      content="Actually, I've switched to tea",
      metadata={
        "semantic_metadata": json.dumps(
          [
            {"key": "type", "value": "preference"},
            {"key": "category", "value": "beverage"},
            {"key": "temporal", "value": "morning"},
            {
              "key": "related_id",
              "value": knowledge_id,
            },  # Added relationship tracking
          ]
        )
      },  # Updated metadata structure
    )
    logger.info("Test 2: Storing preference update")

    update_response = {}
    async for response in specialist.process(
      update_msg
    ):
      if response.metadata.get("streaming"):
        continue
      update_response = json.loads(response.content)

    logger.debug(
      f"Update response received: {update_response}"
    )

    # Validate update response
    assert "content" in update_response
    assert "metadata" in update_response
    assert (
      update_response["id"] != knowledge_id
    )  # Should be new entry
    assert "tea" in update_response["content"].lower()
