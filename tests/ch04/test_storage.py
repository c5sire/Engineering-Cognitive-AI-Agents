"""Test knowledge storage functionality."""

import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.memory.storage import (
  KnowledgeStorage,
)


@pytest.fixture
async def storage():
  """Fixture to create a temporary KnowledgeStorage instance."""
  with tempfile.TemporaryDirectory() as temp_dir:
    logger.debug(
      "Creating temporary KnowledgeStorage instance."
    )
    yield KnowledgeStorage(Path(temp_dir))


@pytest.mark.asyncio
async def test_knowledge_storage_operations(storage):
  """Test basic CRUD operations of KnowledgeStorage."""
  logger.info(
    "Starting test for knowledge storage operations."
  )

  # Test storing
  logger.debug("Storing knowledge content.")
  knowledge_id = await storage.store(
    content="Test knowledge",
    context={"type": "test"},
  )
  assert knowledge_id is not None
  logger.info(
    f"Stored knowledge with ID: {knowledge_id}"
  )

  # Test loading
  logger.debug(
    f"Loading knowledge with ID: {knowledge_id}."
  )
  knowledge = await storage.load(knowledge_id)
  assert knowledge.content == "Test knowledge"
  assert knowledge.context["type"] == "test"
  logger.info("Successfully loaded knowledge content.")

  # Test updating
  logger.debug("Updating knowledge content.")
  updated = await storage.update(
    knowledge_id,
    content="Updated knowledge",
    context={"status": "updated"},
  )
  assert updated.content == "Updated knowledge"
  assert updated.context["status"] == "updated"
  logger.info(
    "Successfully updated knowledge content."
  )

  # Test listing
  logger.debug("Listing all knowledge entries.")
  all_entries = await storage.list_all()
  assert len(all_entries) == 1
  assert any(
    entry.id == knowledge_id for entry in all_entries
  )
  logger.info(
    "Successfully listed all knowledge entries."
  )
