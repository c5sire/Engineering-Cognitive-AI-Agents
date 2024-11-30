"""Test knowledge storage functionality."""

import tempfile
from pathlib import Path

import pytest

from winston.core.memory.storage import (
  KnowledgeStorage,
)


@pytest.fixture
async def storage():
  """Fixture to create a temporary KnowledgeStorage instance."""
  with tempfile.TemporaryDirectory() as temp_dir:
    yield KnowledgeStorage(Path(temp_dir))


@pytest.mark.asyncio
async def test_knowledge_storage_operations(storage):
  """Test basic CRUD operations of KnowledgeStorage."""
  # Test storing
  knowledge_id = await storage.store(
    content="Test knowledge",
    context={"type": "test"},
  )
  assert knowledge_id is not None

  # Test loading
  knowledge = await storage.load(knowledge_id)
  assert knowledge.content == "Test knowledge"
  assert knowledge.context["type"] == "test"

  # Test updating
  updated = await storage.update(
    knowledge_id,
    content="Updated knowledge",
    context={"status": "updated"},
  )
  assert updated.content == "Updated knowledge"
  assert updated.context["status"] == "updated"

  # Test listing
  all_entries = await storage.list_all()
  assert len(all_entries) == 1
  assert any(
    entry.id == knowledge_id for entry in all_entries
  )
