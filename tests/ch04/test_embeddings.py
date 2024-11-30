"""Test embedding store functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from winston.core.memory.embeddings import (
  EmbeddingStore,
)
from winston.core.memory.storage import Knowledge


@pytest.fixture
async def embedding_store():
  """Create temporary embedding store."""
  with tempfile.TemporaryDirectory() as temp_dir:
    yield EmbeddingStore(Path(temp_dir))


@pytest.mark.asyncio
async def test_embedding_operations(embedding_store):
  """Test basic embedding store operations."""
  # Create test knowledge entries
  knowledge1 = Knowledge(
    id="test1",
    content="The quick brown fox jumps over the lazy dog",
    context={"type": "test"},
    created_at=datetime.now(),
    updated_at=datetime.now(),
  )

  knowledge2 = Knowledge(
    id="test2",
    content="The lazy dog sleeps while the fox runs",
    context={"type": "test"},
    created_at=datetime.now(),
    updated_at=datetime.now(),
  )

  # Add embeddings
  await embedding_store.add_embedding(knowledge1)
  await embedding_store.add_embedding(knowledge2)

  # Test similarity search
  matches = await embedding_store.find_similar(
    "fox jumping over dog", limit=2
  )
  assert len(matches) == 2
  assert matches[0].id in ["test1", "test2"]

  # Test with filters
  matches = await embedding_store.find_similar(
    "fox jumping over dog",
    filters={"type": "test"},
    limit=1,
  )
  assert len(matches) == 1

  # Test update
  knowledge1.content = (
    "The quick red fox jumps over the sleeping dog"
  )
  await embedding_store.update_embedding(knowledge1)

  # Test delete
  await embedding_store.delete_embedding("test1")
  matches = await embedding_store.find_similar(
    "fox jumping", limit=2
  )
  assert len(matches) == 1
  assert matches[0].id == "test2"
