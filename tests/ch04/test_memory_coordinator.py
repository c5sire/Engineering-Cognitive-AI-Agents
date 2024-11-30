"""Test memory coordinator integration."""

import tempfile
from pathlib import Path

import pytest

from winston.core.agent import AgentConfig
from winston.core.memory.coordinator import (
  MemoryCoordinator,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.fixture
def system():
  """Create test system."""
  return AgentSystem()


@pytest.fixture
def paths():
  """Create temporary paths."""
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    project_root = Path(__file__).parent.parent.parent

    yield AgentPaths(
      root=temp_root,
      system_root=project_root,
    )


@pytest.mark.asyncio
async def test_memory_coordinator(system, paths):
  """Test memory coordinator storing and retrieving knowledge."""

  # Load config from YAML
  config = AgentConfig.from_yaml(
    paths.system_agents_config
    / "memory_coordinator.yaml"
  )

  # Register memory coordinator
  coordinator = MemoryCoordinator(
    system=system,
    config=config,
    paths=paths,
  )

  # Test storing knowledge
  store_message = Message(
    content=(
      "I usually drink coffee in the morning, "
      "like my father used to"
    )
  )

  # Accumulate complete responses
  accumulated_store = []
  current_response = []

  async for response in system.invoke_conversation(
    "memory_coordinator",
    store_message.content,
  ):
    if response.metadata.get("streaming", False):
      current_response.append(response.content)
    else:
      # Non-streaming response (like tool results)
      if current_response:
        accumulated_store.append(
          "".join(current_response)
        )
        current_response = []
      accumulated_store.append(response.content)

  # Add any remaining content
  if current_response:
    accumulated_store.append("".join(current_response))

  print(
    "Complete responses from storing:",
    accumulated_store,
  )

  assert len(accumulated_store) > 0
  assert any(
    "stored" in r.lower() for r in accumulated_store
  )

  # Debug: Inspect stored files
  knowledge_dir = paths.workspaces / "knowledge"
  embedding_dir = paths.workspaces / "embeddings"

  print("\nKnowledge files:")
  for file in knowledge_dir.glob("*.json"):
    print(f"\nFile: {file.name}")
    print(f"Content: {file.read_text()}")

  print("\nChromaDB files:")
  for file in embedding_dir.glob("*"):
    print(f"File: {file.name}")

  # Test retrieving knowledge
  retrieve_message = Message(
    content=(
      "Please check your knowledge store and tell me "
      "what you know about my morning routine"
    )
  )

  # Accumulate complete responses
  accumulated_retrieve = []
  current_response = []

  async for response in system.invoke_conversation(
    "memory_coordinator",
    retrieve_message.content,
  ):
    if response.metadata.get("streaming", False):
      current_response.append(response.content)
    else:
      # Non-streaming response (like tool results)
      if current_response:
        accumulated_retrieve.append(
          "".join(current_response)
        )
        current_response = []
      accumulated_retrieve.append(response.content)

  # Add any remaining content
  if current_response:
    accumulated_retrieve.append(
      "".join(current_response)
    )

  print(
    "Complete responses from retrieval:",
    accumulated_retrieve,
  )

  assert len(accumulated_retrieve) > 0
  assert any(
    "coffee" in r.lower() for r in accumulated_retrieve
  )
