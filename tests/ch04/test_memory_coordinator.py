"""Test memory coordinator integration."""

import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.coordinator import (
  MemoryCoordinator,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.workspace import WorkspaceManager


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

    paths = AgentPaths(
      root=temp_root,
      system_root=project_root,
    )

    # Ensure workspaces directory exists
    paths.workspaces.mkdir(parents=True, exist_ok=True)

    yield paths


@pytest.mark.asyncio
async def test_memory_coordinator(system, paths):
  """Test memory coordinator orchestration through multiple episodes."""

  # Create and initialize shared workspace
  shared_workspace = (
    paths.workspaces / "shared_test.md"
  )
  workspace_manager = WorkspaceManager()
  workspace_manager.initialize_workspace(
    shared_workspace
  )

  # Initialize coordinator
  config = AgentConfig.from_yaml(
    paths.system_agents_config
    / "memory_coordinator.yaml"
  )
  coordinator = MemoryCoordinator(
    system, config, paths
  )

  # Test 1: Initial Statement
  initial_msg = Message(
    content="I usually drink coffee in the morning, like my father used to",
    metadata={"shared_workspace": shared_workspace},
  )

  logger.info("Test 1: Processing initial statement")
  responses = []
  async for response in coordinator.process(
    initial_msg
  ):
    if response.metadata.get("streaming"):
      continue

    logger.debug(f"Response: {response.content}")
    responses.append(response)

  # Verify initial processing
  workspace_content = shared_workspace.read_text()
  logger.debug(
    f"Workspace content after initial statement:\n{workspace_content}"
  )

  assert any(
    "coffee" in r.content.lower() for r in responses
  )
  assert "coffee" in workspace_content.lower()
  assert "morning" in workspace_content.lower()
  assert "father" in workspace_content.lower()

  # Test 2: Preference Update (Same Episode)
  update_msg = Message(
    content="Actually, I've switched to tea",
    metadata={"shared_workspace": shared_workspace},
  )

  logger.info("Test 2: Processing preference update")
  responses = []
  async for response in coordinator.process(
    update_msg
  ):
    if response.metadata.get("streaming"):
      continue
    logger.debug(f"Response: {response.content}")
    responses.append(response)

  # Verify preference update
  workspace_content = shared_workspace.read_text()
  logger.debug(
    f"Workspace content after preference update:\n{workspace_content}"
  )

  assert "tea" in workspace_content.lower()
  assert any(
    term in workspace_content.lower()
    for term in [
      "switched",
      "changed",
      "previous",
      "transition",
    ]
  )
  assert "morning" in workspace_content.lower()
  assert any(
    term in workspace_content.lower()
    for term in [
      "father",
      "family",
    ]
  )

  # Test 3: New Episode
  new_topic_msg = Message(
    content="Let's discuss the home renovation project",
    metadata={"shared_workspace": shared_workspace},
  )

  logger.info("Test 3: Processing new episode")
  responses = []
  async for response in coordinator.process(
    new_topic_msg
  ):
    if response.metadata.get("streaming"):
      continue
    logger.debug(f"Response: {response.content}")
    responses.append(response)

  # Verify new episode handling
  workspace_content = shared_workspace.read_text()
  logger.debug(
    f"Workspace content after new episode:\n{workspace_content}"
  )

  assert "renovation" in workspace_content.lower()
  assert "project" in workspace_content.lower()

  archived_content = workspace_content.lower()
  assert any(
    section in archived_content
    for section in [
      "previous",
      "history",
      "background",
      "archive",
    ]
  )
