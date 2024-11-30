"""Test workspace maintenance through MemoryCoordinator."""

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


@pytest.mark.asyncio
async def test_shared_workspace_lifecycle():
  """Test shared workspace maintenance using coffee preference example."""
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
      / "memory_coordinator.yaml"
    )
    coordinator = MemoryCoordinator(
      system, config, paths
    )

    # Create shared workspace path
    shared_workspace = (
      paths.workspaces / "shared_test.md"
    )

    # 1. Initial statement
    message = Message(
      content="I usually drink coffee in the morning, like my father used to",
      metadata={"shared_workspace": shared_workspace},
    )

    # Process message
    async for _ in coordinator.process(message):
      pass

    # Verify shared workspace content
    content = shared_workspace.read_text()
    assert "coffee" in content.lower()
    assert "morning" in content.lower()
    assert "father" in content.lower()

    # 2. Update preference
    update = Message(
      content="Actually, I've switched to tea",
      metadata={"shared_workspace": shared_workspace},
    )

    # Process update
    async for _ in coordinator.process(update):
      pass

    # Verify workspace reflects change while maintaining context
    updated = shared_workspace.read_text()
    assert "tea" in updated.lower()
    assert "switched from coffee" in updated.lower()
    assert (
      "morning" in updated.lower()
    )  # Maintains time context
    assert (
      "father" in updated.lower()
    )  # Maintains family context
