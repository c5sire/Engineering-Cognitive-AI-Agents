"""Test episode analysis capabilities."""

import tempfile
from pathlib import Path

import pytest

from winston.core.agent import AgentConfig
from winston.core.memory.episode_analyst import (
  EpisodeAnalyst,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_episode_analysis():
  """Test episode analysis and tool selection."""
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
      / "episode_analyst.yaml"
    )
    analyst = EpisodeAnalyst(system, config, paths)

    # Initial workspace content
    workspace_content = """
        ## Current Context
        User drinks coffee in the morning, following a family tradition.
        Father used to have the same routine.

        ## Preferences
        - Morning beverage: Coffee
        - Time: Early morning
        - Context: Family tradition
        """

    # Test 1: Information Update
    update_msg = Message(
      content="Actually, I've switched to tea",
      metadata={
        "current_workspace": workspace_content
      },
    )

    async for response in analyst.process(update_msg):
      # Should indicate update (not new episode)
      # Should preserve morning/family context
      # Should call update_content tool (not clear_workspace)
      print(f"Update response: {response.content}")
      print(f"Metadata: {response.metadata}")

    # Test 2: New Episode
    new_topic_msg = Message(
      content="Let's discuss the home renovation project",
      metadata={
        "current_workspace": workspace_content
      },
    )

    async for response in analyst.process(
      new_topic_msg
    ):
      # Should indicate new episode
      # Should not maintain previous context
      # Should call clear_workspace tool
      print(f"New topic response: {response.content}")
      print(f"Metadata: {response.metadata}")
