"""Test working memory specialist functionality."""

import json
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from winston.core.agent import AgentConfig
from winston.core.memory.working_memory import (
  WorkingMemorySpecialist,
)
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem


@pytest.mark.asyncio
async def test_working_memory_operations():
  """Test working memory update and integration."""
  logger.info(
    "Starting test_working_memory_operations"
  )

  # Setup
  with tempfile.TemporaryDirectory() as temp_dir:
    logger.debug(
      f"Temporary directory created at: {temp_dir}"
    )
    temp_root = Path(temp_dir)
    project_root = Path(__file__).parent.parent.parent
    paths = AgentPaths(
      root=temp_root,
      system_root=project_root,
    )
    logger.debug(
      f"Agent paths set up with root: {temp_root} and system root: {project_root}"
    )

    system = AgentSystem()
    config = AgentConfig.from_yaml(
      paths.system_agents_config
      / "working_memory.yaml"
    )
    logger.info(
      "Agent system and configuration loaded."
    )

    specialist = WorkingMemorySpecialist(
      system, config, paths
    )
    logger.info("WorkingMemorySpecialist initialized.")

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
    logger.debug("Initial workspace content set.")

    # Test 1: Basic Update
    update_msg = Message(
      content="Actually, I've switched to tea",
      metadata={
        "current_workspace": workspace_content,
        "relevance_query": "morning beverage preferences",
      },
    )
    logger.debug("Test 1 message created.")

  async for response in specialist.process(update_msg):
    logger.debug("Processing update message...")
    updated = response.content
    logger.debug(f"Response received: {updated}")

    # Parse the JSON response to get workspace content
    response_data = json.loads(updated)
    workspace_content = response_data[
      "updated_content"
    ]

    # Now assert against the actual workspace content
    assert "tea" in workspace_content.lower()
    assert (
      "morning" in workspace_content.lower()
    )  # Maintains time context
    assert (
      "family" in workspace_content.lower()
    )  # Maintains family context
    assert (
      "father" in workspace_content.lower()
    )  # Notes change
    # Test 2: Update with Retrieved Context
    retrieved_context = """
        ## Historical Preferences
        Previously noted tea preferences:
        - Prefers herbal tea in evenings
        - Enjoys green tea after meals
        """
    logger.debug("Retrieved context set.")

    context_msg = Message(
      content="I like chamomile tea before bed",
      metadata={
        "current_workspace": workspace_content,
        "retrieved_context": retrieved_context,
        "relevance_query": "tea preferences and timing",
      },
    )
    logger.debug("Test 2 message created.")

    async for response in specialist.process(
      context_msg
    ):
      logger.debug("Processing context message...")
      updated = response.content
      logger.debug(f"Response received: {updated}")

      # Parse JSON response
      response_data = json.loads(updated)
      workspace_content = response_data[
        "updated_content"
      ]

      assert "chamomile" in workspace_content.lower()
      assert "before bed" in workspace_content.lower()
      assert (
        "herbal tea" in workspace_content.lower()
      )  # Incorporates relevant history
      assert (
        "morning" in workspace_content.lower()
      )  # Maintains morning context

    # Test 3: Workspace Organization
    complex_msg = Message(
      content="""
            I've been trying different teas:
            - Green tea in the morning (too bitter)
            - Chamomile at night (perfect)
            - Earl Grey with lunch (nice)
            """,
      metadata={
        "current_workspace": workspace_content,
        "relevance_query": "tea preferences and experiences",
      },
    )
    logger.debug("Test 3 message created.")

    async for response in specialist.process(
      complex_msg
    ):
      logger.debug("Processing complex message...")
      updated = response.content
      logger.debug(f"Response received: {updated}")

      # Parse JSON response
      response_data = json.loads(updated)
      workspace_content = response_data[
        "updated_content"
      ]

      assert any(
        section in workspace_content
        for section in [
          "Preferences",
          "Current Preferences",
        ]
      )  # Has some form of preferences section
      assert (
        "morning" in workspace_content.lower()
      )  # Time context
      assert (
        "tea" in workspace_content.lower()
      )  # Current preference
      assert (
        "perfect" in workspace_content.lower()
      )  # Includes evaluations
  logger.info(
    "Finished test_working_memory_operations"
  )
