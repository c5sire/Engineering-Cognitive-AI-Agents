# src/winston/core/workspace.py
from pathlib import Path
from textwrap import dedent
from typing import ClassVar, Union

from jinja2 import Template
from loguru import logger

from winston.core.messages import Message
from winston.core.protocols import Agent

DEFAULT_INITIAL_TEMPLATE = dedent("""
# Cognitive Workspace

## User Preferences

## Recent Interactions

## Current Context

## Working Memory
""").strip()


DEFAULT_UPDATE_TEMPLATE = dedent("""
Update the `Current Workspace` to incorporate the new information effectively.

# Steps

1. **Integrate Key Insights**: Extract critical insights and relationships from `Update Content` and integrate them into the existing workspace.
2. **Specific Updates**:
   - If `Message Type` is "Interaction": Update the relevant workspace sections, such as context and preferences.
   - For other types, create or append sections as necessary to reflect the new content appropriately (i.e., if it is new, retain all details as provided)
3. **Preserve Structure**: Maintain existing sections, formatting, and overall organization to ensure clarity and conciseness.

# Output Format

Provide the updated workspace in a clear and structured markdown format, maintaining existing layout and introducing necessary modifications or additions to incorporate new content.

# Examples

**Example 1**

- **Input**:
  - `msg_type`: "Interaction"
  - `content_format`: "User interacted with new features."
  - `workspace`: "## Existing Context..."

- **Output**:
  - `workspace`: "## Existing Context... \n\n### Updated Features...\n- User interacted with new features...\n"

(Note: realistic examples should include detailed content and integration reflecting the actual updates for clarity and completeness.)

Message Type: {{ msg_type }}
Update Content:
```markdown
{{ content_format }}
```

Current Workspace:
```markdown
{{ workspace }}
```
""").strip()


class WorkspaceManager:
  """Singleton manager for all workspaces."""

  _instance: ClassVar[
    Union["WorkspaceManager", None]
  ] = None

  def __new__(cls) -> "WorkspaceManager":
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._initialize()
    return cls._instance

  def _initialize(self) -> None:
    """Initialize the workspace manager."""
    logger.trace(
      "Initializing new WorkspaceManager instance"
    )
    self._workspaces: dict[Path, str] = {}

  def initialize_workspace(
    self,
    workspace_path: Path,
    content: str | None = None,
  ) -> None:
    """Initialize a workspace if it doesn't exist."""
    if not workspace_path.exists():
      logger.info(
        f"Creating new workspace at {workspace_path}"
      )
      workspace_path.parent.mkdir(
        parents=True, exist_ok=True
      )
      try:
        workspace_path.write_text(
          content or DEFAULT_INITIAL_TEMPLATE
        )
        logger.debug(
          f"Workspace initialized successfully at {workspace_path}"
        )
      except Exception as e:
        logger.exception(
          f"Failed to create workspace at {workspace_path}: {e}"
        )
        raise

  def load_workspace(
    self,
    workspace_path: Path,
  ) -> str:
    """Load workspace content, initializing if needed."""
    logger.debug(
      f"Loading workspace from {workspace_path}"
    )
    if workspace_path not in self._workspaces:
      try:
        self.initialize_workspace(workspace_path)
        self._workspaces[workspace_path] = (
          workspace_path.read_text()
        )
        logger.trace(
          f"Workspace loaded and cached: {workspace_path}"
        )
      except Exception as e:
        logger.exception(
          f"Failed to load workspace {workspace_path}: {e}"
        )
        raise
    return self._workspaces[workspace_path]

  def save_workspace(
    self,
    workspace_path: Path,
    content: str,
  ) -> None:
    """Save workspace content."""
    logger.debug(
      f"Saving workspace to {workspace_path}"
    )
    try:
      workspace_path.write_text(content)
      self._workspaces[workspace_path] = content
      logger.trace(
        f"Workspace saved successfully: {workspace_path}"
      )
    except Exception as e:
      logger.exception(
        f"Failed to save workspace {workspace_path}: {e}"
      )
      raise

  async def update_workspace(
    self,
    workspace_path: Path,
    message: Message,
    agent: Agent,
    update_template: str | None = None,
    update_category: str | None = None,
  ) -> str:
    """Update workspace with new interaction."""
    logger.info(
      f"Updating workspace: {workspace_path}"
    )
    logger.debug(
      f"Update category: {update_category}, Message: {message}"
    )

    try:
      workspace = self.load_workspace(workspace_path)

      if update_category is None:
        update_category = "Interaction"

      content_format = (
        message.content
        if update_category != "Interaction"
        else f"User: {message.content}"
      )

      template = Template(
        update_template or DEFAULT_UPDATE_TEMPLATE
      )
      update_prompt = template.render(
        msg_type=update_category.replace("_", " "),
        content_format=content_format,
        workspace=workspace,
      ).strip()

      logger.trace(
        f"Generated update prompt: {update_prompt}"
      )

      response = await agent.generate_response(
        Message(
          content=update_prompt,
          metadata=message.metadata,
        )
      )

      self.save_workspace(
        workspace_path, response.content
      )
      logger.debug(
        "Workspace update completed successfully"
      )
      return response.content

    except Exception as e:
      logger.exception(
        f"Failed to update workspace {workspace_path}: {e}"
      )
      raise
