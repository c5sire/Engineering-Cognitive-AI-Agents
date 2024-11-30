"""Working memory specialist agent."""

from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.messages import Message
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class WorkspaceUpdateRequest(BaseModel):
  """Parameters for workspace update."""

  content: str = Field(
    description="New content to integrate"
  )
  current_workspace: str = Field(
    description="Current workspace content"
  )
  retrieved_context: str | None = Field(
    default=None,
    description="Optional retrieved context to incorporate",
  )


class WorkspaceUpdateResponse(BaseModel):
  """Response from workspace update."""

  updated_content: str = Field(
    description="Updated workspace content"
  )
  sections_modified: list[str] = Field(
    default_factory=list,
    description="Workspace sections that were modified",
  )


class WorkingMemorySpecialist(BaseAgent):
  """Specialist agent for working memory maintenance."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    # Register the update tool
    self.system.register_tool(
      Tool(
        name="update_workspace",
        description="Update workspace content while maintaining context",
        handler=self._handle_workspace_update,
        input_model=WorkspaceUpdateRequest,
        output_model=WorkspaceUpdateResponse,
      )
    )

    # Grant self access
    self.system.grant_tool_access(
      self.id, ["update_workspace"]
    )

  async def _handle_workspace_update(
    self, request: WorkspaceUpdateRequest
  ) -> WorkspaceUpdateResponse:
    """Handle workspace update request."""

    print(
      f"Updating workspace with content: {request.content}"
    )
    print(
      f"Retrieved context: {request.retrieved_context}"
    )

    # Construct prompt for content integration
    update_prompt = f"""
        Update this workspace content:
        ```markdown
        {request.current_workspace}
        ```

        With this new information:
        {request.content}

        {f'''
        And incorporate this relevant context:
        {request.retrieved_context}
        ''' if request.retrieved_context else ''}

        Follow these guidelines:
        1. Maintain clear section structure
        2. Preserve important relationships and context
        3. Note significant changes
        4. Keep information organized and focused

        Provide the complete updated workspace content in markdown format.
        """

    # Generate updated content
    response = await self.generate_response(
      Message(
        content=update_prompt,
        metadata={"type": "Workspace Update"},
      )
    )

    # Track which sections were modified (could parse markdown)
    sections_modified = []  # TODO: Parse sections from diff

    return WorkspaceUpdateResponse(
      updated_content=response.content,
      sections_modified=sections_modified,
    )
