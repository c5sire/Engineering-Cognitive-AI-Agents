"""Episode analysis specialist agent."""

from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.messages import Response
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class ClearWorkspaceRequest(BaseModel):
  """Parameters for clearing workspace."""

  preserve_elements: list[str] = Field(
    default_factory=list,
    description="Elements to preserve when clearing",
  )


class UpdateContentRequest(BaseModel):
  """Parameters for updating workspace content."""

  updates: dict[str, str] = Field(
    description="Content updates to apply"
  )
  preserve_context: list[str] = Field(
    description="Context elements to preserve"
  )


class RetrievalRequest(BaseModel):
  """Parameters for knowledge retrieval."""

  query_topics: list[str] = Field(
    description="Topics to retrieve information about"
  )


class EpisodeAnalyst(BaseAgent):
  """Specialist agent for episode and context analysis."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    # Register tools
    self.system.register_tool(
      Tool(
        name="clear_workspace",
        description="Clear workspace while optionally preserving elements",
        handler=self._handle_clear_workspace,
        input_model=ClearWorkspaceRequest,
        output_model=Response,
      )
    )

    self.system.register_tool(
      Tool(
        name="update_content",
        description="Update workspace content while preserving relationships",
        handler=self._handle_update_content,
        input_model=UpdateContentRequest,
        output_model=Response,
      )
    )

    self.system.register_tool(
      Tool(
        name="request_retrieval",
        description="Request additional context from long-term memory",
        handler=self._handle_retrieval,
        input_model=RetrievalRequest,
        output_model=Response,
      )
    )

    # Grant self access to tools
    self.system.grant_tool_access(
      self.id,
      [
        "clear_workspace",
        "update_content",
        "request_retrieval",
      ],
    )

  async def _handle_clear_workspace(
    self, request: ClearWorkspaceRequest
  ) -> Response:
    """Handle workspace clearing."""
    # Implementation will interact with WorkspaceManager
    return Response(
      content="Workspace cleared",
      metadata={
        "preserved": request.preserve_elements
      },
    )

  async def _handle_update_content(
    self, request: UpdateContentRequest
  ) -> Response:
    """Handle content updates."""
    # Implementation will update workspace while maintaining context
    return Response(
      content="Content updated",
      metadata={
        "updates": request.updates,
        "preserved_context": request.preserve_context,
      },
    )

  async def _handle_retrieval(
    self, request: RetrievalRequest
  ) -> Response:
    """Handle retrieval requests."""
    # Implementation will use memory coordinator for retrieval
    return Response(
      content="Retrieved information",
      metadata={"topics": request.query_topics},
    )
