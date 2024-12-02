"""Episode analysis specialist agent."""

from loguru import logger
from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.messages import Response
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class EpisodeBoundaryResponse(BaseModel):
  """Response from episode boundary detection."""

  is_new_episode: bool = Field(
    description="Whether this represents a new episode"
  )
  context_elements: list[str] = Field(
    default_factory=list,
    description="Context elements to preserve",
  )


class EpisodeAnalyst(BaseAgent):
  """Specialist agent for episode boundary detection."""

  def __init__(
    self,
    system: AgentSystem,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    super().__init__(system, config, paths)

    logger.info("Initializing EpisodeAnalyst agent.")

    # Register the action tool
    self.system.register_tool(
      Tool(
        name="report_episode_boundary",
        description="Report episode boundary detection results",
        handler=self._handle_boundary_report,
        input_model=EpisodeBoundaryResponse,
        output_model=Response,
      )
    )

    logger.debug(
      "Registered tool: report_episode_boundary"
    )

    # Grant self access
    self.system.grant_tool_access(
      self.id, ["report_episode_boundary"]
    )

    logger.info(
      "Granted tool access for report_episode_boundary."
    )

  async def _handle_boundary_report(
    self, report: EpisodeBoundaryResponse
  ) -> Response:
    """Handle the boundary detection report."""
    logger.trace(f"Handling boundary report: {report}")

    try:
      response = Response(
        content="Episode boundary analyzed",
        metadata={
          "is_new_episode": report.is_new_episode,
          "preserve_context": report.context_elements,
        },
      )
      logger.info(
        "Successfully analyzed episode boundary."
      )
      return response

    except Exception as e:
      logger.exception(
        f"Error while handling boundary report: {e}"
      )
      raise
