"""Episode Analyst: Specialist agent for cognitive episode boundary detection.

The Episode Analyst plays a crucial role in Winston's memory system by detecting
natural boundaries in cognitive context. Just as humans naturally segment experiences
into distinct episodes, this specialist analyzes interactions to determine when
context shifts occur and what information should be preserved across these boundaries.

Architecture Overview:
```mermaid
graph TD
    EA[Episode Analyst] -->|Analyzes| I[Input Message]
    EA -->|Considers| C[Current Context]

    subgraph "Analysis Process"
        I -->|Topic Shift?| TS[Topic Analysis]
        I -->|Context Change?| CC[Context Analysis]
        I -->|Time Pattern?| TP[Temporal Analysis]

        TS --> D[Decision Making]
        CC --> D
        TP --> D
    end

    D -->|Determines| NE[New Episode?]
    D -->|Identifies| PC[Preserve Context]

    NE -->|Signals| MC[Memory Coordinator]
    PC -->|Informs| WM[Working Memory]
```

Design Philosophy:
The Episode Analyst addresses a fundamental challenge in cognitive architectures:
determining when to maintain current context versus starting fresh. This mirrors
human cognitive processes where we naturally segment experiences into distinct
episodes while maintaining relevant connections between them.

Example Scenarios:

1. Preference Update
   Input: "Actually, I've switched to tea"
   Context: Previous discussion about coffee preferences
   Analysis: Same episode (beverage preferences)
   Preserve: Morning routine context, family patterns

2. Topic Shift
   Input: "Let's discuss the home renovation project"
   Context: Previous beverage preference discussion
   Analysis: New episode (complete context shift)
   Preserve: None (clean context break)

3. Related Shift
   Input: "Speaking of morning routines, I've started exercising"
   Context: Previous discussion about morning beverages
   Analysis: Partial new episode
   Preserve: Temporal context (morning patterns)

Key Architectural Principles:
- Focus purely on episode detection (single responsibility)
- Provide clear boundary signals to memory coordinator
- Identify critical context to preserve
- Support natural cognitive segmentation

The specialist's system prompt guides the LLM to:
1. Analyze message content against current context
2. Identify significant context shifts
3. Determine what context remains relevant
4. Signal clear episode boundaries

This design enables sophisticated context management while maintaining:
- Clean separation of concerns
- Clear cognitive boundaries
- Natural context preservation
- Flexible episode detection

Implementation Note:
While the Episode Analyst makes sophisticated cognitive decisions about
context boundaries, it performs no direct memory operations. It simply
signals its analysis to the Memory Coordinator, which orchestrates any
necessary memory updates. This separation of concerns ensures the analyst
can focus purely on its cognitive role while leaving memory management
to appropriate specialists.
"""

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
