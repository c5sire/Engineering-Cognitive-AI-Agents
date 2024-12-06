"""Working Memory Specialist: Expert agent for cognitive workspace maintenance.

The Working Memory Specialist manages Winston's immediate cognitive context,
serving as the bridge between current interactions and long-term knowledge.
Unlike simple storage, working memory requires sophisticated organization and
maintenance of active cognitive elements - from current conversation topics
to emerging patterns and relationships.

Architecture Overview:
```mermaid
graph TD
    WM[Working Memory Specialist]

    subgraph "Input Analysis"
        NI[New Information] -->|Analyze| RA[Relevance Analysis]
        EC[Existing Context] -->|Consider| RA
        RC[Retrieved Context] -->|Integrate| RA
    end

    subgraph "Workspace Management"
        CE[Current Episode]
        UM[User Model]
        TP[Temporal Patterns]
        CW[Context Web]
        WA[Working Analysis]
        HC[Historical Context]
        MN[Metacognitive Notes]
    end

    RA -->|Updates| CE
    RA -->|Refines| UM
    RA -->|Adjusts| TP
    RA -->|Maintains| CW
    RA -->|Informs| WA
    RA -->|Archives| HC
    RA -->|Records| MN

    WM -->|Manages| CE
    WM -->|Maintains| UM
    WM -->|Tracks| TP
    WM -->|Weaves| CW
    WM -->|Develops| WA
    WM -->|Preserves| HC
    WM -->|Updates| MN
```

Design Philosophy:
Working memory in cognitive architectures mirrors human cognitive processes,
maintaining not just current information but organizing it into a coherent
understanding that supports ongoing operations. The Working Memory Specialist
manages this dynamic cognitive workspace through several key functions:

1. Current Episode Management
   - Active topic/context tracking
   - Recent interactions and insights
   - Emerging patterns and preferences
   - Immediate concerns or questions

2. User Model Maintenance
   - Current preferences and patterns
   - Recent behavioral observations
   - Important relationships
   - Key characteristics

3. Temporal Pattern Tracking
   - Time-based preferences
   - Regular patterns
   - Recent changes
   - Historical progression

4. Context Web Management
   - Active relationships between concepts
   - Important connections
   - Recent context shifts
   - Relevant background

5. Working Analysis Development
   - Current understanding
   - Ongoing evaluations
   - Emerging hypotheses
   - Points needing clarification

Example Operations:

1. Preference Update
   Input: "I've switched to tea"
   Operations:
   - Update current preferences
   - Note temporal change
   - Maintain routine context
   - Track preference evolution

2. Pattern Recognition
   Input: "Like most mornings lately..."
   Operations:
   - Identify temporal pattern
   - Link to existing routines
   - Update user model
   - Note behavioral consistency

3. Context Integration
   Input: Retrieved knowledge about past preferences
   Operations:
   - Integrate with current context
   - Maintain temporal relationships
   - Update user model
   - Note pattern evolution

Key Architectural Principles:

1. Cognitive Utility
   - Organize for understanding
   - Support reasoning processes
   - Enable future interactions
   - Preserve context awareness

2. Knowledge Evolution
   - Track preference changes
   - Note pattern developments
   - Maintain change history
   - Preserve important transitions

3. Relationship Preservation
   - Between concepts
   - Across time
   - With context
   - Through changes

4. Operational Support
   - Enable pattern recognition
   - Support reasoning
   - Aid planning
   - Guide interactions

Implementation Note:
While the Working Memory Specialist makes sophisticated decisions about
context organization and maintenance, it operates within clear architectural
boundaries. It focuses purely on workspace management, leaving long-term
storage to semantic memory specialists and episode boundary detection to
the episode analyst. This separation of concerns ensures clean cognitive
architecture while enabling complex memory operations.
"""

from loguru import logger
from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.paths import AgentPaths
from winston.core.system import AgentSystem
from winston.core.tools import Tool


class WorkspaceUpdateResult(BaseModel):
  """Parameters for workspace update."""

  updated_workspace: str = Field(
    description="Full, complete, final, consolidated, comprehensive updated workspace"
  )
  rationale: str = Field(
    description="Rationale for the updates you made to the workspace"
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

    logger.info(
      "Initializing WorkingMemorySpecialist."
    )
    # Register the update tool
    tool = Tool(
      name="update_workspace",
      description="Update workspace content while maintaining context",
      handler=self._handle_workspace_update,
      input_model=WorkspaceUpdateResult,
      output_model=WorkspaceUpdateResult,
    )
    self.system.register_tool(tool)

    # Grant self access
    self.system.grant_tool_access(self.id, [tool.name])

    logger.info(
      "WorkingMemorySpecialist initialization complete"
    )

  async def _handle_workspace_update(
    self, result: WorkspaceUpdateResult
  ) -> WorkspaceUpdateResult:
    """Handle workspace update request."""

    logger.trace(
      f"Received workspace update: {result}"
    )
    return result
