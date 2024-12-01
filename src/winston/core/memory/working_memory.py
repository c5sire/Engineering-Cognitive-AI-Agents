"""Working memory specialist agent."""

from jinja2 import Template
from loguru import logger
from pydantic import BaseModel, Field

from winston.core.agent import BaseAgent
from winston.core.agent_config import AgentConfig
from winston.core.messages import Message, Response
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
    self.system.register_tool(
      Tool(
        name="update_workspace",
        description="Update workspace content while maintaining context",
        handler=self._handle_workspace_update,
        input_model=WorkspaceUpdateRequest,
        output_model=Response,
      )
    )

    logger.debug("Registered update_workspace tool.")
    # Grant self access
    self.system.grant_tool_access(
      self.id, ["update_workspace"]
    )
    logger.info(
      "Granted tool access for update_workspace."
    )

  async def _handle_workspace_update(
    self, request: WorkspaceUpdateRequest
  ) -> Response:
    """Handle workspace update request."""

    logger.trace("Received workspace update request.")
    logger.info(
      f"Updating workspace with content: {request.content}"
    )
    logger.debug(
      f"Retrieved context: {request.retrieved_context}"
    )

    # Define the Jinja2 template for the update prompt
    update_prompt_template = Template("""
Your task is to generate a COMPLETE updated workspace that will serve as YOUR cognitive working memory. This is not just about recording information - it's about organizing YOUR understanding in a way that supports YOUR cognitive operations.

NEW CONTENT TO INTEGRATE:
{{ content }}

CURRENT WORKSPACE:
{{ current_workspace }}

{% if retrieved_context %}
RETRIEVED CONTEXT:
{{ retrieved_context }}
{% endif %}

IMPORTANT: Generate the ENTIRE workspace content in markdown format, structured to support YOUR cognitive processes.

WORKSPACE ORGANIZATION:

## Current Episode
- Active topic/context
- Recent interactions and insights
- Emerging patterns and preferences
- Immediate concerns or open questions

## User Model
- Current preferences and patterns
- Recent behavioral observations
- Important relationships
- Key characteristics

## Temporal Patterns
- Time-based preferences
- Regular patterns
- Recent changes
- Historical progression

## Context Web
- Active relationships between concepts
- Important connections
- Recent context shifts
- Relevant background

## Working Analysis
- Current understanding
- Ongoing evaluations
- Emerging hypotheses
- Points needing clarification

## Historical Context
- Significant past events
- Previous episodes (when relevant)
- Pattern evolution
- Important changes

## Metacognitive Notes
- Interaction effectiveness
- Understanding gaps
- Successful approaches
- Areas needing attention

CRITICAL REQUIREMENTS:

1. Maintain Cognitive Utility
   - Organize for YOUR understanding
   - Support YOUR reasoning processes
   - Enable YOUR future interactions
   - Preserve YOUR context awareness

2. Show Knowledge Evolution
   - Track preference changes
   - Note pattern developments
   - Maintain change history
   - Preserve important transitions

3. Preserve Relationships
   - Between concepts
   - Across time
   - With context
   - Through changes

4. Support Future Operations
   - Enable pattern recognition
   - Support reasoning
   - Aid planning
   - Guide interactions

EXAMPLE FORMAT:
```markdown
## Current Episode
- User now prefers tea in the morning
- Recently switched from coffee
- Shows openness to change
- Maintains morning routine structure

## User Model
- Values morning beverage routine
- Open to changing preferences
- Maintains family traditions
- Appreciates temporal structure

## Temporal Patterns
- Morning beverage ritual important
- Recent shift: coffee → tea
- Consistent early timing
- Pattern stability despite change

## Context Web
- Morning routine → Family tradition
- Beverage choice → Time of day
- Change pattern → Adaptation capacity
- Routine importance → Stability value

[Continue with other sections...]
```

REMEMBER:
- This is YOUR working memory
- YOU need this for cognitive operations
- Include ALL relevant sections
- Maintain COMPLETE context
- Preserve ALL important relationships

Generate the FULL, COMPLETE workspace content now, incorporating all sections and maintaining all important context and relationships.
""")

    # Render the template with the request data
    update_prompt = update_prompt_template.render(
      content=request.content,
      current_workspace=request.current_workspace,
      retrieved_context=request.retrieved_context,
    )

    return await self.generate_response(
      Message(
        content=update_prompt,
        metadata={"type": "Workspace Update"},
      )
    )
