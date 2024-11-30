# Agent Implementation Guidelines

## 1. Core Requirements

```python
class SpecialistAgent(BaseAgent):
    """Each specialist agent must:
    1. Inherit from BaseAgent
    2. Have a clear, focused cognitive role
    3. Provide configuration YAML
    """
```

## 2. Configuration (YAML)

```yaml
# required: config/agents/{agent_id}.yaml
id: agent_id # Unique identifier
model: gpt-4 # Model to use
system_prompt: | # Core intelligence
  You are a {SPECIALIST} agent in a Society of Mind system.
  Your role is to {SPECIFIC_RESPONSIBILITY}.

  For each input:
  1. {KEY_ANALYSIS_STEPS}
  2. {DECISION_CRITERIA}
  3. {ACTION_GUIDELINES}
temperature: 0.7 # Optional parameters
stream: true
```

## 3. Tool Implementation

```python
from pydantic import BaseModel, Field
from winston.core.tools import Tool

# A. Define Request/Response Models
class AnalyzeRequest(BaseModel):
    """Each tool needs input validation."""
    content: str = Field(
        description="Content to analyze"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

class AnalyzeResponse(BaseModel):
    """And structured output."""
    analysis: str = Field(
        description="Analysis results"
    )
    confidence: float = Field(
        description="Confidence score"
    )

# B. Implement Tool Handler
async def analyze_content(
    request: AnalyzeRequest
) -> AnalyzeResponse:
    """Implement the actual tool logic."""
    # ... tool implementation
    return AnalyzeResponse(
        analysis="Results...",
        confidence=0.95
    )

# C. Optional: Format Results
def format_analysis(
    result: AnalyzeResponse
) -> str:
    """Format tool results for user display."""
    return f"Analysis: {result.analysis} ({result.confidence:.0%} confident)"

# D. Create Tool Instance
analyze_tool = Tool(
    name="analyze_content",
    description="Analyze content with specific criteria",
    handler=analyze_content,
    input_model=AnalyzeRequest,
    output_model=AnalyzeResponse,
    format_response=format_analysis  # Optional
)

# E. Register in Agent
class SpecialistAgent(BaseAgent):
    def __init__(self, system: System, config: AgentConfig, paths: AgentPaths):
        super().__init__(system, config, paths)

        # Register tool with system
        system.register_tool(analyze_tool)

        # Grant this agent access
        system.grant_tool_access(
            config.id,
            ["analyze_content"]
        )
```

That's it! The BaseAgent handles everything else:

- Message processing
- LLM interaction
- Tool execution
- Response streaming
