# Agent Implementation Guidelines

## 1. Core Design Philosophy

The fundamental principle of specialist agent design is that **cognitive logic lives in the prompt**. The language model, guided by a carefully crafted system prompt, performs all analysis and decision-making. Tools simply represent the concrete actions available based on that reasoning.

## 2. Core Requirements

```python
class SpecialistAgent(BaseAgent):
    """Each specialist agent must:
    1. Inherit from BaseAgent
    2. Have a clear, focused cognitive role
    3. Provide configuration YAML
    4. Maintain clean separation of concerns
    """
```

## 3. Configuration (YAML)

```yaml
# required: config/agents/{agent_id}.yaml
id: agent_id # Unique identifier
model: gpt-4 # Model to use
system_prompt: | # Core intelligence
  You are a {SPECIALIST} agent in a Society of Mind system.

  Your ONLY role is to {SPECIFIC_COGNITIVE_FUNCTION}.

  Given input, analyze:
  1. {KEY_ANALYSIS_POINTS}
  2. {DECISION_CRITERIA}

  Based on your analysis, select the appropriate action:
  - Use tool_a when {CONDITION_A}
  - Use tool_b when {CONDITION_B}

  Always explain your reasoning before taking action.

temperature: 0.7 # Optional parameters
stream: true
```

## 4. Tool Implementation

```python
from pydantic import BaseModel, Field
from winston.core.tools import Tool

# A. Define Request/Response Models
class ActionRequest(BaseModel):
    """Each tool needs input validation."""
    content: str = Field(
        description="Content to analyze"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

class ActionResponse(BaseModel):
    """And structured output."""
    result: str = Field(
        description="Action results"
    )
    metadata: dict = Field(
        description="Additional metadata"
    )

# B. Implement Tool Handler(s)
async def handle_action(
    request: ActionRequest
) -> ActionResponse:
    """Implement concrete action logic only."""
    # ... tool implementation
    return ActionResponse(
        result="Action completed",
        metadata={"status": "success"}
    )

# C. Optional: Format Results
def format_result(
    result: ActionResponse
) -> str:
    """Format tool results for user display."""
    return f"Result: {result.result}"

# D. Create Tool Instance
action_tool = Tool(
    name="perform_action",
    description="Execute specific action based on LLM analysis",
    handler=handle_action,
    input_model=ActionRequest,
    output_model=ActionResponse,
    format_response=format_result  # Optional
)
```

## 5. Agent Implementation

```python
class SpecialistAgent(BaseAgent):
    """Each specialist performs a specific cognitive function."""

    def __init__(
        self,
        system: System,
        config: AgentConfig,
        paths: AgentPaths,
    ) -> None:
        super().__init__(system, config, paths)

        # Tools represent possible actions based on LLM reasoning
        self.system.register_tool(Tool(
            name="action_a",
            description="Take action A when analysis indicates X",
            handler=self._handle_action_a,
            input_model=ActionARequest,
            output_model=ActionAResponse
        ))

        self.system.register_tool(Tool(
            name="action_b",
            description="Take action B when analysis indicates Y",
            handler=self._handle_action_b,
            input_model=ActionBRequest,
            output_model=ActionBResponse
        ))

        # Grant self access to own tools
        self.system.grant_tool_access(
            self.id,
            ["action_a", "action_b"]
        )
```

## 6. Separation of Concerns

1. **System Prompt**

   - Defines the cognitive role and decision process
   - Specifies analysis criteria and decision logic
   - Establishes conditions for tool usage
   - Contains all reasoning patterns

2. **Language Model**

   - Performs all analysis and reasoning
   - Makes decisions based on prompt guidance
   - Explains reasoning before taking action
   - Selects appropriate tools based on analysis

3. **Tools**

   - Execute concrete actions only
   - Implement no decision logic
   - Process structured inputs
   - Return structured outputs
   - Format results for display (optional)

4. **Agent Class**
   - Inherits from BaseAgent
   - Registers available tools
   - Grants tool access
   - Provides no additional logic

The BaseAgent handles all core functionality:

- Message processing
- LLM interaction
- Tool execution
- Response streaming

This clean separation ensures that cognitive logic remains in the prompt/LLM layer while tools serve purely as action executors based on that reasoning.

## Anti-Pattern: "RPC-like" Control Flow

❌ **NEVER** use metadata or message fields to direct agent behavior like this:

```python
# WRONG: Using metadata as RPC-like control flow
message = Message(
    content="Some content",
    metadata={
        "type": "do_analysis",  # NO! This is not how we direct behavior
        "operation": "store",   # NO! This is not how we select actions
        "mode": "quick"        # NO! This is not how we control processing
    }
)
```

This approach:

- Bypasses the LLM's cognitive capabilities
- Creates brittle, procedural control flow
- Violates the core architectural principle that cognitive logic belongs in the prompt

## Key Principles

✅ **CRITICAL**: Cognitive Logic Lives in the Prompt

1. **Cognitive Logic in Prompt**

   - The system prompt defines the agent's cognitive role
   - LLM reasoning determines appropriate actions
   - Tools represent conclusions/decisions, not control flow

2. **Tools as Decision Points**

   - Each tool represents a possible conclusion from analysis
   - Tool parameters capture the details of the decision
   - Multiple tools represent distinct cognitive branches

3. **Clean Separation**
   - Prompt: Contains ALL cognitive logic and decision criteria
   - LLM: Performs analysis and selects appropriate action
   - Tools: Execute concrete actions based on LLM decisions
   - Metadata: Carries context, NOT control flow

## Remember

- If you find yourself using metadata or messages to control agent behavior, STOP
- If you're creating complex control flows between agents, STOP
- Return to the core principle: Cognitive logic lives in the prompt
- Let the LLM's reasoning, guided by the prompt, drive tool selection and action

This architectural principle ensures our agents remain truly cognitive rather than degrading into procedural RPC endpoints.
