# Chapter 4 TODO

## Streaming intermediate results

This is a great observation about the need for better streaming and visibility of intermediate results. Let's design a more comprehensive approach that distinguishes between different types of outputs while maintaining clean streaming throughout the call chain.

Here's a proposed design:

1. First, let's enhance our `Response` model to better distinguish message types:

```python
from enum import StrEnum
from typing import Any

class ResponseType(StrEnum):
    """Types of responses in the system."""
    USER = "user_message"      # Final output for user
    TOOL_RESULT = "tool_result"        # Tool execution results
    INTERNAL_STEP = "internal_step"    # Internal processing steps
    DEBUG_INFO = "debug_info"          # Debug/diagnostic information

class Response(BaseModel):
    """Enhanced response model."""
    content: str
    response_type: ResponseType
    step_name: str | None = None  # For grouping related steps
    metadata: dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
```

2. Then, let's create a context manager for handling steps consistently:

```python
from contextlib import asynccontextmanager

class ProcessingStep:
    """Context manager for handling processing steps."""
    def __init__(
        self,
        name: str,
        step_type: str,
        show_input: bool = True
    ):
        self.name = name
        self.step_type = step_type
        self.show_input = show_input
        self.cl_step = None

    async def __aenter__(self):
        self.cl_step = cl.Step(
            name=self.name,
            type=self.step_type,
            show_input=self.show_input
        )
        await self.cl_step.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cl_step.__aexit__(exc_type, exc_val, exc_tb)

    async def stream_response(self, response: Response):
        """Stream response based on its type."""
        if response.response_type == ResponseType.TOOL_RESULT:
            self.cl_step.input = response.metadata.get("tool_input")
            self.cl_step.output = response.content
        elif response.streaming:
            await self.cl_step.stream_token(response.content)
        else:
            self.cl_step.output = response.content
```

3. Now we can modify the Memory Coordinator to use these enhancements:

```python
async def process(self, message: Message) -> AsyncIterator[Response]:
    """Enhanced process method with better streaming and step visibility."""

    # Episode Analysis Step
    async with ProcessingStep("Episode Analysis", "analysis") as step:
        async for response in self.episode_analyst.process(message):
            if response.streaming:
                await step.stream_response(response)
            else:
                episode_analysis = json.loads(response.content)
                yield Response(
                    content=response.content,
                    response_type=ResponseType.INTERNAL_STEP,
                    step_name="episode_analysis",
                    metadata=episode_analysis
                )

    # Semantic Memory Operations
    async with ProcessingStep("Semantic Memory", "memory") as step:
        async for response in self.semantic_memory.process(message):
            await step.stream_response(response)
            if not response.streaming:
                yield Response(
                    content=response.content,
                    response_type=ResponseType.INTERNAL_STEP,
                    step_name="semantic_memory",
                    metadata=response.metadata
                )

    # Working Memory Update
    async with ProcessingStep("Working Memory", "memory") as step:
        async for response in self.working_memory.process(message):
            await step.stream_response(response)
            if not response.streaming:
                # Final user-facing response
                yield Response(
                    content=response.content,
                    response_type=ResponseType.USER,
                    metadata=response.metadata
                )
```

4. Update the Chainlit app to handle different response types:

```python
async def handle_message(self, message: cl.Message):
    """Enhanced message handler with step visualization."""
    system = cl.user_session.get("system")
    agent_id = cl.user_session.get("agent_id")

    async for response in system.invoke_conversation(
        agent_id,
        message.content,
        context={"history": cl.user_session.get("history", [])}
    ):
        if response.response_type == ResponseType.USER:
            if not hasattr(self, 'current_msg'):
                self.current_msg = cl.Message(content="")
                await self.current_msg.send()

            if response.streaming:
                await self.current_msg.stream_token(response.content)
            else:
                await self.current_msg.update()

        elif response.response_type == ResponseType.TOOL_RESULT:
            async with cl.Step(
                name=response.metadata.get("tool_name", "Tool"),
                type="tool"
            ) as step:
                step.input = response.metadata.get("tool_input")
                step.output = response.content
```

This design:

- Clearly distinguishes between user messages, tool results, and internal steps
- Provides consistent streaming throughout the call chain
- Uses Chainlit's Step feature to visualize internal processing
- Maintains clean separation of concerns
- Supports both streaming and non-streaming responses
- Preserves metadata and context throughout the process

The enhanced visibility helps with:

- Debugging and development
- Understanding the system's decision-making process
- Monitoring tool execution and results
- Tracking the flow of information through the system

Would you like me to expand on any part of this design or show how it would be implemented in other components of the system?

## Typed result handling and metadata variables

You're absolutely right about using Pydantic validation. Let's organize this properly:

1. New Files Structure:

```
src/winston/core/responses/
├── __init__.py           # Exports core types
├── models.py            # Response models and types
└── steps.py             # ProcessingStep implementation

tests/core/responses/
├── __init__.py
├── test_models.py       # Test response models
└── test_steps.py        # Test step handling
```

2. Places needing Pydantic validation in the memory system:

a) Memory Coordinator (`winston/core/memory/coordinator.py`):

```python
# Current:
episode_analysis = json.loads(response.content)

# Should be:
from winston.core.memory.episode_analyst import EpisodeBoundaryResponse
episode_analysis = EpisodeBoundaryResponse.model_validate_json(response.content)
```

b) Semantic Memory Coordinator (`winston/core/memory/semantic/coordinator.py`):

```python
# Current:
result = json.loads(response.content)

# Should be:
from winston.core.memory.semantic.retrieval import RetrieveKnowledgeResponse
from winston.core.memory.semantic.storage import StoreKnowledgeResponse

# For retrieval results
retrieval_result = RetrieveKnowledgeResponse.model_validate_json(response.content)

# For storage results
storage_result = StoreKnowledgeResponse.model_validate_json(response.content)
```

c) Working Memory Specialist (`winston/core/memory/working_memory.py`):

```python
# Current:
working_memory_results = json.loads(response.content)

# Should be:
from winston.core.memory.working_memory import WorkspaceUpdateResponse

class WorkspaceUpdateResponse(BaseModel):
    """Structured response from workspace updates."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

working_memory_results = WorkspaceUpdateResponse.model_validate_json(response.content)
```

Here's the implementation of the new response system:

```python
# src/winston/core/responses/models.py
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field

class ResponseType(StrEnum):
    """Types of responses in the system."""
    USER = "user_message"
    TOOL_RESULT = "tool_result"
    INTERNAL_STEP = "internal_step"
    DEBUG_INFO = "debug_info"

class Response(BaseModel):
    """Enhanced response model."""
    content: str
    response_type: ResponseType
    step_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
```

```python
# src/winston/core/responses/steps.py
from contextlib import asynccontextmanager
import chainlit as cl
from .models import Response, ResponseType

class ProcessingStep:
    """Context manager for handling processing steps."""
    def __init__(
        self,
        name: str,
        step_type: str,
        show_input: bool = True
    ):
        self.name = name
        self.step_type = step_type
        self.show_input = show_input
        self.cl_step = None

    async def __aenter__(self):
        self.cl_step = cl.Step(
            name=self.name,
            type=self.step_type,
            show_input=self.show_input
        )
        await self.cl_step.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cl_step.__aexit__(exc_type, exc_val, exc_tb)

    async def stream_response(self, response: Response):
        """Stream response based on its type."""
        if response.response_type == ResponseType.TOOL_RESULT:
            self.cl_step.input = response.metadata.get("tool_input")
            self.cl_step.output = response.content
        elif response.streaming:
            await self.cl_step.stream_token(response.content)
        else:
            self.cl_step.output = response.content
```

Updated Memory Coordinator example:

```python
# winston/core/memory/coordinator.py
from winston.core.responses.models import Response, ResponseType
from winston.core.responses.steps import ProcessingStep
from winston.core.memory.episode_analyst import EpisodeBoundaryResponse
from winston.core.memory.semantic.retrieval import RetrieveKnowledgeResponse
from winston.core.memory.semantic.storage import StoreKnowledgeResponse
from winston.core.memory.working_memory import WorkspaceUpdateResponse

async def process(self, message: Message) -> AsyncIterator[Response]:
    """Enhanced process method with better streaming and step visibility."""

    # Episode Analysis Step
    async with ProcessingStep("Episode Analysis", "analysis") as step:
        async for response in self.episode_analyst.process(message):
            if response.streaming:
                await step.stream_response(response)
            else:
                episode_analysis = EpisodeBoundaryResponse.model_validate_json(
                    response.content
                )
                yield Response(
                    content=response.content,
                    response_type=ResponseType.INTERNAL_STEP,
                    step_name="episode_analysis",
                    metadata=episode_analysis.model_dump()
                )

    # Similar updates for semantic memory and working memory...
```

This ensures type safety throughout the system while providing clear visualization of the processing steps. Would you like me to show the implementation for any other components or expand on the testing approach?

## Streaming tool handlers

Yes, we can modify the tool handling system to support streaming responses. This would require changes in several places:

1. First, update the `Tool` class in `winston/core/tools.py` to support streaming handlers:

```python
class Tool(BaseModel, Generic[T]):
    """Tool definition using Pydantic."""
    name: str = Field(..., description="Unique name for the tool")
    description: str = Field(..., description="Clear description for the LLM")
    handler: Callable[[Any], Awaitable[T] | AsyncIterator[T]] = Field(
        ...,
        description="Async function that implements the tool logic",
    )
    input_model: type[BaseModel] = Field(
        ...,
        description="Pydantic model defining the input parameters",
    )
    output_model: type[T] = Field(
        ...,
        description="Pydantic model defining the return type",
    )
    streams: bool = Field(
        default=False,
        description="Whether this tool streams its responses",
    )
```

2. Update the tool execution in `BaseAgent` (`winston/core/agent.py`):

```python
async def execute_tool(
    self, function_call: dict[str, Any]
) -> AsyncIterator[Response]:
    """Execute a tool call with streaming support."""
    try:
        tool_name = function_call["name"]
        tool = self.system.get_agent_tools(self.agent_id)[tool_name]

        # Parse arguments using input model
        args = tool.input_model.model_validate_json(
            function_call["arguments"]
        )

        # Execute handler
        result = await tool.handler(args)

        if tool.streams:
            # Handle streaming results
            async for item in result:
                if not isinstance(item, tool.output_model):
                    raise ValueError(
                        f"Handler yielded {type(item)}, expected {tool.output_model}"
                    )
                yield Response(
                    content=item.model_dump_json(),
                    metadata={
                        "tool_call": True,
                        "tool_name": tool_name,
                        "tool_args": function_call["arguments"],
                        "streaming": True,
                    },
                )
        else:
            # Handle single response
            if not isinstance(result, tool.output_model):
                raise ValueError(
                    f"Handler returned {type(result)}, expected {tool.output_model}"
                )
            yield Response(
                content=result.model_dump_json(),
                metadata={
                    "tool_call": True,
                    "tool_name": tool_name,
                    "tool_args": function_call["arguments"],
                },
            )

    except Exception as e:
        logger.exception(e)
        yield Response(
            content=f"Error executing function: {str(e)}",
            metadata={"error": True},
        )
```

3. Example of a streaming tool implementation:

```python
class WorkspaceUpdateResponse(BaseModel):
    """Workspace update response."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

async def handle_workspace_update(
    request: WorkspaceUpdateRequest,
) -> AsyncIterator[WorkspaceUpdateResponse]:
    """Handle workspace update with streaming."""
    # Initial analysis response
    yield WorkspaceUpdateResponse(
        content="Starting workspace analysis...",
        metadata={"phase": "analysis_start"},
    )

    # Generate the update prompt
    update_prompt = render_update_template(request)

    # Stream the LLM response
    async for response in generate_streaming_response(
        Message(
            content=update_prompt,
            metadata={"type": "Workspace Update"},
        )
    ):
        yield WorkspaceUpdateResponse(
            content=response.content,
            metadata={"phase": "update_generation", "streaming": True},
        )

    # Final completion response
    yield WorkspaceUpdateResponse(
        content="Workspace update complete",
        metadata={"phase": "complete"},
    )

# Register the streaming tool
workspace_update_tool = Tool(
    name="update_workspace",
    description="Update workspace content while maintaining context",
    handler=handle_workspace_update,
    input_model=WorkspaceUpdateRequest,
    output_model=WorkspaceUpdateResponse,
    streams=True,
)
```

4. Update the tool processing in the agent's process method:

```python
async def _process_tool_calls(
    self,
    tool_calls: list[ChatCompletionDeltaToolCall],
) -> AsyncIterator[Response]:
    """Process tool calls with streaming support."""
    for tool_call in tool_calls:
        async for response in self.execute_tool({
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }):
            yield response
```

This design:

- Allows tools to stream responses when needed
- Maintains type safety through Pydantic models
- Preserves the existing non-streaming tool functionality
- Provides clear metadata about streaming status
- Supports progressive updates during tool execution

The streaming capability is particularly useful for:

- Long-running operations
- Progressive updates
- Real-time feedback
- Complex processing that generates intermediate results

Would you like me to show how this would be implemented in other parts of the system or provide more examples?
