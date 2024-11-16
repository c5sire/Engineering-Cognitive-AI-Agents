import inspect
import json
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, cast

from prompt import CompletionConfig, PromptHandler
from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
  """Types of capabilities the system can create and manage"""

  TOOL = "tool"
  INSTRUCTION = "instruction"
  COMPOSITE = (
    "composite"  # New: Composed of other capabilities
  )


class CapabilityMetadata(BaseModel):
  """Metadata about a capability for tracking and management"""

  name: str
  type: CapabilityType
  version: str = Field(default="1.0.0")
  created_at: str  # ISO timestamp
  description: str
  inputs: list[str]
  outputs: list[str]
  performance_metrics: dict[str, float] = Field(
    default_factory=dict
  )
  usage_count: int = Field(default=0)
  success_rate: float = Field(default=0.0)
  dependencies: list[str] = Field(default_factory=list)
  tags: list[str] = Field(default_factory=list)


class CapabilityStore(BaseModel):
  """Persistent storage and management of capabilities"""

  tools: dict[
    str, tuple[Callable[..., Any], CapabilityMetadata]
  ]
  instructions: dict[
    str, tuple[str, CapabilityMetadata]
  ]
  composites: dict[
    str, tuple[list[str], CapabilityMetadata]
  ]

  def add_tool(
    self,
    name: str,
    func: Callable[..., Any],
    metadata: CapabilityMetadata,
  ) -> None:
    """Add a new tool with metadata"""
    self.tools[name] = (func, metadata)

  def add_instruction(
    self,
    name: str,
    template: str,
    metadata: CapabilityMetadata,
  ) -> None:
    """Add a new instruction with metadata"""
    self.instructions[name] = (template, metadata)

  def add_composite(
    self,
    name: str,
    components: list[str],
    metadata: CapabilityMetadata,
  ) -> None:
    """Add a new composite capability"""
    self.composites[name] = (components, metadata)

  def get_capability(
    self,
    name: str,
  ) -> tuple[Any, CapabilityMetadata] | None:
    """Retrieve a capability by name"""
    if name in self.tools:
      return self.tools[name]
    if name in self.instructions:
      return self.instructions[name]
    if name in self.composites:
      return self.composites[name]
    return None

  def update_metrics(
    self,
    name: str,
    execution_time: float,
    success: bool,
  ) -> None:
    """Update performance metrics for a capability"""
    capability = self.get_capability(name)
    if capability:
      _, metadata = capability
      metadata.usage_count += 1
      metadata.performance_metrics[
        "avg_execution_time"
      ] = (
        metadata.performance_metrics.get(
          "avg_execution_time", 0
        )
        * (metadata.usage_count - 1)
        + execution_time
      ) / metadata.usage_count
      metadata.success_rate = (
        metadata.success_rate
        * (metadata.usage_count - 1)
        + (1.0 if success else 0.0)
      ) / metadata.usage_count

  def get_all_capabilities(
    self,
  ) -> dict[
    str, tuple[CapabilityType, CapabilityMetadata]
  ]:
    """Gets all stored capabilities with their metadata"""
    capabilities: dict[
      str, tuple[CapabilityType, CapabilityMetadata]
    ] = {}

    # Add tools
    for name, (_, metadata) in self.tools.items():
      capabilities[name] = (
        CapabilityType.TOOL,
        metadata,
      )

    # Add instructions
    for name, (
      _,
      metadata,
    ) in self.instructions.items():
      capabilities[name] = (
        CapabilityType.INSTRUCTION,
        metadata,
      )

    # Add composites
    for name, (_, metadata) in self.composites.items():
      capabilities[name] = (
        CapabilityType.COMPOSITE,
        metadata,
      )

    return capabilities


class CapabilityTypeDecision(BaseModel):
  """Model for deciding capability implementation type"""

  capability_type: CapabilityType
  reasoning: str
  requirements: list[str]
  suggested_dependencies: list[str]
  performance_notes: str


class ToolSpecification(BaseModel):
  """Specification for a new tool implementation"""

  function_name: str
  description: str
  parameters: list[str]
  parameter_descriptions: list[str]
  return_type: str
  implementation: str
  test_cases: list[str]
  error_cases: list[str]
  dependencies: list[str] = Field(default_factory=list)


class InstructionSpecification(BaseModel):
  """Specification for a new instruction template"""

  template: str
  description: str
  required_inputs: list[str]
  example_outputs: list[str]
  validation_criteria: list[str]
  error_cases: list[str]
  dependencies: list[str] = Field(default_factory=list)


class CompositeSpecification(BaseModel):
  """Specification for a composite capability"""

  name: str
  description: str
  components: list[str]
  input_mappings: list[str]
  output_mappings: list[str]
  execution_order: list[str]
  error_cases: list[str]


class CapabilityManager:
  """Manages creation, optimization, and execution of capabilities"""

  def __init__(
    self,
    handler: PromptHandler,
    store: CapabilityStore,
  ):
    self.handler = handler
    self.store = store

  async def resolve_capability(
    self,
    name: str,
    context: dict[str, Any],
  ) -> tuple[CapabilityType, Any]:
    """Resolves a capability by finding or creating an appropriate implementation"""
    print(f"\nResolving capability: {name}")

    # Check if it exists
    existing = self.store.get_capability(name)
    if existing:
      print(f"Found existing capability: {name}")
      return existing[1].type, existing[0]

    # Determine best implementation approach
    print(
      f"Determining implementation approach for: {name}"
    )
    decision = await self._determine_capability_type(
      name, context
    )

    print(
      f"Will implement as {decision.capability_type}: {name}"
    )
    print(f"Reasoning: {decision.reasoning}")

    # Create appropriate implementation
    if decision.capability_type == CapabilityType.TOOL:
      return await self._create_tool(
        name, context, decision
      )
    elif (
      decision.capability_type
      == CapabilityType.INSTRUCTION
    ):
      return await self._create_instruction(
        name, context, decision
      )
    elif (
      decision.capability_type
      == CapabilityType.COMPOSITE
    ):
      return await self._create_composite(
        name, context, decision
      )
    else:
      raise ValueError(
        f"Unknown capability type: {decision.capability_type}"
      )

  async def execute_capability(
    self,
    name: str,
    inputs: dict[str, Any],
  ) -> Any:
    """Executes a capability with performance tracking"""
    capability = self.store.get_capability(name)
    if not capability:
      raise ValueError(f"Capability not found: {name}")

    start_time = time.time()
    success = True
    try:
      if capability[1].type == CapabilityType.TOOL:
        result = capability[0](**inputs)
      elif (
        capability[1].type
        == CapabilityType.INSTRUCTION
      ):
        template = cast(str, capability[0])
        result = self.handler.complete(
          template.format(**inputs),
          config=CompletionConfig(),
        )
      else:  # COMPOSITE
        result = await self._execute_composite(
          name, inputs
        )
      return result

    except Exception:
      success = False
      raise

    finally:
      execution_time = time.time() - start_time
      self.store.update_metrics(
        name, execution_time, success
      )

  async def optimize_capability(
    self,
    name: str,
  ) -> None:
    """Optimizes a capability based on usage patterns and performance"""
    capability = self.store.get_capability(name)
    if not capability:
      return

    _, metadata = capability

    # Skip if not enough usage data
    if metadata.usage_count < 100:
      return

    # Check if performance is poor
    if (
      metadata.success_rate < 0.95
      or metadata.performance_metrics.get(
        "avg_execution_time", 0
      )
      > 1.0
    ):
      # Generate optimization suggestions
      suggestions = (
        await self._generate_optimization_suggestions(
          name, metadata
        )
      )

      # Implement optimizations
      if metadata.type == CapabilityType.TOOL:
        await self._optimize_tool(name, suggestions)
      elif metadata.type == CapabilityType.INSTRUCTION:
        await self._optimize_instruction(
          name, suggestions
        )
      else:
        await self._optimize_composite(
          name, suggestions
        )

  async def _determine_capability_type(
    self,
    name: str,
    context: dict[str, Any],
  ) -> CapabilityTypeDecision:
    """Determines whether to implement as tool, instruction, or composite"""
    instruction = f"""
    Determine the most appropriate implementation type for this capability:

    Name: {name}
    Context: {json.dumps(context, indent=2)}

    Consider:
    1. Complexity and nature of the operation
    2. Need for external system interactions
    3. Data processing requirements
    4. Whether operation is primarily analytical or creative
    5. Performance requirements
    6. Potential for reuse and composition

    Available capability types:
    - TOOL: Python function for procedural/computational tasks
    - INSTRUCTION: Prompt template for creative/analytical tasks
    - COMPOSITE: Combination of existing capabilities

    Existing capabilities:
    Tools: {list(self.store.tools.keys())}
    Instructions: {list(self.store.instructions.keys())}
    Composites: {list(self.store.composites.keys())}

    Provide:
    1. The most appropriate capability type
    2. Reasoning for this choice
    3. List of specific requirements
    4. List of suggested dependencies (if any)
    5. Any performance considerations as a single text note
    """

    config = CompletionConfig(
      response_format=CapabilityTypeDecision,
      system_message="You are an expert system architect who understands the tradeoffs between different implementation approaches.",
    )

    return self.handler.complete(
      instruction, config=config
    )

  async def _create_tool(
    self,
    name: str,
    context: dict[str, Any],
    decision: CapabilityTypeDecision,
  ) -> tuple[CapabilityType, Any]:
    """Creates a new tool implementation"""
    print(
      f"\nCreating tool implementation for: {name}"
    )
    print(f"Requirements: {decision.requirements}")
    print(
      f"Dependencies: {decision.suggested_dependencies}"
    )

    instruction = f"""
    Create a Python function implementation for:

    Name: {name}
    Context: {json.dumps(context, indent=2)}
    Requirements: {json.dumps(decision.requirements, indent=2)}
    Dependencies: {json.dumps(decision.suggested_dependencies, indent=2)}
    Performance Considerations: {decision.performance_notes}

    Provide:
    1. Function name
    2. Description
    3. List of parameter names
    4. Description for each parameter (will be mapped to parameters)
    5. Return type
    6. Implementation code (use only standard library, no external dependencies)
    7. List of test case descriptions
    8. List of error case descriptions
    9. List of dependencies

    IMPORTANT: Use only Python standard library. Do not use external packages.
    Ensure all parameters are documented and the implementation includes proper type hints and error handling.
    """

    print("Generating tool specification...")
    config = CompletionConfig(
      response_format=ToolSpecification,
      system_message="You are an expert Python developer focused on creating reliable, well-tested functions using only the standard library.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )

    print(
      f"Creating function from specification: {spec.function_name}"
    )
    try:
      # Create function from specification
      namespace: dict[str, Any] = {}
      exec(spec.implementation, namespace)
      func = namespace[spec.function_name]
    except Exception as e:
      print(f"Error creating function: {str(e)}")
      print("Implementation:")
      print(spec.implementation)
      raise

    print("Adding tool to store...")
    # Add to store with metadata
    metadata = CapabilityMetadata(
      name=name,
      type=CapabilityType.TOOL,
      created_at=datetime.now().isoformat(),
      description=spec.description,
      inputs=spec.parameters,
      outputs=[spec.return_type],
      dependencies=spec.dependencies,
    )
    self.store.add_tool(name, func, metadata)
    print(f"Successfully created tool: {name}")

    return CapabilityType.TOOL, func

  async def _create_instruction(
    self,
    name: str,
    context: dict[str, Any],
    decision: CapabilityTypeDecision,
  ) -> tuple[CapabilityType, str]:
    """Creates a new instruction template"""
    instruction = f"""
    Create a prompt template for:

    Name: {name}
    Context: {json.dumps(context, indent=2)}
    Requirements: {json.dumps(decision.requirements, indent=2)}
    Dependencies: {json.dumps(decision.suggested_dependencies, indent=2)}

    The template should:
    1. Clearly specify required inputs
    2. Provide clear guidance for the model
    3. Include validation criteria
    4. Handle potential error cases
    5. Produce well-structured output
    """

    config = CompletionConfig(
      response_format=InstructionSpecification,
      system_message="You are an expert prompt engineer focused on creating clear, effective instructions.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )

    # Add to store with metadata
    metadata = CapabilityMetadata(
      name=name,
      type=CapabilityType.INSTRUCTION,
      created_at=datetime.now().isoformat(),
      description=spec.description,
      inputs=spec.required_inputs,
      outputs=spec.example_outputs,
      dependencies=spec.dependencies,
    )
    self.store.add_instruction(
      name, spec.template, metadata
    )

    return CapabilityType.INSTRUCTION, spec.template

  async def _create_composite(
    self,
    name: str,
    context: dict[str, Any],
    decision: CapabilityTypeDecision,
  ) -> tuple[CapabilityType, list[str]]:
    """Creates a new composite capability"""
    instruction = f"""
    Design a composite capability for:

    Name: {name}
    Context: {json.dumps(context, indent=2)}
    Requirements: {json.dumps(decision.requirements, indent=2)}
    Available Components: {json.dumps({
      "tools": list(self.store.tools.keys()),
      "instructions": list(self.store.instructions.keys()),
      "composites": list(self.store.composites.keys())
    }, indent=2)}

    Specify:
    1. Component capabilities to use
    2. Input/output mappings
    3. Execution order
    4. Error handling strategy
    """

    config = CompletionConfig(
      response_format=CompositeSpecification,
      system_message="You are an expert system integrator focused on combining capabilities effectively.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )

    # Validate all components exist
    for component in spec.components:
      if not self.store.get_capability(component):
        raise ValueError(
          f"Component not found: {component}"
        )

    # Add to store with metadata
    metadata = CapabilityMetadata(
      name=name,
      type=CapabilityType.COMPOSITE,
      created_at=datetime.now().isoformat(),
      description=spec.description,
      inputs=[
        m.split(":")[0] for m in spec.input_mappings
      ],
      outputs=[
        m.split(":")[0] for m in spec.output_mappings
      ],
      dependencies=spec.components,
    )
    self.store.add_composite(
      name, spec.components, metadata
    )

    return CapabilityType.COMPOSITE, spec.components

  async def _execute_composite(
    self,
    name: str,
    inputs: dict[str, Any],
  ) -> Any:
    """Executes a composite capability"""
    capability = self.store.get_capability(name)
    if (
      not capability
      or capability[1].type != CapabilityType.COMPOSITE
    ):
      raise ValueError(
        f"Composite capability not found: {name}"
      )

    components, metadata = capability
    results: dict[str, Any] = {}

    # Execute components in order
    for component in components:
      # Map inputs
      component_inputs = {
        k: inputs.get(v, results.get(v))
        for k, v in metadata.inputs
      }

      # Execute component
      result = await self.execute_capability(
        component, component_inputs
      )
      results[component] = result

    # Map final output
    return results[components[-1]]

  async def _generate_optimization_suggestions(
    self,
    name: str,
    metadata: CapabilityMetadata,
  ) -> list[str]:
    """Generates suggestions for optimizing a capability"""
    instruction = f"""
    Analyze this capability for potential optimizations:

    Name: {name}
    Type: {metadata.type}
    Performance Metrics: {json.dumps(metadata.performance_metrics, indent=2)}
    Success Rate: {metadata.success_rate}
    Usage Count: {metadata.usage_count}

    Consider:
    1. Performance bottlenecks
    2. Error patterns
    3. Resource usage
    4. Potential simplifications
    5. Caching opportunities
    """

    config = CompletionConfig(
      system_message="You are an expert performance engineer focused on optimization.",
    )

    response = self.handler.complete(
      instruction, config=config
    )
    return response.split("\n")

  async def _optimize_tool(
    self,
    name: str,
    suggestions: list[str],
  ) -> None:
    """Implements optimizations for a tool"""
    capability = self.store.tools.get(name)
    if not capability:
      return

    func, metadata = capability
    source = inspect.getsource(func)

    instruction = f"""
    Optimize this Python function based on these suggestions:

    Current Implementation:
    {source}

    Optimization Suggestions:
    {json.dumps(suggestions, indent=2)}

    Maintain:
    1. Original functionality
    2. Type safety
    3. Error handling
    4. Test coverage
    """

    config = CompletionConfig(
      response_format=ToolSpecification,
      system_message="You are an expert Python performance optimizer.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )

    # Update implementation
    namespace: dict[str, Any] = {}
    exec(spec.implementation, namespace)
    self.store.tools[name] = (
      namespace[spec.function_name],
      metadata,
    )

  async def _optimize_instruction(
    self,
    name: str,
    suggestions: list[str],
  ) -> None:
    """Implements optimizations for an instruction"""
    capability = self.store.instructions.get(name)
    if not capability:
      return

    template, metadata = capability

    instruction = f"""
    Optimize this instruction template based on these suggestions:

    Current Template:
    {template}

    Optimization Suggestions:
    {json.dumps(suggestions, indent=2)}

    Maintain:
    1. Clear guidance
    2. Required inputs
    3. Output structure
    4. Error handling
    """

    config = CompletionConfig(
      response_format=InstructionSpecification,
      system_message="You are an expert prompt optimization engineer.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )
    self.store.instructions[name] = (
      spec.template,
      metadata,
    )

  async def _optimize_composite(
    self,
    name: str,
    suggestions: list[str],
  ) -> None:
    """Implements optimizations for a composite capability"""
    capability = self.store.composites.get(name)
    if not capability:
      return

    components, metadata = capability

    instruction = f"""
    Optimize this composite capability based on these suggestions:

    Current Components: {json.dumps(components, indent=2)}
    Metadata: {json.dumps(metadata.dict(), indent=2)}

    Optimization Suggestions:
    {json.dumps(suggestions, indent=2)}

    Consider:
    1. Component ordering
    2. Parallel execution opportunities
    3. Input/output optimizations
    4. Error handling improvements
    """

    config = CompletionConfig(
      response_format=CompositeSpecification,
      system_message="You are an expert system integration optimizer.",
    )

    spec = self.handler.complete(
      instruction, config=config
    )
    self.store.composites[name] = (
      spec.components,
      metadata,
    )
