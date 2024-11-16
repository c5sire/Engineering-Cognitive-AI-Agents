import json
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from capabilities import (
  CapabilityManager,
  CapabilityMetadata,
  CapabilityStore,
  CapabilityType,
)
from matplotlib.figure import Figure
from matplotlib.text import Text
from networkx import DiGraph
from prompt import CompletionConfig, PromptHandler
from pydantic import BaseModel, Field


class PlanInput(BaseModel):
  """Input parameter for a task"""

  key: str = Field(
    ...,
    description="Parameter name required by the tool",
  )
  source_key: str = Field(
    ...,
    description="Key in working memory or output from another task",
  )
  description: str = Field(
    ...,
    description="Description of what this input represents",
  )


class TaskInput(BaseModel):
  """Represents an input to a task"""

  key: str  # The name of the input as expected by the task
  description: str
  source_key: str | None = Field(
    default=None,
    description="The name of the output/memory key that maps to this input",
  )


class Task(BaseModel):
  """Declarative task definition"""

  name: str
  inputs: list[TaskInput]
  output_key: str
  description: str
  capability_name: str = Field(
    ...,
    description="Name of the capability this task requires",
  )

  def get_input_mapping(self) -> dict[str, str]:
    """Returns a mapping of task input names to their source keys"""
    return {
      input.key: input.source_key or input.key
      for input in self.inputs
    }


class TaskContext(BaseModel):
  """Runtime context for task execution"""

  working_memory: dict[str, Any] = Field(
    default_factory=dict
  )
  capabilities: dict[
    str, tuple[CapabilityType, Any]
  ] = Field(default_factory=dict)
  prompt_configs: dict[str, CompletionConfig] = Field(
    default_factory=dict
  )

  def get_capability(
    self, name: str
  ) -> tuple[CapabilityType, Any] | None:
    """Gets a capability by name"""
    return self.capabilities.get(name)

  def add_capability(
    self,
    name: str,
    capability_type: CapabilityType,
    implementation: Any,
  ) -> None:
    """Adds a capability to the context"""
    self.capabilities[name] = (
      capability_type,
      implementation,
    )


class TaskSystem(BaseModel):
  """Complete task system with validation"""

  tasks: dict[str, Task]

  def validate_dependencies(
    self, context: TaskContext
  ) -> None:
    """Validates that all task dependencies can be resolved"""
    G: DiGraph = self.get_task_graph()
    if not nx.is_directed_acyclic_graph(G):
      cycles = list(nx.simple_cycles(G))
      raise PlanValidationError(
        ValidationErrorType.CYCLE_DETECTED,
        "Task graph contains cycles",
        details={"cycles": cycles},
      )

    # Track all available outputs from tasks
    available_outputs = {
      task.output_key for task in self.tasks.values()
    }

    # Ensure all inputs have producers or are in working memory
    for task in self.tasks.values():
      input_mapping = task.get_input_mapping()
      for input in task.inputs:
        source_key = input_mapping[input.key]
        if (
          source_key not in context.working_memory
          and source_key not in available_outputs
        ):
          raise PlanValidationError(
            ValidationErrorType.MISSING_INPUT,
            f"No task produces required input '{source_key}' and it's not present in working memory",
            task_name=task.name,
            details={
              "input_key": input.key,
              "source_key": source_key,
              "available_outputs": list(
                available_outputs
              ),
              "working_memory_keys": list(
                context.working_memory.keys()
              ),
            },
          )

      # Validate capabilities
      if not isinstance(task, PlanTask):
        capability = context.get_capability(
          task.capability_name
        )
        if not capability:
          raise PlanValidationError(
            ValidationErrorType.MISSING_CAPABILITY,
            f"Capability '{task.capability_name}' not found",
            task_name=task.name,
            details={
              "missing_name": task.capability_name,
              "available_capabilities": {
                name: str(cap_type)
                for name, (
                  cap_type,
                  _,
                ) in context.capabilities.items()
              },
            },
          )

  def get_task_graph(self) -> DiGraph:
    """Creates a directed graph of task dependencies.

    Returns:
        A directed graph with task names as nodes and task metadata
    """
    G = nx.DiGraph()

    # Add all tasks as nodes
    for task in self.tasks.values():
      node_attrs = {
        "output": task.output_key,
        "type": "plan_task"
        if isinstance(task, PlanTask)
        else "task",
      }
      G.add_node(task.name, **node_attrs)

      # If it's a PlanTask and we want to include nested tasks
      if isinstance(task, PlanTask):
        nested_graph = task.plan.get_task_graph()
        # Add prefix to nested task names to avoid conflicts
        nested_graph = nx.relabel_nodes(
          nested_graph,
          lambda x: f"{task.name}/{x}",
        )
        G = nx.compose(G, nested_graph)
        # Add edge from nested plan's output to parent task
        output_task = next(
          t
          for t in task.plan.tasks.values()
          if t.output_key
          == task.plan.desired_outputs[0]
        )
        G.add_edge(
          f"{task.name}/{output_task.name}",
          task.name,
        )

    # Add edges for dependencies
    for task in self.tasks.values():
      for input in task.inputs:
        producer = next(
          (
            t
            for t in self.tasks.values()
            if t.output_key == input.key
          ),
          None,
        )
        if producer:
          G.add_edge(producer.name, task.name)

    return G

  def visualize(self) -> None:
    """Visualizes the task dependency graph"""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    G: DiGraph = self.get_task_graph()
    pos: dict[str, tuple[float, float]] = (
      nx.spring_layout(G, seed=42)
    )

    _: Figure = plt.figure(figsize=(12, 8))

    # Draw nodes with different colors based on type
    node_data = dict(G.nodes(data=True))
    plan_tasks: list[str] = [
      n
      for n, d in node_data.items()
      if d.get("type") == "plan_task"
    ]
    regular_tasks: list[str] = [
      n
      for n, d in node_data.items()
      if d.get("type") == "task"
    ]

    nx.draw_networkx_nodes(
      G,
      pos,
      nodelist=regular_tasks,
      node_color="lightblue",
      node_size=2000,
    )
    nx.draw_networkx_nodes(
      G,
      pos,
      nodelist=plan_tasks,
      node_color="lightgreen",
      node_size=2000,
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Add labels with output keys
    labels: dict[str, str] = {
      task: f"{task}\n({node_data[task]['output']})"
      for task in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels)

    _: Text = plt.title("Task Dependency Graph")
    plt.show()


class Plan(BaseModel):
  """Represents a high-level goal with its execution context"""

  name: str
  description: str
  desired_outputs: list[str]  # Keys we want to produce
  tasks: dict[str, Task]
  functions: dict[str, Callable[..., Any]] = Field(
    default_factory=dict
  )
  instructions: dict[str, str] = Field(
    default_factory=dict
  )
  prompt_configs: dict[str, CompletionConfig] = Field(
    default_factory=dict
  )
  initial_context: dict[str, Any] = Field(
    default_factory=dict
  )

  def create_task_system(self) -> TaskSystem:
    """Creates a TaskSystem from this plan"""
    return TaskSystem(tasks=self.tasks)

  def create_context(self) -> TaskContext:
    """Creates initial TaskContext for this plan"""
    return TaskContext(
      working_memory=self.initial_context.copy(),
      functions=self.functions,
      instructions=self.instructions,
      prompt_configs=self.prompt_configs,
    )

  def visualize(
    self, include_nested: bool = True
  ) -> None:
    """Visualizes the plan's task dependency graph"""
    import networkx as nx
    from networkx import DiGraph

    G: DiGraph = self.get_task_graph(include_nested)
    pos: dict[str, tuple[float, float]] = (
      nx.spring_layout(G, seed=42)
    )

    _: Figure = plt.figure(figsize=(12, 8))

    # Draw nodes with different colors based on type
    plan_tasks: list[str] = [
      n
      for n, d in G.nodes(data=True)
      if d.get("type") == "plan_task"
    ]
    regular_tasks: list[str] = [
      n
      for n, d in G.nodes(data=True)
      if d.get("type") == "task"
    ]

    nx.draw_networkx_nodes(
      G,
      pos,
      nodelist=regular_tasks,
      node_color="lightblue",
      node_size=2000,
    )
    nx.draw_networkx_nodes(
      G,
      pos,
      nodelist=plan_tasks,
      node_color="lightgreen",
      node_size=2000,
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Add labels with output keys
    labels: dict[str, str] = {
      task: f"{task}\n({G.nodes[task]['output']})"
      for task in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels)

    _: Text = plt.title(
      f"Task Dependency Graph for Plan: {self.name}"
    )
    plt.show()

  def get_task_graph(
    self, include_nested: bool = True
  ) -> DiGraph:
    """Creates a directed graph of task dependencies"""
    import networkx as nx
    from networkx import DiGraph

    G: DiGraph = nx.DiGraph()

    # Add all tasks as nodes
    for task in self.tasks.values():
      node_attrs = {
        "output": task.output_key,
        "type": "plan_task"
        if isinstance(task, PlanTask)
        else "task",
      }
      G.add_node(task.name, **node_attrs)

      # If it's a PlanTask and we want to include nested tasks
      if include_nested and isinstance(task, PlanTask):
        nested_graph = task.plan.get_task_graph()
        # Add prefix to nested task names to avoid conflicts
        nested_graph = nx.relabel_nodes(
          nested_graph,
          lambda x: f"{task.name}/{x}",
        )
        G = nx.compose(G, nested_graph)
        # Add edge from nested plan's output to parent task
        output_task = next(
          t
          for t in task.plan.tasks.values()
          if t.output_key
          == task.plan.desired_outputs[0]
        )
        G.add_edge(
          f"{task.name}/{output_task.name}",
          task.name,
        )

    # Add edges for dependencies
    for task in self.tasks.values():
      for input in task.inputs:
        producer = next(
          (
            t
            for t in self.tasks.values()
            if t.output_key == input.key
          ),
          None,
        )
        if producer:
          G.add_edge(producer.name, task.name)

    return G


class PlanTask(Task):
  """A Task that executes a Plan"""

  plan: Plan = Field(
    ..., description="The plan this task executes"
  )

  @classmethod
  def from_plan(cls, plan: Plan) -> "PlanTask":
    """Creates a PlanTask from a Plan"""
    input_keys = set(plan.initial_context.keys())
    for task in plan.tasks.values():
      input_keys.update(k.key for k in task.inputs)

    return cls(
      name=plan.name,
      inputs=[
        TaskInput(key=k, description=f"Input {k}")
        for k in input_keys
      ],
      output_key=plan.desired_outputs[0],
      description=plan.description,
      plan=plan,
    )


class ConceptualTask(BaseModel):
  """Represents a high-level task in the conceptual plan"""

  name: str = Field(
    ..., description="Name of the task"
  )
  description: str = Field(
    ..., description="What the task accomplishes"
  )
  required_inputs: list[str] = Field(
    ..., description="Information needed by this task"
  )
  expected_outputs: list[str] = Field(
    ...,
    description="Information/artifacts this task produces",
  )
  dependencies: list[str] = Field(
    default_factory=list,
    description="Names of tasks this depends on",
  )
  purpose: str = Field(
    ..., description="Why this task is necessary"
  )


class ContextItem(BaseModel):
  """Represents a single context item with a key and value"""

  key: str = Field(..., description="Context item key")
  value: str = Field(
    ..., description="Context item value"
  )


class ConceptualPlan(BaseModel):
  """High-level logical plan before mapping to specific tools"""

  goal: str = Field(
    ...,
    description="What this plan aims to accomplish",
  )
  context_items: list[ContextItem] = Field(
    ...,
    description="Relevant context items and constraints",
  )
  tasks: list[ConceptualTask] = Field(
    ...,
    description="Logical steps to accomplish the goal",
  )


class PlanSpecification(BaseModel):
  """Specification for dynamically creating a Plan"""

  name: str = Field(
    ..., description="Name of the plan"
  )
  description: str = Field(
    ...,
    description="Description of what the plan accomplishes",
  )
  desired_outputs: list[str] = Field(
    ...,
    description="Keys of outputs this plan should produce",
  )
  tasks: list[Task] = Field(
    ..., description="Tasks to execute in this plan"
  )


class ValidationErrorType(str, Enum):
  MISSING_CAPABILITY = "missing_capability"
  MISSING_INPUT = "missing_input"
  CIRCULAR_DEPENDENCY = "circular_dependency"
  CYCLE_DETECTED = "cycle_detected"
  UNKNOWN = "unknown"


class ValidationErrorModel(BaseModel):
  """Pydantic model for validation errors"""

  error_type: ValidationErrorType
  message: str
  task_name: str | None = None
  details: dict[str, Any] | None = None


class PlanValidationError(Exception):
  """Custom exception for plan validation errors"""

  def __init__(
    self,
    error_type: ValidationErrorType,
    message: str,
    task_name: str | None = None,
    details: dict[str, Any] | None = None,
  ):
    self.error_type = error_type
    self.task_name = task_name
    self.details = details or {}
    super().__init__(message)

  def to_model(self) -> ValidationErrorModel:
    """Convert exception to Pydantic model"""
    return ValidationErrorModel(
      error_type=self.error_type,
      message=str(self.args[0]),
      task_name=self.task_name,
      details=self.details,
    )

  def __str__(self) -> str:
    task_info = (
      f" in task '{self.task_name}'"
      if self.task_name
      else ""
    )
    error_msg = f"\n[{self.error_type}]{task_info}:\n{self.args[0]}"

    if self.details:
      error_msg += "\n\nDetails:"
      for key, value in self.details.items():
        error_msg += f"\n  {key}: {value}"

    if (
      self.error_type
      == ValidationErrorType.MISSING_INPUT
    ):
      error_msg += "\n\nTo fix this:"
      error_msg += (
        "\n1. Add a task that produces this output, or"
      )
      error_msg += (
        "\n2. Provide the value in initial_context, or"
      )
      error_msg += (
        "\n3. Update the task to use available inputs"
      )

    return error_msg


class PlanAttempt(BaseModel):
  """Represents a previous plan attempt with validation errors"""

  plan: PlanSpecification = Field(
    ..., description="The attempted plan specification"
  )
  errors: list[ValidationErrorModel] = Field(
    ..., description="Validation errors encountered"
  )


class TaskExecutor:
  """Executes tasks with capability management"""

  def __init__(
    self,
    system: TaskSystem,
    context: TaskContext,
    prompt_handler: PromptHandler,
    capability_manager: CapabilityManager,
  ):
    self.system = system
    self.context = context
    self.prompt_handler = prompt_handler
    self.capability_manager = capability_manager
    self._execution_stack: set[str] = set()

  def _gather_input_values(
    self, task: Task
  ) -> dict[str, Any]:
    """Gathers input values for a task from working memory or task outputs.

    Args:
        task: The task to gather inputs for

    Returns:
        Dictionary mapping input names to their values

    Raises:
        ValueError: If a required input value is not available
    """
    input_values: dict[str, Any] = {}
    input_mapping = task.get_input_mapping()

    for input in task.inputs:
      source_key = input_mapping[input.key]
      if source_key not in self.context.working_memory:
        raise ValueError(
          f"Required input '{source_key}' not found in working memory for task '{task.name}'"
        )
      input_values[input.key] = (
        self.context.working_memory[source_key]
      )

    return input_values

  async def execute_task(self, task: Task) -> Any:
    """Executes a single task with capability management."""
    print(f"\nExecuting task: {task.name}")

    if task.name in self._execution_stack:
      raise RecursionError(
        f"Circular dependency detected: {task.name}"
      )

    self._execution_stack.add(task.name)
    try:
      # Get input values
      input_values = self._gather_input_values(task)

      # Execute based on task type
      if isinstance(task, PlanTask):
        result = await execute_plan_task(
          task, **input_values
        )
      else:
        # Use capability manager for execution
        result = await self.capability_manager.execute_capability(
          task.capability_name,
          input_values,
        )

        # Optimize if needed
        await (
          self.capability_manager.optimize_capability(
            task.capability_name
          )
        )

      print(
        f"  ✓ Task completed. Output: {str(result)[:100]}..."
      )
      self.context.working_memory[task.output_key] = (
        result
      )
      return result

    finally:
      self._execution_stack.remove(task.name)

  async def resolve_and_execute(
    self, task_name: str
  ) -> Any:
    """Resolves and executes a task and all its dependencies.

    Args:
        task_name: Name of the task to execute

    Returns:
        Result of task execution

    Raises:
        ValueError: If task not found or execution fails
        RecursionError: If circular dependencies detected
    """
    task = self.system.tasks.get(task_name)
    if not task:
      raise ValueError(f"Task not found: {task_name}")

    # Check for circular dependencies
    if task_name in self._execution_stack:
      raise RecursionError(
        f"Circular dependency detected: {task_name}"
      )

    # Get input mappings
    input_mapping = task.get_input_mapping()

    # Execute dependencies first
    for input in task.inputs:
      source_key = input_mapping[input.key]
      if source_key not in self.context.working_memory:
        # Find task that produces this output
        producer = next(
          (
            t
            for t in self.system.tasks.values()
            if t.output_key == source_key
          ),
          None,
        )
        if producer:
          print(
            f"\nExecuting dependency: {producer.name}"
          )
          await self.resolve_and_execute(producer.name)

    # Now execute this task
    return await self.execute_task(task)


class PlanExecutor:
  """Executes plans with dynamic capability creation"""

  def __init__(
    self,
    prompt_handler: PromptHandler | None = None,
    capability_store: CapabilityStore | None = None,
  ):
    self.prompt_handler = (
      prompt_handler or PromptHandler()
    )
    self.capability_store = (
      capability_store
      or CapabilityStore(
        tools={},
        instructions={},
        composites={},
      )
    )
    self.capability_manager = CapabilityManager(
      self.prompt_handler,
      self.capability_store,
    )

  async def execute(
    self,
    plan: Plan,
    initial_context: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    try:
      system = plan.create_task_system()

      # Use provided initial_context or fall back to plan's initial_context
      working_memory = (
        initial_context or plan.initial_context
      ).copy()

      context = TaskContext(
        working_memory=working_memory,
        capabilities=self.capability_store.get_all_capabilities(),
        prompt_configs={},
      )

      # Validate dependencies (capabilities should already exist)
      system.validate_dependencies(context)

      # Execute tasks with capability manager
      executor = TaskExecutor(
        system,
        context,
        self.prompt_handler,
        self.capability_manager,
      )

      results = {}
      for output in plan.desired_outputs:
        producer_task = next(
          task
          for task in plan.tasks.values()
          if task.output_key == output
        )
        # Use resolve_and_execute instead of execute_task
        result = await executor.resolve_and_execute(
          producer_task.name
        )
        results[output] = result

      return results

    except Exception as e:
      print(f"\n❌ Execution Error: {str(e)}")
      raise


def execute_plan_task(
  plan_task: PlanTask, **inputs: Any
) -> Any:
  """
  Function to execute a PlanTask

  Args:
      plan_task: The PlanTask to execute
      **inputs: Input values for the plan

  Returns:
      The result of executing the plan
  """
  updated_plan = Plan(
    name=plan_task.plan.name,
    description=plan_task.plan.description,
    desired_outputs=plan_task.plan.desired_outputs,
    tasks=plan_task.plan.tasks,
    functions=plan_task.plan.functions,
    instructions=plan_task.plan.instructions,
    prompt_configs=plan_task.plan.prompt_configs,
    initial_context=inputs,
  )

  executor = PlanExecutor()
  results = executor.execute(updated_plan)
  return results[updated_plan.desired_outputs[0]]


# ------------------------------------------------------------------------------


def generate_conceptual_plan(
  handler: PromptHandler,
  goal: str,
  context: dict[str, str],
  initial_inputs: dict[str, str],
) -> ConceptualPlan:
  """Generates a high-level conceptual plan for achieving a goal."""
  # Format context items for the prompt
  context_str = "\n".join(
    f"- {k}: {v}" for k, v in context.items()
  )

  # Format initial inputs more naturally
  inputs_str = "\n".join(
    f"- {k} ({v})" for k, v in initial_inputs.items()
  )

  instruction = f"""
  Please create a detailed, logical plan to accomplish the following goal:

  GOAL:
  {goal}

  CONTEXT:
  {context_str}

  INPUTS AVAILABLE AT START:
  {inputs_str}

  Create a structured plan that:
  1. Breaks down the goal into logical steps/tasks
  2. Identifies key dependencies between tasks
  3. Specifies what information/resources each task needs
  4. Describes what each task produces/outputs
  5. Explains why each task is necessary

  For each task, include:
  - Task name/description
  - Required inputs (use exact names from available inputs or outputs from other tasks)
  - Expected outputs
  - Dependencies (if any)
  - Purpose/justification

  The response should include:
  - The main goal
  - Context items as a list of key-value pairs
  - A list of tasks with their details

  IMPORTANT: Tasks can only use:
  1. The exact input names listed above (e.g., 'project_id')
  2. Outputs produced by previous tasks

  Focus on the logical flow and completeness rather than technical implementation.
  """

  config = CompletionConfig(
    response_format=ConceptualPlan,
    system_message="You are a planning expert who breaks down complex goals into logical steps.",
  )

  result = handler.complete(instruction, config=config)
  if not isinstance(result, ConceptualPlan):
    raise ValueError("Failed to parse conceptual plan")

  return result


def convert_conceptual_to_specification(
  handler: PromptHandler,
  conceptual_plan: ConceptualPlan,
  available_inputs: dict[str, str],
  capability_store: CapabilityStore,
  previous_attempt: dict[str, Any] | None = None,
) -> PlanSpecification:
  """Converts a conceptual plan into a formal plan specification."""
  # Get available capabilities from store with metadata
  available_capabilities = {
    name: {
      "type": str(capability_type),
      "description": metadata.description,
      "inputs": metadata.inputs,  # Show actual function parameters
      "input_mappings": getattr(
        metadata, "input_mappings", {}
      ),  # Show mappings if available
    }
    for name, (capability_type, metadata) in (
      capability_store.get_all_capabilities().items()
    )
  }

  instruction = f"""
  Please convert this conceptual plan into a formal specification:

  CONCEPTUAL PLAN:
  {conceptual_plan.model_dump_json(indent=2)}

  AVAILABLE INPUTS:
  {json.dumps(available_inputs, indent=2)}

  AVAILABLE CAPABILITIES:
  {json.dumps(available_capabilities, indent=2)}

  Create a formal plan specification that:
  1. Maps conceptual tasks to required capabilities
  2. Ensures all required inputs are properly sourced
  3. Maintains the logical flow and dependencies
  4. Produces all necessary outputs

  For each task:
  - Identify the core capability needed from the available capabilities
  - Use the exact input parameter names required by the capability
  - Map task inputs to capability parameters using input_mappings if provided
  - Define the data flow between tasks

  IMPORTANT INPUT HANDLING:
  - Use the exact parameter names required by each capability
  - Map task outputs to capability inputs according to input_mappings
  - Available inputs must be used exactly as named
  - Tasks can only use:
    1. Inputs listed in AVAILABLE INPUTS
    2. Outputs from previous tasks in the plan

  Focus on WHAT each task needs to accomplish rather than HOW it will be implemented.
  """

  if previous_attempt:
    instruction += f"""
        Previous attempt failed. Here are the details:

        Previous Plan Specification:
        {json.dumps(previous_attempt['plan'].model_dump(), indent=2)}

        Validation Errors:
        {json.dumps(previous_attempt['errors'], indent=2)}

        Please fix these issues in the new specification.
        """

  config = CompletionConfig(
    response_format=PlanSpecification,
    system_message="You are a planning expert who creates executable task workflows.",
  )

  result = handler.complete(instruction, config=config)
  if not isinstance(result, PlanSpecification):
    raise ValueError(
      "Failed to parse plan specification"
    )

  return result


async def dynamic_plan_test():
  # Example functions
  def get_project_request(project_id: str) -> str:
    return f"Project request details for {project_id}"

  def generate_response(
    request_text: str,
    sentiment: str,
  ) -> str:
    return f"Thank you for your request. Based on your {sentiment} message: {request_text}, we will process it accordingly."

  def analyze_sentiment(text: str) -> str:
    return "positive"

  # Initialize the prompt handler and capability store
  handler = PromptHandler()
  store = CapabilityStore(
    tools={},
    instructions={},
    composites={},
  )

  # Add predefined capabilities to store with clearer metadata
  store.add_tool(
    "get_project_request",
    get_project_request,
    CapabilityMetadata(
      name="get_project_request",
      type=CapabilityType.TOOL,
      created_at=datetime.now().isoformat(),
      description="Retrieves project request details given a project ID",
      inputs=[
        "project_id"
      ],  # Be explicit about parameter name
      outputs=[
        "project_request_details"
      ],  # Name the output
      dependencies=[],
    ),
  )

  store.add_tool(
    "analyze_sentiment",
    analyze_sentiment,
    CapabilityMetadata(
      name="analyze_sentiment",
      type=CapabilityType.TOOL,
      created_at=datetime.now().isoformat(),
      description="Analyzes sentiment of given text. Input 'text' should be the project request details.",
      inputs=[
        "text"
      ],  # The function expects 'text', not 'project_request_details'
      outputs=["sentiment"],
      dependencies=[],
      input_mappings={  # Add input mappings to help LLM understand parameter mapping
        "project_request_details": "text",  # Map from task input to function parameter
      },
    ),
  )

  store.add_tool(
    "generate_response",
    generate_response,
    CapabilityMetadata(
      name="generate_response",
      type=CapabilityType.TOOL,
      created_at=datetime.now().isoformat(),
      description="Generates a response based on request text and sentiment. Input 'request_text' should be the analyzed content.",
      inputs=["request_text", "sentiment"],
      outputs=["response"],
      dependencies=[],
      input_mappings={
        "analyzed_content": "request_text",
        "request_sentiment": "sentiment",
      },
    ),
  )

  # Add a new tool specifically for content analysis
  def analyze_content(
    project_request_details: str,
  ) -> str:
    return f"Analysis of: {project_request_details}"

  store.add_tool(
    "analyze_content",
    analyze_content,
    CapabilityMetadata(
      name="analyze_content",
      type=CapabilityType.TOOL,
      created_at=datetime.now().isoformat(),
      description="Analyzes the content of a project request to extract key points",
      inputs=[
        "project_request_details"
      ],  # Match the expected input name
      outputs=["content_analysis"],
      dependencies=[],
    ),
  )

  capability_manager = CapabilityManager(
    handler, store
  )
  executor = PlanExecutor(
    prompt_handler=handler,
    capability_store=store,
  )

  # Step 1: Generate conceptual plan
  goal = """
    Process a project request and generate an appropriate response that considers both
    the content and sentiment of the request.
    """

  context = {
    "requirements": """
        - Must retrieve and analyze the full project request
        - Must consider the emotional tone/sentiment of the request
        - Must generate a professional and appropriate response
        - Response should acknowledge both content and tone
        """,
    "quality_guidelines": """
        - Responses should be professional and empathetic
        - Must address all points in the original request
        - Should maintain appropriate tone based on request sentiment
        """,
  }

  # Define initial inputs
  initial_inputs = {
    "project_id": "Project identifier used to retrieve the request details",
  }

  print("\nGenerating conceptual plan...")
  conceptual_plan = generate_conceptual_plan(
    handler,
    goal,
    context,
    initial_inputs,
  )

  print("\nConceptual Plan:")
  print(f"Goal: {conceptual_plan.goal}")
  print("\nContext Items:")
  for item in conceptual_plan.context_items:
    print(f"  {item.key}: {item.value}")
  print("\nTasks:")
  for task in conceptual_plan.tasks:
    print(f"\n  {task.name}:")
    print(f"    Description: {task.description}")
    print(
      f"    Required Inputs: {task.required_inputs}"
    )
    print(
      f"    Expected Outputs: {task.expected_outputs}"
    )
    print(f"    Dependencies: {task.dependencies}")
    print(f"    Purpose: {task.purpose}")

  # Step 2: Convert to formal specification
  available_inputs = {
    "project_id": "string",
  }

  print("\nConverting to formal specification...")
  plan_spec = convert_conceptual_to_specification(
    handler,
    conceptual_plan,
    available_inputs,
    store,
  )

  print("\nFormal Plan Specification:")
  print(f"Name: {plan_spec.name}")
  print(f"Description: {plan_spec.description}")
  print(
    f"Desired Outputs: {plan_spec.desired_outputs}"
  )

  print("\nTasks:")
  for task in plan_spec.tasks:
    print(f"\n  {task.name}:")
    print(f"    Description: {task.description}")
    print(f"    Output: {task.output_key}")
    print(f"    Capability: {task.capability_name}")
    print("    Inputs:")
    for input in task.inputs:
      mapping_info = (
        f" -> {input.source_key}"
        if input.source_key
        else ""
      )
      print(
        f"      - {input.key}{mapping_info}: {input.description}"
      )

  print("\nCreating and validating plan...")
  plan = await create_and_validate_plan(
    specification=plan_spec,
    initial_context={"project_id": "PROJ123"},
    capability_manager=capability_manager,
  )

  # Visualize the plan
  print("\nVisualizing plan...")
  plan.visualize()

  # Execute the plan using the same executor
  print("\nExecuting plan...")
  outputs = await executor.execute(plan)

  print("\nPlan Execution Results:")
  for key, value in outputs.items():
    print(f"{key}: {value}")


async def create_and_validate_plan(
  specification: PlanSpecification,
  initial_context: dict[str, Any],
  capability_manager: CapabilityManager,
) -> Plan:
  """Creates and validates a Plan from a specification."""
  # Convert tasks list to dictionary
  tasks_dict = {
    task.name: task for task in specification.tasks
  }

  # Create Plan instance
  plan = Plan(
    name=specification.name,
    description=specification.description,
    desired_outputs=specification.desired_outputs,
    tasks=tasks_dict,
    initial_context=initial_context,
  )

  # Create context
  context = TaskContext(
    working_memory=initial_context.copy(),
    capabilities={},
    prompt_configs={},
  )

  # Validate with capability resolution
  system = plan.create_task_system()
  attempted_capabilities: set[str] = set()

  while True:
    try:
      system.validate_dependencies(context)
      break
    except PlanValidationError as e:
      if (
        e.error_type
        != ValidationErrorType.MISSING_CAPABILITY
      ):
        raise

      missing_name = e.details.get("missing_name")
      if not missing_name:
        raise ValueError(
          "Missing capability name in error details"
        )

      if missing_name in attempted_capabilities:
        raise ValueError(
          f"Failed to create capability after attempt: {missing_name}"
        )

      attempted_capabilities.add(missing_name)
      print(f"\nResolving capability: {missing_name}")

      # Resolve the capability
      (
        capability_type,
        implementation,
      ) = await capability_manager.resolve_capability(
        missing_name,
        context.working_memory,
      )

      # Add to context
      context.add_capability(
        missing_name, capability_type, implementation
      )

  return plan


if __name__ == "__main__":
  import asyncio

  asyncio.run(dynamic_plan_test())
