import json
from collections.abc import Callable
from typing import (
  Any,
  TypeVar,
  cast,
)

import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from openai.types.chat import (
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionToolParam,
  ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


# ------------------------------------------------------------------------------


class CompletionConfig(BaseModel):
  """Configuration for completions with type safety"""

  model: str = "gpt-4o-mini"
  temperature: float = Field(
    default=0.7, ge=0.0, le=2.0
  )
  max_tokens: int | None = Field(default=None, gt=0)
  system_message: str | None = None
  tools: list[ChatCompletionToolParam] | None = None
  response_format: type[T] | None = None


class PromptHandler:
  """Unified prompt handler that automatically selects appropriate mode:
  1. If response_format provided -> Structured Pydantic response
  2. If tools provided -> Tool calling
  3. Otherwise -> Plain text completion
  """

  def __init__(self, api_key: str | None = None):
    self.client = OpenAI(api_key=api_key)

  def _build_messages(
    self,
    instruction: str,
    config: CompletionConfig,
  ) -> list[ChatCompletionMessageParam]:
    """Builds message list for completion request"""
    messages: list[ChatCompletionMessageParam] = []

    if config.system_message:
      messages.append(
        ChatCompletionSystemMessageParam(
          role="system",
          content=config.system_message,
        ),
      )

    messages.append(
      ChatCompletionUserMessageParam(
        role="user",
        content=instruction,
      ),
    )

    return messages

  def complete[T](
    self,
    instruction: str,
    *,
    config: CompletionConfig | None = None,
    context_functions: dict[str, Callable[..., Any]]
    | None = None,
  ) -> str | T:
    """Unified completion method that handles all response types

    Args:
        instruction: The prompt instruction
        config: Optional completion configuration
        context_functions: Required when using tools

    Returns:
        String response or parsed Pydantic model

    Raises:
        ValueError: If tools used without context_functions
        ValueError: If no content in response
    """
    config = config or CompletionConfig()
    messages = self._build_messages(
      instruction, config
    )

    # Case 1: Structured Pydantic response
    if config.response_format is not None:
      response = (
        self.client.beta.chat.completions.parse(
          model=config.model,
          messages=messages,
          response_format=config.response_format,
          temperature=config.temperature,
          max_tokens=config.max_tokens,
        )
      )

      parsed = response.choices[0].message.parsed
      if parsed is None:
        raise ValueError("Failed to parse response")

      return parsed

    # Case 2: Tool calling
    if config.tools is not None:
      if context_functions is None:
        raise ValueError(
          "context_functions required when using tools"
        )

      response = self.client.chat.completions.create(
        model=config.model,
        messages=messages,
        tools=config.tools,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
      )

      if response.choices[0].message.tool_calls:
        for tool_call in response.choices[
          0
        ].message.tool_calls:
          function_name = tool_call.function.name
          function_args = json.loads(
            tool_call.function.arguments
          )

          if function_name not in context_functions:
            raise ValueError(
              f"Tool '{function_name}' not found"
            )

          result = context_functions[function_name](
            **function_args
          )
          messages.append(
            ChatCompletionToolMessageParam(
              role="tool",
              tool_call_id=tool_call.id,
              content=str(result),
            ),
          )

        response = self.client.chat.completions.create(
          model=config.model,
          messages=messages,
          temperature=config.temperature,
          max_tokens=config.max_tokens,
        )

    # Case 3: Plain text completion
    response = self.client.chat.completions.create(
      model=config.model,
      messages=messages,
      temperature=config.temperature,
      max_tokens=config.max_tokens,
    )

    content = response.choices[0].message.content
    if content is None:
      raise ValueError("No content in response")

    return content


# ------------------------------------------------------------------------------


class TaskContext(BaseModel):
  """Runtime context for task execution"""

  working_memory: dict[str, Any] = Field(
    default_factory=dict
  )
  functions: dict[str, Any]
  instructions: dict[str, str]
  prompt_configs: dict[str, CompletionConfig] = Field(
    default_factory=dict
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
  tool_name: str | None = Field(
    default=None,
    description="Name of the tool this task uses, if any",
  )

  def get_input_mapping(self) -> dict[str, str]:
    """Returns a mapping of task input names to their source keys"""
    return {
      input.key: input.source_key or input.key
      for input in self.inputs
    }


class TaskSystem(BaseModel):
  """Complete task system with validation"""

  tasks: dict[str, Task]

  def get_task_graph(self) -> Any:
    """Creates a directed graph of task dependencies"""
    G: Any = nx.DiGraph()

    # Add all tasks as nodes with explicit typing for attributes
    for task in self.tasks.values():
      G.add_node(task.name, output=task.output_key)

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
    G: Any = self.get_task_graph()
    pos: Any = nx.spring_layout(G, seed=42)

    _ = plt.figure(figsize=(12, 8))
    nx.draw(
      G,
      pos,
      with_labels=True,
      node_color="lightblue",
      node_size=2000,
      font_size=10,
      font_weight="bold",
    )

    # Add output keys as node labels
    output_labels = {
      task.name: f"{task.name}\n({task.output_key})"
      for task in self.tasks.values()
    }
    nx.draw_networkx_labels(G, pos, output_labels)

    _ = plt.title("Task Dependency Graph")
    plt.show()

  def validate_dependencies(
    self, context: TaskContext
  ):
    """Validates that all task dependencies can be resolved

    Args:
        context: TaskContext containing working memory and functions
    """
    G: Any = self.get_task_graph()
    if not nx.is_directed_acyclic_graph(G):
      raise ValueError("Task graph contains cycles")

    # Track all available outputs from tasks
    available_outputs = {
      task.output_key for task in self.tasks.values()
    }

    # Ensure all inputs have producers or are in working memory
    for task in self.tasks.values():
      input_mapping = task.get_input_mapping()
      for input in task.inputs:
        source_key = input_mapping[
          input.key
        ]  # Get the mapped source key
        if (
          source_key not in context.working_memory
          and source_key not in available_outputs
        ):
          raise ValueError(
            f"No task produces required input '{source_key}' (mapped from '{input.key}') "
            f"for task '{task.name}' and it's not present in working memory"
          )


class TaskExecutor:
  """Handles task execution with dependency resolution"""

  def __init__(
    self,
    system: TaskSystem,
    context: TaskContext,
    prompt_handler: PromptHandler | None = None,
  ):
    self.system = system
    self.context = context
    self.prompt_handler = (
      prompt_handler or PromptHandler()
    )
    self._execution_stack: set[str] = set()

  def is_prompt_task(self, task_name: str) -> bool:
    """Determines if a task is a prompt task"""
    task = self.system.tasks[task_name]
    if isinstance(task, PlanTask):
      return False
    return task_name not in self.context.functions

  def execute_task(self, task: Task) -> str:
    """Executes a single task with input mapping"""
    # Get the input mapping
    input_mapping = task.get_input_mapping()

    # Gather input values with mapping
    input_values = {}
    for input in task.inputs:
      source_key = input_mapping[input.key]
      if source_key not in self.context.working_memory:
        raise ValueError(
          f"Missing required input: {source_key}"
        )
      input_values[input.key] = (
        self.context.working_memory[source_key]
      )

    if isinstance(task, PlanTask):
      result = execute_plan_task(task, **input_values)
    elif task.tool_name:  # Execute tool
      if task.tool_name not in self.context.functions:
        raise ValueError(
          f"Tool not found: {task.tool_name}"
        )
      result = cast(
        str,
        self.context.functions[task.tool_name](
          **input_values
        ),
      )
    else:  # Execute prompt
      if task.name not in self.context.instructions:
        raise ValueError(
          f"No instructions found for prompt task: {task.name}"
        )

      config = self.context.prompt_configs.get(
        task.name, CompletionConfig()
      )
      instruction = self.context.instructions[
        task.name
      ].format(**input_values)
      result = self.prompt_handler.complete(
        instruction,
        config=config,
        context_functions=self.context.functions,
      )

    self.context.working_memory[task.output_key] = (
      result
    )
    return result

  def resolve_and_execute(self, task_name: str) -> str:
    """Resolves dependencies and executes a task"""
    if task_name in self._execution_stack:
      raise ValueError(
        f"Circular dependency detected for task: {task_name}"
      )

    self._execution_stack.add(task_name)
    task = self.system.tasks[task_name]

    try:
      # Resolve dependencies first
      for input in task.inputs:
        if (
          input.key not in self.context.working_memory
        ):
          producer_task = next(
            (
              t
              for t in self.system.tasks.values()
              if t.output_key == input.key
            ),
            None,
          )
          if producer_task:
            self.resolve_and_execute(
              producer_task.name
            )

      return self.execute_task(task)

    finally:
      self._execution_stack.remove(task_name)


class Plan(BaseModel):
  """Represents a high-level goal with its execution context"""

  name: str
  description: str
  desired_outputs: list[str]  # Keys we want to produce
  tasks: dict[str, Task]
  functions: dict[str, Callable[..., Any]] = Field(
    default_factory=dict,
  )
  instructions: dict[str, str] = Field(
    default_factory=dict,
  )
  prompt_configs: dict[str, CompletionConfig] = Field(
    default_factory=dict,
  )
  initial_context: dict[str, Any] = Field(
    default_factory=dict,
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

  def get_task_graph(
    self, include_nested: bool = True
  ) -> nx.DiGraph:
    """Creates a directed graph of task dependencies

    Args:
        include_nested: If True, includes tasks from nested plans

    Returns:
        A directed graph representing task dependencies
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

  def visualize(
    self, include_nested: bool = True
  ) -> None:
    """Visualizes the plan's task dependency graph

    Args:
        include_nested: If True, includes tasks from nested plans
    """
    G = self.get_task_graph(include_nested)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 8))

    # Draw nodes with different colors based on type
    plan_tasks = [
      n
      for n, d in G.nodes(data=True)
      if d.get("type") == "plan_task"
    ]
    regular_tasks = [
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
    labels = {
      task: f"{task}\n({G.nodes[task]['output']})"
      for task in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels)

    plt.title(
      f"Task Dependency Graph for Plan: {self.name}"
    )
    plt.show()


class PlanTask(Task):
  """A Task that executes a Plan"""

  # Add plan as a model field
  plan: Plan = Field(
    ...,  # ... means the field is required
    description="The plan this task executes",
  )

  @classmethod
  def from_plan(cls, plan: Plan) -> "PlanTask":
    """Creates a PlanTask from a Plan

    Args:
        plan: The Plan to execute

    Returns:
        A PlanTask instance
    """
    # Get input keys from both initial context and task inputs
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
      plan=plan,  # Pass plan as a field
    )


class PlanExecutor:
  """Executes plans to achieve desired outputs"""

  def __init__(
    self,
    prompt_handler: PromptHandler | None = None,
  ):
    self.prompt_handler = (
      prompt_handler or PromptHandler()
    )

  def execute(
    self,
    plan: Plan,
    initial_context: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """
    Executes a plan to produce desired outputs

    Args:
        plan: The Plan to execute
        initial_context: Optional initial context to merge with plan's context

    Returns:
        Dictionary mapping output keys to their values

    Raises:
        ValueError: If no task produces a required output
        ValueError: If dependencies cannot be resolved
    """
    system = plan.create_task_system()
    context = plan.create_context()

    # Merge provided initial context if any
    if initial_context:
      context.working_memory.update(initial_context)

    # Validate plan can produce desired outputs
    available_outputs = {
      task.output_key for task in plan.tasks.values()
    }
    for output in plan.desired_outputs:
      if output not in available_outputs:
        raise ValueError(
          f"No task produces required output: {output}"
        )

    # Validate dependencies
    system.validate_dependencies(context)

    # Create executor
    executor = TaskExecutor(
      system, context, self.prompt_handler
    )

    # Find and execute tasks that produce desired outputs
    results = {}
    for output in plan.desired_outputs:
      producer_task = next(
        task
        for task in plan.tasks.values()
        if task.output_key == output
      )
      result = executor.resolve_and_execute(
        producer_task.name
      )
      results[output] = result

    return results


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


# Example function
def get_project_request(project_id: str) -> str:
  return f"Project request details for {project_id}"


# ------------------------------------------------------------------------------


def main() -> None:
  # Define a nested plan for handling project requests
  request_plan = Plan(
    name="get_project_details",
    description="Retrieves and formats project request details",
    desired_outputs=["project_request"],
    tasks={
      "get_project_request": Task(
        name="get_project_request",
        inputs=[
          TaskInput(
            key="project_id",
            description="Project ID",
          )
        ],
        output_key="project_request",
        description="Retrieves project request details",
        tool_name="get_project_request",
      ),
    },
    functions={
      "get_project_request": get_project_request
    },
  )

  # Create main proposal plan that uses the nested plan
  proposal_plan = Plan(
    name="create_project_proposal",
    description="Creates a detailed project proposal from a project ID",
    desired_outputs=["proposal_plan"],
    tasks={
      "get_project_details": PlanTask.from_plan(
        request_plan,
      ),
      "create_proposal_plan": Task(
        name="create_proposal_plan",
        inputs=[
          TaskInput(
            key="project_request",
            description="Project request",
          )
        ],
        output_key="proposal_plan",
        description="Creates a proposal plan",
      ),
    },
    instructions={
      "create_proposal_plan": """
      Based on the following project request, create a detailed proposal plan:

      Project Request: {project_request}

      Please include:
      1. Project overview
      2. Timeline
      3. Resource requirements
      4. Risk assessment
      5. Budget estimation
      """
    },
    prompt_configs={
      "create_proposal_plan": CompletionConfig(
        system_message="You are an expert project manager helping to create detailed project proposals.",
        temperature=0.7,
        model="gpt-4o-mini",
        max_tokens=1000,
      )
    },
    # Set initial context at the top level
    initial_context={"project_id": "123"},
  )

  # Visualize the plan structure
  print("\nVisualization with nested tasks:")
  proposal_plan.visualize(include_nested=True)

  print("\nVisualization of top-level tasks only:")
  proposal_plan.visualize(include_nested=False)

  # Execute plan
  executor = PlanExecutor()
  results = executor.execute(proposal_plan)

  print("\nProposal Plan:")
  print(results["proposal_plan"])


if __name__ == "__main__":
  main()
