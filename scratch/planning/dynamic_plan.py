import json
from typing import (
  Any,
)

from plan import (
  Plan,
  PlanExecutor,
  Task,
)
from prompt import PromptHandler
from pydantic import BaseModel

# ------------------------------------------------------------------------------


class PlanSpecification(BaseModel):
  """Specification for dynamically creating a Plan"""

  name: str
  description: str
  desired_outputs: list[str]
  tasks: list[Task]


def generate_plan_specification(
  handler: PromptHandler,
  available_inputs: dict[str, str],
  available_tools: dict[str, Any],
  available_instructions: dict[str, str],
  previous_attempt: dict[str, Any] | None = None,
) -> PlanSpecification:
  """Generates a plan specification with validation error feedback and previous attempt"""
  instruction = f"""
    Please create a plan for handling project requests with the following available tools and instructions:

    Available Inputs:
    {json.dumps(available_inputs, indent=2)}

    Available Tools:
    {json.dumps(available_tools, indent=2)}

    Available Instructions:
    {json.dumps(available_instructions, indent=2)}

    The plan should:
    1. Retrieve project request details using get_project_request tool
    2. Analyze the sentiment of the request using analyze_sentiment tool
    3. Generate a summarized response considering both the request and its sentiment using the generate_response instruction

    IMPORTANT: For each task input, you must specify:
    - key: The parameter name required by the tool (e.g., 'text' for analyze_sentiment)
    - source_key: The key in working memory or output from another task that provides the value (e.g., 'request_text')
    - description: Description of the input

    For prompt tasks (those without a tool), use the instruction name as the task name to ensure proper matching.

    Example task using analyze_sentiment tool:
    {{
      "name": "Analyze Request Sentiment",
      "tool_name": "analyze_sentiment",
      "inputs": [
        {{
          "key": "text",           # Parameter name required by analyze_sentiment tool
          "source_key": "request_text",  # Maps to output from Retrieve Project Request task
          "description": "The request text to analyze"
        }}
      ],
      "output_key": "sentiment",
      "description": "Analyzes sentiment of the request"
    }}

    The task's inputs must map correctly between:
    1. What the tool expects (key)
    2. What's available in working memory or from other tasks (source_key)

    For the final response generation task, make sure to use 'generate_response' as the task name to match the instruction.
    """

  if previous_attempt:
    instruction += f"""

    Previous attempt failed. Here are the details:

    Previous Plan Specification:
    {json.dumps(previous_attempt['plan'].model_dump(), indent=2)}

    Validation Errors:
    {json.dumps(previous_attempt['errors'], indent=2)}

    Please analyze the previous plan and its errors carefully, then create a new plan that addresses these issues.
    Focus particularly on fixing the dependency chain and ensuring all inputs are properly sourced.
    """

  response = handler.client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "system",
        "content": """You are a planning expert who creates task workflows.
            Create plans that effectively utilize available tools and instructions.
            When given a failed attempt, carefully analyze the errors and previous plan to create an improved version.""",
      },
      {"role": "user", "content": instruction},
    ],
    response_format=PlanSpecification,
  )

  result = response.choices[0].message.parsed
  if result is None:
    raise ValueError("Failed to parse response")

  return result


def test():
  # Example function
  def get_project_request(project_id: str) -> str:
    return f"Project request details for {project_id}"

  def generate_response(
    request_text: str, sentiment: str
  ) -> str:
    """Generates a response considering the request text and sentiment.

    Args:
        request_text: The project request text
        sentiment: The analyzed sentiment of the request

    Returns:
        str: Generated response
    """
    return f"Thank you for your request. Based on your {sentiment} message: {request_text}, we will process it accordingly."

  def analyze_sentiment(text: str) -> str:
    return "positive"

  # Initialize the prompt handler
  handler = PromptHandler()

  # Example inputs
  available_inputs = {
    "project_id": "string",
  }

  # Example available tools and instructions
  available_tools = {
    "get_project_request": {
      "name": "get_project_request",
      "description": "Retrieves project request details",
      "parameters": {"project_id": "string"},
      "returns": "string",
    },
    "analyze_sentiment": {
      "name": "analyze_sentiment",
      "description": "Analyzes sentiment of text",
      "parameters": {"text": "string"},
      "returns": "string",
    },
  }

  available_instructions = {
    "summarize_request": "Summarize the following project request: {request_text}",
    "generate_response": """Generate a comprehensive response for the project request.

Input context:
- Project Request: {request_text}
- Sentiment Analysis: {sentiment}

Please generate a professional and appropriate response that:
1. Acknowledges the request
2. Considers the detected sentiment
3. Provides a clear and constructive response
4. Maintains a professional tone

Format the response in a clear, structured manner.""",
  }

  MAX_RETRIES = 3
  previous_attempt: dict[str, Any] | None = None
  plan: Plan | None = None  # Track successful plan
  result: PlanSpecification | None = None
  system: Any | None = (
    None  # Using Any since we don't have the TaskSystem type
  )
  context: Any | None = (
    None  # Using Any since we don't have the TaskContext type
  )

  for attempt in range(MAX_RETRIES):
    try:
      print(f"\nAttempt {attempt + 1}/{MAX_RETRIES}")

      # Generate plan specification with validation feedback
      result = generate_plan_specification(
        handler,
        available_inputs,
        available_tools,
        available_instructions,
        previous_attempt,
      )

      # Print the parsed result
      print("\nParsed Result:")
      print(f"Name: {result.name}")
      print(f"Description: {result.description}")
      print(
        f"Desired Outputs: {result.desired_outputs}"
      )

      print("\nTasks:")
      for task in result.tasks:
        print(f"\n  {task.name}:")
        print(f"    Description: {task.description}")
        print(f"    Output: {task.output_key}")
        if task.tool_name:
          print(f"    Tool: {task.tool_name}")
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

      # Convert list of tasks to dictionary for Plan creation
      tasks_dict = {
        task.name: task for task in result.tasks
      }

      # Create actual Plan instance from specification
      plan = Plan(
        name=result.name,
        description=result.description,
        desired_outputs=result.desired_outputs,
        tasks=tasks_dict,
        functions={
          "get_project_request": get_project_request,
          "analyze_sentiment": analyze_sentiment,
          "generate_response": generate_response,
        },
        instructions=available_instructions,
        prompt_configs={},
        initial_context={"project_id": "PROJ123"},
      )

      # Create TaskSystem and TaskContext for validation
      system = plan.create_task_system()
      context = plan.create_context()

      print("\nValidating plan dependencies...")
      system.validate_dependencies(context)
      print("✓ Plan validation successful!")
      break  # Exit loop on success

    except ValueError as e:
      validation_errors = [str(e)]
      print(f"✗ Plan validation failed: {e}")

      # Store the failed attempt and its errors for the next iteration
      previous_attempt = {
        "plan": result
        if result is not None
        else PlanSpecification(
          name="",
          description="",
          desired_outputs=[],
          tasks=[],
        ),
        "errors": validation_errors,
      }

      if attempt == MAX_RETRIES - 1:
        print(
          "\nMax retries reached. Final validation errors:"
        )
        print("\n".join(validation_errors))

        if system is not None:
          print(
            "\nTask Graph (showing problematic dependencies):"
          )
          system.visualize()

          print(
            "\nDetailed task input/output analysis:"
          )
          for task_name, task in system.tasks.items():
            print(f"\nTask: {task_name}")
            print(f"Output Key: {task.output_key}")
            if task.tool_name:
              print(f"Tool: {task.tool_name}")
            print("Input Keys:")
            input_mapping = task.get_input_mapping()
            for input in task.inputs:
              source_key = input_mapping[input.key]
              producers = [
                t
                for t in system.tasks.values()
                if t.output_key == source_key
              ]
              if producers:
                print(
                  f"  - {input.key} -> {source_key} (provided by: {[t.name for t in producers]})"
                )
              elif (
                context is not None
                and source_key
                in context.working_memory
              ):
                print(
                  f"  - {input.key} -> {source_key} (from working memory)"
                )
              else:
                print(
                  f"  - {input.key} -> {source_key} (no producer found)"
                )
        return  # Exit on max retries

    except Exception as e:
      print(f"Unexpected error: {e}")
      if attempt == MAX_RETRIES - 1:
        raise

  # Only proceed with visualization and execution if we have a valid plan
  if plan is not None:
    print("\nVisualizing plan...")
    plan.visualize()

    print("\nExecuting plan...")
    executor = PlanExecutor()
    outputs = executor.execute(plan)

    print("\nPlan Execution Results:")
    for key, value in outputs.items():
      print(f"{key}: {value}")


if __name__ == "__main__":
  test()
