import json
from datetime import datetime
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import (
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionToolParam,
  ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field


class CompletionConfig(BaseModel):
  """Configuration for completions with type safety"""

  model: str = "gpt-4o-mini"
  temperature: float = Field(
    default=0.7, ge=0.0, le=2.0
  )
  max_tokens: int | None = Field(default=None, gt=0)
  system_message: str | None = None
  tools: list[ChatCompletionToolParam] | None = None
  response_format: type[BaseModel] | None = None


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

  def complete(
    self,
    instruction: str,
    *,
    config: CompletionConfig | None = None,
    context_functions: dict[str, Any] | None = None,
  ) -> str | BaseModel:
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


def test():
  # 1. Basic Models
  class User(BaseModel):
    name: str
    age: int

  class Address(BaseModel):
    street: str
    city: str
    country: str

  class DetailedUser(BaseModel):
    name: str
    age: int
    address: Address

  class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

  class ProductList(BaseModel):
    products: list[Product]

  class OrderItem(BaseModel):
    product_name: str
    quantity: int
    price_per_unit: float

  class Order(BaseModel):
    order_id: str
    customer_name: str
    items: list[OrderItem]
    total_amount: float
    discount: float | None = None
    order_date: str

  class MovieRecommendation(BaseModel):
    title: str
    genre: str
    rating: float
    description: str

  # Initialize handler
  handler = PromptHandler()

  # Test 1: Basic User
  print("\n=== Basic User Test ===")
  user_config = CompletionConfig(
    response_format=User,
  )
  user = handler.complete(
    "Generate a random user",
    config=user_config,
  )
  print(f"Basic User: {user}")

  # Test 2: Detailed User
  print("\n=== Detailed User Test ===")
  detailed_config = CompletionConfig(
    response_format=DetailedUser,
  )
  detailed_user = handler.complete(
    "Generate a user with address details",
    config=detailed_config,
  )
  print(f"Detailed User: {detailed_user}")

  # Test 3: Product List
  print("\n=== Product List Test ===")
  product_config = CompletionConfig(
    response_format=ProductList,
  )
  products = handler.complete(
    "Generate 3 random products",
    config=product_config,
  )
  print(f"Products: {products}")

  # Test 4: Complex Order
  print("\n=== Order Test ===")
  order_config = CompletionConfig(
    response_format=Order,
    max_tokens=8192,
  )
  order = handler.complete(
    "Generate a random order with the date in ISO format (YYYY-MM-DD)",
    config=order_config,
  )

  # Convert order date to datetime if needed
  if isinstance(order, Order):
    order.order_date = datetime.fromisoformat(
      order.order_date,
    ).isoformat()
  print(f"Order: {order}")

  # Test 5: Movie Recommendation with System Message
  print("\n=== Movie Recommendation Test ===")
  movie_config = CompletionConfig(
    response_format=MovieRecommendation,
    system_message="You are a movie recommendation expert.",
    max_tokens=8192,
  )
  movie = handler.complete(
    "Suggest an action movie",
    config=movie_config,
  )
  print(f"Movie Recommendation: {movie}")

  # Test 6: Error Handling Example
  print("\n=== Error Handling Test ===")
  try:
    # Intentionally using an invalid configuration
    bad_config = CompletionConfig(
      response_format=User,
      max_tokens=-1,  # Invalid value
    )
    result = handler.complete(
      "This should fail",
      config=bad_config,
    )
  except ValueError as e:
    print(f"Expected error caught: {e}")

  # Original TaskBreakdown test
  print("\n=== Task Breakdown Test ===")

  class TaskItem(BaseModel):
    title: str = Field(
      ..., description="Title of the task"
    )
    description: str = Field(
      ...,
      description="Detailed description of what needs to be done",
    )
    estimated_hours: float = Field(
      ..., description="Estimated hours to complete"
    )
    priority: int = Field(
      ...,
      description="Priority from 1 (highest) to 5 (lowest)",
    )

  class TaskBreakdown(BaseModel):
    project_name: str = Field(
      ..., description="Name of the project"
    )
    total_tasks: int = Field(
      ..., description="Total number of tasks"
    )
    tasks: list[TaskItem] = Field(
      ..., description="List of tasks"
    )
    estimated_total_hours: float = Field(
      ..., description="Total estimated hours"
    )

  project_description = """
    Create a mobile app for tracking daily water intake. The app should:
    - Allow users to log their water consumption
    - Send reminders throughout the day
    - Show progress visualizations
    - Support multiple user profiles
    """

  task_config = CompletionConfig(
    temperature=0.7,
    system_message="""You are a project management expert who breaks down projects
        into detailed tasks. Provide structured task breakdowns that are realistic and actionable.""",
    response_format=TaskBreakdown,
  )

  instruction = f"""
    Please break down the following project into detailed tasks:

    Project Description:
    {project_description}

    Provide a structured breakdown including task descriptions, estimated hours,
    and priorities. Be specific and realistic with the estimates.
    """

  try:
    result = cast(
      TaskBreakdown,
      handler.complete(
        instruction,
        config=task_config,
      ),
    )

    print(f"\nProject: {result.project_name}")
    print(f"Total Tasks: {result.total_tasks}")
    print(
      f"Total Estimated Hours: {result.estimated_total_hours}"
    )
    print("\nTask Breakdown:")

    for i, task in enumerate(result.tasks, 1):
      print(
        f"\n{i}. {task.title} (Priority: {task.priority})"
      )
      print(f"   Description: {task.description}")
      print(
        f"   Estimated Hours: {task.estimated_hours}"
      )

  except Exception as e:
    print(f"Error: {e}")


if __name__ == "__main__":
  test()
