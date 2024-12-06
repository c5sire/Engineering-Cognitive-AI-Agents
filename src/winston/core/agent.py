"""Core agent interfaces and base implementation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, cast

import litellm
from litellm import acompletion
from litellm.types.completion import (
  ChatCompletionMessageParam,
)
from litellm.types.utils import (
  ChatCompletionDeltaToolCall,
  Function,
  ModelResponse,
  StreamingChoices,
)
from litellm.utils import CustomStreamWrapper
from loguru import logger

from winston.core.agent_config import AgentConfig
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import (
  Agent,
  System,
)
from winston.core.tools import (
  ToolManager,
)
from winston.core.vision import (
  create_vision_messages,
)
from winston.core.workspace import WorkspaceManager

# litellm.set_verbose = True
litellm.drop_params = True


@dataclass
class AgentState:
  """State management for the agent."""

  last_response: str | None = None
  last_error: str | None = None
  context: dict[str, Any] = field(default_factory=dict)
  tool_calls: dict[str, Any] = field(
    default_factory=dict
  )


class BaseAgent(Agent):
  """Base agent implementation with LLM support."""

  def __init__(
    self,
    system: System,
    config: AgentConfig,
    paths: AgentPaths,
  ) -> None:
    """Initialize an Agent instance."""
    self.system = system
    self._config = config
    self.paths = paths
    self.state = AgentState()
    self.tool_manager = ToolManager(system, config.id)

    # Register agent first
    system.register_agent(self)

  @property
  def id(self) -> str:
    """Get the agent ID."""
    return self.config.id

  @property
  def workspace_path(self) -> Path:
    """Get path to this agent's workspace."""
    return self.paths.workspaces / f"{self.id}.md"

  @property
  def config(self) -> AgentConfig:
    """Get the agent configuration."""
    return self._config

  @classmethod
  def can_handle(cls, message: Message) -> bool:
    """Check if this agent can handle the message."""
    return True

  def _get_workspaces(
    self,
    message: Message,
  ) -> tuple[str, str | None]:
    """Get private and shared workspace content.

    Parameters
    ----------
    message : Message
        Message that may contain shared workspace path

    Returns
    -------
    tuple[str, str | None]
        Private workspace content and optional shared workspace content
    """
    # Get private workspace content
    workspace_manager = WorkspaceManager()
    private_workspace = (
      workspace_manager.load_workspace(
        self.workspace_path
      )
    )

    # Check for shared workspace
    shared_workspace_path = message.metadata.get(
      "shared_workspace"
    )
    if shared_workspace_path:
      shared_workspace = (
        workspace_manager.load_workspace(
          shared_workspace_path
        )
      )
    else:
      shared_workspace = None

    return private_workspace, shared_workspace

  async def _update_workspaces(
    self,
    message: Message,
    private_update_template: str | None = None,
    shared_update_template: str | None = None,
    update_category: str | None = None,
  ) -> tuple[str, str | None]:
    """Update workspaces with new content."""
    workspace_manager = WorkspaceManager()

    private_workspace = (
      await workspace_manager.update_workspace(
        self.workspace_path,
        message,
        self,
        private_update_template,
        update_category,
      )
    )

    # Check for shared workspace
    shared_workspace_path = message.metadata.get(
      "shared_workspace"
    )
    if shared_workspace_path:
      shared_workspace = (
        await workspace_manager.update_workspace(
          shared_workspace_path,
          message,
          self,
          shared_update_template,
          update_category,
        )
      )
    else:
      shared_workspace = None

    return private_workspace, shared_workspace

  async def _handle_conversation(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Handle conversation with LLM integration."""
    logger.debug(
      f"Handling conversation for message: {message}"
    )
    messages = self._prepare_messages(message)
    tools = self.tool_manager.get_tools_schema()
    logger.trace(f"Tools schema: {tools}")
    tool_choice = None
    if self.config.required_tool:
      # Validate required tool exists in schema
      available_tools = {
        tool["function"]["name"]
        for tool in tools
        if tool.get("type") == "function"
        and "function" in tool
      }
      if (
        self.config.required_tool
        not in available_tools
      ):
        raise ValueError(
          f"Required tool '{self.config.required_tool}' not found in available tools: {available_tools}"
        )

      tool_choice = {
        "type": "function",  # OpenAI format
        "function": {
          "name": self.config.required_tool
        },
      }
    logger.trace(f"Tool choice: {tool_choice}")
    try:
      logger.info("Starting LLM conversation...")
      response = await acompletion(
        model=self.config.model,
        messages=messages,
        temperature=self.config.temperature,
        stream=self.config.stream,
        tools=tools if tools else None,
        tool_choice=tool_choice,
        timeout=self.config.timeout,
      )
      if self.config.stream:
        async for (
          chunk
        ) in self._process_streaming_response(
          cast(CustomStreamWrapper, response)
        ):
          yield chunk
      else:
        yield await self._process_single_response(
          cast(ModelResponse, response)
        )

    except Exception as e:
      logger.exception(
        f"Exception occurred during conversation handling: {str(e)}"
      )
      yield Response(
        content=f"Error generating response: {str(e)}",
        metadata={"error": True},
      )

  def _prepare_messages(
    self, message: Message
  ) -> list[ChatCompletionMessageParam]:
    """Prepare messages for LLM."""
    messages: list[ChatCompletionMessageParam] = []

    if self.config.system_prompt_template:
      # Render template with message metadata
      system_prompt = self.config.render_system_prompt(
        message.metadata
      )
      messages.append(
        Message.system(
          system_prompt
        ).to_chat_completion_message()
      )

    if history := message.metadata.get("history", []):
      messages.extend(
        Message.from_history(
          msg
        ).to_chat_completion_message()
        for msg in history
      )

    messages.append(
      message.to_chat_completion_message()
    )

    return messages

  async def _process_streaming_response(
    self,
    response: CustomStreamWrapper,
  ) -> AsyncIterator[Response]:
    """Process streaming LLM response."""
    tool_calls: list[ChatCompletionDeltaToolCall] = []
    accumulated_content: str = ""

    async for chunk in response:
      choices_list = cast(
        list[StreamingChoices],
        chunk["choices"],
      )
      choices: StreamingChoices = choices_list[0]
      # logger.trace(f"Choices: {choices}")
      # Handle tool calls completion
      if (
        hasattr(choices, "finish_reason")
        and choices["finish_reason"]
        in ("tool_calls", "stop")
        and not tool_calls
      ):
        logger.debug(
          "Stream completed with no tool calls present"
        )
        # Ensure tool calls are present if a tool choice was specified
        if self.config.required_tool:
          raise ValueError(
            "Expected tool call in response but none found."
          )
      elif (
        hasattr(choices, "finish_reason")
        and choices["finish_reason"]
        in ("tool_calls", "stop")
        and tool_calls
      ):
        results = []
        for tool_call in tool_calls:
          result = (
            await self.tool_manager.execute_tool(
              {
                "name": tool_call.function["name"],
                "arguments": tool_call.function[
                  "arguments"
                ],
              }
            )
          )
          results.append(result)

          # Update the last response with the tool call result
          if formatted_response := result.metadata.get(
            "formatted_response"
          ):
            self.state.last_response = (
              formatted_response
            )
          else:
            self.state.last_response = result.content

        # Yield all results
        for result in results:
          yield result
        return

      delta = choices.delta
      if delta.tool_calls:
        for tool_call in cast(
          list[ChatCompletionDeltaToolCall],
          delta["tool_calls"],
        ):
          if len(tool_calls) <= tool_call.index:
            assert len(tool_calls) == tool_call.index
            tool_calls.append(
              ChatCompletionDeltaToolCall(
                function=Function(
                  arguments=None,
                ),
                index=tool_call.index,
              )
            )

          if tool_call.function.name:
            tool_calls[
              tool_call.index
            ].function.name = tool_call.function.name

          if tool_call.function.arguments:
            tool_calls[
              tool_call.index
            ].function.arguments += (
              tool_call.function.arguments
            )

      # Handle content
      if (
        hasattr(delta, "content")
        and delta["content"] is not None
      ):
        accumulated_content += str(delta["content"])
        yield Response(
          content=delta["content"],
          metadata={"streaming": True},
        )

    if accumulated_content:
      logger.debug(
        f"Accumulated content: {accumulated_content}"
      )
      self.state.last_response = accumulated_content

  async def _process_single_response(
    self, response: ModelResponse
  ) -> Response:
    """Process non-streaming LLM response."""
    if not response.choices:
      raise ValueError("No choices in response")

    message = response.choices[0].message
    if not message:
      raise ValueError("No message in response")

    if not (
      hasattr(message, "tool_calls")
      and message.tool_calls
    ):
      logger.debug("No tool calls present in response")
      # Ensure tool calls are present if a tool choice was specified
      if self.config.required_tool:
        raise ValueError(
          "Expected tool call in response but none found."
        )
    elif (
      hasattr(message, "tool_calls")
      and message.tool_calls
    ):
      results = []
      for tool_call in message.tool_calls:
        result = await self.tool_manager.execute_tool(
          {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
          }
        )
        results.append(result)
      return Response(
        content=str(results),
        metadata={"tool_calls": True},
      )

    if message.content:
      self.state.last_response = message.content
      return Response(content=message.content)

    raise ValueError("No content in message")

  async def _handle_function(
    self, message: Message
  ) -> Response:
    """Handle function calls."""
    if not isinstance(message.content, dict):
      return Response(
        content="Invalid function call format",
        metadata={"error": True},
      )

    try:
      return await self.tool_manager.execute_tool(
        message.content
      )
    except Exception as e:
      logger.error(
        "Function execution error", exc_info=True
      )
      return Response(
        content=f"Error executing function: {str(e)}",
        metadata={"error": True},
      )

  async def _handle_event(
    self, message: Message
  ) -> Response:
    """Handle events."""
    event_type = message.metadata.get(
      "event_type", "unknown"
    )
    self.state.context[f"last_{event_type}"] = (
      message.content
    )
    return Response(
      content=f"Processed event: {event_type}",
      metadata={"event_type": event_type},
    )

  async def generate_response(
    self, message: Message
  ) -> Response:
    """Generate a non-streaming response from the LLM."""
    logger.debug(
      f"Generating response for message: {message}"
    )
    messages = self._prepare_messages(message)
    try:
      logger.info(
        "Starting LLM response generation..."
      )
      response = await acompletion(
        model=self.config.model,
        messages=messages,
        temperature=self.config.temperature,
        stream=False,
        timeout=self.config.timeout,
      )
      model_response = cast(ModelResponse, response)
      if not model_response.choices:
        logger.error("No choices in response")
        raise ValueError("No choices in response")

      message = model_response.choices[0].message
      if not message:
        logger.error("No message in response")
        raise ValueError("No message in response")

      if message.content:
        return Response(content=message.content)
      else:
        logger.error("No content in message")
        raise ValueError("No content in message")

    except Exception as e:
      logger.exception(f"LLM error occurred: {str(e)}")
      return Response(
        content=f"Error generating response: {str(e)}",
        metadata={"error": True},
      )

  async def generate_streaming_response(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Generate a streaming response from the LLM.

    Parameters
    ----------
    message : Message
        The input message to process

    Yields
    -------
    Response
        Streamed responses from the LLM
    """
    messages = self._prepare_messages(message)
    try:
      response = await acompletion(
        model=self.config.model,
        messages=messages,
        temperature=self.config.temperature,
        stream=True,
        timeout=self.config.timeout,
      )

      async for (
        chunk
      ) in self._process_streaming_response(
        cast(CustomStreamWrapper, response)
      ):
        yield chunk

    except Exception as e:
      logger.error("LLM error", exc_info=True)
      yield Response(
        content=f"Error generating response: {str(e)}",
        metadata={"error": True},
      )

  async def generate_vision_response(
    self,
    prompt: str,
    image_path: str | Path,
  ) -> Response:
    """Generate a non-streaming response from the vision model.

    Parameters
    ----------
    prompt : str
        Text prompt for the vision model
    image_path : str | Path
        Path to image file

    Returns
    -------
    Response
        The generated response from the vision model
    """
    messages = create_vision_messages(
      prompt, image_path
    )

    try:
      model = (
        self.config.vision_model or self.config.model
      )
      response = await acompletion(
        model=model,
        messages=messages,
        temperature=self.config.temperature,
        stream=False,
        timeout=self.config.timeout,
      )

      model_response = cast(ModelResponse, response)
      if not model_response.choices:
        raise ValueError("No choices in response")

      message = model_response.choices[0].message
      if not message:
        raise ValueError("No message in response")

      if message.content:
        return Response(content=message.content)
      else:
        raise ValueError("No content in message")

    except Exception as e:
      logger.error("Vision model error", exc_info=True)
      return Response(
        content=f"Error processing image: {str(e)}",
        metadata={"error": True},
      )

  async def generate_streaming_vision_response(
    self,
    prompt: str,
    image_path: str | Path,
  ) -> AsyncIterator[Response]:
    """Generate a streaming response from the vision model."""
    messages = create_vision_messages(
      prompt, image_path
    )

    try:
      model = (
        self.config.vision_model or self.config.model
      )
      response = await acompletion(
        model=model,
        messages=messages,
        temperature=self.config.temperature,
        stream=True,
        timeout=self.config.timeout,
      )

      # Process streaming response directly
      async for chunk in cast(
        CustomStreamWrapper, response
      ):
        choices = chunk["choices"][0]
        if content := choices.delta.get("content"):
          yield Response(content=content)

    except Exception as e:
      logger.error("Vision model error", exc_info=True)
      yield Response(
        content=f"Error processing image: {str(e)}",
        metadata={"error": True},
      )

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Process messages to coordinate memory operations.

    Parameters
    ----------
    message : Message
        The message to process

    Yields
    ------
    Response
        Responses from memory operations
    """
    # Let the LLM evaluate the message using system prompt and tools
    async for response in self._handle_conversation(
      message
    ):
      yield response
