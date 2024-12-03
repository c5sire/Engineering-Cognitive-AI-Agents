# src/winston/ui/chainlit_app.py
"""Chainlit integration for generic agent chat interface."""

from typing import cast

import chainlit as cl
from loguru import logger

from winston.core.messages import (
  Message,
  Response,
  ResponseType,
)
from winston.core.protocols import Agent, System
from winston.core.steps import ProcessingStep
from winston.core.system import AgentSystem


class AgentChat:
  """Generic Chainlit-based chat interface for any agent."""

  def __init__(self) -> None:
    """
    Initialize AgentChat with system and register Chainlit handlers.
    """
    self.system = AgentSystem()

    # Register class methods as Chainlit handlers
    cl.on_chat_start(self.start)
    cl.on_message(self.handle_message)

  async def start(self) -> None:
    """
    Initialize the chat session.

    Creates and registers an agent instance and initializes the chat history.
    """
    agent = self.create_agent(self.system)

    # Store system and agent_id in session
    cl.user_session.set("system", self.system)  # type: ignore
    cl.user_session.set("agent_id", agent.id)  # type: ignore
    cl.user_session.set("history", [])  # type: ignore

  async def handle_message(
    self,
    message: cl.Message,
  ) -> None:
    """
    Handle incoming chat messages with enhanced step visualization.

    Parameters
    ----------
    message : cl.Message
        The incoming message from the user.
    """
    system: AgentSystem = cast(
      AgentSystem,
      cl.user_session.get("system"),
    )
    agent_id: str = cast(
      str,
      cl.user_session.get("agent_id"),
    )

    # Get history in proper format
    history = cl.user_session.get("history", [])
    metadata = {"history": history}

    # Handle image uploads
    if message.elements and len(message.elements) > 0:
      image = message.elements[0]
      if isinstance(image, cl.Image):
        metadata["image_path"] = image.path

    current_msg: cl.Message | None = None
    accumulated_content: list[str] = []

    try:
      async for response in system.invoke_conversation(
        agent_id,
        message.content,
        context=metadata,
      ):
        if (
          response.response_type
          == ResponseType.USER_MESSAGE
        ):
          # Create message if it doesn't exist
          if current_msg is None:
            current_msg = cl.Message(content="")
            await current_msg.send()

          # Always stream tokens for user messages
          await current_msg.stream_token(
            response.content
          )
          accumulated_content.append(response.content)

        elif (
          response.response_type
          == ResponseType.TOOL_RESULT
        ):
          async with ProcessingStep(
            name=response.metadata.get(
              "tool_name", "Tool"
            ),
            step_type="tool",
          ) as step:
            await step.stream_response(response)

        elif (
          response.response_type
          == ResponseType.INTERNAL_STEP
        ):
          async with ProcessingStep(
            name=response.step_name or "Processing",
            step_type="run",
          ) as step:
            await step.stream_response(response)

      # Ensure final content is updated
      if current_msg is not None:
        await current_msg.update()

      # Update history with conversation
      history = cast(
        list[dict[str, str]],
        cl.user_session.get("history", []),
      )

      user_message = Message(content=message.content)
      assistant_message = Response(
        content="".join(accumulated_content)
      )

      history.extend(
        [
          user_message.to_history_format(),
          assistant_message.to_history_format(),
        ]
      )

      cl.user_session.set("history", history)  # type: ignore

    except Exception as e:
      logger.exception("Error processing message")
      await cl.Message(
        content=f"An error occurred: {str(e)}"
      ).send()

  def create_agent(
    self,
    system: System,
  ) -> Agent:
    """
    Create the agent instance.

    Returns
    -------
    Agent
        The agent instance to use for chat.

    Raises
    ------
    NotImplementedError
        This method must be implemented by subclasses.
    """
    raise NotImplementedError(
      "Subclasses must implement create_agent method"
    )
