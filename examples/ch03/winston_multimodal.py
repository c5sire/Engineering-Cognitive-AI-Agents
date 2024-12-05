"""Winston with cognitive workspace and multi-modal capabilities."""

from pathlib import Path
from typing import AsyncIterator

from winston.core.agent import AgentConfig, BaseAgent
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import Agent, System
from winston.ui.chainlit_app import AgentChat


class MultimodalAgent(BaseAgent):
  """Agent with multi-modal cognitive capabilities."""

  @classmethod
  def can_handle(cls, message: Message) -> bool:
    """Check if this agent can handle the message."""
    return "image_path" in message.metadata

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Process an incoming message."""
    # Accumulate the complete vision response
    visual_observation = ""

    # Handle image analysis if present
    if self.can_handle(message):
      async for (
        response
      ) in self.generate_streaming_vision_response(
        "Generate a detailed description of the image.",
        message.metadata["image_path"],
      ):
        visual_observation += response.content
        yield response

    # Get workspaces
    private_workspace, shared_workspace = (
      self._get_workspaces(message)
    )

    # Render response prompt
    response_prompt = self.config.render_system_prompt(
      {
        "message": message.content,
        "private_workspace": private_workspace,
        "shared_workspace": shared_workspace,
        "visual_observation": visual_observation,
      }
    )

    # Stream response
    accumulated_content = ""
    async for (
      response
    ) in self.generate_streaming_response(
      Message(
        content=response_prompt,
        metadata=message.metadata,
      )
    ):
      accumulated_content += response.content
      yield response

    # Update workspaces
    await self._update_workspaces(
      Message(
        content=accumulated_content,
        metadata=message.metadata,
      ),
      update_category="Visual observation"
      if visual_observation
      else None,
    )


class MultimodalWinstonChat(AgentChat):
  """Chat interface for multi-modal Winston."""

  def __init__(self) -> None:
    self.paths = AgentPaths(root=Path(__file__).parent)
    super().__init__()

  def create_agent(self, system: System) -> Agent:
    config = AgentConfig.from_yaml(
      self.paths.config
      / "agents/winston_multimodal.yaml",
    )
    return MultimodalAgent(
      system=system,
      config=config,
      paths=self.paths,
    )


# Create the application
app = MultimodalWinstonChat()
