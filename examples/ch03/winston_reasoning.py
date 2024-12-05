"""Winston with cognitive workspace and reasoning capabilities."""

from pathlib import Path
from typing import AsyncIterator

from winston.core.agent import AgentConfig, BaseAgent
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import Agent, System
from winston.ui.chainlit_app import AgentChat


class ReasoningAgent(BaseAgent):
  """Agent with reasoning capabilities."""

  @classmethod
  def can_handle(cls, message: Message) -> bool:
    """Check if this agent can handle the message."""
    return any(
      trigger in message.content.lower()
      for trigger in [
        "analyze",
        "understand",
        "explain why",
        "what's causing",
        "help me understand",
        "struggling with",
        "having trouble",
      ]
    )

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Process message using private and optionally shared context."""

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
      update_category=(
        "Situational reasoning"
        if self.can_handle(message)
        else "General reasoning"
      ),
    )


class ReasoningWinstonChat(AgentChat):
  """Chat interface for reasoning Winston."""

  def __init__(self) -> None:
    # Set up paths relative to this file's location
    self.paths = AgentPaths(root=Path(__file__).parent)
    super().__init__()

  def create_agent(self, system: System) -> Agent:
    config = AgentConfig.from_yaml(
      self.paths.config
      / "agents/winston_reasoning.yaml"
    )
    return ReasoningAgent(
      system=system,
      config=config,
      paths=self.paths,
    )


# Create the application
app = ReasoningWinstonChat()
