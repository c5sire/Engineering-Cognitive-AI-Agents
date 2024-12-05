"""Winston with cognitive workspace and basic memory capabilities."""

from pathlib import Path
from typing import AsyncIterator

from winston.core.agent import AgentConfig, BaseAgent
from winston.core.messages import Message, Response
from winston.core.paths import AgentPaths
from winston.core.protocols import Agent, System
from winston.ui.chainlit_app import AgentChat


class MemoryAgent(BaseAgent):
  """Agent with basic memory capabilities."""

  async def process(
    self,
    message: Message,
  ) -> AsyncIterator[Response]:
    """Process message in private memory workspace."""
    print(
      f"MemoryWinston processing: {message.content}"
    )

    private_workspace, shared_workspace = (
      self._get_workspaces(message)
    )

    response_prompt = self.config.render_system_prompt(
      {
        "private_workspace": private_workspace,
        "shared_workspace": shared_workspace,
        "message": message.content,
      }
    )

    # Stream responses and accumulate content
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

    # After processing, update workspace(s)
    if not accumulated_content:
      return

    await self._update_workspaces(
      Message(
        content=accumulated_content,
        metadata=message.metadata,
      ),
    )


class MemoryWinstonChat(AgentChat):
  """Chat interface for cognitive Winston."""

  def __init__(self) -> None:
    # Set up paths relative to this file's location
    self.paths = AgentPaths(root=Path(__file__).parent)
    super().__init__()

  def create_agent(self, system: System) -> Agent:
    config = AgentConfig.from_yaml(
      self.paths.config / "agents/winston_memory.yaml"
    )
    return MemoryAgent(
      system=system,
      config=config,
      paths=self.paths,
    )


# Create the application
app = MemoryWinstonChat()
