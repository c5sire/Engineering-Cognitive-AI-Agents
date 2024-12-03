"""Step management utilities for Winston's processing pipeline."""

from typing import Literal

import chainlit as cl

from winston.core.messages import (
  Response,
  ResponseType,
)

StepType = Literal[
  "run",
  "tool",
  "llm",
  "embedding",
  "retrieval",
  "rerank",
]


class ProcessingStep:
  """Context manager for handling processing steps."""

  def __init__(
    self,
    name: str,
    step_type: StepType,
    show_input: bool = True,
  ):
    """
    Initialize a processing step.

    Parameters
    ----------
    name : str
        Name of the step
    step_type : StepType
        Type of the step (must be one of Chainlit's supported types)
    show_input : bool, optional
        Whether to show input in UI, by default True
    """
    self.name = name
    self.step_type = step_type
    self.show_input = show_input
    self.cl_step: cl.Step | None = None

  async def __aenter__(self) -> "ProcessingStep":
    """Enter the context manager."""
    self.cl_step = cl.Step(
      name=self.name,
      type=self.step_type,
      show_input=self.show_input,
    )
    await self.cl_step.__aenter__()
    return self

  async def __aexit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: object | None,
  ) -> None:
    """Exit the context manager."""
    if self.cl_step:
      await self.cl_step.__aexit__(
        exc_type, exc_val, exc_tb
      )

  async def stream_response(
    self, response: Response
  ) -> None:
    """
    Stream response based on its type.

    Parameters
    ----------
    response : Response
        Response to stream
    """
    if not self.cl_step:
      return

    if (
      response.response_type
      == ResponseType.TOOL_RESULT
    ):
      self.cl_step.input = response.metadata.get(
        "tool_input", ""
      )
      self.cl_step.output = response.content
    elif response.streaming:
      await self.cl_step.stream_token(response.content)
    else:
      self.cl_step.output = response.content
