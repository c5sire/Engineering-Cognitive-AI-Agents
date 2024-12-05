"""Step management utilities for Winston's processing pipeline."""

import json
from contextvars import ContextVar, Token
from typing import Literal

import chainlit as cl

from winston.core.messages import (
  Response,
)

StepType = Literal[
  "run",
  "tool",
  "llm",
  "embedding",
  "retrieval",
  "rerank",
]

current_step: ContextVar["ProcessingStep | None"] = (
  ContextVar(
    "current_step",
    default=None,
  )
)


class ProcessingStep:
  """Context manager for handling processing steps."""

  def __init__(
    self,
    name: str,
    step_type: StepType,
    show_input: bool = True,
  ) -> None:
    """
    Initialize a processing step.

    Parameters
    ----------
    name : str
        Name of the step
    step_type : StepType
        Type of the step (must be one of Chainlit's supported types)
    show_input : bool
        Whether to show input in UI
    """
    self.name = name
    self.step_type = step_type
    self.show_input = show_input
    self.cl_step: cl.Step | None = None
    self.token: Token | None = None

  async def __aenter__(self) -> "ProcessingStep":
    """
    Enter the context manager.

    Returns
    -------
    ProcessingStep
        The processing step instance
    """
    self.cl_step = cl.Step(
      name=self.name,
      type=self.step_type,
      show_input=self.show_input,
    )

    self.token = current_step.set(self)
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
        exc_type,
        exc_val,
        exc_tb,
      )

    if self.token:
      current_step.reset(self.token)

  async def show_response(
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

    if response.metadata:
      self.cl_step.input = (
        "```json\n"
        + json.dumps(
          response.metadata, indent=2, default=str
        )
        + "\n```"
      ).strip()
    if response.streaming and response.content:
      await self.cl_step.stream_token(response.content)
    elif response.content:
      self.cl_step.output = response.content
