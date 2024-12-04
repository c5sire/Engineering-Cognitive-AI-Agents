"""Core agent interfaces and base implementation."""

import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
  """Enhanced agent configuration with validation."""

  id: str = Field(
    ..., description="Unique identifier for the agent"
  )
  model: str = Field(
    ..., description="Model to use for the agent"
  )
  system_prompt_template: str = Field(
    ...,
    description="Jinja2 template for system prompt",
    alias="system_prompt",
  )
  temperature: float = Field(
    default=0.7,
    description="Temperature for model sampling",
  )
  stream: bool = Field(
    default=True,
    description="Whether to stream responses",
  )
  vision_model: str | None = Field(
    default=None,
    description="Model to use for vision tasks",
  )
  max_retries: int = Field(
    default=3, description="Maximum number of retries"
  )
  timeout: int = Field(
    default=60, description="Timeout in seconds"
  )
  workspace_template: str = Field(
    default="default",
    description="Workspace template to use",
  )
  required_tool: str | None = Field(
    default=None,
    description="Tool that must be called during processing",
  )

  def render_system_prompt(
    self, metadata: dict[str, Any]
  ) -> str:
    """Render system prompt template with metadata.

    Parameters
    ----------
    metadata : dict[str, Any]
        Metadata to render the template with

    Returns
    -------
    str
        Rendered system prompt
    """
    template = Template(self.system_prompt_template)
    return template.render(**metadata)

  @classmethod
  def from_yaml(
    cls, path: str | Path
  ) -> "AgentConfig":
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file

    Returns
    -------
    AgentConfig
        Loaded and validated configuration

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist
    ValueError
        If the configuration is invalid
    """
    path = Path(path)
    with path.open() as f:
      config_data = yaml.safe_load(f)

    return cls.model_validate(config_data)

  @classmethod
  def from_json(
    cls, path: str | Path
  ) -> "AgentConfig":
    """Load configuration from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the JSON configuration file

    Returns
    -------
    AgentConfig
        Loaded and validated configuration

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist
    ValueError
        If the configuration is invalid
    """
    path = Path(path)
    with path.open() as f:
      config_data = json.load(f)

    return cls.model_validate(config_data)
