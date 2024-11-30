"""Path management utilities for Winston agents."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentPaths:
  """Manages paths relative to the agent's root directory."""

  root: Path
  system_root: Path | None = None

  @property
  def config(self) -> Path:
    """Get the config directory path."""
    return self.root / "config"

  @property
  def workspaces(self) -> Path:
    """Get the workspaces directory path."""
    return self.root / "workspaces"

  @property
  def system_config(self) -> Path:
    """Get the system config directory path.

    If system_root is specified, uses that as the base for system config.
    Otherwise, falls back to the legacy behavior of using root/config/system.
    """
    if self.system_root is not None:
      return self.system_root / "config"
    return self.root / "config" / "system"

  @property
  def system_agents_config(self) -> Path:
    """Get the system-wide agents config directory path."""
    return self.system_config / "agents"
