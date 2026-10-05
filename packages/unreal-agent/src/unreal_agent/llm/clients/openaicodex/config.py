from dataclasses import dataclass

from unreal_agent.llm.clients._base import Trace

__all__ = ["Config"]


@dataclass(slots=True)
class Config:
    """Codex subscription settings. `auth_file` is read once at construction and is
    exclusive with inline credentials."""

    access_token: str = ""
    account_id: str = ""
    auth_file: str = ""
    base_url: str = ""
    max_attempts: int | None = None
    trace: Trace | None = None
