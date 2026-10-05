"""Runner configuration: the request parser, tool factory, and LLM providers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from unreal_agent.llm import Adapter
from unreal_agent.operation import RemoteJobHandler
from unreal_agent.runner.request import Request
from unreal_agent.runner.streams import Reader
from unreal_agent.session import ID as SessionID
from unreal_agent.tool import Registry, StaticTranslators

__all__ = [
    "Client",
    "Config",
    "Getenv",
    "Provider",
    "RequestParser",
    "ToolConfig",
    "ToolFactory",
    "Tools",
]

type Getenv = Callable[[str], str]
"""Go's os.Getenv: an unset variable reads as the empty string."""


class Client(Adapter, Protocol):
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class Provider:
    """An LLM provider. An empty `api_key_environment` delegates authentication to `new_client`.

    `new_client(api_key, base_url, max_attempts, getenv)` raises on invalid configuration.
    """

    name: str
    new_client: Callable[[str, str, int, Getenv], Client] | None
    base_url: str = ""
    default_model: str = ""
    api_key_environment: str = ""


@dataclass(slots=True)
class ToolConfig:
    translators: StaticTranslators = field(default_factory=StaticTranslators)
    names: list[str] = field(default_factory=list)
    session_id: SessionID = ""
    getenv: Getenv = lambda _: ""


@dataclass(slots=True)
class Tools:
    registry: Registry
    remote_jobs: list[RemoteJobHandler] = field(default_factory=list)
    close: Callable[[], Awaitable[None]] | None = None


type ToolFactory = Callable[[ToolConfig], Awaitable[Tools]]
type RequestParser = Callable[[Reader], tuple[Request, ToolFactory | None]]
"""Decode a request and return the factory for its tools. Raises on invalid input."""


@dataclass(slots=True)
class Config:
    name: str = ""
    providers: list[Provider] = field(default_factory=list)
    parse_request: RequestParser | None = None
