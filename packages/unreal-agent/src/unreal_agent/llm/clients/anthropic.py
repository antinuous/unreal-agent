from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Self

from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.clients._base import base_url, require
from unreal_agent.llm.messagesapi import Config as AdapterConfig
from unreal_agent.llm.messagesapi import Exchange, MessagesAdapter
from unreal_agent.llm.messagesapi.adapter import check_config
from unreal_agent.llm.model import Request, Response
from unreal_agent.primitives import RemoteClient

__all__ = ["DEFAULT_BASE_URL", "Client", "Config", "Exchange"]

DEFAULT_BASE_URL: Final = "https://api.anthropic.com/v1"


@dataclass(slots=True)
class Config:
    api_key: str = ""
    base_url: str = ""
    fallback: bool | None = None
    cache_ttl: str = ""
    max_attempts: int | None = None
    trace: Callable[[Exchange], None] | None = None


class Client:
    """A Messages API adapter that owns its remote client. Close it with `aclose`."""

    def __init__(self, config: Config) -> None:
        require(config.api_key, "anthropic API key must be set")
        url = base_url(config.base_url) or DEFAULT_BASE_URL
        adapter_config = AdapterConfig(
            endpoint=url + "/messages",
            headers={
                "X-Api-Key": [config.api_key],
                "Anthropic-Version": ["2023-06-01"],
                "Content-Type": ["application/json"],
            },
            fallback=config.fallback,
            cache_ttl=config.cache_ttl,
            max_attempts=config.max_attempts,
            trace=config.trace,
        )
        # Validate before creating the remote client: closing it is async, so a
        # failed constructor could not release it.
        check_config(adapter_config)
        self.remote = RemoteClient()
        self.adapter = MessagesAdapter(self.remote, adapter_config)

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        return await self.adapter.respond(request, options)

    async def aclose(self) -> None:
        await self.remote.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
