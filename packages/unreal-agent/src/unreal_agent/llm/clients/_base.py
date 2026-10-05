from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Self

from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.model import Request, Response
from unreal_agent.llm.responsesapi import CacheKeyPlacement, Config, Exchange, ResponsesAdapter
from unreal_agent.llm.responsesapi._wire import trim_space_text
from unreal_agent.primitives import RemoteClient

type Trace = Callable[[Exchange], None]


class ResponsesClient:
    """A Responses API adapter that owns its remote client. Close it with `aclose`."""

    def __init__(
        self,
        remote: RemoteClient,
        endpoint: str,
        headers: Mapping[str, Sequence[str]],
        *,
        cache_key_placement: CacheKeyPlacement | None = None,
        max_attempts: int | None = None,
        trace: Trace | None = None,
        extensions: Mapping[str, bytes] | None = None,
    ) -> None:
        self.remote = remote
        self.adapter = ResponsesAdapter(
            remote,
            Config(
                endpoint=endpoint,
                headers=headers,
                cache_key_placement=cache_key_placement or CacheKeyPlacement(),
                max_attempts=max_attempts,
                trace=trace,
                extensions=extensions or {},
            ),
        )

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


def require(value: str, message: str) -> str:
    value = trim_space_text(value)
    if not value:
        raise ValueError(message)
    return value


def base_url(value: str) -> str:
    return trim_space_text(value).rstrip("/")
