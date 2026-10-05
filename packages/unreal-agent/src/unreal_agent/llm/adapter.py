from dataclasses import dataclass
from typing import Protocol

from unreal_agent.llm.model import Request, Response

__all__ = ["Adapter", "RequestOptions"]


@dataclass(frozen=True, slots=True)
class RequestOptions:
    CacheKey: str = ""


class Adapter(Protocol):
    """A model provider. Cancelling the calling task cancels the request."""

    async def respond(self, request: Request, options: RequestOptions) -> Response: ...
