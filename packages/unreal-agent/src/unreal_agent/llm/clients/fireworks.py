from dataclasses import dataclass

import msgspec

from unreal_agent.gojson import Int64, JSONError
from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.clients._base import ResponsesClient, Trace, base_url, require
from unreal_agent.llm.model import Request, Response
from unreal_agent.llm.responsesapi import CacheKeyPlacement
from unreal_agent.llm.responsesapi._wire import decode_object
from unreal_agent.primitives import RemoteClient

__all__ = ["Client", "Config"]


@dataclass(slots=True)
class Config:
    api_key: str = ""
    base_url: str = ""
    max_attempts: int | None = None
    trace: Trace | None = None


class _PromptTokensDetails(msgspec.Struct):
    cached_tokens: Int64 | None = None


class _CompletionTokensDetails(msgspec.Struct):
    reasoning_tokens: Int64 | None = None


class _ChatUsage(msgspec.Struct):
    prompt_tokens: Int64 | None = None
    completion_tokens: Int64 | None = None
    prompt_tokens_details: _PromptTokensDetails | None = None
    completion_tokens_details: _CompletionTokensDetails | None = None


class Client(ResponsesClient):
    """Fireworks reports chat-completions usage fields, which override the Responses ones."""

    def __init__(self, config: Config) -> None:
        require(config.api_key, "fireworks API key must be set")
        url = base_url(config.base_url)
        if not url:
            raise ValueError("fireworks base URL must be set")
        super().__init__(
            RemoteClient(),
            url + "/responses",
            {"Authorization": ["Bearer " + config.api_key], "Content-Type": ["application/json"]},
            cache_key_placement=CacheKeyPlacement(header="x-session-affinity"),
            max_attempts=config.max_attempts,
            trace=config.trace,
        )

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        response = await self.adapter.respond(request, options)
        if len(response.Usage.Raw) == 0:
            return response
        try:
            usage = decode_object(bytes(response.Usage.Raw), _ChatUsage)
        except JSONError as error:
            raise ValueError(f"decode Fireworks response usage: {error}") from error
        if usage.prompt_tokens is not None:
            response.Usage.InputTokens = usage.prompt_tokens
        if usage.completion_tokens is not None:
            response.Usage.OutputTokens = usage.completion_tokens
        details = usage.prompt_tokens_details
        if details is not None and details.cached_tokens is not None:
            response.Usage.CachedInputTokens = details.cached_tokens
        completion = usage.completion_tokens_details
        if completion is not None and completion.reasoning_tokens is not None:
            response.Usage.ReasoningTokens = completion.reasoning_tokens
        return response
