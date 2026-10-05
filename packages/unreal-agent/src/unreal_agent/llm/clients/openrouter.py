from dataclasses import dataclass

from unreal_agent.llm.clients._base import ResponsesClient, Trace, base_url, require
from unreal_agent.llm.responsesapi import CacheKeyPlacement
from unreal_agent.primitives import RemoteClient

__all__ = ["Client", "Config"]


@dataclass(slots=True)
class Config:
    api_key: str = ""
    base_url: str = ""
    max_attempts: int | None = None
    trace: Trace | None = None


class Client(ResponsesClient):
    def __init__(self, config: Config) -> None:
        require(config.api_key, "OpenRouter API key must be set")
        url = base_url(config.base_url)
        if not url:
            raise ValueError("OpenRouter base URL must be set")
        super().__init__(
            RemoteClient(),
            url + "/responses",
            {"Authorization": ["Bearer " + config.api_key], "Content-Type": ["application/json"]},
            cache_key_placement=CacheKeyPlacement(header="x-session-id"),
            max_attempts=config.max_attempts,
            trace=config.trace,
            # OpenRouter only caches prompts for upstreams that need explicit breakpoints
            # (Anthropic among them) when the request opts in; the top-level field places
            # the breakpoint on the last cacheable block and moves it forward each turn.
            # With the x-session-id header, every turn of a session lands on one warm upstream.
            # The one-hour TTL survives the gaps an agent session produces (a long tool
            # call, a many-minute reasoning turn, the tool heartbeat); it costs 2x instead
            # of 1.25x on the small per-turn cache writes and saves a full-prefix rewrite
            # per gap.
            extensions={"cache_control": b'{"type":"ephemeral","ttl":"1h"}'},
        )
