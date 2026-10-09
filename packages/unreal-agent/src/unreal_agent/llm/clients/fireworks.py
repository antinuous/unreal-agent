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
        require(config.api_key, "fireworks API key must be set")
        url = base_url(config.base_url)
        if not url:
            raise ValueError("fireworks base URL must be set")
        super().__init__(
            RemoteClient,
            url + "/responses",
            {"Authorization": ["Bearer " + config.api_key], "Content-Type": ["application/json"]},
            cache_key_placement=CacheKeyPlacement(header="x-session-affinity"),
            max_attempts=config.max_attempts,
            trace=config.trace,
        )
