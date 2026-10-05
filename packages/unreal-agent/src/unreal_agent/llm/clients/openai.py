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
        require(config.api_key, "OpenAI API key must be set")
        url = base_url(config.base_url)
        if not url:
            raise ValueError("OpenAI base URL must be set")
        super().__init__(
            RemoteClient,
            url + "/responses",
            {"Authorization": ["Bearer " + config.api_key], "Content-Type": ["application/json"]},
            cache_key_placement=CacheKeyPlacement(use_prompt_cache_key_field=True),
            max_attempts=config.max_attempts,
            trace=config.trace,
        )
