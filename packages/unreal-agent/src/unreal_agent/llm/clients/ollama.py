from dataclasses import dataclass
from typing import Final

from unreal_agent.llm.clients._base import ResponsesClient, base_url
from unreal_agent.primitives import RemoteClient

__all__ = ["BASE_URL", "Client", "Config"]

BASE_URL: Final = "http://localhost:11434/v1"


@dataclass(slots=True)
class Config:
    base_url: str = ""
    max_attempts: int | None = None


class Client(ResponsesClient):
    """A local Ollama server. Requests carry no credentials."""

    def __init__(self, config: Config) -> None:
        url = base_url(config.base_url) or BASE_URL
        super().__init__(
            RemoteClient,
            url + "/responses",
            {"Content-Type": ["application/json"]},
            max_attempts=config.max_attempts,
        )
