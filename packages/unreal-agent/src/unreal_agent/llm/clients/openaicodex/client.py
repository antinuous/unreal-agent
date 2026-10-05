import asyncio
import ipaddress
import re
from typing import Final
from urllib.parse import urlsplit

from unreal_agent._go import go_trim_space
from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.clients._base import ResponsesClient
from unreal_agent.llm.clients.openaicodex.config import Config
from unreal_agent.llm.clients.openaicodex.credentials import credentials
from unreal_agent.llm.model import Request, Response
from unreal_agent.llm.responsesapi import APIError, CacheKeyPlacement
from unreal_agent.primitives import RemoteClient, new_remote_http_client
from unreal_agent.primitives.primitive_event import cancel_requested

__all__ = ["BASE_URL", "Client"]

BASE_URL: Final = "https://chatgpt.com/backend-api/codex"
_PORT: Final = re.compile(r"(:[0-9]*)?")


class Client(ResponsesClient):
    def __init__(self, config: Config) -> None:
        url = validate_base_url(config.base_url)
        if config.max_attempts is not None and config.max_attempts <= 0:
            raise ValueError("max attempts must be positive")
        found = credentials(config)
        super().__init__(
            _remote_without_redirects,
            url + "/responses",
            {
                "Authorization": ["Bearer " + found.access_token],
                "ChatGPT-Account-ID": [found.account_id],
                "Content-Type": ["application/json"],
                "originator": ["unreal-agent"],
                "User-Agent": ["unreal-agent"],
            },
            cache_key_placement=CacheKeyPlacement(
                header="session-id", use_prompt_cache_key_field=True
            ),
            max_attempts=config.max_attempts,
            trace=config.trace,
        )

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        if cancel_requested():
            raise asyncio.CancelledError
        if request.Model.MaxOutputTokens is not None:
            raise ValueError("codex does not support max_output_tokens")
        try:
            return await self.adapter.respond(request, options)
        except APIError as error:
            if error.status_code == 401:
                raise error.wrap(
                    "codex credentials rejected; renew them externally and recreate the client"
                ) from error
            raise


def _remote_without_redirects() -> RemoteClient:
    # Never forward subscription credentials through redirects.
    http_client = new_remote_http_client()
    http_client.follow_redirects = False
    return RemoteClient(http_client)


def validate_base_url(value: str) -> str:
    value = go_trim_space(value).rstrip("/")
    if not value or value == BASE_URL:
        return BASE_URL
    try:
        parsed = urlsplit(value)
    except ValueError:
        parsed = None
    if (
        parsed is not None
        and "@" not in parsed.netloc
        and not parsed.query
        and not parsed.fragment
        and parsed.scheme in ("http", "https")
        and _is_loopback(parsed.netloc)
    ):
        return value
    raise ValueError(
        "codex base URL must be https://chatgpt.com/backend-api/codex or an explicit "
        "loopback IP endpoint"
    )


def _is_loopback(netloc: str) -> bool:
    if netloc.startswith("["):
        host, _, rest = netloc[1:].partition("]")
    else:
        host, _, port = netloc.partition(":")
        rest = ":" + port if port or netloc.endswith(":") else ""
    if not _PORT.fullmatch(rest) or "%" in host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
