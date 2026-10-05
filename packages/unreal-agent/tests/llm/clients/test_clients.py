"""Client tests against a local HTTP/1.1 server, like Go's httptest servers."""

import asyncio
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import Message as HeaderMessage
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Request,
    RequestOptions,
    Role,
    StopReason,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
)
from unreal_agent.llm.clients import (
    ResponsesClient,
    fireworks,
    ollama,
    openai,
    openaicodex,
    openrouter,
)
from unreal_agent.llm.responsesapi import APIError, Exchange

from tests.llm.clients._helpers import write_test_auth

CACHE_KEY = "84097828fc31a8c8d29210df48901a85de7fd013f686b17be77d1be29cb7a98b"
COMPLETED = (
    b'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}'
    b"\n\n"
)

# --- Local HTTP/1.1 test server -------------------------------------------------


@dataclass
class ServerRequest:
    method: str
    path: str
    headers: HeaderMessage
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


class ResponseWriter:
    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler
        self.headers: list[tuple[str, str]] = []
        self.status: int | None = None
        self._chunked = False

    def set_header(self, name: str, value: str) -> None:
        self.headers = [(key, item) for key, item in self.headers if key.lower() != name.lower()]
        self.headers.append((name, value))

    def write_header(self, status: int) -> None:
        if self.status is not None:
            return
        self.status = status
        lines = [f"HTTP/1.1 {status} {HTTPStatus(status).phrase}"]
        lines.extend(f"{name}: {value}" for name, value in self.headers)
        if not any(name.lower() == "content-length" for name, _ in self.headers):
            self._chunked = True
            lines.append("Transfer-Encoding: chunked")
        self._handler.wfile.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        self._handler.wfile.flush()

    def write(self, data: bytes) -> None:
        self.write_header(200)
        if not data:
            return
        if self._chunked:
            self._handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            self._handler.wfile.write(data)
        self._handler.wfile.flush()

    def finish(self) -> None:
        if self.status is None:
            self.set_header("Content-Length", "0")
            self.write_header(200)
        if self._chunked:
            self._handler.wfile.write(b"0\r\n\r\n")
            self._handler.wfile.flush()


type Handler = Callable[[ServerRequest, ResponseWriter], None]


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: Handler) -> None:
        self.test_handler = handler
        self.errors: list[BaseException] = []
        super().__init__(("127.0.0.1", 0), _RequestHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request = ServerRequest(self.command, self.path, self.headers, body)
        writer = ResponseWriter(self)
        server = cast(Server, self.server)
        try:
            server.test_handler(request, writer)
            writer.finish()
        except BrokenPipeError, ConnectionResetError:
            self.close_connection = True
        except BaseException as error:
            server.errors.append(error)
            self.close_connection = True

    do_GET = do_POST = _dispatch


@contextmanager
def serve(handler: Handler) -> Iterator[Server]:
    server = Server(handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.daemon = True
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
    if server.errors:
        raise server.errors[0]


def write_sse(writer: ResponseWriter, body: bytes) -> None:
    writer.set_header("Content-Type", "text/event-stream")
    writer.write(body)


def hello(model: str) -> Request:
    return Request(
        Model=Model(ID=model),
        Input=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello"))],
    )


# --- openai, openrouter, fireworks ----------------------------------------------


@pytest.mark.parametrize(
    ("new_client", "name"),
    [
        (lambda key, url: openai.Client(openai.Config(api_key=key, base_url=url)), "OpenAI"),
        (
            lambda key, url: openrouter.Client(openrouter.Config(api_key=key, base_url=url)),
            "OpenRouter",
        ),
        (
            lambda key, url: fireworks.Client(fireworks.Config(api_key=key, base_url=url)),
            "fireworks",
        ),
    ],
    ids=["openai", "openrouter", "fireworks"],
)
def test_new_client_requires_api_key_and_base_url(
    new_client: Callable[[str, str], ResponsesClient], name: str
) -> None:
    with pytest.raises(ValueError, match=f"^{name} API key must be set$"):
        new_client(" ", "")
    with pytest.raises(ValueError, match=f"^{name} base URL must be set$"):
        new_client("test-key", " ")


async def test_openai_client_calls_responses_api() -> None:
    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        assert request.path == "/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Accept"] == "text/event-stream"
        body = request.json()
        assert body["model"] == "gpt-test"
        assert body["stream"] is True
        assert body["prompt_cache_key"] == CACHE_KEY
        write_sse(
            writer,
            b'data: {"type":"response.completed","response":{"id":"resp-1","status":"completed",'
            b'"output":[],"usage":{}}}\n\n',
        )

    with serve(handler) as server:
        async with openai.Client(openai.Config(api_key="test-key", base_url=server.url)) as client:
            response = await client.respond(hello("gpt-test"), RequestOptions(CacheKey="session-1"))
    assert (response.ID, response.Stop) == ("resp-1", StopReason.COMPLETE)


async def test_openrouter_client_calls_responses_api() -> None:
    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        assert request.path == "/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Accept"] == "text/event-stream"
        body = request.json()
        assert body["model"] == "openai/gpt-test"
        assert body["stream"] is True
        assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert request.headers["x-session-id"] == CACHE_KEY
        assert "prompt_cache_key" not in body
        write_sse(
            writer,
            b'data: {"type":"response.completed","response":{"id":"resp-1","status":"completed",'
            b'"output":[],"usage":{}}}\n\n',
        )

    with serve(handler) as server:
        config = openrouter.Config(api_key="test-key", base_url=server.url + "/")
        async with openrouter.Client(config) as client:
            response = await client.respond(
                hello("openai/gpt-test"), RequestOptions(CacheKey="session-1")
            )
    assert (response.ID, response.Stop) == ("resp-1", StopReason.COMPLETE)


async def test_fireworks_client_calls_responses_api() -> None:
    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        assert request.path == "/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Accept"] == "text/event-stream"
        body = request.json()
        assert body["model"] == "accounts/fireworks/models/glm-5p3-flash"
        assert body["store"] is False
        assert body["stream"] is True
        assert request.headers["x-session-affinity"] == CACHE_KEY
        assert "prompt_cache_key" not in body
        write_sse(
            writer,
            b'data: {"type":"response.completed","response":{"id":"resp-1","status":"completed",'
            b'"output":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,'
            b'"prompt_tokens_details":{"cached_tokens":3},'
            b'"completion_tokens_details":{"reasoning_tokens":5}}}}\n\n',
        )

    with serve(handler) as server:
        config = fireworks.Config(api_key="test-key", base_url=server.url + "/")
        async with fireworks.Client(config) as client:
            response = await client.respond(
                hello("accounts/fireworks/models/glm-5p3-flash"),
                RequestOptions(CacheKey="session-1"),
            )
    assert (response.ID, response.Stop) == ("resp-1", StopReason.COMPLETE)
    usage = response.Usage
    assert (
        usage.InputTokens,
        usage.CachedInputTokens,
        usage.OutputTokens,
        usage.ReasoningTokens,
    ) == (
        11,
        3,
        7,
        5,
    )


# --- ollama ---------------------------------------------------------------------


async def test_ollama_client_uses_responses_without_authentication() -> None:
    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        assert (request.method, request.path) == ("POST", "/v1/responses")
        assert request.headers["Authorization"] is None
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Accept"] == "text/event-stream"
        body = request.json()
        assert body["model"] == "local-model:tag"
        assert body["stream"] is True
        assert body["store"] is False
        assert "prompt_cache_key" not in body
        write_sse(writer, COMPLETED)

    with serve(handler) as server:
        async with ollama.Client(ollama.Config(base_url=server.url + "/v1/")) as client:
            response = await client.respond(
                Request(Model=Model(ID="local-model:tag")), RequestOptions(CacheKey="session")
            )
    assert (response.ID, response.Stop) == ("r", StopReason.COMPLETE)


def test_ollama_default_base_url() -> None:
    client = ollama.Client(ollama.Config())
    assert client.adapter.endpoint == "http://localhost:11434/v1/responses"


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_TEST_MODEL"),
    reason="set OLLAMA_TEST_MODEL to an installed tool-capable model",
)
@pytest.mark.timeout(180)
async def test_ollama_live_tool_round_trip() -> None:
    from unreal_agent.llm import ReasoningEffort

    model = os.environ["OLLAMA_TEST_MODEL"]
    secret = "orchid-7319"
    config = ollama.Config(base_url=os.environ.get("OLLAMA_TEST_BASE_URL", ""), max_attempts=1)
    async with ollama.Client(config) as client:
        request = Request(
            Model=Model(ID=model, ReasoningEffort=ReasoningEffort.MEDIUM),
            Input=[
                Item(
                    Type=ItemType.MESSAGE,
                    Data=Message(
                        Role=Role.SYSTEM,
                        Text="Call read_secret exactly once to answer the user. Once you receive "
                        "its result, reply with the secret and no other text.",
                    ),
                ),
                Item(
                    Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="What is the secret?")
                ),
            ],
            Tools=[
                Tool(
                    Type=ToolType.FUNCTION,
                    Name="read_secret",
                    Description="Read the secret.",
                    Parameters={"type": "object", "properties": {}},
                )
            ],
        )
        response = await client.respond(request, RequestOptions())
        assert response.Failure is None
        request.Input.extend(response.Output)
        calls = 0
        for item in response.Output:
            if isinstance(item.Data, ToolCall):
                assert item.Data.Name == "read_secret"
                assert item.Data.CallID != ""
                calls += 1
                request.Input.append(
                    Item(
                        Type=ItemType.TOOL_RESULT,
                        Data=ToolResult(
                            CallID=item.Data.CallID,
                            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=secret)],
                        ),
                    )
                )
        assert calls == 1
        response = await client.respond(request, RequestOptions())
        assert response.Failure is None
        assert response.Stop == StopReason.COMPLETE
        assert any(
            isinstance(item.Data, Message) and secret in item.Data.Text for item in response.Output
        ), "model did not return the tool result"


# --- openaicodex ----------------------------------------------------------------


async def test_codex_client_uses_subscription_protocol() -> None:
    requests = 0

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        nonlocal requests
        requests += 1
        assert (request.method, request.path) == ("POST", "/responses")
        for key, want in {
            "Authorization": "Bearer access-token",
            "ChatGPT-Account-ID": "account",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "originator": "unreal-agent",
        }.items():
            assert request.headers[key] == want, key
        body = request.json()
        assert (body["model"], body["stream"], body["store"]) == ("gpt-test", True, False)
        assert len(body["input"]) == 2
        assert body["input"][0]["role"] == "system"
        assert body["input"][0]["content"] == "Our own instructions"
        assert body["prompt_cache_key"] == request.headers["session-id"]
        assert request.headers["session_id"] is None
        write_sse(writer, COMPLETED)

    with serve(handler) as server:
        config = openaicodex.Config(
            access_token="access-token", account_id="account", base_url=server.url
        )
        async with openaicodex.Client(config) as client:
            request = Request(
                Model=Model(ID="gpt-test"),
                Input=[
                    Item(
                        Type=ItemType.MESSAGE,
                        Data=Message(Role=Role.SYSTEM, Text="Our own instructions"),
                    ),
                    Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello")),
                ],
            )
            response = await client.respond(request, RequestOptions(CacheKey="session"))
    assert (response.ID, response.Stop, requests) == ("r", StopReason.COMPLETE, 1)


async def test_codex_client_loads_credentials_once(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    write_test_auth(path, "token-1", "account-1")
    seen: list[tuple[str, str]] = []

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        seen.append((request.headers["Authorization"], request.headers["ChatGPT-Account-ID"]))
        assert request.headers["session-id"]
        assert request.headers["session_id"] is None
        write_sse(writer, COMPLETED)

    async def check(client: openaicodex.Client, token: str, account: str) -> None:
        await client.respond(Request(), RequestOptions(CacheKey="session"))
        assert seen.pop() == ("Bearer " + token, account), "unexpected credential snapshot"

    with serve(handler) as server:
        async with openaicodex.Client(
            openaicodex.Config(auth_file=str(path), base_url=server.url)
        ) as client:
            path.unlink()
            await check(client, "token-1", "account-1")
            path.write_text("invalid JSON")
            path.chmod(0o600)
            await check(client, "token-1", "account-1")
            write_test_auth(path, "token-2", "account-2")
            await check(client, "token-1", "account-1")
            async with openaicodex.Client(
                openaicodex.Config(auth_file=str(path), base_url=server.url)
            ) as updated:
                await check(updated, "token-2", "account-2")
                await check(client, "token-1", "account-1")


async def test_codex_client_rejects_redirects() -> None:
    redirected = 0

    def target(_: ServerRequest, writer: ResponseWriter) -> None:
        nonlocal redirected
        redirected += 1
        writer.write_header(200)

    with serve(target) as target_server:

        def handler(_: ServerRequest, writer: ResponseWriter) -> None:
            writer.set_header("Location", target_server.url)
            writer.set_header("Content-Length", "0")
            writer.write_header(307)

        with serve(handler) as server:
            config = openaicodex.Config(
                access_token="opaque", account_id="account", base_url=server.url
            )
            async with openaicodex.Client(config) as client:
                with pytest.raises(APIError) as caught:
                    await client.respond(Request(), RequestOptions())
    assert caught.value.status_code == 307
    assert redirected == 0


async def test_codex_client_unauthorized_does_not_fallback() -> None:
    requests = 0

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        nonlocal requests
        requests += 1
        body = b'{"error":{"message":"unauthorized","code":"invalid_token"}}'
        writer.set_header("Content-Length", str(len(body)))
        writer.write_header(401)
        writer.write(body)

    with serve(handler) as server:
        config = openaicodex.Config(
            access_token="opaque", account_id="account", base_url=server.url
        )
        async with openaicodex.Client(config) as client:
            with pytest.raises(APIError) as caught:
                await client.respond(Request(), RequestOptions())
            assert caught.value.status_code == 401
            assert "renew them externally" in str(caught.value)
            assert requests == 1

            async def cancelled() -> None:
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                await client.respond(Request(), RequestOptions())

            with pytest.raises(asyncio.CancelledError):
                await asyncio.create_task(cancelled())
            assert requests == 1, "canceled request sent"


def test_codex_client_rejects_max_output_tokens_before_sending() -> None:
    async def run() -> None:
        config = openaicodex.Config(access_token="opaque", account_id="account")
        async with openaicodex.Client(config) as client:
            with pytest.raises(ValueError, match="codex does not support max_output_tokens"):
                await client.respond(Request(Model=Model(MaxOutputTokens=1)), RequestOptions())

    asyncio.run(run())


def test_codex_client_configuration_validation() -> None:
    base = openaicodex.BASE_URL
    for url in ("", base, base + "/", "http://127.0.0.1:1234/codex", "http://[::1]:1234"):
        openaicodex.validate_base_url(url)
    for url in (
        "https://api.openai.com/v1",
        "http://chatgpt.com/backend-api/codex",
        "https://chatgpt.com.evil.example/backend-api/codex",
        base + "?token=bad",
        base + "#fragment",
        "http://localhost:1234",
        "http://user:pass@127.0.0.1:1234",
        "file:///tmp/auth",
    ):
        with pytest.raises(ValueError, match="codex base URL must be"):
            openaicodex.validate_base_url(url)
    for config in (
        openaicodex.Config(),
        openaicodex.Config(access_token="opaque", account_id="account", max_attempts=0),
        openaicodex.Config(
            access_token="opaque", account_id="account", base_url="https://evil.example"
        ),
    ):
        with pytest.raises(ValueError, match="must be"):
            openaicodex.Client(config)


# --- providers_stream_test.go ---------------------------------------------------

TOOL_CALL = (
    b'{"type":"function_call","id":"fc-1","call_id":"call-1","name":"Bash",'
    b'"arguments":"{\\"command\\":\\"pwd\\"}","status":"completed"}'
)
PROVIDER_RESPONSE = (
    b'{"id":"r","status":"completed","output":['
    + TOOL_CALL
    + b',{"type":"message","id":"m-1","role":"assistant","phase":"final_answer",'
    b'"status":"completed","content":[{"type":"output_text","text":"final text"}]}],'
    b'"usage":{"input_tokens":12,"output_tokens":8,"input_tokens_details":{"cached_tokens":5},'
    b'"output_tokens_details":{"reasoning_tokens":4}}}'
)

type ProviderFactory = Callable[[str, Callable[[Exchange], None]], ResponsesClient]

PROVIDERS: dict[str, ProviderFactory] = {
    "openai": lambda url, trace: openai.Client(
        openai.Config(api_key="test-key", base_url=url, trace=trace)
    ),
    "openrouter": lambda url, trace: openrouter.Client(
        openrouter.Config(api_key="test-key", base_url=url, trace=trace)
    ),
    "fireworks": lambda url, trace: fireworks.Client(
        fireworks.Config(api_key="test-key", base_url=url, trace=trace)
    ),
    "openai-codex": lambda url, trace: openaicodex.Client(
        openaicodex.Config(
            access_token="subscription-token", account_id="account", base_url=url, trace=trace
        )
    ),
}


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_providers_only_return_terminal_response(name: str) -> None:
    prefix_sent = threading.Event()
    release = threading.Event()

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        assert request.headers["Accept"] == "text/event-stream"
        assert request.json()["stream"] is True
        writer.set_header("Content-Type", "text/event-stream")
        writer.write(b'data: {"type":"response.created","response":{"id":"r"}}\n\n')
        writer.write(
            b'data: {"type":"response.output_text.delta",'
            b'"delta":"partial text must not escape"}\n\n'
        )
        writer.write(
            b'data: {"type":"response.output_item.done","output_index":0,"item":'
            + TOOL_CALL
            + b"}\n\n"
        )
        prefix_sent.set()
        if not release.wait(5):
            return
        writer.write(
            b'data: {"type":"response.completed","response":' + PROVIDER_RESPONSE + b"}\n\n"
        )

    traces: list[Exchange] = []
    with serve(handler) as server:
        async with PROVIDERS[name](server.url, traces.append) as client:
            task = asyncio.create_task(
                client.respond(Request(Model=Model(ID="test")), RequestOptions())
            )
            assert await asyncio.to_thread(prefix_sent.wait, 5), "stream did not start"
            done, _ = await asyncio.wait([task], timeout=0.05)
            assert not done, "Respond returned before the terminal event"
            assert traces == [], "trace emitted a partial exchange"
            release.set()
            response = await asyncio.wait_for(task, 5)
    assert response.Stop == StopReason.COMPLETE
    assert len(response.Output) == 2
    assert len(traces) == 1
    assert traces[0].status_code == 200
    assert traces[0].response_body == PROVIDER_RESPONSE
    call = response.Output[0].Data
    message = response.Output[1].Data
    assert isinstance(call, ToolCall)
    assert isinstance(message, Message)
    assert call.CallID == "call-1"
    assert (message.Text, message.Phase) == ("final text", "final_answer")
    usage = response.Usage
    assert (
        usage.InputTokens,
        usage.OutputTokens,
        usage.CachedInputTokens,
        usage.ReasoningTokens,
    ) == (
        12,
        8,
        5,
        4,
    )
    assert "partial" not in message.Text
