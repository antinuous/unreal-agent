import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import httpx
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Request,
    Role,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.llm.messagesapi import Config, MessagesAdapter, request_body
from unreal_agent.llm.messagesapi.stream import StreamingState
from unreal_agent.llm.responsesapi._wire import trim_space
from unreal_agent.primitives import RemoteClient, sse_data

TESTDATA = Path(__file__).parent / "testdata"

STREAM_MESSAGE_START = (
    '{"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant",'
    '"content":[],"usage":{"input_tokens":1,"output_tokens":0}}}'
)


def message(role: str, text: str) -> Item:
    return Item(Type=ItemType.MESSAGE, Data=Message(Role=cast(Role, role), Text=text))


def tool_call(call_id: str) -> Item:
    return Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID=call_id, Name="Bash", Arguments="{}"))


def tool_result(call_id: str, text: str, running: bool) -> Item:
    return Item(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID=call_id,
            Running=running,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=text)],
        ),
    )


def valid_request() -> Request:
    return Request(Model=Model(ID="claude-test"), Input=[message(Role.USER, "Hello")])


def live_stream(name: str) -> bytes:
    return (TESTDATA / f"live-{name}.sse").read_bytes()


def response_body(stop: str, content: str) -> bytes:
    return (
        '{"type":"message","id":"msg-1","role":"assistant","model":"claude-test",'
        f'"stop_reason":{json.dumps(stop)},"content":{content},'
        '"usage":{"input_tokens":10,"output_tokens":5}}'
    ).encode()


def encode_request(request: Request) -> tuple[bytes, dict[str, Any]]:
    body = request_body(request)
    return body, json.loads(body)


def observe_stream(state: StreamingState, stream: bytes) -> None:
    for frame in stream.split(b"\n\n"):
        data = trim_space(sse_data(frame) or b"")
        if data:
            state.observe(data)


def observe_events(state: StreamingState, *events: str) -> None:
    for event in events:
        state.observe(event.encode())


class Stream(httpx.AsyncByteStream):
    """A response body: chunks, then an optional error or a wait for cancellation."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: BaseException | None = None,
        hang: asyncio.Event | None = None,
        closed: asyncio.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.hang = hang
        self.closed = closed

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            for chunk in self.chunks:
                await asyncio.sleep(0)
                yield chunk
            if self.hang is not None:
                self.hang.set()
                await asyncio.Event().wait()
            if self.error is not None:
                raise self.error
        finally:
            if self.closed is not None:
                self.closed.set()


def reply(
    status: int = 200,
    body: bytes | list[bytes] = b"",
    *,
    content_type: str | None = "text/event-stream",
    headers: dict[str, str] | None = None,
    error: BaseException | None = None,
    hang: asyncio.Event | None = None,
    closed: asyncio.Event | None = None,
) -> httpx.Response:
    all_headers = dict(headers or {})
    if content_type is not None:
        all_headers["Content-Type"] = content_type
    chunks = body if isinstance(body, list) else ([body] if body else [])
    return httpx.Response(
        status,
        headers=all_headers,
        stream=Stream(chunks, error=error, hang=hang, closed=closed),
    )


type Handler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def new_adapter(handler: Handler, **config: Any) -> MessagesAdapter:
    remote = RemoteClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    config.setdefault("endpoint", "https://example.test/v1/messages")
    return MessagesAdapter(remote, Config(**config))
