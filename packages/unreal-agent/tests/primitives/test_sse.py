import asyncio
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from unreal_agent.primitives import (
    TERMINAL_EVENT_TYPES,
    PrimitiveEvent,
    PrimitiveEventType,
    RemoteClient,
    RemoteSSEOptions,
    SSEFrameDelimiter,
    default_remote_request,
    sse_data,
)
from unreal_agent.primitives.channel import Channel
from unreal_agent.primitives.remote_sse import RemoteOutputFrame, RemoteSSEFramer

SSE_DATA_CASES: list[tuple[str, str, bytes | None]] = [
    ("empty", "", None),
    ("separators only", "\n\n", None),
    ("comment", ": data: ignored", None),
    ("other fields", "event: message\nid: 1\nretry: 20\nunknown: value", None),
    ("bare unknown field", "unknown", None),
    ("single", "data: hello", b"hello"),
    ("no space", "data:hello", b"hello"),
    ("one space only", "data:  hello", b" hello"),
    ("leading tab", "data:\thello", b"\thello"),
    ("trailing whitespace", "data: hello \t", b"hello \t"),
    (
        "colons in value",
        "data: https://example.test:123/:foo",
        b"https://example.test:123/:foo",
    ),
    ("case sensitive", "Data: ignored\nDATA: ignored\ndata: kept", b"kept"),
    ("field whitespace", " data: ignored\ndata : ignored\ndata\t: ignored", None),
    ("empty data", "data:", b""),
    ("empty data with space", "data: ", b""),
    ("bare data", "data", b""),
    ("whitespace data", "data:  \t", b" \t"),
    ("multiple fields", "data: first\ndata: second", b"first\nsecond"),
    ("leading empty", "data:\ndata: next", b"\nnext"),
    ("bare leading empty", "data\ndata: next", b"\nnext"),
    ("middle empty", "data: first\ndata\ndata: last", b"first\n\nlast"),
    ("trailing empty", "data: first\ndata:", b"first\n"),
    ("all empty", "data:\ndata\ndata: ", b"\n\n"),
    (
        "interleaved fields",
        "data: first\n: comment\nevent: message\nid: 1\ndata: second",
        b"first\nsecond",
    ),
    (
        "no implicit continuation",
        "data: first\nnot a data line\ndata: second",
        b"first\nsecond",
    ),
    ("delimiter retained", "data: first\ndata: second\n\n", b"first\nsecond"),
    ("Unicode", "data: 会話\u2028café\u2029", "会話\u2028café\u2029".encode()),
    ("NUL in data", "data: a\x00b", b"a\x00b"),
    ("BOM in data", "data: ﻿text", "﻿text".encode()),
    ("leading BOM", "﻿data: hello", b"hello"),
    ("leading BOM and BOM in data", "﻿data: ﻿text", "﻿text".encode()),
    ("opaque bytes", "data: \udcff\udcfe", b"\xff\xfe"),
]


@pytest.mark.parametrize(
    ("frame", "want", "ending"),
    [
        pytest.param(frame, want, ending, id=f"{name}-{ending!r}")
        for name, frame, want in SSE_DATA_CASES
        for ending in (["\n", "\r\n", "\r"] if "\n" in frame else ["\n"])
    ],
)
def test_sse_data(frame: str, want: bytes | None, ending: str) -> None:
    encoded = frame.replace("\n", ending).encode("utf-8", "surrogateescape")
    assert sse_data(encoded) == want


def test_sse_data_mixed_line_endings() -> None:
    got = sse_data(b"event: message\r\ndata: one\ndata:\rdata: two\r\n")
    assert got == b"one\n\ntwo"


def test_sse_data_large_field() -> None:
    want = b"x" * (1 << 20)
    assert sse_data(b"data: " + want) == want


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=repr)
def test_sse_data_with_framer_chunk_boundaries(ending: str) -> None:
    wire = (
        "﻿data: first\n\n: heartbeat\n\n﻿data\n\ndata:\ndata: hello\n"
        ": comment\ndata:\n\ndata: [DONE]\n\ndata: unfinished"
    )
    encoded = wire.replace("\n", ending).encode()
    want = [b"first", None, b"", b"\nhello\n", b"[DONE]"]
    for split in range(len(encoded) + 1):
        framer = RemoteSSEFramer(len(encoded), SSEFrameDelimiter.STRIP)
        got: list[bytes | None] = []
        for chunk in (encoded[:split], encoded[split:]):
            frames, error = framer.append(chunk)
            assert error is None
            got.extend(sse_data(frame.data) for frame in frames)
        got.extend(sse_data(frame.data) for frame in framer.finish())
        assert got == want, f"split={split}"


async def test_sse_data_from_remote_primitive() -> None:
    wire = (
        b": comment\r\rdata:\rdata: first\r\ndata: last\n\ndata\n\ndata: [DONE]\n\ndata: unfinished"
    )

    async def body():
        for index in range(len(wire)):
            yield wire[index : index + 1]

    import httpx

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body())

    client = RemoteClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = default_remote_request("sse", "fields", "https://example.test")
    request.sse = RemoteSSEOptions(max_frame_size=1024, frame_delimiter=SSEFrameDelimiter.STRIP)
    events: Channel[PrimitiveEvent] = Channel()
    client.send_request(request, events)
    collected = await _collect(events)
    await client.aclose()
    got = [
        sse_data(event.result.data)
        for event in collected
        if event.type == PrimitiveEventType.REMOTE_OUTPUT
    ]
    assert got == [None, b"\nfirst\nlast", b"", b"[DONE]"]
    assert collected[-1].type == PrimitiveEventType.REMOTE_COMPLETED


async def _collect(events: Channel[PrimitiveEvent]) -> list[PrimitiveEvent]:
    collected: list[PrimitiveEvent] = []
    while True:
        event = await asyncio.wait_for(events.get(), 10)
        collected.append(event)
        if event.type in TERMINAL_EVENT_TYPES:
            return collected


@settings(max_examples=200, deadline=None)
@given(
    data=st.binary(max_size=4096), ending_index=st.integers(0, 255), chunk_size=st.integers(0, 255)
)
@example(data=b"", ending_index=0, chunk_size=1)
@example(data=b"hello", ending_index=0, chunk_size=1)
@example(data=b"\nleading", ending_index=0, chunk_size=1)
@example(data=b"trailing\n", ending_index=0, chunk_size=1)
@example(data=b"\n\n", ending_index=0, chunk_size=1)
@example(data=b": colons: preserved", ending_index=0, chunk_size=1)
@example(data=b" \t whitespace \t", ending_index=0, chunk_size=1)
@example(data="会話\u2028".encode(), ending_index=0, chunk_size=1)
@example(data=b"\xff\x00", ending_index=0, chunk_size=1)
@example(data=b"[DONE]", ending_index=0, chunk_size=1)
def test_sse_data_round_trip(data: bytes, ending_index: int, chunk_size: int) -> None:
    want = re.sub(rb"\r\n?", b"\n", data)
    ending = [b"\n", b"\r\n", b"\r"][ending_index % 3]
    wire = bytearray(b"event: message" + ending)
    for line in want.split(b"\n"):
        wire += b"data: " + line + ending
    wire += ending
    encoded = bytes(wire)
    assert sse_data(encoded) == want

    framer = RemoteSSEFramer(len(encoded), SSEFrameDelimiter.STRIP)
    frames: list[RemoteOutputFrame] = []
    step = chunk_size + 1
    for start in range(0, len(encoded), step):
        out, error = framer.append(encoded[start : start + step])
        assert error is None
        frames.extend(out)
    frames.extend(framer.finish())
    assert len(frames) == 1
    assert sse_data(frames[0].data) == want
