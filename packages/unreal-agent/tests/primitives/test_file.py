import asyncio
import fcntl
import os
from pathlib import Path

import pytest
from unreal_agent.primitives import (
    IO_READ_CHUNK_SIZE,
    CorrelationID,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
    read_file,
)
from unreal_agent.primitives.channel import Channel

from tests.primitives._helpers import MAX_INT64, collect_events, write_test_file


async def read_events(request: IOReadRequest) -> list[PrimitiveEvent]:
    events: Channel[PrimitiveEvent] = Channel()
    read_file(request, events)
    return await collect_events(events)


async def test_read_file_streams_range_and_completes_with_full_size(tmp_path: Path) -> None:
    contents = bytes(index % 251 for index in range(IO_READ_CHUNK_SIZE * 3 + 17))
    path = write_test_file(tmp_path, contents)
    offset = IO_READ_CHUNK_SIZE // 2
    count = IO_READ_CHUNK_SIZE * 2 + 11
    request = IOReadRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("read-1"),
        path=path,
        offset=offset,
        count=count,
    )

    output = bytearray()
    next_offset = offset
    output_events = 0
    completed: IOReadCompletedResult | None = None
    for event in await read_events(request):
        assert (event.source, event.correlation_id) == (request.source, request.correlation_id)
        if event.type == PrimitiveEventType.IO_READ_OUTPUT:
            result = event.result
            assert isinstance(result, IOReadOutputResult)
            assert result.offset == next_offset
            assert len(result.data) <= IO_READ_CHUNK_SIZE
            output += result.data
            next_offset += len(result.data)
            output_events += 1
        elif event.type == PrimitiveEventType.IO_READ_COMPLETED:
            completed = event.result
        else:
            pytest.fail(f"unexpected event type {event.type!r}")

    assert output_events >= 2
    assert bytes(output) == contents[offset : offset + count]
    assert completed == IOReadCompletedResult(len(contents))


async def test_read_file_clamps_range_to_end_of_file(tmp_path: Path) -> None:
    path = write_test_file(tmp_path, b"0123456789")
    events = await read_events(IOReadRequest(path=path, offset=7, count=MAX_INT64))
    output = b"".join(
        event.result.data for event in events if event.type == PrimitiveEventType.IO_READ_OUTPUT
    )
    assert output == b"789"


async def test_read_file_with_zero_count_emits_only_completion(tmp_path: Path) -> None:
    contents = b"contents"
    path = write_test_file(tmp_path, contents)
    events = await read_events(IOReadRequest(path=path))
    assert len(events) == 1
    assert events[0].type == PrimitiveEventType.IO_READ_COMPLETED
    assert events[0].result == IOReadCompletedResult(len(contents))


async def test_read_file_canceled_before_start() -> None:
    events: Channel[PrimitiveEvent] = Channel()
    task = read_file(IOReadRequest(path="unused"), events)
    task.cancel()
    collected = await collect_events(events)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [event.type for event in collected] == [PrimitiveEventType.CANCELED]


async def test_read_file_cancellation_terminates_stream(tmp_path: Path) -> None:
    contents = b"a" * (IO_READ_CHUNK_SIZE * 3)
    path = write_test_file(tmp_path, contents)
    request = IOReadRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("read-1"),
        path=path,
        count=len(contents),
    )
    events: Channel[PrimitiveEvent] = Channel()
    task = read_file(request, events)

    output_events = 0
    canceled_events = 0
    while True:
        event = await events.get()
        if event.type == PrimitiveEventType.IO_READ_OUTPUT:
            output_events += 1
            task.cancel()
        elif event.type == PrimitiveEventType.CANCELED:
            canceled_events += 1
            break
        elif event.type == PrimitiveEventType.IO_READ_COMPLETED:
            pytest.fail("canceled read completed")
        else:
            pytest.fail(f"unexpected event type {event.type!r}")

    assert 0 < output_events < 3
    assert canceled_events == 1
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_read_file_failure_is_terminal(tmp_path: Path) -> None:
    request = IOReadRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("read-1"),
        path=str(tmp_path / "missing"),
        count=1,
    )
    events = await read_events(request)
    assert len(events) == 1
    assert events[0].type == PrimitiveEventType.FAILED
    result = events[0].result
    assert isinstance(result, PrimitiveFailureResult)
    assert "missing" in result.error


async def test_read_file_ignores_advisory_file_lock(tmp_path: Path) -> None:
    contents = b"locked contents"
    path = write_test_file(tmp_path, contents)
    locked = os.open(path, os.O_RDWR)
    fcntl.flock(locked, fcntl.LOCK_EX)
    try:
        async with asyncio.timeout(1):
            events = await read_events(IOReadRequest(path=path, count=len(contents)))
    finally:
        fcntl.flock(locked, fcntl.LOCK_UN)
        os.close(locked)
    assert [event.type for event in events] == [
        PrimitiveEventType.IO_READ_OUTPUT,
        PrimitiveEventType.IO_READ_COMPLETED,
    ]


async def test_read_file_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    path = str(tmp_path / "input.fifo")
    os.mkfifo(path, 0o600)
    events: Channel[PrimitiveEvent] = Channel()
    task = read_file(IOReadRequest(path=path, count=1), events)
    async with asyncio.timeout(1):
        event = await events.get()
    assert event.type == PrimitiveEventType.FAILED
    result = event.result
    assert isinstance(result, PrimitiveFailureResult)
    assert "unsupported file type" in result.error
    await task
    assert events.empty()


async def test_read_file_rejects_negative_range() -> None:
    events = await read_events(IOReadRequest(path="unused", offset=-1))
    assert [event.type for event in events] == [PrimitiveEventType.FAILED]
    assert "offset must not be negative" in events[0].result.error


async def test_read_file_rejects_negative_count() -> None:
    events = await read_events(IOReadRequest(path="unused", count=-1))
    assert [event.type for event in events] == [PrimitiveEventType.FAILED]
    assert "count must not be negative" in events[0].result.error
