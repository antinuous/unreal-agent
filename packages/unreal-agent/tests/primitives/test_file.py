import asyncio
import fcntl
import os
import stat
from pathlib import Path

import pytest
from unreal_agent._go import go_quote
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
from unreal_agent.primitives.file import file_type

from tests.primitives._helpers import collect_events, write_test_file


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


@pytest.mark.parametrize("path", ["input\x00name", "input\ud800name"], ids=["nul", "surrogate"])
async def test_read_file_reports_invalid_path(path: str) -> None:
    async with asyncio.timeout(1):
        events = await read_events(IOReadRequest(path=path, count=1))
    assert [event.type for event in events] == [PrimitiveEventType.FAILED]
    assert events[0].result.error == f"open {go_quote(path)}: open {path}: invalid argument"


async def test_read_file_open_failure_includes_path_error(tmp_path: Path) -> None:
    path = str(tmp_path / "missing")
    request = IOReadRequest(
        source=SourceID("operation-1"), correlation_id=CorrelationID("read-1"), path=path, count=1
    )
    events = await read_events(request)
    assert len(events) == 1
    assert events[0].type == PrimitiveEventType.FAILED
    assert (events[0].source, events[0].correlation_id) == (request.source, request.correlation_id)
    assert events[0].result.error == f'open "{path}": open {path}: no such file or directory'


@pytest.mark.parametrize(
    ("path", "want"),
    [pytest.param(None, "d---------", id="directory"), pytest.param("/dev/null", "Dc---------")],
)
async def test_read_file_reports_unsupported_file_type_like_go(
    tmp_path: Path, path: str | None, want: str
) -> None:
    path = path or str(tmp_path)
    events = await read_events(IOReadRequest(path=path, count=1))
    assert events[0].result.error == f'open "{path}": unsupported file type {want}'


@pytest.mark.parametrize(
    ("mode", "want"),
    [
        (stat.S_IFDIR, "d---------"),
        (stat.S_IFLNK, "L---------"),
        (stat.S_IFBLK, "D---------"),
        (stat.S_IFCHR, "Dc---------"),
        (stat.S_IFIFO, "p---------"),
        (stat.S_IFSOCK, "S---------"),
        (0, "?---------"),
    ],
)
def test_file_type_matches_go_file_mode(mode: int, want: str) -> None:
    assert file_type(mode | 0o644) == want
