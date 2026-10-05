import asyncio
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
    SourceID,
)
from unreal_agent.primitives._errors import (
    ContextCanceledError,
    DeadlineExceededError,
    FileClosedError,
    is_only_context_cancellation,
    join,
    wrap,
)
from unreal_agent.primitives._fileio import File
from unreal_agent.primitives.channel import Channel
from unreal_agent.primitives.file import ReadableFile, inspect_and_stream_file, stream_open_file

from tests.primitives._helpers import MAX_INT64, collect_events, failure_text, single_event

type OpenReadFile = Callable[[bytes], File]


@pytest.fixture
def open_read_file(tmp_path: Path) -> Iterator[OpenReadFile]:
    opened: list[File] = []

    def open_file(contents: bytes) -> File:
        descriptor, path = tempfile.mkstemp(dir=tmp_path, prefix="input")
        os.close(descriptor)
        Path(path).write_bytes(contents)
        file = File.open(path, os.O_RDWR)
        opened.append(file)
        return file

    yield open_file
    for file in opened:
        if not file.closed:
            file.close()


class WrappedFile:
    def __init__(self, file: File) -> None:
        self.file = file
        self.name = file.name

    def read_at(self, size: int, offset: int) -> bytes:
        return self.file.read_at(size, offset)

    def stat(self) -> os.stat_result:
        return self.file.stat()

    def close(self) -> None:
        self.file.close()


class ReadFailureFile(WrappedFile):
    def __init__(self, file: File, error: Exception) -> None:
        super().__init__(file)
        self.error = error

    def read_at(self, size: int, offset: int) -> bytes:
        raise self.error


class StatFailureFile(WrappedFile):
    def __init__(self, file: File, stat_error: Exception, close_error: Exception) -> None:
        super().__init__(file)
        self.stat_error = stat_error
        self.close_error = close_error

    def stat(self) -> os.stat_result:
        raise self.stat_error

    def close(self) -> None:
        self.file.close()
        raise self.close_error


class CloseFailureFile(WrappedFile):
    def __init__(self, file: File, error: Exception | None) -> None:
        super().__init__(file)
        self.error = error

    def close(self) -> None:
        self.file.close()
        if self.error is not None:
            raise self.error


class BlockingCloseFile(WrappedFile):
    def __init__(self, file: File, close_error: Exception | None = None) -> None:
        super().__init__(file)
        self.started = threading.Event()
        self.release = threading.Event()
        self.close_error = close_error

    def close(self) -> None:
        self.started.set()
        self.release.wait()
        self.file.close()
        if self.close_error is not None:
            raise self.close_error


class CancelingFile(WrappedFile):
    def __init__(self, file: File) -> None:
        super().__init__(file)
        self.cancel: Callable[[], None] = lambda: None

    def read_at(self, size: int, offset: int) -> bytes:
        data = self.file.read_at(size, offset)
        self.cancel()
        return data


@dataclass
class FileReadRange:
    offset: int
    count: int


class ObservedReadFile(WrappedFile):
    def __init__(self, file: File) -> None:
        super().__init__(file)
        self.reads: list[FileReadRange] = []
        self.stats = 0
        self.closed = False

    def read_at(self, size: int, offset: int) -> bytes:
        self.reads.append(FileReadRange(offset, size))
        return self.file.read_at(size, offset)

    def stat(self) -> os.stat_result:
        self.stats += 1
        return self.file.stat()

    def close(self) -> None:
        self.closed = True
        self.file.close()


class PausingReadFile(WrappedFile):
    def __init__(self, file: File) -> None:
        super().__init__(file)
        self.read = threading.Event()
        self.release = threading.Event()

    def read_at(self, size: int, offset: int) -> bytes:
        data = self.file.read_at(size, offset)
        self.read.set()
        self.release.wait()
        return data


class InterruptedReadFile(CloseFailureFile):
    def __init__(self, file: File, close_error: Exception | None, read_error: Exception | None):
        super().__init__(file, close_error)
        self.ready = threading.Event()
        self.release = threading.Event()
        self.read_error = read_error

    def read_at(self, size: int, offset: int) -> bytes:
        if offset >= IO_READ_CHUNK_SIZE:
            self.ready.set()
            self.release.wait()
            if self.read_error is not None:
                raise self.read_error
        return self.file.read_at(size, offset)


def start_file_stream(
    request: IOReadRequest, file: ReadableFile
) -> tuple[asyncio.Task[None], Channel[PrimitiveEvent]]:
    events: Channel[PrimitiveEvent] = Channel()
    task = asyncio.create_task(inspect_and_stream_file(request, file, events))
    return task, events


def start_canceled_file_stream(
    request: IOReadRequest, file: ReadableFile
) -> tuple[asyncio.Task[None], Channel[PrimitiveEvent]]:
    events: Channel[PrimitiveEvent] = Channel()

    async def body() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await inspect_and_stream_file(request, file, events)

    return asyncio.create_task(body()), events


async def collect_stream(
    task: asyncio.Task[None], events: Channel[PrimitiveEvent]
) -> list[PrimitiveEvent]:
    """Collect through the terminal event, then require the stream to end with nothing more."""
    collected = await collect_events(events)
    await task
    assert events.empty()
    return collected


def assert_closed(file: File) -> None:
    with pytest.raises(FileClosedError):
        file.stat()


async def test_inspect_and_stream_file_reports_stat_failure(open_read_file: OpenReadFile) -> None:
    file = StatFailureFile(open_read_file(b""), Exception("stat failed"), Exception("close failed"))
    events: Channel[PrimitiveEvent] = Channel(1)
    await inspect_and_stream_file(IOReadRequest(path="input"), file, events)
    error = failure_text(single_event(await collect_events(events), PrimitiveEventType.FAILED))
    assert "stat failed" in error
    assert "close failed" in error


async def test_stream_open_file_reports_read_failure(open_read_file: OpenReadFile) -> None:
    file = ReadFailureFile(open_read_file(b"x"), Exception("read failed"))
    task, events = start_file_stream(IOReadRequest(path="input", count=1), file)
    error = failure_text(
        single_event(await collect_stream(task, events), PrimitiveEventType.FAILED)
    )
    assert "read failed" in error


async def test_stream_open_file_reports_completion_close_failure(
    open_read_file: OpenReadFile,
) -> None:
    file = CloseFailureFile(open_read_file(b""), Exception("close failed"))
    task, events = start_file_stream(IOReadRequest(path="input"), file)
    error = failure_text(
        single_event(await collect_stream(task, events), PrimitiveEventType.FAILED)
    )
    assert "close failed" in error


@pytest.mark.parametrize(
    ("close_error", "event_type"),
    [
        pytest.param(None, PrimitiveEventType.CANCELED, id="closed"),
        pytest.param(Exception("close failed"), PrimitiveEventType.FAILED, id="close failure"),
    ],
)
async def test_stream_open_file_cancellation_while_sending_output(
    open_read_file: OpenReadFile,
    close_error: Exception | None,
    event_type: PrimitiveEventType,
) -> None:
    file = BlockingCloseFile(open_read_file(b"contents"), close_error)
    task, events = start_file_stream(IOReadRequest(path=file.name, count=8), file)

    # Keep output blocked in the unbuffered send until cancellation reaches close.
    while events.qsize() == 0:  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.to_thread(file.started.wait)
    file.release.set()

    event = single_event(await collect_stream(task, events), event_type)
    if close_error is not None:
        assert "close failed" in failure_text(event)
    assert_closed(file.file)


async def test_stream_open_file_cancellation_after_close(open_read_file: OpenReadFile) -> None:
    file = BlockingCloseFile(open_read_file(b""))
    task, events = start_file_stream(IOReadRequest(path="input"), file)
    await asyncio.to_thread(file.started.wait)
    task.cancel()
    file.release.set()
    single_event(await collect_stream(task, events), PrimitiveEventType.CANCELED)


async def test_stream_open_file_cancellation_during_read(open_read_file: OpenReadFile) -> None:
    file = CancelingFile(open_read_file(b"content"))
    task, events = start_file_stream(IOReadRequest(path="input", count=7), file)
    loop = asyncio.get_running_loop()

    def cancel() -> None:
        loop.call_soon_threadsafe(task.cancel)

    file.cancel = cancel
    single_event(await collect_stream(task, events), PrimitiveEventType.CANCELED)


@pytest.mark.parametrize("count", [0, 1], ids=["count=0", "count=1"])
async def test_stream_open_file_cancellation_before_read(
    open_read_file: OpenReadFile, count: int
) -> None:
    file = ObservedReadFile(open_read_file(b"x"))
    task, events = start_canceled_file_stream(IOReadRequest(path="input", count=count), file)
    single_event(await collect_stream(task, events), PrimitiveEventType.CANCELED)
    assert file.reads == []
    assert file.closed


@pytest.mark.parametrize(
    ("error", "want"),
    [
        pytest.param(ContextCanceledError(), True, id="canceled"),
        pytest.param(DeadlineExceededError(), True, id="deadline"),
        pytest.param(wrap("read", ContextCanceledError()), True, id="wrapped"),
        pytest.param(
            join(ContextCanceledError(), DeadlineExceededError()), True, id="joined cancellations"
        ),
        pytest.param(join(ContextCanceledError(), Exception("cleanup failed")), False, id="mixed"),
        pytest.param(Exception("cleanup failed"), False, id="other"),
    ],
)
def test_is_only_context_cancellation(error: BaseException, want: bool) -> None:
    assert is_only_context_cancellation(error) is want


_SIZE = IO_READ_CHUNK_SIZE * 3 + 17


@pytest.mark.parametrize(
    ("size", "offset", "count", "want"),
    [
        pytest.param(_SIZE, 0, 7, 7, id="prefix"),
        pytest.param(
            _SIZE,
            13,
            IO_READ_CHUNK_SIZE * 2 + 3,
            IO_READ_CHUNK_SIZE * 2 + 3,
            id="middle across chunks",
        ),
        pytest.param(_SIZE, _SIZE - 5, 5, 5, id="tail"),
        pytest.param(_SIZE, _SIZE - 7, MAX_INT64, 7, id="clamped overflowing range"),
        pytest.param(_SIZE, 11, 0, 0, id="zero count"),
        pytest.param(_SIZE, _SIZE, 5, 0, id="at EOF"),
        pytest.param(_SIZE, MAX_INT64, MAX_INT64, 0, id="beyond EOF"),
        pytest.param(0, 0, MAX_INT64, 0, id="empty file"),
    ],
)
async def test_read_file_accesses_only_requested_range(
    open_read_file: OpenReadFile, size: int, offset: int, count: int, want: int
) -> None:
    file = ObservedReadFile(open_read_file(b""))
    file.file.truncate(size)
    request = IOReadRequest(
        source=SourceID("source"),
        correlation_id=CorrelationID("read"),
        path=file.name,
        offset=offset,
        count=count,
    )
    task, channel = start_file_stream(request, file)
    events = await collect_stream(task, channel)

    next_offset = offset
    for read in file.reads:
        assert read.offset == next_offset
        assert 0 < read.count <= IO_READ_CHUNK_SIZE
        next_offset += read.count
    assert next_offset - offset == want
    assert file.stats == 2
    assert file.closed

    next_offset = offset
    for event in events:
        assert (event.source, event.correlation_id) == (request.source, request.correlation_id)
        if event.type == PrimitiveEventType.IO_READ_OUTPUT:
            output = event.result
            assert isinstance(output, IOReadOutputResult)
            assert output.offset == next_offset
            assert output.data
            next_offset += len(output.data)
        elif event.type == PrimitiveEventType.IO_READ_COMPLETED:
            assert event.result == IOReadCompletedResult(size)
        else:
            pytest.fail(f"unexpected event: {event!r}")
    assert next_offset - offset == want


async def test_stream_open_file_rejects_premature_eof(open_read_file: OpenReadFile) -> None:
    file = open_read_file(b"contents")
    size = file.stat().st_size
    file.truncate(3)
    events: Channel[PrimitiveEvent] = Channel(2)
    await stream_open_file(IOReadRequest(path=file.name, offset=2, count=4), file, size, events)
    error = failure_text(single_event(await collect_events(events), PrimitiveEventType.FAILED))
    assert "unexpected EOF" in error
    assert_closed(file)


@pytest.mark.parametrize("size", [20, 4], ids=["growth", "shrinkage"])
async def test_stream_open_file_detects_size_changes_outside_range(
    open_read_file: OpenReadFile, size: int
) -> None:
    file = PausingReadFile(open_read_file(b"contents"))
    task, events = start_file_stream(IOReadRequest(path=file.name, count=2), file)
    await asyncio.to_thread(file.read.wait)
    try:
        os.truncate(file.name, size)
    finally:
        file.release.set()
    output = await events.get()
    assert output.type == PrimitiveEventType.IO_READ_OUTPUT
    assert output.result.data == b"co"
    error = failure_text(
        single_event(await collect_stream(task, events), PrimitiveEventType.FAILED)
    )
    assert "file size changed" in error
    assert_closed(file.file)


async def test_stream_open_file_reports_final_stat_and_close_failures(
    open_read_file: OpenReadFile,
) -> None:
    file = StatFailureFile(open_read_file(b""), Exception("stat failed"), Exception("close failed"))
    events: Channel[PrimitiveEvent] = Channel(1)
    await stream_open_file(IOReadRequest(path=file.name), file, 0, events)
    error = failure_text(single_event(await collect_events(events), PrimitiveEventType.FAILED))
    for want in ("after reading", "stat failed", "close failed"):
        assert want in error


@pytest.mark.skip(
    reason="Go-only io.ReaderAt contract: ReadAt may return io.EOF with a full count; "
    "File.read_at returns bytes and has no EOF signal"
)
def test_stream_open_file_accepts_full_read_with_eof() -> None:
    pass


@pytest.mark.parametrize(
    ("read_error", "close_error", "want"),
    [
        pytest.param(None, None, "unexpected EOF", id="truncated second chunk"),
        pytest.param(Exception("read failed"), None, "read failed", id="second read fails"),
        pytest.param(
            Exception("read failed"),
            Exception("close failed"),
            "read failed",
            id="second read and close fail",
        ),
    ],
)
async def test_stream_open_file_failure_after_output(
    open_read_file: OpenReadFile,
    read_error: Exception | None,
    close_error: Exception | None,
    want: str,
) -> None:
    contents = b"x" * (IO_READ_CHUNK_SIZE + 17)
    file = InterruptedReadFile(open_read_file(contents), close_error, read_error)
    request = IOReadRequest(
        source=SourceID("source"),
        correlation_id=CorrelationID("read"),
        path=file.name,
        count=len(contents),
    )
    task, events = start_file_stream(request, file)
    first = await events.get()
    assert first.type == PrimitiveEventType.IO_READ_OUTPUT
    assert first.result.offset == 0
    assert first.result.data == contents[:IO_READ_CHUNK_SIZE]

    await asyncio.to_thread(file.ready.wait)
    try:
        if read_error is None:
            file.file.truncate(IO_READ_CHUNK_SIZE + 3)
    finally:
        file.release.set()
    failure = single_event(await collect_stream(task, events), PrimitiveEventType.FAILED)
    error = failure_text(failure)
    assert want in error
    if close_error is not None:
        assert "close failed" in error
    for event in (first, failure):
        assert (event.source, event.correlation_id) == (request.source, request.correlation_id)
    assert_closed(file.file)
