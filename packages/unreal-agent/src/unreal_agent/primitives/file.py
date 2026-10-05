import asyncio
import os
import stat
from dataclasses import dataclass
from typing import Final, Protocol

from unreal_agent._go import go_quote
from unreal_agent.primitives._errors import PrimitiveError, UnexpectedEOFError, join, wrap
from unreal_agent.primitives._fileio import File
from unreal_agent.primitives.primitive_event import (
    NO_CORRELATION,
    NO_SOURCE,
    CorrelationID,
    EventSink,
    PrimitiveEvent,
    PrimitiveEventType,
    SourceID,
    cancel_requested,
    primitive_canceled,
    primitive_failure,
    run_blocking,
    send_event,
    send_terminal,
    start_primitive,
)

IO_READ_CHUNK_SIZE: Final = 32 * 1024
_GO_FILE_TYPES: Final = {
    stat.S_IFDIR: "d",
    stat.S_IFLNK: "L",
    stat.S_IFBLK: "D",
    stat.S_IFCHR: "Dc",
    stat.S_IFIFO: "p",
    stat.S_IFSOCK: "S",
}


@dataclass(slots=True)
class IOReadRequest:
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    path: str = ""
    offset: int = 0
    count: int = 0


@dataclass(slots=True, frozen=True)
class IOReadOutputResult:
    offset: int
    data: bytes


@dataclass(slots=True, frozen=True)
class IOReadCompletedResult:
    size: int


class ReadableFile(Protocol):
    name: str

    def read_at(self, size: int, offset: int) -> bytes: ...

    def stat(self) -> os.stat_result: ...

    def close(self) -> None: ...


def read_file(request: IOReadRequest, events: EventSink) -> asyncio.Task[None]:
    """Read a range bounded by the file's initial metadata size.

    Virtual files whose reported size does not describe their content are unsupported.
    """
    return start_primitive(stream_file(request, events), name="io.read")


async def stream_file(request: IOReadRequest, events: EventSink) -> None:
    error = _validate_io_read_request(request)
    if error is not None:
        await send_terminal(events, _read_failure(request, error))
        return
    if cancel_requested():
        await send_terminal(events, _read_canceled(request))
        return
    try:
        file = await run_blocking(File.open, request.path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as error:
        await send_terminal(
            events,
            _read_failure(
                request, wrap(f"open {go_quote(request.path)}: open {request.path}", error)
            ),
        )
        return
    await inspect_and_stream_file(request, file, events)


async def inspect_and_stream_file(
    request: IOReadRequest, file: ReadableFile, events: EventSink
) -> None:
    try:
        info = await run_blocking(file.stat)
    except Exception as error:
        failure = join(wrap(f"inspect {go_quote(request.path)}", error), await _close(file))
        await send_terminal(events, _read_failure(request, failure))
        return
    if not stat.S_ISREG(info.st_mode):
        failure = join(
            PrimitiveError(
                f"open {go_quote(request.path)}: unsupported file type {file_type(info.st_mode)}"
            ),
            await _close(file),
        )
        await send_terminal(events, _read_failure(request, failure))
        return
    await stream_open_file(request, file, info.st_size, events)


async def stream_open_file(
    request: IOReadRequest, file: ReadableFile, size: int, events: EventSink
) -> None:
    remaining = min(request.count, size - request.offset) if request.offset < size else 0
    file_offset = request.offset
    path = go_quote(request.path)
    while remaining > 0:
        if cancel_requested():
            await _finish_canceled_read(events, request, file)
            return
        want = min(remaining, IO_READ_CHUNK_SIZE)
        read_error: BaseException | None = None
        data = b""
        try:
            data = await run_blocking(file.read_at, want, file_offset)
        except Exception as error:
            read_error = error
        if cancel_requested():
            await _finish_canceled_read(events, request, file)
            return
        if read_error is None and len(data) != want:
            read_error = UnexpectedEOFError()
        if read_error is not None:
            failure = join(wrap(f"read {path}", read_error), await _close(file))
            await send_terminal(events, _read_failure(request, failure))
            return
        event = PrimitiveEvent(
            PrimitiveEventType.IO_READ_OUTPUT,
            request.source,
            request.correlation_id,
            IOReadOutputResult(file_offset, data),
        )
        if not await send_event(events, event):
            await _finish_canceled_read(events, request, file)
            return
        file_offset += len(data)
        remaining -= len(data)

    if cancel_requested():
        await _finish_canceled_read(events, request, file)
        return
    try:
        info = await run_blocking(file.stat)
    except Exception as error:
        failure = join(wrap(f"inspect {path} after reading", error), await _close(file))
        await send_terminal(events, _read_failure(request, failure))
        return
    if info.st_size != size:
        failure = join(
            PrimitiveError(f"read {path}: file size changed from {size} to {info.st_size}"),
            await _close(file),
        )
        await send_terminal(events, _read_failure(request, failure))
        return
    close_error = await _close(file)
    if close_error is not None:
        await send_terminal(events, _read_failure(request, wrap(f"close {path}", close_error)))
        return
    if cancel_requested():
        await send_terminal(events, _read_canceled(request))
        return
    await send_terminal(
        events,
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            request.source,
            request.correlation_id,
            IOReadCompletedResult(size),
        ),
    )


async def _close(file: ReadableFile) -> BaseException | None:
    try:
        await run_blocking(file.close)
    except Exception as error:
        return error
    return None


async def _finish_canceled_read(
    events: EventSink, request: IOReadRequest, file: ReadableFile
) -> None:
    close_error = await _close(file)
    if close_error is not None:
        await send_terminal(
            events, _read_failure(request, wrap(f"close {go_quote(request.path)}", close_error))
        )
        return
    await send_terminal(events, _read_canceled(request))


def file_type(mode: int) -> str:
    return _GO_FILE_TYPES.get(stat.S_IFMT(mode), "?") + "---------"


def _validate_io_read_request(request: IOReadRequest) -> PrimitiveError | None:
    if request.offset < 0:
        return PrimitiveError(f"read {go_quote(request.path)}: offset must not be negative")
    if request.count < 0:
        return PrimitiveError(f"read {go_quote(request.path)}: count must not be negative")
    return None


def _read_failure(request: IOReadRequest, error: BaseException | None) -> PrimitiveEvent:
    assert error is not None
    return primitive_failure(request.source, request.correlation_id, error)


def _read_canceled(request: IOReadRequest) -> PrimitiveEvent:
    return primitive_canceled(request.source, request.correlation_id)
