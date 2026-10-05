import asyncio
import errno
import os
import stat
import sys
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from unreal_agent._go import go_quote
from unreal_agent.primitives._errors import (
    ContextCanceledError,
    PrimitiveError,
    is_only_context_cancellation,
    join,
    wrap,
)
from unreal_agent.primitives._fileio import File, invalid_path_error
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
    send_terminal,
    start_primitive,
)

MAX_CREATE_CONTENTION_RETRIES: Final = 16
_SUPPORTED_MODE_BITS: Final = 0o7777


class IOCreateKind(IntEnum):
    REGULAR_FILE = 1
    DIRECTORY = 2


@dataclass(slots=True)
class IOCreateRequest:
    """`mode` is used only when creating a new entry and is subject to umask."""

    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    kind: int = 0
    path: str = ""
    mode: int = 0


@dataclass(slots=True, frozen=True)
class IOCreateResult:
    kind: int = 0


def create(request: IOCreateRequest, events: EventSink) -> asyncio.Task[None]:
    return start_primitive(create_path(request, events), name="io.create")


async def create_path(request: IOCreateRequest, events: EventSink) -> None:
    if cancel_requested():
        await send_terminal(events, primitive_canceled(request.source, request.correlation_id))
        return
    canceled = threading.Event()
    try:
        result = await run_blocking(create_new_path, request, canceled, on_cancel=canceled.set)
    except Exception as error:
        if is_only_context_cancellation(error):
            event = primitive_canceled(request.source, request.correlation_id)
        else:
            event = primitive_failure(request.source, request.correlation_id, error)
        await send_terminal(events, event)
        return
    await send_terminal(
        events,
        PrimitiveEvent(
            PrimitiveEventType.IO_CREATE_COMPLETED,
            request.source,
            request.correlation_id,
            result,
        ),
    )


def create_new_path(request: IOCreateRequest, canceled: threading.Event) -> IOCreateResult:
    """Create with openat/mkdirat semantics, or accept an existing entry of the same kind."""
    path = go_quote(request.path)
    if request.kind not in (IOCreateKind.REGULAR_FILE, IOCreateKind.DIRECTORY):
        raise PrimitiveError(f"create {path}: unsupported kind {int(request.kind)}")
    try:
        mode = unix_file_mode(request.mode)
    except PrimitiveError as error:
        raise wrap(f"create {path}", error) from None

    target = request.path
    if request.kind == IOCreateKind.DIRECTORY:
        target = trim_trailing_path_separators(target)
    separator = target.rfind(os.sep)
    parent_path, name = target[: separator + 1], target[separator + 1 :]
    if not parent_path:
        parent_path = "."

    try:
        parent = open_create_directory(parent_path)
    except OSError as error:
        raise wrap(
            f"open parent directory {go_quote(parent_path)}: open {parent_path}", error
        ) from None
    return create_or_use_path(request, name, parent_path, parent, mode, canceled)


def create_or_use_path(
    request: IOCreateRequest,
    name: str,
    parent_path: str,
    parent: File,
    mode: int,
    canceled: threading.Event,
) -> IOCreateResult:
    path = go_quote(request.path)
    retries = 0
    while True:
        if canceled.is_set():
            raise _raised(
                join(ContextCanceledError(), close_file("parent directory", parent_path, parent))
            )
        if request.kind == IOCreateKind.REGULAR_FILE:
            try:
                file = open_new_file_at(parent, name, mode)
            except FileExistsError:
                pass
            except Exception as error:
                raise _raised(
                    join(
                        wrap(f"create {path}", error),
                        close_file("parent directory", parent_path, parent),
                    )
                ) from None
            else:
                return persist_created_file(request, parent_path, file, parent)
        else:
            try:
                make_new_directory_at(parent, name, mode)
            except FileExistsError:
                pass
            except Exception as error:
                raise _raised(
                    join(
                        wrap(f"create {path}", error),
                        close_file("parent directory", parent_path, parent),
                    )
                ) from None
            else:
                return persist_created_entry(request, parent_path, parent)

        result = use_existing_path(request, name, parent_path, parent)
        if result is not None:
            return result
        if retries == MAX_CREATE_CONTENTION_RETRIES:
            raise _raised(
                join(
                    PrimitiveError(
                        f"create {path}: path remained unstable after "
                        f"{MAX_CREATE_CONTENTION_RETRIES} retries"
                    ),
                    close_file("parent directory", parent_path, parent),
                )
            )
        retries += 1


def use_existing_path(
    request: IOCreateRequest, name: str, parent_path: str, parent: File
) -> IOCreateResult | None:
    """Accept an existing entry of the requested kind. None asks the caller to retry."""
    path = go_quote(request.path)
    try:
        info = os.stat(name, dir_fd=parent.fileno(), follow_symlinks=False)
    except FileNotFoundError:
        return None
    except Exception as error:
        raise _raised(
            join(
                wrap(f"inspect existing {path}", error),
                close_file("parent directory", parent_path, parent),
            )
        ) from None
    kind_error = validate_existing_path(request, info.st_mode)
    if kind_error is not None:
        raise _raised(
            join(
                wrap(f"use existing {path}", kind_error),
                close_file("parent directory", parent_path, parent),
            )
        )
    return persist_created_entry(request, parent_path, parent)


def validate_existing_path(request: IOCreateRequest, mode: int) -> PrimitiveError | None:
    if request.kind == IOCreateKind.REGULAR_FILE and not stat.S_ISREG(mode):
        return PrimitiveError("existing path is not a regular file")
    if request.kind == IOCreateKind.DIRECTORY and not stat.S_ISDIR(mode):
        return PrimitiveError("existing path is not a directory")
    return None


def trim_trailing_path_separators(path: str) -> str:
    trimmed = path.rstrip(os.sep)
    return trimmed if trimmed else path


def persist_created_file(
    request: IOCreateRequest, parent_path: str, file: File, parent: File
) -> IOCreateResult:
    try:
        sync_created_file(file, parent)
    except PrimitiveError as error:
        raise _raised(
            join(
                wrap(f"persist {go_quote(request.path)}", error),
                close_file("file", request.path, file),
                close_file("parent directory", parent_path, parent),
            )
        ) from None
    # The creation is already durable; descriptor cleanup cannot undo the commit.
    close_file("file", request.path, file)
    close_file("parent directory", parent_path, parent)
    return IOCreateResult(request.kind)


def persist_created_entry(
    request: IOCreateRequest, parent_path: str, parent: File
) -> IOCreateResult:
    try:
        sync_created_directory(parent)
    except PrimitiveError as error:
        raise _raised(
            join(
                wrap(f"persist {go_quote(request.path)}", error),
                close_file("parent directory", parent_path, parent),
            )
        ) from None
    # The creation is already durable; descriptor cleanup cannot undo the commit.
    close_file("parent directory", parent_path, parent)
    return IOCreateResult(request.kind)


def open_create_directory(path: str) -> File:
    return File.open(path, os.O_RDONLY | os.O_DIRECTORY)


def open_new_file_at(parent: File, name: str, mode: int) -> File:
    return File.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=parent.fileno())


def make_new_directory_at(parent: File, name: str, mode: int) -> None:
    try:
        os.mkdir(name, mode, dir_fd=parent.fileno())
    except ValueError:
        raise invalid_path_error(name) from None


def unix_file_mode(mode: int) -> int:
    unsupported = mode & ~_SUPPORTED_MODE_BITS
    if unsupported:
        raise PrimitiveError(f"unsupported mode bits {unsupported:#o}")
    return mode


def close_file(kind: str, path: str, file: File) -> BaseException | None:
    try:
        file.close()
    except Exception as error:
        return wrap(f"close {kind} {go_quote(path)}", error)
    return None


def sync_created_file(file: File, parent: File) -> None:
    try:
        file.sync()
    except Exception as error:
        raise wrap("sync file", error) from None
    sync_created_directory(parent)


def sync_created_directory(parent: File) -> None:
    try:
        parent.sync()
    except Exception as error:
        raise wrap("sync parent directory", error) from None
    if sys.platform == "darwin":
        import fcntl

        try:
            fcntl.fcntl(parent.fileno(), fcntl.F_FULLFSYNC)
        except OSError as error:
            failure = full_sync_result(error)
            if failure is not None:
                raise failure from None
        except Exception as error:
            raise wrap("flush storage device", error) from None


def full_sync_result(error: BaseException | None) -> PrimitiveError | None:
    if error is None or (isinstance(error, OSError) and error.errno == errno.ENOTSUP):
        return None
    return wrap("flush storage device", error)


def _raised(error: BaseException | None) -> BaseException:
    assert error is not None
    return error
