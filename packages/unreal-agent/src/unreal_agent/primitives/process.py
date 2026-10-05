import asyncio
import errno
import os
import signal
import stat
import subprocess
from collections.abc import Buffer, Callable
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Final, Protocol

from unreal_agent.primitives._errors import (
    ContextCanceledError,
    FileClosedError,
    PrimitiveError,
    describe,
    go_quote,
    has_errno,
    is_error,
    join,
    wrap,
)
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
    run_in_thread,
    send_event,
    send_terminal,
    start_primitive,
    wait_ignoring_cancellation,
)
from unreal_agent.primitives.process_signal import normalize_process_group_signal_error

PROCESS_OUTPUT_CHUNK_SIZE: Final = 32 * 1024
PROCESS_TERMINATION_GRACE_PERIOD: Final = 5.0


class ProcessPipeSet(IntFlag):
    NONE = 0
    STDIN = 1
    STDOUT = 2
    STDERR = 4
    ALL = 7


class ProcessStream(IntEnum):
    STDOUT = 1
    STDERR = 2


class ProcessDoneError(PrimitiveError):
    def __init__(self, message: str = "process invocation is done") -> None:
        super().__init__(message)


class ProcessFinishedError(PrimitiveError):
    def __init__(self, message: str = "os: process already finished") -> None:
        super().__init__(message)


class ReadDeadlineError(PrimitiveError):
    def __init__(self, message: str = "i/o timeout") -> None:
        super().__init__(message)


@dataclass(slots=True)
class ProcessStartRequest:
    """`path` must be absolute.

    Output capture paths must be absolute, name existing regular files, and are
    truncated before start. Each is mutually exclusive with its pipe.
    `environment=None` inherits the parent environment; an empty list clears it.
    """

    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    path: str = ""
    arguments: list[str] = field(default_factory=list)
    directory: str = ""
    environment: list[str] | None = None
    pipes: int = ProcessPipeSet.NONE
    stdout_path: str = ""
    stderr_path: str = ""


@dataclass(slots=True, frozen=True)
class ProcessStartedResult:
    pid: int


@dataclass(slots=True, frozen=True)
class ProcessOutputResult:
    stream: ProcessStream
    offset: int
    data: bytes


@dataclass(slots=True, frozen=True)
class ProcessStreamFailureResult:
    stream: ProcessStream
    error: str


@dataclass(slots=True, frozen=True)
class ProcessExitResult:
    """`exit_code` is -1 and `signal` is nonzero when a signal ended the process."""

    exit_code: int = 0
    signal: int = 0


@dataclass(slots=True)
class ProcessWriteRequest:
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    data: bytes = b""


@dataclass(slots=True, frozen=True)
class ProcessWriteResult:
    count: int


@dataclass(slots=True, frozen=True)
class ProcessWriteFailureResult:
    count: int
    error: str


@dataclass(slots=True)
class ProcessCloseInputRequest:
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION


@dataclass(slots=True)
class ProcessSignalRequest:
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    signal: int = 0
    propagate_to_children: bool = False


class _Closer(Protocol):
    def close(self) -> None: ...


class PipeReader:
    """Nonblocking pipe reader on the event loop. `finish` acts as a past read deadline."""

    __slots__ = ("_deadline", "_waiter", "file")

    def __init__(self, file: File) -> None:
        os.set_blocking(file.fileno(), False)
        self.file = file
        self._deadline = False
        self._waiter: asyncio.Future[None] | None = None

    @property
    def name(self) -> str:
        return self.file.name

    def fileno(self) -> int:
        return self.file.fileno()

    async def read(self, size: int) -> bytes:
        """Return available bytes or b"" at end of file."""
        while True:
            if self._deadline:
                raise ReadDeadlineError()
            fd = self.file.fileno()
            try:
                return os.read(fd, size)
            except BlockingIOError:
                pass
            await self._wait_readable(fd)

    async def _wait_readable(self, fd: int) -> None:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiter = waiter
        loop.add_reader(fd, _resolve, waiter)
        try:
            await waiter
        finally:
            self._waiter = None
            if not self.file.closed:
                loop.remove_reader(fd)

    def finish(self) -> None:
        self._deadline = True
        if self._waiter is not None:
            _resolve(self._waiter)

    def close(self) -> None:
        fd = self.file.fileno()
        if self._waiter is not None:
            asyncio.get_running_loop().remove_reader(fd)
            _resolve(self._waiter)
        self.file.close()


class PipeWriteError(PrimitiveError):
    def __init__(self, count: int, error: BaseException) -> None:
        super().__init__(describe(error))
        self.count = count
        self.__cause__ = error


class PipeWriter:
    """Nonblocking pipe writer. A started write ignores cancellation until done or closed."""

    __slots__ = ("_registered", "_waiters", "file")

    def __init__(self, file: File) -> None:
        os.set_blocking(file.fileno(), False)
        self.file = file
        self._waiters: set[asyncio.Future[None]] = set()
        self._registered = False

    def fileno(self) -> int:
        return self.file.fileno()

    async def write(self, data: bytes) -> int:
        view = memoryview(data)
        written = 0
        while written < len(view):
            try:
                fd = self.file.fileno()
            except FileClosedError as error:
                raise PipeWriteError(written, error) from None
            try:
                written += os.write(fd, view[written:])
                continue
            except BlockingIOError:
                pass
            except OSError as error:
                raise PipeWriteError(written, error) from None
            await wait_ignoring_cancellation(self._writable(fd))
        return written

    def _writable(self, fd: int) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters.add(waiter)
        if not self._registered:
            loop.add_writer(fd, self._wake)
            self._registered = True
        return waiter

    def _wake(self) -> None:
        if self._registered:
            asyncio.get_running_loop().remove_writer(self.file.fileno())
            self._registered = False
        waiters, self._waiters = self._waiters, set()
        for waiter in waiters:
            _resolve(waiter)

    def close(self) -> None:
        self.file.fileno()
        self._wake()
        self.file.close()


def _resolve(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


@dataclass(slots=True)
class ProcessCommand:
    argv: list[str]
    directory: str | None
    environment: dict[bytes, str] | None
    stdin: File | None = None
    stdout: File | None = None
    stderr: File | None = None


@dataclass(slots=True)
class ProcessParentPipes:
    stdin: PipeWriter | None = None
    stdout: PipeReader | None = None
    stderr: PipeReader | None = None
    stdout_capture: File | None = None
    stderr_capture: File | None = None

    def close_all(self) -> BaseException | None:
        return join(
            close_process_file(self.stdin),
            close_process_file(self.stdout),
            close_process_file(self.stderr),
        )

    def sync_captures(self) -> BaseException | None:
        return join(
            sync_process_capture(self.stdout_capture, "stdout"),
            sync_process_capture(self.stderr_capture, "stderr"),
        )

    def close_captures(self) -> BaseException | None:
        return join(
            close_process_file(self.stdout_capture), close_process_file(self.stderr_capture)
        )

    def finish_output(self) -> None:
        """Wake reads held open by descendants; each reader drains the bytes already buffered."""
        for reader in (self.stdout, self.stderr):
            if reader is not None:
                reader.finish()

    def close_output(self) -> BaseException | None:
        return join(close_process_file(self.stdout), close_process_file(self.stderr))


@dataclass(slots=True)
class ProcessPipes:
    stdin_read: File | None = None
    stdin_write: PipeWriter | None = None
    stdout_read: PipeReader | None = None
    stdout_write: File | None = None
    stdout_capture: File | None = None
    stderr_read: PipeReader | None = None
    stderr_write: File | None = None
    stderr_capture: File | None = None

    def parent(self) -> ProcessParentPipes:
        return ProcessParentPipes(
            stdin=self.stdin_write,
            stdout=self.stdout_read,
            stderr=self.stderr_read,
            stdout_capture=self.stdout_capture,
            stderr_capture=self.stderr_capture,
        )

    def close_child_ends(self) -> BaseException | None:
        return join(
            close_process_file(self.stdin_read),
            close_process_file(self.stdout_write),
            close_process_file(self.stderr_write),
        )

    def close_all(self) -> BaseException | None:
        return join(
            close_process_file(self.stdin_read),
            close_process_file(self.stdin_write),
            close_process_file(self.stdout_read),
            close_process_file(self.stdout_write),
            close_process_file(self.stdout_capture),
            close_process_file(self.stderr_read),
            close_process_file(self.stderr_write),
            close_process_file(self.stderr_capture),
        )


@dataclass(slots=True, frozen=True)
class WaitOutcome:
    returncode: int | None
    error: BaseException | None = None


class ProcessInvocation:
    """A started process plus its controls. Cancel `task` to cancel the process."""

    def __init__(self) -> None:
        self._ready = asyncio.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.stdin: PipeWriter | None = None
        self.start_error: BaseException | None = None
        self.task: asyncio.Task[None] | None = None

    def write_input(self, request: ProcessWriteRequest, events: EventSink) -> asyncio.Task[None]:
        """Cancellation is honored until the pipe write starts.

        Once started, the write owns its data until it completes or the process
        or input lifecycle interrupts the pipe. Concurrent writes may interleave;
        callers that require ordering dispatch the next write after its event.
        """
        return start_primitive(self._write_input(request, events), name="process.write_input")

    def close_input(
        self, request: ProcessCloseInputRequest, events: EventSink
    ) -> asyncio.Task[None]:
        """Idempotent; interrupts any active `write_input`."""
        return start_primitive(self._close_input(request, events), name="process.close_input")

    def signal(self, request: ProcessSignalRequest, events: EventSink) -> asyncio.Task[None]:
        return start_primitive(self._signal(request, events), name="process.signal")

    def _mark_ready(self) -> None:
        self._ready.set()

    def _start_canceled(self) -> bool:
        task = self.task
        return task is not None and (task.cancelled() or task.cancelling() > 0)

    async def _write_input(self, request: ProcessWriteRequest, events: EventSink) -> None:
        try:
            await self._prepare_control(process_group=False)
        except PrimitiveError as error:
            if is_error(error, ContextCanceledError):
                event = primitive_canceled(request.source, request.correlation_id)
            else:
                event = _input_write_failure(request, 0, error)
            await send_terminal(events, event)
            return
        stdin = self.stdin
        if stdin is None:
            await send_terminal(
                events,
                _input_write_failure(request, 0, PrimitiveError("stdin pipe is unavailable")),
            )
            return
        try:
            count = await stdin.write(request.data)
        except PipeWriteError as error:
            await send_terminal(events, _input_write_failure(request, error.count, error))
            return
        await send_terminal(
            events,
            PrimitiveEvent(
                PrimitiveEventType.PROCESS_INPUT_WRITTEN,
                request.source,
                request.correlation_id,
                ProcessWriteResult(count),
            ),
        )

    async def _close_input(self, request: ProcessCloseInputRequest, events: EventSink) -> None:
        try:
            await self._prepare_control(process_group=False)
        except PrimitiveError as error:
            await send_terminal(
                events,
                _control_failure(
                    request.source, request.correlation_id, "close process input", error
                ),
            )
            return
        closed = PrimitiveEvent(
            PrimitiveEventType.PROCESS_INPUT_CLOSED, request.source, request.correlation_id
        )
        error = close_process_file(self.stdin)
        if error is not None:
            closed = primitive_failure(
                request.source, request.correlation_id, wrap("close process input", error)
            )
        await send_terminal(events, closed)

    async def _signal(self, request: ProcessSignalRequest, events: EventSink) -> None:
        try:
            await self._prepare_control(process_group=request.propagate_to_children)
        except PrimitiveError as error:
            await send_terminal(
                events,
                _control_failure(request.source, request.correlation_id, "signal process", error),
            )
            return
        process = self.process
        assert process is not None
        if request.propagate_to_children:
            error = _kill(-process.pid, request.signal)
        else:
            error = signal_direct(process, request.signal)
        if error is not None:
            event = primitive_failure(
                request.source, request.correlation_id, wrap("signal process", error)
            )
        else:
            event = PrimitiveEvent(
                PrimitiveEventType.PROCESS_SIGNALED, request.source, request.correlation_id
            )
        await send_terminal(events, event)

    async def _prepare_control(self, *, process_group: bool) -> None:
        if not self._ready.is_set():
            try:
                await self._ready.wait()
            except asyncio.CancelledError:
                raise ContextCanceledError() from None
        if cancel_requested():
            raise ContextCanceledError()
        if self.start_error is not None:
            # A canceled start does not cancel this independently owned control invocation.
            raise ProcessDoneError(f"process invocation is done: {describe(self.start_error)}")
        if self._start_canceled():
            raise ProcessDoneError()
        process = self.process
        assert process is not None
        if process_group:
            exists, error = process_group_exists(process.pid)
            if error is not None:
                raise wrap("inspect process group", error)
            if not exists:
                raise ProcessDoneError()
            return
        if process_wait_completed(process):
            raise ProcessDoneError()

    async def run(self, request: ProcessStartRequest, events: EventSink) -> None:
        error = validate_process_start_request(request)
        if error is not None:
            self.start_error = error
            self._mark_ready()
            await send_terminal(events, _failure(request, error))
            return
        if cancel_requested():
            self.start_error = ContextCanceledError()
            self._mark_ready()
            await send_terminal(events, _canceled(request))
            return

        try:
            command, pipes = await run_blocking(prepare_process, request)
        except PrimitiveError as error:
            self.start_error = error
            self._mark_ready()
            await send_terminal(events, _failure(request, error))
            return
        if cancel_requested():
            cleanup_error = pipes.close_all()
            self.start_error = ContextCanceledError()
            self._mark_ready()
            await send_terminal(
                events, process_completion_event(request, _canceled(request), cleanup_error)
            )
            return
        try:
            process = await run_blocking(spawn_process, command)
        except Exception as error:
            start_error = join(
                wrap(f"start process {go_quote(request.path)}", error), pipes.close_all()
            )
            assert start_error is not None
            self.start_error = start_error
            self._mark_ready()
            await send_terminal(events, _failure(request, start_error))
            return

        parent = pipes.parent()
        self.process = process
        self.stdin = parent.stdin
        self._mark_ready()
        child_pipe_close_error = pipes.close_child_ends()
        wait_completed = run_in_thread(lambda: wait_process(process))

        await send_event(
            events,
            PrimitiveEvent(
                PrimitiveEventType.PROCESS_STARTED,
                request.source,
                request.correlation_id,
                ProcessStartedResult(process.pid),
            ),
        )
        workers: list[asyncio.Task[None]] = []
        for stream, reader in (
            (ProcessStream.STDOUT, parent.stdout),
            (ProcessStream.STDERR, parent.stderr),
        ):
            if reader is not None:
                workers.append(
                    start_primitive(
                        run_process_output(request, ProcessOutputState(stream, reader), events),
                        name="process.output",
                    )
                )

        def cancel_workers() -> None:
            for worker in workers:
                worker.cancel()

        if cancel_requested():
            cancel_workers()
        outcome, canceled, completion_error = await await_process_completion(
            process, parent, wait_completed, PROCESS_TERMINATION_GRACE_PERIOD, cancel_workers
        )
        capture_sync_error = await run_blocking(parent.sync_captures, on_cancel=cancel_workers)
        capture_close_error = parent.close_captures()
        stdin_close_error = close_process_file(parent.stdin)
        if not canceled:
            parent.finish_output()
        if workers:
            await wait_ignoring_cancellation(
                asyncio.gather(*workers, return_exceptions=True), cancel_workers
            )
        output_close_error = parent.close_output()

        exit_result, exit_error = process_result(request.path, outcome)
        terminal_error = join(
            child_pipe_close_error,
            capture_sync_error,
            capture_close_error,
            stdin_close_error,
            output_close_error,
            completion_error,
            exit_error,
        )
        if canceled:
            await send_terminal(
                events, process_completion_event(request, _canceled(request), terminal_error)
            )
            return
        await send_terminal(
            events,
            process_completion_event(
                request,
                PrimitiveEvent(
                    PrimitiveEventType.PROCESS_EXITED,
                    request.source,
                    request.correlation_id,
                    exit_result,
                ),
                terminal_error,
            ),
        )


def start_process(request: ProcessStartRequest, events: EventSink) -> ProcessInvocation:
    invocation = ProcessInvocation()
    invocation.task = start_primitive(invocation.run(request, events), name="process.start")
    return invocation


async def await_process_completion(
    process: subprocess.Popen[bytes],
    pipes: ProcessParentPipes,
    wait_completed: asyncio.Future[WaitOutcome],
    grace_period: float,
    on_cancel: Callable[[], None] | None = None,
) -> tuple[WaitOutcome, bool, BaseException | None]:
    """Return the wait outcome, whether the invocation was canceled, and cleanup errors."""

    def canceled() -> None:
        if on_cancel is not None:
            on_cancel()

    if cancel_requested():
        canceled()
    else:
        try:
            await asyncio.wait([wait_completed])
        except asyncio.CancelledError:
            canceled()
        else:
            completion_error = await wait_ignoring_cancellation(
                terminate_process(process, ProcessParentPipes(), grace_period), canceled
            )
            return wait_completed.result(), False, completion_error

    if process_wait_completed(process):
        completion_error = await wait_ignoring_cancellation(
            terminate_process(process, ProcessParentPipes(), grace_period), canceled
        )
        return await wait_ignoring_cancellation(wait_completed, canceled), False, completion_error
    cancellation_error = await wait_ignoring_cancellation(
        terminate_process(process, pipes, grace_period), canceled
    )
    return await wait_ignoring_cancellation(wait_completed, canceled), True, cancellation_error


def process_wait_completed(process: subprocess.Popen[bytes] | None) -> bool:
    """Whether the process has been reaped."""
    return process is not None and process.returncode is not None


def wait_process(process: subprocess.Popen[bytes]) -> WaitOutcome:
    try:
        return WaitOutcome(process.wait())
    except Exception as error:
        return WaitOutcome(process.returncode, error)


def spawn_process(command: ProcessCommand) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command.argv,
        executable=command.argv[0],
        stdin=_child_descriptor(command.stdin),
        stdout=_child_descriptor(command.stdout),
        stderr=_child_descriptor(command.stderr),
        cwd=command.directory,
        env=command.environment,
        close_fds=True,
        process_group=0,
    )


def _child_descriptor(file: File | None) -> int:
    return subprocess.DEVNULL if file is None else file.fileno()


class _VerbatimEntry(bytes):
    """An environment entry Popen must pass unchanged.

    Go passes entries without a name or without "=" to the child as-is. Popen
    builds each entry as `key + b"=" + value` and rejects keys containing "=",
    so such an entry is a bytes key that bypasses both steps, paired with an
    empty value. Identity equality keeps repeated entries apart, as Go does. This
    relies on CPython's POSIX `Popen._execute_child`.
    """

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __contains__(self, item: object) -> bool:
        return False

    def __add__(self, value: Buffer, /) -> bytes:
        return bytes(self)


def _environment(entries: list[str] | None) -> dict[bytes, str] | None:
    """Deduplicate like Go's os/exec: the last entry for a name wins, in its position."""
    if entries is None:
        return None
    kept: list[str] = []
    seen: set[str] = set()
    for entry in reversed(entries):
        index = entry.find("=")
        if index == 0:
            index = entry.find("=", 1)
            index = 0 if index < 0 else index
        if index < 0:
            if entry:
                kept.append(entry)
            continue
        name = entry[:index]
        if name in seen:
            continue
        seen.add(name)
        kept.append(entry)
    environment: dict[bytes, str] = {}
    for entry in reversed(kept):
        name, separator, value = entry.partition("=")
        if separator and name:
            environment[os.fsencode(name)] = value
        else:
            environment[_VerbatimEntry(os.fsencode(entry))] = ""
    return environment


def _pipe(path: str, stream: str) -> tuple[File, File]:
    try:
        read, write = os.pipe()
    except OSError as error:
        raise wrap(f"start process {go_quote(path)}: create {stream} pipe", error) from None
    return File(read, "|0"), File(write, "|1")


def prepare_process(request: ProcessStartRequest) -> tuple[ProcessCommand, ProcessPipes]:
    pipes = ProcessPipes()
    path = go_quote(request.path)
    stdout_info: os.stat_result | None = None
    stderr_info: os.stat_result | None = None

    def fail(error: BaseException) -> PrimitiveError:
        joined = join(error, pipes.close_all())
        assert isinstance(joined, PrimitiveError)
        return joined

    if request.pipes & ProcessPipeSet.STDIN:
        read, write = _pipe(request.path, "stdin")
        pipes.stdin_read = read
        pipes.stdin_write = PipeWriter(write)

    if request.pipes & ProcessPipeSet.STDOUT:
        try:
            read, write = _pipe(request.path, "stdout")
        except PrimitiveError as error:
            raise fail(error) from None
        pipes.stdout_read = PipeReader(read)
        pipes.stdout_write = write
    elif request.stdout_path:
        try:
            pipes.stdout_capture, stdout_info = open_process_capture(request.stdout_path, "stdout")
        except PrimitiveError as error:
            raise fail(wrap(f"start process {path}", error)) from None

    if request.pipes & ProcessPipeSet.STDERR:
        try:
            read, write = _pipe(request.path, "stderr")
        except PrimitiveError as error:
            raise fail(error) from None
        pipes.stderr_read = PipeReader(read)
        pipes.stderr_write = write
    elif request.stderr_path:
        try:
            pipes.stderr_capture, stderr_info = open_process_capture(request.stderr_path, "stderr")
        except PrimitiveError as error:
            raise fail(wrap(f"start process {path}", error)) from None

    if (
        stdout_info is not None
        and stderr_info is not None
        and (stdout_info.st_dev, stdout_info.st_ino) == (stderr_info.st_dev, stderr_info.st_ino)
    ):
        raise fail(
            PrimitiveError(
                f"start process {path}: stdout and stderr capture paths identify the same file"
            )
        )
    for capture, capture_path, stream in (
        (pipes.stdout_capture, request.stdout_path, "stdout"),
        (pipes.stderr_capture, request.stderr_path, "stderr"),
    ):
        if capture is None:
            continue
        try:
            capture.truncate(0)
        except Exception as error:
            raise fail(
                wrap(
                    f"start process {path}: truncate process {stream} capture "
                    f"{go_quote(capture_path)}",
                    error,
                )
            ) from None

    command = ProcessCommand(
        argv=[request.path, *request.arguments],
        directory=request.directory or None,
        environment=_environment(request.environment),
        stdin=pipes.stdin_read,
        stdout=pipes.stdout_write or pipes.stdout_capture,
        stderr=pipes.stderr_write or pipes.stderr_capture,
    )
    return command, pipes


def open_process_capture(path: str, stream: str) -> tuple[File, os.stat_result]:
    quoted = go_quote(path)
    try:
        file = File.open(path, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        raise wrap(f"open process {stream} capture {quoted}", error) from None
    try:
        info = file.stat()
    except OSError as error:
        raise _raised(
            join(
                wrap(f"inspect process {stream} capture {quoted}", error), close_process_file(file)
            )
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise _raised(
            join(
                PrimitiveError(
                    f"open process {stream} capture {quoted}: path is not a regular file"
                ),
                close_process_file(file),
            )
        )
    try:
        os.set_blocking(file.fileno(), True)
    except OSError as error:
        raise _raised(
            join(
                wrap(f"open process {stream} capture {quoted}: set blocking", error),
                close_process_file(file),
            )
        ) from None
    return file, info


def sync_process_capture(file: File | None, stream: str) -> BaseException | None:
    if file is None:
        return None
    try:
        file.sync()
    except Exception as error:
        return wrap(f"sync process {stream} capture", error)
    return None


def close_process_file(file: _Closer | None) -> BaseException | None:
    if file is None:
        return None
    try:
        file.close()
    except FileClosedError:
        return None
    except Exception as error:
        return error
    return None


def _kill(pid: int, signal_number: int) -> OSError | None:
    try:
        os.kill(pid, signal_number)
    except OSError as error:
        return error
    return None


def signal_direct(process: subprocess.Popen[bytes], signal_number: int) -> BaseException | None:
    """Signal only the direct child, like Go's `os.Process.Signal`."""
    if process_wait_completed(process):
        return ProcessFinishedError()
    return _kill(process.pid, signal_number)


def signal_process_invocation(
    process: subprocess.Popen[bytes], signal_number: int
) -> BaseException | None:
    error = normalize_process_group_signal_error(process.pid, _kill(-process.pid, signal_number))
    if error is None:
        return None
    if not has_errno(error, errno.ESRCH):
        return error
    error = signal_direct(process, signal_number)
    if error is None or isinstance(error, ProcessFinishedError) or has_errno(error, errno.ESRCH):
        return None
    return error


async def terminate_process(
    process: subprocess.Popen[bytes], pipes: ProcessParentPipes, grace_period: float
) -> BaseException | None:
    """Signal the group with SIGTERM, wait up to the grace period, then SIGKILL."""
    term_error = signal_process_invocation(process, signal.SIGTERM)
    exited, wait_error = await wait_for_process_invocation(process, grace_period)
    # Unlike Darwin's exiting-member check, this forgives EPERM only after confirmed completion.
    if exited and has_errno(term_error, errno.EPERM):
        term_error = None
    kill_error = None
    if not exited:
        kill_error = signal_process_invocation(process, signal.SIGKILL)
    return join(term_error, wait_error, kill_error, pipes.close_all())


async def wait_for_process_invocation(
    process: subprocess.Popen[bytes], grace_period: float
) -> tuple[bool, BaseException | None]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_period
    delay = 0.001
    while True:
        exists, error = process_group_exists(process.pid)
        if error is not None:
            return False, error
        if not exists and process_wait_completed(process):
            return True, None
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False, None
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.05)


def process_group_exists(process_group_id: int) -> tuple[bool, BaseException | None]:
    error = _kill(-process_group_id, 0)
    if error is None or error.errno == errno.EPERM:
        return True, None
    if error.errno == errno.ESRCH:
        return False, None
    return False, error


@dataclass(slots=True)
class ProcessOutputState:
    stream: ProcessStream
    reader: PipeReader


async def run_process_output(
    request: ProcessStartRequest, output: ProcessOutputState, events: EventSink
) -> None:
    error = await stream_process_output(request, output, events)
    if error is not None:
        await send_event(
            events,
            PrimitiveEvent(
                PrimitiveEventType.PROCESS_STREAM_FAILED,
                request.source,
                request.correlation_id,
                ProcessStreamFailureResult(output.stream, describe(error)),
            ),
        )


async def stream_process_output(
    request: ProcessStartRequest, output: ProcessOutputState, events: EventSink
) -> BaseException | None:
    offset = 0
    name = process_stream_name(output.stream)
    while True:
        try:
            data = await output.reader.read(PROCESS_OUTPUT_CHUNK_SIZE)
        except asyncio.CancelledError:
            # Keep reading so a terminating process never blocks on a full pipe; the
            # recorded cancellation drops every later event.
            continue
        except ReadDeadlineError:
            try:
                await drain_process_output(request, output.stream, output.reader, offset, events)
            except PrimitiveError as error:
                return error
            return None
        except Exception as error:
            if cancel_requested() and isinstance(error, FileClosedError):
                return None
            return wrap(f"read process {name}", error)
        if not data:
            return None
        await send_process_output(request, output.stream, offset, data, events)
        offset += len(data)


class _HasFileno(Protocol):
    def fileno(self) -> int: ...


async def drain_process_output(
    request: ProcessStartRequest,
    stream: ProcessStream,
    reader: _HasFileno,
    offset: int,
    events: EventSink,
) -> int:
    """Read what is already buffered without blocking. Returns the next offset."""
    name = process_stream_name(stream)
    try:
        descriptor = reader.fileno()
        os.set_blocking(descriptor, False)
    except Exception as error:
        raise wrap(f"drain process {name}: set nonblocking", error) from None
    while True:
        try:
            data = os.read(descriptor, PROCESS_OUTPUT_CHUNK_SIZE)
        except BlockingIOError:
            return offset
        except OSError as error:
            raise wrap(f"drain process {name}", error) from None
        if not data:
            return offset
        if not await send_process_output(request, stream, offset, data, events):
            return offset
        offset += len(data)


async def send_process_output(
    request: ProcessStartRequest,
    stream: ProcessStream,
    offset: int,
    data: bytes,
    events: EventSink,
) -> bool:
    return await send_event(
        events,
        PrimitiveEvent(
            PrimitiveEventType.PROCESS_OUTPUT,
            request.source,
            request.correlation_id,
            ProcessOutputResult(stream, offset, data),
        ),
    )


def process_result(
    path: str, outcome: WaitOutcome
) -> tuple[ProcessExitResult, BaseException | None]:
    if outcome.returncode is None or outcome.error is not None:
        cause = outcome.error if outcome.error is not None else PrimitiveError("no exit status")
        return ProcessExitResult(), wrap(f"wait for process {go_quote(path)}", cause)
    if outcome.returncode < 0:
        return ProcessExitResult(exit_code=-1, signal=-outcome.returncode), None
    return ProcessExitResult(exit_code=outcome.returncode), None


def process_stream_name(stream: ProcessStream) -> str:
    return "stderr" if stream == ProcessStream.STDERR else "stdout"


def validate_process_start_request(request: ProcessStartRequest) -> PrimitiveError | None:
    if not request.path:
        return PrimitiveError("start process: path must be set")
    if not os.path.isabs(request.path):
        return PrimitiveError("start process: path must be absolute")
    if int(request.pipes) & ~int(ProcessPipeSet.ALL):
        return PrimitiveError(f"start process: unsupported pipe selection {int(request.pipes):#x}")
    for name, path, pipe in (
        ("stdout", request.stdout_path, ProcessPipeSet.STDOUT),
        ("stderr", request.stderr_path, ProcessPipeSet.STDERR),
    ):
        if not path:
            continue
        if not os.path.isabs(path):
            return PrimitiveError(f"start process: {name} capture path must be absolute")
        if request.pipes & pipe:
            return PrimitiveError(
                f"start process: {name} cannot use both a pipe and a capture path"
            )
    if (
        request.stdout_path
        and request.stderr_path
        and os.path.normpath(request.stdout_path) == os.path.normpath(request.stderr_path)
    ):
        return PrimitiveError("start process: stdout and stderr capture paths must differ")
    return None


def process_completion_event(
    request: ProcessStartRequest, event: PrimitiveEvent, error: BaseException | None
) -> PrimitiveEvent:
    if error is not None:
        return _failure(request, error)
    return event


def _failure(request: ProcessStartRequest, error: BaseException) -> PrimitiveEvent:
    return primitive_failure(request.source, request.correlation_id, error)


def _canceled(request: ProcessStartRequest) -> PrimitiveEvent:
    return primitive_canceled(request.source, request.correlation_id)


def _control_failure(
    source: SourceID, correlation_id: CorrelationID, action: str, error: BaseException
) -> PrimitiveEvent:
    if is_error(error, ContextCanceledError):
        return primitive_canceled(source, correlation_id)
    return primitive_failure(source, correlation_id, wrap(action, error))


def _input_write_failure(
    request: ProcessWriteRequest, count: int, error: BaseException
) -> PrimitiveEvent:
    return PrimitiveEvent(
        PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED,
        request.source,
        request.correlation_id,
        ProcessWriteFailureResult(count, describe(wrap("write process input", error))),
    )


def _raised(error: BaseException | None) -> BaseException:
    assert error is not None
    return error
