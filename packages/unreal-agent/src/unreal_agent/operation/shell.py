import os
import posixpath
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import msgspec
from msgspec.structs import replace

from unreal_agent.gojson import Int64, JSONError, Raw, Struct, decode, encode
from unreal_agent.operation.operation import (
    ID,
    Operation,
    OperationError,
    PrimitiveDispatch,
    Spec,
    Status,
    Step,
    unsupported,
)
from unreal_agent.operation.output import MAX_OUTPUT_LENGTH, _bound_output, bound_output
from unreal_agent.primitives import (
    CorrelationID,
    IOCreateKind,
    IOCreateRequest,
    IOCreateResult,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    ProcessExitResult,
    ProcessStartedResult,
    ProcessStartRequest,
    SourceID,
)
from unreal_agent.primitives._errors import go_quote

__all__ = [
    "SHELL_ERR_FILENAME",
    "SHELL_OUT_FILENAME",
    "TYPE_SHELL",
    "VERSION_SHELL",
    "Shell",
    "ShellInput",
    "ShellPhase",
    "ShellResult",
    "ShellState",
    "decode_shell_state",
    "new_shell_spec",
]

TYPE_SHELL: Final = "shell"
VERSION_SHELL: Final = 3
SHELL_OUT_FILENAME: Final = "out"
SHELL_ERR_FILENAME: Final = "err"
_UTF_MAX: Final = 4


class ShellPhase(StrEnum):
    NONE = ""
    CREATE_DIRECTORY = "create_directory"
    CREATE_OUT = "create_out"
    CREATE_ERR = "create_err"
    PROCESS = "process"
    READ_OUT = "read_out"
    READ_OUT_TAIL = "read_out_tail"
    READ_ERR = "read_err"
    READ_ERR_TAIL = "read_err_tail"


_READ_PHASES: Final = frozenset(
    {ShellPhase.READ_OUT, ShellPhase.READ_ERR, ShellPhase.READ_OUT_TAIL, ShellPhase.READ_ERR_TAIL}
)
_CREATE_PHASES: Final = frozenset(
    {ShellPhase.CREATE_DIRECTORY, ShellPhase.CREATE_OUT, ShellPhase.CREATE_ERR}
)
_TAIL_PHASES: Final = frozenset({ShellPhase.READ_OUT_TAIL, ShellPhase.READ_ERR_TAIL})


class ShellInput(Struct):
    Command: str = ""
    Shell: str = ""
    Directory: str = ""


class ShellResult(Struct):
    Out: str = ""
    Err: str = ""
    OutSize: Int64 = 0
    ErrSize: Int64 = 0
    ExitCode: Int64 = 0


class ShellState(Struct):
    Input: ShellInput = msgspec.field(default_factory=ShellInput)
    BaseDirectory: str = ""

    Phase: str = ShellPhase.NONE
    ProcessGroupID: Int64 = 0
    PendingExitCode: Int64 | None = None
    OutSize: Int64 = 0
    ErrSize: Int64 = 0
    InlineOut: bytes = b""
    InlineErr: bytes = b""
    InlineOutTail: bytes = b""
    InlineErrTail: bytes = b""
    Result: ShellResult | None = None
    TerminalError: str = ""
    ErrorTruncated: bool = False
    OutTruncated: bool = False
    ErrTruncated: bool = False
    OutPath: str = ""
    ErrPath: str = ""


def new_shell_spec(input: ShellInput, base_directory: str, max_output_length: int) -> Spec:
    if max_output_length <= 0 or max_output_length > MAX_OUTPUT_LENGTH:
        raise OperationError("max output length is out of range")
    state = ShellState(
        Input=input, BaseDirectory=base_directory, OutTruncated=True, ErrTruncated=True
    )
    _validate_shell_state(state)
    return Spec(
        MaxOutputLength=max_output_length,
        Type=TYPE_SHELL,
        Version=VERSION_SHELL,
        State=Raw(encode(state)),
    )


def decode_shell_state(current: Operation) -> ShellState:
    operation_id = go_quote(current.ID)
    if current.Type != TYPE_SHELL:
        raise unsupported(
            f"advance shell operation {operation_id}: unsupported type {go_quote(current.Type)}"
        )
    if current.Version != VERSION_SHELL:
        raise unsupported(
            f"advance shell operation {operation_id}: unsupported version {current.Version}"
        )
    if current.MaxOutputLength <= 0 or current.MaxOutputLength > MAX_OUTPUT_LENGTH:
        raise OperationError("max output length is out of range")
    try:
        return decode(bytes(current.State), ShellState)
    except JSONError as error:
        raise OperationError(f"decode shell operation {operation_id} state: {error}") from error


def _shell_operation_state(current: Operation) -> ShellState:
    state = decode_shell_state(current)
    try:
        _validate_shell_state(state)
    except OperationError as error:
        raise OperationError(
            f"validate shell operation {go_quote(current.ID)} state: {error}"
        ) from error
    limit = _read_limit(current)
    if (
        len(state.InlineOut) + len(state.InlineOutTail) > limit
        or len(state.InlineErr) + len(state.InlineErrTail) > limit
    ):
        raise OperationError("inline shell result exceeds the configured limit")
    return state


def _read_limit(current: Operation) -> int:
    return current.MaxOutputLength * _UTF_MAX


def _tail_read_limit(current: Operation) -> int:
    limit = current.MaxOutputLength
    return (limit - limit // 2) * _UTF_MAX


@dataclass(slots=True, frozen=True)
class _ShellPaths:
    directory: str
    out: str
    err: str


class Shell:
    """The shell actor. Read chunks are transient; checkpoints hold only complete reads."""

    __slots__ = ("_chunks", "_current", "_read_size", "_state")

    def __init__(self, current: Operation) -> None:
        self._state = _shell_operation_state(current)
        self._current = replace(current, State=Raw(b""))
        self._chunks: list[bytes] = []
        self._read_size = 0

    def handle(self, event: PrimitiveEvent | None) -> Step:
        current, state = self._current, self._state
        operation_id = go_quote(current.ID)
        match current.Status:
            case Status.READY:
                if event is not None:
                    raise OperationError(
                        f"advance ready shell operation {operation_id}: unexpected primitive event"
                    )
                if state.Phase or state.Result is not None or state.TerminalError:
                    raise OperationError(
                        f"advance ready shell operation {operation_id}: state is not initial"
                    )
                try:
                    paths = _new_shell_paths(state.BaseDirectory, current.ID)
                except OperationError as error:
                    return self.fail(error)
                return self._next(ShellPhase.CREATE_DIRECTORY, paths)
            case Status.AWAITING:
                if event is None:
                    return self._resume()
                return self._handle_awaiting(event)
            case Status.CANCELING:
                if event is not None:
                    raise OperationError(
                        f"advance canceling shell operation {operation_id}: "
                        "unexpected primitive event"
                    )
                return self.cancel()
            case _:
                raise OperationError(
                    f"advance shell operation {operation_id}: "
                    f"terminal status {go_quote(current.Status)}"
                )

    def _resume(self) -> Step:
        state = self._state
        if state.Phase == ShellPhase.PROCESS:
            if state.ProcessGroupID == 0:
                return self.fail(
                    "shell execution outcome is unknown because process start was not recorded"
                )
            return self.fail("shell execution was interrupted before an exit status was recorded")
        try:
            paths = _new_shell_paths(state.BaseDirectory, self._current.ID)
        except OperationError as error:
            return self.fail(error)
        self._chunks, self._read_size = [], 0
        return self._dispatch_phase(paths)

    def _next(self, phase: ShellPhase, paths: _ShellPaths) -> Step:
        self._state.Phase = phase
        return self._dispatch_phase(paths)

    def _dispatch_phase(self, paths: _ShellPaths) -> Step:
        current, state = self._current, self._state
        correlation = CorrelationID(state.Phase)
        match state.Phase:
            case ShellPhase.CREATE_DIRECTORY:
                return self._dispatch(
                    _create_shell_path(
                        current.ID, correlation, IOCreateKind.DIRECTORY, paths.directory, 0o700
                    )
                )
            case ShellPhase.CREATE_OUT:
                return self._dispatch(_create_shell_file(current.ID, correlation, paths.out))
            case ShellPhase.CREATE_ERR:
                return self._dispatch(_create_shell_file(current.ID, correlation, paths.err))
            case ShellPhase.PROCESS:
                return self._dispatch(_start_shell_process(current.ID, state, paths))
            case phase if phase in _READ_PHASES:
                try:
                    request = self._read_request(paths)
                except OperationError as error:
                    return self.fail(error)
                return self._dispatch(PrimitiveDispatch(PrimitiveDispatchType.IO_READ, request))
            case _:
                return self.fail(f"shell operation has invalid phase {go_quote(state.Phase)}")

    def _handle_awaiting(self, event: PrimitiveEvent) -> Step:
        current, state = self._current, self._state
        if event.source != current.ID:
            return self.fail(
                f"shell primitive event source is {go_quote(event.source)}, "
                f"want {go_quote(current.ID)}"
            )
        if event.type == PrimitiveEventType.CANCELED:
            return self.cancel()
        if event.type == PrimitiveEventType.FAILED:
            return self.fail(_shell_primitive_failure(event))
        if event.correlation_id != state.Phase:
            return self.fail(
                f"shell primitive event correlation is {go_quote(event.correlation_id)}, "
                f"want {go_quote(state.Phase)}"
            )
        try:
            paths = _new_shell_paths(state.BaseDirectory, current.ID)
        except OperationError as error:
            return self.fail(error)
        if state.Phase in _CREATE_PHASES:
            return self._created(event, paths)
        if state.Phase == ShellPhase.PROCESS:
            return self._process_event(event, paths)
        if state.Phase in _READ_PHASES:
            return self._read_event(event, paths)
        return self.fail(f"shell operation has invalid phase {go_quote(state.Phase)}")

    def _created(self, event: PrimitiveEvent, paths: _ShellPaths) -> Step:
        state = self._state
        kind = (
            IOCreateKind.DIRECTORY
            if state.Phase == ShellPhase.CREATE_DIRECTORY
            else IOCreateKind.REGULAR_FILE
        )
        result = event.result
        if (
            event.type != PrimitiveEventType.IO_CREATE_COMPLETED
            or not isinstance(result, IOCreateResult)
            or result.kind != kind
        ):
            return self.fail("create shell artifact returned an invalid result")
        match state.Phase:
            case ShellPhase.CREATE_DIRECTORY:
                return self._next(ShellPhase.CREATE_OUT, paths)
            case ShellPhase.CREATE_OUT:
                state.OutPath = paths.out
                return self._next(ShellPhase.CREATE_ERR, paths)
            case _:
                state.ErrPath = paths.err
                return self._next(ShellPhase.PROCESS, paths)

    def _process_event(self, event: PrimitiveEvent, paths: _ShellPaths) -> Step:
        state = self._state
        match event.type:
            case PrimitiveEventType.PROCESS_STARTED:
                started = event.result
                if not isinstance(started, ProcessStartedResult) or started.pid <= 1:
                    return self.fail("start shell returned an invalid result")
                state.ProcessGroupID = started.pid
                return self._await()
            case PrimitiveEventType.PROCESS_EXITED:
                exit_result = event.result
                if not isinstance(exit_result, ProcessExitResult):
                    return self.fail("shell returned an invalid exit result")
                state.ProcessGroupID = 0
                state.PendingExitCode = _shell_exit_status(exit_result)
                return self._next(ShellPhase.READ_OUT, paths)
            case PrimitiveEventType.PROCESS_OUTPUT:
                return self.fail("captured shell process produced an unexpected output event")
            case PrimitiveEventType.PROCESS_STREAM_FAILED:
                return self.fail("captured shell process returned an unexpected stream failure")
            case _:
                return self.fail(f"shell process returned unexpected event {go_quote(event.type)}")

    def _read_request(self, paths: _ShellPaths) -> IOReadRequest:
        current, state = self._current, self._state
        request = IOReadRequest(
            source=SourceID(current.ID),
            correlation_id=CorrelationID(state.Phase),
            count=_read_limit(current),
        )
        if state.Phase in (ShellPhase.READ_OUT, ShellPhase.READ_OUT_TAIL):
            request.path, size = paths.out, state.OutSize
        elif state.Phase in (ShellPhase.READ_ERR, ShellPhase.READ_ERR_TAIL):
            request.path, size = paths.err, state.ErrSize
        else:
            raise OperationError(f"shell operation has invalid read phase {go_quote(state.Phase)}")
        if state.Phase in _TAIL_PHASES:
            if size <= _read_limit(current):
                raise OperationError("invalid shell tail read size")
            request.count = _tail_read_limit(current)
            request.offset = size - request.count
        return request

    def _read_event(self, event: PrimitiveEvent, paths: _ShellPaths) -> Step:
        try:
            request = self._read_request(paths)
        except OperationError as error:
            return self.fail(error)
        read_size = self._read_size
        match event.type:
            case PrimitiveEventType.IO_READ_OUTPUT:
                output = event.result
                if (
                    not isinstance(output, IOReadOutputResult)
                    or output.offset != request.offset + read_size
                ):
                    return self.fail("shell read returned invalid output")
                if len(output.data) > request.count - read_size:
                    return self.fail("shell read exceeded its configured limit")
                self._chunks.append(output.data)
                self._read_size += len(output.data)
                return Step()
            case PrimitiveEventType.IO_READ_COMPLETED:
                result = event.result
                if (
                    not isinstance(result, IOReadCompletedResult)
                    or result.size < request.offset
                    or read_size != min(request.count, result.size - request.offset)
                ):
                    return self.fail("shell read returned an invalid completion")
                if (
                    self._state.Phase in _TAIL_PHASES
                    and result.size != request.offset + request.count
                ):
                    return self.fail("shell capture size changed while reading its tail")
                data = b"".join(self._chunks)
                self._chunks, self._read_size = [], 0
                return self._read_completed(data, result.size, paths)
            case _:
                return self.fail(f"shell read returned unexpected event {go_quote(event.type)}")

    def _read_completed(self, data: bytes, size: int, paths: _ShellPaths) -> Step:
        current, state = self._current, self._state
        needs_tail = size > _read_limit(current)
        if needs_tail and state.Phase in (ShellPhase.READ_OUT, ShellPhase.READ_ERR):
            data = data[: _read_limit(current) - _tail_read_limit(current)]
        match state.Phase:
            case ShellPhase.READ_OUT:
                state.InlineOut, state.OutSize, state.InlineOutTail = data, size, b""
                if needs_tail:
                    return self._next(ShellPhase.READ_OUT_TAIL, paths)
                return self._next(ShellPhase.READ_ERR, paths)
            case ShellPhase.READ_OUT_TAIL:
                state.InlineOutTail = data
                return self._next(ShellPhase.READ_ERR, paths)
            case ShellPhase.READ_ERR:
                state.InlineErr, state.ErrSize, state.InlineErrTail = data, size, b""
                if needs_tail:
                    return self._next(ShellPhase.READ_ERR_TAIL, paths)
                return self._finish(paths)
            case _:
                state.InlineErrTail = data
                return self._finish(paths)

    def _finish(self, paths: _ShellPaths) -> Step:
        current, state = self._current, self._state
        if state.PendingExitCode is None:
            return self.fail("shell execution completed without an exit status")
        out, state.OutTruncated = _bound_output(
            state.InlineOut, state.InlineOutTail, state.OutSize, current.MaxOutputLength, paths.out
        )
        err, state.ErrTruncated = _bound_output(
            state.InlineErr, state.InlineErrTail, state.ErrSize, current.MaxOutputLength, paths.err
        )
        state.Result = ShellResult(
            Out=out,
            Err=err,
            OutSize=state.OutSize,
            ErrSize=state.ErrSize,
            ExitCode=state.PendingExitCode,
        )
        state.OutPath, state.ErrPath = paths.out, paths.err
        state.PendingExitCode = None
        state.InlineOut = state.InlineErr = state.InlineOutTail = state.InlineErrTail = b""
        state.Phase = ShellPhase.NONE
        self._current = replace(current, Status=Status.COMPLETED)
        return self._checkpoint()

    def _dispatch(self, dispatch: PrimitiveDispatch) -> Step:
        step = self._await()
        step.dispatches = [dispatch]
        return step

    def _await(self) -> Step:
        self._current = replace(self._current, Status=Status.AWAITING)
        return self._checkpoint()

    def _checkpoint(self) -> Step:
        current, state = self._current, self._state
        if not state.ErrorTruncated:
            state.TerminalError, state.ErrorTruncated = bound_output(
                state.TerminalError, current.MaxOutputLength
            )
        return Step(operation=replace(current, State=Raw(encode(state))))

    def fail(self, error: BaseException | str) -> Step:
        self._chunks, self._read_size = [], 0
        state = self._state
        state.Phase = ShellPhase.NONE
        state.TerminalError = str(error)
        state.ErrorTruncated = False
        self._current = replace(self._current, Status=Status.FAILED)
        return self._checkpoint()

    def cancel(self) -> Step:
        self._chunks, self._read_size = [], 0
        state = self._state
        state.Phase = ShellPhase.NONE
        state.TerminalError = "shell operation canceled"
        state.ErrorTruncated = False
        self._current = replace(self._current, Status=Status.CANCELED)
        return self._checkpoint()


def _go_base(path: str) -> str:
    if not path:
        return "."
    trimmed = path.rstrip("/")
    if not trimmed:
        return "/"
    return trimmed[trimmed.rfind("/") + 1 :]


def _go_join(*elements: str) -> str:
    joined = "/".join(element for element in elements if element)
    if not joined:
        return ""
    cleaned = posixpath.normpath(joined)
    return "/" + cleaned.lstrip("/") if cleaned.startswith("//") else cleaned


def _new_shell_paths(base_directory: str, operation_id: ID) -> _ShellPaths:
    name = operation_id
    if name in ("", ".", "..") or os.path.isabs(name) or _go_base(name) != name:
        raise OperationError(f"operation ID {go_quote(operation_id)} is not a path component")
    directory = _go_join(base_directory, name)
    return _ShellPaths(
        directory=directory,
        out=_go_join(directory, SHELL_OUT_FILENAME),
        err=_go_join(directory, SHELL_ERR_FILENAME),
    )


def _validate_shell_state(state: ShellState) -> None:
    if not state.Input.Shell or not os.path.isabs(state.Input.Shell):
        raise OperationError("shell path must be absolute")
    if not state.BaseDirectory or not os.path.isabs(state.BaseDirectory):
        raise OperationError("base directory must be absolute")
    if state.ProcessGroupID < 0 or state.ProcessGroupID == 1:
        raise OperationError("shell process group ID must be zero or greater than one")
    if state.OutSize < 0 or state.ErrSize < 0:
        raise OperationError("captured output sizes must not be negative")


def _create_shell_path(
    operation_id: ID, correlation: CorrelationID, kind: IOCreateKind, path: str, mode: int
) -> PrimitiveDispatch:
    return PrimitiveDispatch(
        PrimitiveDispatchType.IO_CREATE,
        IOCreateRequest(
            source=SourceID(operation_id),
            correlation_id=correlation,
            kind=kind,
            path=path,
            mode=mode,
        ),
    )


def _create_shell_file(
    operation_id: ID, correlation: CorrelationID, path: str
) -> PrimitiveDispatch:
    return _create_shell_path(operation_id, correlation, IOCreateKind.REGULAR_FILE, path, 0o600)


def _start_shell_process(
    operation_id: ID, state: ShellState, paths: _ShellPaths
) -> PrimitiveDispatch:
    return PrimitiveDispatch(
        PrimitiveDispatchType.PROCESS_START,
        ProcessStartRequest(
            source=SourceID(operation_id),
            correlation_id=CorrelationID(ShellPhase.PROCESS),
            path=state.Input.Shell,
            arguments=["-c", state.Input.Command],
            directory=state.Input.Directory,
            stdout_path=paths.out,
            stderr_path=paths.err,
        ),
    )


def _shell_exit_status(exit_result: ProcessExitResult) -> int:
    if exit_result.signal != 0:
        return 128 + exit_result.signal
    return exit_result.exit_code


def _shell_primitive_failure(event: PrimitiveEvent) -> str:
    if not isinstance(event.result, PrimitiveFailureResult):
        return "shell primitive failed with an invalid result"
    return event.result.error
