import asyncio
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw, encode
from unreal_agent.operation import (
    SHELL_ERR_FILENAME,
    SHELL_OUT_FILENAME,
    Operation,
    OperationError,
    PrimitiveDispatch,
    Shell,
    ShellInput,
    ShellPhase,
    ShellState,
    Status,
    new_shell_spec,
)
from unreal_agent.primitives import (
    Channel,
    CorrelationID,
    IOCreateKind,
    IOCreateRequest,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    ProcessExitResult,
    ProcessStartedResult,
    ProcessStartRequest,
    SourceID,
    create,
    read_file,
    start_process,
)

from tests.operation._helpers import (
    advance_shell_once,
    assert_process_group_gone,
    create_shell_artifacts,
    dispatch_data,
    new_shell_operation,
    one_dispatch_data,
    shell_operation_with_state,
    shell_state,
)

TEST_SHELL_PATH: Final = "/bin/sh"


_TERMINAL_EVENTS: Final = frozenset(
    {
        PrimitiveEventType.IO_CREATE_COMPLETED,
        PrimitiveEventType.IO_READ_COMPLETED,
        PrimitiveEventType.PROCESS_EXITED,
        PrimitiveEventType.FAILED,
        PrimitiveEventType.CANCELED,
    }
)


def assert_file_contents(path: str, want: bytes) -> None:
    with open(path, "rb") as file:
        assert file.read() == want, path


@dataclass(slots=True)
class PrimitiveInvocation:
    source: SourceID
    correlation_id: CorrelationID
    events: Channel[PrimitiveEvent]
    task: asyncio.Task[None]

    async def next(self, stop: asyncio.Event | None) -> PrimitiveEvent | None:
        """Return the next event, or None when `stop` is set first."""
        if stop is None:
            return await self.events.get()
        if stop.is_set():
            return None
        receive = asyncio.ensure_future(self.events.get())
        stopped = asyncio.ensure_future(stop.wait())
        await asyncio.wait([receive, stopped], return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
        if receive.done():
            return receive.result()
        receive.cancel()
        await asyncio.wait([receive])
        if not receive.cancelled():
            return receive.result()
        return None

    async def drain(self) -> None:
        self.task.cancel()
        while (await self.events.get()).type not in _TERMINAL_EVENTS:
            pass


def start_primitive(dispatch: PrimitiveDispatch) -> PrimitiveInvocation:
    events: Channel[PrimitiveEvent] = Channel()
    data: Any = dispatch.data
    match dispatch.type:
        case PrimitiveDispatchType.IO_CREATE:
            assert isinstance(data, IOCreateRequest)
            task = create(data, events)
        case PrimitiveDispatchType.IO_READ:
            assert isinstance(data, IOReadRequest)
            task = read_file(data, events)
        case PrimitiveDispatchType.PROCESS_START:
            assert isinstance(data, ProcessStartRequest)
            started = start_process(data, events).task
            assert started is not None
            task = started
        case _:
            pytest.fail(f"unexpected dispatch type {dispatch.type!r}")
    return PrimitiveInvocation(data.source, data.correlation_id, events, task)


def primitive_event_is_terminal(event: PrimitiveEvent) -> bool:
    return event.type in _TERMINAL_EVENTS


async def receive_only_primitive_event(events: Channel[PrimitiveEvent]) -> PrimitiveEvent:
    async with asyncio.timeout(15):
        event = await events.get()
    await asyncio.sleep(0.01)
    assert events.empty(), "primitive returned another event"
    return event


type Observer = Callable[[Operation, PrimitiveEvent], None]


async def run_shell_actor(
    current: Operation, observe: Observer | None = None, stop: asyncio.Event | None = None
) -> Operation:
    async with asyncio.timeout(30):
        shell = Shell(current)
        step = shell.handle(None)
        assert step.operation is not None
        current = step.operation
        active: PrimitiveInvocation | None = None
        try:
            while current.Status == Status.AWAITING:
                assert len(step.dispatches) <= 1, step.dispatches
                if step.dispatches:
                    assert active is None, f"primitive {active!r} is still active"
                    active = start_primitive(step.dispatches[0])
                assert active is not None, "shell actor is awaiting no active primitive"
                event = await active.next(stop)
                if event is None:
                    await active.drain()
                    active = None
                    step = shell.handle(
                        PrimitiveEvent(PrimitiveEventType.CANCELED, SourceID(current.ID))
                    )
                else:
                    assert (event.source, event.correlation_id) == (
                        active.source,
                        active.correlation_id,
                    )
                    if primitive_event_is_terminal(event):
                        active = None
                    step = shell.handle(event)
                    if step.operation is not None and observe is not None:
                        observe(step.operation, event)
                if step.operation is not None:
                    current = step.operation
        finally:
            if active is not None:
                await active.drain()
        return current


async def test_shell_actor_captures_artifacts_and_completes(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_directory = str(tmp_path_factory.mktemp("base"))
    working_directory = str(tmp_path_factory.mktemp("work"))
    stdout = "stdout-" * (8 * 1024)
    stderr = "stderr-" * (7 * 1024)
    monkeypatch.setenv("SHELL_STDOUT", stdout)
    monkeypatch.setenv("SHELL_STDERR", stderr)
    current = new_shell_operation(
        "shell-complete",
        ShellInput(
            Shell=TEST_SHELL_PATH,
            Command="printf '%s' \"$SHELL_STDOUT\"; printf '%s' \"$SHELL_STDERR\" >&2; exit 7",
            Directory=working_directory,
        ),
        base_directory,
        19,
    )
    completed = await run_shell_actor(current)
    assert completed.Status == Status.COMPLETED
    state = shell_state(completed)
    assert state.Result is not None
    assert state.TerminalError == ""
    assert state.Phase == ""
    result = state.Result
    assert result.Out == (
        stdout[:9]
        + f"...57325 bytes truncated; complete output in {state.OutPath}..."
        + stdout[-10:]
    )
    assert result.Err == (
        stderr[:9]
        + f"...50157 bytes truncated; complete output in {state.ErrPath}..."
        + stderr[-10:]
    )
    assert (result.OutSize, result.ErrSize, result.ExitCode) == (len(stdout), len(stderr), 7)

    directory = os.path.join(base_directory, current.ID)
    assert_file_contents(os.path.join(directory, SHELL_OUT_FILENAME), stdout.encode())
    assert_file_contents(os.path.join(directory, SHELL_ERR_FILENAME), stderr.encode())
    assert len(os.listdir(directory)) == 2


def test_shell_spec_rejects_zero_inline_length(tmp_path: Path) -> None:
    with pytest.raises(OperationError):
        new_shell_spec(ShellInput(Shell=TEST_SHELL_PATH), str(tmp_path), 0)


async def test_shell_actor_starts_shell_directly_with_capture_paths(tmp_path: Path) -> None:
    base_directory = str(tmp_path)
    current = new_shell_operation(
        "shell-request",
        ShellInput(Shell="/bin/zsh", Command="printf direct", Directory="/tmp"),
        base_directory,
        10,
    )
    step = advance_shell_once(current, None)
    assert step.operation is not None
    initial = shell_state(step.operation)
    assert initial.OutPath == ""
    assert initial.ErrPath == ""
    directory = os.path.join(base_directory, current.ID)
    for index in range(3):
        request = one_dispatch_data(step, PrimitiveDispatchType.IO_CREATE, IOCreateRequest)
        events: Channel[PrimitiveEvent] = Channel()
        create(request, events)
        event = await receive_only_primitive_event(events)
        assert step.operation is not None
        step = advance_shell_once(step.operation, event)
        assert step.operation is not None
        state = shell_state(step.operation)
        want_out = os.path.join(directory, SHELL_OUT_FILENAME) if index >= 1 else ""
        want_err = os.path.join(directory, SHELL_ERR_FILENAME) if index >= 2 else ""
        assert (state.OutPath, state.ErrPath) == (want_out, want_err), index

    request = one_dispatch_data(step, PrimitiveDispatchType.PROCESS_START, ProcessStartRequest)
    assert request.path == "/bin/zsh"
    assert request.arguments == ["-c", "printf direct"]
    assert request.directory == "/tmp"
    assert request.environment is None
    assert request.pipes == 0
    assert request.stdout_path == os.path.join(directory, SHELL_OUT_FILENAME)
    assert request.stderr_path == os.path.join(directory, SHELL_ERR_FILENAME)


def test_shell_actor_persists_signal_exit_status_before_reading(tmp_path: Path) -> None:
    base_directory = str(tmp_path)
    operation_id = "shell-signaled"
    directory = os.path.join(base_directory, operation_id)
    create_shell_artifacts(directory, b"", b"")
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base_directory,
            Phase=ShellPhase.PROCESS,
            ProcessGroupID=4321,
        ),
    )
    event = PrimitiveEvent(
        PrimitiveEventType.PROCESS_EXITED,
        SourceID(operation_id),
        CorrelationID("process"),
        ProcessExitResult(exit_code=-1, signal=signal.SIGTERM),
    )
    step = advance_shell_once(current, event)
    request = one_dispatch_data(step, PrimitiveDispatchType.IO_READ, IOReadRequest)
    assert request.path == os.path.join(directory, SHELL_OUT_FILENAME)
    assert step.operation is not None
    state = shell_state(step.operation)
    assert state.Phase == ShellPhase.READ_OUT
    assert state.ProcessGroupID == 0
    assert state.PendingExitCode == 143


def test_shell_actor_fails_unknown_process_outcome_without_restarting(tmp_path: Path) -> None:
    current = shell_operation_with_state(
        "shell-unknown",
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=str(tmp_path),
            Phase=ShellPhase.PROCESS,
        ),
    )
    step = advance_shell_once(current, None)
    assert step.operation is not None
    state = shell_state(step.operation)
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    assert "outcome is unknown" in state.TerminalError


def test_shell_actor_preserves_recorded_process_when_failing_recovery(tmp_path: Path) -> None:
    current = shell_operation_with_state(
        "shell-recover",
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=str(tmp_path),
            Phase=ShellPhase.PROCESS,
            ProcessGroupID=4321,
        ),
    )
    step = advance_shell_once(current, None)
    assert step.operation is not None
    state = shell_state(step.operation)
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    assert state.ProcessGroupID == 4321
    assert "interrupted before an exit status" in state.TerminalError


async def test_shell_actor_resumes_output_read_with_recorded_exit_status(tmp_path: Path) -> None:
    base_directory = str(tmp_path)
    operation_id = "shell-resume-exit"
    create_shell_artifacts(os.path.join(base_directory, operation_id), b"out", b"err")
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base_directory,
            Phase=ShellPhase.READ_OUT,
            PendingExitCode=5,
        ),
    )
    completed = await run_shell_actor(current)
    result = shell_state(completed).Result
    assert completed.Status == Status.COMPLETED
    assert result is not None
    assert (result.Out, result.Err, result.ExitCode) == ("out", "err", 5)


@pytest.mark.parametrize("output", [b"prefixMOREtail", b""])
async def test_shell_actor_rereads_and_replaces_inline(tmp_path: Path, output: bytes) -> None:
    base_directory = str(tmp_path)
    operation_id = "shell-resume-read"
    create_shell_artifacts(os.path.join(base_directory, operation_id), output, b"error")
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base_directory,
            Phase=ShellPhase.READ_OUT,
            PendingExitCode=4,
            InlineOut=b"stale",
            InlineOutTail=b"stale tail",
        ),
    )
    current = replace(current, MaxOutputLength=10)
    resumed = advance_shell_once(current, None)
    request = one_dispatch_data(resumed, PrimitiveDispatchType.IO_READ, IOReadRequest)
    assert (request.offset, request.count) == (0, 40)

    completed = await run_shell_actor(current)
    state = shell_state(completed)
    result = state.Result
    want = ""
    if output:
        want = f"prefi...4 bytes truncated; complete output in {state.OutPath}...Etail"
    assert completed.Status == Status.COMPLETED
    assert result is not None
    assert (result.Out, result.OutSize, result.Err, result.ExitCode) == (
        want,
        len(output),
        "error",
        4,
    )


@pytest.mark.parametrize(
    ("name", "state", "dispatch", "correlation", "artifact", "kind", "mode", "offset", "count"),
    [
        (
            "create directory",
            ShellState(Phase=ShellPhase.CREATE_DIRECTORY),
            PrimitiveDispatchType.IO_CREATE,
            "create_directory",
            "",
            IOCreateKind.DIRECTORY,
            0o700,
            0,
            0,
        ),
        (
            "create stdout",
            ShellState(Phase=ShellPhase.CREATE_OUT),
            PrimitiveDispatchType.IO_CREATE,
            "create_out",
            SHELL_OUT_FILENAME,
            IOCreateKind.REGULAR_FILE,
            0o600,
            0,
            0,
        ),
        (
            "create stderr",
            ShellState(Phase=ShellPhase.CREATE_ERR),
            PrimitiveDispatchType.IO_CREATE,
            "create_err",
            SHELL_ERR_FILENAME,
            IOCreateKind.REGULAR_FILE,
            0o600,
            0,
            0,
        ),
        (
            "read stdout",
            ShellState(Phase=ShellPhase.READ_OUT, PendingExitCode=5, InlineOut=b"ou"),
            PrimitiveDispatchType.IO_READ,
            "read_out",
            SHELL_OUT_FILENAME,
            0,
            0,
            0,
            40,
        ),
        (
            "read stderr",
            ShellState(Phase=ShellPhase.READ_ERR, PendingExitCode=5, InlineErr=b"err"),
            PrimitiveDispatchType.IO_READ,
            "read_err",
            SHELL_ERR_FILENAME,
            0,
            0,
            0,
            40,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and " " in value else "",
)
def test_shell_actor_resumes_every_replayable_phase(
    tmp_path: Path,
    name: str,
    state: ShellState,
    dispatch: PrimitiveDispatchType,
    correlation: str,
    artifact: str,
    kind: int,
    mode: int,
    offset: int,
    count: int,
) -> None:
    base_directory = str(tmp_path)
    operation_id = "resume-" + name.replace(" ", "-")
    state = replace(state, Input=ShellInput(Shell=TEST_SHELL_PATH), BaseDirectory=base_directory)
    current = replace(shell_operation_with_state(operation_id, state), MaxOutputLength=10)
    step = advance_shell_once(current, None)
    assert step.operation is not None
    assert step.operation.Status == Status.AWAITING
    assert len(step.dispatches) == 1
    assert step.dispatches[0].type == dispatch
    directory = os.path.join(base_directory, operation_id)
    path = os.path.join(directory, artifact) if artifact else directory
    if dispatch == PrimitiveDispatchType.IO_CREATE:
        create_request = dispatch_data(step.dispatches[0], IOCreateRequest)
        assert create_request.source == operation_id
        assert create_request.correlation_id == correlation
        assert (create_request.path, create_request.kind, create_request.mode) == (path, kind, mode)
    else:
        read_request = dispatch_data(step.dispatches[0], IOReadRequest)
        assert read_request.source == operation_id
        assert read_request.correlation_id == correlation
        assert (read_request.path, read_request.offset, read_request.count) == (
            path,
            offset,
            count,
        )


async def test_shell_actor_requires_recorded_exit_status(tmp_path: Path) -> None:
    base = str(tmp_path)
    operation_id = "missing-exit-status"
    create_shell_artifacts(os.path.join(base, operation_id), b"", b"")
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH), BaseDirectory=base, Phase=ShellPhase.READ_OUT
        ),
    )
    failed = await run_shell_actor(current)
    state = shell_state(failed)
    assert failed.Status == Status.FAILED
    assert "without an exit status" in state.TerminalError


async def test_shell_actor_fails_when_shell_cannot_start(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    current = new_shell_operation(
        "shell-start-failure",
        ShellInput(Shell=str(tmp_path_factory.mktemp("shell") / "missing-shell")),
        str(tmp_path_factory.mktemp("base")),
        10,
    )
    failed = await run_shell_actor(current)
    state = shell_state(failed)
    assert failed.Status == Status.FAILED
    assert state.TerminalError
    assert state.Result is None


@pytest.mark.parametrize(
    ("input", "base_directory", "message"),
    [
        (ShellInput(), "/tmp", "shell path must be absolute"),
        (ShellInput(Shell="sh"), "/tmp", "shell path must be absolute"),
        (ShellInput(Shell=TEST_SHELL_PATH), "relative", "base directory must be absolute"),
    ],
)
def test_new_shell_spec_rejects_invalid_configuration(
    input: ShellInput, base_directory: str, message: str
) -> None:
    with pytest.raises(OperationError, match=f"^{message}$"):
        new_shell_spec(input, base_directory, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        {"ProcessGroupID": -1},
        {"ProcessGroupID": 1},
        {"OutSize": -1},
        {"ErrSize": -1},
        {"InlineOut": b"xxxxx"},
        {"InlineErr": b"xxxxx"},
    ],
)
def test_shell_actor_validates_persisted_state(mutation: dict[str, object]) -> None:
    valid = ShellState(
        Input=ShellInput(Shell=TEST_SHELL_PATH), BaseDirectory="/tmp", Phase=ShellPhase.READ_OUT
    )
    current = shell_operation_with_state("shell-invalid", replace(valid, **mutation))
    with pytest.raises(OperationError):
        advance_shell_once(replace(current, MaxOutputLength=1), None)


def test_shell_actor_fails_identity_that_is_not_a_path_component() -> None:
    valid = ShellState(
        Input=ShellInput(Shell=TEST_SHELL_PATH), BaseDirectory="/tmp", Phase=ShellPhase.READ_OUT
    )
    step = advance_shell_once(shell_operation_with_state("../invalid", valid), None)
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED


def test_shell_actor_cancellation_is_terminal(tmp_path: Path) -> None:
    current = new_shell_operation(
        "shell-cancel", ShellInput(Shell=TEST_SHELL_PATH), str(tmp_path), 10
    )
    step = advance_shell_once(current, None)
    assert step.operation is not None
    canceled = advance_shell_once(replace(step.operation, Status=Status.CANCELING), None)
    assert canceled.operation is not None
    state = shell_state(canceled.operation)
    assert canceled.operation.Status == Status.CANCELED
    assert not canceled.dispatches
    assert state.TerminalError


def test_shell_actor_cancellation_preserves_recorded_process(tmp_path: Path) -> None:
    current = shell_operation_with_state(
        "shell-cancel-process",
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=str(tmp_path),
            Phase=ShellPhase.PROCESS,
            ProcessGroupID=4321,
        ),
    )
    canceled = advance_shell_once(replace(current, Status=Status.CANCELING), None)
    assert canceled.operation is not None
    state = shell_state(canceled.operation)
    assert canceled.operation.Status == Status.CANCELED
    assert not canceled.dispatches
    assert state.ProcessGroupID == 4321
    assert state.Phase == ""


async def test_shell_actor_cancellation_stops_active_process(tmp_path: Path) -> None:
    stop = asyncio.Event()
    current = new_shell_operation(
        "shell-cancel-active",
        ShellInput(Shell=TEST_SHELL_PATH, Command="exec sleep 30"),
        str(tmp_path),
        10,
    )
    process_group_id = 0

    def observe(update: Operation, event: PrimitiveEvent) -> None:
        nonlocal process_group_id
        if event.type != PrimitiveEventType.PROCESS_STARTED:
            return
        started = event.result
        assert isinstance(started, ProcessStartedResult)
        assert started.pid > 1
        process_group_id = started.pid
        assert shell_state(update).ProcessGroupID == process_group_id
        stop.set()

    canceled = await run_shell_actor(current, observe, stop)
    state = shell_state(canceled)
    assert canceled.Status == Status.CANCELED
    assert state.Phase == ""
    assert state.ProcessGroupID == process_group_id
    assert state.Result is None
    assert_process_group_gone(process_group_id)


@pytest.mark.parametrize(
    ("text", "want", "limit"),
    [
        pytest.param(
            b"HEAD" + b"x" * (1024 * 1024) + b"TAIL",
            "HEAD...1048575 bytes truncated...xTAIL",
            9,
            id="large ASCII",
        ),
        pytest.param(
            "🙂".encode() * 20, "🙂🙂...64 bytes truncated...🙂🙂", 4, id="four byte characters"
        ),
        pytest.param(
            "🙂".encode() * 20, "...76 bytes truncated...🙂", 1, id="one four byte character"
        ),
        pytest.param(
            "🙂".encode() * 20, "🙂...68 bytes truncated...🙂🙂", 3, id="odd Unicode budget"
        ),
        pytest.param(
            ("界é" + "🙂" * 20 + "好é界").encode(),
            "界é🙂...76 bytes truncated...好é界",
            6,
            id="split UTF8 boundaries",
        ),
        pytest.param(
            b"AB" + b"x" * 100 + b"\xff\xfeZ",
            "AB...100 bytes truncated...��Z",
            5,
            id="invalid UTF8",
        ),
        pytest.param(
            b'\n"' + b"x" * 100 + b"\x00\\\t",
            '\n"xxxx...93 bytes truncated...xxx\x00\\\t',
            12,
            id="control characters",
        ),
        pytest.param(
            b"start" + b"x" * 100 + b"end", "...107 bytes truncated...d", 1, id="one character"
        ),
    ],
)
async def test_shell_reads_actual_tail_with_bounded_state(
    tmp_path: Path, text: bytes, want: str, limit: int
) -> None:
    base = str(tmp_path)
    operation_id = "tail"
    create_shell_artifacts(os.path.join(base, operation_id), text, text)
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base,
            Phase=ShellPhase.READ_OUT,
            PendingExitCode=0,
        ),
    )
    current = replace(current, MaxOutputLength=limit)
    tail_reads = 0

    def observe(update: Operation, event: PrimitiveEvent) -> None:
        nonlocal tail_reads
        state = shell_state(update)
        assert len(state.InlineOut) + len(state.InlineOutTail) <= 4 * limit
        assert len(state.InlineErr) + len(state.InlineErrTail) <= 4 * limit
        if event.type == PrimitiveEventType.IO_READ_COMPLETED and event.correlation_id in (
            "read_out_tail",
            "read_err_tail",
        ):
            tail_reads += 1

    completed = await run_shell_actor(current, observe)
    state = shell_state(completed)
    assert completed.Status == Status.COMPLETED
    assert state.Result is not None
    want_out = want.replace(
        " bytes truncated...", f" bytes truncated; complete output in {state.OutPath}...", 1
    )
    want_err = want.replace(
        " bytes truncated...", f" bytes truncated; complete output in {state.ErrPath}...", 1
    )
    assert (state.Result.Out, state.Result.Err) == (want_out, want_err)
    assert state.OutTruncated
    assert state.ErrTruncated
    assert tail_reads == 2
    assert (state.Result.OutSize, state.Result.ErrSize) == (len(text), len(text))
    assert not state.InlineOut
    assert not state.InlineErr
    assert not state.InlineOutTail
    assert not state.InlineErrTail
    assert_file_contents(state.OutPath, text)
    assert_file_contents(state.ErrPath, text)


@pytest.mark.parametrize(
    ("out", "err", "want_out", "want_err", "tail_reads"),
    [
        pytest.param("", "", "", "", [], id="empty"),
        pytest.param(
            "abcdefghijklmnop",
            "🙂🙂🙂🙂",
            "ab...12 bytes truncated; complete output in {path}...op",
            "🙂🙂🙂🙂",
            [],
            id="byte budget boundary",
        ),
        pytest.param(
            "abcdefghijklmnopq",
            "err",
            "ab...13 bytes truncated; complete output in {path}...pq",
            "err",
            ["read_out_tail"],
            id="stdout only",
        ),
        pytest.param(
            "界é🙂",
            "abcdefghijklmnopq",
            "界é🙂",
            "ab...13 bytes truncated; complete output in {path}...pq",
            ["read_err_tail"],
            id="stderr only",
        ),
    ],
)
async def test_shell_reads_tail_only_for_streams_exceeding_the_byte_budget(
    tmp_path: Path, out: str, err: str, want_out: str, want_err: str, tail_reads: list[str]
) -> None:
    base = str(tmp_path)
    operation_id = "mixed-output"
    create_shell_artifacts(os.path.join(base, operation_id), out.encode(), err.encode())
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base,
            Phase=ShellPhase.READ_OUT,
            PendingExitCode=0,
        ),
    )
    current = replace(current, MaxOutputLength=4)
    observed: list[str] = []

    def observe(_: Operation, event: PrimitiveEvent) -> None:
        if event.type == PrimitiveEventType.IO_READ_COMPLETED and event.correlation_id in (
            "read_out_tail",
            "read_err_tail",
        ):
            observed.append(event.correlation_id)

    completed = await run_shell_actor(current, observe)
    state = shell_state(completed)
    assert completed.Status == Status.COMPLETED
    assert state.Result is not None
    assert state.Result.Out == want_out.replace("{path}", state.OutPath)
    assert state.Result.Err == want_err.replace("{path}", state.ErrPath)
    assert observed == tail_reads


@pytest.mark.parametrize("phase", [ShellPhase.READ_OUT_TAIL, ShellPhase.READ_ERR_TAIL])
async def test_shell_rereads_and_replaces_tail(tmp_path: Path, phase: ShellPhase) -> None:
    text = b"HEAD" + b"x" * 100 + b"TAIL!"
    base = str(tmp_path)
    operation_id = "resume-tail"
    create_shell_artifacts(os.path.join(base, operation_id), text, text)
    state = ShellState(
        Input=ShellInput(Shell=TEST_SHELL_PATH),
        BaseDirectory=base,
        Phase=phase,
        PendingExitCode=0,
        InlineOut=text[:20],
        OutSize=len(text),
        InlineOutTail=b"stale",
    )
    if phase == ShellPhase.READ_ERR_TAIL:
        state = replace(
            state,
            InlineOutTail=text[-20:],
            InlineErr=text[:20],
            ErrSize=len(text),
            InlineErrTail=b"stale",
        )
    current = replace(shell_operation_with_state(operation_id, state), MaxOutputLength=10)
    step = advance_shell_once(current, None)
    request = one_dispatch_data(step, PrimitiveDispatchType.IO_READ, IOReadRequest)
    assert (request.offset, request.count) == (len(text) - 20, 20)
    assert step.operation is not None
    completed = await run_shell_actor(step.operation)
    completed_state = shell_state(completed)
    result = completed_state.Result
    assert completed.Status == Status.COMPLETED
    assert result is not None
    assert result.Out == (
        f"HEADx...99 bytes truncated; complete output in {completed_state.OutPath}...TAIL!"
    )
    assert result.Err == (
        f"HEADx...99 bytes truncated; complete output in {completed_state.ErrPath}...TAIL!"
    )


@pytest.mark.parametrize(
    ("name", "event_type", "result"),
    [
        ("wrong offset", PrimitiveEventType.IO_READ_OUTPUT, IOReadOutputResult(79, b"x")),
        ("oversized chunk", PrimitiveEventType.IO_READ_OUTPUT, IOReadOutputResult(80, b"x" * 21)),
        ("incomplete tail", PrimitiveEventType.IO_READ_COMPLETED, IOReadCompletedResult(100)),
        ("changed size", PrimitiveEventType.IO_READ_COMPLETED, IOReadCompletedResult(101)),
    ],
)
def test_shell_rejects_invalid_tail_events(
    tmp_path: Path, name: str, event_type: PrimitiveEventType, result: object
) -> None:
    state = ShellState(
        Input=ShellInput(Shell=TEST_SHELL_PATH),
        BaseDirectory=str(tmp_path),
        Phase=ShellPhase.READ_OUT_TAIL,
        OutSize=100,
        InlineOut=b"h" * 20,
    )
    current = replace(shell_operation_with_state("invalid-tail", state), MaxOutputLength=10)
    shell = Shell(current)
    source, correlation = SourceID("invalid-tail"), CorrelationID("read_out_tail")
    if name == "changed size":
        shell.handle(
            PrimitiveEvent(
                PrimitiveEventType.IO_READ_OUTPUT,
                source,
                correlation,
                IOReadOutputResult(80, b"t" * 20),
            )
        )
    step = shell.handle(PrimitiveEvent(event_type, source, correlation, result))
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches


@pytest.mark.parametrize(
    "event",
    [
        None,
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            SourceID("shell-unknown-phase"),
            CorrelationID("unknown"),
            IOReadCompletedResult(0),
        ),
    ],
)
def test_shell_actor_fails_unknown_persisted_phase(
    tmp_path: Path, event: PrimitiveEvent | None
) -> None:
    state = encode(ShellState(Input=ShellInput(Shell=TEST_SHELL_PATH), BaseDirectory=str(tmp_path)))
    current = replace(
        shell_operation_with_state("shell-unknown-phase", ShellState()),
        State=Raw(state.replace(b'"Phase":""', b'"Phase":"unknown"')),
    )
    step = advance_shell_once(current, event)
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    failed = shell_state(step.operation)
    assert failed.TerminalError == 'shell operation has invalid phase "unknown"'
    assert failed.Phase == ""
