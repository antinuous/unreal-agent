# ruff: noqa: ASYNC240
import asyncio
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest
from unreal_agent.primitives import (
    PROCESS_OUTPUT_CHUNK_SIZE,
    TERMINAL_EVENT_TYPES,
    CorrelationID,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    ProcessCloseInputRequest,
    ProcessExitResult,
    ProcessInvocation,
    ProcessOutputResult,
    ProcessPipeSet,
    ProcessSignalRequest,
    ProcessStartedResult,
    ProcessStartRequest,
    ProcessStream,
    ProcessWriteFailureResult,
    ProcessWriteRequest,
    ProcessWriteResult,
    SourceID,
    read_file,
    start_process,
)
from unreal_agent.primitives.channel import Channel

type Events = Channel[PrimitiveEvent]


class Process:
    def __init__(self, request: ProcessStartRequest) -> None:
        self.events: Events = Channel(1)
        self.invocation: ProcessInvocation = start_process(request, self.events)

    def cancel(self) -> None:
        assert self.invocation.task is not None
        self.invocation.task.cancel()

    def write_input(self, request: ProcessWriteRequest) -> Events:
        events: Events = Channel(1)
        self.invocation.write_input(request, events)
        return events

    def close_input(self, request: ProcessCloseInputRequest) -> Events:
        events: Events = Channel(1)
        self.invocation.close_input(request, events)
        return events

    def signal(self, request: ProcessSignalRequest) -> Events:
        events: Events = Channel(1)
        self.invocation.signal(request, events)
        return events


def start(request: ProcessStartRequest) -> Process:
    return Process(request)


def start_with_all_pipes(request: ProcessStartRequest) -> Process:
    request.pipes = ProcessPipeSet.ALL
    return Process(request)


async def collect_events(events: Events, limit: float = 30.0) -> list[PrimitiveEvent]:
    collected: list[PrimitiveEvent] = []
    async with asyncio.timeout(limit):
        while True:
            event = await events.get()
            collected.append(event)
            if event.type in TERMINAL_EVENT_TYPES:
                return collected


async def collect_process_events(events: Events) -> list[PrimitiveEvent]:
    collected: list[PrimitiveEvent] = []
    async with asyncio.timeout(10):
        while True:
            event = await events.get()
            collected.append(event)
            if event.type in (
                PrimitiveEventType.PROCESS_EXITED,
                PrimitiveEventType.FAILED,
                PrimitiveEventType.CANCELED,
            ):
                return collected


async def receive_event(events: Events) -> PrimitiveEvent:
    async with asyncio.timeout(10):
        return await events.get()


def single_event(events: list[PrimitiveEvent], event_type: PrimitiveEventType) -> PrimitiveEvent:
    assert len(events) == 1, events
    assert events[0].type == event_type, events[0]
    return events[0]


def result[T](event: PrimitiveEvent, kind: type[T]) -> T:
    assert isinstance(event.result, kind), event.result
    return event.result


async def expect_started(process: Process) -> None:
    started = await receive_event(process.events)
    assert started.type == PrimitiveEventType.PROCESS_STARTED, started


def parse_process_pid(data: bytes) -> tuple[int, bool]:
    marker = data.decode(errors="replace")
    if not marker.endswith("\n"):
        return 0, False
    try:
        pid = int(marker.strip())
    except ValueError:
        return 0, False
    if pid <= 0:
        return 0, False
    return pid, True


async def wait_for_process_pid(path: Path) -> int:
    deadline = time.monotonic() + 10
    delay = 0.001
    while time.monotonic() < deadline:
        try:
            pid, ready = parse_process_pid(path.read_bytes())
            if ready:
                return pid
        except FileNotFoundError:
            pass
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.05)
    pytest.fail("timed out waiting for process PID")


async def wait_for_file_content(path: Path, want: str) -> None:
    deadline = time.monotonic() + 10
    delay = 0.001
    while time.monotonic() < deadline:
        try:
            if path.read_text() == want:
                return
        except FileNotFoundError:
            pass
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.05)
    pytest.fail(f"timed out waiting for {path} to contain {want!r}")


async def wait_for_process_gone(pid: int) -> None:
    deadline = time.monotonic() + 10
    delay = 0.001
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.05)
    pytest.fail("timed out waiting for process exit")


@contextmanager
def cleanup_process_pid(pid: int) -> Iterator[Callable[[], None]]:
    gone = False

    def mark_gone() -> None:
        nonlocal gone
        gone = True

    try:
        yield mark_gone
    finally:
        if not gone:
            for number in (signal.SIGCONT, signal.SIGKILL):
                with suppress(ProcessLookupError):
                    os.kill(pid, number)


def assert_identity(events: list[PrimitiveEvent], source: str, correlation_id: str) -> None:
    for event in events:
        assert (event.source, event.correlation_id) == (source, correlation_id), event


def assert_completed_output(
    events: list[PrimitiveEvent], source: str, correlation_id: str, want: str
) -> None:
    assert len(events) == 3, events
    assert_identity(events, source, correlation_id)
    assert [event.type for event in events] == [
        PrimitiveEventType.PROCESS_STARTED,
        PrimitiveEventType.PROCESS_OUTPUT,
        PrimitiveEventType.PROCESS_EXITED,
    ]
    assert result(events[1], ProcessOutputResult).data == want.encode()


def assert_event_types(events: list[PrimitiveEvent], want: dict[PrimitiveEventType, int]) -> None:
    got: dict[PrimitiveEventType, int] = {}
    for event in events:
        got[event.type] = got.get(event.type, 0) + 1
    assert got == want


def request(**kwargs: Any) -> ProcessStartRequest:
    kwargs.setdefault("source", SourceID("operation-1"))
    kwargs.setdefault("correlation_id", CorrelationID("process-1"))
    return ProcessStartRequest(**kwargs)


def write(correlation_id: str, data: bytes) -> ProcessWriteRequest:
    return ProcessWriteRequest(SourceID("operation-1"), CorrelationID(correlation_id), data)


def close(correlation_id: str) -> ProcessCloseInputRequest:
    return ProcessCloseInputRequest(SourceID("operation-1"), CorrelationID(correlation_id))


def send_signal(
    correlation_id: str, number: int, *, propagate: bool = False
) -> ProcessSignalRequest:
    return ProcessSignalRequest(
        SourceID("operation-1"), CorrelationID(correlation_id), number, propagate
    )


BLOCKED_WRITER = "head -c 1 >/dev/null; printf '%d\\n' $$ > \"$1\"; exec sleep 30"


async def test_process_captures_output_into_paths(tmp_path: Path) -> None:
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    for path in (stdout_path, stderr_path):
        path.write_bytes(b"stale")
    events = await collect_events(
        start(
            request(
                path="/bin/sh",
                arguments=["-c", "printf captured-out; printf captured-err >&2; exit 7"],
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        ).events
    )
    assert [event.type for event in events] == [
        PrimitiveEventType.PROCESS_STARTED,
        PrimitiveEventType.PROCESS_EXITED,
    ]
    assert result(events[1], ProcessExitResult) == ProcessExitResult(7, 0)
    assert stdout_path.read_bytes() == b"captured-out"
    assert stderr_path.read_bytes() == b"captured-err"


async def test_process_cancellation_completes_captured_output(tmp_path: Path) -> None:
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    for path in (stdout_path, stderr_path):
        path.write_bytes(b"")
    process = start(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "trap 'printf final; printf error >&2; exit 0' TERM; printf ready; "
                "while :; do :; done",
            ],
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
    )
    await expect_started(process)
    await wait_for_file_content(stdout_path, "ready")
    process.cancel()
    single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
    assert stdout_path.read_text() == "readyfinal"
    assert stderr_path.read_text() == "error"


async def test_process_rejects_invalid_capture_configuration(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    path.write_bytes(b"")
    for invalid in (
        ProcessStartRequest(path="/bin/true", stdout_path="relative"),
        ProcessStartRequest(path="/bin/true", stderr_path="relative"),
        ProcessStartRequest(path="/bin/true", pipes=ProcessPipeSet.STDOUT, stdout_path=str(path)),
        ProcessStartRequest(path="/bin/true", pipes=ProcessPipeSet.STDERR, stderr_path=str(path)),
        ProcessStartRequest(path="/bin/true", stdout_path=str(path), stderr_path=str(path)),
    ):
        event = single_event(await collect_events(start(invalid).events), PrimitiveEventType.FAILED)
        assert result(event, PrimitiveFailureResult).error


@pytest.mark.parametrize("stderr_path", ["/{path}", "{dir}//./capture", "{dir}/x/../capture"])
async def test_process_compares_capture_paths_after_go_clean(
    tmp_path: Path, stderr_path: str
) -> None:
    path = tmp_path / "capture"
    request = ProcessStartRequest(
        path="/bin/true",
        stdout_path=str(path),
        stderr_path=stderr_path.format(path=path, dir=tmp_path),
    )
    event = single_event(await collect_events(start(request).events), PrimitiveEventType.FAILED)
    assert result(event, PrimitiveFailureResult).error == (
        "start process: stdout and stderr capture paths must differ"
    )


async def test_process_rejects_aliased_capture_paths_before_truncating(tmp_path: Path) -> None:
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    stdout_path.write_bytes(b"preserved")
    os.link(stdout_path, stderr_path)
    event = single_event(
        await collect_events(
            start(
                ProcessStartRequest(
                    path="/bin/true", stdout_path=str(stdout_path), stderr_path=str(stderr_path)
                )
            ).events
        ),
        PrimitiveEventType.FAILED,
    )
    assert "same file" in result(event, PrimitiveFailureResult).error
    assert stdout_path.read_bytes() == b"preserved"


async def test_process_rejects_invalid_capture_path(tmp_path: Path) -> None:
    events = await collect_events(
        start(ProcessStartRequest(path="/bin/true", stdout_path=str(tmp_path))).events
    )
    assert result(single_event(events, PrimitiveEventType.FAILED), PrimitiveFailureResult).error


async def test_process_streams_output_and_exit() -> None:
    stdout = "stdout-" * (PROCESS_OUTPUT_CHUNK_SIZE // 4)
    stderr = "stderr-" * (PROCESS_OUTPUT_CHUNK_SIZE // 4)
    events = await collect_events(
        start_with_all_pipes(
            request(
                path="/bin/sh",
                arguments=[
                    "-c",
                    "printf '%s' \"$PROCESS_STDOUT\"; printf '%s' \"$PROCESS_STDERR\" >&2; exit 7",
                ],
                environment=[f"PROCESS_STDOUT={stdout}", f"PROCESS_STDERR={stderr}"],
            )
        ).events
    )
    assert len(events) >= 4, events
    assert_identity(events, "operation-1", "process-1")
    assert events[0].type == PrimitiveEventType.PROCESS_STARTED
    assert result(events[0], ProcessStartedResult).pid > 0
    output = {ProcessStream.STDOUT: bytearray(), ProcessStream.STDERR: bytearray()}
    for event in events[1:-1]:
        assert event.type == PrimitiveEventType.PROCESS_OUTPUT, event
        chunk = result(event, ProcessOutputResult)
        assert len(chunk.data) <= PROCESS_OUTPUT_CHUNK_SIZE
        assert chunk.offset == len(output[chunk.stream])
        output[chunk.stream] += chunk.data
    assert output[ProcessStream.STDOUT] == stdout.encode()
    assert output[ProcessStream.STDERR] == stderr.encode()
    assert events[-1].type == PrimitiveEventType.PROCESS_EXITED
    assert result(events[-1], ProcessExitResult) == ProcessExitResult(7, 0)


async def test_process_input() -> None:
    process = start_with_all_pipes(request(path="/bin/cat"))
    await expect_started(process)
    written = single_event(
        await collect_events(process.write_input(write("write-1", b"input data"))),
        PrimitiveEventType.PROCESS_INPUT_WRITTEN,
    )
    assert result(written, ProcessWriteResult).count == len(b"input data")
    closed = single_event(
        await collect_events(process.close_input(close("close-1"))),
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
    )
    assert closed.result is None

    rest = await collect_events(process.events)
    assert_event_types(
        rest, {PrimitiveEventType.PROCESS_OUTPUT: 1, PrimitiveEventType.PROCESS_EXITED: 1}
    )
    assert rest[-1].type == PrimitiveEventType.PROCESS_EXITED
    for event in rest:
        if event.type == PrimitiveEventType.PROCESS_OUTPUT:
            assert result(event, ProcessOutputResult) == ProcessOutputResult(
                ProcessStream.STDOUT, 0, b"input data"
            )


async def test_file_read_composes_with_process_input(tmp_path: Path) -> None:
    data = b"composed input\n" * (8 * 1024)
    path = tmp_path / "input"
    path.write_bytes(data)
    process = start_with_all_pipes(request(path="/bin/cat"))
    process_events = asyncio.create_task(collect_events(process.events))

    read_events: Events = Channel()
    read_file(
        IOReadRequest(SourceID("operation-1"), CorrelationID("read-1"), str(path), 0, len(data)),
        read_events,
    )
    writes = 0
    while True:
        event = await receive_event(read_events)
        if event.type == PrimitiveEventType.IO_READ_OUTPUT:
            chunk = result(event, IOReadOutputResult)
            written = single_event(
                await collect_events(process.write_input(write(f"write-{writes}", chunk.data))),
                PrimitiveEventType.PROCESS_INPUT_WRITTEN,
            )
            assert result(written, ProcessWriteResult).count == len(chunk.data)
            writes += 1
        elif event.type == PrimitiveEventType.IO_READ_COMPLETED:
            assert result(event, IOReadCompletedResult).size == len(data)
            break
        else:
            pytest.fail(f"read event = {event}")
    assert writes >= 2
    single_event(
        await collect_events(process.close_input(close("close-1"))),
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
    )
    stdout = bytearray()
    exited = False
    for event in await process_events:
        if event.type == PrimitiveEventType.PROCESS_OUTPUT:
            output = result(event, ProcessOutputResult)
            if output.stream == ProcessStream.STDOUT:
                stdout += output.data
        elif event.type == PrimitiveEventType.PROCESS_EXITED:
            exited = result(event, ProcessExitResult).exit_code == 0
    assert exited
    assert stdout == data


async def test_process_signals_remain_responsive_during_blocked_input_write(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "writer-started"
    process = start_with_all_pipes(
        request(path="/bin/sh", arguments=["-c", BLOCKED_WRITER, "sh", str(marker)])
    )
    await expect_started(process)
    write_events = process.write_input(write("write-1", bytes(1024 * 1024)))
    await wait_for_process_pid(marker)
    signaled = single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.PROCESS_SIGNALED,
    )
    assert signaled.correlation_id == "signal-1"
    rest = await collect_process_events(process.events)
    assert rest[-1].type == PrimitiveEventType.PROCESS_EXITED, rest[-1]
    assert result(rest[-1], ProcessExitResult).signal == signal.SIGTERM
    failed = single_event(
        await collect_events(write_events), PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED
    )
    failure = result(failed, ProcessWriteFailureResult)
    assert failure.count > 0
    assert failure.error


async def test_process_close_interrupts_blocked_input_write(tmp_path: Path) -> None:
    marker = tmp_path / "writer-started"
    process = start_with_all_pipes(
        request(path="/bin/sh", arguments=["-c", BLOCKED_WRITER, "sh", str(marker)])
    )
    await expect_started(process)
    write_events = process.write_input(write("write-1", bytes(1024 * 1024)))
    await wait_for_process_pid(marker)
    single_event(
        await collect_events(process.close_input(close("close-1"))),
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
    )
    failed = single_event(
        await collect_events(write_events), PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED
    )
    failure = result(failed, ProcessWriteFailureResult)
    assert failure.count > 0
    assert failure.error
    single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.PROCESS_SIGNALED,
    )
    rest = await collect_process_events(process.events)
    assert rest[-1].type == PrimitiveEventType.PROCESS_EXITED


async def test_process_close_input_ends_input() -> None:
    process = start_with_all_pipes(request(path="/bin/cat"))
    await expect_started(process)
    closed = single_event(
        await collect_events(process.close_input(close("close-1"))),
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
    )
    assert closed.correlation_id == "close-1"
    rest = await collect_events(process.events)
    exited = single_event(rest, PrimitiveEventType.PROCESS_EXITED)
    assert result(exited, ProcessExitResult).exit_code == 0


async def test_process_input_remains_closed() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    for correlation_id in ("close-1", "close-2"):
        event = single_event(
            await collect_events(process.close_input(close(correlation_id))),
            PrimitiveEventType.PROCESS_INPUT_CLOSED,
        )
        assert (event.source, event.correlation_id) == ("operation-1", correlation_id)
    event = single_event(
        await collect_events(process.write_input(write("write-1", b"input after close"))),
        PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED,
    )
    assert (event.source, event.correlation_id) == ("operation-1", "write-1")
    failure = result(event, ProcessWriteFailureResult)
    assert failure.count == 0
    assert failure.error
    process.cancel()
    await collect_events(process.events)


async def test_process_signal() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.PROCESS_SIGNALED,
    )
    exited = single_event(await collect_events(process.events), PrimitiveEventType.PROCESS_EXITED)
    assert result(exited, ProcessExitResult) == ProcessExitResult(-1, signal.SIGTERM)


async def test_process_signal_propagates_to_children(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=["-c", "sleep 30 & printf '%d\\n' $! > \"$1\"; wait", "sh", str(marker)],
        )
    )
    await expect_started(process)
    descendant = await wait_for_process_pid(marker)
    with cleanup_process_pid(descendant) as descendant_gone:
        single_event(
            await collect_events(
                process.signal(send_signal("signal-1", signal.SIGTERM, propagate=True))
            ),
            PrimitiveEventType.PROCESS_SIGNALED,
        )
        single_event(await collect_events(process.events), PrimitiveEventType.PROCESS_EXITED)
        await wait_for_process_gone(descendant)
        descendant_gone()


async def test_process_signal_does_not_propagate_by_default(tmp_path: Path) -> None:
    pid_marker, alive_marker = tmp_path / "pid", tmp_path / "alive"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                'trap \':\' TERM; /bin/sh -c "$3" sh "$1" "$2" & while :; do wait; done',
                "sh",
                str(pid_marker),
                str(alive_marker),
                'trap \'printf "alive\\n" > "$2"; exit 0\' USR1; '
                "printf '%d\\n' $$ > \"$1\"; while :; do :; done",
            ],
        )
    )
    await expect_started(process)
    descendant = await wait_for_process_pid(pid_marker)
    with cleanup_process_pid(descendant) as descendant_gone:
        single_event(
            await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
            PrimitiveEventType.PROCESS_SIGNALED,
        )
        os.kill(descendant, signal.SIGUSR1)
        await wait_for_file_content(alive_marker, "alive\n")
        await wait_for_process_gone(descendant)
        descendant_gone()
    process.cancel()
    single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)


async def test_cancel_process() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    process.cancel()
    canceled = single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
    assert (canceled.source, canceled.correlation_id, canceled.result) == (
        "operation-1",
        "process-1",
        None,
    )


async def test_completed_process_is_not_canceled_after_cancellation_stops(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "pid"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "printf '%d\\n' $$ > \"$1\"; printf output; exit 7",
                "sh",
                str(marker),
            ],
        )
    )
    collected = asyncio.create_task(collect_events(process.events))
    pid = await wait_for_process_pid(marker)
    await wait_for_process_gone(pid)
    process.cancel()
    events = await collected
    assert len(events) >= 2, events
    assert events[0].type == PrimitiveEventType.PROCESS_STARTED
    assert events[-1].type == PrimitiveEventType.PROCESS_EXITED
    assert result(events[-1], ProcessExitResult).exit_code == 7


async def test_cancel_process_allows_graceful_shutdown(tmp_path: Path) -> None:
    ready, terminated = tmp_path / "ready", tmp_path / "terminated"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "trap '/bin/dd if=/dev/zero bs=1024 count=96 2>/dev/null; "
                'printf "terminated\\n" > "$2"; exit 0\' TERM; '
                "printf '%d\\n' $$ > \"$1\"; while :; do :; done",
                "sh",
                str(ready),
                str(terminated),
            ],
        )
    )
    await expect_started(process)
    await wait_for_process_pid(ready)
    process.cancel()
    canceled = single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
    assert canceled.result is None
    assert terminated.read_text() == "terminated\n"


async def test_cancel_during_blocked_start_event_drains_shutdown_output(tmp_path: Path) -> None:
    ready, terminated = tmp_path / "ready", tmp_path / "terminated"
    events: Events = Channel(1)
    await events.put(PrimitiveEvent(PrimitiveEventType.PROCESS_OUTPUT))
    invocation = start_process(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "trap '/bin/dd if=/dev/zero bs=1024 count=96 2>/dev/null; "
                'printf "terminated\\n" > "$2"; exit 0\' TERM; '
                "printf '%d\\n' $$ > \"$1\"; while :; do :; done",
                "sh",
                str(ready),
                str(terminated),
            ],
            pipes=ProcessPipeSet.ALL,
        ),
        events,
    )
    await wait_for_process_pid(ready)
    assert invocation.task is not None
    invocation.task.cancel()
    started = time.monotonic()
    assert (await events.get()).type == PrimitiveEventType.PROCESS_OUTPUT
    canceled = single_event(await collect_events(events), PrimitiveEventType.CANCELED)
    assert canceled.result is None
    assert time.monotonic() - started < 4
    assert terminated.read_text() == "terminated\n"


async def test_blocked_input_write_defers_cancellation_until_process_stops(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "writer-started"
    process = start_with_all_pipes(
        request(path="/bin/sh", arguments=["-c", BLOCKED_WRITER, "sh", str(marker)])
    )
    await expect_started(process)
    write_events: Events = Channel(1)
    write_task = process.invocation.write_input(write("write-1", bytes(1024 * 1024)), write_events)
    await wait_for_process_pid(marker)
    write_task.cancel()
    process.cancel()
    canceled = single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
    assert canceled.result is None
    failed = single_event(
        await collect_events(write_events), PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED
    )
    failure = result(failed, ProcessWriteFailureResult)
    assert failure.count > 0
    assert failure.error


async def test_process_deadline_while_output_is_backpressured() -> None:
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=["-c", "/bin/dd if=/dev/zero bs=1024 count=96 2>/dev/null; sleep 30"],
        )
    )
    await expect_started(process)
    await asyncio.sleep(2)
    process.cancel()
    rest = await collect_events(process.events)
    assert rest[-1].type == PrimitiveEventType.CANCELED, rest[-1]
    assert rest[-1].result is None


async def test_cancel_process_terminates_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=["-c", "sleep 30 & printf '%d\\n' $! > \"$1\"; wait", "sh", str(marker)],
        )
    )
    await expect_started(process)
    descendant = await wait_for_process_pid(marker)
    with cleanup_process_pid(descendant) as descendant_gone:
        process.cancel()
        canceled = single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
        assert canceled.result is None
        await wait_for_process_gone(descendant)
        descendant_gone()


async def test_processes_from_same_source_are_independent() -> None:
    first = start_with_all_pipes(
        request(correlation_id="process-1", path="/bin/sh", arguments=["-c", "printf first"])
    )
    second = start_with_all_pipes(
        request(correlation_id="process-2", path="/bin/sh", arguments=["-c", "printf second"])
    )
    assert_completed_output(await collect_events(first.events), "operation-1", "process-1", "first")
    assert_completed_output(
        await collect_events(second.events), "operation-1", "process-2", "second"
    )


async def test_process_exit_terminates_descendants_without_waiting_for_inherited_pipes() -> None:
    process = start_with_all_pipes(
        request(path="/bin/sh", arguments=["-c", "sleep 30 & printf 'output:%d' \"$!\""])
    )
    collected: list[PrimitiveEvent] = []
    output = bytearray()
    descendant = 0
    while True:
        event = await receive_event(process.events)
        collected.append(event)
        if event.type == PrimitiveEventType.PROCESS_OUTPUT:
            output += result(event, ProcessOutputResult).data
            with suppress(ValueError):
                descendant = int(output.decode().removeprefix("output:"))
        if event.type == PrimitiveEventType.PROCESS_EXITED:
            break
    assert len(collected) == 3, collected
    assert output.startswith(b"output:")
    assert process.events.empty()
    assert descendant > 0
    with cleanup_process_pid(descendant) as descendant_gone:
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
        descendant_gone()


async def test_process_drains_buffered_output_after_exit(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "/bin/dd if=/dev/zero bs=1024 count=96 2>/dev/null; "
                "printf '%d\\n' $$ > \"$1\"; read -r line",
                "sh",
                str(marker),
            ],
        )
    )
    await expect_started(process)
    pid, output_bytes = await wait_for_process_pid_and_output(marker, process.events)
    with cleanup_process_pid(pid) as process_gone:
        single_event(
            await collect_events(process.write_input(write("release-1", b"\n"))),
            PrimitiveEventType.PROCESS_INPUT_WRITTEN,
        )
        await wait_for_process_gone(pid)
        process_gone()
    rest = await collect_events(process.events)
    for event in rest:
        if event.type == PrimitiveEventType.PROCESS_OUTPUT:
            output_bytes += len(result(event, ProcessOutputResult).data)
    assert output_bytes == 96 * 1024
    assert rest[-1].type == PrimitiveEventType.PROCESS_EXITED


async def wait_for_process_pid_and_output(path: Path, events: Events) -> tuple[int, int]:
    deadline = time.monotonic() + 10
    delay = 0.001
    output_bytes = 0
    while time.monotonic() < deadline:
        try:
            pid, ready = parse_process_pid(path.read_bytes())
            if ready:
                return pid, output_bytes
        except FileNotFoundError:
            pass
        try:
            async with asyncio.timeout(delay):
                event = await events.get()
        except TimeoutError:
            pass
        else:
            assert event.type == PrimitiveEventType.PROCESS_OUTPUT, event
            output_bytes += len(result(event, ProcessOutputResult).data)
        delay = min(delay * 2, 0.05)
    pytest.fail("timed out waiting for process PID")


async def test_process_exit_interrupts_pending_input_write(tmp_path: Path) -> None:
    marker, release, write_started = tmp_path / "pid", tmp_path / "release", tmp_path / "started"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=[
                "-c",
                "exec 3<&0; (head -c 1 >/dev/null; printf '%d\\n' $$ > \"$3\"; sleep 30) <&3 & "
                'printf \'%d\\n\' $! > "$1"; while [ ! -e "$2" ]; do sleep 0.01; done',
                "sh",
                str(marker),
                str(release),
                str(write_started),
            ],
        )
    )
    await expect_started(process)
    descendant = await wait_for_process_pid(marker)
    with cleanup_process_pid(descendant):
        write_events = process.write_input(write("write-1", bytes(1024 * 1024)))
        await wait_for_process_pid(write_started)
        release.write_bytes(b"")
        rest = await collect_process_events(process.events)
        failed = single_event(
            await collect_events(write_events), PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED
        )
        failure = result(failed, ProcessWriteFailureResult)
        assert failure.count > 0
        assert failure.error
        assert rest[-1].type == PrimitiveEventType.PROCESS_EXITED


async def test_process_uses_directory_and_environment(tmp_path: Path) -> None:
    canonical = os.path.realpath(tmp_path)
    events = await collect_events(
        start_with_all_pipes(
            request(
                path="/bin/sh",
                arguments=["-c", 'printf \'%s:%s:%s\' "$PWD" "$VALUE" "$1"', "sh", "argument"],
                directory=str(tmp_path),
                environment=["VALUE=environment"],
            )
        ).events
    )
    assert len(events) == 3, events
    assert (
        result(events[1], ProcessOutputResult).data == f"{canonical}:environment:argument".encode()
    )


@pytest.mark.parametrize(
    ("environment", "want"),
    [
        pytest.param(None, "inherited:parent", id="nil inherits"),
        pytest.param([], ":", id="empty"),
        pytest.param(["HARNESS_TEST_PROCESS_VALUE=explicit"], "explicit:", id="explicit replaces"),
    ],
)
async def test_process_environment_inheritance(
    monkeypatch: pytest.MonkeyPatch, environment: list[str] | None, want: str
) -> None:
    monkeypatch.setenv("HARNESS_TEST_PROCESS_VALUE", "inherited")
    monkeypatch.setenv("HARNESS_TEST_PROCESS_PARENT_ONLY", "parent")
    events = await collect_events(
        start_with_all_pipes(
            request(
                path="/bin/sh",
                arguments=[
                    "-c",
                    "printf '%s:%s' \"$HARNESS_TEST_PROCESS_VALUE\" "
                    '"$HARNESS_TEST_PROCESS_PARENT_ONLY"',
                ],
                environment=environment,
            )
        ).events
    )
    assert [event.type for event in events] == [
        PrimitiveEventType.PROCESS_STARTED,
        PrimitiveEventType.PROCESS_OUTPUT,
        PrimitiveEventType.PROCESS_EXITED,
    ]
    assert result(events[1], ProcessOutputResult).data == want.encode()
    assert result(events[2], ProcessExitResult) == ProcessExitResult(0, 0)


async def test_process_passes_environment_entries_like_go() -> None:
    events = await collect_events(
        start_with_all_pipes(
            request(
                path="/usr/bin/env",
                environment=["FOO", "A=1", "=x=y", "", "A=2", "BAR", "BAR"],
            )
        ).events
    )
    assert [event.type for event in events] == [
        PrimitiveEventType.PROCESS_STARTED,
        PrimitiveEventType.PROCESS_OUTPUT,
        PrimitiveEventType.PROCESS_EXITED,
    ]
    assert result(events[1], ProcessOutputResult).data == b"FOO\n=x=y\nA=2\nBAR\nBAR\n"


async def test_process_selects_parent_pipes() -> None:
    events = await collect_events(
        start(
            request(
                path="/bin/sh",
                arguments=[
                    "-c",
                    "if read -r value; then printf 'stdin:%s' \"$value\"; "
                    "else printf stdout; fi; printf stderr >&2",
                ],
                pipes=ProcessPipeSet.STDOUT,
            )
        ).events
    )
    assert [event.type for event in events] == [
        PrimitiveEventType.PROCESS_STARTED,
        PrimitiveEventType.PROCESS_OUTPUT,
        PrimitiveEventType.PROCESS_EXITED,
    ]
    output = result(events[1], ProcessOutputResult)
    assert (output.stream, output.data) == (ProcessStream.STDOUT, b"stdout")


async def test_process_without_stdin_pipe_closes_input_idempotently() -> None:
    process = start(
        request(
            path="/bin/sleep",
            arguments=["30"],
            pipes=ProcessPipeSet.STDOUT | ProcessPipeSet.STDERR,
        )
    )
    await expect_started(process)
    closed = single_event(
        await collect_events(process.close_input(close("close-1"))),
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
    )
    assert closed.result is None
    process.cancel()
    single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)


async def test_process_rejects_invalid_pipe_selection() -> None:
    event = single_event(
        await collect_events(start(request(path="/bin/true", pipes=1 << 7)).events),
        PrimitiveEventType.FAILED,
    )
    assert "pipe selection" in result(event, PrimitiveFailureResult).error


async def test_process_requires_absolute_executable_path(tmp_path: Path) -> None:
    executable = tmp_path / "requested-command"
    executable.write_text("#!/bin/sh\nprintf unexpected")
    executable.chmod(0o700)
    event = single_event(
        await collect_events(
            start_with_all_pipes(
                request(path="requested-command", environment=[f"PATH={tmp_path}"])
            ).events
        ),
        PrimitiveEventType.FAILED,
    )
    assert "path must be absolute" in result(event, PrimitiveFailureResult).error


@pytest.mark.parametrize(
    ("start_request", "want"),
    [
        pytest.param(request(), "path must be set", id="empty path"),
        pytest.param(
            request(path="/missing/process/executable"), "start process", id="missing executable"
        ),
    ],
)
async def test_process_start_failures(start_request: ProcessStartRequest, want: str) -> None:
    event = single_event(
        await collect_events(start_with_all_pipes(start_request).events),
        PrimitiveEventType.FAILED,
    )
    assert want in result(event, PrimitiveFailureResult).error


async def test_start_process_with_canceled_context() -> None:
    process = start_with_all_pipes(request(path="/bin/sh"))
    process.cancel()
    event = single_event(await collect_events(process.events), PrimitiveEventType.CANCELED)
    assert event.result is None
    control = single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.FAILED,
    )
    assert "context canceled" in result(control, PrimitiveFailureResult).error


async def test_process_control_fails_after_start_failure() -> None:
    process = start_with_all_pipes(request(path="/missing/process/executable"))
    event = single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.FAILED,
    )
    assert "start process" in result(event, PrimitiveFailureResult).error
    await collect_events(process.events)


async def test_process_controls_fail_after_exit_with_unread_events(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    process = start_with_all_pipes(
        request(
            path="/bin/sh",
            arguments=["-c", "printf '%d\\n' $$ > \"$1\"; printf output", "sh", str(marker)],
        )
    )
    pid = await wait_for_process_pid(marker)
    await wait_for_process_gone(pid)

    controls = [
        (
            "write-1",
            PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED,
            process.write_input(write("write-1", b"input")),
        ),
        ("close-1", PrimitiveEventType.FAILED, process.close_input(close("close-1"))),
        (
            "signal-1",
            PrimitiveEventType.FAILED,
            process.signal(send_signal("signal-1", signal.SIGTERM)),
        ),
        (
            "signal-group-1",
            PrimitiveEventType.FAILED,
            process.signal(send_signal("signal-group-1", signal.SIGTERM, propagate=True)),
        ),
    ]
    for correlation_id, event_type, events in controls:
        event = single_event(await collect_events(events), event_type)
        assert (event.source, event.correlation_id) == ("operation-1", correlation_id)
        if event.type == PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED:
            failure = result(event, ProcessWriteFailureResult).error
        else:
            failure = result(event, PrimitiveFailureResult).error
        assert "process invocation is done" in failure
    assert_completed_output(
        await collect_events(process.events), "operation-1", "process-1", "output"
    )


async def test_process_signal_failure() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    event = single_event(
        await collect_events(process.signal(send_signal("signal-1", -1, propagate=True))),
        PrimitiveEventType.FAILED,
    )
    assert "invalid argument" in result(event, PrimitiveFailureResult).error
    process.cancel()
    await collect_events(process.events)


async def test_process_control_honors_its_context() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    controls: list[tuple[str, Events]] = []
    for correlation_id, launch in (
        ("write-1", lambda e: process.invocation.write_input(write("write-1", b"input"), e)),
        ("close-1", lambda e: process.invocation.close_input(close("close-1"), e)),
        (
            "signal-1",
            lambda e: process.invocation.signal(send_signal("signal-1", signal.SIGTERM), e),
        ),
    ):
        events: Events = Channel(1)
        launch(events).cancel()
        controls.append((correlation_id, events))
    for correlation_id, events in controls:
        event = single_event(await collect_events(events), PrimitiveEventType.CANCELED)
        assert (event.source, event.correlation_id) == ("operation-1", correlation_id)
    process.cancel()
    await collect_events(process.events)


async def test_process_control_fails_after_process_cancellation() -> None:
    process = start_with_all_pipes(request(path="/bin/sh", arguments=["-c", "exec sleep 30"]))
    await expect_started(process)
    process.cancel()
    await collect_events(process.events)
    event = single_event(
        await collect_events(process.signal(send_signal("signal-1", signal.SIGTERM))),
        PrimitiveEventType.FAILED,
    )
    assert "process invocation is done" in result(event, PrimitiveFailureResult).error
