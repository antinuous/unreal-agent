# ruff: noqa: ASYNC220
import asyncio
import errno
import os
import resource
import signal
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from pathlib import Path

import pytest
from unreal_agent.primitives import process as process_module
from unreal_agent.primitives._errors import PrimitiveError, has_errno
from unreal_agent.primitives._fileio import File
from unreal_agent.primitives.channel import Channel
from unreal_agent.primitives.primitive_event import (
    CorrelationID,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
    send_event,
    send_terminal,
)
from unreal_agent.primitives.process import (
    PROCESS_OUTPUT_CHUNK_SIZE,
    PipeReader,
    ProcessExitResult,
    ProcessInvocation,
    ProcessOutputResult,
    ProcessOutputState,
    ProcessParentPipes,
    ProcessPipeSet,
    ProcessSignalRequest,
    ProcessStartRequest,
    ProcessStream,
    ProcessStreamFailureResult,
    WaitOutcome,
    await_process_completion,
    close_process_file,
    drain_process_output,
    prepare_process,
    process_completion_event,
    process_group_exists,
    process_result,
    run_process_output,
    signal_process_invocation,
    spawn_process,
    stream_process_output,
    sync_process_capture,
    terminate_process,
    wait_for_process_invocation,
)

type Events = Channel[PrimitiveEvent]

REQUEST = ProcessStartRequest(SourceID("operation-1"), CorrelationID("process-1"))


async def with_cancel_requested[T](function: Callable[[], Awaitable[T]]) -> T:
    """Run `function` in a task whose cancellation is already requested."""
    box: list[T] = []

    async def body() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        box.append(await function())

    task = asyncio.create_task(body())
    with suppress(asyncio.CancelledError):
        await task
    assert box, "function did not return"
    return box[0]


@pytest.fixture
def pipe() -> Iterator[tuple[File, File]]:
    read, write = os.pipe()
    reader, writer = File(read, "|0"), File(write, "|1")
    yield reader, writer
    for file in (reader, writer):
        close_process_file(file)


def assert_channel_empty(events: Events) -> None:
    assert events.empty()


async def test_drain_process_output(pipe: tuple[File, File]) -> None:
    reader, writer = pipe
    data = b"buffered process output"
    assert writer.write(data) == len(data)
    events: Events = Channel(1)
    offset = await drain_process_output(REQUEST, ProcessStream.STDERR, reader, 17, events)
    event = await events.get()
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.PROCESS_OUTPUT,
        "operation-1",
        "process-1",
    )
    assert event.result == ProcessOutputResult(ProcessStream.STDERR, 17, data)
    assert offset == 17 + len(data)


async def test_drain_process_output_has_no_byte_limit(tmp_path: Path) -> None:
    data = bytes(2 * 1024 * 1024 + 1)
    path = tmp_path / "output"
    path.write_bytes(data)
    reader = File.open(str(path), os.O_RDONLY)
    try:
        events: asyncio.Queue[PrimitiveEvent] = asyncio.Queue()
        offset = await drain_process_output(
            ProcessStartRequest(), ProcessStream.STDOUT, reader, 0, events
        )
        assert offset == len(data)
        assert events.qsize() == len(data) // PROCESS_OUTPUT_CHUNK_SIZE + 1
    finally:
        close_process_file(reader)


async def test_drain_process_output_stops_when_pipe_has_no_buffered_data(
    pipe: tuple[File, File],
) -> None:
    reader, _ = pipe
    offset = await drain_process_output(
        ProcessStartRequest(), ProcessStream.STDOUT, reader, 0, Channel(1)
    )
    assert offset == 0


async def test_drain_process_output_stops_at_eof(pipe: tuple[File, File]) -> None:
    reader, writer = pipe
    writer.close()
    offset = await drain_process_output(
        ProcessStartRequest(), ProcessStream.STDOUT, reader, 0, Channel(1)
    )
    assert offset == 0


async def test_drain_process_output_honors_cancellation(pipe: tuple[File, File]) -> None:
    reader, writer = pipe
    writer.write(b"output")
    events: Events = Channel(1)
    offset = await with_cancel_requested(
        lambda: drain_process_output(ProcessStartRequest(), ProcessStream.STDOUT, reader, 0, events)
    )
    assert offset == 0
    assert_channel_empty(events)


async def test_drain_process_output_reports_invalid_reader(pipe: tuple[File, File]) -> None:
    reader, _ = pipe
    reader.close()
    with pytest.raises(PrimitiveError, match="set nonblocking"):
        await drain_process_output(
            ProcessStartRequest(), ProcessStream.STDOUT, reader, 0, Channel(1)
        )


async def test_stream_process_output_drains_after_deadline(pipe: tuple[File, File]) -> None:
    reader, writer = pipe
    data = b"available before the direct process exited"
    assert writer.write(data) == len(data)
    pipe_reader = PipeReader(reader)
    pipe_reader.finish()
    events: Events = Channel(1)
    error = await stream_process_output(
        REQUEST, ProcessOutputState(ProcessStream.STDOUT, pipe_reader), events
    )
    assert error is None
    assert (await events.get()).result == ProcessOutputResult(ProcessStream.STDOUT, 0, data)


async def test_stream_process_output_reports_read_failure(pipe: tuple[File, File]) -> None:
    reader, _ = pipe
    pipe_reader = PipeReader(reader)
    pipe_reader.close()
    events: Events = Channel(1)
    await run_process_output(REQUEST, ProcessOutputState(ProcessStream.STDERR, pipe_reader), events)
    event = await events.get()
    assert event.type == PrimitiveEventType.PROCESS_STREAM_FAILED
    assert isinstance(event.result, ProcessStreamFailureResult)
    assert event.result.stream == ProcessStream.STDERR
    assert "read process stderr" in event.result.error


async def test_stream_process_output_preserves_failure_during_cancellation(
    tmp_path: Path,
) -> None:
    reader = PipeReader(File.open(str(tmp_path), os.O_RDONLY))
    try:
        error = await with_cancel_requested(
            lambda: stream_process_output(
                ProcessStartRequest(), ProcessOutputState(ProcessStream.STDOUT, reader), Channel(1)
            )
        )
        assert error is not None
        assert "read process stdout" in str(error)
    finally:
        close_process_file(reader)


async def test_stream_process_output_honors_cancellation(pipe: tuple[File, File]) -> None:
    reader, writer = pipe
    writer.write(b"output")
    writer.close()
    events: Events = Channel(1)
    error = await with_cancel_requested(
        lambda: stream_process_output(
            ProcessStartRequest(),
            ProcessOutputState(ProcessStream.STDOUT, PipeReader(reader)),
            events,
        )
    )
    assert error is None
    assert_channel_empty(events)


async def test_stream_process_output_treats_canceled_close_as_completion(
    pipe: tuple[File, File],
) -> None:
    reader, _ = pipe
    pipe_reader = PipeReader(reader)
    pipe_reader.close()
    error = await with_cancel_requested(
        lambda: stream_process_output(
            ProcessStartRequest(), ProcessOutputState(ProcessStream.STDOUT, pipe_reader), Channel(1)
        )
    )
    assert error is None


async def test_drain_process_output_reports_read_failure(tmp_path: Path) -> None:
    directory = File.open(str(tmp_path), os.O_RDONLY)
    try:
        with pytest.raises(PrimitiveError, match="drain process stdout"):
            await drain_process_output(
                ProcessStartRequest(), ProcessStream.STDOUT, directory, 0, Channel(1)
            )
    finally:
        close_process_file(directory)


def test_process_result_errors() -> None:
    _, error = process_result("/bin/sh", WaitOutcome(None, RuntimeError("wait failed")))
    assert error is not None
    completed = subprocess.run(["/bin/sh", "-c", "exit 0"], check=True)
    _, error = process_result(
        "/bin/sh", WaitOutcome(completed.returncode, RuntimeError("wait failed"))
    )
    assert error is not None


async def test_process_control_cancellation_while_waiting_for_start() -> None:
    invocation = ProcessInvocation()
    events: Events = Channel()
    task = invocation.signal(
        ProcessSignalRequest(SourceID("operation-1"), CorrelationID("signal-1"), signal.SIGTERM),
        events,
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    event = await events.get()
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.CANCELED,
        "operation-1",
        "signal-1",
    )
    assert_channel_empty(events)


async def test_process_control_cancellation_after_start_becomes_ready() -> None:
    invocation = ProcessInvocation()
    invocation._mark_ready()
    events: Events = Channel()
    invocation.signal(
        ProcessSignalRequest(SourceID("operation-1"), CorrelationID("signal-1"), signal.SIGTERM),
        events,
    ).cancel()
    event = await events.get()
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.CANCELED,
        "operation-1",
        "signal-1",
    )
    assert_channel_empty(events)


async def test_run_process_canceled_before_start() -> None:
    request = ProcessStartRequest(SourceID("operation-1"), CorrelationID("process-1"), "/bin/sh")
    events: Events = Channel(1)
    invocation = ProcessInvocation()
    await with_cancel_requested(lambda: invocation.run(request, events))
    event = await events.get()
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.CANCELED,
        "operation-1",
        "process-1",
    )
    assert_channel_empty(events)


async def test_run_process_canceled_after_preparing_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    preparing = threading.Event()
    release = threading.Event()
    prepared: list[process_module.ProcessPipes] = []

    def blocking_prepare(
        request: ProcessStartRequest,
    ) -> tuple[process_module.ProcessCommand, process_module.ProcessPipes]:
        command, pipes = prepare_process(request)
        prepared.append(pipes)
        preparing.set()
        release.wait(10)
        return command, pipes

    monkeypatch.setattr(process_module, "prepare_process", blocking_prepare)
    request = ProcessStartRequest(
        SourceID("operation-1"), CorrelationID("process-1"), "/bin/sh", pipes=ProcessPipeSet.ALL
    )
    events: Events = Channel(1)
    invocation = ProcessInvocation()
    task = asyncio.create_task(invocation.run(request, events))
    await asyncio.to_thread(preparing.wait, 10)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with suppress(asyncio.CancelledError):
        await task
    event = await events.get()
    assert event.type == PrimitiveEventType.CANCELED
    assert invocation.process is None
    pipes = prepared[0]
    assert pipes.stdin_read is not None
    assert pipes.stdin_read.closed
    assert pipes.stdout_write is not None
    assert pipes.stdout_write.closed
    assert_channel_empty(events)


def test_prepare_process_uses_no_pipes_for_zero_selection() -> None:
    command, pipes = prepare_process(ProcessStartRequest(path="/usr/bin/true"))
    assert (command.stdin, command.stdout, command.stderr) == (None, None, None)
    assert pipes == process_module.ProcessPipes()


async def test_process_capture_descriptors_remain_open_until_wait_and_sync(
    tmp_path: Path,
) -> None:
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    for path in (stdout_path, stderr_path):
        path.write_bytes(b"")
    command, pipes = prepare_process(
        ProcessStartRequest(
            path="/bin/sh",
            arguments=["-c", "printf output; printf error >&2"],
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        parent = pipes.parent()
        assert parent.stdout_capture is not None
        assert parent.stderr_capture is not None
        assert pipes.stdout_write is None
        assert pipes.stderr_write is None
        process = spawn_process(command)
        assert pipes.close_child_ends() is None
        assert process.wait() == 0
        assert parent.sync_captures() is None
        assert parent.close_captures() is None
        assert stdout_path.read_bytes() == b"output"
        assert stderr_path.read_bytes() == b"error"
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            process.wait()
        assert pipes.close_all() is None


def test_sync_process_capture_reports_closed_descriptor(tmp_path: Path) -> None:
    file = File.open(str(tmp_path / "capture"), os.O_WRONLY | os.O_CREAT, 0o600)
    file.close()
    error = sync_process_capture(file, "stdout")
    assert error is not None
    assert "sync process stdout capture" in str(error)


async def test_prepare_process_closes_pipes_after_allocation_failure() -> None:
    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    limited = min(original[0], 64)
    if limited < 16:
        pytest.skip(f"descriptor limit {limited} is too small for a safe exhaustion test")
    descriptors: list[int] = []

    def exhaust() -> int:
        opened = 0
        while True:
            try:
                descriptors.append(os.open(os.devnull, os.O_RDONLY))
            except OSError as error:
                if error.errno != errno.EMFILE:
                    raise
                return opened
            opened += 1

    def release(count: int) -> None:
        for _ in range(count):
            os.close(descriptors.pop())

    request = ProcessStartRequest(
        SourceID("operation-1"), CorrelationID("process-1"), "/bin/sh", pipes=ProcessPipeSet.ALL
    )
    events: Events = Channel(1)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limited, original[1]))
    try:
        if exhaust() < 4:
            pytest.skip("too few descriptors were available below the reduced limit")
        await ProcessInvocation().run(request, events)
        event = events.get_nowait()
        assert event.type == PrimitiveEventType.FAILED
        assert isinstance(event.result, PrimitiveFailureResult)
        assert "create stdin pipe" in event.result.error

        release(2)
        with pytest.raises(PrimitiveError, match="create stdout pipe"):
            prepare_process(request)
        assert exhaust() == 2

        release(4)
        with pytest.raises(PrimitiveError, match="create stderr pipe"):
            prepare_process(request)
        assert exhaust() == 4
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, original)


async def test_process_lifecycle_helpers() -> None:
    assert close_process_file(None) is None
    process = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    process.wait()
    assert await terminate_process(process, ProcessParentPipes(), 0) is None


def start_waiting(process: subprocess.Popen[bytes]) -> asyncio.Future[int]:
    return asyncio.ensure_future(asyncio.to_thread(process.wait))


async def test_terminate_process_falls_back_to_direct_pid() -> None:
    process = subprocess.Popen(["/bin/sh", "-c", "while :; do :; done"])
    waited = start_waiting(process)
    try:
        assert await terminate_process(process, ProcessParentPipes(), 1.0) is None
        assert await waited < 0
    finally:
        if process.returncode is None:
            process.kill()
            await waited


async def test_signal_process_invocation_reports_invalid_signal() -> None:
    process = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], process_group=0)
    try:
        error = signal_process_invocation(process, -1)
        assert has_errno(error, errno.EINVAL), error
    finally:
        process.kill()
        assert process.wait() < 0


async def test_terminate_process_escalates_after_grace_period(tmp_path: Path) -> None:
    marker = tmp_path / "ready"
    process = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            "trap '' TERM; printf ready > \"$1\"; while :; do :; done",
            "sh",
            str(marker),
        ],
        process_group=0,
    )
    waited: asyncio.Future[int] | None = None
    try:
        deadline = time.monotonic() + 10
        while marker.read_bytes() != b"ready" if marker.exists() else True:
            assert time.monotonic() < deadline, "timed out waiting for ready marker"
            await asyncio.sleep(0.01)
        waited = start_waiting(process)
        grace = 0.05
        started = time.monotonic()
        assert await terminate_process(process, ProcessParentPipes(), grace) is None
        assert time.monotonic() - started >= grace
        assert await waited == -signal.SIGKILL
    finally:
        if process.returncode is None:
            process.kill()
            if waited is not None:
                await waited
            else:
                process.wait()


async def test_terminate_process_does_not_require_killed_process_to_be_reaped() -> None:
    process = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], process_group=0)
    try:
        assert await terminate_process(process, ProcessParentPipes(), 0) is None
        assert process.wait() != 0
    finally:
        if process.returncode is None:
            process.kill()
            process.wait()


async def test_await_process_completion_cleans_descendants_after_leader_exit() -> None:
    process = subprocess.Popen(["/bin/sh", "-c", "sleep 30 &"], process_group=0)
    assert process.wait() == 0
    group = process.pid
    exists, error = process_group_exists(group)
    assert error is None
    assert exists, "background descendant exited before the test"
    cleaned = False
    try:
        wait_completed: asyncio.Future[WaitOutcome] = asyncio.get_running_loop().create_future()
        completed = asyncio.ensure_future(
            with_cancel_requested(
                lambda: await_process_completion(process, ProcessParentPipes(), wait_completed, 1.0)
            )
        )
        exited, error = await wait_for_process_invocation(process, 1.0)
        assert error is None
        assert exited, "process group still exists after completion cleanup"
        cleaned = True
        wait_completed.set_result(WaitOutcome(0))
        outcome, canceled, completion_error = await completed
        assert (outcome, canceled, completion_error) == (WaitOutcome(0), False, None)
    finally:
        if not cleaned:
            with suppress(ProcessLookupError):
                os.kill(-group, signal.SIGKILL)


def test_process_completion_event() -> None:
    request = ProcessStartRequest(SourceID("operation-1"), CorrelationID("process-1"))
    exited = PrimitiveEvent(
        PrimitiveEventType.PROCESS_EXITED,
        request.source,
        request.correlation_id,
        ProcessExitResult(0),
    )
    assert process_completion_event(request, exited, None).type == PrimitiveEventType.PROCESS_EXITED
    got = process_completion_event(request, exited, RuntimeError("cleanup failed"))
    assert got.type == PrimitiveEventType.FAILED
    assert isinstance(got.result, PrimitiveFailureResult)
    assert "cleanup failed" in got.result.error


async def test_canceled_process_event_delivery() -> None:
    events: Events = Channel(1)
    delivered = await with_cancel_requested(
        lambda: send_event(events, PrimitiveEvent(PrimitiveEventType.PROCESS_STARTED))
    )
    assert not delivered

    await send_terminal(
        events,
        PrimitiveEvent(PrimitiveEventType.PROCESS_EXITED, REQUEST.source, REQUEST.correlation_id),
    )
    event = await events.get()
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.PROCESS_EXITED,
        "operation-1",
        "process-1",
    )
    await send_terminal(
        events,
        process_module.primitive_failure(
            REQUEST.source, REQUEST.correlation_id, RuntimeError("failed")
        ),
    )
    assert (await events.get()).type == PrimitiveEventType.FAILED


async def test_blocked_process_event_delivery_unblocks_on_cancellation() -> None:
    events: Events = Channel(1)
    await events.put(PrimitiveEvent(PrimitiveEventType.PROCESS_OUTPUT))
    delivered = asyncio.create_task(
        send_event(events, PrimitiveEvent(PrimitiveEventType.PROCESS_OUTPUT))
    )
    await asyncio.sleep(0)
    delivered.cancel()
    assert not await delivered
    assert events.qsize() == 1


@pytest.mark.parametrize(
    "terminal", [PrimitiveEventType.PROCESS_EXITED, PrimitiveEventType.CANCELED]
)
@pytest.mark.parametrize("cancel_sender", [False, True])
async def test_blocked_process_terminal_delivery_preserves_queued_event(
    terminal: PrimitiveEventType, cancel_sender: bool
) -> None:
    events: Events = Channel(1)
    await events.put(PrimitiveEvent(PrimitiveEventType.PROCESS_OUTPUT))
    delivered = asyncio.create_task(send_terminal(events, PrimitiveEvent(terminal)))
    await asyncio.sleep(0)
    if cancel_sender:
        delivered.cancel()
        await asyncio.sleep(0)
    assert (await events.get()).type == PrimitiveEventType.PROCESS_OUTPUT
    await delivered
    assert (await events.get()).type == terminal
    assert_channel_empty(events)
