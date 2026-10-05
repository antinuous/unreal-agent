import asyncio
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw, encode
from unreal_agent.operation import (
    TYPE_SHELL,
    VERSION_SHELL,
    ClosedError,
    LocalOperationManager,
    Operation,
    OperationError,
    PrimitiveDispatch,
    RemoteJobPlan,
    Shell,
    ShellInput,
    ShellPhase,
    ShellState,
    Status,
    Step,
    decode_shell_state,
    new_remote_job_spec,
)
from unreal_agent.operation.local_manager import (
    _Actor,
    _AddRequest,
    _Close,
    _RunningOperation,
    start_local_primitive,
)
from unreal_agent.primitives import (
    ComputeRequest,
    CorrelationID,
    IOCreateKind,
    IOCreateRequest,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
)

from tests.operation._helpers import Panic


def _shell_operation(state: ShellState, max_output_length: int) -> Operation:
    return Operation(
        ID="shell",
        Type=TYPE_SHELL,
        Version=VERSION_SHELL,
        Status=Status.AWAITING,
        State=Raw(encode(state)),
        MaxOutputLength=max_output_length,
    )


@pytest.mark.parametrize(
    "phase",
    [ShellPhase.READ_OUT, ShellPhase.READ_ERR, ShellPhase.READ_OUT_TAIL, ShellPhase.READ_ERR_TAIL],
)
def test_shell_read_chunks_are_transient_and_recovery_rereads(
    tmp_path: Path, phase: ShellPhase
) -> None:
    state = ShellState(
        Input=ShellInput(Shell="/bin/sh"),
        BaseDirectory=str(tmp_path),
        Phase=phase,
        PendingExitCode=0,
    )
    contents = b"abcdefghijkl"
    full_size = len(contents)
    match phase:
        case ShellPhase.READ_OUT:
            state.InlineOut = b"stale"
        case ShellPhase.READ_ERR:
            state.InlineErr = b"stale"
        case ShellPhase.READ_OUT_TAIL:
            state.InlineOutTail = b"stale"
            state.InlineOut, state.OutSize = b"h" * 12, 100
            full_size = 100
        case _:
            state.InlineErrTail = b"stale"
            state.InlineErr, state.ErrSize = b"h" * 12, 100
            full_size = 100
    execution = Shell(_shell_operation(state, 6))
    start = execution.handle(None)
    assert start.operation is not None
    durable = start.operation
    request = start.dispatches[0].data
    assert isinstance(request, IOReadRequest)
    original = bytes(durable.State)
    chunks = [contents[:2], contents[2:]]
    offset = request.offset
    for chunk in chunks:
        step = execution.handle(
            PrimitiveEvent(
                PrimitiveEventType.IO_READ_OUTPUT,
                request.source,
                request.correlation_id,
                IOReadOutputResult(offset, chunk),
            )
        )
        assert step.operation is None
        assert bytes(durable.State) == original
        offset += len(chunk)
    assert len(execution._chunks) == 2
    assert execution._chunks[0] is chunks[0]
    assert execution._chunks[1] is chunks[1]

    recovered = Shell(durable)
    replay = recovered.handle(None)
    assert replay.operation is not None
    assert replay.dispatches[0].data == request, "recovery skipped transient chunks"
    step = recovered.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_OUTPUT,
            request.source,
            request.correlation_id,
            IOReadOutputResult(request.offset, contents),
        )
    )
    assert step.operation is None
    completion = PrimitiveEvent(
        PrimitiveEventType.IO_READ_COMPLETED,
        request.source,
        request.correlation_id,
        IOReadCompletedResult(full_size),
    )
    finished = execution.handle(completion)
    assert finished.operation is not None
    assembled = decode_shell_state(finished.operation)
    match phase:
        case ShellPhase.READ_OUT:
            assert assembled.InlineOut == contents
        case ShellPhase.READ_OUT_TAIL:
            assert assembled.InlineOutTail == contents
        case _:
            assert finished.operation.Status == Status.COMPLETED
            assert assembled.Result is not None
            assert assembled.Result.ExitCode == 0
            want = f"abc...6 bytes truncated; complete output in {request.path}...jkl"
            if phase == ShellPhase.READ_ERR_TAIL:
                want = f"hhh...94 bytes truncated; complete output in {request.path}...jkl"
            assert assembled.Result.Err == want
            assert len(assembled.InlineErr) + len(assembled.InlineErrTail) == 0
    replayed = recovered.handle(completion)
    assert replayed == finished


@pytest.mark.parametrize(
    "scenario",
    [
        "wrong source",
        "wrong correlation",
        "wrong offset",
        "oversized",
        "invalid payload",
        "incomplete read",
        "invalid completion",
        "canceled",
        "failed",
    ],
)
def test_shell_read_validates_chunks_and_discards_them_on_termination(
    tmp_path: Path, scenario: str
) -> None:
    state = ShellState(
        Input=ShellInput(Shell="/bin/sh"), BaseDirectory=str(tmp_path), Phase=ShellPhase.READ_OUT
    )
    execution = Shell(_shell_operation(state, 3))
    start = execution.handle(None)
    assert start.operation is not None, "read start has no checkpoint"
    event = PrimitiveEvent(
        PrimitiveEventType.IO_READ_OUTPUT,
        SourceID("shell"),
        CorrelationID("read_out"),
        IOReadOutputResult(0, b"ab"),
    )
    assert execution.handle(event).operation is None
    event = replace_event(event, result=IOReadOutputResult(2, b"c"))
    match scenario:
        case "wrong source":
            event = replace_event(event, source=SourceID("other"))
        case "wrong correlation":
            event = replace_event(event, correlation_id=CorrelationID("read_err"))
        case "wrong offset":
            event = replace_event(event, result=IOReadOutputResult(0, b"c"))
        case "oversized":
            event = replace_event(event, result=IOReadOutputResult(2, b"c" * 11))
        case "invalid payload":
            event = replace_event(event, result="invalid")
        case "incomplete read":
            event = replace_event(
                event, type=PrimitiveEventType.IO_READ_COMPLETED, result=IOReadCompletedResult(3)
            )
        case "invalid completion":
            event = replace_event(
                event, type=PrimitiveEventType.IO_READ_COMPLETED, result="invalid"
            )
        case "canceled":
            event = replace_event(event, type=PrimitiveEventType.CANCELED, result=None)
        case "failed":
            event = replace_event(
                event,
                type=PrimitiveEventType.FAILED,
                result=PrimitiveFailureResult("read failed"),
            )
    step = execution.handle(event)
    assert step.operation is not None
    assert step.operation.Status in (Status.CANCELED, Status.FAILED)
    assert decode_shell_state(step.operation).InlineOut == b""
    assert execution._chunks == []


def replace_event(event: PrimitiveEvent, **changes: Any) -> PrimitiveEvent:
    fields = {
        "type": event.type,
        "source": event.source,
        "correlation_id": event.correlation_id,
        "result": event.result,
    }
    fields.update(changes)
    return PrimitiveEvent(**fields)


async def test_local_operation_manager_starts_multiple_primitives(tmp_path: Path) -> None:
    operation_id = "local-multiple-primitives"
    actor = _Actor()
    current = _RunningOperation(operation=Operation(ID=operation_id, Status=Status.AWAITING))
    actor.operations[operation_id] = current
    assert actor.accept_local_step(current, Step()) == 0

    def dispatch(correlation: str, name: str) -> PrimitiveDispatch:
        return PrimitiveDispatch(
            PrimitiveDispatchType.IO_CREATE,
            IOCreateRequest(
                source=SourceID(operation_id),
                correlation_id=CorrelationID(correlation),
                kind=IOCreateKind.REGULAR_FILE,
                path=str(tmp_path / name),
            ),
        )

    started = actor.accept_local_step(
        current, Step(dispatches=[dispatch("first", "first"), dispatch("second", "second")])
    )
    assert started == 2
    assert actor.updates.empty()
    assert current.operation.Status == Status.AWAITING
    assert not current.canceled

    completed = replace(current.operation, Status=Status.COMPLETED)
    actor.accept_local_step(current, Step(operation=completed))
    assert actor.operations == {}
    assert actor.updates.qsize() == 1
    assert actor.updates.get_nowait().Status == Status.COMPLETED
    async with asyncio.timeout(15):
        await actor.drain_local_primitives(started)


async def test_local_operation_manager_dispatches_compute_and_drains_after_cancellation() -> None:
    actor = _Actor()
    current = _RunningOperation(operation=Operation(ID="compute", Status=Status.AWAITING))
    actor.operations[current.operation.ID] = current
    started = threading.Event()
    release = threading.Event()

    def run(_: threading.Event) -> str:
        started.set()
        release.wait()
        return "finished"

    step = Step(
        dispatches=[
            PrimitiveDispatch(
                PrimitiveDispatchType.COMPUTE,
                ComputeRequest(
                    source=SourceID(current.operation.ID),
                    correlation_id=CorrelationID("prepare"),
                    run=run,
                ),
            )
        ]
    )
    try:
        active = actor.accept_local_step(current, step)
        assert active == 1
        assert await asyncio.to_thread(started.wait, 10)
        current.cancel()
        drained = asyncio.create_task(actor.drain_local_primitives(active))
        await asyncio.sleep(0.2)
        assert not drained.done(), "shutdown finished before canceled callback returned"
        assert actor.updates.empty()
        assert current.operation.Status == Status.AWAITING
    finally:
        release.set()
    async with asyncio.timeout(15):
        await drained


@pytest.mark.parametrize(
    "outcome", ["completed", "failed", "canceled", "panicked", "exited", "missing callback"]
)
async def test_local_operation_manager_drains_every_compute_outcome(outcome: str) -> None:
    actor = _Actor()

    def run(_: threading.Event) -> int:
        match outcome:
            case "failed":
                raise RuntimeError("failed work")
            case "panicked":
                raise Panic("failed callback")
            case "exited":
                raise SystemExit
        return 42

    request = ComputeRequest(run=None if outcome == "missing callback" else run)
    task = start_local_primitive(
        PrimitiveDispatch(PrimitiveDispatchType.COMPUTE, request), actor.mailbox
    )
    if outcome == "canceled":
        task.cancel()
    async with asyncio.timeout(15):
        await actor.drain_local_primitives(1)


async def test_local_operation_manager_rejects_invalid_compute_payload() -> None:
    actor = _Actor()
    current = _RunningOperation(operation=Operation(ID="rejected", Status=Status.AWAITING))
    failure = ""

    def handle(event: PrimitiveEvent | None) -> Step:
        nonlocal failure
        if (
            event is None
            or event.type != PrimitiveEventType.FAILED
            or event.source != current.operation.ID
        ):
            raise OperationError("unexpected rejection event")
        if not isinstance(event.result, PrimitiveFailureResult):
            raise OperationError("invalid rejection result")
        failure = event.result.error
        return Step(operation=replace(current.operation, Status=Status.FAILED))

    current.handle = handle
    actor.operations[current.operation.ID] = current
    dispatch = PrimitiveDispatch(PrimitiveDispatchType.COMPUTE)
    assert actor.accept_local_step(current, Step(dispatches=[dispatch])) == 0
    assert "compute dispatch data" in failure
    assert actor.operations == {}
    assert actor.updates.qsize() == 1
    assert actor.updates.get_nowait().Status == Status.FAILED
    assert current.canceled


class _BlockingRemoteJobHandler:
    def __init__(self) -> None:
        self.adding = asyncio.Event()
        self.updates: asyncio.Queue[Operation] = asyncio.Queue()

    def remote_job_plan_type(self) -> str:
        return "test"

    def remote_job_plan_version(self) -> int:
        return 1

    async def add_remote_job(self, operation: Operation) -> None:
        self.adding.set()
        await asyncio.Event().wait()

    async def cancel_remote_job(self, id: str, reason: str) -> None:
        pass

    async def remote_job_updates(self) -> AsyncGenerator[Operation]:
        while True:
            try:
                yield await self.updates.get()
            except asyncio.QueueShutDown:
                return

    def stop(self) -> None:
        self.updates.shutdown()


async def test_local_operation_manager_request_returns_when_actor_stops_mid_request() -> None:
    handler = _BlockingRemoteJobHandler()
    manager = LocalOperationManager(handler)
    spec = new_remote_job_spec(RemoteJobPlan(Type="test", Version=1, Data=Raw(b"{}")))
    operation = Operation(
        ID="remote",
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.READY,
        MaxOutputLength=spec.MaxOutputLength,
        State=spec.State,
    )
    adding = asyncio.create_task(manager.add(operation))
    async with asyncio.timeout(5):
        await handler.adding.wait()
    manager._task.cancel()
    async with asyncio.timeout(5):
        with pytest.raises(ClosedError):
            await adding
        with pytest.raises(asyncio.CancelledError):
            await manager.wait_closed()


async def test_local_operation_manager_ends_updates_when_cancelled_during_drain() -> None:
    actor = _Actor()
    actor.active_primitives = 1
    task = asyncio.create_task(actor.run())
    actor.mailbox.post(_Close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    actor.mailbox.post(_AddRequest(Operation(ID="late"), result))
    await asyncio.sleep(0)
    task.cancel()
    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(asyncio.QueueShutDown):
            await actor.updates.get()
        with pytest.raises(ClosedError):
            await result
