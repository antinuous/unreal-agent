import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw
from unreal_agent.operation import (
    TYPE_SHELL,
    LocalOperationManager,
    Operation,
    OperationError,
    RemoteJobPlan,
    Status,
    UnsupportedError,
    decode_remote_job_state,
    new_remote_job_spec,
    update_remote_job,
)

type NewManager = Callable[..., LocalOperationManager]


@dataclass(slots=True, frozen=True)
class Cancellation:
    id: str
    reason: str


@dataclass(slots=True, eq=False)
class FakeRemoteJobHandler:
    plan_type: str = "test"
    plan_version: int = 1
    add_error: Exception | None = None
    cancel_error: Exception | None = None
    adds: asyncio.Queue[Operation] = field(default_factory=lambda: asyncio.Queue(1))
    cancellations: asyncio.Queue[Cancellation] = field(default_factory=lambda: asyncio.Queue(1))
    updates: asyncio.Queue[Operation] = field(default_factory=lambda: asyncio.Queue(1))

    def remote_job_plan_type(self) -> str:
        return self.plan_type

    def remote_job_plan_version(self) -> int:
        return self.plan_version

    async def add_remote_job(self, operation: Operation) -> None:
        if self.add_error is not None:
            raise self.add_error
        await self.adds.put(operation)

    async def cancel_remote_job(self, id: str, reason: str) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error
        await self.cancellations.put(Cancellation(id, reason))

    async def remote_job_updates(self) -> AsyncIterator[Operation]:
        while True:
            try:
                yield await self.updates.get()
            except asyncio.QueueShutDown:
                return

    def stop(self) -> None:
        self.updates.shutdown()


def new_remote_job_operation(operation_id: str, plan_type: str = "test") -> Operation:
    spec = new_remote_job_spec(RemoteJobPlan(Type=plan_type, Version=1, Data=Raw(b"{}")))
    return Operation(
        ID=operation_id,
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.READY,
        MaxOutputLength=spec.MaxOutputLength,
        State=spec.State,
    )


async def receive(queue: asyncio.Queue[Operation] | asyncio.Queue[Cancellation]) -> object:
    async with asyncio.timeout(5):
        return await queue.get()


async def receive_update(manager: LocalOperationManager) -> Operation:
    async with asyncio.timeout(5):
        return await anext(manager.updates())


async def test_local_operation_manager_routes_remote_jobs(new_manager: NewManager) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("remote-job")
    await manager.add(current)
    added = cast(Operation, await receive(handler.adds))
    assert added.ID == current.ID

    state = decode_remote_job_state(current)
    state.TerminalResult = '{"content":[]}'
    completed = update_remote_job(current, state, Status.COMPLETED)
    assert completed.operation is not None
    await handler.updates.put(completed.operation)
    assert (await receive_update(manager)).Status == Status.COMPLETED

    second = new_remote_job_operation("remote-cancel")
    await manager.add(second)
    await receive(handler.adds)
    await manager.cancel(second.ID, "requested")
    assert await receive(handler.cancellations) == Cancellation(second.ID, "requested")


async def test_local_operation_manager_rejects_invalid_remote_job_updates(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("remote-invalid")
    await manager.add(current)
    await receive(handler.adds)
    await handler.updates.put(replace(current, Status=Status.AWAITING, State=Raw(b"{")))

    failed = await receive_update(manager)
    state = decode_remote_job_state(failed)
    assert failed.Status == Status.FAILED
    assert "validate remote job handler update" in state.TerminalError
    assert cast(Cancellation, await receive(handler.cancellations)).id == current.ID


async def test_local_operation_manager_rejects_unsupported_plan(new_manager: NewManager) -> None:
    manager = new_manager()
    with pytest.raises(UnsupportedError):
        await manager.add(new_remote_job_operation("unsupported"))


async def test_local_operation_manager_rejects_ambiguous_plan(new_manager: NewManager) -> None:
    manager = new_manager(FakeRemoteJobHandler(), FakeRemoteJobHandler())
    with pytest.raises(OperationError, match="multiple handlers"):
        await manager.add(new_remote_job_operation("ambiguous"))


async def test_local_operation_manager_propagates_remote_add_error(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler(add_error=RuntimeError("cannot add"))
    manager = new_manager(handler)
    with pytest.raises(RuntimeError) as error:
        await manager.add(new_remote_job_operation("add-error"))
    assert error.value is handler.add_error


async def test_local_operation_manager_skips_missing_handlers(new_manager: NewManager) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(None, handler)
    await manager.add(new_remote_job_operation("after-missing-handler"))
    assert cast(Operation, await receive(handler.adds)).ID == "after-missing-handler"


async def test_local_operation_manager_propagates_remote_cancellation_error(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler(cancel_error=RuntimeError("cannot cancel"))
    manager = new_manager(handler)
    current = new_remote_job_operation("cancel-error")
    await manager.add(current)
    await receive(handler.adds)
    with pytest.raises(RuntimeError) as error:
        await manager.cancel(current.ID, "requested")
    assert error.value is handler.cancel_error


async def test_local_operation_manager_fails_jobs_when_remote_handler_stops(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("handler-stopped")
    await manager.add(current)
    await receive(handler.adds)
    handler.stop()

    failed = await receive_update(manager)
    state = decode_remote_job_state(failed)
    assert failed.Status == Status.FAILED
    assert "remote job handler 0 stopped" in state.TerminalError
    with pytest.raises(OperationError, match="remote job handler 0 stopped"):
        await manager.add(new_remote_job_operation("handler-already-stopped"))
    assert handler.adds.empty(), "stopped handler accepted an operation"


async def test_local_operation_manager_isolates_stopped_remote_handler(
    new_manager: NewManager,
) -> None:
    first = FakeRemoteJobHandler()
    second = FakeRemoteJobHandler(plan_type="other")
    manager = new_manager(first, second)
    first_job = new_remote_job_operation("first-handler")
    second_job = new_remote_job_operation("second-handler", "other")
    await manager.add(first_job)
    await receive(first.adds)
    await manager.add(second_job)
    await receive(second.adds)
    first.stop()
    failed = await receive_update(manager)
    assert (failed.ID, failed.Status) == (first_job.ID, Status.FAILED)

    state = decode_remote_job_state(second_job)
    state.TerminalResult = "{}"
    completed = update_remote_job(second_job, state, Status.COMPLETED)
    assert completed.operation is not None
    await second.updates.put(completed.operation)
    update = await receive_update(manager)
    assert (update.ID, update.Status) == (second_job.ID, Status.COMPLETED)


async def test_local_operation_manager_rejects_update_from_wrong_remote_handler(
    new_manager: NewManager,
) -> None:
    first = FakeRemoteJobHandler()
    second = FakeRemoteJobHandler(plan_type="other")
    manager = new_manager(first, second)
    current = new_remote_job_operation("wrong-handler")
    await manager.add(current)
    await receive(first.adds)
    awaiting = update_remote_job(current, decode_remote_job_state(current), Status.AWAITING)
    assert awaiting.operation is not None
    await second.updates.put(awaiting.operation)

    failed = await receive_update(manager)
    assert failed.Status == Status.FAILED
    assert "wrong handler" in decode_remote_job_state(failed).TerminalError
    assert cast(Cancellation, await receive(first.cancellations)).id == current.ID


def _change_plan(update: Operation) -> Operation:
    state = decode_remote_job_state(update)
    state.Plan = replace(state.Plan, Data=Raw(b'{"changed":true}'))
    step = update_remote_job(update, state, Status.AWAITING)
    assert step.operation is not None
    return step.operation


@pytest.mark.parametrize(
    ("name", "mutate", "want"),
    [
        (
            "output limit",
            lambda update: replace(update, MaxOutputLength=update.MaxOutputLength + 1),
            "changed output limit",
        ),
        ("type", lambda update: replace(update, Type=TYPE_SHELL), "type or version"),
        ("version", lambda update: replace(update, Version=update.Version + 1), "type or version"),
        (
            "status",
            lambda update: replace(update, Status=cast(Status, "unknown")),
            "invalid status",
        ),
        ("plan", _change_plan, "changed plan"),
    ],
)
async def test_local_operation_manager_validates_remote_job_update_identity(
    new_manager: NewManager, name: str, mutate: Callable[[Operation], Operation], want: str
) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("invalid-" + name)
    await manager.add(current)
    await receive(handler.adds)
    await handler.updates.put(mutate(replace(current, Status=Status.AWAITING)))

    failed = await receive_update(manager)
    state = decode_remote_job_state(failed)
    assert failed.Status == Status.FAILED
    assert failed.MaxOutputLength == current.MaxOutputLength
    assert want in state.TerminalError
    assert cast(Cancellation, await receive(handler.cancellations)).id == current.ID


async def test_local_operation_manager_rejects_remote_job_status_regression(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("status-regression")
    await manager.add(current)
    await receive(handler.adds)
    state = decode_remote_job_state(current)
    awaiting = update_remote_job(current, state, Status.AWAITING)
    assert awaiting.operation is not None
    await handler.updates.put(awaiting.operation)
    assert (await receive_update(manager)).Status == Status.AWAITING

    regressed = update_remote_job(awaiting.operation, state, Status.READY)
    assert regressed.operation is not None
    await handler.updates.put(regressed.operation)
    failed = await receive_update(manager)
    assert failed.Status == Status.FAILED
    assert "invalid status transition" in decode_remote_job_state(failed).TerminalError
    assert cast(Cancellation, await receive(handler.cancellations)).id == current.ID


async def test_local_operation_manager_reports_remote_cancellation_failure_after_invalid_update(
    new_manager: NewManager,
) -> None:
    handler = FakeRemoteJobHandler(cancel_error=RuntimeError("cannot cancel"))
    manager = new_manager(handler)
    current = new_remote_job_operation("invalid-cancel-error")
    await manager.add(current)
    await receive(handler.adds)
    await handler.updates.put(replace(current, Status=Status.AWAITING, State=Raw(b"{")))

    failed = await receive_update(manager)
    state = decode_remote_job_state(failed)
    assert failed.Status == Status.FAILED
    assert "validate remote job handler update" in state.TerminalError
    assert "cannot cancel" in state.TerminalError


async def test_local_operation_manager_adds_remote_job_once(new_manager: NewManager) -> None:
    handler = FakeRemoteJobHandler()
    manager = new_manager(handler)
    current = new_remote_job_operation("idempotent")
    await manager.add(current)
    await receive(handler.adds)
    await manager.add(current)
    assert handler.adds.empty(), "duplicate add"
