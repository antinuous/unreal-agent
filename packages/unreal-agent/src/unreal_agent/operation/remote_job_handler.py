import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from unreal_agent._go import go_quote
from unreal_agent.operation.operation import ID, Operation, OperationError, Status, unsupported
from unreal_agent.operation.remote_job import RemoteJobHandler, decode_remote_job_state


@dataclass(slots=True)
class RemoteJobHandlerUpdate:
    handler_index: int
    operation: Operation | None = None
    closed: bool = False


type _Deliver = Callable[[RemoteJobHandlerUpdate], Awaitable[None]]


class RemoteJobHandlers:
    """Routes remote jobs to the single handler registered for their plan type and version."""

    __slots__ = ("closed", "handlers", "workers")

    def __init__(self, handlers: Sequence[RemoteJobHandler | None]) -> None:
        self.handlers = list(handlers)
        self.closed = [False] * len(self.handlers)
        self.workers: list[asyncio.Task[None]] = []

    def start(self, deliver: _Deliver) -> None:
        for index, handler in enumerate(self.handlers):
            if handler is not None:
                self.workers.append(
                    asyncio.create_task(
                        self._forward(index, handler, deliver), name=f"remote-job-handler-{index}"
                    )
                )

    def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()

    async def wait(self) -> None:
        await asyncio.gather(*self.workers, return_exceptions=True)

    async def add(self, current: Operation) -> int:
        state = decode_remote_job_state(current)
        plan = state.Plan
        matched = -1
        for index, handler in enumerate(self.handlers):
            if (
                handler is None
                or handler.remote_job_plan_type() != plan.Type
                or handler.remote_job_plan_version() != plan.Version
            ):
                continue
            if matched != -1:
                raise OperationError(
                    "local operation manager has multiple handlers for remote job plan "
                    f"{go_quote(plan.Type)} version {plan.Version}"
                )
            matched = index
        if matched == -1:
            raise unsupported(
                "local operation manager does not support remote job plan "
                f"{go_quote(plan.Type)} version {plan.Version}"
            )
        if self.closed[matched]:
            raise OperationError(f"remote job handler {matched} stopped")
        handler = self.handlers[matched]
        assert handler is not None
        await handler.add_remote_job(current)
        return matched

    def mark_closed(self, index: int) -> None:
        self.closed[index] = True

    async def cancel(self, index: int, operation_id: ID, reason: str) -> None:
        handler = self.handlers[index]
        assert handler is not None
        await handler.cancel_remote_job(operation_id, reason)

    @staticmethod
    async def _forward(index: int, handler: RemoteJobHandler, deliver: _Deliver) -> None:
        try:
            async with contextlib.aclosing(handler.remote_job_updates()) as updates:
                async for operation in updates:
                    await deliver(RemoteJobHandlerUpdate(index, operation))
        except Exception:
            # A failed update stream stops the handler as if the stream had ended.
            pass
        await deliver(RemoteJobHandlerUpdate(index, closed=True))


def validate_remote_job_update(current: Operation, updated: Operation) -> None:
    operation_id = go_quote(current.ID)
    if updated.MaxOutputLength != current.MaxOutputLength:
        raise OperationError(
            f"remote job handler changed output limit for operation {operation_id}"
        )
    if updated.Type != current.Type or updated.Version != current.Version:
        raise OperationError(
            f"remote job handler changed operation {operation_id} type or version from "
            f"{go_quote(current.Type)} version {current.Version} to "
            f"{go_quote(updated.Type)} version {updated.Version}"
        )
    try:
        current_state = decode_remote_job_state(current)
    except OperationError as error:
        raise OperationError(f"validate current remote job: {error}") from error
    try:
        updated_state = decode_remote_job_state(updated)
    except OperationError as error:
        raise OperationError(f"validate remote job handler update: {error}") from error
    if (
        updated_state.Plan.Type != current_state.Plan.Type
        or updated_state.Plan.Version != current_state.Plan.Version
        or bytes(updated_state.Plan.Data) != bytes(current_state.Plan.Data)
    ):
        raise OperationError(f"remote job handler changed plan for operation {operation_id}")
    if not valid_remote_job_status_transition(current.Status, updated.Status):
        raise OperationError(
            "remote job handler produced invalid status transition for operation "
            f"{operation_id} from {go_quote(current.Status)} to {go_quote(updated.Status)}"
        )


def valid_remote_job_status_transition(current: Status | str, updated: Status | str) -> bool:
    match current:
        case Status.READY | Status.AWAITING:
            return updated in (
                Status.AWAITING,
                Status.CANCELING,
                Status.COMPLETED,
                Status.FAILED,
                Status.CANCELED,
            )
        case Status.CANCELING:
            return updated in (Status.CANCELING, Status.FAILED, Status.CANCELED)
        case _:
            return False
