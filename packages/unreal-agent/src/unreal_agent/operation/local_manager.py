import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Self

from msgspec.structs import replace

from unreal_agent._go import go_quote
from unreal_agent.operation.image import TYPE_VIEW_IMAGE, ViewImage
from unreal_agent.operation.operation import (
    ID,
    Operation,
    OperationError,
    PrimitiveDispatch,
    Status,
    Step,
    unsupported,
)
from unreal_agent.operation.remote_job import TYPE_REMOTE_JOB, RemoteJobHandler, fail_remote_job
from unreal_agent.operation.remote_job_handler import (
    RemoteJobHandlers,
    RemoteJobHandlerUpdate,
    validate_remote_job_update,
)
from unreal_agent.operation.shell import TYPE_SHELL, Shell
from unreal_agent.operation.skill_use import (
    TYPE_SKILL_USE,
    advance_skill_use,
    decode_skill_use,
    fail_skill_use,
)
from unreal_agent.operation.value import TYPE_VALUE, advance_value
from unreal_agent.primitives import (
    ComputeRequest,
    EventSink,
    IOCreateRequest,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    ProcessStartRequest,
    SourceID,
    compute,
    create,
    read_file,
    start_process,
)

__all__ = ["ClosedError", "LocalOperationManager"]

_LOCAL_COMPLETED_EVENTS: Final = frozenset(
    {
        PrimitiveEventType.IO_CREATE_COMPLETED,
        PrimitiveEventType.IO_READ_COMPLETED,
        PrimitiveEventType.PROCESS_EXITED,
        PrimitiveEventType.FAILED,
        PrimitiveEventType.CANCELED,
        PrimitiveEventType.COMPUTE_COMPLETED,
    }
)
_FINISHED: Final = frozenset({Status.COMPLETED, Status.FAILED, Status.CANCELED})


class ClosedError(Exception):
    """The operation manager was closed (Go returns the context error)."""

    def __init__(self) -> None:
        super().__init__("operation manager is closed")


@dataclass(slots=True)
class _AddRequest:
    operation: Operation
    result: asyncio.Future[None]


@dataclass(slots=True)
class _CancelRequest:
    id: ID
    reason: str
    result: asyncio.Future[None]


@dataclass(slots=True)
class _Delivery:
    item: PrimitiveEvent | RemoteJobHandlerUpdate
    taken: asyncio.Future[None]


class _Close:
    pass


type _Message = _AddRequest | _CancelRequest | _Delivery | _Close


class _Mailbox:
    """The actor's only input. Deliveries rendezvous like sends on an unbuffered Go channel."""

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_Message] = asyncio.Queue()

    def post(self, message: _Message) -> None:
        self._queue.put_nowait(message)

    async def deliver(self, item: PrimitiveEvent | RemoteJobHandlerUpdate) -> None:
        """Wait until the actor takes `item`. A cancelled delivery is withdrawn untaken."""
        taken: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Delivery(item, taken))
        try:
            await taken
        except asyncio.CancelledError:
            if taken.done() and not taken.cancelled():
                return
            taken.cancel()
            raise

    async def put(self, event: PrimitiveEvent) -> None:
        await self.deliver(event)

    async def receive(
        self,
    ) -> _AddRequest | _CancelRequest | _Close | PrimitiveEvent | RemoteJobHandlerUpdate:
        while True:
            message = await self._queue.get()
            if isinstance(message, _Delivery):
                if message.taken.done():
                    continue
                message.taken.set_result(None)
                return message.item
            return message

    def receive_nowait(self) -> _Message:
        return self._queue.get_nowait()


type _Handler = Callable[[PrimitiveEvent | None], Step]


def _unset_handler(_: PrimitiveEvent | None) -> Step:
    raise OperationError("operation handler is not initialized")


@dataclass(slots=True, eq=False)
class _RunningOperation:
    operation: Operation
    remote_job_handler_index: int = -1
    handle: _Handler = _unset_handler
    canceled: bool = False
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def initialize(self) -> None:
        if self.operation.Type == TYPE_SHELL:
            self.handle = Shell(self.operation).handle
        elif self.operation.Type == TYPE_VIEW_IMAGE:
            self.handle = ViewImage(self.operation).handle
        else:
            self.handle = lambda event: advance_local_operation(self.operation, event)

    def cancel(self) -> None:
        """The equivalent of cancelling the operation's Go context."""
        self.canceled = True
        for task in list(self.tasks):
            task.cancel()

    def track(self, task: asyncio.Task[Any]) -> None:
        if self.canceled:
            task.cancel()
        if not task.done():
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)


class _Actor:
    """All manager state, owned by one task."""

    def __init__(self, remote_job_handlers: tuple[RemoteJobHandler | None, ...] = ()) -> None:
        self.mailbox = _Mailbox()
        self.remote_jobs = RemoteJobHandlers(remote_job_handlers)
        self.updates: asyncio.Queue[Operation] = asyncio.Queue()
        self.operations: dict[ID, _RunningOperation] = {}
        self.accepted: set[ID] = set()
        self.active_primitives = 0
        self.closing = False

    async def run(self) -> None:
        self.remote_jobs.start(self.mailbox.deliver)
        try:
            while True:
                message = await self.mailbox.receive()
                match message:
                    case _Close():
                        return
                    case _AddRequest():
                        await self._add(message)
                    case _CancelRequest():
                        await self._cancel(message)
                    case RemoteJobHandlerUpdate():
                        await self._remote_job_update(message)
                    case PrimitiveEvent():
                        self._primitive_event(message)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self.closing = True
        for current in list(self.operations.values()):
            current.cancel()
        self.remote_jobs.stop()
        try:
            await self.drain_local_primitives(self.active_primitives)
            self.active_primitives = 0
            await self.remote_jobs.wait()
        finally:
            while True:
                try:
                    message = self.mailbox.receive_nowait()
                except asyncio.QueueEmpty:
                    break
                _reject(message)
            self.updates.shutdown(immediate=True)

    async def _add(self, request: _AddRequest) -> None:
        current = request.operation
        if current.ID in self.accepted:
            _resolve(request.result)
            return
        running = _RunningOperation(operation=current)
        if current.Type == TYPE_REMOTE_JOB:
            try:
                running.remote_job_handler_index = await self.remote_jobs.add(current)
            except Exception as error:
                _resolve(request.result, error)
                return
            self.accepted.add(current.ID)
            self.operations[current.ID] = running
            _resolve(request.result)
            return
        try:
            running.initialize()
            step = running.handle(None)
        except Exception as error:
            running.cancel()
            _resolve(request.result, error)
            return
        self.accepted.add(current.ID)
        self.operations[current.ID] = running
        self.active_primitives += self.accept_local_step(running, step)
        _resolve(request.result)

    async def _cancel(self, request: _CancelRequest) -> None:
        running = self.operations.get(request.id)
        if running is None:
            _resolve(request.result)
            return
        if running.remote_job_handler_index >= 0:
            try:
                await self.remote_jobs.cancel(
                    running.remote_job_handler_index, request.id, request.reason
                )
            except Exception as error:
                _resolve(request.result, error)
                return
            _resolve(request.result)
            return
        running.cancel()
        _resolve(request.result)

    async def _remote_job_update(self, update: RemoteJobHandlerUpdate) -> None:
        index = update.handler_index
        if update.closed or update.operation is None:
            self.remote_jobs.mark_closed(index)
            for running in list(self.operations.values()):
                if running.remote_job_handler_index == index:
                    self.fail_local_operation(
                        running, OperationError(f"remote job handler {index} stopped")
                    )
            return
        updated = update.operation
        running = self.operations.get(updated.ID)
        if running is None:
            return
        if running.remote_job_handler_index != index:
            await self._fail_remote_job_operation(
                running,
                OperationError(
                    f"remote job operation {go_quote(updated.ID)} "
                    "update came from the wrong handler"
                ),
            )
            return
        try:
            validate_remote_job_update(running.operation, updated)
        except OperationError as error:
            await self._fail_remote_job_operation(running, error)
            return
        running.operation = updated
        self._append_update(updated)
        if updated.Status in _FINISHED:
            running.cancel()
            self.operations.pop(updated.ID, None)

    def _primitive_event(self, event: PrimitiveEvent) -> None:
        completed = event.type in _LOCAL_COMPLETED_EVENTS
        if completed:
            self.active_primitives -= 1
        running = self.operations.get(event.source)
        if running is None:
            return
        if completed and running.canceled:
            event = PrimitiveEvent(PrimitiveEventType.CANCELED, event.source, event.correlation_id)
        try:
            step = running.handle(event)
        except Exception as error:
            self.fail_local_operation(running, error)
            return
        self.active_primitives += self.accept_local_step(running, step)

    def accept_local_step(self, running: _RunningOperation, step: Step) -> int:
        """Publish the checkpoint and start its dispatches; return how many primitives started."""
        if step.operation is not None:
            running.operation = step.operation
            self._append_update(step.operation)
            if step.operation.Status in _FINISHED:
                running.cancel()
                self.operations.pop(step.operation.ID, None)
                return 0
        started = 0
        for dispatch in step.dispatches:
            try:
                task = start_local_primitive(dispatch, self.mailbox)
            except OperationError as error:
                rejection = PrimitiveEvent(
                    PrimitiveEventType.FAILED,
                    SourceID(running.operation.ID),
                    result=PrimitiveFailureResult(str(error)),
                )
                try:
                    failed = running.handle(rejection)
                except Exception as advance_error:
                    self.fail_local_operation(running, advance_error)
                    return started
                return started + self.accept_local_step(running, failed)
            running.track(task)
            started += 1
        return started

    def fail_local_operation(self, running: _RunningOperation, error: BaseException) -> None:
        self._append_update(fail_operation(running.operation, error))
        running.cancel()
        self.operations.pop(running.operation.ID, None)

    async def _fail_remote_job_operation(
        self, running: _RunningOperation, cause: BaseException
    ) -> None:
        if running.remote_job_handler_index >= 0:
            try:
                await self.remote_jobs.cancel(
                    running.remote_job_handler_index, running.operation.ID, str(cause)
                )
            except Exception as error:
                cause = OperationError(f"{cause}\ncancel rejected remote job: {error}")
        self.fail_local_operation(running, cause)

    async def drain_local_primitives(self, active: int) -> None:
        while active != 0:
            message = await self.mailbox.receive()
            if isinstance(message, PrimitiveEvent):
                if message.type in _LOCAL_COMPLETED_EVENTS:
                    active -= 1
            elif isinstance(message, _AddRequest | _CancelRequest):
                _reject(message)

    def _append_update(self, operation: Operation) -> None:
        self.updates.put_nowait(replace(operation))


def _resolve(result: asyncio.Future[None], error: BaseException | None = None) -> None:
    if result.done():
        return
    if error is None:
        result.set_result(None)
    else:
        result.set_exception(error)


def _reject(message: _Message) -> None:
    match message:
        case _AddRequest() | _CancelRequest():
            _resolve(message.result, ClosedError())
        case _Delivery():
            if not message.taken.done():
                message.taken.set_result(None)
        case _:
            pass


def advance_local_operation(current: Operation, event: PrimitiveEvent | None) -> Step:
    if current.Type == TYPE_VALUE:
        return advance_value(current, event)
    if current.Type == TYPE_SKILL_USE:
        return advance_skill_use(current, event)
    raise unsupported(f"local operation manager does not support type {go_quote(current.Type)}")


def fail_operation(current: Operation, error: BaseException | str) -> Operation:
    """Return `current` failed with `error`, recording the error in its state when possible."""
    failed: Operation | None = None
    try:
        if current.Type == TYPE_SKILL_USE:
            state = decode_skill_use(current)
            failed = fail_skill_use(current, state, error).operation
        elif current.Type == TYPE_SHELL:
            failed = Shell(current).fail(error).operation
        elif current.Type == TYPE_VIEW_IMAGE:
            failed = ViewImage(current).fail(error).operation
        elif current.Type == TYPE_REMOTE_JOB:
            failed = fail_remote_job(current, error).operation
    except OperationError:
        failed = None
    return failed if failed is not None else replace(current, Status=Status.FAILED)


def start_local_primitive(dispatch: PrimitiveDispatch, events: EventSink) -> asyncio.Task[None]:
    """Start a dispatched primitive. Raises OperationError when the dispatch is invalid."""
    data = dispatch.data
    match dispatch.type:
        case PrimitiveDispatchType.IO_CREATE:
            if not isinstance(data, IOCreateRequest):
                raise _invalid_data(dispatch, "primitives.IOCreateRequest")
            return create(data, events)
        case PrimitiveDispatchType.IO_READ:
            if not isinstance(data, IOReadRequest):
                raise _invalid_data(dispatch, "primitives.IOReadRequest")
            return read_file(data, events)
        case PrimitiveDispatchType.PROCESS_START:
            if not isinstance(data, ProcessStartRequest):
                raise _invalid_data(dispatch, "primitives.ProcessStartRequest")
            task = start_process(data, events).task
            assert task is not None
            return task
        case PrimitiveDispatchType.COMPUTE:
            if not isinstance(data, ComputeRequest):
                raise _invalid_data(dispatch, "primitives.ComputeRequest")
            return compute(data, events)
        case _:
            raise OperationError(
                f"local operation manager does not support dispatch {go_quote(dispatch.type)}"
            )


def _invalid_data(dispatch: PrimitiveDispatch, want: str) -> OperationError:
    kind = "<nil>" if dispatch.data is None else type(dispatch.data).__name__
    return OperationError(f"{dispatch.type} dispatch data is {kind}, want {want}")


class _Updates:
    __slots__ = ("_queue",)

    def __init__(self, queue: asyncio.Queue[Operation]) -> None:
        self._queue = queue

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Operation:
        try:
            return await self._queue.get()
        except asyncio.QueueShutDown:
            raise StopAsyncIteration from None


class LocalOperationManager:
    """Runs operations on local primitives and routes remote jobs to their handlers.

    Construct inside a running event loop. One actor task owns all state; `add`
    and `cancel` are request/response calls into it. `close` (or leaving the
    async context) cancels every operation, waits for every started primitive
    to finish, and then ends `updates()`.
    """

    __slots__ = ("_actor", "_task")

    def __init__(self, *remote_job_handlers: RemoteJobHandler | None) -> None:
        self._actor = _Actor(remote_job_handlers)
        self._task = asyncio.create_task(self._actor.run(), name="local-operation-manager")

    async def add(self, operation: Operation) -> None:
        """Start an operation at most once for each ID during the manager's lifetime."""
        await self._request(lambda result: _AddRequest(operation, result))

    async def cancel(self, id: ID, reason: str) -> None:
        await self._request(lambda result: _CancelRequest(id, reason, result))

    def updates(self) -> AsyncIterator[Operation]:
        """Every checkpoint in order. The buffer is unbounded and ends after `close`."""
        return _Updates(self._actor.updates)

    def close(self) -> None:
        if not self._actor.closing:
            self._actor.closing = True
            self._actor.mailbox.post(_Close())

    async def wait_closed(self) -> None:
        await asyncio.shield(self._task)

    async def aclose(self) -> None:
        self.close()
        await self.wait_closed()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _request(
        self, build: Callable[[asyncio.Future[None]], _AddRequest | _CancelRequest]
    ) -> None:
        if self._actor.closing or self._task.done():
            raise ClosedError
        result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._actor.mailbox.post(build(result))
        await asyncio.wait((result, self._task), return_when=asyncio.FIRST_COMPLETED)
        if not result.done():
            result.cancel()
            raise ClosedError
        await result
