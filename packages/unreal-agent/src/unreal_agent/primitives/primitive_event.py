import asyncio
import contextlib
import contextvars
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, NewType, Protocol

from unreal_agent.primitives._errors import describe

SourceID = NewType("SourceID", str)
CorrelationID = NewType("CorrelationID", str)
NO_SOURCE: Final = SourceID("")
NO_CORRELATION: Final = CorrelationID("")


class PrimitiveEventType(StrEnum):
    FAILED = "primitive.failed"
    CANCELED = "primitive.canceled"

    COMPUTE_COMPLETED = "compute.completed"
    TIMER_FIRED = "timer.fired"

    IO_CREATE_COMPLETED = "io.create_completed"
    IO_READ_OUTPUT = "io.read_output"
    IO_READ_COMPLETED = "io.read_completed"

    PROCESS_STARTED = "process.started"
    PROCESS_OUTPUT = "process.output"
    PROCESS_STREAM_FAILED = "process.stream_failed"
    PROCESS_EXITED = "process.exited"
    PROCESS_INPUT_WRITTEN = "process.input_written"
    PROCESS_INPUT_WRITE_FAILED = "process.input_write_failed"
    PROCESS_INPUT_CLOSED = "process.input_closed"
    PROCESS_SIGNALED = "process.signaled"

    REMOTE_RESPONSE_STARTED = "remote.response_started"
    REMOTE_OUTPUT = "remote.output"
    REMOTE_STREAM_FAILED = "remote.stream_failed"
    REMOTE_RETRY_SCHEDULED = "remote.retry_scheduled"
    REMOTE_COMPLETED = "remote.completed"


TERMINAL_EVENT_TYPES = frozenset(
    {
        PrimitiveEventType.FAILED,
        PrimitiveEventType.CANCELED,
        PrimitiveEventType.COMPUTE_COMPLETED,
        PrimitiveEventType.TIMER_FIRED,
        PrimitiveEventType.IO_CREATE_COMPLETED,
        PrimitiveEventType.IO_READ_COMPLETED,
        PrimitiveEventType.PROCESS_EXITED,
        PrimitiveEventType.PROCESS_INPUT_WRITTEN,
        PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED,
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
        PrimitiveEventType.PROCESS_SIGNALED,
        PrimitiveEventType.REMOTE_COMPLETED,
    }
)


@dataclass(slots=True)
class PrimitiveEvent:
    type: PrimitiveEventType
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    result: Any = None


@dataclass(slots=True, frozen=True)
class PrimitiveFailureResult:
    error: str = ""


class EventSink(Protocol):
    """Where a primitive delivers events. `asyncio.Queue` satisfies it.

    `put` must block for backpressure and must not enqueue the item when the
    waiting call is cancelled.
    """

    async def put(self, item: PrimitiveEvent, /) -> None: ...


def primitive_failure(
    source: SourceID, correlation_id: CorrelationID, error: BaseException | str
) -> PrimitiveEvent:
    message = error if isinstance(error, str) else describe(error)
    return PrimitiveEvent(
        PrimitiveEventType.FAILED, source, correlation_id, PrimitiveFailureResult(message)
    )


def primitive_canceled(source: SourceID, correlation_id: CorrelationID) -> PrimitiveEvent:
    return PrimitiveEvent(PrimitiveEventType.CANCELED, source, correlation_id)


_running: set[asyncio.Task[Any]] = set()


def start_primitive[T](
    coroutine: Coroutine[Any, Any, T], name: str | None = None
) -> asyncio.Task[T]:
    """Start a primitive eagerly so cancellation can never skip its body.

    The body starts on the next loop iteration, so a cancellation requested
    before then yields `primitive.canceled` without doing any work. The module
    keeps a strong reference until the task finishes, so callers may drop the
    task the way Go callers drop a goroutine.
    """
    task = asyncio.create_task(_finish_cancellation(coroutine), name=name, eager_start=True)
    if not task.done():
        _running.add(task)
        task.add_done_callback(_running.discard)
    return task


async def _finish_cancellation[T](coroutine: Coroutine[Any, Any, T]) -> T:
    await _absorb_pending_cancellation()
    result = await coroutine
    if cancel_requested():
        raise asyncio.CancelledError
    return result


def cancel_requested() -> bool:
    """Report an outstanding cancellation request on the current task (Go `ctx.Err() != nil`)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


async def _absorb_pending_cancellation() -> None:
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(0)


async def send_event(events: EventSink, event: PrimitiveEvent) -> bool:
    """Send a non-terminal event. Returns False, dropping the event, on cancellation."""
    if cancel_requested():
        await _absorb_pending_cancellation()
        return False
    try:
        await events.put(event)
    except asyncio.CancelledError:
        return False
    return True


async def send_terminal(events: EventSink, event: PrimitiveEvent) -> None:
    """Send a terminal event. Cancellation never drops it; the request stays recorded."""
    while True:
        try:
            await events.put(event)
        except asyncio.CancelledError:
            continue
        return


async def wait_ignoring_cancellation[T](
    awaitable: Awaitable[T], on_cancel: Callable[[], None] | None = None
) -> T:
    """Wait for completion even if the task is cancelled; the request stays recorded.

    `on_cancel` runs for each cancellation delivered while waiting.
    """
    future = asyncio.ensure_future(awaitable)
    while not future.done():
        try:
            await asyncio.wait([future])
        except asyncio.CancelledError:
            if on_cancel is not None:
                on_cancel()
    return future.result()


async def run_blocking[T](
    function: Callable[..., T], *args: Any, on_cancel: Callable[[], None] | None = None
) -> T:
    """Run blocking work on the default executor and wait for it despite cancellation."""
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    future = loop.run_in_executor(None, lambda: context.run(function, *args))
    return await wait_ignoring_cancellation(future, on_cancel)


def run_in_thread[T](function: Callable[[], T]) -> asyncio.Future[T]:
    """Run long-lived blocking work on a dedicated thread so it cannot starve the executor."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    context = contextvars.copy_context()

    def deliver(outcome: Callable[[], None]) -> None:
        if not future.done():
            outcome()

    def target() -> None:
        try:
            value = context.run(function)
        except BaseException as error:
            failure = error
            _call_soon(loop, lambda: deliver(lambda: future.set_exception(failure)))
        else:
            _call_soon(loop, lambda: deliver(lambda: future.set_result(value)))

    threading.Thread(target=target, daemon=True, name="primitive-worker").start()
    return future


def _call_soon(loop: asyncio.AbstractEventLoop, callback: Callable[[], None]) -> None:
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(callback)
