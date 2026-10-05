import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from unreal_agent.primitives._errors import describe
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
    run_in_thread,
    send_terminal,
    start_primitive,
    wait_ignoring_cancellation,
)

type ComputeFunction = Callable[[threading.Event], Any]


@dataclass(slots=True)
class ComputeRequest:
    """Offloads CPU-heavy or blocking third-party calls to a dedicated thread.

    `run` receives an event that is set when the primitive is cancelled. It
    returns the value or raises to report a failure.
    """

    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    run: ComputeFunction | None = None


@dataclass(slots=True, frozen=True)
class ComputeResult:
    value: Any = None


def compute(request: ComputeRequest, events: EventSink) -> asyncio.Task[None]:
    return start_primitive(_run_compute(request, events), name="compute")


async def _run_compute(request: ComputeRequest, events: EventSink) -> None:
    run = request.run
    if run is None:
        await send_terminal(
            events,
            primitive_failure(request.source, request.correlation_id, "compute: Run must be set"),
        )
        return
    if cancel_requested():
        await send_terminal(events, primitive_canceled(request.source, request.correlation_id))
        return

    canceled = threading.Event()
    outcome = await wait_ignoring_cancellation(
        run_in_thread(lambda: _call(run, canceled)), on_cancel=canceled.set
    )
    if cancel_requested():
        await send_terminal(events, primitive_canceled(request.source, request.correlation_id))
        return
    value, error = outcome
    if error is not None:
        await send_terminal(
            events, primitive_failure(request.source, request.correlation_id, error)
        )
        return
    await send_terminal(
        events,
        PrimitiveEvent(
            PrimitiveEventType.COMPUTE_COMPLETED,
            request.source,
            request.correlation_id,
            ComputeResult(value),
        ),
    )


def _call(run: ComputeFunction, canceled: threading.Event) -> tuple[Any, str | None]:
    try:
        return run(canceled), None
    except SystemExit:
        return None, "compute callback exited without returning"
    except (Exception, asyncio.CancelledError) as error:
        return None, describe(error)
    except BaseException as error:
        return None, f"compute callback panicked: {describe(error)}"
