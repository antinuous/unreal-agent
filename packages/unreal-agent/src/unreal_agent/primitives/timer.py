import asyncio
import time
from dataclasses import dataclass, field
from typing import Final

from unreal_agent.gojson import Timestamp
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
    send_terminal,
    start_primitive,
)

TIMER_MAXIMUM_SLEEP: Final = 5.0


@dataclass(slots=True)
class TimerRequest:
    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    deadline: Timestamp = field(default_factory=Timestamp)


@dataclass(slots=True, frozen=True)
class TimerResult:
    deadline: Timestamp
    fired_at: Timestamp


def wall_clock() -> Timestamp:
    return Timestamp(time.time_ns())


def schedule_timer(request: TimerRequest, events: EventSink) -> asyncio.Task[None]:
    return start_primitive(_run_timer(request, events), name="timer")


async def _run_timer(request: TimerRequest, events: EventSink) -> None:
    if request.deadline.is_zero():
        await send_terminal(
            events,
            primitive_failure(
                request.source, request.correlation_id, "schedule timer: deadline must be set"
            ),
        )
        return
    if cancel_requested():
        await send_terminal(events, primitive_canceled(request.source, request.correlation_id))
        return

    deadline = request.deadline
    delay = timer_delay(wall_clock(), deadline)
    try:
        while True:
            await asyncio.sleep(delay)
            now = wall_clock()
            if now >= deadline:
                break
            delay = timer_delay(now, deadline)
    except asyncio.CancelledError:
        await send_terminal(events, primitive_canceled(request.source, request.correlation_id))
        raise
    await send_terminal(
        events,
        PrimitiveEvent(
            PrimitiveEventType.TIMER_FIRED,
            request.source,
            request.correlation_id,
            TimerResult(request.deadline, now),
        ),
    )


def timer_delay(now: Timestamp, deadline: Timestamp) -> float:
    """Bound each sleep so every wait rechecks the absolute wall-clock deadline."""
    remaining = deadline.unix_ns - now.unix_ns
    return min(max(remaining, 0), int(TIMER_MAXIMUM_SLEEP * 1e9)) / 1e9
