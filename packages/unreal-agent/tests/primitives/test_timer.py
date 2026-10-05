import asyncio
from collections.abc import Callable

import pytest
from unreal_agent.gojson import Timestamp
from unreal_agent.primitives import (
    Channel,
    CorrelationID,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
    TimerRequest,
    TimerResult,
    schedule_timer,
)
from unreal_agent.primitives import timer as timer_module
from unreal_agent.primitives.timer import TIMER_MAXIMUM_SLEEP, timer_delay

SOURCE = SourceID("operation-1")
CORRELATION = CorrelationID("timer-1")
MINUTE_NS = 60 * 10**9
HOUR_NS = 60 * MINUTE_NS
BASE_NS = 1_785_758_400 * 10**9


@pytest.fixture
def fake_wall_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Timestamp]:
    """Drive the wall clock from looptime's fake loop time, in whole microseconds."""
    starts: dict[asyncio.AbstractEventLoop, float] = {}

    def now() -> Timestamp:
        loop = asyncio.get_running_loop()
        start = starts.setdefault(loop, loop.time())
        return Timestamp(BASE_NS + round((loop.time() - start) * 10**6) * 1000)

    monkeypatch.setattr(timer_module, "wall_clock", now)
    return now


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def single_event(events: Channel[PrimitiveEvent], kind: PrimitiveEventType) -> PrimitiveEvent:
    event = await events.get()
    assert event.type == kind
    assert events.empty()
    return event


@pytest.mark.looptime
async def test_schedule_timer_fires_at_absolute_deadline(
    fake_wall_clock: Callable[[], Timestamp],
) -> None:
    started_at = fake_wall_clock()
    request = TimerRequest(SOURCE, CORRELATION, Timestamp(started_at.unix_ns + 15 * MINUTE_NS))
    events: Channel[PrimitiveEvent] = Channel()
    schedule_timer(request, events)

    await settle()
    assert events.empty()
    await asyncio.sleep(15 * 60 - 0.001)
    await settle()
    assert events.empty()

    await asyncio.sleep(0.001)
    event = await single_event(events, PrimitiveEventType.TIMER_FIRED)
    assert (event.source, event.correlation_id) == (SOURCE, CORRELATION)
    assert event.result == TimerResult(request.deadline, request.deadline)


@pytest.mark.looptime
async def test_schedule_timer_fires_immediately_at_expired_deadline(
    fake_wall_clock: Callable[[], Timestamp],
) -> None:
    started_at = fake_wall_clock()
    request = TimerRequest(SOURCE, CORRELATION, Timestamp(started_at.unix_ns - HOUR_NS))
    events: Channel[PrimitiveEvent] = Channel()
    schedule_timer(request, events)

    event = await single_event(events, PrimitiveEventType.TIMER_FIRED)
    assert event.result == TimerResult(request.deadline, started_at)


@pytest.mark.looptime
async def test_schedule_timer_canceled_before_start(
    fake_wall_clock: Callable[[], Timestamp],
) -> None:
    request = TimerRequest(SOURCE, CORRELATION, Timestamp(fake_wall_clock().unix_ns + HOUR_NS))
    events: Channel[PrimitiveEvent] = Channel()
    task = schedule_timer(request, events)
    task.cancel()

    event = await single_event(events, PrimitiveEventType.CANCELED)
    assert event == PrimitiveEvent(PrimitiveEventType.CANCELED, SOURCE, CORRELATION)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.looptime
async def test_schedule_timer_cancellation_interrupts_wait(
    fake_wall_clock: Callable[[], Timestamp],
) -> None:
    started_at = fake_wall_clock()
    request = TimerRequest(SOURCE, CORRELATION, Timestamp(started_at.unix_ns + HOUR_NS))
    events: Channel[PrimitiveEvent] = Channel()
    task = schedule_timer(request, events)
    await settle()
    assert events.empty()

    task.cancel()
    event = await single_event(events, PrimitiveEventType.CANCELED)
    assert event == PrimitiveEvent(PrimitiveEventType.CANCELED, SOURCE, CORRELATION)
    assert fake_wall_clock() == started_at


async def test_schedule_timer_rejects_unset_deadline() -> None:
    events: Channel[PrimitiveEvent] = Channel()
    schedule_timer(TimerRequest(SOURCE, CORRELATION), events)

    event = await single_event(events, PrimitiveEventType.FAILED)
    assert (event.source, event.correlation_id) == (SOURCE, CORRELATION)
    assert isinstance(event.result, PrimitiveFailureResult)
    assert "deadline must be set" in event.result.error


NOW = Timestamp.parse("2026-08-03T12:00:00Z")


@pytest.mark.parametrize(
    ("deadline", "want"),
    [
        (Timestamp(NOW.unix_ns + HOUR_NS), TIMER_MAXIMUM_SLEEP),
        (Timestamp(NOW.unix_ns + 2 * 10**9), 2.0),
        (NOW, 0.0),
        (Timestamp(NOW.unix_ns - HOUR_NS), 0.0),
    ],
    ids=["distant", "near", "now", "past"],
)
def test_timer_delay_is_bounded_by_maximum_sleep(deadline: Timestamp, want: float) -> None:
    assert timer_delay(NOW, deadline) == want
