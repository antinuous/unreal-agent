import asyncio
import threading
from typing import Any

import pytest
from unreal_agent.primitives import (
    Channel,
    ComputeRequest,
    ComputeResult,
    CorrelationID,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
    compute,
)

SOURCE = SourceID("image")
RESIZE = CorrelationID("resize")
DECODE = CorrelationID("decode")


class Panic(BaseException):
    pass


async def receive(events: Channel[PrimitiveEvent]) -> PrimitiveEvent:
    async with asyncio.timeout(10):
        return await events.get()


async def assert_no_event(events: Channel[PrimitiveEvent], task: asyncio.Task[None]) -> None:
    await asyncio.wait([task], timeout=10)
    assert events.empty()


async def finish(task: asyncio.Task[None]) -> None:
    await asyncio.wait([task], timeout=10)
    assert task.done()


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (None, PrimitiveEventType.COMPUTE_COMPLETED),
        (ValueError("encoding failed"), PrimitiveEventType.FAILED),
        (asyncio.CancelledError("context canceled"), PrimitiveEventType.FAILED),
    ],
    ids=["success", "partial failure", "callback cancellation error"],
)
async def test_compute_returns_value_or_failure(
    error: BaseException | None, kind: PrimitiveEventType
) -> None:
    value = [640, 480]

    def run(canceled: threading.Event) -> Any:
        if not isinstance(canceled, threading.Event) or canceled.is_set():
            raise RuntimeError("callback did not receive an active cancellation signal")
        if error is not None:
            raise error
        return value

    events: Channel[PrimitiveEvent] = Channel(2)
    task = compute(ComputeRequest(SOURCE, RESIZE, run), events)
    event = await receive(events)
    assert (event.type, event.source, event.correlation_id) == (kind, SOURCE, RESIZE)
    if error is not None:
        assert event.result == PrimitiveFailureResult(str(error))
    else:
        assert event.result == ComputeResult(value)
    await assert_no_event(events, task)


async def test_compute_canceled_before_start() -> None:
    ran = False

    def run(_: threading.Event) -> None:
        nonlocal ran
        ran = True

    events: Channel[PrimitiveEvent] = Channel(2)
    task = compute(ComputeRequest(SOURCE, RESIZE, run), events)
    task.cancel()
    event = await receive(events)
    assert event == PrimitiveEvent(PrimitiveEventType.CANCELED, SOURCE, RESIZE)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ran
    assert events.empty()


async def test_compute_cancellation_waits_for_callback() -> None:
    started = threading.Event()
    release = threading.Event()

    def run(_: threading.Event) -> str:
        started.set()
        release.wait()
        raise ValueError("incomplete image")

    events: Channel[PrimitiveEvent] = Channel(2)
    task = compute(ComputeRequest(SOURCE, RESIZE, run), events)
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        assert events.empty()
        assert not task.done()
    finally:
        release.set()
    event = await receive(events)
    assert event == PrimitiveEvent(PrimitiveEventType.CANCELED, SOURCE, RESIZE)
    await finish(task)
    assert task.cancelled()
    assert events.empty()


async def test_compute_callback_observes_cancellation() -> None:
    started = threading.Event()

    def run(canceled: threading.Event) -> None:
        started.set()
        canceled.wait()
        raise asyncio.CancelledError

    events: Channel[PrimitiveEvent] = Channel(1)
    task = compute(ComputeRequest(run=run), events)
    assert await asyncio.to_thread(started.wait, 10)
    task.cancel()
    event = await receive(events)
    assert event.type == PrimitiveEventType.CANCELED
    assert event.result is None
    await finish(task)


async def test_compute_reports_invalid_request() -> None:
    events: Channel[PrimitiveEvent] = Channel()
    task = compute(ComputeRequest(SOURCE, RESIZE), events)
    event = await receive(events)
    assert (event.type, event.source, event.correlation_id) == (
        PrimitiveEventType.FAILED,
        SOURCE,
        RESIZE,
    )
    assert event.result == PrimitiveFailureResult("compute: Run must be set")
    await assert_no_event(events, task)


@pytest.mark.parametrize("blocked", ["callback", "event delivery"])
async def test_compute_invocations_run_independently(blocked: str) -> None:
    release = threading.Event()

    def first_run(_: threading.Event) -> str:
        if blocked == "callback":
            release.wait()
        return "first"

    first: Channel[PrimitiveEvent] = Channel()
    first_task = compute(ComputeRequest(run=first_run), first)
    try:
        await asyncio.sleep(0.01)
        second: Channel[PrimitiveEvent] = Channel(1)
        compute(ComputeRequest(run=lambda _: "second"), second)
        assert (await receive(second)).result == ComputeResult("second")
    finally:
        release.set()
    assert (await receive(first)).result == ComputeResult("first")
    await finish(first_task)


@pytest.mark.parametrize(
    ("error", "message", "cancel"),
    [
        (Panic("broken codec"), "broken codec", False),
        (KeyboardInterrupt("interrupted"), "interrupted", False),
        (Panic("broken codec"), "", True),
    ],
    ids=["base exception", "keyboard interrupt", "canceled"],
)
async def test_compute_recovers_callback_panics(
    error: BaseException, message: str, cancel: bool
) -> None:
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Task[None]] = []

    def run(_: threading.Event) -> None:
        if cancel:
            loop.call_soon_threadsafe(tasks[0].cancel)
        raise error

    events: Channel[PrimitiveEvent] = Channel(2)
    tasks.append(compute(ComputeRequest(SOURCE, DECODE, run), events))
    event = await receive(events)
    want = PrimitiveEventType.CANCELED if cancel else PrimitiveEventType.FAILED
    assert (event.type, event.source, event.correlation_id) == (want, SOURCE, DECODE)
    if cancel:
        assert event.result is None
    else:
        assert event.result == PrimitiveFailureResult(f"compute callback panicked: {message}")
    await assert_no_event(events, tasks[0])


@pytest.mark.parametrize("canceled", [False, True], ids=["failed", "canceled"])
async def test_compute_reports_exit(canceled: bool) -> None:
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Task[None]] = []
    cleaned_up = threading.Event()

    def run(_: threading.Event) -> None:
        try:
            if canceled:
                loop.call_soon_threadsafe(tasks[0].cancel)
            raise SystemExit
        finally:
            cleaned_up.set()

    events: Channel[PrimitiveEvent] = Channel(2)
    tasks.append(compute(ComputeRequest(SOURCE, DECODE, run), events))
    event = await receive(events)
    want = PrimitiveEventType.CANCELED if canceled else PrimitiveEventType.FAILED
    assert (event.type, event.source, event.correlation_id) == (want, SOURCE, DECODE)
    assert cleaned_up.is_set()
    if canceled:
        assert event.result is None
    else:
        assert event.result == PrimitiveFailureResult("compute callback exited without returning")
    await assert_no_event(events, tasks[0])
