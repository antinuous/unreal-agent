import asyncio
from typing import Any, cast

import pytest
from unreal_agent.coordinator.loop import (
    CLOSED,
    SLURP_IDLE_TIMEOUT,
    SLURP_MAX_ITEMS,
    Closed,
    Source,
    slurp,
)

slurp_synctest = pytest.mark.looptime(resolution=1e-9)

SLURP_IDLE_TIMEOUT_NANOSECONDS = round(SLURP_IDLE_TIMEOUT * 1e9)


async def slurp_wait() -> None:
    """Go's `synctest.Wait`: run the loop until no callback is ready or due."""
    loop = asyncio.get_running_loop()
    internals = cast(Any, loop)
    while True:
        await asyncio.sleep(0)
        if internals._ready:
            continue
        scheduled = internals._scheduled
        if scheduled and not scheduled[0].cancelled() and scheduled[0].when() <= loop.time():
            continue
        return


async def slurp_sleep(nanoseconds: int) -> None:
    if nanoseconds > 0:
        await asyncio.sleep(nanoseconds / 1e9)
    await slurp_wait()


def slurp_now() -> int:
    return round(asyncio.get_running_loop().time() * 1e9)


def queue_source[T](queue: asyncio.Queue[T]) -> Source[T]:
    """A Go channel read: values in order, then `CLOSED` once shut down and drained."""

    async def receive() -> T | Closed:
        try:
            return await queue.get()
        except asyncio.QueueShutDown:
            return CLOSED

    return Source(receive, lambda: None)


async def close_source[T](source: Source[T]) -> None:
    task = source.cancel()
    if task is not None:
        await asyncio.wait((task,))


@pytest.mark.parametrize("count", [0, 3])
@slurp_synctest
async def test_slurp_channel_preserves_order(count: int) -> None:
    output: asyncio.Queue[int] = asyncio.Queue(count)
    want: list[int] = []
    for value in range(count):
        output.put_nowait(value)
        want.append(value)
    source = queue_source(output)
    started_at = slurp_now()
    got = await slurp(source)
    assert got == want, f"slurped values = {got}, want {want}"
    elapsed = slurp_now() - started_at
    assert elapsed == SLURP_IDLE_TIMEOUT_NANOSECONDS, (
        f"slurp duration = {elapsed}, want {SLURP_IDLE_TIMEOUT_NANOSECONDS}"
    )
    await close_source(source)


@slurp_synctest
async def test_slurp_channel_resets_idle_timeout() -> None:
    output: asyncio.Queue[int] = asyncio.Queue(1)
    output.put_nowait(1)
    source = queue_source(output)
    task = asyncio.create_task(slurp(source))
    await slurp_wait()
    await slurp_sleep(SLURP_IDLE_TIMEOUT_NANOSECONDS // 2)
    output.put_nowait(2)
    await slurp_wait()
    await slurp_sleep(SLURP_IDLE_TIMEOUT_NANOSECONDS - 1)
    assert not task.done(), "slurp returned before a full idle interval after the last value"
    await slurp_sleep(1)
    assert task.done(), "slurp did not return after a full idle interval"
    values = task.result()
    assert values == [1, 2], f"slurp result: values={values}"
    await close_source(source)


@pytest.mark.parametrize("count", [99, 100, 101])
@slurp_synctest
async def test_slurp_channel_caps_batch(count: int) -> None:
    output: asyncio.Queue[int] = asyncio.Queue(count)
    want = list(range(count))
    for value in want:
        output.put_nowait(value)
    source = queue_source(output)
    started_at = slurp_now()
    got = await slurp(source)
    elapsed = slurp_now() - started_at
    batch_size = min(count, SLURP_MAX_ITEMS)
    assert got == want[:batch_size], f"slurped values = {got}, want {want[:batch_size]}"
    assert output.qsize() == count - batch_size, (
        f"remaining values = {output.qsize()}, want {count - batch_size}"
    )
    for value in want[batch_size:]:
        remaining = output.get_nowait()
        assert remaining == value, f"remaining value = {remaining}, want {value}"
    want_duration = 0
    if count < SLURP_MAX_ITEMS:
        want_duration = SLURP_IDLE_TIMEOUT_NANOSECONDS
    assert elapsed == want_duration, f"slurp duration = {elapsed}, want {want_duration}"
    await close_source(source)


@pytest.mark.parametrize("buffered", [False, True])
@slurp_synctest
async def test_slurp_channel_returns_accumulated_values_on_closure(buffered: bool) -> None:
    output: asyncio.Queue[str] = asyncio.Queue(2)
    want: list[str] = []
    if buffered:
        output.put_nowait("second")
        output.put_nowait("third")
        want.extend(["second", "third"])
    output.shutdown()
    started_at = slurp_now()
    got = await slurp(queue_source(output))
    assert got == want, f"slurp result = {got}; want {want}"
    elapsed = slurp_now() - started_at
    assert elapsed == 0, f"closure advanced time by {elapsed}"


@slurp_synctest
async def test_slurp_channel_returns_cancellation_cause() -> None:
    cause = "session stopped"
    source = queue_source(asyncio.Queue[str]())
    task = asyncio.create_task(slurp(source))
    await slurp_wait()
    started_at = slurp_now()
    task.cancel(cause)
    await slurp_wait()
    assert task.cancelled(), f"slurp error = {task!r}, want {cause!r}"
    with pytest.raises(asyncio.CancelledError) as caught:
        task.result()
    assert caught.value.args == (cause,), f"slurp error = {caught.value!r}, want {cause!r}"
    elapsed = slurp_now() - started_at
    assert elapsed == 0, f"cancellation advanced time by {elapsed}"
    await close_source(source)
