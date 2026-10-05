import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

_TEARDOWN_TIMEOUT = 1.0


@pytest.fixture(autouse=True)
async def exact_timers() -> AsyncIterator[None]:
    """Fire timers only at their deadline, and require every task to exit afterwards.

    asyncio runs a timer early when it is within one clock resolution of
    now, which breaks nanosecond fake-time deadlines. Go's fake_time also
    fails a test whose bubble goroutines do not exit once the test context
    is canceled; teardown cancels leftover tasks and fails on any that stay
    pending.
    """
    cast(Any, asyncio.get_running_loop())._clock_resolution = 1e-12
    yield
    current = asyncio.current_task()
    leftover = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in leftover:
        task.cancel()
    if leftover:
        _, pending = await asyncio.wait(leftover, timeout=_TEARDOWN_TIMEOUT)
        assert not pending, f"tasks did not exit after cancellation: {pending!r}"
