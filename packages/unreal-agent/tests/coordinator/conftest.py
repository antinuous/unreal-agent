import asyncio
from typing import Any, cast

import pytest


@pytest.fixture(autouse=True)
async def exact_timers() -> None:
    """Fire timers only at their deadline, as Go's synctest does.

    asyncio runs a timer early when it is within one clock resolution of
    now, which breaks nanosecond fake-time deadlines.
    """
    cast(Any, asyncio.get_running_loop())._clock_resolution = 1e-12
