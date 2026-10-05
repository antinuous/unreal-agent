import asyncio

import pytest
from unreal_agent.llm import Adapter, Request, RequestOptions, Response, StopReason


class _Blocking:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.options: RequestOptions | None = None

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        self.options = options
        self.started.set()
        await asyncio.Event().wait()
        return Response(Stop=StopReason.COMPLETE)


async def test_adapter_cancellation_maps_to_task_cancellation() -> None:
    fake = _Blocking()
    adapter: Adapter = fake
    task = asyncio.create_task(adapter.respond(Request(), RequestOptions(CacheKey="key")))
    await fake.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.options == RequestOptions(CacheKey="key")
