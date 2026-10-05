import asyncio
from collections.abc import AsyncIterator, Iterable

from msgspec.structs import replace

from unreal_agent.gojson import Raw
from unreal_agent.inbox.inbox import ID, Input

__all__ = ["ClosedError", "Inbox"]


class ClosedError(Exception):
    def __init__(self) -> None:
        super().__init__("inbox is closed")


class Inbox:
    def __init__(self, seen_ids: Iterable[ID] = ()) -> None:
        seen: set[ID] = set()
        for input_id in seen_ids:
            if input_id == "":
                raise ValueError("seen input ID is empty")
            seen.add(input_id)
        self._seen = seen
        self._queue: asyncio.Queue[Input] = asyncio.Queue()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def submit(self, input: Input) -> None:
        try:
            input.validate()
        except ValueError as error:
            raise ValueError(f"submit input: {error}") from error
        if self._closed:
            raise ClosedError
        if input.ID in self._seen:
            return
        self._seen.add(input.ID)
        self._queue.put_nowait(replace(input, Payload=Raw(bytes(input.Payload))))

    async def get(self) -> Input:
        try:
            return await self._queue.get()
        except asyncio.QueueShutDown:
            raise ClosedError from None

    def get_nowait(self) -> Input:
        if self._closed:
            raise ClosedError
        return self._queue.get_nowait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.shutdown(immediate=True)

    def __aiter__(self) -> AsyncIterator[Input]:
        return self

    async def __anext__(self) -> Input:
        try:
            return await self.get()
        except ClosedError:
            raise StopAsyncIteration from None
