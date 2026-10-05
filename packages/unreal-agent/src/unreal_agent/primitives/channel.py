import asyncio
from collections import deque


class Channel[T]:
    """An event sink with Go channel capacity semantics.

    With capacity 0 a `put` completes only when a `get` takes the item, like an
    unbuffered Go channel. A positive capacity behaves like a buffered channel.
    A cancelled `put` never enqueues its item, and a `put` whose item was
    already taken returns normally even if cancelled at the same moment.
    """

    __slots__ = ("_buffer", "_capacity", "_getters", "_putters")

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError("channel capacity must not be negative")
        self._capacity = capacity
        self._buffer: deque[T] = deque()
        self._getters: deque[asyncio.Future[T]] = deque()
        self._putters: deque[tuple[T, asyncio.Future[None]]] = deque()

    def qsize(self) -> int:
        """Items that a `get` could take without waiting."""
        return len(self._buffer) + len(self._putters)

    def empty(self) -> bool:
        return self.qsize() == 0

    async def put(self, item: T) -> None:
        if self._hand_off(item):
            return
        if len(self._buffer) < self._capacity:
            self._buffer.append(item)
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (item, waiter)
        self._putters.append(entry)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                return
            if entry in self._putters:
                self._putters.remove(entry)
            raise

    def _hand_off(self, item: T) -> bool:
        if self._buffer:
            return False
        while self._getters:
            getter = self._getters.popleft()
            if not getter.done():
                getter.set_result(item)
                return True
        return False

    def get_nowait(self) -> T:
        if self._buffer:
            item = self._buffer.popleft()
            self._refill()
            return item
        while self._putters:
            item, waiter = self._putters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return item
        raise asyncio.QueueEmpty

    def _refill(self) -> None:
        while self._putters and len(self._buffer) < self._capacity:
            item, waiter = self._putters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                self._buffer.append(item)

    async def get(self) -> T:
        try:
            return self.get_nowait()
        except asyncio.QueueEmpty:
            pass
        getter: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        self._getters.append(getter)
        try:
            return await getter
        except asyncio.CancelledError:
            if getter.done() and not getter.cancelled():
                item = getter.result()
                self._buffer.appendleft(item)
            elif getter in self._getters:
                self._getters.remove(getter)
            raise
