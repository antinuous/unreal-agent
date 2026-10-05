import asyncio

import pytest
from unreal_agent.primitives import Channel


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_unbuffered_put_waits_for_receiver() -> None:
    channel: Channel[int] = Channel()
    put = asyncio.create_task(channel.put(1))
    await settle()
    assert not put.done()
    assert channel.qsize() == 1
    assert await channel.get() == 1
    await put
    assert channel.empty()


async def test_unbuffered_put_hands_off_to_waiting_receiver() -> None:
    channel: Channel[int] = Channel()
    get = asyncio.create_task(channel.get())
    await settle()
    await channel.put(7)
    assert await get == 7


async def test_buffered_put_blocks_when_full() -> None:
    channel: Channel[int] = Channel(1)
    await channel.put(1)
    put = asyncio.create_task(channel.put(2))
    await settle()
    assert not put.done()
    assert channel.get_nowait() == 1
    await settle()
    assert put.done()
    assert channel.get_nowait() == 2


async def test_canceled_put_does_not_enqueue() -> None:
    channel: Channel[int] = Channel()
    put = asyncio.create_task(channel.put(1))
    await settle()
    put.cancel()
    with pytest.raises(asyncio.CancelledError):
        await put
    assert channel.empty()
    with pytest.raises(asyncio.QueueEmpty):
        channel.get_nowait()


async def test_put_taken_before_cancellation_completes() -> None:
    channel: Channel[int] = Channel()
    put = asyncio.create_task(channel.put(1))
    await settle()
    assert channel.get_nowait() == 1
    put.cancel()
    await put


async def test_canceled_get_keeps_received_item() -> None:
    channel: Channel[int] = Channel()
    get = asyncio.create_task(channel.get())
    await settle()
    await channel.put(3)
    get.cancel()
    with pytest.raises(asyncio.CancelledError):
        await get
    assert channel.get_nowait() == 3


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        Channel(-1)
