import asyncio
from typing import cast

import pytest
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ClosedError, Inbox, Input, InputKind


def make_input(input_id: str) -> Input:
    return Input(ID=input_id, Kind=InputKind.EXTERNAL)


async def receive(inbox: Inbox) -> Input:
    async with asyncio.timeout(1):
        return await inbox.get()


async def submit_and_receive(inbox: Inbox, input: Input) -> Input:
    await inbox.submit(input)
    return await receive(inbox)


def assert_no_input(inbox: Inbox) -> None:
    with pytest.raises(asyncio.QueueEmpty):
        inbox.get_nowait()


async def test_inbox_outputs_every_input_kind() -> None:
    inbox = Inbox()
    for input in [
        Input(ID="external", Kind=InputKind.EXTERNAL),
        Input(ID="control", Kind=InputKind.CONTROL, Payload=Raw(b'{"Mode":"hard"}')),
        Input(ID="crash", Kind=InputKind.CRASH),
    ]:
        assert await submit_and_receive(inbox, input) == input


async def test_inbox_deduplicates_input_id() -> None:
    inbox = Inbox()
    first = Input(ID="same-input", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"message":"first"}'))
    duplicate = Input(
        ID="same-input",
        Kind=InputKind.CONTROL,
        Payload=Raw(b'{"Mode":"hard","Reason":"different"}'),
    )
    second = make_input("next-input")
    assert await submit_and_receive(inbox, first) == first
    await inbox.submit(duplicate)
    assert await submit_and_receive(inbox, second) == second
    assert_no_input(inbox)


async def test_inbox_deduplicates_queued_input_id() -> None:
    inbox = Inbox()
    first = Input(ID="same-input", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"value":"first"}'))
    duplicate = Input(
        ID="same-input", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"value":"duplicate"}')
    )
    await inbox.submit(first)
    await inbox.submit(duplicate)
    assert await receive(inbox) == first
    assert_no_input(inbox)


async def test_inbox_deduplicates_concurrent_submissions() -> None:
    inbox = Inbox()
    async with asyncio.TaskGroup() as group:
        for _ in range(100):
            group.create_task(inbox.submit(make_input("same-input")))
    assert (await receive(inbox)).ID == "same-input"
    assert_no_input(inbox)


async def test_inbox_accepts_concurrent_submissions() -> None:
    inbox = Inbox()
    count = 100

    async def consume() -> list[int]:
        return [int((await receive(inbox)).ID) for _ in range(count)]

    async with asyncio.TaskGroup() as group:
        consumer = group.create_task(consume())
        for index in range(count):
            group.create_task(inbox.submit(make_input(str(index))))
    assert sorted(consumer.result()) == list(range(count))


async def test_inbox_submit_does_not_wait_for_output_consumer() -> None:
    inbox = Inbox()
    for index in range(100):
        await inbox.submit(make_input(str(index)))
    for index in range(100):
        assert str(index) == (await receive(inbox)).ID


async def test_inbox_discards_recovered_ids() -> None:
    inbox = Inbox(["recovered", "recovered"])
    await inbox.submit(Input(ID="recovered", Kind=InputKind.EXTERNAL))
    assert (await submit_and_receive(inbox, Input(ID="new", Kind=InputKind.EXTERNAL))).ID == "new"
    assert_no_input(inbox)


async def test_inbox_honors_canceled_submission() -> None:
    inbox = Inbox()
    task = asyncio.create_task(inbox.submit(make_input("input-1")))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_no_input(inbox)
    assert (await submit_and_receive(inbox, make_input("input-1"))).ID == "input-1"


async def test_inbox_clones_submitted_input() -> None:
    inbox = Inbox()
    buffer = bytearray(b'{"value":"original"}')
    input = Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(buffer))
    got = await submit_and_receive(inbox, input)
    buffer[10] = ord("X")
    input.Payload = Raw(b'{"value":"changed"}')
    assert bytes(got.Payload) == b'{"value":"original"}'
    assert got is not input


async def test_inbox_stops_when_closed() -> None:
    inbox = Inbox()
    pending = asyncio.create_task(inbox.get())
    await asyncio.sleep(0)
    inbox.close()
    async with asyncio.timeout(1):
        with pytest.raises(ClosedError):
            await pending
    with pytest.raises(ClosedError):
        await inbox.get()
    assert [input async for input in inbox] == []
    with pytest.raises(ClosedError):
        await inbox.submit(make_input("after-stop"))


async def test_inbox_stops_with_queued_inputs() -> None:
    inbox = Inbox()
    for index in range(10):
        await inbox.submit(make_input(str(index)))
    inbox.close()
    async with asyncio.timeout(1):
        assert [input async for input in inbox] == []
    with pytest.raises(ClosedError):
        inbox.get_nowait()


async def test_inbox_async_iteration_yields_in_order() -> None:
    inbox = Inbox()
    for index in range(3):
        await inbox.submit(make_input(str(index)))
    got = []
    async for input in inbox:
        got.append(input.ID)
        if len(got) == 3:
            inbox.close()
    assert got == ["0", "1", "2"]


async def test_inbox_cancelled_get_does_not_consume() -> None:
    inbox = Inbox()
    pending = asyncio.create_task(inbox.get())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await inbox.submit(make_input("kept"))
    assert (await receive(inbox)).ID == "kept"


@pytest.mark.parametrize(
    ("input", "message"),
    [
        (Input(Kind=InputKind.EXTERNAL, Payload=Raw(b"{}")), "input ID is empty"),
        (Input(ID="unknown-kind", Kind=cast(InputKind, "unknown")), 'unsupported kind "unknown"'),
        (
            Input(ID="invalid-json", Kind=InputKind.EXTERNAL, Payload=Raw(b"{")),
            'input "invalid-json" payload is not valid JSON',
        ),
    ],
)
async def test_inbox_rejects_invalid_input(input: Input, message: str) -> None:
    with pytest.raises(ValueError, match=f"^submit input: .*{message}"):
        await Inbox().submit(input)


async def test_inbox_validates_before_closed_check() -> None:
    inbox = Inbox()
    inbox.close()
    with pytest.raises(ValueError, match="input ID is empty"):
        await inbox.submit(Input(Kind=InputKind.EXTERNAL))
    with pytest.raises(ClosedError):
        await inbox.submit(make_input("valid"))


def test_inbox_rejects_invalid_seen_id() -> None:
    with pytest.raises(ValueError, match="seen input ID is empty"):
        Inbox([""])
