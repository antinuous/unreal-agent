# ruff: noqa: ASYNC230
"""Property test ported from Go's FuzzCoordinatorLogMatchesExecution.

Each example runs in a fresh looptime event loop (one `fake_time.Test`
bubble) with the store's blocking work inlined, so no worker threads let
fake time jump. The Go seed corpus is the `@example` list.
"""

import asyncio
import io
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, cast

import looptime
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from msgspec.structs import replace
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import Coordinator, Dependencies
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind
from unreal_agent.llm import Message, Model, RequestOptions, Response, Role
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.operation import LocalOperationManager
from unreal_agent.runner.run import SessionObserver, open_datetime_log
from unreal_agent.runner.streams import TeeWriter
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Item, ModelResponse
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.tool import StaticTranslators, new_registry

from tests.runner._helpers import (
    copy_log_response,
    decode_log_items,
    fuzz_log_response,
    fuzz_log_usage,
    log_json,
    read_log_history,
    to_valid_utf8,
    wait,
    with_log_fuzz_seeds,
    without_threads,
)


@dataclass(slots=True)
class LogModelCall:
    index: int
    request: LLMRequest
    response: asyncio.Future[Response]
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class GatedLogModel:
    def __init__(self) -> None:
        self.calls: asyncio.Queue[LogModelCall] = asyncio.Queue()
        self.requests = 0
        self.returned: dict[int, Response] = {}

    async def respond(self, request: LLMRequest, options: RequestOptions) -> Response:
        index = self.requests
        self.requests += 1
        call = LogModelCall(index, request, asyncio.get_running_loop().create_future())
        try:
            self.calls.put_nowait(call)
            response = await call.response
        except asyncio.CancelledError:
            call.cancelled = True
            raise
        finally:
            call.finished.set()
        self.returned[index] = copy_log_response(response)
        return response


def assert_coordinator_execution_log(
    items: list[Item], submitted: list[Input], requests: int, returned: dict[int, Response]
) -> None:
    inputs: list[Input] = []
    previous = ""
    recorded_at = None
    turns = 0
    responses: dict[int, Response] = {}
    turn_ids: set[str] = set()
    for index, item in enumerate(items):
        assert item.Sequence == index + 1, item
        assert not item.RecordedAt.is_zero(), item
        assert recorded_at is None or not item.RecordedAt < recorded_at, item
        recorded_at = item.RecordedAt
        match item.Data:
            case Input() as value:
                inputs.append(value)
            case Turn() as value:
                assert value.ID, f"invalid turn chain: {value!r} after {previous!r}"
                assert value.ID not in turn_ids, f"invalid turn chain: {value!r} after {previous!r}"
                assert value.PreviousTurnID == previous, (
                    f"invalid turn chain: {value!r} after {previous!r}"
                )
                assert value.Type == TurnType.REGULAR, (
                    f"invalid turn chain: {value!r} after {previous!r}"
                )
                previous = value.ID
                turn_ids.add(value.ID)
                turns += 1
            case ModelResponse() as value:
                assert turns > 0, f"response duplicated or attached to the wrong turn: {value!r}"
                assert turns - 1 not in responses, (
                    f"response duplicated or attached to the wrong turn: {value!r}"
                )
                assert value.TurnID == previous, (
                    f"response duplicated or attached to the wrong turn: {value!r}"
                )
                responses[turns - 1] = value.Response
            case _:
                raise AssertionError(f"unexpected logged item: {item!r}")
    assert inputs == submitted, "logged inputs lost, changed, or duplicated submitted inputs"
    assert turns == requests, "logged turns differ from actual model requests"
    assert responses == returned, (
        "logged responses differ from completed model responses by request (including usage)"
    )


async def run_coordinator_log_case(
    actions: bytes, text: str, input: int, cached: int, written: int, output: int, reasoning: int
) -> None:
    loop = asyncio.get_running_loop()
    # Fire timers only at their deadline, as Go's fake_time does.
    cast(Any, loop)._clock_resolution = 1e-12
    text = to_valid_utf8(text)
    usage = fuzz_log_usage(text, input, cached, written, output, reasoning)
    id = "coordinator-execution"
    with (
        tempfile.TemporaryDirectory() as directory,
        tempfile.TemporaryDirectory() as log_directory,
        without_threads(),
    ):
        async with asyncio.timeout(60):
            store = Store(directory)
            await store.create(id)
            inputs = Inbox()
            file = open_datetime_log(log_directory, time.time())
            stdout = io.BytesIO()
            tasks: list[asyncio.Task[None]] = []

            def cancel() -> None:
                for task in tasks:
                    task.cancel()

            observer = SessionObserver(id, TeeWriter(file, stdout), cancel)
            observer_id = store.add_observer(observer.observe)
            builder = new_builder()
            builder.set_model(Model(ID="journal-model"))
            model = GatedLogModel()
            operations = LocalOperationManager()
            coordinator = Coordinator(
                Dependencies(
                    session_id=id,
                    inbox=inputs,
                    sessions=store,
                    context_builder=builder,
                    llm=model,
                    tools=new_registry(StaticTranslators()),
                    operations=operations,
                )
            )
            submitted: list[Input] = []
            messages: list[str] = []

            async def submit(value: Input) -> None:
                source = replace(value)
                await inputs.submit(value)
                submitted.append(source)

            async def submit_message() -> None:
                message = f"{len(messages)}: {text}"
                await submit(
                    Input(
                        ID=f"input-{len(messages)}",
                        Kind=InputKind.EXTERNAL,
                        Payload=log_json(message),
                    )
                )
                messages.append(message)

            await submit_message()
            done = asyncio.create_task(coordinator.run())
            tasks.append(done)
            requests = 0

            async def next_call() -> LogModelCall:
                nonlocal requests
                getter = asyncio.create_task(model.calls.get())
                await asyncio.wait((getter, done), return_when=asyncio.FIRST_COMPLETED)
                if not getter.done():
                    getter.cancel()
                    raise AssertionError(
                        f"coordinator stopped before the next model request: {done!r}"
                    )
                call = getter.result()
                assert call.index == requests, "model request index"
                requests += 1
                delivered = [
                    item.Data.Text
                    for item in call.request.Input
                    if isinstance(item.Data, Message) and item.Data.Role == Role.USER
                ]
                assert delivered == messages, (
                    "model request lost, changed, or duplicated submitted messages"
                )
                return call

            async def complete(call: LogModelCall, variant: int) -> None:
                assert not call.finished.is_set(), "model request was interrupted before completion"
                call.response.set_result(fuzz_log_response(text, usage, call.index, variant))
                # The coordinator ends the request when it accepts its result.
                # Wait for that boundary before submitting the next input.
                await call.finished.wait()
                await wait()

            pending = await next_call()
            for action in actions:
                match action % 3:
                    case 0:
                        await complete(pending, action)
                        await submit_message()
                        pending = await next_call()
                    case 1:
                        interrupted = pending
                        await submit_message()
                        pending = await next_call()
                        assert interrupted.cancelled, (
                            "steering did not interrupt the pending model request"
                        )
                    case _:
                        duplicate = replace(submitted[-1], Payload=log_json("duplicate: " + text))
                        await inputs.submit(duplicate)
                        await wait()
                        assert not pending.finished.is_set(), (
                            "duplicate input interrupted the model request"
                        )
            mode = actions[-1] % 3 if actions else 0
            if mode == 2:
                done.cancel()
            else:
                stop = ControlMode.STOP_HARD if mode == 1 else ControlMode.STOP_WHEN_IDLE
                await submit(
                    Input(
                        ID="stop",
                        Kind=InputKind.CONTROL,
                        Payload=log_json(ControlMessage(Mode=stop, Reason=text)),
                    )
                )
                if mode == 0:
                    await complete(pending, len(actions))
            await asyncio.wait([done])
            if mode == 2:
                assert done.cancelled(), "coordinator error, want cancellation"
            else:
                assert not done.cancelled(), done
                assert done.exception() is None, done
            inputs.close()
            await operations.aclose()
            await wait()
            store.remove_observer(observer_id)
            file.close()
            assert observer.error is None, observer.error
            with open(file.name, "rb") as reopened_file:
                logged = reopened_file.read()
            assert logged == stdout.getvalue(), "file log differs from stdout"
            items = decode_log_items(logged)
            reopened = Store(directory)
            history = await read_log_history(reopened, id, 1 + len(actions) % 7)
            assert history == items, "file log differs from reopened session history"
            assert model.requests == requests, "model requests differ from driver requests"
            assert_coordinator_execution_log(items, submitted, requests, model.returned)


def run_in_bubble(body: Any) -> None:
    with asyncio.Runner(loop_factory=lambda: looptime.new_event_loop(resolution=1e-9)) as runner:
        runner.run(body)


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@with_log_fuzz_seeds
@given(
    st.binary(max_size=16),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=512),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
)
def test_coordinator_log_matches_execution(
    actions: bytes, text: str, input: int, cached: int, written: int, output: int, reasoning: int
) -> None:
    size = len(text.encode())
    assume(len(actions) <= 16 and size <= 128 << 10 and len(actions) * size <= 256 << 10)
    run_in_bubble(
        run_coordinator_log_case(actions, text, input, cached, written, output, reasoning)
    )
