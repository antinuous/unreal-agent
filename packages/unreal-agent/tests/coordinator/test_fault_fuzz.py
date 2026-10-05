"""Property test for coordinator dependency faults, ported from Go's FuzzCoordinatorFaults.

The Go seed corpus is the fuzz target's `f.Add` calls. Each example runs in a
fresh looptime event loop, the equivalent of one `fake_time.Test` bubble, so
hour-long operation delays cost no real time. The real Responses adapter runs
over an `httpx.MockTransport`.
"""

import asyncio
import contextlib
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Final, Self, cast

import httpx
import looptime
import msgspec
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent._go import go_quote
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import Coordinator, Dependencies
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind
from unreal_agent.llm import Model, ToolCall
from unreal_agent.llm.responsesapi import Config, ResponsesAdapter
from unreal_agent.operation import ID as OperationID
from unreal_agent.operation import Operation, SkillUseState, Status, decode_skill_use
from unreal_agent.primitives import RemoteClient
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Session, Turn, TurnID
from unreal_agent.sessionstore import (
    Item,
    ItemKind,
    ModelResponse,
    Observer,
    ObserverID,
    Page,
    ResumeState,
    Sequence,
    SessionInfo,
    Snapshot,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.tool import SKILL_USE_NAME, Skill, StaticTranslators, new_registry

from tests.coordinator._helpers import chain

MILLISECOND: Final = 0.001
HOUR: Final = 3600.0
MAX_INT64: Final = 2**63 - 1


class CoordinatorFuzzFault(Exception):
    """Go's errCoordinatorFuzzFault; each trace raises its own instance."""

    def __init__(self) -> None:
        super().__init__("injected coordinator dependency failure")


@dataclass(slots=True)
class CoordinatorFaultEvent:
    kind: str
    value: Any


@dataclass(slots=True)
class CoordinatorFaultTrace:
    site: str
    remaining: int
    events: list[CoordinatorFaultEvent] = field(default_factory=list)
    failed: bool = False
    error: CoordinatorFuzzFault = field(default_factory=CoordinatorFuzzFault)

    def record(self, kind: str, value: Any) -> None:
        self.events.append(CoordinatorFaultEvent(kind, value))

    def fail(self, site: str) -> CoordinatorFuzzFault | None:
        if self.failed or site != self.site:
            return None
        if self.remaining > 0:
            self.remaining -= 1
            return None
        self.failed = True
        self.events.append(CoordinatorFaultEvent("failure", site))
        return self.error


class FaultContext:
    """The fuzz run's Go context: cancelling it runs every registered cancel function."""

    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.done = asyncio.Event()
        self._callbacks: list[Callable[[], object]] = []

    def on_cancel(self, callback: Callable[[], object]) -> None:
        if self.error is not None:
            callback()
            return
        self._callbacks.append(callback)

    def cancel(self, error: BaseException) -> None:
        if self.error is not None:
            return
        self.error = error
        self.done.set()
        for callback in self._callbacks:
            callback()

    def check(self) -> None:
        if isinstance(self.error, TimeoutError):
            raise TimeoutError
        if self.error is not None:
            raise asyncio.CancelledError


def clone_fault_operation(value: Operation) -> Operation:
    return replace(value)


def clone_fault_item(item: Item) -> Item:
    data = item.Data
    if isinstance(data, Input | Turn):
        data = replace(data)
    elif isinstance(data, ModelResponse):
        response = data.Response
        data = replace(
            data,
            Response=replace(
                response,
                Usage=replace(response.Usage),
                Output=[replace(output) for output in (response.Output or [])],
            ),
        )
    elif isinstance(data, ToolCallStatus):
        data = replace(
            data,
            Status=replace(data.Status, WaitingFor=list(data.Status.WaitingFor or [])),
            Operations=[clone_fault_operation(value) for value in data.Operations],
        )
    return replace(item, Data=data)


class FaultMemoryStore:
    """A single-session store that commits whole values or fails without changing history.

    It has no coordinator lifecycle or scheduling logic.
    """

    def __init__(self, ctx: FaultContext, trace: CoordinatorFaultTrace) -> None:
        self.ctx = ctx
        self.trace = trace
        self.snapshot = Snapshot()
        self.history: list[Item] = []
        self.operations: dict[OperationID, Operation] = {}
        self.observers: dict[ObserverID, Observer] = {}

    def add_observer(self, observer: Observer | None, /) -> ObserverID:
        id = uuid.uuid4()
        if observer is not None:
            self.observers[id] = observer
        return id

    def remove_observer(self, id: ObserverID, /) -> None:
        self.observers.pop(id, None)

    async def create(self, id: SessionID, /) -> Snapshot:
        self.ctx.check()
        if id == "" or self.snapshot.session.ID != "":
            raise StoreError("fuzz store needs one fresh session")
        self.snapshot.session = Session(ID=id, CreatedAt=gojson.Timestamp.now())
        self.operations = {}
        return Snapshot(self.snapshot.session)

    async def inspect(self, id: SessionID, /) -> Snapshot:
        self._check(id)
        return Snapshot(self.snapshot.session)

    async def list_sessions(self) -> list[SessionInfo]:
        self.ctx.check()
        if self.snapshot.session.ID == "":
            return []
        updated = self.snapshot.session.CreatedAt
        if self.history:
            updated = self.history[-1].RecordedAt
        return [SessionInfo(id=self.snapshot.session.ID, last_updated_at=updated)]

    async def items(self, id: SessionID, after: Sequence, limit: int, /) -> Page:
        self._check(id)
        if (error := self.trace.fail("history")) is not None:
            raise error
        if limit <= 0:
            raise StoreError("invalid page size")
        start = min(after, len(self.history))
        end = min(start + limit, len(self.history))
        page = Page(next_after=after, more=end < len(self.history))
        for item in self.history[start:end]:
            page.items.append(clone_fault_item(item))
            page.next_after = item.Sequence
        return page

    async def append_input(self, id: SessionID, input: Input, /) -> None:
        await self._append(id, "input", ItemKind.INPUT, input)

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        await self._append(id, "turn", ItemKind.TURN, turn)

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        await self._append(id, "response", ItemKind.MODEL_RESPONSE, response)

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        await self._append(id, "status", ItemKind.TOOL_CALL_STATUS, status)

    async def _append(
        self,
        id: SessionID,
        site: str,
        kind: ItemKind,
        value: Input | Turn | ModelResponse | ToolCallStatus,
    ) -> None:
        self._check(id)
        if (error := self.trace.fail(site)) is not None:
            raise error
        item = clone_fault_item(
            Item(
                Sequence=len(self.history) + 1,
                RecordedAt=gojson.Timestamp.now(),
                Kind=kind,
                Data=value,
            )
        )
        if isinstance(item.Data, ToolCallStatus):
            for operation in item.Data.Operations:
                self.operations[operation.ID] = clone_fault_operation(operation)
        self.history.append(item)
        self.trace.record("commit", clone_fault_item(item))
        for observe in list(self.observers.values()):
            observe(id, clone_fault_item(item))

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        self._check(id)
        if operation.ID not in self.operations:
            raise StoreError(f"operation {go_quote(operation.ID)} was never committed")
        if (error := self.trace.fail("save")) is not None:
            raise error
        self.operations[operation.ID] = clone_fault_operation(operation)
        self.trace.record("save", clone_fault_operation(operation))

    async def resume(self, id: SessionID, /) -> ResumeState:
        raise StoreError("fuzz store does not support resume")

    async def fork(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID, /
    ) -> Snapshot:
        raise StoreError("fuzz store does not support forks")

    def _check(self, id: SessionID) -> None:
        self.ctx.check()
        if id == "" or id != self.snapshot.session.ID:
            raise StoreError("unknown fuzz session")


@dataclass(slots=True)
class FaultOperationPlan:
    path: str
    result: str
    delay: float
    fail: bool
    repeat: bool


class FaultOperationUpdates:
    """The manager's never-closed update channel as an async iterator."""

    def __init__(self, queue: asyncio.Queue[Operation]) -> None:
        self._queue = queue

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Operation:
        return await self._queue.get()


class ControlledFaultOperations:
    def __init__(self, ctx: FaultContext, trace: CoordinatorFaultTrace, capacity: int) -> None:
        self.ctx = ctx
        self.trace = trace
        self.plans: dict[str, FaultOperationPlan] = {}
        self.queue: asyncio.Queue[Operation] = asyncio.Queue(capacity)
        self.started = asyncio.Event()
        self.cancels: dict[OperationID, asyncio.Event] = {}
        self.tasks: list[asyncio.Task[None]] = []
        ctx.on_cancel(self._cancel_all)

    def _cancel_all(self) -> None:
        for cancelled in self.cancels.values():
            cancelled.set()

    async def add(self, operation: Operation, /) -> None:
        self.trace.record("add", clone_fault_operation(operation))
        if (error := self.trace.fail("add")) is not None:
            raise error
        state = decode_skill_use(operation)
        plan = self.plans.get(state.Path)
        if plan is None:
            raise RuntimeError("operation has no execution plan")
        if operation.ID in self.cancels:
            return
        cancelled = asyncio.Event()
        if self.ctx.error is not None:
            cancelled.set()
        self.cancels[operation.ID] = cancelled
        self.trace.record("start", clone_fault_operation(operation))
        self.started.set()
        self.tasks.append(
            asyncio.create_task(self._execute(replace(operation), state, plan, cancelled))
        )

    async def _execute(
        self,
        value: Operation,
        state: SkillUseState,
        plan: FaultOperationPlan,
        cancelled: asyncio.Event,
    ) -> None:
        try:
            value.Status = Status.AWAITING
            await self._emit(value)
            try:
                async with asyncio.timeout(plan.delay):
                    await cancelled.wait()
            except TimeoutError:
                if plan.fail:
                    value.Status = Status.FAILED
                    state.TerminalError = plan.result
                else:
                    value.Status = Status.COMPLETED
                    state.Content = plan.result.encode()
            else:
                if self.ctx.error is not None:
                    return
                value.Status = Status.CANCELED
                state.TerminalError = "operation canceled"
            value.State = gojson.Raw(gojson.encode(state))
            await self._emit(value)
            if plan.repeat:
                await self._emit(value)
        finally:
            cancelled.set()

    async def cancel(self, id: OperationID, reason: str, /) -> None:
        self.trace.record("cancel", id)
        if (error := self.trace.fail("cancel")) is not None:
            raise error
        cancelled = self.cancels.get(id)
        if cancelled is None:
            raise RuntimeError("operation canceled before dispatch")
        cancelled.set()

    async def _emit(self, value: Operation) -> None:
        self.trace.record("update", clone_fault_operation(value))
        if not self.queue.full():
            self.queue.put_nowait(clone_fault_operation(value))
            return
        put = asyncio.create_task(self.queue.put(clone_fault_operation(value)))
        done = asyncio.create_task(self.ctx.done.wait())
        await asyncio.wait((put, done), return_when=asyncio.FIRST_COMPLETED)
        put.cancel()
        done.cancel()
        await asyncio.wait((put, done))

    def updates(self) -> FaultOperationUpdates:
        return FaultOperationUpdates(self.queue)


class CoordinatorFaultStream(httpx.AsyncByteStream):
    """Go's coordinatorFaultReader: every read fails with the injected fault."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aiter__(self) -> Any:
        raise self.error
        yield b""


class CoordinatorFaultTransport:
    def __init__(self, trace: CoordinatorFaultTrace, block: bool) -> None:
        self.trace = trace
        self.bodies: list[bytes] = []
        self.started = asyncio.Event()
        self.block = block
        self.calls = 0

    async def round_trip(self, request: httpx.Request) -> httpx.Response:
        encoded = await request.aread()
        self.trace.record("request", encoded)
        index = self.calls
        self.calls += 1
        if index == 0:
            self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if (error := self.trace.fail("transport")) is not None:
            raise httpx.ConnectError(str(error), request=request) from error
        if self.trace.fail("http") is not None:
            return httpx.Response(503, content=b'{"error":{"message":"injected HTTP failure"}}')
        if (error := self.trace.fail("body")) is not None:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=CoordinatorFaultStream(error),
            )
        if index >= len(self.bodies):
            raise httpx.ConnectError("coordinator requested an extra model response")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=self.bodies[index],
        )


def fault_json(value: Any) -> bytes:
    return gojson.encode(value, deterministic=True)


class FaultRequestOutput(msgspec.Struct):
    type: str | None = None
    text: str | None = None


class FaultRequestInput(msgspec.Struct):
    type: str | None = None
    call_id: str | None = None
    output: list[FaultRequestOutput] | None = None


class FaultRequest(msgspec.Struct):
    input: list[FaultRequestInput] | None = None


def assert_coordinator_fault_trace(
    events: list[CoordinatorFaultEvent],
    results: dict[str, str],
    usage: bytes,
    tokens: int,
    hard_stop: bool,
    settled: bool,
) -> None:
    previous: TurnID = ""
    turns = requests = responses = 0
    failed = False
    calls: dict[str, TurnID] = {}
    committed: dict[OperationID, Operation] = {}
    started: set[OperationID] = set()
    updates: dict[OperationID, list[Operation]] = {}
    statuses: dict[OperationID, Operation] = {}
    responded: set[TurnID] = set()
    for event in events:
        continued = failed and event.kind in ("commit", "save", "request", "add")
        assert not continued, f"coordinator continued {event.kind!r} after dependency failure"
        match event.kind:
            case "failure":
                failed = True
            case "commit":
                data = cast(Item, event.value).Data
                if isinstance(data, Turn):
                    message = f"broken turn chain: {data!r}"
                    assert data.ID != "", message
                    assert data.PreviousTurnID == previous, message
                    previous = data.ID
                    turns += 1
                elif isinstance(data, ModelResponse):
                    message = f"response duplicated or associated with the wrong request: {data!r}"
                    assert data.TurnID not in responded, message
                    assert data.TurnID == previous, message
                    assert f"response-{requests - 1}" == data.Response.ID, message
                    responded.add(data.TurnID)
                    responses += 1
                    got = data.Response.Usage
                    message = f"usage changed between HTTP response and persistence: {got!r}"
                    assert got.InputTokens == tokens & MAX_INT64, message
                    assert got.OutputTokens == tokens % 997, message
                    assert bytes(got.Raw) == usage, message
                    for output in data.Response.Output or []:
                        if isinstance(output.Data, ToolCall):
                            calls[output.Data.CallID] = data.TurnID
                elif isinstance(data, ToolCallStatus):
                    message = f"status lacks its committed call or operation: {data!r}"
                    assert data.CallID in calls, message
                    assert data.TurnID == calls[data.CallID], message
                    assert data.Status.Error == "", message
                    assert len(data.Operations) == 1, message
                    assert len(data.Status.WaitingFor or []) == 1, message
                    assert (data.Status.WaitingFor or [])[0] == data.Operations[0].ID, message
                    op = data.Operations[0]
                    prior = committed.get(op.ID)
                    if prior is not None:
                        assert op == prior, "tool status differs from the saved operation"
                    else:
                        assert op.Status == Status.READY, (
                            "operation was initialized after execution"
                        )
                    committed[op.ID] = statuses[op.ID] = op
            case "request":
                requests += 1
                message = "HTTP request preceded its committed turn or followed a hard stop"
                assert requests == turns, message
                assert not (hard_stop and requests > 1), message
                if requests > 1:
                    request = msgspec.json.decode(cast(bytes, event.value), type=FaultRequest)
                    delivered: dict[str, str] = {}
                    for item in request.input or []:
                        if item.type == "function_call_output":
                            call_id = item.call_id or ""
                            assert call_id not in delivered, "tool result delivered more than once"
                            output = item.output or []
                            assert len(output) == 1, f"unexpected tool output: {output!r}"
                            assert output[0].type == "input_text", (
                                f"unexpected tool output: {output!r}"
                            )
                            delivered[call_id] = output[0].text or ""
                    assert delivered == results, (
                        f"model received wrong tool results: got {delivered!r}, want {results!r}"
                    )
            case "add":
                value = cast(Operation, event.value)
                assert committed.get(value.ID) == value, (
                    "operation dispatched before its status and state committed"
                )
            case "start":
                value = cast(Operation, event.value)
                assert value.ID not in started, "operation started more than once"
                started.add(value.ID)
            case "cancel":
                assert cast(OperationID, event.value) in started, (
                    "operation canceled before the manager knew it"
                )
            case "update":
                value = cast(Operation, event.value)
                updates.setdefault(value.ID, []).append(value)
            case "save":
                value = cast(Operation, event.value)
                found = any(update == value for update in updates.get(value.ID, []))
                message = "saved operation was never reported by the manager"
                assert value.ID in started, message
                assert found, message
                committed[value.ID] = value
    assert turns == requests, f"committed turns={turns}, HTTP requests={requests}"
    if settled:
        want_responses = 1 if hard_stop else 2
        message = f"settled execution: responses={responses}, started operations={len(started)}"
        assert responses == want_responses, message
        assert len(started) == len(results), message
        for id in started:
            value = committed.get(id, Operation())
            message = "coordinator stopped before recording terminal operation and tool status"
            assert value.Status in (Status.COMPLETED, Status.FAILED, Status.CANCELED), message
            assert statuses.get(id, Operation()) == value, message
            assert not hard_stop or value.Status == Status.CANCELED, (
                "hard stop did not cancel the pending operation"
            )


async def run_coordinator_fault_case(
    actions: bytes, mode: int, occurrence: int, text: str, tokens: int
) -> None:
    loop = asyncio.get_running_loop()
    # Fire timers only at their deadline, as Go's fake_time does (see conftest.exact_timers).
    cast(Any, loop)._clock_resolution = 1e-12
    if not actions:
        actions = b"\x00"
    mode %= 14
    hard_stop = mode in (8, 12)
    ctx = FaultContext()
    deadline = loop.call_later(60, lambda: ctx.cancel(TimeoutError()))
    sites = ["", "history", "input", "turn", "response", "status", "save", "add", "cancel"]
    sites += ["transport", "http", "body", "", ""]
    n = len(actions)
    counts = [1, 1, 2, 2, 2, 2 * n, 2 * n, n, n, 2, 2, 2, 1, 1]
    trace = CoordinatorFaultTrace(site=sites[mode], remaining=occurrence % counts[mode])
    store = FaultMemoryStore(ctx, trace)
    id: SessionID = "faults"
    await store.create(id)
    inputs = Inbox()
    ctx.on_cancel(inputs.close)
    await inputs.submit(
        Input(ID="user", Kind=InputKind.EXTERNAL, Payload=gojson.Raw(fault_json(text)))
    )
    await inputs.submit(
        Input(
            ID="idle",
            Kind=InputKind.CONTROL,
            Payload=gojson.Raw(fault_json(ControlMessage(Mode=ControlMode.STOP_WHEN_IDLE))),
        )
    )
    registry = new_registry(StaticTranslators(), SKILL_USE_NAME)
    manager = ControlledFaultOperations(ctx, trace, n * 4)
    wire_calls: list[dict[str, Any]] = []
    results: dict[str, str] = {}
    for index, action in enumerate(actions):
        name = f"skill-{index}"
        plan = FaultOperationPlan(
            path="/fuzz/" + name,
            result=f"result-{index}: {text}",
            delay=(1 + action % 16) * MILLISECOND,
            fail=action & 16 != 0,
            repeat=action & 32 != 0,
        )
        if hard_stop:
            plan.delay = HOUR
        manager.plans[plan.path] = plan
        registry.register_skill(
            Skill(name=name, description="Controlled operation fixture.", path=plan.path)
        )
        call_id = f"call-{index}"
        results[call_id] = plan.result
        wire_calls.append(
            {
                "type": "function_call",
                "id": "provider-" + call_id,
                "call_id": call_id,
                "name": SKILL_USE_NAME,
                "arguments": fault_json({"name": name}).decode(),
            }
        )
    usage = fault_json({"input_tokens": tokens & MAX_INT64, "output_tokens": tokens % 997})
    transport = CoordinatorFaultTransport(trace, block=mode == 13)
    for index, output in enumerate([wire_calls, []]):
        response = {
            "id": f"response-{index}",
            "status": "completed",
            "output": output,
            "usage": gojson.Raw(usage),
        }
        event = fault_json({"type": "response.completed", "response": response})
        transport.bodies.append(b"data: " + event + b"\n\n")
    remote = RemoteClient(httpx.AsyncClient(transport=httpx.MockTransport(transport.round_trip)))
    adapter = ResponsesAdapter(
        remote, Config(endpoint="https://fuzz.invalid/responses", max_attempts=1)
    )
    builder = new_builder(*registry.skills())
    builder.set_model(Model(ID="fuzz-model"))
    for definition in registry.static_definitions():
        builder.add_tool(definition.tool)
    current = Coordinator(
        Dependencies(
            session_id=id,
            inbox=inputs,
            sessions=store,
            context_builder=builder,
            llm=adapter,
            tools=registry,
            operations=manager,
        )
    )
    background: list[asyncio.Task[None]] = []
    if hard_stop:
        background.append(asyncio.create_task(submit_fault_hard_stop(ctx, manager, inputs)))
    if mode == 13:
        background.append(asyncio.create_task(cancel_fault_run(ctx, transport)))

    run = asyncio.create_task(current.run())
    ctx.on_cancel(run.cancel)
    err: BaseException | None = None
    try:
        await run
    except asyncio.CancelledError:
        err = ctx.error if isinstance(ctx.error, TimeoutError) else asyncio.CancelledError()
    except Exception as error:
        err = error
    ctx.cancel(asyncio.CancelledError())
    deadline.cancel()
    await asyncio.gather(*background, *manager.tasks)
    await remote.aclose()

    if mode == 13:
        assert isinstance(err, asyncio.CancelledError), f"canceled HTTP request: {err!r}"
    elif trace.site != "":
        causes = chain(err)
        message = f"fault {trace.site!r} reached={trace.failed}, coordinator error={err!r}"
        assert trace.failed, message
        assert err is not None, message
        assert not any(isinstance(value, TimeoutError) for value in causes), message
        if mode < 9:
            assert any(value is trace.error for value in causes), (
                f"coordinator lost dependency error: {err!r}"
            )
    else:
        assert err is None, repr(err)
    assert_coordinator_fault_trace(trace.events, results, usage, tokens, hard_stop, err is None)


async def submit_fault_hard_stop(
    ctx: FaultContext, manager: ControlledFaultOperations, inputs: Inbox
) -> None:
    if not await wait_fault_event(ctx, manager.started):
        return
    payload = gojson.Raw(b'{"Mode":"hard","Reason":"fuzz stop"}')
    with contextlib.suppress(Exception):
        await inputs.submit(Input(ID="hard", Kind=InputKind.CONTROL, Payload=payload))


async def cancel_fault_run(ctx: FaultContext, transport: CoordinatorFaultTransport) -> None:
    if await wait_fault_event(ctx, transport.started):
        ctx.cancel(asyncio.CancelledError())


async def wait_fault_event(ctx: FaultContext, event: asyncio.Event) -> bool:
    """Go's `select` on an event and ctx.Done(); True when the event won."""
    waiter = asyncio.create_task(event.wait())
    done = asyncio.create_task(ctx.done.wait())
    await asyncio.wait((waiter, done), return_when=asyncio.FIRST_COMPLETED)
    waiter.cancel()
    done.cancel()
    await asyncio.wait((waiter, done))
    return event.is_set() and ctx.error is None


def run_in_fault_bubble(body: Callable[[], Coroutine[Any, Any, None]]) -> None:
    with asyncio.Runner(loop_factory=lambda: looptime.new_event_loop(resolution=1e-9)) as runner:
        runner.run(body())


def with_coordinator_fault_seeds[F: Callable[..., Any]](test: F) -> F:
    for mode in range(14):
        test = example(bytes([0, 17, 35]), mode, 0, 'result\n"会話', (1 << 53) + 1)(test)
        test = example(bytes([48, 2, 19]), mode, 255, "", MAX_INT64)(test)
    return example(b"", 0, 0, "", 0)(test)


@settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@with_coordinator_fault_seeds
@given(
    st.binary(max_size=8),
    st.integers(0, 255),
    st.integers(0, 255),
    st.text(max_size=4096),
    st.integers(0, 2**64 - 1),
)
def test_coordinator_faults(
    actions: bytes, mode: int, occurrence: int, text: str, tokens: int
) -> None:
    assume(len(actions) <= 8 and len(text.encode()) <= 4096)
    run_in_fault_bubble(lambda: run_coordinator_fault_case(actions, mode, occurrence, text, tokens))
