"""Shared fakes and harnesses for the coordinator tests.

`@fake_time` runs a test under looptime with nanosecond resolution, `wait()`
runs the loop until nothing is ready, and `sleep_ns` advances fake time and
then waits.
"""

import asyncio
import bisect
import copy
import dataclasses
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Self, cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent._go import go_quote
from unreal_agent.contextbuilder import Builder, Result, new_builder
from unreal_agent.coordinator import Coordinator, Dependencies
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind, Settings
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    Request,
    RequestOptions,
    Response,
    Role,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.operation import ID as OperationID
from unreal_agent.operation import TYPE_SHELL, Operation, Spec, Status, new_value_spec
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Session, Turn, TurnID, TurnType
from unreal_agent.sessionstore import (
    Item,
    ItemData,
    ItemKind,
    ModelResponse,
    Observer,
    ObserverID,
    Page,
    ResumeState,
    SessionInfo,
    Snapshot,
    Store,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.sessionstore.localfile.state import (
    StoredState,
    fork_stored_state,
    new_stored_state,
    validate_operation,
)
from unreal_agent.tool import (
    BASH_NAME,
    VIEW_IMAGE_NAME,
    CallStatus,
    Context,
    Registry,
    SkillUseResult,
    StaticTranslators,
)
from unreal_agent.tool import Result as ToolOutcome

# Nanosecond resolution so fake-time deadlines land exactly.
fake_time = pytest.mark.looptime(resolution=1e-9)

NANOSECOND = 1
MILLISECOND = 1_000_000
SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
SLURP_IDLE_TIMEOUT_NS = MILLISECOND
TOOL_CALL_RUN_GRACE_PERIOD_NS = SECOND


async def wait() -> None:
    """Run the loop until no callback is ready or due."""
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


def now_ns() -> int:
    return round(asyncio.get_running_loop().time() * SECOND)


async def sleep_ns(duration: int) -> None:
    if duration > 0:
        await asyncio.sleep(duration / SECOND)
    await wait()


async def advance(duration: int) -> None:
    await wait()
    await sleep_ns(duration)
    await wait()


def chain(error: BaseException | None) -> list[BaseException]:
    """The error with every cause, context, and group member."""
    seen: list[BaseException] = []
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or any(current is value for value in seen):
            continue
        seen.append(current)
        pending.extend((current.__cause__, current.__context__))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return seen


def caused_by(error: BaseException | None, target: BaseException | type[BaseException]) -> bool:
    for value in chain(error):
        if isinstance(target, type):
            if isinstance(value, target):
                return True
        elif value is target:
            return True
    return False


def clone[T](value: T, kind: type[T]) -> T:
    """A deep copy through the Go-compatible codec."""
    return gojson.decode(gojson.encode(value), kind)


def encode_json(value: Any) -> gojson.Raw:
    return gojson.Raw(gojson.encode(value))


def external_event(id: str, text: str) -> Input:
    return Input(ID=id, Kind=InputKind.EXTERNAL, Payload=encode_json(text))


def stored_item(sequence: int, kind: ItemKind | str, data: Any) -> Item:
    return Item(Sequence=sequence, Kind=cast(ItemKind, kind), Data=data)


def stop_input(id: str, mode: ControlMode | str) -> Input:
    message = ControlMessage(Mode=cast(ControlMode, mode), Reason="user requested stop")
    return Input(ID=id, Kind=InputKind.CONTROL, Payload=encode_json(message))


def heartbeat_input(id: str) -> Input:
    message = ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="Heartbeat: waiting for tools.")
    return Input(ID=id, Kind=InputKind.CONTROL, Payload=encode_json(message))


def settings_input(id: str, settings: Settings) -> Input:
    message = ControlMessage(Mode=ControlMode.UPDATE_SETTINGS, Parameters=settings)
    return Input(ID=id, Kind=InputKind.CONTROL, Payload=encode_json(message))


def text_response(text: str) -> Response:
    return Response(
        Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=text))]
    )


def text_result(call_id: str, value: str, running: bool = False) -> LLMItem:
    return LLMItem(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID=call_id,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)],
            Running=running,
        ),
    )


def user_message(text: str) -> LLMItem:
    return LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text))


def with_preamble(*items: LLMItem) -> list[LLMItem]:
    """The items after the preamble every context builder starts with."""
    return [*new_builder().build().request.Input, *items]


def value_spec(value: bytes = b"1") -> Spec:
    return new_value_spec(gojson.Raw(value))


class TestTranslator:
    __test__ = False

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        return SkillUseResult(Error="error:" + status.Error)


class SubmittingTranslator:
    def __init__(
        self, specs: list[Spec] | None = None, on_translate: Callable[[], None] | None = None
    ) -> None:
        self.specs: list[Spec] = list(specs or [])
        self.calls: list[ToolCall] = []
        self.on_translate = on_translate

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        self.calls.append(call)
        if self.on_translate is not None:
            self.on_translate()
        return CallStatus(WaitingFor=[ctx.submit(spec) for spec in self.specs])

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        return SkillUseResult(CallID=call_id, Error=status.Error)


class OperationStatusTranslator:
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        statuses = ",".join(str(value.Status) for value in operations)
        return SkillUseResult(Content=statuses.encode())


class FailingResultTranslator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        raise self.error


class TerminalResultTranslator:
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        for value in operations:
            if value.Status in (Status.COMPLETED, Status.FAILED, Status.CANCELED):
                raise RuntimeError("terminal result failed")
        return SkillUseResult(Content=b"pending")


class SubmissionTranslator(SubmittingTranslator):
    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        return OperationStatusTranslator().translate_result(call_id, status, operations)


class SubmittingResultErrorTranslator(SubmittingTranslator):
    def __init__(self, error: Exception, specs: list[Spec] | None = None) -> None:
        super().__init__(specs)
        self.error = error

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolOutcome:
        raise self.error


@dataclass(slots=True)
class ItemRequest:
    after: int
    limit: int


@dataclass(eq=False)
class FakeStore:
    resume: ResumeState = field(default_factory=ResumeState)
    history: list[Item] = field(default_factory=list)
    items_error: Exception | None = None
    items_page: Page | None = None
    item_requests: list[ItemRequest] = field(default_factory=list)
    appended_inputs: list[Input] = field(default_factory=list)
    appended_turns: list[Turn] = field(default_factory=list)
    appended_responses: list[ModelResponse] = field(default_factory=list)
    appended_statuses: list[ToolCallStatus] = field(default_factory=list)
    saved_operations: list[Operation] = field(default_factory=list)
    append_input_error: Exception | None = None
    append_turn_error: Exception | None = None
    append_model_response_error: Exception | None = None
    append_status_error: Exception | None = None
    save_operation_error: Exception | None = None
    unexpected_mutations: list[str] = field(default_factory=list)
    on_append_input: Callable[[Input], None] | None = None
    on_append_turn: Callable[[Turn], None] | None = None
    on_append_model_response: Callable[[ModelResponse], None] | None = None
    on_append_tool_call_status: Callable[[ToolCallStatus], None] | None = None
    on_save_operation: Callable[[Operation], None] | None = None

    def add_observer(self, observer: Observer, /) -> ObserverID:
        return uuid.uuid4()

    def remove_observer(self, id: ObserverID, /) -> None:
        pass

    async def create(self, id: SessionID, /) -> Snapshot:
        self.unexpected_mutations.append("create")
        raise StoreError("unexpected create")

    async def list_sessions(self) -> list[SessionInfo]:
        raise StoreError("unexpected list sessions")

    async def inspect(self, id: SessionID, /) -> Snapshot:
        return self.resume.snapshot

    async def items(self, id: SessionID, after: int, limit: int, /) -> Page:
        self.item_requests.append(ItemRequest(after, limit))
        if self.items_error is not None:
            raise self.items_error
        if self.items_page is not None:
            return self.items_page
        start = bisect.bisect_right([item.Sequence for item in self.history], after)
        end = min(start + limit, len(self.history))
        items = list(self.history[start:end])
        next_after = items[-1].Sequence if items else after
        return Page(items=items, next_after=next_after, more=end < len(self.history))

    async def append_input(self, id: SessionID, input: Input, /) -> None:
        self.appended_inputs.append(copy.deepcopy(input))
        if self.on_append_input is not None:
            self.on_append_input(input)
        if self.append_input_error is not None:
            raise self.append_input_error

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        self.appended_turns.append(copy.deepcopy(turn))
        if self.on_append_turn is not None:
            self.on_append_turn(turn)
        if self.append_turn_error is not None:
            raise self.append_turn_error

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        self.appended_responses.append(copy.deepcopy(response))
        if self.on_append_model_response is not None:
            self.on_append_model_response(response)
        if self.append_model_response_error is not None:
            raise self.append_model_response_error

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        self.appended_statuses.append(copy.deepcopy(status))
        if self.on_append_tool_call_status is not None:
            self.on_append_tool_call_status(status)
        if self.append_status_error is not None:
            raise self.append_status_error

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        self.saved_operations.append(copy.deepcopy(operation))
        if self.on_save_operation is not None:
            self.on_save_operation(operation)
        if self.save_operation_error is not None:
            raise self.save_operation_error

    async def fork(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID, /
    ) -> Snapshot:
        self.unexpected_mutations.append("fork")
        raise StoreError("unexpected fork")


def empty_fake_store() -> FakeStore:
    return FakeStore(resume=ResumeState(snapshot=Snapshot(session=Session(ID="session-1"))))


def as_store(store: object) -> Store:
    return cast(Store, store)


class MemoryStore:
    """The localfile store's state machine without files or threads.

    Values are copied through the JSON codec on every write and read, as the
    file round trip does.
    """

    def __init__(self) -> None:
        self._states: dict[SessionID, StoredState] = {}
        self._observers: dict[ObserverID, Observer] = {}

    def add_observer(self, observer: Observer, /) -> ObserverID:
        id = uuid.uuid4()
        self._observers[id] = observer
        return id

    def remove_observer(self, id: ObserverID, /) -> None:
        self._observers.pop(id, None)

    async def create(self, id: SessionID, /) -> Snapshot:
        state = new_stored_state(id, gojson.Timestamp.now())
        self._states[id] = state
        return dataclasses.replace(state.head.snapshot)

    async def list_sessions(self) -> list[SessionInfo]:
        return [SessionInfo(id=id) for id in self._states]

    async def inspect(self, id: SessionID, /) -> Snapshot:
        return dataclasses.replace(self._state(id).head.snapshot)

    async def items(self, id: SessionID, after: int, limit: int, /) -> Page:
        if limit <= 0:
            raise StoreError("item page limit must be positive")
        if after < 0:
            raise StoreError("item page cursor is negative")
        state = self._state(id)
        count = len(state.items)
        start = min(after, count)
        end = min(start + limit, count)
        items = [clone(item, Item) for item in state.items[start:end]]
        return Page(
            items=items, next_after=items[-1].Sequence if items else after, more=end < count
        )

    async def append_input(self, id: SessionID, input: Input, /) -> None:
        state = self._state(id)
        state.append_input(clone(input, Input), gojson.Timestamp.now())
        self._notify(id, state.items[-1])

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        state = self._state(id)
        state.append_turn(clone(turn, Turn), gojson.Timestamp.now())
        self._notify(id, state.items[-1])

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        state = self._state(id)
        state.append_model_response(clone(response, ModelResponse), gojson.Timestamp.now())
        self._notify(id, state.items[-1])

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        state = self._state(id)
        status = clone(status, ToolCallStatus)
        state.append_tool_call_status(status, list(status.Operations), gojson.Timestamp.now())
        self._notify(id, state.items[-1])

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        validate_operation(operation)
        self._state(id).save_operation(clone(operation, Operation))

    async def resume(self, id: SessionID, /) -> ResumeState:
        resumed = self._state(id).resume()
        return ResumeState(
            snapshot=dataclasses.replace(resumed.snapshot),
            operations=[clone(value, Operation) for value in resumed.operations],
            external_input_ids=list(resumed.external_input_ids),
        )

    async def fork(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID, /
    ) -> Snapshot:
        if id == parent_id:
            raise StoreError(f"fork session {go_quote(id)} onto itself")
        state = fork_stored_state(
            self._state(parent_id), id, previous_turn_id, gojson.Timestamp.now()
        )
        self._states[id] = state
        self._notify(id, state.items[-1])
        return dataclasses.replace(state.head.snapshot)

    def _state(self, id: SessionID) -> StoredState:
        state = self._states.get(id)
        if state is None:
            raise StoreError(f"read session {go_quote(id)}: file does not exist")
        return state

    def _notify(self, id: SessionID, item: Item) -> None:
        for observer in list(self._observers.values()):
            observer(id, clone(item, Item))


class FailingStore:
    """Wraps a store and fails selected writes."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.input_error: Exception | None = None
        self.turn_error: Exception | None = None
        self.response_error: Exception | None = None
        self.status_error: Exception | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    async def append_input(self, id: SessionID, input: Input, /) -> None:
        if self.input_error is not None:
            raise self.input_error
        await self.store.append_input(id, input)

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        if self.turn_error is not None:
            raise self.turn_error
        await self.store.append_turn(id, turn)

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        if self.response_error is not None:
            raise self.response_error
        await self.store.append_model_response(id, response)

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        if self.status_error is not None:
            raise self.status_error
        await self.store.append_tool_call_status(id, status)


class QueueIterator[T]:
    def __init__(self, queue: asyncio.Queue[T]) -> None:
        self._queue = queue

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            return await self._queue.get()
        except asyncio.QueueShutDown:
            raise StopAsyncIteration from None


class FakeOperationManager:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Operation] = asyncio.Queue()
        self.adds: list[Operation] = []
        self.add_error: Callable[[Operation], Exception | None] | None = None
        self.mutate_adds = False
        self.cancels: list[OperationID] = []
        self.cancel_reasons: list[str] = []
        self.cancel_error: Exception | None = None

    async def add(self, operation: Operation, /) -> None:
        self.adds.append(operation)
        if self.mutate_adds:
            operation.State = gojson.Raw(b"!" + bytes(operation.State)[1:])
            operation.Idempotency = gojson.Raw(b"!" + bytes(operation.Idempotency)[1:])
        if self.add_error is not None and (error := self.add_error(operation)) is not None:
            raise error

    async def cancel(self, id: OperationID, reason: str, /) -> None:
        self.cancels.append(id)
        self.cancel_reasons.append(reason)
        if self.cancel_error is not None:
            raise self.cancel_error

    def updates(self) -> AsyncIterator[Operation]:
        return QueueIterator(self.queue)

    def send(self, operation: Operation) -> None:
        self.queue.put_nowait(operation)

    def close(self) -> None:
        self.queue.shutdown()


type RespondFunction = Callable[[Request], Awaitable[Response]]


class FakeAdapter:
    def __init__(self, respond: RespondFunction | None = None) -> None:
        self.requests: list[Request] = []
        self.request_options: list[RequestOptions] = []
        self.respond_function = respond

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        self.requests.append(request)
        self.request_options.append(options)
        if self.respond_function is None:
            raise RuntimeError("unexpected respond")
        return await self.respond_function(request)


class FailingBuilder:
    """A builder whose `build` fails; every other call goes to `builder`."""

    def __init__(self, builder: Builder, error: Exception) -> None:
        self.builder = builder
        self.error = error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.builder, name)

    def build(self) -> Result:
        raise self.error


def as_builder(builder: object) -> Builder:
    return cast(Builder, builder)


async def ignoring_cancellation[T](future: asyncio.Future[T]) -> T:
    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done():
                return future.result()


def new_test_inbox() -> Inbox:
    return Inbox()


async def submit_test_input(inputs: Inbox, input: Input) -> None:
    await inputs.submit(input)


def new_test_coordinator(
    store: FakeStore | Store,
    inputs: Inbox,
    operations: FakeOperationManager,
    builder: Builder,
    registry: Registry,
    adapter: FakeAdapter | None = None,
) -> Coordinator:
    resume = store.resume if isinstance(store, FakeStore) else ResumeState()
    return Coordinator(
        Dependencies(
            session_id="session-1",
            inbox=inputs,
            restored=resume,
            sessions=as_store(store),
            context_builder=builder,
            llm=adapter if adapter is not None else FakeAdapter(),
            tools=registry,
            operations=operations,
        )
    )


def deps(current: Coordinator) -> Dependencies:
    return current._dependencies


def independent_tool_calls(count: int) -> tuple[FakeStore, Registry]:
    registry = Registry(StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME)
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(2, ItemKind.MODEL_RESPONSE, ModelResponse()),
    ]
    output: list[LLMItem] = []
    response = ModelResponse(TurnID="turn-1", Response=Response(Output=output))
    for index in range(count):
        value = Operation(
            ID=f"operation-{index}", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING
        )
        call = ToolCall(CallID=f"call-{index}", Name=VIEW_IMAGE_NAME, Arguments="{}")
        output.append(LLMItem(Type=ItemType.TOOL_CALL, Data=call))
        store.resume.operations.append(replace(value))
        store.history.append(
            stored_item(
                len(store.history) + 1,
                ItemKind.TOOL_CALL_STATUS,
                ToolCallStatus(
                    TurnID="turn-1",
                    CallID=call.CallID,
                    Status=CallStatus(WaitingFor=[value.ID]),
                    Operations=[replace(value)],
                ),
            )
        )
    store.history[1].Data = response
    return store, registry


def assert_completed_results(request: Request, count: int) -> None:
    completed: dict[str, int] = {}
    for item in request.Input:
        if item.Type == ItemType.TOOL_RESULT:
            result = cast(ToolResult, item.Data)
            if result.Output[0].Value == Status.COMPLETED:
                completed[result.CallID] = completed.get(result.CallID, 0) + 1
    assert len(completed) == count, f"completed results = {completed}, want {count}"
    for index in range(count):
        assert completed.get(f"call-{index}") == 1, (
            f"completed results = {completed}, want each call once"
        )


def assert_stop_result(request: Request, call_id: str, want: str) -> None:
    for item in request.Input:
        if item.Type == ItemType.TOOL_RESULT:
            result = cast(ToolResult, item.Data)
            if result.CallID == call_id and result.Output[0].Value == want:
                return
    raise AssertionError(f"request lacks result {want!r} for {call_id!r}")


@dataclass(eq=False)
class StopTestCall:
    task: asyncio.Task[Any]
    request: Request
    response: asyncio.Future[Response]

    @property
    def canceled(self) -> bool:
        return self.task.cancelled() or self.task.cancelling() > 0


class StopTestRun:
    def __init__(
        self,
        current: Coordinator,
        store: FakeStore,
        inputs: Inbox,
        operations: FakeOperationManager,
    ) -> None:
        self.current = current
        self.store = store
        self.inputs = inputs
        self.operations = operations
        self.calls: list[StopTestCall] = []
        self.task: asyncio.Task[None] | None = None

    @property
    def dependencies(self) -> Dependencies:
        return self.current._dependencies

    def go(self) -> asyncio.Task[None]:
        """Start `run` without waiting for it."""
        self.task = asyncio.create_task(self.current.run())
        return self.task

    async def start(self) -> None:
        self.go()
        await wait()
        await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        await wait()

    async def input(self, *inputs: Input) -> None:
        for input in inputs:
            await submit_test_input(self.inputs, input)
        await advance_after_event()

    async def update(self, index: int, status: Status) -> None:
        self.operations.send(replace(self.store.resume.operations[index], Status=status))
        await advance_after_event()

    async def respond(self, index: int, response: Response) -> None:
        assert index < len(self.calls), f"request {index} has not started"
        future = self.calls[index].response
        assert not future.done(), f"request {index} is no longer waiting"
        future.set_result(response)
        await advance_after_event()

    def assert_running(self) -> None:
        assert self.task is not None
        if self.task.done():
            raise AssertionError(f"Run exited early: {self.finished_error()!r}")

    def assert_stopped(self) -> None:
        assert self.task is not None
        assert self.task.done(), "Run did not stop"
        error = self.finished_error()
        assert error is None, f"Run error = {error!r}"

    def finished_error(self) -> BaseException | None:
        """The error `run` returned; fails when `run` has not returned."""
        assert self.task is not None
        assert self.task.done(), "Run did not return"
        try:
            self.task.result()
        except BaseException as error:
            return error
        return None

    async def error(self) -> BaseException | None:
        assert self.task is not None
        await asyncio.wait((self.task,))
        return self.finished_error()

    def cancel(self, message: str | None = None) -> None:
        assert self.task is not None
        self.task.cancel(message)

    def request_count(self) -> int:
        return len(cast(FakeAdapter, self.dependencies.llm).requests)


async def advance_after_event() -> None:
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()


def new_stop_test_run[R: StopTestRun](pending: int, kind: type[R] = StopTestRun) -> R:
    store, registry = independent_tool_calls(pending)
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry)
    run = kind(current, store, inputs, operations)

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        run.calls.append(call)
        return await call.response

    run.dependencies.llm = FakeAdapter(respond)
    return run


def stop_test_adapter(run: StopTestRun, *, ignore_cancellation: bool = False) -> FakeAdapter:
    """The stop-test model; with `ignore_cancellation` it answers even after interruption."""

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        run.calls.append(call)
        if ignore_cancellation:
            return await ignoring_cancellation(call.response)
        return await call.response

    return FakeAdapter(respond)


class HeartbeatTestRun(StopTestRun):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.recorded_inputs: list[Input] = []
        self.store.on_append_input = self.recorded_inputs.append


def new_heartbeat_test_run(pending: int) -> HeartbeatTestRun:
    return new_stop_test_run(pending, HeartbeatTestRun)


def assert_heartbeat_count(run: HeartbeatTestRun, want: int) -> None:
    count = 0
    ids: set[str] = set()
    for input in run.recorded_inputs:
        if input.Kind != InputKind.CONTROL:
            continue
        if input.decode_control_message().Mode == ControlMode.HEARTBEAT:
            count += 1
            assert input.ID != "", "heartbeat IDs must be nonempty"
            assert input.ID not in ids, "heartbeat IDs must be unique"
            ids.add(input.ID)
    assert count == want, f"heartbeats = {count}, want {want}"


def latest_heartbeat_reason(run: HeartbeatTestRun) -> str:
    for input in reversed(run.recorded_inputs):
        if input.Kind != InputKind.CONTROL:
            continue
        control = input.decode_control_message()
        if control.Mode == ControlMode.HEARTBEAT:
            return control.Reason
    raise AssertionError("no recorded heartbeat")


def count_heartbeat_messages(request: Request) -> int:
    count = 0
    for item in request.Input:
        if item.Type == ItemType.MESSAGE:
            message = cast(Message, item.Data)
            if message.Role == Role.USER and message.Text.startswith("Heartbeat:"):
                count += 1
    return count


async def advance_heartbeat_time(duration: int) -> None:
    await advance(duration)


def new_tool_grace_test_run() -> StopTestRun:
    run = new_stop_test_run(0)
    translator = SubmissionTranslator([value_spec()])
    run.dependencies.tools = Registry(
        StaticTranslators(bash=translator, view_image=OperationStatusTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    return run


def tool_grace_response(*call_ids: str) -> Response:
    return Response(
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID=id, Name=BASH_NAME, Arguments="{}"),
            )
            for id in call_ids
        ]
    )


async def update_tool_grace_call(run: StopTestRun, call_id: str, terminal: Status) -> None:
    for status in run.store.appended_statuses:
        if status.CallID == call_id:
            run.operations.send(replace(status.Operations[0], Status=terminal))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
            return
    raise AssertionError(f"tool call {call_id!r} was not scheduled")


async def persist_test_run(run: StopTestRun) -> MemoryStore:
    store = MemoryStore()
    await store.create("session-1")
    await store.append_turn("session-1", cast(Turn, run.store.history[0].Data))
    await store.append_model_response("session-1", cast(ModelResponse, run.store.history[1].Data))
    for item in run.store.history[2:]:
        await store.append_tool_call_status("session-1", cast(ToolCallStatus, item.Data))
    return store


async def restore_test_run(run: StopTestRun, store: Store | MemoryStore | FailingStore) -> None:
    restored = await as_store(store).resume("session-1")
    run.dependencies.sessions = as_store(store)
    run.dependencies.restored = restored


def item_data(item: Item) -> ItemData:
    assert item.Data is not None
    return item.Data
