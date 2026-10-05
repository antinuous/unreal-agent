"""Ports of the Go coordinator tests in harness/coordinator.

Fake time follows Go's testing/synctest: `@synctest` runs a test under
looptime with nanosecond resolution, `wait()` runs the loop until nothing
is ready, and `sleep_ns` advances fake time and then waits.
"""

import asyncio
import bisect
import contextlib
import dataclasses
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD, Builder, Result, new_builder
from unreal_agent.coordinator import Coordinator, CoordinatorError, Dependencies
from unreal_agent.coordinator.loop import (
    HISTORY_PAGE_SIZE,
    ToolCallKey,
    ToolCallState,
    _closed_input_error,
)
from unreal_agent.inbox import (
    ClosedError,
    ControlMessage,
    ControlMode,
    Inbox,
    Input,
    InputKind,
    Settings,
)
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    Model,
    Reasoning,
    ReasoningEffort,
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
from unreal_agent.operation import (
    TYPE_SHELL,
    LocalOperationManager,
    Operation,
    ShellInput,
    ShellState,
    Spec,
    Status,
    UnsupportedError,
    new_shell_spec,
    new_value_spec,
)
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Session, Turn, TurnID, TurnType
from unreal_agent.sessionstore import (
    BEFORE_FIRST,
    NIL_OBSERVER_ID,
    Fork,
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
from unreal_agent.sessionstore.localfile import Store as LocalFileStore
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
    StaticTranslators,
    new_registry,
)
from unreal_agent.tool._go import go_quote

# --- coordinator_support ------------------------------------------------------

# Go's testing/synctest: fake time with nanosecond resolution so deadlines land exactly.
synctest = pytest.mark.looptime(resolution=1e-9)

NANOSECOND = 1
MILLISECOND = 1_000_000
SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
SLURP_IDLE_TIMEOUT_NS = MILLISECOND
TOOL_CALL_RUN_GRACE_PERIOD_NS = SECOND


# --- Fake time -------------------------------------------------------------------


async def wait() -> None:
    """Go's `synctest.Wait`: run the loop until no callback is ready or due."""
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
    """Go's `synctest.Sleep`: advance fake time by `duration`, then `wait()`."""
    if duration > 0:
        await asyncio.sleep(duration / SECOND)
    await wait()


async def advance(duration: int) -> None:
    await wait()
    await sleep_ns(duration)
    await wait()


# --- Errors ----------------------------------------------------------------------


def chain(error: BaseException | None) -> list[BaseException]:
    """The error with every cause, context, and group member, like Go's errors.Is walk."""
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


# --- Values ----------------------------------------------------------------------


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


def text_result(call_id: str, value: str) -> LLMItem:
    return LLMItem(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID=call_id, Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]
        ),
    )


def user_message(text: str) -> LLMItem:
    return LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text))


def with_preamble(*items: LLMItem) -> list[LLMItem]:
    """The items after the preamble every context builder starts with."""
    return [*new_builder().build().request.Input, *items]


def value_spec(value: bytes = b"1") -> Spec:
    return new_value_spec(gojson.Raw(value))


# --- Translators -----------------------------------------------------------------


class TestTranslator:
    __test__ = False

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        return ToolResult(
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="error:" + status.Error)]
        )


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

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        return ToolResult(
            CallID=call_id,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=status.Error)],
        )


class OperationStatusTranslator:
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        statuses = ",".join(str(value.Status) for value in operations)
        return ToolResult(Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=statuses)])


class FailingResultTranslator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        raise self.error


class TerminalResultTranslator:
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        for value in operations:
            if value.Status in (Status.COMPLETED, Status.FAILED, Status.CANCELED):
                raise RuntimeError("terminal result failed")
        return ToolResult(Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="pending")])


class SubmissionTranslator(SubmittingTranslator):
    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        return OperationStatusTranslator().translate_result(call_id, status, operations)


class SubmittingResultErrorTranslator(SubmittingTranslator):
    def __init__(self, error: Exception, specs: list[Spec] | None = None) -> None:
        super().__init__(specs)
        self.error = error

    def translate_result(self, call_id: str, status: CallStatus, operations: Any, /) -> ToolResult:
        raise self.error


# --- Fake store --------------------------------------------------------------------


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

    def add_observer(self, observer: Observer | None, /) -> ObserverID:
        return NIL_OBSERVER_ID

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
        self.appended_inputs.append(input)
        if self.on_append_input is not None:
            self.on_append_input(input)
        if self.append_input_error is not None:
            raise self.append_input_error

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        self.appended_turns.append(turn)
        if self.on_append_turn is not None:
            self.on_append_turn(turn)
        if self.append_turn_error is not None:
            raise self.append_turn_error

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        self.appended_responses.append(response)
        if self.on_append_model_response is not None:
            self.on_append_model_response(response)
        if self.append_model_response_error is not None:
            raise self.append_model_response_error

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        self.appended_statuses.append(status)
        if self.on_append_tool_call_status is not None:
            self.on_append_tool_call_status(status)
        if self.append_status_error is not None:
            raise self.append_status_error

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        self.saved_operations.append(operation)
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


# --- In-memory session store -------------------------------------------------------


class MemoryStore:
    """The localfile store's state machine without files or threads.

    Values are copied through the JSON codec on every write and read, as the
    file round trip does.
    """

    def __init__(self) -> None:
        self._states: dict[SessionID, StoredState] = {}
        self._observers: dict[ObserverID, Observer] = {}

    def add_observer(self, observer: Observer | None, /) -> ObserverID:
        if observer is None:
            return NIL_OBSERVER_ID
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
    """Wraps a store and fails selected writes, like the Go embedding fakes."""

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


# --- Fake operation manager and model ----------------------------------------------


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
    """Wait for `future` even if the waiting task is cancelled, like a Go call that ignores ctx."""
    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done():
                return future.result()


# --- Coordinator construction --------------------------------------------------------


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
    registry = new_registry(
        StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME
    )
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(2, ItemKind.MODEL_RESPONSE, ModelResponse()),
    ]
    response = ModelResponse(TurnID="turn-1")
    for index in range(count):
        value = Operation(
            ID=f"operation-{index}", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING
        )
        call = ToolCall(CallID=f"call-{index}", Name=VIEW_IMAGE_NAME, Arguments="{}")
        response.Response.Output.append(LLMItem(Type=ItemType.TOOL_CALL, Data=call))
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


# --- Running coordinators --------------------------------------------------------------


@dataclass(eq=False)
class StopTestCall:
    task: asyncio.Task[Any]
    request: Request
    response: asyncio.Future[Response]

    @property
    def canceled(self) -> bool:
        """Go's `ctx.Err() != nil` for the request context."""
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
        """Start `run` without waiting, like `go func() { run.done <- Run(ctx) }()`."""
        self.task = asyncio.create_task(self.current.run())
        return self.task

    async def start(self) -> None:
        self.go()
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
        """The error `run` returned; fails when `run` has not returned (Go's select default)."""
        assert self.task is not None
        assert self.task.done(), "Run did not return"
        try:
            self.task.result()
        except BaseException as error:
            return error
        return None

    async def error(self) -> BaseException | None:
        """Wait for `run` to return its error (Go's `<-run.done`)."""
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


# --- Heartbeats --------------------------------------------------------------------------


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


# --- Tool grace ------------------------------------------------------------------------


def new_tool_grace_test_run() -> StopTestRun:
    run = new_stop_test_run(0)
    translator = SubmissionTranslator([value_spec()])
    run.dependencies.tools = new_registry(
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


# --- Persistence -------------------------------------------------------------------------


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


# --- test_loop_state ----------------------------------------------------------


def loop_state_store(*items: Item, operations: list[Operation] | None = None) -> FakeStore:
    return FakeStore(
        resume=ResumeState(
            snapshot=Snapshot(session=Session(ID="session-1")),
            operations=list(operations or []),
        ),
        history=list(items),
    )


def loop_state_user_message(text: str) -> LLMItem:
    return LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text))


def loop_state_text_result(call_id: str, value: str) -> LLMItem:
    return LLMItem(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID=call_id, Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]
        ),
    )


async def test_coordinator_restores_session() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    input = external_event("input-1", "hello")
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    response = Response(
        ID="response-1",
        Output=[
            LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="working")),
            LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        ],
    )
    status = ToolCallStatus(
        TurnID=turn.ID, CallID=call.CallID, Status=CallStatus(Error="invalid arguments")
    )
    resumed_operation = Operation(
        ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING
    )
    store = loop_state_store(
        stored_item(1, ItemKind.INPUT, input),
        stored_item(2, ItemKind.TURN, turn),
        stored_item(3, ItemKind.MODEL_RESPONSE, ModelResponse(TurnID=turn.ID, Response=response)),
        stored_item(4, ItemKind.TOOL_CALL_STATUS, status),
        operations=[resumed_operation],
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()

    state = current._state
    assert state.current_turn_id == turn.ID, f"restored state = {state!r}"
    assert state.current_turn_type == TurnType.REGULAR, f"restored state = {state!r}"
    assert state.operations.get(resumed_operation.ID) == resumed_operation, (
        f"restored state = {state!r}"
    )

    built = builder.build()
    want_input = with_preamble(
        loop_state_user_message("hello"),
        response.Output[0],
        response.Output[1],
        loop_state_text_result(call.CallID, "error:invalid arguments"),
    )
    assert built.request.Input == want_input, (
        f"built request = {built.request!r}, want input {want_input!r}"
    )
    assert store.item_requests == [ItemRequest(0, HISTORY_PAGE_SIZE)], (
        f"item requests = {store.item_requests!r}"
    )


def test_coordinator_applies_turn_types() -> None:
    current = new_stop_test_run(0).current
    for turn_type in (
        TurnType.REGULAR,
        TurnType.COMPACTION,
        TurnType.REGULAR,
        TurnType.COMPACTION,
    ):
        turn = Turn(ID="turn", Type=turn_type)
        current._add_item_to_local_state(Item(Kind=ItemKind.TURN, Data=turn))
        assert current._state.current_turn_type == turn_type, (
            f"current turn type = {current._state.current_turn_type!r}, want {turn_type!r}"
        )


async def test_coordinator_restores_ordinary_response_during_compaction() -> None:
    run = new_stop_test_run(0)
    response = text_response("Ordinary response")
    response.Output.append(
        LLMItem(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="old-call", Name="unknown"))
    )
    run.store.history = [
        stored_item(1, ItemKind.INPUT, external_event("input", "hello")),
        stored_item(2, ItemKind.TURN, Turn(ID="ordinary", Type=TurnType.REGULAR)),
        stored_item(
            3,
            ItemKind.TURN,
            Turn(ID="compact", PreviousTurnID="ordinary", Type=TurnType.COMPACTION),
        ),
        stored_item(
            4, ItemKind.MODEL_RESPONSE, ModelResponse(TurnID="ordinary", Response=response)
        ),
    ]
    await run.current._load_history()
    built = run.dependencies.context_builder.build()
    want = [*with_preamble(loop_state_user_message("hello")), *response.Output]
    message = "earlier ordinary response was lost or treated as a compaction response"
    assert built.request.Input == want, message
    assert run.current._state.current_turn_type == TurnType.COMPACTION, message
    assert run.current._pending_inputs() == 1, message
    assert len(run.current._state.tool_calls) == 1, message


async def test_coordinator_restores_paginated_fork_history() -> None:
    parent_input = external_event("parent-input", "parent")
    items = [stored_item(1, ItemKind.INPUT, parent_input)]
    for sequence in range(2, HISTORY_PAGE_SIZE + 1):
        items.append(
            stored_item(sequence, ItemKind.TURN, Turn(ID=f"turn-{sequence}", Type=TurnType.REGULAR))
        )
    items.append(
        stored_item(
            HISTORY_PAGE_SIZE + 1,
            ItemKind.FORK,
            Fork(ParentID="parent", PreviousTurnID="turn-256"),
        )
    )
    items.append(
        stored_item(HISTORY_PAGE_SIZE + 2, ItemKind.INPUT, external_event("child-input", "child"))
    )
    store = loop_state_store(*items)
    builder = new_builder()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        builder,
        new_registry(StaticTranslators()),
    )

    await current._restore()

    assert store.item_requests == [
        ItemRequest(0, HISTORY_PAGE_SIZE),
        ItemRequest(HISTORY_PAGE_SIZE, HISTORY_PAGE_SIZE),
    ], f"item requests = {store.item_requests!r}"
    assert current._state.current_turn_id == "turn-256", (
        f"current turn ID = {current._state.current_turn_id!r}, want 'turn-256'"
    )
    built = builder.build()
    want = with_preamble(loop_state_user_message("parent"), loop_state_user_message("child"))
    assert built.request.Input == want, f"replayed input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_returns_session_history_error() -> None:
    store = empty_fake_store()
    store.items_error = RuntimeError("disk unavailable")
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )

    with pytest.raises(CoordinatorError) as caught:
        await current.run()
    assert str(caught.value) == "load session history after 0: disk unavailable", (
        f"restore error = {caught.value!r}"
    )


async def test_coordinator_rejects_session_history_without_progress() -> None:
    store = empty_fake_store()
    store.items_page = Page(more=True, next_after=BEFORE_FIRST)
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )

    with pytest.raises(CoordinatorError) as caught:
        await current._restore()
    assert str(caught.value) == "load session history did not advance after 0", (
        f"restore error = {caught.value!r}"
    )


async def test_coordinator_keeps_unreplayable_tool_status_in_local_state() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    status = ToolCallStatus(
        TurnID=turn.ID,
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=["terminal-operation"]),
    )
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, turn),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID=turn.ID,
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, status),
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()
    call_state = current._state.tool_calls.get(ToolCallKey(status.TurnID, status.CallID))
    assert call_state is not None, f"tool call state = {call_state!r}"
    assert call_state.status is not None, f"tool call state = {call_state!r}"
    assert call_state.status == status.Status, f"tool call state = {call_state!r}"
    built = builder.build()
    assert built.request.Input == with_preamble(LLMItem(Type=ItemType.TOOL_CALL, Data=call)), (
        f"replayed input = {built.request.Input!r}"
    )


async def test_coordinator_restores_completed_tool_call_from_status_snapshots() -> None:
    registry = new_registry(
        StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    initial = Operation(ID="operation-1", Type="test", Version=1, Status=Status.READY)
    completed = replace(initial, Status=Status.COMPLETED)
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[initial.ID])
    )
    initial_status = replace(status, Operations=[initial])
    completed_status = replace(status, Operations=[completed])
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, initial_status),
        stored_item(4, ItemKind.TOOL_CALL_STATUS, completed_status),
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()
    assert ToolCallKey("turn-1", call.CallID) not in current._state.tool_calls, (
        "completed tool call remains in local state"
    )
    assert current._state.operations.get(completed.ID) == completed, (
        f"restored operation = {current._state.operations.get(completed.ID)!r}, want {completed!r}"
    )
    built = builder.build()
    want = with_preamble(
        LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        loop_state_text_result(call.CallID, "completed"),
    )
    assert built.request.Input == want, f"replayed input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_overlays_resumed_operations_after_history_snapshots() -> None:
    registry = new_registry(
        StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    first = Operation(ID="operation-1", Type="test", Version=1, Status=Status.READY)
    second = Operation(ID="operation-2", Type="test", Version=1, Status=Status.READY)
    resumed_second = replace(second, Status=Status.AWAITING)
    status = ToolCallStatus(
        TurnID="turn-1",
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=[first.ID, second.ID]),
        Operations=[replace(first), replace(second)],
    )
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, status),
        operations=[resumed_second],
    )
    operations = FakeOperationManager()
    current = new_test_coordinator(store, new_test_inbox(), operations, new_builder(), registry)

    await current._restore()
    restored = current._state.operations
    assert restored.get(first.ID) == first, f"restored operations = {restored!r}"
    assert restored.get(second.ID) == resumed_second, f"restored operations = {restored!r}"
    await current._dispatch_operations_to_manager()
    want = {first.ID: first, second.ID: resumed_second}
    got = {value.ID: value for value in operations.adds}
    assert got == want, f"dispatched operations = {got!r}, want {want!r}"


def test_coordinator_tracks_tool_calls() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    first = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    second = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments='{"second":true}')
    for response in (
        ModelResponse(
            TurnID="turn-1",
            Response=Response(
                Output=[
                    LLMItem(
                        Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="working")
                    ),
                    LLMItem(Type=ItemType.TOOL_CALL, Data=first),
                ]
            ),
        ),
        ModelResponse(
            TurnID="turn-2",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=second)]),
        ),
    ):
        current._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))

    want = {
        ToolCallKey("turn-1", first.CallID): ToolCallState(tool_call=first, operations=set()),
        ToolCallKey("turn-2", second.CallID): ToolCallState(tool_call=second, operations=set()),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )

    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(
                TurnID="turn-1",
                CallID=first.CallID,
                Status=CallStatus(Error="invalid arguments"),
            ),
        )
    )
    want = {
        ToolCallKey("turn-2", second.CallID): ToolCallState(tool_call=second, operations=set()),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )

    waiting_for = ["operation-1", "operation-2"]
    waiting_status = CallStatus(WaitingFor=waiting_for)
    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(TurnID="turn-2", CallID=second.CallID, Status=waiting_status),
        )
    )
    want = {
        ToolCallKey("turn-2", second.CallID): ToolCallState(
            tool_call=second,
            status=waiting_status,
            operations={waiting_for[0], waiting_for[1]},
        ),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )


def test_coordinator_tool_call_operations_are_terminal() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    assert current._tool_call_operations_are_terminal("missing-turn", "missing-call"), (
        "tool call without operations was not recognized as terminal"
    )

    key = ToolCallKey("turn-1", "call-1")
    current._state.tool_calls[key] = ToolCallState(
        tool_call=ToolCall(), operations={"completed", "failed", "canceled"}
    )
    for id, status in {
        "completed": Status.COMPLETED,
        "failed": Status.FAILED,
        "canceled": Status.CANCELED,
    }.items():
        current._add_operation_to_local_state(Operation(ID=id, Status=status))

    assert current._tool_call_operations_are_terminal(key.turn_id, key.call_id), (
        "terminal operations were not recognized"
    )

    current._add_operation_to_local_state(Operation(ID="failed", Status=Status.AWAITING))
    assert not current._tool_call_operations_are_terminal(key.turn_id, key.call_id), (
        "non-terminal operation was recognized as terminal"
    )


def test_coordinator_adds_tool_result_from_tracked_tool_call() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    builder = new_builder()
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(Error="invalid arguments")
    )
    current._add_tool_result_to_local_state(status)

    built = builder.build()
    want = with_preamble(loop_state_text_result(call.CallID, "error:invalid arguments"))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


def test_coordinator_accepts_orphaned_tool_status() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    status = ToolCallStatus(
        TurnID="turn-1", CallID="missing-call", Status=CallStatus(WaitingFor=["operation-1"])
    )

    item = current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))
    assert item.Data == status, f"returned item = {item.Data!r}, want {status!r}"


def test_coordinator_returns_tool_result_translation_error() -> None:
    registry = new_registry(
        StaticTranslators(view_image=FailingResultTranslator(RuntimeError("translation failed"))),
        VIEW_IMAGE_NAME,
    )
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    with pytest.raises(CoordinatorError) as caught:
        current._add_item_to_local_state(
            Item(
                Kind=ItemKind.TOOL_CALL_STATUS,
                Data=ToolCallStatus(TurnID="turn-1", CallID=call.CallID),
            )
        )
    assert str(caught.value) == 'add tool call "call-1" result to context: translation failed', (
        f"add status error = {caught.value!r}"
    )


def test_coordinator_skips_tool_result_without_available_translator() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    call = ToolCall(CallID="call-1", Name="unavailable-tool")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    current._add_tool_result_to_local_state(ToolCallStatus(TurnID="turn-1", CallID=call.CallID))
    assert ToolCallKey("turn-1", call.CallID) in current._state.tool_calls, (
        "tool call with unavailable translator was removed"
    )


def test_coordinator_skips_tool_result_with_untracked_operation() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME)
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )
    current._add_operation_to_local_state(Operation(ID="operation-1", Status=Status.COMPLETED))

    current._add_tool_result_to_local_state(
        ToolCallStatus(
            TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=["operation-1"])
        )
    )
    assert ToolCallKey("turn-1", call.CallID) in current._state.tool_calls, (
        "tool call with an untracked operation was removed"
    )


async def test_coordinator_keeps_control_input_without_context_projection() -> None:
    input = Input(
        ID="control-input", Kind=InputKind.CONTROL, Payload=gojson.Raw(b'{"Mode":"when_idle"}')
    )
    store = loop_state_store(stored_item(1, ItemKind.INPUT, input))
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, new_registry(StaticTranslators())
    )

    await current._restore()
    built = builder.build()
    assert built.request.Input == with_preamble(), f"replayed input = {built.request.Input!r}"


@pytest.mark.parametrize(
    ("item", "want"),
    [
        pytest.param(stored_item(1, ItemKind.FORK, Turn()), "want sessionstore.Fork", id="fork"),
        pytest.param(stored_item(1, ItemKind.INPUT, Turn()), "want inbox.Input", id="input"),
        pytest.param(
            stored_item(1, ItemKind.INPUT, Input(ID="input-1", Kind=cast(InputKind, "unknown"))),
            "invalid input",
            id="invalid input",
        ),
        pytest.param(stored_item(1, ItemKind.TURN, Input()), "want session.Turn", id="turn"),
        pytest.param(
            stored_item(1, ItemKind.MODEL_RESPONSE, Turn()),
            "want sessionstore.ModelResponse",
            id="response",
        ),
        pytest.param(
            stored_item(1, ItemKind.TOOL_CALL_STATUS, Turn()),
            "want sessionstore.ToolCallStatus",
            id="status",
        ),
        pytest.param(stored_item(1, "unknown", None), 'unsupported item kind "unknown"', id="kind"),
    ],
)
async def test_coordinator_rejects_invalid_session_item_data(item: Item, want: str) -> None:
    store = loop_state_store(item)
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    with pytest.raises(CoordinatorError) as caught:
        await current._restore()
    assert want in str(caught.value), f"error = {caught.value!r}, want substring {want!r}"


@synctest
async def test_coordinator_run_returns_context_cancellation() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    task = asyncio.create_task(current.run())
    await wait()
    task.cancel()
    await asyncio.wait((task,))
    error: BaseException | None = asyncio.CancelledError() if task.cancelled() else task.exception()
    assert isinstance(error, asyncio.CancelledError), (
        f"Run error = {error!r}, want context cancellation"
    )


async def test_closed_input_error_prefers_context_cancellation() -> None:
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    try:
        error = _closed_input_error("input")
    finally:
        task.uncancel()
    assert isinstance(error, asyncio.CancelledError), (
        f"closed input error = {error!r}, want context cancellation"
    )


# --- test_loop_run ------------------------------------------------------------


async def loop_run_receive[T](values: asyncio.Queue[T]) -> T:
    """Go's `receiveTestValue`: a value within one second."""
    try:
        return await asyncio.wait_for(values.get(), 1)
    except TimeoutError:
        raise AssertionError("timed out waiting for value") from None


def loop_run_start(current: Coordinator) -> asyncio.Task[None]:
    return asyncio.create_task(current.run())


async def loop_run_error(task: asyncio.Task[None]) -> BaseException | None:
    """Go's `receiveTestValue(t, done)` for the error `Run` returned."""
    done, _ = await asyncio.wait((task,), timeout=1)
    assert done, "timed out waiting for value"
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


def loop_run_closed_inbox() -> Inbox:
    inputs = new_test_inbox()
    inputs.close()
    return inputs


async def loop_run_returned(current: Coordinator) -> BaseException | None:
    try:
        await current.run()
    except Exception as error:
        return error
    return None


async def test_coordinator_run_calls_model_after_persisted_external_input() -> None:
    store = empty_fake_store()
    store.history = [stored_item(1, ItemKind.TURN, Turn(ID="previous-turn", Type=TurnType.REGULAR))]
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    started: asyncio.Queue[Request] = asyncio.Queue()
    request_canceled: asyncio.Queue[BaseException] = asyncio.Queue()
    order: list[str] = []
    store.on_append_input = lambda _: order.append("input")
    store.on_append_turn = lambda _: order.append("turn")

    async def respond(request: Request) -> Response:
        order.append("respond")
        started.put_nowait(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            request_canceled.put_nowait(error)
            raise
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    builder = new_builder()
    current = new_test_coordinator(
        store, inputs, operations, builder, new_registry(StaticTranslators()), adapter
    )
    task = loop_run_start(current)

    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    request = await loop_run_receive(started)
    built = builder.build()
    want = with_preamble(LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello")))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"
    assert request == built.request, f"model request = {request!r}, want {built.request!r}"
    assert store.appended_inputs == [event], (
        f"appended inputs = {store.appended_inputs!r}, want {[event]!r}"
    )
    assert len(store.appended_turns) == 1, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[0].ID != "", f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[0].PreviousTurnID == "previous-turn", (
        f"appended turns = {store.appended_turns!r}"
    )
    assert order == ["input", "turn", "respond"], f"effect order = {order}"
    assert len(operations.adds) == 0, (
        f"unexpected operation effects: adds={operations.adds} cancels={operations.cancels}"
    )
    assert len(operations.cancels) == 0, (
        f"unexpected operation effects: adds={operations.adds} cancels={operations.cancels}"
    )

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    canceled = await loop_run_receive(request_canceled)
    assert isinstance(canceled, asyncio.CancelledError), (
        f"model request error = {canceled!r}, want context cancellation"
    )


async def test_coordinator_run_slurps_queued_inputs_before_calling_model() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    first = external_event("input-1", "first")
    second = external_event("input-2", "second")
    await submit_test_input(inputs, first)
    await submit_test_input(inputs, second)
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    request = await loop_run_receive(started)
    want = with_preamble(
        LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="first")),
        LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="second")),
    )
    assert request.Input == want, f"model input = {request.Input!r}, want {want!r}"
    assert store.appended_inputs == [first, second], f"appended inputs = {store.appended_inputs!r}"
    assert len(adapter.requests) == 1, f"model requests = {len(adapter.requests)}, want 1"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )


async def test_coordinator_run_does_not_call_model_when_request_build_fails() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        as_builder(FailingBuilder(new_builder(), RuntimeError("context unavailable"))),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    err = await loop_run_error(task)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == "build model request: context unavailable", f"Run error = {err!r}"
    assert store.appended_inputs == [event], (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )
    assert len(store.appended_turns) == 0, (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )


async def test_coordinator_run_does_not_call_model_when_turn_store_fails() -> None:
    store = empty_fake_store()
    store.append_turn_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    err = await loop_run_error(task)
    assert err is not None, f"Run error = {err!r}"
    assert "disk unavailable" in str(err), f"Run error = {err!r}"
    assert len(store.appended_turns) == 1, (
        f"effects after turn store failure: turns={store.appended_turns} "
        f"requests={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"effects after turn store failure: turns={store.appended_turns} "
        f"requests={adapter.requests}"
    )


async def test_coordinator_run_persists_model_response_for_originating_turn() -> None:
    store = empty_fake_store()
    response_stored: asyncio.Queue[ModelResponse] = asyncio.Queue()
    store.on_append_model_response = response_stored.put_nowait
    response = Response(
        ID="response-1",
        Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="done"))],
    )

    async def respond(request: Request) -> Response:
        return response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    stored = await loop_run_receive(response_stored)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_turns) == 1, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert stored.TurnID == store.appended_turns[0].ID, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert stored.Response == response, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert len(adapter.requests) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )
    assert len(store.appended_responses) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )


async def test_coordinator_run_persists_tool_call_before_dispatch() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    store = empty_fake_store()
    dispatched: asyncio.Queue[Operation] = asyncio.Queue()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if len(store.appended_responses) != 1 or len(store.appended_statuses) != 1:
            return RuntimeError("operation dispatched before response and status were stored")
        dispatched.put_nowait(value)
        return None

    operations.add_error = add_error
    response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )

    async def respond(request: Request) -> Response:
        return response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        operations,
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    operation_value = await loop_run_receive(dispatched)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert store.appended_statuses[0].Operations == [operation_value], (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert store.appended_statuses[0].Status.WaitingFor == [operation_value.ID], (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert len(adapter.requests) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_starts_corrective_turn_for_validation_error() -> None:
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    call_count = 0

    async def respond(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        started.put_nowait(request)
        if call_count == 1:
            return first_response
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    first_request = await loop_run_receive(started)
    second_request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    assert store.appended_statuses[0].Status.Error != "", (
        f"appended statuses = {store.appended_statuses!r}"
    )
    assert len(store.appended_turns) == 2, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[1].PreviousTurnID == store.appended_turns[0].ID, (
        f"appended turns = {store.appended_turns!r}"
    )
    assert len(first_request.Input) == 2, f"model requests = {[first_request, second_request]!r}"
    assert len(second_request.Input) == 4, f"model requests = {[first_request, second_request]!r}"
    assert second_request.Input[3].Type == ItemType.TOOL_RESULT, (
        f"model requests = {[first_request, second_request]!r}"
    )


async def test_coordinator_run_starts_continuation_turn_for_completed_tool_call() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    call_count = 0

    async def respond(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        started.put_nowait(request)
        if call_count == 1:
            return first_response
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    operations = FakeOperationManager()
    dispatched: asyncio.Queue[Operation] = asyncio.Queue()

    def add_error(value: Operation) -> Exception | None:
        dispatched.put_nowait(value)
        return None

    operations.add_error = add_error
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        operations,
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    await loop_run_receive(started)
    operation_value = replace(await loop_run_receive(dispatched), Status=Status.COMPLETED)
    operations.send(operation_value)
    continuation_request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert store.saved_operations == [operation_value], (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 2, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert store.appended_statuses[1].Operations == [operation_value], (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    options = list(adapter.request_options)
    assert len(options) == 2, (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert options[0].CacheKey == "session-1", (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert options[1].CacheKey == "session-1", (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert len(continuation_request.Input) == 4, f"continuation request = {continuation_request!r}"
    assert continuation_request.Input[3].Type == ItemType.TOOL_RESULT, (
        f"continuation request = {continuation_request!r}"
    )
    assert len(adapter.requests) == 2, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 2, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_batches_completed_tool_calls_into_one_turn() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    operation_value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    calls = [
        ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}"),
        ToolCall(CallID="call-2", Name=VIEW_IMAGE_NAME, Arguments="{}"),
    ]
    status = CallStatus(WaitingFor=[operation_value.ID])
    store = empty_fake_store()
    store.resume.operations = [replace(operation_value)]
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(Type=ItemType.TOOL_CALL, Data=calls[0]),
                        LLMItem(Type=ItemType.TOOL_CALL, Data=calls[1]),
                    ]
                ),
            ),
        ),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=calls[0].CallID,
                Status=status,
                Operations=[replace(operation_value)],
            ),
        ),
        stored_item(
            4,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=calls[1].CallID,
                Status=status,
                Operations=[replace(operation_value)],
            ),
        ),
    ]
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)
    task = loop_run_start(current)

    operations.send(replace(operation_value, Status=Status.COMPLETED))
    request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 2, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(store.appended_turns) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_turns[0].PreviousTurnID == "turn-1", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(adapter.requests) == 1, f"requests = {adapter.requests!r}"
    assert len(request.Input) == 5, f"requests = {adapter.requests!r}"


async def test_coordinator_run_steers_active_model_request() -> None:
    started: asyncio.Queue[Request] = asyncio.Queue()
    first_canceled: asyncio.Queue[None] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if len(request.Input) == 2:
                first_canceled.put_nowait(None)
            raise
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "first"))
    first_request = await loop_run_receive(started)
    await submit_test_input(inputs, external_event("input-2", "second"))
    second_request = await loop_run_receive(started)
    await loop_run_receive(first_canceled)
    done, _ = await asyncio.wait((task,), timeout=0.02)
    assert not done, f"Run stopped after superseded cancellation: {await loop_run_error(task)!r}"
    assert len(first_request.Input) == 2, f"steered requests = {[first_request, second_request]!r}"
    assert len(second_request.Input) == 3, f"steered requests = {[first_request, second_request]!r}"
    assert len(store.appended_turns) == 2, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[1].PreviousTurnID == store.appended_turns[0].ID, (
        f"appended turns = {store.appended_turns!r}"
    )
    assert len(store.appended_responses) == 0, f"appended responses = {store.appended_responses!r}"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )


async def test_coordinator_run_handles_operation_update_while_model_is_running() -> None:
    started: asyncio.Queue[None] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(None)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    initial = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store = empty_fake_store()
    store.resume.operations = [replace(initial)]
    stored_update: asyncio.Queue[Operation] = asyncio.Queue()
    store.on_save_operation = stored_update.put_nowait
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), new_registry(StaticTranslators()), adapter
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    await loop_run_receive(started)
    update = replace(initial, Status=Status.AWAITING)
    operations.send(update)
    stored = await loop_run_receive(stored_update)
    assert stored == update, f"stored operation = {stored!r}, want {update!r}"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(adapter.requests) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_drops_successful_response_from_superseded_turn() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="stale-call", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    second_response = Response(
        ID="response-2",
        Output=[
            LLMItem(
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="current answer"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    first_returned: asyncio.Queue[None] = asyncio.Queue()
    release_second = asyncio.Event()
    response_stored: asyncio.Queue[ModelResponse] = asyncio.Queue()
    store = empty_fake_store()
    store.on_append_model_response = response_stored.put_nowait

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        if len(request.Input) == 2:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            first_returned.put_nowait(None)
            return first_response
        await release_second.wait()
        return second_response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)
    try:
        await submit_test_input(inputs, external_event("input-1", "first"))
        await loop_run_receive(started)
        await submit_test_input(inputs, external_event("input-2", "second"))
        await loop_run_receive(started)
        await loop_run_receive(first_returned)
        await asyncio.sleep(0.02)
        release_second.set()
        stored = await loop_run_receive(response_stored)
        assert len(store.appended_turns) == 2, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        assert stored.TurnID == store.appended_turns[1].ID, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        assert stored.Response == second_response, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        try:
            unexpected = await asyncio.wait_for(response_stored.get(), 0.02)
        except TimeoutError:
            pass
        else:
            raise AssertionError(f"stored superseded response = {unexpected!r}")

        task.cancel()
        err = await loop_run_error(task)
        assert isinstance(err, asyncio.CancelledError), (
            f"Run error = {err!r}, want context cancellation"
        )
        assert store.appended_responses == [stored], (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
        assert len(translator.calls) == 0, (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
        assert len(store.appended_statuses) == 0, (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
    finally:
        task.cancel()


async def test_coordinator_run_returns_current_model_error() -> None:
    provider_error = RuntimeError("provider unavailable")

    async def respond(request: Request) -> Response:
        raise provider_error

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    err = await loop_run_error(task)
    assert caused_by(err, provider_error), f"Run error = {err!r}"
    assert "call model for turn" in str(err), f"Run error = {err!r}"
    assert len(store.appended_turns) == 1, (
        f"turns = {store.appended_turns!r}, responses = {store.appended_responses!r}"
    )
    assert len(store.appended_responses) == 0, (
        f"turns = {store.appended_turns!r}, responses = {store.appended_responses!r}"
    )


async def test_coordinator_handles_model_response_before_scheduling_tool_calls() -> None:
    spec = value_spec(b'{"value":1}')
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            ID="response-1",
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ],
        ),
    )
    store = empty_fake_store()
    stored_before_translation = False

    def on_translate() -> None:
        nonlocal stored_before_translation
        stored_before_translation = store.appended_responses == [response]

    translator = SubmittingTranslator([spec], on_translate)
    operations = FakeOperationManager()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        operations,
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
    )

    await current._handle_model_response(response)
    assert stored_before_translation, (
        "tool call was translated before its model response was stored"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    assert len(store.appended_statuses[0].Operations) == 1, (
        f"appended statuses = {store.appended_statuses!r}"
    )
    assert len(operations.adds) == 0, f"handler dispatched operations = {operations.adds!r}"


async def test_coordinator_does_not_schedule_tool_calls_when_model_response_store_fails() -> None:
    store = empty_fake_store()
    store.append_model_response_error = RuntimeError("disk unavailable")
    translator = SubmittingTranslator()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
    )
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ]
        ),
    )

    with pytest.raises(CoordinatorError) as raised:
        await current._handle_model_response(response)
    assert str(raised.value) == 'store turn "turn-1" response: disk unavailable', (
        f"handle model response error = {raised.value!r}"
    )
    assert store.appended_responses == [response], (
        f"appended responses = {store.appended_responses!r}"
    )
    assert len(translator.calls) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(store.appended_statuses) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(current._state.operations) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )


async def test_coordinator_handles_model_response_without_tool_calls() -> None:
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="done"))]
        ),
    )

    await current._handle_model_response(response)
    assert store.appended_responses == [response], (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(store.appended_statuses) == 0, (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(current._state.operations) == 0, (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )


async def test_coordinator_run_schedules_tool_calls_without_status_before_dispatch() -> None:
    first_spec = replace(value_spec(b'{"value":1}'), MaxOutputLength=3)
    second_spec = replace(value_spec(b'{"value":2}'), MaxOutputLength=7)
    translator = SubmittingTranslator([first_spec, second_spec])
    registry = new_registry(StaticTranslators(bash=translator), BASH_NAME)
    handled = ToolCall(CallID="call-handled", Name=BASH_NAME, Arguments="{}")
    missing = ToolCall(CallID="call-missing", Name=BASH_NAME, Arguments="{}")
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(Type=ItemType.TOOL_CALL, Data=handled),
                        LLMItem(Type=ItemType.TOOL_CALL, Data=missing),
                    ]
                ),
            ),
        ),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=handled.CallID,
                Status=CallStatus(Error="already handled"),
            ),
        ),
    ]
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if len(store.appended_statuses) != 1 or store.appended_statuses[0].CallID != missing.CallID:
            return RuntimeError("operation dispatched before tool-call status was stored")
        return None

    operations.add_error = add_error
    requests: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        requests.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed inbox error"
    assert str(err) == "inbox output closed", f"Run error = {err!r}, want closed inbox error"
    assert translator.calls == [missing], (
        f"translated calls = {translator.calls!r}, want {[missing]!r}"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    status = store.appended_statuses[0]
    assert status.TurnID == "turn-1", f"appended status = {status!r}"
    assert status.CallID == missing.CallID, f"appended status = {status!r}"
    assert len(status.Status.WaitingFor) == 2, f"appended status = {status!r}"
    assert len(status.Operations) == 2, f"appended status = {status!r}"
    want_operations: dict[str, Operation] = {}
    for index, value in enumerate(status.Operations):
        assert value.ID != "", f"scheduled operation {index} = {value!r}"
        assert status.Status.WaitingFor[index] == value.ID, (
            f"scheduled operation {index} = {value!r}"
        )
        assert value.Status == Status.READY, f"scheduled operation {index} = {value!r}"
        assert value.MaxOutputLength == translator.specs[index].MaxOutputLength, (
            f"scheduled operation {index} = {value!r}"
        )
        want_operations[value.ID] = value
        local = current._state.operations.get(value.ID)
        assert local == value, f"local operation {value.ID!r} = {local!r}, want {value!r}"
    got_operations = {value.ID: value for value in operations.adds}
    assert got_operations == want_operations, (
        f"dispatched operations = {got_operations!r}, want {want_operations!r}"
    )
    await current._schedule_tool_calls()
    assert len(translator.calls) == 1, (
        f"rescheduled call: calls={translator.calls!r} statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 1, (
        f"rescheduled call: calls={translator.calls!r} statuses={store.appended_statuses!r}"
    )
    assert_stop_result(await loop_run_receive(requests), handled.CallID, "already handled")


async def test_coordinator_run_starts_corrective_turn_for_recovered_validation_error() -> None:
    call = ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}")
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
    ]
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_statuses[0].Status.Error != "", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(store.appended_turns) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_turns[0].PreviousTurnID == "turn-1", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(request.Input) == 3, f"corrective request = {request!r}"
    assert request.Input[2].Type == ItemType.TOOL_RESULT, f"corrective request = {request!r}"


async def test_coordinator_run_rejects_external_input_without_text_payload() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    await submit_test_input(
        inputs, Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=gojson.Raw(b"{}"))
    )
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert 'add input "input-1" to context' in str(err), f"Run error = {err!r}"
    assert len(store.appended_inputs) == 0, f"stored inputs = {store.appended_inputs!r}"


async def test_coordinator_run_stores_operation_updates_without_redispatch() -> None:
    store = empty_fake_store()
    initial = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store.resume.operations = [replace(initial)]
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    updated = replace(initial, Status=Status.AWAITING)
    operations.send(updated)
    operations.close()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), new_registry(StaticTranslators()), adapter
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed operation updates error"
    assert str(err) == "operation updates closed", (
        f"Run error = {err!r}, want closed operation updates error"
    )
    local = current._state.operations.get(initial.ID)
    assert local == updated, f"operation = {local!r}, want {updated!r}"
    assert store.saved_operations == [updated], (
        f"saved operations = {store.saved_operations!r}, want {[updated]!r}"
    )
    assert operations.adds == [initial], (
        f"dispatched operations = {operations.adds!r}, want only the initial operation"
    )
    assert len(operations.cancels) == 0, (
        f"unexpected effects: cancels={operations.cancels} model={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"unexpected effects: cancels={operations.cancels} model={adapter.requests}"
    )


async def test_coordinator_reconciles_tool_calls_from_persisted_operation_updates() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    store = empty_fake_store()
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        )
    )
    operations = [
        Operation(ID="operation-1", Status=Status.READY),
        Operation(ID="operation-2", Status=Status.AWAITING),
    ]
    for value in operations:
        current._add_operation_to_local_state(value)
    status = ToolCallStatus(
        TurnID="turn-1",
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=[operations[0].ID, operations[1].ID]),
    )
    current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))

    first = replace(operations[0], Status=Status.COMPLETED)
    await current._handle_operation_update(first)
    await current._reconcile_tool_calls()
    key = ToolCallKey(status.TurnID, status.CallID)
    assert key in current._state.tool_calls, (
        "tool call was removed before every operation became terminal"
    )

    second = replace(operations[1], Status=Status.FAILED)
    await current._handle_operation_update(second)
    await current._reconcile_tool_calls()
    assert key not in current._state.tool_calls, "completed tool call remains in local state"
    assert store.saved_operations == [first, second], (
        f"saved operations = {store.saved_operations!r}, want {[first, second]!r}"
    )
    assert store.appended_statuses == [
        ToolCallStatus(
            TurnID=status.TurnID,
            CallID=status.CallID,
            Status=status.Status,
            Operations=[first, second],
        )
    ], f"appended statuses = {store.appended_statuses!r}"
    built = builder.build()
    want = with_preamble(
        LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        LLMItem(
            Type=ItemType.TOOL_RESULT,
            Data=ToolResult(
                CallID=call.CallID,
                Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="error:")],
            ),
        ),
    )
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_run_returns_reconciliation_error() -> None:
    registry = new_registry(
        StaticTranslators(view_image=TerminalResultTranslator()), VIEW_IMAGE_NAME
    )
    value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[value.ID])
    )
    store = empty_fake_store()
    store.resume.operations = [replace(value)]
    store.history = [
        stored_item(
            1,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(2, ItemKind.TOOL_CALL_STATUS, status),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(TurnID="another-turn", CallID="another-call"),
        ),
        stored_item(4, ItemKind.TURN, Turn(ID="turn-2", Type=TurnType.REGULAR)),
    ]
    completed = replace(value, Status=Status.COMPLETED)
    operations = FakeOperationManager()
    operations.send(completed)
    current = new_test_coordinator(store, new_test_inbox(), operations, new_builder(), registry)

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == ('add tool call "call-1" result to context: terminal result failed'), (
        f"Run error = {err!r}"
    )
    assert store.saved_operations == [completed], (
        f"saved operations = {store.saved_operations!r}, want {[completed]!r}"
    )


async def test_coordinator_does_not_complete_tool_call_before_operation_is_stored() -> None:
    registry = new_registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    store = empty_fake_store()
    store.save_operation_error = RuntimeError("disk unavailable")
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )
    value = Operation(ID="operation-1", Status=Status.READY)
    current._add_operation_to_local_state(value)
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[value.ID])
    )
    current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))

    value = replace(value, Status=Status.COMPLETED)
    with pytest.raises(CoordinatorError) as raised:
        await current._handle_operation_update(value)
    assert str(raised.value) == 'store operation "operation-1": disk unavailable', (
        f"handle operation update error = {raised.value!r}"
    )
    assert ToolCallKey(status.TurnID, status.CallID) in current._state.tool_calls, (
        "tool call was removed before its operation was stored"
    )
    built = builder.build()
    assert len(built.request.Input) == 2, (
        f"tool results = {built.request.Input!r}, want only the initial result"
    )


async def test_coordinator_run_dispatches_restored_non_terminal_operations() -> None:
    store = empty_fake_store()
    statuses = [
        Status.READY,
        Status.AWAITING,
        Status.CANCELING,
        Status.COMPLETED,
        Status.FAILED,
        Status.CANCELED,
    ]
    for status in statuses:
        store.resume.operations.append(
            Operation(ID=status.value, Type=TYPE_SHELL, Version=1, Status=status)
        )
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), new_registry(StaticTranslators()), adapter
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed inbox error"
    assert str(err) == "inbox output closed", f"Run error = {err!r}, want closed inbox error"
    adds = sorted(operations.adds, key=lambda value: value.ID)
    want = [
        Operation(ID="awaiting", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING),
        Operation(ID="canceling", Type=TYPE_SHELL, Version=1, Status=Status.CANCELING),
        Operation(ID="ready", Type=TYPE_SHELL, Version=1, Status=Status.READY),
    ]
    assert adds == want, f"dispatched operations = {adds!r}, want {want!r}"
    assert len(adapter.requests) == 0, f"model requests = {adapter.requests!r}"


async def test_coordinator_run_returns_operation_dispatch_error() -> None:
    store = empty_fake_store()
    value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store.resume.operations = [value]
    operations = FakeOperationManager()
    operations.add_error = lambda _: RuntimeError("dispatch failed")
    current = new_test_coordinator(
        store, new_test_inbox(), operations, new_builder(), new_registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'dispatch operation "operation-1": dispatch failed', f"Run error = {err!r}"


async def test_coordinator_run_returns_unsupported_recovered_operation() -> None:
    store = empty_fake_store()
    store.resume.operations = [
        Operation(ID="unsupported", Type="remote", Version=1, Status=Status.READY),
        Operation(ID="supported", Type=TYPE_SHELL, Version=1, Status=Status.READY),
    ]
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if value.ID == "unsupported":
            return UnsupportedError(
                f"manager does not support {go_quote(value.Type)}: unsupported operation"
            )
        return None

    operations.add_error = add_error
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), new_registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert caused_by(err, UnsupportedError), f"Run error = {err!r}, want unsupported operation"
    assert "unsupported" in current._state.operations, "unsupported operation was not retained"


async def test_coordinator_clones_operation_data_before_dispatch() -> None:
    operations = FakeOperationManager()
    operations.mutate_adds = True
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        operations,
        new_builder(),
        new_registry(StaticTranslators()),
    )
    value = current._add_operation_to_local_state(
        Operation(
            ID="operation-1",
            Type=TYPE_SHELL,
            Version=1,
            Status=Status.READY,
            State=gojson.Raw(b'{"state":"original"}'),
            Idempotency=gojson.Raw(b'{"key":"original"}'),
        )
    )

    await current._dispatch_operations_to_manager()
    stored = current._state.operations[value.ID]
    assert bytes(stored.State) == b'{"state":"original"}', (
        f"stored operation was mutated: {stored!r}"
    )
    assert bytes(stored.Idempotency) == b'{"key":"original"}', (
        f"stored operation was mutated: {stored!r}"
    )


async def test_coordinator_run_returns_input_store_error_after_updating_local_state() -> None:
    store = empty_fake_store()
    store.append_input_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    current = new_test_coordinator(
        store, inputs, FakeOperationManager(), new_builder(), new_registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'store input "input-1": disk unavailable', f"Run error = {err!r}"
    built = deps(current).context_builder.build()
    want = with_preamble(LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello")))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_run_returns_operation_store_error_after_updating_local_state() -> None:
    store = empty_fake_store()
    store.save_operation_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    update = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING)
    operations.send(update)
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), new_registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'store operation "operation-1": disk unavailable', f"Run error = {err!r}"
    local = current._state.operations.get(update.ID)
    assert local == update, f"operation = {local!r}, want {update!r}"
    assert len(operations.adds) == 0, (
        f"operations dispatched before store commit = {operations.adds!r}"
    )


async def test_coordinator_stores_every_session_item_kind() -> None:
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    event = external_event("input-1", "hello")
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    response = ModelResponse(TurnID=turn.ID, Response=Response(ID="response-1"))
    value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    current._add_operation_to_local_state(value)
    status = ToolCallStatus(
        TurnID=turn.ID,
        CallID="call-1",
        Status=CallStatus(WaitingFor=[value.ID]),
        Operations=[value],
    )
    items = [
        Item(Kind=ItemKind.INPUT, Data=event),
        Item(Kind=ItemKind.TURN, Data=turn),
        Item(Kind=ItemKind.MODEL_RESPONSE, Data=response),
        Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status),
    ]
    for item in items:
        await current._store_item_in_session_store(item)

    assert store.appended_inputs == [event], (
        f"stored items: inputs={store.appended_inputs!r} turns={store.appended_turns!r} "
        f"responses={store.appended_responses!r} statuses={store.appended_statuses!r}"
    )
    assert store.appended_turns == [turn], (
        f"stored items: inputs={store.appended_inputs!r} turns={store.appended_turns!r} "
        f"responses={store.appended_responses!r} statuses={store.appended_statuses!r}"
    )
    assert store.appended_responses == [response], (
        f"stored items: inputs={store.appended_inputs!r} turns={store.appended_turns!r} "
        f"responses={store.appended_responses!r} statuses={store.appended_statuses!r}"
    )
    assert store.appended_statuses == [status], (
        f"stored items: inputs={store.appended_inputs!r} turns={store.appended_turns!r} "
        f"responses={store.appended_responses!r} statuses={store.appended_statuses!r}"
    )


def _loop_run_fail_turn(store: FakeStore, _: Coordinator) -> None:
    store.append_turn_error = RuntimeError("disk unavailable")


def _loop_run_fail_model_response(store: FakeStore, _: Coordinator) -> None:
    store.append_model_response_error = RuntimeError("disk unavailable")


def _loop_run_fail_status(store: FakeStore, _: Coordinator) -> None:
    store.append_status_error = RuntimeError("disk unavailable")


def _loop_run_configure_nothing(store: FakeStore, _: Coordinator) -> None:
    pass


_LOOP_RUN_TURN = Turn(ID="turn-1", Type=TurnType.REGULAR)


@pytest.mark.parametrize(
    ("item", "configure", "want"),
    [
        pytest.param(
            Item(Kind=ItemKind.TURN, Data=_LOOP_RUN_TURN),
            _loop_run_fail_turn,
            'store turn "turn-1": disk unavailable',
            id="turn",
        ),
        pytest.param(
            Item(Kind=ItemKind.MODEL_RESPONSE, Data=ModelResponse(TurnID=_LOOP_RUN_TURN.ID)),
            _loop_run_fail_model_response,
            'store turn "turn-1" response: disk unavailable',
            id="model response",
        ),
        pytest.param(
            Item(
                Kind=ItemKind.TOOL_CALL_STATUS,
                Data=ToolCallStatus(
                    TurnID=_LOOP_RUN_TURN.ID,
                    CallID="call-1",
                    Status=CallStatus(WaitingFor=["operation-1"]),
                ),
            ),
            _loop_run_fail_status,
            'store tool call "call-1" status: disk unavailable',
            id="tool status",
        ),
        pytest.param(
            Item(Kind=ItemKind.FORK, Data=Fork()),
            _loop_run_configure_nothing,
            'unsupported local item kind "fork"',
            id="unsupported item",
        ),
    ],
)
async def test_coordinator_returns_session_item_store_errors(
    item: Item, configure: Callable[[FakeStore, Coordinator], None], want: str
) -> None:
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators()),
    )
    configure(store, current)

    with pytest.raises(CoordinatorError) as raised:
        await current._store_item_in_session_store(item)
    assert str(raised.value) == want, f"store error = {raised.value!r}, want {want!r}"


# --- test_stop ----------------------------------------------------------------


@synctest
async def test_coordinator_remains_available_until_stop() -> None:
    run = new_stop_test_run(0)
    await run.start()
    run.assert_running()
    for index in range(2):
        await run.input(external_event(f"input-{index}", "hello"))
        await run.respond(index, text_response("hello"))
        run.assert_running()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()
    assert len(run.calls) == 2, f"requests = {len(run.calls)}, want 2"


@pytest.mark.parametrize(
    ("terminal", "during_request"),
    [
        pytest.param(terminal, during, id=f"{terminal}/during_request={str(during).lower()}")
        for terminal in (Status.COMPLETED, Status.FAILED, Status.CANCELED)
        for during in (False, True)
    ],
)
@synctest
async def test_coordinator_when_idle_delivers_pending_results(
    terminal: Status, during_request: bool
) -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(external_event("progress", "check progress"))
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(run.calls) == 1, "when_idle interrupted the model request"
    assert not run.calls[0].canceled, "when_idle interrupted the model request"
    if during_request:
        await run.update(0, terminal)
    await run.respond(0, text_response("The check is still running."))
    run.assert_running()
    if not during_request:
        assert len(run.calls) == 1, "requested model response before tool completed"
        await run.update(0, terminal)
    assert len(run.calls) == 2, f"requests = {len(run.calls)}, want 2"
    assert_stop_result(run.calls[1].request, "call-0", str(terminal))
    await run.respond(1, text_response("All results received."))
    run.assert_stopped()
    assert len(run.operations.cancels) == 0, "when_idle canceled an operation"


@synctest
async def test_coordinator_stops_after_cancellation_is_recorded() -> None:
    run = new_stop_test_run(2)
    await run.start()
    await run.input(external_event("progress", "check progress"))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    assert run.calls[0].canceled, "stop did not interrupt active model request"
    assert len(run.operations.cancels) == 2, (
        f"cancellations = {run.operations.cancels}, want both operations"
    )
    for reason in run.operations.cancel_reasons:
        assert reason == "user requested stop", f"cancellation reason = {reason!r}"
    run.assert_running()
    await run.update(0, Status.CANCELED)
    run.assert_running()
    await run.update(1, Status.COMPLETED)
    assert len(run.store.saved_operations) == 2, (
        "stop did not persist terminal updates and tool results"
    )
    assert len(run.store.appended_statuses) == 2, (
        "stop did not persist terminal updates and tool results"
    )
    assert len(run.calls) == 1, "hard stop started another request"
    run.assert_stopped()
    assert len(run.operations.cancels) == 2, "operation cancellation was requested more than once"


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, "steer"], ids=[ControlMode.STOP_HARD, "steer"]
)
@synctest
async def test_coordinator_cancels_compaction_without_recording_a_response(mode: str) -> None:
    run = new_stop_test_run(0)
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("first", "hello"))
    run.current._state.current_turn_type = TurnType.COMPACTION
    if mode == "steer":
        await run.input(external_event("second", "change course"))
    else:
        await run.input(stop_input("stop", mode))
    want_type = TurnType.REGULAR
    if mode == ControlMode.STOP_HARD:
        want_type = TurnType.COMPACTION
    assert run.calls[0].canceled, (
        "cancellation did not preserve the recorded turn kind until the next turn"
    )
    assert run.current._state.current_turn_type == want_type, (
        "cancellation did not preserve the recorded turn kind until the next turn"
    )
    await run.respond(0, text_response("Late summary"))
    assert len(run.store.appended_responses) == 0, (
        "canceled compaction recorded a response or delivered input"
    )
    assert run.current._state.delivered_inputs == 0, (
        "canceled compaction recorded a response or delivered input"
    )
    if mode != ControlMode.STOP_HARD:
        assert len(run.calls) == 2, "late response affected the replacement request"
        assert run.current._cancel_model is not None, (
            "late response affected the replacement request"
        )
        await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
        await run.respond(1, text_response("Done"))
    run.assert_stopped()


@synctest
async def test_coordinator_hard_stop_overrides_when_idle() -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(external_event("first", "hello"))
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    assert not run.calls[0].canceled, "when_idle interrupted the model"
    await run.input(stop_input("hard", ControlMode.STOP_HARD))
    await run.input(
        stop_input("idle-again", ControlMode.STOP_WHEN_IDLE),
        stop_input("hard-again", ControlMode.STOP_HARD),
        heartbeat_input("heartbeat"),
        external_event("late", "more work"),
    )
    run.assert_running()
    assert run.current._stop.request.Mode == ControlMode.STOP_HARD, (
        "later inputs changed the stop or repeated cancellation"
    )
    assert len(run.operations.cancels) == 1, (
        "later inputs changed the stop or repeated cancellation"
    )
    assert len(run.calls) == 1, (
        "hard stop did not interrupt the model or later input started another request"
    )
    assert run.calls[0].canceled, (
        "hard stop did not interrupt the model or later input started another request"
    )
    await run.update(0, Status.CANCELED)
    run.assert_stopped()
    assert len(run.operations.cancels) == 1, (
        "stop repeated cancellation or started another model request"
    )
    assert len(run.calls) == 1, "stop repeated cancellation or started another model request"


@synctest
async def test_coordinator_hard_stop_discards_late_model_response() -> None:
    run = new_stop_test_run(1)
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("first", "hello"))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    run.assert_running()
    await run.respond(
        0,
        Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="late", Name=VIEW_IMAGE_NAME, Arguments="{}"),
                )
            ]
        ),
    )
    assert len(run.store.appended_responses) == 0, (
        "late response was recorded or translated during cancellation"
    )
    assert len(run.store.appended_statuses) == 0, (
        "late response was recorded or translated during cancellation"
    )
    assert len(run.calls) == 1, "late response was recorded or translated during cancellation"
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@pytest.mark.parametrize("failure", ["input", "cancel", "operation", "tool-result"])
@synctest
async def test_coordinator_stop_propagates_errors(failure: str) -> None:
    run = new_stop_test_run(2)
    want = RuntimeError("test failure")
    match failure:
        case "input":
            run.store.append_input_error = want
        case "cancel":
            run.operations.cancel_error = want
        case "operation":
            run.store.save_operation_error = want
        case "tool-result":
            run.store.append_status_error = want
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    if failure in ("operation", "tool-result"):
        await run.update(0, Status.CANCELED)
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    if failure == "cancel":
        assert len(run.operations.cancels) == 2, (
            "cancellation error prevented attempting the other operation"
        )


@pytest.mark.parametrize("source", ["inbox", "operation updates"])
@synctest
async def test_coordinator_cancellation_while_collecting_updates(source: str) -> None:
    run = new_stop_test_run(1)
    run.go()
    await wait()
    if source == "inbox":
        await submit_test_input(run.inputs, external_event("input", "hello"))
    else:
        run.operations.send(replace(run.store.resume.operations[0], Status=Status.COMPLETED))
    await wait()
    run.cancel("caller disconnected")
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want collection cancellation cause"
    )
    want_inputs, want_updates = 1, 0
    if source == "operation updates":
        want_inputs, want_updates = 0, 1
    assert len(run.store.appended_inputs) == want_inputs, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )
    assert len(run.store.saved_operations) == want_updates, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )
    assert len(run.calls) == 0, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )


@synctest
async def test_coordinator_drains_inbox_and_operations_before_calling_model() -> None:
    run = new_stop_test_run(1)
    await run.start()
    await submit_test_input(run.inputs, external_event("input", "check completed work"))
    run.operations.send(replace(run.store.resume.operations[0], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()

    assert len(run.calls) == 1, (
        f"model requests = {len(run.calls)}, want one request containing both events"
    )
    assert_completed_results(run.calls[0].request, 1)
    found_input = False
    for item in run.calls[0].request.Input:
        if (
            item.Type == ItemType.MESSAGE
            and cast(Message, item.Data).Text == "check completed work"
        ):
            found_input = True
    assert found_input, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )
    assert len(run.store.appended_inputs) == 1, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )
    assert len(run.store.saved_operations) == 1, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )


@synctest
async def test_coordinator_cancellation_takes_precedence_over_model_error() -> None:
    run = new_stop_test_run(0)
    results: asyncio.Future[Exception] = asyncio.get_running_loop().create_future()

    async def respond(request: Request) -> Response:
        raise await results

    run.dependencies.llm = FakeAdapter(respond)
    run.go()
    await wait()
    await run.input(external_event("input", "hello"))
    cancel_model = run.current._cancel_model
    assert cancel_model is not None, "model request did not start"

    def cancel_run_and_model() -> object:
        run.cancel()
        return cancel_model()

    run.current._cancel_model = cancel_run_and_model
    results.set_result(RuntimeError("model failed"))
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(run.store.appended_responses) == 0, "failed model response was persisted"


@pytest.mark.parametrize(
    "mode",
    [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE],
    ids=[ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE],
)
@synctest
async def test_coordinator_stop_controls_do_not_repeat_on_resume(mode: ControlMode) -> None:
    run = new_stop_test_run(0)
    run.store.history.append(stored_item(3, ItemKind.INPUT, stop_input("old-stop", mode)))
    await run.start()
    run.assert_running()
    await run.input(external_event("new", "continue"))
    await run.respond(0, text_response("Continuing."))
    run.assert_running()
    await run.input(stop_input("new-stop", ControlMode.STOP_HARD))
    run.assert_stopped()


@pytest.mark.parametrize("stage", ["completed", "request-started", "response-recorded"])
@synctest
async def test_coordinator_when_idle_delivers_restored_results(stage: str) -> None:
    run = new_stop_test_run(1)
    status = cast(ToolCallStatus, run.store.history[2].Data)
    status.Operations[0] = replace(status.Operations[0], Status=Status.COMPLETED)
    run.store.history[2].Data = status
    run.store.resume.operations = []
    run.dependencies.restored = run.store.resume
    if stage != "completed":
        run.store.history.append(
            stored_item(
                4,
                ItemKind.TURN,
                Turn(ID="delivered", PreviousTurnID="turn-1", Type=TurnType.REGULAR),
            )
        )
    if stage == "response-recorded":
        run.store.history.append(
            stored_item(
                5,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="delivered", Response=text_response("Done.")),
            )
        )
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    if stage != "response-recorded":
        assert len(run.calls) == 1, f"requests = {len(run.calls)}, want restored result delivery"
        assert_stop_result(run.calls[0].request, "call-0", str(Status.COMPLETED))
        await run.respond(0, text_response("Done."))
    else:
        assert len(run.calls) == 0, "already delivered result caused another turn"
    run.assert_stopped()


# --- test_heartbeat -----------------------------------------------------------


@synctest
async def test_coordinator_heartbeats_while_waiting_for_tools() -> None:
    run = new_heartbeat_test_run(2)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.AWAITING)
    await advance_heartbeat_time(30 * SECOND - (2 * SLURP_IDLE_TIMEOUT_NS) - NANOSECOND)
    assert run.request_count() == 0, "heartbeat arrived before the interval elapsed"
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    want_reason = (
        "Heartbeat: waited 60 seconds for tool calls.\nRunning: "
        '[{"CallID":"call-0","Name":"ViewImage","Arguments":"{}"},'
        '{"CallID":"call-1","Name":"ViewImage","Arguments":"{}"}]'
    )
    reason = latest_heartbeat_reason(run)
    assert reason == want_reason, f"heartbeat reason = {reason!r}, want {want_reason!r}"
    assert run.request_count() == 1, (
        "heartbeat did not produce one model request with a check-in message"
    )
    assert count_heartbeat_messages(run.calls[0].request) == 1, (
        "heartbeat did not produce one model request with a check-in message"
    )
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 1)
    assert run.request_count() == 1, "heartbeat interrupted or overlapped an active model request"
    assert not run.calls[0].canceled, "heartbeat interrupted or overlapped an active model request"
    await run.respond(0, text_response("Still waiting."))
    await advance_heartbeat_time(MINUTE - NANOSECOND)
    assert_heartbeat_count(run, 1)
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 2)
    assert run.request_count() == 2, "next heartbeat did not append exactly one check-in"
    assert count_heartbeat_messages(run.calls[1].request) == 2, (
        "next heartbeat did not append exactly one check-in"
    )
    await run.update(0, Status.COMPLETED)
    await run.update(1, Status.COMPLETED)
    assert run.request_count() == 2, "tool completion interrupted a heartbeat response"
    assert not run.calls[1].canceled, "tool completion interrupted a heartbeat response"
    await run.respond(1, text_response("Waiting for results."))
    assert run.request_count() == 3, "completed results were not delivered after the model response"
    assert_completed_results(run.calls[2].request, 2)
    await run.respond(2, text_response("Done."))
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 2)
    assert run.request_count() == 3, (
        "heartbeats continued without pending calls or canceled operations"
    )
    assert len(run.operations.cancels) == 0, (
        "heartbeats continued without pending calls or canceled operations"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()


@pytest.mark.parametrize(
    ("pending", "interval"),
    [
        pytest.param(1, 0, id="disabled"),
        pytest.param(0, SECOND, id="idle"),
    ],
)
@synctest
async def test_coordinator_heartbeat_disabled_or_idle(pending: int, interval: int) -> None:
    run = new_heartbeat_test_run(pending)
    run.dependencies.tool_heartbeat_interval = interval / SECOND
    await run.start()
    await advance_heartbeat_time(HOUR)
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "unexpected model request"


@synctest
async def test_coordinator_heartbeat_yields_to_steering_and_results() -> None:
    run = new_heartbeat_test_run(2)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.input(external_event("steering", "check progress"))
    await advance_heartbeat_time(MINUTE)
    assert_heartbeat_count(run, 0)
    await run.respond(0, text_response("Waiting."))
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.COMPLETED)
    assert run.request_count() == 2, "result waited for a heartbeat"
    await run.respond(1, text_response("One remains."))
    await advance_heartbeat_time(MINUTE - NANOSECOND)
    assert_heartbeat_count(run, 0)
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    want_reason = (
        "Heartbeat: waited 60 seconds for tool calls.\nRunning: "
        '[{"CallID":"call-1","Name":"ViewImage","Arguments":"{}"}]'
    )
    reason = latest_heartbeat_reason(run)
    assert reason == want_reason, (
        f"heartbeat reason = {reason!r}, want only the remaining call: {want_reason!r}"
    )


@synctest
async def test_heartbeat_discards_previous_deadline() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    old_deadline = now_ns() + MINUTE
    await advance_heartbeat_time(20 * SECOND)
    await run.input(external_event("steering", "continue"))
    await run.respond(0, text_response("Waiting."))
    new_deadline = now_ns() + MINUTE
    await advance_heartbeat_time(old_deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 1, "discarded deadline started a turn"
    await advance_heartbeat_time(new_deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)


@synctest
async def test_heartbeat_interval_shorter_than_inbox_batch() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = NANOSECOND / SECOND
    await run.start()
    for index in range(3):
        await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
        assert_heartbeat_count(run, index + 1)
        assert run.request_count() == index + 1, "heartbeat prevented inbox delivery"
        await advance_heartbeat_time(SECOND)
        assert_heartbeat_count(run, index + 1)
        await run.respond(index, text_response("Waiting."))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@synctest
async def test_queued_heartbeat_preserves_active_model_response() -> None:
    run = new_heartbeat_test_run(1)
    await run.start()
    await run.input(external_event("steering", "check progress"))
    await run.input(heartbeat_input("queued-heartbeat"))
    assert run.request_count() == 1, "queued heartbeat interrupted the model"
    assert not run.calls[0].canceled, "queued heartbeat interrupted the model"
    await run.respond(0, text_response("Working."))
    assert run.request_count() == 2, "queued heartbeat was not delivered in the next turn"
    assert count_heartbeat_messages(run.calls[1].request) == 1, (
        "queued heartbeat was not delivered in the next turn"
    )
    await run.respond(1, text_response("Waiting."))
    await advance_heartbeat_time(HOUR)
    assert run.request_count() == 2, "queued heartbeat was delivered twice"


@synctest
async def test_heartbeat_waits_for_all_operations_of_call() -> None:
    run = new_heartbeat_test_run(2)
    response = cast(ModelResponse, run.store.history[1].Data)
    run.store.history[1].Data = replace(
        response, Response=replace(response.Response, Output=response.Response.Output[:1])
    )
    status = cast(ToolCallStatus, run.store.history[2].Data)
    operations = list(run.store.resume.operations)
    run.store.history[2].Data = replace(
        status,
        Operations=operations,
        Status=replace(status.Status, WaitingFor=[operations[0].ID, operations[1].ID]),
    )
    run.store.history = run.store.history[:3]
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    deadline = now_ns() + MINUTE
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.COMPLETED)
    assert run.request_count() == 0, "partial completion started a turn"
    await advance_heartbeat_time(deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    await run.respond(0, text_response("Waiting."))
    await run.update(1, Status.COMPLETED)
    assert run.request_count() == 2, "last operation did not complete the call"
    assert_stop_result(run.calls[1].request, "call-0", "completed,completed")
    await run.respond(1, text_response("Done."))
    await advance_heartbeat_time(MINUTE)
    assert_heartbeat_count(run, 1)


@synctest
async def test_coordinator_heartbeat_stops_during_cancellation() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "heartbeat ran during cancellation"
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@synctest
async def test_heartbeat_control_preserves_when_idle_stop() -> None:
    run = new_heartbeat_test_run(1)
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE), heartbeat_input("heartbeat"))
    assert run.current._stop.request.Mode == ControlMode.STOP_WHEN_IDLE, (
        "heartbeat changed the stop or failed to wake the model"
    )
    assert run.request_count() == 1, "heartbeat changed the stop or failed to wake the model"
    await run.respond(0, text_response("Waiting."))
    await run.update(0, Status.COMPLETED)
    await run.respond(1, text_response("Done."))
    run.assert_stopped()


@synctest
async def test_coordinator_heartbeat_propagates_submission_failure() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = SECOND / SECOND
    stopped_inbox = Inbox()
    stopped_inbox.close()

    def swap_inbox(_: Operation) -> None:
        run.dependencies.inbox = stopped_inbox

    run.store.on_save_operation = swap_inbox
    await run.start()
    await run.update(0, Status.AWAITING)

    await advance_heartbeat_time(SECOND)
    err = run.finished_error()
    assert caused_by(err, ClosedError), f"Run error = {err!r}, want {ClosedError.__name__}"
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "model ran after heartbeat submission failed"


@synctest
async def test_coordinator_heartbeat_requires_persistence() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = SECOND / SECOND
    want = RuntimeError("heartbeat storage failed")
    run.store.append_input_error = want
    await run.start()
    await advance_heartbeat_time(SECOND)
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert run.request_count() == 0, "model ran before heartbeat was persisted"


@pytest.mark.parametrize("stage", ["input", "turn", "response"])
@synctest
async def test_coordinator_replays_heartbeat(stage: str) -> None:
    run = new_heartbeat_test_run(1)
    run.store.history.append(stored_item(4, ItemKind.INPUT, heartbeat_input("heartbeat-1")))
    if stage != "input":
        run.store.history.append(
            stored_item(5, ItemKind.TURN, Turn(ID="heartbeat-turn", Type=TurnType.REGULAR))
        )
    if stage == "response":
        run.store.history.append(
            stored_item(
                6,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="heartbeat-turn", Response=text_response("Waiting.")),
            )
        )
    await run.start()
    built = run.dependencies.context_builder.build()
    assert count_heartbeat_messages(built.request) == 1, "heartbeat was not replayed exactly once"
    if stage == "response":
        assert run.request_count() == 0, "delivered heartbeat woke the model again on resume"
    else:
        assert run.request_count() == 1, "undelivered heartbeat was not resumed"
        assert built.request == run.calls[0].request, "undelivered heartbeat was not resumed"
    assert_heartbeat_count(run, 0)


async def test_coordinator_rejects_negative_heartbeat_interval() -> None:
    run = new_heartbeat_test_run(0)
    run.dependencies.tool_heartbeat_interval = -SECOND / SECOND
    with pytest.raises(CoordinatorError, match="heartbeat interval"):
        await run.current.run()


# --- test_grace ---------------------------------------------------------------


@pytest.mark.parametrize(
    "terminal", [Status.COMPLETED, Status.FAILED, Status.CANCELED], ids=lambda value: str(value)
)
@synctest
async def test_coordinator_tool_grace_batches_completions_until_all_calls_finish(
    terminal: Status,
) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    deadline = now_ns() + SECOND
    await update_tool_grace_call(run, "A", terminal)
    assert run.request_count() == 1, "first completion was not persisted and deferred"
    assert len(run.store.appended_statuses) == 3, "first completion was not persisted and deferred"
    await sleep_ns(100 * MILLISECOND)
    await update_tool_grace_call(run, "B", terminal)
    assert run.request_count() == 2, "last completion did not end the grace period early"
    assert now_ns() < deadline, "last completion did not end the grace period early"
    assert_stop_result(run.calls[1].request, "A", str(terminal))
    assert_stop_result(run.calls[1].request, "B", str(terminal))
    await run.respond(1, text_response("Done."))
    await sleep_ns(deadline - now_ns() + SECOND)
    assert run.request_count() == 2, "expired grace period delivered results again"


@pytest.mark.parametrize("end_grace", ["steering", "expiry"])
@synctest
async def test_coordinator_tool_grace_waits_only_for_latest_turn(end_grace: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run three tools"))
    await run.respond(0, tool_grace_response("A", "B", "C"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    if end_grace == "steering":
        await run.input(external_event("steering", "run two more tools"))
    else:
        await sleep_ns(TOOL_CALL_RUN_GRACE_PERIOD_NS)
    await run.respond(1, tool_grace_response("D", "E"))
    deadline = now_ns() + TOOL_CALL_RUN_GRACE_PERIOD_NS
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 2, "older completion ended the new turn's grace period"
    await update_tool_grace_call(run, "D", Status.COMPLETED)
    assert run.request_count() == 2, "partial completion ended the new turn's grace period"
    await update_tool_grace_call(run, "E", Status.COMPLETED)
    assert run.request_count() == 3, (
        "older running call prevented the new turn's grace period from ending early"
    )
    assert now_ns() < deadline, (
        "older running call prevented the new turn's grace period from ending early"
    )
    for call_id in ["A", "B", "D", "E"]:
        assert_stop_result(run.calls[2].request, call_id, str(Status.COMPLETED))
    assert_stop_result(run.calls[2].request, "C", TOOL_CALL_RUNNING_PAYLOAD)
    await run.respond(2, text_response("Waiting for C."))
    await update_tool_grace_call(run, "C", Status.COMPLETED)
    assert run.request_count() == 4, (
        "older completion was deferred after the new turn's grace period ended"
    )
    assert_stop_result(run.calls[3].request, "C", str(Status.COMPLETED))


@synctest
async def test_coordinator_tool_grace_deadline_does_not_reset_on_completion() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run three tools"))
    await run.respond(0, tool_grace_response("A", "B", "C"))
    deadline = now_ns() + SECOND
    await sleep_ns(100 * MILLISECOND)
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await sleep_ns(400 * MILLISECOND)
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    await sleep_ns(deadline - now_ns() - NANOSECOND)
    assert run.request_count() == 1, "partial completions started a turn before the grace deadline"
    await sleep_ns(NANOSECOND)
    assert run.request_count() == 2, (
        "pending results were not delivered at the original grace deadline"
    )
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "C", TOOL_CALL_RUNNING_PAYLOAD)
    await sleep_ns(SECOND)
    assert run.request_count() == 2, "grace deadline repeated or interrupted the continuation"
    assert not run.calls[1].canceled, "grace deadline repeated or interrupted the continuation"


@synctest
async def test_coordinator_tool_grace_expiry_without_results_does_not_start_turn() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await sleep_ns(SECOND)
    assert run.request_count() == 1, "grace expiry started a turn without pending inputs"
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    assert run.request_count() == 2, "completion after grace expiry was deferred"
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize("kind", ["external", "heartbeat"])
@synctest
async def test_coordinator_inbox_ends_tool_grace_period(kind: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    input = external_event("steering", "check progress")
    if kind == "heartbeat":
        input = heartbeat_input("heartbeat")
    await run.input(input)
    assert run.request_count() == 2, "inbox input did not end the tool grace period"
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", TOOL_CALL_RUNNING_PAYLOAD)
    await run.respond(1, text_response("Waiting."))
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 3, (
        "completion was deferred after inbox input ended the grace period"
    )
    assert_stop_result(run.calls[2].request, "B", str(Status.COMPLETED))


@synctest
async def test_coordinator_tool_grace_discards_previous_deadline() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    old_deadline = now_ns() + SECOND
    await sleep_ns(250 * MILLISECOND)
    await run.input(external_event("steering", "run another tool"))
    await run.respond(1, tool_grace_response("C"))
    new_deadline = now_ns() + SECOND
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await sleep_ns(old_deadline - now_ns() + NANOSECOND)
    assert run.request_count() == 2, "previous deadline ended the new grace period"
    await sleep_ns(new_deadline - now_ns())
    assert run.request_count() == 3, "new grace deadline did not deliver pending results"
    assert_stop_result(run.calls[2].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[2].request, "B", TOOL_CALL_RUNNING_PAYLOAD)
    assert_stop_result(run.calls[2].request, "C", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize("name", [VIEW_IMAGE_NAME, "unavailable"])
@synctest
async def test_coordinator_immediate_tool_status_bypasses_grace(name: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    response = tool_grace_response("A")
    response.Output.append(
        LLMItem(
            Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="immediate", Name=name, Arguments="{}")
        )
    )
    await run.respond(0, response)
    assert run.request_count() == 2, (
        "immediate tool status was deferred or prevented valid work from dispatching"
    )
    assert len(run.operations.adds) == 1, (
        "immediate tool status was deferred or prevented valid work from dispatching"
    )
    want = ""
    if name == "unavailable":
        want = 'tool "unavailable" is not available'
    assert_stop_result(run.calls[1].request, "immediate", want)
    assert_stop_result(run.calls[1].request, "A", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE], ids=lambda value: str(value)
)
@synctest
async def test_coordinator_stop_during_tool_grace_period(mode: ControlMode) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    await run.input(stop_input("stop", mode))
    terminal = Status.CANCELED
    if mode == ControlMode.STOP_WHEN_IDLE:
        terminal = Status.COMPLETED
        assert len(run.operations.cancels) == 0, "when-idle stop canceled pending work"
    else:
        assert len(run.operations.cancels) == 1, (
            "stop waited for grace expiry before canceling work"
        )
    assert run.request_count() == 1, "stop started a turn before the operation finished"
    await update_tool_grace_call(run, "A", terminal)
    if mode != ControlMode.STOP_HARD:
        assert run.request_count() == 2, (
            "when_idle waited for grace expiry before delivering the result"
        )
        assert_stop_result(run.calls[1].request, "A", str(terminal))
        await run.respond(1, text_response("Done."))
    else:
        assert run.request_count() == 1, "hard stop started a final turn"
    run.assert_stopped()


# --- test_settings ------------------------------------------------------------


@synctest
async def test_coordinator_settings_do_not_wake_or_interrupt_model() -> None:
    run = new_stop_test_run(0)
    await run.start()
    await run.input(settings_input("initial", Settings(ReasoningEffort=ReasoningEffort.LOW)))
    assert run.request_count() == 0, "settings woke an idle model"
    assert run.current._pending_inputs() == 0, "settings woke an idle model"
    run.assert_running()
    await run.input(external_event("prompt", "hello"))
    assert run.request_count() == 1, "next turn did not use settings"
    assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.LOW, (
        "next turn did not use settings"
    )
    await run.input(settings_input("next", Settings(ReasoningEffort=ReasoningEffort.HIGH)))
    assert run.request_count() == 1, "settings interrupted or altered the active request"
    assert not run.calls[0].canceled, "settings interrupted or altered the active request"
    assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.LOW, (
        "settings interrupted or altered the active request"
    )
    await run.respond(0, text_response("Hello."))
    assert run.request_count() == 1, "settings caused an extra turn after the response"
    assert run.current._pending_inputs() == 0, "settings caused an extra turn after the response"
    await run.input(external_event("follow-up", "continue"))
    assert run.request_count() == 2, "following turn did not use updated settings"
    assert run.calls[1].request.Model.ReasoningEffort == ReasoningEffort.HIGH, (
        "following turn did not use updated settings"
    )
    await run.respond(1, text_response("Done."))
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()


@synctest
async def test_coordinator_settings_preserve_tool_grace_and_apply_to_continuation() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("prompt", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    grace = run.current._state.grace
    await run.input(settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.MAX)))
    assert run.request_count() == 1, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert len(run.current._state.grace_tool_calls) == 1, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert run.current._state.grace is grace, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert len(run.operations.cancels) == 0, (
        "settings disturbed pending operations or ended their grace period"
    )
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 2, "automatic continuation did not use updated settings"
    assert run.calls[1].request.Model.ReasoningEffort == ReasoningEffort.MAX, (
        "automatic continuation did not use updated settings"
    )
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", str(Status.COMPLETED))


@synctest
async def test_coordinator_settings_follow_inbox_order_and_deduplicate() -> None:
    run = new_heartbeat_test_run(0)
    await run.start()
    first = settings_input("first", Settings(ReasoningEffort=ReasoningEffort.LOW))
    last = settings_input("last", Settings(ReasoningEffort=ReasoningEffort.HIGH))
    await run.input(first, last, first)
    assert run.recorded_inputs == [first, last], f"recorded settings = {run.recorded_inputs!r}"
    await run.input(external_event("prompt", "hello"))
    assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.HIGH, (
        "duplicate rolled back the latest settings"
    )


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE], ids=lambda value: str(value)
)
@synctest
async def test_coordinator_settings_preserve_pending_stop(mode: ControlMode) -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(
        stop_input("stop", mode),
        settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.HIGH)),
    )
    run.assert_running()
    assert run.current._stop.request.Mode == mode, (
        "settings changed the pending stop or started a turn"
    )
    assert run.request_count() == 0, "settings changed the pending stop or started a turn"
    if mode == ControlMode.STOP_HARD:
        assert len(run.operations.cancels) == 1, "settings prevented operation cancellation"
        await run.update(0, Status.CANCELED)
        assert run.request_count() == 0, "hard stop started a final turn"
    else:
        assert len(run.operations.cancels) == 0, "settings canceled pending work"
        await run.update(0, Status.COMPLETED)
        assert run.request_count() == 1, "final turn did not use updated settings"
        assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.HIGH, (
            "final turn did not use updated settings"
        )
        await run.respond(0, text_response("Done."))
    run.assert_stopped()


@synctest
async def test_coordinator_settings_require_persistence() -> None:
    run = new_stop_test_run(0)
    want = RuntimeError("settings storage failed")
    run.store.append_input_error = want
    await run.start()
    await run.input(settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.HIGH)))
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert run.request_count() == 0, "model ran after settings persistence failed"


@pytest.mark.parametrize("mode", ["resume", "fork"])
@synctest
async def test_coordinator_settings_replay_on_resume_and_fork(mode: str) -> None:
    parent = new_stop_test_run(0)
    store = await persist_test_run(parent)
    for input in [
        settings_input("first", Settings(ReasoningEffort=ReasoningEffort.LOW)),
        settings_input("last", Settings(ReasoningEffort=ReasoningEffort.HIGH)),
    ]:
        await store.append_input("session-1", input)
    turn = Turn(ID="parent-turn", PreviousTurnID="turn-1", Type=TurnType.REGULAR)
    await store.append_turn("session-1", turn)
    await store.append_model_response(
        "session-1", ModelResponse(TurnID=turn.ID, Response=text_response("Done."))
    )
    id = "session-1"
    if mode == "fork":
        id = "child"
        await store.fork(id, "session-1", turn.ID)
    restored = await store.resume(id)
    run = new_stop_test_run(0)
    run.dependencies.session_id = id
    run.dependencies.sessions = as_store(store)
    run.dependencies.restored = restored
    run.dependencies.context_builder.set_model(
        Model(ID="model", ReasoningEffort=ReasoningEffort.MEDIUM)
    )
    await run.start()
    assert run.request_count() == 0, "replayed settings started a turn"
    await run.input(external_event("prompt", "continue"))
    assert run.calls[0].request.Model.ID == "model", (
        "latest recorded settings did not override initial configuration"
    )
    assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.HIGH, (
        "latest recorded settings did not override initial configuration"
    )


# --- test_scheduling ----------------------------------------------------------


@synctest
async def test_coordinator_run_returns_dispatch_error_for_new_operation() -> None:
    run = new_tool_grace_test_run()
    want = RuntimeError("dispatch failed")
    run.operations.add_error = lambda _: want
    await run.start()
    await run.input(external_event("input", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(run.calls) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert len(run.store.appended_statuses) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert len(run.operations.adds) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert run.store.appended_statuses[0].Operations == run.operations.adds, (
        "dispatched operation differs from its committed state"
    )


@synctest
async def test_coordinator_dispatches_only_new_tool_operations() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("first", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    await run.input(external_event("second", "run another tool"))
    await run.respond(1, tool_grace_response("B"))
    await update_tool_grace_call(run, "A", Status.AWAITING)
    await run.input(heartbeat_input("heartbeat"))
    await run.respond(2, text_response("Waiting."))
    assert len(run.operations.adds) == 2, (
        f"operation dispatches = {run.operations.adds!r}, want each new operation once"
    )
    assert run.operations.adds[0].ID != run.operations.adds[1].ID, (
        f"operation dispatches = {run.operations.adds!r}, want each new operation once"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    await run.respond(3, text_response("A completed."))
    await run.respond(4, text_response("B completed."))
    run.assert_stopped()
    assert len(run.operations.adds) == 2, "result delivery redispatched operations"


@synctest
async def test_coordinator_scheduling_result_failure_prevents_dispatch() -> None:
    run = new_stop_test_run(0)
    want = RuntimeError("cannot translate result")
    translator = SubmittingResultErrorTranslator(want, [value_spec()])
    run.dependencies.tools = new_registry(StaticTranslators(bash=translator), BASH_NAME)

    async def respond(request: Request) -> Response:
        return Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ]
        )

    adapter = FakeAdapter(respond)
    run.dependencies.llm = adapter
    await run.start()
    await run.input(external_event("input", "run it"))
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want result translation error"
    assert len(translator.calls) == 1, "model response was not persisted and translated"
    assert len(run.store.appended_responses) == 1, "model response was not persisted and translated"
    assert len(run.store.appended_statuses) == 0, (
        "failed result translation committed a status, dispatched work, or started another request"
    )
    assert len(run.operations.adds) == 0, (
        "failed result translation committed a status, dispatched work, or started another request"
    )
    assert len(adapter.requests) == 1, (
        "failed result translation committed a status, dispatched work, or started another request"
    )


async def test_coordinator_reconciliation_rejects_untranslated_call() -> None:
    run = new_stop_test_run(1)
    run.store.history = run.store.history[:2]
    await run.current._load_history()
    statuses: list[ToolCallStatus] = []
    with pytest.raises(CoordinatorError) as caught:
        statuses = await run.current._reconcile_tool_calls()
    assert str(caught.value) == 'reconcile untranslated tool call "call-0" in turn "turn-1"', (
        f"reconciliation error = {caught.value!r}, want untranslated-call error"
    )
    assert len(statuses) == 0, "reconciliation completed an untranslated call"
    assert len(run.store.appended_statuses) == 0, "reconciliation completed an untranslated call"
    assert len(run.current._state.tool_calls) == 1, "reconciliation completed an untranslated call"
    assert run.current._pending_inputs() == 0, "reconciliation completed an untranslated call"


@pytest.mark.parametrize(
    ("start_with_input", "steer"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        "input=false/steer=false",
        "input=false/steer=true",
        "input=true/steer=false",
        "input=true/steer=true",
    ],
)
@synctest
async def test_coordinator_run_defers_completions_until_model_finishes(
    start_with_input: bool, steer: bool
) -> None:
    store, registry = independent_tool_calls(3)
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    calls: list[StopTestCall] = []

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        calls.append(call)
        return await call.response

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)
    run_task = asyncio.create_task(current.run())
    try:

        async def finish_operation(index: int) -> None:
            operations.send(replace(store.resume.operations[index], Status=Status.COMPLETED))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
            assert len(store.appended_statuses) == index + 1, (
                f"completed tool statuses = {len(store.appended_statuses)}, want {index + 1}"
            )
            status = store.appended_statuses[index]
            assert status.CallID == f"call-{index}", f"completed call = {status.CallID!r}"

        first_pending = 0
        if start_with_input:
            await submit_test_input(inputs, external_event("progress", "check progress"))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        else:
            await finish_operation(0)
            first_pending = 1
        assert len(calls) == 1, f"model requests = {len(calls)}, want 1"
        first = calls[0]
        for index in range(first_pending, 3):
            await finish_operation(index)
            assert not first.canceled, f"operation {index} canceled active model request"
            assert len(calls) == 1, f"model requests = {len(calls)}, want 1"
        response = Response(
            ID="progress-response",
            Output=[
                LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="progress"))
            ],
        )
        if steer:
            await submit_test_input(inputs, external_event("steering", "summarize results"))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        else:
            first.response.set_result(response)
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        assert len(calls) == 2, f"model requests = {len(calls)}, want 2"
        next = calls[1]
        assert_completed_results(next.request, 3)
        if steer:
            assert first.canceled, "steering did not cancel active model request"
            last = next.request.Input[-1]
            assert last.Type == ItemType.MESSAGE, f"steering missing from request: {next.request!r}"
            assert cast(Message, last.Data).Text == "summarize results", (
                f"steering missing from request: {next.request!r}"
            )
        next.response.set_result(Response(ID="final-response"))
        await wait()
        assert len(calls) == 2, (
            f"model requests = {len(calls)}, turns = {len(store.appended_turns)}, want 2 each"
        )
        assert len(store.appended_turns) == 2, (
            f"model requests = {len(calls)}, turns = {len(store.appended_turns)}, want 2 each"
        )
        if not steer:
            assert store.appended_responses[0].Response == response, (
                f"active model response was not preserved: {store.appended_responses!r}"
            )
        run_task.cancel()
        await wait()
        assert run_task.cancelled(), f"Run error = {run_task!r}, want context cancellation"
    finally:
        run_task.cancel()


@synctest
async def test_coordinator_run_slurps_independent_operation_completions() -> None:
    store, registry = independent_tool_calls(3)
    operations = FakeOperationManager()
    for value in store.resume.operations:
        operations.send(replace(value, Status=Status.COMPLETED))

    async def respond(request: Request) -> Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(
        store, new_test_inbox(), operations, new_builder(), registry, adapter
    )
    run_task = asyncio.create_task(current.run())
    try:
        await wait()
        await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        requests = adapter.requests
        assert len(requests) == 1, f"model requests = {len(requests)}, want 1"
        assert_completed_results(requests[0], 3)
        assert len(store.saved_operations) == 3, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        assert len(store.appended_statuses) == 3, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        assert len(store.appended_turns) == 1, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        run_task.cancel()
        await wait()
        assert run_task.cancelled(), f"Run error = {run_task!r}, want context cancellation"
    finally:
        run_task.cancel()


async def test_coordinator_run_persists_completed_updates_before_closure() -> None:
    store, registry = independent_tool_calls(3)
    operations = FakeOperationManager()
    completed: list[Operation] = []
    for value in store.resume.operations:
        value = replace(value, Status=Status.COMPLETED)
        completed.append(value)
        operations.send(value)
    operations.close()

    async def respond(request: Request) -> Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    builder = new_builder()
    current = new_test_coordinator(store, new_test_inbox(), operations, builder, registry, adapter)
    with pytest.raises(CoordinatorError) as caught:
        await current.run()
    assert str(caught.value) == "operation updates closed", (
        f"Run error = {caught.value!r}, want closed operation updates error"
    )
    assert store.saved_operations == completed, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 3, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    built = builder.build()
    assert_completed_results(built.request, 3)


# --- test_submission ----------------------------------------------------------

COMPLETED = str(Status.COMPLETED)


@pytest.mark.parametrize(
    "stage", ["in flight", "response committed", "turn write failed", "response write failed"]
)
@synctest
async def test_coordinator_submission_boundary_survives_recovery(stage: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    faults = FailingStore(store)
    await restore_test_run(run, faults)
    run.go()
    await wait()
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 1, "completion before submission was not included immediately"
    assert run.current._state.current_turn_inputs == 1, (
        "completion before submission was not included immediately"
    )
    sent = run.calls[0].request
    original = list(sent.Input)
    assert_stop_result(sent, "call-1", COMPLETED)
    assert_stop_result(sent, "call-0", TOOL_CALL_RUNNING_PAYLOAD)
    for item in sent.Input:
        if item.Type != ItemType.TOOL_RESULT:
            continue
        result = cast(ToolResult, item.Data)
        assert not (
            result.CallID == "call-1" and result.Output[0].Value == TOOL_CALL_RUNNING_PAYLOAD
        ), "completion before submission retained its running placeholder"
    await run.update(0, Status.COMPLETED)
    await run.input(stop_input("heartbeat", ControlMode.HEARTBEAT))
    assert len(run.calls) == 1, "late inputs interrupted or replaced the request"
    assert not run.calls[0].canceled, "late inputs interrupted or replaced the request"
    suffix = [text_result("call-0", COMPLETED), user_message("user requested stop")]
    want = replace(sent, Input=[*original, *suffix])
    response = text_response("Waiting for A.")
    failure = RuntimeError("injected write failure")
    if stage == "turn write failed":
        faults.turn_error = failure
    elif stage == "response write failed":
        faults.response_error = failure
    if stage != "in flight":
        await run.respond(0, response)
        if stage != "response write failed":
            want = replace(want, Input=[*original, *response.Output, *suffix])
    if stage == "response committed":
        assert len(run.calls) == 2, "next request did not place the response before late inputs"
        assert run.calls[1].request == want, (
            "next request did not place the response before late inputs"
        )
        assert run.current._state.delivered_inputs == 1, (
            "late inputs were counted as delivered by the earlier response"
        )
        assert run.current._state.current_turn_inputs == 3, (
            "late inputs were counted as delivered by the earlier response"
        )
    assert sent.Input == original, "in-flight request was mutated"
    if stage in ("turn write failed", "response write failed"):
        err = await run.error()
        assert caused_by(err, failure), f"Run error = {err!r}, want {failure!r}"
        assert len(run.calls) == 1, "failed persistence started another model request"
    else:
        run.cancel()
        await wait()
        err = await run.error()
        assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, (
        "resume did not reconstruct submitted history and include pending inputs"
    )
    assert resumed.calls[0].request == want, (
        "resume did not reconstruct submitted history and include pending inputs"
    )
    assert_completed_results(resumed.calls[0].request, 2)
    assert len(resumed.operations.adds) == 0, "resume redispatched completed operations"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "completion delivery did not settle in one response"
    )
    assert len(resumed.calls) == 1, "completion delivery did not settle in one response"


@synctest
async def test_coordinator_steering_commits_pending_suffix_before_new_response() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    await run.start()
    await run.input(external_event("first", "first input"))
    initial = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    await run.input(external_event("steer", "steering input"))
    assert len(run.calls) == 2, "steering did not replace the request"
    assert run.calls[0].canceled, "steering did not replace the request"
    want = replace(
        initial,
        Input=[*initial.Input, text_result("call-0", COMPLETED), user_message("steering input")],
    )
    assert run.calls[1].request == want, "replacement request omitted or reordered pending inputs"
    await run.update(1, Status.COMPLETED)
    response = text_response("Current response.")
    await run.respond(1, response)
    want = replace(want, Input=[*want.Input, *response.Output, text_result("call-1", COMPLETED)])
    assert len(run.calls) == 3, "new response did not land at the replacement request boundary"
    assert run.calls[2].request == want, (
        "new response did not land at the replacement request boundary"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await run.respond(2, text_response("Done."))
    run.assert_stopped()


@pytest.mark.parametrize("kind", ["external", "heartbeat", "hard stop"])
@synctest
async def test_coordinator_recovers_pending_results_after_input_write_failure(kind: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    faults = FailingStore(store)
    await restore_test_run(run, faults)
    await run.start()
    await run.update(1, Status.COMPLETED)
    want = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    want = replace(want, Input=[*want.Input, text_result("call-0", COMPLETED)])
    input = external_event("unpersisted", "unpersisted input")
    if kind == "heartbeat":
        input = stop_input("unpersisted", ControlMode.HEARTBEAT)
    elif kind == "hard stop":
        input = stop_input("unpersisted", ControlMode.STOP_HARD)
    failure = RuntimeError("input write failed")
    faults.input_error = failure
    await run.input(input)
    err = await run.error()
    assert caused_by(err, failure), f"Run error = {err!r}, want {failure!r}"
    assert len(run.calls) == 1, "unpersisted input triggered a request or cancellation"
    assert len(run.operations.cancels) == 0, "unpersisted input triggered a request or cancellation"
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, "resume lost persisted results or retained the failed input"
    assert resumed.calls[0].request == want, (
        "resume lost persisted results or retained the failed input"
    )
    assert resumed.current._state.available_inputs == 2, (
        "resume miscounted results or redispatched completed operations"
    )
    assert len(resumed.operations.adds) == 0, (
        "resume miscounted results or redispatched completed operations"
    )
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "restored results were not delivered exactly once"
    )
    assert len(resumed.calls) == 1, "restored results were not delivered exactly once"


@synctest
async def test_coordinator_starts_new_tool_while_delivering_previous_completion() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    translator = SubmissionTranslator([value_spec(b"1")])
    run.dependencies.tools = new_registry(
        StaticTranslators(bash=translator, view_image=OperationStatusTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    await restore_test_run(run, store)
    run.go()
    await wait()
    await run.update(1, Status.COMPLETED)
    first = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    dispatches = len(run.operations.adds)
    response = Response(
        Output=[
            LLMItem(
                ProviderID="reasoning",
                Type=ItemType.REASONING,
                Data=Reasoning(
                    Raw=gojson.Raw(
                        b'{"id":"reasoning","type":"reasoning","summary":[],"encrypted_content":"opaque"}'
                    )
                ),
            ),
            LLMItem(
                ProviderID="item-C",
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="C", Name=BASH_NAME, Arguments="{}"),
            ),
        ]
    )
    await run.respond(0, response)
    assert run.request_count() == 1, (
        "previous completion started a turn instead of waiting for the new tool"
    )
    await sleep_ns(SECOND - 1)
    assert run.request_count() == 1, (
        "previous completion started a turn before the tool grace period elapsed"
    )
    await sleep_ns(1)
    want = replace(
        first,
        Input=[
            *first.Input,
            *response.Output,
            text_result("call-0", COMPLETED),
            text_result("C", TOOL_CALL_RUNNING_PAYLOAD),
        ],
    )
    assert len(run.calls) == 2, (
        "new tool response, old completion, and new running result were reordered"
    )
    assert run.calls[1].request == want, (
        "new tool response, old completion, and new running result were reordered"
    )
    work = (
        f"new work: translations={len(translator.calls)}, dispatches={len(run.operations.adds)}, "
        f"delivered={run.current._state.delivered_inputs}, "
        f"submitted={run.current._state.current_turn_inputs}"
    )
    assert len(translator.calls) == 1, work
    assert len(run.operations.adds) > dispatches, work
    assert run.current._state.delivered_inputs == 1, work
    assert run.current._state.current_turn_inputs == 2, work
    run.operations.send(replace(run.operations.adds[dispatches], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()
    second_response = text_response("Waiting for C.")
    await run.respond(1, second_response)
    want = replace(want, Input=[*want.Input, *second_response.Output, text_result("C", COMPLETED)])
    assert len(run.calls) == 3, "new completion was not delivered after its in-flight response"
    assert run.calls[2].request == want, (
        "new completion was not delivered after its in-flight response"
    )
    assert run.current._state.delivered_inputs == 2, (
        "new completion was not delivered after its in-flight response"
    )
    run.cancel()
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"
    resumed = new_stop_test_run(0)
    resumed.dependencies.tools = new_registry(
        StaticTranslators(bash=translator, view_image=OperationStatusTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, "replay changed the request or repeated completed work"
    assert resumed.calls[0].request == want, "replay changed the request or repeated completed work"
    assert len(resumed.operations.adds) == 0, (
        "replay changed the request or repeated completed work"
    )
    assert len(translator.calls) == 1, "replay changed the request or repeated completed work"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "completed calls were not delivered exactly once"
    assert resumed.current._state.available_inputs == 3, (
        "completed calls were not delivered exactly once"
    )
    assert len(resumed.calls) == 1, "completed calls were not delivered exactly once"


@synctest
async def test_coordinator_hard_stop_preserves_pending_inputs_on_replay() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    await run.start()
    await run.input(external_event("first", "check progress"))
    want = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    await run.input(stop_input("hard", ControlMode.STOP_HARD))
    assert len(run.calls) == 1, "hard stop failed to interrupt and cancel only pending work"
    assert run.calls[0].canceled, "hard stop failed to interrupt and cancel only pending work"
    assert len(run.operations.cancels) == 1, (
        "hard stop failed to interrupt and cancel only pending work"
    )
    await run.input(external_event("late", "follow up after stop"))
    await run.update(1, Status.CANCELED)
    run.assert_stopped()
    assert len(run.calls) == 1, "hard stop started another model request"
    want = replace(
        want,
        Input=[
            *want.Input,
            text_result("call-0", COMPLETED),
            user_message("follow up after stop"),
            text_result("call-1", str(Status.CANCELED)),
        ],
    )
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert resumed.calls[0].request == want, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert len(resumed.operations.adds) == 0, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert len(resumed.operations.cancels) == 0, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    await resumed.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Follow-up received."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "resumed stop history did not settle in one response"
    )
    assert len(resumed.calls) == 1, "resumed stop history did not settle in one response"


# --- test_recovery ------------------------------------------------------------


@pytest.mark.parametrize("stage", ["untranslated", "rejected", "accepted"])
@synctest
async def test_coordinator_resumes_unavailable_tool(stage: str) -> None:
    run = new_stop_test_run(1)
    want_error = 'tool "ViewImage" is not available'
    if stage == "untranslated":
        run.store.history = run.store.history[:2]
    elif stage == "rejected":
        status = cast(ToolCallStatus, run.store.history[2].Data)
        run.store.history[2].Data = replace(
            status, Status=CallStatus(Error=want_error), Operations=[]
        )
    store = await persist_test_run(run)
    before = await store.items("session-1", BEFORE_FIRST, 100)
    await restore_test_run(run, store)
    run.dependencies.tools = new_registry(StaticTranslators())
    await run.start()
    if stage != "accepted":
        await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
        assert len(run.calls) == 1, (
            "unavailable tool did not produce one corrective request without dispatch"
        )
        assert len(run.operations.adds) == 0, (
            "unavailable tool did not produce one corrective request without dispatch"
        )
        assert_stop_result(run.calls[0].request, "call-0", want_error)
        await run.respond(0, text_response("Corrected."))
        run.assert_stopped()

        resumed = new_stop_test_run(0)
        resumed.dependencies.tools = new_registry(StaticTranslators())
        await restore_test_run(resumed, store)
        await resumed.start()
        await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
        resumed.assert_stopped()
        assert len(resumed.calls) == 0, "settled validation error started work on resume"
        assert len(resumed.operations.adds) == 0, "settled validation error started work on resume"
        return
    err = run.finished_error()
    assert err is not None, f"Run error = {err!r}, want unavailable-tool error"
    assert 'tool "ViewImage" required by recorded call "call-0" is not available' in str(err), (
        f"Run error = {err!r}, want unavailable-tool error"
    )
    assert len(run.calls) == 0, "unresolvable tool started work"
    assert len(run.operations.adds) == 0, "unresolvable tool started work"
    after = await store.items("session-1", BEFORE_FIRST, 100)
    assert before == after, "failed resume changed persisted history"


@pytest.mark.parametrize(
    "stage", ["input", "turn", "response", "compaction", "compaction response"]
)
@synctest
async def test_coordinator_resumes_unanswered_input(stage: str) -> None:
    run = new_stop_test_run(0)
    run.store.history = [stored_item(1, ItemKind.INPUT, external_event("user", "hello"))]
    if stage != "input":
        turn = Turn(ID="first", Type=TurnType.REGULAR)
        if stage in ("compaction", "compaction response"):
            turn.Type = TurnType.COMPACTION
        run.store.history.append(stored_item(2, ItemKind.TURN, turn))
    if stage in ("response", "compaction response"):
        run.store.history.append(
            stored_item(
                3,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="first", Response=text_response("Hello.")),
            )
        )
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    if stage == "response":
        assert len(run.calls) == 0, "answered input caused another request"
    else:
        assert len(run.calls) == 1, f"requests = {len(run.calls)}, want resumed input delivery"
        assert run.current._state.current_turn_type == TurnType.REGULAR, (
            "resume did not start ordinary delivery of pending input"
        )
        assert run.current._state.current_turn_id != "first", (
            "resume did not start ordinary delivery of pending input"
        )
        assert run.current._pending_inputs() == 1, (
            "resume did not start ordinary delivery of pending input"
        )
        want = with_preamble(
            LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello"))
        )
        assert run.calls[0].request.Input == want, "resume changed pending input context"
        await run.respond(0, text_response("Hello."))
    assert run.current._pending_inputs() == 0, "ordinary response did not deliver pending input"
    run.assert_stopped()


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ControlMode.STOP_HARD, id=str(ControlMode.STOP_HARD)),
        pytest.param(ControlMode.STOP_WHEN_IDLE, id=str(ControlMode.STOP_WHEN_IDLE)),
    ],
)
@synctest
async def test_coordinator_stop_rejects_unsupported_operation(mode: ControlMode) -> None:
    run = new_stop_test_run(1)
    manager = LocalOperationManager()
    try:
        run.dependencies.operations = manager
        await submit_test_input(run.inputs, stop_input("stop", mode))
        await run.start()
        err = run.finished_error()
        assert caused_by(err, UnsupportedError), f"Run error = {err!r}, want unsupported operation"
        assert len(run.store.saved_operations) == 0, (
            "unsupported operation changed state or started a model request"
        )
        assert len(run.calls) == 0, "unsupported operation changed state or started a model request"
    finally:
        await manager.aclose()


@pytest.mark.parametrize(
    "terminal",
    [
        pytest.param(Status.COMPLETED, id=str(Status.COMPLETED)),
        pytest.param(Status.FAILED, id=str(Status.FAILED)),
        pytest.param(Status.CANCELED, id=str(Status.CANCELED)),
    ],
)
@synctest
async def test_coordinator_resumes_saved_terminal_operation(terminal: Status) -> None:
    run = new_stop_test_run(1)
    store = MemoryStore()
    await store.create("session-1")
    await store.append_turn("session-1", cast(Turn, run.store.history[0].Data))
    await store.append_model_response("session-1", cast(ModelResponse, run.store.history[1].Data))
    await store.append_tool_call_status(
        "session-1", cast(ToolCallStatus, run.store.history[2].Data)
    )
    completed = replace(run.store.resume.operations[0], Status=terminal)
    await store.save_operation("session-1", completed)
    restored = await store.resume("session-1")
    run.dependencies.sessions, run.dependencies.restored = store, restored
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(run.operations.adds) == 0, "saved terminal operation was dispatched again"
    assert len(run.calls) == 1, f"requests = {len(run.calls)}, want restored result delivery"
    assert_stop_result(run.calls[0].request, "call-0", str(terminal))
    await run.respond(0, text_response("Done."))
    run.assert_stopped()

    resumed = new_stop_test_run(0)
    restored = await store.resume("session-1")
    assert len(restored.operations) == 0, (
        f"settled operations retained for resume: {restored.operations!r}"
    )
    resumed.dependencies.sessions, resumed.dependencies.restored = store, restored
    await resumed.start()
    await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 0, "delivered terminal result caused work on the next resume"
    assert len(resumed.operations.adds) == 0, (
        "delivered terminal result caused work on the next resume"
    )
    resumed.assert_stopped()


# --- test_recovery_sequences --------------------------------------------------


@pytest.mark.parametrize("stage", ["steered-request", "followup-request"])
@synctest
async def test_coordinator_resumes_mixed_delivery_after_steering(stage: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    run_canceled = False

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        run.calls.append(call)
        if len(run.calls) == 1:
            while True:
                try:
                    return await asyncio.shield(call.response)
                except asyncio.CancelledError:
                    if call.response.done():
                        return call.response.result()
                    if run_canceled:
                        raise
        return await call.response

    run.dependencies.llm = FakeAdapter(respond)
    run.go()
    await wait()
    await run.input(external_event("first", "first input"))
    await run.update(0, Status.COMPLETED)
    await run.input(external_event("steer", "steering input"))
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 2, "steering did not replace the first request"
    assert run.calls[0].canceled, "steering did not replace the first request"
    await run.respond(0, text_response("Stale response."))
    assert run.current._state.delivered_inputs == 0, "stale response consumed pending inputs"
    pending = 4
    if stage == "followup-request":
        await run.respond(1, text_response("Current response."))
        pending = 1
        assert len(run.calls) == 3, "completion during the steered request did not trigger delivery"
    run_canceled = True
    run.cancel()
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    got = resumed.current._pending_inputs()
    assert got == pending, f"replayed balance = {got}, want {pending}"
    assert len(resumed.calls) == 1, f"resumed requests = {len(resumed.calls)}, want 1"
    request = resumed.calls[0].request
    assert_completed_results(request, 2)
    messages: dict[str, int] = {}
    for item in request.Input:
        if item.Type == ItemType.MESSAGE:
            text = cast(Message, item.Data).Text
            messages[text] = messages.get(text, 0) + 1
    assert messages.get("first input", 0) == 1, f"replayed messages = {messages}"
    assert messages.get("steering input", 0) == 1, f"replayed messages = {messages}"
    assert messages.get("Stale response.", 0) == 0, f"replayed messages = {messages}"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "resumed delivery did not settle exactly once"
    assert len(resumed.calls) == 1, "resumed delivery did not settle exactly once"


@synctest
async def test_coordinator_resumes_partially_completed_tool_call() -> None:
    run = new_stop_test_run(1)
    second = replace(run.store.resume.operations[0], ID="operation-1")
    run.store.resume.operations.append(second)
    status = cast(ToolCallStatus, run.store.history[2].Data)
    status = replace(
        status,
        Status=replace(status.Status, WaitingFor=[*status.Status.WaitingFor, second.ID]),
        Operations=[*status.Operations, second],
    )
    run.store.history[2].Data = status
    store = await persist_test_run(run)
    first = replace(status.Operations[0], Status=Status.COMPLETED)
    await store.save_operation("session-1", first)
    await restore_test_run(run, store)
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_running()
    assert len(run.calls) == 0, "partial completion produced a finished tool result"
    assert run.current._pending_inputs() == 0, "partial completion produced a finished tool result"
    assert len(run.operations.adds) != 0, "unfinished operation was not dispatched"
    for dispatched in run.operations.adds:
        assert dispatched == second, f"redispatched operation = {dispatched!r}, want {second!r}"
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 1, f"requests = {len(run.calls)}, want one completed tool result"
    assert_stop_result(run.calls[0].request, "call-0", "completed,completed")
    completed_results = 0
    for item in run.calls[0].request.Input:
        if (
            item.Type == ItemType.TOOL_RESULT
            and cast(ToolResult, item.Data).Output[0].Value == "completed,completed"
        ):
            completed_results += 1
    assert completed_results == 1, f"completed tool results = {completed_results}, want 1"
    await run.update(1, Status.COMPLETED)
    assert run.current._pending_inputs() == 1, (
        "duplicate terminal update changed the delivery balance"
    )
    await run.respond(0, text_response("Done."))
    run.assert_stopped()
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    assert len(resumed.dependencies.restored.operations) == 0, (
        "settled operations were retained for recovery"
    )
    await resumed.start()
    await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
    resumed.assert_stopped()
    assert len(resumed.calls) == 0, "settled tool call ran again on resume"
    assert len(resumed.operations.adds) == 0, "settled tool call ran again on resume"


@pytest.mark.parametrize("failure", ["reconciliation", "initial-build", "initial-turn"])
@synctest
async def test_coordinator_retries_recovery_persistence_failures(failure: str) -> None:
    run = new_stop_test_run(1)
    store = await persist_test_run(run)
    want = RuntimeError("injected failure")
    faults = FailingStore(store)
    terminal = Status.COMPLETED
    value = replace(run.store.resume.operations[0], Status=terminal)
    await store.save_operation("session-1", value)
    if failure == "reconciliation":
        faults.status_error = want
    elif failure == "initial-build":
        run.dependencies.context_builder = as_builder(
            FailingBuilder(run.dependencies.context_builder, want)
        )
    elif failure == "initial-turn":
        faults.turn_error = want
    await restore_test_run(run, faults)
    await run.start()
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(run.calls) == 0, "model request started before its prerequisites were committed"
    page = await store.items("session-1", BEFORE_FIRST, 100)
    turns = sum(1 for item in page.items if item.Kind == ItemKind.TURN)
    assert turns == 1, "failed recovery persisted an extra turn"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 1, "retry did not deliver the saved result without redispatch"
    assert len(resumed.operations.adds) == 0, (
        "retry did not deliver the saved result without redispatch"
    )
    assert_stop_result(resumed.calls[0].request, "call-0", str(terminal))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "retry did not settle delivery exactly once"
    assert len(resumed.calls) == 1, "retry did not settle delivery exactly once"


@synctest
async def test_coordinator_retries_initial_tool_status_persistence_failure() -> None:
    run = new_stop_test_run(1)
    run.store.history = run.store.history[:2]
    store = await persist_test_run(run)
    translator = SubmittingTranslator([value_spec(b"1")])
    registry = new_registry(StaticTranslators(view_image=translator), VIEW_IMAGE_NAME)
    run.dependencies.tools = registry
    want = RuntimeError("cannot persist initial tool status")
    faults = FailingStore(store)
    faults.status_error = want
    await restore_test_run(run, faults)
    await run.start()
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(translator.calls) == 1, (
        "failed status commit did not prevent dispatch and model requests"
    )
    assert len(run.operations.adds) == 0, (
        "failed status commit did not prevent dispatch and model requests"
    )
    assert len(run.calls) == 0, "failed status commit did not prevent dispatch and model requests"
    page = await store.items("session-1", BEFORE_FIRST, 100)
    assert len(page.items) == 2, f"history after failed commit = {page.items!r}"
    assert page.items[1].Kind == ItemKind.MODEL_RESPONSE, (
        f"history after failed commit = {page.items!r}"
    )

    resumed = new_stop_test_run(0)
    resumed.dependencies.tools = registry
    await restore_test_run(resumed, store)
    assert len(resumed.dependencies.restored.operations) == 0, (
        "failed status commit persisted operations"
    )
    await resumed.start()
    assert len(translator.calls) == 2, "retry did not translate and dispatch the unfinished call"
    assert len(resumed.operations.adds) == 1, (
        "retry did not translate and dispatch the unfinished call"
    )
    restored = await store.resume("session-1")
    assert restored.operations == resumed.operations.adds, (
        "dispatched operation differs from committed operation"
    )
    resumed.operations.send(replace(resumed.operations.adds[0], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 1, (
        f"requests = {len(resumed.calls)}, want completed result delivery"
    )
    assert_stop_result(resumed.calls[0].request, "call-0", "")
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()


# --- test_delivery ------------------------------------------------------------

DELIVERY_STEPS: list[tuple[str, ItemKind, str, int, int, int, int]] = [
    ("first completion", ItemKind.TOOL_CALL_STATUS, "first", 1, 1, 0, 1),
    ("request starts", ItemKind.TURN, "turn-1", 1, 1, 0, 1),
    ("second completes during request", ItemKind.TOOL_CALL_STATUS, "second", 0, 2, 0, 2),
    ("duplicate completion", ItemKind.TOOL_CALL_STATUS, "second", 0, 2, 0, 2),
    ("request completes", ItemKind.MODEL_RESPONSE, "response-1", 0, 2, 1, 1),
    ("next request starts", ItemKind.TURN, "turn-2", 0, 2, 1, 1),
    ("next request completes", ItemKind.MODEL_RESPONSE, "response-2", 0, 2, 2, 0),
]


@pytest.mark.parametrize(
    "step_index", range(len(DELIVERY_STEPS)), ids=[step[0] for step in DELIVERY_STEPS]
)
async def test_coordinator_replays_tool_result_balance(step_index: int) -> None:
    store, registry = independent_tool_calls(2)
    store.resume.operations = []

    def completion(index: int) -> ToolCallStatus:
        status = cast(ToolCallStatus, store.history[index + 2].Data)
        value = replace(status.Operations[0], Status=Status.COMPLETED)
        return replace(status, Operations=[value])

    first, second = completion(0), completion(1)
    data: dict[str, Any] = {
        "first": first,
        "second": second,
        "turn-1": Turn(ID="delivery-1", PreviousTurnID="turn-1", Type=TurnType.REGULAR),
        "response-1": ModelResponse(TurnID="delivery-1", Response=text_response("Checking.")),
        "turn-2": Turn(ID="delivery-2", PreviousTurnID="delivery-1", Type=TurnType.REGULAR),
        "response-2": ModelResponse(TurnID="delivery-2", Response=text_response("Done.")),
    }
    for _, kind, key, *_counts in DELIVERY_STEPS[: step_index + 1]:
        store.history.append(stored_item(len(store.history) + 1, kind, data[key]))
    _, _, _, pending_calls, completed, delivered, pending_results = DELIVERY_STEPS[step_index]

    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    await current._restore()
    got = len(current._state.tool_calls)
    assert got == pending_calls, f"pending calls = {got}, want {pending_calls}"
    got = current._state.available_inputs
    assert got == completed, f"completed results = {got}, want {completed}"
    built = current._dependencies.context_builder.build()
    appended = 0
    for item in built.request.Input:
        if (
            item.Type == ItemType.TOOL_RESULT
            and cast(ToolResult, item.Data).Output[0].Value == Status.COMPLETED
        ):
            appended += 1
    assert appended == current._state.available_inputs, (
        f"appended completions = {appended}, available inputs = {current._state.available_inputs}"
    )
    got = current._state.delivered_inputs
    assert got == delivered, f"delivered results = {got}, want {delivered}"
    got = current._pending_inputs()
    assert got == pending_results, f"pending results = {got}, want {pending_results}"


async def test_coordinator_compaction_preserves_pending_inputs_on_replay(tmp_path: Path) -> None:
    run = new_stop_test_run(2)
    store = LocalFileStore(tmp_path)
    await store.create("session-1")
    current = run.current
    current._dependencies.sessions = store
    status = cast(ToolCallStatus, run.store.history[2].Data)
    completed = replace(status.Operations[0], Status=Status.COMPLETED)
    status = replace(status, Operations=[completed])
    items = [
        *run.store.history,
        Item(Kind=ItemKind.INPUT, Data=external_event("before", "Before compaction")),
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="turn-1", Type=TurnType.COMPACTION),
        ),
        Item(Kind=ItemKind.INPUT, Data=external_event("during", "During compaction")),
        Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status),
    ]
    for item in items:
        current._add_item_to_local_state(item)
        await current._store_item_in_session_store(item)
    before = current._dependencies.context_builder.build()
    response = ModelResponse(TurnID="compact", Response=text_response("Summary"))
    response.Response.Output.append(
        LLMItem(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="summary-call", Name="unknown"))
    )
    current._state.call_model = True
    statuses = await current._handle_model_response(response)
    message = "compaction response changed ordinary scheduling"
    assert len(statuses) == 0, message
    assert current._state.call_model, message
    current._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))
    state = current._state
    message = f"compaction changed pending work: {state}"
    assert state.current_turn_type == TurnType.COMPACTION, message
    assert state.delivered_inputs == 0, message
    assert current._pending_inputs() == 3, message
    assert len(state.tool_calls) == 1, message
    assert len(state.operations) == 2, message
    reopened = LocalFileStore(tmp_path)
    page = await reopened.items("session-1", 0, 100)
    assert page.items[-1].Data == response, "compaction response was not persisted"
    replayed = new_stop_test_run(0).current
    replayed._dependencies.sessions = reopened
    await replayed._load_history()
    want_state = dataclasses.replace(current._state, call_model=False)
    assert replayed._state == want_state, f"replayed state = {replayed._state}, want {want_state}"
    builders: list[Builder] = [
        current._dependencies.context_builder,
        replayed._dependencies.context_builder,
    ]
    for builder in builders:
        built = builder.build()
        assert built.request == before.request, "compaction changed active context"


# --- test_fork ----------------------------------------------------------------

FORK_PARENT_STAGES = ["unscheduled", "running", "completed", "compaction", "compaction response"]


@pytest.mark.parametrize(
    ("parent_stage", "child_tool"),
    [(stage, tool) for stage in FORK_PARENT_STAGES for tool in (False, True)],
    ids=[
        f"parent={stage}/child-tool={str(tool).lower()}"
        for stage in FORK_PARENT_STAGES
        for tool in (False, True)
    ],
)
@synctest
async def test_coordinator_fork_stops_when_child_is_idle(
    parent_stage: str, child_tool: bool
) -> None:
    parent = new_stop_test_run(1)
    if parent_stage == "unscheduled":
        parent.store.history = parent.store.history[:2]
    store = await persist_test_run(parent)
    previous_turn: TurnID = "turn-1"
    if parent_stage == "completed":
        status = cast(ToolCallStatus, parent.store.history[2].Data)
        status.Operations[0].Status = Status.COMPLETED
        await store.save_operation("session-1", status.Operations[0])
        await store.append_tool_call_status("session-1", status)
        previous_turn = "parent-final"
        await store.append_turn(
            "session-1", Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.REGULAR)
        )
        await store.append_model_response(
            "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Parent done."))
        )
    compaction = parent_stage in ("compaction", "compaction response")
    if compaction:
        previous_turn = "compact"
        await store.append_turn(
            "session-1",
            Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.COMPACTION),
        )
        if parent_stage == "compaction response":
            await store.append_model_response(
                "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Summary"))
            )
    await store.fork("child", "session-1", previous_turn)
    restored = await store.resume("child")
    child = new_stop_test_run(0)
    child.dependencies.session_id = "child"
    child.dependencies.sessions = cast(Store, store)
    child.dependencies.restored = restored
    spec = value_spec(b'{"value":1}')
    registry = new_registry(
        StaticTranslators(bash=SubmittingTranslator([spec]), view_image=TestTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    child.dependencies.tools = registry
    await child.start()
    child.assert_running()
    message = "inherited tool call became active child work"
    assert len(child.operations.adds) == 0, message
    assert len(child.current._state.tool_calls) == 0, message
    assert child.current._state.current_turn_id == previous_turn, (
        "fork lost the parent turn boundary"
    )
    message = "fork changed the parent turn kind or started inherited work"
    assert (child.current._state.current_turn_type == TurnType.COMPACTION) == compaction, message
    assert child.current._pending_inputs() == 0, message
    assert len(child.calls) == 0, message
    await child.input(
        external_event("child-user", "hello"),
        stop_input("child-stop", ControlMode.STOP_WHEN_IDLE),
    )
    assert child.current._state.current_turn_type == TurnType.REGULAR, (
        "child turn did not establish its own type"
    )
    inherited_call = False
    for item in child.calls[0].request.Input:
        if item.Type == ItemType.TOOL_CALL and cast(ToolCall, item.Data).CallID == "call-0":
            inherited_call = True
    assert inherited_call, "fork lost its parent context"
    if child_tool:
        await child.respond(
            0,
            Response(
                Output=[
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(CallID="child-call", Name=BASH_NAME, Arguments="{}"),
                    )
                ]
            ),
        )
        child.assert_running()
        message = "child tool call was not tracked"
        assert len(child.operations.adds) == 1, message
        assert len(child.current._state.tool_calls) == 1, message
        completed = replace(child.operations.adds[0], Status=Status.COMPLETED)
        child.operations.send(completed)
        await wait()
        await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        child.assert_running()
        await child.respond(1, text_response("Child tool done."))
    else:
        await child.respond(0, text_response("Child done."))
    child.assert_stopped()


# --- test_unavailable_tool ----------------------------------------------------


@pytest.mark.parametrize(
    ("status", "operations"),
    [
        pytest.param(CallStatus(), [], id="successful translation"),
        pytest.param(CallStatus(WaitingFor=["operation-1"]), [], id="pending operation"),
        pytest.param(
            CallStatus(WaitingFor=["operation-1"]),
            [Operation(ID="operation-1", Status=Status.COMPLETED)],
            id="completed operation",
        ),
        pytest.param(
            CallStatus(Error="error", WaitingFor=["operation-1"]), [], id="error with operation"
        ),
        pytest.param(
            CallStatus(Error="error"),
            [Operation(ID="operation-1", Status=Status.COMPLETED)],
            id="error with recorded operation but no waiting IDs",
        ),
    ],
)
async def test_coordinator_rejects_restore_when_recorded_call_requires_unavailable_tool(
    status: CallStatus, operations: list[Operation]
) -> None:
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(
                            Type=ItemType.TOOL_CALL,
                            Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                        )
                    ]
                ),
            ),
        ),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(TurnID="turn-1", CallID="call-1", Status=status, Operations=operations),
        ),
    ]
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(bash=SubmittingTranslator())),
    )
    error: Exception | None = None
    try:
        await current._restore()
    except Exception as raised:
        error = raised
    assert error is not None, f"restore error = {error!r}"
    assert 'tool "Bash" required by recorded call "call-1" is not available' in str(error), (
        f"restore error = {error!r}"
    )


async def test_coordinator_schedules_available_call_alongside_unavailable_call() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        new_registry(StaticTranslators(bash=translator), BASH_NAME),
    )
    valid = ToolCall(CallID="valid", Name=BASH_NAME, Arguments="{}")
    statuses = await current._handle_model_response(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(
                Output=[
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(CallID="unknown", Name="unknown-tool"),
                    ),
                    LLMItem(Type=ItemType.TOOL_CALL, Data=valid),
                ]
            ),
        )
    )
    message = (
        f"statuses = {statuses!r}, recorded = {store.appended_statuses!r}, "
        f"translated = {translator.calls!r}"
    )
    assert len(statuses) == 2, message
    assert len(store.appended_statuses) == 2, message
    assert translator.calls == [valid], message
    for status in store.appended_statuses:
        match status.CallID:
            case "unknown":
                message = f"unavailable call status = {status!r}"
                assert status.Status.Error == 'tool "unknown-tool" is not available', message
                assert len(status.Operations) == 0, message
                assert len(status.Status.WaitingFor) == 0, message
            case "valid":
                message = f"available call status = {status!r}"
                assert status.Status.Error == "", message
                assert len(status.Operations) == 1, message
                assert len(status.Status.WaitingFor) == 1, message
            case _:
                raise AssertionError(f"unexpected call status = {status!r}")


@pytest.mark.parametrize("recorded", [False, True], ids=["untranslated", "rejected"])
async def test_coordinator_restores_unavailable_tool_call(recorded: bool) -> None:
    call = ToolCall(CallID="call-1", Name="unknown-tool", Arguments="{}")
    want_error = 'tool "unknown-tool" is not available'
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
    ]
    if recorded:
        store.history.append(
            stored_item(
                3,
                ItemKind.TOOL_CALL_STATUS,
                ToolCallStatus(
                    TurnID="turn-1", CallID=call.CallID, Status=CallStatus(Error=want_error)
                ),
            )
        )
    builder = new_builder()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        builder,
        new_registry(StaticTranslators()),
    )
    await current._restore()
    want_new_statuses = 0 if recorded else 1
    statuses = await current._schedule_tool_calls()
    assert len(statuses) == want_new_statuses, f"scheduled statuses = {statuses!r}"
    built = builder.build()
    want = with_preamble(
        LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        LLMItem(
            Type=ItemType.TOOL_RESULT,
            Data=ToolResult(
                CallID=call.CallID,
                Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=want_error)],
            ),
        ),
    )
    assert built.request.Input == want, f"restored input = {built.request.Input!r}, want {want!r}"
    statuses = await current._schedule_tool_calls()
    assert len(statuses) == 0, f"rejected call was rescheduled: statuses = {statuses!r}"
    message = (
        f"rejected call remained pending: state = {current._state!r}, "
        f"statuses = {store.appended_statuses!r}"
    )
    assert len(current._state.tool_calls) == 0, message
    assert len(store.appended_statuses) == want_new_statuses, message


# --- test_stop_integration ----------------------------------------------------


def stop_integration_submit_now(inputs: Inbox, input: Input) -> None:
    """Submit from a synchronous callback; Go's Submit never blocks."""
    coroutine = inputs.submit(input)
    try:
        coroutine.send(None)
    except StopIteration:
        return
    coroutine.close()
    raise AssertionError("inbox submit suspended")


async def test_coordinator_stop_cancels_shell_process(tmp_path: Path) -> None:
    store, registry = independent_tool_calls(1)
    spec = new_shell_spec(ShellInput(Shell="/bin/sh", Command="exec sleep 30"), str(tmp_path), 64)
    value = replace(
        store.resume.operations[0],
        Status=Status.READY,
        State=spec.State,
        Type=spec.Type,
        Version=spec.Version,
        MaxOutputLength=spec.MaxOutputLength,
    )
    store.resume.operations[0] = value
    status = cast(ToolCallStatus, store.history[2].Data)
    status.Operations[0] = value
    store.history[2].Data = status
    inputs = Inbox()
    stop = stop_input("stop", ControlMode.STOP_HARD)
    process_group_id = 0

    def on_save_operation(update: Operation) -> None:
        nonlocal process_group_id
        try:
            state = gojson.decode(bytes(update.State), ShellState)
        except Exception as error:
            store.save_operation_error = error
            return
        if process_group_id == 0 and state.ProcessGroupID > 1:
            process_group_id = state.ProcessGroupID
            stop_integration_submit_now(inputs, stop)

    store.on_save_operation = on_save_operation

    async def respond(request: Request) -> Response:
        return text_response("Stopped.")

    adapter = FakeAdapter(respond)
    manager = LocalOperationManager()
    try:
        current = Coordinator(
            Dependencies(
                session_id="session-1",
                inbox=inputs,
                restored=store.resume,
                sessions=as_store(store),
                context_builder=new_builder(),
                llm=adapter,
                tools=registry,
                operations=manager,
            )
        )
        async with asyncio.timeout(10):
            await current.run()
        assert process_group_id > 1, "shell process never started"
        with pytest.raises(ProcessLookupError):
            os.killpg(process_group_id, 0)
        last = store.appended_statuses[-1]
        assert last.Operations[0].Status == Status.CANCELED, (
            f"persisted operation status = {last.Operations[0].Status}"
        )
        assert len(adapter.requests) == 0, "hard stop requested a model response"
    finally:
        await manager.aclose()


@synctest
async def test_coordinator_stop_after_tool_commit_settles_operation() -> None:
    run = new_tool_grace_test_run()
    store: MemoryStore = await persist_test_run(run)
    await restore_test_run(run, store)
    manager = LocalOperationManager()
    run.dependencies.operations = manager
    stop_queued = False

    def observer(session_id: SessionID, item: Item) -> None:
        nonlocal stop_queued
        if item.Kind != ItemKind.TOOL_CALL_STATUS or stop_queued:
            return
        status = cast(ToolCallStatus, item.Data)
        if len(status.Operations) == 0:
            return
        stop_queued = True
        stop_integration_submit_now(run.inputs, stop_input("stop", ControlMode.STOP_HARD))

    observer_id = store.add_observer(observer)
    try:
        await run.start()
        await run.input(external_event("input", "run tool"))
        await run.respond(0, tool_grace_response("A"))
        run.assert_stopped()
        message = "stop was not queued at the commit boundary or started another turn"
        assert stop_queued, message
        assert len(run.calls) == 1, message
        page = await store.items("session-1", BEFORE_FIRST, 100)
        last = page.items[-1]
        assert last.Kind == ItemKind.TOOL_CALL_STATUS, (
            f"last item = {last.Kind}, want settled tool call"
        )
        status = cast(ToolCallStatus, last.Data)
        message = f"persisted tool result = {status!r}, want completed value operation"
        assert status.CallID == "A", message
        assert len(status.Operations) == 1, message
        assert status.Operations[0].Status == Status.COMPLETED, message
        restored = await store.resume("session-1")
        assert len(restored.operations) == 0, (
            "stop left an operation unsettled in the session store"
        )
    finally:
        store.remove_observer(observer_id)
        await manager.aclose()
