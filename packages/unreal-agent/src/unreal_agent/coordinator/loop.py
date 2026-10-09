import asyncio
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final, NamedTuple, cast

from msgspec.structs import replace

from unreal_agent import gojson
from unreal_agent._go import go_quote, go_type_name
from unreal_agent.coordinator.coordinator import CoordinatorError, Dependencies
from unreal_agent.inbox import ClosedError as InboxClosedError
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind
from unreal_agent.llm import (
    ItemType,
    Request,
    RequestOptions,
    Response,
    ToolCall,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.operation import ID as OperationID
from unreal_agent.operation import Operation, Spec, Status
from unreal_agent.session import Turn, TurnID, TurnType
from unreal_agent.sessionstore import (
    BEFORE_FIRST,
    Fork,
    Item,
    ItemKind,
    ModelResponse,
    ToolCallStatus,
)
from unreal_agent.tool import CallStatus, error_status

__all__ = [
    "HISTORY_PAGE_SIZE",
    "SLURP_IDLE_TIMEOUT",
    "SLURP_MAX_ITEMS",
    "TOOL_CALL_RUN_GRACE_PERIOD",
    "Coordinator",
]

HISTORY_PAGE_SIZE: Final = 256
SLURP_IDLE_TIMEOUT: Final = 0.001
SLURP_MAX_ITEMS: Final = 100
TOOL_CALL_RUN_GRACE_PERIOD: Final = 1.0

_G_EXPONENT_LIMIT: Final = 6

_TERMINAL: Final = frozenset({Status.COMPLETED, Status.FAILED, Status.CANCELED})
_GO_TYPE_NAMES: Final[dict[type, str]] = {
    Fork: "sessionstore.Fork",
    Input: "inbox.Input",
    Turn: "session.Turn",
    ModelResponse: "sessionstore.ModelResponse",
    ToolCallStatus: "sessionstore.ToolCallStatus",
}


class Closed:
    __slots__ = ()


CLOSED: Final = Closed()


class Source[T]:
    """At most one pending read of an async source, kept across loop iterations.

    A read starts only when the loop needs to wait, so values the loop has
    not asked for stay in the source. Once the source ends, `take` keeps
    returning `CLOSED`.
    """

    __slots__ = ("_receive", "_task", "_wake")

    def __init__(
        self, receive: Callable[[], Coroutine[Any, Any, T | Closed]], wake: Callable[[], None]
    ) -> None:
        self._receive = receive
        self._wake = wake
        self._task: asyncio.Task[T | Closed] | None = None

    @property
    def ready(self) -> bool:
        return self._task is not None and self._task.done()

    def arm(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._receive())
            self._task.add_done_callback(lambda _: self._wake())

    def take(self) -> T | Closed:
        task = self._task
        assert task is not None
        value = task.result()
        if not isinstance(value, Closed):
            self._task = None
        return value

    async def wait(self, within: float) -> bool:
        """Wait up to `within` seconds for a value; cancellation leaves the read pending."""
        self.arm()
        task = self._task
        assert task is not None
        if task.done():
            return True
        done, _ = await asyncio.wait((task,), timeout=within)
        return bool(done)

    def cancel(self) -> asyncio.Task[T | Closed] | None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        return task


def inbox_source(inbox: Inbox, wake: Callable[[], None]) -> Source[Input]:
    async def receive() -> Input | Closed:
        try:
            return await inbox.get()
        except InboxClosedError:
            return CLOSED

    return Source(receive, wake)


def iterator_source[T](values: AsyncIterator[T], wake: Callable[[], None]) -> Source[T]:
    async def receive() -> T | Closed:
        try:
            return await anext(values)
        except StopAsyncIteration:
            return CLOSED

    return Source(receive, wake)


async def slurp[T](source: Source[T]) -> list[T]:
    """Drain values until a 1ms idle gap, 100 values, or the end of the source.

    Cancellation discards the batch.
    """
    values: list[T] = []
    while len(values) < SLURP_MAX_ITEMS:
        if not await source.wait(SLURP_IDLE_TIMEOUT):
            return values
        value = source.take()
        if isinstance(value, Closed):
            return values
        values.append(value)
    return values


class Timer:
    """A one-shot deadline whose expiry wakes the loop until it is consumed."""

    __slots__ = ("_handle", "_wake", "deadline", "fired")

    def __init__(self, delay: float, wake: Callable[[], None]) -> None:
        loop = asyncio.get_running_loop()
        self.fired = False
        self._wake = wake
        self.deadline = loop.time() + delay
        self._handle = loop.call_at(self.deadline, self._fire)

    def _fire(self) -> None:
        self.fired = True
        self._wake()

    def consume(self) -> bool:
        fired, self.fired = self.fired, False
        return fired

    def cancel(self) -> None:
        self._handle.cancel()
        self.fired = False


class ToolCallKey(NamedTuple):
    turn_id: TurnID
    call_id: str


@dataclass(slots=True)
class ToolCallState:
    tool_call: ToolCall
    status: CallStatus | None = None
    operations: set[OperationID] = field(default_factory=set)


@dataclass(slots=True)
class LoopState:
    current_turn_id: TurnID = ""
    current_turn_type: TurnType = TurnType.NONE
    tool_calls: dict[ToolCallKey, ToolCallState] = field(default_factory=dict)
    operations: dict[OperationID, Operation] = field(default_factory=dict)
    available_inputs: int = 0
    delivered_inputs: int = 0
    current_turn_inputs: int = 0
    call_model: bool = False
    grace: Timer | None = None
    grace_tool_calls: set[ToolCallKey] = field(default_factory=set)


@dataclass(slots=True)
class StopState:
    request: ControlMessage = field(default_factory=ControlMessage)
    cancellation_requested: bool = False
    """Once set, no new nonterminal operations may enter coordinator state."""


@dataclass(slots=True)
class ModelResponseResult:
    turn_id: TurnID
    response: Response | None = None
    error: BaseException | None = None


class ToolCallContext:
    """The `tool.Context` for one translation; it only records submitted specs."""

    __slots__ = ("operations",)

    def __init__(self) -> None:
        self.operations: list[Operation] = []

    def submit(self, spec: Spec, /) -> OperationID:
        id = str(uuid.uuid4())
        self.operations.append(
            Operation(
                MaxOutputLength=spec.MaxOutputLength,
                ID=id,
                Type=spec.Type,
                Version=spec.Version,
                Status=Status.READY,
                State=spec.State,
                Idempotency=spec.Idempotency,
            )
        )
        return id


class Coordinator:
    """Owns one session's decision loop.

    Some failed dependencies raise their own error instead of `CoordinatorError`.
    All state is owned by the task running `run`. Each model request runs in
    its own task and is cancelled when interrupted.
    """

    def __init__(self, dependencies: Dependencies) -> None:
        self._dependencies = dependencies
        self._state = LoopState()
        self._stop = StopState()
        self._cancel_model: Callable[[], object] | None = None
        self._model_tasks: set[asyncio.Task[ModelResponseResult]] = set()
        self._model_results: deque[ModelResponseResult] = deque()
        self._wake = asyncio.Event()
        self._inputs: Source[Input] | None = None
        self._updates: Source[Operation] | None = None
        self._heartbeat: Timer | None = None
        self._next_source = 0
        self._started = False

    async def run(self) -> None:
        if self._started:
            raise RuntimeError("coordinator run is single-use")
        self._started = True
        if self._dependencies.tool_heartbeat_interval < 0:
            raise CoordinatorError("tool heartbeat interval must not be negative")
        await self._restore()
        self._inputs = inbox_source(self._dependencies.inbox, self._wake.set)
        self._updates = iterator_source(self._dependencies.operations.updates(), self._wake.set)
        try:
            await self._run()
        finally:
            await self._close()

    async def _run(self) -> None:
        statuses = await self._schedule_tool_calls()
        await self._dispatch_operations_to_manager()
        self._state.call_model = _statuses_require_model_response(statuses)

        while True:
            call_model = await self._process_events()
            if self._stop.request.Mode == ControlMode.STOP_HARD:
                if await self._handle_stop():
                    return
            elif call_model:
                await self._request_model_response()
                self._clear_tool_grace()
            if self._stop.request.Mode == ControlMode.STOP_WHEN_IDLE and self._is_idle():
                return

            if not self._is_waiting_for_only_tool_calls():
                self._cancel_heartbeat()
            elif self._heartbeat is None and self._dependencies.tool_heartbeat_interval > 0:
                self._heartbeat = Timer(self._dependencies.tool_heartbeat_interval, self._wake.set)

            await self._handle_next_event()

    async def _handle_next_event(self) -> None:
        inputs, updates = self._sources()
        inputs.arm()
        updates.arm()
        while True:
            for offset in range(5):
                index = (self._next_source + offset) % 5
                if await self._try_event(index, inputs, updates):
                    self._next_source = (index + 1) % 5
                    return
            self._wake.clear()
            await self._wake.wait()

    async def _try_event(
        self, index: int, inputs: Source[Input], updates: Source[Operation]
    ) -> bool:
        match index:
            case 0 if inputs.ready:
                received = inputs.take()
                if isinstance(received, Closed):
                    raise _closed_input_error("inbox output")
                await self._process_inputs([received])
            case 1 if updates.ready:
                update = updates.take()
                if isinstance(update, Closed):
                    raise _closed_input_error("operation updates")
                await self._process_operations([update])
            case 2 if self._heartbeat is not None and self._heartbeat.consume():
                await self._post_heartbeat()
            case 3 if self._state.grace is not None and self._state.grace.consume():
                self._clear_tool_grace()
            case 4 if self._model_results:
                result = self._model_results.popleft()
                if self._cancel_model is not None and result.turn_id == self._state.current_turn_id:
                    await self._process_model_response(result)
            case _:
                return False
        return True

    def _sources(self) -> tuple[Source[Input], Source[Operation]]:
        assert self._inputs is not None
        assert self._updates is not None
        return self._inputs, self._updates

    async def _close(self) -> None:
        self._interrupt_model()
        self._cancel_heartbeat()
        self._clear_tool_grace()
        pending: list[asyncio.Task[object]] = []
        for source in (self._inputs, self._updates):
            if source is not None and (task := source.cancel()) is not None:
                pending.append(cast(asyncio.Task[object], task))
        for task in self._model_tasks:
            task.cancel()
            pending.append(cast(asyncio.Task[object], task))
        if pending:
            await asyncio.wait(pending)

    async def _process_events(self) -> bool:
        inputs, updates = self._sources()
        await self._process_inputs(await slurp(inputs))
        await self._process_operations(await slurp(updates))
        await self._reconcile_tool_calls()
        return self._state.call_model or (
            self._pending_inputs() > 0
            and self._cancel_model is None
            and not self._state.grace_tool_calls
        )

    async def _process_inputs(self, inputs: Sequence[Input]) -> None:
        await self._handle_inbox_inputs(inputs)

    async def _process_operations(self, updates: Sequence[Operation]) -> None:
        for update in updates:
            await self._handle_operation_update(update)

    async def _process_model_response(self, result: ModelResponseResult) -> None:
        self._interrupt_model()
        if result.error is not None:
            if _cancelling():
                raise asyncio.CancelledError
            raise CoordinatorError(
                f"call model for turn {go_quote(result.turn_id)}: {result.error}"
            ) from result.error
        assert result.response is not None
        statuses = await self._handle_model_response(
            ModelResponse(TurnID=result.turn_id, Response=result.response)
        )
        for status in statuses:
            for value in status.Operations:
                await self._dispatch_operation_to_manager(value)
        if not self._state.call_model and statuses:
            for status in statuses:
                self._state.grace_tool_calls.add(ToolCallKey(status.TurnID, status.CallID))
            if self._state.grace is not None:
                self._state.grace.cancel()
            self._state.grace = Timer(TOOL_CALL_RUN_GRACE_PERIOD, self._wake.set)

    def _clear_tool_grace(self) -> None:
        self._state.grace_tool_calls.clear()
        if self._state.grace is not None:
            self._state.grace.cancel()
            self._state.grace = None

    def _cancel_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            self._heartbeat = None

    async def _handle_stop(self) -> bool:
        if not self._stop.cancellation_requested:
            self._interrupt_model()
            await self._cancel_operations()
            self._stop.cancellation_requested = True
        return not self._has_pending_operations()

    def _is_idle(self) -> bool:
        return (
            self._cancel_model is None
            and self._pending_inputs() == 0
            and not self._state.tool_calls
            and not self._has_pending_operations()
        )

    def _is_waiting_for_only_tool_calls(self) -> bool:
        return (
            self._cancel_model is None
            and self._stop.request.Mode != ControlMode.STOP_HARD
            and self._pending_inputs() == 0
            and bool(self._state.tool_calls)
        )

    async def _post_heartbeat(self) -> None:
        calls = sorted(
            (call.tool_call for call in self._state.tool_calls.values()),
            key=lambda call: call.CallID.encode("utf-8", "surrogateescape"),
        )
        try:
            running = gojson.encode(calls).decode("utf-8", "surrogateescape")
        except gojson.JSONError as error:
            raise CoordinatorError(f"encode heartbeat tool calls: {error}") from error
        interval = format_seconds(self._dependencies.tool_heartbeat_interval)
        reason = f"Heartbeat: waited {interval} seconds for tool calls.\nRunning: {running}"
        try:
            payload = gojson.encode(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason=reason))
        except gojson.JSONError as error:
            raise CoordinatorError(f"encode heartbeat: {error}") from error
        await self._dependencies.inbox.submit(
            Input(ID=str(uuid.uuid4()), Kind=InputKind.CONTROL, Payload=gojson.Raw(payload))
        )

    def _interrupt_model(self) -> None:
        if self._cancel_model is not None:
            self._cancel_model()
            self._cancel_model = None

    def _accept_stop(self, request: ControlMessage) -> None:
        if self._stop.request.Mode == ControlMode.STOP_HARD:
            return
        self._stop.request = request

    def _pending_inputs(self) -> int:
        return self._state.available_inputs - self._state.delivered_inputs

    def _has_pending_operations(self) -> bool:
        return any(value.Status not in _TERMINAL for value in self._state.operations.values())

    async def _cancel_operations(self) -> None:
        failures: list[CoordinatorError] = []
        for id, value in list(self._state.operations.items()):
            if value.Status in _TERMINAL:
                continue
            try:
                await self._dependencies.operations.cancel(id, self._stop.request.Reason)
            except Exception as error:
                failure = CoordinatorError(f"cancel operation {go_quote(id)}: {error}")
                failure.__cause__ = error
                failures.append(failure)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise CoordinatorError("\n".join(str(failure) for failure in failures)) from (
                ExceptionGroup("cancel operations", failures)
            )

    async def _request_model_response(self) -> None:
        self._interrupt_model()
        try:
            built = self._dependencies.context_builder.build()
        except Exception as error:
            raise CoordinatorError(f"build model request: {error}") from error
        turn = Turn(
            ID=str(uuid.uuid4()),
            PreviousTurnID=self._state.current_turn_id,
            Type=TurnType.REGULAR,
        )
        item = self._add_item_to_local_state(Item(Kind=ItemKind.TURN, Data=turn))
        await self._store_item_in_session_store(item)

        options = RequestOptions(CacheKey=self._dependencies.session_id)
        task = asyncio.create_task(self._respond(turn.ID, built.request, options))
        self._model_tasks.add(task)
        task.add_done_callback(partial(self._model_done, turn.ID))
        self._cancel_model = task.cancel
        self._state.call_model = False

    async def _respond(
        self, turn_id: TurnID, request: Request, options: RequestOptions
    ) -> ModelResponseResult:
        try:
            response = await self._dependencies.llm.respond(request, options)
        except Exception as error:
            return ModelResponseResult(turn_id, error=error)
        return ModelResponseResult(turn_id, response=response)

    def _model_done(self, turn_id: TurnID, task: asyncio.Task[ModelResponseResult]) -> None:
        self._model_tasks.discard(task)
        if task.cancelled():
            result = ModelResponseResult(turn_id, error=asyncio.CancelledError())
        else:
            result = task.result()
        self._model_results.append(result)
        self._wake.set()

    async def _handle_inbox_input(self, input: Input) -> None:
        item = self._add_item_to_local_state(Item(Kind=ItemKind.INPUT, Data=input))
        await self._store_item_in_session_store(item)
        if input.Kind == InputKind.CONTROL:
            request = input.decode_control_message()
            match request.Mode:
                case ControlMode.UPDATE_SETTINGS:
                    return
                case ControlMode.STOP_HARD | ControlMode.STOP_WHEN_IDLE:
                    self._accept_stop(request)
                case _:
                    pass
        self._clear_tool_grace()

    async def _handle_inbox_inputs(self, inputs: Sequence[Input]) -> None:
        for input in inputs:
            await self._handle_inbox_input(input)
            if input.Kind == InputKind.EXTERNAL:
                self._state.call_model = True

    async def _handle_model_response(self, response: ModelResponse) -> list[ToolCallStatus]:
        item = self._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))
        await self._store_item_in_session_store(item)
        if (
            self._state.current_turn_type == TurnType.COMPACTION
            and response.TurnID == self._state.current_turn_id
        ):
            return []
        statuses = await self._schedule_tool_calls()
        self._state.call_model = _statuses_require_model_response(statuses)
        return statuses

    async def _handle_operation_update(self, update: Operation) -> None:
        update = self._add_operation_to_local_state(update)
        await self._store_operation_in_session_store(update)

    async def _restore(self) -> None:
        await self._load_history()
        for value in self._dependencies.restored.operations:
            self._add_operation_to_local_state(value)

    async def _load_history(self) -> None:
        after = BEFORE_FIRST
        while True:
            try:
                page = await self._dependencies.sessions.items(
                    self._dependencies.session_id, after, HISTORY_PAGE_SIZE
                )
            except Exception as error:
                raise CoordinatorError(f"load session history after {after}: {error}") from error
            for item in page.items:
                try:
                    self._restore_item(item)
                except Exception as error:
                    raise CoordinatorError(f"load session item {item.Sequence}: {error}") from error
            if not page.more:
                return
            if page.next_after <= after:
                raise CoordinatorError(f"load session history did not advance after {after}")
            after = page.next_after

    def _restore_item(self, item: Item) -> None:
        status = item.Data
        if (
            item.Kind == ItemKind.TOOL_CALL_STATUS
            and isinstance(status, ToolCallStatus)
            and _requires_translator(status)
        ):
            call = self._state.tool_calls.get(ToolCallKey(status.TurnID, status.CallID))
            if call is not None and self._dependencies.tools.resolve(call.tool_call.Name) is None:
                raise CoordinatorError(
                    f"tool {go_quote(call.tool_call.Name)} required by recorded call "
                    f"{go_quote(status.CallID)} is not available"
                )
        self._add_item_to_local_state(item)

    def _add_item_to_local_state(self, item: Item) -> Item:
        """Apply one history item to local state and the context builder.

        Raises `CoordinatorError` for mismatched data, or the context
        builder's error for an undecodable control payload.
        """
        data = item.Data
        match item.Kind:
            case ItemKind.FORK:
                if not isinstance(data, Fork):
                    raise _wrong_data("fork", data, Fork)
                # FIXME: Forks leave inherited calls without results and retain
                # pending-input accounting.
                self._state.tool_calls.clear()
                self._state.operations.clear()
                self._clear_tool_grace()

            case ItemKind.INPUT:
                if not isinstance(data, Input):
                    raise _wrong_data("input", data, Input)
                try:
                    data.validate()
                except ValueError as error:
                    raise CoordinatorError(f"invalid input: {error}") from error
                if data.Kind == InputKind.EXTERNAL:
                    try:
                        self._dependencies.context_builder.add_external_input(data)
                    except Exception as error:
                        raise CoordinatorError(
                            f"add input {go_quote(data.ID)} to context: {error}"
                        ) from error
                    self._state.available_inputs += 1
                if data.Kind == InputKind.CONTROL:
                    request = data.decode_control_message()
                    self._dependencies.context_builder.add_control_message(request)
                    if request.Mode == ControlMode.HEARTBEAT:
                        self._state.available_inputs += 1

            case ItemKind.TURN:
                if not isinstance(data, Turn):
                    raise _wrong_data("turn", data, Turn)
                self._state.current_turn_id = data.ID
                self._state.current_turn_type = data.Type
                self._state.current_turn_inputs = self._state.available_inputs
                self._dependencies.context_builder.commit()

            case ItemKind.MODEL_RESPONSE:
                if not isinstance(data, ModelResponse):
                    raise _wrong_data("model response", data, ModelResponse)
                if (
                    self._state.current_turn_type == TurnType.COMPACTION
                    and data.TurnID == self._state.current_turn_id
                ):
                    return item
                self._dependencies.context_builder.add_model_response(data.Response)
                if data.TurnID == self._state.current_turn_id:
                    self._state.delivered_inputs = self._state.current_turn_inputs
                self._add_tool_calls_to_local_state(data)

            case ItemKind.TOOL_CALL_STATUS:
                if not isinstance(data, ToolCallStatus):
                    raise _wrong_data("tool-call status", data, ToolCallStatus)
                for value in data.Operations:
                    self._add_operation_to_local_state(value)
                self._add_tool_call_operations_to_local_state(data)
                self._add_tool_result_to_local_state(data)

            case _:
                raise CoordinatorError(f"unsupported item kind {go_quote(item.Kind)}")
        return item

    def _add_tool_calls_to_local_state(self, response: ModelResponse) -> None:
        for output in response.Response.Output or ():
            if output.Type != ItemType.TOOL_CALL:
                continue
            call = cast(ToolCall, output.Data)
            self._state.tool_calls[ToolCallKey(response.TurnID, call.CallID)] = ToolCallState(
                tool_call=call
            )

    def _add_tool_call_operations_to_local_state(self, status: ToolCallStatus) -> None:
        call = self._state.tool_calls.get(ToolCallKey(status.TurnID, status.CallID))
        if call is None:
            return
        waiting_for = status.Status.WaitingFor
        call.status = replace(
            status.Status, WaitingFor=None if waiting_for is None else list(waiting_for)
        )
        call.operations.update(waiting_for or ())

    def _finish_tool_call(self, turn_id: TurnID, call_id: str) -> None:
        key = ToolCallKey(turn_id, call_id)
        self._state.tool_calls.pop(key, None)
        self._state.grace_tool_calls.discard(key)
        if not self._state.grace_tool_calls:
            self._clear_tool_grace()
        self._state.available_inputs += 1

    def _tool_call_operations_are_terminal(self, turn_id: TurnID, call_id: str) -> bool:
        call = self._state.tool_calls.get(ToolCallKey(turn_id, call_id))
        if call is None:
            return True
        for id in call.operations:
            value = self._state.operations.get(id)
            if value is None or value.Status not in _TERMINAL:
                return False
        return True

    def _add_tool_result_to_local_state(self, status: ToolCallStatus) -> None:
        call = self._state.tool_calls.get(ToolCallKey(status.TurnID, status.CallID))
        if call is None:
            return
        translator = self._dependencies.tools.resolve(call.tool_call.Name)
        if translator is None:
            if not _requires_translator(status):
                self._dependencies.context_builder.add_tool_result(
                    status.CallID,
                    [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=status.Status.Error)],
                    False,
                )
                self._finish_tool_call(status.TurnID, status.CallID)
            return

        operations: list[Operation] = []
        for id in status.Status.WaitingFor or ():
            if id not in call.operations:
                return
            value = self._state.operations.get(id)
            if value is None:
                return
            operations.append(value)
        try:
            result = translator.translate_result(status.CallID, status.Status, operations)
        except Exception as error:
            raise CoordinatorError(
                f"add tool call {go_quote(status.CallID)} result to context: {error}"
            ) from error
        model_result = result.to_llm_result()
        running = not self._tool_call_operations_are_terminal(status.TurnID, status.CallID)
        self._dependencies.context_builder.add_tool_result(
            status.CallID, model_result.Output, running
        )
        if not running:
            self._finish_tool_call(status.TurnID, status.CallID)

    def _add_operation_to_local_state(self, value: Operation) -> Operation:
        stored = replace(value)
        self._state.operations[value.ID] = stored
        return stored

    async def _schedule_tool_calls(self) -> list[ToolCallStatus]:
        statuses: list[ToolCallStatus] = []
        for key in list(self._state.tool_calls):
            call = self._state.tool_calls.get(key)
            if call is None or call.status is not None:
                continue
            statuses.append(await self._schedule_tool_call(key, call.tool_call))
        return statuses

    async def _schedule_tool_call(self, key: ToolCallKey, call: ToolCall) -> ToolCallStatus:
        translator = self._dependencies.tools.resolve(call.Name)
        context = ToolCallContext()
        if translator is not None:
            status = translator.translate(context, call)
        else:
            status = error_status(f"tool {go_quote(call.Name)} is not available", 0)
        operations = [self._add_operation_to_local_state(value) for value in context.operations]
        tool_call_status = ToolCallStatus(
            TurnID=key.turn_id, CallID=key.call_id, Status=status, Operations=operations
        )
        item = self._add_item_to_local_state(
            Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=tool_call_status)
        )
        await self._store_item_in_session_store(item)
        return tool_call_status

    async def _reconcile_tool_calls(self) -> list[ToolCallStatus]:
        completed: list[ToolCallStatus] = []
        for key in list(self._state.tool_calls):
            call = self._state.tool_calls.get(key)
            if call is None or not self._tool_call_operations_are_terminal(*key):
                continue
            if call.status is None:
                raise CoordinatorError(
                    f"reconcile untranslated tool call {go_quote(key.call_id)} "
                    f"in turn {go_quote(key.turn_id)}"
                )
            status = ToolCallStatus(
                TurnID=key.turn_id,
                CallID=key.call_id,
                Status=call.status,
                Operations=[self._state.operations[id] for id in call.status.WaitingFor or ()],
            )
            item = self._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))
            await self._store_item_in_session_store(item)
            if key not in self._state.tool_calls:
                completed.append(status)
        return completed

    async def _store_item_in_session_store(self, item: Item) -> None:
        sessions = self._dependencies.sessions
        id = self._dependencies.session_id
        data = item.Data
        match item.Kind:
            case ItemKind.INPUT:
                input = cast(Input, data)
                try:
                    await sessions.append_input(id, input)
                except Exception as error:
                    raise CoordinatorError(f"store input {go_quote(input.ID)}: {error}") from error
            case ItemKind.TURN:
                turn = cast(Turn, data)
                try:
                    await sessions.append_turn(id, turn)
                except Exception as error:
                    raise CoordinatorError(f"store turn {go_quote(turn.ID)}: {error}") from error
            case ItemKind.MODEL_RESPONSE:
                response = cast(ModelResponse, data)
                try:
                    await sessions.append_model_response(id, response)
                except Exception as error:
                    raise CoordinatorError(
                        f"store turn {go_quote(response.TurnID)} response: {error}"
                    ) from error
            case ItemKind.TOOL_CALL_STATUS:
                status = cast(ToolCallStatus, data)
                try:
                    await sessions.append_tool_call_status(id, status)
                except Exception as error:
                    raise CoordinatorError(
                        f"store tool call {go_quote(status.CallID)} status: {error}"
                    ) from error
            case _:
                raise CoordinatorError(f"unsupported local item kind {go_quote(item.Kind)}")

    async def _store_operation_in_session_store(self, value: Operation) -> None:
        try:
            await self._dependencies.sessions.save_operation(self._dependencies.session_id, value)
        except Exception as error:
            raise CoordinatorError(f"store operation {go_quote(value.ID)}: {error}") from error

    async def _dispatch_operations_to_manager(self) -> None:
        for value in list(self._state.operations.values()):
            await self._dispatch_operation_to_manager(value)

    async def _dispatch_operation_to_manager(self, value: Operation) -> None:
        if value.Status in _TERMINAL:
            return
        try:
            await self._dependencies.operations.add(replace(value))
        except Exception as error:
            raise CoordinatorError(f"dispatch operation {go_quote(value.ID)}: {error}") from error


def _requires_translator(status: ToolCallStatus) -> bool:
    return status.Status.Error == "" or bool(status.Status.WaitingFor) or bool(status.Operations)


def _statuses_require_model_response(statuses: Sequence[ToolCallStatus]) -> bool:
    return any(status.Status.Error != "" or not status.Status.WaitingFor for status in statuses)


def _wrong_data(label: str, data: object, want: type) -> CoordinatorError:
    return CoordinatorError(
        f"{label} data is {go_type_name(data, _GO_TYPE_NAMES)}, want {_GO_TYPE_NAMES[want]}"
    )


def _cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _closed_input_error(name: str) -> BaseException:
    if _cancelling():
        return asyncio.CancelledError()
    return CoordinatorError(f"{name} closed")


def format_seconds(seconds: float) -> str:
    """Format like Go's `%g` verb on a float64: shortest digits, exponent below 1e-4 or from 1e6."""
    if seconds == 0:
        return "0"
    sign = "-" if seconds < 0 else ""
    mantissa, _, exponent = repr(abs(seconds)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    point = len(whole) + int(exponent or 0) - (len(whole + fraction) - len(digits))
    digits = digits.rstrip("0")
    exp = point - 1
    if exp < -4 or exp >= _G_EXPONENT_LIMIT:
        body = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        return f"{sign}{body}e{'+' if exp >= 0 else '-'}{abs(exp):02d}"
    if point <= 0:
        return f"{sign}0.{'0' * -point}{digits}"
    if point >= len(digits):
        return sign + digits + "0" * (point - len(digits))
    return f"{sign}{digits[:point]}.{digits[point:]}"
