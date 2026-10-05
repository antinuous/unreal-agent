import pytest
from unreal_agent.contextbuilder import new_builder
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Response,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.operation import Operation, Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.tool import BASH_NAME, CallStatus, StaticTranslators, new_registry

from tests.coordinator._helpers import (
    FakeOperationManager,
    SubmittingTranslator,
    empty_fake_store,
    new_test_coordinator,
    new_test_inbox,
    stored_item,
    value_spec,
    with_preamble,
)


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
