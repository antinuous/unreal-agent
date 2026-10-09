import pytest
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.contextbuilder.builder import _Builder
from unreal_agent.inbox import ControlMessage, ControlMode
from unreal_agent.llm import Item, ItemType, Message, ToolCall, ToolResult

from tests.contextbuilder._helpers import (
    add_compaction_test_turn,
    assert_compaction_turn,
    build_compaction_context,
    new_internal_builder,
    text,
    tool_result,
    usable_compaction_response,
    user,
    with_preamble,
)


@pytest.mark.parametrize(
    ("totals", "turn"),
    [
        pytest.param([], -1, id="no responses"),
        pytest.param([1_000, 10_000, 20_000], -1, id="whole history fits"),
        pytest.param([55_000], 1, id="one oversized response"),
        pytest.param([10_000, 18_000, 25_000], 1, id="largest fitting suffix"),
        pytest.param([1_000, 101_000, 104_000, 109_000, 119_000], 2, id="C D and E fit at 18k"),
        pytest.param([10_000, 18_000, 31_000, 43_000, 55_000], 4, id="D and E exceed budget"),
        pytest.param([10_000, 18_000, 35_000, 43_000, 55_000], 3, id="D and E exactly fit"),
        pytest.param([10_000, 18_000, 31_000, 43_000, 63_000], 4, id="exact final turn"),
        pytest.param([10_000, 18_000, 31_000, 43_000, 100_000], 5, id="oversized final turn"),
        pytest.param([10_000, 0, 31_000, 0, 55_000], 5, id="zero intermediate usage"),
        pytest.param([10_000, 18_000, 31_000, 43_000, 0], -1, id="zero latest usage"),
        pytest.param([0, 0, 0], -1, id="all usage missing"),
    ],
)
def test_compaction_selects_whole_turns_from_usage(totals: list[int], turn: int) -> None:
    current = new_internal_builder()
    for index, total in enumerate(totals):
        add_compaction_test_turn(current, index, total)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="staged heartbeat")
    )
    before = current.build()
    assert_compaction_turn(current, turn)
    after = current.build()
    assert before == after, "selection changed the request"
    assert len(current._staged_suffix) == 1, "selection committed staged arrivals"


@pytest.mark.parametrize(
    ("call_turn", "result_turn", "running", "want_turn"),
    [
        pytest.param(1, -1, False, 3, id="pending B"),
        pytest.param(0, -1, False, 3, id="pending A"),
        pytest.param(1, 2, False, 3, id="completed before suffix"),
        pytest.param(1, 3, False, 3, id="completed in D"),
        pytest.param(1, 5, False, 3, id="staged completion"),
        pytest.param(1, 2, True, 3, id="committed running result"),
        pytest.param(1, 5, True, 3, id="staged running result"),
        pytest.param(4, -1, False, 3, id="pending call already in E"),
        pytest.param(-1, 3, False, 3, id="orphan result in D"),
        pytest.param(-1, 5, False, 3, id="orphan staged result"),
        pytest.param(-1, 1, False, 3, id="orphan only in discarded prefix"),
    ],
)
def test_compaction_carries_pending_tool_calls(
    call_turn: int, result_turn: int, running: bool, want_turn: int
) -> None:
    current = new_internal_builder()
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        if turn == result_turn:
            current.add_tool_result("call", text("done"), running)
        output: list[Item] = []
        if turn == call_turn:
            output = [Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="call", Name="test"))]
        add_compaction_test_turn(current, turn, total, *output)
    if result_turn == 5:
        current.add_tool_result("call", text("done"), running)
    assert_compaction_turn(current, want_turn)
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction rejected"
    built = rebuilt.build()
    calls = 0
    for item in built.request.Input:
        if item.Type == ItemType.TOOL_CALL:
            calls += 1
        if item.Type == ItemType.MESSAGE:
            assert isinstance(item.Data, Message)
            assert item.Data.Text not in ("A input", "B input", "C input"), (
                f"compaction retained old message {item.Data.Text!r}"
            )
    want_calls = 0
    if call_turn >= 0 and (call_turn >= 3 or result_turn < 0 or result_turn >= 3 or running):
        want_calls = 1
    assert calls == want_calls, f"retained calls = {calls}, want {want_calls}"


def test_compaction_carries_only_unfinished_chained_calls() -> None:
    current = new_internal_builder()
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        output: list[Item] = []
        match turn:
            case 1:
                output = [Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="B", Name="test"))]
            case 2:
                current.add_tool_result("B", text("done B"), False)
                output = [Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="C", Name="test"))]
            case 3:
                current.add_tool_result("C", text("done C"), False)
            case _:
                pass
        add_compaction_test_turn(current, turn, total, *output)
    assert_compaction_turn(current, 3)
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction rejected"
    calls = [
        item.Data.CallID
        for item in rebuilt.build().request.Input
        if item.Type == ItemType.TOOL_CALL and isinstance(item.Data, ToolCall)
    ]
    assert calls == ["C"], f"carried calls = {calls}, want C"


def test_compaction_keeps_interrupted_turns_after_the_cut() -> None:
    current = new_internal_builder()
    add_compaction_test_turn(current, 0, 10_000)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="interrupted turn")
    )
    current.commit()
    current.commit()
    add_compaction_test_turn(current, 1, 25_000)
    start = current._compaction_cutoff()
    assert start == 0, f"selection = {start}, want checkpoint 0 before the interrupted turn"
    first = current._committed_prefix[current._prefix_tokens[start].committed_prefix_index].Data
    assert isinstance(first, Message)
    assert first.Text == "interrupted turn", f"first retained input = {first.Text!r}"


def test_compaction_uses_explicit_result_terminality() -> None:
    current = new_internal_builder()
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        output: list[Item] = []
        if turn == 0:
            output = [
                Item(
                    ProviderID=call_id,
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID=call_id, Name="test"),
                )
                for call_id in ("finished", "pending")
            ]
        if turn == 1:
            current.add_tool_result("finished", text(""), True)
            current.add_tool_result("finished", text(TOOL_CALL_RUNNING_PAYLOAD), False)
            current.add_tool_result("pending", text(""), True)
        add_compaction_test_turn(current, turn, total, *output)
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction rejected"
    request = build_compaction_context(rebuilt).request
    assert request.Input[2].ProviderID == "pending", f"carried call = {request.Input[2]!r}"
    for item in request.Input:
        assert not (
            item.Type == ItemType.TOOL_CALL
            and isinstance(item.Data, ToolCall)
            and item.Data.CallID == "finished"
        ), "terminal result was treated as running because of its text"
    assert isinstance(rebuilt, _Builder)
    carried = rebuilt
    result = carried._committed_prefix[3].Data
    assert isinstance(result, ToolResult)
    assert result.Running, "carried running result lost its lifecycle flag"
    # The carried call must survive a second reconstruction.
    add_compaction_test_turn(carried, 5, 20_000)
    add_compaction_test_turn(carried, 6, 35_000)
    again = carried.compact(usable_compaction_response("second summary"))
    assert again is not None, "second compaction lost the carried call"
    assert build_compaction_context(again).request.Input[2].ProviderID == "pending", (
        "second compaction lost the carried call"
    )


def test_compaction_summarizes_oversized_newest_turn() -> None:
    current = new_internal_builder()
    call = Item(
        ProviderID="pending", Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="call", Name="test")
    )
    add_compaction_test_turn(current, 0, 10_000, call)
    add_compaction_test_turn(current, 1, 40_000)
    current.add_tool_result("call", text("done"), False)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="arrived during compaction")
    )
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction rejected"
    want = with_preamble(
        user("summary"),
        call,
        tool_result("call", "done"),
        user("arrived during compaction"),
    )
    assert build_compaction_context(rebuilt).request.Input == want


def test_compaction_preserves_ordered_pending_calls_and_results() -> None:
    current = new_internal_builder()
    first = Item(
        ProviderID="provider-z",
        Type=ItemType.TOOL_CALL,
        Data=ToolCall(CallID="z", Name="first", Arguments='{"path":"first"}'),
    )
    second = Item(
        ProviderID="provider-a",
        Type=ItemType.TOOL_CALL,
        Data=ToolCall(CallID="a", Name="second", Arguments='{"path":"second"}'),
    )
    add_compaction_test_turn(current, 0, 10_000, first)
    current.add_tool_result("z", text(""), True)
    add_compaction_test_turn(current, 1, 40_000, second)
    current.add_tool_result("a", text(""), True)
    add_compaction_test_turn(current, 2, 70_000)
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction rejected"
    want = with_preamble(
        user("summary"),
        first,
        tool_result("z", TOOL_CALL_RUNNING_PAYLOAD, running=True),
        second,
        tool_result("a", TOOL_CALL_RUNNING_PAYLOAD, running=True),
    )
    assert build_compaction_context(rebuilt).request.Input == want
