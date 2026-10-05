from typing import Any, cast

import pytest
from unreal_agent import gojson
from unreal_agent.gojson import JSONError, Raw, Timestamp
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Response, Role
from unreal_agent.operation import Operation, Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Fork, Item, ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.tool import CallStatus

RECORDED_AT = Timestamp.parse("2026-08-27T12:00:00Z")

ROUND_TRIP_ITEMS = [
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.FORK,
        Data=Fork(ParentID="parent", PreviousTurnID="turn-1"),
    ),
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.INPUT,
        Data=Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"message":"hello"}')),
    ),
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.TURN,
        Data=Turn(ID="turn-1", Type=TurnType.REGULAR),
    ),
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.TURN,
        Data=Turn(ID="turn-1", Type=TurnType.COMPACTION),
    ),
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.MODEL_RESPONSE,
        Data=ModelResponse(
            TurnID="turn-1",
            Response=Response(
                Output=[
                    LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="hello"))
                ]
            ),
        ),
    ),
    Item(
        Sequence=1,
        RecordedAt=RECORDED_AT,
        Kind=ItemKind.TOOL_CALL_STATUS,
        Data=ToolCallStatus(
            TurnID="turn-1",
            CallID="call-1",
            Status=CallStatus(Error="in…3 chars truncated…id", ErrorTruncated=True),
            Operations=[
                Operation(
                    ID="operation-1",
                    Type="test",
                    Version=1,
                    Status=Status.COMPLETED,
                    MaxOutputLength=123,
                )
            ],
        ),
    ),
]


@pytest.mark.parametrize(
    "want",
    ROUND_TRIP_ITEMS,
    ids=[f"{item.Kind}-{index}" for index, item in enumerate(ROUND_TRIP_ITEMS)],
)
def test_item_json_round_trip(want: Item) -> None:
    encoded = gojson.encode(want)
    got = gojson.decode(encoded, Item)
    assert got == want
    assert type(got.Data) is type(want.Data)
    assert gojson.encode(got) == encoded


def test_item_json_preserves_nil_response_output() -> None:
    want = Item(
        Sequence=1,
        Kind=ItemKind.MODEL_RESPONSE,
        Data=ModelResponse(TurnID="turn-1", Response=Response()),
    )
    encoded = gojson.encode(want)
    assert b'"Output"' not in encoded
    assert gojson.decode(encoded, Item) == want


@pytest.mark.parametrize(
    ("kind", "data", "want"),
    [
        (ItemKind.FORK, Turn(), "sessionstore.Fork"),
        (ItemKind.INPUT, Turn(), "inbox.Input"),
        (ItemKind.TURN, Input(), "session.Turn"),
        (ItemKind.MODEL_RESPONSE, Turn(), "sessionstore.ModelResponse"),
        (ItemKind.TOOL_CALL_STATUS, Turn(), "sessionstore.ToolCallStatus"),
        ("unknown", object(), 'unsupported item kind "unknown"'),
    ],
)
def test_item_json_rejects_invalid_tag_on_encode(kind: str, data: Any, want: str) -> None:
    with pytest.raises(JSONError, match=want):
        gojson.encode(Item(Kind=cast(ItemKind, kind), Data=data))


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"Kind":[],"Data":{}}',
        b'{"Sequence":[],"Kind":"turn","Data":{}}',
        b'{"Kind":"unknown","Data":{}}',
        b'{"Kind":"fork","Data":[]}',
        b'{"Kind":"input","Data":[]}',
        b'{"Kind":"turn","Data":[]}',
        b'{"Kind":"model_response","Data":[]}',
        b'{"Kind":"tool_call_status","Data":[]}',
        b'{"Kind":"fork","Data":null}',
        b'{"Kind":"input","Data":null}',
        b'{"Kind":"turn","Data":null}',
        b'{"Kind":"model_response","Data":null}',
        b'{"Kind":"tool_call_status","Data":null}',
    ],
)
def test_item_json_rejects_invalid_tag_and_data_on_decode(encoded: bytes) -> None:
    with pytest.raises(JSONError):
        gojson.decode(encoded, Item)


def test_tool_call_status_keeps_empty_waiting_for_like_go() -> None:
    assert gojson.encode(CallStatus(Error="x")) == b'{"Error":"x"}'
    assert gojson.encode(CallStatus(WaitingFor=[])) == b'{"Error":"","WaitingFor":[]}'
    assert gojson.decode(b'{"WaitingFor":[]}', CallStatus).WaitingFor == []
    assert gojson.decode(b"{}", CallStatus).WaitingFor is None
