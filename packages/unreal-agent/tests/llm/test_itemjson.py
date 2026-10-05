from typing import cast

import pytest
from unreal_agent import gojson
from unreal_agent.gojson import JSONError, Raw
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Reasoning,
    Role,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)

ROUND_TRIP_ITEMS = [
    Item(
        ProviderID="message-1",
        Type=ItemType.MESSAGE,
        Data=Message(Role=Role.ASSISTANT, Text="hello", Phase="commentary"),
    ),
    Item(
        ProviderID="call-1",
        Type=ItemType.TOOL_CALL,
        Data=ToolCall(CallID="call-1", Name="test", Arguments="{}"),
    ),
    Item(
        ProviderID="result-1",
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID="call-1", Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="done")]
        ),
    ),
    Item(
        ProviderID="result-image",
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID="call-image",
            Output=[ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="image-data")],
        ),
    ),
    Item(
        ProviderID="result-mixed",
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID="call-mixed",
            Output=[
                ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Dimensions: 2000x1500"),
                ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,aGVsbG8="),
            ],
        ),
    ),
    Item(
        ProviderID="reasoning-1",
        Type=ItemType.REASONING,
        Data=Reasoning(Summary=["inspect"], Raw=Raw(b'{"encrypted":"opaque"}')),
    ),
    Item(ProviderID="reasoning-empty", Type=ItemType.REASONING, Data=Reasoning()),
]


@pytest.mark.parametrize("want", ROUND_TRIP_ITEMS, ids=lambda item: item.ProviderID)
def test_item_json_round_trip(want: Item) -> None:
    encoded = gojson.encode(want)
    assert b'"ProviderID"' in encoded
    assert b'"Data":{' in encoded
    got = gojson.decode(encoded, Item)
    assert got == want
    assert type(got.Data) is type(want.Data)
    assert gojson.encode(got) == encoded


def test_reasoning_raw_is_compacted_on_encode() -> None:
    item = Item(Type=ItemType.REASONING, Data=Reasoning(Raw=Raw(b'{ "encrypted" : [1, 2.50] }')))
    assert (
        gojson.encode(item)
        == b'{"ProviderID":"","Type":"reasoning","Data":{"Raw":{"encrypted":[1,2.50]}}}'
    )


@pytest.mark.parametrize(
    ("item", "want"),
    [
        (Item(Type=ItemType.MESSAGE, Data=ToolCall()), "llm.Message"),
        (Item(Type=ItemType.TOOL_CALL, Data=Message()), "llm.ToolCall"),
        (Item(Type=ItemType.TOOL_RESULT, Data=Message()), "llm.ToolResult"),
        (Item(Type=ItemType.REASONING, Data=Message()), "llm.Reasoning"),
        (Item(Type=cast(ItemType, "unknown"), Data=Message()), 'unsupported item type "unknown"'),
    ],
)
def test_item_json_rejects_invalid_tag_and_data_on_encode(item: Item, want: str) -> None:
    with pytest.raises(JSONError, match=want):
        gojson.encode(item)
    with pytest.raises(ValueError, match=want):
        item.validate()


@pytest.mark.parametrize(
    "encoded",
    [
        "{",
        '{"Type":[],"Data":{}}',
        '{"ProviderID":[],"Type":"message","Data":{"Role":"user"}}',
        '{"Type":"unknown","Data":{}}',
        '{"Type":"message","Data":[]}',
        '{"Type":"tool_call","Data":[]}',
        '{"Type":"tool_result","Data":[]}',
        '{"Type":"tool_result","Data":{"Output":"done"}}',
        '{"Type":"tool_result","Data":{"Output":42}}',
        '{"Type":"tool_result","Data":{"Output":{}}}',
        '{"Type":"tool_result","Data":{"Output":["done"]}}',
        '{"Type":"tool_result","Data":{"Output":[{"Value":42}]}}',
        '{"Type":"reasoning","Data":[]}',
        '{"Type":"message","Data":null}',
        '{"Type":"tool_call","Data":null}',
        '{"Type":"tool_result","Data":null}',
        '{"Type":"reasoning","Data":null}',
        "{}",
        '{"Type":"message"}',
        '{"Type":"tool_call"}',
        '{"Type":"tool_result"}',
        '{"Type":"reasoning"}',
        '{"ProviderID":"p","Type":"message"}',
    ],
)
def test_item_json_rejects_invalid_encoding(encoded: str) -> None:
    with pytest.raises(JSONError):
        gojson.decode(encoded, Item)


def test_item_json_null_data_message() -> None:
    with pytest.raises(JSONError, match="message data must not be null"):
        gojson.decode('{"Type":"message","Data":null}', Item)
