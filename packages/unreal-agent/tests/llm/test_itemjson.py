from typing import cast

import pytest
from unreal_agent import gojson
from unreal_agent.gojson import JSONError, Raw
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
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
        ProviderID="result-running",
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID="call-running",
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="running")],
            Running=True,
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
        Type=ItemType.PROVIDER,
        Data=ProviderItem(
            Type="reasoning",
            Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text="inspect"),
            Raw=Raw(b'{"encrypted":"opaque"}'),
        ),
    ),
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


def test_provider_raw_is_compacted_on_encode() -> None:
    item = Item(Type=ItemType.PROVIDER, Data=ProviderItem(Raw=Raw(b'{ "encrypted" : [1, 2.50] }')))
    assert (
        gojson.encode(item)
        == b'{"ProviderID":"","Type":"provider","Data":{"Type":"","Raw":{"encrypted":[1,2.50]}}}'
    )


@pytest.mark.parametrize(
    ("item", "want"),
    [
        (Item(Type=ItemType.MESSAGE, Data=ToolCall()), "llm.Message"),
        (Item(Type=ItemType.TOOL_CALL, Data=Message()), "llm.ToolCall"),
        (Item(Type=ItemType.TOOL_RESULT, Data=Message()), "llm.ToolResult"),
        (Item(Type=ItemType.PROVIDER, Data=Message()), "llm.ProviderItem"),
        (Item(Type=ItemType.PROVIDER, Data=ProviderItem()), "provider item Raw is required"),
        (
            Item(Type=ItemType.PROVIDER, Data=ProviderItem(Raw=Raw(b"null"))),
            "provider item Raw is required",
        ),
        (
            Item(
                Type=ItemType.PROVIDER,
                Data=ProviderItem(
                    Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text="visible")
                ),
            ),
            "provider item Raw is required",
        ),
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
        '{"Type":"tool_result","Data":{"Output":"done"}}',
        '{"Type":"tool_result","Data":{"Output":42}}',
        '{"Type":"tool_result","Data":{"Output":{}}}',
        '{"Type":"tool_result","Data":{"Output":["done"]}}',
        '{"Type":"tool_result","Data":{"Output":[{"Value":42}]}}',
        '{"Type":"reasoning","Data":[]}',
        '{"Type":"provider","Data":[]}',
        '{"Type":"provider","Data":null}',
        '{"Type":"provider","Data":{}}',
        '{"Type":"provider","Data":{"Raw":null}}',
        '{"Type":"provider","Data":{"Display":{"Kind":"reasoning","Text":"visible"}}}',
        "{}",
        '{"Type":"message"}',
    ],
)
def test_item_json_rejects_invalid_encoding(encoded: str) -> None:
    with pytest.raises(JSONError):
        gojson.decode(encoded, Item)


def test_item_json_null_data_message() -> None:
    with pytest.raises(JSONError, match="message data must not be null"):
        gojson.decode('{"Type":"message","Data":null}', Item)


def _reasoning_display(text: str) -> ProviderDisplay:
    return ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text=text)


@pytest.mark.parametrize(
    ("data", "want"),
    [
        pytest.param(
            '{"Summary":["first","second"],"Raw":{"type":"reasoning",'
            '"encrypted_content":"opaque","extra":9007199254740993}}',
            ProviderItem(
                Type="reasoning",
                Raw=Raw(
                    b'{"type":"reasoning","encrypted_content":"opaque","extra":9007199254740993}'
                ),
                Display=_reasoning_display("first\n\nsecond"),
            ),
            id="responses",
        ),
        pytest.param(
            '{"Raw":{"type":"thinking","thinking":"original","signature":"signed"}}',
            ProviderItem(
                Type="thinking",
                Raw=Raw(b'{"type":"thinking","thinking":"original","signature":"signed"}'),
            ),
            id="thinking",
        ),
        pytest.param(
            '{"Raw":{"type":"redacted_thinking","data":"opaque"}}',
            ProviderItem(
                Type="redacted_thinking", Raw=Raw(b'{"type":"redacted_thinking","data":"opaque"}')
            ),
            id="redacted",
        ),
        pytest.param(
            '{"Summary":["visible"]}',
            ProviderItem(Display=_reasoning_display("visible")),
            id="summary only",
        ),
        pytest.param("{}", ProviderItem(), id="empty"),
        pytest.param(
            '{"Raw":{"type":42,"state":"opaque"}}',
            ProviderItem(Raw=Raw(b'{"type":42,"state":"opaque"}')),
            id="unrecognised payload",
        ),
        pytest.param('{"Raw":[1,2]}', ProviderItem(Raw=Raw(b"[1,2]")), id="nonobject payload"),
    ],
)
def test_item_json_migrates_legacy_reasoning(data: str, want: ProviderItem) -> None:
    item = gojson.decode(
        '{"ProviderID":"original-id","Type":"reasoning","Data":' + data + "}", Item
    )
    expected = Item(ProviderID="original-id", Type=ItemType.PROVIDER, Data=want)
    assert item == expected
    assert type(item.Type) is ItemType
    if not bytes(want.Raw):
        with pytest.raises(JSONError, match="provider item Raw is required"):
            gojson.encode(item)
        return
    encoded = gojson.encode(item)
    assert b'"Type":"reasoning","Data"' not in encoded, "wrote retired representation"
    assert b'"Summary"' not in encoded, "wrote retired representation"
    assert gojson.decode(encoded, Item) == expected, "migration changed on reload"


def test_item_json_rejects_malformed_legacy_reasoning() -> None:
    with pytest.raises(JSONError, match="decode reasoning data"):
        gojson.decode('{"Type":"reasoning","Data":{"Summary":"text"}}', Item)
