from typing import Any

import pytest
from unreal_agent import gojson
from unreal_agent.contextbuilder import new_builder
from unreal_agent.gojson import Raw
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    ItemType,
    Model,
    ProviderItem,
    Response,
    StopReason,
    ToolCall,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.llm.messagesapi import CacheTTL, decode_response, request_body
from unreal_agent.llm.messagesapi._wire import parse_any
from unreal_agent.llm.messagesapi.stream import StreamingState

from tests.llm.messagesapi._helpers import STREAM_MESSAGE_START, observe_events, response_body

FIRST_FALLBACK = (
    '{"type":"fallback","from":{"model":"primary"},"to":{"model":"second"},'
    '"trigger":{"type":"refusal","category":null},"extra":9007199254740993}'
)
LAST_FALLBACK = (
    '{"type":"fallback","from":{"model":"second"},"to":{"model":"final"},'
    '"trigger":{"type":"refusal","category":"cyber"},"extra":1e1000}'
)

BLOCKS = [
    '{"type":"thinking","thinking":"discard unsigned thinking"}',
    '{"type":"text","text":"First partial answer"}',
    '{"type":"tool_use","id":"call","name":"discard","input":[],"toolset_name":"unsupported"}',
    '{"type":"redacted_thinking"}',
    FIRST_FALLBACK,
    '{"type":"thinking","thinking":"discard second model"}',
    '{"type":"text","text":"Second partial answer"}',
    '{"type":"tool_use","id":"call","name":"discard","input":{},"caller":{"type":"unsupported"}}',
    LAST_FALLBACK,
    '{"type":"thinking","thinking":"Final thinking","signature":"signed","extra":9007199254740993}',
    '{"type":"redacted_thinking","data":"opaque"}',
    '{"type":"tool_use","id":"call","name":"Bash","input":{"n":9007199254740993}}',
    '{"type":"text","text":"Final answer"}',
]


def assert_json_equal(got: bytes | str, want: bytes | str) -> None:
    assert parse_any(got) == parse_any(want), f"JSON = {got!r}, want {want!r}"


@pytest.mark.parametrize("streaming", [False, True], ids=["streaming=false", "streaming=true"])
def test_fallback_continuation_persists_and_replays(streaming: bool) -> None:
    want_indices = [4, 8, 9, 10, 11, 12]
    body = response_body(
        "tool_use", "[" + ",".join([FIRST_FALLBACK, LAST_FALLBACK, *BLOCKS[9:]]) + "]"
    )
    if streaming:
        want_indices = [1, 4, 6, 8, 9, 10, 11, 12]
        state = StreamingState()
        observe_events(state, STREAM_MESSAGE_START)
        for index, block in enumerate(BLOCKS):
            observe_events(
                state,
                f'{{"type":"content_block_start","index":{index * 2},"content_block":{block}}}',
                f'{{"type":"content_block_stop","index":{index * 2}}}',
            )
        observe_events(
            state,
            '{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4,'
            '"iterations":[{"type":"fallback_message","model":"final","output_tokens":4}]}}',
            '{"type":"message_stop"}',
        )
        body = state.unwrap()
    response = decode_response(body)
    output = response.Output or []
    assert len(output) == len(want_indices)
    calls = [item.Data for item in output if item.Type == ItemType.TOOL_CALL]
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCall)
    assert (calls[0].Name, calls[0].CallID) == ("Bash", "call"), "superseded tool call survived"
    persisted = gojson.encode(response)
    assert b"discard" not in persisted
    restored = gojson.decode(persisted, Response)
    assert restored == response, "persistence changed the response"
    builder = new_builder()
    builder.set_model(Model(ID="primary"))
    builder.add_external_input(Input(ID="user", Kind=InputKind.EXTERNAL, Payload=Raw(b'"Start"')))
    builder.commit()
    builder.add_model_response(restored)
    builder.add_tool_result(
        "call", [ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Done")], False
    )
    encoded = request_body(builder.build().request, fallback=True, cache_ttl=CacheTTL.FIVE_MINUTES)
    replay: dict[str, Any] = parse_any(encoded)
    messages = replay["messages"]
    assert len(messages) == 3, encoded
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["content"]) == len(want_indices)
    assert len(messages[2]["content"]) == 1
    for position, index in enumerate(want_indices):
        assert_json_equal(gojson.encode(messages[1]["content"][position]), BLOCKS[index])


def test_fallback_stream_filters_before_building_tool_input() -> None:
    state = StreamingState()
    observe_events(
        state,
        STREAM_MESSAGE_START,
        '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use",'
        '"id":"discard","name":"Bash","input":{}}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"unfinished\\":"}}',
        '{"type":"content_block_stop","index":0}',
        '{"type":"content_block_start","index":1,"content_block":' + FIRST_FALLBACK + "}",
        '{"type":"content_block_stop","index":1}',
        '{"type":"content_block_start","index":2,"content_block":{"type":"tool_use",'
        '"id":"kept","name":"Bash","input":{}}}',
        '{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"n\\":9007199254740993}"}}',
        '{"type":"content_block_stop","index":2}',
        '{"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
        '{"type":"message_stop"}',
    )
    response = decode_response(state.unwrap())
    output = response.Output or []
    assert len(output) == 2
    boundary = output[0].Data
    assert isinstance(boundary, ProviderItem)
    assert_json_equal(bytes(boundary.Raw), FIRST_FALLBACK)
    call = output[1].Data
    assert isinstance(call, ToolCall)
    assert (call.CallID, call.Arguments) == ("kept", '{"n":9007199254740993}')


@pytest.mark.parametrize(
    "block",
    [
        '{"type":"thinking","thinking":"unsigned"}',
        '{"type":"redacted_thinking"}',
        '{"type":"tool_use","id":"call","name":"Bash","input":[]}',
        '{"type":"fallback","from":{"model":"primary"}}',
        '{"type":"fallback","to":{"model":"fallback"}}',
    ],
)
def test_fallback_preserves_final_response_validation(block: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        decode_response(response_body("end_turn", "[" + FIRST_FALLBACK + "," + block + "]"))


@pytest.mark.parametrize(
    "kind",
    [
        "server_tool_use",
        "connector_text",
        "container_upload",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
    ],
)
def test_fallback_rejects_unsupported_blocks(kind: str) -> None:
    block = f'{{"type":"{kind}"}}'
    for content in (block, block + "," + FIRST_FALLBACK, FIRST_FALLBACK + "," + block):
        with pytest.raises(ValueError, match="unsupported output block type"):
            decode_response(response_body("end_turn", "[" + content + "]"))
    state = StreamingState()
    observe_events(state, STREAM_MESSAGE_START)
    with pytest.raises(ValueError, match="unsupported output block type"):
        state.observe(
            f'{{"type":"content_block_start","index":0,"content_block":{block}}}'.encode()
        )


@pytest.mark.parametrize(
    ("stop", "want"),
    [
        ("end_turn", StopReason.COMPLETE),
        ("max_tokens", StopReason.MAX_OUTPUT_TOKENS),
        ("refusal", StopReason.REFUSED),
    ],
)
def test_fallback_block_does_not_change_stop_reason(stop: str, want: StopReason) -> None:
    response = decode_response(response_body(stop, "[" + FIRST_FALLBACK + "]"))
    assert response.Stop == want
    assert (response.Failure is not None) == (want == StopReason.REFUSED)
