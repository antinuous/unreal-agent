import hashlib
import json

import pytest
from unreal_agent.gojson import Raw, decode
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    ReasoningEffort,
    Role,
    StopReason,
    ToolCall,
)
from unreal_agent.llm.messagesapi import StreamEndedError, decode_response, request_body
from unreal_agent.llm.messagesapi._wire import parse_any
from unreal_agent.llm.messagesapi.builder import BlockContentBuilder
from unreal_agent.llm.messagesapi.stream import StreamingState
from unreal_agent.primitives._errors import UnexpectedEOFError

from tests.llm.messagesapi._helpers import (
    STREAM_MESSAGE_START,
    live_stream,
    message,
    observe_events,
    observe_stream,
    valid_request,
)

SIGNATURE_SHA256 = "e16bd2907c90a268ba2ea1a8f3b9601e6eceae98bdfc19f0a105174099b5035b"


@pytest.mark.parametrize(
    ("name", "input_tokens", "output_tokens", "reasoning", "items", "stop"),
    [
        ("text", 18, 4, 0, 1, StopReason.COMPLETE),
        ("tools", 397, 92, 0, 2, StopReason.COMPLETE),
        ("tool-results", 553, 35, 0, 1, StopReason.COMPLETE),
        ("thinking", 74, 547, 210, 2, StopReason.COMPLETE),
        ("truncated-tool", 401, 32, 0, 0, StopReason.MAX_OUTPUT_TOKENS),
    ],
)
def test_stream_fixtures(
    name: str, input_tokens: int, output_tokens: int, reasoning: int, items: int, stop: StopReason
) -> None:
    state = StreamingState()
    observe_stream(state, live_stream(name))
    response = decode_response(state.unwrap())
    output = response.Output or []
    usage = response.Usage
    assert response.Stop == stop
    assert len(output) == items
    assert (usage.InputTokens, usage.OutputTokens, usage.ReasoningTokens) == (
        input_tokens,
        output_tokens,
        reasoning,
    )
    assert b'"service_tier":"standard"' in bytes(usage.Raw)
    assert b'"cache_creation":' in bytes(usage.Raw)
    match name:
        case "text":
            assert output[0] == message(Role.ASSISTANT, "hello")
        case "tools":
            assert output == [
                Item(
                    Type=ItemType.TOOL_CALL,
                    ProviderID="toolu_01RiEXhooJeGyuotGN6zWi4p",
                    Data=ToolCall(
                        CallID="toolu_01RiEXhooJeGyuotGN6zWi4p",
                        Name="capture",
                        Arguments='{"value":"alpha"}',
                    ),
                ),
                Item(
                    Type=ItemType.TOOL_CALL,
                    ProviderID="toolu_01Y4rMvNFq6LJFGnoBUxBruV",
                    Data=ToolCall(
                        CallID="toolu_01Y4rMvNFq6LJFGnoBUxBruV",
                        Name="capture",
                        Arguments='{"value":"beta"}',
                    ),
                ),
            ]
        case "tool-results":
            assert output[0] == message(
                Role.ASSISTANT,
                'I ran both captures at the same time, and both worked: one captured "alpha" '
                'and the other captured "beta".',
            )
        case "thinking":
            thinking = output[0].Data
            assert output[0].Type == ItemType.PROVIDER
            assert isinstance(thinking, ProviderItem)
            assert thinking.Display is None
            answer = output[1].Data
            assert isinstance(answer, Message)
            assert "7417" in answer.Text
            signature = json.loads(bytes(thinking.Raw))["signature"]
            assert hashlib.sha256(signature.encode()).hexdigest() == SIGNATURE_SHA256
            request = valid_request()
            request.Model.ReasoningEffort = ReasoningEffort.HIGH
            request.Input.extend(output)
            request.Input.append(message(Role.USER, "Continue"))
            replay = json.loads(request_body(request))["messages"]
            assert len(replay) == 3
            assert replay[1]["role"] == "assistant"
            assert len(replay[1]["content"]) == 2
            replayed = replay[1]["content"][0]
            assert replayed["type"] == "thinking"
            assert replayed["signature"] == signature


def test_stream_interleaves_sparse_blocks_in_index_order() -> None:
    state = StreamingState()
    observe_events(
        state,
        STREAM_MESSAGE_START,
        '{"type":"content_block_start","index":9,"content_block":{"type":"tool_use","id":"b",'
        '"name":"capture","input":{}}}',
        '{"type":"content_block_start","index":3,"content_block":{"type":"text",'
        '"text":"Starting "}}',
        '{"type":"content_block_start","index":6,"content_block":{"type":"tool_use","id":"a",'
        '"name":"capture","input":{}}}',
        '{"type":"content_block_delta","index":9,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"n\\":"}}',
        '{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"both"}}',
        '{"type":"ping"}',
        '{"type":"future_event","data":{"ignored":true}}',
        '{"type":"content_block_delta","index":6,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"value\\":\\"a\\"}"}}',
        '{"type":"content_block_delta","index":9,"delta":{"type":"input_json_delta",'
        '"partial_json":"9007199254740993}"}}',
        '{"type":"content_block_stop","index":9}',
        '{"type":"content_block_stop","index":9}',
        '{"type":"content_block_stop","index":3}',
        '{"type":"content_block_stop","index":6}',
        '{"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
        '{"type":"message_stop"}',
    )
    response = decode_response(state.unwrap())
    assert response.Output == [
        Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="Starting both")),
        Item(
            Type=ItemType.TOOL_CALL,
            ProviderID="a",
            Data=ToolCall(CallID="a", Name="capture", Arguments='{"value":"a"}'),
        ),
        Item(
            Type=ItemType.TOOL_CALL,
            ProviderID="b",
            Data=ToolCall(CallID="b", Name="capture", Arguments='{"n":9007199254740993}'),
        ),
    ]


@pytest.mark.parametrize(
    "stop", ["end_turn", "tool_use", "max_tokens", "model_context_window_exceeded", "refusal"]
)
def test_stream_omits_unfinalized_blocks_regardless_of_stop_reason(stop: str) -> None:
    state = StreamingState()
    observe_events(
        state,
        STREAM_MESSAGE_START,
        '{"type":"content_block_start","index":0,"content_block":{"type":"text",'
        '"text":"unfinalized"}}',
        '{"type":"content_block_start","index":1,"content_block":{"type":"thinking",'
        '"thinking":"unfinalized","signature":"signed"}}',
        '{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"open",'
        '"name":"capture","input":{}}}',
        '{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"incomplete\\":"}}',
        '{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"ready",'
        '"name":"capture","input":{}}}',
        '{"type":"content_block_stop","index":3}',
        '{"type":"content_block_start","index":7,"content_block":{"type":"redacted_thinking",'
        '"data":"unfinalized"}}',
        '{"type":"content_block_start","index":8,"content_block":{"type":"tool_use",'
        '"id":"valid-but-open","name":"capture","input":{}}}',
        '{"type":"message_delta","delta":{"stop_reason":' + json.dumps(stop) + "}}",
        '{"type":"message_stop"}',
    )
    response = decode_response(state.unwrap())
    output = response.Output or []
    assert len(output) == 1
    assert isinstance(output[0].Data, ToolCall)
    assert output[0].Data.CallID == "ready"


def test_stream_merges_message_updates_and_cumulative_usage() -> None:
    state = StreamingState()
    observe_events(
        state,
        '{"type":"message_start","message":{"id":"original","type":"message","role":"assistant",'
        '"content":[],"metadata":{"initial":true},"usage":{"input_tokens":9,'
        '"cache_read_input_tokens":80,"output_tokens":1,"details":{"a":9007199254740993,"b":2}}}}',
        '{"type":"message_delta","delta":{"id":"updated","type":"message","role":"assistant",'
        '"model":"new","metadata":null,"content":["replaced at flush"]},"usage":{"input_tokens":0,'
        '"cache_read_input_tokens":null,"output_tokens":7,"details":{"b":0,"c":1e1000}}}',
        '{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal",'
        '"category":"policy_violation","explanation":"Provider explanation"}},'
        '"usage":{"output_tokens":8,"details":{"a":null},'
        '"output_tokens_details":{"thinking_tokens":2}}}',
        '{"type":"message_delta","delta":{"id":null,"stop_reason":null,"stop_details":null,'
        '"absent":null}}',
        '{"type":"message_stop"}',
    )
    body = state.unwrap()
    want = (
        b'{"content":[],"id":"updated","metadata":{"initial":true},"model":"new",'
        b'"role":"assistant","stop_details":{"category":"policy_violation",'
        b'"explanation":"Provider explanation","type":"refusal"},"stop_reason":"refusal",'
        b'"type":"message","usage":{"cache_read_input_tokens":80,'
        b'"details":{"a":9007199254740993,"b":0,"c":1e1000},"input_tokens":0,'
        b'"output_tokens":8,"output_tokens_details":{"thinking_tokens":2}}}'
    )
    assert body == want
    response = decode_response(body)
    assert response.Usage.InputTokens == 80
    assert response.Usage.OutputTokens == 8
    assert response.Usage.ReasoningTokens == 2
    assert response.Failure is not None
    assert response.Failure.Message == "Provider explanation"


def test_stream_preserves_thinking_for_replay() -> None:
    state = StreamingState()
    observe_events(
        state,
        STREAM_MESSAGE_START,
        '{"type":"content_block_start","index":4,"content_block":{"type":"thinking",'
        '"thinking":"first ","signature":"start-","provider_state":{"n":9007199254740993}}}',
        '{"type":"content_block_delta","index":4,"delta":{"type":"thinking_delta",'
        '"thinking":"second"}}',
        '{"type":"content_block_delta","index":4,"delta":{"type":"signature_delta",'
        '"signature":"end"}}',
        '{"type":"content_block_stop","index":4}',
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        '{"type":"message_stop"}',
    )
    response = decode_response(state.unwrap())
    reasoning = (response.Output or [])[0].Data
    assert isinstance(reasoning, ProviderItem)
    assert reasoning.Display == ProviderDisplay(
        Kind=ProviderDisplayKind.REASONING, Text="first second"
    )
    assert b'"signature":"start-end"' in bytes(reasoning.Raw)
    assert b"9007199254740993" in bytes(reasoning.Raw)
    request = valid_request()
    request.Input.extend(response.Output or [])
    replay = request_body(request)
    assert b'"signature":"start-end"' in replay
    assert b"9007199254740993" in replay


_START = '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'
_DELTA = '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}'
_STOP = '{"type":"content_block_stop","index":0}'


@pytest.mark.parametrize(
    "events",
    [
        [_START],
        ['{"type":"message_delta","delta":{}}'],
        ['{"type":"message_stop"}'],
        [STREAM_MESSAGE_START, STREAM_MESSAGE_START],
        [STREAM_MESSAGE_START, _START, _START],
        [STREAM_MESSAGE_START, _DELTA],
        [STREAM_MESSAGE_START, _STOP],
        [STREAM_MESSAGE_START, _START, _STOP, _DELTA],
        [STREAM_MESSAGE_START, _START, _STOP, _START],
        [STREAM_MESSAGE_START, '{"type":"content_block_start","content_block":{"type":"text"}}'],
        [
            STREAM_MESSAGE_START,
            '{"type":"content_block_start","index":-1,"content_block":{"type":"text"}}',
        ],
        [
            STREAM_MESSAGE_START,
            _START,
            '{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta",'
            '"signature":"a"}}',
        ],
        [
            STREAM_MESSAGE_START,
            _START,
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":42}}',
        ],
        ["{"],
        ["{}"],
    ],
    ids=[
        "block before message",
        "delta before message",
        "stop before message",
        "duplicate message",
        "duplicate block",
        "nonexistent delta",
        "nonexistent stop",
        "finalized delta",
        "recreated block",
        "missing index",
        "negative index",
        "wrong delta",
        "invalid payload",
        "invalid JSON",
        "missing type",
    ],
)
def test_stream_rejects_invalid_transitions(events: list[str]) -> None:
    state = StreamingState()
    with pytest.raises(ValueError):  # noqa: PT011
        observe_events(state, *events)


def test_stream_error_messages() -> None:
    state = StreamingState()
    with pytest.raises(ValueError, match=r"^content_block_start before message_start$"):
        observe_events(state, _START)
    observe_events(state, STREAM_MESSAGE_START, _START, _STOP)
    with pytest.raises(
        ValueError,
        match=r'^content block 0: content block is finalized, cannot handle delta "\{',
    ):
        observe_events(state, _DELTA)
    with pytest.raises(ValueError, match=r"^content block 0 already exists$"):
        observe_events(state, _START)
    with pytest.raises(ValueError, match=r"^content_block_stop for nonexistent block 1$"):
        observe_events(state, '{"type":"content_block_stop","index":1}')
    with pytest.raises(ValueError, match=r"^messages stream event has no type$"):
        observe_events(state, "{}")


def test_stream_requires_message_stop() -> None:
    state = StreamingState()
    observe_events(state, STREAM_MESSAGE_START)
    with pytest.raises(StreamEndedError) as raised:
        state.unwrap()
    assert str(raised.value) == "messages stream ended without message_stop: unexpected EOF"
    assert isinstance(raised.value.__cause__, UnexpectedEOFError)


def test_stream_rejects_finalized_invalid_tool_input() -> None:
    state = StreamingState()
    observe_events(
        state,
        STREAM_MESSAGE_START,
        '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a",'
        '"name":"capture","input":{}}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"unfinished\\":"}}',
        '{"type":"content_block_stop","index":0}',
        '{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}',
    )
    with pytest.raises(
        ValueError, match=r"^content block 0: streamed tool input is not a complete JSON object$"
    ):
        observe_events(state, '{"type":"message_stop"}')


@pytest.mark.parametrize(
    ("kind", "raw"),
    [
        ("text", '{"type":"text_delta","text":"A short text chunk."}'),
        ("thinking", '{"type":"thinking_delta","thinking":"A short thinking chunk."}'),
        ("thinking", '{"type":"signature_delta","signature":"A short signature chunk."}'),
        ("tool_use", '{"type":"input_json_delta","partial_json":"{\\"value\\":\\"alpha\\"}"}'),
    ],
)
def test_block_accepts_matching_delta(kind: str, raw: str) -> None:
    BlockContentBuilder(kind=kind).handle_delta(raw.encode())


def test_assembled_body_matches_fixture_shape() -> None:
    state = StreamingState()
    observe_stream(state, live_stream("text"))
    fields = decode(state.unwrap(), dict[str, Raw])
    assert list(fields) == sorted(fields)
    assert parse_any(bytes(fields["content"])) == [{"text": "hello", "type": "text"}]
