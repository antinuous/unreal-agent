import json

import pytest
from unreal_agent.gojson import Raw, encode
from unreal_agent.llm import (
    Failure,
    Item,
    ItemType,
    Message,
    ProviderItem,
    Response,
    Role,
    StopReason,
    TokenUsage,
    Usage,
)
from unreal_agent.llm.responsesapi import decode_response


def test_response_converts_web_search_response() -> None:
    response = decode_response(b"""{
        "id":"response-1",
        "status":"completed",
        "output":[
            {
                "id":"reasoning-1",
                "type":"reasoning",
                "summary":[],
                "encrypted_content":"opaque"
            },
            {
                "id":"search-1",
                "type":"web_search_call",
                "status":"completed",
                "action":{"type":"search","query":"latest news"}
            },
            {
                "id":"message-1",
                "type":"message",
                "role":"assistant",
                "status":"completed",
                "content":[{
                    "type":"output_text",
                    "text":"Current answer.",
                    "annotations":[],
                    "logprobs":[]
                }]
            }
        ]
    }""")
    assert len(response.Output or []) == 2
    reasoning = (response.Output or [])[0].Data
    assert isinstance(reasoning, ProviderItem)
    raw = json.loads(bytes(reasoning.Raw))
    assert raw["encrypted_content"] == "opaque"
    assert raw["id"] == "reasoning-1"
    message = (response.Output or [])[1].Data
    assert isinstance(message, Message)
    assert message.Text == "Current answer."


def test_response_allows_tool_call_without_status() -> None:
    response = decode_response(b"""{
        "id":"response-1",
        "status":"completed",
        "output":[{
            "id":"function-1",
            "type":"function_call",
            "call_id":"call-1",
            "name":"weather",
            "arguments":"{}"
        }]
    }""")
    assert len(response.Output or []) == 1


def test_response_ignores_unfinished_tool_call() -> None:
    response = decode_response(b"""{
        "id":"response-1",
        "status":"completed",
        "output":[{
            "id":"function-1",
            "type":"function_call",
            "call_id":"call-1",
            "name":"weather",
            "arguments":"{}",
            "status":"incomplete"
        }]
    }""")
    assert response.Output == []


def test_response_preserves_raw_usage() -> None:
    raw = (
        b'{"input_tokens":7,"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},'
        b'"output_tokens":5,"output_tokens_details":{"reasoning_tokens":4},"total_tokens":12,'
        b'"cost":0.0042,"provider_usage":{"prompt_tokens":7}}'
    )
    response = decode_response(
        b'{"id":"response-1","model":"served-model","status":"completed","output":[],"usage":'
        + raw
        + b"}"
    )
    usage = response.Usage
    assert (
        usage.InputTokens,
        usage.CachedInputTokens,
        usage.CacheWriteInputTokens,
        usage.OutputTokens,
        usage.ReasoningTokens,
    ) == (7, 3, 2, 5, 4)
    assert bytes(usage.Raw) == raw
    want = TokenUsage(
        InputTokens=7,
        CachedInputTokens=3,
        CacheWriteInputTokens=2,
        OutputTokens=5,
        ReasoningTokens=4,
    )
    assert usage.ByModel == {"served-model": want}


@pytest.mark.parametrize("model", ["", '"model":null,', '"model":"",'])
@pytest.mark.parametrize("usage", ["", "null", "{}", '{"input_tokens":7,"output_tokens":5}'])
def test_response_usage_without_model(model: str, usage: str) -> None:
    body = '{"id":"response-1","status":"completed","output":[],' + model + '"object":"response"'
    want: dict[str, TokenUsage] | None = None
    if usage:
        body += ',"usage":' + usage
        if usage != "null":
            want = {"": TokenUsage()}
            if usage != "{}":
                want[""] = TokenUsage(InputTokens=7, OutputTokens=5)
    response = decode_response((body + "}").encode())
    assert response.Usage.ByModel == want


@pytest.mark.parametrize(
    ("body", "want"),
    [
        pytest.param(
            b'{"id":"resp-1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},'
            b'"output":[],"usage":{"input_tokens":2,"input_tokens_details":{},"output_tokens":1,'
            b'"output_tokens_details":{}}}',
            Response(
                ID="resp-1",
                Stop=StopReason.MAX_OUTPUT_TOKENS,
                Output=[],
                Usage=Usage(
                    InputTokens=2,
                    OutputTokens=1,
                    ByModel={"": TokenUsage(InputTokens=2, OutputTokens=1)},
                    Raw=Raw(
                        b'{"input_tokens":2,"input_tokens_details":{},"output_tokens":1,'
                        b'"output_tokens_details":{}}'
                    ),
                ),
            ),
            id="truncated",
        ),
        pytest.param(
            b'{"id":"resp-2","status":"incomplete","incomplete_details":{"reason":"content_filter"},'
            b'"output":[],"usage":{"input_tokens":2,"input_tokens_details":{},"output_tokens":0,'
            b'"output_tokens_details":{}}}',
            Response(
                ID="resp-2",
                Stop=StopReason.REFUSED,
                Output=[],
                Usage=Usage(
                    InputTokens=2,
                    ByModel={"": TokenUsage(InputTokens=2)},
                    Raw=Raw(
                        b'{"input_tokens":2,"input_tokens_details":{},"output_tokens":0,'
                        b'"output_tokens_details":{}}'
                    ),
                ),
            ),
            id="refused",
        ),
        pytest.param(
            b'{"id":"resp-3","status":"failed","error":{"code":"server_error","message":"failed"},'
            b'"output":[],"usage":{"input_tokens":2,"input_tokens_details":{},"output_tokens":0,'
            b'"output_tokens_details":{}}}',
            Response(
                ID="resp-3",
                Output=[],
                Usage=Usage(
                    InputTokens=2,
                    ByModel={"": TokenUsage(InputTokens=2)},
                    Raw=Raw(
                        b'{"input_tokens":2,"input_tokens_details":{},"output_tokens":0,'
                        b'"output_tokens_details":{}}'
                    ),
                ),
                Failure=Failure(Code="server_error", Message="failed"),
            ),
            id="failed",
        ),
    ],
)
def test_response_separates_stop_reasons_from_failures(body: bytes, want: Response) -> None:
    assert decode_response(body) == want


def test_response_keeps_output_when_truncated() -> None:
    got = decode_response(b"""{
        "id":"resp-1",
        "status":"incomplete",
        "incomplete_details":{"reason":"max_output_tokens"},
        "output":[{"id":"message-1","type":"message","role":"assistant","status":"incomplete",
            "content":[{"type":"output_text","text":"partial","annotations":[],"logprobs":[]}]}]
    }""")
    assert got == Response(
        ID="resp-1",
        Stop=StopReason.MAX_OUTPUT_TOKENS,
        Output=[
            Item(
                ProviderID="message-1",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="partial"),
            )
        ],
    )


def test_response_reads_refusal_as_message_text() -> None:
    got = decode_response(b"""{
        "id":"resp-1",
        "status":"completed",
        "output":[{"id":"message-1","type":"message","role":"assistant","status":"completed",
            "content":[{"type":"refusal","refusal":"I cannot help with that."}]}]
    }""")
    assert len(got.Output or []) == 1
    assert (got.Output or [])[0].Data == Message(
        Role=Role.ASSISTANT, Text="I cannot help with that."
    )


def test_response_carries_unknown_message_phase() -> None:
    got = decode_response(b"""{
        "id":"resp-1",
        "status":"completed",
        "output":[{"id":"message-1","type":"message","role":"assistant","status":"completed",
            "phase":"analysis",
            "content":[{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}]}]
    }""")
    message = (got.Output or [])[0].Data
    assert isinstance(message, Message)
    assert message.Phase == "analysis"


def test_response_rejects_unfinished_status() -> None:
    with pytest.raises(ValueError, match='unsupported response status "queued"'):
        decode_response(b'{"id":"resp-1","status":"queued","output":[]}')


def test_response_rejects_unsupported_incomplete_reason() -> None:
    with pytest.raises(ValueError, match='unsupported incomplete reason "unknown"'):
        decode_response(
            b'{"id":"resp-1","status":"incomplete","incomplete_details":{"reason":"unknown"},'
            b'"output":[]}'
        )


def test_response_rejects_unsupported_output() -> None:
    with pytest.raises(ValueError, match='unsupported output item type "file_search_call"'):
        decode_response(
            b'{"id":"resp-1","status":"completed","output":[{"type":"file_search_call",'
            b'"id":"search-1"}]}'
        )


# Edge cases checked against the Go implementation.


def test_response_keeps_raw_reasoning_and_usage_byte_for_byte() -> None:
    response = decode_response(
        b'{"status":"completed","output":[ { "id" : "x", "type":"reasoning", "summary":[] } ],'
        b' "usage" : { "input_tokens" : 1 } }'
    )
    reasoning = (response.Output or [])[0].Data
    assert isinstance(reasoning, ProviderItem)
    assert bytes(reasoning.Raw) == b'{ "id" : "x", "type":"reasoning", "summary":[] }'
    assert bytes(response.Usage.Raw) == b'{ "input_tokens" : 1 }'


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b'{"status":"completed","output":[],"usage":{"input_tokens":1.0}}', "decode response:"),
        (b'{"status":"completed","output":[],"usage":{"input_tokens":1e2}}', "decode response:"),
        (b'{"status":"completed","created_at":"x","output":[]}', "decode response:"),
        (
            b'{"status":"completed","output":[null]}',
            'output item 0: unsupported output item type ""',
        ),
        (
            b'{"status":"completed","output":[{"type":"message","content":[null]}]}',
            'output item 0: unsupported message content type ""',
        ),
        (b'{"output":[]}', 'unsupported response status ""'),
        (b'{"status":"incomplete","output":[]}', 'unsupported incomplete reason ""'),
    ],
)
def test_response_rejects_like_go(body: bytes, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        decode_response(body)


def test_response_accepts_null_members_like_go() -> None:
    response = decode_response(
        b'{"status":"completed","output":[{"type":"reasoning","summary":[null]}],"usage":null}'
    )
    assert (response.Output or [])[0].Data == ProviderItem(
        Type="reasoning", Raw=Raw(b'{"type":"reasoning","summary":[null]}')
    )
    assert response.Usage == Usage(Raw=Raw(b"null"))
    failed = decode_response(b'{"status":"failed","output":[]}')
    assert failed.Failure == Failure(Message="response failed")


def test_response_encodes_empty_output_and_provider_item_like_go() -> None:
    failed = decode_response(b'{"status":"failed"}')
    assert failed.Output == []
    assert b'"Output":[]' in encode(failed)
    response = decode_response(b'{"status":"completed","output":[{"type":"reasoning"}]}')
    assert encode((response.Output or [])[0]) == (
        b'{"ProviderID":"","Type":"provider","Data":{"Type":"reasoning","Raw":{"type":"reasoning"}}}'
    )
