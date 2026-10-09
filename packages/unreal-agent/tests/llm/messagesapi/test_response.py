import pytest
from unreal_agent.gojson import Raw, decode
from unreal_agent.llm import (
    Failure,
    ItemType,
    Message,
    Model,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Request,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from unreal_agent.llm.messagesapi import decode_response
from unreal_agent.llm.messagesapi._wire import Usage as WireUsage
from unreal_agent.llm.messagesapi._wire import parse_any
from unreal_agent.llm.messagesapi.response import response_usage

from tests.llm.messagesapi._helpers import encode_request, message, response_body, tool_result

THINKING = (
    '{"type":"thinking","thinking":"Considering tools","signature":"signed",'
    '"provider_state":{"v":2}}'
)
REDACTED = '{"type":"redacted_thinking","data":"opaque"}'
OMITTED = '{"type":"thinking","thinking":"","signature":"omitted-but-signed"}'


def test_response_preserves_block_order_and_thinking() -> None:
    decoded = decode_response(
        response_body(
            "tool_use",
            "["
            + THINKING
            + """,
		{"type":"text","text":"First"},
		{"type":"tool_use","id":"a","name":"Bash","input":{"number":9007199254740993},"caller":{"type":"direct"}},
		"""
            + REDACTED
            + """,
		{"type":"text","text":"Second"},
		"""
            + OMITTED
            + """,
		{"type":"tool_use","id":"b","name":"Bash","input":{}}
	]""",
        )
    )
    assert decoded.ID == "msg-1"
    assert decoded.Stop == StopReason.COMPLETE
    assert decoded.Failure is None
    output = decoded.Output or []
    assert [item.Type for item in output] == [
        ItemType.PROVIDER,
        ItemType.MESSAGE,
        ItemType.TOOL_CALL,
        ItemType.PROVIDER,
        ItemType.MESSAGE,
        ItemType.PROVIDER,
        ItemType.TOOL_CALL,
    ]
    for item in output:
        item.validate()
    for index, want in {0: THINKING, 3: REDACTED, 5: OMITTED}.items():
        reasoning = output[index].Data
        assert isinstance(reasoning, ProviderItem)
        assert bytes(reasoning.Raw) == want.encode()
        assert reasoning.Type == ("redacted_thinking" if index == 3 else "thinking")
        if index == 0:
            assert reasoning.Display == ProviderDisplay(
                Kind=ProviderDisplayKind.REASONING, Text="Considering tools"
            )
        else:
            assert reasoning.Display is None, "invented reasoning display"
    assert output[1].Data == Message(Role=Role.ASSISTANT, Text="First")
    assert output[4].Data == Message(Role=Role.ASSISTANT, Text="Second")
    assert output[2].ProviderID == "a"
    assert output[2].Data == ToolCall(
        CallID="a", Name="Bash", Arguments='{"number":9007199254740993}'
    )

    body, _ = encode_request(
        Request(
            Model=Model(ID="claude-test"),
            Input=[
                message(Role.USER, "Start"),
                *output,
                tool_result("b", "B", False),
                tool_result("a", "A", False),
            ],
        )
    )
    messages = decode(bytes(decode(body, dict[str, Raw])["messages"]), list[dict[str, Raw]])
    content = decode(bytes(messages[1]["content"]), list[Raw])
    for index, want in {0: THINKING, 3: REDACTED, 5: OMITTED}.items():
        assert parse_any(bytes(content[index])) == parse_any(want.encode())


@pytest.mark.parametrize(
    ("stop", "want"),
    [
        ("end_turn", StopReason.COMPLETE),
        ("tool_use", StopReason.COMPLETE),
        ("stop_sequence", StopReason.COMPLETE),
        ("max_tokens", StopReason.MAX_OUTPUT_TOKENS),
        ("model_context_window_exceeded", StopReason.MAX_OUTPUT_TOKENS),
        ("refusal", StopReason.REFUSED),
    ],
)
def test_response_stop_reasons(stop: str, want: StopReason) -> None:
    decoded = decode_response(response_body(stop, '[{"type":"text","text":"Answer"}]'))
    assert decoded.Stop == want
    assert len(decoded.Output or []) == 1
    assert (decoded.Failure is not None) == (want == StopReason.REFUSED)


@pytest.mark.parametrize("stop", ["pause_turn", "unknown", ""])
def test_response_rejects_unsupported_stop_reasons(stop: str) -> None:
    with pytest.raises(ValueError, match=r"unsupported stop reason"):
        decode_response(response_body(stop, "[]"))


@pytest.mark.parametrize(
    ("details", "want"),
    [
        ("", Failure(Code="refusal", Message="response refused")),
        (',"stop_details":null', Failure(Code="refusal", Message="response refused")),
        (
            ',"stop_details":{"type":"refusal","category":"policy_violation",'
            '"explanation":"Provider explanation"}',
            Failure(Code="policy_violation", Message="Provider explanation"),
        ),
        (
            ',"stop_details":{"type":"refusal","category":"policy_violation","explanation":null}',
            Failure(Code="policy_violation", Message="response refused"),
        ),
        (
            ',"stop_details":{"type":"refusal","category":null,'
            '"explanation":"Provider explanation"}',
            Failure(Code="refusal", Message="Provider explanation"),
        ),
        (
            ',"stop_details":{"type":"refusal","category":null,"explanation":null}',
            Failure(Code="refusal", Message="response refused"),
        ),
        (
            ',"stop_details":{"type":"refusal","category":"","explanation":""}',
            Failure(Code="refusal", Message="response refused"),
        ),
    ],
    ids=[
        "absent",
        "null",
        "full",
        "category only",
        "explanation only",
        "null fields",
        "empty fields",
    ],
)
def test_response_refusal_details(details: str, want: Failure) -> None:
    body = response_body("refusal", "[]")
    decoded = decode_response(body[:-1] + details.encode() + b"}")
    assert decoded.Stop == StopReason.REFUSED
    assert decoded.Output == []
    assert decoded.Failure == want


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded"])
@pytest.mark.parametrize("tool_input", ["{}", '{"command":"ls"}'])
def test_response_preserves_final_calls_at_token_limit(stop: str, tool_input: str) -> None:
    decoded = decode_response(
        response_body(
            stop,
            """[
				{"type":"tool_use","id":"complete","name":"Bash","input":{"command":"pwd"}},
				{"type":"text","text":"Another command"},
				{"type":"tool_use","id":"last","name":"Bash","input":"""
            + tool_input
            + """}
			]""",
        )
    )
    output = decoded.Output or []
    assert decoded.Stop == StopReason.MAX_OUTPUT_TOKENS
    assert len(output) == 3
    assert isinstance(output[0].Data, ToolCall)
    assert output[0].Data.CallID == "complete"
    assert output[1].Type == ItemType.MESSAGE
    assert isinstance(output[2].Data, ToolCall)
    assert output[2].Data.CallID == "last"


@pytest.mark.parametrize(
    "tool_input",
    [
        "{}",
        """{
  "z": [9007199254740993, 1e1000, -0, 1.2300],
  "a": {"escaped": "\\u0061\\/", "value": null}
}""",
    ],
)
def test_response_preserves_tool_arguments(tool_input: str) -> None:
    decoded = decode_response(
        response_body(
            "tool_use", '[{"type":"tool_use","id":"a","name":"Bash","input":' + tool_input + "}]"
        )
    )
    call = (decoded.Output or [])[0].Data
    assert isinstance(call, ToolCall)
    assert call.Arguments == tool_input


@pytest.mark.parametrize(
    "content",
    [
        '{"type":"server_tool_use","id":"server-1","name":"web_search","input":{}}',
        '{"type":"web_search_tool_result","tool_use_id":"server-1","content":[]}',
        '{"type":"text","text":"Cited answer","citations":[{"type":"web_search_result_location",'
        '"encrypted_index":"reference"}]}',
        '{"type":"tool_use","id":"a","name":"click","toolset_name":"browser","input":{}}',
        '{"type":"tool_use","id":"a","name":"Bash","input":{},'
        '"caller":{"type":"code_execution_20260120","tool_id":"server-1"}}',
        '{"type":"unknown"}',
        '{"type":"thinking","thinking":"Not signed"}',
        '{"type":"redacted_thinking"}',
        '{"type":"tool_use","id":"a","name":"Bash"}',
        '{"type":"tool_use","id":"a","name":"Bash","input":[]}',
        '{"type":"tool_use","id":"a","name":"Bash","input":null}',
        '{"type":"tool_use","id":"a","name":"Bash","input":"{}"}',
        '{"type":"tool_use","id":"a","name":"Bash","input":42}',
        '{"type":"tool_use","id":"a","name":"Bash","input":true}',
        '{"type":"tool_use","id":"a","name":"Bash","input":{"nested":{"x":1,"x":2}}}',
        '{"type":"tool_use","id":"a","name":"Bash","input":{"x":}}',
        '{"type":"tool_use","name":"Bash","input":{}}',
    ],
    ids=[
        "server tool",
        "server result",
        "citations",
        "toolset",
        "programmatic call",
        "unknown block",
        "missing signature",
        "missing ciphertext",
        "missing input",
        "nonobject input",
        "null input",
        "string input",
        "number input",
        "boolean input",
        "duplicate argument",
        "malformed input",
        "missing call id",
    ],
)
def test_response_rejects_unsupported_blocks(content: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        decode_response(response_body("end_turn", "[" + content + "]"))


def test_response_rejects_duplicate_calls() -> None:
    call = '{"type":"tool_use","id":"a","name":"Bash","input":{}}'
    with pytest.raises(ValueError, match=r'^output block 1: duplicate tool call "a"$'):
        decode_response(response_body("tool_use", "[" + call + "," + call + "]"))


def test_response_usage() -> None:
    usage = (
        '{"input_tokens":7,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,'
        '"output_tokens":11,"output_tokens_details":{"thinking_tokens":5},"extra":{"cost":0.25}}'
    )
    decoded = decode_response(
        (
            '{"id":"msg-1","type":"message","role":"assistant","stop_reason":"end_turn",'
            '"content":[],"usage":' + usage + "}"
        ).encode()
    )
    assert decoded.Usage == Usage(
        InputTokens=57,
        CachedInputTokens=30,
        CacheWriteInputTokens=20,
        OutputTokens=11,
        ReasoningTokens=5,
        Raw=Raw(usage.encode()),
    )
    without_cache = response_usage(WireUsage(input_tokens=7, output_tokens=11))
    assert without_cache == Usage(InputTokens=7, OutputTokens=11)


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "{}",
        '{"type":"error","error":{"message":"Failure"}}',
        '{"type":"message","role":"assistant","stop_reason":null}',
        '{"type":"message","role":"assistant","stop_reason":"end_turn","usage":[]}',
        '{"type":"message","role":"assistant","stop_reason":"end_turn",'
        '"usage":{"input_tokens":"invalid"}}',
        '{"type":"message","role":"assistant","stop_reason":"end_turn","usage":{},"usage":null}',
    ],
)
def test_response_rejects_invalid_envelopes(body: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        decode_response(body.encode())


@pytest.mark.parametrize("usage", ["", "null", "{}"])
def test_response_optional_usage(usage: str) -> None:
    body = '{"type":"message","role":"assistant","stop_reason":"end_turn","content":[]'
    if usage:
        body += ',"usage":' + usage
    decoded = decode_response((body + "}").encode())
    assert decoded.Usage == Usage(Raw=Raw(usage.encode()))


def test_response_error_messages() -> None:
    with pytest.raises(
        ValueError, match=r'^expected an assistant message, got type "" and role ""$'
    ):
        decode_response(b"{}")
    with pytest.raises(ValueError, match=r"^response must have a stop reason$"):
        decode_response(b'{"type":"message","role":"assistant"}')
    with pytest.raises(ValueError, match=r'^output block 0: unsupported toolset "browser"$'):
        decode_response(
            response_body(
                "end_turn",
                '[{"type":"tool_use","id":"a","name":"click","toolset_name":"browser","input":{}}]',
            )
        )
