import json
from typing import cast

import msgspec
import pytest
from unreal_agent.contextbuilder import new_builder
from unreal_agent.gojson import Raw, decode, encode, validate
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Reasoning,
    ReasoningEffort,
    Request,
    Response,
    Role,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
)
from unreal_agent.llm.responsesapi import decode_response, request_body
from unreal_agent.llm.responsesapi.request import request_input_item
from unreal_agent.operation import Spec
from unreal_agent.sessionstore import Item as StoreItem
from unreal_agent.sessionstore import ItemKind, ModelResponse
from unreal_agent.tool import bash

from tests.llm.responsesapi._helpers import detailed_request, valid_request

INVALID_BYTE = b"\xff".decode("utf-8", "surrogateescape")


@pytest.mark.parametrize(
    ("arguments", "valid"),
    [
        pytest.param("{}", True, id="empty object"),
        pytest.param(' \n{"command": "echo \\u0061"}\t', True, id="whitespace and escapes"),
        pytest.param('{"command":42}', True, id="schema error"),
        pytest.param('{"array":[null,true,1e1000],"object":{}}', True, id="nested values"),
        pytest.param('{"command":"apt-get install -y r-base', False, id="truncated string"),
        pytest.param('{"command":"pwd"', False, id="truncated object"),
        pytest.param("{} []", False, id="trailing data"),
        pytest.param('{"command":"pwd","command":"ls"}', False, id="duplicate names"),
        pytest.param("null", False, id="null"),
        pytest.param(" \nnull\t", False, id="null with whitespace"),
        pytest.param("[]", False, id="array"),
        pytest.param('"pwd"', False, id="string"),
        pytest.param("42", False, id="number"),
        pytest.param("true", False, id="boolean"),
        pytest.param("", False, id="empty"),
        pytest.param(" \n\t", False, id="whitespace"),
    ],
)
def test_request_input_item_replays_tool_call_arguments(arguments: str, valid: bool) -> None:
    call = ToolCall(CallID="call-1", Name="Bash", Arguments=arguments)
    source = Item(ProviderID="function-1", Type=ItemType.TOOL_CALL, Data=call)
    wire = json.loads(bytes(request_input_item(source)))
    assert wire["type"] == "function_call"
    assert (wire["id"], wire["call_id"], wire["name"]) == ("function-1", "call-1", "Bash")
    replayed = validate(wire["arguments"])
    assert isinstance(replayed, dict)
    if valid:
        assert wire["arguments"] == arguments
    else:
        assert replayed == {"invalid_arguments": arguments}
    assert source.Data == ToolCall(CallID="call-1", Name="Bash", Arguments=arguments)


def test_request_body_reports_tool_argument_encoding_error() -> None:
    request = Request(
        Input=[
            Item(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name="Bash", Arguments=INVALID_BYTE),
            )
        ]
    )
    with pytest.raises(ValueError, match='input item 0: encode tool call "call-1" arguments:'):
        request_body(request, "")


def test_request_body_replaces_invalid_utf8_in_model_and_plain_input_text() -> None:
    def escaped(data: bytes) -> str:
        return data.decode("utf-8", "surrogateescape")

    request = Request(
        Model=Model(ID=escaped(b"a\xffb\xe2\x82")),
        Input=[
            Item(
                Type=ItemType.MESSAGE,
                Data=Message(
                    Role=Role.USER, Text=escaped(b"x\xe2\x82\xacy\xf0\x9f\x98<\xef\xbf\xbd")
                ),
            )
        ],
    )
    replacement = "\ufffd".encode()
    assert request_body(request, "") == (
        b'{"include":["reasoning.encrypted_content"],"input":[{"content":"x\xe2\x82\xacy'
        + replacement * 3
        + b"<"
        + replacement
        + b'","role":"user"}],"model":"a'
        + replacement
        + b"b"
        + replacement * 2
        + b'","store":false,"stream":true}'
    )


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(Message(Role=Role.USER, Text="ok", Phase=INVALID_BYTE), id="phase"),
        pytest.param(Message(Role=cast(Role, INVALID_BYTE), Text="ok"), id="role"),
    ],
)
def test_request_body_rejects_invalid_utf8_plain_input_fields(message: Message) -> None:
    with pytest.raises(ValueError, match=r"^input item 0: .*invalid UTF-8"):
        request_body(Request(Input=[Item(Type=ItemType.MESSAGE, Data=message)]), "")


def test_request_body_rejects_invalid_utf8_tool_result() -> None:
    output = ToolResultOutput(Kind=ToolResultKind.TEXT, Value=f"bad{INVALID_BYTE}byte")
    request = Request(
        Input=[Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID="call-1", Output=[output]))]
    )
    with pytest.raises(ValueError, match=r"^input item 0: .*invalid UTF-8"):
        request_body(request, "")


@pytest.mark.parametrize(
    ("output", "want"),
    [
        pytest.param(
            [ToolResultOutput(Kind=ToolResultKind.TEXT, Value="done")],
            '[{"type":"input_text","text":"done"}]',
            id="text",
        ),
        pytest.param(
            [ToolResultOutput(Kind=ToolResultKind.TEXT)],
            '[{"type":"input_text","text":""}]',
            id="empty text",
        ),
        pytest.param([], "[]", id="no outputs"),
        pytest.param(
            [ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="https://example.com/image.png")],
            '[{"type":"input_image","image_url":"https://example.com/image.png"}]',
            id="image URL",
        ),
        pytest.param(
            [ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,aGVsbG8=")],
            '[{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="}]',
            id="image data URL",
        ),
        pytest.param(
            [
                ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Original dimensions: 4000x3000"),
                ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,Zmlyc3Q="),
                ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Resized dimensions: 2000x1500"),
                ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="https://example.com/second.png"),
            ],
            '[{"type":"input_text","text":"Original dimensions: 4000x3000"},'
            '{"type":"input_image","image_url":"data:image/png;base64,Zmlyc3Q="},'
            '{"type":"input_text","text":"Resized dimensions: 2000x1500"},'
            '{"type":"input_image","image_url":"https://example.com/second.png"}]',
            id="mixed content in order",
        ),
    ],
)
def test_request_body_encodes_tool_result_outputs(
    output: list[ToolResultOutput], want: str
) -> None:
    body = request_body(
        Request(
            Model=Model(ID="gpt-test"),
            Input=[
                Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID="call-1", Output=output))
            ],
        ),
        "",
    )
    assert json.loads(body)["input"] == [
        {"type": "function_call_output", "call_id": "call-1", "output": json.loads(want)}
    ]


@pytest.mark.parametrize(
    ("item", "want"),
    [
        pytest.param(
            Item(Type=ItemType.MESSAGE, Data=None),
            "input item 0: message item data must be llm.Message, got <nil>",
            id="message",
        ),
        pytest.param(
            Item(Type=ItemType.TOOL_CALL, Data=None),
            "input item 0: tool_call item data must be llm.ToolCall, got <nil>",
            id="tool call",
        ),
        pytest.param(
            Item(Type=ItemType.TOOL_RESULT, Data=None),
            "input item 0: tool_result item data must be llm.ToolResult, got <nil>",
            id="tool result",
        ),
        pytest.param(
            Item(
                Type=ItemType.TOOL_RESULT,
                Data=ToolResult(Output=[ToolResultOutput(Value="done")]),
            ),
            'input item 0: unsupported tool result kind ""',
            id="unset tool result kind",
        ),
        pytest.param(
            Item(
                Type=ItemType.TOOL_RESULT,
                Data=ToolResult(Output=[ToolResultOutput(Kind=cast(ToolResultKind, "unknown"))]),
            ),
            'input item 0: unsupported tool result kind "unknown"',
            id="unsupported tool result kind",
        ),
        pytest.param(
            Item(Type=ItemType.REASONING, Data=None),
            "input item 0: reasoning item data must be llm.Reasoning, got <nil>",
            id="reasoning",
        ),
        pytest.param(
            Item(Type=ItemType.REASONING, Data=Reasoning(Summary=["thought"])),
            "input item 0: reasoning item must carry the provider item in Raw",
            id="reasoning without raw",
        ),
        pytest.param(
            Item(Type=ItemType.MESSAGE, Data=ToolCall()),
            "input item 0: message item data must be llm.Message, got llm.ToolCall",
            id="mismatched data",
        ),
    ],
)
def test_request_body_rejects_invalid_item_payloads(item: Item, want: str) -> None:
    with pytest.raises(ValueError, match=r"^input item 0: ") as caught:
        request_body(Request(Input=[item]), "")
    assert str(caught.value) == want


@pytest.mark.parametrize("role", [Role.USER, Role.SYSTEM])
def test_request_input_item_preserves_identified_input_message_roles(role: Role) -> None:
    item = Item(
        ProviderID="message-1", Type=ItemType.MESSAGE, Data=Message(Role=role, Text="hello")
    )
    got = json.loads(bytes(request_input_item(item)))
    assert "id" not in got
    assert "status" not in got
    assert (got["role"], got["type"]) == (str(role), "message")
    assert got["content"] == [{"text": "hello", "type": "input_text"}]


def test_request_input_item_replays_raw_reasoning_verbatim() -> None:
    raw = (
        b'{"id":"reasoning-1","type":"reasoning","status":"completed","summary":[],'
        b'"content":[{"type":"reasoning_text","text":"verbatim"}],"encrypted_content":"opaque"}'
    )
    item = request_input_item(
        Item(
            ProviderID="ignored",
            Type=ItemType.REASONING,
            Data=Reasoning(Summary=["stale summary"], Raw=Raw(raw)),
        )
    )
    assert bytes(item) == raw


def test_request_input_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="unsupported input item type") as caught:
        request_input_item(Item(Type=cast(ItemType, "image"), Data=None))
    assert str(caught.value) == 'unsupported input item type "image"'


def test_request_body_omits_unset_max_output_tokens() -> None:
    assert "max_output_tokens" not in json.loads(request_body(valid_request(), ""))


def test_request_body_encodes_reasoning_effort() -> None:
    body = request_body(
        Request(Model=Model(ID="gpt-test", ReasoningEffort=ReasoningEffort.HIGH)), ""
    )
    assert json.loads(body)["reasoning"] == {"effort": "high", "summary": "auto"}


def test_request_body_rejects_unsupported_reasoning_effort() -> None:
    request = Request(Model=Model(ID="gpt-test", ReasoningEffort=cast(ReasoningEffort, "maximum")))
    with pytest.raises(ValueError, match="unsupported reasoning effort") as caught:
        request_body(request, "")
    assert str(caught.value) == 'unsupported reasoning effort "maximum"'


def test_request_body_encodes_hosted_web_search() -> None:
    body = request_body(
        Request(
            Model=Model(ID="gpt-test"),
            Input=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="latest news"))],
            Tools=[Tool(Type=ToolType.HOSTED, Name="web_search")],
        ),
        "",
    )
    assert json.loads(body)["tools"] == [{"type": "web_search"}]


def test_request_body_rejects_unsupported_hosted_tool() -> None:
    with pytest.raises(ValueError, match="unsupported hosted tool name") as caught:
        request_body(Request(Tools=[Tool(Type=ToolType.HOSTED, Name="unknown")]), "")
    assert str(caught.value) == 'tool 0: unsupported hosted tool name "unknown"'


def test_request_body_rejects_unsupported_tool_type() -> None:
    with pytest.raises(ValueError, match="unsupported tool type") as caught:
        request_body(Request(Tools=[Tool(Name="weather")]), "")
    assert str(caught.value) == 'tool 0: unsupported tool type ""'


def test_request_body_is_byte_stable_across_encodings() -> None:
    request = valid_request()
    request.Tools = [
        Tool(
            Type=ToolType.FUNCTION,
            Name="Bash",
            Description="Execute a shell command.",
            Parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command."},
                    "max_output_chars": {"type": "integer", "description": "Inline budget."},
                    "timeout_seconds": {"type": "integer", "description": "Deadline."},
                },
                "required": ["command"],
            },
        )
    ]
    first = request_body(request, "session-1")
    for _ in range(64):
        assert request_body(request, "session-1") == first


def test_request_body_encodes_max_reasoning_effort() -> None:
    body = request_body(
        Request(Model=Model(ID="gpt-test", ReasoningEffort=ReasoningEffort.MAX)), ""
    )
    assert json.loads(body)["reasoning"]["effort"] == "max"


def test_request_body_merges_extensions() -> None:
    body = request_body(
        valid_request(),
        "",
        {"cache_control": b'{"type":"ephemeral"}', "provider": b'{"only":["anthropic"]}'},
    )
    request = json.loads(body)
    assert request["model"] == "gpt-test"
    assert request["stream"] is True
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["provider"]["only"] == ["anthropic"]


@pytest.mark.parametrize(
    "extensions",
    [
        pytest.param({"model": b'"other"'}, id="standard field"),
        pytest.param({"provider": b'{"only":'}, id="invalid JSON"),
    ],
)
def test_request_body_rejects_extensions_overriding_standard_fields(
    extensions: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError, match="request extension"):
        request_body(valid_request(), "", extensions)


# Golden bodies produced by the Go implementation.


def test_request_body_matches_go_for_detailed_request() -> None:
    assert request_body(detailed_request(), "abc") == (
        b'{"include":["reasoning.encrypted_content"],"input":[{"content":"Be concise.",'
        b'"role":"system"},{"content":"weather?","role":"user"},{"content":"Use tools.",'
        b'"role":"system"},{"content":[{"annotations":[],"logprobs":[],"text":"Checking.",'
        b'"type":"output_text"}],"id":"previous-message","phase":"commentary","role":"assistant",'
        b'"status":"completed","type":"message"},{"id":"previous-reasoning","type":"reasoning",'
        b'"status":"completed","summary":[{"type":"summary_text","text":"Checked the request."}],'
        b'"content":[{"type":"reasoning_text","text":"verbatim"}],'
        b'"encrypted_content":"previous-opaque"},{"arguments":"{\\"city\\":\\"London\\"}",'
        b'"call_id":"call-1","id":"previous-tool-call","name":"weather","type":"function_call"},'
        b'{"call_id":"call-1","output":[{"text":"sunny","type":"input_text"}],'
        b'"type":"function_call_output"}],"max_output_tokens":128,"model":"gpt-test",'
        b'"prompt_cache_key":"abc","store":false,"stream":true,"tools":[{"description":'
        b'"Get weather","name":"weather","parameters":{"additionalProperties":false,'
        b'"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},'
        b'"strict":false,"type":"function"}]}'
    )


def test_request_body_matches_go_for_empty_request() -> None:
    assert request_body(Request(), "") == (
        b'{"include":["reasoning.encrypted_content"],"store":false,"stream":true}'
    )


def test_request_body_matches_go_for_edge_values() -> None:
    request = Request(
        Model=Model(ID="m", ReasoningEffort=ReasoningEffort.HIGH),
        Tools=[
            Tool(Type=ToolType.FUNCTION, Name="x"),
            Tool(Type=ToolType.HOSTED, Name="web_search"),
        ],
        Input=[
            Item(Type=ItemType.MESSAGE, Data=Message()),
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="<&>\u2028", Phase="p")),
            Item(ProviderID="m", Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT)),
            Item(
                ProviderID="m",
                Type=ItemType.MESSAGE,
                Data=Message(Role=cast(Role, "developer"), Text="d", Phase="x"),
            ),
            Item(
                Type=ItemType.REASONING,
                Data=Reasoning(Raw=Raw(b'{ "z" : 1 , "a" : "<\\u003c" }')),
            ),
            Item(Type=ItemType.TOOL_CALL, Data=ToolCall(Arguments="nope")),
            Item(Type=ItemType.TOOL_RESULT, Data=ToolResult()),
        ],
    )
    body = request_body(request, "", {"b_ext": b' { "y" : [ 1 , 2 ] } '})
    assert (
        body
        == (
            '{"b_ext":{"y":[1,2]},"include":["reasoning.encrypted_content"],"input":['
            '{"content":"","role":""},{"content":"<&>\u2028","phase":"p","role":"user"},'
            '{"content":[],"id":"m","role":"assistant","status":"completed","type":"message"},'
            '{"content":[{"text":"d","type":"input_text"}],"role":"developer","type":"message"},'
            '{"z":1,"a":"<<"},{"arguments":"{\\"invalid_arguments\\":\\"nope\\"}","call_id":"",'
            '"name":"","type":"function_call"},{"output":[],"type":"function_call_output"}],'
            '"model":"m","reasoning":{"effort":"high","summary":"auto"},"store":false,'
            '"stream":true,"tools":[{"name":"x","parameters":{},"strict":false,"type":"function"},'
            '{"type":"web_search"}]}'
        ).encode()
    )


class _WireContent(msgspec.Struct):
    type: str = ""
    text: str = ""


class _WireItem(msgspec.Struct):
    type: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    output: list[_WireContent] = []


class _WireRequest(msgspec.Struct):
    input: list[_WireItem] = []


class _ReplayToolContext:
    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, spec: Spec, /) -> str:
        self.submissions += 1
        return "unexpected-operation"


def _text(value: str) -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]


def test_request_body_replays_rejected_tool_call_from_history() -> None:
    response = decode_response(
        b'{"id":"response-1","status":"completed","output":[{"type":"function_call",'
        b'"id":"function-1","call_id":"call-1","name":"Bash",'
        b'"arguments":"{\\"command\\":\\"apt-get install -y r-base"}]}'
    )
    history = encode(
        StoreItem(
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(TurnID="turn-1", Response=response),
        )
    )
    restored = decode(history, StoreItem)
    assert isinstance(restored.Data, ModelResponse)
    response = restored.Data.Response
    call = response.Output[0].Data
    assert isinstance(call, ToolCall)
    translator = bash.new(bash.Config())
    tool_context = _ReplayToolContext()
    status = translator.translate(tool_context, call)
    assert "decode Bash arguments:" in status.Error
    assert tool_context.submissions == 0
    result = translator.translate_result(call.CallID, status, [])

    builder = new_builder()
    builder.add_model_response(response)
    builder.add_tool_result(result.CallID, result.Output, False)
    builder.commit()
    valid_call = ToolCall(CallID="call-2", Name="Search", Arguments=' {"query":"weather"} ')
    builder.add_model_response(Response(Output=[Item(Type=ItemType.TOOL_CALL, Data=valid_call)]))
    builder.add_tool_result(valid_call.CallID, _text("found"), False)
    built = builder.build()

    wire = msgspec.json.decode(request_body(built.request, ""), type=_WireRequest).input
    assert len(wire) == 5, "want system message and two call/result pairs"
    assert json.loads(wire[1].arguments) == {"invalid_arguments": call.Arguments}
    assert wire[1:] == [
        _WireItem(
            type="function_call",
            call_id=call.CallID,
            name=call.Name,
            arguments=wire[1].arguments,
        ),
        _WireItem(
            type="function_call_output",
            call_id=call.CallID,
            output=[_WireContent(type="input_text", text=result.Output[0].Value)],
        ),
        _WireItem(
            type="function_call",
            call_id=valid_call.CallID,
            name=valid_call.Name,
            arguments=valid_call.Arguments,
        ),
        _WireItem(
            type="function_call_output",
            call_id=valid_call.CallID,
            output=[_WireContent(type="input_text", text="found")],
        ),
    ]
    assert encode(restored) == history, "replay changed persisted history"


_REASONING = (
    b'{"id":"reasoning","type":"reasoning","summary":[{"type":"summary_text",'
    b'"text":"Keep working."}],"encrypted_content":"opaque"}'
)


class _RawInput(msgspec.Struct):
    input: list[msgspec.Raw] = []


@pytest.mark.parametrize(
    ("output", "want"),
    [
        pytest.param(
            b'[{"id":"message","type":"message","role":"assistant","status":"completed",'
            b'"phase":"final_answer","content":[{"type":"output_text","text":"Waiting.",'
            b'"annotations":[],"logprobs":[]}]}]',
            b'[{"id":"message","type":"message","role":"assistant","status":"completed",'
            b'"phase":"final_answer","content":[{"type":"output_text","text":"Waiting.",'
            b'"annotations":[],"logprobs":[]}]}]',
            id="assistant message",
        ),
        pytest.param(
            b"["
            + _REASONING
            + b',{"id":"item-C","type":"function_call","call_id":"C","name":"test",'
            b'"arguments":"{}","status":"completed"}]',
            b"["
            + _REASONING
            + b',{"id":"item-C","type":"function_call","call_id":"C","name":"test",'
            b'"arguments":"{}"}]',
            id="reasoning and tool call",
        ),
        pytest.param(b"[]", b"[]", id="empty output"),
    ],
)
def test_submitted_prefix_and_late_result_order_on_wire(output: bytes, want: bytes) -> None:
    builder = new_builder()
    builder.set_model(Model(ID="test"))
    builder.add_model_response(
        Response(
            Output=[
                Item(
                    Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="A", Name="test", Arguments="{}")
                ),
                Item(
                    Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="B", Name="test", Arguments="{}")
                ),
            ]
        )
    )
    builder.add_tool_result("A", _text(""), True)
    builder.add_tool_result("B", _text("done B"), False)
    before_body = request_body(builder.build().request, "session")
    builder.commit()
    builder.add_tool_result("A", _text("done A"), False)
    builder.add_model_response(
        decode_response(
            b'{"id":"response","object":"response","status":"completed","output":' + output + b"}"
        )
    )
    after_body = request_body(builder.build().request, "session")

    sent = [bytes(item) for item in msgspec.json.decode(before_body, type=_RawInput).input]
    following = [bytes(item) for item in msgspec.json.decode(after_body, type=_RawInput).input]
    assert following[: len(sent)] == sent, "serialized submitted prefix changed"
    want_tail = [
        *json.loads(want),
        {
            "type": "function_call_output",
            "call_id": "A",
            "output": [{"type": "input_text", "text": "done A"}],
        },
    ]
    assert [json.loads(item) for item in following[len(sent) :]] == want_tail
