import json
import re
from typing import Any, cast

import pytest
from unreal_agent.gojson import Raw, decode
from unreal_agent.llm import (
    Item,
    ItemType,
    Model,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    ReasoningEffort,
    Request,
    Role,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
)
from unreal_agent.llm.messagesapi import CacheTTL, request_body
from unreal_agent.llm.messagesapi._wire import parse_any
from unreal_agent.llm.messagesapi.request import request_input, request_tool_output

from tests.llm.messagesapi._helpers import (
    encode_request,
    message,
    tool_call,
    tool_result,
    valid_request,
)


def _content(body: bytes, index: int) -> list[bytes]:
    messages = decode(bytes(decode(body, dict[str, Raw])["messages"]), list[dict[str, Raw]])
    return [bytes(block) for block in decode(bytes(messages[index]["content"]), list[Raw])]


def test_request_settings_and_tools() -> None:
    request = Request(
        Model=Model(ID="claude-test", MaxOutputTokens=16384, ReasoningEffort=ReasoningEffort.HIGH),
        Input=[
            message(Role.SYSTEM, "System instructions"),
            message(Role.USER, "First"),
            message(Role.USER, "Second"),
        ],
        Tools=[
            Tool(
                Type=ToolType.FUNCTION,
                Name="Bash",
                Description="Run a command",
                Parameters={
                    "type": "object",
                    "required": ["command"],
                    "additionalProperties": False,
                    "properties": {"command": {"type": "string"}},
                },
            )
        ],
    )
    _, wire = encode_request(request)
    assert wire["model"] == "claude-test"
    assert wire["max_tokens"] == 16384
    assert wire["stream"] is True
    assert wire["cache_control"]["type"] == "ephemeral"
    assert wire["thinking"]["type"] == "adaptive"
    assert wire["output_config"]["effort"] == "high"
    assert [block["text"] for block in wire["system"]] == ["System instructions"]
    assert len(wire["messages"]) == 1
    assert wire["messages"][0]["role"] == "user"
    assert len(wire["messages"][0]["content"]) == 2
    [tool] = wire["tools"]
    assert (tool["type"], tool["name"], tool["description"], tool["strict"]) == (
        "custom",
        "Bash",
        "Run a command",
        False,
    )
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["command"]


def test_request_body_bytes() -> None:
    request = Request(
        Model=Model(ID="claude-test", ReasoningEffort=ReasoningEffort.LOW),
        Input=[
            message(Role.SYSTEM, "System"),
            message(Role.USER, "Start"),
            tool_call("a"),
            tool_result("a", "done", False),
        ],
        Tools=[Tool(Type=ToolType.FUNCTION, Name="Bash", Parameters={"type": "object"})],
    )
    assert request_body(request) == (
        b'{"cache_control":{"type":"ephemeral"},"max_tokens":128000,"messages":['
        b'{"content":[{"text":"Start","type":"text"}],"role":"user"},'
        b'{"content":[{"id":"a","input":{},"name":"Bash","type":"tool_use"}],"role":"assistant"},'
        b'{"content":[{"content":[{"text":"done","type":"text"}],"tool_use_id":"a",'
        b'"type":"tool_result"}],"role":"user"}],"model":"claude-test",'
        b'"output_config":{"effort":"low"},"stream":true,'
        b'"system":[{"text":"System","type":"text"}],"thinking":{"type":"adaptive"},'
        b'"tools":[{"input_schema":{"type":"object"},"name":"Bash","strict":false,'
        b'"type":"custom"}]}'
    )


@pytest.mark.parametrize("effort", ["", "low", "medium", "high", "xhigh", "max"])
def test_request_defaults_and_efforts(effort: str) -> None:
    body, wire = encode_request(
        Request(
            Model=Model(ID="claude-test", ReasoningEffort=ReasoningEffort(effort)),
            Input=[message(Role.USER, "Hello")],
        )
    )
    assert wire["max_tokens"] == 128_000
    if effort == "":
        assert "thinking" not in wire, body
        assert "output_config" not in wire, body
    else:
        assert wire["thinking"]["type"] == "adaptive", body
        assert wire["output_config"]["effort"] == effort, body
    assert "tools" not in wire


@pytest.mark.parametrize(
    ("ttl", "fallback"),
    [
        pytest.param(CacheTTL.NONE, False, id="unset"),
        pytest.param(CacheTTL.FIVE_MINUTES, False, id="5m"),
        pytest.param(CacheTTL.ONE_HOUR, False, id="1h"),
        pytest.param(CacheTTL.FIVE_MINUTES, True, id="fallback"),
    ],
)
def test_request_cache_ttl(ttl: CacheTTL, fallback: bool) -> None:
    wire = json.loads(request_body(valid_request(), fallback=fallback, cache_ttl=ttl))
    want = {"type": "ephemeral"}
    if ttl:
        want["ttl"] = str(ttl)
    assert wire["cache_control"] == want
    assert (wire.get("fallbacks") == "default") == fallback


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("running", [False, True])
def test_request_orders_primary_results_by_calls(order: tuple[str, str], running: bool) -> None:
    request = Request(
        Model=Model(ID="claude-test"),
        Input=[
            message(Role.USER, "Start"),
            tool_call("a"),
            message(Role.ASSISTANT, "Also checking"),
            tool_call("b"),
            message(Role.USER, "Steering one"),
            tool_result(order[0], order[0] + " result", running),
            message(Role.USER, "Steering two"),
            tool_result(order[1], order[1] + " result", running),
        ],
    )
    _, wire = encode_request(request)
    assistant = wire["messages"][1]["content"]
    assert len(assistant) == 3
    assert (assistant[0]["id"], assistant[1]["text"], assistant[2]["id"]) == (
        "a",
        "Also checking",
        "b",
    )
    user = wire["messages"][2]["content"]
    assert len(user) == 4
    assert (user[0]["tool_use_id"], user[1]["tool_use_id"]) == ("a", "b")
    assert (user[2]["text"], user[3]["text"]) == ("Steering one", "Steering two")
    assert user[0]["content"][0]["text"] == "a result"
    assert user[1]["content"][0]["text"] == "b result"


def test_request_repeated_results_become_user_content() -> None:
    _, wire = encode_request(
        Request(
            Model=Model(ID="claude-test"),
            Input=[
                message(Role.USER, "Start"),
                tool_call("a"),
                tool_result("a", "still running", True),
                message(Role.USER, "Steering"),
                tool_result("a", "done", False),
                message(Role.ASSISTANT, "Acknowledged"),
                tool_result("a", "another update", True),
            ],
        )
    )
    user = wire["messages"][2]["content"]
    assert len(user) == 4
    assert user[0]["type"] == "tool_result"
    assert user[0]["content"][0]["text"] == "still running"
    assert user[1]["text"] == "Steering"
    assert user[2]["text"] == '"Bash" (call "a", completed)\n'
    assert user[3]["text"] == "done"
    user = wire["messages"][4]["content"]
    assert len(user) == 2
    assert user[0]["type"] == "text"
    assert 'call "a", running' in user[0]["text"]
    assert user[1]["text"] == "another update"


def test_request_tool_output_preserves_images_and_empty_results() -> None:
    output = [
        ToolResultOutput(Kind=ToolResultKind.TEXT, Value="First image"),
        ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,aGVsbG8="),
        ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Second image"),
        ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="https://example.com/image.jpg"),
    ]
    result = Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID="a", Output=output))
    _, wire = encode_request(
        Request(
            Model=Model(ID="claude-test"),
            Input=[
                message(Role.USER, "Start"),
                tool_call("a"),
                tool_call("b"),
                result,
                Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID="b")),
                message(Role.ASSISTANT, "More"),
                result,
            ],
        )
    )
    native = wire["messages"][2]["content"][0]["content"]
    update = wire["messages"][4]["content"][1:]
    assert native == update
    assert len(native) == 4
    assert (native[0]["text"], native[2]["text"]) == ("First image", "Second image")
    assert native[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}
    assert native[3]["source"] == {"type": "url", "url": "https://example.com/image.jpg"}
    assert "content" not in wire["messages"][2]["content"][1], (
        "empty tool output was replaced with invented content"
    )


@pytest.mark.parametrize(
    ("items", "want"),
    [
        ([], "messages must not be empty"),
        ([Item(Type=ItemType.TOOL_CALL, Data=None)], "data must be llm.ToolCall"),
        ([Item(Type=cast(ItemType, "unknown"), Data=None)], "unsupported item type"),
        ([message("tool", "bad")], "unsupported role"),
        (
            [message(Role.USER, "hi"), message(Role.SYSTEM, "late")],
            "system message must be the first input item",
        ),
        (
            [message(Role.SYSTEM, "first"), message(Role.SYSTEM, "second")],
            "system message must be the first input item",
        ),
        ([tool_call("a")], "missing immediately following results"),
        (
            [tool_call("a"), tool_call("b"), tool_result("a", "done", False)],
            'result for tool call "b"',
        ),
        (
            [
                tool_call("a"),
                message(Role.USER, "hi"),
                message(Role.ASSISTANT, "wait"),
                tool_result("a", "done", False),
            ],
            'result for tool call "a"',
        ),
        ([tool_result("a", "done", False)], "unknown tool call"),
        ([tool_call("a"), tool_call("a")], "duplicate tool call"),
        ([tool_call("")], "must have an ID and name"),
        (
            [
                Item(
                    Type=ItemType.PROVIDER,
                    Data=ProviderItem(
                        Display=ProviderDisplay(
                            Kind=ProviderDisplayKind.REASONING, Text="only summary"
                        )
                    ),
                )
            ],
            "provider item Raw is required",
        ),
    ],
    ids=[
        "empty",
        "wrong payload",
        "unknown tag",
        "unknown role",
        "late system",
        "second system",
        "missing result",
        "missing one result",
        "result too late",
        "unknown call",
        "duplicate call",
        "missing call id",
        "reasoning summary",
    ],
)
def test_request_rejects_invalid_history(items: list[Item], want: str) -> None:
    with pytest.raises(ValueError, match=re.escape(want)):
        request_input(items)


def test_request_missing_results_message() -> None:
    want = 'missing immediately following results for tool calls ["a" "b"]'
    with pytest.raises(ValueError, match=f"^{re.escape(want)}$"):
        request_input([tool_call("a"), tool_call("b")])


def _modify_model(**changes: Any) -> Request:
    request = valid_request()
    for name, value in changes.items():
        setattr(request.Model, name, value)
    return request


def _with_tools(*tools: Tool) -> Request:
    request = valid_request()
    request.Tools = list(tools)
    return request


@pytest.mark.parametrize(
    ("request_", "want"),
    [
        (_modify_model(ID=" "), "model must be set"),
        (_modify_model(MaxOutputTokens=-1), "invalid max output tokens -1"),
        (_modify_model(MaxOutputTokens=0), "invalid max output tokens 0"),
        (_modify_model(ReasoningEffort="extreme"), 'unsupported reasoning effort "extreme"'),
        (
            _with_tools(Tool(Type=ToolType.HOSTED, Name="web_search")),
            'tool 0: unsupported tool type "hosted"',
        ),
        (
            _with_tools(Tool(Type=cast(ToolType, "unknown"))),
            'tool 0: unsupported tool type "unknown"',
        ),
        (_with_tools(Tool(Type=ToolType.FUNCTION)), "tool 0: name must be set"),
        (
            _with_tools(Tool(Type=ToolType.FUNCTION, Name="Bash", Parameters={"type": "array"})),
            "tool 0: input schema must have type object",
        ),
    ],
    ids=[
        "model",
        "negative max",
        "zero max",
        "effort",
        "hosted tool",
        "tool type",
        "tool name",
        "schema",
    ],
)
def test_request_rejects_unsupported_settings(request_: Request, want: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(want)}$"):
        request_body(request_)


_INVALID_BASE64 = "image data URL must contain valid nonempty base64 data"


@pytest.mark.parametrize(
    ("part", "want"),
    [
        (
            ToolResultOutput(Kind=cast(ToolResultKind, "audio"), Value="unsupported"),
            'unsupported tool result kind "audio"',
        ),
        (
            ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="file:///tmp/image.png"),
            "image must be an HTTP(S) URL or a supported base64 data URL",
        ),
        (
            ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/svg+xml;base64,aGVsbG8="),
            "unsupported image data URL",
        ),
        (
            ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,%%%"),
            _INVALID_BASE64,
        ),
        (
            ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,"),
            _INVALID_BASE64,
        ),
    ],
)
def test_request_rejects_unsupported_tool_output(part: ToolResultOutput, want: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(want)}$"):
        request_tool_output([part])


def test_request_is_stable_and_preserves_numbers() -> None:
    arguments = '{"large":9007199254740993,"precise":0.1234567890123456789,"huge":1e1000}'
    request = Request(
        Model=Model(ID="claude-test"),
        Input=[
            message(Role.USER, "Start"),
            Item(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="a", Name="Bash", Arguments=arguments),
            ),
            tool_result("a", "done", False),
        ],
        Tools=[
            Tool(
                Type=ToolType.FUNCTION,
                Name="Bash",
                Parameters={"type": "object", "const": 9007199254740993, "maximum": Raw(b"1e1000")},
            )
        ],
    )
    original = list(request.Input)
    first = request_body(request)
    tool_input = decode(_content(first, 1)[0], dict[str, Raw])["input"]
    assert (
        bytes(tool_input)
        == b'{"huge":1e1000,"large":9007199254740993,"precise":0.1234567890123456789}'
    )
    assert b'"const":9007199254740993' in first
    assert b'"maximum":1e1000' in first
    for _ in range(10):
        assert request_body(request) == first, "request encoding is not deterministic"
    assert request.Input == original, "request encoding mutated input"


@pytest.mark.parametrize("arguments", ['{"unfinished":', "[]", "null", ""])
def test_request_replays_malformed_arguments_as_an_object(arguments: str) -> None:
    _, wire = encode_request(
        Request(
            Model=Model(ID="claude-test"),
            Input=[
                message(Role.USER, "Start"),
                Item(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="a", Name="Bash", Arguments=arguments),
                ),
                tool_result("a", "invalid arguments", False),
            ],
        )
    )
    assert wire["messages"][1]["content"][0]["input"] == {"invalid_arguments": arguments}


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"thinking","thinking":"escaped \\\\ \\" \\u2028","signature":"\\u0073igned",'
        '"data":{"future":9007199254740993}}',
        '{"type":"thinking","thinking":null,"signature":"signed","unknown":[1e1000]}',
        '{"type":"redacted_thinking","data":"opaque","signature":{"future":true},"thinking":42}',
        '{"type":"thinking","signature":null}',
        '{"type":"redacted_thinking","data":""}',
        '{"type":"reasoning","unknown":{"value":1e1000}}',
    ],
)
def test_request_replays_opaque_reasoning(raw: str) -> None:
    request = valid_request()
    request.Input.append(Item(Type=ItemType.PROVIDER, Data=ProviderItem(Raw=Raw(raw.encode()))))
    replayed = _content(request_body(request), 1)[0]
    assert parse_any(replayed) == parse_any(raw.encode())


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"thinking","signature":"signed","signature":"duplicate"}',
        '{"type":"redacted_thinking","data":"unfinished',
    ],
)
def test_request_rejects_malformed_reasoning_json(raw: str) -> None:
    request = valid_request()
    request.Input.append(Item(Type=ItemType.PROVIDER, Data=ProviderItem(Raw=Raw(raw.encode()))))
    with pytest.raises(ValueError, match=r"^encode messages request: "):
        request_body(request)
