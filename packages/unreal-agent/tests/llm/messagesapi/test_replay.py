from typing import Any

import pytest
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD, new_builder
from unreal_agent.gojson import Raw, decode, encode
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    Item,
    ItemType,
    Model,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Request,
    Response,
    Role,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.llm.messagesapi import decode_response
from unreal_agent.llm.messagesapi.request import request_input_block

from tests.llm.messagesapi._helpers import (
    encode_request,
    message,
    response_body,
    tool_call,
    tool_result,
)


def _assert_persisted_request_replay(request: Request, expected: bytes) -> None:
    restored = decode(encode(request), Request)
    body, _ = encode_request(restored)
    assert body == expected, "request changed after persistence"


def _text(value: str) -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]


def test_replay_preserves_committed_results_while_delivering_old_completions() -> None:
    first_encoding: bytes | None = None
    for order in (("a", "b"), ("b", "a")):
        builder = new_builder()
        builder.set_model(Model(ID="claude-test"))
        builder.add_external_input(
            Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b'"Start"'))
        )
        builder.commit()
        calls = decode_response(
            response_body(
                "tool_use",
                """[
			{"type":"thinking","thinking":"Two calls","signature":"signed"},
			{"type":"tool_use","id":"a","name":"Bash","input":{}},
			{"type":"tool_use","id":"b","name":"Bash","input":{}}
		]""",
            )
        )
        builder.add_model_response(calls)
        for call_id in order:
            builder.add_tool_result(call_id, [], True)
        builder.add_tool_result("b", _text("B completed"), False)
        first_request = builder.build().request
        first_body, first = encode_request(first_request)
        _assert_persisted_request_replay(first_request, first_body)
        if first_encoding is None:
            first_encoding = first_body
        else:
            assert first_body == first_encoding, "running result arrival order changed the wire"
        results: list[dict[str, Any]] = first["messages"][2]["content"]
        assert len(results) == 2
        assert results[0]["tool_use_id"] == "a"
        assert results[0]["content"][0]["text"] == TOOL_CALL_RUNNING_PAYLOAD
        assert results[1]["tool_use_id"] == "b"
        assert results[1]["content"][0]["text"] == "B completed"
        builder.commit()

        # A finishes while a model request is in flight. The response adds C
        # ahead of the staged completion, just as the coordinator does.
        builder.add_tool_result("a", _text("A completed"), False)
        builder.add_model_response(Response(Output=[tool_call("c")]))
        builder.add_tool_result("c", [], True)
        second_request = builder.build().request
        second_body, second = encode_request(second_request)
        _assert_persisted_request_replay(second_request, second_body)
        assert first["messages"] == second["messages"][: len(first["messages"])], (
            "a late completion changed the committed prefix"
        )
        results = second["messages"][4]["content"]
        assert len(results) == 3
        assert results[0]["tool_use_id"] == "c"
        assert results[0]["content"][0]["text"] == TOOL_CALL_RUNNING_PAYLOAD
        assert results[1]["type"] == "text"
        assert 'call "a", completed' in results[1]["text"]
        assert results[2]["text"] == "A completed"
        builder.commit()

        builder.add_model_response(Response(Output=[message(Role.ASSISTANT, "Waiting for C")]))
        builder.add_tool_result("c", _text("C completed"), False)
        third_request = builder.build().request
        third_body, third = encode_request(third_request)
        _assert_persisted_request_replay(third_request, third_body)
        assert second["messages"] == third["messages"][: len(second["messages"])], (
            "C's completion changed an earlier result"
        )
        results = third["messages"][6]["content"]
        assert len(results) == 2
        assert results[0]["type"] == "text"
        assert 'call "c", completed' in results[0]["text"]
        assert results[1]["text"] == "C completed"


def test_request_uses_raw_thinking_instead_of_display() -> None:
    raw = Raw(b'{"type":"thinking","thinking":"original","signature":"signed"}')
    stale = ProviderItem(
        Type="thinking",
        Raw=raw,
        Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text="stale display text"),
    )
    _, wire = encode_request(
        Request(
            Model=Model(ID="claude-test"),
            Input=[
                message(Role.USER, "Start"),
                Item(Type=ItemType.PROVIDER, Data=stale),
                tool_call("a"),
                tool_result("a", "done", False),
            ],
        )
    )
    assert wire["messages"][1]["content"][0]["type"] == "thinking"
    block = request_input_block(Item(Type=ItemType.PROVIDER, Data=stale))
    thinking = decode(bytes(block), dict[str, str])
    assert (thinking["thinking"], thinking["signature"]) == ("original", "signed")


@pytest.mark.parametrize("kind", ["tool_results", "thinking"])
def test_request_body_with_large_history(kind: str) -> None:
    request = Request(Model=Model(ID="claude-test"), Input=[message(Role.USER, "Hello")])
    for index in range(32):
        call_id = str(index)
        if kind == "thinking":
            request.Input.append(
                Item(
                    Type=ItemType.PROVIDER,
                    Data=ProviderItem(
                        Type="thinking",
                        Raw=Raw(
                            b'{"type":"thinking","thinking":"","signature":"'
                            + b"x" * (16 << 10)
                            + b'","provider_state":{"v":2}}'
                        ),
                    ),
                )
            )
        request.Input.extend([tool_call(call_id), tool_result(call_id, "x" * (4 << 10), False)])
    _, wire = encode_request(request)
    assert len(wire["messages"]) == 65
