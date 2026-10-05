import json

import pytest
from unreal_agent.gojson import Raw, decode
from unreal_agent.llm import Message, Reasoning, StopReason, ToolCall
from unreal_agent.llm.responsesapi import StreamEndedError, decode_response
from unreal_agent.llm.responsesapi.stream import ResponseState
from unreal_agent.primitives import sse_data
from unreal_agent.primitives._errors import UnexpectedEOFError

from tests.llm.responsesapi._helpers import COMPLETED_RESPONSE, FALLBACK_CALL, FALLBACK_MESSAGE

FALLBACK_REASONING = (
    b'{"id":"rs-1","type":"reasoning","summary":[],"encrypted_content":"opaque",'
    b'"vendor_field":"kept"}'
)


def assert_unexpected_eof(state: ResponseState) -> None:
    with pytest.raises(StreamEndedError) as caught:
        state.unwrap()
    assert isinstance(caught.value.__cause__, UnexpectedEOFError)


@pytest.mark.parametrize(
    ("status", "extra", "stop", "failure"),
    [
        ("completed", "", StopReason.COMPLETE, False),
        (
            "incomplete",
            ',"incomplete_details":{"reason":"max_output_tokens"}',
            StopReason.MAX_OUTPUT_TOKENS,
            False,
        ),
        (
            "incomplete",
            ',"incomplete_details":{"reason":"content_filter"}',
            StopReason.REFUSED,
            False,
        ),
        ("failed", ',"error":{"code":"server_error","message":"failed"}', StopReason.NONE, True),
    ],
)
def test_responses_terminal_statuses(
    status: str, extra: str, stop: StopReason, failure: bool
) -> None:
    frame = (
        f'data: {{"type":"response.{status}","response":{{"id":"r","status":"{status}",'
        f'"output":[]{extra}}}}}'
    ).encode()
    state = ResponseState()
    payload = sse_data(frame)
    assert payload is not None
    state.observe(payload)
    response = decode_response(state.unwrap())
    assert response.Stop == stop
    assert (response.Failure is not None) == failure


def test_responses_terminal_preserves_provider_usage() -> None:
    raw = (
        b'{"id":"r","status":"completed","output":[],"usage":{"prompt_tokens":12,'
        b'"completion_tokens":3,"provider_extension":{"cost":0.25}}}'
    )
    state = ResponseState()
    state.observe(b'{"type":"response.completed","response":' + raw + b"}")
    body = state.unwrap()
    assert body == raw
    assert b'"provider_extension":{"cost":0.25}' in bytes(decode_response(body).Usage.Raw)


@pytest.mark.parametrize(
    "data",
    [
        b'{"response":{}}',
        b'{"type":"","response":{}}',
        b'{"type":1,"response":{}}',
        b'{"type":"other.completed","response":{}}',
        b'{"type":"response.","response":{}}',
        b'{"type":"response.completed"}',
        b'{"type":"response.completed","response":null}',
        b'{"type":"response.completed","response":[]}',
        b'{"type":"response.completed","response":"not an object"}',
        b'{"type":"response.output_text.delta","delta":"unfinished"}',
        b"null",
        b"[]",
    ],
)
def test_response_state_validates_envelope(data: bytes) -> None:
    state = ResponseState()
    try:
        state.observe(data)
    except ValueError:
        return
    with pytest.raises(StreamEndedError):
        state.unwrap()


def test_response_state_leaves_status_to_response_decoder() -> None:
    raw = b'{"id":"r","status":"in_progress","output":[]}'
    state = ResponseState()
    state.observe(b'{"type":"response.completed","response":' + raw + b"}")
    body = state.unwrap()
    assert body == raw
    with pytest.raises(ValueError, match="unsupported response status"):
        decode_response(body)


@pytest.mark.parametrize(
    ("kind", "terminal"),
    [
        ("response.created", False),
        ("response.queued", False),
        ("response.in_progress", False),
        ("response.completed", True),
        ("response.failed", True),
        ("response.incomplete", True),
        ("response.output_item.done", False),
        ("response.output_text.done", False),
    ],
)
def test_response_state_captures_only_terminal_types(kind: str, terminal: bool) -> None:
    raw = f'{{"id":"r","status":"{kind.removeprefix("response.")}","output":[]}}'.encode()
    event = f'{{"type":"{kind}","sequence_number":7,"response":{raw.decode()}}}'.encode()
    if kind == "response.output_item.done":
        event = (
            f'{{"type":"{kind}","sequence_number":7,"output_index":0,'
            '"item":{"type":"reasoning","id":"rs","summary":[]}}'
        ).encode()
    state = ResponseState()
    state.observe(event)
    if terminal:
        assert state.unwrap() == raw
    else:
        assert state.response is None
        assert_unexpected_eof(state)


def test_response_state_keeps_captured_terminal_response() -> None:
    state = ResponseState()
    state.observe(b'{"type":"response.completed","response":' + COMPLETED_RESPONSE + b"}")
    for kind in (
        "response.in_progress",
        "response.queued",
        "response.created",
        "response.output_text.delta",
    ):
        state.observe(
            f'{{"type":"{kind}","response":{{"id":"not-terminal","status":"in_progress"}}}}'.encode()
        )
        assert state.unwrap() == COMPLETED_RESPONSE


@pytest.mark.parametrize(
    "output",
    [
        b"",
        b',"output":[]',
        b',"output":[ ]',
        b',"output":null',
        b',"output":[' + FALLBACK_MESSAGE + b"]",
    ],
)
def test_collected_items_fallback(output: bytes) -> None:
    state = ResponseState()
    for index, body in (
        (2, FALLBACK_CALL),
        (0, FALLBACK_REASONING),
        (1, FALLBACK_MESSAGE),
        (1, FALLBACK_MESSAGE),
    ):
        state.observe(
            b'{"type":"response.output_item.done","output_index":%d,"item":%s}' % (index, body)
        )
    assert_unexpected_eof(state)
    usage = b'{"input_tokens":12,"output_tokens":8,"cost":0.25,"vendor_usage":{"tokens":20}}'
    terminal = (
        b'{"id":"r","status":"completed","vendor":"preserved","usage":' + usage + output + b"}"
    )
    state.observe(b'{"type":"response.completed","response":' + terminal + b"}")
    body = state.unwrap()
    assert state.response == terminal, "captured terminal response mutated"
    fields = decode(body, dict[str, Raw])
    assert bytes(fields["vendor"]) == b'"preserved"'
    assert bytes(fields["usage"]) == usage
    response = decode_response(body)
    if output == b',"output":[' + FALLBACK_MESSAGE + b"]":
        assert body == terminal
        assert len(response.Output or []) == 1
    else:
        assert len(response.Output or []) == 3
        reasoning = (response.Output or [])[0].Data
        assert isinstance(reasoning, Reasoning)
        assert b'"vendor_field":"kept"' in bytes(reasoning.Raw)
        message = (response.Output or [])[1].Data
        assert isinstance(message, Message)
        assert (message.Text, message.Phase) == ("fallback", "final_answer")
        call = (response.Output or [])[2].Data
        assert isinstance(call, ToolCall)
        assert (call.CallID, call.Arguments) == ("call-1", '{"command":"pwd"}')
    assert response.ID == "r"
    assert response.Stop == StopReason.COMPLETE
    assert response.Usage.InputTokens == 12
    assert state.unwrap() == body, "unwrap is not stable"


@pytest.mark.parametrize(
    ("status", "extra", "stop", "failure"),
    [
        (
            "incomplete",
            b',"incomplete_details":{"reason":"max_output_tokens"}',
            StopReason.MAX_OUTPUT_TOKENS,
            False,
        ),
        ("failed", b',"error":{"code":"server_error","message":"failed"}', StopReason.NONE, True),
    ],
)
def test_collected_items_preserve_terminal_status(
    status: str, extra: bytes, stop: StopReason, failure: bool
) -> None:
    state = ResponseState()
    state.observe(
        b'{"type":"response.output_item.done","output_index":0,"item":' + FALLBACK_MESSAGE + b"}"
    )
    state.observe(
        b'{"type":"response.%s","response":{"id":"r","status":"%s","output":[]%s}}'
        % (status.encode(), status.encode(), extra)
    )
    response = decode_response(state.unwrap())
    assert response.Stop == stop
    assert (response.Failure is not None) == failure
    assert len(response.Output or []) == 1


def test_collected_items_splice_matches_go() -> None:
    state = ResponseState()
    state.observe(
        b'{"type":"response.output_item.done","output_index":0,'
        b'"item": { "id" : "x", "type":"reasoning", "s":"\\u003c<" } }'
    )
    state.observe(
        b'{"type":"response.completed","response": { "z":1, "id":"r", "a" : { "q" : 2 }, '
        b'"output" : [ ] } }'
    )
    assert state.unwrap() == (
        b'{"a":{"q":2},"id":"r","output":[{"id":"x","type":"reasoning","s":"<<"}],"z":1}'
    )


@pytest.mark.parametrize(
    ("event", "code", "message", "error_type"),
    [
        (
            b'{"type":"error","code":"usage_limit_reached","message":"limit reached"}',
            "usage_limit_reached",
            "limit reached",
            "error",
        ),
        (
            b'{"type":"error","error":{"type":"insufficient_quota","code":"credit_balance_exhausted",'
            b'"message":"You have no credits remaining."}}',
            "credit_balance_exhausted",
            "You have no credits remaining.",
            "insufficient_quota",
        ),
    ],
)
def test_response_state_reads_error_events(
    event: bytes, code: str, message: str, error_type: str
) -> None:
    state = ResponseState()
    state.observe(event)
    assert state.err is not None
    assert (state.err.status_code, state.err.code, state.err.message, state.err.type) == (
        200,
        code,
        message,
        error_type,
    )
    with pytest.raises(type(state.err)):
        state.unwrap()
    assert json.loads(event)["type"] == "error"
