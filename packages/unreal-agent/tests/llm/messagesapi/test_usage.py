from typing import Any

import pytest
from unreal_agent.llm import TokenUsage
from unreal_agent.llm.messagesapi import decode_response
from unreal_agent.llm.messagesapi.stream import StreamingState

from tests.llm.messagesapi._helpers import observe_events

_COUNTERS = (
    '"input_tokens":7,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,'
    '"output_tokens":11,"output_tokens_details":{"thinking_tokens":5}'
)
_TOTAL = TokenUsage(
    InputTokens=57,
    CachedInputTokens=30,
    CacheWriteInputTokens=20,
    OutputTokens=11,
    ReasoningTokens=5,
)
_FINAL = TokenUsage(InputTokens=57, CachedInputTokens=30, CacheWriteInputTokens=20, OutputTokens=11)

_CASES: list[Any] = [
    pytest.param("served", "end_turn", "", {"served": _TOTAL}, id="single model"),
    pytest.param("", "end_turn", "", {"": _TOTAL}, id="empty model"),
    pytest.param("served", "end_turn", "null", {"served": _TOTAL}, id="null iterations"),
    pytest.param("served", "end_turn", "[]", {"served": _TOTAL}, id="empty iterations"),
    pytest.param(
        "primary",
        "end_turn",
        "["
        '{"type":"message","model":"primary","input_tokens":10,"cache_read_input_tokens":3,'
        '"cache_creation_input_tokens":2,"output_tokens":2},'
        '{"type":"message","model":"primary","input_tokens":5,"cache_read_input_tokens":7,'
        '"output_tokens":1},'
        '{"type":"message","model":"second","input_tokens":2,"output_tokens":0},'
        '{"type":"fallback_message","model":"final","input_tokens":7,'
        '"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"output_tokens":11}'
        "]",
        {
            "primary": TokenUsage(
                InputTokens=27, CachedInputTokens=10, CacheWriteInputTokens=2, OutputTokens=3
            ),
            "second": TokenUsage(InputTokens=2),
            "final": _FINAL,
        },
        id="multiple hops and repeated model",
    ),
    pytest.param(
        "served",
        "end_turn",
        '[{"type":"fallback_message","model":"served","input_tokens":7,'
        '"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"output_tokens":11}]',
        {"served": _FINAL},
        id="sticky routing",
    ),
    pytest.param(
        "final",
        "refusal",
        '[{"type":"message","model":"primary","input_tokens":9,"output_tokens":2},'
        '{"type":"fallback_message","model":"final","input_tokens":7,"output_tokens":11}]',
        {
            "primary": TokenUsage(InputTokens=9, OutputTokens=2),
            "final": TokenUsage(InputTokens=7, OutputTokens=11),
        },
        id="all models declined",
    ),
    pytest.param(
        "served",
        "end_turn",
        '[{"type":"message","model":null,"input_tokens":9,"output_tokens":2},'
        '{"type":"fallback_message","model":"served","input_tokens":7,"output_tokens":11}]',
        {
            "": TokenUsage(InputTokens=9, OutputTokens=2),
            "served": TokenUsage(InputTokens=7, OutputTokens=11),
        },
        id="unattributed iteration",
    ),
]


@pytest.mark.parametrize("streaming", [False, True], ids=["streaming=false", "streaming=true"])
@pytest.mark.parametrize(("model", "stop", "iterations", "want"), _CASES)
def test_usage_by_model(
    model: str, stop: str, iterations: str, want: dict[str, TokenUsage], streaming: bool
) -> None:
    usage = "{" + _COUNTERS
    if iterations:
        usage += ',"iterations":' + iterations
    usage += "}"
    body = (
        f'{{"type":"message","id":"msg","role":"assistant","model":"{model}",'
        f'"stop_reason":"{stop}","content":[],"usage":{usage}}}'
    ).encode()
    raw = usage.encode()
    if streaming:
        state = StreamingState()
        observe_events(
            state,
            '{"type":"message_start","message":{"type":"message","id":"msg","role":"assistant",'
            f'"model":"{model}","content":[],"usage":{{"input_tokens":1,"output_tokens":0}}}}}}',
            f'{{"type":"message_delta","delta":{{"stop_reason":"{stop}"}},"usage":{usage}}}',
            '{"type":"message_stop"}',
        )
        body = state.unwrap()
        raw = body[body.index(b'"usage":') + len(b'"usage":') :].removesuffix(b"}")
    response = decode_response(body)
    assert response.Usage.ByModel == want
    assert response.Usage.token_usage == _TOTAL, "top-level usage changed"
    assert bytes(response.Usage.Raw) == raw, "raw usage changed"


@pytest.mark.parametrize(
    "iterations", ["{}", '[{"model":42}]', '[{"model":"served","input_tokens":"bad"}]']
)
def test_usage_rejects_malformed_iterations(iterations: str) -> None:
    body = (
        '{"type":"message","role":"assistant","stop_reason":"end_turn","usage":{"iterations":'
        + iterations
        + "}}"
    )
    with pytest.raises(ValueError):  # noqa: PT011
        decode_response(body.encode())
