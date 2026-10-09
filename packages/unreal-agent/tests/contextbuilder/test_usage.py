from typing import cast

import pytest
from unreal_agent.contextbuilder import new_builder
from unreal_agent.contextbuilder.builder import _Builder, _PrefixToken
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode, Settings
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    ReasoningEffort,
    Response,
    Role,
    ToolCall,
    Usage,
)

from tests.contextbuilder._helpers import (
    add_compaction_test_turn,
    build_compaction_context,
    new_compaction_history,
    new_internal_builder,
    reasoning,
    text,
    usable_compaction_response,
    user,
    with_preamble,
)


@pytest.mark.parametrize(
    ("threshold", "usage", "want"),
    [
        pytest.param(83_616, [], False, id="no checkpoint"),
        pytest.param(0, [Usage(InputTokens=100_000)], True, id="zero threshold"),
        pytest.param(
            83_616, [Usage(InputTokens=80_000, OutputTokens=3_615)], False, id="below threshold"
        ),
        pytest.param(
            83_616, [Usage(InputTokens=80_000, OutputTokens=3_616)], True, id="at threshold"
        ),
        pytest.param(
            83_616, [Usage(InputTokens=80_000, OutputTokens=3_617)], True, id="above threshold"
        ),
        pytest.param(
            83_616, [Usage(InputTokens=90_000), Usage(InputTokens=40_000)], False, id="latest only"
        ),
        pytest.param(
            83_616, [Usage(InputTokens=90_000), Usage()], False, id="missing latest usage"
        ),
        pytest.param(
            83_616,
            [
                Usage(
                    InputTokens=80_000,
                    OutputTokens=3_615,
                    CachedInputTokens=50_000,
                    CacheWriteInputTokens=10_000,
                    ReasoningTokens=2_000,
                )
            ],
            False,
            id="included tokens",
        ),
        pytest.param(1_000, [Usage(InputTokens=1_000)], True, id="small threshold"),
        pytest.param(8_000, [Usage()], False, id="small threshold missing usage"),
    ],
)
def test_needs_compaction_uses_latest_checkpoint(
    threshold: int, usage: list[Usage], want: bool
) -> None:
    current = new_builder()
    current.set_model(Model(CompactionThreshold=threshold))
    for entry in usage:
        current.add_model_response(Response(Usage=entry))
    assert current.needs_compaction() == want
    current.add_tool_result("call", text("result" * 100_000), False)
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="new input"))
    assert current.needs_compaction() == want, "staged arrivals changed needs_compaction()"
    current.commit()
    assert current.needs_compaction() == want, "committing arrivals changed needs_compaction()"


def test_needs_compaction_uses_updated_threshold() -> None:
    current = new_builder()
    current.set_model(Model(ID="model", CompactionThreshold=200_000))
    current.add_model_response(Response(Usage=Usage(InputTokens=90_000)))
    assert not current.needs_compaction(), "usage below threshold needs compaction"
    current.add_control_message(
        ControlMessage(
            Mode=ControlMode.UPDATE_SETTINGS, Parameters=Settings(CompactionThreshold=80_000)
        )
    )
    assert current.needs_compaction(), "lowering the threshold did not enable compaction"
    current.add_control_message(
        ControlMessage(
            Mode=ControlMode.UPDATE_SETTINGS,
            Parameters=Settings(Model="model", CompactionThreshold=200_000),
        )
    )
    assert not current.needs_compaction(), "raising the threshold kept the lower threshold"


@pytest.mark.parametrize(
    ("settings", "reset"),
    [
        pytest.param(Settings(), False, id="omitted"),
        pytest.param(Settings(Model="initial", CompactionThreshold=50_000), False, id="same model"),
        pytest.param(Settings(CompactionThreshold=50_000), False, id="threshold only"),
        pytest.param(Settings(ReasoningEffort=ReasoningEffort.HIGH), False, id="reasoning effort"),
        pytest.param(Settings(Model="next"), True, id="different model"),
        pytest.param(
            Settings(Model="next", CompactionThreshold=50_000),
            True,
            id="different model and threshold",
        ),
    ],
)
def test_model_settings_reset_checkpoints_only_when_model_changes(
    settings: Settings, reset: bool
) -> None:
    current = new_compaction_history()
    current.set_model(Model(ID="initial", CompactionThreshold=50_000))
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="pending"))
    before = build_compaction_context(current).request.Input
    checkpoints = list(current._prefix_tokens)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.UPDATE_SETTINGS, Parameters=settings)
    )
    assert build_compaction_context(current).request.Input == before, (
        "settings changed conversation history"
    )
    if not reset:
        message = "settings for the same model discarded measured usage"
        assert current._prefix_tokens == checkpoints, message
        assert current.needs_compaction(), message
        return
    assert len(current._prefix_tokens) == 0, "model switch retained old usage checkpoints"
    assert not current.needs_compaction(), "model switch retained old usage checkpoints"
    with pytest.raises(ValueError, match="no history eligible for compaction"):
        current.build_compaction()
    add_compaction_test_turn(current, 5, 30_000)
    want = [_PrefixToken(len(current._committed_prefix), 30_000)]
    assert current._prefix_tokens == want
    assert current._compaction_cutoff() == 0, (
        "new model did not establish a fresh compaction cutoff"
    )


def test_needs_compaction_resets_with_reconstruction() -> None:
    current = new_compaction_history()
    current.set_model(Model(CompactionThreshold=50_000))
    assert current.needs_compaction(), "expected compaction before reconstruction"
    summary = usable_compaction_response("summary")
    summary.Usage = Usage(InputTokens=200_000, OutputTokens=4_000)
    rebuilt = current.compact(summary)
    assert rebuilt is not None, "compaction failed"
    assert not rebuilt.needs_compaction(), "rebuilt context reused old or summary-request usage"
    rebuilt.commit()
    rebuilt.add_model_response(Response(Usage=Usage(InputTokens=55_000)))
    assert rebuilt.needs_compaction(), "rebuilt context did not use the new ordinary checkpoint"


def test_builder_records_token_checkpoints() -> None:
    current = new_internal_builder()
    first = Response(
        Output=[
            Item(ProviderID="reasoning", Type=ItemType.PROVIDER, Data=reasoning("thinking")),
            Item(
                ProviderID="message",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="Working."),
            ),
            Item(
                ProviderID="call", Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="A", Name="test")
            ),
        ],
        Usage=Usage(
            InputTokens=100,
            OutputTokens=20,
            CachedInputTokens=70,
            CacheWriteInputTokens=10,
            ReasoningTokens=5,
            Raw=Raw(b'{"input_tokens":100,"output_tokens":20}'),
        ),
    )
    current.add_model_response(first)
    second = Response(
        Output=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="Continuing."))],
        Usage=Usage(InputTokens=150, OutputTokens=10),
    )
    current.add_model_response(second)
    assert first.Output is not None
    assert second.Output is not None
    assert current._committed_prefix == [*with_preamble(*first.Output), *second.Output]
    assert current._prefix_tokens == [_PrefixToken(4, 120), _PrefixToken(5, 160)]


def test_builder_records_empty_response_checkpoints() -> None:
    current = new_internal_builder()
    current.commit()
    current.commit()
    current.add_model_response(Response(Usage=Usage(InputTokens=100)))
    current.commit()
    current.add_model_response(Response())
    assert current._prefix_tokens == [_PrefixToken(1, 100), _PrefixToken(1, 0)]
    assert current.build().request.Input == with_preamble(), "empty entries changed request input"


def test_compaction_starts_new_token_checkpoints() -> None:
    current = new_compaction_history()
    current.add_tool_result("call", text("large result"), False)
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="new input"))
    original = list(current._prefix_tokens)
    summary = usable_compaction_response("summary")
    summary.Usage = Usage(InputTokens=200_000, OutputTokens=4_000)
    fresh = current.compact(summary)
    assert fresh is not None, "compaction failed"
    rebuilt = cast(_Builder, fresh)
    assert len(rebuilt._prefix_tokens) == 0, "compaction retained checkpoints from the old context"
    assert rebuilt._compaction_cutoff() is None, "compaction reused unmeasured context"
    before = build_compaction_context(rebuilt).request.Input
    rebuilt.commit()
    response = usable_compaction_response("F output")
    response.Usage = Usage(InputTokens=49_000, OutputTokens=1_000)
    rebuilt.add_model_response(response)
    assert response.Output is not None
    want = [*before, *response.Output]
    assert rebuilt._committed_prefix == want, "recording the first checkpoint changed context items"
    checkpoints = [_PrefixToken(len(want), 50_000)]
    assert rebuilt._prefix_tokens == checkpoints
    cutoff = rebuilt._compaction_cutoff()
    message = "first response's input was excluded from the compaction cut"
    assert cutoff is not None, message
    assert rebuilt._prefix_tokens[cutoff].committed_prefix_index == len(want), message
    add_compaction_test_turn(rebuilt, 6, 60_000)
    checkpoints.append(_PrefixToken(len(want) + 3, 60_000))
    assert rebuilt._prefix_tokens == checkpoints
    second = rebuilt.compact(summary)
    assert second is not None, "second compaction did not reset checkpoints"
    assert len(cast(_Builder, second)._prefix_tokens) == 0, (
        "second compaction did not reset checkpoints"
    )
    want = [
        *with_preamble(user("summary")),
        *rebuilt._committed_prefix[len(before) + len(response.Output) :],
    ]
    assert build_compaction_context(second).request.Input == want
    assert current._prefix_tokens == original, (
        "compaction changed the original builder's checkpoints"
    )
