import re
from typing import Final

import pytest
from unreal_agent.gojson import decode, encode
from unreal_agent.llm import ToolCall
from unreal_agent.operation import DEFAULT_MAX_OUTPUT_LENGTH, MAX_OUTPUT_LENGTH
from unreal_agent.tool import (
    BASH_NAME,
    CallStatus,
    Registry,
    StaticTranslators,
    Translator,
    bash,
    parse_max_output_length,
)

from tests.tool._helpers import RecordingContext

_TRUNCATION_MARKER: Final = re.compile(r"\.\.\.[0-9]+ bytes truncated\.\.\.")


def new_bash() -> Translator:
    return bash.BashTranslator(bash.Config(shell="/bin/sh", base_directory="/operations"))


def test_output_length_maximum_matches_tool_schemas() -> None:
    assert parse_max_output_length(str(MAX_OUTPUT_LENGTH).encode()) == MAX_OUTPUT_LENGTH
    registry = Registry(StaticTranslators(), BASH_NAME)
    checked = 0
    for definition in registry.static_definitions():
        if definition.tool.Name != BASH_NAME:
            continue
        schema = definition.tool.Parameters["properties"]["max_output_length"]
        assert schema["maximum"] == MAX_OUTPUT_LENGTH
        assert schema["default"] == DEFAULT_MAX_OUTPUT_LENGTH
        checked += 1
    assert checked == 1


_LONG_NAME: Final = "é" * (DEFAULT_MAX_OUTPUT_LENGTH + 1)
_QUOTED_LONG_NAME: Final = encode(_LONG_NAME).decode()


@pytest.mark.parametrize(
    ("arguments", "limit", "truncated"),
    [
        pytest.param('{"command":null,"max_output_length":10}', 10, True, id="Bash requested"),
        pytest.param('{"command":null}', DEFAULT_MAX_OUTPUT_LENGTH, False, id="Bash untruncated"),
        pytest.param(
            f"{{{_QUOTED_LONG_NAME}:0,{_QUOTED_LONG_NAME}:0}}",
            DEFAULT_MAX_OUTPUT_LENGTH,
            True,
            id="Bash malformed",
        ),
        pytest.param(
            f'{{"max_output_length":{_QUOTED_LONG_NAME}}}',
            DEFAULT_MAX_OUTPUT_LENGTH,
            False,
            id="Bash invalid limit",
        ),
    ],
)
def test_validation_errors_are_bounded_before_result_translation(
    arguments: str, limit: int, truncated: bool
) -> None:
    translator = new_bash()
    status = translator.translate(RecordingContext(), ToolCall(Arguments=arguments))
    retained = _TRUNCATION_MARKER.sub("", status.Error)
    assert not status.WaitingFor
    assert status.Error
    status.Error.encode("utf-8")
    assert len(retained) <= limit
    assert status.ErrorTruncated == truncated

    status = decode(encode(status), CallStatus)
    result = translator.translate_result("invalid-call", status, []).to_llm_result()
    assert result.Output[0].Value.startswith("Error: ")
    assert result.Output[0].Value.removeprefix("Error: ") == status.Error
