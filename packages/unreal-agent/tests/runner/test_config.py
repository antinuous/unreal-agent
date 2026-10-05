"""Runner configuration tests."""

import io

import pytest
from unreal_agent.runner import Config, Request, RunnerError, run
from unreal_agent.runner.streams import Reader

from tests.runner._helpers import must_not_read_environment, parse_test_request, reader


def _no_tool_factory(_: Reader) -> tuple[Request, None]:
    return Request(prompt="hello"), None


@pytest.mark.parametrize(
    ("config", "want"),
    [
        pytest.param(
            Config(parse_request=parse_test_request), "runner name must be set", id="missing name"
        ),
        pytest.param(
            Config(name=" \t", parse_request=parse_test_request),
            "runner name must be set",
            id="blank name",
        ),
        pytest.param(Config(name="test-runner"), "request parser must be set", id="missing parser"),
        pytest.param(
            Config(name="test-runner", parse_request=_no_tool_factory),
            "request parser returned no tool factory",
            id="missing tool factory",
        ),
    ],
)
async def test_run_rejects_incomplete_configuration(config: Config, want: str) -> None:
    with pytest.raises(RunnerError) as raised:
        await run(
            [],
            must_not_read_environment,
            reader('{"prompt":"hello"}'),
            io.BytesIO(),
            io.BytesIO(),
            config,
        )
    assert str(raised.value) == want
