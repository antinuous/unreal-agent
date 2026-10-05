"""Max-attempts resolution tests."""

import io

import pytest
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.llm import Response, StopReason
from unreal_agent.runner import Client, RunnerError, run_main
from unreal_agent.runner.run import (
    LLM_API_KEY_ENVIRONMENT,
    LLM_MAX_ATTEMPTS_ENVIRONMENT,
    resolve_max_attempts,
)

from tests.runner._helpers import FakeClient, environment, reader, runner_config


@pytest.mark.parametrize(
    ("requested", "value", "want"),
    [
        pytest.param(None, "", 5, id="default"),
        pytest.param(None, " 3 ", 3, id="environment"),
        pytest.param(None, "1", 1, id="environment disables retries"),
        pytest.param(2, "3", 2, id="request override"),
        pytest.param(1, "3", 1, id="request disables retries"),
        pytest.param(1, "bad", 1, id="unused invalid environment"),
        pytest.param(-1, "", None, id="negative request"),
        pytest.param(None, "-1", None, id="negative environment"),
        pytest.param(None, "1.5", None, id="invalid environment"),
        pytest.param(0, "", None, id="zero request"),
        pytest.param(None, "0", None, id="zero environment"),
        pytest.param(None, "999999999999999999999999999", None, id="environment overflow"),
    ],
)
def test_resolve_max_attempts(requested: int | None, value: str, want: int | None) -> None:
    if want is None:
        with pytest.raises(RunnerError):
            resolve_max_attempts(requested, lambda _: value)
    else:
        assert resolve_max_attempts(requested, lambda _: value) == want


async def test_run_main_request_disables_retries(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    def respond(_: LLMRequest) -> Response:
        return Response(ID="response-1", Stop=StopReason.COMPLETE)

    client = FakeClient(respond)
    config = runner_config(client)

    def new_client(_: str, __: str, max_attempts: int, ___: object) -> Client:
        if max_attempts != 1:
            raise ValueError("request did not disable retries")
        return client

    config.providers[0].new_client = new_client
    code = await run_main(
        [
            "-workspace",
            str(tmp_path_factory.mktemp("workspace")),
            "-session-directory",
            str(tmp_path_factory.mktemp("sessions")),
        ],
        environment({LLM_API_KEY_ENVIRONMENT: "secret", LLM_MAX_ATTEMPTS_ENVIRONMENT: "3"}),
        reader('{"prompt":"hello","max_attempts":1}'),
        io.BytesIO(),
        io.BytesIO(),
        config,
    )
    assert (code, client.closed) == (0, True)
