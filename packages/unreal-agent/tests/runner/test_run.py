# ruff: noqa: ASYNC240
import asyncio
import glob
import os
import time
import uuid
from pathlib import Path

import pytest
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.inbox import ControlMode, Input, Settings
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    ReasoningEffort,
    Response,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.runner import Client, Config, Provider, Request, RequestMessage, RunnerError
from unreal_agent.runner.request import reasoning_effort, validate_request
from unreal_agent.runner.run import (
    LLM_API_KEY_ENVIRONMENT,
    LLM_BASE_URL_ENVIRONMENT,
    LLM_MAX_ATTEMPTS_ENVIRONMENT,
    LLM_MODEL_ENVIRONMENT,
    LLM_PROVIDER_ENVIRONMENT,
    load_dotenv,
    resolve_session_directory,
)
from unreal_agent.sessionstore import ItemKind

from tests.runner._helpers import (
    FakeClient,
    call_run_main,
    decode_lines,
    environment,
    event_types,
    input_ids,
    item_kinds,
    item_sequence,
    message_response,
    parse_test_request,
    reader,
    runner_config,
)


async def test_run_main_executes_batched_messages(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    requests: list[LLMRequest] = []

    def respond(request: LLMRequest) -> Response:
        requests.append(request)
        response = message_response("done", "response-1")
        response.Usage = Usage(InputTokens=4, OutputTokens=2)
        return response

    client = FakeClient(respond)
    workspace = tmp_path_factory.mktemp("workspace")
    skill_path = workspace / ".harness" / "skills" / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, mode=0o700)
    skill_path.write_text("---\nname: review\ndescription: Review code.\n---\n")
    sessions = tmp_path_factory.mktemp("sessions")
    log_directory = tmp_path_factory.mktemp("logs") / "logs"
    code, stdout, stderr = await call_run_main(
        [
            "-workspace",
            str(workspace),
            "-session-directory",
            str(sessions),
            "-log-directory",
            str(log_directory),
        ],
        environment({"OPENAI_API_KEY": "secret", "SHELL": "/bin/sh"}),
        reader("""{
            "messages":[
                {
                    "role":"user",
                    "content":"first",
                    "message_id":"69621f8d-4f4d-49a5-8f7d-3b24fd855c01"
                },
                {"role":"user","content":"second"}
            ],
            "system_prompt":"be concise",
            "model":"gpt-test",
            "thinking_level":"medium"
        }"""),
        runner_config(client),
    )
    assert code == 0, (stderr, stdout)
    request = requests[0]
    assert request.Model.ID == "gpt-test"
    assert request.Model.ReasoningEffort == ReasoningEffort.MEDIUM
    messages = [item.Data for item in request.Input if isinstance(item.Data, Message)]
    assert len(messages) == 3
    assert messages[0].Role == Role.SYSTEM
    assert "<name>review</name>" in messages[0].Text
    assert f"<location>{skill_path}</location>" in messages[0].Text
    assert messages[0].Text.endswith("\n\nbe concise")
    assert messages[1:] == [
        Message(Role=Role.USER, Text="first"),
        Message(Role=Role.USER, Text="second"),
    ]
    assert sorted(tool.Name for tool in request.Tools) == ["Bash", "SkillUse", "ViewImage"]
    assert item_sequence(stdout) in (
        "input.control input.external input.external input.control turn model_response",
        "input.control input.external input.external turn input.control model_response",
        "input.control input.external input.external turn model_response input.control",
    )
    items = decode_lines(stdout)
    first = items[0].Data
    assert isinstance(first, Input)
    control = first.decode_control_message()
    assert control.Mode == ControlMode.UPDATE_SETTINGS
    assert control.Parameters == Settings(ReasoningEffort=ReasoningEffort.MEDIUM)
    ids = input_ids(stdout)
    assert len(ids) == 2
    assert ids[0] == "69621f8d-4f4d-49a5-8f7d-3b24fd855c01"
    for id in ids:
        uuid.UUID(id)
    logs = glob.glob(str(log_directory / "*.jsonl"))
    assert len(logs) == 1
    time.strptime(os.path.basename(logs[0]), "%Y%m%d-%H%M%S.jsonl")
    assert Path(logs[0]).read_text() == stdout
    assert client.closed


@pytest.mark.parametrize(
    ("key_environment", "generic_key", "provider_key", "want_key", "want_error"),
    [
        pytest.param(
            "CUSTOM_CREDENTIAL", "", "provider-secret", "provider-secret", False, id="provider key"
        ),
        pytest.param(
            "CUSTOM_CREDENTIAL",
            "generic-secret",
            "provider-secret",
            "generic-secret",
            False,
            id="generic key takes precedence",
        ),
        pytest.param(
            "",
            "generic-secret",
            "provider-secret",
            "",
            False,
            id="delegated authentication ignores API keys",
        ),
        pytest.param("CUSTOM_CREDENTIAL", "", "", "", True, id="missing required key"),
    ],
)
async def test_run_main_uses_provider_authentication_configuration(
    key_environment: str,
    generic_key: str,
    provider_key: str,
    want_key: str,
    want_error: bool,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    client = FakeClient(lambda _: Response(ID="response-1", Stop=StopReason.COMPLETE))
    created = False

    def new_client(api_key: str, _: str, __: int, ___: object) -> Client:
        nonlocal created
        created = True
        if api_key != want_key:
            raise ValueError("unexpected API key")
        return client

    providers = [
        Provider(
            name="custom",
            default_model="test-model",
            api_key_environment=key_environment,
            new_client=new_client,
        )
    ]
    code, _, stderr = await call_run_main(
        [
            "-workspace",
            str(tmp_path_factory.mktemp("workspace")),
            "-session-directory",
            str(tmp_path_factory.mktemp("sessions")),
        ],
        environment(
            {
                LLM_PROVIDER_ENVIRONMENT: "custom",
                LLM_API_KEY_ENVIRONMENT: generic_key,
                "CUSTOM_CREDENTIAL": provider_key,
                "CUSTOM_API_KEY": "must-not-use",
            }
        ),
        reader('{"prompt":"hello"}'),
        Config(name="unreal-agent-runner", parse_request=parse_test_request, providers=providers),
    )
    if want_error:
        assert code == 1, stderr
        assert not created, stderr
        assert key_environment in stderr, stderr
    else:
        assert code == 0, stderr
        assert created, stderr


async def test_run_main_uses_llm_configuration_from_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    def respond(request: LLMRequest) -> Response:
        if request.Model.ID != "environment-model":
            raise ValueError(f"model = {request.Model.ID!r}")
        return Response(ID="response-1", Stop=StopReason.COMPLETE)

    client = FakeClient(respond)
    selected = False

    def default_client(_: str, __: str, ___: int, ____: object) -> Client:
        raise ValueError("default provider selected")

    def router_client(api_key: str, base_url: str, max_attempts: int, _: object) -> Client:
        nonlocal selected
        if (api_key, base_url, max_attempts) != ("custom-secret", "https://custom.example/v1", 2):
            raise ValueError("unexpected OpenRouter configuration")
        selected = True
        return client

    providers = [
        Provider(name="openai", api_key_environment="OPENAI_API_KEY", new_client=default_client),
        Provider(
            name="openrouter",
            base_url="https://default.example/v1",
            default_model="router-model",
            api_key_environment="OPENROUTER_API_KEY",
            new_client=router_client,
        ),
    ]
    code, _, stderr = await call_run_main(
        [
            "-workspace",
            str(tmp_path_factory.mktemp("workspace")),
            "-session-directory",
            str(tmp_path_factory.mktemp("sessions")),
        ],
        environment(
            {
                LLM_PROVIDER_ENVIRONMENT: "openrouter",
                LLM_API_KEY_ENVIRONMENT: "custom-secret",
                "OPENROUTER_API_KEY": "provider-secret",
                LLM_BASE_URL_ENVIRONMENT: "https://custom.example/v1",
                LLM_MODEL_ENVIRONMENT: "environment-model",
                LLM_MAX_ATTEMPTS_ENVIRONMENT: "2",
            }
        ),
        reader('{"messages":[{"role":"user","content":"hello"}]}'),
        Config(name="unreal-agent-runner", parse_request=parse_test_request, providers=providers),
    )
    assert code == 0, stderr
    assert selected, stderr


@pytest.mark.parametrize(
    ("arguments", "want", "truncated"),
    [
        pytest.param('{"command":"printf hello"}', "hello", False, id="default limit"),
        pytest.param(
            '{"command":"printf hello","max_output_length":3}',
            "h...2 bytes truncated; complete output in {path}...lo",
            True,
            id="requested limit",
        ),
    ],
)
async def test_run_main_executes_bash_tool_to_completion(
    arguments: str,
    want: str,
    truncated: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    monkeypatch.setenv(LLM_API_KEY_ENVIRONMENT, "secret")
    client = FakeClient()

    def respond(request: LLMRequest) -> Response:
        client.calls += 1
        if client.calls == 1:
            return Response(
                ID="response-1",
                Stop=StopReason.COMPLETE,
                Output=[
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(CallID="call-1", Name="Bash", Arguments=arguments),
                    )
                ],
            )
        found = False
        for item in request.Input:
            result = item.Data
            if not isinstance(result, ToolResult):
                continue
            value = result.Output[0].Value
            if result.CallID != "call-1" or value == TOOL_CALL_RUNNING_PAYLOAD:
                continue
            if truncated:
                prefix, _, suffix = want.partition("{path}")
                if not value.startswith(prefix) or not value.endswith(suffix):
                    raise ValueError(f"unexpected truncated Bash output: {value}")
                path = value.removeprefix(prefix).removesuffix(suffix)
                if not os.path.isabs(path):
                    raise ValueError(f"Bash capture path is not absolute: {path!r}")
                if Path(path).read_bytes() != b"hello":
                    raise ValueError("unexpected Bash capture")
            elif value != want:
                raise ValueError(f"unexpected Bash output: {value}")
            found = True
        if not found:
            raise ValueError("completed Bash result is missing")
        return message_response("finished", "response-2")

    client.respond_function = respond
    async with asyncio.timeout(10):
        code, stdout, stderr = await call_run_main(
            [
                "-workspace",
                str(tmp_path_factory.mktemp("workspace")),
                "-session-directory",
                str(tmp_path_factory.mktemp("sessions")),
            ],
            environment({LLM_API_KEY_ENVIRONMENT: "secret", "SHELL": "/bin/sh"}),
            reader('{"messages":[{"role":"user","content":"run it"}],"model":"gpt-test"}'),
            runner_config(client),
        )
    assert code == 0, (stderr, stdout)
    assert "PATH=" not in stdout, "persisted session items contain the process environment"
    assert "secret" not in stdout, "persisted session items contain the process environment"
    assert item_sequence(stdout) in (
        "input.control input.external input.control turn model_response tool_call_status "
        "tool_call_status turn model_response",
        "input.control input.external turn input.control model_response tool_call_status "
        "tool_call_status turn model_response",
        "input.control input.external turn model_response tool_call_status input.control "
        "tool_call_status turn model_response",
        "input.control input.external turn model_response tool_call_status tool_call_status "
        "turn input.control model_response",
        "input.control input.external turn model_response tool_call_status tool_call_status "
        "turn model_response input.control",
    )


async def test_run_main_emits_validation_error() -> None:
    code, stdout, stderr = await call_run_main(
        [],
        lambda _: "secret",
        reader('{"messages":[],"thinking_level":"maximum"}'),
        runner_config(FakeClient()),
    )
    assert code == 1
    assert event_types(stdout) == ["error"]
    assert "thinking_level must be one of" in stderr


def test_validate_request_rejects_non_uuid_message_id() -> None:
    with pytest.raises(RunnerError) as raised:
        validate_request(
            Request(messages=[RequestMessage(content="hello", message_id="message-1")])
        )
    assert str(raised.value) == "messages[0].message_id must be a UUID"


def test_load_dotenv_uses_scoped_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_RUNNER_EXISTING", "outer")
    monkeypatch.setenv("SANDBOX_EGRESS_PROXY", "outer-proxy")
    monkeypatch.setenv("HTTPS_PROXY", "outer-https")
    path = tmp_path / ".env"
    path.write_text("HARNESS_RUNNER_EXISTING=inner\nSANDBOX_EGRESS_PROXY=https://proxy.example\n")
    scope = load_dotenv(str(path))
    assert os.environ["HARNESS_RUNNER_EXISTING"] == "outer"
    assert os.environ["SANDBOX_EGRESS_PROXY"] == "https://proxy.example"
    assert os.environ["HTTPS_PROXY"] == "https://proxy.example"
    scope.close()
    assert os.environ["SANDBOX_EGRESS_PROXY"] == "outer-proxy"
    assert os.environ["HTTPS_PROXY"] == "outer-https"


def test_reasoning_effort_maps_every_thinking_level() -> None:
    cases = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
        "xhigh": ReasoningEffort.XHIGH,
        "max": ReasoningEffort.MAX,
        "": ReasoningEffort.HIGH,
    }
    for level, want in cases.items():
        assert reasoning_effort(level) == want, level
    for level in ("xhigh", "max"):
        validate_request(Request(prompt="", thinking_level=level))


@pytest.mark.parametrize(
    "args",
    [
        pytest.param([], id="unset"),
        pytest.param(["-log-directory", ""], id="empty"),
        pytest.param(["-log-directory", "  "], id="whitespace"),
    ],
)
async def test_run_main_without_log_directory_writes_only_stdout(
    args: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = tmp_path_factory.mktemp("workspace")
    monkeypatch.chdir(workspace)
    client = FakeClient(lambda _: message_response("done"))
    code, stdout, stderr = await call_run_main(
        [
            "-workspace",
            str(workspace),
            "-session-directory",
            str(tmp_path_factory.mktemp("sessions")),
            "-p",
            "hello",
            *args,
        ],
        environment({LLM_API_KEY_ENVIRONMENT: "secret"}),
        reader(""),
        runner_config(client),
    )
    assert code == 0, stderr
    assert ItemKind.MODEL_RESPONSE in item_kinds(stdout), "stdout is missing the model response"
    assert os.listdir(workspace) == [], "want no log files or directories"


async def test_run_main_uses_default_session_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = tmp_path_factory.mktemp("workspace")
    state_home = tmp_path_factory.mktemp("state")
    monkeypatch.chdir(workspace)
    started = False

    def respond(_: LLMRequest) -> Response:
        nonlocal started
        if started:
            return Response()
        started = True
        return Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(
                        CallID="call-1", Name="Bash", Arguments='{"command":"printf hello"}'
                    ),
                )
            ]
        )

    async with asyncio.timeout(10):
        code, _, stderr = await call_run_main(
            ["-workspace", str(workspace), "-p", "hello"],
            environment(
                {
                    LLM_API_KEY_ENVIRONMENT: "secret",
                    "XDG_STATE_HOME": str(state_home),
                    "SHELL": "/bin/sh",
                }
            ),
            reader(""),
            runner_config(FakeClient(respond)),
        )
    assert code == 0, stderr
    sessions = state_home / "unreal-agent" / "sessions"
    for pattern, want_content in {
        "*.session.jsonl": "model_response",
        os.path.join("operations", "*", "*", "out"): "hello",
    }.items():
        paths = glob.glob(str(sessions / pattern))
        assert len(paths) == 1, pattern
        assert want_content in Path(paths[0]).read_text()
    assert os.listdir(workspace) == [], "want no state in workspace"


@pytest.mark.parametrize(
    ("configured", "state_home", "user_home", "want"),
    [
        pytest.param(
            "", "/state", "/home/user", "/state/unreal-agent/sessions", id="XDG state home"
        ),
        pytest.param(
            "",
            "",
            "/home/user",
            "/home/user/.local/state/unreal-agent/sessions",
            id="home fallback",
        ),
        pytest.param(
            "",
            "",
            "",
            "/process/home/.local/state/unreal-agent/sessions",
            id="process home fallback",
        ),
        pytest.param(
            "",
            "relative/state",
            "/home/user",
            "/home/user/.local/state/unreal-agent/sessions",
            id="relative XDG ignored",
        ),
        pytest.param(" /sessions ", "/state", "/home/user", "/sessions", id="absolute override"),
        pytest.param("sessions", "/state", "", None, id="relative override"),
        pytest.param("//sessions/", "", "", "/sessions", id="override cleaned like go"),
        pytest.param("//a/../sessions", "", "", "/sessions", id="override dot-dot cleaned"),
    ],
)
def test_resolve_session_directory(
    configured: str,
    state_home: str,
    user_home: str,
    want: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", "/process/home")
    monkeypatch.chdir(tmp_path)
    got = resolve_session_directory(
        configured, environment({"XDG_STATE_HOME": state_home, "HOME": user_home})
    )
    assert got == (want if want is not None else os.path.join(os.getcwd(), "sessions"))


@pytest.mark.parametrize("user_home", ["", "relative/home"])
def test_resolve_session_directory_rejects_missing_state_location(
    user_home: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "")
    with pytest.raises(RunnerError, match="-session-directory"):
        resolve_session_directory(
            "", environment({"XDG_STATE_HOME": "relative/state", "HOME": user_home})
        )


def test_resolve_workspace_cleans_like_go(tmp_path: Path) -> None:
    from unreal_agent.runner.run import _resolve_workspace

    assert _resolve_workspace(" /" + str(tmp_path) + "//x/.. ") == str(tmp_path)
