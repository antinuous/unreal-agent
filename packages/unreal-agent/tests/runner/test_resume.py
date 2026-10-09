import asyncio
import io
from typing import Any

import pytest
from unreal_agent import gojson
from unreal_agent.inbox import ControlMode, Input, InputKind, Settings
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, ReasoningEffort, Response, ToolCall, ToolResult
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.runner import Client
from unreal_agent.runner.run import LLM_API_KEY_ENVIRONMENT
from unreal_agent.sessionstore import BEFORE_FIRST, ItemKind
from unreal_agent.sessionstore.localfile import Store

from tests.runner._helpers import (
    SECRET_ENVIRONMENT,
    FailingItemWriter,
    FakeClient,
    call_run,
    call_run_main,
    decode_lines,
    environment,
    input_ids,
    is_error,
    item_kinds,
    message_response,
    reader,
    runner_config,
)


@pytest.mark.parametrize(
    ("name", "kind", "input_kind"),
    [
        ("settings", ItemKind.INPUT, InputKind.CONTROL),
        ("message", ItemKind.INPUT, InputKind.EXTERNAL),
        ("response", ItemKind.MODEL_RESPONSE, InputKind.NONE),
    ],
)
async def test_run_resumes_after_output_failure(
    name: str, kind: ItemKind, input_kind: InputKind, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = str(tmp_path_factory.mktemp("workspace"))
    sessions = str(tmp_path_factory.mktemp("sessions"))
    request = (
        '{"session_id":"output-failure","messages":[{"role":"user","content":"hello",'
        '"message_id":"69621f8d-4f4d-49a5-8f7d-3b24fd855c01"}]}'
    )
    requests: list[asyncio.Task[Any]] = []

    def respond(_: LLMRequest) -> Response:
        task = asyncio.current_task()
        assert task is not None
        requests.append(task)
        return message_response("Hello.")

    client = FakeClient(respond)

    async def run(output: Any) -> BaseException | None:
        return await call_run(
            ["-workspace", workspace, "-session-directory", sessions],
            SECRET_ENVIRONMENT,
            reader(request),
            output,
            runner_config(client),
        )

    async with asyncio.timeout(10):
        want = OSError("output unavailable")
        error = await run(FailingItemWriter(kind, want, input_kind))
        assert is_error(error, want), f"Run error = {error!r}, want original output error"
        store = Store(sessions)
        page = await store.items("output-failure", BEFORE_FIRST, 100)
        assert any(item.Kind == kind for item in page.items), (
            "item whose output failed was not committed"
        )
        output = io.BytesIO()
        assert await run(output) is None
        want_inputs = 1 if name == "settings" else 0
        assert len(input_ids(output.getvalue().decode())) == want_inputs, (
            "retry persisted the wrong number of user inputs"
        )
        assert len(requests) == 1, "model calls across failure and retry, want 1"
        assert requests.pop().done(), "model request outlived its run"
        assert await run(io.BytesIO()) is None
        assert not requests, "settled retry started another model request"


def system_text(request: LLMRequest) -> str:
    first = request.Input[0].Data
    return first.Text if isinstance(first, Message) else ""


@pytest.mark.parametrize("with_tool", [False, True], ids=["first-response", "tool-result"])
async def test_run_main_resumes_interrupted_delivery_with_duplicate_input(
    with_tool: bool, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = str(tmp_path_factory.mktemp("workspace"))
    sessions = str(tmp_path_factory.mktemp("sessions"))
    model, effort = "initial-model", ReasoningEffort.LOW
    system_prompt = "initial prompt"
    message_id = "69621f8d-4f4d-49a5-8f7d-3b24fd855c01"

    async def run(client: Client) -> tuple[int, str, str]:
        request = gojson.encode(
            {
                "model": model,
                "thinking_level": str(effort),
                "system_prompt": system_prompt,
                "session_id": "resume-delivery",
                "messages": [{"role": "user", "content": "run it", "message_id": message_id}],
            }
        )
        return await call_run_main(
            ["-workspace", workspace, "-session-directory", sessions],
            environment({LLM_API_KEY_ENVIRONMENT: "secret", "SHELL": "/bin/sh"}),
            reader(request),
            runner_config(client),
        )

    def has_result(request: LLMRequest) -> bool:
        return any(
            isinstance(item.Data, ToolResult)
            and item.Data.CallID == "call-1"
            and item.Data.Output[0].Value == "hello"
            for item in request.Input
        )

    interrupted = FakeClient()

    def interrupted_respond(request: LLMRequest) -> Response:
        interrupted.calls += 1
        if model != request.Model.ID or request.Model.ReasoningEffort != effort:
            raise ValueError(f"initial settings = {request.Model!r}")
        if not system_text(request).endswith("\n\n" + system_prompt):
            raise ValueError("initial request did not use the caller's system prompt")
        if with_tool and interrupted.calls == 1:
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
        if with_tool and not has_result(request):
            raise ValueError("missing completed result")
        raise ValueError("interrupted delivery")

    interrupted.respond_function = interrupted_respond
    async with asyncio.timeout(10):
        code, _, stderr = await run(interrupted)
        want_calls = 2 if with_tool else 1
        assert code == 1, stderr
        assert interrupted.calls == want_calls, stderr
        assert "interrupted delivery" in stderr

        model, effort = "resumed-model", ReasoningEffort.MAX
        system_prompt = "resumed prompt"
        resumed = FakeClient()

        def resumed_respond(request: LLMRequest) -> Response:
            resumed.calls += 1
            if model != request.Model.ID or request.Model.ReasoningEffort != effort:
                raise ValueError(f"recovered request settings = {request.Model!r}")
            if not system_text(request).endswith("\n\n" + system_prompt):
                raise ValueError("recovered request did not use the caller's system prompt")
            if with_tool and not has_result(request):
                raise ValueError("missing restored result")
            return message_response("Done.")

        resumed.respond_function = resumed_respond
        code, stdout, stderr = await run(resumed)
        assert code == 0, stderr
        assert resumed.calls == 1, stderr
        assert input_ids(stdout) == [], "duplicate input was recorded again"
        settings = 0
        for item in decode_lines(stdout):
            if isinstance(item.Data, Input) and item.Data.Kind == InputKind.CONTROL:
                control = item.Data.decode_control_message()
                if control.Mode == ControlMode.UPDATE_SETTINGS:
                    settings += 1
                    assert control.Parameters == Settings(
                        Model=model, CompactionThreshold=0, ReasoningEffort=effort
                    )
        assert settings == 1, "recorded settings controls, want 1"
        assert item_kinds(stdout).count(ItemKind.MODEL_RESPONSE) == 1

        resumed.calls = 0
        code, _, stderr = await run(resumed)
        assert code == 0, stderr
        assert resumed.calls == 0, stderr

        message_id = "49621f8d-4f4d-49a5-8f7d-3b24fd855c02"

        def follow_up(request: LLMRequest) -> Response:
            resumed.calls += 1
            if model != request.Model.ID or request.Model.ReasoningEffort != effort:
                raise ValueError(f"following request settings = {request.Model!r}")
            return Response()

        resumed.respond_function = follow_up
        code, _, stderr = await run(resumed)
        assert code == 0, stderr
        assert resumed.calls == 1, stderr


async def test_run_resumes_session_without_recorded_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    workspace = str(tmp_path_factory.mktemp("workspace"))
    sessions = str(tmp_path_factory.mktemp("sessions"))
    store = Store(sessions)
    await store.create("legacy")
    await store.append_input(
        "legacy",
        Input(
            ID="69621f8d-4f4d-49a5-8f7d-3b24fd855c01",
            Kind=InputKind.EXTERNAL,
            Payload=gojson.Raw(b'"hello"'),
        ),
    )
    client = FakeClient()

    def respond(request: LLMRequest) -> Response:
        client.calls += 1
        if (
            request.Model.ID != "selected-model"
            or request.Model.ReasoningEffort != ReasoningEffort.MEDIUM
        ):
            raise ValueError(f"legacy session request settings = {request.Model!r}")
        return Response()

    client.respond_function = respond
    output = io.BytesIO()
    error = await call_run(
        ["-workspace", workspace, "-session-directory", sessions],
        SECRET_ENVIRONMENT,
        reader("""{
            "session_id":"legacy", "model":"selected-model", "thinking_level":"medium",
            "messages":[{"role":"user","content":"hello",
                "message_id":"69621f8d-4f4d-49a5-8f7d-3b24fd855c01"}]
        }"""),
        output,
        runner_config(client),
    )
    assert error is None, error
    assert client.calls == 1, error
    assert input_ids(output.getvalue().decode()) == [], "legacy input was recorded again"
