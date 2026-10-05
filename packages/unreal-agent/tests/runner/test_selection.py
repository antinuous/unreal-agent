# ruff: noqa: ASYNC240
"""Ports of cmd/internal/agentrunner/selection_test.go."""

import os
from pathlib import Path

import pytest
from unreal_agent import gojson
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Response, StopReason, ToolCall, ToolResult
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.sessionstore import ToolCallStatus

from tests.runner._helpers import (
    FakeClient,
    call_run_main,
    decode_lines,
    environment,
    message_response,
    reader,
    runner_config,
)

_SKILL = "---\nname: review\ndescription: Review code.\n---\n"


@pytest.mark.parametrize(
    ("name", "skill", "disallowed", "want"),
    [
        ("no integrations", "", None, ["Bash", "ViewImage"]),
        ("valid skill", _SKILL, None, ["Bash", "ViewImage", "SkillUse"]),
        ("disallowed SkillUse", _SKILL, ["SkillUse"], ["Bash", "ViewImage"]),
        ("disallowed ViewImage", "", ["ViewImage"], ["Bash"]),
        ("malformed skill", "invalid", None, ["Bash", "ViewImage"]),
        ("incomplete skill", "---\nname: review\n---\n", None, ["Bash", "ViewImage"]),
        ("disallowed tools", _SKILL, ["Bash", "ViewImage", "SkillUse"], []),
        ("empty selection", "", ["Bash", "ViewImage"], []),
    ],
)
async def test_run_selects_tools_from_startup_configuration(
    name: str,
    skill: str,
    disallowed: list[str] | None,
    want: list[str],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    workspace = tmp_path_factory.mktemp("workspace")
    skill_path = workspace / ".harness" / "skills" / "review" / "SKILL.md"
    if skill:
        skill_path.parent.mkdir(parents=True, mode=0o700)
        skill_path.write_text(skill)
    encoded = gojson.encode({"prompt": "hello", "disallowed_tools": disallowed})
    requests: list[LLMRequest] = []

    def respond(request: LLMRequest) -> Response:
        requests.append(request)
        return message_response("done", "response-1")

    code, stdout, stderr = await call_run_main(
        ["-workspace", str(workspace), "-session-directory", str(tmp_path_factory.mktemp("s"))],
        environment({"OPENAI_API_KEY": "secret"}),
        reader(encoded),
        runner_config(FakeClient(respond)),
    )
    assert code == 0, (stderr, stdout)
    request = requests[0]
    assert [tool.Name for tool in request.Tools] == want
    system = request.Input[0].Data
    assert isinstance(system, Message)
    want_skills = "SkillUse" in want
    for text in [
        "Use SkillUse",
        "<available_skills>",
        "<name>review</name>",
        "Review code.",
        str(skill_path),
    ]:
        assert (text in system.Text) == want_skills, text
    if name in ("malformed skill", "incomplete skill"):
        assert "skill error>" in stderr, "invalid skill was not reported"


@pytest.mark.parametrize("name", ["Bash", "unknown-tool"])
async def test_run_continues_after_unavailable_tool_call(
    name: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = tmp_path
    requests: list[LLMRequest] = []

    def respond(request: LLMRequest) -> Response:
        if len(requests) >= 2:
            raise ValueError("unexpected extra model turn")
        requests.append(request)
        if len(requests) == 1:
            return Response(
                ID="response-1",
                Stop=StopReason.COMPLETE,
                Output=[
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(
                            CallID="unavailable-call",
                            Name=name,
                            Arguments='{"command":"touch executed"}',
                        ),
                    )
                ],
            )
        return message_response("done", "response-2")

    code, stdout, stderr = await call_run_main(
        ["-workspace", str(workspace), "-session-directory", str(tmp_path_factory.mktemp("s"))],
        environment({"OPENAI_API_KEY": "secret"}),
        reader('{"prompt":"hello","disallowed_tools":["Bash"]}'),
        runner_config(FakeClient(respond)),
    )
    assert code == 0, (stderr, stdout)
    assert len(requests) == 2, "model requests, want initial and corrective turns"
    want_error = f'tool "{name}" is not available'
    corrective = requests[1]
    assert any(
        isinstance(item.Data, ToolResult)
        and item.Data.CallID == "unavailable-call"
        and item.Data.Output[0].Value == want_error
        for item in corrective.Input
    ), f"corrective request omitted tool error: {corrective.Input!r}"
    statuses = [item.Data for item in decode_lines(stdout) if isinstance(item.Data, ToolCallStatus)]
    assert len(statuses) == 1, "recorded statuses, want one"
    status = statuses[0]
    assert status.CallID == "unavailable-call"
    assert status.Status.Error == want_error
    assert not status.Status.WaitingFor
    assert not status.Operations
    assert not os.path.exists(workspace / "executed"), "excluded Bash call executed"
