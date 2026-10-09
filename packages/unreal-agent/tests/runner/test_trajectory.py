"""Feeds Python runner output through the Harbor trajectory parser.

The parser is loaded from `benchmarks/harbor/src/harness_harbor/trajectory.py`
with stand-ins for the `harbor` models, which the workspace does not install.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from unreal_agent.gojson import Raw
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Response,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.runner.run import LLM_API_KEY_ENVIRONMENT

from tests.runner._helpers import FakeClient, call_run_main, environment, reader, runner_config

TRAJECTORY = Path(__file__).parents[4] / "benchmarks/harbor/src/harness_harbor/trajectory.py"


class _Model:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Step(_Model):
    observation: Any = None


@pytest.fixture
def trajectory(monkeypatch: pytest.MonkeyPatch) -> Any:
    models = types.ModuleType("harbor.models.trajectories")
    classes: dict[str, type] = {
        name: type(name, (_Model,), {})
        for name in [
            "Agent",
            "ContentPart",
            "FinalMetrics",
            "ImageSource",
            "Metrics",
            "Observation",
            "ObservationResult",
            "ToolCall",
            "Trajectory",
        ]
    }
    classes["Step"] = _Step
    for name, value in classes.items():
        setattr(models, name, value)
    monkeypatch.setitem(sys.modules, "harbor", types.ModuleType("harbor"))
    monkeypatch.setitem(sys.modules, "harbor.models", types.ModuleType("harbor.models"))
    monkeypatch.setitem(sys.modules, "harbor.models.trajectories", models)
    spec = importlib.util.spec_from_file_location("harness_harbor_trajectory", TRAJECTORY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_runner_output_converts_to_trajectory(
    trajectory: Any, tmp_path_factory: pytest.TempPathFactory
) -> None:
    turns = 0

    def respond(_: LLMRequest) -> Response:
        nonlocal turns
        turns += 1
        usage = Usage(
            InputTokens=10,
            CachedInputTokens=4,
            CacheWriteInputTokens=1,
            OutputTokens=3,
            ReasoningTokens=2,
        )
        if turns == 1:
            return Response(
                ID="response-1",
                Stop=StopReason.COMPLETE,
                Usage=usage,
                Output=[
                    LLMItem(
                        Type=ItemType.PROVIDER,
                        Data=ProviderItem(
                            Type="reasoning",
                            Raw=Raw(b'{"type":"reasoning"}'),
                            Display=ProviderDisplay(
                                Kind=ProviderDisplayKind.REASONING, Text="plan the command"
                            ),
                        ),
                    ),
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(
                            CallID="call-1", Name="Bash", Arguments='{"command":"printf hello"}'
                        ),
                    ),
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(CallID="call-2", Name="Missing", Arguments="{}"),
                    ),
                ],
            )
        # The tool error and the Bash result may arrive in one turn or two.
        return Response(
            ID=f"response-{turns}",
            Stop=StopReason.COMPLETE,
            Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="done"))],
        )

    async with asyncio.timeout(10):
        code, stdout, stderr = await call_run_main(
            [
                "-workspace",
                str(tmp_path_factory.mktemp("workspace")),
                "-session-directory",
                str(tmp_path_factory.mktemp("sessions")),
                "-p",
                "run it",
            ],
            environment({LLM_API_KEY_ENVIRONMENT: "secret", "SHELL": "/bin/sh"}),
            reader(""),
            runner_config(FakeClient(respond)),
        )
    assert code == 0, stderr
    agent = trajectory.Agent(name="unreal-agent", version="test", model_name="gpt-default")
    result = trajectory.convert(stdout.splitlines(), agent, "session")
    sources = [step.source for step in result.steps]
    assert sources == ["user"] + ["agent"] * turns
    assert result.steps[0].message == "run it"
    first = result.steps[1]
    assert first.reasoning_content == "plan the command"
    assert [call.tool_call_id for call in first.tool_calls] == ["call-1", "call-2"]
    assert first.tool_calls[0].arguments == {"command": "printf hello"}
    contents = {item.source_call_id: item.content for item in first.observation.results}
    assert contents["call-1"] == "hello"
    assert contents["call-2"] == 'tool "Missing" is not available'
    assert result.steps[-1].message == "done"
    metrics = result.final_metrics
    assert metrics.total_prompt_tokens == 10
    assert metrics.total_completion_tokens == 3
    assert metrics.total_cached_tokens == 4
    assert metrics.total_steps == 1 + turns
    assert metrics.extra == {"reasoning_tokens": 2, "cache_write_tokens": 1}
    assert result.extra == {"runner_errors": []}


async def test_runner_error_event_converts_to_trajectory(trajectory: Any) -> None:
    code, stdout, _ = await call_run_main(
        [], lambda _: "", reader('{"messages":[]}'), runner_config(FakeClient())
    )
    assert code == 1
    agent = trajectory.Agent(name="unreal-agent", version="test", model_name="gpt-default")
    result = trajectory.convert(stdout.splitlines(), agent, "session")
    assert result.steps == []
    assert result.extra == {"runner_errors": ["messages must not be empty"]}
