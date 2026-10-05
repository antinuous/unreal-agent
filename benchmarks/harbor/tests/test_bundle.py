import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harness_harbor.agent import UnrealAgent
from harness_harbor.bundle import ARCHIVE, Bundle


def test_changed_archive_is_rejected_after_construction(make_bundle):
    path = make_bundle(data=b"original")
    bundle = Bundle.load(str(path))
    assert bundle.read_archive() == b"original"
    assert bundle.python_version == "3.14.8"
    agent = UnrealAgent(bundle=str(path), logs_dir=path, model_name="openai/test")
    assert agent.version() == "a" * 40
    (path / ARCHIVE).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="checksum"):
        bundle.read_archive()
    with pytest.raises(ValueError, match="checksum"):
        Bundle.load(str(path))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"revision": "A" * 40}, "revision"),
        ({"sha256": "0" * 63}, "sha256"),
        ({"os": "darwin"}, "Linux"),
        ({"arch": "riscv64"}, "Linux"),
        ({"python_version": "3.14"}, "python_version"),
    ],
)
def test_invalid_manifest_is_rejected(make_bundle, fields, message):
    with pytest.raises(ValueError, match=message):
        Bundle.load(str(make_bundle(**fields)))


def test_bad_trajectory_is_not_silently_accepted(make_bundle):
    path = make_bundle()
    agent = UnrealAgent(bundle=str(path), logs_dir=path, model_name="openai/test")
    (path / "runner.jsonl").write_text("{")
    with pytest.raises(ValueError, match="line 1"):
        agent.populate_context_post_run(AgentContext())
    assert not (path / "trajectory.json").exists()


def test_missing_runner_log_is_recorded_not_raised(make_bundle):
    path = make_bundle()
    agent = UnrealAgent(bundle=str(path), logs_dir=path, model_name="openai/test")
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.metadata is not None
    assert context.metadata["agent_logs_missing"]
    assert context.metadata["revision"] == "a" * 40
    assert context.n_input_tokens is None
    assert not (path / "trajectory.json").exists()


def test_metadata_records_bundle_provenance(make_bundle):
    path = make_bundle()
    agent = UnrealAgent(bundle=str(path), logs_dir=path, model_name="openai/test")
    record = {
        "Sequence": 1,
        "Kind": "input",
        "Data": {"Kind": "external", "Payload": "task"},
        "RecordedAt": "2026-09-10T10:00:00Z",
    }
    (path / "runner.jsonl").write_text(json.dumps(record) + "\n")
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.metadata is not None
    assert context.metadata["bundle_sha256"] == Bundle.load(str(path)).sha256
    assert context.metadata["arch"] == "amd64"
    assert context.metadata["python_version"] == "3.14.8"


@pytest.mark.parametrize(
    ("prefix", "key"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("fireworks_ai", "FIREWORKS_AI_API_KEY"),
    ],
)
def test_provider_keys_and_nested_model_paths(make_bundle, prefix, key):
    path = make_bundle()
    agent = UnrealAgent(
        bundle=str(path),
        logs_dir=path,
        model_name=f"{prefix}/organization/model",
        extra_env={key: "test-key"},
    )
    assert agent._model == "organization/model"
    assert agent.model_connection.api_key == "test-key"


@pytest.mark.parametrize("model", ["test", "unsupported/test", "openai/"])
def test_invalid_model_fails_before_installation(model):
    with pytest.raises(ValueError, match=r"(?i)model"):
        UnrealAgent(bundle="missing", logs_dir=Path(), model_name=model)


def test_invalid_reasoning_fails_before_installation():
    with pytest.raises(ValueError, match="thinking_level"):
        UnrealAgent(
            bundle="missing",
            logs_dir=Path(),
            model_name="openai/test",
            thinking_level="extreme",
        )


def test_task_mcp_servers_and_skills_are_recorded_not_rejected(make_bundle):
    path = make_bundle(revision="b" * 40)
    agent = UnrealAgent(
        bundle=str(path),
        logs_dir=path,
        model_name="openai/test",
        mcp_servers=[
            MCPServerConfig(name="playwright", transport="sse", url="http://mcp:3080/sse")
        ],
        skills_dir="/app/.agents/skills",
    )
    assert agent._not_offered == {
        "task_mcp_servers": ["playwright"],
        "task_skills_dir": "/app/.agents/skills",
    }
