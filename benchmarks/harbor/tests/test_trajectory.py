import json

import pytest
from harbor.models.trajectories import Agent, Trajectory
from harness_harbor.trajectory import RUNNING, bash_result, convert


def record(sequence, kind, data):
    return json.dumps(
        {
            "Sequence": sequence,
            "Kind": kind,
            "Data": data,
            "RecordedAt": "2026-09-10T10:00:00Z",
        }
    )


def response(turn, output):
    return {
        "TurnID": turn,
        "Response": {
            "ID": turn,
            "Stop": "complete",
            "Output": output,
            "Usage": {
                "InputTokens": 10,
                "CachedInputTokens": 4,
                "CacheWriteInputTokens": 0,
                "OutputTokens": 3,
                "ReasoningTokens": 1,
            },
        },
    }


def status(state="completed", stdout="test", **fields):
    return {
        "CallID": "call-1",
        "Status": {"WaitingFor": ["op-1"]},
        "Operations": [
            {
                "ID": "op-1",
                "Type": "shell",
                "Status": state,
                "State": {
                    "Result": {"Out": stdout, "Err": "", "ExitCode": 0},
                    **fields,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("reasoning", {"Summary": ["first", "second"]}, "first\n\nsecond"),
        (
            "provider",
            {"Display": {"Kind": "reasoning", "Text": "first\n\nsecond"}},
            "first\n\nsecond",
        ),
        (
            "provider",
            {"Raw": {"type": "reasoning", "summary": [{"text": "raw only"}]}},
            None,
        ),
        ("provider", {"Display": None}, None),
        ("provider", {"Display": {"Kind": "reasoning", "Text": ""}}, None),
        ("provider", {"Display": {"Kind": "tool", "Text": "Searching"}}, None),
    ],
)
def test_provider_and_legacy_reasoning_preserve_trajectory_and_usage(kind, value, expected):
    output = [
        {"Type": kind, "Data": value},
        {"Type": "message", "Data": {"Text": "done"}},
    ]
    trajectory = convert(
        [record(1, "model_response", response("turn-1", output))],
        Agent(name="unreal-agent", version="test"),
        "session",
    )
    step = trajectory.steps[0]
    assert step.message == "done"
    assert step.reasoning_content == expected
    assert step.metrics is not None
    assert step.metrics.extra is not None
    assert step.metrics.extra["reasoning_tokens"] == 1
    metrics = trajectory.final_metrics
    assert metrics is not None
    assert metrics.total_prompt_tokens == 10
    assert metrics.total_completion_tokens == 3
    assert metrics.total_cached_tokens == 4
    assert metrics.extra is not None
    assert metrics.extra["reasoning_tokens"] == 1
    assert Trajectory.model_validate(trajectory.to_json_dict()) == trajectory


@pytest.mark.parametrize(
    "text", ["test", "1234", "aGVsbG8=", "😃\n", "", "{not JSON", '  "hi"\n\n']
)
def test_plain_text_is_not_base64_decoded(text):
    assert bash_result(status(stdout=text)) == (text or "(no output)", False)


def test_preserves_inline_truncation_without_duplicate_metadata():
    stdout = "head...92 bytes truncated; complete output in /out...tail"
    stderr = "head...92 bytes truncated; complete output in /err...tail"
    data = status(
        Result={"Out": stdout, "Err": stderr, "ExitCode": 7},
        OutTruncated=True,
        ErrTruncated=True,
        OutPath="/out",
        ErrPath="/err",
    )
    assert bash_result(data) == (f"{stdout}\nStderr:\n{stderr}\nExit code: 7", False)


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit_code", "expected"),
    [
        ("", "warning\n", 0, "Stderr:\nwarning\n"),
        ("out\n", "err\n", 0, "out\n\nStderr:\nerr\n"),
        ("out", "err", 7, "out\nStderr:\nerr\nExit code: 7"),
        ("", "failed\n", 7, "Stderr:\nfailed\n\nExit code: 7"),
        ("", "", 1, "Exit code: 1"),
    ],
)
def test_labels_stderr_and_nonzero_exit_codes(stdout, stderr, exit_code, expected):
    data = status(Result={"Out": stdout, "Err": stderr, "ExitCode": exit_code})
    assert bash_result(data) == (expected, False)


@pytest.mark.parametrize("state", ["ready", "awaiting", "canceling"])
def test_running_operations(state):
    assert bash_result(status(state=state, Result=None)) == (RUNNING, True)


@pytest.mark.parametrize("state", ["failed", "canceled"])
@pytest.mark.parametrize("error", ["", "shell stopped", "err...100 bytes truncated...end"])
def test_failed_operations(state, error):
    data = status(state=state, Result=None, TerminalError=error)
    assert bash_result(data) == (f"Error: {error or 'shell operation ' + state}", False)


@pytest.mark.parametrize("state", ["failed", "canceled"])
@pytest.mark.parametrize(
    ("fields", "prefix"),
    [
        ({}, ""),
        ({"OutPath": "/out"}, "Stdout capture: /out\n"),
        ({"ErrPath": "/err"}, "Stderr capture: /err\n"),
        (
            {"OutPath": "/out", "ErrPath": "/err"},
            "Stdout capture: /out\nStderr capture: /err\n",
        ),
    ],
)
def test_failure_and_cancellation_preserve_available_captures(state, fields, prefix):
    data = status(state=state, Result=None, **fields)
    assert bash_result(data) == (prefix + f"Error: shell operation {state}", False)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"Status": {}}, "has 0 operations, want 1"),
        ({"Status": {"WaitingFor": ["op-1", "op-2"]}}, "has 2 operations, want 1"),
        (
            {**status(), "Status": {"Error": "bad arguments"}},
            "both a validation error and operations",
        ),
        (status(Result=None), "Completed shell operation has no result"),
        (status(state="unknown"), "Unsupported operation status"),
        (
            status(Result={"Out": None, "Err": "", "ExitCode": 0}),
            "Expected current runner text output",
        ),
    ],
)
def test_rejects_invalid_shell_results(data, message):
    with pytest.raises(ValueError, match=message):
        bash_result(data)


def test_truncated_validation_error_is_plain_text():
    data = {
        "Status": {
            "Error": "bad...100 bytes truncated...JSON",
            "ErrorTruncated": True,
        }
    }
    assert bash_result(data) == ("Error: bad...100 bytes truncated...JSON", False)


def test_internal_control_and_delayed_observations():
    tool_call = {
        "Type": "tool_call",
        "Data": {
            "CallID": "call-1",
            "Name": "Bash",
            "Arguments": '{"command":"printf test"}',
        },
    }
    lines = [
        record(1, "input", {"Kind": "external", "Payload": '"literal quotes"'}),
        record(2, "input", {"Kind": "control", "Payload": {"Mode": "when_idle"}}),
        record(3, "turn", {"ID": "turn-1"}),
        record(4, "model_response", response("turn-1", [tool_call])),
        record(5, "tool_call_status", status(state="ready")),
        record(6, "turn", {"ID": "turn-2"}),
        record(7, "model_response", response("turn-2", [])),
        record(8, "tool_call_status", status()),
        record(9, "turn", {"ID": "turn-3"}),
        record(10, "model_response", response("turn-3", [])),
    ]
    trajectory = convert(lines, Agent(name="unreal-agent", version="test"), "session")
    assert len(trajectory.steps) == 4
    assert trajectory.steps[0].message == '"literal quotes"'
    observation = trajectory.steps[1].observation
    assert observation is not None
    observations = observation.results
    first_extra, second_extra = observations[0].extra, observations[1].extra
    assert first_extra is not None
    assert second_extra is not None
    assert observations[0].content == RUNNING
    assert first_extra["running"] is True
    assert first_extra["available_before_turn"] == "turn-2"
    assert second_extra["available_before_turn"] == "turn-3"
    assert observations[1].content == "test"
    assert second_extra["running"] is False
    metrics = trajectory.final_metrics
    assert metrics is not None
    assert metrics.total_prompt_tokens == 30
    assert metrics.total_cached_tokens == 12
    assert metrics.total_completion_tokens == 9
    assert metrics.total_cost_usd is None
    Trajectory.model_validate(trajectory.to_json_dict())


@pytest.mark.parametrize("state", ["ready", "completed"])
def test_only_latest_pending_running_result_is_marked_available(state):
    calls = [
        {
            "Type": "tool_call",
            "Data": {
                "CallID": call_id,
                "Name": "Bash",
                "Arguments": '{"command":"printf test"}',
            },
        }
        for call_id in ("call-1", "call-2")
    ]
    other_status = {**status(state="ready"), "CallID": "call-2"}
    lines = [
        record(1, "model_response", response("turn-1", calls)),
        record(2, "tool_call_status", status(state="ready")),
        record(3, "tool_call_status", other_status),
        record(4, "input", {"Kind": "external", "Payload": "continue"}),
        record(5, "tool_call_status", status(state=state)),
        record(6, "turn", {"ID": "turn-2"}),
    ]
    trajectory = convert(lines, Agent(name="unreal-agent", version="test"), "session")
    observation = trajectory.steps[0].observation
    assert observation is not None
    observations = observation.results
    assert len(observations) == 3
    first_extra, second_extra, third_extra = (result.extra for result in observations)
    assert first_extra is not None
    assert second_extra is not None
    assert third_extra is not None
    assert observations[0].content == RUNNING
    assert "available_before_turn" not in first_extra
    assert observations[1].source_call_id == "call-2"
    assert observations[1].content == RUNNING
    assert first_extra["running"] is True
    assert second_extra["running"] is True
    assert second_extra["available_before_turn"] == "turn-2"
    content, running = bash_result(status(state=state))
    assert observations[2].content == content
    assert third_extra["running"] is running
    assert third_extra["available_before_turn"] == "turn-2"
    assert third_extra["sequence"] == 5


def test_completed_output_matching_running_payload_remains_available():
    call = {
        "Type": "tool_call",
        "Data": {"CallID": "call-1", "Name": "Bash", "Arguments": "{}"},
    }
    lines = [
        record(1, "model_response", response("turn-1", [call])),
        record(2, "tool_call_status", status(stdout=RUNNING)),
        record(3, "tool_call_status", status(state="ready")),
        record(4, "tool_call_status", status()),
        record(5, "turn", {"ID": "turn-2"}),
    ]
    trajectory = convert(lines, Agent(name="unreal-agent", version="test"), "session")
    observation = trajectory.steps[0].observation
    assert observation is not None
    observations = observation.results
    assert len(observations) == 3
    first_extra, second_extra, third_extra = (result.extra for result in observations)
    assert first_extra is not None
    assert second_extra is not None
    assert third_extra is not None
    assert observations[0].content == RUNNING
    assert first_extra["running"] is False
    assert first_extra["available_before_turn"] == "turn-2"
    assert second_extra["running"] is True
    assert "available_before_turn" not in second_extra
    assert observations[2].content == "test"
    assert third_extra["running"] is False
    assert third_extra["available_before_turn"] == "turn-2"
    assert Trajectory.model_validate(trajectory.to_json_dict()) == trajectory


@pytest.mark.parametrize("lines", [["{"], ['{"Kind":']])
def test_malformed_record_is_an_error_including_partial_final_line(lines):
    with pytest.raises(ValueError, match="line 1"):
        convert(lines, Agent(name="unreal-agent", version="test"), "session")


def test_validation_error_and_incomplete_run_keep_usage():
    lines = [
        record(
            1,
            "model_response",
            response(
                "turn-1",
                [
                    {
                        "Type": "tool_call",
                        "Data": {
                            "CallID": "call-1",
                            "Name": "Bash",
                            "Arguments": "{",
                        },
                    }
                ],
            ),
        ),
        record(
            2,
            "tool_call_status",
            {"CallID": "call-1", "Status": {"Error": "bad JSON"}},
        ),
        json.dumps({"type": "error", "message": "provider disconnected"}),
    ]
    trajectory = convert(lines, Agent(name="unreal-agent", version="test"), "session")
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 10
    assert trajectory.extra is not None
    assert trajectory.extra["runner_errors"] == ["provider disconnected"]
    assert trajectory.steps[0].observation is not None
    assert trajectory.steps[0].observation.results[0].content == "Error: bad JSON"
