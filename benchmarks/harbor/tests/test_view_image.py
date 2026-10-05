import base64
from typing import Any

import pytest
from harbor.models.trajectories import Agent, Trajectory
from harness_harbor.trajectory import RUNNING, convert, view_image_result
from tests.test_trajectory import record, response


def image_status(state="completed", **fields) -> dict[str, Any]:
    return {
        "CallID": "image-call",
        "Status": {"WaitingFor": ["image-operation"]},
        "Operations": [
            {
                "ID": "image-operation",
                "Type": "view_image",
                "Status": state,
                "State": {
                    "Result": {
                        "Content": base64.b64encode(b"returned image bytes").decode(),
                        "OriginalMIMEType": "image/png",
                        "EncodedMIMEType": "image/png",
                        "OriginalWidth": 4000,
                        "OriginalHeight": 2000,
                        "ScaleRatio": 1,
                        "Error": "",
                        **fields,
                    }
                },
            }
        ],
    }


def invalid_results():
    wrong_type = image_status()
    wrong_type["Operations"][0]["Type"] = "shell"
    no_result = image_status()
    no_result["Operations"][0]["State"]["Result"] = None
    return [
        {"Status": {}},
        {"Status": {"WaitingFor": ["one", "two"]}},
        {**image_status(), "Status": {"Error": "invalid"}},
        wrong_type,
        no_result,
        image_status("unknown"),
        image_status(Content=""),
        image_status(Content="not base64!"),
        image_status(EncodedMIMEType="image/gif"),
        image_status(ScaleRatio=0),
        image_status(ScaleRatio=2),
        image_status(Error="failed"),
    ]


@pytest.mark.parametrize(
    ("original", "encoded", "ratio", "text"),
    [
        ("image/png", "image/png", 1, None),
        ("image/jpeg", "image/jpeg", 1, None),
        ("image/bmp", "image/png", 1, "original MIME type: image/bmp"),
        (
            "image/png",
            "image/png",
            0.5,
            "original dimensions: 4000x2000; multiply coordinates by 2.00 to approximate original",
        ),
        (
            "image/tiff",
            "image/png",
            0.25,
            "original MIME type: image/tiff; original dimensions: 4000x2000; "
            "multiply coordinates by 4.00 to approximate original",
        ),
        (
            "image/png",
            "image/png",
            1 / 3.25,
            "original dimensions: 4000x2000; multiply coordinates by 3.25 to approximate original",
        ),
        (
            "image/png",
            "image/png",
            1 / 3.456,
            "original dimensions: 4000x2000; multiply coordinates by 3.46 to approximate original",
        ),
        (
            "image/png",
            "image/png",
            1 / 120,
            "original dimensions: 4000x2000; multiply coordinates by 120.00 "
            "to approximate original",
        ),
    ],
)
def test_exports_image_and_only_applicable_metadata(tmp_path, original, encoded, ratio, text):
    data = image_status(
        OriginalMIMEType=original,
        EncodedMIMEType=encoded,
        ScaleRatio=ratio,
    )
    parts, running = view_image_result(data, tmp_path)
    assert running is False
    assert isinstance(parts, list)
    assert parts[0].type == "image"
    source = parts[0].source
    assert source is not None
    assert source.media_type == encoded
    path = tmp_path / source.path
    assert path.parent == tmp_path / "images"
    assert path.read_bytes() == b"returned image bytes"
    extension = ".jpg" if encoded == "image/jpeg" else ".png"
    assert path.suffix == extension
    if text is None:
        assert len(parts) == 1
    else:
        assert len(parts) == 2
        assert parts[1].type == "text"
        assert parts[1].text == text
    assert view_image_result(data, tmp_path) == (parts, False)
    assert len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("state", ["ready", "awaiting", "canceling"])
def test_running_remains_text(state):
    assert view_image_result(image_status(state), None) == (RUNNING, True)


@pytest.mark.parametrize("state", ["failed", "canceled"])
def test_terminal_errors_remain_text(state):
    data = image_status(state, Error="exceeded MaxSize by 100 bytes")
    assert view_image_result(data, None) == (
        "Error: exceeded MaxSize by 100 bytes; "
        "original MIME type: image/png; original dimensions: 4000x2000",
        False,
    )
    data["Operations"][0]["State"]["Result"] = None
    assert view_image_result(data, None) == ("Error: view-image operation " + state, False)


def test_validation_and_header_errors_remain_text():
    assert view_image_result({"Status": {"Error": "invalid path"}}, None) == (
        "Error: invalid path",
        False,
    )
    assert view_image_result(
        image_status(
            "failed",
            OriginalMIMEType="",
            OriginalWidth=0,
            OriginalHeight=0,
            Error="decode image header; original dimensions unavailable",
        ),
        None,
    ) == ("Error: decode image header; original dimensions unavailable", False)


@pytest.mark.parametrize("data", invalid_results())
def test_rejects_invalid_results(tmp_path, data):
    with pytest.raises(ValueError):  # noqa: PT011
        view_image_result(data, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_rejects_missing_output_directory():
    with pytest.raises(ValueError, match="require an output directory"):
        view_image_result(image_status(), None)


def test_multimodal_observations_preserve_async_availability(tmp_path):
    call = {
        "Type": "tool_call",
        "Data": {
            "CallID": "image-call",
            "Name": "ViewImage",
            "Arguments": '{"path":"image.bmp"}',
        },
    }
    lines = [
        record(1, "model_response", response("turn-1", [call])),
        record(2, "tool_call_status", image_status("ready")),
        record(3, "turn", {"ID": "turn-2"}),
        record(4, "model_response", response("turn-2", [])),
        record(
            5,
            "tool_call_status",
            image_status(OriginalMIMEType="image/bmp", ScaleRatio=0.5),
        ),
        record(6, "turn", {"ID": "turn-3"}),
    ]
    trajectory = convert(
        lines,
        Agent(name="unreal-agent", version="test"),
        "session",
        output_dir=tmp_path,
    )
    observation = trajectory.steps[0].observation
    assert observation is not None
    observations = observation.results
    first_extra, second_extra = observations[0].extra, observations[1].extra
    assert first_extra is not None
    assert second_extra is not None
    assert observations[0].content == RUNNING
    assert first_extra["running"] is True
    assert first_extra["available_before_turn"] == "turn-2"
    assert second_extra["available_before_turn"] == "turn-3"
    assert observations[1].source_call_id == "image-call"
    assert second_extra["running"] is False
    content = observations[1].content
    assert isinstance(content, list)
    assert content[0].type == "image"
    assert content[1].type == "text"
    restored = Trajectory.model_validate(trajectory.to_json_dict())
    assert restored == trajectory
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 20


@pytest.mark.parametrize(
    "state", ["ready", "awaiting", "canceling", "completed", "failed", "canceled"]
)
def test_only_latest_pending_image_result_is_marked_available(tmp_path, state):
    call = {
        "Type": "tool_call",
        "Data": {
            "CallID": "image-call",
            "Name": "ViewImage",
            "Arguments": '{"path":"image.png"}',
        },
    }
    lines = [
        record(1, "model_response", response("turn-1", [call])),
        record(2, "tool_call_status", image_status("ready")),
        record(3, "tool_call_status", image_status(state)),
        record(4, "turn", {"ID": "turn-2"}),
    ]
    trajectory = convert(
        lines,
        Agent(name="unreal-agent", version="test"),
        "session",
        output_dir=tmp_path,
    )
    observation = trajectory.steps[0].observation
    assert observation is not None
    observations = observation.results
    assert len(observations) == 2
    first_extra, second_extra = observations[0].extra, observations[1].extra
    assert first_extra is not None
    assert second_extra is not None
    assert observations[0].content == RUNNING
    assert first_extra["running"] is True
    assert "available_before_turn" not in first_extra
    content, running = view_image_result(image_status(state), tmp_path)
    assert observations[1].content == content
    assert second_extra["running"] is running
    assert second_extra["available_before_turn"] == "turn-2"
