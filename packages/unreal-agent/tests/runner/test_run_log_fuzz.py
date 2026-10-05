# ruff: noqa: ASYNC230
"""Property test ported from Go's FuzzRunLogMatchesExecution.

Each example drives three runner invocations against one session in a fresh
looptime event loop, with the store's and the file primitive's blocking work
inlined so no worker threads let fake time jump. The Go seed corpus is the
`@example` list.
"""

import asyncio
import io
import os
import tempfile
from typing import Any, cast

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from unreal_agent import gojson
from unreal_agent.inbox import ControlMode, Input, InputKind, Settings
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, ReasoningEffort, Response, ToolCall
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.operation import Status, decode_skill_use
from unreal_agent.runner import run
from unreal_agent.runner.run import LLM_API_KEY_ENVIRONMENT
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Item, ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.tool import SKILL_USE_NAME

from tests.runner._helpers import (
    FakeClient,
    copy_log_response,
    decode_log_items,
    environment,
    fuzz_log_response,
    fuzz_log_usage,
    is_error,
    read_log_history,
    reader,
    runner_config,
    to_valid_utf8,
    with_log_fuzz_seeds,
    without_threads,
)
from tests.runner.test_coordinator_log_fuzz import run_in_bubble


class FuzzLogOutput:
    """Fails the write of the model response after `responses` successful ones."""

    def __init__(self, output: io.BytesIO, responses: int, error: Exception) -> None:
        self.output = output
        self.responses = responses
        self.error = error

    def write(self, data: bytes, /) -> int:
        item = gojson.decode(data, Item)
        if item.Kind == ItemKind.MODEL_RESPONSE:
            if self.responses == 0:
                raise self.error
            self.responses -= 1
        return self.output.write(data)

    def flush(self) -> None:
        pass


def assert_execution_log(
    items: list[Item],
    returned: list[Response],
    requests: int,
    text: str,
    skill_content: str,
    valid_calls: set[str],
    runs: int,
    settled: bool,
) -> None:
    responses: list[Response] = []
    turns: list[Turn] = []
    previous = ""
    inputs = stops = settings_count = 0
    input_ids: set[str] = set()
    calls: dict[str, str] = {}
    statuses: dict[str, ToolCallStatus] = {}
    responded: set[str] = set()
    for index, item in enumerate(items):
        assert item.Sequence == index + 1, item
        assert not item.RecordedAt.is_zero(), item
        match item.Data:
            case Input() as value:
                assert value.ID, "empty or duplicate input ID"
                assert value.ID not in input_ids, "empty or duplicate input ID"
                input_ids.add(value.ID)
                match value.Kind:
                    case InputKind.EXTERNAL:
                        inputs += 1
                        assert gojson.decode(bytes(value.Payload), str) == text, (
                            "external input changed"
                        )
                    case InputKind.CONTROL:
                        control = value.decode_control_message()
                        match control.Mode:
                            case ControlMode.STOP_WHEN_IDLE:
                                stops += 1
                            case ControlMode.UPDATE_SETTINGS:
                                settings_count += 1
                                assert control.Parameters == Settings(
                                    ReasoningEffort=ReasoningEffort.HIGH
                                ), "logged settings differ from the runner request"
                            case _:
                                raise AssertionError(f"unexpected control: {control!r}")
                    case _:
                        raise AssertionError(f"unexpected input kind {value.Kind!r}")
            case Turn() as value:
                assert value.ID, f"invalid turn chain: {value!r} after {previous!r}"
                assert value.PreviousTurnID == previous, (
                    f"invalid turn chain: {value!r} after {previous!r}"
                )
                assert value.Type == TurnType.REGULAR, (
                    f"invalid turn chain: {value!r} after {previous!r}"
                )
                previous = value.ID
                turns.append(value)
            case ModelResponse() as value:
                assert value.TurnID == previous, "response duplicated or attached to the wrong turn"
                assert value.TurnID not in responded, (
                    "response duplicated or attached to the wrong turn"
                )
                responded.add(value.TurnID)
                responses.append(value.Response)
                for output in value.Response.Output:
                    if isinstance(output.Data, ToolCall):
                        calls[output.Data.CallID] = value.TurnID
            case ToolCallStatus() as value:
                assert calls.get(value.CallID) == value.TurnID, (
                    "tool status preceded its call or changed turn"
                )
                if value.CallID not in valid_calls:
                    assert value.Status.Error, (
                        f"invalid call was not recorded as an error: {value!r}"
                    )
                    assert not value.Operations, (
                        f"invalid call was not recorded as an error: {value!r}"
                    )
                    assert not value.Status.WaitingFor, (
                        f"invalid call was not recorded as an error: {value!r}"
                    )
                else:
                    assert not value.Status.Error, f"tool status lost its operation: {value!r}"
                    assert len(value.Operations) == 1, f"tool status lost its operation: {value!r}"
                    assert value.Status.WaitingFor == [value.Operations[0].ID], (
                        f"tool status lost its operation: {value!r}"
                    )
                    prior = statuses.get(value.CallID)
                    assert prior is None or prior.Operations[0].ID == value.Operations[0].ID, (
                        "tool operation identity changed"
                    )
                    if value.Operations[0].Status == Status.COMPLETED:
                        state = decode_skill_use(value.Operations[0])
                        assert state.Content == skill_content.encode(), (
                            "logged tool result differs from the file read"
                        )
                statuses[value.CallID] = value
            case _:
                raise AssertionError(f"unexpected logged item: {item!r}")
    assert inputs == 1, "logged external inputs, want one despite retries"
    assert (stops, settings_count) == (runs, runs), "logged stops and settings differ from runs"
    assert len(turns) == requests, "logged turns differ from actual model requests"
    assert responses == returned, (
        "logged responses differ from completed model responses (including usage)"
    )
    if settled:
        for call_id in calls:
            status = statuses.get(call_id)
            assert status is not None, f"settled execution lost the result for call {call_id!r}"
            if call_id in valid_calls:
                assert status.Operations[0].Status == Status.COMPLETED, (
                    f"settled execution lost the result for call {call_id!r}"
                )


async def run_log_case(
    actions: bytes, text: str, input: int, cached: int, written: int, output: int, reasoning: int
) -> None:
    loop = asyncio.get_running_loop()
    cast(Any, loop)._clock_resolution = 1e-12
    text = to_valid_utf8(text)
    usage = fuzz_log_usage(text, input, cached, written, output, reasoning)
    with (
        tempfile.TemporaryDirectory() as workspace,
        tempfile.TemporaryDirectory() as sessions,
        without_threads(),
    ):
        skill_path = os.path.join(workspace, ".harness", "skills", "journal", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), mode=0o700)
        skill_content = "---\nname: journal\ndescription: Read the journal fixture.\n---\n" + text
        with open(skill_path, "w") as file:
            file.write(skill_content)
        id = "execution"
        message_id = "69621f8d-4f4d-49a5-8f7d-3b24fd855c01"
        message = {"role": "user", "content": text, "message_id": message_id}
        encoded_request = gojson.encode(
            {"session_id": id, "model": "journal-model", "messages": [message, message]}
        )
        responses: list[Response] = []
        valid_calls: set[str] = set()
        for index, action in enumerate(actions):
            response = fuzz_log_response(text, usage, index, action)
            for call_index in range(1 + action % 2):
                call = ToolCall(
                    CallID=f"call-{index}-{call_index}",
                    Name=SKILL_USE_NAME,
                    Arguments='{"name":"journal"}',
                )
                match action % 4:
                    case 0:
                        call.Name = "UnknownTool"
                    case 1:
                        call.Arguments = "{"
                    case _:
                        valid_calls.add(call.CallID)
                response.Output.append(
                    LLMItem(ProviderID=call.CallID, Type=ItemType.TOOL_CALL, Data=call)
                )
            responses.append(response)
        responses.append(fuzz_log_response(text, usage, len(responses), len(actions)))

        mode = actions[0] % 4 if actions else 0
        fail_at = actions[-1] % len(responses) if actions else 0
        injected = False
        provider_failure = ValueError("generated provider failure")
        output_failure = OSError("generated stdout failure")
        returned: list[Response] = []
        requests = 0
        history: list[Item] = []
        all_logged: list[Item] = []
        previous_log = b""
        log_directory = os.path.join(workspace, "logs")
        running: list[asyncio.Task[None]] = []
        for run_index in range(3):
            running.clear()

            def respond(request: LLMRequest) -> Response:
                nonlocal requests, injected
                if (
                    request.Model.ID != "journal-model"
                    or request.Model.ReasoningEffort != ReasoningEffort.HIGH
                ):
                    raise ValueError(f"unexpected request settings: {request.Model!r}")
                requests += 1
                if not injected and len(returned) == fail_at and mode in (1, 2):
                    injected = True
                    if mode == 2:
                        running[0].cancel()
                        raise asyncio.CancelledError
                    raise provider_failure
                if len(returned) == len(responses):
                    raise ValueError("settled execution requested another model response")
                response = responses[len(returned)]
                returned.append(response)
                return copy_log_response(response)

            client = FakeClient(respond)
            stdout = io.BytesIO()
            destination: Any = stdout
            if run_index == 0 and mode == 3:
                destination = FuzzLogOutput(stdout, fail_at, output_failure)
            run_task: asyncio.Task[None] = asyncio.create_task(
                run(
                    [
                        "-workspace",
                        workspace,
                        "-session-directory",
                        sessions,
                        "-log-directory",
                        log_directory,
                        "-tool-heartbeat-interval",
                        "0",
                    ],
                    environment({LLM_API_KEY_ENVIRONMENT: "secret"}),
                    reader(encoded_request),
                    destination,
                    io.BytesIO(),
                    runner_config(client),
                )
            )
            running.append(run_task)
            async with asyncio.timeout(60):
                await asyncio.wait([run_task])
            error = None if run_task.cancelled() else run_task.exception()
            if run_index == 0 and mode == 2:
                assert run_task.cancelled(), f"run {run_index}: error = {error!r}, want cancel"
            elif run_index == 0 and mode == 1:
                assert is_error(error, provider_failure), f"run {run_index}: error = {error!r}"
            elif run_index == 0 and mode == 3:
                assert is_error(error, output_failure), f"run {run_index}: error = {error!r}"
            else:
                assert not run_task.cancelled(), f"run {run_index}: {error!r}"
                assert error is None, f"run {run_index}: {error!r}"
            assert client.closed, "runner did not close the client"
            entries = sorted(os.listdir(log_directory))
            assert entries, "no log files"
            combined = b""
            for entry in entries:
                with open(os.path.join(log_directory, entry), "rb") as file:
                    combined += file.read()
            assert combined.startswith(previous_log), "resuming changed the previous file log"
            logged = combined[len(previous_log) :]
            previous_log = combined
            if run_index == 0 and mode == 3:
                assert logged.startswith(stdout.getvalue()), (
                    "file log did not preserve the record whose stdout write failed"
                )
                assert len(logged) != len(stdout.getvalue()), (
                    "file log did not preserve the record whose stdout write failed"
                )
            else:
                assert logged == stdout.getvalue(), "file log differs from stdout"
            store = Store(sessions)
            items = decode_log_items(logged)
            next_history = await read_log_history(store, id, 1 + len(actions) % 7)
            assert next_history[: len(history)] == history, "resuming changed existing history"
            assert next_history[len(history) :] == items, (
                "file log does not exactly match newly persisted history"
            )
            history = next_history
            all_logged.extend(items)
            assert_execution_log(
                all_logged,
                returned,
                requests,
                text,
                skill_content,
                valid_calls,
                run_index + 1,
                run_index > 0 or mode == 0,
            )
            if run_index > 0:
                resumed = await store.resume(id)
                assert not resumed.operations, "settled run has unfinished operations"
        assert len(returned) == len(responses), "execution did not return every planned response"


@settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@with_log_fuzz_seeds
@given(
    st.binary(max_size=16),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=512),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
    st.integers(0, 2**64 - 1),
)
def test_run_log_matches_execution(
    actions: bytes, text: str, input: int, cached: int, written: int, output: int, reasoning: int
) -> None:
    size = len(text.encode())
    assume(len(actions) <= 16 and size <= 128 << 10 and len(actions) * size <= 256 << 10)
    run_in_bubble(run_log_case(actions, text, input, cached, written, output, reasoning))
