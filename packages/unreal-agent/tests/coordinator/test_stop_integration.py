import asyncio
import os
from pathlib import Path
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import Coordinator, Dependencies
from unreal_agent.inbox import ControlMode, Inbox, Input
from unreal_agent.llm import Request, Response
from unreal_agent.operation import (
    LocalOperationManager,
    Operation,
    ShellInput,
    ShellState,
    Status,
    new_shell_spec,
)
from unreal_agent.session import ID as SessionID
from unreal_agent.sessionstore import BEFORE_FIRST, Item, ItemKind, ToolCallStatus

from tests.coordinator._helpers import (
    FakeAdapter,
    MemoryStore,
    as_store,
    external_event,
    fake_time,
    independent_tool_calls,
    new_tool_grace_test_run,
    persist_test_run,
    restore_test_run,
    stop_input,
    text_response,
    tool_grace_response,
)


def stop_integration_submit_now(inputs: Inbox, input: Input) -> None:
    """Submit from a synchronous callback; submission never waits for the consumer."""
    coroutine = inputs.submit(input)
    try:
        coroutine.send(None)
    except StopIteration:
        return
    coroutine.close()
    raise AssertionError("inbox submit suspended")


async def test_coordinator_stop_cancels_shell_process(tmp_path: Path) -> None:
    store, registry = independent_tool_calls(1)
    spec = new_shell_spec(ShellInput(Shell="/bin/sh", Command="exec sleep 30"), str(tmp_path), 64)
    value = replace(
        store.resume.operations[0],
        Status=Status.READY,
        State=spec.State,
        Type=spec.Type,
        Version=spec.Version,
        MaxOutputLength=spec.MaxOutputLength,
    )
    store.resume.operations[0] = value
    status = cast(ToolCallStatus, store.history[2].Data)
    status.Operations[0] = value
    store.history[2].Data = status
    inputs = Inbox()
    stop = stop_input("stop", ControlMode.STOP_HARD)
    process_group_id = 0

    def on_save_operation(update: Operation) -> None:
        nonlocal process_group_id
        try:
            state = gojson.decode(bytes(update.State), ShellState)
        except Exception as error:
            store.save_operation_error = error
            return
        if process_group_id == 0 and state.ProcessGroupID > 1:
            process_group_id = state.ProcessGroupID
            stop_integration_submit_now(inputs, stop)

    store.on_save_operation = on_save_operation

    async def respond(request: Request) -> Response:
        return text_response("Stopped.")

    adapter = FakeAdapter(respond)
    manager = LocalOperationManager()
    try:
        current = Coordinator(
            Dependencies(
                session_id="session-1",
                inbox=inputs,
                restored=store.resume,
                sessions=as_store(store),
                context_builder=new_builder(),
                llm=adapter,
                tools=registry,
                operations=manager,
            )
        )
        async with asyncio.timeout(10):
            await current.run()
        assert process_group_id > 1, "shell process never started"
        with pytest.raises(ProcessLookupError):
            os.killpg(process_group_id, 0)
        last = store.appended_statuses[-1]
        assert last.Operations[0].Status == Status.CANCELED, (
            f"persisted operation status = {last.Operations[0].Status}"
        )
        assert len(adapter.requests) == 0, "hard stop requested a model response"
    finally:
        await manager.aclose()


@fake_time
async def test_coordinator_stop_after_tool_commit_settles_operation() -> None:
    run = new_tool_grace_test_run()
    store: MemoryStore = await persist_test_run(run)
    await restore_test_run(run, store)
    manager = LocalOperationManager()
    run.dependencies.operations = manager
    stop_queued = False

    def observer(session_id: SessionID, item: Item) -> None:
        nonlocal stop_queued
        if item.Kind != ItemKind.TOOL_CALL_STATUS or stop_queued:
            return
        status = cast(ToolCallStatus, item.Data)
        if len(status.Operations) == 0:
            return
        stop_queued = True
        stop_integration_submit_now(run.inputs, stop_input("stop", ControlMode.STOP_HARD))

    observer_id = store.add_observer(observer)
    try:
        await run.start()
        await run.input(external_event("input", "run tool"))
        await run.respond(0, tool_grace_response("A"))
        run.assert_stopped()
        message = "stop was not queued at the commit boundary or started another turn"
        assert stop_queued, message
        assert len(run.calls) == 1, message
        page = await store.items("session-1", BEFORE_FIRST, 100)
        last = page.items[-1]
        assert last.Kind == ItemKind.TOOL_CALL_STATUS, (
            f"last item = {last.Kind}, want settled tool call"
        )
        status = cast(ToolCallStatus, last.Data)
        message = f"persisted tool result = {status!r}, want completed value operation"
        assert status.CallID == "A", message
        assert len(status.Operations) == 1, message
        assert status.Operations[0].Status == Status.COMPLETED, message
        restored = await store.resume("session-1")
        assert len(restored.operations) == 0, (
            "stop left an operation unsettled in the session store"
        )
    finally:
        store.remove_observer(observer_id)
        await manager.aclose()
