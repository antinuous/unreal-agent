import os
from pathlib import Path

from unreal_agent.primitives import (
    TERMINAL_EVENT_TYPES,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
)
from unreal_agent.primitives.channel import Channel

MAX_INT64 = 2**63 - 1


def failure_text(event: PrimitiveEvent) -> str:
    assert isinstance(event.result, PrimitiveFailureResult)
    return event.result.error


async def collect_events(events: Channel[PrimitiveEvent]) -> list[PrimitiveEvent]:
    collected: list[PrimitiveEvent] = []
    while True:
        event = await events.get()
        collected.append(event)
        if event.type in TERMINAL_EVENT_TYPES:
            return collected


def single_event(events: list[PrimitiveEvent], event_type: PrimitiveEventType) -> PrimitiveEvent:
    assert len(events) == 1, events
    assert events[0].type == event_type, events
    return events[0]


def write_test_file(tmp_path: Path, contents: bytes, name: str = "input") -> str:
    path = tmp_path / name
    path.write_bytes(contents)
    os.chmod(path, 0o600)
    return str(path)
