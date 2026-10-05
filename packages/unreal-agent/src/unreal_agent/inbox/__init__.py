"""Input deduplication for one session.

`Inbox` is an unbounded FIFO of validated, deduplicated `Input` values with one
consumer. All methods run on the event loop thread.

- `Inbox(seen_ids=())` seeds the IDs already recorded for the session. An empty
  ID raises `ValueError`.
- `await inbox.submit(input)` validates first and raises `ValueError` prefixed
  with "submit input: ", then raises `ClosedError` if the inbox is closed. A
  previously seen ID is accepted and dropped. Otherwise a copy of the input is
  queued. Submit never waits for the consumer. Cancelling the submitting task
  before it runs leaves the inbox unchanged.
- `await inbox.get()` returns the next input in submission order and raises
  `ClosedError` once the inbox is closed. Cancelling a pending `get()` does not
  consume an input. `get_nowait()` raises `asyncio.QueueEmpty` when nothing is
  queued. `async for input in inbox` ends when the inbox is closed.
- `inbox.close()` discards queued
  inputs, wakes a pending `get()` with `ClosedError`, and makes later submits
  raise `ClosedError`. It is idempotent.

`Input.decode_control_message()` parses control payloads strictly and rejects
unknown members in the envelope and in settings parameters.
"""

from unreal_agent.inbox.inbox import (
    ID,
    ControlMessage,
    ControlMode,
    Input,
    InputKind,
    Settings,
)
from unreal_agent.inbox.local import ClosedError, Inbox

__all__ = [
    "ID",
    "ClosedError",
    "ControlMessage",
    "ControlMode",
    "Inbox",
    "Input",
    "InputKind",
    "Settings",
]
