"""The owner of one session's central event loop.

`Coordinator(dependencies).run()` restores the session from
`dependencies.sessions` and `dependencies.restored`, then waits on the inbox,
operation updates, the tool heartbeat, the tool grace deadline, and the active
model request. After each event it drains queued inputs and then queued
operation updates (stopping after a 1ms idle gap or 100 values), reconciles
finished tool calls, and starts a model request when one is due. `run` is
single-use and returns None once a stop control completes.

Cancellation: cancelling the task that awaits `run` is Go's context
cancellation. `run` cancels its model request and pending reads and raises
`CancelledError`, which takes precedence over a model error or a closed
source. The caller closes the inbox after `run` returns.

Ordering: every effect follows the persistence it depends on. An input is
stored before it can wake the model, a turn before its model request, a
response before its tool calls are translated, a tool-call status with its
operation snapshots before those operations are dispatched, and an operation
update before the tool call it settles is reconciled. Local state changes
before each store write, and a failed dependency ends `run` with
`CoordinatorError` without further writes.
"""

from unreal_agent.coordinator.coordinator import CoordinatorError, Dependencies
from unreal_agent.coordinator.loop import Coordinator

__all__ = ["Coordinator", "CoordinatorError", "Dependencies"]
