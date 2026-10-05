"""Asynchronous primitives: compute, timers, file read and create, processes, and remote HTTP.

Starting a primitive
    Each start function (`compute`, `schedule_timer`, `read_file`, `create`,
    `RemoteClient.send_request`, `start_process`, and the `ProcessInvocation`
    controls) must be called inside a running event loop. It starts the work
    as an eager `asyncio.Task` and returns it (`start_process` returns the
    `ProcessInvocation`, whose `task` runs the process). The module keeps the
    task alive, so callers may drop it the way Go callers drop a goroutine.
    Awaiting the task is optional.

Event delivery
    The caller owns the event sink: any object with `async put(event)`, such
    as `asyncio.Queue` or `Channel` (Go channel semantics, including
    unbuffered rendezvous). Primitives never close or drain it. `put` provides
    backpressure; a bounded queue blocks the primitive until the consumer
    catches up. Several primitives may share one sink and are told apart by
    `source` and `correlation_id`.

    Every primitive sends exactly one terminal event, always last. Terminal
    types are listed in `TERMINAL_EVENT_TYPES`: `primitive.failed`,
    `primitive.canceled`, and the per-primitive completion types. Process
    controls (`write_input`, `close_input`, `signal`) are separate primitives
    with their own terminal events. A process stream failure
    (`process.stream_failed`) and remote events before `remote.completed` are
    not terminal.

    Remote requests keep the response idle timer paused while an event is
    being delivered, so consumer backpressure never counts as an idle response.

Cancellation
    Cancelling the returned task is the equivalent of cancelling the Go
    context. Cancellation is a request recorded on the task
    (`Task.cancelling()`), not an immediate exit:

    * Non-terminal events not yet delivered are dropped.
    * The terminal event is never dropped. A cancelled primitive normally ends
      with `primitive.canceled`. It ends with `primitive.failed` when cleanup
      fails, and with its completion event when the work had already
      committed (a finished process, a created path) before cancellation took
      effect.
    * Blocking work already running (a file system call, a compute callback,
      an in-flight pipe write) finishes before the terminal event. Compute
      callbacks receive a `threading.Event` that is set on cancellation.
    * A cancelled process gets SIGTERM on its process group, a 5 second grace
      period, then SIGKILL.
    * After the terminal event is delivered the task re-raises
      `asyncio.CancelledError`, so awaiting a cancelled primitive raises it.
    * Process controls own their cancellation. Cancelling the process task
      does not cancel a control already running; later controls fail with
      "process invocation is done".

Durations are float seconds. Wall-clock instants are `gojson.Timestamp`.
"""

from unreal_agent.primitives.channel import Channel
from unreal_agent.primitives.compute import ComputeFunction, ComputeRequest, ComputeResult, compute
from unreal_agent.primitives.file import (
    IO_READ_CHUNK_SIZE,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    read_file,
)
from unreal_agent.primitives.file_create import (
    IO_CREATE_DEFAULT_MODE,
    IOCreateKind,
    IOCreateRequest,
    IOCreateResult,
    create,
)
from unreal_agent.primitives.primitive_dispatch import PrimitiveDispatchType
from unreal_agent.primitives.primitive_event import (
    TERMINAL_EVENT_TYPES,
    CorrelationID,
    EventSink,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
)
from unreal_agent.primitives.process import (
    PROCESS_OUTPUT_CHUNK_SIZE,
    ProcessCloseInputRequest,
    ProcessExitResult,
    ProcessInvocation,
    ProcessOutputResult,
    ProcessPipeSet,
    ProcessSignalRequest,
    ProcessStartedResult,
    ProcessStartRequest,
    ProcessStream,
    ProcessStreamFailureResult,
    ProcessWriteFailureResult,
    ProcessWriteRequest,
    ProcessWriteResult,
    start_process,
)
from unreal_agent.primitives.remote import (
    DEFAULT_REMOTE_MAX_ATTEMPTS,
    DEFAULT_RETRYABLE_STATUS_CODES,
    RemoteClient,
    RemoteCompletedResult,
    RemoteOutputResult,
    RemoteRequest,
    RemoteResponseStartedResult,
    RemoteRetryPolicy,
    RemoteRetryScheduledResult,
    RemoteSSEOptions,
    RemoteStreamFailureResult,
    default_remote_request,
    new_remote_http_client,
)
from unreal_agent.primitives.remote_sse import SSEFrameDelimiter
from unreal_agent.primitives.sse import sse_data
from unreal_agent.primitives.timer import TimerRequest, TimerResult, schedule_timer

__all__ = [
    "DEFAULT_REMOTE_MAX_ATTEMPTS",
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "IO_CREATE_DEFAULT_MODE",
    "IO_READ_CHUNK_SIZE",
    "PROCESS_OUTPUT_CHUNK_SIZE",
    "TERMINAL_EVENT_TYPES",
    "Channel",
    "ComputeFunction",
    "ComputeRequest",
    "ComputeResult",
    "CorrelationID",
    "EventSink",
    "IOCreateKind",
    "IOCreateRequest",
    "IOCreateResult",
    "IOReadCompletedResult",
    "IOReadOutputResult",
    "IOReadRequest",
    "PrimitiveDispatchType",
    "PrimitiveEvent",
    "PrimitiveEventType",
    "PrimitiveFailureResult",
    "ProcessCloseInputRequest",
    "ProcessExitResult",
    "ProcessInvocation",
    "ProcessOutputResult",
    "ProcessPipeSet",
    "ProcessSignalRequest",
    "ProcessStartRequest",
    "ProcessStartedResult",
    "ProcessStream",
    "ProcessStreamFailureResult",
    "ProcessWriteFailureResult",
    "ProcessWriteRequest",
    "ProcessWriteResult",
    "RemoteClient",
    "RemoteCompletedResult",
    "RemoteOutputResult",
    "RemoteRequest",
    "RemoteResponseStartedResult",
    "RemoteRetryPolicy",
    "RemoteRetryScheduledResult",
    "RemoteSSEOptions",
    "RemoteStreamFailureResult",
    "SSEFrameDelimiter",
    "SourceID",
    "TimerRequest",
    "TimerResult",
    "compute",
    "create",
    "default_remote_request",
    "new_remote_http_client",
    "read_file",
    "schedule_timer",
    "sse_data",
    "start_process",
]
