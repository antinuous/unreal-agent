from unreal_agent.primitives import TERMINAL_EVENT_TYPES, PrimitiveEventType


def test_terminal_event_types_match_go() -> None:
    assert {
        PrimitiveEventType.FAILED,
        PrimitiveEventType.CANCELED,
        PrimitiveEventType.IO_CREATE_COMPLETED,
        PrimitiveEventType.IO_READ_COMPLETED,
        PrimitiveEventType.PROCESS_EXITED,
        PrimitiveEventType.PROCESS_INPUT_WRITTEN,
        PrimitiveEventType.PROCESS_INPUT_WRITE_FAILED,
        PrimitiveEventType.PROCESS_INPUT_CLOSED,
        PrimitiveEventType.PROCESS_SIGNALED,
        PrimitiveEventType.REMOTE_COMPLETED,
        PrimitiveEventType.TIMER_FIRED,
        PrimitiveEventType.COMPUTE_COMPLETED,
    } == TERMINAL_EVENT_TYPES
