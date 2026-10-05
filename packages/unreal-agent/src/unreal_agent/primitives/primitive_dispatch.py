from enum import StrEnum


class PrimitiveDispatchType(StrEnum):
    PROCESS_START = "process.start"
    PROCESS_WRITE_INPUT = "process.write_input"
    PROCESS_CLOSE_INPUT = "process.close_input"
    PROCESS_SIGNAL = "process.signal"
    IO_CREATE = "io.create"
    IO_READ = "io.read"
    REMOTE_REQUEST = "remote.request"
    TIMER_SCHEDULE = "timer.schedule"
    COMPUTE = "compute"
