from enum import StrEnum


class PrimitiveDispatchType(StrEnum):
    PROCESS_START = "process.start"
    IO_CREATE = "io.create"
    IO_READ = "io.read"
    COMPUTE = "compute"
