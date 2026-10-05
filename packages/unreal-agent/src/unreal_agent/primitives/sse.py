import re
from typing import Final

_LINE_BREAKS: Final = re.compile(rb"[\r\n]+")
_BOM: Final = "﻿".encode()


def sse_data(frame: bytes) -> bytes | None:
    """Extract the data fields from one already-framed SSE event.

    None means the frame has no data fields; an empty data field returns b"".
    """
    frame = frame.removeprefix(_BOM)
    data: bytearray | None = None
    for line in _LINE_BREAKS.split(frame):
        if not line:
            continue
        field, _, value = line.partition(b":")
        if field != b"data":
            continue
        if data is None:
            data = bytearray()
        data += value.removeprefix(b" ")
        data += b"\n"
    if data is None:
        return None
    return bytes(data[:-1])
