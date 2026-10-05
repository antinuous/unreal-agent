import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from unreal_agent.primitives._errors import PrimitiveError

_LINE_ENDING: Final = re.compile(rb"[\r\n]")


class SSEFrameDelimiter(IntEnum):
    STRIP = 0
    PASS_THROUGH = 1


@dataclass(slots=True)
class RemoteOutputFrame:
    offset: int
    data: bytes


class RemoteSSEFramer:
    """Splits an SSE byte stream into events at blank lines (CR, LF, or CRLF).

    The size bound counts wire bytes of the frame being assembled, including a
    CRLF whose LF arrives in a later chunk.
    """

    __slots__ = (
        "_after_cr",
        "_delimiter_start",
        "_emitted_offset",
        "_ended_frame_wire_size",
        "_frame",
        "_frame_ended_at_cr",
        "_frame_wire_size",
        "_line_has_data",
        "frame_delimiter",
        "maximum_frame_size",
    )

    def __init__(self, maximum_frame_size: int, frame_delimiter: SSEFrameDelimiter) -> None:
        self.maximum_frame_size = maximum_frame_size
        self.frame_delimiter = frame_delimiter
        self._frame_wire_size = 0
        self._emitted_offset = 0
        self._frame = bytearray()
        self._delimiter_start = 0
        self._line_has_data = False
        self._after_cr = False
        self._frame_ended_at_cr = False
        self._ended_frame_wire_size = 0

    def append(self, encoded: bytes) -> tuple[list[RemoteOutputFrame], PrimitiveError | None]:
        """Return the complete frames, plus an error once a frame exceeds the size bound."""
        frames: list[RemoteOutputFrame] = []
        position = 0
        end = len(encoded)
        pass_through = self.frame_delimiter == SSEFrameDelimiter.PASS_THROUGH
        while position < end:
            if self._after_cr:
                if encoded[position] == 0x0A:
                    if self._frame_ended_at_cr:
                        if self._ended_frame_wire_size >= self.maximum_frame_size:
                            return frames, self._frame_size_error()
                        self._ended_frame_wire_size += 1
                        if pass_through:
                            if frames:
                                last = frames[-1]
                                last.data += b"\n"
                            else:
                                frames.append(RemoteOutputFrame(self._emitted_offset, b"\n"))
                            self._emitted_offset += 1
                        self._after_cr = False
                        self._frame_ended_at_cr = False
                        self._ended_frame_wire_size = 0
                        position += 1
                        continue
                    error = self._append_frame_bytes(encoded[position : position + 1])
                    if error is not None:
                        return frames, error
                    self._after_cr = False
                    position += 1
                    continue
                self._after_cr = False
                self._frame_ended_at_cr = False
                self._ended_frame_wire_size = 0

            match = _LINE_ENDING.search(encoded, position)
            if match is None:
                error = self._append_frame_bytes(encoded[position:])
                if error is not None:
                    return frames, error
                self._line_has_data = True
                break
            line_ending = match.start()
            error = self._append_frame_bytes(encoded[position : line_ending + 1])
            if error is not None:
                return frames, error
            if line_ending != position:
                self._line_has_data = True
            character = encoded[line_ending]
            position = line_ending + 1
            if character == 0x0D:
                self._frame_ended_at_cr = not self._line_has_data
                if self._line_has_data:
                    self._delimiter_start = len(self._frame) - 1
                self._line_has_data = False
                self._after_cr = True
                if self._frame_ended_at_cr:
                    self._ended_frame_wire_size = self._frame_wire_size
                    frames.append(self._take_frame(self._delimiter_start))
            else:
                if self._line_has_data:
                    self._delimiter_start = len(self._frame) - 1
                else:
                    frames.append(self._take_frame(self._delimiter_start))
                self._line_has_data = False
        return frames, None

    def finish(self) -> list[RemoteOutputFrame]:
        self._after_cr = False
        self._frame_ended_at_cr = False
        self._ended_frame_wire_size = 0
        if not self._frame or self.frame_delimiter == SSEFrameDelimiter.STRIP:
            self._frame_wire_size = 0
            self._frame = bytearray()
            self._delimiter_start = 0
            return []
        return [self._take_frame(len(self._frame))]

    def _append_frame_bytes(self, encoded: bytes) -> PrimitiveError | None:
        if len(encoded) > self.maximum_frame_size - self._frame_wire_size:
            return self._frame_size_error()
        self._frame += encoded
        self._frame_wire_size += len(encoded)
        return None

    def _take_frame(self, delimiter_start: int) -> RemoteOutputFrame:
        data = bytes(self._frame)
        if self.frame_delimiter == SSEFrameDelimiter.STRIP:
            data = data[:delimiter_start]
        frame = RemoteOutputFrame(self._emitted_offset, data)
        self._emitted_offset += len(data)
        self._frame_wire_size = 0
        self._frame = bytearray()
        self._delimiter_start = 0
        return frame

    def _frame_size_error(self) -> PrimitiveError:
        return PrimitiveError(f"remote SSE frame exceeds {self.maximum_frame_size} bytes")
