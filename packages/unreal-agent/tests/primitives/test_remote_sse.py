import pytest
from unreal_agent.primitives.remote_sse import (
    RemoteOutputFrame,
    RemoteSSEFramer,
    SSEFrameDelimiter,
)

NEWLINES = ["\n", "\r", "\r\n"]
BOUNDARIES = [
    (line, blank) for line in NEWLINES for blank in NEWLINES if not (line == "\r" and blank == "\n")
]


def _append_all(framer: RemoteSSEFramer, chunks: list[bytes]) -> list[RemoteOutputFrame]:
    frames: list[RemoteOutputFrame] = []
    for chunk in chunks:
        decoded, error = framer.append(chunk)
        assert error is None
        frames.extend(decoded)
    return frames


def test_remote_sse_framer_emits_complete_frames_across_chunks() -> None:
    framer = RemoteSSEFramer(1024, SSEFrameDelimiter.PASS_THROUGH)
    frames = _append_all(
        framer, [b"data: one\r", b"\n\r", b"\ndata: two\n\n: keepalive\r\rpartial"]
    )
    want = [b"data: one\r\n\r", b"\n", b"data: two\n\n", b": keepalive\r\r"]
    assert [frame.data for frame in frames] == want
    offset = 0
    for frame in frames:
        assert frame.offset == offset
        offset += len(frame.data)


def test_remote_sse_framer_emits_frame_ending_in_bare_cr() -> None:
    framer = RemoteSSEFramer(1024, SSEFrameDelimiter.PASS_THROUGH)
    frames, error = framer.append(b"data: one\r\r")
    assert error is None
    assert frames == [RemoteOutputFrame(0, b"data: one\r\r")]


def test_remote_sse_framer_preserves_incomplete_bytes_at_eof() -> None:
    framer = RemoteSSEFramer(1024, SSEFrameDelimiter.PASS_THROUGH)
    frames, error = framer.append(b"data: partial")
    assert error is None
    assert frames == []
    assert framer.finish() == [RemoteOutputFrame(0, b"data: partial")]


def test_remote_sse_framer_does_not_emit_incomplete_frame_when_stripping_delimiter() -> None:
    framer = RemoteSSEFramer(1024, SSEFrameDelimiter.STRIP)
    frames, error = framer.append(b"data: partial")
    assert error is None
    assert frames + framer.finish() == []


def test_remote_sse_framer_emits_every_empty_frame_when_stripping_delimiter() -> None:
    framer = RemoteSSEFramer(1024, SSEFrameDelimiter.STRIP)
    frames, error = framer.append(b"\n\ndata: x\n\n")
    assert error is None
    assert frames == [
        RemoteOutputFrame(0, b""),
        RemoteOutputFrame(0, b""),
        RemoteOutputFrame(0, b"data: x"),
    ]


def test_remote_sse_framer_bounds_incomplete_frame() -> None:
    framer = RemoteSSEFramer(4, SSEFrameDelimiter.STRIP)
    _, error = framer.append(b"1234")
    assert error is None
    _, error = framer.append(b"5")
    assert error is not None


@pytest.mark.parametrize(("line_ending", "blank_line"), BOUNDARIES, ids=repr)
def test_remote_sse_framer_passes_through_every_newline_boundary(
    line_ending: str, blank_line: str
) -> None:
    encoded = f"data: one{line_ending}{blank_line}data: two{line_ending}{blank_line}".encode()
    for cut in range(len(encoded) + 1):
        framer = RemoteSSEFramer(len(encoded), SSEFrameDelimiter.PASS_THROUGH)
        frames = _append_all(framer, [encoded[:cut], encoded[cut:]])
        frames.extend(framer.finish())
        output = b""
        for frame in frames:
            assert frame.offset == len(output), f"cut {cut}"
            output += frame.data
        assert output == encoded, f"cut {cut}"


@pytest.mark.parametrize(("line_ending", "blank_line"), BOUNDARIES, ids=repr)
def test_remote_sse_framer_strips_every_frame_delimiter(line_ending: str, blank_line: str) -> None:
    encoded = f"data: one{line_ending}{blank_line}data: two{line_ending}{blank_line}".encode()
    for cut in range(len(encoded) + 1):
        framer = RemoteSSEFramer(len(encoded), SSEFrameDelimiter.STRIP)
        frames = _append_all(framer, [encoded[:cut], encoded[cut:]])
        frames.extend(framer.finish())
        assert frames == [
            RemoteOutputFrame(0, b"data: one"),
            RemoteOutputFrame(len(b"data: one"), b"data: two"),
        ], f"cut {cut}"


def test_remote_sse_framer_returns_complete_frames_before_size_failure() -> None:
    framer = RemoteSSEFramer(5, SSEFrameDelimiter.STRIP)
    frames, error = framer.append(b"one\n\n123456")
    assert error is not None
    assert frames == [RemoteOutputFrame(0, b"one")]


def test_remote_sse_framer_counts_split_crlf_in_frame_size() -> None:
    framer = RemoteSSEFramer(5, SSEFrameDelimiter.STRIP)
    frames, error = framer.append(b"one\n\r")
    assert error is None
    assert [frame.data for frame in frames] == [b"one"]
    _, error = framer.append(b"\n")
    assert error is not None
