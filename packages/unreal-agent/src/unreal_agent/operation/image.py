import base64
import io
import struct
import threading
import zlib
from bisect import bisect_right
from collections.abc import Buffer, Sequence
from typing import IO, Annotated, Final, Protocol, cast

import msgspec
from msgspec.structs import replace
from PIL import Image, UnidentifiedImageError

from unreal_agent.gojson import Int64, JSONError, OmitZero, Raw, Struct, decode, encode
from unreal_agent.operation.operation import (
    Operation,
    OperationError,
    PrimitiveDispatch,
    Spec,
    Status,
    Step,
    unsupported,
)
from unreal_agent.primitives import (
    ComputeRequest,
    ComputeResult,
    CorrelationID,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
)
from unreal_agent.primitives._errors import ContextCanceledError, go_quote

__all__ = [
    "DEFAULT_MAX_VIEW_IMAGE_SOURCE_PIXELS",
    "MAX_VIEW_IMAGE_SOURCE_BYTES",
    "TYPE_VIEW_IMAGE",
    "VERSION_VIEW_IMAGE",
    "ViewImage",
    "ViewImageConfig",
    "ViewImageResult",
    "ViewImageState",
    "decode_view_image_state",
    "new_view_image_spec",
]

# The pixel budget is enforced from the header before decoding, so Pillow's
# own decompression bomb guard would only reject images the budget allows.
Image.MAX_IMAGE_PIXELS = None

TYPE_VIEW_IMAGE: Final = "view_image"
VERSION_VIEW_IMAGE: Final = 1
DEFAULT_MAX_VIEW_IMAGE_SOURCE_PIXELS: Final = 32_000_000
MAX_VIEW_IMAGE_SOURCE_BYTES: Final = 256 * 1024 * 1024

VIEW_IMAGE_READ_CORRELATION: Final = CorrelationID("view-image-read")
VIEW_IMAGE_PROCESS_CORRELATION: Final = CorrelationID("view-image-process")


class ViewImageConfig(Struct):
    MaxSize: Int64 = 0
    """Limits the base64 content in bytes, excluding any data URL prefix."""
    MaxHeight: Int64 = 0
    MaxWidth: Int64 = 0
    MaxSourcePixels: Annotated[Int64, OmitZero] = 0
    """Bounds decoding work; zero uses DEFAULT_MAX_VIEW_IMAGE_SOURCE_PIXELS."""


class ViewImageResult(Struct):
    Content: str = ""
    OriginalWidth: Int64 = 0
    OriginalHeight: Int64 = 0
    OriginalMIMEType: str = ""
    EncodedMIMEType: str = ""
    ScaleRatio: float = 0.0
    Error: str = ""


class ViewImageState(Struct):
    Path: str = ""
    Config: ViewImageConfig = msgspec.field(default_factory=ViewImageConfig)
    Result: ViewImageResult | None = None


def new_view_image_spec(path: str, config: ViewImageConfig) -> Spec:
    state = ViewImageState(Path=path, Config=config)
    _validate_view_image_state(state)
    return Spec(Type=TYPE_VIEW_IMAGE, Version=VERSION_VIEW_IMAGE, State=Raw(encode(state)))


def decode_view_image_state(current: Operation) -> ViewImageState:
    operation_id = go_quote(current.ID)
    if current.Type != TYPE_VIEW_IMAGE or current.Version != VERSION_VIEW_IMAGE:
        raise unsupported(
            f"decode view-image operation {operation_id}: "
            f"type {go_quote(current.Type)} version {current.Version}"
        )
    try:
        state = decode(bytes(current.State), ViewImageState)
    except JSONError as error:
        raise OperationError(
            f"decode view-image operation {operation_id} state: {error}"
        ) from error
    try:
        _validate_view_image_state(state)
    except OperationError as error:
        raise OperationError(
            f"validate view-image operation {operation_id} state: {error}"
        ) from error
    return state


def fail_view_image(current: Operation, cause: BaseException | str) -> Step:
    return ViewImage(current).fail(cause)


def _validate_view_image_state(state: ViewImageState) -> None:
    if not state.Path:
        raise OperationError("image path must be set")
    config = state.Config
    if config.MaxSize <= 0 or config.MaxHeight <= 0 or config.MaxWidth <= 0:
        raise OperationError("image MaxSize, MaxHeight, and MaxWidth must be positive")
    if config.MaxSourcePixels < 0:
        raise OperationError("image MaxSourcePixels must not be negative")


class ViewImage:
    """The view-image actor. Source bytes stay transient; recovery rereads the whole file."""

    __slots__ = ("_chunks", "_current", "_processing", "_read_size", "_state")

    def __init__(self, current: Operation) -> None:
        self._state = decode_view_image_state(current)
        self._current = replace(current, State=Raw(b""))
        self._chunks: list[bytes] = []
        self._read_size = 0
        self._processing = False

    @property
    def state(self) -> ViewImageState:
        return self._state

    def handle(self, event: PrimitiveEvent | None) -> Step:
        current = self._current
        operation_id = go_quote(current.ID)
        match current.Status:
            case Status.READY | Status.AWAITING:
                if self._state.Result is not None:
                    raise OperationError(
                        f"advance view-image operation {operation_id}: state is terminal"
                    )
                if event is None:
                    if self._processing:
                        raise OperationError(
                            f"advance view-image operation {operation_id}: "
                            "processing is already in progress"
                        )
                    return self._read()
                if current.Status == Status.READY:
                    raise OperationError(
                        f"advance ready view-image operation {operation_id}: "
                        "unexpected primitive event"
                    )
                return self._handle_event(event)
            case Status.CANCELING:
                if event is not None:
                    raise OperationError(
                        f"advance canceling view-image operation {operation_id}: "
                        "unexpected primitive event"
                    )
                return self.cancel()
            case _:
                raise OperationError(
                    f"advance view-image operation {operation_id}: "
                    f"terminal status {go_quote(current.Status)}"
                )

    def _read(self) -> Step:
        self._chunks, self._read_size = [], 0
        self._current = replace(self._current, Status=Status.AWAITING)
        step = self._checkpoint()
        step.dispatches = [
            PrimitiveDispatch(
                PrimitiveDispatchType.IO_READ,
                IOReadRequest(
                    source=SourceID(self._current.ID),
                    correlation_id=VIEW_IMAGE_READ_CORRELATION,
                    path=self._state.Path,
                    count=MAX_VIEW_IMAGE_SOURCE_BYTES,
                ),
            )
        ]
        return step

    def _handle_event(self, event: PrimitiveEvent) -> Step:
        if event.source != self._current.ID:
            raise OperationError(
                f"image event source is {go_quote(event.source)}, want {go_quote(self._current.ID)}"
            )
        correlation = (
            VIEW_IMAGE_PROCESS_CORRELATION if self._processing else VIEW_IMAGE_READ_CORRELATION
        )
        # Dispatch rejection has no correlation because the primitive never started.
        dispatch_rejected = event.type == PrimitiveEventType.FAILED and event.correlation_id == ""
        if event.correlation_id != correlation and not dispatch_rejected:
            raise OperationError(
                f"image event correlation is {go_quote(event.correlation_id)}, "
                f"want {go_quote(correlation)}"
            )
        if event.type == PrimitiveEventType.CANCELED:
            if event.result is not None:
                raise OperationError("image cancellation returned an invalid result")
            return self.cancel()
        if event.type == PrimitiveEventType.FAILED:
            failure = event.result
            if not isinstance(failure, PrimitiveFailureResult):
                raise OperationError("image primitive failed with an invalid result")
            if self._processing or dispatch_rejected:
                return self.fail(f"image primitive failed: {failure.error}")
            cause = OperationError(f"read image: {failure.error}")
            if self._read_size == 0:
                return self.fail(cause)
            return self._process(cause)
        if self._processing:
            return self._processed(event)
        return self._handle_read(event)

    def _handle_read(self, event: PrimitiveEvent) -> Step:
        match event.type:
            case PrimitiveEventType.IO_READ_OUTPUT:
                output = event.result
                if (
                    not isinstance(output, IOReadOutputResult)
                    or output.offset != self._read_size
                    or len(output.data) > MAX_VIEW_IMAGE_SOURCE_BYTES - self._read_size
                ):
                    raise OperationError("image read returned invalid output")
                self._chunks.append(output.data)
                self._read_size += len(output.data)
                return Step()
            case PrimitiveEventType.IO_READ_COMPLETED:
                completed = event.result
                if (
                    not isinstance(completed, IOReadCompletedResult)
                    or completed.size < 0
                    or min(completed.size, MAX_VIEW_IMAGE_SOURCE_BYTES) != self._read_size
                ):
                    raise OperationError("image read returned an invalid completion")
                if completed.size > MAX_VIEW_IMAGE_SOURCE_BYTES:
                    return self._process(
                        OperationError(
                            f"image source is {completed.size} bytes; source limit is "
                            f"{MAX_VIEW_IMAGE_SOURCE_BYTES} bytes (exceeded by "
                            f"{completed.size - MAX_VIEW_IMAGE_SOURCE_BYTES} bytes)"
                        )
                    )
                return self._process(None)
            case _:
                raise OperationError(f"image read returned unexpected event {go_quote(event.type)}")

    def _process(self, cause: BaseException | None) -> Step:
        source, chunks = SourceID(self._current.ID), self._chunks
        config = replace(self._state.Config)
        self._chunks, self._read_size = [], 0
        self._processing = True

        def run(canceled: threading.Event) -> ViewImageResult:
            result, error = prepare_view_image(canceled, chunks, config, cause)
            if error is not None:
                result.Content = ""
                result.Error = str(error)
            # Image failures are values so compute preserves their diagnostic metadata.
            return result

        return Step(
            dispatches=[
                PrimitiveDispatch(
                    PrimitiveDispatchType.COMPUTE,
                    ComputeRequest(
                        source=source, correlation_id=VIEW_IMAGE_PROCESS_CORRELATION, run=run
                    ),
                )
            ]
        )

    def _processed(self, event: PrimitiveEvent) -> Step:
        if event.type != PrimitiveEventType.COMPUTE_COMPLETED:
            raise OperationError(
                f"image processing returned unexpected event {go_quote(event.type)}"
            )
        computed = event.result
        if not isinstance(computed, ComputeResult):
            raise OperationError("image processing returned an invalid compute result")
        result = computed.value
        if not isinstance(result, ViewImageResult):
            raise OperationError("image processing returned an invalid image result")
        result = replace(result)
        status = Status.COMPLETED
        if result.Error:
            result.Content = ""
            status = Status.FAILED
        return self._finish(result, status)

    def fail(self, error: BaseException | str) -> Step:
        return self._finish(ViewImageResult(ScaleRatio=1.0, Error=str(error)), Status.FAILED)

    def cancel(self) -> Step:
        return self._finish(
            ViewImageResult(
                ScaleRatio=1.0, Error=f"view-image operation canceled: {ContextCanceledError()}"
            ),
            Status.CANCELED,
        )

    def _checkpoint(self) -> Step:
        return Step(operation=replace(self._current, State=Raw(encode(self._state))))

    def _finish(self, result: ViewImageResult, status: Status) -> Step:
        self._state.Result = result
        self._current = replace(self._current, Status=status)
        self._chunks, self._read_size = [], 0
        self._processing = False
        return self._checkpoint()


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


class _ImageError(Exception):
    pass


def _check_canceled(canceled: CancellationSignal) -> None:
    if canceled.is_set():
        raise ContextCanceledError


_PILLOW_FORMATS: Final = {
    "jpeg": "JPEG",
    "png": "PNG",
    "gif": "GIF",
    "bmp": "BMP",
    "tiff": "TIFF",
    "webp": "WEBP",
}
_PNG_MODES: Final = frozenset({"1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA"})


def prepare_view_image(
    canceled: CancellationSignal,
    chunks: Sequence[bytes],
    config: ViewImageConfig,
    cause: BaseException | None,
) -> tuple[ViewImageResult, BaseException | None]:
    result = ViewImageResult(ScaleRatio=1.0)
    # Inspect the buffered header even when reading failed.
    try:
        image_format = _inspect_view_image(_ChunkReader(chunks), result)
    except _ImageError as error:
        return result, _join(cause, error)
    if cause is not None:
        return result, cause
    try:
        content = _encode_view_image(canceled, chunks, config, result, image_format)
    except Exception as error:
        return result, error
    result.Content = content
    return result, None


def _join(first: BaseException | None, second: BaseException) -> BaseException:
    if first is None:
        return second
    return _ImageError(f"{first}\n{second}")


def _inspect_view_image(reader: _ChunkReader, result: ViewImageResult) -> str:
    header = reader.read(16)
    reader.seek(0)
    image_format = _sniff(header)
    if image_format is None:
        raise _ImageError(
            "decode image header; original dimensions unavailable: image: unknown format"
        )
    try:
        with Image.open(cast(IO[bytes], reader), formats=[_PILLOW_FORMATS[image_format]]) as source:
            width, height = source.size
    except Exception as error:
        dimensions = _header_dimensions(image_format, reader)
        if dimensions is None:
            raise _ImageError(
                "decode image header; original dimensions unavailable: "
                f"{_describe(error, image_format)}"
            ) from error
        width, height = dimensions
    result.OriginalWidth, result.OriginalHeight = width, height
    if width <= 0 or height <= 0:
        raise _ImageError("image dimensions must be positive")
    result.OriginalMIMEType = "image/" + image_format
    if image_format == "gif":
        raise _ImageError('unsupported image format "gif"')
    return image_format


def _encode_view_image(
    canceled: CancellationSignal,
    chunks: Sequence[bytes],
    config: ViewImageConfig,
    result: ViewImageResult,
    image_format: str,
) -> str:
    max_pixels = config.MaxSourcePixels or DEFAULT_MAX_VIEW_IMAGE_SOURCE_PIXELS
    if result.OriginalWidth > max_pixels // result.OriginalHeight:
        raise _ImageError(
            f"refuse to decode {result.OriginalWidth}x{result.OriginalHeight} {image_format} "
            f"image: exceeds MaxSourcePixels {max_pixels}"
        )
    try:
        with Image.open(
            cast(IO[bytes], _ChunkReader(chunks, canceled)), formats=[_PILLOW_FORMATS[image_format]]
        ) as opened:
            opened.load()
            source = opened.copy()
    except Exception as error:
        raise _ImageError(
            f"decode {image_format} pixels: {_describe(error, image_format)}"
        ) from error
    width, height, ratio = view_image_dimensions(
        result.OriginalWidth, result.OriginalHeight, config
    )
    result.ScaleRatio = ratio
    _check_canceled(canceled)
    if ratio < 1:
        source = source.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)

    result.EncodedMIMEType = "image/jpeg" if image_format == "jpeg" else "image/png"
    content = ImageContentWriter(canceled, config.MaxSize)
    encoder = _Base64Writer(content)
    error: BaseException | None = None
    try:
        if image_format == "jpeg" and ratio == 1:
            for chunk in chunks:
                encoder.write(chunk)
        elif image_format == "jpeg":
            source.convert("RGB").save(cast(IO[bytes], encoder), "JPEG", quality=90)
        else:
            if source.mode not in _PNG_MODES:
                source = source.convert("RGBA")
            source.save(cast(IO[bytes], encoder), "PNG")
    except Exception as failure:
        error = failure
    for finish in (encoder.close, lambda: _check_canceled(canceled)):
        try:
            finish()
        except Exception as failure:
            error = failure if error is None else _ImageError(f"{error}\n{failure}")
    if error is not None:
        raise _ImageError(
            f"encode {result.EncodedMIMEType}: {_describe(error)}; produced {content.size} "
            f"base64 bytes before stopping, final size unavailable (MaxSize {config.MaxSize} bytes)"
        ) from error

    if content.size > config.MaxSize:
        excess = content.size - config.MaxSize
        raise _ImageError(
            f"{result.EncodedMIMEType} image after resizing {result.OriginalWidth}x"
            f"{result.OriginalHeight} to {width}x{height} (scale {ratio:.6g}) needs "
            f"{content.size} base64 bytes; MaxSize is {config.MaxSize} bytes, exceeded by "
            f"{excess} bytes ({100 * excess / config.MaxSize:.1f}%)"
        )
    return content.content()


def view_image_dimensions(
    width: int, height: int, config: ViewImageConfig
) -> tuple[int, int, float]:
    ratio = min(1.0, config.MaxWidth / width, config.MaxHeight / height)
    return max(1, _round_half_away(width * ratio)), max(1, _round_half_away(height * ratio)), ratio


def _round_half_away(value: float) -> int:
    whole = int(value)
    return whole + (1 if value - whole >= 0.5 else 0)


def _describe(error: BaseException, image_format: str = "") -> str:
    if isinstance(error, UnidentifiedImageError):
        return f"{image_format}: invalid format"
    text = str(error)
    return text if text else type(error).__name__


def _sniff(header: bytes) -> str | None:
    """Match the magic prefixes Go's image package registers for each format."""
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(header) >= 6 and header[:4] == b"GIF8" and header[5:6] == b"a":
        return "gif"
    if len(header) >= 10 and header[:2] == b"BM" and header[6:10] == b"\x00\x00\x00\x00":
        return "bmp"
    if header.startswith((b"II\x2a\x00", b"MM\x00\x2a")):
        return "tiff"
    if len(header) >= 15 and header[:4] == b"RIFF" and header[8:15] == b"WEBPVP8":
        return "webp"
    return None


def _header_dimensions(image_format: str, reader: _ChunkReader) -> tuple[int, int] | None:
    """Read dimensions from a header Pillow rejects because later data is missing or invalid.

    Go reads only the header when inspecting, so a partial read still reports
    the original dimensions.
    """
    reader.seek(0)
    data = reader.read(64 * 1024)
    try:
        match image_format:
            case "png":
                return _png_dimensions(data)
            case "gif":
                return _gif_dimensions(data)
            case "jpeg":
                return _jpeg_dimensions(data)
    except struct.error, IndexError:
        return None
    return None


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 33 or data[8:16] != b"\x00\x00\x00\x0dIHDR":
        return None
    if zlib.crc32(data[12:29]) != struct.unpack(">I", data[29:33])[0]:
        return None
    width, height = struct.unpack(">ii", data[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 13:
        return None
    width, height = struct.unpack("<HH", data[6:10])
    flags = data[10]
    if flags & 0x80 and len(data) < 13 + 3 * (1 << ((flags & 0x07) + 1)):
        return None
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):
            position += 2
            continue
        (length,) = struct.unpack(">H", data[position + 2 : position + 4])
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[position + 5 : position + 9])
            return width, height
        position += 2 + length
    return None


class _ChunkReader(io.RawIOBase):
    """A seekable reader over read chunks that observes cancellation on every read."""

    def __init__(self, chunks: Sequence[bytes], canceled: CancellationSignal | None = None) -> None:
        super().__init__()
        self._chunks = [chunk for chunk in chunks if chunk]
        self._starts: list[int] = []
        total = 0
        for chunk in self._chunks:
            self._starts.append(total)
            total += len(chunk)
        self._size = total
        self._position = 0
        self._canceled = canceled

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        match whence:
            case io.SEEK_SET:
                position = offset
            case io.SEEK_CUR:
                position = self._position + offset
            case io.SEEK_END:
                position = self._size + offset
            case _:
                raise ValueError(f"invalid whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def readinto(self, buffer: Buffer, /) -> int:
        if self._canceled is not None:
            _check_canceled(self._canceled)
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self._position < self._size:
            index = bisect_right(self._starts, self._position) - 1
            chunk = self._chunks[index]
            start = self._position - self._starts[index]
            count = min(len(chunk) - start, len(view) - written)
            view[written : written + count] = chunk[start : start + count]
            written += count
            self._position += count
        return written


class ImageContentWriter:
    """Accumulates base64 content up to `limit` bytes and keeps counting beyond it."""

    __slots__ = ("_canceled", "_content", "limit", "size")

    def __init__(self, canceled: CancellationSignal, limit: int) -> None:
        self._canceled = canceled
        self.limit = limit
        self.size = 0
        self._content: list[bytes] = []

    def write(self, data: bytes) -> int:
        _check_canceled(self._canceled)
        self.size += len(data)
        if self.size > self.limit:
            # Keep counting to report the exact excess without retaining oversized output.
            self._content.clear()
            return len(data)
        self._content.append(data)
        return len(data)

    def content(self) -> str:
        return b"".join(self._content).decode("ascii")

    def retained(self) -> int:
        return sum(len(part) for part in self._content)


class _Base64Writer:
    """Streams standard base64 into a writer, like Go's `base64.NewEncoder`."""

    __slots__ = ("_pending", "_writer")

    def __init__(self, writer: ImageContentWriter) -> None:
        self._writer = writer
        self._pending = b""

    def write(self, data: bytes | bytearray | memoryview) -> int:
        buffered = self._pending + bytes(data)
        whole = len(buffered) - len(buffered) % 3
        self._pending = buffered[whole:]
        if whole:
            self._writer.write(base64.b64encode(buffered[:whole]))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pending, self._pending = self._pending, b""
        if pending:
            self._writer.write(base64.b64encode(pending))
