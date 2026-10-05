import asyncio
import base64
import errno
import io
import os
import struct
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from msgspec.structs import replace
from PIL import Image
from unreal_agent._go import go_quote
from unreal_agent.operation import (
    MAX_VIEW_IMAGE_SOURCE_BYTES,
    LocalOperationManager,
    Operation,
    OperationError,
    Status,
    Step,
    ViewImage,
    ViewImageConfig,
    ViewImageResult,
    decode_view_image_state,
    new_view_image_spec,
)
from unreal_agent.operation.image import (
    VIEW_IMAGE_PROCESS_CORRELATION,
    VIEW_IMAGE_READ_CORRELATION,
    ImageContentWriter,
    _ChunkReader,
)
from unreal_agent.operation.local_manager import _Actor, _RunningOperation
from unreal_agent.primitives import (
    IO_READ_CHUNK_SIZE,
    Channel,
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
    compute,
)
from unreal_agent.primitives._errors import ContextCanceledError

from tests.operation._helpers import Panic

WEBP_FIXTURE = base64.b64decode("UklGRh4AAABXRUJQVlA4TBIAAAAvAkAAAA+w//Mf8x8V5iCi/xE=")
# Two 8x4 frames at (6, 4) on a 24x12 logical screen, encoded by Go's image/gif.
ANIMATED_GIF_FIXTURE = base64.b64decode(
    "R0lGODlhGAAMAIAAAAAAAP///yH/C05FVFNDQVBFMi4wAwEAAAAh+QQACgAAACwGAAQACAAEAAACBYSPqYsFACH5"
    "BAAKAAAALAYABAAIAAQAAAIFhI+piwUAOw=="
)


_PILLOW_FORMATS = {"png": "PNG", "jpeg": "JPEG", "gif": "GIF", "bmp": "BMP", "tiff": "TIFF"}


def view_image_fixture(width: int, height: int) -> Image.Image:
    source = Image.new("RGBA", (width, height))
    source.putdata(
        [
            ((x * 37 + y * 17) & 0xFF, (x * 13 + y * 47) & 0xFF, (x * 71 + y * 19) & 0xFF, 255)
            for y in range(height)
            for x in range(width)
        ]
    )
    return source


def encode_view_image_fixture(source: Image.Image, image_format: str) -> bytes:
    if image_format == "webp":
        return WEBP_FIXTURE
    buffer = io.BytesIO()
    match image_format:
        case "jpeg":
            source.convert("RGB").save(buffer, "JPEG", quality=90)
        case "bmp":
            source.convert("RGB").save(buffer, "BMP")
        case "gif":
            source.convert("RGB").save(buffer, "GIF")
        case _:
            source.save(buffer, _PILLOW_FORMATS[image_format])
    return buffer.getvalue()


def view_image_test_operation(directory: Path, data: bytes, config: ViewImageConfig) -> Operation:
    path = directory / "image.unknown"
    path.write_bytes(data)
    spec = new_view_image_spec(str(path), config)
    return Operation(
        ID="view-image", Type=spec.Type, Version=spec.Version, State=spec.State, Status=Status.READY
    )


async def run_view_image_test(current: Operation) -> tuple[Operation, ViewImageResult]:
    async with LocalOperationManager() as manager, asyncio.timeout(10):
        await manager.add(current)
        async for updated in manager.updates():
            state = decode_view_image_state(updated)
            if updated.Status not in (Status.COMPLETED, Status.FAILED, Status.CANCELED):
                assert state.Result is None, "intermediate checkpoint contains an image result"
                continue
            assert state.Result is not None, "terminal state is missing result"
            return updated, state.Result
    raise AssertionError("manager stopped before image completed")


def decode_view_image_content(
    result: ViewImageResult, want_format: str
) -> tuple[bytes, Image.Image]:
    data = base64.b64decode(result.Content, validate=True)
    decoded = Image.open(io.BytesIO(data))
    decoded.load()
    assert decoded.format == want_format.upper(), (decoded.format, result.Error)
    assert result.EncodedMIMEType == "image/" + want_format
    return data, decoded


def view_image_test_actor(current: Operation) -> ViewImage:
    return ViewImage(current)


def view_image_compute_request(step: Step) -> ComputeRequest:
    assert step.operation is None, step
    assert len(step.dispatches) == 1, step
    assert step.dispatches[0].type == PrimitiveDispatchType.COMPUTE
    request = step.dispatches[0].data
    assert isinstance(request, ComputeRequest)
    assert request.run is not None
    return request


def run_view_image_compute(step: Step) -> PrimitiveEvent:
    request = view_image_compute_request(step)
    assert request.run is not None
    value = request.run(threading.Event())
    return PrimitiveEvent(
        PrimitiveEventType.COMPUTE_COMPLETED,
        request.source,
        request.correlation_id,
        ComputeResult(value),
    )


@pytest.mark.parametrize("image_format", ["jpeg", "png", "bmp", "tiff", "webp"])
async def test_view_image_recognizes_formats_from_content(
    tmp_path: Path, image_format: str
) -> None:
    source = view_image_fixture(24, 12)
    data = encode_view_image_fixture(source, image_format)
    width, height = (3, 2) if image_format == "webp" else (24, 12)
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=100_000, MaxWidth=100, MaxHeight=100)
    )
    completed, result = await run_view_image_test(current)
    assert completed.Status == Status.COMPLETED
    assert result.Error == ""
    assert (result.OriginalWidth, result.OriginalHeight) == (width, height)
    assert result.OriginalMIMEType == "image/" + image_format
    assert result.ScaleRatio == 1
    want_format = "jpeg" if image_format == "jpeg" else "png"
    encoded, actual = decode_view_image_content(result, want_format)
    assert actual.size == (width, height)
    if image_format == "jpeg":
        assert encoded == data, "unresized JPEG was re-encoded"
    if image_format in ("png", "bmp", "tiff"):
        assert actual.convert("RGBA").tobytes() == source.tobytes()


@pytest.mark.parametrize("name", ["static", "animated"])
async def test_view_image_rejects_gif(tmp_path: Path, name: str) -> None:
    data = (
        encode_view_image_fixture(view_image_fixture(24, 12), "gif")
        if name == "static"
        else ANIMATED_GIF_FIXTURE
    )
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=12, MaxHeight=12)
    )
    completed, result = await run_view_image_test(current)
    assert completed.Status == Status.FAILED
    assert result.Content == ""
    assert result.EncodedMIMEType == ""
    assert (result.OriginalWidth, result.OriginalHeight) == (24, 12)
    assert result.OriginalMIMEType == "image/gif"
    assert result.ScaleRatio == 1
    assert result.Error == 'unsupported image format "gif"'


@pytest.mark.parametrize(
    ("width", "height", "max_width", "max_height", "want_width", "want_height", "ratio"),
    [
        pytest.param(120, 60, 40, 40, 40, 20, 1 / 3, id="landscape"),
        pytest.param(60, 120, 40, 40, 20, 40, 1 / 3, id="portrait"),
        pytest.param(60, 120, 10, 100, 10, 20, 1 / 6, id="width limits portrait"),
        pytest.param(120, 60, 100, 10, 20, 10, 1 / 6, id="height limits landscape"),
        pytest.param(60, 60, 30, 40, 30, 30, 0.5, id="square"),
        pytest.param(30, 20, 30, 40, 30, 20, 1, id="already fits"),
        pytest.param(3, 2, 100, 100, 3, 2, 1, id="no upscale"),
        pytest.param(7, 3, 4, 4, 4, 2, 4 / 7, id="round pixels"),
        pytest.param(100, 1, 1, 1, 1, 1, 0.01, id="thin image"),
    ],
)
@pytest.mark.parametrize("image_format", ["png", "jpeg"])
async def test_view_image_resizes_to_both_bounds(
    tmp_path: Path,
    image_format: str,
    width: int,
    height: int,
    max_width: int,
    max_height: int,
    want_width: int,
    want_height: int,
    ratio: float,
) -> None:
    data = encode_view_image_fixture(view_image_fixture(width, height), image_format)
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=100_000, MaxWidth=max_width, MaxHeight=max_height)
    )
    completed, result = await run_view_image_test(current)
    assert completed.Status == Status.COMPLETED, result.Error
    _, actual = decode_view_image_content(result, image_format)
    assert actual.size == (want_width, want_height)
    assert result.ScaleRatio == ratio
    assert (result.OriginalWidth, result.OriginalHeight) == (width, height)


async def test_view_image_preserves_transparency(tmp_path: Path) -> None:
    source = Image.new("RGBA", (8, 4), (120, 60, 30, 128))
    data = encode_view_image_fixture(source, "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=4, MaxHeight=4)
    )
    _, result = await run_view_image_test(current)
    _, decoded = decode_view_image_content(result, "png")
    pixel = decoded.convert("RGBA").getpixel((1, 1))
    assert isinstance(pixel, tuple)
    assert pixel[3] == 128


@pytest.mark.parametrize("image_format", ["jpeg", "bmp"])
async def test_view_image_max_size_counts_base64_after_resizing(
    tmp_path_factory: pytest.TempPathFactory, image_format: str
) -> None:
    data = encode_view_image_fixture(view_image_fixture(120, 80), image_format)
    config = ViewImageConfig(MaxSize=100_000, MaxWidth=30, MaxHeight=30)
    _, baseline = await run_view_image_test(
        view_image_test_operation(tmp_path_factory.mktemp("image"), data, config)
    )
    size = len(baseline.Content)
    assert size, baseline.Error
    for limit in (size, size - 1, 1):
        config.MaxSize = limit
        completed, result = await run_view_image_test(
            view_image_test_operation(tmp_path_factory.mktemp("image"), data, config)
        )
        if limit == size:
            assert completed.Status == Status.COMPLETED, result.Error
            assert result.Content == baseline.Content
            continue
        assert completed.Status == Status.FAILED
        assert result.Content == ""
        assert (result.OriginalWidth, result.OriginalHeight) == (120, 80)
        assert result.OriginalMIMEType == "image/" + image_format
        assert result.EncodedMIMEType == baseline.EncodedMIMEType
        assert result.ScaleRatio == 0.25
        for text in (
            "120x80 to 30x20",
            f"{size} base64 bytes",
            f"MaxSize is {limit} bytes",
            f"exceeded by {size - limit} bytes",
        ):
            assert text in result.Error


async def test_view_image_reports_exact_max_size_message(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(120, 80), "bmp")
    config = ViewImageConfig(MaxSize=100_000, MaxWidth=30, MaxHeight=30)
    _, baseline = await run_view_image_test(view_image_test_operation(tmp_path, data, config))
    size = len(baseline.Content)
    config.MaxSize = 3
    _, result = await run_view_image_test(view_image_test_operation(tmp_path, data, config))
    assert result.Error == (
        f"image/png image after resizing 120x80 to 30x20 (scale 0.25) needs {size} base64 bytes; "
        f"MaxSize is 3 bytes, exceeded by {size - 3} bytes ({100 * (size - 3) / 3:.1f}%)"
    )


async def test_view_image_rejects_oversized_source_before_decoding(tmp_path: Path) -> None:
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, 54)
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<I", header, 18, 1_000_000_000)
    struct.pack_into("<I", header, 22, 1_000_000_000)
    struct.pack_into("<H", header, 26, 1)
    struct.pack_into("<H", header, 28, 24)
    current = view_image_test_operation(
        tmp_path, bytes(header), ViewImageConfig(MaxSize=1000, MaxWidth=1, MaxHeight=1)
    )
    completed, result = await run_view_image_test(current)
    assert completed.Status == Status.FAILED
    assert result.Content == ""
    assert result.EncodedMIMEType == ""
    assert (result.OriginalWidth, result.OriginalHeight) == (1_000_000_000, 1_000_000_000)
    assert result.OriginalMIMEType == "image/bmp"
    for message in (
        "refuse to decode 1000000000x1000000000 bmp image",
        "MaxSourcePixels 32000000",
    ):
        assert message in result.Error


@pytest.mark.parametrize("max_pixels", [31, 32, 33])
async def test_view_image_source_pixel_budget_boundary(tmp_path: Path, max_pixels: int) -> None:
    data = encode_view_image_fixture(view_image_fixture(8, 4), "bmp")
    current = view_image_test_operation(
        tmp_path,
        data,
        ViewImageConfig(MaxSize=1000, MaxWidth=1, MaxHeight=1, MaxSourcePixels=max_pixels),
    )
    completed, result = await run_view_image_test(current)
    if max_pixels < 32:
        assert completed.Status == Status.FAILED
        assert "MaxSourcePixels 31" in result.Error
        assert result.ScaleRatio == 1
        return
    assert completed.Status == Status.COMPLETED
    assert result.ScaleRatio == 0.125
    _, decoded = decode_view_image_content(result, "png")
    assert decoded.size == (1, 1)


@pytest.mark.parametrize(
    ("size", "png", "status"),
    [
        pytest.param(MAX_VIEW_IMAGE_SOURCE_BYTES - 1, True, Status.COMPLETED, id="below limit"),
        pytest.param(MAX_VIEW_IMAGE_SOURCE_BYTES, True, Status.COMPLETED, id="at limit"),
        pytest.param(MAX_VIEW_IMAGE_SOURCE_BYTES + 1, True, Status.FAILED, id="oversized PNG"),
        pytest.param(
            MAX_VIEW_IMAGE_SOURCE_BYTES + 123, False, Status.FAILED, id="oversized invalid file"
        ),
    ],
)
def test_view_image_source_byte_limit(tmp_path: Path, size: int, png: bool, status: Status) -> None:
    png_data = encode_view_image_fixture(view_image_fixture(1, 1), "png")
    data = png_data if png else b"not an image"
    current = view_image_test_operation(
        tmp_path,
        data,
        ViewImageConfig(MaxSize=1024, MaxWidth=1, MaxHeight=1, MaxSourcePixels=1),
    )
    view = view_image_test_actor(current)
    start = view.handle(None)
    request = start.dispatches[0].data
    assert isinstance(request, IOReadRequest)
    assert (request.count, request.offset) == (256 * 1024 * 1024, 0)
    # Reuse immutable padding to exercise the boundary without allocating 256 MiB.
    padding = bytes(IO_READ_CHUNK_SIZE)
    chunk = data
    read_size = min(size, request.count)
    offset = 0
    while offset < read_size:
        chunk = chunk[: min(len(chunk), read_size - offset)]
        step = view.handle(
            PrimitiveEvent(
                PrimitiveEventType.IO_READ_OUTPUT,
                request.source,
                request.correlation_id,
                IOReadOutputResult(offset, chunk),
            )
        )
        assert step.operation is None
        assert not view._processing
        offset += len(chunk)
        chunk = padding
    assert view._read_size == read_size
    processing = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            request.source,
            request.correlation_id,
            IOReadCompletedResult(size),
        )
    )
    finished = view.handle(run_view_image_compute(processing))
    assert finished.operation is not None
    assert finished.operation.Status == status
    result = view.state.Result
    assert result is not None
    if png:
        assert (result.OriginalWidth, result.OriginalHeight) == (1, 1)
        assert result.OriginalMIMEType == "image/png"
    if status == Status.COMPLETED:
        decode_view_image_content(result, "png")
        return
    assert result.Content == ""
    assert result.EncodedMIMEType == ""
    for message in (
        f"image source is {size} bytes",
        "source limit is 268435456 bytes",
        f"exceeded by {size - MAX_VIEW_IMAGE_SOURCE_BYTES} bytes",
    ):
        assert message in result.Error


@pytest.mark.parametrize(
    ("name", "message", "truncate", "width", "height"),
    [
        ("empty", "decode image header", 0, 0, 0),
        ("unknown", "unknown format", -1, 0, 0),
        ("truncated pixels", "decode png pixels", 33, 17, 11),
    ],
)
async def test_view_image_failures_retain_available_metadata(
    tmp_path: Path, name: str, message: str, truncate: int, width: int, height: int
) -> None:
    data = encode_view_image_fixture(view_image_fixture(17, 11), "png")
    source = b"not an image" if truncate < 0 else data[:truncate]
    current = view_image_test_operation(
        tmp_path, source, ViewImageConfig(MaxSize=1000, MaxWidth=100, MaxHeight=100)
    )
    completed, result = await run_view_image_test(current)
    assert completed.Status == Status.FAILED
    assert result.Content == ""
    assert (result.OriginalWidth, result.OriginalHeight) == (width, height), name
    assert message in result.Error
    assert "MaxSize" not in result.Error


async def test_view_image_reports_missing_file(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(17, 11), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=100, MaxHeight=100)
    )
    state = decode_view_image_state(current)
    os.remove(state.Path)
    completed, result = await run_view_image_test(current)
    want = (
        f"read image: open {go_quote(state.Path)}: open {state.Path}: "
        f"{os.strerror(errno.ENOENT).lower()}"
    )
    assert completed.Status == Status.FAILED
    assert result.Error == want


@pytest.mark.parametrize(
    ("path", "config"),
    [
        ("", ViewImageConfig(MaxSize=1, MaxWidth=1, MaxHeight=1)),
        ("image", ViewImageConfig(MaxSize=0, MaxWidth=1, MaxHeight=1)),
        ("image", ViewImageConfig(MaxSize=1, MaxWidth=0, MaxHeight=1)),
        ("image", ViewImageConfig(MaxSize=1, MaxWidth=1, MaxHeight=0)),
        ("image", ViewImageConfig(MaxSize=-1, MaxWidth=1, MaxHeight=1)),
        ("image", ViewImageConfig(MaxSize=1, MaxWidth=-1, MaxHeight=1)),
        ("image", ViewImageConfig(MaxSize=1, MaxWidth=1, MaxHeight=-1)),
        ("image", ViewImageConfig(MaxSize=1, MaxWidth=1, MaxHeight=1, MaxSourcePixels=-1)),
    ],
)
def test_view_image_spec_rejects_invalid_configuration(path: str, config: ViewImageConfig) -> None:
    with pytest.raises(OperationError):
        new_view_image_spec(path, config)


def test_local_operation_manager_view_image_handle_failure() -> None:
    spec = new_view_image_spec(
        "image.jpg", ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view = view_image_test_actor(
        Operation(
            ID="view-image",
            Type=spec.Type,
            Version=spec.Version,
            State=spec.State,
            Status=Status.READY,
        )
    )
    start = view.handle(None)
    assert start.operation is not None
    original = bytes(start.operation.State)
    current = _RunningOperation(operation=start.operation, handle=view.handle)
    with pytest.raises(OperationError) as handle_error:
        current.handle(
            PrimitiveEvent(
                PrimitiveEventType.COMPUTE_COMPLETED,
                SourceID(current.operation.ID),
                VIEW_IMAGE_PROCESS_CORRELATION,
            )
        )
    actor = _Actor()
    actor.operations[current.operation.ID] = current
    actor.fail_local_operation(current, handle_error.value)
    assert actor.updates.qsize() == 1
    assert actor.operations == {}
    assert current.canceled
    failed = actor.updates.get_nowait()
    state = decode_view_image_state(failed)
    assert failed.Status == Status.FAILED
    assert state.Result == ViewImageResult(ScaleRatio=1, Error=str(handle_error.value))
    assert bytes(start.operation.State) == original


def test_image_content_writer_counts_beyond_limit() -> None:
    writer = ImageContentWriter(threading.Event(), 5)
    for data in (b"abc", b"de", b"fghi", b"j"):
        assert writer.write(data) == len(data)
        assert writer.retained() <= writer.limit, "retained oversized content"
    assert writer.size == 10
    assert writer.retained() == 0


def test_image_reader_and_writer_honor_cancellation() -> None:
    canceled = threading.Event()
    canceled.set()
    reader = _ChunkReader([b"image"], canceled)
    with pytest.raises(ContextCanceledError):
        reader.read()
    writer = ImageContentWriter(canceled, 100)
    with pytest.raises(ContextCanceledError):
        writer.write(b"image")


class PanicSignal:
    def is_set(self) -> bool:
        raise Panic("injected reader panic")


def _read_event(current: Operation, data: bytes, offset: int = 0) -> PrimitiveEvent:
    return PrimitiveEvent(
        PrimitiveEventType.IO_READ_OUTPUT,
        SourceID(current.ID),
        VIEW_IMAGE_READ_CORRELATION,
        IOReadOutputResult(offset, data),
    )


def view_image_ready_to_compute(current: Operation, data: bytes) -> tuple[ViewImage, Step]:
    view = view_image_test_actor(current)
    view.handle(None)
    view.handle(_read_event(current, data))
    step = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            SourceID(current.ID),
            VIEW_IMAGE_READ_CORRELATION,
            IOReadCompletedResult(len(data)),
        )
    )
    return view, step


def assert_view_image_canceled(step: Step) -> None:
    assert step.operation is not None
    assert step.operation.Status == Status.CANCELED
    assert not step.dispatches
    state = decode_view_image_state(step.operation)
    assert state.Result is not None
    assert state.Result.Content == ""
    assert state.Result.Error == "view-image operation canceled: context canceled"


async def test_view_image_dispatches_compute_and_recovery_rereads(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(160, 120), "bmp")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=16, MaxHeight=12)
    )
    view = view_image_test_actor(current)
    start = view.handle(None)
    assert start.operation is not None
    request = start.dispatches[0].data
    assert isinstance(request, IOReadRequest)
    original = bytes(start.operation.State)
    for offset in range(0, len(data), 1024):
        step = view.handle(_read_event(current, data[offset : offset + 1024], offset))
        assert step.operation is None
        assert not step.dispatches
    processing = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            request.source,
            request.correlation_id,
            IOReadCompletedResult(len(data)),
        )
    )
    compute_request = view_image_compute_request(processing)
    assert compute_request.source == request.source
    assert compute_request.correlation_id == VIEW_IMAGE_PROCESS_CORRELATION
    assert view._chunks == []
    assert view._read_size == 0
    assert view._processing
    assert view.state.Result is None

    # The callback owns only transient inputs, even if the actor changes before it runs.
    config = replace(view.state.Config)
    view.state.Config.MaxSize = 1
    event = run_view_image_compute(processing)
    view.state.Config = config
    assert view.state.Result is None
    assert view._current.Status == Status.AWAITING
    assert bytes(start.operation.State) == original
    completed = view.handle(event)
    assert completed.operation is not None
    assert completed.operation.Status == Status.COMPLETED
    assert not completed.dispatches
    result = view.state.Result
    assert result is not None
    _, decoded = decode_view_image_content(result, "png")
    assert decoded.size == (16, 12)
    assert len(result.Content) > 1
    with pytest.raises(OperationError):
        view.handle(event)

    recovered = view_image_test_actor(start.operation)
    restarted = recovered.handle(None)
    assert restarted.dispatches[0].data == request
    resumed, resumed_result = await run_view_image_test(start.operation)
    assert resumed.Status == Status.COMPLETED
    assert resumed_result.OriginalWidth == 160
    assert resumed_result.ScaleRatio == 0.1


@pytest.mark.parametrize("phase", ["reading", "computing", "converted"])
def test_view_image_cancellation_does_not_dispatch_compute(tmp_path: Path, phase: str) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    correlation = VIEW_IMAGE_READ_CORRELATION
    if phase == "reading":
        view = view_image_test_actor(current)
        view.handle(None)
        view.handle(_read_event(current, data))
    else:
        view, step = view_image_ready_to_compute(current, data)
        correlation = VIEW_IMAGE_PROCESS_CORRELATION
        if phase == "converted":
            run_view_image_compute(step)
    step = view.handle(
        PrimitiveEvent(PrimitiveEventType.CANCELED, SourceID(current.ID), correlation)
    )
    assert_view_image_canceled(step)
    assert view._chunks == []
    assert view._read_size == 0
    assert not view._processing


def test_view_image_recovers_canceling_status(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    recovered = view_image_test_actor(replace(current, Status=Status.CANCELING))
    assert_view_image_canceled(recovered.handle(None))


@pytest.mark.parametrize("phase", ["reading", "computing"])
def test_view_image_validates_events_before_updating_state(tmp_path: Path, phase: str) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view = view_image_test_actor(current)
    start = view.handle(None)
    assert start.operation is not None
    correlation = VIEW_IMAGE_READ_CORRELATION
    if phase == "computing":
        view, _ = view_image_ready_to_compute(current, data)
        correlation = VIEW_IMAGE_PROCESS_CORRELATION
        with pytest.raises(OperationError):
            view.handle(None)
    original = bytes(start.operation.State)
    for kind in (
        PrimitiveEventType.COMPUTE_COMPLETED,
        PrimitiveEventType.FAILED,
        PrimitiveEventType.CANCELED,
    ):
        for scenario in ("wrong source", "wrong correlation", "invalid payload"):
            event = PrimitiveEvent(kind, SourceID(current.ID), correlation)
            match scenario:
                case "wrong source":
                    event.source = SourceID("another-operation")
                case "wrong correlation":
                    event.correlation_id = CorrelationID("another-invocation")
                case _:
                    event.result = IOReadCompletedResult(0)
            with pytest.raises(OperationError):
                view.handle(event)
            assert view.state.Result is None, (kind, scenario)
    if phase == "computing":
        for event in (
            PrimitiveEvent(
                PrimitiveEventType.COMPUTE_COMPLETED,
                result=ComputeResult("not an image result"),
            ),
            PrimitiveEvent(
                PrimitiveEventType.IO_READ_COMPLETED, result=IOReadCompletedResult(len(data))
            ),
        ):
            event.source, event.correlation_id = SourceID(current.ID), correlation
            with pytest.raises(OperationError):
                view.handle(event)
    assert bytes(start.operation.State) == original


def test_view_image_read_failure_without_data_does_not_dispatch_compute(tmp_path: Path) -> None:
    current = view_image_test_operation(
        tmp_path, b"", ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view = view_image_test_actor(current)
    view.handle(None)
    step = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.FAILED,
            SourceID(current.ID),
            VIEW_IMAGE_READ_CORRELATION,
            PrimitiveFailureResult("file not found"),
        )
    )
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    state = decode_view_image_state(step.operation)
    assert state.Result is not None
    assert state.Result.Error == "read image: file not found"


def test_view_image_read_failure_retains_partial_header_dimensions(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view = view_image_test_actor(current)
    view.handle(None)
    view.handle(_read_event(current, data[:33]))
    processing = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.FAILED,
            SourceID(current.ID),
            VIEW_IMAGE_READ_CORRELATION,
            PrimitiveFailureResult("read interrupted"),
        )
    )
    step = view.handle(run_view_image_compute(processing))
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    result = view.state.Result
    assert result is not None
    assert (result.OriginalWidth, result.OriginalHeight) == (30, 20)
    assert result.OriginalMIMEType == "image/png"
    assert result.Error == "read image: read interrupted"


async def test_view_image_compute_reports_processing_panic(tmp_path: Path) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view, processing = view_image_ready_to_compute(current, data)
    request = view_image_compute_request(processing)
    run = cast(Any, request.run)

    def panicking(_: threading.Event) -> object:
        return run(PanicSignal())

    request.run = panicking
    events: Channel[PrimitiveEvent] = Channel(1)
    compute(request, events)
    async with asyncio.timeout(10):
        event = await events.get()
    assert event.type == PrimitiveEventType.FAILED, event
    step = view.handle(event)
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    result = view.state.Result
    assert result is not None
    assert result.Content == ""
    assert (
        result.Error == "image primitive failed: compute callback panicked: injected reader panic"
    )


@pytest.mark.parametrize("correlation", [VIEW_IMAGE_PROCESS_CORRELATION, CorrelationID("")])
def test_view_image_compute_failures_terminate_without_redispatch(
    tmp_path: Path, correlation: CorrelationID
) -> None:
    data = encode_view_image_fixture(view_image_fixture(30, 20), "png")
    current = view_image_test_operation(
        tmp_path, data, ViewImageConfig(MaxSize=1000, MaxWidth=10, MaxHeight=10)
    )
    view, _ = view_image_ready_to_compute(current, data)
    step = view.handle(
        PrimitiveEvent(
            PrimitiveEventType.FAILED,
            SourceID(current.ID),
            correlation,
            PrimitiveFailureResult("compute failed"),
        )
    )
    assert step.operation is not None
    assert step.operation.Status == Status.FAILED
    assert not step.dispatches
    state = decode_view_image_state(step.operation)
    assert state.Result is not None
    assert state.Result.Content == ""
    assert state.Result.Error == "image primitive failed: compute failed"
