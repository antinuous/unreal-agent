import asyncio
import base64
import io
from collections.abc import Callable
from pathlib import Path

import pytest
from msgspec.structs import replace
from PIL import Image
from unreal_agent.gojson import Raw, encode
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    TYPE_SHELL,
    TYPE_VIEW_IMAGE,
    VERSION_VIEW_IMAGE,
    LocalOperationManager,
    Operation,
    Spec,
    Status,
    ViewImageConfig,
    ViewImageResult,
    ViewImageState,
    decode_view_image_state,
)
from unreal_agent.tool import (
    VIEW_IMAGE_NAME,
    CallStatus,
    StaticTranslators,
    TranslationError,
    new_registry,
    viewimage,
)


class RecordingContext:
    def __init__(self) -> None:
        self.specs: list[Spec] = []

    def submit(self, spec: Spec, /) -> str:
        self.specs.append(spec)
        return "image-operation"


def image_operation(status: Status, result: ViewImageResult | None) -> Operation:
    state = ViewImageState(
        Path="image.png",
        Config=ViewImageConfig(MaxSize=1000, MaxWidth=30, MaxHeight=20),
        Result=result,
    )
    return Operation(
        ID="image-operation",
        Type=TYPE_VIEW_IMAGE,
        Version=VERSION_VIEW_IMAGE,
        Status=status,
        State=Raw(encode(state)),
    )


def default_limits(**overrides: int) -> ViewImageConfig:
    limits = ViewImageConfig(
        MaxSize=viewimage.DEFAULT_MAX_SIZE,
        MaxWidth=viewimage.DEFAULT_MAX_WIDTH,
        MaxHeight=viewimage.DEFAULT_MAX_HEIGHT,
    )
    return replace(limits, **overrides)


@pytest.mark.parametrize(
    ("path", "limits", "want_path", "want_limits"),
    [
        pytest.param(
            "image.png",
            ViewImageConfig(),
            "/workspace/image.png",
            default_limits(),
            id="relative path",
        ),
        pytest.param(
            "/outside/image.png",
            ViewImageConfig(MaxSize=1000, MaxWidth=20, MaxHeight=10, MaxSourcePixels=100),
            "/outside/image.png",
            ViewImageConfig(MaxSize=1000, MaxWidth=20, MaxHeight=10, MaxSourcePixels=100),
            id="absolute path",
        ),
        pytest.param(
            "link/../ image.png ",
            ViewImageConfig(MaxWidth=123),
            "/workspace/link/../ image.png ",
            default_limits(MaxWidth=123),
            id="preserve path components",
        ),
    ],
)
def test_translator_submits_image_operation(
    path: str, limits: ViewImageConfig, want_path: str, want_limits: ViewImageConfig
) -> None:
    translator = viewimage.ViewImageTranslator(
        viewimage.Config(directory="/workspace", limits=limits)
    )
    ctx = RecordingContext()
    status = translator.translate(
        ctx, ToolCall(Name=VIEW_IMAGE_NAME, Arguments=encode({"path": path}).decode())
    )
    assert status.Error == ""
    assert status.WaitingFor == ["image-operation"]
    assert len(ctx.specs) == 1
    spec = ctx.specs[0]
    state = decode_view_image_state(
        Operation(Type=spec.Type, Version=spec.Version, State=spec.State)
    )
    assert state.Path == want_path
    assert state.Config == want_limits
    assert state.Result is None


@pytest.mark.parametrize(
    "arguments",
    [
        "",
        "{",
        "null",
        "[]",
        "{}",
        '{"path":null}',
        '{"path":false}',
        '{"path":12}',
        '{"path":""}',
        '{"path":"  "}',
        '{"path":"a\\u0000b"}',
    ],
)
def test_translator_rejects_invalid_arguments(arguments: str) -> None:
    translator = viewimage.ViewImageTranslator(viewimage.Config())
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments=arguments))
    assert status.Error
    assert ctx.specs == []
    assert not status.WaitingFor
    result = translator.translate_result("call", status, [])
    assert result.CallID == "call"
    assert len(result.Output) == 1
    assert result.Output[0].Kind == ToolResultKind.TEXT
    assert status.Error in result.Output[0].Value


def test_translator_rejects_invalid_limits() -> None:
    ctx = RecordingContext()
    translator = viewimage.ViewImageTranslator(viewimage.Config(limits=ViewImageConfig(MaxSize=-1)))
    status = translator.translate(ctx, ToolCall(Arguments='{"path":"image.png"}'))
    assert status.Error
    assert ctx.specs == []


@pytest.mark.parametrize(
    ("original", "encoded", "ratio", "text"),
    [
        pytest.param("image/png", "image/png", 1.0, "", id="unchanged PNG"),
        pytest.param("image/jpeg", "image/jpeg", 1.0, "", id="unchanged JPEG"),
        pytest.param(
            "image/bmp", "image/png", 1.0, "original MIME type: image/bmp", id="converted only"
        ),
        pytest.param(
            "image/jpeg",
            "image/jpeg",
            0.25,
            "original dimensions: 120x80; multiply coordinates by 4.00 to approximate original",
            id="resized only",
        ),
        pytest.param(
            "image/tiff",
            "image/png",
            0.5,
            "original MIME type: image/tiff; original dimensions: 120x80; "
            "multiply coordinates by 2.00 to approximate original",
            id="converted and resized",
        ),
        pytest.param(
            "image/png",
            "image/png",
            1 / 3.25,
            "original dimensions: 120x80; multiply coordinates by 3.25 to approximate original",
            id="fractional coordinates",
        ),
        pytest.param(
            "image/png",
            "image/png",
            1 / 3.456,
            "original dimensions: 120x80; multiply coordinates by 3.46 to approximate original",
            id="rounded coordinates",
        ),
        pytest.param(
            "image/png",
            "image/png",
            1.0 / 120,
            "original dimensions: 120x80; multiply coordinates by 120.00 to approximate original",
            id="large coordinates",
        ),
    ],
)
def test_translator_formats_conditional_image_metadata(
    original: str, encoded: str, ratio: float, text: str
) -> None:
    current = image_operation(
        Status.COMPLETED,
        ViewImageResult(
            Content="aW1hZ2U=",
            OriginalWidth=120,
            OriginalHeight=80,
            OriginalMIMEType=original,
            EncodedMIMEType=encoded,
            ScaleRatio=ratio,
        ),
    )
    result = viewimage.ViewImageTranslator(viewimage.Config()).translate_result(
        "call", CallStatus(), [current]
    )
    want = ToolResult(
        CallID="call",
        Output=[
            ToolResultOutput(Kind=ToolResultKind.IMAGE, Value=f"data:{encoded};base64,aW1hZ2U=")
        ],
    )
    if text:
        want.Output.append(ToolResultOutput(Kind=ToolResultKind.TEXT, Value=text))
    assert result == want


@pytest.mark.parametrize(
    ("status", "result", "want"),
    [
        pytest.param(Status.READY, None, "Image is still loading.", id="ready"),
        pytest.param(Status.AWAITING, None, "Image is still loading.", id="awaiting"),
        pytest.param(Status.CANCELING, None, "Image is still loading.", id="canceling"),
        pytest.param(
            Status.FAILED,
            ViewImageResult(Error="not an image"),
            "Error: not an image",
            id="failed before header",
        ),
        pytest.param(
            Status.FAILED,
            ViewImageResult(
                Content="must not return this",
                OriginalWidth=120,
                OriginalHeight=80,
                OriginalMIMEType="image/png",
                EncodedMIMEType="image/png",
                ScaleRatio=1,
                Error="needs 1200 base64 bytes; MaxSize is 1000 bytes, exceeded by 200 bytes",
            ),
            "Error: needs 1200 base64 bytes; MaxSize is 1000 bytes, exceeded by 200 bytes; "
            "original MIME type: image/png; original dimensions: 120x80",
            id="failed after header",
        ),
        pytest.param(
            Status.CANCELED,
            ViewImageResult(Error="canceled while reading"),
            "Error: canceled while reading",
            id="canceled",
        ),
        pytest.param(
            Status.CANCELED,
            None,
            "Error: view-image operation canceled",
            id="canceled without result",
        ),
        pytest.param(
            Status.FAILED, None, "Error: view-image operation failed", id="failed without result"
        ),
    ],
)
def test_translator_formats_failures_and_pending_operations(
    status: Status, result: ViewImageResult | None, want: str
) -> None:
    got = viewimage.ViewImageTranslator(viewimage.Config()).translate_result(
        "call", CallStatus(), [image_operation(status, result)]
    )
    assert got == ToolResult(
        CallID="call", Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=want)]
    )


_VALID = image_operation(
    Status.COMPLETED, ViewImageResult(Content="aW1hZ2U=", EncodedMIMEType="image/png", ScaleRatio=1)
)


@pytest.mark.parametrize(
    ("status", "operations"),
    [
        pytest.param(CallStatus(), [], id="no operations"),
        pytest.param(CallStatus(), [_VALID, _VALID], id="multiple operations"),
        pytest.param(CallStatus(Error="invalid"), [_VALID], id="validation error with operation"),
        pytest.param(CallStatus(), [replace(_VALID, Type=TYPE_SHELL)], id="wrong type"),
        pytest.param(CallStatus(), [replace(_VALID, State=Raw(b"{"))], id="malformed state"),
        pytest.param(
            CallStatus(),
            [replace(_VALID, Status="unknown")],
            id="unknown status",
        ),
        pytest.param(CallStatus(), [image_operation(Status.COMPLETED, None)], id="missing result"),
        pytest.param(
            CallStatus(),
            [
                image_operation(
                    Status.COMPLETED, ViewImageResult(EncodedMIMEType="image/png", ScaleRatio=1)
                )
            ],
            id="missing content",
        ),
        pytest.param(
            CallStatus(),
            [
                image_operation(
                    Status.COMPLETED, ViewImageResult(Content="data", EncodedMIMEType="image/png")
                )
            ],
            id="invalid ratio",
        ),
    ],
)
def test_translator_rejects_invalid_operation_results(
    status: CallStatus, operations: list[Operation]
) -> None:
    with pytest.raises(TranslationError):
        viewimage.ViewImageTranslator(viewimage.Config()).translate_result(
            "call", status, operations
        )


async def test_view_image_reads_workspace_file_and_returns_resized_data_url(
    tmp_path: Path, new_manager: Callable[[], LocalOperationManager]
) -> None:
    Image.new("RGBA", (12, 8)).save(tmp_path / "image.bmp", format="BMP")
    (tmp_path / "image.bmp").chmod(0o600)
    translator = viewimage.ViewImageTranslator(
        viewimage.Config(directory=str(tmp_path), limits=ViewImageConfig(MaxWidth=6, MaxHeight=6))
    )
    registry = new_registry(StaticTranslators(view_image=translator), VIEW_IMAGE_NAME)
    definitions = registry.static_definitions()
    assert len(definitions) == 1
    assert definitions[0].tool.Name == VIEW_IMAGE_NAME
    assert definitions[0].tool.Parameters["required"] == ["path"]
    assert len(definitions[0].tool.Parameters["properties"]) == 1
    resolved = registry.resolve(VIEW_IMAGE_NAME)
    assert resolved is not None
    ctx = RecordingContext()
    status = resolved.translate(
        ctx, ToolCall(Name=VIEW_IMAGE_NAME, Arguments='{"path":"image.bmp"}')
    )
    assert status.Error == ""
    assert len(ctx.specs) == 1
    spec = ctx.specs[0]
    manager = new_manager()
    await manager.add(
        Operation(
            ID=(status.WaitingFor or [])[0],
            Type=spec.Type,
            Version=spec.Version,
            State=spec.State,
            Status=Status.READY,
        )
    )
    async with asyncio.timeout(5):
        async for current in manager.updates():
            if current.Status in (Status.READY, Status.AWAITING):
                continue
            result = resolved.translate_result("call", status, [current])
            assert current.Status == Status.COMPLETED
            assert len(result.Output) == 2
            assert result.Output[0].Kind == ToolResultKind.IMAGE
            prefix = "data:image/png;base64,"
            assert result.Output[0].Value.startswith(prefix)
            data = base64.b64decode(result.Output[0].Value.removeprefix(prefix), validate=True)
            with Image.open(io.BytesIO(data)) as decoded:
                assert decoded.format == "PNG"
                assert decoded.size == (6, 4)
            assert result.Output[1] == ToolResultOutput(
                Kind=ToolResultKind.TEXT,
                Value="original MIME type: image/bmp; original dimensions: 12x8; "
                "multiply coordinates by 2.00 to approximate original",
            )
            return
    pytest.fail("operation manager stopped before completing the image")
