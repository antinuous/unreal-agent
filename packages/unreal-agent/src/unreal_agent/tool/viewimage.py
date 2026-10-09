from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as replace_dataclass
from typing import Final

from msgspec.structs import replace

from unreal_agent._go import go_bytes, go_quote, go_trim_space
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    Operation,
    OperationError,
    Status,
    ViewImageConfig,
    ViewImageResult,
    decode_view_image_state,
    new_view_image_spec,
)
from unreal_agent.tool._jsonv2 import GoJSONError, decode_string_field
from unreal_agent.tool.output import error_status
from unreal_agent.tool.tool import CallStatus, Context, TranslationError

__all__ = [
    "DEFAULT_MAX_HEIGHT",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_MAX_WIDTH",
    "Config",
    "Result",
    "ViewImageTranslator",
]

DEFAULT_MAX_SIZE: Final = 5_000_000 - 1_000
DEFAULT_MAX_WIDTH: Final = 2000
DEFAULT_MAX_HEIGHT: Final = 2000

_ARGUMENTS_TYPE: Final = 'struct { Path string "json:\\"path\\"" }'


@dataclass(frozen=True, slots=True)
class Config:
    directory: str = ""
    limits: ViewImageConfig = field(default_factory=ViewImageConfig)


@dataclass(frozen=True, slots=True)
class Result:
    CallID: str = ""
    Running: bool = False
    Image: ViewImageResult | None = None
    Error: str = ""

    def to_llm_result(self) -> ToolResult:
        output: list[ToolResultOutput] = []
        details: list[str] = []
        if self.Running:
            details.append("Image is still loading.")
        else:
            failed = self.Error != ""
            if failed:
                details.append("Error: " + self.Error)
            if (image := self.Image) is not None:
                if not failed:
                    output.append(
                        ToolResultOutput(
                            Kind=ToolResultKind.IMAGE,
                            Value=f"data:{image.EncodedMIMEType};base64,{image.Content}",
                        )
                    )
                if image.OriginalMIMEType and (
                    failed or image.OriginalMIMEType != image.EncodedMIMEType
                ):
                    details.append("original MIME type: " + image.OriginalMIMEType)
                if (
                    image.OriginalWidth > 0
                    and image.OriginalHeight > 0
                    and (failed or image.ScaleRatio < 1)
                ):
                    details.append(
                        f"original dimensions: {image.OriginalWidth}x{image.OriginalHeight}"
                    )
                if not failed and image.ScaleRatio < 1:
                    details.append(
                        f"multiply coordinates by {1 / image.ScaleRatio:.2f} "
                        "to approximate original"
                    )
        if details:
            output.append(ToolResultOutput(Kind=ToolResultKind.TEXT, Value="; ".join(details)))
        return ToolResult(CallID=self.CallID, Output=output, Running=self.Running)


class ViewImageTranslator:
    """Zero MaxSize, MaxWidth, and MaxHeight limits take the package defaults."""

    __slots__ = ("_directory", "_limits")

    def __init__(self, config: Config) -> None:
        limits = config.limits
        self._directory = config.directory
        self._limits = replace(
            limits,
            MaxSize=limits.MaxSize or DEFAULT_MAX_SIZE,
            MaxWidth=limits.MaxWidth or DEFAULT_MAX_WIDTH,
            MaxHeight=limits.MaxHeight or DEFAULT_MAX_HEIGHT,
        )

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        try:
            path = decode_string_field(go_bytes(call.Arguments), "path", _ARGUMENTS_TYPE)
        except GoJSONError as error:
            return error_status(f"decode ViewImage arguments: {error}", DEFAULT_MAX_OUTPUT_LENGTH)
        if go_trim_space(path) == "":
            return CallStatus(Error='ViewImage argument "path" must be set')
        offset = go_bytes(path).find(b"\x00")
        if offset >= 0:
            return CallStatus(
                Error=f'ViewImage argument "path" contains a NUL byte at offset {offset}'
            )
        if self._directory and not path.startswith("/"):
            # Preserve .. for filesystem resolution across symlinks; joining would clean it.
            path = self._directory + "/" + path
        try:
            spec = new_view_image_spec(path, replace(self._limits))
        except OperationError as error:
            return CallStatus(Error=f"build ViewImage operation: {error}")
        return CallStatus(WaitingFor=[ctx.submit(spec)])

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> Result:
        if status.Error:
            if operations:
                raise TranslationError(
                    f"ViewImage call {go_quote(call_id)} has both a validation error and operations"
                )
            return Result(CallID=call_id, Error=status.Error)
        if len(operations) != 1:
            raise TranslationError(
                f"ViewImage call {go_quote(call_id)} has {len(operations)} operations, want 1"
            )
        current = operations[0]
        try:
            state = decode_view_image_state(current)
        except OperationError as error:
            raise TranslationError(
                f"decode ViewImage call {go_quote(call_id)} result: {error}"
            ) from error
        result = Result(CallID=call_id, Image=state.Result)
        match current.Status:
            case Status.READY | Status.AWAITING | Status.CANCELING:
                return replace_dataclass(result, Running=True)
            case Status.FAILED | Status.CANCELED:
                error = "view-image operation " + current.Status
                if state.Result is not None and state.Result.Error:
                    error = state.Result.Error
                return replace_dataclass(result, Error=error)
            case Status.COMPLETED:
                image = state.Result
                if (
                    image is None
                    or not image.Content
                    or not image.EncodedMIMEType
                    or image.ScaleRatio <= 0
                    or image.ScaleRatio > 1
                    or image.Error
                ):
                    raise TranslationError(
                        f"ViewImage call {go_quote(call_id)} completed operation "
                        f"{go_quote(current.ID)} has an invalid image result"
                    )
            case _:
                raise TranslationError(
                    f"ViewImage call {go_quote(call_id)} operation {go_quote(current.ID)} "
                    f"has invalid status {go_quote(current.Status)}"
                )
        return result
