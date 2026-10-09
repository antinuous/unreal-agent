from enum import StrEnum
from typing import Annotated

from unreal_agent._go import go_trim_space, json_quote
from unreal_agent.gojson import Int64, JSONError, OmitZero, Raw, Struct, decode, is_valid_json
from unreal_agent.llm import ReasoningEffort

__all__ = [
    "ID",
    "ControlMessage",
    "ControlMode",
    "Input",
    "InputKind",
    "Settings",
]

type ID = str


class InputKind(StrEnum):
    NONE = ""
    EXTERNAL = "external"
    CONTROL = "control"
    CRASH = "crash"


class ControlMode(StrEnum):
    NONE = ""
    STOP_HARD = "hard"
    STOP_WHEN_IDLE = "when_idle"
    HEARTBEAT = "heartbeat"
    UPDATE_SETTINGS = "settings"


_ReasoningEffort = ReasoningEffort


class Settings(Struct):
    SystemPrompt: Annotated[str | None, OmitZero] = None
    Model: Annotated[str, OmitZero] = ""
    MaxOutputTokens: Annotated[Int64 | None, OmitZero] = None
    ReasoningEffort: Annotated[_ReasoningEffort, OmitZero] = _ReasoningEffort.NONE


class ControlMessage(Struct):
    Mode: ControlMode = ControlMode.NONE
    Reason: str = ""
    Parameters: Annotated[Settings | None, OmitZero] = None


class _ControlEnvelope(Struct, forbid_unknown_fields=True):
    Mode: str | None = None
    Reason: str | None = None
    Parameters: Raw = Raw(b"")


class _SettingsParameters(Struct, forbid_unknown_fields=True):
    SystemPrompt: str | None = None
    Model: str | None = None
    MaxOutputTokens: Int64 | None = None
    ReasoningEffort: str | None = None


class Input(Struct):
    ID: str = ""
    Kind: InputKind = InputKind.NONE
    Payload: Annotated[Raw, OmitZero] = Raw(b"")

    def validate(self) -> None:
        if self.ID == "":
            raise ValueError("input ID is empty")
        match self.Kind:
            case InputKind.EXTERNAL | InputKind.CRASH:
                pass
            case InputKind.CONTROL:
                self.decode_control_message()
            case _:
                raise ValueError(
                    f"input {json_quote(self.ID)} has unsupported kind {json_quote(self.Kind)}"
                )
        if len(self.Payload) != 0 and not is_valid_json(bytes(self.Payload)):
            raise ValueError(f"input {json_quote(self.ID)} payload is not valid JSON")

    def decode_control_message(self) -> ControlMessage:
        """Decode a control payload, rejecting unknown members. Raises ValueError."""
        if self.Kind != InputKind.CONTROL:
            raise ValueError(f"control message input has kind {json_quote(self.Kind)}")
        try:
            envelope = decode(bytes(self.Payload), _ControlEnvelope | None) or _ControlEnvelope()
        except JSONError as error:
            raise ValueError(f"decode control message: {error}") from error
        mode = envelope.Mode or ""
        reason = envelope.Reason or ""
        if mode != ControlMode.UPDATE_SETTINGS and len(envelope.Parameters) != 0:
            raise ValueError(f"control mode {json_quote(mode)} does not accept parameters")
        match mode:
            case ControlMode.STOP_HARD | ControlMode.STOP_WHEN_IDLE:
                return ControlMessage(Mode=ControlMode(mode), Reason=reason)
            case ControlMode.HEARTBEAT:
                if reason == "":
                    raise ValueError("heartbeat reason is empty")
                return ControlMessage(Mode=ControlMode.HEARTBEAT, Reason=reason)
            case ControlMode.UPDATE_SETTINGS:
                raw = bytes(envelope.Parameters)
                if len(raw) == 0 or raw == b"null":
                    raise ValueError("settings parameters are missing")
                try:
                    parameters = decode(raw, _SettingsParameters)
                except JSONError as error:
                    raise ValueError(f"decode settings parameters: {error}") from error
                effort = parameters.ReasoningEffort or ""
                if effort and not ReasoningEffort.is_valid(effort):
                    raise ValueError(f"unsupported reasoning effort {json_quote(effort)}")
                model = parameters.Model or ""
                if model and not go_trim_space(model):
                    raise ValueError("model ID is empty")
                if parameters.MaxOutputTokens is not None and parameters.MaxOutputTokens <= 0:
                    raise ValueError("max output tokens must be positive")
                return ControlMessage(
                    Mode=ControlMode.UPDATE_SETTINGS,
                    Reason=reason,
                    Parameters=Settings(
                        SystemPrompt=parameters.SystemPrompt,
                        Model=model,
                        MaxOutputTokens=parameters.MaxOutputTokens,
                        ReasoningEffort=ReasoningEffort(effort),
                    ),
                )
            case _:
                raise ValueError(f"unsupported control mode {json_quote(mode)}")
