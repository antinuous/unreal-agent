import json
from enum import StrEnum
from typing import Annotated, Protocol

from unreal_agent.gojson import JSONError, OmitZero, Raw, Struct, decode, is_valid_json
from unreal_agent.llm import ReasoningEffort

__all__ = [
    "ID",
    "ControlMessage",
    "ControlMode",
    "Input",
    "InputKind",
    "Settings",
    "Writer",
]

type ID = str


class InputKind(StrEnum):
    EXTERNAL = "external"
    CONTROL = "control"
    CRASH = "crash"


class ControlMode(StrEnum):
    STOP_HARD = "hard"
    STOP_WHEN_IDLE = "when_idle"
    HEARTBEAT = "heartbeat"
    UPDATE_SETTINGS = "settings"


_ReasoningEffort = ReasoningEffort


class Settings(Struct):
    ReasoningEffort: Annotated[_ReasoningEffort, OmitZero] = ""


class ControlMessage(Struct):
    Mode: ControlMode = ""
    Reason: str = ""
    Parameters: Annotated[Settings | None, OmitZero] = None


class _ControlEnvelope(Struct, forbid_unknown_fields=True):
    Mode: str | None = None
    Reason: str | None = None
    Parameters: Raw = Raw(b"")


class _SettingsParameters(Struct, forbid_unknown_fields=True):
    ReasoningEffort: str | None = None


class Input(Struct):
    ID: str = ""
    Kind: InputKind = ""
    Payload: Annotated[Raw, OmitZero] = Raw(b"")

    def validate(self) -> None:
        """Raise ValueError when the input cannot be accepted."""
        if self.ID == "":
            raise ValueError("input ID is empty")
        match self.Kind:
            case InputKind.EXTERNAL | InputKind.CRASH:
                pass
            case InputKind.CONTROL:
                self.decode_control_message()
            case _:
                raise ValueError(
                    f"input {_quote(self.ID)} has unsupported kind {_quote(self.Kind)}"
                )
        if len(self.Payload) != 0 and not is_valid_json(bytes(self.Payload)):
            raise ValueError(f"input {_quote(self.ID)} payload is not valid JSON")

    def decode_control_message(self) -> ControlMessage:
        """Decode a control payload, rejecting unknown members. Raises ValueError."""
        if self.Kind != InputKind.CONTROL:
            raise ValueError(f"control message input has kind {_quote(self.Kind)}")
        try:
            envelope = decode(bytes(self.Payload), _ControlEnvelope | None) or _ControlEnvelope()
        except JSONError as error:
            raise ValueError(f"decode control message: {error}") from error
        mode = envelope.Mode or ""
        reason = envelope.Reason or ""
        if mode != ControlMode.UPDATE_SETTINGS and len(envelope.Parameters) != 0:
            raise ValueError(f"control mode {_quote(mode)} does not accept parameters")
        match mode:
            case ControlMode.STOP_HARD | ControlMode.STOP_WHEN_IDLE:
                return ControlMessage(Mode=ControlMode(mode), Reason=reason)
            case ControlMode.HEARTBEAT:
                if reason == "":
                    raise ValueError("heartbeat reason is empty")
                return ControlMessage(Mode=ControlMode.HEARTBEAT, Reason=reason)
            case ControlMode.UPDATE_SETTINGS:
                try:
                    parameters = (
                        decode(bytes(envelope.Parameters), _SettingsParameters | None)
                        or _SettingsParameters()
                    )
                except JSONError as error:
                    raise ValueError(f"decode settings parameters: {error}") from error
                effort = parameters.ReasoningEffort or ""
                if not ReasoningEffort.is_valid(effort):
                    raise ValueError(f"unsupported reasoning effort {_quote(effort)}")
                return ControlMessage(
                    Mode=ControlMode.UPDATE_SETTINGS,
                    Reason=reason,
                    Parameters=Settings(ReasoningEffort=ReasoningEffort(effort)),
                )
            case _:
                raise ValueError(f"unsupported control mode {_quote(mode)}")


class Writer(Protocol):
    async def submit(self, input: Input) -> None: ...


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
