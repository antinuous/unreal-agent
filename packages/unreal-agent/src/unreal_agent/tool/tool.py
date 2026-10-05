from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Protocol
from uuid import UUID

from unreal_agent.gojson import OmitZero, Raw, Struct
from unreal_agent.llm import Tool, ToolCall, ToolResult
from unreal_agent.operation import ID, Operation, Spec

__all__ = [
    "CallStatus",
    "Context",
    "Definition",
    "RegistrationID",
    "Registry",
    "ResultTranslator",
    "Skill",
    "TranslationError",
    "Translator",
]


class CallStatus(Struct):
    """The persisted outcome of translating one tool call."""

    Error: str = ""
    ErrorTruncated: Annotated[bool, OmitZero] = False
    WaitingFor: Annotated[list[str], OmitZero] = []


class Context(Protocol):
    """Turn-local and coordinator-owned.

    `submit` allocates an ID and records inert data without performing I/O or
    handing work to another queue.
    """

    def submit(self, spec: Spec, /) -> ID: ...


class TranslationError(Exception):
    """A call's status and operations cannot be turned into a tool result."""


class ResultTranslator(Protocol):
    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> ToolResult:
        """Raise `TranslationError` when the status and operations are inconsistent."""
        ...


class Translator(ResultTranslator, Protocol):
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        """Validate a call and submit its operations. Runs synchronously without I/O."""
        ...


@dataclass(slots=True)
class Definition:
    tool: Tool
    metadata: Raw = field(default_factory=lambda: Raw(b""))


type RegistrationID = UUID


@dataclass(frozen=True, slots=True)
class Skill:
    name: str = ""
    description: str = ""
    path: str = ""


class Registry(Protocol):
    def static_definitions(self) -> list[Definition]: ...

    def resolve(self, name: str, /) -> Translator | None: ...

    def register_skill(self, skill: Skill, /) -> RegistrationID:
        """Raise `ValueError` for an invalid skill or a duplicate path or name."""
        ...

    def unregister_skill(self, id: RegistrationID, /) -> None: ...

    def skills(self) -> list[Skill]: ...
