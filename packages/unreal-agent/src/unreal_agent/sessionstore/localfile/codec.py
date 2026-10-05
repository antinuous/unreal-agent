from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Final

from msgspec import field as struct_field
from msgspec.structs import replace

from unreal_agent.gojson import Int64, JSONError, OmitEmpty, Raw, Struct, decode, encode, validate
from unreal_agent.operation import Operation
from unreal_agent.session import Session
from unreal_agent.sessionstore import Item, ItemKind, StoreError, ToolCallStatus
from unreal_agent.sessionstore.localfile._errors import validate_session_id, wrap
from unreal_agent.sessionstore.localfile.state import StoredState, new_stored_state
from unreal_agent.tool._go import go_quote

FORMAT_VERSION: Final = 2


class RecordType(StrEnum):
    SESSION = "session"
    ITEM = "item"
    OPERATION = "operation"


class _LogRecord(Struct):
    Type: str = struct_field(default="", name="type")
    Data: Raw = struct_field(default=Raw(b""), name="data")


_Session = Session
_Item = Item
_Operation = Operation


class SessionRecord(Struct):
    Version: Int64 = 0
    Session: _Session = struct_field(default_factory=Session)


class ItemRecord(Struct):
    Item: _Item = struct_field(default_factory=Item)
    Operations: Annotated[list[Operation], OmitEmpty] = []


class OperationRecord(Struct):
    Operation: _Operation = struct_field(default_factory=Operation)


@dataclass(slots=True)
class _DecodedRecord:
    type: RecordType
    item: ItemRecord = field(default_factory=ItemRecord)
    operation: Operation = field(default_factory=Operation)


def encode_initial_log(value: Session, items: Sequence[Item]) -> bytes:
    validate_session_id(value.ID)
    try:
        encoded = encode_record(
            RecordType.SESSION, SessionRecord(Version=FORMAT_VERSION, Session=value)
        )
    except StoreError as error:
        raise StoreError(f"encode session: {error}") from error

    lines = [encoded]
    for index, item in enumerate(items):
        if item.Sequence != index + 1:
            raise StoreError(f"item {index} has non-contiguous sequence {item.Sequence}")
        if item.RecordedAt.is_zero():
            raise StoreError(f"item {index} recorded time is zero")
        record = ItemRecord(Item=item)
        if item.Kind == ItemKind.TOOL_CALL_STATUS and isinstance(item.Data, ToolCallStatus):
            record = ItemRecord(
                Item=replace(item, Data=replace(item.Data, Operations=[])),
                Operations=item.Data.Operations,
            )
        try:
            lines.append(encode_record(RecordType.ITEM, record))
        except StoreError as error:
            raise StoreError(f"encode item {index}: {error}") from error
    return b"".join(lines)


def decode_log(encoded: bytes) -> tuple[StoredState, int]:
    """Decode the committed records; bytes after the last newline are ignored."""
    last_newline = encoded.rfind(b"\n")
    if last_newline < 0:
        raise StoreError("session log has no committed records")
    committed_size = last_newline + 1
    lines = encoded[:last_newline].split(b"\n")

    value: StoredState | None = None
    records: list[_DecodedRecord] = []
    item_count = 0
    inherited_item_count = 0
    for line_index, line in enumerate(lines):
        try:
            record = decode(line, _LogRecord)
        except JSONError as error:
            raise StoreError(f"decode record {line_index + 1}: {error}") from error
        match record.Type:
            case RecordType.SESSION:
                if line_index != 0:
                    raise StoreError("session record must be first")
                try:
                    header = decode(record.Data, SessionRecord)
                except JSONError as error:
                    raise StoreError(f"decode session record: {error}") from error
                if header.Version == 1:
                    raise StoreError("legacy session format version 1 cannot be resumed")
                if header.Version != FORMAT_VERSION:
                    raise StoreError(f"unsupported session format version {header.Version}")
                validate_session_id(header.Session.ID)
                if header.Session.CreatedAt.is_zero():
                    raise StoreError(f"session {go_quote(header.Session.ID)} creation time is zero")
                value = new_stored_state(header.Session.ID, header.Session.CreatedAt)
            case RecordType.ITEM:
                if line_index == 0:
                    raise StoreError("session record must be first")
                try:
                    item_record = decode(record.Data, ItemRecord)
                    item_record.Item.validate_data()
                except (JSONError, ValueError) as error:
                    raise StoreError(f"decode item record: {error}") from error
                item = item_record.Item
                if item.Sequence != item_count + 1:
                    raise StoreError(
                        f"item {item_count} has sequence {item.Sequence}, want {item_count + 1}"
                    )
                if item.RecordedAt.is_zero():
                    raise StoreError(f"item {item_count} recorded time is zero")
                if item.Kind != ItemKind.TOOL_CALL_STATUS and item_record.Operations:
                    raise StoreError(
                        f"item {item_count} of kind {go_quote(item.Kind)} initializes operations"
                    )
                item_count += 1
                if item.Kind == ItemKind.FORK:
                    inherited_item_count = item_count
                records.append(_DecodedRecord(type=RecordType.ITEM, item=item_record))
            case RecordType.OPERATION:
                if line_index == 0:
                    raise StoreError("session record must be first")
                try:
                    update = decode(record.Data, OperationRecord)
                except JSONError as error:
                    raise StoreError(f"decode operation record: {error}") from error
                records.append(
                    _DecodedRecord(type=RecordType.OPERATION, operation=update.Operation)
                )
            case _:
                raise StoreError(f"unsupported record type {go_quote(record.Type)}")

    assert value is not None
    replayed_items = 0
    for record_index, record in enumerate(records):
        if record.type == RecordType.ITEM:
            item_index = replayed_items
            replayed_items += 1
            if item_index < inherited_item_count:
                if record.item.Operations:
                    raise StoreError(f"inherited item {item_index} initializes operations")
                value.inherit_item(record.item.Item)
                continue
            try:
                _replay_item(value, record.item)
            except StoreError as error:
                raise wrap(f"apply item {item_index}", error) from error
        else:
            if replayed_items < inherited_item_count:
                raise StoreError(
                    f"operation record {record_index + 2} appears in inherited history"
                )
            try:
                value.save_operation(record.operation)
            except StoreError as error:
                raise wrap("apply operation record", error) from error
    return value, committed_size


def _replay_item(state: StoredState, record: ItemRecord) -> None:
    item = record.Item
    data: Any = item.Data
    match item.Kind:
        case ItemKind.INPUT:
            state.append_input(data, item.RecordedAt)
        case ItemKind.TURN:
            state.append_turn(data, item.RecordedAt)
        case ItemKind.MODEL_RESPONSE:
            state.append_model_response(data, item.RecordedAt)
        case ItemKind.TOOL_CALL_STATUS:
            state.append_tool_call_status(data, record.Operations, item.RecordedAt)
        case _:
            raise StoreError(f"unexpected owned item kind {go_quote(item.Kind)}")


def encode_record(kind: str, value: Any) -> bytes:
    """Encode one JSONL record, validating embedded raw JSON like Go's encoder."""
    try:
        data = encode(value)
        encoded = encode(_LogRecord(Type=kind, Data=Raw(data)))
        validate(encoded)
    except (JSONError, ValueError, OverflowError, IndexError) as error:
        raise StoreError(str(error)) from error
    if b"\n" in encoded:
        raise StoreError("encoded record contains a newline")
    return encoded + b"\n"
