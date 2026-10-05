"""Append-only session history and operation state.

`Item.Data` holds the type selected by `Item.Kind`: `Fork`, `inbox.Input`,
`session.Turn`, `ModelResponse`, or `ToolCallStatus`. Encoding an item with
`gojson.encode` validates that pairing, and decoding selects the data type
from the tag.

`Store` is the async storage interface consumed by the coordinator and the
runner. `unreal_agent.sessionstore.localfile.Store` implements it with one
JSONL file per session. Failures raise `StoreError`. `NotExistError` (a
`FileNotFoundError`) marks a missing session, turn, or operation, and
`ExistError` (a `FileExistsError`) marks a duplicate.
"""

from unreal_agent.sessionstore.sessionstore import (
    BEFORE_FIRST,
    NIL_OBSERVER_ID,
    ExistError,
    Fork,
    Item,
    ItemData,
    ItemKind,
    ModelResponse,
    NotExistError,
    Observer,
    ObserverID,
    Page,
    ResumeState,
    Sequence,
    SessionInfo,
    Snapshot,
    Store,
    StoreError,
    ToolCallStatus,
)

__all__ = [
    "BEFORE_FIRST",
    "NIL_OBSERVER_ID",
    "ExistError",
    "Fork",
    "Item",
    "ItemData",
    "ItemKind",
    "ModelResponse",
    "NotExistError",
    "Observer",
    "ObserverID",
    "Page",
    "ResumeState",
    "Sequence",
    "SessionInfo",
    "Snapshot",
    "Store",
    "StoreError",
    "ToolCallStatus",
]
