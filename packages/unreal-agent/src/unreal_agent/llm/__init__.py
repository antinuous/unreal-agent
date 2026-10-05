"""Provider-neutral model requests, responses, and items.

`Item.Data` holds the type selected by `Item.Type`. Encoding an item validates
that pairing, and decoding selects the data type from the tag. Reasoning.Raw
and Usage.Raw pass provider JSON through byte for byte (compacted on encode).
"""

from unreal_agent.llm.adapter import Adapter, RequestOptions
from unreal_agent.llm.model import (
    Failure,
    Item,
    ItemData,
    ItemType,
    Message,
    Model,
    Reasoning,
    ReasoningEffort,
    Request,
    Response,
    Role,
    StopReason,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
    Usage,
)

__all__ = [
    "Adapter",
    "Failure",
    "Item",
    "ItemData",
    "ItemType",
    "Message",
    "Model",
    "Reasoning",
    "ReasoningEffort",
    "Request",
    "RequestOptions",
    "Response",
    "Role",
    "StopReason",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolResultKind",
    "ToolResultOutput",
    "ToolType",
    "Usage",
]
