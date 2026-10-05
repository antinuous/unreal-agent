"""Provider clients built on the Responses API adapter.

Each client owns its remote HTTP client: call `respond`, then `aclose` (or use
it as an async context manager).
"""

from unreal_agent.llm.clients._base import ResponsesClient

__all__ = ["ResponsesClient"]
