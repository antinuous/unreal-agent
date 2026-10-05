from dataclasses import dataclass, field

from unreal_agent.contextbuilder import Builder
from unreal_agent.inbox import Inbox
from unreal_agent.llm import Adapter
from unreal_agent.operation import Manager
from unreal_agent.session import ID as SessionID
from unreal_agent.sessionstore import ResumeState, Store
from unreal_agent.tool import Registry

__all__ = ["CoordinatorError", "Dependencies"]


class CoordinatorError(Exception):
    """The coordinator stopped on a failed dependency or invalid history.

    The failing dependency's exception is chained as `__cause__`.
    """


@dataclass(slots=True)
class Dependencies:
    """Everything one coordinator run consumes.

    `tool_heartbeat_interval` is in seconds; zero disables heartbeats.
    """

    inbox: Inbox
    sessions: Store
    context_builder: Builder
    llm: Adapter
    tools: Registry
    operations: Manager
    session_id: SessionID = ""
    restored: ResumeState = field(default_factory=ResumeState)
    tool_heartbeat_interval: float = 0.0
