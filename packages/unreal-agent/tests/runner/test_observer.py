"""Session observer and log output tests."""

import io

from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Response, Role
from unreal_agent.runner.run import SessionObserver
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Item, ItemKind, ModelResponse

from tests.runner._helpers import FailingItemWriter


class Cancel:
    def __init__(self) -> None:
        self.cancelled = False

    def __call__(self) -> None:
        self.cancelled = True


def test_session_observer_does_not_stop_on_text_response() -> None:
    output = io.BytesIO()
    cancel = Cancel()
    observer = SessionObserver("session", output, cancel)
    observer.observe(
        "session",
        Item(
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                Response=Response(
                    Output=[
                        LLMItem(
                            Type=ItemType.MESSAGE,
                            Data=Message(Role=Role.ASSISTANT, Text="The checks are still running."),
                        )
                    ]
                )
            ),
        ),
    )
    assert observer.error is None
    assert output.getvalue(), "model response was not written"
    assert not cancel.cancelled, "observer canceled the run"


def test_session_observer_stops_on_output_failure() -> None:
    want = OSError("output unavailable")
    output = FailingItemWriter(ItemKind.TURN, want)
    cancel = Cancel()
    observer = SessionObserver("session", output, cancel)
    item = Item(Kind=ItemKind.TURN, Data=Turn(ID="turn-1", Type=TurnType.REGULAR))
    observer.observe("other-session", item)
    assert output.writes == 0, "unrelated session affected the observer"
    assert not cancel.cancelled, "unrelated session affected the observer"
    observer.observe("session", item)
    assert observer.error is not None
    assert observer.error.__cause__ is want
    assert cancel.cancelled
    output.error = OSError("later failure")
    observer.observe("session", item)
    assert output.writes == 1, "observer wrote again or replaced its first error"
    assert observer.error.__cause__ is want, "observer wrote again or replaced its first error"
