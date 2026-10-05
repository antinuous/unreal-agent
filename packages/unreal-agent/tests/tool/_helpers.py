from unreal_agent.operation import Spec


class RecordingContext:
    def __init__(self) -> None:
        self.specs: list[Spec] = []

    def submit(self, spec: Spec, /) -> str:
        self.specs.append(spec)
        return "operation-1"
