"""In-memory, I/O-free model request construction.

`new_builder(*skills)` returns a `Builder`. Every request starts with one
system message, `TrimSpace(preamble + "\\n\\n" + system_prompt)`, where the
preamble is the bundled harness prompt followed by the skill preamble and an
`<available_skills>` block when skills are given.

The input is a committed prefix followed by a staged suffix. Model responses
append to the committed prefix, so a response lands before inputs that were
staged but not yet submitted. `commit()` moves the staged suffix into the
prefix. A running tool call shows `TOOL_CALL_RUNNING_PAYLOAD` until a later
result for the same call replaces its staged placeholder. `build()` returns a
copy that later builder calls do not change.

Each model response records its input plus output tokens. Once the latest
count reaches the model's compaction threshold, `build_compaction()` asks for
a summary of the history before the most recent 20,000 tokens of whole turns,
and `compact(response)` returns a new builder holding that summary, the tool
calls still pending at the cutoff, and the retained history.
"""

from unreal_agent.contextbuilder.builder import PREAMBLE, TOOL_CALL_RUNNING_PAYLOAD, new_builder
from unreal_agent.contextbuilder.contextbuilder import Builder, Change, ChangeKind, Report, Result

__all__ = [
    "PREAMBLE",
    "TOOL_CALL_RUNNING_PAYLOAD",
    "Builder",
    "Change",
    "ChangeKind",
    "Report",
    "Result",
    "new_builder",
]
