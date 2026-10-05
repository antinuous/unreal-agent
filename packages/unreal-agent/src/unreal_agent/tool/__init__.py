"""Model-visible tool selection and pure translation between tool calls and operations.

- `Translator.translate(ctx, call) -> CallStatus` validates a call and submits
  operation specs through `ctx.submit(spec) -> ID`. It runs synchronously and
  performs no I/O.
- `Translator.translate_result(call_id, status, operations) -> ToolResult`
  renders the model-visible result and raises `TranslationError` when the
  status and operations are inconsistent.
- `CallStatus` is persisted and encodes byte-identically to Go with
  `gojson.encode`.
- `Registry(StaticTranslators(bash=..., view_image=...), *enabled)`
  advertises and resolves only the enabled static tools. Skills register
  under UUIDs and are unique by path and by name.
- `discover_skills(directory)` reads `<directory>/*/SKILL.md` frontmatter.

Tool constructors live in `unreal_agent.tool.bash` and
`unreal_agent.tool.viewimage`. Argument errors match Go's `encoding/json/v2`
text because they are shown to the model.
"""

from unreal_agent.tool.output import error_status, parse_max_output_length
from unreal_agent.tool.registry import (
    NIL_REGISTRATION_ID,
    Registry,
    StaticTranslators,
    discover_skills,
)
from unreal_agent.tool.skill_use import SkillUseTranslator
from unreal_agent.tool.static import BASH_NAME, SKILL_USE_NAME, VIEW_IMAGE_NAME
from unreal_agent.tool.tool import (
    CallStatus,
    Context,
    Definition,
    RegistrationID,
    ResultTranslator,
    Skill,
    TranslationError,
    Translator,
)

__all__ = [
    "BASH_NAME",
    "NIL_REGISTRATION_ID",
    "SKILL_USE_NAME",
    "VIEW_IMAGE_NAME",
    "CallStatus",
    "Context",
    "Definition",
    "RegistrationID",
    "Registry",
    "ResultTranslator",
    "Skill",
    "SkillUseTranslator",
    "StaticTranslators",
    "TranslationError",
    "Translator",
    "discover_skills",
    "error_status",
    "parse_max_output_length",
]
