import threading
import uuid
from dataclasses import dataclass
from typing import Final

import yaml

from unreal_agent._filepath import go_join
from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.primitives._errors import describe
from unreal_agent.tool._glob import BadPatternError, glob
from unreal_agent.tool.skill_use import SkillUseTranslator
from unreal_agent.tool.static import (
    BASH_NAME,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    UnavailableTranslator,
    static_definitions,
)
from unreal_agent.tool.tool import Definition, RegistrationID, Skill, Translator

__all__ = ["NIL_REGISTRATION_ID", "Registry", "StaticTranslators", "discover_skills"]

NIL_REGISTRATION_ID: Final = uuid.UUID(int=0)
_MAX_SCAN_TOKEN_SIZE: Final = 64 * 1024


@dataclass(slots=True)
class StaticTranslators:
    bash: Translator | None = None
    view_image: Translator | None = None


class Registry:
    """Advertise and resolve only the `enabled` static tools.

    A missing Bash or ViewImage translator resolves to one that reports the
    tool is not configured.
    """

    __slots__ = (
        "_enabled",
        "_lock",
        "_skill_ids_by_path",
        "_skill_order",
        "_skills",
        "_static_translators",
    )

    def __init__(self, configured: StaticTranslators, *enabled: str) -> None:
        self._enabled = frozenset(enabled)
        self._lock = threading.Lock()
        self._skills: dict[RegistrationID, Skill] = {}
        self._skill_ids_by_path: dict[str, RegistrationID] = {}
        self._skill_order: list[RegistrationID] = []
        self._static_translators: dict[str, Translator] = {
            BASH_NAME: UnavailableTranslator(BASH_NAME)
            if configured.bash is None
            else configured.bash,
            VIEW_IMAGE_NAME: UnavailableTranslator(VIEW_IMAGE_NAME)
            if configured.view_image is None
            else configured.view_image,
            SKILL_USE_NAME: SkillUseTranslator(self.resolve_skill),
        }

    def static_definitions(self) -> list[Definition]:
        return [
            definition
            for definition in static_definitions()
            if definition.tool.Name in self._enabled
        ]

    def resolve(self, name: str, /) -> Translator | None:
        if name not in self._enabled:
            return None
        return self._static_translators.get(name)

    def register_skill(self, skill: Skill, /) -> RegistrationID:
        """Raise `ValueError` for an invalid skill or a duplicate path or name."""
        validate_skill(skill)
        with self._lock:
            if skill.path in self._skill_ids_by_path:
                raise ValueError(f"skill path {go_quote(skill.path)} is already registered")
            if any(registered.name == skill.name for registered in self._skills.values()):
                raise ValueError(f"skill name {go_quote(skill.name)} is already registered")
            id = uuid.uuid4()
            self._skills[id] = skill
            self._skill_ids_by_path[skill.path] = id
            self._skill_order.append(id)
            return id

    def unregister_skill(self, id: RegistrationID, /) -> None:
        if id == NIL_REGISTRATION_ID:
            return
        with self._lock:
            skill = self._skills.pop(id, None)
            if skill is None:
                return
            del self._skill_ids_by_path[skill.path]
            self._skill_order.remove(id)

    def skills(self) -> list[Skill]:
        with self._lock:
            return [self._skills[id] for id in self._skill_order]

    def resolve_skill(self, name: str) -> Skill | None:
        with self._lock:
            for id in self._skill_order:
                if self._skills[id].name == name:
                    return self._skills[id]
        return None


def validate_skill(skill: Skill) -> None:
    if go_trim_space(skill.path) == "":
        raise ValueError("skill path must be set")
    if go_trim_space(skill.name) == "":
        raise ValueError("skill name must be set")
    if go_trim_space(skill.description) == "":
        raise ValueError("skill description must be set")


def discover_skills(directory: str) -> tuple[list[Skill], list[Exception]]:
    """Load `<directory>/*/SKILL.md` frontmatter. Invalid and duplicate skills become errors."""
    try:
        paths = glob(go_join(directory, "*", "SKILL.md"))
    except BadPatternError as error:
        return [], [ValueError(f"find skills: {error}")]

    skills: list[Skill] = []
    errors: list[Exception] = []
    names: set[str] = set()
    for path in paths:
        try:
            contents = _read_file(path)
        except _ReadError as error:
            errors.append(ValueError(f"read skill {go_quote(path)}: {error}"))
            continue
        try:
            name, description = _parse_skill_frontmatter(contents)
        except ValueError as error:
            errors.append(ValueError(f"parse skill {go_quote(path)}: {error}"))
            continue
        skill = Skill(name=go_trim_space(name), description=go_trim_space(description), path=path)
        try:
            validate_skill(skill)
        except ValueError as error:
            errors.append(ValueError(f"validate skill {go_quote(path)}: {error}"))
            continue
        if skill.name in names:
            errors.append(
                ValueError(f"skill {go_quote(path)}: duplicate name {go_quote(skill.name)}")
            )
            continue
        names.add(skill.name)
        skills.append(skill)
    return skills, errors


class _ReadError(Exception):
    pass


def _read_file(path: str) -> bytes:
    """os.ReadFile, with Go's error text."""
    try:
        file = open(path, "rb")  # noqa: SIM115
    except IsADirectoryError as error:
        raise _ReadError(f"read {path}: {describe(error)}") from error
    except OSError as error:
        raise _ReadError(f"open {path}: {describe(error)}") from error
    with file:
        try:
            return file.read()
        except OSError as error:
            raise _ReadError(f"read {path}: {describe(error)}") from error


def _scan_lines(contents: bytes) -> tuple[list[str], bool]:
    """bufio.Scanner with ScanLines: the lines and whether a line exceeded the token limit."""
    lines: list[str] = []
    start = 0
    while start < len(contents):
        end = contents.find(b"\n", start)
        line = contents[start:] if end < 0 else contents[start:end]
        if len(line) >= _MAX_SCAN_TOKEN_SIZE:
            return lines, True
        lines.append(line.removesuffix(b"\r").decode("utf-8", "surrogateescape"))
        if end < 0:
            break
        start = end + 1
    return lines, False


def _parse_skill_frontmatter(contents: bytes) -> tuple[str, str]:
    lines, too_long = _scan_lines(contents)
    if not lines or go_trim_space(lines[0]) != "---":
        raise ValueError("missing opening YAML frontmatter delimiter")
    for index, line in enumerate(lines[1:], start=1):
        if go_trim_space(line) == "---":
            frontmatter = "".join(entry + "\n" for entry in lines[1:index])
            try:
                return _decode_frontmatter(frontmatter)
            except ValueError as error:
                raise ValueError(f"decode YAML frontmatter: {error}") from error
    if too_long:
        raise ValueError("read YAML frontmatter: bufio.Scanner: token too long")
    raise ValueError("missing closing YAML frontmatter delimiter")


_NULL_TAG: Final = "tag:yaml.org,2002:null"
_GO_YAML_PROBLEMS: Final = {
    "mapping values are not allowed here": "mapping values are not allowed in this context",
}
_FRONTMATTER_TYPE: Final = "tool.skillFrontmatter"


def _decode_frontmatter(frontmatter: str) -> tuple[str, str]:
    """go-yaml v3 `Unmarshal` into a struct of `name` and `description` strings."""
    try:
        document = next(yaml.compose_all(frontmatter, Loader=yaml.SafeLoader), None)
    except yaml.reader.ReaderError as error:
        raise ValueError(f"yaml: {_reader_problem(error)}") from error
    except yaml.MarkedYAMLError as error:
        mark = error.problem_mark or error.context_mark
        line = f"line {mark.line + 1}: " if mark is not None else ""
        problem = error.problem or error.context or "unknown problem parsing YAML content"
        raise ValueError(f"yaml: {line}{_GO_YAML_PROBLEMS.get(problem, problem)}") from error
    if document is None or (isinstance(document, yaml.ScalarNode) and document.tag == _NULL_TAG):
        return "", ""
    if not isinstance(document, yaml.MappingNode):
        raise _unmarshal_errors([_type_error(document, _FRONTMATTER_TYPE)])
    duplicates = [
        f"line {later.start_mark.line + 1}: mapping key {go_quote(_node_value(later))} "
        f"already defined at line {earlier.start_mark.line + 1}"
        for position, (earlier, _) in enumerate(document.value)
        for later, _ in document.value[position + 1 :]
        if type(earlier) is type(later) and _node_value(earlier) == _node_value(later)
    ]
    if duplicates:
        raise _unmarshal_errors(duplicates)
    fields = {"name": "", "description": ""}
    errors: list[str] = []
    for key, value in document.value:
        if not isinstance(key, yaml.ScalarNode) or key.value not in fields:
            continue
        if isinstance(value, yaml.ScalarNode):
            fields[key.value] = "" if value.tag == _NULL_TAG else value.value
        else:
            errors.append(_type_error(value, "string"))
    if errors:
        raise _unmarshal_errors(errors)
    return fields["name"], fields["description"]


def _reader_problem(error: yaml.reader.ReaderError) -> str:
    character = error.character
    if isinstance(character, int) and 0xDC80 <= character <= 0xDCFF:
        octet = character - 0xDC00
        if octet < 0xC0 or octet >= 0xF8:
            return "invalid leading UTF-8 octet"
        return "invalid trailing UTF-8 octet"
    return "control characters are not allowed"


def _node_value(node: yaml.Node) -> str:
    return node.value if isinstance(node, yaml.ScalarNode) else ""


def _type_error(node: yaml.Node, target: str) -> str:
    tag = node.tag.replace("tag:yaml.org,2002:", "!!", 1)
    value = ""
    if isinstance(node, yaml.ScalarNode):
        value = f" `{node.value[:7]}...`" if len(node.value) > 10 else f" `{node.value}`"
    return f"line {node.start_mark.line + 1}: cannot unmarshal {tag}{value} into {target}"


def _unmarshal_errors(errors: list[str]) -> ValueError:
    return ValueError("yaml: unmarshal errors:\n  " + "\n  ".join(errors))
