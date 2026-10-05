import threading
import uuid
from dataclasses import dataclass
from typing import Final

from unreal_agent.primitives._errors import describe
from unreal_agent.tool._glob import BadPatternError, glob
from unreal_agent.tool._go import go_join, go_quote, go_trim_space
from unreal_agent.tool.skill_use import SkillUseTranslator
from unreal_agent.tool.static import (
    BASH_NAME,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    UnavailableTranslator,
    static_definitions,
)
from unreal_agent.tool.tool import Definition, RegistrationID, Registry, Skill, Translator

__all__ = ["NIL_REGISTRATION_ID", "StaticTranslators", "discover_skills", "new_registry"]

NIL_REGISTRATION_ID: Final = uuid.UUID(int=0)
_MAX_SCAN_TOKEN_SIZE: Final = 64 * 1024


@dataclass(slots=True)
class StaticTranslators:
    bash: Translator | None = None
    view_image: Translator | None = None


class _Registry:
    __slots__ = (
        "_enabled",
        "_lock",
        "_skill_ids_by_path",
        "_skill_order",
        "_skills",
        "_static_translators",
    )

    def __init__(self, configured: StaticTranslators, enabled: tuple[str, ...]) -> None:
        self._enabled = frozenset(enabled)
        self._lock = threading.Lock()
        self._skills: dict[RegistrationID, Skill] = {}
        self._skill_ids_by_path: dict[str, RegistrationID] = {}
        self._skill_order: list[RegistrationID] = []
        self._static_translators: dict[str, Translator] = {
            BASH_NAME: configured.bash or UnavailableTranslator(BASH_NAME),
            VIEW_IMAGE_NAME: configured.view_image or UnavailableTranslator(VIEW_IMAGE_NAME),
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


def new_registry(configured: StaticTranslators, *enabled: str) -> Registry:
    """Advertise and resolve only the `enabled` static tools.

    A missing Bash or ViewImage translator resolves to one that reports the
    tool is not configured.
    """
    return _Registry(configured, enabled)


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
    name = description = ""
    for line in lines[1:]:
        if go_trim_space(line) == "---":
            return name, description
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key == "name":
            name = go_trim_space(value)
        elif key == "description":
            description = go_trim_space(value)
    if too_long:
        raise ValueError("read YAML frontmatter: bufio.Scanner: token too long")
    raise ValueError("missing closing YAML frontmatter delimiter")
