"""Go's `filepath.Glob` on Unix.

Python's glob skips dot entries, uses `[!...]` instead of `[^...]`, and does
not report malformed patterns, so skill discovery ports Go's version.
"""

import os
from typing import Final

from unreal_agent._filepath import go_join
from unreal_agent._go import RUNE_ERROR, decode_rune

__all__ = ["BadPatternError", "glob"]

_SEPARATOR: Final = 0x2F
_PATH_SEPARATORS_LIMIT: Final = 10000


class BadPatternError(ValueError):
    def __init__(self) -> None:
        super().__init__("syntax error in pattern")


def glob(pattern: str) -> list[str]:
    """Return matching paths in Go's order. Raises BadPatternError."""
    return [os.fsdecode(path) for path in _glob(os.fsencode(pattern), 0)]


def _glob(pattern: bytes, depth: int) -> list[bytes]:
    if depth == _PATH_SEPARATORS_LIMIT:
        raise BadPatternError
    match(pattern, b"")
    if not _has_meta(pattern):
        try:
            os.lstat(pattern)
        except OSError:
            return []
        return [pattern]
    split = pattern.rfind(b"/") + 1
    directory, file = pattern[:split], pattern[split:]
    directory = _clean_glob_path(directory)
    if not _has_meta(directory):
        return _glob_directory(directory, file, [])
    if directory == pattern:
        raise BadPatternError
    matches: list[bytes] = []
    for candidate in _glob(directory, depth + 1):
        matches = _glob_directory(candidate, file, matches)
    return matches


def _clean_glob_path(path: bytes) -> bytes:
    if path == b"":
        return b"."
    if path == b"/":
        return path
    return path[:-1]


def _glob_directory(directory: bytes, pattern: bytes, matches: list[bytes]) -> list[bytes]:
    try:
        if not os.path.isdir(directory):
            return matches
        names = sorted(os.listdir(directory))
    except OSError:
        return matches
    for name in names:
        if match(pattern, name):
            matches.append(os.fsencode(go_join(os.fsdecode(directory), os.fsdecode(name))))
    return matches


def _has_meta(path: bytes) -> bool:
    return any(character in path for character in b"*?[\\")


def match(pattern: bytes, name: bytes) -> bool:
    """filepath.Match. Raises BadPatternError."""
    while pattern:
        star, chunk, pattern = _scan_chunk(pattern)
        if star and not chunk:
            return _SEPARATOR not in name
        rest, ok, bad = _match_chunk(chunk, name)
        if ok and (not rest or pattern):
            name = rest
            continue
        if bad:
            raise BadPatternError
        if star:
            matched = False
            index = 0
            while index < len(name) and name[index] != _SEPARATOR:
                rest, ok, bad = _match_chunk(chunk, name[index + 1 :])
                if ok:
                    if not pattern and rest:
                        index += 1
                        continue
                    name = rest
                    matched = True
                    break
                if bad:
                    raise BadPatternError
                index += 1
            if matched:
                continue
        return False
    return not name


def _scan_chunk(pattern: bytes) -> tuple[bool, bytes, bytes]:
    star = False
    while pattern and pattern[0] == 0x2A:
        pattern = pattern[1:]
        star = True
    in_range = False
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == 0x5C:
            if index + 1 < len(pattern):
                index += 1
        elif character == 0x5B:
            in_range = True
        elif character == 0x5D:
            in_range = False
        elif character == 0x2A and not in_range:
            return star, pattern[:index], pattern[index:]
        index += 1
    return star, pattern, b""


def _match_chunk(chunk: bytes, text: bytes) -> tuple[bytes, bool, bool]:
    """Return (rest, ok, bad pattern)."""
    failed = False
    while chunk:
        failed = failed or not text
        head = chunk[0]
        if head == 0x5B:
            rune = 0
            if not failed:
                rune, size = decode_rune(text)
                text = text[size:]
            chunk = chunk[1:]
            negated = False
            if chunk and chunk[0] == 0x5E:
                negated = True
                chunk = chunk[1:]
            matched = False
            ranges = 0
            while True:
                if chunk and chunk[0] == 0x5D and ranges > 0:
                    chunk = chunk[1:]
                    break
                low, chunk, bad = _get_escape(chunk)
                if bad:
                    return b"", False, True
                high = low
                if chunk[0] == 0x2D:
                    high, chunk, bad = _get_escape(chunk[1:])
                    if bad:
                        return b"", False, True
                matched = matched or low <= rune <= high
                ranges += 1
            failed = failed or matched == negated
        elif head == 0x3F:
            if not failed:
                failed = text[0] == _SEPARATOR
                text = text[decode_rune(text)[1] :]
            chunk = chunk[1:]
        else:
            if head == 0x5C:
                chunk = chunk[1:]
                if not chunk:
                    return b"", False, True
            if not failed:
                failed = chunk[0] != text[0]
                text = text[1:]
            chunk = chunk[1:]
    if failed:
        return b"", False, False
    return text, True, False


def _get_escape(chunk: bytes) -> tuple[int, bytes, bool]:
    if not chunk or chunk[0] in (0x2D, 0x5D):
        return 0, chunk, True
    if chunk[0] == 0x5C:
        chunk = chunk[1:]
        if not chunk:
            return 0, chunk, True
    rune, size = decode_rune(chunk)
    bad = rune == RUNE_ERROR and size == 1
    rest = chunk[size:]
    return rune, rest, bad or not rest
