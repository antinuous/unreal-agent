"""Go `path/filepath` semantics for Unix paths.

`os.path.normpath` keeps a leading `//`, and `os.path.abspath` uses the
resolved working directory. Go's `Clean` reduces `//x` to `/x`, and `Abs`
joins with `os.Getwd`, which prefers a `$PWD` that names the working directory.
"""

import os

__all__ = ["go_abs", "go_base", "go_clean", "go_getwd", "go_join"]


def go_clean(path: str) -> str:
    if not path:
        return "."
    rooted = path.startswith("/")
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not rooted:
                parts.append("..")
            continue
        parts.append(part)
    cleaned = "/".join(parts)
    if rooted:
        return "/" + cleaned
    return cleaned or "."


def go_join(*elements: str) -> str:
    present = [element for element in elements if element]
    if not present:
        return ""
    return go_clean("/".join(present))


def go_getwd() -> str:
    """os.Getwd: `$PWD` when it is absolute and names the working directory."""
    pwd = os.environ.get("PWD", "")
    if pwd.startswith("/"):
        try:
            if os.path.samestat(os.stat(pwd), os.stat(".")):
                return pwd
        except OSError:
            pass
    return os.getcwd()


def go_abs(path: str) -> str:
    """filepath.Abs. Raises OSError when the working directory is unavailable."""
    if path.startswith("/"):
        return go_clean(path)
    return go_join(go_getwd(), path)


def go_base(path: str) -> str:
    """filepath.Base."""
    if not path:
        return "."
    trimmed = path.rstrip("/")
    if not trimmed:
        return "/"
    return trimmed[trimmed.rfind("/") + 1 :]
