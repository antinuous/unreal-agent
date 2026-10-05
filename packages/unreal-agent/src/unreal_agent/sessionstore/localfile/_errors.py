import os

from unreal_agent.session import ID as SessionID
from unreal_agent.sessionstore import ExistError, NotExistError, StoreError
from unreal_agent.tool._go import go_quote


def wrap(prefix: str, error: BaseException, op: str = "") -> StoreError:
    """Prefix an error's message, keeping whether it marks a missing or duplicate value."""
    if isinstance(error, FileNotFoundError):
        kind: type[StoreError] = NotExistError
    elif isinstance(error, FileExistsError):
        kind = ExistError
    else:
        kind = StoreError
    wrapped = kind(f"{prefix}: {message(error, op)}")
    wrapped.__cause__ = error
    return wrapped


def join(first: StoreError, second: StoreError) -> StoreError:
    """Combine two failures like Go's errors.Join, keeping the first one's kind."""
    joined = type(first)(f"{first}\n{second}")
    joined.__cause__ = first
    joined.__context__ = second
    return joined


def message(error: BaseException, op: str = "") -> str:
    """Format an OS error the way Go's *PathError prints, as `op path: reason`."""
    if isinstance(error, StoreError) or not isinstance(error, OSError):
        return str(error)
    text = _strerror(error)
    subject = " ".join(str(part) for part in (op, error.filename, error.filename2) if part)
    return f"{subject}: {text}" if subject else text


def _strerror(error: OSError) -> str:
    if error.errno is None:
        return error.strerror or str(error)
    text = os.strerror(error.errno)
    return text[:1].lower() + text[1:]


def validate_session_id(id: SessionID) -> None:
    if id == "":
        raise StoreError("session ID is empty")
    if not all(_is_id_character(character) for character in id):
        raise StoreError(
            f"session ID {go_quote(id)} must contain only ASCII letters, digits, and dashes"
        )


def _is_id_character(character: str) -> bool:
    return character.isascii() and (character.isalnum() or character == "-")
