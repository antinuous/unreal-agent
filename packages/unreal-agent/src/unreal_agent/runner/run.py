"""One runner invocation: parse arguments, open the session, and drive the coordinator."""

import asyncio
import contextlib
import io
import os
import re
import stat
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import BinaryIO, Final

from unreal_agent import gojson
from unreal_agent._filepath import go_abs, go_join
from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import Coordinator, Dependencies
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind, Settings
from unreal_agent.llm import Model
from unreal_agent.llm.responsesapi import DEFAULT_MAX_ATTEMPTS
from unreal_agent.operation import LocalOperationManager
from unreal_agent.runner.config import Client, Config, Getenv, Provider, ToolConfig
from unreal_agent.runner.errors import RunnerError, describe, join, wrap
from unreal_agent.runner.flags import HelpError, parse_flags
from unreal_agent.runner.request import reasoning_effort, validate_request
from unreal_agent.runner.streams import Reader, TeeWriter, Writer, write_flushed
from unreal_agent.session import ID as SessionID
from unreal_agent.sessionstore import Item, ResumeState, Snapshot
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.tool import (
    BASH_NAME,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    StaticTranslators,
    bash,
    discover_skills,
    viewimage,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_SYSTEM_PROMPT",
    "LLM_API_KEY_ENVIRONMENT",
    "LLM_BASE_URL_ENVIRONMENT",
    "LLM_MAX_ATTEMPTS_ENVIRONMENT",
    "LLM_MODEL_ENVIRONMENT",
    "LLM_PROVIDER_ENVIRONMENT",
    "EnvironmentScope",
    "SessionObserver",
    "load_dotenv",
    "open_datetime_log",
    "resolve_max_attempts",
    "resolve_session_directory",
    "run",
    "run_main",
]

DEFAULT_PROVIDER: Final = "openai"
DEFAULT_SESSION_DIRECTORY: Final = "unreal-agent/sessions"
LLM_API_KEY_ENVIRONMENT: Final = "UNREAL_HARNESS_LLM_API_KEY"
LLM_BASE_URL_ENVIRONMENT: Final = "UNREAL_HARNESS_LLM_BASE_URL"
LLM_MODEL_ENVIRONMENT: Final = "UNREAL_HARNESS_LLM_MODEL"
LLM_PROVIDER_ENVIRONMENT: Final = "UNREAL_HARNESS_LLM_PROVIDER"
LLM_MAX_ATTEMPTS_ENVIRONMENT: Final = "UNREAL_HARNESS_LLM_MAX_ATTEMPTS"

DEFAULT_SYSTEM_PROMPT: Final = """\
You are an AI agent running inside an isolated sandbox container.

## Guidelines
- Save output files to the workspace root.
- For large datasets, inspect a sample first before processing everything.
"""

_ATOI: Final = re.compile(r"[+-]?[0-9]+")
_MAX_INT64: Final = 2**63 - 1


class _ErrorEvent(gojson.Struct):
    type: str = ""
    message: str = ""


async def run_main(
    args: list[str],
    getenv: Getenv,
    input: Reader,
    output: Writer,
    stderr: Writer,
    config: Config,
) -> int:
    """Run and map the outcome to an exit code.

    0 on success or help, 130 when the calling task is cancelled (no error
    event), and otherwise 1 after writing an error event to `output` and the
    error to `stderr`.
    """
    try:
        await run(args, getenv, input, output, stderr, config)
    except HelpError:
        return 0
    except asyncio.CancelledError:
        return 130
    except Exception as error:
        task = asyncio.current_task()
        if task is not None and task.cancelling() > 0:
            return 130
        failure: BaseException = error
        try:
            encoded = gojson.encode(_ErrorEvent(type="error", message=describe(error)))
        except gojson.JSONError as encode_error:
            failure = join(failure, RunnerError(f"encode error event: {encode_error}"))
        else:
            try:
                write_flushed(output, encoded + b"\n")
            except Exception as write_error:
                failure = join(failure, RunnerError(f"write error event: {describe(write_error)}"))
        prefix = config.name + ": " if config.name else ""
        try:
            write_flushed(
                stderr, f"{prefix}{describe(failure)}\n".encode("utf-8", "surrogateescape")
            )
        except Exception:
            return 1
        return 1
    return 0


type _Cleanup = Callable[[], Awaitable[None]]


async def run(
    args: list[str],
    getenv: Getenv,
    input: Reader,
    output: Writer,
    flag_output: Writer,
    config: Config,
) -> None:
    """Execute one request. Cancelling the calling task cancels the run.

    Raises HelpError for -h, RunnerError (or the failing dependency's error)
    on failure, and CancelledError when cancelled. Cleanup failures are joined
    to the run's error.
    """
    cleanups: list[_Cleanup] = []
    failure: Exception | None = None
    try:
        await _run(args, getenv, input, output, flag_output, config, cleanups)
    except Exception as error:
        failure = error
    except BaseException:
        with contextlib.suppress(asyncio.CancelledError):
            await _clean_up(cleanups)
        raise
    cleanup_error = await _clean_up(cleanups)
    error = join(failure, cleanup_error)
    if error is not None:
        raise error


async def _clean_up(cleanups: list[_Cleanup]) -> BaseException | None:
    """Run every cleanup to completion, like Go's defers.

    A cancellation that lands during cleanup does not interrupt it and does not
    change the outcome: Go decides the exit code from the run's result. The
    task stays marked as cancelling, so a failed run still exits 130.
    """
    error: BaseException | None = None
    cancelled = False
    for cleanup in reversed(cleanups):
        task = asyncio.ensure_future(cleanup())
        while not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait((task,))
        if task.cancelled():
            cancelled = True
        elif (failure := task.exception()) is not None:
            if not isinstance(failure, Exception):
                raise failure
            error = join(error, failure)
    cleanups.clear()
    if cancelled:
        raise asyncio.CancelledError
    return error


async def _run(
    args: list[str],
    getenv: Getenv,
    input: Reader,
    output: Writer,
    flag_output: Writer,
    config: Config,
    cleanups: list[_Cleanup],
) -> None:
    if go_trim_space(config.name) == "":
        raise RunnerError("runner name must be set")
    if config.parse_request is None:
        raise RunnerError("request parser must be set")
    options = parse_flags(config.name, args, flag_output)
    if len(options.arguments) > 1:
        raise RunnerError("expected at most one positional JSON request")
    if options.prompt is not None and options.arguments:
        raise RunnerError("-p cannot be combined with a positional JSON request")
    if options.tool_heartbeat_interval < 0:
        raise RunnerError("tool heartbeat interval must not be negative")

    if options.prompt is not None:
        try:
            encoded = gojson.encode({"prompt": options.prompt})
        except gojson.JSONError as error:
            raise RunnerError(f"encode prompt request: {error}") from error
        input = io.BytesIO(encoded)
    elif options.arguments:
        input = io.BytesIO(os.fsencode(options.arguments[0]))
    parsed, new_tools = config.parse_request(input)
    if new_tools is None:
        raise RunnerError("request parser returned no tool factory")
    messages = validate_request(parsed)
    workspace = _resolve_workspace(options.workspace)
    environment = load_dotenv(go_join(workspace, ".env"))
    cleanups.append(environment.aclose)
    max_attempts = resolve_max_attempts(parsed.max_attempts, getenv)
    provider_name = go_trim_space(getenv(LLM_PROVIDER_ENVIRONMENT)) or DEFAULT_PROVIDER
    selected, new_client = _select_provider(config.providers, provider_name)
    base_url = go_trim_space(getenv(LLM_BASE_URL_ENVIRONMENT)) or selected.base_url
    model = (
        go_trim_space(parsed.model)
        or go_trim_space(getenv(LLM_MODEL_ENVIRONMENT))
        or selected.default_model
    )
    if model == "":
        raise RunnerError(f"model must be set in the request or {LLM_MODEL_ENVIRONMENT}")
    api_key = ""
    if selected.api_key_environment:
        api_key = getenv(LLM_API_KEY_ENVIRONMENT)
        if go_trim_space(api_key) == "":
            api_key = getenv(selected.api_key_environment)
        if go_trim_space(api_key) == "":
            raise RunnerError(
                f"{LLM_API_KEY_ENVIRONMENT} or {selected.api_key_environment} must be set"
            )
    try:
        client = new_client(api_key, base_url, max_attempts, getenv)
    except Exception as error:
        raise wrap(f"create {selected.name} client", error) from error
    cleanups.append(_closer(client, selected.name))

    try:
        store_directory = resolve_session_directory(options.session_directory, getenv)
    except RunnerError as error:
        raise wrap("resolve session directory", error) from error
    try:
        store = Store(store_directory)
    except Exception as error:
        raise wrap("open session store", error) from error
    session_id, restored = await _open_session(store, parsed.session_id)
    observed_output: Writer = output
    log_directory = go_trim_space(options.log_directory)
    if log_directory:
        log_file = open_datetime_log(log_directory, time.time())
        cleanups.append(_log_closer(log_file))
        observed_output = TeeWriter(log_file, output)

    coordinator_task: list[asyncio.Task[None]] = []

    def cancel() -> None:
        for task in coordinator_task:
            task.cancel()

    async def stop_coordinator() -> None:
        for task in coordinator_task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    cleanups.append(stop_coordinator)
    operation_directory = go_join(store_directory, "operations", session_id)
    try:
        _make_directories(operation_directory)
    except OSError as error:
        raise wrap("create operation directory", error, "mkdir") from error
    shell = go_trim_space(getenv("SHELL")) or "/bin/sh"
    skills, skill_errors = discover_skills(go_join(workspace, ".harness", "skills"))
    names = [BASH_NAME, VIEW_IMAGE_NAME]
    if skills:
        names.append(SKILL_USE_NAME)
    tool_config = ToolConfig(
        session_id=session_id,
        getenv=getenv,
        names=names,
        translators=StaticTranslators(
            bash=bash.BashTranslator(
                bash.Config(shell=shell, directory=workspace, base_directory=operation_directory)
            ),
            view_image=viewimage.ViewImageTranslator(viewimage.Config(directory=workspace)),
        ),
    )
    configured = await new_tools(tool_config)
    if configured.close is not None:
        cleanups.append(configured.close)
    registry = configured.registry
    if registry.resolve(SKILL_USE_NAME) is not None:
        for skill in skills:
            try:
                registry.register_skill(skill)
            except Exception as error:
                raise wrap(f"register skill {go_quote(skill.path)}", error) from error
    for skill_error in skill_errors:
        try:
            write_flushed(
                flag_output,
                f"skill error> {skill_error}\n".encode("utf-8", "surrogateescape"),
            )
        except Exception as error:
            raise wrap("write skill error", error) from error

    operations = LocalOperationManager(*configured.remote_jobs)
    cleanups.append(operations.aclose)
    try:
        inputs = Inbox(restored.external_input_ids)
    except ValueError as error:
        raise wrap("open inbox", error) from error

    async def close_inbox() -> None:
        inputs.close()

    cleanups.append(close_inbox)
    effort = reasoning_effort(parsed.thinking_level)
    settings = ControlMessage(
        Mode=ControlMode.UPDATE_SETTINGS, Parameters=Settings(ReasoningEffort=effort)
    )
    await _submit(inputs, InputKind.CONTROL, gojson.encode(settings), "submit settings")
    for index, message in enumerate(messages):
        try:
            payload = gojson.encode(message.content)
        except gojson.JSONError as error:
            raise RunnerError(f"encode message {index}: {error}") from error
        message_id = (
            go_trim_space(message.message_id)
            if message.message_id is not None
            else str(uuid.uuid4())
        )
        await _submit(inputs, InputKind.EXTERNAL, payload, f"submit message {index}", message_id)
    stop = ControlMessage(Mode=ControlMode.STOP_WHEN_IDLE)
    await _submit(inputs, InputKind.CONTROL, gojson.encode(stop), "submit stop request")

    builder = new_builder(*registry.skills())
    builder.set_model(Model(ID=model, ReasoningEffort=effort))
    builder.set_system_prompt(
        parsed.system_prompt if parsed.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    )
    for definition in registry.static_definitions():
        builder.add_tool(definition.tool)

    observer = SessionObserver(session_id, observed_output, cancel)
    observer_id = store.add_observer(observer.observe)

    async def remove_observer() -> None:
        store.remove_observer(observer_id)

    cleanups.append(remove_observer)
    coordinator = Coordinator(
        Dependencies(
            tool_heartbeat_interval=options.tool_heartbeat_interval / 1e9,
            session_id=session_id,
            inbox=inputs,
            restored=restored,
            sessions=store,
            context_builder=builder,
            llm=client,
            tools=registry,
            operations=operations,
        )
    )
    task = asyncio.create_task(coordinator.run(), name="coordinator")
    coordinator_task.append(task)
    coordinator_error: BaseException | None = None
    try:
        await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
        coordinator_error = RunnerError("context canceled")
    except Exception as error:
        coordinator_error = error
    if observer.error is not None:
        raise observer.error
    if coordinator_error is not None:
        raise wrap("run coordinator", coordinator_error) from coordinator_error


def _resolve_workspace(configured: str) -> str:
    workspace = go_abs(go_trim_space(configured))
    try:
        info = os.stat(workspace)
    except OSError as error:
        raise wrap("inspect workspace", error, "stat") from error
    if not stat.S_ISDIR(info.st_mode):
        raise RunnerError(f"workspace {go_quote(workspace)} is not a directory")
    return workspace


async def _submit(
    inputs: Inbox,
    kind: InputKind,
    payload: bytes,
    label: str,
    input_id: str | None = None,
) -> None:
    try:
        await inputs.submit(
            Input(ID=input_id or str(uuid.uuid4()), Kind=kind, Payload=gojson.Raw(payload))
        )
    except Exception as error:
        raise wrap(label, error) from error


def _closer(client: Client, name: str) -> _Cleanup:
    async def close() -> None:
        try:
            await client.aclose()
        except Exception as error:
            raise wrap(f"close {name} client", error) from error

    return close


def _log_closer(file: BinaryIO) -> _Cleanup:
    async def close() -> None:
        try:
            file.close()
        except Exception as error:
            raise wrap("close session log", error) from error

    return close


def resolve_max_attempts(requested: int | None, getenv: Getenv) -> int:
    max_attempts = DEFAULT_MAX_ATTEMPTS
    if requested is not None:
        max_attempts = requested
    elif value := go_trim_space(getenv(LLM_MAX_ATTEMPTS_ENVIRONMENT)):
        if _ATOI.fullmatch(value) is None:
            raise RunnerError(
                f"parse {LLM_MAX_ATTEMPTS_ENVIRONMENT}: strconv.Atoi: parsing "
                f"{go_quote(value)}: invalid syntax"
            )
        max_attempts = int(value)
        if not -_MAX_INT64 - 1 <= max_attempts <= _MAX_INT64:
            raise RunnerError(
                f"parse {LLM_MAX_ATTEMPTS_ENVIRONMENT}: strconv.Atoi: parsing "
                f"{go_quote(value)}: value out of range"
            )
    if max_attempts <= 0:
        raise RunnerError("max attempts must be positive")
    return max_attempts


def _select_provider(
    providers: list[Provider], name: str
) -> tuple[Provider, Callable[[str, str, int, Getenv], Client]]:
    for provider in providers:
        if provider.name == name:
            if provider.new_client is None:
                raise RunnerError(f"provider {go_quote(name)} has no client factory")
            return provider, provider.new_client
    available = ", ".join(provider.name for provider in providers)
    raise RunnerError(f"unsupported provider {go_quote(name)}; available providers: {available}")


def resolve_session_directory(configured: str, getenv: Getenv) -> str:
    """Raises RunnerError when no absolute state location is available."""
    if configured := go_trim_space(configured):
        return go_abs(configured)
    state_home = getenv("XDG_STATE_HOME")
    if not os.path.isabs(state_home):
        user_home = getenv("HOME") or os.environ.get("HOME", "")
        if user_home == "":
            raise RunnerError(
                "find home directory: $HOME is not defined; specify -session-directory to override"
            )
        if not os.path.isabs(user_home):
            raise RunnerError(
                "set an absolute XDG_STATE_HOME or HOME, or specify -session-directory"
            )
        state_home = go_join(user_home, ".local", "state")
    return go_join(state_home, DEFAULT_SESSION_DIRECTORY)


def open_datetime_log(directory: str, now: float) -> BinaryIO:
    """Open `<directory>/YYYYMMDD-HHMMSS.jsonl` (UTC) for appending, creating it with mode 0600."""
    try:
        _make_directories(directory)
    except OSError as error:
        raise wrap("create log directory", error, "mkdir") from error
    name = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + ".jsonl"
    path = go_join(directory, name)
    try:
        return open(path, "ab", buffering=0, opener=_open_private)
    except OSError as error:
        raise wrap("open session log", error, "open") from error


def _open_private(path: str, flags: int) -> int:
    return os.open(path, flags | os.O_CLOEXEC, 0o600)


def _make_directories(path: str) -> None:
    """Go's os.MkdirAll with mode 0700: every created directory gets the mode."""
    if os.path.isdir(path):
        return
    parent = os.path.dirname(path.rstrip("/"))
    if parent and parent != path:
        _make_directories(parent)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if not os.path.isdir(path):
            raise


@dataclass(slots=True)
class _EnvironmentChange:
    name: str
    value: str | None


@dataclass(slots=True)
class EnvironmentScope:
    """Process environment changes made from `.env`, undone in reverse by `close`."""

    changes: list[_EnvironmentChange] = field(default_factory=list)

    def set(self, name: str, value: str) -> None:
        previous = os.environ.get(name)
        try:
            os.environ[name] = value
        except (ValueError, OSError) as error:
            raise wrap(f"set environment variable {go_quote(name)}", error) from error
        self.changes.append(_EnvironmentChange(name, previous))

    def close(self) -> None:
        error: BaseException | None = None
        for change in reversed(self.changes):
            try:
                if change.value is None:
                    os.environ.pop(change.name, None)
                else:
                    os.environ[change.name] = change.value
            except (ValueError, OSError) as failure:
                error = join(
                    error,
                    wrap(f"restore environment variable {go_quote(change.name)}", failure),
                )
        self.changes.clear()
        if error is not None:
            raise error

    async def aclose(self) -> None:
        self.close()


def load_dotenv(path: str) -> EnvironmentScope:
    """Apply `NAME=value` lines from `path` without overriding the process environment.

    SANDBOX_EGRESS_PROXY is the exception: it always applies and is mirrored to
    HTTPS_PROXY. A missing file changes nothing.
    """
    try:
        with open(path, "rb") as file:
            encoded = file.read()
    except FileNotFoundError:
        return EnvironmentScope()
    except OSError as error:
        raise wrap("read environment file", error, "open") from error
    values: dict[str, str] = {}
    for raw_line in encoded.decode("utf-8", "surrogateescape").split("\n"):
        line = go_trim_space(raw_line)
        if line == "" or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = go_trim_space(name)
        if name == "":
            continue
        values[name] = go_trim_space(value)
    scope = EnvironmentScope()
    try:
        for name, value in values.items():
            if name in os.environ and name != "SANDBOX_EGRESS_PROXY":
                continue
            scope.set(name, value)
        if proxy := os.environ.get("SANDBOX_EGRESS_PROXY", ""):
            scope.set("HTTPS_PROXY", proxy)
    except RunnerError as error:
        try:
            scope.close()
        except Exception as close_error:
            raise join(error, close_error) from error
        raise
    return scope


async def _open_session(store: Store, requested: str | None) -> tuple[SessionID, ResumeState]:
    session_id = go_trim_space(requested) if requested is not None else str(uuid.uuid4())
    try:
        return session_id, await store.resume(session_id)
    except FileNotFoundError:
        pass
    except Exception as error:
        raise wrap(f"open session {go_quote(session_id)}", error) from error
    try:
        snapshot: Snapshot = await store.create(session_id)
    except Exception as error:
        raise wrap(f"create session {go_quote(session_id)}", error) from error
    return session_id, ResumeState(snapshot=snapshot)


class SessionObserver:
    """Writes each item persisted for one session as a JSON line.

    The first write failure is kept in `error` and cancels the run; later
    items are not written.
    """

    __slots__ = ("cancel", "error", "output", "session_id")

    def __init__(self, session_id: SessionID, output: Writer, cancel: Callable[[], None]) -> None:
        self.session_id = session_id
        self.output = output
        self.cancel = cancel
        self.error: RunnerError | None = None

    def observe(self, session_id: SessionID, item: Item) -> None:
        if session_id != self.session_id or self.error is not None:
            return
        try:
            write_session_item(self.output, item)
        except RunnerError as error:
            self.error = error
            self.cancel()


def write_session_item(output: Writer, item: Item) -> None:
    try:
        encoded = gojson.encode(item)
    except gojson.JSONError as error:
        raise wrap(f"encode session item {item.Sequence}", error) from error
    try:
        write_flushed(output, encoded + b"\n")
    except Exception as error:
        raise wrap(f"write session item {item.Sequence}", error) from error
