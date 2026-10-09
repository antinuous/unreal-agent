import asyncio
import contextlib
import copy
import http.server
import io
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast
from unittest import mock

from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent._go import go_int64
from unreal_agent.inbox import ID as InputID
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    Failure,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    RequestOptions,
    Response,
    Role,
    StopReason,
    TokenUsage,
    Usage,
)
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.runner import (
    Client,
    Config,
    Provider,
    Reader,
    Request,
    ToolConfig,
    ToolFactory,
    Tools,
    decode_request,
    run,
    run_main,
)
from unreal_agent.runner.run import LLM_API_KEY_ENVIRONMENT
from unreal_agent.session import ID as SessionID
from unreal_agent.sessionstore import BEFORE_FIRST, Item, ItemKind
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.tool import Registry

MAX_INT64: Final = 2**63 - 1

type Respond = Callable[[LLMRequest], Response]


class FakeClient:
    """`respond` runs in the model request task and may raise."""

    def __init__(self, respond: Respond | None = None) -> None:
        self.respond_function = respond
        self.calls = 0
        self.closed = False

    async def respond(self, request: LLMRequest, options: RequestOptions) -> Response:
        assert self.respond_function is not None, "unexpected model request"
        return self.respond_function(request)

    async def aclose(self) -> None:
        self.closed = True


def parse_test_request(input: Reader) -> tuple[Request, ToolFactory]:
    parsed = decode_request(input)

    async def new_tools(config: ToolConfig) -> Tools:
        return Tools(registry=Registry(config.translators, *parsed.enabled_tools(*config.names)))

    return parsed, new_tools


def runner_config(client: Client) -> Config:
    def new_client(api_key: str, base_url: str, max_attempts: int, _: object) -> Client:
        if api_key != "secret" or base_url != "https://example.com" or max_attempts != 5:
            raise ValueError("unexpected provider configuration")
        return client

    return Config(
        name="unreal-agent-runner",
        parse_request=parse_test_request,
        providers=[
            Provider(
                name="openai",
                base_url="https://example.com",
                default_model="gpt-default",
                api_key_environment="OPENAI_API_KEY",
                new_client=new_client,
            )
        ],
    )


def environment(values: dict[str, str]) -> Callable[[str], str]:
    return lambda name: values.get(name, "")


SECRET_ENVIRONMENT: Final = environment({LLM_API_KEY_ENVIRONMENT: "secret"})


def must_not_read_environment(name: str) -> str:
    raise AssertionError(f"invalid arguments reached environment setup ({name})")


class ErrorReader:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or OSError("stdin must not be read")

    def read(self) -> bytes:
        raise self.error


def reader(text: str | bytes) -> io.BytesIO:
    return io.BytesIO(text.encode() if isinstance(text, str) else text)


async def call_run_main(
    args: list[str],
    getenv: Callable[[str], str],
    input: Reader,
    config: Config,
) -> tuple[int, str, str]:
    stdout, stderr = io.BytesIO(), io.BytesIO()
    code = await run_main(args, getenv, input, stdout, stderr, config)
    return code, stdout.getvalue().decode(), stderr.getvalue().decode()


async def call_run(
    args: list[str],
    getenv: Callable[[str], str],
    input: Reader,
    output: Any,
    config: Config,
) -> BaseException | None:
    try:
        await run(args, getenv, input, output, io.BytesIO(), config)
    except Exception as error:
        return error
    return None


def causes(error: BaseException | None) -> Iterator[BaseException]:
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None:
            continue
        yield current
        pending.extend(cast(tuple[BaseException, ...], getattr(current, "errors", ())))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        pending.append(current.__cause__)


def is_error(error: BaseException | None, target: BaseException) -> bool:
    return any(cause is target for cause in causes(error))


def decode_log_items(encoded: bytes) -> list[Item]:
    if not encoded:
        return []
    assert encoded.endswith(b"\n"), "log has an unterminated final record"
    return [gojson.decode(line, Item) for line in encoded[:-1].split(b"\n")]


def decode_lines(output: str) -> list[Item]:
    return [gojson.decode(line, Item) for line in output.splitlines() if line]


def event_types(output: str) -> list[str]:
    return [cast(str, gojson.validate(line)["type"]) for line in output.splitlines() if line]


def item_kinds(output: str) -> list[ItemKind]:
    return [item.Kind for item in decode_lines(output)]


def item_sequence(output: str) -> str:
    kinds: list[str] = []
    for item in decode_lines(output):
        kind = str(item.Kind)
        if isinstance(item.Data, Input):
            kind += "." + str(item.Data.Kind)
        kinds.append(kind)
    return " ".join(kinds)


def input_ids(output: str) -> list[InputID]:
    return [
        item.Data.ID
        for item in decode_lines(output)
        if isinstance(item.Data, Input) and item.Data.Kind == InputKind.EXTERNAL
    ]


def message_response(text: str, id: str = "") -> Response:
    return Response(
        ID=id,
        Stop=StopReason.COMPLETE if id else StopReason.NONE,
        Output=[
            LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=text)),
        ],
    )


def user_messages(request: LLMRequest) -> list[str]:
    return [
        item.Data.Text
        for item in request.Input
        if isinstance(item.Data, Message) and item.Data.Role == Role.USER
    ]


class FailingItemWriter:
    """Fails each write of an item of `kind` (and, for inputs, `input_kind`)."""

    def __init__(
        self, kind: ItemKind, error: Exception, input_kind: InputKind = InputKind.NONE
    ) -> None:
        self.kind = kind
        self.input_kind = input_kind
        self.error = error
        self.writes = 0
        self.output = io.BytesIO()

    def write(self, data: bytes, /) -> int:
        self.writes += 1
        item = gojson.decode(data, Item)
        if item.Kind == self.kind:
            if (
                self.input_kind != InputKind.NONE
                and isinstance(item.Data, Input)
                and item.Data.Kind != self.input_kind
            ):
                return self.output.write(data)
            raise self.error
        return self.output.write(data)

    def flush(self) -> None:
        pass


async def wait() -> None:
    """Run the loop until no callback is ready or due."""
    loop = asyncio.get_running_loop()
    internals = cast(Any, loop)
    while True:
        await asyncio.sleep(0)
        if internals._ready:
            continue
        scheduled = internals._scheduled
        if scheduled and not scheduled[0].cancelled() and scheduled[0].when() <= loop.time():
            continue
        return


async def _inline_to_thread[T](function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    return function(*args, **kwargs)


def _inline_thread[T](function: Callable[[], T]) -> asyncio.Future[T]:
    future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    try:
        future.set_result(function())
    except Exception as error:
        future.set_exception(error)
    return future


async def _inline_blocking[T](
    function: Callable[..., T], *args: Any, on_cancel: Callable[[], None] | None = None
) -> T:
    return function(*args)


@contextlib.contextmanager
def without_threads() -> Iterator[None]:
    """Run the store's, the file primitive's, and the stdin read's blocking work inline.

    looptime advances fake time whenever the loop is idle, including while a
    worker thread runs, so fake-time tests keep all work on the loop.
    """
    with (
        mock.patch.object(asyncio, "to_thread", _inline_to_thread),
        mock.patch("unreal_agent.primitives.file.run_blocking", _inline_blocking),
        mock.patch("unreal_agent.runner.run.run_in_thread", _inline_thread),
    ):
        yield


def log_json(value: Any) -> gojson.Raw:
    return gojson.Raw(gojson.encode(value, deterministic=True))


def to_valid_utf8(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8")


def reaches_compaction_threshold(input: int, output: int) -> bool:
    """Whether the fuzzed token count reaches the fixtures' maximum compaction threshold."""
    return go_int64((input & MAX_INT64) + (output & MAX_INT64)) == MAX_INT64


def write_user_settings(config_home: Path, contents: str) -> None:
    directory = config_home / "unreal-agent"
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "settings.json").write_text(contents)


def fuzz_log_usage(
    text: str, input: int, cached: int, written: int, output: int, reasoning: int
) -> Usage:
    return Usage(
        InputTokens=input & MAX_INT64,
        CachedInputTokens=cached & MAX_INT64,
        CacheWriteInputTokens=written & MAX_INT64,
        OutputTokens=output & MAX_INT64,
        ReasoningTokens=reasoning & MAX_INT64,
        ByModel={
            "primary": TokenUsage(InputTokens=input & MAX_INT64, OutputTokens=output & MAX_INT64),
            "fallback": TokenUsage(
                InputTokens=input // 2,
                CachedInputTokens=cached // 2,
                CacheWriteInputTokens=written // 2,
                OutputTokens=output // 2,
                ReasoningTokens=reasoning // 2,
            ),
        },
        Raw=gojson.Raw(
            gojson.encode(
                _UsageRaw(
                    Tokens=input,
                    Cost=gojson.Raw(b"0.000000000123456789"),
                    Details=[text, None, True, gojson.Raw(b'{"nested":9007199254740993}')],
                )
            )
        ),
    )


class _UsageRaw(gojson.Struct):
    Tokens: int = 0
    Cost: gojson.Raw = gojson.Raw(b"null")
    Details: list[Any] = []


class _ReasoningRaw(gojson.Struct):
    Encrypted: str = ""
    Index: int = 0


def fuzz_log_response(text: str, usage: Usage, index: int, variant: int) -> Response:
    response = Response(
        ID=f"response-{index}",
        Usage=replace(usage, ByModel=None if usage.ByModel is None else dict(usage.ByModel)),
        Stop=[StopReason.COMPLETE, StopReason.MAX_OUTPUT_TOKENS, StopReason.REFUSED][variant % 3],
        Output=[
            LLMItem(
                ProviderID=f"reasoning-{index}",
                Type=ItemType.PROVIDER,
                Data=ProviderItem(
                    Type="reasoning",
                    Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text=text),
                    Raw=gojson.Raw(gojson.encode(_ReasoningRaw(Encrypted=text, Index=index))),
                ),
            ),
            LLMItem(
                ProviderID=f"message-{index}",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text=text, Phase="final_answer"),
            ),
        ],
    )
    if variant & 1:
        response.Failure = Failure(Code="provider_failure", Message=text)
    if variant & 2:
        response.Usage.Raw = gojson.Raw(b"")
    if variant & 4:
        response.Output = []
    return response


def copy_log_response(value: Response) -> Response:
    """Keep the oracle independent of values whose ownership passes to the coordinator."""
    return copy.deepcopy(value)


async def read_log_history(store: Store, id: SessionID, limit: int) -> list[Item]:
    items: list[Item] = []
    after = BEFORE_FIRST
    while True:
        page = await store.items(id, after, limit)
        items.extend(page.items)
        if not page.more:
            return items
        assert page.next_after > after, "history pagination did not advance"
        after = page.next_after


LOG_FUZZ_SEEDS: Final[list[tuple[bytes, str, int]]] = [
    (b"", "", 0),
    (
        bytes([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 13, 14, 15]),
        'quotes"\\\n\r\t\x00 会話\u2028',
        1,
    ),
    (bytes([3, 4, 8, 5, 8, 6, 8, 3, 9]), "precision", (1 << 53) + 1),
    (bytes([10, 3, 9, 8, 11, 8, 3]), "�invalid UTF-8", MAX_INT64),
    (bytes([4, 6, 6, 6, 6, 6, 8, 3]), "large\n" * 200, 1000000),
    (bytes([1, 2, 3, 4, 8, 3]), "provider error and resume", 12345),
    (bytes([0, 1, 2, 0, 1]), "hard stop during request", 9),
    (bytes([0, 1, 2, 0, 2]), "cancel during request", 17),
    (bytes([0, 3]), "x" * (70 << 10), 42),
]


def with_log_fuzz_seeds[F: Callable[..., Any]](test: F) -> F:
    from hypothesis import example

    for actions, text, tokens in LOG_FUZZ_SEEDS:
        test = example(actions, text, tokens, tokens // 3, tokens // 5, tokens, tokens // 2)(test)
    return test


@dataclass(slots=True)
class HTTPRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class HTTPResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@contextlib.contextmanager
def http_server(handle: Callable[[HTTPRequest], HTTPResponse]) -> Iterator[str]:
    """Serve `handle` on a loopback port and yield the base URL."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            response = handle(
                HTTPRequest(
                    self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body
                )
            )
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
