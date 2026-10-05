"""The agent runner CLI: one JSON request in, persisted session items out as JSONL.

`await run_main(args, getenv, input, output, stderr, config)` parses Go-style
flags (`-p`, `-session-directory`, `-workspace`, `-log-directory`,
`-tool-heartbeat-interval`), reads the request from `-p`, the one positional
argument, or `input`, and runs a coordinator until the session is idle. It
returns 0 on success or help, 130 when the calling task is cancelled, and 1
after writing `{"type":"error","message":...}` to `output` and
`<name>: <error>` to `stderr`. `run` is the same without the exit-code mapping.

Wiring: validate the request, resolve the workspace, apply `<workspace>/.env`
to the process environment for the run (existing variables win, except
SANDBOX_EGRESS_PROXY, which is mirrored to HTTPS_PROXY), create the provider
client, open or create the session in the localfile store, build the Bash and
ViewImage translators and discover skills from `<workspace>/.harness/skills`,
seed the inbox with the restored input IDs, and submit UpdateSettings, the
user messages, and StopWhenIdle before running the coordinator.

Output: every item persisted for the session is one `gojson` line, written
and flushed to the optional `<log-directory>/YYYYMMDD-HHMMSS.jsonl` and then
to `output`. The first write failure cancels the run and becomes its error.

Streams are binary: `Reader.read() -> bytes` and `Writer.write(bytes)` with
`flush()`. Writes happen synchronously on the event loop thread, like Go's
observer callback.
"""

from unreal_agent.runner.config import (
    Client,
    Config,
    Getenv,
    Provider,
    RequestParser,
    ToolConfig,
    ToolFactory,
    Tools,
)
from unreal_agent.runner.errors import JoinedError, RunnerError
from unreal_agent.runner.flags import FlagError, HelpError
from unreal_agent.runner.providers import default_providers
from unreal_agent.runner.request import Request, RequestMessage, decode_request
from unreal_agent.runner.run import DEFAULT_SYSTEM_PROMPT, run, run_main
from unreal_agent.runner.streams import Reader, Writer

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Client",
    "Config",
    "FlagError",
    "Getenv",
    "HelpError",
    "JoinedError",
    "Provider",
    "Reader",
    "Request",
    "RequestMessage",
    "RequestParser",
    "RunnerError",
    "ToolConfig",
    "ToolFactory",
    "Tools",
    "Writer",
    "decode_request",
    "default_providers",
    "run",
    "run_main",
]
