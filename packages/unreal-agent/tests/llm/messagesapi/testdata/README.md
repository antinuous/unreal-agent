These responses were captured with curl from `https://api.anthropic.com/v1/messages`
using `claude-opus-5-5` and `anthropic-version: 2023-06-01` on 2026-09-25.
The SSE files contain the response bodies as received, including padding and pings.

- `live-text.sse`: max_tokens 96, “Reply with exactly: hello”.
- `live-tools.sse`: max_tokens 2048, adaptive thinking with low effort; ask for two
  parallel `capture` calls with values `alpha` and `beta`. The tool schema is an
  object with one required string property, `value`.
- `live-tool-results.sse`: replay the preceding assistant calls and answer them
  with results in reverse call order, max_tokens 256, same thinking settings.
- `live-thinking.sse`: max_tokens 2048, adaptive thinking with high effort; find
  the smallest positive n with remainders 5, 7, 11 modulo 17, 19, 23 respectively.
  The thinking block has no display text but carries a signature.
- `live-truncated-tool.sse`: max_tokens 32, automatic tool selection, refreshed
  with `genv bash -c 'curl ...'`. The `capture` tool description is
  "Capture a string value." and its input schema requires a string `value`.
  Prompt: "Call capture directly with value containing the numbers 1 through 200
  as a comma-separated string. Do not write explanatory text."
  The response ends with an open tool block and no `content_block_stop`.
