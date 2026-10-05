"""Session history and operation state in local JSONL files.

`Store(directory)` creates the directory (0700) and stores each session in
`<id>.session.jsonl`. The first record is the version 2 session header; each
later record is an item or an operation update. Records are appended with
O_APPEND and synced. A torn tail after the last newline is ignored on read and
truncated before the next append. `create` and `fork` publish a complete file
by atomic rename and silently replace an existing session.

Reading a session replays and validates every transition after the last fork
record; items up to and including that fork are inherited without validation.
"""

from unreal_agent.sessionstore.localfile.store import SESSION_FILE_SUFFIX, Store

__all__ = ["SESSION_FILE_SUFFIX", "Store"]
