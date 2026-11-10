# Chat hooks

Hooks are shell commands that you configure for native `vade chat` agent
turns. They run on the host that runs the harness. They do not run in direct,
demo, or delegated chat. Cards, workloads, and conformance runs do not load
hooks.

## File and format

Put `hooks.conf` in the project's chat state directory:

- By default: `VADE_HOME/projects/NAME-HASH/hooks.conf`. Without `VADE_HOME`,
  the Vade home is `~/.vade`.
- With `--vade-home DIR`: `DIR/projects/NAME-HASH/hooks.conf`.
- With `--state-in-project`: `WORKSPACE/.vade/hooks.conf`.

This is the same directory that holds the project's sessions and history.
[Project instruction files](project-instructions.md) keep their existing
lookup rules. In particular, `.vade/instructions.md` is a source-tree file.
It does not move into the default chat state directory.

The file uses the existing JSON format. For example:

```json
{
  "hooks": [
    {"point": "before-tool", "command": "/bin/sh scripts/check-tool.sh"},
    {"point": "after-tool", "command": "/bin/sh scripts/tool-note.sh", "timeout_ms": 5000},
    {"point": "on-stop", "command": "printf 'Turn ended.\\n'"}
  ]
}
```

The harness reads the file once at chat start. Restart chat to load changes.
A missing file has no effect. A broken file produces one startup notice.
The harness ignores the entire broken file for that chat.

Only `point`, `command`, and `timeout_ms` are allowed in an entry. The command
must not be empty. The default timeout is 30000 milliseconds. An override
must be an integer from 1 through 300000 milliseconds. The file can hold up
to 32 entries and 64 KiB of UTF-8 text. Links and nonregular files are refused.
These limits are defined in [the hook reader](../src/hooks.rs).

## Hook points

Hooks at the same point run in file order. The tool call has a saved
`unknown_outcome` journal record before any hook can run.

| Point | Input on stdin | Effect |
| --- | --- | --- |
| `before-tool` | A JSON object with `tool` and `arguments` | Exit 0 allows the tool. Exit 2 blocks it. The model receives stderr as the reason. An empty reason gets a fixed message. Other exits, errors, and timeouts allow the tool with a warning to the user. A block stops later before hooks. |
| `after-tool` | A JSON object with `tool`, `arguments`, and `result_summary` | Runs after a dispatched tool, including one that returns an error. Stdout is appended to its result under `[after-tool hook note]`. A blocked or skipped tool has no after hook. Failures produce a warning. |
| `on-stop` | An empty JSON object | Runs once when the native agent turn ends, including error or cancellation. Stdout is shown to the user under `[on-stop hook]`. It does not enter model context. |

`arguments` is the parsed JSON object from the tool call. For example:

```json
{"tool":"workspace_read","arguments":{"path":"README.md"}}
```

The before and after points cover the parent's tool calls, child-control
calls, and automatic check calls. Child agents do not load hooks of their own.
Invalid tool names or invalid argument JSON are refused before hooks run.

Each output stream is capped at 4096 bytes. A cut stream has a note. The
result summary is a prefix of at most 4096 bytes. All after notes for a call,
including their label, share a 4096-byte cap. They also fit the remaining
tool result budget. The original tool result is kept. A cut or omitted note
produces a warning. JSON object results receive a labeled `vade_hook_note`
field so their error fields still work. If that field already exists, the
new name gets trailing underscores. See [the output tests](../src/hooks/tests.rs).

## Shell and environment

The harness uses `/bin/sh -c COMMAND`. The selected workspace root is the
working directory and the value of `VADE_PROJECT`. Hooks inherit the host
environment. The harness sets these variables for each entry:

| Variable | Value |
| --- | --- |
| `VADE_HOOK_POINT` | `before-tool`, `after-tool`, or `on-stop` |
| `VADE_TOOL` | The tool name, or an empty value for `on-stop` |
| `VADE_PROJECT` | The full canonical workspace root path |

These are user-configured host commands. They use your host permissions and
do not use the model's command sandbox or approval policy. Configure only
commands that you trust. A hook does not grant the model any extra tool
permission.

Each command starts in its own process group. Timeout and cancellation stop
that group and reap the shell. The group is also stopped after normal exit.
The shell's exit status is kept when a child holds an output pipe open.
After exit, output collection waits at most one more second, as defined in
[the hook runner](../src/hooks.rs). Incomplete output gets a labeled note.
Programs that leave the group can outlive it. Pipe reads and stdin writes do
not hold up the deadline. Before and after hooks honor turn cancellation.
Stop hooks run with their own deadlines after turn cancellation.

**Back:** [Harness docs](README.md)
