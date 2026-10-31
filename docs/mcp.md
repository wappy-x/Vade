# MCP tools in Vade chat

Vade chat can use the tools of MCP servers. MCP is the Model Context Protocol.
An MCP server is a program that offers tools, for example a GitHub, database
or browser tool. Many tools you already use have an MCP server. Add the server
once, and the model in `vade chat` can call its tools next to the native
workspace tools.

## What is supported

| Part | State |
| --- | --- |
| Transport | stdio only. Vade starts the server as a child process and sends JSON-RPC 2.0 messages, one per line, on its stdin and stdout. HTTP and SSE servers are not supported. |
| Protocol revision | Vade asks for MCP `2025-06-18`. It also accepts a server that answers `2025-03-26` or `2024-11-05`. A server that answers another revision does not start. |
| Tools | `tools/list` (up to 16 pages and 256 tools) and `tools/call`. |
| Resources | `resources/list` only. `vade mcp test` shows the list. Vade does not read resources yet, and the model does not see them. |
| Prompts | `prompts/list` only. `vade mcp test` shows the list. Vade does not get prompts yet, and the model does not see them. |
| Progress | `notifications/progress` for a running call restarts the call timer. See [Time limits](#time-limits-and-size-limits). |
| Server requests | Vade answers `ping`. It answers every other request (sampling, roots, elicitation) with error `-32601`, because Vade does not offer these features. |
| Child agents | Child agents (`--children`) get the native tools only. |

## Add a server

A server is a program on the computer where chat runs. Vade starts it and
talks to it over standard input and output.

### A first server with no extra install

The repository has a small example server,
[`examples/mcp_word_count.rs`](../examples/mcp_word_count.rs). It has one tool,
`word_count`, and needs no token, no network and no runtime besides the Rust
that `scripts/install.sh` installs. From the checkout root:

```sh
cargo build --release --offline --locked --manifest-path tools/Cargo.toml --example mcp-word-count
vade mcp add words -- "$PWD/tools/target/release/examples/mcp-word-count"
vade mcp test words
```

`vade mcp test words` shows `Tools (1):` and `words__word_count`. In chat, ask
for example "Use the word_count tool on: one two three". Vade asks before the
call; answer `y`.

### A server you install

Most servers need their own program, and often a token. For example, the
GitHub server needs the `github-mcp-server` program and a GitHub token:

```sh
vade mcp add github --env GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_TOKEN -- github-mcp-server stdio
vade mcp test github
vade tools
```

### Options

`vade mcp add NAME [options] -- COMMAND [ARG...]` writes the server to your
user list. Everything after `--` is the command and its arguments. Put
the options and the name before `--`; any other word there is an error. The
options are:

| Option | What it does |
| --- | --- |
| `--project` | Write to `.vade/mcp.json` in the current project instead of your user list. |
| `--env KEY=VALUE` | Set an environment variable for the server. Repeat it for more variables, each with a different name. |
| `--cwd DIR` | Run the server in `DIR`, a folder inside the project folder. `DIR` is relative, without `..`, and a link in it may not lead out of the project. |
| `--timeout-ms N` | The time limit for one call, from 1 to 600000 ms. The default is 60000. |
| `--disabled` | Add the server, but do not start it. |

The other commands:

| Command | What it does |
| --- | --- |
| `vade mcp list` | Shows each server, its list (user or project), its state (enabled, disabled or not trusted) and its command. It shows environment variable names, not their values. |
| `vade mcp test NAME` | Starts the server, does the handshake, lists its tools, resources and prompts, and stops it. Use it after you add a server. |
| `vade mcp remove NAME [--project]` | Removes the server from your user list, or from the project list with `--project`. |
| `vade mcp trust NAME` | Lets a server from the project list start in chat. See [Project servers need your yes](#project-servers-need-your-yes). |
| `vade tools` | Lists the native workspace and memory tools, then the tools of each server. The servers start for the list and stop at the end. `VADE_MEMORY=0` leaves out the memory tools. |

All commands take `--workspace DIR` to name the project. The default is the
current folder. `vade tools` and `vade mcp` run the installed harness. With a
source build, run `vade-harness tools` and `vade-harness mcp` directly.

## The files

Vade reads two files when chat starts:

- The user list: `mcp.json` in the Vade home. That is `$VADE_HOME/mcp.json`,
  or `~/.vade/mcp.json` when `VADE_HOME` is not set. Chat uses the Vade home
  that keeps its history (`--vade-home`), which `vade chat` and `vade launch`
  set to the same folder. With `VADE_CHAT_IN_PROJECT=1`, history goes to the
  project, and these commands give chat the Vade home with `--mcp-home`, so
  chat still reads this file.
- The project list: `.vade/mcp.json` in the project folder.

A project entry replaces a user entry with the same name. The format is:

```json
{
  "servers": {
    "github": {
      "command": "github-mcp-server",
      "args": ["stdio"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_TOKEN"},
      "cwd": ".",
      "enabled": true,
      "timeout_ms": 60000
    }
  }
}
```

Only `command` is necessary. A file can use the key `mcpServers` in place of
`servers`, so you can copy a server list from another MCP client. A file may
not have both keys. Vade ignores keys it does not know. An entry with `url`,
`headers` or a `type` other than `stdio` is skipped with a note.

A server name has 1 to 32 ASCII letters, digits or hyphens, and its first
character is a letter or a digit. Underscores are not permitted, because
`__` separates the server name from the tool name.

`vade mcp add` writes the file with owner-only permissions (mode 600),
because `env` can hold tokens. If you edit the file by hand, keep it private.
A broken file is never overwritten: `vade mcp add` stops with an error. In
chat, Vade skips a broken file or entry and prints one line about it.

Vade refuses a config file, or its folder, that is a symlink, and a config
file that is not a regular file (for example a FIFO). The Vade home may not
contain `..`. Run one `vade mcp` command at a time: the commands take no
lock, so two changes at the same moment can lose one of them.

## Project servers need your yes

A project list comes with the project. If you clone a project, its
`.vade/mcp.json` could name any command. So a server from the project list
does not start until you say yes with `vade mcp trust NAME`. Chat prints one
line for each server that waits for your yes.

Vade keeps the yes in your user list, under `trusted`, with a copy of the
entry. If the project changes the entry (for example its command or its
arguments), the yes no longer applies, and you must run `vade mcp trust NAME`
again. `vade mcp add --project` records the yes for you, because you wrote the
entry yourself.

## How the model sees the tools

Each server tool gets the name `<server>__<tool>`. Characters outside
`A-Z a-z 0-9 _ -` change to `_`. For example, tool `read.file` on server
`fs` is `fs__read_file`. The description starts with
`MCP server <server>, tool <tool>.` and the parameters are the server's input
schema.

The native tools come first, unchanged. The MCP tools follow, in the order of
the servers and of their tool lists. Vade skips a tool and prints a note when:

- the full name is longer than 64 characters,
- two tools get the same name,
- the input schema is not an object schema,
- the tool definition is larger than 16 KiB, or
- the server has more than 128 tools (Vade keeps the first 128).

Chat reads the files and starts the servers once, when chat starts. The
tool list of a conversation is fixed when the conversation starts. A resumed
session keeps its list, MCP tools included. If a server does not run or no
longer has a tool, a call returns an `mcp_unavailable` error to the model. A
server that did not start when chat began stays out of that chat. `/new`
keeps the running servers and their tool lists. To use a changed file or a
server's new tools, end chat and start it again.

## Approvals

MCP calls follow `--approval`:

| Setting | MCP calls |
| --- | --- |
| `ask` (default) | Vade shows the server, the tool and the arguments, and waits for your yes before each call. |
| `auto` | Vade shows the call and runs it. |
| `never` | Vade denies every MCP call. |

When a call does not run, Vade tells the model why: you declined it, Vade
could not show the question (for example, the arguments hold characters that
the terminal would hide), no one could answer the question, or the setting is
`never`. The server is not asked.
A run without a terminal (for example `--prompt`, or chat with input from a
pipe) cannot answer a question. Under `ask`, such a run declines each MCP call
and the model gets an `approval_required` result. Use `--yes` to let such a
run make MCP calls.

Vade does not use the tool annotations (for example `readOnlyHint`) to skip
a question, because the server supplies them.

## Time limits and size limits

| Limit | Value |
| --- | --- |
| Start and handshake | The server's `timeout_ms` for each step |
| One call | The server's `timeout_ms` (default 60000 ms) |
| One call with progress | Each progress notification for the call restarts the timer, up to 10 times `timeout_ms` in total. Progress does not extend the start, the handshake or the lists. |
| One message from the server | 8 MiB |
| One result for the model | 32 KiB, or the smaller per-call output limit of the run |
| Tools for each server | 128 |
| Nesting of one tool definition | 32 levels; a deeper tool is left out with a note |
| All MCP tool definitions together | 8192 JSON values and 512 KiB; later tools are left out with a note, so the chat history keeps its room |

When a call passes its time limit, Vade sends `notifications/cancelled`,
stops the server's process group and tells the model that the outcome is
unknown. The next call starts the server again. A stop at a time limit does
not use the one restart of a server that stopped by itself. When
you cancel a run, Vade sends the same notification. This includes a cancel
during a restart, the handshake or the lists. The notification does not prove
that the server stopped its work. A call that you cancel after Vade sent it
is saved with an unknown outcome, as after a time limit. Check the server's
state before you run that call again.

Vade writes to each server from a separate thread. If a server stops reading
its input, a call still ends at its time limit or when you cancel it. When a
request is still unwritten at that time (the pipe to the server is full),
Vade stops that server. After a time limit, the next call starts it again;
after a cancel, the next call starts it again once. A server that
reads nothing but has room left in the pipe looks like a slow server: each
call ends at its time limit, and Vade stops the server.

A result keeps its text parts. An image or audio part becomes one line that
names its type and size. A binary resource becomes one line with its URI. A result with `isError` reaches the model
as `{"ok":false,"error":"mcp_tool_error","message":...}`. A result that is
too large is cut, and the last line tells how many bytes were kept.

## When a server fails

- Before servers start, chat prints one line with their number, for
  example `vade: starting 2 MCP servers`.
- A server that does not start is left out. Chat prints one line with the
  reason and the end of the server's error output. The other servers start.
- When Vade quotes a server's error output (to you or to the model), it
  shows each configured `env` value of 4 or more characters as
  `[env value]`, also a value that runs over several lines. Vade hides each
  value as the output arrives, before it cuts old output. Output that may
  still be the start of a value is held back until the next output decides
  it. Where values overlap, Vade hides all the characters that any of them
  covers, so a value that arrives in parts, starts with a shorter value or
  starts inside another value is hidden whole. Output that stops inside a
  value shows no part of it. Shorter values are not
  hidden, because they also match ordinary words. Do not put a secret
  shorter than 4 characters in `env`.
- A line on stdout that does not start with `{` (for example a startup
  banner, in any encoding) is skipped. A line that starts with `{` but is
  not valid JSON fails the current call. The server stays connected.
- A server stops when its stdout closes. If a server stops, Vade starts it
  again once, at the next call to it. If it
  stops again, its calls fail until you restart the chat. A restart keeps the
  tools the chat started with; it adds no new tool. A call that was
  running when the server stopped has an unknown outcome, and the model is
  told so.
- When chat ends, Vade closes each server's stdin, waits up to 500 ms for
  its stdout to close, and then stops the server's process group. A server
  that Vade stops for a full input pipe, or before a restart, also stops
  with its process group. So the child of a wrapper command stops too, also
  when the wrapper exited first.

Servers run in their own process group, so Ctrl-C in chat does not reach
them. They get only `HOME`, `PATH`, `TMPDIR`, `USER`, `LOGNAME`, `SHELL`,
`LANG`, `LC_ALL`, `LC_CTYPE` and `TERM` from the chat, plus their `env`
entries. Give a server the other variables it needs, such as a token, in its
`env` entry. They run with your
permissions, outside the macOS sandbox that `workspace_shell` uses. Add only
servers you trust.

To start chat without any MCP server, set `VADE_MCP=off`.

## Where the code is

| Item | What it does |
| --- | --- |
| [`harness/src/mcp.rs`](../harness/src/mcp.rs) | Starts the servers, names the tools, renders results and sends MCP calls from chat |
| [`harness/src/mcp/client.rs`](../harness/src/mcp/client.rs) | One server process: the JSON-RPC client over stdio |
| [`harness/src/mcp/config.rs`](../harness/src/mcp/config.rs) | The two files, their entries and the trust records |
| [`harness/src/mcp/cli.rs`](../harness/src/mcp/cli.rs) | `vade mcp` and `vade tools` |
| [`harness/tests/mcp.rs`](../harness/tests/mcp.rs) | Tests against a small fake stdio server in the same test program |
