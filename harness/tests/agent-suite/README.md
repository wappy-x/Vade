# harness/tests/agent-suite

**Fixtures for the agent task suite: 12 mixed tasks for the native coding
agent, each with a deterministic check.** The suite measures the harness and
the served model together on real work, not on one kind of task. `cargo test`
does not run the tasks. A suite runner copies each task to a new folder,
starts `vade-harness agent` against a running server and checks the result.

The fixtures are part of this repository and use its [license](../../../LICENSE)
(Apache-2.0). They hold no secrets, credentials or machine facts.

[`agent_suite_fixtures.rs`](../agent_suite_fixtures.rs) checks in `cargo test` that
every input and fixture file a task names exists and is tracked by Git, and
that no suite file is untracked or ignored. It also keeps every suite file
under 5 MB and refuses the file types that the release scan blocks (`.ptx`,
`.safetensors`, `.log`, `.pem`, `.key`). For this reason the log inputs of
`shell-count` end in `.log.txt`. A runner copies only Git-tracked files into a
task workspace.

## The tasks

| Task | What the agent must do | How the result is checked |
| --- | --- | --- |
| `fix-failing-test` | Find and fix a bug in a small Rust crate so that its tests pass | `cargo test` exits 0; `tests/` unchanged |
| `implement-from-spec` | Write a function from a Markdown spec | Hidden tests copied in after the run pass |
| `multi-file-rename` | Rename a type in every file of a crate | No old name left; exact count of the new name; `cargo test` exits 0 |
| `markdown-spec-qa` | Read a spec and answer five questions | Exact answer file |
| `pdf-table` | Read a PDF invoice and write its table as CSV | Exact CSV file |
| `shell-count` | List, search and count log files with shell commands | Exact report file |
| `write-unit-test` | Write and run a test file for a working function | `cargo test` exits 0; the tests fail against each of four broken copies of the function |
| `follow-agents-md` | Make a change that follows the rules in `AGENTS.md` | Exact changelog line, a required file header, a protected folder unchanged |
| `long-context-digest` | Answer questions about a generated repository digest of 241,865 bytes | Exact answer file |
| `cancel-resume` | Keep working after a cancel in the middle of the task | The cancel took effect; exact output files after `continue` |
| `session-resume` | Use a fact that only the first run of the session read; the file is gone before the resumed run | The first run writes nothing; exact file content after the resumed run |
| `structured-output` | Answer one chat request in JSON mode (`response_format` `json_object`) | The reply parses as JSON and equals the expected object |

`pdf-table` is easier when the harness reads PDF files as text. Without that,
the agent can still read the uncompressed PDF source. `structured-output`
needs a server that accepts `response_format`; a runner reports a refusal
(HTTP 400, 404, 422 or 501) as unsupported, not as a failure. A server that
ignores the field can still pass when the model follows the system message.

## One task folder

| Item | What it holds |
| --- | --- |
| `task.txt` | The task: id, title, steps with their prompts, and the checks |
| `workspace/` | The files the agent starts with |
| `expected/`, `hidden/`, `mutants/`, `request.json` | Exact expected files, tests the agent never sees, broken copies of a source file for `fails-with`, and the request for a direct task |
| `solve.script` | A correct scripted model transcript. A replay of it must pass every check; this proves the checks with no model |
| `wrong.script` | Optional: a near miss, such as a changed test file or a wrong answer line. A replay of it must fail at least one check |

`task.txt` uses `key: value` lines. A key with an empty value takes the
indented lines after it (two leading spaces removed). `#` starts a comment.

| Key | Meaning |
| --- | --- |
| `id`, `title` | The task name (the same as the folder) and one line about it |
| `mode` | `agent` (default) or `direct`, one chat request with no harness |
| `budget_seconds` | Active work time for each harness run |
| `generate` | A generated input file. `repo-digest-v1` adds `DIGEST.md` |
| `request` | For `direct`: the request body; the runner sets `model` and `stream` |
| `inputs` | The workspace files that the task reads. Each must exist and be tracked by Git |
| `needs` | A feature that the task uses, for the card |
| `step` | Starts one harness run: `new` (a new session), `resume` (the same session) or `cancel` (a new session that is canceled) |
| `prompt` | The prompt of the last step |
| `cancel_at_call` | For `cancel`: the model call to hold; the runner sends `/cancel` while it waits |
| `remove_before` | Workspace paths that the runner deletes before the last step starts |
| `step_check` | One check that runs right after the last step, on the workspace as that step left it |
| `check` | One check that runs after the final step; all checks must pass |

The harness runs with `--approval auto`, a private `--vade-home`, `--session
suite` (or `--resume suite`) and `--prompt`. A `cancel` step sends its prompt
on standard input instead, then `/cancel`.

## Checks

Paths are relative to the task workspace after the run; `FIXTURE` paths are
relative to the task folder. File compares ignore trailing spaces, carriage
returns and trailing empty lines.

| Check | Passes when |
| --- | --- |
| `last-exit-zero` | The last harness run exited 0 |
| `held-closed` | The harness closed the held model call (the cancel took effect) |
| `exists PATH` | The path exists |
| `file-equals PATH FIXTURE` | The file text equals the fixture |
| `json-equals PATH FIXTURE` | Both parse as JSON and are equal; member order is free |
| `contains PATH TEXT` | The file holds the text |
| `starts-with PATH TEXT` | The file starts with the text |
| `absent PATH TEXT` | No file under the path holds the text |
| `count PATH N TEXT` | The text occurs exactly N times under the path |
| `count-min PATH N TEXT` | The text occurs at least N times under the path |
| `unchanged PATH` | The file or folder is byte for byte the same as at the start |
| `reply-contains TEXT` | The printed final reply holds the text |
| `run ARGV` | The command exits 0 in the workspace |
| `run-with FIXTURE DEST -- ARGV` | With the fixture copied over DEST, the command exits 0 |
| `fails-with FIXTURE DEST -- ARGV` | The command exits 0 as is, and exits nonzero with the fixture copied over DEST |

`run-with` and `fails-with` put the original file back afterward.

## Scripted transcripts

In `solve.script` and `wrong.script`, each line that starts with `>>> ` begins one model reply:

| Line | Reply |
| --- | --- |
| `>>> call NAME` | One tool call; the JSON arguments follow on the next lines |
| `>>> patch` | One `apply_patch` call; the raw patch text follows |
| `>>> answer` | The final answer; its text follows |
| `>>> json` | A raw chat completion body, sent as is |

## The card

A run gives one card in JSON and Markdown. Each task row has the status
(`PASS`, `FAIL`, `UNSUPPORTED` or `ERROR`), the checks passed, the model
calls, the tool turns from the harness `Native run:` line, the prompt and
completion tokens from the server `usage` objects, and the wall time.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
