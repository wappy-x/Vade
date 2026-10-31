# harness/tests/agent-suite

**Fixtures for the agent task suite: 22 mixed tasks for the native coding
agent, each with a deterministic check.** The suite measures the harness and
the served model together on real work, not on one kind of task.
`vade-harness suite` copies each task to a new folder, starts
`vade-harness agent` against a model endpoint and checks the result. Its dry
run replays each task's scripts with no model, and `cargo test` runs that dry
run ([`suite_dry_run.rs`](../suite_dry_run.rs)). How to run the suite, add a
task and read the grade: [Agent task suite](../../../docs/harness-suite.md).

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

| Task | Category | What the agent must do | How the result is checked |
| --- | --- | --- | --- |
| `edit-indent-scope` | `edit-in-place` | Change one of two identical lines that differ only in indentation | Exact count of each line; `tests/` unchanged; `cargo test` exits 0 |
| `edit-lines-two-files` | `edit-in-place` | Read two files, then change one line in each with line edits | Exact files; `cargo test` exits 0 |
| `follow-agents-md` | `edit-in-place` | Make a change that follows the rules in `AGENTS.md` | Exact changelog line, a required file header, a protected folder unchanged |
| `multi-file-rename` | `multi-file-refactor` | Rename a type in every file of a crate | No old name left; exact count of the new name; `cargo test` exits 0 |
| `extract-module` | `multi-file-refactor` | Move three functions into a new module and keep the old paths | Old definitions gone; `cargo test` and a hidden test of the new paths exit 0 |
| `fix-failing-test` | `bug-fix` | Find and fix a bug in a small Rust crate so that its tests pass | `cargo test` exits 0; `tests/` unchanged |
| `fix-utf8-slice-boundary` | `bug-fix` | Fix a byte slice that panics inside a multi-byte character | `cargo test` exits 0; `tests/` unchanged |
| `implement-from-spec` | `new-feature` | Write a function from a Markdown spec | Hidden tests copied in after the run pass |
| `write-unit-test` | `new-feature` | Write and run a test file for a working function | `cargo test` exits 0; the tests fail against each of four broken copies of the function |
| `markdown-spec-qa` | `document-reading` | Read a spec and answer five questions | Exact answer file |
| `pdf-table` | `document-reading` | Read a PDF invoice and write its table as CSV | Exact CSV file |
| `docx-policy-qa` | `document-reading` | Read a Word document and answer five questions | Exact answer file |
| `html-rate-table` | `document-reading` | Read an HTML page and write one of its two tables as JSON | The JSON equals the expected array |
| `shell-count` | `shell-build` | List, search and count log files with shell commands | Exact report file |
| `build-and-run` | `shell-build` | Fix a build error, run the tool and save its output | Exact output file; `cargo build` exits 0 |
| `long-context-digest` | `long-context` | Answer questions about a generated repository digest of 241,865 bytes | Exact answer file |
| `recover-stale-path` | `tool-error-recovery` | The prompt names a file that has moved; find it and fix it there | No file at the old path; `cargo test` exits 0 |
| `recover-failed-patch` | `tool-error-recovery` | A patch written from the prompt does not apply; read the file and patch again | `cargo test` exits 0; `tests/` unchanged |
| `cancel-resume` | `session-resume` | Keep working after a cancel in the middle of the task | The cancel took effect; exact output files after `continue` |
| `session-resume` | `session-resume` | Use a fact that only the first run of the session read; the file is gone before the resumed run | The first run writes nothing; exact file content after the resumed run |
| `refuse-outside-path` | `permission` | The task needs a file outside the workspace; change nothing and say so | Workspace unchanged; the reply names the file |
| `structured-output` | `structured-output` | Answer one chat request in JSON mode (`response_format` `json_object`) | The reply parses as JSON and equals the expected object |

`pdf-table` is easier when the harness reads PDF files as text. Without that,
the agent can still read the uncompressed PDF source. `structured-output`
needs a server that accepts `response_format`; the runner reports a refusal
(HTTP 400, 404, 422 or 501) as `UNSUPPORTED`, not as `FAIL`. An
`UNSUPPORTED` run is not completed, so the card lists it with the failed
runs. A server that ignores the field can still pass when the model follows
the system message. The runner sends the request with `"stream": true`, so
that the live card can compare its token IDs.

## One task folder

| Item | What it holds |
| --- | --- |
| `task.txt` | The task: id, title, steps with their prompts, and the checks |
| `workspace/` | The files the agent starts with |
| `expected/`, `hidden/`, `mutants/`, `request.json` | Exact expected files, tests the agent never sees, broken copies of a source file for `fails-with`, and the request for a direct task |
| `solve.script` | A correct scripted model transcript. A replay of it must pass every check; this proves the checks with no model |
| `wrong.script` | A near miss, such as a changed test file or a wrong answer line. A replay of it must fail at least one check |

`task.txt` uses `key: value` lines. A key with an empty value takes the
indented lines after it (two leading spaces removed). `#` starts a comment.

| Key | Meaning |
| --- | --- |
| `id`, `title` | The task name (the same as the folder) and one line about it |
| `category` | One word that groups the task on the card, such as `bug-fix` |
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
| `missing PATH` | The path does not exist |
| `file-equals PATH FIXTURE` | The file text equals the fixture |
| `lines-equal PATH FIXTURE` | The file has the same non-blank lines as the fixture, in the same order; blank lines are not compared |
| `json-equals PATH FIXTURE` | Both parse as JSON and are equal; member order is free |
| `contains PATH TEXT` | The file holds the text |
| `starts-with PATH TEXT` | The file starts with the text |
| `absent PATH TEXT` | No file under the path holds the text |
| `count PATH N TEXT` | The text occurs exactly N times under the path |
| `count-min PATH N TEXT` | The text occurs at least N times under the path |
| `unchanged PATH` | The file or folder is byte for byte the same as at the start. Hidden files count; `target` and `.git` folders are not compared, and a symbolic link is compared as a link |
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

In a reply, `{{edit_tag PATH}}` becomes the edit tag that
`workspace_read_lines` gives for that file as it is when the reply goes out.
A model copies this tag from the read result; the script cannot know it in
advance.

## The card

Each live run gives one card in JSON and Markdown. Each task row has the
category, the status (`PASS`, `FAIL`, `UNSUPPORTED` or `ERROR`), the checks
passed, the model calls, the tool turns from the harness `Native run:` line,
the prompt and completion tokens from the server `usage` objects, a digest
of the streamed token IDs and the wall time. A summary card compares the
runs and gives the grade; see [Agent task suite](../../../docs/harness-suite.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
