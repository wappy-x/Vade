# Agent task suite

The agent task suite measures the Vade coding agent on mixed work, not on one
kind of task. It has 22 tasks in 11 categories: edits in place, refactors
over several files, bug fixes with a failing test, new features with tests,
document reading, shell and build work, a long context, recovery after a tool
error, a resumed session, permission limits and JSON output. The tasks live
in [harness/tests/agent-suite](../harness/tests/agent-suite/README.md).

`vade-harness suite` runs the suite in two ways:

| Run | What it needs | What it proves |
| --- | --- | --- |
| `--dry-run` | No model and no GPU | Every task's checks and rubric are right: a correct script passes with quality 3, and a near miss fails |
| `--live` | A served model | How the agent and the model do on each task: completion, quality, time, tokens and, over several runs, the same token IDs |

Both runs start the real `vade-harness agent` for each task, in a new copy of
the task's workspace. The tools run for real: files change, and on macOS
commands run in the command sandbox. Only the model differs. In a dry run, a
loopback endpoint streams the replies of a script. In a live run, a
recording proxy passes each call to the model server.

## Run the dry run

From the repository root, after a build of the harness:

```sh
cd harness
cargo +1.99.0 build --offline
./target/debug/vade-harness suite --dry-run
```

Each task prints one line, and the last line gives the result:

```text
agent-suite-2 dry run | recover-stale-path | tool-error-recovery | solve PASS 7/7 quality 3/3 (2 changed lines) | wrong FAIL 4/7 quality 0/3 (missing src/v1 -> present) | OK
DRY RUN PASS: 22 of 22 tasks pass with solve.script and fail with wrong.script, 0 skipped (agent-suite-2)
```

A task passes the dry-run gate when all of these are true:

- Its `solve.script` passes every check and scores 3 of 3 on its rubric.
- The harness asked for every reply in `solve.script`, and no step timed out.
- Its `wrong.script` fails at least one check. An error in the run is not a
  fail.

The exit status is 0 when every task passes the gate, 1 when one does not,
and 2 on a usage or setup error. Without `--out`, the runner uses a new
temporary folder and removes it at the end, also after a failure. `--out DIR`
keeps the folder and prints its path; `--only ID,ID` runs only some tasks. On the lab Mac on 2026-10-30, the full dry run
of 22 tasks took 35.58 s of wall time (`/usr/bin/time -p`, `real 35.58`).

`cargo test` runs the same dry run in
[`harness/tests/suite_dry_run.rs`](../harness/tests/suite_dry_run.rs). The
command sandbox is on macOS only. On another host, a task whose scripts call
`workspace_shell` shows `SKIP`; it is not counted as a pass.

## Run the suite on a served model

Run the runner on a Mac. The command sandbox, and with it `workspace_shell`,
exists only on macOS. On another host the agent cannot run commands, so the
shell tasks fail and the other tasks cannot run their tests: the card then
measures the host, not the model.

Start the model, build the runner, then give it the server address and model
name. On one Spark, from the repository root:

```sh
vade up --sparks 1
cd harness
cargo +1.99.0 build --offline --release
./target/release/vade-harness suite --live \
  --endpoint http://127.0.0.1:18080 \
  --model qwen3.8-flash-next-fp4-local \
  --runs 3 --out ~/suite-cards/one-spark
```

Use the address that `vade status` prints and the model name that
`GET /v1/models` gives. For a pair of Sparks, start the pair and use the same
command with the pair's address and a new `--out` folder. The runner has no
other setting for the setup, so two cards compare row by row. Each card
records the address and the model.

Every run uses the same workspace path for a task,
`$TMPDIR/vade-harness-suite-live`, so that the prompts can be the same from
run to run. Only one live run can use this path at a time: when it exists,
the runner stops with exit status 2 and makes no `--out` folder. The runner
removes it at the end of a run. When a run is stopped by a signal, remove it
by hand. The long-context task needs a server context of 131072 tokens.

The runner writes these files:

| File | What it holds |
| --- | --- |
| `OUT/card.md`, `OUT/card.json` | The summary over all runs: the card row, one row per category, one line per failed task run, and one row per task with its quality and token-ID comparison |
| `OUT/run-N/card.md`, `OUT/run-N/card.json` | One run: the same category rows and failure lines, and one row per task with checks, quality, model calls, tool turns, tokens, the token digest and the wall time |
| `OUT/run-N/tasks/ID/` | The workspace before and after, the harness output of each step, the check log and `transcript.jsonl` (every model call and reply, raw) |

## What the card shows

Each task run gets two grades:

- **Completed:** every `check:` and `step_check:` line of `task.txt` passes.
  The checks are exact. No model judges an answer.
- **Quality, from 0 to 3:** from the rubric in `rubric.txt` next to
  `task.txt`. The score is 0 when the task is not completed. Otherwise it is
  3 minus one for each rule that the run misses:
  - **Minimal:** the added and removed lines over all changed files are at
    most `max_changed_lines`.
  - **No collateral edits:** every file that changed, or that the agent
    made, is under a `may_change` path. Hidden files count. Files in
    `target` and `.git` folders, and a `Cargo.lock` that cargo made, are
    not counted.
  - **Clear final answer:** the last reply holds each `answer_mentions` text
    and has at most `answer_max_chars` characters.

The category table has, for each category and for all tasks: the number of
tasks, the task runs completed, the quality mean, the median wall time and
the prompt and completion tokens from the server's `usage` objects.

Each failed task run gets one line with its first wrong tool call. That is
the first call that the harness refused or that failed (the result says
`refused`, `denied`, `invalid arguments`, `unknown tool`, `unsupported` or
starts with `error`), or the first call that writes a path outside the
rubric's `may_change`. When no call is wrong by these rules, the line names
the last tool call.

Over several runs, the summary compares the streamed token IDs of each task.
To keep the requests the same from run to run, the runner gives each agent
run its own empty home folder (so your `~/.vade/instructions.md` is not in
the prompt) and a fixed reply cap of 4096 tokens (`--reply-tokens`).

The comparison goes call by call against the first run. While a call has the
same request as in the first run, it must stream the same token IDs. The
request is compared without the fields that change on every run: the
`prompt_cache_key` and `cancel` tokens and the tool call IDs that the server
makes. Some tool output also changes on every run. For example, a
`workspace_shell` result gives `elapsed_ms`, and cargo prints its build time.
When a request differs, the conversation is no longer the same, and the
comparison for that run stops there. A task is exact when all of these
are true:

- Every run streamed token IDs.
- No model call gave other token IDs for the same request.
- In each run, the first request is the same as in the first run, so at
  least one call was compared.
- When no request differs, every run made the same number of model calls.

The exactness column of the task row gives one of these texts:

| Text | Exact | Meaning |
|---|---|---|
| `yes` | yes | Every call had the same request and the same token IDs. |
| `yes up to call N (its request differs: tool output)` | yes | Call N is the first call whose request differs. The calls before it are the same. |
| `no (model call N differs with the same request)` | no | Call N got the same request but gave other token IDs. |
| `no (one run made more calls; the first N are the same)` | no | The first N calls are the same, but one run made more calls than the first run, or fewer. |
| `not compared (the first request differs)` | no | The first request of a run differs, so no call was compared. |
| `no token IDs` | no | A run streamed no token IDs. |
| `-` | no | The suite ran only once, so nothing was compared. |

After each step, the runner waits up to 5 s for the model calls that are
still running. When the wait ends first, the task note on the run card says
"a model call was still running after 5 s, so the call list may lack a call".
A row that says `one run made more calls` can then come from the missing call.

## What grade A means

The summary gives one card row, for example:

```text
| agent-suite-2 | qwen3.8-flash-next-fp4-local | 22 tasks | 3 runs | 66/66 task runs pass | quality 3.00 | 22/22 tasks exact | grade A |
```

This example shows the form only; it is not a measured result.

| Grade | Rule |
| --- | --- |
| A | At least 3 runs. Every task run is completed and has quality 2 or more. The quality mean is 2.5 or more. Every task is exact. No step timed out. |
| B | At least 3 runs. At least 90 % of the task runs are completed. The quality mean is 2.0 or more. Every task is exact. |
| F | Any other result with at least 3 runs. |

With fewer than 3 runs, the card gives no grade. A grade is valid only for a
suite whose dry run passes at the same revision. The card JSON records the
Git revision of the fixtures.

## Add a task

1. Make the folder `harness/tests/agent-suite/tasks/<id>/` with `task.txt`,
   `rubric.txt`, `workspace/`, `solve.script` and `wrong.script`. Put exact
   expected files in `expected/` and tests that the agent never sees in
   `hidden/`. The [suite README](../harness/tests/agent-suite/README.md)
   lists every key of `task.txt`, every check and the script lines.
2. Give the task a `category:` line. Use a category from the table in the
   suite README when one fits.
3. Write `solve.script` as a correct agent would work: read, change, check,
   answer. Write `wrong.script` as a near miss that a real agent could make,
   such as an edit to a test file, a fix in the wrong function or one wrong
   answer line.
4. Write `rubric.txt`. Set `may_change` to the files the task needs. Set
   `max_changed_lines` to about twice what the solve changes: the dry-run
   line prints that number.
5. Add the files to Git: the runner refuses untracked files. Then run
   `vade-harness suite --dry-run --only <id>` until the line ends in `OK`.
6. Add a row to the task table in the suite README, and raise the task count
   in [`agent_suite_fixtures.rs`](../harness/tests/agent_suite_fixtures.rs).

Keep every fixture under 5 MB and avoid the file types that the release scan
blocks (`.ptx`, `.safetensors`, `.log`, `.pem`, `.key`). Fixture crates use
the Rust standard library only and build with `cargo test --offline`.

**Back:** [Documentation index](README.md) · **Harness:** [README](../harness/README.md)
