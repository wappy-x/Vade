# How `vade` talks to you

This page is the contract for everything `vade` prints. New commands follow
it, and the tests in [`cli/tests/`](../tests) hold the existing ones to it.

## Chat

`vade chat` is where you spend your time, so it gets the most care. It is
minimal on purpose: two voices, one quiet line of numbers, nothing else.

```
▌ Why is the sky blue?

▸ thinking 12:02:19–12:02:21 · ≈42 tokens · 1.5s · Ctrl+T expand

Sunlight is a mix of colours. Air molecules scatter short wavelengths far more than long ones
(about 1/λ⁴), so blue light is redirected across the whole sky and reaches your eye from every
direction.
╰ 12:02:21 · 1.5s · 312 tokens out · 222 tok/s · prefill ≈153k tok/s · first token 0.0s
qwen3.8-flash-next-fp4-local · ctx 1.1k/32.8k · in 790 · out 312 · 222 tok/s · 1 Spark, drafting on
›
```

(One exchange against an offline mock server; the [transcript](transcript.md)
shows the whole screen.)

1. **Who is who.** Everything you wrote sits behind a cyan `▌` bar, in bold.
   The model's answer has no bar and no header. The bar reads the same without
   color, so the two voices never blur on a light theme or with `NO_COLOR`.
2. **Thinking is not the answer.** A thought is one dim, italic line: when it
   started and ended, how many tokens, how long. Ctrl+T opens it; its text
   then sits behind a dim `┆` bar, never in the answer's column.
3. **One line of numbers per answer.** Under each answer, `╰` ends the turn
   with its clock, elapsed time, tokens out, decode speed, prefill speed, time
   to first token, context used of the window and cached prompt tokens. Raw
   counts, no percentages.
4. **One status strip.** Above the prompt, one dim line holds the model,
   context used of the window, tokens in and out, the last decode speed, the
   Spark count and whether drafting is on. While the model works, the strip
   shows what it is doing, the elapsed time and the live tokens per second.
5. **No boxes and no banners.** The prompt is a `›`. The welcome is one dim
   line. There are no rules between turns.
6. **Words stay whole.** Answers and your turns wrap between words, and your
   bar continues on every wrapped row. Approval previews (diffs) wrap at exact
   columns instead, so a wrapped fragment never looks like a diff line.
7. **Fast.** The strip and the live answer redraw only the rows that changed,
   at most 30 times a second; the transcript above is never repainted per
   token.
8. **Plain when it should be.** When standard input or output is not a
   terminal or `TERM=dumb` (or the chat program runs with `--plain`), chat
   prints plain lines: your turns keep their `▌ ` prefix, answers stream as
   text, and the numbers print as one line under each answer.
9. **Narrow terminals keep what matters.** The strip drops facts from the
   middle first; the model name and the Spark and drafting setting stay.

`vade chat` tells the chat program the Spark count and drafting setting
through `VADE_CHAT_SETUP` (for example `2 Sparks, drafting on`); the chat
program cannot ask the server for them.

## Commands

| Group | Command | What it does |
| --- | --- | --- |
| Start and stop | `vade launch` | Install if missing, start on this Spark, open chat |
| | `vade up` | Start without chat (this Spark, or Sparks from a controller) |
| | `vade down` | Stop what vade started |
| Use | `vade chat` | Chat with the running engine |
| Look | `vade status` | What is running, where, and whether it is healthy |
| | `vade config` | Which setup is selected and what it holds |
| | `vade doctor` | One line per check, then PASS or FAIL |
| | `vade detect` | This Spark's machine, memory, disk, driver and model |
| | `vade logs` | Recent engine logs |
| Prove | `vade verify` | Same token IDs every run, drafting ON and OFF |
| | `vade bench` | Speed measured from the client |
| | `vade test` | Build, determinism and long-context gate checks |
| Maintain | `vade install`, `vade update` | Put binaries in place, or refresh them |
| | `vade run qwen3-8b` | The separate legacy Qwen3-8B workflow |
| | `vade --version` | The CLI version |

Bare `vade` prints the same short status as `vade status` plus the next
command to run.

## Output

1. **One fact per line.** A line says one thing: a value, a check result or a
   step. Explanations go in `--help`, not in normal output.
2. **Stable columns.** Labeled values line up in one column (`Model     ...`).
   Check rows are `state  name  detail` with fixed widths, so you can read them
   by eye and cut them with `awk`. Wording and column order change only with a
   snapshot update in the same commit.
3. **Results on stdout, problems on stderr.** Errors always go to stderr. With
   `--json`, stdout holds only the JSON object. `vade verify` prints exactly one
   result line on stdout and its progress on stderr.
4. **Color only on a terminal.** Color is on when the stream is a terminal
   and `TERM` is not `dumb`. `NO_COLOR` turns it off and `CLICOLOR_FORCE=1`
   turns it on for pipes; `NO_COLOR` wins. Color never carries meaning on its
   own: every colored word reads the same in plain text. JSON, receipts and
   `key=value` records are never colored.
5. **`--json` on every read-only command.** `status`, `doctor`, `config`,
   `detect` and `version` take `--json` and print one object on one line, with
   a `schema` field (`vade.status.v1`, `vade.doctor.v1`, `vade.config.v1`,
   `vade.detect.v1`, `vade.version.v1`). Fields are only ever added, never
   renamed. The exit code is the same as without `--json`, and so are the
   checks. A usage error (an unknown option) still goes to stderr; any other
   failure is in the object. A doctor that stops early is a report with one
   failed row. `config`, `detect` and `version` failures carry `error` and
   `next`. `status` keeps its own shape: a top-level `errors` list and an
   `error` on each failed check, with no `next`. Dry runs answer in the same
   schema. `logs` prints the
   engine's own text, `verify` writes its JSON receipt with `--receipt` and
   `bench` writes its JSON receipt with `--output`.
6. **Errors say what failed, then the exact next command.**

   ```
   error: Port 18080 is in use.
    hint: Close the application using it and retry vade up.
   ```

   The first line is what failed, with the path, port or machine it failed on.
   The `hint:` line starts with a verb and names a command you can paste.
7. **Progress for anything over 2 seconds.** A step prints its name
   right-aligned in an 8-column field, then what it is doing. While it runs it
   prints `still ... (N s elapsed)` at most every 15 seconds, and it ends with
   `done in N s` (milliseconds below one second, tenths of a second above).
8. **Raw numbers.** Bytes, tokens, tokens per second, milliseconds and seconds,
   with their units. No percentages, ratings or rounding to "about".
9. **No emoji and no banners.** The status marks `●` (ready), `○` (stopped)
   and `▲` (needs attention) always sit next to the word that says the same
   thing.
10. **Dry runs change nothing.** `--dry-run` says so on its first line and then
    shows the plan: commands, paths and limits, one per line.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. `status` exits 0 for both ready and stopped, `verify` for IDENTICAL, `doctor` for PASS |
| 1 | Anything failed: a bad argument, a failed step, a doctor FAIL, a `verify` DIFFERENT or ERROR, an unhealthy `status --json` |
| 2 | `vade test` was given inputs it cannot use (the gate commands keep their own codes) |
| 130 | `vade bench` was interrupted with Ctrl-C |

When the reader of stdout goes away (for example `vade status | head -1`), a
command that would have succeeded still exits 0.

## Help text

Every command has a page, and `vade help <command>` prints the same page as
`vade <command> --help`. A page has, in this order:

1. **One line** that says what the command does, starting with a verb.
2. **Examples:** one to three commands you can paste, the common case first.
3. **Usage:** the syntax.
4. **Options:** one row per flag, the flag in one column and its range and
   default in the other.
5. **Notes:** only what you need to know before running it, one item each.

Pages wrap at 80 columns and never depend on the terminal width. The tests keep
a snapshot of every page in [`cli/tests/snapshots/`](../tests/snapshots).
