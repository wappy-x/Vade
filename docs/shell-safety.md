# Shell safety

The Vade agent runs commands with the `workspace_shell` tool. This page
tells you what Vade does with each command, what the prompt shows, and how
to let a set of commands run without a prompt. It is written for the
person who runs Vade and for the agent that asks to run commands.

## Summary

| You want | Do this |
| --- | --- |
| Approve one command | Answer `y` at the prompt. |
| Approve the same exact command for the rest of the session | Answer `s`. |
| Approve the same exact command in this project from now on | Answer `p`. It goes into `.vade/permissions.json`. |
| Allow a family of commands, such as `ls -la` on any workspace path | Write a rule with a `<path>` slot in a permissions file (see [Allow lists](#allow-lists)). |
| Turn all prompts off | Start with `--yes`. Blocked commands stay blocked. |
| Deny all commands | Start with `--approval never`. |

## What happens to a command

Vade takes these steps for each command, in this order:

1. **Blocked.** Vade's command checker reads the command and refuses some
   forms outright, such as `rm -rf /`, `rm -rf ~`, a write to a disk
   device (`dd of=/dev/disk2`, `> /dev/sda`) and a fork bomb. A blocked
   command never runs, also not with `--yes`. This check can only refuse
   more. It does not find every dangerous form (see [Limits](#limits)).
2. **Allow list.** If a rule in an allow list matches the command's argv,
   the command runs without a prompt. The allow lists are empty by default.
3. **Prompt.** Every other command waits for your answer. With `--yes`,
   it runs. With no terminal, it does not run (see
   [Runs without a terminal](#runs-without-a-terminal)).
4. **Sandbox.** Every command that runs, runs inside the macOS sandbox, in
   the workspace folder. The sandbox lets it write only in the workspace
   and in the session build folder. It denies the network and credential
   files.

Vade does not decide that a command is harmless. By default, also `ls` and
`git status` ask. An approved command can still run code from the
workspace: `cargo test` runs build scripts and tests, and a rule for
`./build.sh` runs the current text of that script.

## The prompt

The prompt shows exactly what will run:

| Line | What it shows |
| --- | --- |
| `command argv=[...]` | The argv as the agent sent it, as JSON, with the timeout in milliseconds. Nonprinting characters are escaped, so the JSON decodes to the exact words. |
| `Runs as: [...]` | Only when Vade runs a fixed system form instead. For example, `git status` runs as `/usr/bin/git --no-pager ... status ...` with hooks, pagers and hidden or credential files left out. |
| `Folder:` | The working folder: the workspace root. |
| `Environment (only these ...)` | Every variable that the command gets. Vade clears your shell's environment and sets only this list. `<build>` is the session build folder. |
| `Scan exclusions` | Folders that the command can neither read nor write, at every depth. |
| `Build environment` | The build folder and the Rust toolchain that the command uses. |
| `Safety:` | Why the command needs approval, and the answers you can give. |

The answers:

| Key | Command | Effect |
| --- | --- | --- |
| `y` | `/approve` | Run this command once. |
| `s` | `/approve session` | Run it, and allow the same exact argv without a prompt for the rest of this session. |
| `p` | `/approve project` | Run it, and save the same exact argv to `.vade/permissions.json`. |
| `n` | `/deny` | Do not run it. The agent gets a denial. |

`s` and `p` save only the exact argv. After `s` for `cargo test --offline`,
the command `cargo test --offline` runs without a prompt, but
`cargo test`, `cargo test --offline -p x` and
`/bin/sh -c "cargo test --offline"` still ask. An Allow list notice after
the answer says which rule was added, or why none was. Session rules end
with the session: `/new`, resume, switch and fork start without them.

`s` and `p` offer a rule only for a command whose words are all plain
(see [Rule words](#rule-words)) and that may go on a list (see
[What never goes on a list](#what-never-goes-on-a-list)). For other
commands, they approve once.

## Allow lists

Vade reads two files when a session starts, and again at `/new`, resume,
switch and fork:

- `.vade/permissions.json` in the workspace: the project list.
- `permissions.json` in the Vade home (`VADE_HOME`, or `~/.vade`): the user
  list, for every project.

A rule in either file, or a session rule, lets a command run without a
prompt. A project list is part of the project, so read
`.vade/permissions.json` before you work in a project that you did not
write.

### Example

This allow list is a good start for a Rust project. Vade's tests load this
exact block from this page and check what it matches.

```json
{
  "version": 1,
  "allow": [
    "cargo test",
    "cargo test --offline",
    "cargo check --offline --all-targets",
    "git status",
    "git status --short",
    "git diff",
    "git diff --stat",
    "git log --oneline -n 20",
    "ls -la",
    "ls -la <path>",
    "wc -l <path>"
  ]
}
```

With this list, these commands run without a prompt: `cargo test`,
`git status`, `ls -la`, `ls -la src`, `ls -la harness/src/shell` and
`wc -l src/main.rs`. These still ask: `cargo test --offline -p x` (more
words), `git status -uall` (other words), `ls -la /` and `ls -la ..`
(outside the workspace), `ls -la .env` (a hidden file), `ls -la -R` (an
option, not a path) and `wc -l a.rs b.rs` (two paths for one slot).

### Rule words

A rule is the words of one command, separated by spaces, program first.
The array form (`["cargo", "test"]`) means the same. A rule matches a
command only when the command has the same number of words and each word
matches:

- **A plain word** matches only the same word. A plain word has 1 to 4096
  ASCII letters, digits and `- _ . , / : = + @`. A word with `$`, a
  backquote, `(`, `)`, `{`, `}`, `[`, `]`, `*`, `?`, `~`, `<`, `>`, a
  quote, a space or any other character is not plain, so a rule cannot
  hold it, and no rule matches a command that has it. A shell script
  (`/bin/sh -c "cargo test"`) has a space in its script word, so it always
  asks.
- **`<path>`** matches one word that names a path inside the workspace. The
  word must be plain, must not start with `-`, must be relative (no
  leading `/`), and must not hold `..` or a part that is hidden (starts
  with `.`), a credential file name or non-ASCII. `.` is the workspace
  itself. If a part of the path already exists and is a link, the word
  does not match, because a link can point outside the workspace. A path
  that does not exist yet matches. Vade checks the path when the command
  asks to run. `<path>` cannot be the program word. Use one `<path>` for
  each path word: `wc -l <path> <path>` matches two paths.

There are no other slots and no prefix rules: `git push *` does not load.
Use the program name as the agent writes it: `ls` and `/bin/ls` are
different rules.

A `<path>` slot lets the command act on any workspace path that it
matches, also `.`. Think before you write `rm -rf <path>`: it can delete
the whole workspace.

### What never goes on a list

Some commands never run from a rule, even with plain words. A rule for
them does not load, `s` and `p` approve them once only, and they always
ask:

- A word that names a hidden file, a credential file or a
  `permissions.json` file in any part. Vade splits the word at
  `/ = : @ ,` and also looks inside a value attached to an option, so
  `cat .env`, `tar -tfid_rsa`, `git show HEAD:.env` and
  `curl -d @id_rsa URL` all ask. This also means that `git diff .github`
  always asks.
- A command that the checker finds runs as root (`sudo`, `su`), sends text
  into a shell (`curl ... | sh`) or runs text as code (`eval`).
- A blocked command.

### File rules

A file that Vade cannot read stops startup with the file path and the
problem. At a later session start, Vade uses an empty list for that file,
so its commands ask, and a notice names the file. Fix the file or remove
it. Vade refuses a `.vade` folder or a permissions file that is a link, and
a permissions file that is not a regular file. A file can hold at most 512
rules and 64 KiB, and a rule at most 128 words.

When you answer `p`, Vade reads the project file again, adds the rule and
saves the file. From then on, Vade uses the file as it read it, also when
the save fails, so a rule that you removed by hand no longer applies. Other
edits by hand take effect at the next session start or the next `p`
answer. If the new file would not load or would pass a limit, Vade keeps
the old file. Two sessions that save the same file at the same time take
turns: Vade locks the file (`.permissions.json.lock` beside it) for the
read and the save. If the lock stays busy for two seconds, Vade does not
save the rule, and the command is approved once.

## Turning prompts off

`--yes`, or its long name `--dangerously-skip-prompts`, turns all prompts
off. It is the same as `--approval auto`: edits and commands run without
asking, inside the sandbox. Blocked commands stay blocked.

```sh
vade-harness agent --endpoint URL --model NAME --workspace . --yes
```

`--approval never` denies all edits and commands.

## Runs without a terminal

When no person can answer, for example with `--prompt` or when standard
input is not a terminal, Vade does not wait. Commands that a rule allows
run. Every other command does not run, and the agent gets this tool error:

```json
{"ok":false,"error":"approval_required","message":"Command not run: it needs approval (every command needs approval unless an allow list names it), and no one can approve it in this non-interactive run. ..."}
```

To let such a run use a command, add a rule for it, or pass `--yes`.

## For agents

- Send the argv that you mean, without a shell: `["cargo", "test"]`, not
  `["/bin/sh", "-c", "cargo test"]`. A rule can match the plain argv. A
  shell script with a space always asks.
- Use plain words. A word with a variable, a pattern, a quote or `~`
  always asks.
- After a denial or an `approval_required` error, do not send the same
  command again. Finish without it, or tell the person which rule would
  allow it, for example `"cargo test --offline"` or `"ls -la <path>"` in
  `.vade/permissions.json`.
- Do not change `.vade/permissions.json` yourself. The person writes the
  allow list.

## The journal

Each decision about a command that needs approval, and each blocked
command, is a `shell_approval` record in the session journal. A record
holds the tool call ID, the argv, the class, the reasons, the decision and
the rules that applied:

| Decision | Meaning |
| --- | --- |
| `approved` | A person said yes. `rules` shows `once` or the rule that the person chose to add, for example `session: git push origin main`. |
| `denied` | A person said no. |
| `allowed_by_rule` | An allow-list rule matched, for example `project: ls -la <path>`. |
| `allowed_by_flag` | Prompts were off (`--yes` or `--approval auto`). |
| `denied_no_terminal` | No one could answer. |
| `denied_by_policy` | `--approval never` is set. |
| `blocked` | The command is blocked. |

Vade writes the record before it saves a rule. If Vade cannot write the
record for an approval, the command does not run and no rule is saved. Use
`/journal` to inspect raw records.

## The experimental checker

The checker can also put a command in a safe class: read-only, build and
test commands that it reads fully. That safe class is off by default. Set
`VADE_SHELL_CHECKER=1` to let commands in it run without a prompt. Reviews
still find inputs that it calls safe and that are not, so turn it on only
when you accept that risk. [The shell checker](../harness/docs/shell-checker.md)
describes how it reads a command and its known gaps.

## Limits

- The default policy asks before every command that no rule names, so it
  does not depend on the checker reading a command correctly. The sandbox
  is the security boundary.
- The block check knows a fixed set of programs and forms. Some dangerous
  forms are not blocked and only ask, for example a disk write through
  `tee /dev/sd?`, a program name from a pattern (`/bin/ba* -c ...`) and
  `su root -c ...`. With `--yes`, such a command runs inside the sandbox.
  Vade's tests keep every input from the reviews of this checker as a
  case that never runs without a person.
- A rule matches words, not what the program does with them. A rule runs
  the program and the files as they are when the command runs.

The source is in
[`harness/src/shell/policy.rs`](../harness/src/shell/policy.rs) (the
default policy and rule matching),
[`harness/src/shell/permissions.rs`](../harness/src/shell/permissions.rs)
(the allow lists) and
[`harness/src/shell/safety.rs`](../harness/src/shell/safety.rs) (the
checker).
