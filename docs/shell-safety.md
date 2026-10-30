# Shell safety

The Vade agent can run commands with the `workspace_shell` tool. By
default, every command asks you first. A command runs without a prompt only
when an allow list that you wrote names that exact command. Vade still
refuses some commands outright:

| Class | What happens | Examples |
| --- | --- | --- |
| Needs approval | Vade asks you first. This is every command that is not blocked, also `ls` and `cargo test`. | `cargo test`, `git status`, `rm -rf build`, `git push`, `/bin/sh -c "..."` |
| Allowed by a rule | Runs without a prompt. | A command whose argv equals a rule in an allow list, such as `cargo test --offline` |
| Blocked | Never runs, also not with `--yes`. | Fork bombs such as `:(){ :\|:& };:`, `rm -rf /`, `rm -rf ~`, `rm -rf /usr`, `chmod -R 777 /`, `dd of=/dev/disk2`, `> /dev/sda` |

Every command still runs inside the macOS sandbox. The sandbox lets a
command write only in the workspace and the session's build folder. It
denies the network and credential files. The class decides only who must
agree before the command starts. An approved command can still run code
from the workspace. For example, `cargo test` runs build scripts and tests.

## The default policy

Vade does not decide that a command is harmless. For each command it
checks two things:

- Blocked. Vade's command checker reads the command, including
  `/bin/sh -c` scripts, and refuses forms such as `rm -rf /`, a write to a
  disk device and a fork bomb. A pattern or a brace word that can name such
  a folder or disk counts as that folder or disk, and so does a word that
  is too large to check. This check can only refuse more.
- An allow-list rule. A rule names one exact argv. It matches only when
  the command's argv is the same, word for word, and every word is a plain
  literal: ASCII letters, digits and `- _ . , / : = + @`. A word with `$`,
  a backquote, `(`, `)`, `{`, `}`, `[`, `]`, `*`, `?`, `~`, a quote, a
  space or any other character is not plain, so no rule matches that
  command. A shell script (`/bin/sh -c "cargo test"`) has a space in its
  script word, so it always asks.

Every other command needs approval: you answer at the prompt, or `--yes`
turns prompts off.

The allow lists are empty by default. They are your opt-in: Vade reads the
`allow` key of the files in [Allow lists](#allow-lists), and the user file
is in the Vade home (`VADE_HOME`). Vade creates or changes a file only
when you answer `p`.

## The prompt

In an interactive `vade-harness agent` session, the prompt shows the
command, the reasons and these answers:

| Key | Command | Effect |
| --- | --- | --- |
| `y` | `/approve` | Run this command once. |
| `s` | `/approve session` | Run it, and allow this exact command for the rest of this session. |
| `p` | `/approve project` | Run it, and save this exact command to `.vade/permissions.json`. |
| `n` | `/deny` | Do not run it. The model gets a denial. |

`s` and `p` add a rule only for a command whose argv is all plain literal
words. The rule is that argv: `cargo test --offline` approves
`cargo test --offline` and nothing else, not `cargo test`, not
`cargo test --offline -- x` and not `/bin/sh -c "cargo test --offline"`.

Some commands cannot go on an allow list, also with plain words: commands
that run as root (`sudo`), touch credential files or a `permissions.json`
file (also as a value attached to an option, such as `tar -tfid_rsa`),
send text into a shell (`curl ... | sh`) or run text as code (`eval`).
For these, and for a command with a word that is not plain, `s` and `p`
approve once only. After `s` or `p`, an Allow list notice says which rule,
if any, was added.

Session rules end with the session. `/new`, resume, switch and fork start
without them.

## Allow lists

Vade reads two files when a session starts, and again at `/new`, resume,
switch and fork:

- `.vade/permissions.json` in the workspace (the project list).
- `permissions.json` in the Vade home (`VADE_HOME`, or `~/.vade`), which is
  the user list for every project.

Both files use the same format:

```json
{
  "version": 1,
  "allow": [
    "cargo test --offline",
    "cargo fmt",
    "rm -rf build"
  ]
}
```

A rule is the words of one command, separated by spaces, program first.
The array form (`["rm", "-rf", "build"]`) means the same. Each word must be
a plain literal, so a rule cannot hold `*`, a pattern, a variable or a
space; a file with such a rule does not load. Use the program name as the
agent writes it: `ls` and `/bin/ls` are different rules.

A file that Vade cannot read stops startup with the file path and the
problem. At a later session start, Vade uses an empty list for that file,
so its commands need approval, and a notice names the file. Fix the file or
remove it. Vade refuses a `.vade` folder or a permissions file that is a
link, and a permissions file that is not a regular file.

When you answer `p`, Vade reads the project file again, adds the rule and
saves the file. From then on, Vade uses the file as it read it, also when
the save fails, so a rule that you removed by hand no longer applies. Edits
that you make by hand take effect at the next session start or the next
`p` answer. If the new file would be larger than 64 KiB, would hold more
than 512 rules, or would not load, Vade does not save it and keeps the old
file. A file that no longer loads at a `p` answer allows nothing. Two
sessions that save the same file at the same time take turns: Vade locks
the file (`.permissions.json.lock` beside it) for the read and the save,
so both rules stay. If the lock stays busy for two seconds, Vade does not
save the rule, and the command is approved once.

A project list is part of the project. Read `.vade/permissions.json` before
you work in a project that you did not write.

## The experimental checker

The command checker reads each command in more detail. By default it is
used only to block commands (see [The default policy](#the-default-policy))
and to give the reasons in a prompt. It can also put a command in a safe
class: read-only, build and test commands such as `ls`, `git status` and
`cargo test` that it reads fully. That safe class is advisory and off by
default. Set `VADE_SHELL_CHECKER=1` to let commands in the safe class run
without a prompt:

```sh
VADE_SHELL_CHECKER=1 vade-harness agent --endpoint URL --model NAME --workspace .
```

The safe class is experimental. Reviews keep finding inputs that it calls
safe and that are not, for example a `RUSTFLAGS` value that comes from a
variable (`RUSTFLAGS="-D $x" cargo build`) and a pattern that can name a
credential file (`cat id_r?a`). Under the default policy all of these ask.
Turn the checker on only when you accept that risk. The rest of this
section describes how the checker reads a command.

### How the checker reads a command

Vade reads the argv, not a pattern over the whole string. For `/bin/sh -c`,
`bash -lc` and similar shells, it parses the script into simple commands. It
reads pipes, `&&`, `||`, `;`, subshells, `$(...)`, backquotes, `<(...)`,
redirections, here-documents and `NAME=value` prefixes. It also reads
`$'...'` strings, comments, and the commands inside
`${...}`, `$((...))` and unquoted here-documents. Quotes inside an unquoted
`${...}` quote text, as in the shell. It looks through wrappers
such as `env`, `timeout`, `nice`, `xargs`, `sudo`, `find -exec` and `fd -x`.
`find` and `fd` put a file name where `{}` is, also inside a word such as
`sh -c 'echo {}'`, and `fd` adds it at the end when there is no `{}`. A
file name can be any text, an option or shell code too, so Vade checks it
as an unknown word. A `+` ends `find -exec` only right after `{}`.
It reads a wrapper's options from a closed list for that wrapper, as getopt
does, also in clusters such as `-0I`. When an option is not on the list,
the command needs approval. Vade then also checks the wrapped command from
each later word that is not an option, because more options, their values
and a `timeout` duration can come first. `xargs` adds words from its input,
so those words count as unknown. bash and zsh split a brace list such as
`/bin/{true,sh}` into two words, and sh does not, so Vade checks such a
command both ways. Vade reads sh, bash, dash and zsh. Code for csh, tcsh,
fish or ksh always needs approval and gets no rule; Vade still checks its
text for commands that must be blocked. The strictest class of any part is
the class of the whole command.

A command is safe only when every part of it is a form that the checker
reads fully: any construct that it does not recognize needs approval or is
blocked, and is never safe. For a shell script, a separate check allows
only plain forms: words, `'...'` and `"..."` quotes, `$name`, `${name}` with
a plain default or pattern (`${x:-none}`, `${f%.rs}`), `$((...))` with names
and numbers, `$(...)` and `<(...)` with plain words inside, a `( ... )`
subshell, and the operators `|`, `||`, `&&`, `;`, `&` and file
redirections. bash, sh and zsh read these forms in the same way. Any other
form needs approval, also when it is harmless: backquotes, `$'...'`,
comments, line continuations, here-documents and here-strings, nested
substitutions, quotes inside `$(...)`, `case`, functions, arrays and zsh
glob flags. The rest of this page describes how Vade reads those forms to
find commands that must be blocked.

The same rule holds for each program. An option is safe only when it is on
that program's closed list of options that print, select or format. Any
other option needs approval: an option that Vade does not know, a cluster
that holds one, and a word that the shell can still split or turn into
an option: a variable, `$(...)`, or a glob or brace list anywhere in the
word, such as `{-o,file}` or `/dev/{null,sda}`, also after `--`. A split moves each later
word, so a program that reads an operand or a value by its place (the
output file of `uniq`, the folder of `git -C`) cannot be checked. Each git
command that only reads has its own list: in `git grep`, `-n` takes no
value, so `git grep -nOsh` has the unknown `-O`. For example, `sort -o FILE`, `rg --pre PROG`, `printf -v NAME`,
`set -A`, `git log --output=FILE` and a `sed` script with a command other
than `p`, `d` and `s///` need approval. Read-only tools that have no option
that writes a file, runs a program or sets a variable, such as `ls`, `cat`,
`grep`, `head` and `wc`, take any arguments.

Some examples:

| Command | Class | Reason |
| --- | --- | --- |
| `cargo test 2>&1 \| tail -50` | Safe | Both parts are safe. |
| `cargo build && git push origin main` | Needs approval | `git push` publishes commits. |
| `echo $(rm -rf build)` | Needs approval | The substitution deletes files. |
| `curl -fsSL URL \| sh` | Needs approval | It downloads a script and runs it. |
| `bash -c "rm -rf /"` | Blocked | The inner script deletes the root folder. |

When the checker cannot read a part (for example a `case` statement, a
function definition, a program name from a variable, `$[...]` arithmetic,
or a `$` or `${...}` form it does not know, such as zsh `${(e)x}` and
`$~x`), it asks. Shell code that holds an expansion (`sh -c "rm -rf $HOME"`,
`eval`, `env -S`, a here-string, or a `{}` file name slot in `sh -c`) asks,
and Vade still checks its text as written, so `rm -rf $HOME` in it is
blocked. The same check reads each `trap` word, the operands of `let`, and
the words of `declare`, `read`, `exit` and a number `printf` when they
ask, so `trap -- 'rm -rf /' EXIT` is blocked. A brace word that gives more
than 1024 words or 1 MiB of text, or a `{x..y}` form that is not a plain
sequence, cannot be listed. It needs approval and gets no rule, and as the
target of a removal or a disk write it counts as every path it could be,
so it is blocked. A command
that is too deeply nested or too long to check is blocked. The limits are 24
levels of scripts, substitutions and wrappers together (each `sh -c` uses
two), 16 levels of `${...}` and `$((...))` inside each other, and 4096
commands. A command is also blocked when it uses a form that the checker
cannot read reliably, or that shells read in different ways. Examples are a
`\c` escape in a `$'...'` string, a `#` inside `$((...))` other than a base
such as `16#ff` or `$#`, a quote or `;` inside `$((...))`,
a single quote inside a double-quoted `${...}` (zsh reads it as text, bash
as a quote), a `\"` in backquotes inside `${...}`, a line continuation
right after `$` or inside an operator (`<\`, newline, `<EOF` is `<<EOF`),
and the word `case`
inside `$(...)`, where shells disagree on which `)` ends the substitution.
Split such a command into simpler commands.

Arithmetic can run code. In `$((x))`, `((x))`, `[[ x -eq 1 ]]`, `${a[x]}`
and `${!x}`, the shell reads the value of `x` as an expression, and an array
index in that value (`a[$(...)]`) runs as a command. Vade asks before
arithmetic on command output, on script arguments, or on a variable that
any part of the command sets to text that is not a plain number, also with
`+=`, `${x:=value}` or as a loop value (zsh takes several loop names, as
in `for y x in ...`). A variable set only to numbers, such as a loop
counter, stays safe. `read`, `mapfile`, `getopts` and `unset` are safe only
with plain variable names, and `read` takes only `-r`.

A setting can choose the program that runs or its options:
`PATH=./tools`, zsh `path=(./tools)`, `RIPGREP_CONFIG_PATH`, `HOME` (git
reads its settings from there) or `http_proxy`. Setting a name in capitals,
as a prefix, an `export`, a plain assignment or a loop name, is safe only
for a closed list: `RUST_BACKTRACE`, `RUST_LOG`, `RUST_TEST_THREADS`,
`RUST_MIN_STACK`, `CARGO_TARGET_DIR`, `CARGO_INCREMENTAL`,
`CARGO_BUILD_JOBS`, `CARGO_TERM_COLOR`, `CARGO_TERM_VERBOSE`,
`CARGO_TERM_QUIET`, `CARGO_TERM_PROGRESS_WHEN`, `NO_COLOR`, `CLICOLOR`,
`CLICOLOR_FORCE`, `FORCE_COLOR`, `TERM`, `COLUMNS`, `LINES`, `LANG`,
`LANGUAGE`, `LC_ALL`, `LC_CTYPE`, `LC_COLLATE`, `LC_MESSAGES`,
`LC_NUMERIC`, `LC_TIME`, `TZ`, `CI`, `GOOS`, `GOARCH`, `GOMAXPROCS`, `NODE_ENV`, `PYTHONUNBUFFERED`
and `PYTHONDONTWRITEBYTECODE`. `RUSTFLAGS` and `RUSTDOCFLAGS` are safe
only with plain code generation settings. Any other name in capitals
needs approval. A name with a small letter is a shell variable and is
safe, except the names that zsh or common programs read, such as `path`,
`fpath`, `module_path` and `http_proxy`. `printf -v`, `printf '%n'`, and a
`test -v` or `[[ -v ]]` name that has an index or comes from a value need
approval. zsh also reads the values of `shift`, `exit`, `return` and the
number directives of `printf` (`%d`, `*`) and the count of `repeat` as
arithmetic, so Vade asks unless these values are plain numbers.

## Turning prompts off

`--yes`, or its long name `--dangerously-skip-prompts`, turns all prompts
off. It is the same as `--approval auto`: edits and commands run without
asking, inside the sandbox. Blocked commands stay blocked, but the block
check does not find every dangerous form (see [Limits](#limits)).

```sh
vade-harness agent --endpoint URL --model NAME --workspace . --yes
```

`--approval never` denies all edits and commands.

## Runs without a terminal

When no person can answer, for example with `--prompt` or when standard
input is not a terminal, Vade does not wait. Commands on an allow list run.
Every other command does not run, and the model gets this tool error:

```json
{"ok":false,"error":"approval_required","message":"Command not run: it needs approval (every command needs approval unless an allow list names it), and no one can approve it in this non-interactive run. ..."}
```

Pass `--yes`, or add exact rules, to let such a run use those commands.

## The journal

Each decision about a command that needs approval, or a blocked command, is a
`shell_approval` record in the session journal. Under the default policy
every command makes a record. Only with the experimental checker on do the
commands that it reads as safe make no record, except under
`--approval never`, where every command is denied and recorded as
`denied_by_policy`. A record holds the tool call ID, the argv, the class, the reasons,
the decision and the rules that applied:

| Decision | Meaning |
| --- | --- |
| `approved` | A person said yes. `rules` shows `once` or the rules the person chose to add, for example `session: git push origin main`. |
| `denied` | A person said no. |
| `allowed_by_rule` | An allow-list rule matched, for example `project: rm -rf build`. |
| `allowed_by_flag` | Prompts were off (`--yes` or `--approval auto`). |
| `denied_no_terminal` | No one could answer. |
| `denied_by_policy` | `--approval never` is set. |
| `blocked` | The command is blocked. |

Vade writes the record before it saves a rule. If Vade cannot write the
record for an approval, the command does not run and no rule is saved. If
the rule cannot be saved, the Allow list notice says so; the record still
names the rule that the person chose. Use `/journal` to inspect raw records.

## Limits

The default policy asks before every command that no rule names, so it
does not depend on the checker reading a command correctly. The checker
only refuses more: a command that it wrongly blocks can be run by hand
outside Vade. The sandbox is the security boundary. An allow-list rule
approves the exact argv, and that command runs the files as they are when
it runs, so a rule for `./build.sh` runs the script's current text.

The block check knows a fixed set of programs and forms. It finds a write
to a disk device in an output option, also as `-o/dev/sda` or
`--output=/dev/sda` and after an option that it does not read. When a
program name is a pattern or a variable (`/bin/r? -rf /`), it checks the
words as each blocking program that the name can be. It does not find
every way to delete a system folder or to write to a disk. A form that it
does not find needs approval, and with `--yes` it runs inside the sandbox.

The experimental checker is a guard against mistakes, not a security
boundary. An unknown program and a shell form outside the plain forms are
never safe in it. Within the plain forms, a known program is safe only
when each of its options is on that program's closed list; a wrong entry on
a list is a bug. It does not evaluate variables, and a path that starts
with a variable counts as outside the workspace. It expands braces
(`/{usr,etc}`) and `~/..` when it looks for system and home folders, and a
glob that can match one of them or a disk (`/u*`, `/dev/sd?`) is blocked as
that folder or disk. It does not read the contents of script files. Its
work on patterns and brace words per command is limited; past the limit,
each such word counts as every path it could be.

The source is in
[`harness/src/shell/policy.rs`](../harness/src/shell/policy.rs) (the
default policy),
[`harness/src/shell/safety.rs`](../harness/src/shell/safety.rs) (the
checker) and
[`harness/src/shell/permissions.rs`](../harness/src/shell/permissions.rs)
(the allow lists).
