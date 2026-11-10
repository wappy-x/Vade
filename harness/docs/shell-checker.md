# The shell checker (experimental)

This page describes how Vade's command checker reads a `workspace_shell`
command. By default the checker only blocks commands and gives the reasons
in a prompt; every other command asks (see
[shell safety](../../docs/shell-safety.md)). With `VADE_SHELL_CHECKER=1`,
the commands that it puts in its safe class also run without a prompt.
That safe class is experimental: reviews keep finding inputs that it calls
safe and that are not, for example a `RUSTFLAGS` value that comes from a
variable (`RUSTFLAGS="-D $x" cargo build`) and a pattern that can name a
credential file (`cat id_r?a`).

## How the checker reads a command

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
blocked. The same check reads each `trap` word, so
`trap -- 'rm -rf /' EXIT` is blocked. It also reads the operands of `let`
and the words of `declare`, `read`, `exit` and a number `printf`, but a
`$(...)` inside them whose text holds `=` is not found: for example
`let 'a[$(X=1 rm -rf /)]'` only asks. A brace word that gives more than
1024 words or 1 MiB of text, or a `{x..y}` form that is not a plain
sequence, cannot be listed. It needs approval and gets no rule. As the
target of a removal or a redirection, it counts as every path it could be,
so it is blocked. As an operand of other programs it only asks, for
example `tee {/dev/{0..1024},/dev/sda}`. A command that is too deeply nested or too long to check is blocked. The limits are 24
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

## Known gaps

The checker is a guard against mistakes, not a security boundary. An
unknown program and a shell form outside the plain forms are never safe in
it. Within the plain forms, a known program is safe only when each of its
options is on that program's closed list; a wrong entry on a list is a
bug. It does not evaluate variables, and a path that starts with a
variable counts as outside the workspace. It expands braces (`/{usr,etc}`)
and `~/..` when it looks for system and home folders. It does not read the
contents of script files. Its work on patterns and brace words per command
is limited; past the limit, each such word counts as every path it could
be.

The block check finds a write to a disk device in an output option, also
as `-o/dev/sda` or `--output=/dev/sda`, and a glob that can match a system
folder or a disk (`/u*`, `/dev/sd?`) in a removal or a redirection. These
forms are known not to be blocked; under the default policy they ask:

- A disk write through an operand or an option of `tee`, `base64`, `time`
  or `go build -o`, also after an option that the checker does not read
  (`base64 --unknown -o /dev/sda -i in`, `tee /dev/sd?`).
- A shell or wrapper name from a pattern or a variable
  (`/bin/ba* -c 'rm -rf /'`, `$SHELL -c 'rm -rf /'`). A pattern name is
  checked only as the blocking programs that it can name, such as
  `/bin/r? -rf /`.
- `su root -c '...'`, and a `case` pattern, a `<<$'EOF'` delimiter or a
  zsh or extglob `(...)` in a path, which the lexer does not read as the
  shell does.

The corpus tests in
[`harness/src/shell/safety/corpus_tests.rs`](../src/shell/safety/corpus_tests.rs)
keep every review input, and each one asks or is blocked under the
default policy.
