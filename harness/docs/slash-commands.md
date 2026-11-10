# Slash commands

Type a slash command on its own input line. Built-in commands take
precedence over user templates, even when the built-in is unavailable in the
current mode. `/help` lists commands. `/help all` lists all built-ins for the
current mode and the user commands. `/help NAME` shows a command's details.

## Built-in commands

`/help` and `/compact` already serve chat help and native context compaction.
`/clear` is an alias for `/new`. It starts fresh conversation state and keeps
old native journals. It waits for active work to finish. `/memory` reports
that saved memory is unavailable in this build. It does not save or remove
files. Use [project instructions](project-instructions.md) for saved rules.

The full built-in list and mode rules are in the harness
[command table](../README.md#commands-and-keys) and `/help all`.

## User templates

Create a UTF-8 file named `NAME.md` in one of these directories:

1. `STATE/commands/`, for the current project.
2. `VADE_HOME/commands/`, for all projects.

The first matching name wins. Built-ins come before both directories.
`STATE` is `VADE_HOME/projects/NAME-HASH` by default. With
`--state-in-project`, it is `WORKSPACE/.vade`. `--vade-home DIR` sets the
Vade home. Otherwise `VADE_HOME` sets it, with `~/.vade` as the fallback.
This follows the existing session and history locations.

For example, save this as `commands/review.md` in the project state directory:

```text
description: Review files for defects
Review $ARGUMENTS. Explain each defect and suggest a small fix.
```

Then type:

```text
/review src/main.rs
```

The prompt sent to the model is:

```text
Review src/main.rs. Explain each defect and suggest a small fix.
```

The optional first line `description:` supplies the text for `/help`. That
line is removed from the prompt. The rest of the file is the prompt body.
Each literal `$ARGUMENTS` is replaced with the text after the command name
and its first separating whitespace character. With no arguments, it becomes
empty text. Shell variables are not expanded. The result is sent as prompt
text. It is not parsed again as a slash command or file attachment.

Names can contain ASCII letters, digits, hyphens, and underscores. They are
case-sensitive and have at most 64 bytes. Each file and expanded prompt must
fit the existing 64 KiB input limit. Each directory scan is bounded, and its
template bodies can total at most 1 MiB. Links and nonregular files are
refused. The [template reader](../src/slash_commands.rs) defines these limits.
Files are read again on each use, so edits apply to the next command.

An unknown slash command reports the error and lists available names. It
does not send a prompt. Use `//text` to send the literal prompt `/text`.
Multiline paste keeps its existing behavior. Absolute file attachments and
input that starts with a path or API route also keep their existing behavior.
The raw terminal's command completion list contains built-ins. Use `/help`
to find user templates.

**Back:** [Harness docs](README.md)
