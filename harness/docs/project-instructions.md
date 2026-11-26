# Project instructions

**The instruction files Vade reads at session start and puts in the system
prompt.** This page covers direct and agent mode. Codex and Hermes modes hand the
turn to an external agent, and that agent reads its own files.

## Which files load

Vade looks in each folder from the git root down to the workspace folder. The
git root is the nearest folder, at or above the workspace, that holds `.git`.
If no folder holds `.git`, Vade looks only in the workspace folder. It never
reads folders above the git root. In this guide, the project folder is the git
root, or the workspace folder when there is no git root.

In each folder, Vade tries these names in this order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `.vade/instructions.md`

The first file that loads wins, and Vade does not read the other names in that
folder. If a file is skipped, Vade tries the next name in the same folder.

Vade also reads one user-level file, `~/.vade/instructions.md`, for rules that
apply to all of your projects.

The files go into the prompt in this order: the user-level file, then the git
root, then each folder nearer to the workspace. The nearest file comes last, so
it is the most specific. Each file starts with one header line that gives its
path relative to the workspace, for example `## ../AGENTS.md` or
`## ~/.vade/instructions.md`. The prompt never contains an absolute path.

## Links and skipped files

An instruction file can be a symbolic link, but only to a file in the same
folder or below it. Vade checks every link that it meets on the way, including
a link inside a link target and a folder link:

- The link must not be absolute, and it must not go up with `..`.
- Each name in the link must pass the same rule as a workspace read: no hidden
  name such as `.env`, no credential name such as `credentials.json`, no key
  suffix such as `.pem`, and only ASCII.
- A link can reach a `.vade` folder only as the documented path
  `.vade/instructions.md`, at the end of the link. Inside any `.vade` folder,
  including `~/.vade`, Vade opens only `instructions.md`, and that file must not
  be a link. A `.vade` folder also holds private data, such as the input
  history of `--state-in-project`, and no link can reach it.
- Each name that Vade opens must be the exact name in the folder. On a
  filesystem that ignores letter case, a link to `pick.txt` does not open a
  file named `picK.txt`.
- Vade follows at most 8 links for one file.

The user-level folder `~/.vade` must be a real folder, not a link. Thus a link
cannot put a file from another place, such as a key, a secret or a private
note, into the prompt.

Vade does not open a file by its path. It starts at the project folder (or
`~/.vade`) and opens one name at a time from the folder that it opened before,
and it never follows a link at that open. A link is resolved only by its text,
after the checks above. Thus a folder that another process replaces with a
link after a check cannot lead the read out of the project. The open also
never waits for a writer, so a path that changes to a FIFO is skipped and
cannot stop a turn.

A file that Vade cannot use is skipped, and `/instructions` gives the reason:

- A link is absolute or goes up with `..`.
- A link names a hidden, credential or non-ASCII file.
- A name in a `.vade` folder is not `instructions.md`, or a link is there.
- A name is not the exact name in its folder.
- A folder on the way is a link, or `~/.vade` is a link.
- A link points to a missing file, or the chain has more than 8 links.
- The file has more than one hard link.
- It is not a regular file, for example a folder or a FIFO.
- It is not UTF-8 text, or it cannot be read.
- It is the same file as one that already loaded. For example, the workspace
  is `app/web` and `app/CLAUDE.md` links down to `web/AGENTS.md`. The text goes
  in once, under `../CLAUDE.md`. This case counts as loaded, so Vade does not
  try the next name.

## Where the text goes

All the files go in one section of the system prompt. The section starts with
the line `# Project instructions` and comes after Vade's built-in prompt. The
built-in text stays first, so every project shares the same start of the
prompt.

The section tells the model that the files do not change tool permissions or
the approval policy, and that an explicit user request comes first.

In agent mode, the root folder's `AGENTS.md` is in this section, not in the
frozen project rules. Files from `--rules-scope` folders and the skills catalog
stay in the frozen project rules. See [context and
projects](context-and-projects.md).

## Size limits

Vade reads at most 32 KiB (32,768 bytes) of one file. The section, with its
header lines, is also at most 32 KiB. Vade never drops text without a note in
the section:

- **A file over 32 KiB.** Vade keeps the first 32 KiB and puts one line after
  it, for example `[AGENTS.md is cut: Vade reads at most 32768 bytes of one
  file, and this file has 40000 bytes. The rest of the file is not sent.]`
- **A section over 32 KiB.** Vade cuts the section and ends it with one line
  that gives the bytes shown and the total. The line names the file whose end
  is cut and each file that is not sent. If that list is too long, the line
  gives only the number of files and the first of them; `/instructions` lists
  them all. The files later in the list are cut first.

`/instructions` marks each cut file with `cut` and lists the files that are not
complete. A reload that leaves a file cut says so in its status line. Shorten
the files to keep all of them.

## Changes between turns

Before each turn, Vade checks every name that it looks for: the text of each
link on the way, and the identity, modified time and size of the file at the
end. A new, changed or removed file, or a link that changes, makes Vade read
the files again. Thus a link that you repair takes effect on the next turn. If
the section text is different, Vade shows one status line, for example:

```text
Project instructions changed; reloaded ../AGENTS.md, CLAUDE.md.
```

- **Direct mode** sends the new section in the system prompt of the next
  request.
- **Agent mode** keeps the system message frozen, so the cached prompt prefix
  and the journal stay valid. The new section goes with the turn in one user
  message named `vade_project_instructions`. That message tells the model that
  it replaces the section in the system prompt and any root `AGENTS.md` in the
  frozen project rules. If no file applies now, the message tells the model to
  ignore both.
- **Child agents** get the parent's latest `vade_project_instructions` message
  after the frozen prefix, so a child follows the rules that its parent follows
  now. No other parent history goes to the child.
- **Resume** compares the files with the text that the saved session carries:
  the last update message, else the section in the system prompt. If they
  differ, the first turn after the resume carries the new section.
- **Sessions saved before this section existed** have the root `AGENTS.md` in
  their frozen project rules. On the first turn after the resume, Vade sends
  one update message that replaces that copy, also when the file is unchanged
  or removed.
- **`/new`** reads the files again, and the new session has the current section
  in its system prompt.

## Commands and options

| Command or option | What it does |
| --- | --- |
| `/instructions` | Lists each file that loaded or was skipped, in prompt order, with its size, path and reason, and the section size. It reads the files again first if they changed. |
| `--no-instructions` | Direct and agent mode: load no instruction file. The root `AGENTS.md` is then not in the frozen project rules either. On `--resume`, the first turn sends one update message that tells the model to ignore the saved instructions. |

**Back:** [Harness docs](README.md) · **Docs:** [index](../../docs/README.md)
