# Saved chat sessions

Each agent chat is a saved session. You can list your sessions, continue one
later, export one as a readable transcript and delete the ones you do not need.
No engine is needed to list, export or delete. Direct text chat saves no
session. `vade chat` opens the agent when the engine context is 8192 tokens or
more, and direct text chat below that.

```sh
vade chat --name "login bug"          # start a session with a name
vade sessions                         # list the sessions of this folder
vade chat --resume "login bug"        # continue it later
vade chat --resume last               # or continue the newest one
vade sessions export last --markdown > login-bug.md
vade sessions delete session-1793250720000-4242-0
```

## What a session is

`vade chat` opens the Vade coding agent (`vade-harness agent`) in the current
folder. The agent writes each message, tool call and tool result to a private
journal before it continues. The journal is the session. There is no second
store: list, resume, export and delete all read the same journal.

Each session has:

- **An ID.** The agent makes one when the session starts, for example
  `session-1793250720000-4242-0`. The ID is the journal file name. The harness
  flag `--session ID` chooses your own ID.
- **A name (optional).** Give one with `--name NAME`. A name has 1 to 64
  characters, with no control characters, no `/` or `\`, and no spaces at the
  start or end. It cannot start with `-`, because a command line reads that as
  an option. The word `last` is reserved. Two sessions can have the same name.

A session belongs to the folder where it started. `vade sessions` shows the
sessions of the current folder, so run it, `export` and `delete` from that
folder. A copy or a fresh clone of the folder is a different folder.

Direct text chat does not save sessions. `vade chat` uses direct text chat only
when the engine runs with a context below 8192 tokens, and the
`vade run qwen3-8b` chat is always direct. Direct chat keeps the conversation in
memory only, so `--resume` and `--name` are refused there.

## Where sessions are kept

Sessions are private files outside your project, so git never sees them:

```text
~/.vade/projects/<folder name>-<hash>/sessions/<ID>.journal
```

`VADE_HOME` changes `~/.vade`. With `VADE_CHAT_IN_PROJECT=1`, sessions go to
the project's git-ignored `.vade/sessions/` folder instead. The
[CLI reference](../cli/README.md#where-vade-keeps-your-files) explains both
locations.

## List sessions

```sh
vade sessions          # this folder
vade sessions --all    # every project under the Vade home
```

`--all` reads the Vade home, so it does not work when `VADE_CHAT_IN_PROJECT=1`
keeps sessions in the project folder. Run `vade sessions` in each project
instead.

The list has one row per session, newest first:

| Column | Meaning |
| --- | --- |
| ID | The session ID |
| NAME | The name, or `-` |
| STARTED | When the session started |
| LAST TURN | The last write to the journal: the end of the last turn or step |
| TURNS | How many prompts you sent. A resume note is not a turn |
| MODEL | The model ID the session used |
| PROJECT | The folder the session belongs to |

Times are in UTC. Sessions from older builds have no saved start time. For
those, the list uses the time in a generated ID, then the file creation time
when the system keeps one, else `-`.

The list runs the same checks on each journal as a resume does: the version,
sequence number, kept history, options, message list and compaction source of
each record, the record count, and the folder the session started in. A
journal that fails these checks shows `cannot resume` and the reason in the
MODEL column, and `last` skips it. The other rows still show. A session that
started in a different folder at the same path, for example a fresh clone,
shows `cannot resume: journal belongs to a different workspace directory`. You
can still export it. `vade sessions --all` does not compare folders, because it
does not open each project.

The list reads up to 4096 entries of a session folder (a deleted session leaves
its empty lock file, which counts too) and up to 1024 projects under the Vade
home. When a folder has more, the list ends with a warning that it may miss
sessions. A name or `last` then does not select a session from that folder,
because the newest one may be missing. Use the session ID. `--all` also warns
about a project session folder that it cannot open, and lists the others.

Inside a chat, `/sessions` prints the same table.

## Resume a session

```sh
vade chat --resume ID
vade chat --resume NAME
vade chat --resume last
```

- An exact ID comes first.
- `last` selects the newest session that has at least one turn, that no other
  chat has open, and that a resume accepts. When no session qualifies, the
  error says so. `vade sessions` shows why for each one.
- A name selects the newest session with that name.

A resume replays the journal into the model context: the system prompt, every
message, tool call and tool result, in the same order. The model and the tool
definitions must match the saved session. A resume does not send a prompt on
its own, does not run a tool again and does not restore files. The agent adds
one note that tells the model that workspace files may have changed since.

Add `--name NAME` to a resume to rename the session.

Inside a chat, `/resume ID|NAME|last` switches to another session after the
current work stops. The harness takes the same values:
`vade-harness agent ... --resume ID|NAME|last`.

### Compacted sessions

When the agent compacts the context, the journal keeps the original records and
adds a compaction record. A resume restores the compacted context, as the model
saw it last. A record kind that this build does not know is ignored, so a
journal from a newer build still resumes when its messages are compatible.

## Export a transcript

```sh
vade sessions export ID|NAME|last --markdown > chat.md
```

The export is Markdown. It starts with the ID, name, model, project, start
time, last turn and turn count. Then it shows the conversation:

- each prompt under **User** and each answer under **Assistant**, including an
  answer that the model gave as a refusal;
- each tool call with its name and arguments, in a code block;
- each tool result in a code block;
- a line where the session was resumed;
- a note that Vade added for the model, such as a project instructions update,
  with its text in a code block (it is not a turn);
- a line where the context was compacted, with the summary the model continued
  from.

The export keeps the turns that a compaction removed from the model context, so
it is the full conversation. A step that the agent replaced before it finished
shows only once, in its final form. The system prompt and the frozen project
context are left out. Control characters are removed.

An export only reads the journal. It works while the session is open in a chat.
It runs the same checks as a resume, except the folder check, and refuses a
damaged journal.

## Delete a session

```sh
vade sessions delete ID
```

Delete takes an ID only, not a name. It removes the journal file. It refuses a
session that a chat has open. Close that chat first. Engine cache hints that the
session saved expire on their own.

## Commands at a glance

| Command | What it does |
| --- | --- |
| `vade chat --name NAME` | Starts a session with a name |
| `vade chat --resume ID\|NAME\|last` | Continues a saved session |
| `vade sessions` | Lists the sessions of this folder, newest first |
| `vade sessions --all` | Lists the sessions of every project |
| `vade sessions export ID\|NAME\|last --markdown` | Prints a Markdown transcript |
| `vade sessions delete ID` | Deletes a closed session |
| `/sessions`, `/resume ID\|NAME\|last` | The same inside a chat |
| `vade-harness sessions ...` | The harness command behind `vade sessions`; it adds `--workspace PATH`, `--vade-home DIR` and `--state-in-project` |

The [harness README](../harness/README.md#sessions-and-journals) covers the
journal itself: crash recovery, limits, forks and branches.
