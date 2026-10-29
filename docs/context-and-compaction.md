# Context and compaction

A chat in the native agent (`vade chat`, or `vade-harness agent`) does not
stop at the end of the model's context window. When the next turn would not
fit, the model writes a short summary of the older turns, the most recent
turns stay word for word, and the chat continues. You see one status line,
where N is the number of kept turns and M is the size of the summary:

```text
[context compacted: kept N turns, summary M tokens]
```

This page tells you how the harness counts the context, when it compacts, what
it keeps, and what it records. The work happens only in the harness. The
engine is not changed, and the harness changes only the prompt that it sends.

## The context budget

After every turn, the harness records one budget entry:

- **Prompt tokens.** The tokens that the conversation now uses. When the
  engine reports usage, this is the reported prompt tokens plus the reply
  tokens of the last response, an exact count. Otherwise it is the local
  estimate: request bytes divided by 3, plus framing and a safety margin.
- **Window.** The model's context window. The engine reports it, and
  `--context-tokens N` can make it smaller.
- **Reply reservation.** The room kept for the next reply: the requested
  output, but at most 8,192 tokens and at most a quarter of the window.

`/context` shows the newest 8 entries first, one line per turn. This example
line has made-up numbers:

```text
context after turn 7: 98304 of 131072 prompt tokens (75%), 8192 kept for the reply, 24576 left
```

An estimate starts with "about". The status strip also shows `ctx used/window`
after each reply.

## When the harness compacts

The harness checks the budget before it sends a request, and again before
each request inside a long task:

1. **At 80% of the window.** The harness compacts once. It compacts again
   only after the use has gone below 60%. This prevents a rewrite on every
   turn.
2. **When the next turn would not fit.** The harness always tries to compact,
   at any use level. Before it saves a new message, it also checks that a
   compaction can make the message fit. If no compaction can, it does not send
   the message (see [the hard stop](#the-hard-stop)).

`--no-compact` turns off both automatic triggers. (`--no-auto-compact` is an
older name for the same flag.) `/compact` still works.

## What a rollover keeps

A rollover replaces the older turns with one state block. A turn starts at a
user message.

- **The frozen prefix.** The options, the tool definitions and the leading
  system and developer messages keep every byte.
- **The state block.** One user message, named `vade_context_summary`, that
  comes directly after the frozen prefix.
- **The recent turns.** The newest 2 user turns, with every assistant reply,
  tool call and tool result in them, word for word. If 2 turns do not fit,
  the harness keeps 1 turn.

All other messages leave the prompt. The journal keeps them.

### The kept-turns rule

The harness tries 2 kept turns, then 1. For each choice, it counts the frozen
prefix plus the kept turns. It then adds room for a full state block. The
first choice that fits the window, and that is smaller than the current
prompt, wins. The model is not asked for a summary before a choice fits.

### The state block

The state block has four parts:

1. **A header.** It tells the model that the block is lossy data and not a
   new instruction, that tool outputs are references, and that it must read a
   file again before it edits that file.
2. **The model's summary.** The model writes it in these sections: Plan,
   Decisions, Open tasks, Files (path, what changed or was read, and its
   current state) and Facts. The harness keeps at most 3/8 of a byte per
   window token, from 1 KiB to 8 KiB. A longer reply is cut, and a note says
   so.
3. **Tool outputs by reference.** The harness writes this part, not the
   model. Each file gets one line with the line ranges and byte ranges that
   were read ("bytes from 0" for a byte read with no range; "listed or
   searched" for a listing or a search) and the number of edit calls and
   applied edits. Each of the last 8 commands gets its arguments and its
   status line, also when the output was captured to an artifact. No tool
   output text is copied. This part is at most 3 KiB.
4. **The user asks.** The user messages of the summarized turns, word for
   word, oldest first, up to 4 KiB in total. The harness note of a resumed
   or forked chat is not an ask and is left out. Asks that an earlier state
   block carried come first, so a second rollover keeps them too. The last
   summarized ask always stays; if it is longer than 4 KiB, it is cut and a
   note says so. When older asks do not fit, a note gives their number
   (with the asks that an earlier block left out), and the model's summary
   covers them. The newest ask is always in the kept turns.

### How the model writes the summary

The summary request is an ordinary request to the same model, with the same
options and tool definitions:

- It starts with the frozen prefix, so the engine can reuse its cached
  tokens.
- Then comes one user message with the instructions and a digest of the
  older turns. In the digest, text is shortened, tool calls show only their
  path and line ranges or their arguments, and tool results show only their
  size and, for a receipt, its status line. The digest gets the room that
  the window has after the prefix, the reply reservation and 1,024 tokens
  for the instructions, from 1 KiB to 64 KiB. When it is too long, the
  oldest entries go first, and earlier summaries stay.
- The reply must be text. A reply with a tool call, or with no text, is
  refused, and no tool runs.

The harness then counts the new prompt again. It must fit, and it must be
smaller than the old prompt. `summary M tokens` in the status line is the
count of the new prompt minus the count without the state block. It is exact
when the engine counts tokens, and an estimate otherwise.

## Fallbacks

The harness never deletes history that it cannot replace safely.

1. **Model rollover.** This is the first choice.
2. **Local excerpts.** If the model reply fails, if it calls a tool, if a
   tool outcome in an older turn is unknown, or if one kept turn and a full
   state block do not fit, the harness uses its older planner. That planner
   keeps the user messages and the latest edit and check receipts word for
   word and excerpts older results. In an automatic compaction, the notice
   names the reason that the model summary was not used.
3. **The hard stop.** If neither one fits, see the next section.

A rollover does not summarize a turn that holds a tool call with an unknown
outcome. If such a call is in an older turn, the local excerpt planner keeps
that call and its result word for word. Look at the workspace before you repeat that call.

## The hard stop

If even a compacted prompt does not fit, the harness sends nothing and says so.
This example has made-up numbers:

```text
This chat cannot continue: even after compaction the prompt and the room for the reply need about 140000 tokens and the context window holds 131072, so it was not sent. Start a new chat with /new, or send a shorter message.
```

The number includes the room for the reply. With `--max-output-tokens`, or
when the count is the local estimate, this is the full requested output (at
least the reserve). An automatic reply that the engine counts shrinks with
the prompt, so it needs only the reply reservation and 256 tokens. The same
rule picks the kept turns. This happens when the newest
message alone, or the newest turn alone, is too large for the window. The
history stays as it was.

## Commands and flags

| Command or flag | What it does |
| --- | --- |
| `/context` | The budget after each turn, the per-message estimates and the last request's admission |
| `/compact` | A model rollover now, with local excerpts as the fallback |
| `/compact KEEP SUMMARY` | Your own summary; keeps the newest KEEP user turns |
| `/compact-preview [KEEP SUMMARY]` | Preview the local excerpt plan, or your own summary, and change nothing |
| `/compaction` | The latest compaction: who wrote the summary, the kept turns, the summary and the source records |
| `--no-compact` | No automatic compaction; `/compact` still works |

## The journal record

Every compaction appends one record to the session journal. The record holds
the new messages and a `compaction` object with:

- `summary_origin`: `model`, `local excerpts` or `user`;
- `kept_turns` and `summary_tokens` for a model rollover;
- `summary`: the full state block;
- `removed_message_indices` and `retained_message_indices`: what was
  summarized and what was kept;
- `exact_input_tokens`: the counts before and after, when the engine counted
  them;
- the source record, the context generation and the unchanged prefix length.

The records before it do not change. A resumed session replays the
compaction, so it continues from the same compacted prompt. `/journal RECORD
OFFSET LENGTH` reads the original records.

## Exactness

Compaction changes only the prompt that the harness sends. The engine, its
arithmetic and its token IDs do not change. For the same prompt and settings,
the engine still returns the same token IDs. After a compaction the prompt is
different, so the next reply can differ from the reply to the full history.
This is expected.

A compaction makes the engine read the new suffix again, so the next turn has
fewer cached tokens. The frozen prefix keeps its bytes, so its cached tokens
can still be used. The next response's reported usage gives the real number.

## Limits

- The quality of the summary depends on the model. The harness checks the
  structure, the size and the fit, not the meaning.
- `vade chat` does not pass `--no-compact` yet. Use
  `vade-harness agent --no-compact` directly.
- Direct text chat, below an 8,192-token window, has no automatic compaction.

The code is in `harness/src/context/rollover.rs`. The older planner and the
budget arithmetic are in `harness/src/context.rs`. The
[harness context guide](../harness/docs/context-and-projects.md) covers
project rules, branches and the local excerpt planner in more detail.
