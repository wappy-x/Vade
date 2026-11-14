# Answer quality and exactness cost cards

`vade-harness quality-card` runs one fixed set of prompts against a served
model through the chat API and prints two cards in the RigMark card style:

- **Quality.** 12 prompts in three classes (code, structured output, prose),
  run once with drafting as served. One score per class plus a total.
- **Exactness cost.** The same 12 prompts with strict drafting, then with
  plain decoding. The streamed token ids are compared, the two decode rates
  sit side by side, and the verify cost comes from the server draft receipts.

Every check is exact. No model judges an answer.

A second set, `--set engine-1`, prints the engine card: five checks that
the served engine behaves. See [The engine card](#the-engine-card-engine-1).

## Run it

Build once from `harness/`. Give the endpoint and the model it serves. Use a
new JSON filename for each run; the receipt is never overwritten.

```sh
cargo build --release --offline
./target/release/vade-harness quality-card \
  --endpoint http://127.0.0.1:18082 --model "$ENGINE_MODEL" \
  --json quality-live.json --card quality-live.card.txt \
  --server-log server.log
```

`quality-card --help` lists every flag. `--only quality` or
`--only exactness` runs one card. `--print-prompts` prints the fixed set as
JSON without any request.

## The fixed set (quality-set-1)

| Class | Prompts | How an answer is scored |
| --- | --- | --- |
| Code | 4 small Rust functions | One fenced block, the exact signature, the answer's own tests present and passing, then fixed hidden unit tests compiled and run with `rustc --test` |
| Structured | 4 extraction and classification tasks | The answer itself parses as one JSON document (no code fence, no other text); the field sets are exact and each named field equals one exact value |
| Prose | 4 short writing tasks | A rubric of exact rules: required and forbidden text, sentence and line counts, word range, fixed form |

The prompts and their checks live in `harness/src/quality/cases.rs`. The set
carries a version string on every card. Change the set only with a new
version; scores across versions are not comparable.

### How the score is counted

- Each case yields a list of named checks. A check passes or fails; a check
  that could not run (code tests with `--no-code-run`, or no compiler) stays
  out of every denominator and the card labels it.
- Class score and total score are `100 × passed checks / run checks`.
- The card sets no pass mark. It prints raw counts next to every score.
- A low score is a measurement, not a failed run.

### Requests

One request at a time. Greedy (`temperature: 0`), thinking off, cache off,
fixed seed (default 20261027), per-case output caps of 200 to 700 tokens.
The quality card omits the `draft` field, as served, and reports how many
replies carry `draft_used: true` in their `vade` receipt.

### Code answers run on your machine

Code scoring writes the answer's block plus the hidden tests to a temporary
file, compiles it with `rustc --test` (unsafe code forbidden), and runs the
tests. The compile and the test run for one case share a single 60-second
bound, and a timed-out check stops its whole process group. This runs
model-written code on the machine that runs the card command.
Use `--no-code-run` to score code with
the text checks only; the card and receipt then say the unit tests did not
run. Code checks need a Rust compiler on `PATH` (`--rustc PATH` picks one).

## The exactness cost card

The served FlashNext path accepts only `draft: "auto"`, `"plain"` or
`"strict"`; `draft: "fast"` is refused at request validation, so no relaxed
verify path exists to measure against. This card therefore prices exactness
as strict against plain:

- **Ids identical N/N.** Both arms stream `token_id` per chunk. One check
  in the code (`evidence_failure`) defines complete token evidence. A
  reply is complete only when every streamed id is an exact integer (a
  fractional, negative or out-of-range id is never converted; it fails the
  cell), the stream ended with the terminal `[DONE]` event, and the reply
  reports a completion count and a finish reason. The number of streamed
  ids must equal the completion count. One exception: the engine counts
  its end-of-sequence token without streaming its id, so a `stop` reply
  (or a `tool_calls` reply, which ends the same way) with one fewer id is
  complete; a `length` reply streams its last token
  and gets no allowance. Both arms must also agree on the finish reason
  and the completion count, and report the draft modes back as strict then
  plain. Only then are the id lists and answer texts compared. Each
  failure prints one reason that names the failed condition, for example
  `no token ids`, `invalid token id`, `no terminal event`, `missing
  completion token count`, or a token id count mismatch with the numbers.
  A strict arm downgraded to plain by the engine fails as a draft mode
  mismatch with both modes printed. The card prints the complete-evidence
  facts per passed case, and the first differing token is named in the
  receipt.
- **Decode rates.** Median strict and plain tok/s per class, estimated from
  streamed token times (`(tokens − 1) / (total − first token)`), plus the
  strict/plain gain. This is what drafting buys.
- **Verify cost.** The API receipt carries no accepted-token or verify-time
  fields, so the cost comes from the server log: one `draft id=...` line per
  request with `cycles`, `accepted`, `proposed`, `target_ms` and `draft_ms`.
  Pass `--server-log FILE`. Completion ids repeat across server runs, so
  each strict completion id must match exactly one line in the log; an
  absent or repeated id makes the cost unavailable, and unrelated lines are
  never summed as this run's cost. The matched lines are saved in the JSON
  receipt. The card prints verify ms per step (`target_ms / cycles`) and
  accepted tokens per step. Without a log, or with no exact match, the card
  says why the cost was not measured; it never estimates it. A log that
  cannot be read is recorded in the receipt and never discards the answers.

## Outputs and exit codes

| Output | Holds |
| --- | --- |
| Standard output | The two cards as text |
| `--card FILE` | The same card text in a new file |
| `--json FILE` | The `vade.quality-card.v1` receipt: prompts, raw answers, token ids, per-check results, timings, draft receipts |

Both files use `create_new` with private Unix mode 0600 and are checked
before any request is sent. The receipt holds no endpoint credential; the
global `--api-key-file` (or `VADE_API_KEY_FILE`) key is sent as a Bearer header and never written down.

| Exit | Meaning |
| --- | --- |
| 0 | Every request completed; strict and plain token ids are present and identical, with returned draft modes strict then plain |
| 1 | A request failed, token ids are missing or differ, draft modes mismatch, or an output write failed |
| 2 | Invalid flags, or an output destination that exists or cannot be created |

## The engine card (engine-1)

`quality-card --set engine-1` grades five checks against one live server.
It does not score answer quality; it checks that the engine reads long
prompts, keeps to the requested output form, and gives the same token ids
whatever the cache or the batch.

```sh
./target/release/vade-harness quality-card --set engine-1 \
  --endpoint http://127.0.0.1:18082 --model "$ENGINE_MODEL" \
  --json engine-live.json --card engine-live.card.txt
```

`--hardware TEXT`, `--recipe TEXT`, `--seed N` and `--request-seconds N`
work as for the quality card. `--only`, `--server-log`, `--rustc`,
`--no-code-run` and `--print-prompts` apply to quality-set-1 only and are
refused with engine-1.

| # | Check | Requests | Pass rule | Number |
| --- | --- | --- | --- | --- |
| 1 | Long-context recall | 2 probes, then one prompt of at least 8,192 and one of at least 32,768 tokens | The answer holds the planted code verbatim and the server counted at least the target prompt tokens | n/2 |
| 2 | JSON mode validity | 50: 10 schemas, 5 source texts each, `response_format` `json_schema` (strict); 40 prompts name the keys, 10 do not | The whole answer parses as one JSON value and the schema accepts it | n/50 |
| 3 | Tool-call correctness | 20: 5 tools offered on each, `tool_choice` `auto`, `parallel_tool_calls` false | Exactly one call, the exact tool name, and arguments that parse to exactly the expected object | n/20 |
| 4 | Cache invariance | 20 prompts: cold (`"cache": false`), then prime and warm (`"cache": true`) | Cold read 0 cached tokens, warm reused more than 0, and cold and warm have identical token ids | n/20 |
| 5 | Concurrency invariance | 3 prompts: one alone, then 8 copies at once | Each copy has the token ids of the run alone, and all the copies that answered streamed at one shared moment; a failed copy counts as a miss | n/24, shown per prompt |

How each check is built:

- **Needle.** The filler is plain sentences from fixed word lists and a
  seeded generator, made at run time; no file is checked in. Two probe
  requests of 64 and 320 sentences measure the server's own tokens per
  sentence from `usage.prompt_tokens`, so the card sizes its prompts for any
  tokenizer. The planted sentence sits at the middle of the filler; each
  depth has its own code. The card prints the prompt tokens the server
  counted. The filler has a 3 percent margin plus 8 sentences. At 12 tokens
  per sentence and a 132-token frame that is 2,810 sentences, 33,852 prompt
  tokens for the 32K needle.
- **Needle size check.** Before it sends a needle, the card sends the exact
  request body to the server's `/v1/admission` check. The server counts the
  prompt with its chat template and compares it with its context and input
  limits. It runs no model and keeps no memory. The prompt must leave 32
  tokens for the answer. If the server would refuse the calibrated prompt,
  the card changes the filler with the server's own count until the prompt
  holds from the target up to the largest size the server admits, in at most
  6 checks. If the server's limits are below the target, the card does not
  send the needle. The card shows `NOT RUN`, the server's limits and the
  context to use. The pass rule does not change: a needle that did not run
  is not a pass, so the grade is C. `vade up` starts with a 32,768-token
  context, prompt and answer together, so the 32K needle needs a larger
  one: `vade up --context 40960`. A server without `/v1/admission` gets the
  calibrated size unchecked, and the card says so.
- **JSON mode.** The schemas use the subset that a constrained JSON mode can
  enforce: `type`, `properties`, `required`, `additionalProperties: false`,
  `items`, `enum`, `minimum` and `maximum`. For four source texts of each
  schema the prompt names the keys too, as a client would. For the fifth it
  names no key, so only `response_format` carries the schema: a server that
  ignores the field cannot pass those 10 items by prompt following alone.
  The card prints both counts. Vade ignores `response_format` today (see
  [Structured output](../../docs/openai-api.md#structured-output)), so the
  10 schema-only items will most likely miss; two or more misses in one
  check is grade C. A C from this check on current Vade is a known gap, not
  a regression. A code fence, a sentence before the JSON, or
  a missing key is a miss. The card checks form and schema, not the
  extracted values.
- **Tool calls.** The prompts name every argument value; one holds a
  non-ASCII city name, so the streamed argument text must keep its UTF-8.
  Key order and number spelling are free (`250` equals `250.0`); every
  other difference is a miss.
- **Cache.** The cold pass sends `"cache": false` on all 20 prompts first:
  the server neither looks up nor stores the prompt. Then each prompt is sent
  twice with `"cache": true`: the first run stores the prompt and the second
  must report `usage.prompt_tokens_details.cached_tokens` above 0. The
  served FlashNext prefix cache is opt-in: a request that omits the field
  runs uncached (`prefix_cache_opt_in` in
  `engine/crates/serve/src/flashnext.rs`), so the card asks for the cache
  explicitly. Each prompt holds 96 filler sentences. The prime run is kept
  in the receipt and is not graded. The check needs the prefix cache on: it
  is off with `--prefix-cache off`, with `VADE_FLASHNEXT_PREFIX_CACHE=0`,
  and by default on the layer chain
  (`engine/crates/serve/src/prefix_switch.rs`). With it off, every warm
  run reads 0 cached tokens and the grade is C.
- **Concurrency.** Each request in this check sends `"cache": false`, so only
  the batch differs. Eight threads wait on one barrier and send together.
  Open sockets are not enough: of the copies that answered, the last first
  token must arrive before the first one ends, or the prompt counts 0 of 8
  and the card says the copies did not overlap. A failed copy counts as a
  miss and is left out of that window. The check needs a server that runs 8
  chats at once (`--max-active 8` or more; with an owner key and three or
  more slots, one slot is kept for the owner, so use 9). A server with fewer
  slots queues the copies, and the prompt counts 0 of 8.

Every request is greedy (`temperature: 0`), thinking off, with the `draft`
field omitted (drafting as served). Every request except the prime and warm
runs of the cache check sends `"cache": false`. A pair of replies is
identical only with complete token evidence on both (the same rule as the
exactness card): valid integer ids, the terminal event, counts that match
the ids, the same finish reason and count, then the same ids and text. The draft mode may differ: with the field
omitted the server drafts strict for one chat and plain above its chat
limit, and drafting on and off must give the same ids. The card prints the
modes of the run alone and of the copies.

The card reads the build, Spark count and engine settings from the server's
own `vade` reply object: `build.git_sha`, `numerics.world_size`,
`numerics.nvfp4`, the drafting depth and the first 16 characters of
`checkpoint.sha256`. It prints `unknown` for a field the server does not
send. The receipt also keeps `checkpoint.directory`; the card leaves out
that host path, so the card can be shared.

**Grade.**

| Grade | Rule |
| --- | --- |
| A+ | All five checks pass with no miss |
| A | One soft miss: exactly one item short in JSON mode or in tool calls |
| B | Two soft misses: one short in JSON mode and one short in tool calls |
| C | Anything else: a missed or not-run needle, any cache or concurrency item that is not identical, or two or more items short in one check |

The output is the boxed card, then a markdown table with the build sha,
Spark count, model, drafting, NVFP4 flag, wall time, grade and one row per
check. The receipt (`vade.engine-card.v1`, written with `create_new`) holds
every graded reply with its token ids, the prime runs of the cache check,
and every verdict. The two calibration probes keep only their prompt token
counts. Long generated prompts are
named by seed, sentence count and an FNV-1a 64 hash instead of their text.
Exit 0 means grade A+. Exit 1 means a lower grade, a failed write, or a
setup error after the output files exist (for example an unreadable
`VADE_API_KEY_FILE`); that run leaves
an empty JSON file, so retry with a new `--json` path. Exit 2 means invalid
flags or an output file that exists.

## Check it yourself

No GPU is needed:

```sh
cargo test --offline --all-targets
cargo test --offline quality:: -- --nocapture
cargo test --offline --test quality_card_mock -- --nocapture
cargo test --offline --test engine_card_mock -- --nocapture
RUSTFLAGS='-D warnings' cargo check --offline --all-targets
```

The fixtures cover a golden answer for every case (code goldens run under
the hidden tests), failing answers, failed answer-owned unit tests, exact
JSON field sets, SSE parsing, draft receipt parsing, verify-cost matching
(unique, repeated and unrelated completion ids), card width, and full runs
against a loopback mock server: identical ids, differing ids, missing token
ids, fractional token ids, a stream with no terminal event, a downgraded
strict arm, the real CLI sending the configured API key, a clean retry after
a blocked card file, and a trickling stream against the request deadline.
Unit fixtures pin each incomplete-evidence condition by its failure reason:
short id list, short id list on a length finish, missing completion count,
missing finish reason, no terminal event, invalid streamed ids, mismatched
draft modes, and arms that disagree on finish or count.

The engine card tests run a loopback server that answers every check. It
shows a clean server at A+, and one defect at a time: one fenced JSON answer
(A), one more wrong tool argument (B), two wrong tool arguments, a missed
32K needle, a 32K needle that a 32,768-token context cannot hold (NOT RUN,
not sent), a warm-cache id change, cold runs that read the cache, an id
that changes when five requests are in flight, and a server that streams
one request at a time (each C). A context of 33,400 tokens makes the card
size the 32K needle down to fit (A+), and a server without `/v1/admission`
gets the calibrated size (A+). Unit tests pin the schema checker, the tool
call match, streamed tool call parts, the pair rule, the grade table, the
calibration and the card width.

**Where it has run so far:** on the build Mac against loopback mocks in the
test suite, and on one DGX Spark for filed cards. Filed cards and receipts
live with the operator's run records outside this repository.
