# Answer quality and exactness cost cards

`vade-harness quality-card` runs one fixed set of prompts against a served
model through the chat API and prints two cards in the RigMark card style:

- **Quality.** 12 prompts in three classes (code, structured output, prose),
  run once with drafting as served. One score per class plus a total.
- **Exactness cost.** The same 12 prompts with strict drafting, then with
  plain decoding. The streamed token ids are compared, the two decode rates
  sit side by side, and the verify cost comes from the server draft receipts.

Every check is exact. No model judges an answer.

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
  with one fewer id is complete; a `length` reply streams its last token
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

## Check it yourself

No GPU is needed:

```sh
cargo test --offline --all-targets
cargo test --offline quality:: -- --nocapture
cargo test --offline --test quality_card_mock -- --nocapture
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

**Where it has run so far:** on the build Mac against loopback mocks in the
test suite, and on one DGX Spark for filed cards. Filed cards and receipts
live with the operator's run records outside this repository.
