# Qwen3.8-Flash-Next-FP8 reference fixtures

These are retained reference fixtures for Vade's Qwen3.8-Flash-Next-FP8 engine.
Flash-Next now serves HTTP requests through `flash-serve` across two Sparks;
follow the [installation guide](../../../docs/INSTALL.md) and
[server guide](../../docs/flash-server.md) to run it. Current measured results
and their limits are in [Benchmarks](../../../docs/benchmarks.md).

This folder is for engine developers. It contains made-up conversations and
reference outputs, with no model weights or real conversations. The files
retain the original reference arithmetic described below; they are not a
reference for today's FP8 expert multiplication or a serving qualification.
See the [arithmetic contract](../../docs/flash-fp8-math.md) and
[validation record](../../docs/validation.md).

## Reference data

**FP32 references for three made-up chats, plus a BF16 run of the short
case.**

<dl>
<dt>Model</dt>
<dd>Official Qwen3.8-Flash-Next-FP8.</dd>
<dt>Made with</dt>
<dd>Hugging Face Transformers on the CPU, loading layers and experts a few at a
time, with plain ("eager") attention. Vade took no part.</dd>
<dt>Reference precision</dt>
<dd>FP32. The FP8 weights were converted to BF16 with the official code, then
promoted to FP32. The short case also has a BF16 run.</dd>
<dt>Settings</dt>
<dd>Thinking off; each answer token is the highest-scoring one. Library versions
and checkpoint file sizes are in each JSON file.</dd>
<dt>Cases</dt>
<dd>Three made-up chats: short, medium and long.</dd>
<dt>Used by</dt>
<dd><code>load-check --flash-session</code></dd>
</dl>

FP8, BF16 and FP32 are 8-bit, 16-bit and 32-bit number formats.

## Which program reads them

**`load-check --flash-session` compares the Flash engine with the case
files and score rows.**

```text
load-check --flash-session PTX MODEL FIXTURE.json ADDR0 ADDR1 [RANK]
```

- `ADDR0` and `ADDR1` are the two ranks' addresses. A rank is one Spark's
  process in the two-Spark engine.
- On two Sparks, run it on each with its `RANK`, 0 or 1.
- Leave out `RANK` to run both ranks locally.

The routing and layer files are reference data for debugging. The shipped
check does not read them.

<details>
<summary><b>What is in the folder</b></summary>

<dl>
<dt><code>short.json</code>, <code>medium.json</code>,
<code>long.json</code></dt>
<dd>The messages, the formatted prompt, its token IDs, the reference answer, and
the top 20 scores at each checked position.</dd>
<dt><code>short-0.f32</code>, <code>short-7.f32</code>, and so on</dt>
<dd>The full row of scores at one position, as little-endian FP32 numbers.</dd>
<dt><code>short-routing.json</code>, and so on</dt>
<dd>The experts the router (the small network that picks which experts each
token uses) chose, and their weights, at two positions.</dd>
<dt><code>short-layers.json</code>, and so on</dt>
<dd>A short summary of every layer's output at the same two positions.</dd>
<dt><code>short-bf16.json</code></dt>
<dd>The short case run in BF16, the makers' own precision.</dd>
<dt><code>short-bf16-0.f32</code>, <code>short-bf16-7.f32</code></dt>
<dd>Its score rows. Scores were promoted to FP32 only for storage.</dd>
</dl>

Scores, or logits, are the model's raw score for every possible next token.

| Case | Prompt tokens | Checked positions |
| --- | ---: | ---: |
| Short | 25 | 8 |
| Medium | 285 | 24 |
| Long | 2,081 | 24 |

</details>

<details>
<summary><b>How far the BF16 run is from the FP32 run</b></summary>

The BF16 short case picks the same 8 of 8 answer tokens as the FP32 run.
Its full score rows differ from the FP32 rows by at most:

| Answer token | Largest difference |
| --- | ---: |
| First | 0.66 |
| Last | 1.32 |

- The difference is the largest absolute score difference, BF16 against
  FP32.
- First: `short-bf16-0.f32` against `short-0.f32`. Last: `short-bf16-7.f32`
  against `short-7.f32`.

That gap is the makers' own precision difference, not a Vade result. The
fixture check allows up to 1.33 (`logits_within_1.33`). A pass supports only
these saved cases; current arithmetic may fail this historical comparison.

</details>

**Where next:** [Flash server](../../docs/flash-server.md) ·
[Validation](../../docs/validation.md) · [History](../../docs/history/README.md)

## Repeated fixtures with resident weights

`flash-bench` loads one distributed rank's weights once and runs the fixture list
in order for each repetition. Every fixture gets a new session, including KV,
recurrent state, PLE history and decode graphs, with the standalone fixture's
capacity. It requires enough `SEQPAR_MEMORY_BYTES` for resident weights and the
largest fixture; it rejects the streamed plan.

```sh
target/release/flash-bench "$PTX" "$MODEL" \
  fixtures/qwen3.8-flash-next-fp8/short-bf16.json,fixtures/qwen3.8-flash-next-fp8/medium.json \
  "$ADDR0" "$ADDR1" "$RANK" --sessions 2
```

Use the same `SEQPAR_CONTROL_SECRET`, `SEQPAR_RDMA` and precision settings as
`load-check --flash-session`. The default world is two ranks. With `--world N`
or `--path`, pass this rank's LISTEN and NEXT endpoints in the address positions;
for the default two-rank ring, both ranks receive ADDR0 and ADDR1 in rank order.
Use `--path` for a line without a closing link.

The original `flash_session` token, logit and memory lines remain. `flash_bench`
boundaries identify each session; summaries add prefill, mean/median decode time,
token IDs and TTFT measured from fresh-session allocation through its first
forward call (excluding weight loading). `FLASH_DUMP_LOGITS` writes separate
`session-00000`, `session-00001`, ... directories, each containing the same
`position-NNNNN-rank-R.f32` format as the standalone driver. A reference mismatch
is reported at the end after all sessions have run, with a nonzero exit status.
