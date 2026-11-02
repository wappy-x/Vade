# Sampling

This page tells you how Vade picks each next token, which request fields
change that choice, and what stays exact when you sample. It covers FP4-Local,
the default model. For the wire format of each field, see the
[OpenAI-compatible API](openai-api.md) and the
[engine server reference](../engine/docs/server.md).

## Greedy is the default

When a request sends no sampling field, Vade decodes greedily: it picks the
token with the highest score at each step. The card, the matrix and every
exactness check use greedy requests. A greedy request gives the same token IDs
on every run, alone or batched, cached or chunked, with drafting on or off,
and on one Spark or more.

When a request sends any one sampling field, the engine takes every other
field from the checkpoint's `generation_config.json`. For FP4-Local that file
asks for temperature 1, top-k 20 and top-p 0.95, so a request that sends only
`seed` gets a sampled reply. For a greedy reply, send no sampling field, or
send `"temperature": 0` with neutral penalties (`repetition_penalty` 1,
`presence_penalty` 0 and `frequency_penalty` 0).

## Fields

| Field | Values | Off value | What it does |
| --- | --- | --- | --- |
| `temperature` | 0 or more | 0 (greedy) | Divides the scores before the draw. A higher value gives a flatter choice. |
| `top_k` | A whole number | 0 | Keeps the `top_k` highest scores. |
| `top_p` | Above 0, at most 1 | 1 | Keeps the smallest set of top tokens whose probability sum reaches `top_p`. |
| `min_p` | 0 to 1 | 0 | Keeps the tokens whose probability is at least `min_p` times the best token's probability. |
| `seed` | An unsigned 64-bit number | None | Selects the random draws. See [Seeds](#seeds). |
| `repetition_penalty` | Above 0 | 1 | Lowers the score of every token already in the prompt or the reply. |
| `presence_penalty` | -2 to 2 | 0 | Subtracts this value from each token already in the reply. |
| `frequency_penalty` | -2 to 2 | 0 | Subtracts this value once for each time a token is in the reply. |

`typical_p`, `logit_bias`, `best_of` and `n` above 1 are not supported. The
engine ignores `typical_p`, `logit_bias` and `best_of`. Chat and completions
refuse `n` above 1 with the error `only n=1 is supported`. Responses and
Messages ignore the value of `n` and give one reply. Each of these fields
still counts as a sampling field, so it brings in the checkpoint's settings.

The four request surfaces (chat, completions, Responses and Messages) accept
the same fields in the table above. The gateways forward them unchanged.

## Order of operations

For each step, the engine applies these steps in this order
([`sample.rs`](../engine/crates/engine/src/sample.rs)):

1. It applies the penalties to the scores. `repetition_penalty` reads the
   prompt and the reply; the other two read the reply only.
2. It finds the best score. At temperature 0 it returns that token. A tie
   goes to the lower token ID.
3. It keeps the `top_k` highest scores.
4. It turns each kept score into a weight, `exp((score - best) / temperature)`.
   The best token has the weight 1.
5. It keeps the `top_p` set.
6. It keeps the tokens whose weight is at least `min_p`. Because the best
   token has the weight 1, this is the same as the probability rule in the
   table, and the best token always stays.
7. It draws one token from the kept weights.

Each sum and the draw run in one fixed order. When the request sets `top_k`,
the kept tokens run from the highest weight down; equal weights go to the
lower token ID first. With `top_k` 0, the sum for the `top_p` limit runs in
token-ID order. The `top_p` set then runs from the highest weight down, as far
as its sum reaches the limit. With `top_k` 0 and `top_p` 1, the draw runs in
token-ID order.
Thus the draw does not depend on how a sort or a partition left the tokens
([`sample.rs`](../engine/crates/engine/src/sample.rs), `nucleus`).

This is the order that the Hugging Face `transformers` library uses for
temperature, top-k, top-p and min-p. The `min_p` step compares each weight
with a fixed number. It adds no sum, so its result does not depend on the
order of the tokens.

## Seeds

Each draw is a function of the seed and the token's absolute position in the
conversation, and of nothing else
([`sample.rs`](../engine/crates/engine/src/sample.rs), `uniform`). The
generator is the SplitMix64 finalizer, written in the repository with the
standard library only. It keeps no state between draws. Thus a retry, a
replay or a rejected draft cannot move a later draw.

If you send `seed`, the engine uses it. If you do not, the engine picks a
random seed for the request. Every reply reports the seed it used in
`vade.sampling.seed`, and a streamed reply also sends it in the
`X-Sampling-Seed` header. To repeat a sampled reply, send the same request
again with that seed.

## Where sampling runs

Sampling runs on the host of rank 0, the Spark that serves HTTP, from the full
scores of the output head.

- **One Spark.** The one engine process samples.
- **Two Sparks (the tensor split).** For a greedy step, the two ranks split
  the output head by vocabulary halves and rank 0 merges the two best scores.
  A step that samples does not split the head: rank 0 runs the whole head and
  samples from its full scores
  ([`head_split.rs`](../engine/crates/engine/src/flash/head_split.rs)). Rank 1
  never samples. It gets the chosen token as the input of the next step, so
  the two ranks always agree on the token.
- **Four Sparks (the tensor split in parts).** As on two Sparks: a greedy step
  splits the output head into four vocabulary parts, and a step that samples
  runs the whole head on rank 0, which samples. The other ranks never sample.
  On two and on four Sparks, a greedy prompt pass and a greedy verify pass of
  more than one chat or more than 8 rows also run the whole head on rank 0.
- **The chain (three or more Sparks without the split).** The front rank keeps
  the output head and samples.
- **Four Sparks with `--tensor-map`.** This older rank path accepts greedy
  requests only. It refuses a sampled request before admission
  ([next-http](../engine/docs/next-http.md)).

## Drafting

By default, a sampled request does not draft. The engine decodes it one token
at a time, and the reply's `vade.draft` is `plain`. Greedy requests still
draft.

The test switch `VADE_MTP_SAMPLED=1` keeps drafting on for sampled requests
on one chat. The model then checks each drafted row with the same rule as an
undrafted step: it draws its own token from that row's scores with the same
seed and position, and it keeps the drafts only up to the first one that is
different. The token IDs are the same as with drafting off; the drafts only
change how many rows each step checks
([`mtp_batch.rs`](../engine/crates/serve/src/mtp_batch.rs)).

The classic speculative-sampling rule accepts a draft with a probability
ratio. It gives the right distribution, but not the same token IDs as an
undrafted run, so Vade does not use it. The `"draft":"sampled"` mode also
checks each row with the same draw as an undrafted step; only its proposals
come from the draft head's own draw
([speculative decoding](../engine/docs/speculative-decoding.md)).

## What is exact

For the same request, settings and seed, a sampled request gives the same
token IDs on every run with the same Spark count. These conditions hold on
the host and are covered by the engine's host tests:

- The same seed draws the same IDs twice, in a new sampler or in a sampler
  that served another request.
- A position draws the same ID when the engine samples it again out of order.
- Top-k, top-p and min-p keep the expected tokens on known scores, and the
  compact top-k path draws the same IDs as the full-score path.
- On random scores and settings, the engine draws the same IDs as a plain
  reference that applies the orders above with full sorts and no partition.
- `"draft":"sampled"` commits the same IDs as drafting off, with any
  proposals and when another chat holds the drafter.
- A request with no sampling field gets the greedy settings, the same reply
  echo bytes and the same IDs as before `min_p` existed.

On a Spark, the IDs also depend on the GPU giving the same scores on every
run. The engine's exactness rules require that: no floating-point atomics and
no reduction order that depends on timing. To check it on your Sparks, send
the same seeded request twice to `/v1/completions` with `"stream": true`.
Each streamed chunk carries its token's `token_id`, so you can compare the two
runs ID for ID. `vade verify` does the same for greedy requests only.

The rules also ask for the same IDs across Spark counts. For a sampled
request, that holds only when one Spark and the tensor split give
bit-identical scores, not only the same best token. Check it with a seeded
request on both setups before you depend on it.

A sampled request and the same request with another seed, or with greedy
settings, are different requests. Their IDs are not expected to match.
