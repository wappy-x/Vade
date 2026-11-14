# Bounded text service

Normal startup keeps the checkpoint execution refusal. The default-off
`--diagnostic-unqualified-text` command-line option admits this adapter for
loopback HTTP diagnostics only. It selects no precision or zero-sign policy.
Health reports `qualification: unqualified` and the diagnostic option name.
Source qualification and product admission remain closed.

```sh
flash-serve RANK WORLD LISTEN NEXT PTX MODEL 4096 127.0.0.1:18080 0 --mtp 0 --diagnostic-unqualified-text
vade up --sparks 2 --context 32768 --output 8192 --mtp 0 --diagnostic-unqualified-text
```

`--mtp 0` loads no draft head. `--mtp 1`, `2` or `3` loads the predictor
layer at that depth before admission counts the resident bytes. With `--mtp 3`,
a greedy request that names no draft mode and allows more than `64` output
tokens drafts after a trial. It decodes plain steps up to token
`64` and makes no proposal before then. From token `64` on, the trial makes
three proposals of depth `3` and checks them against the plain tokens. When
the think block ends before token `64`, the trial first skips the `16` tokens
after it. `8` plain tokens lie between one proposal and the next. Only when
all three match in full does the request draft at depth `3`. A miss, or
another active chat, ends the trial, and the request decodes plain steps from
then on. A request that drafts stops for the rest of its answer when its
drafts keep too few tokens. With `--mtp 1` or `2`, without a predictor, with
`max_tokens` of `64` or less, or with sampling, such a request decodes plain
steps. `"draft":"plain"` or `"mtp_depth":0` turns drafting off. A request
also drafts when it sets `"draft":"strict"`, or when it sends `"mtp_depth"`
`1`, `2` or `3` without `"draft"`. Strict drafting requires greedy sampling,
and its `mtp_depth` must not exceed the loaded depth (`0` by default when
omitted). Proposed tokens are verified only while one chat runs; when more
chats are active, or on a request that does not draft, decoding uses plain
steps. Every kept token equals the token of plain decoding.

`vade up` records the option. A later bare `vade up` cannot silently keep it.
The diagnostic option must be repeated, or the diagnostic engine must be
stopped before a normal start. HTTP binds outside loopback are refused.

`flash-serve` selects the complete-row text service from the parsed checkpoint
schema. `Cartridge::resident_text` checks the operation metadata, retained
safetensors headers, and storage sidecar. The service checks the tokenizer,
chat template, and complete EOS set before it opens the listener.

Use the existing controller configuration with the checkpoint path on each
rank. The checkpoint metadata selects the actual weight recipes. The
diagnostic option changes admission only. It uses the same resident decoder
arithmetic as the token probe.

Run `vade chat` with the same controller configuration after startup.
The service also accepts `/v1/chat/completions`. Set `stream` to `true` for
SSE. A request without sampling fields is greedy. Explicit sampling uses the
shared request parser and sampler. The GLM renderer and incremental decoder
are the same text components as the checkpoint first-token probe.

The controller sends each prompt group, decode step, or verify pass to all ranks.
Startup negotiates the prompt group size across ranks as the largest of `1024`,
`512`, and `256` rows whose workspace fits the rank budget (`64` rows minimum
while chats share prefill), and groups execute layer by layer.
Independent projections share immutable weight reads across those rows.
Each row keeps its own ascending feature fold and rounding boundaries.
Convolution history, recurrent state, and latent caches advance in token
order. Expert outputs return to their original slots before the existing
ordered fold. The final mean and norm run for every row; served prompt groups
run the vocabulary head on the group's last row, while verify passes run it on
every new verified row. Each successful group commits its completed state on
both ranks.

A single plain decode chat runs the scalar schedule; plain decode chats run a
batched decode step (`2` chats when two sessions are open, or `3` to `8` chats
when three or more sessions are open) that shares weight reads on history-free
steps and runs history steps per chat with bit-equal results. Single-chat
decode steps and two-chat batch steps keep routed expert picks on the device
per chat; batch steps of `3` to `8` chats stay on the multi-row device route
path when the first routed layer has fewer than two picks per distinct expert,
and otherwise use the grouped expert path. Each rank owns complete output rows
of every splittable
projection except the KDA low-rank gate inputs (`f_a` and `g_a`), which stay
whole on each rank. The resident owner's device mailbox and data ring restore
gathered rows in token-ID, head, or feature order before the next operation. A
separate control ring carries session commands, and step barriers ride the
mailbox. It does not sum partial reductions across ranks. State latents, pool
planes, and indexer caches stay replicated, while each rank owns its local KDA
recurrent heads and local latent-attention expanded KV heads. Each step uses
the bounded decoder's existing arithmetic and commits state after the validated
head succeeds.

One-token greedy steps (plain decode and one-row strict verify passes with no
carried rows) reduce each resident head part on the device.
The reduction keeps its value, local ID, and validity word. The ranks exchange
these results instead of the full logits. The host checks every validity word,
adds each part's global row origin, and selects the largest value. Ties select
the lower global ID. Batched plain greedy steps gather the head on the device
and reduce each row there without reading full logits to the host. Prompt,
sampled, and multi-row or carried-row strict verification outputs retain their
full-logit contract. No extra decode step is queued.

`resident-decode-check argmax NEW_OUT` checks the compact reduction with finite
extremes, signed zeros, subnormals, and nonfinite losing parts. The pair probe's
`--check-greedy-head` option compares compact IDs with full logits and advances
the compact state. It retains all raw output files. This option runs both
heads, so its timing is diagnostic only.

The service admits up to eight active requests within the negotiated memory
budget. Each command names its session, or lists the participating sessions in
a batched decode step. Each rank retains that session's state. Only one
synchronous execution—a prompt group, single-chat step, batched decode step, or
strict verify or propose pass—runs at a time. Session creation and removal
complete on all ranks before the controller continues.

The service retains matrix rows and the stored embedding table through the
resident decoder owner across requests.
Prompt prefix retention is off by default and experimental: when `flash-serve`
is started with `--prompt-keep`, each rank enables session forking and the
prompt cache retains up to eight finished prompt states across requests within
the session pool, restoring only at the exact end of a kept prompt without
copying state words and evicting cached prompts before live requests wait. Its
memory bound includes session state, local-head expanded rows, cache append
slack, shared layer and prompt-group scratch, the projection memo, optional
weight staging, and logits (plus predictor state and trail room when `--mtp`
`1..3` is loaded). Shared synchronous scratch is reserved once. Each request
reserves separate committed and unpublished state for its prompt plus requested
output. The smallest remaining session budget across ranks controls HTTP
admission. Startup prints the actual resident allocation, shared workspace,
session pool and full-context session bound. It refuses a context that cannot
fit one full-context request. The requested context must also fit checkpoint
metadata. The default output cap is the smaller of the context and 8192.
Requests stop at checkpoint EOS tokens unless they explicitly ignore EOS.
This diagnostic adapter allows a generation deadline of 24 hours. The shared
server default stays one hour. Queue wait has the same bound, and admission
starts a fresh generation deadline. The deadline remains a separate bound
from the token cap; an unfinished request can still time out.
The CUDA allocator also enforces the smallest negotiated rank ceiling.
`SEQPAR_MEMORY_BYTES` can lower that ceiling. `FLASH_SERVE_OUTPUT_TOKENS`
sets the output cap. These are existing service controls.

Startup compares the rank metadata fingerprints, context, and `--prompt-keep`
setting. The fingerprint covers safetensors header metadata and text artifacts.
It is not a hash of all weight payloads. Metadata reports this limit. A host
pass does not prove device exactness, generated answer quality, or speed.

The host test `scalar_service_generates_four_exact_tokens_across_row_parts_and_new_sessions`
checks complete logit bytes and generated-token feedback against the scalar
unsplit decoder. `interleaved_two_four_eight_sessions_match_alone_full_logits_and_ids`
compares 64 generated IDs and complete logits for each chat alone and with
two, four and eight simultaneous sessions. Its source is `tests/row_split.rs`.
A device gate must still
compare repeated token IDs on the authorized pair before merge.

The diagnostic service currently requires exactly two Sparks. It refuses other
counts at mailbox admission with `diagnostic text service requires exactly 2 Sparks`.
Each admitted chat funds its full output allowance, including an omitted
`max_tokens`. Chats that do not fit wait at admission. Eight short chats fit;
eight chats with 123 prompt tokens and an 8192 output allowance do not all fit
at once. A measured 7,690,818,504-byte pool funds six; other budgets can
fund a different count. Nonfitting chats wait. Long prompts need separate
admission. The context and output caps are not a proof of full-length execution.
A permanent engine failure is sticky. Health reports the stored reason with
status `poisoned` before the process exits. It refuses subsequent work.
