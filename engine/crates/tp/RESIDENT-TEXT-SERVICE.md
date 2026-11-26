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
layer at that depth before admission counts the resident bytes. With `--mtp 3`
(the depth `vade up --diagnostic-unqualified-text` loads when no depth is named),
a greedy request that names no draft mode and allows more than `64` output
tokens drafts at depth `3` while its chat is the only one. Its first two passes are
plain steps, which time its step; it proposes from the third.
Each cycle the draft head proposes up to three tokens. The proposal ends early
when the draft head's own probability of the whole proposal falls under one
half. The check pass takes only the proposed tokens that are likely to pay, by
that probability and this request's own timed passes; when none would pay, the
cycle is a plain step. A request proposes a third token only while third tokens
have paid in its own cycles. It tries one again after 4 cycles, then after 8, 16
and 32 while they still do not pay
([`confidence.rs`](src/row_split/confidence.rs)). A request that names
`"draft":"strict"` proposes its whole depth in every cycle, and its proposal ends
early under a probability of `0.3`. It reads a table of kept shares that no
request without a named draft mode writes, so such a request cannot change its
decisions. While two or three chats
that draft are active together (requests that draft by default, or strict
requests of depth `2` or `3`), they draft together
([`pair.rs`](src/row_split/pair.rs)). Their check rows run in one shared pass.
A set of rows is admitted only when every chat of the group is expected to
keep enough words to cover the time the step takes longer than a plain step of
the same chats, by the draft head's probabilities and the group's own timed
steps and passes. That is a prediction, not a guarantee: the group counts each
chat's tokens against its plain pace on the wall clock and ends a drafting
after a chat has lost 1 token to it, so a drafting that goes wrong can cost a
chat a token or two before it ends. Chats whose text the draft head predicts
well draft together, and a chat beside one that it predicts badly decodes plain
steps (measured, not graded: the lowest chat of any load 0.56 percent under its
plain rate, run `draftm-pre10`). While a group does not draft, one
chat at a time makes a proposal that no row checks (a look). A chat is looked
at only in its answer text, 16 tokens or more after its think block ended.
A look passes when its words would pay and the tokens the chat then decodes
are those words; a look with no probability checks no word either and passes
only when its words come. A chat whose look failed waits for its next look as long
again as it has been in its answer text without one that passed; no other
chat looks while it waits, and it must then be right twice in a row. The
group drafts when every chat has passed, and it stops with the step in which
a chat's words do not pay, or when a chat has lost 1 token to it. The first
two passes of a drafting are slow and are not taken as its price; a row count
timed once is no price yet, and when its first two timed passes disagree by
more than a quarter the lower one is its price. No look is made and no drafting
starts while a chat has fewer than 64 tokens left. Looks are made only while no chat is more than 0.4 percent of its tokens
behind its plain pace, measured on the wall clock, so a group that never
drafts pays about that much (measured: 0.2 to 0.43 percent in requests of
256 tokens; a chat's first look costs more than the later ones). When a chat that does not draft is
active too, or more than three chats, every request
decodes plain steps from then on. With `--mtp 1` or `2`, without a predictor, with
`max_tokens` of `64` or less, or with sampling, such a request decodes plain
steps. `"draft":"plain"` or `"mtp_depth":0` turns drafting off. A request
also drafts when it sets `"draft":"strict"`, or when it sends `"mtp_depth"`
`1`, `2` or `3` without `"draft"`. Strict drafting requires greedy sampling,
and its `mtp_depth` must not exceed the loaded depth (`0` by default when
omitted). Proposed tokens are verified for one chat, or for two or three
drafting chats in one shared pass; with more chats, or on a request that does
not draft, decoding uses plain steps. Every kept token equals the token of
plain decoding.

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
Prompt prefix retention is on by default: unless `flash-serve` is started
with `--no-prompt-keep`, each rank enables session forking and the
prompt cache retains up to eight finished prompt states across requests within
the session pool, restoring only at the exact end of a kept prompt without
copying state words and evicting cached prompts before live requests wait. A
kept prompt stays until memory is needed for a live request or a newer kept
prompt, until eight newer
prompts have replaced it, or until no request has used it for 5 minutes (a
request that touches the cache checks the times at once, and the running
server at most once a second). Every request is looked up, and its prompt is
kept when it fits the cache's room (below), unless it sends `"cache": false`;
no cache field is needed. Callers that use the same
API key share kept prompts, and so do all callers that send no credential
when no key is set; a `cache_salt` of its own keeps a caller's prompts apart. With the cache on,
`/health` shows `"prefix_cache":{"enabled":true}` with no counts and
`"device_bytes":null`, and `/v1/events` leaves out the cache's event lines. A
reply does tell its own caller how much of its own prompt came from the cache:
`prompt_tokens_details.cached_tokens` in the usage block and, on a streamed
reply, the `X-Vade-Cached-Tokens` and `X-Vade-Estimated-First-Token-Ms`
headers. The cache's room is still one pool for all
callers, so a caller's own hits can change when another caller uses it
([what a scope keeps apart](../../docs/server.md)): where callers do not
trust each other, start every rank with `--no-prompt-keep`. A resume that a
rank refuses or cannot start drops that one kept prompt on both ranks, and
the request reads its prompt cold.
The default has a small cost on a first turn and a large gain on a later one.
Measured on one pair against the build before it, at context 16,384 with
`--mtp 3`: a decode step took 0.036 ms longer (0.09 percent of 41.13 ms), a
request took about 13 ms longer once, after its first token, and four first
turns of 2,048 tokens at once took 0.4 to 1.7 percent longer; a second turn
at 8,759 prompt tokens then gave its first token after 2.9 s in place of
36.9 s. The cache holds what fits under its byte cap, the session pool less
one full-size chat; the cap moves with the free memory at start. Measured on
one pair on 2026-11-15 with `--mtp 3`: the cap was 2,543,628,429 bytes at
context 16,384 and 2,666,771,392 bytes at context 32,768 (which `vade up`
starts); six kept prompts of 8,192 tokens held 2,208,055,296 bytes; and at
both contexts a second turn of 8,759 tokens started from its kept prompt and
gave its first token after 2.769 s in place of 34.5 to 36.1 s cold, with the
token IDs of a cold read. `--prompt-keep`
is still accepted and names the default; both names at once are refused.
`vade up` never passes `--prompt-keep`. It passes `--no-prompt-keep` to every
rank only for a cache size of 0: `vade up --cache-bytes 0`, a saved setup
with that 0, or `FLASH_SERVE_KV_BYTES=0` given to `vade up`. With no cache
size (a saved setup without one included) the service it starts keeps
prompts. A rank started by hand reads no cache size from its
environment: give it `--no-prompt-keep`. If `FLASH_SERVE_KV_BYTES=0` is set
and the rank keeps prompts, it prints one `TEXT PROMPT KEEP` line at start
that says so and names the option. A size above 0 does not set this
cache (its byte cap is the engine's), so `vade up` refuses it for this
service.
With `--no-prompt-keep` no session is forked, the cache holds zero bytes and
no request looks a prompt up. With the cache on but a byte cap of 0 (the pool
funds one full-size chat and nothing beside it) every request is still looked
up and nothing is kept. The start line of rank 0 ends with `cache=true`
or `cache=false`, then `kept_cap_bytes=N largest_kept_prompt_tokens=M`: the
cache's byte cap and the longest prompt whose planned charge fits under it:
the state planes at that length, their append room and the short-group output
the state can pin. No prompt exists at start; the cache books a real prompt by
the bytes its state holds on the device, so the real limit can differ from M.
When the cache is on and M is under 1,024 tokens, one `TEXT PROMPT KEEP`
line at start says that the cache keeps little or nothing at this context and
that a smaller `--context` gives it room. Its
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

Startup compares the rank metadata fingerprints, context, and the prompt
retention choice, so two ranks with different choices refuse each other. The
same holds for a rank of a build from before the default changed and a rank of
a build from after it. The fingerprint covers safetensors header metadata and text artifacts.
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
