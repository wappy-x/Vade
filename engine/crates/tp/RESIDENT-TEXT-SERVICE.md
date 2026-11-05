# Bounded text service

Normal startup keeps the checkpoint execution refusal. The default-off
`--diagnostic-unqualified-text` command-line option admits this adapter for
loopback HTTP diagnostics only. It selects no precision or zero-sign policy.
Health reports `qualification: unqualified` and the diagnostic option name.
Source qualification and product admission remain closed.

```sh
flash-serve RANK WORLD LISTEN NEXT PTX MODEL 4096 127.0.0.1:18080 0 --mtp 0 --diagnostic-unqualified-text
vade up --sparks 2 --context 4096 --output 64 --mtp 0 --diagnostic-unqualified-text
```

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

The controller sends each prompt group or decode token to all ranks.
Prompt groups contain at most four rows and execute layer by layer.
Independent projections share immutable weight reads across those rows.
Each row keeps its own ascending feature fold and rounding boundaries.
Convolution history, recurrent state, and latent caches advance in token
order. Expert outputs return to their original slots before the existing
ordered fold. The final mean, norm, and full head still run for every row.
The last successful group commits its completed state on both ranks.

Each decode token follows the scalar schedule. Each rank owns complete output
rows of each projection. The resident owner's data ring restores rows in token-ID or feature
order before the next operation. A separate ring carries session commands. It does not sum partial reductions. Causal
state and the other operations stay replicated. Each token uses the bounded
decoder's existing arithmetic and commits state after the full head succeeds.

The service admits one active request. It retains matrix rows and the
stored embedding table through the resident decoder owner across requests.
It does not retain prompt prefixes or execute draft heads. Its memory bound
includes replicated state, layer scratch, projection staging, and logits.
The CUDA allocator also enforces the smallest negotiated rank ceiling.
`SEQPAR_MEMORY_BYTES` can lower that ceiling. `FLASH_SERVE_OUTPUT_TOKENS`
sets the output cap. These are existing service controls.

Startup compares the rank metadata fingerprints and context. The fingerprint
covers safetensors header metadata and text artifacts. It is not a hash of
all weight payloads. Metadata reports this limit. A host pass does not prove
device exactness, generated answer quality, or speed.

The host test `scalar_service_generates_four_exact_tokens_across_row_parts_and_new_sessions`
checks complete logit bytes and generated-token feedback against the scalar
unsplit decoder. Its source is `tests/row_split.rs`. A device gate must still
compare repeated token IDs on the authorized pair before merge.
