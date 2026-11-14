# Recorded completions responses

[The JSON fixture](completions-engine-2026-10-23.json) contains actual direct
engine responses recorded on 2026-10-23 at 06:34 UTC. It is not a model-output
simulation. The replay tests run entirely on the host and do not execute the
engine or establish numerical correctness beyond the recorded live checks.

Provenance:

- Engine: `73a05e1730804939cdb71a9f9e88a51e3bff54f3`, clean archive build.
- Live core gateway: `8d400add666e8774486398500d54975bb0df26d8`.
- One Spark; model `qwen3.8-flash-next-fp4-local`.
- Checkpoint manifest SHA-256: `d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`.
- PTX SHA-256: `9601dfffd21a6c8c6bff45a8eba0699898e4637d0a5b748368e7a0dbbf880765`.
- Weight-header, tokenizer and template fingerprints, exact UTC date, runner
  digest, source result digest and response digests are retained in the fixture.

Both cases ask the same short plain-text completion and cap output at 32 tokens,
with `cache:false`. The no-field case uses greedy selection; its direct request
may echo a different unused seed from the gateway's fixed seed. The named-field
case supplies only `seed:42` and retains the engine's checkpoint sampling settings
(temperature 1, top-k 20, top-p 0.95). The fixture preserves those actual echoes.

Each case produced 32 visible token IDs. The direct repeat, core versioned route
and core `/completions` alias matched the direct baseline in all six comparisons.
The two non-streaming answer projections also matched. The cell completed in
12.317 seconds. These are two live cases, not a replacement for the historical
27/27 client answer matrix or a general engine exactness gate.

Host replay covers both aliases, both response formats, and core, standalone and
opt-in standalone gateways: 24 cases. Six additional cases prefix the recorded
stream with deliberately opaque UTF-8/multiline events and split the entire
response into one-byte HTTP chunks. Only those extension events are synthetic;
the engine stream remains unchanged. Request bytes, status, end-to-end headers
and the complete recorded body must survive forwarding. The current standalone
alias mapping has host coverage; this recording's live gateway was the core.

From `harness/`:

```sh
cargo test --offline --lib recorded_engine_completions
cargo test --offline --lib opaque_completions_sse
```
