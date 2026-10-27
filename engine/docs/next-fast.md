# Flash-Next execution options

`flash_next_options::Options` reads configuration once, before model allocation.
Every switch accepts only `0` (including the unset default) or `1`. Invalid or
non-Unicode values fail loading. HTTP startup passes that same typed value into
the selected model loader. Sessions and token execution never read these switches.

| Switch | Behavior |
|---|---|
| `VADE_FLASHNEXT_FAST_PROJ` | Reference adapter: native MXFP8 projection storage and dispatch |
| `VADE_FLASHNEXT_FAST_FRONTEND` | Reference adapter: device embedding/hash/token feed and mapped PLE gather |
| `VADE_FLASHNEXT_FP4_EP2` | Multi-rank expert placement; world-one serving refuses it |

Rows execution retains expanded BF16 projections and rejects FAST_PROJ or
FAST_FRONTEND, exactly as the integrated qualification baseline did. Tiled PLE
and wide expert prefill are unconditional; the FAST_PREFILL switch is removed.
Rows are the product service. The BATCHED_ROWS switch and former M=1 HTTP
adapter are deleted. The exact reference engine remains available to tests.
World-one Next MTP defaults to k=3; `mtp_depth` accepts 0..3 per request, and `--mtp` sets
the server default. Integration OFF gates explicitly use k=0.

The rejected world-one MOE, ROUTER, HC, GDN and function-cache experiments have
been removed. The TP/EP adapter still consumes the shared K16 routing/expert and
HC pre/up kernels directly; those kernels remain part of its arithmetic contract.
The exact single-row GPU reference remains available for every A/B gate.

GPU harness controls live in `crates/engine/tests/flashnext_gpu.rs` and
`crates/gpu/tests/next_prototypes.rs`, never in engine model execution. These
include GPU activation/model selection, prompt IDs, token/context/budget limits,
indexer selection, timing, expected tokens, logits and debug captures, CPU dump
tolerance, and the external prototype checker. They are test inputs, not serving
options. The prototype tests cover only kernels with live TP consumers.

For byte qualification compare both adapters against the recorded integration
captures from 9cb9727: the four plain/chat/code/reasoning prompts, 64 generated
IDs and decoded UTF-8 bytes each, without EOS truncation. Use native greedy
sampling (temperature=0, top_p=1, top_k=1, repetition_penalty=1,
presence_penalty=0, frequency_penalty=0), sparse QSA, MTP=0 and SPARK_PROFILE=0.
Measure synchronized C1/C8 decode separately from model loading and prefill.
Retain the exact one-Spark command and capture manifest with the measurement.

Pooled row decode always stages the sixteen packed PLE rows per live chat
through reusable pinned storage before graph replay. Padded rows remain zero.
Prompt tiles of at least 64 rows fuse BF16 input conversion into grouped
projection staging and build stable compact routes cooperatively. Expert
GEMMs retain the qualified prefill TC tiles. Explicit chunks may use up to
2048 rows. The PLE_STAGE and PREFILL_DENSE switches and their replaced
paths are removed; the exact reference adapter remains available.

Host CI builds real stamped PTX and every workspace host crate on Linux.
Device tests are ignored with `skipped: no GPU` reasons; run them explicitly
with `--ignored` and their documented GPU inputs. The `vade-tools ci-no-gpu` command creates
a CI-only link library whose device calls abort; never distribute it.
