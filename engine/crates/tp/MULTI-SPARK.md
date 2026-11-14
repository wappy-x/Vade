# Shared multi-Spark hooks

> **Design note, not built.** This page proposes one shared communication
> layer for the two multi-Spark paths. Today the launchers and protocols are
> still separate. [Multiple Sparks](../../docs/multi-spark.md) describes the
> supported path: dense Qwen3 serving with `tp-serve`.

This page is for developers planning that merge.

## The situation today

**Vade has two ways to use several Sparks, and their protocols differ and
cannot talk to each other.**

- **Tensor parallelism**, in [spark-tp](README.md), splits every layer's
  weights, so all Sparks work on every token.
- **Sequence-parallel prefill**, in [spark-seqpar](../seqpar/README.md),
  splits one long prompt, so each Spark reads one slice.

Each has its own launcher, link discovery and TCP/RDMA ring.

## The proposal

**When sequence parallelism is integrated further, move the shared pieces
below into one communication layer that uses only the Rust standard
library.**

- Keep tensor and sequence partitioning in their own model adapters.
- The transport owns only communication and the lifetime of registered
  buffers.
- The engine's per-layer hooks and sequence-prefill phases keep owning GPU
  work and arithmetic.

Then TP's ordered sums and sequence prefill's exported KV spans can use the
same communicator, without duplicating model weights or GPU buffers.

Terms: a **rank** is one Spark's process; a **rail** is one of the two
interfaces each QSFP port presents; **registered memory** is host memory the
network card is allowed to read and write directly.

## Pieces to share

**Seven pieces, from link discovery to accounting.**

<dl>
<dt>Link discovery and topology</dt>
<dd>One <code>Link</code> type and one discovery, for ring, path and a second
rail.</dd>
<dt>Launch and lifetime</dt>
<dd>One SSH supervisor for both launchers.</dd>
<dt>Handshake</dt>
<dd>One explicit run description that every neighbour checks.</dd>
<dt>Neighbour connections</dt>
<dd>Shared rendezvous, hello checks, barriers and deadlines.</dd>
<dt>Collective buffer contract</dt>
<dd>One bit-preserving gather by offset and range, plus broadcast and
control.</dd>
<dt>Registered memory and rails</dt>
<dd>Shared RDMA registration, deadlines, dual rails and TCP fallback.</dd>
<dt>Accounting</dt>
<dd>Shared byte, memory and time counters for communication.</dd>
</dl>

<details>
<summary><b>Each piece in full: existing code, and the boundary to keep</b></summary>

**Link discovery and topology.** TP and seqpar share `Link` and
`discovery::links`, which reads interface speed and IPv4 prefixes. Let the
launcher choose ordered neighbour pairs, ring or path, and an optional second
rail. Accept SSH aliases and remote folders as inputs.

**Launch and lifetime.** `tp-launch` and `seqpar-launch` both supervise SSH
input pipes. Share folder quoting, rank endpoints, nonce creation, stop-all
clean-up, the end-of-input watcher, the lease deadline and thread pinning.
Each program supplies its own rank program and arguments.

**Handshake.** Replace coupling through environment and arguments with one
explicit run description: protocol version, workload kind, rank count,
topology, nonce, model and PTX arguments, prompt, decode or partition
options, KV format and enabled transport features. Each neighbour checks the
expected rank and the same description before exchanging work. Rank-local
addresses and CPU pinning stay out. Sequence prefill today hashes only its
run ID and topology; it should add the actual prompt IDs and partition
settings, not just a token-file path.

**Neighbour connections.** Share bounded rendezvous, two-way hello checks,
blocking load and control barriers, and deadline-bounded collective I/O. Keep
paths, rings, odd rank counts and zero-length owner partitions working. After
a partial collective failure, the common layer must mark the communicator
unusable.

**Collective buffer contract.** TP's `gather_batch` and `gather_blocks`, and
sequence prefill's `gather`, all move rank-owned byte ranges into fixed
output positions. Share one bit-preserving gather by offset and range, plus
broadcast and control. Keep TP's fixed-rank FP32 sum, and sequence prefill's
token and KV ordering, above this transport contract.

**Registered memory and rails.** Both use `rdma::Endpoint` with pinned
buffers. Share registration, completion deadlines, dual-rail routing and the
optional small-message mailbox. GPU exchange hooks hand over stable host
spans after the stream finishes; deregister before resizing or freeing them.
Choose one transport for all ranks at start-up, and keep the explicit TCP
fallback.

**Accounting.** Share bytes sent and received, owned transport scratch,
registration and mailbox bytes, and communication time. Model adapters add
weights, KV, GPU scratch and process memory. No second benchmark framework is
needed.

</details>

**Where next:** [spark-tp](README.md) · [spark-seqpar](../seqpar/README.md) ·
[Multiple Sparks](../../docs/multi-spark.md)
