# Host conductor ownership

The one-row resident mHC boundary can use a local mailbox and a CPU service.
The checkpoint plan chooses the mailbox extent. No model name selects it.
Rows with more than one token keep the existing device boundary.

## Lifetime

`host_conductor::Conductor` owns the mapped mailbox and its job queue.
The caller creates this owner before the exchange thread scope. Both bounded
decoder callers start the host worker in that scope. The device holds a
borrowed `MhcHostLink`, so it cannot outlive the mailbox owner.

The caller drains the device stream before it stops and joins the worker.
`Worker::finish` reports a service error. Its drop guard also stops and joins
the worker on an early return. The mapped storage can drop only after the
scope has joined every borrower. A service error or unwind sets the mailbox
error word and fails the queue. The resident exchange error check also checks
this local mailbox. Failed output is not an accepted request result.

The residual and coefficient planes are CPU-owned vectors. Each completed
run stores them under its receive ticket. A delivery or a later run names
that ticket, not a boundary address. A queued release follows every earlier
user of those planes. The service can reuse a released allocation. It cannot
reuse a live generation. The immutable feature-major weights have an `Arc`
owner in each queued run and each resident boundary.

Only mailbox payloads need mapped memory at this stage. The CPU-only residual
and weight vectors do not expose raw mapped pointer views. Materialization
delivers the residual and coefficients through the same checked mailbox.

## Publication and slot reuse

One ordered device stream sends inputs. One host service reads them. The
service first acquires the exact send sequence. It then reads the length and
payload words with relaxed atomic loads. It checks the expected byte count,
the slot bound and the sequence again before it uses the input.

The service writes response words with relaxed atomic stores. It retains the
completed planes before it releases the receive sequence. The existing take
kernel acquires that sequence before it copies any payload. Its existing
system-scope publication boundary is unchanged.

The next input put follows the preceding take in the ordered stream. It
therefore proves that the prior response was consumed. Delivery-only jobs
have no put. The service limits their receive publications to the available
slots between puts. It refuses sequence exhaustion instead of wrapping.

## Core placement and arithmetic

The service first narrows itself to the fastest cores in its allowed mask.
It creates the helper pool before it pins itself to one core. Each helper
chooses a separate core index from that inherited mask. This changes only
the calling thread's affinity. It does not change a host policy.

When a run is known, each folding core reads a word from each weight cache
line before the service waits for the input. This prepares that core's cache;
it does not guarantee that the weights remain in L2. Idle workers park and
queue publication wakes them. Cache preparation changes no floating value.
After it publishes an answer, the service also touches the second future run
when that job is already queued. It does this after publication, not before
the current answer. Delivery and release jobs do not change this run count.

Every serial sum starts at `0.0` and visits sources or features in the device
order. Products and adds remain separate. RMS division and square root use
the same FP32 expression order. Norm trees retain their fixed lane order.
Helpers divide independent rows or features, never parts of one reduction.
Host tests compare raw plane bits. Device exactness remains a separate gate.

`mhc_host_parameter_bytes` reports the planned feature-major host allocations,
including padded rows and vector headers. It does not change the existing
service memory admission arithmetic. The mapped mailbox remains part of the
backend's allocation accounting.

## Nonfinite inputs

A valid run never puts NaN or infinity in a boundary input, and no golden holds one. The host
lane is deterministic: the same request on the same setup gives the same bytes. With a
nonfinite input the same words are nonfinite as on the kernel path, and every finite word is
equal, but NaN payload bits can differ from the kernel path (a kernel writes its canonical
NaN; a CPU core keeps the operand's payload). The existing nonfinite checks refuse such a
request exactly as before, because the host lane's put does not check its payload and the
values reach the same later checks. Owner ruling 2026-11-09: payload bits need not match.

## Recorder kinds

On a recording queue the host lane's three launches are recorder kinds, so a boundary stays
inside one graph piece: `PutPlain` (tp_mailbox_put), `ShadowPrefetch` (chain_shadow_prefetch)
and the existing `Take`. `conductor-net MODEL 90 5 2 2 chain-recorded` runs the 90-boundary
chain through the recorder and prints the kind counts.
