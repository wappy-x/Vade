# Tensor ownership and copy transport

This is a host layout contract. It does not admit a deferred model for
execution. The current Flash-Next serving path does not change.

The logical layer source at commit
`1b2470df84f9c199a7fabde2e12605c427a28e6b` defines complete head and output-row
intervals. It requires gathers before projections and replicated folds.
The layer program is present in this tree. This module supplies its checked
layout inputs. The [TP adapter](../tp/LAYER-TRANSPORT.md) binds its declared
gather boundaries. Cast rules and session state remain separate contracts.

## Ownership

[`plan::tensor_split`](src/plan/tensor_split.rs) accepts the global shape,
split axis, complete unit size, and rank intervals. `Ownership::checked`
rejects gaps, overlaps, reversed or empty intervals, unequal parts, incomplete
units, and owners outside rank order. `Ownership::equal` constructs the same
equal intervals as the logical plan. Products use checked integer arithmetic.
The rank limit comes from `shapes::MAX_SPARKS`.

The caller passes the original global shape. A head unit includes all of its
channels. An MLA KV-up unit includes the adjacent key and value rows of that
head. Token and selected-expert axes can precede the split axis.

`ProjectionRows` permits only an output-row split of a matrix. Each row keeps
the full original input K. Its scale grid keeps the original tensor shape,
block dimensions, encoding, and global row origin. `scale_index` addresses
that full grid. A slice that starts inside an FP8 scale block must keep that
shared block and the original row coordinate. A backend that can only rebase
a local scale grid must refuse it.

MXFP8 and NVFP4 scale groups come from the storage format. FP8 block dimensions
come from the storage recipe. Compressed scale pages are never sliced with
a row pointer. Decode the checked full scale plane with the existing loader,
or supply a future backend that retains all original page coordinates.
Global multipliers remain attached to their original projection. These row
descriptors neither decode nor requantize weights. They do not replace
safetensors header and component validation.

## Wire order

[`spark_tp::tensor_layout`](../tp/src/tensor_layout.rs) converts checked
ownership into a rank-major byte wire. Each rank packet contains all of its
outer rows in local row-major order. The offsets cover the wire exactly.
The caller supplies a byte limit before a gather can allocate storage.
This limit is per buffer. It is not aggregate model memory admission.

`GatherLayout::gather` calls the existing `Exchange::gather_bytes` interface.
It then copies each packet to the original global coordinates. It does not
use a floating-point conversion, addition, or reduction. A failed exchange
leaves the destination unchanged. `gather_ring` also checks the concrete
host ring's rank count and uses its current copy collective.

`gather_projection_input` checks that the gathered tensor covers the
projection's full K before communication. The device layer driver must call
this gather at the original program boundary. Each output-row projection
must visit the same complete K in the same order on every topology. A rank
must not compute a partial dot product for another rank to add.

The word mailbox binding describes the same outer rows, global row stride,
owned span, and destination origin. It rejects boundaries inside a word,
oversize packets, and a packet that cannot fit the mailbox length field.
Packed BF16 can use this copy only when each boundary covers whole words.
No padding is added to make an interval fit.

`mailbox_channels` checks the current route header as well. The
`device_route_limits_are_separate_from_logical_topology` test records which
fixture counts fit it. Its larger-count refusal must not select a layer
pipeline or another arithmetic path.

On Linux, `MailboxRows::put` and `take` bind these arguments to the existing
checked backend copy calls. They check the actual mailbox topology, slot
capacity, vector extent, sequence, receiver, and timeout. They retain the
backend's context-ownership checks. A take describes its origin rank and
uses that origin's receive channel. These methods describe individual
copies. The [pair adapter](../tp/LAYER-TRANSPORT.md) supplies group agreement,
persistent sequences, ordered calls, completion, and failure handling on the
existing pair wire. The device operator driver must call those boundaries.

## Tests and remaining proof

[`tensor_ownership`](tests/tensor_ownership.rs) checks interval errors,
products, complete units, original scale origins, and the saved description's
topologies. It also requires continued execution refusal.
[`tensor_layout`](../tp/tests/tensor_layout.rs) checks byte extents, arbitrary
split axes, global ordering, raw bit patterns, full projection inputs,
mailbox arguments, route limits, and repeated local TCP ring/path gathers.
These are CPU tests. They are not inter-Spark or device execution proof.

[`tensor_mailbox`](../tp/tests/tensor_mailbox.rs) is an ignored device copy
test. It compares put packets and gathered destination bytes with fixed
source bytes. A host relay injects packets into the existing mapped receive
channels on a single device. It does not test RDMA, a distributed service,
projection arithmetic, a model stack, or token IDs. Run its prebuilt binary
only after the station has booked the device.

The following model bindings remain open:

- A complete device operator driver that calls the pair adapter at every
  declared boundary and uses checked row views for loaded weights.
- Header-backed device weight views that retain original scale grids and
  global multipliers. Unaligned origins need a qualified backend ABI.
- Physical activation and cache dtypes, projection packing, and complete
  cast and fixed-reduction contracts.
- Aggregate device allocation, session state ownership, operator completion,
  rollback, and cache restore. Pair copy completion is already bound.
- Qualification and storage for larger-count mailbox groups. The pair has
  setup agreement, fixed message order, and sticky failure handling.
- Embedding, final mean/norm/head, request service, and the draft wrapper.

The prebuilt `layer-transport-check` gate tests synthetic bytes across the
actual pair wire. It does not execute the decoder. Its station result is
separate from host checks and complete model proof.

After the remaining bindings exist, compare intermediate bytes and repeated token
IDs across supported Spark counts. Include alone, batch, cache, chunks, and
drafting modes. Host layout tests cannot establish that device gate.
