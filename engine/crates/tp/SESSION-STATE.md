# Resident session storage

[`session_state`](src/session_state.rs) owns bytes for ordered decoder lanes.
It does not admit a model or define arithmetic. The existing execution
refusal remains in force.

## Admission and owners

`Layout` takes qualified byte descriptors for each state plane. Fixed planes
hold recurrent state and convolution history. Row planes hold latent rows,
raw index keys, raw index gates, and complete index pools. The row divisor
comes from the index pool geometry. The raw key and gate rows include the
incomplete tail. No tail is discarded at a chunk or batch boundary.

The caller must include the full physical ABI in these descriptors. For
example, KDA convolution history needs its full retained channel-major row.
Its dtype does not follow from the dtype of the recurrent state. This module
copies bytes without conversion. It cannot infer or qualify that ABI.

`physical_storage` derives rank-local extents from `StatePlan` and explicit
producer dtypes. It reserves the full KDA history row, including the oldest
slot. It requires FP32 recurrent state. Other persistent planes require an
explicit BF16 or FP32 type. Missing or unresolved types fail. The caller
still supplies additional ABI scratch and the packing adapters. This method
checks conditional storage capacity. It does not certify producer dtypes.

Each lane owns a committed bank, an append bank, and scratch. Each saved
boundary owns a separate bank. Scratch includes mHC streams within a token,
operator packing, launch tables, and status storage. mHC streams pass between
layers within a token. They are not recurrent state between tokens. Scratch
is cleared before an append. Its users must finish before the next append.

The layout rounds each allocation for the existing device vector API. The
budget charges the rounded extents, including empty planes. `Budget` checks
weights, runtime storage, reserve, active lanes, and saved banks together.
The caller must use the actual rank weight arena size and account for backend
and transport storage in `runtime`. A byte total does not prove memory is
available. Device allocation can still fail.
`DeviceStore::admit` also checks current resident bytes and the backend's
allocation limit. Other users must not spend this pool's reserved capacity.

`Pool` enforces lane and cache credits. A saved bank retains its credit until
it drops. A snapshot can outlive its original lane. Restore requires the
same pool instance. Equal sizes do not allow snapshots to cross checkpoint,
rank, or backend instances. The service must keep each pool bound to one
immutable checkpoint and rank.

## Append and completion

`begin_append` copies the committed bank to a private working bank. A
producer must write every used state plane before commit. `write` takes a
complete used plane and clears its unused capacity. `run` lets a device
adapter use the resident buffers. The adapter returns the indices of fully
written planes. It must also clear unused padding in those planes.

The callback may use the ordered compute queue and a supplied `Completion`
owner for other users. It must not retain addresses or create users outside
these owners. Before transport reads a compute result, the callback must
wait for the producer. At the end, `run` drains transport first and compute
last. This permits a completed gather to enqueue its final compute copy.
Both waits run when the producer returns an error or panics.

Only a successful completion permits commit. Commit swaps owned banks.
It does not copy over the committed bank. A failed operation, incomplete
commit, explicit rollback, or dropped append leaves that bank intact.

A completion error retires the pool. All later work fails. On drop, the pool
keeps uncertain allocations and their credits instead of freeing buffers
that may still have users. The enclosing service must then retire the device
context and transport. This is failure containment, not an in-place recovery
path. A producer error with successful drains needs no pool retirement.

Save copies into a separate owner. Restore copies into the append bank and
swaps only after completion. Truncate requires a saved boundary. Shortening
a token count cannot reconstruct recurrent state or an index pool tail.
Draft rollback must use these boundaries. This module does not enable a
prediction layer or drafting.

## Current integration limit

`Layout::logical_reference` adapts the existing
[`StatePlan`](../model/STATE-PLAN.md) for opaque byte tests. It uses that
plan's declared logical widths. It is not a physical checkpoint layout.
`Layout::checkpoint` checks the source cast contract and refuses allocation
while the physical contract remains incomplete. It has no readiness flag.

The Linux [`DeviceStore`](src/session_state/device.rs) calls existing safe
backend allocation, clear, copy, upload, read, and completion methods.
It adds no kernel or FFI boundary. Store operations use one ordered stream.
No float operation occurs in a state copy or readback.

The decoder driver must still supply qualified physical cache descriptors,
full physical scratch, and complete state-producing launches. The transport
driver must supply a completion owner for all borrowed buffers. The service
must supply checkpoint identity, rank binding, and actual weight/runtime
charges. No active serving path uses these owners yet.

## Validation

The [host tests](tests/session_state.rs) use queued copies and fault injection.
They cover capacity, slot credits, partial allocation cleanup, append,
save/restore, truncate, panic and error drains, and completion quarantine.
They compare opaque continuation bytes across interleaved lanes, chunk
boundaries, cache restore, and rejected appends. They do not compute token IDs.

The [device gate](src/bin/session-state-bytes.rs) uses synthetic opaque bytes.
It checks allocation, ordered copies, saved owners, rollback, and cleanup on
a CUDA device. Its output records its source revision and raw state files.
It also checks a rejected replacement after every state plane was staged.
It tests an invalid extent and an invalid plane index. Both must block commit
and preserve the prior context and state. A valid append must still work.
It does not execute checkpoint arithmetic, batched kernels, or transport.
A booked device run must compare its raw files across fresh runs and devices.

Full model proof still requires checkpoint-backed continuation state and
identical IDs for all required request modes. A storage kit pass does not
close that gate.

Back: [TP crate](README.md).
