# Checked host state storage

[`plan::state`](src/plan/state.rs) adds byte offsets and capacity checks to
the [deferred layer program](LAYER-PLAN.md). It does not change the program,
its operation order, or model admission. It does not select casts or device
arithmetic. The [cast contract](CAST-CONTRACT.md) states the dtype requirements.

## Build a plan

`StatePlan::new` takes a `TextDescription`, rank topology, token capacity,
and byte limit. It builds each existing `LayerPlan` at that capacity. It
uses the program's buffers and placements to compute local row-major byte
extents. It reads no weights and allocates no activation payload.

The element widths come from the program's `F32` and `U32` declarations.
Offsets use the alignment of `u32`. Each plane has a checked shape, byte
range, and used extent. Products, sums, offset ends, and owner totals must
fit `usize` and `isize`. Capacity must fit the description's position limit
and the program's position ABI. Zero token capacity is invalid. A plane
with an empty extent is valid, including convolution history with no past
tap or a pool plane before its first complete group.

The shapes below come from `Builder::kda` and `Builder::latent` in
[`layer.rs`](src/plan/layer.rs). Let `C` be token capacity, `H_r` local head
count, `H` global head count, and `S` the program's selected-slot capacity.

| Owner | Reserved shape | Used extent at context `t` |
| --- | --- | --- |
| KDA recurrence | `[H_r,head_dim,head_dim]` | Full plane, including at the empty initial boundary |
| KDA history | `[QKV,conv-1,H_r*head_dim]` | Full plane with the program's past-tap order |
| Normalized latent cache | `[C,kv_rank]` | Leading `t` rows |
| Raw normalized index keys | `[C,index_dim]` | Leading `t` rows |
| Raw index gate logits | `[C,index_dim]` | Leading `t` rows |
| Learned complete pools | `[C/pool,index_dim]` | Leading `t/pool` rows, with integer division |
| Temporary selected latent | `[S,kv_rank]` | Program scratch, never saved as session state |
| Temporary keys | `[S,H_r,key_dim]` | Program scratch through its declared last read |
| Temporary values | `[S,H_r,value_dim]` | Program scratch through its declared last read |
| Temporary packed KV expansion | `[S,H*(key_dim+value_dim)]` | Step scratch; the program declares no rank split for it |

`index_tail` returns read-only byte views of the incomplete key and gate
rows. Complete raw rows remain retained. These views add no storage owner.
The complete pool plane has its own nonoverlapping bytes. This plan does
not introduce a compact device indexer representation. The policy that
excludes a tail from selection still needs its key and gate history.

`ByteLayout::for_program` reads `LayerPlan::lifetimes`; it does not compute
another data-flow analysis. It reserves inputs, outputs, and declared step
scratch. Gathered outputs reserve their full shape. Step scratch reserves
the declared shape because it has no split field. The default offsets do
not reuse storage. The state plan adds a physical hold for `pool_keys`
through state copy-out at the call boundary. Its last arithmetic read is
earlier. This hold does not change the logical program or its returned
buffers. `check_offsets` checks alternative offsets against the
same inclusive last-use records. It rejects aliases when both buffers are
live, including a step's inputs, outputs, and scratch. It also rejects
unaligned offsets, short arenas, and address overflow.

`working_layout` separates the committed bank, append bank, and scratch
arena. Its overlap check rejects aliases between these owners. The formula
in `StatePlan::new` is `working_bytes = bank_bytes * 2 + max_layer_scratch`.
Each ordered lane needs that capacity. Layers may reuse the scratch arena
only after the prior layer completes. `check_owners` checks the aggregate
working storage for active lanes and banks for saved boundaries. The caller
must reserve that aggregate budget before it creates those owners.

Weights are outside this limit. Header and upload planning remains in
[`admission`](src/plan/admission.rs). A successful state plan is not complete
model memory admission.

## State transitions

`SessionState` supplies an optional host owner for opaque bytes. It does
not expose typed pointers or establish device base alignment. It does not
execute the model. All state starts at the empty boundary with zeroed
KDA state and history. Latent, raw index, and pool planes have no used rows.

| Action | Ownership and boundary rule |
| --- | --- |
| Append | `begin_append` copies the committed bank into a private bank. The mutable guard holds the lane. The caller must write every used state plane from the next token program. Each write must have the exact byte count. It clears unused capacity. `commit` publishes the bank and advances the context only when all planes are present. |
| Rollback or error | `rollback`, guard drop, or a failed incomplete commit drops the private bank. Committed state and context stay intact. A failed write does not change committed bytes. |
| Cache save | `save` checks a separate cache byte limit and copies the whole bank. A saved boundary owns its bytes and shares only immutable plan metadata. It contains no temporary selected rows, expanded K/V, or other scratch. |
| Restore | `restore` requires the same capacity, rank topology, and program. It makes a new owned copy before it replaces committed bytes. Allocation failure leaves the lane intact. The saved owner remains valid. |
| Truncate | `truncate` requires a saved boundary at or before the current context. It restores recurrence, history, latent rows, keys, gates, pools, and tail together. A length alone cannot recover KDA state. |

An append advances one token in causal order. Chunk boundaries do not
advance or rebuild state. Interleaved requests use separate owners. The
host API has no launch queue. A device owner must wait for ordered kernel
and transport completion before commit, reuse, restore, or release. Dropping
a host append does not cancel a device launch.

## Host checks

Run from `engine/` after the required kernel build:

```sh
RUSTFLAGS='-D warnings' cargo +1.99.0 test --offline --locked \
  -p engine-model --test state_plan -- --nocapture
```

The `STATE_BYTES` output is the source for synthetic capacity values. The
`STATE_LIFECYCLE` output records the continuation test. That test computes
small scalar host forwards. It compares state bytes and host outputs across
alone execution, interleaved lanes, chunks, cache restore, and rollback.
Other tests cover exact byte limits, local heads, pool and tail transitions,
short capacity, overflow, empty planes, and overlapping live storage. None
of these tests produces model token IDs or calls a GPU.

## Resident device owners

The [resident session owners](../tp/SESSION-STATE.md) now supply device banks,
scratch, saved owners, and ordered append, commit, rollback, save, and restore.
Their admission budget includes weights, runtime storage, active lanes, and saved banks.
Their completion interface retains owners through compute and transport drains.
The device store binds the existing safe backend allocation and copy calls.

A future device layer driver uses `spark_tp::session_state::Session` for resident state.
`engine_model::plan::state::SessionState` remains the host reference owner for tests.
The device driver must supply checked physical layouts and complete byte charges.
It must bind kernel producers and transport completion to the resident owners.
The logical host layout does not supply those physical contracts.

## Remaining device allocation work

- Bind the layer program to the resident device owners and admission budget.
  Check each physical ABI's alignment and include all scratch and staging charges.
- Add bounded pack and copy descriptors. The logical KDA past-tap history
  differs from the channel-major convolution ABI with its reserved slot.
  Its prepared record differs from the separate Q, K, V, decay, and beta
  device planes. This host plan does not size those extra ABI planes.
- Size and bind projection packing, MMA input planes, padding, status
  words, launch tables, and transport staging. The program only declares
  logical outputs and selected step scratch. It does not describe these
  physical ABI additions.
- Bind learned pooling to the retained raw histories and complete pool
  plane. A compact pool/tail design needs a separate checked byte plan and
  a kernel contract. No compaction is assumed here.
- Bind latent append, counted gather, local slot IDs, and token-major K/V
  strides. The conservative full expansion scratch can be narrowed only
  when the program declares the correct local split.
- Bind launch and transport completion to the resident completion interface.
  Keep every input, output, state bank, and scratch owner live until completion.
  Do not use a host guard's drop as proof of device completion.
- Qualify continuation state and token IDs for alone, batch, cache, chunks,
  supported tensor splits, and draft rollback. The appended prediction
  layer has no verified program here. Draft state is outside this plan.

The resident owners do not provide a layer driver or model admission.
Their synthetic storage tests do not prove model continuation or token IDs.
The existing primitive and full request gates remain open.
