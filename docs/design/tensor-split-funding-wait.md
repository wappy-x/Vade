# Prepare the sequence split before the mailbox vote

Status: design only. No engine behavior changes with this note.

## Problem

The sequence split uses a GPU mailbox to agree that each rank has allocated
its state and work buffers. A ready rank can enter that wait while another
rank still allocates memory. The wait then includes peer preparation time.
This is a possible failure mode from the code. It is not proof of the cause
of a recorded device failure.

Code references below use base `3a0323ebf4c802a59c3764e4d3775400fb7ac9af`.
Local run evidence belongs in the job report, outside this document.

| Site | Current behavior |
| --- | --- |
| [next_xsplit.rs](../../engine/crates/engine/src/flash/next_xsplit.rs):10 | `funded` allocates KV pages, the row bank, and the workspace. |
| [next_xsplit.rs](../../engine/crates/engine/src/flash/next_xsplit.rs):37 | The front allocates before it posts the request. It then calls `tp_xsplit_agree`. |
| [next_xsplit.rs](../../engine/crates/engine/src/flash/next_xsplit.rs):78 | Each worker allocates before it calls `tp_xsplit_agree`. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):185 | Each rank puts its funding result, takes each peer result, and checks the shared error flag. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):199 | Each take has `rows=1`, `width=1`, and `half=1`. This is a control word, not a prompt tile. |
| [next_tp.rs](../../engine/crates/engine/src/flash/next_tp.rs):229 | `Swap.limit_ns` copies `pp::LINK_TIMEOUT`. |
| [next_pp.rs](../../engine/crates/engine/src/flash/next_pp.rs):161 | `LINK_TIMEOUT` is 10 seconds. Its pipeline use bounds silence between heartbeats. |
| [tp_mailbox.rs](../../engine/crates/gpu/src/tp_mailbox.rs):94 | The launch passes the limit and the receive sequence address to the kernel. |
| [tp_mailbox.rs](../../engine/crates/gpu/kernels/src/tp_mailbox.rs):89 | The GPU starts an elapsed-time clock for each take. A pipeline heartbeat does not extend it. |
| [tp_route.rs](../../engine/crates/gpu/src/tp_route.rs):149 | The relay spreads a shared error flag to its neighbors. The flag does not preserve the cause. |

The current error at `next_tp_split.rs:203` identifies the funding phase.
It does not identify the expired take. A peer failure can produce the same
error before the local take reaches its own time limit. A relay message
that says `mailbox failed before message` is also not a receive-wait record.
The counter can describe a local send or a slot reuse check.
See [tp_route.rs](../../engine/crates/gpu/src/tp_route.rs):172 and :225.

## Proposed fix

Replace the device funding vote with a host preparation barrier on the
existing pipeline control links. Keep the sequence-split plan and its
arithmetic unchanged. Apply this protocol to every topology that uses the
sequence split. Derive ranks and buffer sizes from the topology and plan.
Do not add a model-name check, an environment switch, or a second execution
path for the topology.

Use these states for each transaction:

1. **Prepare.** The front sends a transaction ID, prompt and split row counts,
   tile size, and session slot. A middle forwards before local allocation.
   Each rank allocates the same resources that `funded` allocates today.
   It also drains preparation work from its stream before it reports ready.
2. **Ready or refused.** Each rank reports its result and rank index over
   the control link. A middle combines the results for its subtree. Retain
   the prepared bank and workspace until commit or abort. A ready response
   must not depend on a GPU mailbox take.
3. **Commit.** The front sends commit only after all ranks are ready. A
   middle forwards commit before it starts its plan. Each rank then runs
   the existing plan once, with the prepared resources.
4. **Abort.** On refusal, discard prepared resources on every rank before
   returning the existing funding-refusal result. On an incomplete control
   exchange, stop the cohort through the existing failure path. Do not
   reuse a session whose peer state is unknown.

The host preparation phase needs a bounded progress wait. Use the existing
`LINK_TIMEOUT` as the maximum interval without preparation progress. Track
it per rank and phase. Report progress after KV allocation, bank allocation,
workspace allocation, and stream drain. An ordinary process heartbeat must
not restart this clock.
A phase that exceeds that interval fails with its rank and phase. This
policy can still reject slow preparation. It must give a clear error in
that case. It does not claim that all large prompts will succeed.

Start the progress clock after the ordered prepare send completes. Preserve
the current heartbeat clock for transport silence. Give abort priority over
a late ready response. Reject stale or duplicate commit messages by their
transaction ID. Keep the control reader able to receive an abort while
local preparation runs. Do not free resources still used by the stream.

Log `transaction`, `rank`, `phase`, `prompt_rows`, `split_rows`, `kv_rows`,
`bank_rows`, `elapsed_ms`, `limit_ms`, and `pending_ranks`. On failure, report
whether preparation stopped, the control link stopped, or a later mailbox
operation failed. Use rank indexes only. Do not log machine identifiers.
An expired progress wait names the rank whose progress is unconfirmed.
It does not by itself prove a hardware cause.

This proposal uses safe Rust and the existing backend methods. It adds no
kernel, FFI, dependency, or unsafe site. It does not retry a failed live
mailbox. The chosen fix is the host preparation barrier.

## Implementation sites

| Site | Required change |
| --- | --- |
| [next_xsplit.rs](../../engine/crates/engine/src/flash/next_xsplit.rs):10 | Expose preparation phases and retain the prepared resources. Preserve allocation errors as structured reasons. |
| [next_xsplit.rs](../../engine/crates/engine/src/flash/next_xsplit.rs):42 and :80 | Replace the device funding vote with prepare, ready, commit, and abort. Keep `xsplit_steps` at :100 unchanged. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):105 and :142 | Carry the transaction through the front request and worker dispatch. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):185 | Remove the funding put and takes. Do not advance `sent` or `got` for the host barrier. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):209 | Start the existing plan from the counters after the last completed transaction. All ranks must agree on that boundary. |
| [next_pp.rs](../../engine/crates/engine/src/flash/next_pp.rs):53 and :1170 | Change the protocol version and reject mixed peers during hello. |
| [next_pp.rs](../../engine/crates/engine/src/flash/next_pp.rs):1361 and :1373 | Validate the control messages. Preserve forwarding before local work. |
| [next_pp.rs](../../engine/crates/engine/src/flash/next_pp.rs):1212 and :1389 | Keep liveness separate from preparation progress. Route progress and abort through the ordered control owner. |
| [next_tp_split.rs](../../engine/crates/engine/src/flash/next_tp_split.rs):127 and [next_pp.rs](../../engine/crates/engine/src/flash/next_pp.rs):505 | Preserve cohort failure when control or mailbox state is uncertain. |

The stream wait limit for actual data remains unchanged. Scaling the
funding vote by payload bytes would not account for peer allocation time.
The vote carries a control word at `next_tp_split.rs:199`.

## Validation before implementation can merge

Add host protocol tests for slow preparation, each refusal phase, a lost
peer, stale ready, duplicate commit, and abort during allocation. Use a fake
clock to test the progress limit. Show that ordinary heartbeats cannot
extend it. Show that no rank starts a plan before all ranks are ready.
Show that the mailbox counters stay aligned when the funding vote is
removed. Test the current supported topology sizes and generic rank lists.

An authorized device gate must then compare token IDs with the base. Cover
cold prompts, cached continuations, a fresh cold request after retained
prefixes, batching, chunking, and drafting on and off. Use the repeated and
cross-topology checks required by [AGENTS.md](../../AGENTS.md). Record
preparation progress and every failure. Do not claim a device pass from
host tests. No device run is part of this design-only job.
