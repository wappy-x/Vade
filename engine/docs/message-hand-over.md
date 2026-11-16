# The hand-over of a message between two ranks in a decode step

A one-row decode step of the two-rank resident decoder exchanges 245 messages each way. This note says what one
message costs, where the time went on 2026-11-16, and which settings follow from it. Numbers are from nsys traces of
a served one-chat plain step on one pair (the two traces are aligned by the least put-to-take latency of each step).
Every number is MEASURED in the lab run named beside it: `attn-main-trace-01` is the traced run of main `57f0869746`
(246 steps), `msg-c-fi8-1` an untraced run of this change with 8 side blocks (4 sessions), `msg-g-dg-1` the traced
run of a diagnostic build that changes the form of the side prefetch every 1.07 s (24 steps a form).

## How a message moves

1. The put kernel (`put_join`) copies the rows into the send slot of the mailbox (host memory that the device maps)
   and publishes a sequence word.
2. One host thread a rank (the write doorbell, `tp_mailbox_host::TpMailboxHost::ring`) spins on that word and posts
   two chained RDMA writes: the rows into the peer's receive slot, then the sequence word.
3. The take kernel (`take_scatter`) on the peer polls the sequence word and then copies the rows to their places.
4. In front of 190 of the 245 takes sits a side-stream launch (`shadow_side`, `bus_shadow`) that prefetches up to
   6 MB of the next weights while the take waits.

## What one message cost

| Stage | Microseconds | Run |
|---|---|---|
| put kernel | 3.8 median; 9.2 for the 73,728-byte message | `attn-main-trace-01` |
| from the sender's publish to the end of a waiting take, 4-word message | 5.4 | `attn-main-trace-01` |
| the same, 8 KB message, no prefetch launch beside the take | 6.5 | `attn-main-trace-01` |
| the same, 8 KB message, prefetch launch (8 blocks) still running | 13.6 | `attn-main-trace-01` |
| take kernel whose rows are already there | 3.7 (8 KB) | `attn-main-trace-01` |
| one side prefetch launch with 8 blocks; with 2 blocks | 28.7 median; 36.7 | `msg-g-dg-1` |
| one traced step with the side prefetch as on main (8 blocks); with none | 41.67 median; 43.01 | `msg-g-dg-1` |
| from the return of a message's post to the reap of its write completion (`SERVED WIRE`), mean | 5.4 up to 16 KB; 10.2 above | `msg-c-fi8-1` |

The count of messages is not the cost and neither is the doorbell thread (it keeps its core). The largest single
line was the engine's own prefetch: each of its threads holds one 128-byte line request in the memory system, and
every read of a take that runs beside it waits behind them. The same holds for the small launches that follow the
take (the mHC boundary lasted 30 us beside the prefetch against 15 us alone, medians of `attn-main-trace-01`).
The prefetch itself pays: without it a step is 1.35 ms longer (the weight launches last 4.3 ms longer, the takes and
the small launches 2.0 ms shorter). And it is bound by the memory system, not by its thread count: a quarter of the
threads moves the same lines in 1.3 times the time. So fewer blocks cost little prefetch and shorten the line in
front of a take's reads.

## What follows from it

- `bus_shadow::SIDE_BLOCKS` (2): a side prefetch launch has 2 blocks of 256 threads, not the 8 of a take's tail rows.
- A waiting take reads as little as it can after its message arrived. The scatter take reads its row of the copy table
  before the wait. The three waiting takes read the sticky error word on the first miss and then on every 16th
  (`tp_mailbox::ERROR_EVERY`), not on every miss: a failed peer is seen at most 16 polls later, and the timeout is
  still checked on every miss.
- The sequence word of a message travels inline in its work request (`tp_rdma`), so the NIC does not read 4 bytes
  from host memory before it sends them.
- `SERVED WIRE` (print only, at the end of a session): messages of this rank since the load and their mean and least
  time from the return of the post to the reap of the write completion, split at 16 KiB (`tp_mailbox_host::ACKS`).
  That time holds no kernel time. It holds the way of the rows to the peer, the way of the acknowledgement back and
  the wait until the doorbell thread's next poll reaps the completion. It does not hold the post call itself: the
  clock is read after the post returns. So the mean is a fair size of two crossings of the link plus the thread's
  own wait, and the least is NOT a bound: a thread that loses its core between the post and the clock read measures
  too short (0.3 us was seen once). Read the mean; no one-way latency follows from these counters.

None of these changes a copied word or the order of the words.

## Tools

The lab kit's `MSG-anatomy.py` (the table above from two traces), `MSG-flood.py` (takes by the state of the prefetch
beside them) and `MSG-beside.py` (which launches run beside the prefetch) read the nsys exports of both ranks.
