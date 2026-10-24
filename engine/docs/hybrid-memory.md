# Typed hybrid serving memory

Admission funds resident weights, shared workspaces, then active requests and retained cache.
It refuses with required and available bytes when an owner cannot be funded.
The allocator separately enforces the GPU ceiling; no context or output reduction is implicit.

For plain typed serving with multiple active slots, the adapter owns one 256-row prompt executor.
Every session and cache fork refers to that same pool.
The scheduler calls prompt tiles synchronously; the typed executor completes a tile before returning.
A checked mutable borrow refuses overlapping access instead of lending scratch twice.
Each chat keeps its recurrent state, KV, canonical replay checkpoint and private 16-row executor.
Sixteen private rows cover the public verification interface as well as singleton decode.
Explicit one-slot serving keeps the original private 256-row executor and execution order.
Drafted and untyped serving retain their original ownership policy.

The score bound uses the configured context, rather than the checkpoint position ceiling.
It retains two score spans because allocation growth can overlap the old and new buffers.
KV was already grown to a power of two, floored at 64 and capped by the configured context.
That capacity rule, the BF16 KV format and all attention/recurrence arithmetic remain unchanged.
Host logits are charged at the row capacity of the owner that can produce them.
The shared prompt readback and shared decode readback are each funded once.
Typed graph storage is disabled by the launcher; the executor does not add a graph arena.
The existing small tail and descriptor allowances remain funded.

For the 35B fixture, the old minimum slot bound was 9,758,803,264 bytes:

| Owner | Bytes |
|---|---:|
| One-row session scratch | 4,525,376 |
| Recurrent state and replay snapshot | 129,761,280 |
| BF16 KV for the minimum 64-token capacity | 1,310,720 |
| Partial-page/migration allowance | 83,886,080 |
| Private prompt executor, including score replacement | 9,027,723,264 |
| Host logits | 510,545,920 |
| Capacity bookkeeping | 2,048 |
| Host bookkeeping | 1,048,576 |
| MTP, plain mode | 0 |

The prompt executor consists of vector arenas 264,713,280; GDN 56,787,968;
dense attention 17,825,792; typed MoE 98,461,568; two score spans 8,589,934,592;
and tail allowance 64 bytes. The two score spans are 256 rows × 16 heads × 262,144 positions × 8 bytes.
These are conservative owner bounds, rather than measured simultaneous allocations.
Full-context KV costs 20,480 bytes per token, plus 32 bytes per token of bookkeeping.

Using the same fresh-start budget as the admission receipt, full 32K private slots fall
from 10,429,627,712 to 1,043,576,128 bytes. Shared owners rise from 588,208,128
to 2,174,294,016 bytes, leaving 16,175,553,255 bytes for private owners.
Eight full slots use 8,348,609,024 bytes and fit that pool.
At 16K, sixteen full slots use 10,783,249,408 of 16,745,978,599 available bytes.
Other configurations still require admission arithmetic; sixteen full 32K slots do not fit.

The available-memory input is min(Linux MemAvailable, CUDA total), less the process's CUDA allocations since its first query (a container's MemAvailable may omit them).
CUDA free is not an input: in a container it excludes host page cache that MemAvailable counts.
Startup holds back 10%, at least 1 GiB, for the OS, tokenizer, driver and other non-device owners.
It then applies the configured GPU ceiling and subtracts resident and shared bounds.
Container memory figures cannot establish which host owners consume CUDA-unavailable memory.
The host regression matrix and frozen baseline executor checks do not replace native answer-byte qualification.
