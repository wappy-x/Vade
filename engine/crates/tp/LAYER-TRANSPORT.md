# Ordered layer transport

`layer_transport::PairPlan` binds each declared `LayerPlan` gather to the
existing copy mailbox. It derives intervals from the global output shape,
split axis, and complete unit. It keeps projection input K complete. Weight
rows retain original tensor and scale coordinates. No partial dot product
or rank sum crosses this transport.

The caller supplies each gather's physical dtype from the operator binding.
This input does not authorize a checkpoint cast or resolve an unknown cache
dtype. Packed BF16 boundaries must cover whole words. A refused boundary
does not acquire padding or another execution path.

`PairGroup` owns a dedicated existing control `transport::Ring`. Agreement
checks rank identity, request or lane ID, causal token, persistent sequence,
call count, and complete program bytes. It first exchanges fixed extents.
Then it compares full rank-normalized programs and physical types. The
versioned manifest includes scalar bits. It uses no hash that could hide a
different program. A mismatch fails the group before a device launch.

`tensor_wire` exports the same source used by the current serving wire.
Create its `WriteLink` on a separate peer stream. Bind and start its worker
on the mailbox host endpoint. Wait for attachment before using `DevicePair`.
The worker owns the ordinary RDMA key swap, write completion, and slot reuse.
There is no new wire protocol, GPU kernel, or floating-point reduction.

`DevicePair` borrows one backend and one fresh mailbox for its lifetime.
Message counts persist across layers, tokens, and requests. A new group
cannot claim the mailbox again. Exhausted device sequence bits require a
new group and a fresh mailbox. Counts never restart on an old wire.

After every operator, call `DeviceLayer::after_step` with that step index.
Pass only its declared gather outputs, in declaration order. Each output
must own a complete writable global row-major vector. Its producer writes
only its own interval at the original global offset. The adapter queues
the local put, then takes remote intervals in global rank order. It waits
for completion before it returns. The next operator can then read complete
global features. Steps without gathers still advance the checked order.

The caller must keep its state and operator allocations alive through its
launches. The backend checks vector and mailbox context ownership. Copy
completion protects vectors borrowed by `after_step`. `finish` checks every
step and waits for both ranks. An error or abandoned layer fails the control
group and mailbox. The existing wire propagates failure to its peer. The
adapter drains partial launches before borrowed storage can be released.

The session owns aggregate allocation limits and exclusive lane scheduling.
Request IDs in agreement must distinguish interleaved lanes. The adapter
does not run decoder operators, admit checkpoint execution, or resolve
physical cast contracts. The device driver must call these boundaries.

The host tests check the saved description, ordered layouts, full program
agreement, type and request mismatches, and persistent sequences. The
`layer-transport-check` executable runs synthetic byte patterns through
the actual distributed mailbox wire. It includes raw signed-zero and NaN
bits. Its records name all gathers and compare every global word. It keeps
failed bytes. It refuses an empty selection. It does not load weights or
produce model token IDs.

Device byte proof and complete model exactness require station runs. The
model gate must compare repeated token IDs across supported topologies,
alone, batched, cached, chunked, and with drafting on and off. Host checks
and the synthetic byte gate do not establish those model results.
