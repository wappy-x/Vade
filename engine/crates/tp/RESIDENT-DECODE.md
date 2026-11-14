# Resident decoder diagnostic

`resident-decode-check` runs the bounded checkpoint decoder on a pair. It loads
matrix rows and the embedding table before prompt execution. Sessions share
the immutable device owners. Norm, residual-block, and scalar-operation owners
remain cached after first use. A new session starts with fresh causal state.

Within a session, repeated KV expansion projections reuse completed output
rows when the weight name, rounding contract, and every input float bit match.
This preserves signed zeros. These temporary owners drop with the session.
Their budget leaves the ordinary scratch reserve available and is capped at
1 GiB. The ranks agree on the smallest available budget before each session,
so eviction and collective-skip decisions stay aligned. Request prefix caching
remains disabled in this diagnostic.

Each projection retains its complete feature axis. Ranks own equal output-row
intervals aligned to the storage scale blocks. A matrix that cannot make equal
complete intervals stays replicated. The existing ordered transport copies
finished output rows into global order. It does not add partial dot products.
Routing, expert selection, recurrent state, and residual operations consume
complete gathered values on each rank.

The resident mHC prepare kernel uses one block per token. Its barriers order
the unchanged scalar RMS fold, independent normalized stores, complete
coefficient dots, and the original activation and Sinkhorn folds. Coefficient
warps do not split a dot. The following collapse completes the same ordered
stream before the public pass returns. Prepare has no separate host wait.

The MXFP8 activation operation uses the same BF16 source boundary, E8M0 scale,
and E4M3 nearest-even codes as the independent host quantizer. The existing
checked causal kernel supplies its device entry. The dot-product feature tree
stays unchanged. The calibrated FP4 and block FP8 projections use the reviewed
expert operations with their original ascending feature fold.

KV output layout uses one checked copy operation to separate key and value
planes and write the original zero padding. It keeps slot, head, and feature
order. It adds no floating-point arithmetic. The layout diagnostic checks
unequal key/value widths, signed zero, and empty selections against explicit
expected rows on the device.

The command prefix matches the existing transport handshake:

```text
resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS
resident-decode-check prepare MODEL REQUEST_JSON NEW_IDS
resident-decode-check quantizer NEW_OUT
resident-decode-check layout NEW_OUT
resident-decode-check recurrence NEW_OUT
resident-decode-check convolution NEW_OUT
resident-decode-check prompt-rows NEW_OUT
resident-decode-check compare LEFT_RUN RIGHT_RUN
```

Start the pair through the normal authenticated rank transport. Both ranks
must use the same immutable checkpoint, kernel image, request, prompt IDs, and
output-directory argument. The prepared IDs must match the checked chat
template. The request requires plain greedy decoding, neutral penalties,
disabled caching, explicit zero draft depth, and `ignore_eos`. The output bound
is 256 tokens. The prompt bound is 512 tokens, and the total context bound
is 768 tokens. Public service admission stays
closed.

Each run saves all IDs, decoded text, full generated logit rows, prompt-logit
hashes, and timing. The comparator checks deterministic argmax from every saved
row. It compares the common generated prefix when output lengths differ. Thus
an eight-token staged receipt can check a longer resident run without treating
the extra tokens as a staged-reference proof. The prefix rule is only for that
compare mode. Repeated sessions of one job must give the same token count. A
head-last session compares the same tokens and full rows, and each prompt row
that both runs hashed.

`decode_steps` is the output count minus one. The first output comes from the
last prompt step. Timing starts after that step and includes the remaining
decode steps, logit readbacks, and receipt writes. Weight loading and prompt
processing are outside this interval. `RESIDENT COST` reports projection
compute and ordered-gather wall time for each committed token.
`RESIDENT SESSION` reports setup separately. `prepare` writes checked chat
template IDs without GPU work. Full prompt logits remain enabled for proofs.

The runner accepts `--prompt-rows=N` for a matched scheduling measurement.
The checked range is one through four rows. Its default is four. A bounded
comma-separated list, such as `--prompt-rows=4,1`, cycles across repeated
sessions under the same resident weights. Both ranks bind this schedule in
their diagnostic identity. Each timing receipt records its row count. It changes
only the number of independent prompt rows in a launch. It does not change
the arithmetic of a row. The `prompt-rows` device fixture compares grouped
outputs with scalar device and host outputs, including row and scale tails.

The diagnostic runner option `--cost` adds per-operation wall, driver, and compulsory matrix-byte
counters. `--gpu-cost` also enables existing CUDA events after the
first session. The first session measures wall costs without those events.
Later sessions expose event overhead and the GPU/host split. These diagnostics
do not select different arithmetic. Event times and driver wall intervals can
overlap; they must not be added as independent wall phases.

The bounded device enables the existing vector pool after resident weights
load. It respects `VADE_VECTOR_POOL` and an earlier service configuration.
The backend retains its existing size cap, memory pressure checks, and private
buffer ownership. Each reused vector is written before any consumer reads it.

The recurrent update schedules one independent value column per output unit.
Each column keeps the original ascending key folds, decay, delta, and rounding.
A copy operation restores the original state/output layout. The recurrence
diagnostic compares every state and output bit against the original complete
head operation over three state updates and several dimensions.

Convolution schedules independent channels while retaining the original tap
fold and history order. A copy restores the complete-plane output layout.
The convolution diagnostic compares every output and history bit against the
original device operation over several dimensions, tap counts, and updates.

This diagnostic does not qualify checkpoint precision, publisher parity,
drafting, cache forks, long context, or HTTP serving. It does not remove the
checkpoint runtime refusal. A serving adapter must retain the resident owners
and schedule matching token transactions on both ranks.
