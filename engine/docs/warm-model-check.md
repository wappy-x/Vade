# Resident full-model kernel candidate gate

This is default-off test tooling. It is not a product selector, a sandbox,
or a device-proven gate until its exact build has passed the pair screen.
Build `resident-decode-check` with the TP `kernel-bench` feature.
Use the normal pair command and append:

```text
--jobs=jobs.tsv --warm-images=images.tsv
```

The positional job and `--jobs` entries form the prompt matrix. Each existing
job row is `INPUT_IDS REQUEST_JSON NEW_OUT REPEATS`. Warm mode requires the
canonical15-token prompt and a127-token prompt,64 outputs and3 sessions per
job. Additional prompts have the same output/session bounds. Requests retain
the existing plain greedy, no request cache and no drafting checks. Supply
the actual canonical request and its prepared IDs, not an arbitrary15 IDs.
All prompt logits must be hashed; head-last and cost-window modes refuse.

The image manifest is:

```text
baseline baseline.cubin
candidate_one candidate-one.cubin
candidate_two candidate-two.ptx
```

Optional columns are `PROJECTION_WEIGHT EXPECTED_HEX_BITS_OR_any`, followed
by optional `ENTRY GRID_SCALE THREADS`. Geometry can also be given without
a probe. Both contracts join the image identity agreed by the ranks. A probe
captures the first matching scalar projection during decode, not prefill.
It observes the actual returned GPU vector, including a projection-memo hit.
The vector and its hash are saved as `PROJECTION_MARKER.f32` and metadata.
An explicit marker requires every output word to match. An unobserved or
wrong marker stops the matrix; it is not a valid negative-control verdict.

Image paths are relative to the manifest. Labels contain only ASCII letters,
digits, hyphens and underscores. There are at most16 candidates, each at most
64 MiB. Images must contain the full compatible module, not a partial bench
entry. Both ranks agree on the complete image-content hashes before the load.
The same manifest ordering and jobs must be present on both ranks. File paths
and output directory names are not part of that agreement. Binary identity
and matching host/kernel ABI remain the operator's responsibility.
Every listed image, including the reference baseline, uses the full A/B
module arm. Lazy function tables therefore resolve in that image, not in
the backend's original embedded layer module.

## Execution and evidence

The runner opens one device context per rank and retains one resident weight
owner. Its checkpoint pointer admission prevents a second large matrix load.
It runs baseline, each candidate over all jobs, then final-baseline. Each arm
and job has separate raw output folders, three complete generated-logit sets,
IDs, prompt hashes and timings. Comparisons require the full64 outputs, not
an equal shorter prefix. Every session compares against initial baseline
session1 for that prompt. Each job prints the median of its three decode
rates. The normal runner's CUDA/host timing definition is unchanged.

Module-load failure is voted across both ranks before either starts that
image. A byte mismatch is recorded only after all sessions finish. A rejected
candidate inserts a complete baseline recovery gate before the next image.
If recovery or the final baseline differs, the runner stops and requires a
new process and full reload. CUDA, transport, execution and cleanup failures
are terminal: no later candidate runs in that context. This first version
does not automatically restart a poisoned process. Do not call that path
automatic recovery. `COMPLETE` means the matrix ended, not that every image
passed. Inspect per-arm `VERDICT` and top-level `*.REJECT` files.

## State boundary

Before each swap, each decoder and its state have left scope. That discards
KDA recurrence, latent attention and index state, expanded rows, token/output
buffers and per-session host schedules. Every session starts with a new
decoder and `zero_state()` at position0. Decoder preparation also clears the
projection memo between sessions.

The queue guard drains and closes before a swap. The explicit fixture reset
drains again, clears projection memo/origin tables, standalone recorded
programs, copies, norm owners, mHC boundary parameter owners and uploaded
parameter caches. It destroys backend recorder graphs and indirect tables,
then refuses if any externally owned graph remains. Dense offers and flash
scratch owners are cleared. A new candidate starts with empty module-specific
lazy function tables. No graph may retain an unloaded function handle.

Weights, their immutable lookup/calibration data, device context and transport
ownership remain resident. At a swap, free vector-pool and KV reserve spares
are released, along with retained decode partials. Mailbox sequence
numbers and immutable source identities remain monotonic; resetting them
would break the live peer protocol. Live weight bytes and hardware caches are
not zeroed or made cold. This is a warm-run throughput comparison, not a
fresh-process allocator or cold-cache equivalence proof. The ending baseline
and recovery checks detect drift exercised by the gate, not every possible
write to an unexercised weight. Images remain trusted executable GPU code.
The first version does not hash the complete resident weight inventory.

Host tests exercise manifest admission, recovery ordering, matrix bounds and
full-length raw comparison. They do not prove module lifetime or model bytes
on a GPU. The device acceptance screen must include same-image controls,
invalid/missing-entry and deliberately wrong-arithmetic images, a following
good candidate, both prompts on both ranks, and final baseline equality.

## Module marker control

The default-off Linux command `resident-decode-check warm-control NEW_OUT`
tests module selection and recorder retirement without loading model weights.
It changes one store in the recorded zero operation to the FP32 bits for1.0.
In one context it runs wrong-image FAIL, right-image PASS and baseline-again
PASS three times. Each output plane and image hash is saved. The command also
tests refusal with an active queue, an externally owned graph and a permanent
context error. A separate context is used for the permanent-error test.

This control does not exercise the projection memo or a full model. Its PASS
does not remove the full-model trust block. A changed projection must execute
through the actual recorded entry, not only an unused eager entry. An image
hash alone does not establish which body ran.
