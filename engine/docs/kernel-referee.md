# Native projection referee

`kernel-referee` is a default-off fixture binary. Build it with the
`kernel-bench` feature. It runs no model or transport. It is not a product
projection selector or an integration gate.

On one released, GPU-empty device, run:

```sh
kernel-referee CHECKPOINT BASELINE_CUBIN CANDIDATE_CUBIN ENGINEER NEW_OUT
```

The supported experimental native ABI labels are `opus`, `sonnet`, `fable`,
`sol`, and `astra`. They identify frozen fixture interfaces, not model
selection rules. Supply an image built from the matching interface. A missing
entry fails admission. There is no fallback to the baseline entry.

The fixture requires eight distinct real matrices of each admitted family:
block FP8 and NVFP4 at local shape 1024 by 4096, and plain weights widened to
FP32 at local shape 16384 by 512. It uses rank-zero rows of a two-part metadata
plan. It refuses missing families instead of substituting generated weights.
These are explicit laboratory cases, not production tuning constants.

For counts 1, 2, and 8, each candidate receives the same real matrices,
deterministic finite activations, and original calibration as its paired
production baseline. Quantization is outside projection timing. There are five
repeats, with baseline/candidate order reversed on every other repeat.

Before each sample, the baseline StreamMean operation reads a separate 256 MiB
source and writes a separate 256 MiB destination. It completes before timing.
This is cache pressure, not a guarantee of memory-controller traffic. Report
that distinction. CUDA-event times sum projection launches only. They do not
include public-call allocation, host work, readback, or cache pressure.

Astra has no distinct-matrix descriptor-table ABI. Its count cases use one
native launch per matrix. Fable's plain cases also use separate launches.
Other supported cases use a single native launch. Sonnet has no same-bytes
NVFP4 candidate. The output records these limits instead of changing a kernel.

Same-bytes candidates must match every baseline output byte. New-order output
is retained but marked `host_oracle=PENDING`. It is not a correctness PASS until
the corresponding author's source host reference checks all retained output
bytes. Repeated output stability alone is insufficient. Keep exact source and
image hashes, raw timing samples, and byte-check receipts with the report.

The process checks a 55-second submission budget before each case. An in-flight
CUDA operation cannot be safely canceled by this budget. The caller must own
the device lock and release it immediately after the context closes. Never
claim a pair or load a model for this fixture.
