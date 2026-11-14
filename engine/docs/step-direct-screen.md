# Direct recurrent-state output screen

This test tool compares the original StepColumns plus StepLayout sequence with
one Step launch that writes the final state/output layout. It does not change
the decay, prediction, update or output sums. Key folds keep ascending order.
The final output has the same BF16 rounding boundary. The old state is immutable;
the final output has its own disjoint allocation.

On one assigned empty GPU, run:

```sh
kernel-bench --causal-probe base.cubin candidate.cubin step-direct
```

The baseline sequence runs kernels from the baseline image, not the candidate.
The screen prints both image hashes. It checks all admitted direct shapes:
dimension 128 and head counts 1, 2, 3, 17, 63 and 64. Other shapes keep the
existing split path; host plan tests check that fallback.

Each shape has ten value patterns and three recurrent steps. Patterns include
wide exponents, signed zeros, denormals, infinity, cancellation and distinct
quiet/signaling NaN payloads. Every step compares every final state/output byte
and guard from three candidate paths: direct launch, recorder replay entry,
and the old two-launch composition. The saved baseline state is passed to all
arms for the next step only after every comparison succeeds. Both input planes
must remain byte-identical. Outputs are poisoned before each step.

Success requires 60 `STEPDIRECT BYTES PASS` lines, a summary of 540 output
comparisons, two `STEPDIRECT TIME` lines and the final `STEPDIRECT PASS`.
The host case-list test checks all shapes/patterns. A separate host negative
test changes an output byte and requires the comparison to report its plane
and word.

Timing uses each image's recorder replay entry and native geometry: the
baseline has two launches; the candidate has one. Eleven repeats alternate
arm order, with 480 calls per arm and one synchronization after each batch.
One mode reuses one resident state. The other rotates 24 independently owned
states, so the working set is large. This is not a claim that every load is
cold. Times include host enqueue work and the end-of-batch synchronization.

A screen PASS proves only the tested device arithmetic and memory bounds. It
does not prove complete model logits, token IDs, serving behavior or decode
speed. A separate paired model gate is required before merge.

After the positive screen, run this wrong-image control in a separate process:

```sh
kernel-bench --causal-probe base.cubin base.cubin step-direct
```

The old fast entry does not admit Step. It must trap and exit unsuccessfully,
not print a PASS or reach an outer timeout. Discard that process's CUDA context.
This checks that the candidate arm cannot silently use the baseline result.
