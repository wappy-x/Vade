# TCBF: BF16 latent tensor-core result

State: NOT READY FOR GATE. The twin is corrected and device-proven. The
latent kernel improves prompt groups but fails the one-row decode limit.
No served proof, token baseline change, rebase or merge was performed.
Targets `tc-switch`.
Branch `tc-bf`; tested code `82b9c70c479ea8b58bc19b169a3f7dbe418ee8b0`.
Base timing code `14021c361ce990386d3585339cad56e3da603e1c`.

## Choice and scope

Keep the first writer's four latent folds and the small load-overlap update.
Take the second writer's measured zero-term twin correction. Its async-copy
ring serves other plain matrices, so it cannot replace this latent kernel.
The ring's independent result is in `reports/TCBF2.md`, draft PR 262.
The only inherited unsaved source was the contract, saved in `c421084e82`.

The four operations are query and project against `kv_b_proj`, plus scores
and values against selected latent cache rows. The latter two are attention
products, not weight projections. The lead approved this scope at 21:07.
Other checkpoint-BF16 matrices remain unswitched: indexer wq_b, wk,
weights_proj, index_kpool_compress_gate; KDA f_a_proj, f_b_proj, b_proj,
g_a_proj, g_b_proj; router mlp.gate. KDA q/k/v and dense gate/up are MXFP8.

## Word and twin correction

Take the high BF16 half of each FP32 latent operand; weights retain their
BF16 bits. Fill ascending 16-term MMA steps, padding only a short last step
with positive zero. Split steps into four ranges from the term count alone,
accumulate each from positive zero, and join
`fl(fl(p0+p1)+fl(p2+p3))`. Scores multiply by the existing scale; the other
three operations round once to BF16. NaN places agree; payload is outside
the contract. Candidate B instead rounds general activations to BF16 once.

A zero product or carried zero contributes no alignment exponent. A
subnormal uses its exponent field, without normalization. A sum cut to zero
is positive zero. The inherited new host regression used `f32::powi(-137)`
for an expected value, which underflowed on the host. Its expected value
now uses f64 before conversion, and zero magnitudes are skipped before shifts.
The old model fails that regression (exit 101; `logs/TCBF-twin-old-control.txt`).

## Device arithmetic (MEASURED)

Recovered `tcbf-words-02` failed the crossed-class twin sweep. It is not a
passing receipt. Corrected `tcbf-takeover-words-03` passed on the inherited
kernel image `ce976469a455` at code `a9b2b8dd884e59299915d8bebb347cad6999716c`:

| Check | Words | Differ |
|---|---:|---:|
| Twin, five small shapes plus model samples, 18 class pairs | 947358 | 0 |
| Wrong-word control, old scalar score order | 20000 | 17289 |
| Row alone versus groups and placements, sizes 2 through 1040 | 531467712 | 0 |
| Direct decode versus prompt, 271 tokens | 21990688 | 0 |
| Recorded decode versus prompt, 271 tokens | 21990688 | 0 |

The recorded check took 1084 entries. Final load-overlap image proof:
`runs/tcbf-takeover-words-04/bench.log` PASS at the same tested code and image as the timing run. All five rows of the table above repeat with the same counts and differences; tool exit 0 and image check PASS.
These are kernel-level proofs; they do not prove served token IDs.

## Timing (MEASURED, milliseconds)

Receipt `runs/tcbf-takeover-time-02/bench.log`, same-device A X X A medians.
Candidate image `9afa0b92cbaf162aed20db2a4a0d1fb0e4fd64e48baa846b03b2b7d74d35830a`;
base image `49a26dae1180ae9132a8a657e3ed8fd66efe45bda883ccce1a5124189614c1b5`.
Tool exit 0, image check PASS. Mac load at start 32.95, during 45.31.
No local build or test ran during the timed calls.

These times sum four kernel launch-plus-sync medians. Allocations, host
dispatch and softmax are excluded. They are not served times or the older
`absorb-time` public-call times. The new fixture uses a bounded selection
spread across the cache, so absolute times must not be compared across tools.

| Prompt rows | Base 1 | Candidate 1 | Candidate 2 | Base 2 |
|---:|---:|---:|---:|---:|
| 1 | 0.4179 | 0.2972 | 0.2953 | 0.4197 |
| 2 | 0.4580 | 0.3016 | 0.2999 | 0.4591 |
| 4 | 0.4996 | 0.3452 | 0.3496 | 0.5003 |
| 8 | 0.7488 | 0.4580 | 0.4887 | 0.7550 |
| 256 | 15.2055 | 7.2262 | 6.9374 | 15.2327 |
| 1024 | 61.3937 | 24.8206 | 24.7959 | 61.3787 |

| One-row decode selected rows | Base 1 | Candidate 1 | Candidate 2 | Base 2 |
|---:|---:|---:|---:|---:|
| 2051 | 0.1968 | 0.2923 | 0.2934 | 0.1966 |
| 256 | 0.0771 | 0.1544 | 0.1540 | 0.0771 |
| 16 | 0.0617 | 0.1296 | 0.1304 | 0.0618 |

The one-row prompt result is not the one-row decode result. Ruling D fails.
At 2051 selected rows, values alone rise from 0.0511 to 0.1670 ms; query
and project also regress. Scores improve from 0.1034 to 0.0494 ms.
No served run or integration is justified by the prompt speedup.

## Host validation and state

- Twin tests: 8 PASS (`logs/TCBF-twin-takeover.txt`).
- Absorption tests: 8 PASS (`logs/TCBF-prefetch-absorb-tests.txt`).
- Old-model regression control: 1 expected failure.
- Kernel build and Linux benchmark check: PASS (`logs/TCBF-prefetch-kernels.txt`,
  `logs/TCBF-bench-linux.txt`).
- One-path source check: PASS. Unsafe source check: FAIL, detailed below.
- Full suites and clippy remain pending under the lead's 21:51 hold.

## Takeover: remaining served work, by file

No merge into `tc-switch` is authorized by these kernel receipts alone.

| File or area | Work still required |
|---|---|
| `engine/crates/gpu/kernels/src/latent_absorption.rs` | Final load-overlap sweep PASS. Still meet the decode speed rule at full and short selections. |
| `engine/crates/gpu/src/absorb_tc.rs`, `latent_absorb.rs`, `latent_absorb_rows.rs` | Keep the same four-range word and epilogues across every path. The corrected twin and final kernel image are device-proven on the recorded sweep. |
| `engine/crates/tp/src/resident_weights/device/absorption.rs` | Verify served prompt, single decode, grouped chats and draft rows take the measured entries. Existing wiring has not been tested in a served request. |
| `engine/crates/gpu/src/prompt_launches.rs` | Capture nonzero prompt, direct and recorded counters in served receipts, not only check-tool receipts. |
| `engine/docs/tc-dense-contract.md` | Keep the corrected zero-term rule and final receipts current; explicitly separate latent activation truncation from candidate B activation rounding. |
| `engine/crates/tp/src/bin/resident-decode-check/absorb_tc_rows.rs` | Retain crossed classes, wrong-word control, direct and recorded entry coverage on the final image. |
| `engine/crates/gpu/src/backend/causal_ops.rs`, `backend/prompt_ops.rs`, `small_choice.rs` | For other plain matrices, replace both scalar decode and grouped/prompt dispatch together with the proven candidate B word. They have not switched. |
| `engine/crates/tp/src/resident_weights/device/projection_stage.rs` | Include fused plain projection batches and packed BF16 paths in that switch; select matrices by checkpoint dtype and shape. |
| `engine/crates/gpu/src/backend/resident_launch.rs`, `kernels/src/resident_launch.rs` | Add or update recording/replay for any plain-matrix entry introduced by that switch. |
| `engine/crates/gpu/src/prompt_ops.rs`, `causal_ops.rs` | Update host projection references and tests to the same new word, with unchanged output layout and BF16 boundary. |
| Gate-owned numbered baseline and served receipts | Repeat the same request, alone/group, prompt/decode, continuation/cache, drafted/plain and quality proofs. Keep old baselines untouched. |

The full source checks and touched-crate suites are still pending. The lead's load rule holds full suites and clippy above load 15. No served speed claim is made.

Source audit: `check-one-path` PASS (`logs/TCBF-source-onepath.txt`).
`check-unsafe-sites` FAIL (`logs/TCBF-source-unsafe.txt`): the inherited
`kernels/src/latent_absorption.rs` adds unsafe functions `count_at`, `pair`,
`column`, `weights`, and `run`; three entries for deleted unsafe blocks are
stale in `tools/unsafe-allowlist.txt`. Rework the helpers within the allowed
kernel boundary and review the stale entries. Do not add an allowlist row
to hide the failure. This is a separate integration blocker from decode speed.

Final disposition: measured draft only. Source checks held by Mac load (full clippy and suites); the separately run unsafe check failed and one-path passed. No served request or numbered baseline was made.
