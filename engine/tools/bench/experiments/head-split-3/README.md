REOPENED as a KDA ownership rider by the 2026-11-10 near-miss ruling. Latent remains killed.

Base: rc.8, c55511d249d3ac38a66b7c61172cfa5a5520f68a.
These are modeled savings from rc.8 rows and one-device ratios. They are not measured model gains.
Commit 619cde813 kept rc.8 production execution. The patches below preserve its diagnostic inputs.
The new rider keeps rc.8 kernel arithmetic, admits 32-head planar work, and makes KDA heads local.
It excludes both cache patches. Its rc.8 estimate is 0.777965 ms weighted.
The prior Belle full and local event spans measured 0.757919 ms saved, above the new 0.4 ms guard.
Production gates are recorded separately. The earlier kill ruling is superseded.

| Cell | Latent ms | KDA split ms | KDA split plus static cache ms |
| --- | ---: | ---: | ---: |
| c15 | 0.024732 | 0.771611 | 0.824468 |
| c127 | 0.023427 | 0.775051 | 0.827789 |
| c1040 | 0.023055 | 0.778786 | 0.832046 |
| c4158 | 0.021190 | 0.789844 | 0.843972 |
| Weighted 63/63/31/47 | 0.023258 | 0.777965 | 0.831139 |

Source audit:

- Latent ownership already landed in 28bdbf662 and 64a513cc1. Both are ancestors of rc.8. `tp/src/bounded_decoder.rs` uses owned expansion, local query heads, and an attended gather. `tp/src/bounded_decoder/rows.rs` uses the same local attention heads. The remaining q_b payload is 360,448 bytes per rank per step. It saves no exchange or attention work.
- KDA still runs all 64 heads per rank. Its projection rows are split already. `tp/src/projection_stage.rs` batches f_b and g_b into one frontier. Proposed ownership replaces that gather with the gated gather. Three messages per layer remain three. The model step remains 279 exchanges.
- KDA would save 2,232,576 payload bytes per rank per step and 71,303,168 state bytes per rank per session. It adds no weight bytes. State capacity and logical load counts are not measurements of DRAM traffic.
- Put/take fusion is separate. No fusion or fixed-message gain is included. The earlier fusion premise was rejected. The proposed ownership can use its gather interface if fusion returns later.

The corrected device job ran on one GB10 on 2026-11-10, 11:58:14 to 12:00:21 UTC. Driver API: 13000. Compute capability: 12.1. SM count: 48. It used no checkpoint. The fixture compares finite synthetic tensors only.

`H.txt` uses rc.8 arithmetic with temporary 32-head admission. `C.txt` adds a dynamically indexed cached column. `S.txt` uses literal indices. Both sums keep ascending key order. Each arm alternates 64 and 32 heads over 14 trials. It drops the first four. Every trial visits 34 separate states and outputs. GPU event spans include submission gaps. They are not isolated kernel durations.

At diagnostic commit 619cde813, all 24 per-rank output comparisons passed. All eight cross-image full-output file comparisons passed. This is not the model L0, twin, exchange-stage, or real-pair gate. At that commit, none of those gates ran because neither premise passed the original line. No transport change or production head split was included there. No served gain is claimed.

| Operation, 34 calls | H64 ns | H32 ns | Local-cache 32 ns | Static-cache 32 ns |
| --- | ---: | ---: | ---: | ---: |
| ConvParts | 321152 | 222944 | 220512 | 223039 |
| Prepare | 250752 | 247519 | 252480 | 226111 |
| StepPlanar | 1522719 | 866144 | 1253823 | 835456 |
| GatedNorm | 147039 | 147136 | 147520 | 146272 |

Source: `receipts/{H,C,S}.txt`. The dynamic cache used a 512-byte local PTX array and lost time. Literal indices removed that array. Native register allocation was not measured. The static 32-head step saved only 30,688 ns per 34 calls against H32. Halving heads did not halve Prepare or GatedNorm time.

Price method and limits:

- `receipts/census-order.tsv` retains the first nine columns of one warm c1040 rc.8 step. It gives ConvParts 0.300640, Prepare 0.340064, StepPlanar 1.120128, and GatedNorm 0.176832 ms. The second KDA take is 1.125248 ms and carries prefetch. It is not removable wire time. `source.sha256` records the full source exports.
- `price.rs` scales those operation costs by the measured shape ratios. It uses each cell's detail rows for ConvParts and the Prepare/GatedNorm total. The fixed-shape StepPlanar baseline stays 1.120128 ms at all cells. Quantizer credit is zero.
- Sender cost uses the rc.8 put grid means. Receiver byte cost uses the 1x6 and 1x5 take minima. Wire and peer wait are not separately observed. The resulting KDA transport credit is 0.194719 / 0.196034 / 0.197555 / 0.203290 ms. No message-count or prefetch saving is credited.
- The deliberately larger local transport sensitivity gives 1.110288 ms weighted for the static form. It comes from different exchange shapes, removes more payload, and uses an artificial fixed delay. It is not a measured rc.8 fabric gain or an upper bound. The original kill decision used the requested rc.8 price line; it does not prove that all possible ownership designs are below 1 ms.
- The result files include sample ranges by arm order. They retain the observed timing variation.

Recompute the prices from this directory:

```sh
rustc --edition=2024 order.rs -o /tmp/kda-order
/tmp/kda-order receipts/census-order.tsv
rustc --edition=2024 price.rs -o /tmp/kda-price
/tmp/kda-price receipts/H.txt receipts/S.txt receipts/census-detail.txt receipts/census-order.tsv
/tmp/kda-price receipts/H.txt receipts/C.txt receipts/census-detail.txt receipts/census-order.tsv
```

To reproduce the device premise, use an isolated checkout and an authorized idle GPU. The normal fixture refuses the unadmitted 32-head planar shape. Patch paths start with `engine/tools/bench/experiments/head-split-3/` from the repository root:

1. Apply `admission.patch` with `git apply`. Build kernels from `engine/` with `cargo +1.99.0 kernels`. Save the resulting `spark_layer_kernels.ptx` as H.ptx outside tracked source.
2. From `engine/`, run `cargo +1.99.0 build --offline --locked --release -p spark-gpu-probe --features kernel-bench --example kda-heads`. This host binary must retain temporary admission for every arm.
3. Apply `cache-local.patch`, rebuild kernels, and save C.ptx. Reverse that patch with `git apply -R`.
4. Apply `cache-static.patch`, rebuild kernels, and save S.ptx. Reverse that patch and `admission.patch` after the experiment. The temporary admission patch does not update the production admission tests.
5. Run the same fixture binary as `kda-heads H.ptx NEW_H_DIR`, then C.ptx and S.ptx with distinct new output directories. Compare every corresponding `.f32` file with `cmp`.

Use the pinned toolchains and `--offline --locked` for host Cargo builds. Do not commit generated PTX or binaries. The original output planes and binary are retained in the local lane receipt. The final shell job returned 127 only because its final log filter used unavailable `rg`; all fixture runs and byte comparisons completed first. A later `cmp` of all files passed.
