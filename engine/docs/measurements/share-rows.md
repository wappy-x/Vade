# Several rows share one read of a shared weight (block FP8, MXFP8)

Measured on one Spark with `kernel-bench`, 2026-11-16. Microseconds a launch unless a line says otherwise. Every number is MEASURED (run tag named) unless it says ESTIMATE.
Base image: main `57f0869746`. The served numbers are in the pull request.

## 1. Block FP8: `fp8_rows_share` against the forms of main

Run `shareb-fp8-time-06-1` (`kernel-bench --fp8-small {OLD} {X} sharetime`, head `0dab80bef0`). "One row" is `fp8_rows_small` with one token.
"main" is the form the host of main picks for that count (`fp8_rows_small`, `fp8_rows_few` or `fp8_rows_multi`). Bold: the host rule `fp8_small::share_wins` picks the new entry.

| Table (rows x cols of one rank) | Launches a step | One row | Form | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gate and up 1,024 x 4,096 (two in one launch) | 42 | 42.3 | main | 59.1 | 49.9 | 50.2 | 56.1 | 56.9 | 57.9 | 58.4 |
| | | | `fp8_rows_share` | **46.5** | **43.4** | **44.5** | **50.4** | **52.9** | 57.2 | 59.8 |
| down 2,048 x 2,048 | 42 | 22.9 | main | 25.7 | 27.9 | 28.6 | 35.4 | 34.1 | 34.5 | 35.0 |
| | | | `fp8_rows_share` | **23.4** | **24.1** | **24.9** | **27.7** | **28.7** | **30.7** | **31.9** |
| o_proj 2,048 x 16,384 | 11 | 152.9 | main | 162.4 | 170.8 | 169.6 | 190.7 | 193.2 | 195.7 | 198.4 |
| | | | `fp8_rows_share` | **155.9** | **160.0** | **161.0** | **184.5** | 194.6 | 211.5 | 221.8 |
| q_b 8,192 x 1,536 | 11 | 63.1 | main | 70.0 | 77.9 | 78.4 | 83.9 | 85.2 | 86.2 | 86.9 |
| | | | `fp8_rows_share` | **66.1** | **67.2** | **67.9** | **71.0** | **73.5** | **78.2** | **81.2** |
| q_a 768 x 4,096 | 11 | 20.8 | main | 22.4 | 26.1 | 48.6 | 41.5 | 41.4 | 41.8 | 42.1 |
| | | | `fp8_rows_share` | 24.2 | **24.7** | **25.2** | **26.0** | **26.3** | **27.5** | **28.2** |
| kv_a 256 x 4,096 | 11 | 17.8 | main | 17.8 | 17.8 | 21.1 | 38.6 | 38.6 | 38.9 | 39.2 |
| | | | `fp8_rows_share` | 23.0 | 23.5 | 24.0 | **24.5** | **24.9** | **26.2** | **26.9** |

Notes:
- The first cell of the run (gate and up at 2 rows, main 59.1 and share 46.5) is a warm-up outlier. Run `shareb-fp8-time-05` has 50.0 and 42.4 there.
- A step, ESTIMATE (launch time x launches a step, all six tables), ms: main 6.2 / 6.5 / 6.8 / 7.7 / 7.8 / 7.9 / 8.0 at 2 to 8 rows; with the rule 5.7 / 5.8 / 5.9 / 6.6 / 6.9 / 7.3 / 7.5; one row 5.5.
- Bar of the brief (a form at 2 to 4 rows costs at most 1.15 times the one-row launch): gate and up 1.03 to 1.05 (2-row cell from run 05: 1.01), down 1.02 to 1.09, o_proj 1.02 to 1.05, q_b 1.05 to 1.07 are inside. q_a at 3 and 4 rows is at 1.21 and 1.20. kv_a keeps the small kernel (1.00 to 1.19).

### Why the forms of main cost more
`fp8_rows_few` and `fp8_rows_multi` copy a weight once, but each row's warp decodes and scales it again, in the thread that also runs the sum chain. `fp8_rows_small` runs the one-row tile once a row, so every row copies the weights again.
The new entry copies and decodes a weight once (128 producer threads a block, as the one-row tile) and stores the first product `e4m3(w) * scale`. A chain lane for each (weight row, token) loads that product and its own activation from shared memory, makes the second product and adds it to its own chain in column order.

### Two causes found on the way
1. A chain lane that loads its activation from global memory stalls on every load. Version 1 did that: gate and up at 2 rows 128.9 us (run `shareb-fp8-time-01`). With the activations copied into shared memory by the producers: 42.6 us (run `shareb-fp8-time-03`).
2. Shared memory of a block decides how many blocks an SM holds. Run `shareb-fp8-pad-05` (`sharepad`: the same kernel with more dynamic shared memory than it needs), gate and up at 4 rows: 45.2 us with 30,912 B, 45.7 us with 33,024 B, 70.1 us with 34,080 B and more. So a block must stay at 33,024 B or less (3 blocks an SM). The first layout needed 34,080 B from 5 rows and lost there (65.3 us against 56.9). The layout of this branch copies the activations of 5 to 8 rows one chunk later into 2 stages (30,912 B at 5 and 6 rows, 33,024 B at 7 and 8): 50.4 us at 5 rows.

### Where it does not win, and why
- Gate and up at 7 and 8 rows, o_proj at 6 to 8 rows: a block has four chain warps; the chains are the bound, not the weight read.
- q_a at 2 rows, kv_a at 2 to 4 rows: 16 to 48 blocks in all. The chain of one block is the bound and the small kernel spreads the rows over more blocks.

### Bytes
`kernel-bench --fp8-small {OLD} {X} share`, run `shareb-fp8-bytes-06-1`: PASS. 34,368 comparisons against `expert_group` (all codes; every listed rows x cols x block shape at 2 to 8 tokens, one part and two parts on the same rows; the six served tables with five kinds of activations); 294 checks "a row alone equals that row in a group, in every slot"; 336 wrong-on-purpose controls caught (another slot, reversed rows, one changed row); 168 neighbor checks.

## 2. MXFP8: the tile for 5 to 8 rows of q, k, v and the dense gate and up

Run `shareb-mx-time-03` (`kernel-bench --mx-small {OLD} {X} time`), the entries of main:

| Table | Rows | `mx_rows_small*` (the form of main) | `mx_rows_tile` |
|---|---:|---:|---:|
| q, k, v 4,096 x 4,096 (102 launches a step) | 2 | 79.1 | 96.4 |
| | 3 | 90.1 | 97.7 |
| | 4 | 91.2 | 99.0 |
| | 5 | 103.4 | 99.2 |
| | 8 | 113.6 | 98.5 |
| dense gate, up 6,144 x 4,096 (6 launches a step) | 4 | 133.2 | 140.7 |
| | 5 | 151.7 | 142.6 |
| | 8 | 165.1 | 143.2 |
| o 2,048 x 8,192 (34 launches a step) | 4 | 86.7 | 95.1 |
| | 8 | 97.3 | 95.7 |
| dense down 2,048 x 12,288 (3 launches a step) | 8 | 152.8 | 137.6 |

- The host rule (`small_choice::mx`) now sends 5 to 8 rows of the two 4,096-column tables to the tile. ESTIMATE of a step at 8 rows: (113.6 - 98.5) x 102 + (165.1 - 143.2) x 6 = 1.7 ms. The 8,192 and 12,288-column tables keep `mx_rows_small8w` (no gain worth a rule).
- The tile was not a recorded launch: inside a decode step each tile launch would end the step's launch batch. `Kind::MxRowsTile` records it for groups of at most 8 rows. A prompt's tile launches (9 rows and more) stay direct, as before.
- The output matrix at 3 and 4 rows: the tile (95 us) is slower than `mx_rows_small4` (86 to 87 us) in this bench. The 123 and 137 us that `--shared-forms` shows for `mx_rows_small4` are not what the served engine pays (98 to 100 us in the trace `rows3-trace-takeover-1`). So the "output selector" (tile at 3 and 4 rows) is not taken.
- Negative: loading the activations of the small entries one tile ahead of their FMAs is byte-equal (run `shareb-mx-bytes-03`) and slower (q, k, v at 8 rows 133.2 us against 113.6, run `shareb-mx-time-03`). Not in the tree.
- Bytes: `kernel-bench --mx-small {OLD} {X}`, run `shareb-mx-bytes-05`: PASS, 8,760 comparisons; the tile at 5 to 8 rows equals the small entries on the four served tables (80 comparisons, 16 shifted-rows controls caught). `kernel-bench --mx-rows {OLD} {X}`, run `shareb-mxrows-05`: PASS, 18,525 comparisons (the tile entry after its body moved into a function the wrapper also calls).
