# Dense tensor-core gate tools

The gate collects new-ID evidence after an authorized arithmetic rebase.
It never overwrites an old baseline or golden. The lab kit supplies the
build, serve, waiting-line, stop and release commands.

Run `GATE2-check.sh COMMIT NUMBERED_BASELINE TAG [PAIR]` from the lab kit.
Set `GATE2_CONTROL_COMMIT` to the format writer's control-only commit.
The control commit must never merge. Read the latest pair order first.
Each served turn uses the waiting line and releases its claim.

The pair drivers read the two box names of a pair from a lab file that
is not in this repository: `pair-hosts.tsv` in the lab kit, or the file
named by `GATE2_PAIR_HOSTS`. Each line holds `PAIR RANK0 RANK1`.
Lab paths start at `$HOME`. No box name belongs in this folder.

The first two parts use distinct fresh plain servers. The script compares
their IDs and image hashes and checks their process identities. It then
creates the new numbered baseline folder. If the folder exists, its commit
and IDs must match; it is read only. The later parts check drafting,
continuations, cache/cold equality, row tools, and the original four golden
requests twice. The model pair driver compares full logits for decode
against one-row prompt decode, verification rows2/4/8, and eight chat
batch prefix sets.

The format writer's counter schema uses cumulative snapshots:
`SERVED MX COUNTS rank=R session=S at=prompt|end` with `mx_tc_step`,
`mx_tc_rows_stage`, `mx_tc_rows`, `mx_tc_tile`, and `mx_scalar_layer`.
Build `paths.rs` with `rustc +1.99.0 --edition=2024 -D warnings`.
It pairs snapshots within each fresh server window. It requires positive
end-minus-prompt step counts for plain decode and positive row-stage or
row counts for verification on both ranks. A positive lifetime prompt
count cannot prove decode. Any scalar layer count fails. The output head
is excluded from that count.

Build `quality.rs` the same way. It uses the repository JSON reader.
Pass histories, maker reference, main quality run, and candidate quality
run. It requires all2080 main-history positions, all208 maker positions,
and both cold endpoints of each history. Main must reproduce its histories.
A candidate needs at least99 percent agreement on main histories and no
lower maker score than the main run. Missing evidence fails. The gate
collects main quality when no complete run is supplied in
`GATE2_MAIN_QUALITY`.

Served tile, row and one-row wrong-word controls follow the baseline.
The served checker must catch token differences in each relevant rule.
Format device/twin sweeps and a caught sum-order control remain separate
required receipts. A served token pass alone cannot prove rules2/6.
READY also needs the final speed comparison, source checks and golden
handover. The script does not declare READY.

The `TCGATE-*` tools retain the previous night's partial gate. They are
references, not a complete proof of a new format. Their host checker tests
run with `python3 tools/tc-gate/TCGATE-test.py`.

Counter scope: prompt-tile counts prove entry coverage within each fresh
server window. They are cumulative and do not prove a delta for every
individual prompt. Decode and verification checks use per-session deltas
on each rank. The wired device proof and served token checks are separate
evidence for prompt correctness.

Quality runs after the first drafting part, before duplicate and control
runs. `GATE2_MAIN_COMMIT` selects the current main, and
`GATE2_QUALITY_COMMIT` can select that same word rebuilt on the current
main. The baseline commit remains immutable. A source rebuild is not a
new word; its served and golden checks still remain required.
