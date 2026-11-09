# Official Qwen metadata fixtures

These are the supplied public Qwen3.6-35B-A3B-FP8 and Qwen3.5-122B-A10B-FP8 metadata, without weights. Configs are minified without changing values. Header fixtures preserve every tensor name, shape and dtype; offsets are replaced by shape-derived byte lengths in tests.

A header group replaces dot-separated numeric name components with `{0}`, `{1}`, etc. Each axis is either a count (IDs `0..count`) or an explicit ID list. Expand the Cartesian product. Groups were formed from actual records, and each product was checked against the original names. No inventory is generated from the planner's expectations. The two fixtures expand to 64,196 and 76,656 tensors, including vision and MTP. Total JSON fixtures: 72,000 bytes.

`source` records SHA-256 of the original config and index bytes. `headers_sha256` hashes sorted header filenames, each followed by a NUL byte and its original bytes. `stored_bytes` sums the original tensor offset lengths. The derivation checked every name against its index shard and checked there were no duplicate or missing names.

Run `cargo test --offline -p engine-model --test qwen_metadata -- --nocapture` from `engine`. Set `VADE_QWEN_METADATA` to the supplied metadata directory to additionally compare every expanded record and config with the originals. CI always exercises both complete fixture inventories and all requested split counts and contexts.

`mixed-plan-baseline.txt` stores plan bytes from base revision
`99addd660f1a9d34d529a2690df7470b3d180e4b`. The
`plan::mixed_tests::qwen_fixture_plans_stay_byte_identical` test builds the same
plans and compares their Debug bytes with this file. It covers the checked-in
config fixtures above. It also covers the existing synthetic Flash fixture
with reconciled BF16, FP8 and mixed FP4-Local inventories. Flash has no BF16
config adapter. That case sets the existing typed BF16 format before it builds
the inventory. These plans do not prove device exactness.
