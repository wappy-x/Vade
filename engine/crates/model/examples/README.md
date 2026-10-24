# Model metadata examples

From `engine/`, inspect a ModelOpt mixed Flash checkpoint without any weights:

```sh
cargo run --offline -p engine-model --example flash-storage -- config.json tensor-map.json
```

[`flash-storage.rs`](flash-storage.rs) validates all tensor names, shapes and
types against the existing layer plan. It prints per-class stored bytes and
world-1 memory at 32K, 128K and 262K, with optional PLE residency and MTP.
The file map retains each tensor's source shard for later bounded payload reads.
GPU execution and on-demand PLE disk reads remain separate integration work.

To run the optional metadata regression, set `VADE_FP4_METADATA` to the directory
containing both JSON files and run `cargo test --offline -p engine-model --test
flash_mixed complete_checkpoint_metadata -- --ignored --exact`.

[Back to engine-model](../README.md)
