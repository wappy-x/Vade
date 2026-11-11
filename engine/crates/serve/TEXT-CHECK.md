# Checkpoint text probe

The [text-check program](src/bin/text-check.rs) checks pinned text vectors and
stored text edges. It does not start a decoder or admit a model.
The [token fixture](tests/fixtures/glm-text/source-vectors.json) pins all text
artifact hashes. Its ordinary token IDs come from public vocabulary and merge
data. They do not have an upstream runtime receipt.
The [published prompt test](src/tokenizer/glm_external_tests.rs) supplies a
separate external prompt-byte comparison.

## Text check

Build on a host with the pinned toolchain and current kernels. Run:

```sh
cargo +1.99.0 run --offline --locked --manifest-path engine/Cargo.toml -p spark-serve --bin text-check -- tokens .local/text-contract .local/token-results
```

The command checks each artifact hash before it constructs the tokenizer.
It compares fixed IDs, raw vocabulary bytes, decoded bytes, and streamed bytes.
The [fixture source note](tests/fixtures/glm-text/README.md) gives the derivation.
Run the retained artifact tests for the chat, special-token, and row checks.
Normal host tests never download tokenizer data.

## Stored edge check

The probe reads the config and complete index from an existing checkpoint.
It uses the same header admission and retained file descriptors as
[CheckpointTextEdges](src/text_checkpoint.rs). It does not change weight files.
Dimensions and stored types come from that admission.
The fixed token IDs come from the source fixture.

The host command widens each original BF16 embedding row.
It repeats the row across the configured residual streams.
It projects the original row through every stored head row.
This input is a primitive probe operand. It is not a decoder hidden state.
The [host reference](../gpu/src/text_reference.rs) follows the candidate
single-row kernel's FMA order and fixed lane tree.
It keeps the FP32 operand and the original MXFP8 weight and scale bytes.
The reference and candidate share no projection implementation.

```sh
./text-check host "$CHECKPOINT" host
./text-check device "$CHECKPOINT" host run1
./text-check device "$CHECKPOINT" host run2
./text-check device "$CHECKPOINT" host run3
./text-check compare host run1
./text-check compare run1 run2
./text-check compare run1 run3
```

Each device command starts a fresh process and CUDA context.
The `suite CHECKPOINT NEW_DIR` command runs these checks with child processes.
It also checks text artifacts and saves the device name and driver version.
It reads embedding bytes and complete head logits back from the device.
It saves all raw output planes before byte comparison.
A failed comparison keeps those planes and leaves no completion marker.
The probe checks exact bytes. It uses no tolerance or percentage band.

The device command derives its allocation ceiling from the admitted arena
and FP32 vector extents. It includes the resident norm copy, temporary IDs,
embedding output, input row, and complete logits. It counts these vectors
together as an upper bound. The probe does not configure a vector pool.
Admission also includes the existing pinned file slots and upload staging.
The sizes come from the allocators' shared constants. Staging growth includes
both the old and new buffer. Checked arithmetic refuses overflow.
The command refuses a total above Linux MemAvailable capped by CUDA total.
This check runs before the weight arena allocation. CUDA can still refuse
an allocation if memory changes after admission.
The budget log is an estimate. It does not prove device bytes or runtime.

The identity record binds the code revision, fixture, dimensions, and IDs.
It also binds config, index, and complete text tensor hashes.
The output manifest binds each input, embedding, logit, and selected ID file.
The compare command refuses absent completion markers and changed files.

Run the same prebuilt kit on each authorized Spark. Copy both result trees
to a host with the kit binary for that host, or to an authorized Spark.
Compare their host result folders and all device result folders.
This proves repeat and cross-device primitive bytes only.
The stored edges are replicated. This probe does not test tensor transport.

## Open model proof

The probe does not qualify the final norm, loaded dtypes, or head input policy.
It does not produce model answers or prove decoder token IDs.
It does not cover batched, cached, chunked, or drafted model requests.
The complete exactness gate and upstream tokenizer runtime parity remain open.
The imported launch safety contracts still need reviewer approval before merge.
Do not infer model admission or a device pass from host tests or a built kit.

Back: [text program](TEXT-PROGRAM.md).
