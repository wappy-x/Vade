# GLM tokenizer evidence

The [host checks](../../support/glm_tokenizer_evidence.rs) use the existing
tokenizer and artifact loader. They add no tokenizer implementation.
The [artifact note](../glm-text/README.md) pins the complete checkpoint files.
The files stay in ignored `.local/text-contract/`. Tests do not download them.

## Independent expected values

The source is NVIDIA's
[frontend test](https://github.com/NVIDIA/spark-local-inference/blob/18b16e170bb9eab03e2b1800107c72966c56594f/helios/tests/glm_frontend.rs).
[Its retained copy](nvidia-glm-frontend.rs.txt) is unchanged.
Its SHA256 is `e71c9bdf028b9fbd454dc3aae7ab3d399eee6e57acfff15136498f8fd75f856e`.
The [selected expectations](publisher-values.json) retain its delimiter pairs
and EOS list. Vade does not generate these expected values.

| Expected ID | Expected bytes | Source |
| --- | --- | --- |
| `154841` | `<think>` | Publisher frontend test, `delimiters` assertion |
| `154842` | `</think>` | Publisher frontend test, `delimiters` assertion |
| `154820,154827,154829` | EOS set | Publisher frontend test, `ModelConfig` |

The publisher's
[recipe](https://github.com/NVIDIA/spark-local-inference/blob/18b16e170bb9eab03e2b1800107c72966c56594f/ignition/recipes/glm53/recipe.py#L76-L78)
pins the same NVIDIA checkpoint revision as this test.
The [retained contract](nvidia-glm53.json) also supplies its vocabulary row count.
Its SHA256 is `8b1204a0db52a35dc8f984dffb36419c4f58374cb32d082a40c076bdc2540825`.
The source uses [Apache License 2.0](../../../../../licenses/NVIDIA-spark-local-inference-APACHE-2.0.txt).
The [notice](../../../../../NOTICE) retains the copyright holders.

These are fixed publisher test expectations. They are not a publisher runtime receipt.
The publisher test is ignored by default. Its full prompt IDs come from an external generated file.
That result file is absent from the pinned public tree.
The [generator source](https://github.com/NVIDIA/spark-local-inference/blob/18b16e170bb9eab03e2b1800107c72966c56594f/tests/frontend/generate_glm_fixtures.py)
computes those results with Transformers. This job does not run that program.

The [vLLM fixture](../glm-upstream/README.md) independently supplies prompt bytes.
It supplies no real-checkpoint ID arrays. Its checks keep their exclusions explicit.
The local test now also checks the refusal class and the local error reason.

## Complete source inventory

The row inventory remains source-derived. It reads the complete tokenizer metadata.
It does not split text, apply merges, or run an alternate encoder.
It maps each byte-alphabet spelling directly to raw vocabulary bytes.
The explicit ID sets each row. JSON member order does not set row order.
The test compares every mapped row with the existing tokenizer's bytes and special flag.
It also compares whole and streamed decode bytes with source-derived UTF-8 replacement bytes.

The inventory test's `SOURCE INVENTORY` and `SOURCE DECODE` output supplies these values:

| Item | Raw value |
| --- | --- |
| Base vocabulary IDs | `0..154820`, exclusive end; `154820` entries |
| Added-token IDs | `154820..154856`, exclusive end; `36` entries |
| Special added tokens | `18` |
| Non-special added tokens | `18` |
| Mapped IDs | `154856` |
| Embedding, head, and scale rows | `154880` |
| Unmapped tensor rows | `154856..154880`, exclusive end; `24` rows |
| Raw rows that are not valid UTF-8 alone | `1095` |
| Merge ranks in source order | `321649` |
| `ignore_merges` | `true` |

The test prints each added ID, special flag, and spelling as `ADDED ROW`.
It refuses token-byte and decode reads for every unmapped tensor row.
The general tokenizer permits sparse IDs. These trailing rows have no token mapping.
This inventory does not prove that model execution masks them.
It does not prove semantic embedding or head row contents.

[row-headers.json](row-headers.json) selects metadata from the retained public header ledger.
The source ledger is in Vade commit `23550f94172b449404257d2525173b9219553556`,
at `engine/crates/model/tests/fixtures/glm-mtp/public-headers.json`.
Its SHA256 is `ab6f391ed40f28f7d6211fa8535c6197cdf71c2b17c1ce465ecb49412ee2947b`.
The selected file's SHA256 is `86332684a9ccc25ab203bf54b6f530827ab5dd16a79b27948805f8f7a462a0d5`.
The original header JSON SHA256 is `04ddcf8298ad1cc3301de02637070c0a4b70f03e8be5a42d48f3387cf50fe013`.
That header belongs to `model-00003-of-00040.safetensors` at checkpoint revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.
The checkpoint card names the NVIDIA Open Model Agreement.
This job reads no tensor payload and fetches no model weights.

The raw-row digest encodes ID as little-endian `u32`, special as `u8`,
byte length as little-endian `u32`, then raw bytes. It skips unmapped rows.
Its source-derived SHA256 is `bedb1bc06709b65c1b3565067c6439c520f81ced2167766fa9fbb588746ae1c4`.
The decode digest encodes ID and decoded length as little-endian `u32`, then UTF-8 bytes.
Its source-derived SHA256 is `d3a07fb8ec56532686a5e458a6e35d65d6ee2d4861640318848a58d685d6e390`.
The inventory test pins both values.

## Refusals and repeat command

The normal host suite checks duplicate rows, row collisions, missing row coverage,
out-of-range rows, and changed added-row order in the evidence inventory.
These checks do not add production admission rules.
The tokenizer's synthetic
[mapping checks](../../../../tokenizer/tests/bpe_profiles.rs)
test precise loader errors for duplicate keys, duplicate IDs, bad integer IDs,
missing bytes, conflicting added tokens, and malformed merge mappings.
Each mutation starts from a valid artifact and changes a specific field.

From the repository root, with the complete pinned artifacts retained, run:

```sh
RUSTFLAGS='-D warnings' cargo +1.99.0 test --offline --locked --manifest-path engine/Cargo.toml -p spark-serve --test glm_text_contract -- --include-ignored --nocapture
```

Ordinary-text expected IDs remain source-derived. No pinned external ordinary-text result file was found in the inspected trees.
No upstream tokenizer or Jinja runtime runs here. Runtime provenance and complete tokenizer parity remain open.
Model admission stays deferred. Precision and predictor-source decisions stay open.
No device exactness, model quality, or speed result follows from these host checks.

Back: [server tests](../../README.md).
