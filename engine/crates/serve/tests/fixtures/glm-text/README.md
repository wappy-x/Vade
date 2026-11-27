# GLM text contract sources

These are host checks. They do not run model weights or a device.
The [fixed source vectors](source-vectors.json) record prompt bytes and expected IDs.
They also select IDs for the [checkpoint edge probe](../../../TEXT-CHECK.md).
The [test](../../glm_text_contract.rs) keeps plain requests and chat requests
separate. Its ignored tests require complete public artifacts. Normal host
tests do not download files.

The source is `nvidia/GLM-5.3-Flash-FP4-Local` at revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.
Use the files at that revision. Do not use a prefix of `tokenizer.json`.

| Artifact | SHA256 |
| --- | --- |
| [tokenizer.json](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/tokenizer.json) | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| [tokenizer_config.json](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/tokenizer_config.json) | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| [generation_config.json](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/generation_config.json) | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |
| [chat_template.jinja](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/chat_template.jinja) | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` |

The test checks each hash before it loads the artifacts. The tokenizer hash
also equals the LFS object hash in the public checkpoint tree.
The [template notice](../../../../../NOTICE) gives the upstream revision and license.
No checkpoint file is copied into this test folder.

| Request or check | Expected source | Limit of proof |
| --- | --- | --- |
| Plain text to IDs | `model.vocab`, `model.merges`, `ignore_merges`, and the split pattern | Fixed cases, not an external tokenizer comparison |
| IDs to bytes and streamed UTF-8 | Vocabulary byte spellings | Fixed cases include a token that ends inside UTF-8 |
| Added tokens and skip flags | `added_tokens` content, ID, and special fields | Plain caller controls stay ordinary under the Vade request contract |
| Chat bytes and IDs | Template lines 1-3, 139-164, 255-260; vocabulary IDs | Bytes are read from the template by hand |
| Roles, thoughts, and escaped tool fields | Template lines cited in renderer tests | No external Jinja run |
| EOS list | `generation_config.json` `eos_token_id` | Host generator uses fixed logits, not model logits |
| Explicit stops | Request strings, UTF-8 boundaries, and expected visible prefix | Host stop buffer, not a checkpoint answer |

[derive.rs](derive.rs) reads the complete artifact with the metadata JSON
reader. It does not call the tokenizer crate. It takes fixed pre-token
boundaries from the saved regex. It maps whole vocabulary entries directly.
For other entries, it scans adjacent pairs by merge rank. This gives an
independent way to check the fixed ID arrays. It does not prove regex parity.

From the repository root, place the artifacts in `.local/text-contract/`.
Then run:

```sh
rustc +1.99.0 -D warnings engine/crates/serve/tests/fixtures/glm-text/derive.rs -o .local/text-contract/derive
.local/text-contract/derive .local/text-contract/tokenizer.json
RUSTFLAGS='-D warnings' cargo +1.99.0 test --offline --locked --manifest-path engine/Cargo.toml -p spark-serve --test glm_text_contract -- --ignored --nocapture
```

The [published prompt fixture](../glm-upstream/README.md) supplies external
expected prompt bytes. The [tokenizer evidence](../glm-tokenizer/README.md)
adds fixed publisher delimiter expectations and a complete source row inventory.
The source-derived vectors above retain their original proof label.

The open input is a versioned external tokenizer and Jinja result file for
the same requests and artifact hashes. It must include exact IDs, decoded
bytes, chat bytes, role order, explicit and inline thoughts, escaped tool
fields, Unicode, empty text, and stop outcomes. The current hand-read chat
vectors do not supply that external comparison. Real-artifact regression
vectors for the other supported tokenizer families remain separate proof.

GLM executable admission stays closed in the model plan. These checks do
not qualify GLM output parsing, model arithmetic, drafting, or tensor splits.
The chat request adapter does not expose every upstream template option.
In particular, its `clear_thinking` option remains at the template default.

Back: [server tests](../../README.md).
