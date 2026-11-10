# licenses

**Full license texts for third-party material shipped in the repository.**

| File | Covers |
| --- | --- |
| [`Qwen3-8B-APACHE-2.0.txt`](Qwen3-8B-APACHE-2.0.txt) | Apache-2.0 text for the Qwen3-8B configuration, weight index and chat template; the Qwen3.6 chat-template fixture; and the Qwen3.5/Qwen3.6 configuration and tensor-header fixtures. |
| [`Llama-3.1-COMMUNITY.txt`](Llama-3.1-COMMUNITY.txt) | Meta Llama 3.1 Community License for `crates/tokenizer/templates/llama31.jinja`. |
| [`Llama-3.2-COMMUNITY.txt`](Llama-3.2-COMMUNITY.txt) | Meta Llama 3.2 Community License for `crates/tokenizer/templates/llama3.jinja`. |
| [`Phi-4-MIT.txt`](Phi-4-MIT.txt) | Microsoft Phi-4 MIT License for `crates/tokenizer/templates/phi4.jinja`. |
| [`Contributor-Covenant-CC-BY-4.0.txt`](Contributor-Covenant-CC-BY-4.0.txt) | Creative Commons Attribution 4.0 International for the repository’s Contributor Covenant 2.1 `CODE_OF_CONDUCT.md`. |
| [`../crates/tokenizer/UNICODE-NOTICE.txt`](../crates/tokenizer/UNICODE-NOTICE.txt) | Unicode License V3 and attribution for the generated Unicode character tables. |
| [`../LICENSE`](../LICENSE) | Apache-2.0 text for the Gemma 4 chat template, as identified by its recorded official model source. |

**How it fits:** [`NOTICE`](../NOTICE) lists every third-party item and its license text; [Provenance](../docs/provenance.md) records hashes. The Llama and Phi-4 templates are embedded by the Rust tokenizer and retained for that use.

The Llama 3.1, Llama 3.2 and Phi-4 license texts are exact copies of the official
downloads. `NOTICE` records each license source URL and SHA256. Their upstream
bytes are preserved, including the Phi-4 file's CRLF line endings and missing
final newline.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
