# Provenance

**Vade's engine is its own Rust; a few data files are reused unchanged, with
their license terms identified below.** No adapted third-party code is declared.

This page records each reused file's source, hash and notice, and what
informed the work without supplying code. For maintainers and license
reviewers.

[Credits](CREDITS.md) is the full list of people, papers and projects whose
ideas Vade uses, with main and development scope marked. The parts designed
and built in this project are listed under
[Original work](../../docs/CREDITS.md#original-work).

<details>
<summary><b>Terms on this page</b>: provenance, revision, SHA-256 hash, checkpoint, chat template, fixture, Unicode Character Database</summary>

<dl>
<dt>Provenance</dt>
<dd>Where a file came from, and on what terms.</dd>
<dt>Revision</dt>
<dd>One exact version of a repository, named by a commit hash.</dd>
<dt>SHA-256 hash</dt>
<dd>A fingerprint of a file's bytes; any change alters it.</dd>
<dt>Checkpoint</dt>
<dd>A model's published files: config, tokenizer and weights.</dd>
<dt>Chat template</dt>
<dd>The maker's format for turning a chat into the model's input.</dd>
<dt>Fixture</dt>
<dd>Saved reference data that checks compare Vade's output to.</dd>
<dt>Unicode Character Database</dt>
<dd>The official tables that describe every character.</dd>
</dl>

</details>

## In short

**Nothing outside Vade is built into it except data files, each listed below
with its hash.**

- Vade's engine, kernels, tokenizer, server and tools are written for this
  project in Rust. The exceptions are listed under
  [Tooling outside the engine](#tooling-outside-the-engine).
- No outside Rust crate and no community inference engine is built in.
- A few data files are reused unchanged, with their licenses kept.
- No adapted third-party code is declared.

## Vade's own code

**The Rust model reader, GPU driver declarations and ownership code, Rust GPU
kernels and repository tooling were written for this project.**

These are trusted outside tools, not code Vade claims as its own:

- Rust's official compiler `rustc`, Cargo and the `std` and `core` libraries.
- LLVM, which Rust uses to produce GPU code.
- NVIDIA's driver and its just-in-time PTX compiler.
- The host linker and the operating system's runtime.

## Reused data files

**Four kinds of file are reused as data: Qwen3-8B metadata, five chat
templates, Unicode tables and reference outputs.** Each opens below with its
source and hash.

<details>
<summary><b>Qwen3-8B metadata</b>: config and weight index, Apache-2.0</summary>

`crates/model/evidence/qwen3-8b-config.json` and `qwen3-8b-index.json` are
unmodified Qwen/Qwen3-8B model metadata, used to test the loader. Their bytes
match the official distribution at revision
`b968826d9c46dd6066d109eabc6255188de91218`.

<dl>
<dt>config</dt>
<dd><code>f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30</code></dd>
<dt>index</dt>
<dd><code>f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc</code></dd>
</dl>

- **Source.** [Qwen3-8B at that revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218),
  which identifies the Apache-2.0 license.
- **License.** Kept unmodified at `licenses/Qwen3-8B-APACHE-2.0.txt`.
- **Not bundled.** Model weights and the full vocabulary and merge files.
- **Limit.** Matching metadata says nothing about whether a downloaded weight
  file is authentic.

</details>

<details>
<summary><b>Chat templates</b>: Qwen3 and Gemma 4, with hashes</summary>

All templates are in `crates/tokenizer/templates/`. Llama 3.1/3.2 and Phi-4 sources, hashes and terms are listed in [NOTICE](../NOTICE); their bundled bytes match the retained official tokenizer metadata.

<dl>
<dt><code>qwen3.jinja</code></dt>
<dd><code>a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8</code></dd>
<dt><code>gemma4.jinja</code></dt>
<dd><code>ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4</code></dd>
</dl>

- **`qwen3.jinja`** is Qwen3's chat template, copied byte for byte from the
  official Qwen3-8B `tokenizer_config.json` at the same revision.
- **`gemma4.jinja`** is Google's canonical Gemma 4 template, kept for
  model-folder validation.
- Vade's Rust formatter is authored code. No Jinja template engine runs.
- **`tests/qwen3.6.jinja`** (in `crates/tokenizer/`) is the official
  Qwen3.6-35B-A3B template, kept as a test fixture
  (`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`). The
  tests rebuild the official Qwen3.5 and Qwen3.8 templates from it by clause
  and check their published hashes; see [NOTICE](../NOTICE).
- The Qwen3 template's copied bytes were compared with the official revision.
  Its Apache license is kept in `crates/tokenizer/QWEN-NOTICE.txt`.
- [Gemma 4 31B's official page](https://huggingface.co/google/gemma-4-31B-it)
  is the source of the model, tokenizer and template. The recorded model
  metadata identifies Apache-2.0. Attribution is kept in [NOTICE](../NOTICE)
  and the license text in [LICENSE](../LICENSE).

Model-maker credits and the official assistant's behaviour are listed in
[Credits](CREDITS.md).

</details>

<details>
<summary><b>Unicode tables</b>: generated from the official files, with input hashes</summary>

`crates/tokenizer/src/unicode_data.rs` holds reused Unicode data, generated
from the official files. It is not an authored Unicode database.

- **Generator.** `crates/tokenizer/tools/generate_unicode.rs`. The generation
  algorithm and the normalization code are authored.
- **License.** Kept in `crates/tokenizer/UNICODE-NOTICE.txt`.
- **Check.** The official input files were downloaded separately and their
  hashes checked. Regenerating the table reproduces all checked-in table data.

<dl>
<dt><a href="https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt">16.0
categories</a></dt>
<dd><code>ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f</code></dd>
<dt><a href="https://www.unicode.org/Public/12.1.0/ucd/UnicodeData.txt">12.1
normalization</a></dt>
<dd><code>93ab1acd8fd9d450463b50ae77eab151a7cda48f98b25b56baed8070f80fc936</code></dd>
<dt><a href="https://www.unicode.org/Public/12.1.0/ucd/CompositionExclusions.txt">12.1
composition exclusions</a></dt>
<dd><code>abc8394c5bde62453118b00c1c5842160a04d7fffb2e829ee5426b846596d081</code></dd>
</dl>

The category and normalization versions differ on purpose. They match the
behaviour of the reference tokenizer that Vade's tokenizer is checked
against. This is not a claim to implement every behaviour of a current
Unicode library.

</details>

<details>
<summary><b>Reference outputs</b> in <code>fixtures/</code></summary>

- The expected outputs in `fixtures/` come from official Hugging Face
  Transformers runs, on CPU and on CUDA.
- The newer BF16 CUDA suites are kept separate from the older dense FP32
  fixtures.
- [Validation](validation.md) records versions, manifests, missing revision
  metadata and every strict-agreement failure.
- Reused fixtures do not mean that every Vade answer matches the official
  implementation.

</details>

## Adapted code

**None is declared.** Any later adaptation or reuse must add, before export:

- the source URL and exact revision;
- the paths and changes made;
- the license and any retained notices.

Private development does not waive upstream license obligations. Original
code written for an idea first seen elsewhere is credited as an idea, not
labelled as copied code.

## Ideas and references consulted

**These informed what to evaluate; their code does not run inside Vade, and
their speed or quality claims are not Vade results.**

<details>
<summary><b>The pinned references</b>, and what each informed</summary>

**Community projects**

- **Kindling Spark OS** ([revision 5a8129d](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e)).
  Pinned stack records, observability and conditional memory experiments. No
  OS, allocator or watchdog code imported.
- **TensorFold**, by the TensorFold contributors ([revision 17c73e1](https://github.com/ashhart/TensorFold/tree/17c73e189f5e6a5304cda7ea37f086f9c49b4788)).
  Serial reference comparisons, bounded draft work and per-stream state, as
  evaluation ideas. Its [final-only prompt logits behaviour](https://github.com/ashhart/TensorFold/blob/17c73e189f5e6a5304cda7ea37f086f9c49b4788/src/tensorfold/families/qwen4_exp/cuda/decode.py#L283)
  informed the evaluation of skipping output-head work on middle prompt rows.
- **b12x** ([revision a321f9a](https://github.com/local-inference-lab/b12x/tree/a321f9a6e32f4119486c0ea9dd961595ffc87daf)).
  Keeping reference correctness separate from speed qualification, and
  explicit scratch ownership.
- **Reederey cache-tail analysis** ([revision fd87969](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/blob/fd87969cb43a5ba66573bd2508bc05651fc1579c/docs/17-apc-tail-floor.md)).
  Reachable cache boundaries and cache-resume regression requirements.

**Official references**

- **[Official Rust NVPTX support](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html)**
  and the **[NVIDIA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)**.
  Compiler and binary-interface contracts.
- **[Hugging Face Transformers Qwen3, v4.51.3](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py)**.
  Model semantics. The source was read; no implementation code is
  incorporated.

**About TensorFold specifically**

- The single-token `Session::advance` transaction and its validation runner
  are independently written Rust.
- No TensorFold code was copied or executed.
- TensorFold's performance results are not Vade measurements.

More generally, community claims about speed, quality, OS memory or transport
are not inherited as Vade results. A proposed optimization needs its own
attribution and matching correctness and performance evidence.

</details>

## Tooling outside the engine

**Two kinds of tool are not authored Rust, and neither runs in the engine.**
The runtime, GPU kernels, model adapters, tokenizer, server, export tool,
Unicode generator and kernel build tool (`cargo kernels`, in
`crates/gpu/build-kernels`) are authored Rust.

- **Privately retained Python reference generators** produced the saved
  reference data with the official Transformers code. They are kept outside
  this repository and never run in the engine. See
  [reference provenance](../tools/truth/README.md).
- **Earlier private Python and C numerical mocks** were test evidence. They
  are not part of the runtime or the shareable source.

**Where next:** [Credits](CREDITS.md) · [Export](export.md) ·
[Validation](validation.md)
