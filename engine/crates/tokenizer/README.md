# engine-tokenizer

**This crate turns text into token IDs and back, for Qwen3, Qwen3.8 and
Gemma 4, and builds chat prompts the way each model's official chat template
does.** It is written from scratch in Rust and runs on the CPU, with no
neural model involved.

This page is for people reading or changing tokenization or chat formatting.

<details>
<summary><b>Terms on this page</b>: token, BPE, special token, chat template, normalization</summary>

<dl>
<dt>Token</dt>
<dd>A piece of text the model reads as one unit. Each has a number, its ID.</dd>
<dt>BPE</dt>
<dd>Byte-pair encoding: text is split into small pieces, then the most common
pairs are merged by a ranked list.</dd>
<dt>Special token</dt>
<dd>A control marker such as <code>&lt;|im_start|&gt;</code>, which starts a
chat turn.</dd>
<dt>Chat template</dt>
<dd>The model maker's recipe for laying out a conversation as one prompt.</dd>
<dt>Normalization</dt>
<dd>Rewriting Unicode text into one standard form before splitting it. NFC is
the common composed form.</dd>
</dl>

</details>

## Where it fits

**The server uses it for every chat request; its only dependency is the JSON
reader.**

<dl>
<dt><a href="../model/README.md">engine-model</a></dt>
<dd>The bounded JSON reader.</dd>
<dt><a href="../serve/README.md">spark-serve</a></dt>
<dd>Turns each chat request into token IDs, and streams the answer back as
text.</dd>
<dt><a href="../../tools/bench/README.md">vade-bench</a></dt>
<dd>Sizes benchmark prompts exactly.</dd>
</dl>

## Test it

**The unit tests need no GPU and no model. The `tokenize` example shows the
IDs for any text.**

### Do it yourself

1. Run the unit tests.

   ```sh
   cargo +1.99.0 test -p engine-tokenizer
   ```

2. Try the `tokenize` example. It reads one JSON object per line and prints
   the result. Pass a `tokenizer.json` file for tokenizer-only use, or a model
   folder to also validate its chat template.

   ```sh
   cargo +1.99.0 run --release -p engine-tokenizer --example tokenize -- \
     models/Qwen/Qwen3-8B/tokenizer.json
   ```

3. Type one of these lines and press Enter:

   ```json
   {"text":"Hello"}
   {"ids":[9707]}
   {"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false}
   ```

### Ask your coding agent

```text
In the Vade checkout, run cargo +1.99.0 test -p engine-tokenizer and report
the result. Then run the tokenize example from crates/tokenizer/README.md
against models/Qwen/Qwen3-8B/tokenizer.json with the input {"text":"Hello"}
and show the printed IDs. Do not change code, commit or push.
```

<details>
<summary><b>Notes on the <code>tokenize</code> example</b></summary>

- For tool definitions in order, add `"tools_json":["SERIALIZED_JSON_OBJECT"]`.
- The example sorts the keys of object arguments. For byte-exact comparisons,
  pass tool-call arguments as JSON strings.

</details>

## What it does

**Loads the official tokenizer, splits and joins text, and formats chats as
the official template does.**

- Loads the official `tokenizer.json` and checks the model's chat template.
- Splits text into tokens and decodes token IDs back to text, also while an
  answer is streaming.
- Formats chats, tool calls and thinking as the official template does.

<details>
<summary><b>How the three model families differ</b></summary>

| Item | Qwen3 | Qwen3.8 (hybrid) | Gemma 4 |
| --- | --- | --- | --- |
| Splitting | Byte-level BPE | Byte-level BPE | Character BPE |
| NFC normalization | Yes | Yes | No |
| Template check | Byte for byte | Required markers | Byte for byte |
| Tool calls | Supported | Supported | Supported |
| Special tokens in caller text | Plain text | Plain text | Plain text |

- **Splitting.** Qwen3.8 keeps Unicode marks inside words. Gemma 4 uses
  character BPE with byte fallback.
- **Normalization.** Qwen3 uses NFC, as Qwen's tokenizer specifies, and
  Qwen3.8 does the same. Gemma 4 has no NFC; only ASCII spaces change, into
  the space marker U+2581.
- **Template check.** Qwen3 is checked byte for byte against the official
  Qwen3 template; Qwen3.8 against the required markers of the official
  template; Gemma 4 byte for byte against Google's canonical template.
- **Tool calls.** Qwen3.8 renders tool definitions, assistant function calls
  and grouped tool responses using the hybrid template's XML-style format.
- **Special-token spellings in caller text** stay plain text for Qwen3,
  Qwen3.8 and Gemma 4.

</details>

## Use it from Rust

**Use `chat::encode` for chat inference.**

```rust,no_run
use engine_tokenizer::{Tokenizer, chat::{self, Message, Role, Options}};
let tokenizer = Tokenizer::from_directory("MODEL_DIRECTORY")?;
let options = Options { enable_thinking: false, ..Default::default() };
let ids = chat::encode(&tokenizer, &[Message::new(Role::User, "Hello")], &options)?;
# Ok::<(), engine_tokenizer::Error>(())
```

<details>
<summary><b>All public functions</b></summary>

<dl>
<dt><code>Tokenizer::from_directory</code></dt>
<dd>Loads <code>tokenizer.json</code> and checks supported chat templates.
Other byte-level artifacts support text only.</dd>
<dt><code>Tokenizer::from_file</code></dt>
<dd>Loads <code>tokenizer.json</code> only, with no chat checks.</dd>
<dt><code>encode</code></dt>
<dd>Text to IDs. Special-token spellings in the text become special tokens, as
in the official tokenizer.</dd>
<dt><code>encode_ordinary</code></dt>
<dd>Text to IDs, with special-token spellings treated as plain text.</dd>
<dt><code>chat::encode</code></dt>
<dd>A chat to IDs. <b>Use this for chat inference</b>.</dd>
<dt><code>chat::format</code></dt>
<dd>Shows the formatted prompt as text, for inspection only.</dd>
<dt><code>stream_decoder</code></dt>
<dd>Decodes IDs one at a time while an answer streams.</dd>
<dt><code>contains_id</code></dt>
<dd>Tells whether an ID exists in the tokenizer's vocabulary.</dd>
</dl>

</details>

## Text-only byte-level artifacts

The loader reads normalization, digit splitting and `ignore_merges` from the
artifact fields. Existing Phi and Llama artifacts keep their prior digit group
size, including artifacts with a single-digit pattern.
A null normalizer keeps the input bytes. A ByteLevel
post-processor changes offsets only. It does not add a space to the text.
Direct vocabulary lookup stays inside each piece from the pre-tokenizer.

### The case-aware split pattern

A byte-level artifact can carry a split pattern that cuts words where the
letter case changes. `HelloWorld` gives `Hello` and `World`. `HTTPServer`
stays one piece. A contraction such as `'s` stays with its word. Numbers are
in groups of 1 to 3. A symbol run takes the line breaks and slashes that
follow it.

The loader selects this rule when the pattern text of the artifact equals
`CASED_PATTERN` in `src/lib.rs`. It reads no model name and no folder name.
The rule is in `src/cased_split.rs`. Another pattern is refused with
`unsupported split pattern`. The same pattern in an artifact that is not
byte-level is refused too, because the rule was checked for the byte-level
form only.

An artifact with this pattern is text-only here: `encode`, `encode_ordinary`,
`decode` and streaming decode. A renderer must send caller text through
`encode_ordinary`. Then a control spelling in caller text stays plain text.

`from_directory` records the path to `generation_config.json`.
`eos_token_ids()` reads its `eos_token_id` field on first access and caches the
result. The value can be an ID or a nonempty list. Each ID must exist in the
tokenizer vocabulary. The method returns the full list in file order or an
error for invalid settings. These errors do not prevent tokenizer discovery.
Missing settings keep the default for supported chat families.
Other artifacts have an empty list until settings are loaded.
`eos_token_id()` keeps the legacy chat EOS ID for supported chat families.
For text-only artifacts, it returns the first configured ID, or `u32::MAX`
for an empty or invalid list.
The file and JSON loaders do not read companion settings.

Text-only artifacts can use `encode`, `encode_ordinary`, `decode` and streaming
decode. Their chat calls return an error until a renderer supports the template.

## Safety of chat prompts

**For chat encoding, only markers the template inserts become control
tokens.** Caller text, thinking, tool arguments and tool definitions stay
plain text, even when they spell a control token. This holds for every
supported family, including Phi and Gemma 4.

This protects the prompt's structure. It does not stop a message from trying
to talk the model into something (prompt injection).

<details>
<summary><b>Chats and tools</b></summary>

Messages:

- Roles: `system`, `user`, `assistant` and `tool`. Content is text.
- Assistant messages can carry thinking and several tool calls.
- Tool responses can be grouped, as the template expects.

Tool calls:

- `ToolCall.arguments` is a JSON string, used exactly as given after a strict
  check. Pass object arguments through `json_for_template` first.
- Tool definitions keep their key order, as the official template does.
- Malformed arguments, duplicate JSON keys and unsafe function names are
  rejected with an error.

Callers must reject unknown roles and images themselves. The tokenizer never
runs a tool.

</details>

<details>
<summary><b>Streaming decode</b></summary>

1. `stream_decoder(skip_special)` creates a decoder. It holds back only the
   bytes of a character that is not finished yet.
2. `push(id)` returns any complete text.
3. `finish()` ends the stream and replaces an unfinished character with the
   Unicode replacement character.

The pieces joined together equal `decode` for the same IDs.

The model's output layer has a few more slots than the tokenizer has IDs.
`contains_id` lets the engine skip those padding slots.

</details>

<details>
<summary><b>Built-in limits</b></summary>

| Item | Default limit |
| --- | ---: |
| Input text | 1 MiB |
| Tokens per call | 1,048,576 |
| Decoded text | 4 MiB |
| `tokenizer.json` file | 32 MiB |
| Vocabulary entries, merges | 1,000,000 each |
| Added tokens | 4,096 |
| Formatted chat | 1 MiB |

Callers should set smaller limits for requests. The tokenizer does not treat
`model_max_length` as the model's capacity; the engine enforces its own
context limit.

</details>

<details>
<summary><b>Unicode data, and how to regenerate the tables</b></summary>

Normalization uses Unicode 12.1 data, the version the official `tokenizers`
0.23.2 library reproduces for Qwen3. Character categories use Unicode 16.

<dl>
<dt><code>UNICODE-NOTICE.txt</code></dt>
<dd>Unicode license and data sources.</dd>
<dt><code>QWEN-NOTICE.txt</code></dt>
<dd>License and source of the Qwen3 chat template.</dd>
<dt><code>templates/qwen3.jinja</code></dt>
<dd>The official Qwen3 template, used only to check the Rust formatter against
it.</dd>
<dt><code>templates/gemma4.jinja</code></dt>
<dd>Google's canonical Gemma 4 template, used to validate model folders.</dd>
<dt><code>src/unicode_data.rs</code></dt>
<dd>Generated tables.</dd>
<dt><code>tools/generate_unicode.rs</code></dt>
<dd>The table generator.</dd>
</dl>

The Rust formatter does not run Jinja.

To regenerate the tables:

1. Download the three official files named in `UNICODE-NOTICE.txt`. Name them
   as in the command in step 3.
2. From the repository root, build the generator. It shares the SHA-256 code
   in `tools/export/` to record each input's checksum.

   ```sh
   mkdir -p target
   rustc --edition=2024 -O crates/tokenizer/tools/generate_unicode.rs \
     -o target/generate_unicode
   ```

3. Write the tables.

   ```sh
   ./target/generate_unicode UnicodeData-16.0.0.txt UnicodeData-12.1.0.txt \
     CompositionExclusions-12.1.0.txt crates/tokenizer/src/unicode_data.rs
   ```

</details>

## How it was checked

**Qwen3 and Gemma 4 tokenization matched the official tooling on every
recorded case.**

- **Qwen3.** No mismatch against the official `tokenizers` 0.23.2 library and
  the official Jinja template. This check ran before Qwen3.8 and Gemma 4
  support existed.
- **Gemma 4.** When support was added, the `tokenize` example matched the
  official tooling on 12 tokenization, 8 chat and 4 decoding cases, plus one
  Qwen regression case.
- **Qwen3.8.** No separate tokenizer case count is recorded.
- **The case-aware split pattern.** No mismatch against the `tokenizers`
  0.23.3 library. The counts are in the details below.

These checks cover the tokenizer and template only. They say nothing about
model quality or serving.

<details>
<summary><b>The Qwen3 case counts</b></summary>

| Cases | Count |
| --- | ---: |
| Tokenization | 1,985 |
| Extra Unicode cases | 11,139 |
| Decoding of arbitrary IDs | 1,006 |
| Chat-template formatting | 23 |

The decoding cases ran in both special-token modes and streamed.

</details>

<details>
<summary><b>The case counts of the case-aware split pattern</b></summary>

The checkpoint was Inkling 2 Mini. The SHA-256 of its `tokenizer.json` starts
with `7f332727f9a3`. The reference was the `tokenizers` 0.23.3 library with
the same file. The checks ran on 2026-11-22.

| Check | Cases | Equal |
| --- | ---: | ---: |
| Token IDs and decode, program text and prose | 2,493 | 2,493 |
| Token IDs and decode, strings built from pattern edges | 20,000 | 20,000 |
| Split pieces of the 20,000 edge strings | 123,096 | 123,096 |
| Control spellings kept as text by `encode_ordinary` | 60 | 60 |
| Character class of each code point above U+007F | 1,111,936 | 1,111,936 |

- The first, second and fourth rows are the `PARITY` line that the ignored
  test `tests/inkling_host.rs` prints. The test reads the tokenizer file and
  the corpus from the ignored folder `.local/cased-tokenizer/`. They are not
  in the repository.
- The third row came from the unit test `pieces_equal_the_reference_vectors`
  in `src/cased_split.rs`, run once with the large set in place of its
  fixture. The fixture in the repository has 1,135 strings and 2,967 pieces.
- The fifth row compared the cuts of the reference library on five probe
  strings for each code point with the class from `UnicodeData-16.0.0.txt`.
  The unit test `class_tables_cover_every_code_point` pins the tables to
  that file.

Token IDs alone do not prove a split rule. In one trial a wrong word rule
gave the reference IDs on all 22,493 strings. The comparison of the pieces
found it.

</details>

## Gemma 4

**`from_directory` validates the folder's `chat_template.jinja` against
Google's canonical template.** Like Qwen, Gemma keeps special-token
spellings in caller content as plain text. Only the template's own markers
become control tokens. This differs from Google's official pipeline, which
recognizes those spellings in caller content.

<details>
<summary><b>How Gemma text is split, and what the chat API supports</b></summary>

How Gemma text is split:

- Character BPE and byte fallback, through the same merge engine as Qwen.
- Only ASCII spaces change, into the space marker U+2581. No NFC
  normalization and no prefix space are added.
- Raw encoding adds no beginning-of-text token. The chat template writes
  `<bos>` itself.

How to use it:

<dl>
<dt><code>tokenizer.chat_options()</code></dt>
<dd>The model's defaults. Gemma thinking is off.</dd>
<dt><code>tokenizer.format_chat(...)</code></dt>
<dd>Inspect a formatted prompt.</dd>
<dt><code>chat::encode(...)</code></dt>
<dd>Get the token IDs.</dd>
</dl>

Supported: string content, developer and system turns, reasoning, function
declarations, calls, and OpenAI-style tool responses. Not supported in the
text `Message` API: multimodal content and legacy embedded `tool_responses`.

Byte-fallback decoding buffers consecutive byte tokens until the next
ordinary token or the end of the stream. Joined streamed output equals
whole-string decoding, including malformed byte groups.

</details>

**Where next:** [spark-serve](../serve/README.md) · [Server and API](../../docs/server.md) ·
[Provenance](../../docs/provenance.md)
