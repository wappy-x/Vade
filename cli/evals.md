# Evaluation commands

Build the CLI with `cargo build --offline --manifest-path cli/Cargo.toml`.
`vade test longctx` freezes prompts, captures teacher-forced logits and retrieval
answers, or compares saved captures. `vade test gdn-state` compares complete
recurrent state dumps. These commands work without lifecycle configuration.
Generated prompts, dumps and reports belong in private runtime storage.

The Rust ports preserve the reference tools' arithmetic order, JSON field order
and result formats. Greedy requests omit the legacy sampling seed as described
below. The former Python implementations are retired from the repository; their
originals and saved-input parity evidence remain private. Host replay checks establish
tool behavior; they do not establish inference correctness or performance.

## Long-context options

Use `vade test longctx` for the long-context evaluator.
The `prepare`, `run` and `compare` subcommands keep their names.

| Reference option | Rust equivalent | Default and meaning |
| --- | --- | --- |
| `prepare --tokenizer-command COMMAND [ARG ...]` | Same option | Required; existing persistent CPU tokenizer JSON-lines protocol |
| `prepare --frozen-bodies DIR` | Same option | Required; frozen `pp448.body` and `code.body` |
| `prepare --ifeval-inputs PATH` | Same option | Required; distinct IFEval samples, captured request JSON lines, or an instruction directory |
| `prepare --output DIR` | Same option | Required; new directory for bodies and manifest |
| `prepare --documents N` | Same option | 4; integer 4 through 8 |
| `prepare --seed N` | Same option | 1963583967; dataset-generation seed, including signed and large integers within the reference decimal parser limit; not a request sampling seed |
| `prepare --keys-per-depth K` | Same option | 1; integer 1 through 8 |
| `prepare --checkpoint-hash HASH` | Same option | Pinned digest by default; custom preparation digests are rejected by run/compare |
| `prepare --max-tokens N` | Same option | 256; at least 64 times K |
| `run/compare --manifest FILE` | Same option | Required; pinned `vade.longctx.v1` manifest |
| `run/compare --top-k K` | Same option | 8; integer 1 through 32 |
| `run/compare --floor-factor F` | Same option | 1.5; finite and at least 1 |
| `run/compare --report FILE` | Same option | Optional; new JSON receipt, identical to stdout plus newline |
| `run --a-url URL` | Same option | Required; reference server base |
| `run --b-url URL` | Same option | Required; candidate server base |
| `run --c-url URL` | Same option | Optional; perturbed reference floor server |
| `run --output DIR` | Same option | Required; new A/B/C arm directories |
| `run --timeout SECONDS` | Same option | 3600; socket timeout for HTTP |
| `run --api-key-env NAME` | Same option | Optional; environment variable for a Bearer key |
| `run --a-build SHA` | Same option | `7a4ad78a`; operator assertion saved in provenance |
| `run --b-build SHA` | Same option | `caf433b`; historical default retained for receipt compatibility |
| `run --c-build SHA` | Same option | `7a4ad78a+scoring+layer0-ulp`; operator assertion |
| `compare --a-dumps DIR` | Same option | Required; reference saved NDJSON or native dumps |
| `compare --b-dumps DIR` | Same option | Required; candidate saved dumps |
| `compare --c-dumps DIR` | Same option | Optional; perturbed reference saved dumps |
| `-h`, `--help` | Same options | Command names and help text describe the Rust entry point |

Pass current verified build identities explicitly when capturing. The retained
historical defaults describe the original experiment and do not identify a
current binary. The checkpoint default is
`8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03`.

| Reference output | Rust equivalent |
| --- | --- |
| `prepare` READY JSON on stdout | Same fields and field order: status, teacher_documents, retrieval_cases, manifest |
| `manifest.json` and `CASE.body.json` | Same frozen text, record/query ordering, token counts, source digests and prompt/input digests |
| A/B/C `provenance.json` | Same checkpoint, capability identity, asserted build and verification text |
| `CASE.score.jsonl` | Same header additions and response row bytes; validated before reuse |
| `CASE.score.partial` after failed capture | Same retained partial-file convention |
| A/B `CASE.reply.json` | Same case ID, prompt digest, text, usage and effective sampling fields |
| `captured ARM CASE` on stderr | Same progress line after each successful capture |
| Comparison JSON on stdout and optional report | Same status, reasons, checkpoint, metric, pooled/document buckets and retrieval fields |
| Empty upper buckets at 32k | Same null agreement/difference values |
| PASS/PARTIAL/FAIL and exit codes | 0 for PASS/PARTIAL, 1 for a measured FAIL, 2 for execution errors |
| Execution diagnostic on stderr | `FAIL` plus a validation message or exception category; malformed-stream diagnostics can differ from Python tracebacks |

Truncated NDJSON with missing rows or a missing terminal record exits 2 in
Rust (`FAIL StopIteration`) and 1 in the Python reference (an uncaught
`RuntimeError` traceback). Treat any nonzero exit as failure;
this accepted difference does not change the comparison decision.

Preparation makes teacher documents at exactly 32,768 and 131,072 tokens and
48 retrieval cases: four archives per length, five depth questions and one
multi-key question per archive. Seeded labels, shuffled query order and exact
budgets stay frozen. Retrieval trims only outer whitespace and grants no
substring credit. Responses must return every value in the requested order.

Every generated body includes temperature 0, top_p 1, top_k 0, repetition penalty 1,
presence/frequency penalties 0, plain drafting, cache off and thinking off.
Greedy request bodies omit `seed`. Capture also removes a legacy greedy body's
sampling seed before sending retrieval or scoring requests, without modifying
the frozen file or prompt. The `prepare --seed` value remains in manifest metadata
and controls dataset generation, including labels and query order.
Retrieval captures verify the server's sampling echo, token count and runtime mode.
Teacher rows retain every input position, their top-k candidates and A-aligned
raw logits. Binary dumps remain little-endian u32 IDs and complete row-major
FP32 vocabulary rows. Comparisons stream rows and preserve Python's compensated
summation within each row, followed by the same bucket accumulation order.

The six half-open buckets end at 4k, 8k, 16k, 32k, 64k and 128k. PASS requires
candidate errors within the configured floor and flatness multiplier and no
retrieval loss. A missing floor or reference retrieval misses prevent PASS.
Duplicate cases, incomplete coverage, mismatched IDs, nonfinite logits and
missing aligned candidates fail validation.

Teacher-logit comparisons accept world 2 or world 4, with the same world on
both arms (and the floor arm). Keep topology, ordered rank assignment, official
precision and runtime settings fixed. World-four captures are separate from
two-Spark goldens; accepting their metadata is host tooling support, not native
quality qualification.

HTTP uses Rust's standard library. HTTPS uses the CLI's existing `curl` runtime;
credentials travel through stdin rather than command arguments. HTTPS timeout
handling uses curl's connection and low-speed bounds, so it needs separate
runtime qualification. Base URLs and keys stay out of receipts.

See the preserved [long-context procedure](../ops/scripts/tools/longctx_eval.md)
for experiment provenance and the observer's runtime requirements. Its earlier
build references apply to that experiment.

## GDN state options and output

Use `vade test gdn-state` for recurrent-state comparisons. Supply completed session directories.

| Reference option/input | Rust equivalent | Meaning |
| --- | --- | --- |
| `fp32` | First positional directory | Unperturbed reference |
| `candidate` | Second positional directory | Candidate or FP32 control; perturbed reference in floor mode |
| `split` | Optional third directory | Enables the perturbed-floor comparison |
| `--identity` | Same option | Exact state bytes for two unperturbed arms with matching arithmetic |
| `--legacy-prefill-group N` | Same option | Verified group for old manifests; positive multiple of 64 |
| `-h`, `--help` | Same options | Updated command name in usage/help |
| `--` | Same separator | Subsequent arguments are directory names |

| Reference output | Rust equivalent |
| --- | --- |
| Position, layer, head, world and path summary | Same text and ordering |
| `layer=... head=...` and pooled statistics | Same elements, state bytes, reference magnitude, absolute/relative errors and threshold counts |
| FP32 control diagnostics | Same additional requirement for zero absolute error |
| `comparison=bit-identity` and per-file `identity` rows | Same raw-byte comparison, including signed zero |
| Floor-mode raw head rows | Same floor/split maximum, normalized error, squared sums and state bytes |
| Per-layer floor table and diagnostic rows | Same medians, maxima, relative Frobenius values and ratios; all 36 layers are judged, raw head ratios are diagnostic |
| Closing PASS/FAIL line | Same metric, bound and explanation; exit 0 for pass, 1 for fail |
| Invalid dump diagnostic | Same `FAIL invalid dump:` validation messages; exit 1 |
| Invalid CLI input | Usage and error, exit 2; usage names the Rust command |

The tool checks all 36 recurrent layers and 48 heads, every rank, contiguous
head ownership, complete prompt prefixes and exact file sizes. It does not
infer checkpoint identity from old manifests. Verify that identity using the
original capture receipts. The ordinary bound is 1e-3 normalized head error;
floor mode uses twice the measured floor in every recurrent layer. Zero and
undefined floors retain their reference behavior.

## Unix flag ABI check

```sh
vade test unix-flags-abi
vade test unix-flags-abi --target aarch64-unknown-linux-gnu
```

This standard-library command is the Unix flag ABI check.
It invokes `rustc --target TARGET --crate-type lib --emit metadata` on the
flag definitions embedded when the CLI was built, with compile-time assertions
for all twelve constants against an independent audited ABI table. Rebuild the
CLI after changing its flag definitions. The table follows Linux v6.12's ARM64
and generic UAPI file flags and macOS XNU's file flags, cited in
[unix_flags.rs](src/unix_flags.rs). Before removing the shell, host and Linux
ARM64 assertions were paired with its pinned libc 0.2.189 reference.
No Cargo dependency, C cross compiler, linker or target execution is needed.
The selected compiler's Rust target libraries must already be installed.
Temporary source and metadata are removed when the command finishes.

`cargo test --offline --manifest-path cli/Cargo.toml constants_match_host_system_headers`
also compiles a temporary C program with `cc`, reads the host's system headers,
and compares all twelve constants with the CLI values at runtime.

| Input/output | Rust equivalent |
| --- | --- |
| No arguments, host ABI assertions | `vade test unix-flags-abi`; defaults to `rustc -vV`'s host |
| `CARGO` override | `RUSTC` selects the Rust compiler for the new command |
| Cargo flags such as `--offline` | Build the CLI with Cargo; the check itself uses rustc and never fetches dependencies |
| `--target aarch64-unknown-linux-gnu` compile-only check | Same option; rustc selects Linux ARM64 cfgs even on a Mac |
| Cargo success/failure | Exit 0 with `PASS` for twelve assertions; exit 2 for compilation or argument errors |

Audited targets are `aarch64-unknown-linux-gnu`, `x86_64-unknown-linux-gnu`,
`aarch64-apple-darwin` and `x86_64-apple-darwin`; other targets are rejected.
Compile-only ABI checks do not validate Linux file-operation behavior.
