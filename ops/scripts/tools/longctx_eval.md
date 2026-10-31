# Long-context saved-input evaluation

Use `vade test longctx` to prepare frozen inputs, capture teacher-forced logits
and retrieval answers, and compare saved captures. The [CLI evaluation reference](../../../cli/evals.md)
documents the exact options, defaults, output schema and pass criteria.
The implementation is in [cli/src/evals](../../../cli/src/evals/mod.rs).

## Prepare inputs

Build the engine tokenizer and the CLI from the repository root:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path engine/Cargo.toml -p engine-tokenizer --example tokenize
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
```

Set `TOKENIZER_DIR` to the pinned checkpoint's tokenizer directory,
`FROZEN_BODIES` to the directory containing the retained `pp448.body` and
`code.body`, `IFEVAL_SAMPLES` to the retained instruction inputs, and
`EVAL_OUTPUT` to a new private output directory. The repository does not ship
private request captures. Use trusted saved inputs and keep generated files
outside Git.

```sh
cli/target/release/vade test longctx prepare \
  --tokenizer-command engine/target/release/examples/tokenize "$TOKENIZER_DIR" \
  --frozen-bodies "$FROZEN_BODIES" --ifeval-inputs "$IFEVAL_SAMPLES" \
  --output "$EVAL_OUTPUT/prompts"
```

Preparation records source and tokenized-input digests in its manifest.
Synthetic long documents and retrieval cases are diagnostics, not an official
quality score or unmodified natural long-form text. Run/compare require the
pinned suite and checkpoint identity described in the CLI reference.

## Capture and compare

Prepare idle servers with the same checkpoint, topology, effective request
settings and sufficient context capacity. Verify full checkpoint payloads and
binary/PTX identities separately; header identity alone does not prove them.
Pass the actual build identities explicitly instead of accepting the tool's
historical build defaults.

Use `run --a-url` and `--b-url` for reference and candidate servers, with
`--c-url` for the perturbed-reference floor when applicable. Each arm retains
its own provenance and captured rows. `compare --a-dumps`, `--b-dumps` and
optional `--c-dumps` consume existing captures without contacting a server.

Keep every input position and required rank. Missing, mismatched, truncated or
nonfinite rows are failures. A missing floor or reference retrieval misses
prevent a complete PASS. Preserve partial captures and reported reasons;
do not substitute successful subsets for the requested comparison.

The recurrent-state command, `vade test gdn-state`, is separate. Its `--identity`
mode compares raw bytes, including signed zero. Its numerical and floor modes
have different criteria; consult the [option and result tables](../../../cli/evals.md)
before interpreting a result.

Host tests establish parser and comparison behavior. Native model correctness,
quality and performance need their own complete capture receipts.
