# Greedy chat proof client

`first-tokens` uses the shared tokenizer and benchmark HTTP client.
The same commands work under `vade bench --first-tokens`.
The client adds no model execution path.
It does not start or stop a server.

Build on the host:

```sh
cargo build --offline --locked --release --manifest-path engine/tools/bench/Cargo.toml --bin first-tokens
```

Ship the prebuilt target binary for a device check.
Do not run a compiler on a Spark.
The kit must include `SHA256SUMS` and `RUN.txt`.

## Prepare

Set `MODEL` to an immutable checkpoint directory.
Put the user message in `user.txt` as UTF-8.
Preserve its exact bytes across all runs.
Then run:

```sh
./first-tokens prepare "$MODEL" user.txt prepared
```

The command creates a new directory. It refuses to replace one.
It writes the rendered template, prompt IDs, request, and prepared manifest.
The manifest pins the config and tokenizer file hashes.
The checkpoint's template controls chat formatting.
Unsupported templates fail before any HTTP request.
No weights enter this step.

The proof size comes from `TOKENS` and `REPEATS` in
[`first_tokens.rs`](src/first_tokens.rs).
Capture requests eight output tokens. `run` is an alias for `capture`.
The request asks to disable thinking, cache reuse, and drafting.
The checkpoint template controls the prompt. Some templates retain a thinking prefix.
It uses explicit greedy sampling and ignores EOS for the fixed output length.

## Capture

Start every rank from the same candidate payload.
Use the engine's normal tensor split and resident decoder.
Set `URL` to the local chat endpoint. A base URL or `/v1` also works.
Set `SHA` to the full server commit.
Set `CHECKPOINT_SHA` to the independent checkpoint hash expected in `/health`.
Set `SPARK_COUNT` to the rank count.

```sh
./first-tokens capture "$URL" "$MODEL" prepared/prepared.json run1.json "$SHA" "$CHECKPOINT_SHA" "$SPARK_COUNT"
```

The client checks server identity before and after the request.
It requires a clean build, checkpoint hash, kernel hashes, and the expected rank count.
It also requires a positive server start time.
It checks prompt usage against the local prompt IDs.
Equal prompt lengths alone do not prove equal prompt IDs inside the server.
The normal service tokenizer supplies the chat prompt IDs.

The receipt keeps all output IDs, decoded token text, and reply text.
Capture also prints the output IDs and their little-endian u32 byte SHA256.
The reply text can omit chat control markup that remains in the decoded text.
The receipt also keeps settings, usage, and raw client rates.
The client prints tokens per second. This is no speed claim.

A failed reply writes a failed receipt when the server identity remains readable.
A setup or health error exits before it writes a receipt.
The client refuses to replace an existing receipt.

Stop the owned processes after each capture.
Restart every rank, then capture `run2.json` and `run3.json`.
Keep each rank's start, stop, and completion logs.
Server start times prove only that the observed service start changed.
Peer logs must prove fresh peer processes.
Do not treat new client processes as fresh engine processes.

## Compare

```sh
./first-tokens compare run1.json run2.json run3.json
./first-tokens compare run1.json run2.json run3.json --first-id host-first.id
./first-tokens compare run1.json run2.json run3.json --expected-ids host-output.ids
```

The first command leaves first-step host parity open.
The second command checks the first output ID against an independent host reference.
The file must contain the host's first greedy ID for the same prompt and checkpoint.
Do not use an embedding/head primitive result as a decoder reference.
The last command checks every output ID against an independent host sequence.
Put one unsigned token ID per line in `host-output.ids`.
This file must contain the complete requested output sequence.

Comparison requires identical IDs, decoded text, reply text, requests, and stable server identity.
The comparison output includes that same token byte hash.
It also requires distinct service start times.
It checks the saved greedy and draft echoes again.
It excludes timing from exact comparisons.

A repeat pass covers only this request and topology.
It does not close the cross-topology, batch, cache, chunk, or drafting gate.
The station must supply device logs and any host reference proof.
If the resident decoder is absent, post the exact gap and retain this driver as a partial kit.

## Host check

```sh
./first-tokens --self-test
cargo test --offline --locked --manifest-path engine/tools/bench/Cargo.toml first_tokens
```

The self-test uses a byte tokenizer fixture and local mock replies.
It checks template controls, fixed output length, text decode, repeat failures, and input pins.
These host checks prove no GPU arithmetic, transport, or model output.
