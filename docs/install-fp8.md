# Install Flash-Next FP8 on 2, 4 or 6 Sparks

Vade serves two checkpoints of the same model: FP4-Local and FP8. Vade
supports both, and this page installs FP8.

| | FP4-Local | FP8 |
| --- | --- | --- |
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `Qwen/Qwen3.8-Flash-Next-FP8` |
| Sparks | 1 to 24 | 2, 4 or 6 |
| Measured in our cards | 1, 2 and 4 Sparks | 2 and 4 Sparks |
| Two Sparks | The tensor split | Tensor parallel on a ring |
| Four Sparks | The tensor split in quarters (RDMA) | Tensor parallel on a path |
| Start command | `vade up` | `vade up --fp8` |
| Guide | [INSTALL.md](../INSTALL.md) | This page |

FP8 is the publisher's own FP8 checkpoint. It stores most weights as FP8 in
blocks of 128 x 128. Our notes record 131 files with 185,523,317,458 bytes on
disk. Of those bytes, 174,512,783,360 are FP8 weights
([Flash-Next architecture notes](../engine/docs/history/flash-next-architecture.md#fp8-storage-and-bf16-dequantization-with-fp32-promotion)).

The weights do not fit on one Spark, so FP8 always runs on linked Sparks. FP8
uses tensor parallelism: every Spark runs every layer and holds an equal slice
of it. Vade accepts 2, 4 or 6 Sparks. The layer sizes divide evenly by 2 and
by 4. Six Sparks start, but their slices are uneven. We have not measured six
Sparks.

To start FP8 for the first time, give `vade up` the `--fp8` flag. When Vade
already has a recorded launch, `vade up` with no options keeps that launch,
and that includes FP8. With no recorded launch, `vade up` with no flag starts
FP4-Local.

The older flag `--fp8-legacy` still works and does the same thing. The
`--fp8` flag is new on main, and the v0.2.0 section of the
[changelog](../CHANGELOG.md) lists it. A v0.1.0 checkout knows only
`--fp8-legacy`.

## What you need

- **Sparks.** You need 2, 4 or 6 DGX Sparks. Prepare each one with INSTALL
  [steps 1 and 2](../INSTALL.md#step-1-prerequisites). Each Spark needs disk
  space for the full checkpoint above and for the build.
- **Cables.** Each pair of neighboring Sparks needs a cable, with an address
  on each end ([network setup](network-setup.md)). Four Sparks need three
  cables in a line. Vade calls this layout a path. The cable that closes a
  ring of four is optional.
- **A controller.** The controller is a Linux or macOS computer that reaches
  each Spark over SSH. One of the Sparks can be the controller. Set it up with
  [Route B](../INSTALL.md#route-b-from-a-controller). `vade launch` is the
  one-Spark route, and it serves FP4-Local only.

## Build the engine

FP8 installs from source today. On the controller, `vade install` stages the
source and builds the engine on each Spark. It also builds the CLI and the
chat program on the controller.

No release bundle serves FP8 yet. The v0.1.0 launcher bundle runs only on one
Spark, and a controller ignores it
([bundle details](../cli/docs/prebuilt.md#prebuilt-binaries-for-a-controller-install)).

A controller uses prebuilt binaries only from a bundle of its own commit. You
set that bundle with `prebuilt_dir`. This works for FP8 as it does for
FP4-Local, because the same engine binary serves both checkpoints.

## Put the weights on every Spark

Use the publisher's revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. The
[configuration template](../cli/examples/vade.conf) records the same
revision.

Keep the complete snapshot: the configuration, the tokenizer, the index and
every weight shard. Do not convert the files. Do not mix in FP4-Local files.
Keep the folder name `Qwen3.8-Flash-Next-FP8`. The engine serves the model
under that name in lower case, `qwen3.8-flash-next-fp8`.

To download the weights, run this command from the repository root. The
computer needs Rust 1.99.0, curl and network access:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 \
  236dfdf285828023ca3bcd3f37366c58a3469b13 \
  /absolute/path/to/Qwen3.8-Flash-Next-FP8
```

The helper checks every file before it finishes
([ops/fetch-model.sh](../ops/fetch-model.sh)). Copy the folder to the same
absolute path on every Spark. Then check each copy with the hash steps in
[INSTALL step 3](../INSTALL.md#check-the-files). Use an FP8 manifest that you
trust. `vade install` and `vade up` do not download weights.

## Write the config

On the controller, go to the checkout root. Build the CLI and make a private
config from the FP8 template:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
mkdir -m 700 -p .local
[ -e .local/fp8.conf ] || cp cli/examples/vade.conf .local/fp8.conf
export VADE_CONFIG="$PWD/.local/fp8.conf"
```

Edit `.local/fp8.conf` the same way as INSTALL's
[edit the config](../INSTALL.md#edit-the-config) step:

- Set the local source, stage and state folders.
- Delete the `build_manifest` line, or set it to `build.conf` inside your
  private state folder. Without the line, Vade keeps the file at
  `state_dir/build.conf`. Do not keep the `/path/to/...` placeholder.
- Keep `model = qwen3.8-flash-next-fp8`, its `model_repo` and its pinned
  `model_revision`.
- In each `[machine ...]` section, set `ssh` and the absolute `model_path`.
  Every Spark needs the same expanded checkpoint path.
- For each cable, put a `fabric.<peer>` line in both machine sections. Its
  value is that machine's own address on the link to the peer.
- For four Sparks, add two more machine sections. List the machines in cable
  order.

## Start on two Sparks

Make sure that the GPUs are idle. Then, on the controller, preview each step
and run it:

```sh
vade install --dry-run
vade install
vade up --fp8 --dry-run
vade up --fp8
vade status
vade chat
```

`vade up --fp8` starts on the first two machines in the config. The two
Sparks form one engine on a ring. `vade status` shows the model,
`qwen3.8-flash-next-fp8`, and the limits. Leave chat with `/exit`.

Then run `vade verify`. It checks that drafting on and off give the same token
IDs. Drafting is Vade's speculative decoding: the model's own prediction head
guesses a few tokens ahead, and the full model checks each guess. After
that, run `vade down` to stop the engine.

The published cards used context 32768 and output 16384. To start with the
same limits, run this command:

```sh
vade up --fp8 --context 32768 --output 16384
```

## Start on four Sparks

```sh
vade up --fp8 --sparks 4
```

This command takes the first four machines in config order. To name them in
cable order instead, use `--hosts a,b,c,d`. The four Sparks form one engine on
a path. [Four Sparks](four-sparks.md) shows how to recable two pairs into one
group of four.

## RDMA or TCP

Linked Sparks use RDMA by default. RDMA needs the RDMA device, its user
library and an unlimited locked-memory limit in the SSH session. For TCP, set
this variable before `vade install`. Keep it set for `up` and `doctor`:

```sh
export SEQPAR_RDMA=0
```

The CLI accepts TCP for FP8 on every supported Spark count. The published
cards used RDMA.

## Defaults and drafting

The FP8 defaults are context 131072, output `min(65536, context)` and drafting
depth 2 ([CLI parser](../cli/src/cli.rs)). We have not yet checked the
131072 default on real Sparks. The cards used 32768.

If the engine does not have enough memory for a context, it refuses to start.
Then choose a smaller `--context`. The controller does not lower the context
for you.

- `--context N` accepts 4096 to 262144.
- `--output N` accepts 1024 to 262144, up to the context.
- `--mtp 0` turns drafting off. `--mtp 1` to `--mtp 3` sets the depth.
- A request with `"draft": "plain"` decodes without drafting.

FP8 drafts only when one chat runs alone. Chats that overlap decode without
drafting until they finish. The
[CLI reference](../cli/README.md#fp8-limits-and-drafting) has the details.

**FP8 samples by default.** FP4-Local does not. When a request to the FP8
server leaves out a sampling field, the server takes that field from the
checkpoint's own sampled settings (top_k 20, top_p 0.95, from its
`generation_config.json`). This is also true when the request sends no
sampling fields at all. Without `seed`, the engine draws a new seed for each
sampled request, so the answer can change from run to run.

**Send temperature 0 and neutral penalties for greedy decoding.** A request
with `temperature` 0, `repetition_penalty` 1, `presence_penalty` 0 and
`frequency_penalty` 0 is greedy. The token-ID promises on this page apply to
greedy requests only. This request sends the full greedy set:

```sh
curl -s http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"enable_thinking":false}'
```

The gateways add the greedy fields to a request that sends no sampling field
([OpenAI-compatible API](openai-api.md)).

## What to expect

These rows come from the morning cards of 2026-10-31 at main `dc32b7c`. Each
Spark count had one server start. Each start ran a card with drafting on, then
a card with drafting off ([results and receipts](results.md)).

The rates are completion tokens per second. Cold 8K is the time to the first
token after an uncached prompt of 8192 tokens. Every card used context 32768,
output 16384, MTP (drafting depth) 2, greedy decoding and thinking off.

| Sparks | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | ON | 36.6 | 51.3 | 52.8 | 45.4 | 66.6 | 98.7 | 5.97 |
| 2 | OFF | 27.1 | 26.9 | 27.1 | 26.1 | 66.3 | 98.1 | 5.95 |
| 4 | ON | 49.5 | 69.3 | 72.2 | 61.4 | 88.0 | 131.0 | 6.92 |
| 4 | OFF | 35.9 | 35.6 | 36.0 | 34.6 | 87.2 | 131.1 | 6.87 |

A 32K-token prompt gave its first token in 24.12 s on two Sparks and 27.81 s
on four. The second question on that chat reused 32,064 tokens from the
prefix cache and gave its first token in 0.94 s on two Sparks and 1.06 s on
four.

- **Four Sparks write faster than two.** Each Spark holds and reads a quarter
  of each layer for each token, not a half. With drafting ON, prose went from
  36.6 tok/s on two Sparks to 49.5 on four, and eight chats from 98.7 to
  131.0 tok/s.
- **Drafting helps one chat only.** On two Sparks at `dc32b7c`, code went
  from 26.9 tok/s with drafting OFF to 51.3 with drafting ON. Overlapping
  chats almost never draft. In the `dc32b7c` two-Spark ON card, 1 of 12 C4
  streams and 0 of 24 C8 streams used drafting. In the four-Spark ON card,
  no C4 or C8 stream used drafting (the `draft_used` field of each
  stream). So ON and OFF are close (98.7 and 98.1 tok/s for C8 on two
  Sparks at `dc32b7c`).
- **FP8 reads prompts more slowly than FP4-Local.** At `dc32b7c`, the cold
  8K prompt took 5.97 s on two FP8 Sparks, against 2.15 s on two FP4-Local
  Sparks.

The cards started the engine with our lab launcher, not with `vade up`. The
lab launcher gives each rank the same engine arguments as
`vade up --fp8 --context 32768 --output 16384 --mtp 2`, with the same topology,
transport and output settings. A CLI test checks that plan on two and four
Sparks (`fp8_card_settings_reach_every_rank_on_two_and_four_sparks` in
[cli/src/app/tests.rs](../cli/src/app/tests.rs)). We have not yet run a card
on a server that `vade up --fp8` started.

## Exactness on FP8

Exactness means that the same request, with the same settings, gives the same
token IDs. On each Spark count, FP8 is stable for greedy requests.

The 2026-10-31 cards at `dc32b7c` sent every sampling field. In those cards,
drafting on and off gave the same token IDs in 66 of 66 responses, on two
Sparks and on four Sparks. The 2026-10-28 cards gave the same result. The
cards also
matched earlier FP8 cards of the same Spark count. On two Sparks at main
`cbbb11b`, drafting on and off gave the same token IDs in 64 of 64
responses, in each of two server starts.

**Two and four Sparks give the same token IDs from main `504f885` on.** Before
that main, the two-Spark and four-Spark cards matched in only 7 of 66
responses on 2026-10-28, on 2026-10-30 at `380d8e5` and on 2026-10-31 at
`dc32b7c` (5 of 62 on 2026-10-25). A host test found the cause: on four
Sparks, some projections split their inner sum across the ranks, and the ranks
added their parts in a different order than on two Sparks. Main `504f885`
makes four Sparks add in the two-Spark order. Six Sparks still give other
token IDs than two, so compare six-Spark FP8 results only with six Sparks.
[Benchmarks](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)
has the details.

## Not tested yet

- Nobody has run these steps end to end on blank machines.
- No card has measured six Sparks.
- No card has measured the 131072-token default context.

## More on this route

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [Fresh install](fresh-install.md) | A longer step-by-step checklist for two Sparks |
| [Network setup](network-setup.md#fp8) | The FP8 layouts by Spark count |
| [Client gateway](../INSTALL.md#client-gateway) | The INSTALL gateway steps, which work the same way for FP8 |

</div>

In the client gateway steps, use the model ID `qwen3.8-flash-next-fp8`.
