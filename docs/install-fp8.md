# Install Flash-Next FP8 on 2, 4 or 6 Sparks

Vade serves two checkpoints of the same model. Both are supported:

| | FP4-Local | FP8 |
| --- | --- | --- |
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `Qwen/Qwen3.8-Flash-Next-FP8` |
| Sparks | 1 to 24 | 2, 4 or 6 |
| Two Sparks | The tensor split | Tensor parallel on a ring |
| Four Sparks | The layer chain | Tensor parallel on a path |
| Start command | `vade up` | `vade up --fp8` |
| Guide | [INSTALL.md](../INSTALL.md) | This page |

FP8 is the publisher's own FP8 checkpoint. It stores most weights as FP8 in
128 x 128 blocks. Its weights do not fit on one Spark. Our notes record
185,523,317,458 bytes on disk in 131 files, and 174,512,783,360 bytes of that
are FP8 weights
([Flash-Next architecture notes](../engine/docs/history/flash-next-architecture.md#fp8-storage-and-bf16-dequantization-with-fp32-promotion)).
So FP8 always runs on linked Sparks. With tensor parallelism, every Spark runs
every layer, and each Spark holds an equal slice of each layer. Vade accepts
2, 4 or 6 Sparks. The layer sizes divide evenly by 2 and 4. Six Sparks start,
but the slices are uneven, and we have not measured six.

A new FP8 start needs the `--fp8` flag. When Vade already has a recorded
launch, `vade up` with no options keeps that launch, FP8 included. With no
recorded launch, `vade up` with no flag starts FP4-Local. The older spelling
`--fp8-legacy` still works and does the same thing. The `--fp8` spelling is
new on main, in the v0.2.0 section of the [changelog](../CHANGELOG.md). A v0.1.0
checkout knows only `--fp8-legacy`.

## What you need

- 2, 4 or 6 DGX Sparks, each prepared with INSTALL
  [steps 1 and 2](../INSTALL.md#step-1-prerequisites). Each Spark needs the
  disk space for the full checkpoint above, plus the build.
- A cable between each pair of neighboring Sparks, with an address on each
  end. See [network setup](network-setup.md). Four Sparks need three cables
  in a line (a path). The cable that closes a ring of four is optional.
- A controller: a Linux or macOS computer that reaches each Spark over SSH. One
  of the Sparks can be the controller. Set it up with
  [Route B](../INSTALL.md#route-b-from-a-controller). `vade launch`, the
  one-Spark route, serves FP4-Local only.

## Build the engine

FP8 installs from source today. `vade install` on the controller stages the
source, builds the engine on each Spark and builds the CLI and chat program on
the controller.

No release bundle serves FP8 yet. The v0.1.0 launcher bundle runs only on one
Spark, and a controller ignores it
([bundle details](../cli/docs/prebuilt.md#prebuilt-binaries-for-a-controller-install)).
A controller uses prebuilt binaries only from a bundle of its own commit, set
with `prebuilt_dir`. That works for FP8 as for FP4-Local, because the same
engine binary serves both checkpoints.

## Put the weights on every Spark

Use publisher revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. The
[configuration template](../cli/examples/vade.conf) records the same
revision. Keep the complete snapshot: configuration, tokenizer, index and
every weight shard. Do not convert the files, and do not mix in FP4-Local
files. Keep the folder name `Qwen3.8-Flash-Next-FP8`. The engine serves the
model under that name in lower case, `qwen3.8-flash-next-fp8`.

To download, run this from the repository root on a computer with Rust
1.99.0, curl and network access:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 \
  236dfdf285828023ca3bcd3f37366c58a3469b13 \
  /absolute/path/to/Qwen3.8-Flash-Next-FP8
```

The helper checks every file before it finishes
([ops/fetch-model.sh](../ops/fetch-model.sh)). Copy the folder to the same
absolute path on every Spark. Then check each copy with the hash procedure in
[INSTALL step 3](../INSTALL.md#check-the-files), with an FP8 manifest you
trust. `vade install` and `vade up` do not download weights.

## Write the config

From the controller's checkout root, build the CLI and make a private config
from the FP8 template:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
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

From the controller, with idle GPUs, preview each step and then run it:

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
`qwen3.8-flash-next-fp8`, and the limits. Leave chat with `/exit`. Then run
`vade verify` to check that drafting ON and OFF give the same token IDs, and
`vade down` to stop the engine.

The published cards used context 32768 and output 16384. To start with the
same limits:

```sh
vade up --fp8 --context 32768 --output 16384
```

## Start on four Sparks

```sh
vade up --fp8 --sparks 4
```

This takes the first four machines in config order. To name them in cable
order instead, use `--hosts a,b,c,d`. The four Sparks form one engine on a
path. [Four Sparks](four-sparks.md) shows how to recable two pairs into one
group of four.

## RDMA or TCP

Linked Sparks use RDMA by default. That needs the RDMA device, its user
library and an unlimited locked-memory limit in the SSH session. For TCP, set
this before `vade install` and keep it set for `up` and `doctor`:

```sh
export SEQPAR_RDMA=0
```

The CLI accepts TCP for FP8 on every supported Spark count. The published
cards used RDMA.

## Defaults and drafting

The FP8 defaults are context 131072, output `min(65536, context)` and drafting
depth 2 ([CLI parser](../cli/src/cli.rs)). We have not yet checked the
131072 default on real Sparks; the cards used 32768. If the engine has not
enough memory for a context, it refuses to start, and you choose a smaller
`--context`. The controller does not lower it for you.

- `--context N` accepts 4096 to 262144.
- `--output N` accepts 1024 to 262144, up to the context.
- `--mtp 0` turns drafting off. `--mtp 1` to `--mtp 3` sets the depth.
- A request with `"draft": "plain"` decodes without drafting.

FP8 drafts only when one chat runs alone. Chats that overlap decode without
drafting until they finish. The
[CLI reference](../cli/README.md#fp8-limits-and-drafting) has the details.

**Send temperature 0 and neutral penalties for greedy decoding.** FP8 is
different from FP4-Local here. A request to the FP8 server that leaves out a
sampling field takes that field from the checkpoint's own sampled settings
(top_k 20, top_p 0.95, from its `generation_config.json`). That holds even
when the request sends no sampling fields at all. Without `seed`, the engine
draws a new seed for each sampled request, so the answer can change from run
to run. A request with `temperature` 0, `repetition_penalty` 1,
`presence_penalty` 0 and `frequency_penalty` 0 is greedy. The token-ID
promises on this page apply to greedy requests only. This request sends the
full greedy set:

```sh
curl -s http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"enable_thinking":false}'
```

The gateways add the greedy fields to a request that sends no sampling field
([OpenAI-compatible API](openai-api.md)).

## What to expect

The two-Spark rows are the 2026-10-29 cards at main `cbbb11b`, the mean of
two server starts
([benchmarks](benchmarks.md#current-2026-10-29-main-cbbb11b-home-dgx-sparks)).
The four-Spark rows are the 2026-10-28 end-of-day cards at main `76977df`
([benchmarks](benchmarks.md#fp8)); four Sparks were not measured again.
Rates are completion tokens per second. Cold 8K is the time to the first
token after an uncached 8192-token prompt. Every card used context 32768,
output 16384, MTP 2, greedy decoding and thinking OFF.

| Sparks | Main | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `cbbb11b` | ON | 36.5 | 51.4 | 52.6 | 45.0 | 67.6 | 97.3 | 5.96 |
| 2 | `cbbb11b` | OFF | 26.9 | 26.7 | 26.9 | 26.1 | 66.7 | 97.4 | 5.97 |
| 4 | `76977df` | ON | 46.7 | 65.1 | 68.0 | 58.5 | 86.4 | 128.8 | 6.91 |
| 4 | `76977df` | OFF | 35.3 | 34.7 | 35.3 | 33.9 | 88.2 | 128.2 | 6.95 |

On two Sparks at `cbbb11b`, a 32K-token prompt gave its first token in
24.05 s. The second question on that chat reused 32,064 tokens from the
prefix cache and gave its first token in 0.89 s.

- **Four Sparks write faster than two.** Each Spark holds and reads a quarter
  of each layer for each token, not a half. With drafting ON, in the
  2026-10-28 cards at `76977df`, prose went from 36.6 tok/s on two Sparks to
  46.7 on four, and eight chats from 98.3 to 128.8 tok/s.
- **Drafting helps one chat only.** With drafting ON, code went from 26.7 to
  51.4 tok/s on two Sparks. Overlapping chats almost never draft. In the
  2026-10-28 two-Spark ON card, 1 of 12 C4 streams and 0 of 24 C8 streams
  used drafting. In the four-Spark ON card, no C4 or C8 stream used drafting.
  So ON and OFF are close (97.3 and 97.4 tok/s for C8 on two Sparks at
  `cbbb11b`).
- **FP8 reads prompts more slowly than FP4-Local.** At main `cbbb11b`, the
  cold 8K prompt took 5.96 s on two FP8 Sparks, against 2.145 s on two
  FP4-Local Sparks.

The cards started the engine with our lab launcher, not with `vade up`. That
launcher gives each rank the same engine arguments as
`vade up --fp8 --context 32768 --output 16384 --mtp 2`, with the same topology,
transport and output settings. A CLI test checks that plan on two and four
Sparks (`fp8_card_settings_reach_every_rank_on_two_and_four_sparks` in
[cli/src/app/tests.rs](../cli/src/app/tests.rs)). We have not yet run a card
on a server that `vade up --fp8` started.

## Exactness on FP8

On each Spark count, FP8 is stable for greedy requests. In the 2026-10-28
cards, which sent every sampling field, drafting ON and OFF gave the same
token IDs in 66 of 66 responses, on two and on four Sparks. The cards also
matched earlier FP8 cards of the same Spark count. On two Sparks at main
`cbbb11b`, drafting ON and OFF gave the same token IDs in 64 of 64
responses, in each of two server starts.

**Two and four Sparks give different token IDs.** The two-Spark and
four-Spark cards matched in only 7 of 66 responses (5 of 62 on 2026-10-25).
The rule says that supported Spark counts must agree, so this is an exception.
A host test found the cause: on four Sparks, some projections split their
inner sum across the ranks, and the ranks add their parts in a different order
than on two Sparks. A fix is in screening on Sparks. Until it is on main,
compare FP8 results only at the same Spark count.
[Benchmarks](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)
has the details.

## Not tested yet

- Nobody has run these steps end to end on blank machines.
- No card has measured six Sparks.
- No card has measured the 131072-token default context.

More on this route:

- [Fresh install](fresh-install.md) has a longer step-by-step checklist for
  two Sparks.
- [Network setup](network-setup.md#fp8) lists the FP8 layouts by Spark count.
- INSTALL's [client gateway](../INSTALL.md#client-gateway) section works the
  same way. Use the model ID `qwen3.8-flash-next-fp8` in it.
