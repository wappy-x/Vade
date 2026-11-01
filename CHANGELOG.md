# Changelog

This file lists the changes you would notice when you use Vade, newest first.
Changes that are merged but not yet released are under Unreleased. A short
revision such as `380d8e5` or a number such as #142 names a revision or pull
request of the private development history; this repository does not hold
them ([glossary](docs/GLOSSARY.md#ho), "Main `abc1234`").

## Unreleased

- `scripts/install.sh` tries a prebuilt bundle of main first: it checks Ubuntu 24.04 or newer, glibc and the NVIDIA driver, downloads `latest.json` and the bundle, checks the SHA-256 hashes, unpacks to `~/.vade/bundles` and runs `vade install` from there, with no Rust and no build. When a check or the download fails, it prints one `Prebuilt:` line with the reason and builds from source as before. The bundle is not published yet, so today every install builds from source. `--source` always builds from source; `--check` prints the plan and writes nothing. `vade install` from a bundle now replaces an earlier install whose files have other hashes than the bundle's `launcher-bundle.conf`, from a source build or another bundle; before, it kept the earlier engine. `scripts/make-bundle.sh --from-payload` packages already built binaries into `vade-<version>-<sha7>-linux-aarch64-gb10.tar.gz` with a `MANIFEST`; two runs give the same bytes.
- FP8 on four Sparks now adds its partial sums in the two-Spark order, so four Sparks are made to give the same token IDs as two Sparks. A projection that splits its inputs over the ranks (attention `o_proj`, GDN `out_proj` and the shared expert down) now sums each four-rank part in the same blocks as one half of the two-rank sum. The ranks add their parts in pairs, (rank 0 + rank 1) + (rank 2 + rank 3), on the host and on the GPU, and the routed expert down sum groups its blocks as on two Sparks at every Spark count. Three shared BF16 projection kernels (`flash_bf16_mma`, `flash_bf16_mma1` and `flash_bf16_mma64`) get a second copy of their sum for a four-rank part. FP8 on two Sparks and FP4-Local on every Spark count run the first copy, whose source does not change, so their token IDs must not change; the pre-merge screen checks them on the Sparks. FP8 token IDs on four and six Sparks change; six Sparks still give other IDs than two. The project owner approved this change of token IDs on 2026-10-31. At this merge, the FP8 goldens (saved reference answers) for four and six Sparks are recorded again: the world-four `path` goldens of the [world-four procedure](cli/docs/test.md#world-four-goldens), the six-Spark goldens, and the FP8 four-Spark and six-Spark benchmark cards that later changes compare against. The FP8 fixtures in `engine/fixtures/qwen3.8-flash-next-fp8` come from a CPU reference, not from Vade, and do not change. The drafting layer still adds its routed expert sums by Spark count, so its drafts can differ between two and four Sparks. With `"draft"` set to `auto` (the default), `strict` or `plain`, the model checks each draft and keeps only its own tokens, so the token IDs do not depend on the drafts. The opt-in `fast` and `sampled` draft modes can give other token IDs on two and four Sparks. See [the FP8 arithmetic contract](engine/docs/flash-fp8-math.md#rank-counts).
- `vade install`: a source install on two or more Sparks now copies the harness sources to each Spark too. Before, the CLI on each Spark could not find its `../harness` path dependency, and install stopped with "Helper build failed"; the retained log said "no matching package named `vade-harness` found". Installs from a prebuilt bundle are unchanged.
- `scripts/install.sh`: the driver check now also reads the "CUDA UMD Version:" header that newer NVIDIA drivers print. Before, it found no CUDA version on drivers such as 615 and stopped with "must support CUDA 13.0 or newer". The 13.0 rule is unchanged.
- `vade pull`: a small file no longer fails with "The Hub sent more bytes ... than its listing gives". The Hub answers a file request with a redirect, and curl checked its size limit against the redirect reply too, so a file smaller than that reply (such as a 202-byte `generation_config.json`) failed on every attempt. The limit pull gives curl is now at least 64 KiB. A reply longer than the listing still stops the download and removes the partial file, and a short reply is still not taken as the whole file. The Spark check also reads the CUDA version from newer drivers, whose nvidia-smi header says "CUDA UMD Version:" instead of "CUDA Version:"; before, it stopped with "Cannot read CUDA support from nvidia-smi".
- `vade bench --card --prompts FILE` sends the 66 requests of the published cards to a running server, in their order and with their request bodies (RigMark protocol 1.3.0 at revision `52c29c3`, thinking off). It saves every token ID and the card's `token_ids_sha256` in `./vade-card.json`. With `--receipt FILE` it compares the IDs with a receipt in [bench/receipts/2026-10-31](bench/receipts/2026-10-31/README.md) and prints `ids 66/66 identical` or that they differ; against another card it prints `ids N/66 identical`. `--api-key-file` works as for `vade bench`, Ctrl-C stops the card and keeps every finished chat, 8K pair and concurrency round, and a run that gets no reply saves no file; `--compare` compares two saved files and sends nothing. `FILE` is `prompts.json` from a RigMark checkout at that revision (MIT license); the client refuses a file with another SHA-256. Before, only a private client recorded the token IDs of a card. See [docs/results.md](docs/results.md#what-a-stranger-can-and-cannot-re-run).
- docs: the public pages use the 2026-10-31 cards at main `dc32b7c`, with drafting ON and OFF from the same server start. A new page, [Results and receipts](docs/results.md), gives each number the README quotes with its record, and [bench/receipts/2026-10-31](bench/receipts/2026-10-31/README.md) holds a sanitized extract of every card. The README no longer says that four FP4-Local Sparks write at the speed of one: over RDMA they run the tensor split in parts. The credits now name the one file that follows a reference closely, the CPU reference `refcpu.rs`.
- `vade-harness suite`: the task follow-agents-md now checks the CHANGELOG lines and their order with the new check `lines-equal`, which does not compare blank lines. Before, its `file-equals` check also compared blank lines, so a correct entry failed when the list had a blank line between its items. AGENTS.md in the task states the line, its heading and its order, but no blank-line layout.
- docs: eight-Spark runbook and changelog statements corrected to match the code
- docs: credits and provenance reworded; original-work section added; facts unchanged
- harness: suite_dry_run uses a unique temp folder and cleans up (CI flake fix)
- The engine card (`quality-card --set engine-1`) now checks each needle with the server's `/v1/admission` before it sends it. A needle that does not fit is resized to fit, or shown as `NOT RUN` with the server's limits and the context to use. Before, a server started with `vade up --context 32768` refused the 32K needle with HTTP 400, and the card showed only that status line. The 32K needle needs `vade up --context 40960`. See [the engine card](harness/docs/quality-card.md#the-engine-card-engine-1).
- `vade-harness quality-card --set engine-1` checks how a running Vade server behaves, in five checks: a code planted in prompts of 8K and 32K tokens comes back word for word; 50 replies in JSON mode parse and match their schema; 20 tool calls have the exact name and arguments; the prefix cache and 8 requests at once give the same token IDs. It prints one card with a grade from A+ to C and writes a new JSON receipt each run, and it exits 0 only at A+. Vade ignores `response_format` today, so the JSON check most likely grades C. See [the engine card](harness/docs/quality-card.md#the-engine-card-engine-1).
- FP8 decode now reads the HC up weights and most GDN and QSA attention projections through LBF12 copies on each rank whose weights are resident. The copies decode to the same BF16 bits, so token IDs do not change. They use more memory on each rank, so less room is left for the KV cache. In the lab's two-Spark FP8 screen of `d140b4e` (two starts of each side, drafting depth 2, context 32768), each rank printed `flash_session attention_lbf12 matrices=134 bytes=2059973376` and `flash_session hc_up_lbf12 matrices=100 bytes=502272000`. Prose decode went from 36.72 to 37.30 tokens/s with drafting (1.6 %) and from 27.02 to 27.67 tokens/s without it (2.4 %). The KV budget of the first start went from 8,737,245,434 to 7,564,172,129 bytes. See [architecture](docs/architecture.md#fp4-local-and-its-number-formats).
- `examples/openai_client.sh` now gives curl the `VADE_API_KEY_FILE` key in a header file in its private temporary folder. Before, the key was a curl argument, which other users of the computer can see while the request runs. A key with a line break is refused.
- FP4-Local on four Sparks connected by RDMA now runs the tensor split in parts. Rank 0 runs layers 0 and 1 alone, and all four Sparks work on layers 2 to 47. On a pass of at most 16 rows (32 rows for a drafting verify pass), each Spark computes one quarter of every routed expert. On such a pass with one chat, each Spark also computes a part of most projections. A plain greedy step of at most 16 rows splits the output head; a verify step splits it only for one chat and 1 to 8 rows. A wider pass splits no expert, and a long prompt runs as chain tiles in which each Spark runs 12 whole layers. Before, four Sparks ran the layer chain. Plain `vade status` shows the path as `tensor split (FP4-Local on more than two Sparks whose widths cut into whole parts; projection level 2, relay wire, head split by vocabulary parts)`; `vade status --json` and `/health` name it `tensor_split`. The prefix cache stays off on four Sparks, as on the chain. Four Sparks over TCP, and every other count of three or more Sparks, still run the layer chain. A start on two Sparks whose model has no whole halves now stops with an error. See [docs/architecture.md](docs/architecture.md#several-sparks-the-layer-pipeline-and-the-chain) (**Choosing the layout**).
- `vade up --replicas N --pairs` (N = 2 to 12) starts N two-Spark tensor-split servers on 2N Sparks, behind one local gateway on one address; the first and second machine form pair 1, and so on. It is an option: `vade up --sparks 4` starts one server on four Sparks (the tensor split in parts over RDMA, the layer chain over TCP). With any `--replicas` gateway, every turn of a chat now has one key (`prompt_cache_key`, else a session header, else the prompt through the first user message), and a later turn, a reply or a tool step stays on its server while that server has fewer than 8 requests in flight or is among the least busy. Before, a short chat got a new key on each turn, and a later turn moved whenever its server was busier than another. See [docs/four-sparks.md](docs/four-sparks.md#two-pairs-behind-one-address).
- `vade logs` after `vade down` shows every saved log of the last run: one per Spark, plus the replica gateway log. Before, it showed the newest 6 files, so after a run on eight Sparks two rank logs were missing, and after a run on fewer Sparks logs of older runs filled the list.
- `vade chat` and `vade status` no longer tell you to stop a model that is still loading. While `vade launch` loads it in another terminal, `vade chat` says to wait and names `vade logs`, and `vade status` shows `Vade is starting` (exit 0). Before, both said `Startup is incomplete; run vade down, then vade launch.` `vade chat "a prompt"` and `vade chat --json` now say what to do instead of `Unknown option`. In chat, a message sent during a reply says `Stopping the reply to send your new message` and ends with `sending it now`, not `Native cancellation requested` and `say continue`. When the engine stops during a chat, the stop line is followed by the check to run, such as `vade status`. The Linux and small-context notices say what still works and how to get the agent back. [cli/docs/ux.md](cli/docs/ux.md#what-you-see-when) lists what you see in each case.
- MCP servers now get only `HOME`, `PATH`, `TMPDIR`, `USER`, `LOGNAME`, `SHELL`, `LANG`, `LC_ALL`, `LC_CTYPE` and `TERM` from chat, plus their own `env` entries. A server that read a token such as `GITHUB_TOKEN` from your shell needs that token in its `env` entry now. A server `cwd` must be a folder inside the project, without `..`; chat skips an entry with an absolute or `..` `cwd` and prints one line about it. A call that passes its time limit now stops the server, and the next call starts it again.
- The agent's `apply_patch` no longer applies a hunk at another indent width than the file: a block at another nesting depth no longer matches. When a later file of a multi-file patch fails, Vade puts back the earlier files that still hold the patch's bytes; moves stay. A retry no longer removes a leading `+` from an Add body that mixes `+` lines and bare lines. Shell results keep more output and never split a UTF-8 character, and a malformed tool call repeated three times ends the run.
- `scripts/install.sh` no longer needs the model on disk. With no model, its `vade install` still builds the GPU kernels, the engine and chat, compiles the GPU code and exits 0. It saves no settings and prints the `vade pull` command. The first `vade launch` or `vade up` then saves the settings and checks the model files before the start. The one-Spark quick start is now five commands: clone, `scripts/install.sh`, the `PATH` line, `vade pull`, `vade launch`. Before, the script stopped at the model check, and you ran it again after `vade pull`. On one Spark, `vade setup` now skips `vade install` when the saved settings exist and the installed files were built from the current checkout, so the older steps (`vade setup`, then `vade chat`) still work and no longer run the install twice.
- `vade read` and `workspace_read` now read ZIP, tar and gzip archives (`.zip`, `.tar`, `.tar.gz`, `.tgz`, `.gz`). The Markdown lists each entry with its size and type, then reads each file with the reader for its format after a `<!-- file N of M -->` marker. Links are listed and never followed. The files of one read decode to at most 50 MiB in all, also the parts that office files in the archive unzip, and an archive is opened at most two levels deep. ODT numbered lists now show their numbers, and a damaged `styles.xml` gives bullets, not a refusal. HTML keeps `<li><p>text` on the marker line, lines up a list nested under a numbered item, keeps `colspan` columns and no longer drops the rest of a page after an empty `<title/>` in inline SVG. An XLSX file with a very large shared string table now shows its sheets. A relationship target longer than a ZIP entry name no longer refuses an office file when the reader does not need that part. See [docs/reading-files.md](docs/reading-files.md).
- Before an answer, `vade chat` prints how much of the prompt is cached and about how long the first token takes, for example `cached 7,812 of 8,000 tokens; about 0.4 s to first token (3.7 s without the cache)`. The engine sends the same estimate to API clients in four stream response headers (`X-Vade-Prompt-Tokens`, `X-Vade-Cached-Tokens`, `X-Vade-Estimated-First-Token-Ms`, `X-Vade-Estimated-Cold-Ms`) and in the `vade` object (`prefill_estimated_ms`, `prefill_cold_estimated_ms`). The estimate starts from the one-Spark card and learns from each finished prompt. In its unit test on one-Spark receipts at 8K to 128K, cold and warm, it was within 15.7 % of the engine's time to the first logits (`spark-serve` test `canned_dispatch_inputs_are_within_20_percent`). It changes no token IDs.
- `vade-harness suite` runs the agent task suite with the harness. `--dry-run` needs no model: it replays each task's correct and near-miss scripts through the real tools and checks that the correct one passes with quality 3 and the near miss fails; `cargo test` runs it. `--live` runs the tasks on a served model and writes a card per run and a summary: one row per category (tasks, completed, quality mean from 0 to 3, median wall time, tokens), one line per failed task run with its first wrong tool call, the token IDs compared over runs, and a grade. The suite now has 22 tasks in 11 categories, and each task has a quality rubric. See [docs/harness-suite.md](docs/harness-suite.md).
- Install, setup and doctor stop with a clear message where they used to go on or blame the wrong thing. `scripts/install.sh` checks for a GB10 GPU and a driver that supports CUDA 13.0 or newer before it downloads anything. Local detection shows the first line of the `nvidia-smi` error and names the GPU it found. A controller's `vade install` and `vade doctor` use the same GB10 and CUDA 13.0 rule on each Spark. A controller's `vade install` now exits 1 when a Spark lacks the model; the build stays staged. `vade up` and `vade doctor` say `Cannot reach <Spark> over SSH` instead of `Cannot verify staged artifacts` when SSH does not run. A first local install says when port 18080 is taken and which port it saved instead. It also says when no checksum list sits beside the checkpoint, so its files cannot be checked against the published hashes. README.md and INSTALL.md state the CUDA 13.0 and idle-GPU needs and explain the `PATH` line.
- `vade install` from a Git checkout, and so `scripts/install.sh`, no longer keeps an engine that another checkout or revision built. A source install now records its Git revision. When the installed files have another revision or none, install says `this checkout is <rev>, but the installed engine was built from <rev>; building this checkout`, builds, and ends with `generation <id> is now installed, built from <rev>`. Before, a new clone over an existing `~/.vade` printed `reuse installed files` and kept the old engine.
- `scripts/install.sh` builds the GPU kernels once and prints the Spark check once. It now fetches only the Rust `build-std` files, and `vade install` does the one offline kernel build and the one check. Before, the script also ran `cargo kernels` and `vade detect` first: in the lab's fresh-clone run r2 on one Spark (main `0bad2e6`), the kernels built twice (50.96 s, then 53.6 s) and the check printed twice.
- After the local engine crashes or is killed, `vade up` and `vade launch` clear its record in one line, then start as usual. `vade status` says `the engine is not running (stale record found)`, `vade chat` names `vade up` as the next step, and `vade doctor` checks the port instead of reporting it in use. Before, all four stopped on `The recorded server does not own the loopback listener.` until you ran `vade down`.
- On Linux, closing an MCP server (chat `/exit`, `vade tools`, `vade mcp test`) no longer stops every process of your user, such as the engine and your SSH session. Vade now stops only the server's own process group, with `kill -KILL -- -<group>`. It never signals its own group or the `0` and `-1` targets. Hooks and code checks use the same stop.
- When an MCP call does not run, the model now learns why: you declined it, Vade could not show the question, no one could answer in a run without a terminal, or approvals are `never`. It no longer blames the tool server.
- Chat with input from a pipe treats `/exit` before the provider starts your message like the end of input: the message still runs, and Vade says so on stderr. After the message starts, and during other work such as `/compact`, `/exit` cancels it, as before. Inside `/paste`, `/exit` is text. A sent line of only `y` or `n` answers a pending approval; in the full-screen editor, the key must be typed, not pasted or recalled. `vade mcp test` prints `none` for an empty resource or prompt list.
- `docs/mcp.md` starts with an example MCP server that needs no token and no extra install ([`examples/mcp_word_count.rs`](examples/mcp_word_count.rs)).
- `vade-gateway` answers a malformed head, or a head that its own Host, length and key lines would push past 32 KiB, with HTTP 400 (the Anthropic error shape on Messages routes). Before, it closed the socket or gave a 502. Such a request is not recorded or forwarded; with `--clamp-limits`, one rare case first goes out in an admission request. `--record` makes a folder that other users can open owner-only and says so on stderr, and it stops recording at 100,000 request folders, the replay limit; requests are still forwarded. In `--stats` rows and recordings, a local answer (a refusal, a gateway 502, a CORS preflight or the replica `/health`) now has `"complete":false`, so replay skips it. With several upstreams, a cancel or KV release answers within 1.5 seconds, name lookup included. See [harness/docs/invisible-gateway.md](harness/docs/invisible-gateway.md).
- `vade-gateway --stats` rows count Messages cache reads, streams and first tokens the way the engine does, and the core gateway counts Responses replies. Before, a Messages warm turn and every Messages stream were marked invalid.
- `vade-gateway --stable-prefix` now also adds `cache: true` and a cache key to Messages requests, so Claude Code reuses its prompt prefix. In [one run on one Spark](harness/docs/gateway-measurements.md#claude-code-prefix-reuse-2026-10-30), its call after a tool result read 16,000 cached tokens and gave a first token in 0.47 s instead of 7.8 s.
- Uncommon startup errors name the cause and the next step: a wildcard gateway `--bind`, a `--stats` path in a missing folder, `vade-harness doctor --endpoint`, an unknown session in `vade sessions export`, and a model ID the Hub refuses in `vade pull`.
- `vade read` reads PDF files as Markdown, with a page marker before each
  page. The agent's `workspace_read` keeps its own PDF text path.
- `vade read` and the agent's `workspace_read` read DOCX files as Markdown:
  headings, lists, tables, links and line breaks, with a page marker before
  each page. Images are listed by name. XLSX files read as one table for
  each sheet, with dates as ISO dates, and PPTX files as one section for
  each slide with its title, bullets, tables and speaker notes. EPUB books
  read in spine order with a chapter marker for each chapter. OpenDocument
  text, spreadsheets and presentations (ODT, ODS, ODP) read as pages,
  sheets and slides too; their numbered lists show as bullets, and an ODT
  file has page markers only where LibreOffice marked its pages. PNG, JPEG,
  GIF and WebP images read as one line with their size in pixels. See
  [reading files](docs/reading-files.md).
- New `vade setup` asks which Sparks, which checkpoint (for 2, 4 or 6
  Sparks) and where the model is, with each default shown, then runs
  `vade install`, `vade up` and `vade doctor`. `--yes` asks nothing and
  takes `--hosts`, `--fp8`, `--model` and `--link`. From another computer it
  writes the controller config and never replaces one.
  `vade install --dry-run` now starts with a plan: where, which files, the
  model and where state lives. See the
  [agent quickstart](docs/agents/quickstart.md) and
  [install and flows](docs/design/install-and-flows.md).
- Native chat supports [host hooks](harness/docs/hooks.md), and chat supports [user slash commands](harness/docs/slash-commands.md) and `/clear`.
- The CI workflow runs on a push to `main`, on a pull request and by hand, on GitHub-hosted standard runners with no secret; every job is skipped while the repository is private. Pushes and pull requests check and test the CPU crates (`cli`, `harness`, `engine/tools/bench`) on Linux `aarch64` and `x86_64`, and the repository tools on Linux `aarch64`; the engine and GPU kernel job runs only by hand. Tests still run locally before every merge, as [Local CI](docs/ci.md) describes.
- Chat keeps private project notes across sessions; `VADE_MEMORY=0` turns memory off.
- Groups for many Sparks under one controller. A `[group NAME]` section in
  the controller config names machines in rank order, a local port and a
  layout tag; `VADE_GROUP=NAME vade up`, `status`, `down` and `update` work
  on that group with its own ownership record. A group must be an arc of the
  cabling (each host cabled to the next, in rank order), and groups in one
  layout cannot share machines or ports.
  Without `VADE_GROUP`, `vade status` and `vade down` name the groups that
  still run. New `vade doctor --fabric` checks every configured cable over
  SSH, read-only: the cabling shape, each group ring, each fabric port
  (address, speed, MTU, RDMA device), both rails of each cable on their own
  (two addresses, one subnet, 9000-byte pings both ways) and each clock.
  `vade doctor --fabric --ring` also expects one closed ring: every Spark has
  exactly two neighbors, and both rails of every cable link at one speed. See
  [Eight Sparks](docs/eight-sparks.md), with an example config for one ring
  of eight.

- The agent asks before every shell command. A `workspace_shell` command
  runs without a prompt only when an allow list names its exact argv of
  plain literal words; the lists are empty by default. Vade's command
  checker still blocks fork bombs, `rm -rf /`, disk writes and similar, also
  inside `sh -c` scripts and with `--yes`. At a prompt, `s` allows that
  exact command for the session and `p` saves it to
  `.vade/permissions.json`; the user list is `~/.vade/permissions.json`.
  `--yes` (or `--dangerously-skip-prompts`) turns prompts off. A run
  without a terminal returns a tool error for a command that needs
  approval. Each decision goes to the session journal. The checker's safe
  class (read-only, build and test commands run at once) is experimental
  and off; `VADE_SHELL_CHECKER=1` turns it on. See
  [shell safety](docs/shell-safety.md).
- An allow-list rule that you write can hold `<path>` slots, so one rule
  such as `ls -la <path>` allows that command on any path inside the
  workspace (no `..`, no leading `/`, no hidden, credential or linked
  part). The command prompt shows the folder, every environment variable
  that the command gets and, for a read-only command, the system form that
  runs. A word that names a hidden, credential or `permissions.json` file
  in any part, such as `tar -tfid_rsa` or `git show HEAD:.env`, never goes
  on an allow list. Program paths such as `/bin//rm` and names such as `RM`
  are read as macOS runs them, so `/bin//rm -rf /` is blocked.

- Chat sessions can be listed, continued, exported and deleted. Each agent
  chat is a saved session (direct text chat saves none); `--name NAME` names
  it. `vade sessions` lists the sessions of the current folder (ID, name,
  start, last turn, turn count, model and project), newest first, and `--all`
  lists every project.
  `vade chat --resume ID|NAME|last` replays a saved session into the context
  without running tools again. `vade sessions export ID --markdown` prints a
  readable transcript, and `vade sessions delete ID` removes a closed one. See
  [saved chat sessions](docs/sessions.md).

- `vade read <path>` prints a local file as Markdown with a header line
  (name, format, pages, characters, what was skipped). It reads plain text in
  UTF-8, UTF-16 or Windows-1252, Markdown, HTML, CSV and TSV (as tables),
  JSON, YAML, TOML, XML and source files (as code blocks) and RTF. Other
  known formats print one line that names them, and other binary files one
  line that says so. The agent's `workspace_read` uses
  the same readers for RTF and for text that is not UTF-8; UTF-8 text files
  are returned byte for byte, as before. See
  [reading files](docs/reading-files.md).

- Chat can use your MCP tool servers. `vade mcp add NAME -- COMMAND` adds a
  stdio server to `mcp.json` in the Vade home, and `vade mcp list`, `test`,
  `remove` and `trust` manage the list. Agent chat starts each server and
  offers its tools to the model as `<server>__<tool>`, after the native tools,
  under the `--approval` setting. Calls have a time limit and a size limit, and
  a server that stops is started again once. A server in a project's
  `.vade/mcp.json` starts only after `vade mcp trust NAME`. Resources and
  prompts are listed only. `vade tools` lists the native and MCP tools, and
  `VADE_MCP=off` starts no server. See [MCP tools in chat](docs/mcp.md).

- The native agent has `web_fetch` and `web_search`; harness `--no-web` or `VADE_WEB=0` turns both off.

- Help, errors and startup lines read in plain words. Each error says what
  failed and what to run next; with nothing set up, `vade status`, `vade chat`
  and `vade doctor` name `vade launch`. On a Spark, `vade up` now prints a
  ready line like `vade launch`: the address, context, output and Spark count.
  From a controller, the ready line gives the address, context, drafting depth
  and Spark count. A bug in `vade`, `vade-harness` or `vade-gateway` prints one
  plain error that asks for a report in place of the Rust panic message
  (`RUST_BACKTRACE=1` shows the full report). The agent's file tool errors are
  full sentences, and the glossary covers the new terms.

- On macOS, the agent and the gateway now read a reply that a server sends
  just before it closes the connection, such as a 401 for a wrong key. Before,
  they could report "Invalid argument" in place of the reply.

## v0.2.0 - not released yet

v0.2.0 makes two Sparks much faster than v0.1.0. Two Sparks also admit more
chats at once, and both one and two Sparks serve larger context windows. The
answers do not change: FP4-Local gives the same token IDs on one, two and
four Sparks. The [v0.2.0 release notes](docs/release-notes/0.2.0.md) explain
each change in plain words and list the known limits. The detailed list of
changes follows the highlights.

Each number below names its source. "The end-of-day cards" are the RigMark
cards of 2026-10-28 at main `76977df`
([benchmarks](docs/benchmarks.md#current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks)).
"The v0.1.0 cards" are the release cards of 2026-10-24
([receipts](bench/receipts/2026-10-24/README.md)). Rates are completion
tokens per second, with drafting ON (the default) unless a line says
otherwise. Each card sends its cold 8K prompt with drafting OFF
(`"draft": "plain"`), so the cold 8K times below are drafting OFF.

### Highlights

- **Two Sparks write faster.** Two Sparks now run the tensor split: both
  Sparks work on every token. You set nothing, but the split needs RDMA. On
  two Sparks, one chat writes code at 100.0 in the end-of-day cards, against
  65.1 in the v0.1.0 cards. Prose went from 38.3 to 53.8, and eight chats
  together went from 168.6 to 229.5.
- **Plain decoding is faster on two Sparks too.** With drafting OFF, two
  Sparks write prose at 35.1 in the end-of-day cards, against 27.9 in the
  v0.1.0 cards.
- **Long prompts give their first token sooner.** On two Sparks, a cold
  8K-token prompt sent with drafting OFF gives its first token in 2.19 s in
  the end-of-day cards, against 2.79 s in the v0.1.0 cards. With drafting
  ON, a 129,220-token prompt took 40.37 s at main `285fa8a`, against
  42.09 s on the main before it, `54392da`
  ([confirm run](docs/benchmarks.md#confirm-run-main-285fa8a)).
- **Two Sparks admit 16 chats where they admitted 2.** A chat now reserves
  only the memory for the rows it uses, not for the whole context. At a
  131,072-token context, a test sent 16 short chats, one each second, and
  all 16 were active together, against 2 on main `54392da`. The median
  queue wait of the first nine chats fell from 44.8 s to 0.16 s (capacity
  test of 2026-10-28; the same 0.16 s in both runs of the change).
- **Larger context windows, measured.** At main `285fa8a`, one Spark served
  a 131,072-token context: a 120,624-token prompt gave its first token in
  66.2 to 66.5 s. Two Sparks served a 262,144-token context: a
  201,654-token prompt gave its first token in 68.0 to 68.1 s. At main
  `250fb49`, one Spark also served the 262,144-token context, with the same
  prompt, in 120.3 to 120.8 s. The token IDs were the same in every run of
  each setup. The default context stays 32,768. Ask for more
  with `--context`.
- **The prefix cache works on two Sparks.** With the cache on, the second
  question on a long chat does not read the whole chat again. On a
  32K-token chat in the end-of-day runs, the second question gave its first
  token in 0.76 s on two Sparks (9.95 s with the cache off) and in 0.63 s on
  one Spark (16.03 s with the cache off)
  ([benchmarks](docs/benchmarks.md#a-long-chat-and-its-second-question)).
  The cache is now on by default on one Spark and on two Sparks.
- **The OpenAI-compatible API has a guide.** The new
  [API page](docs/openai-api.md) lists the routes and fields that work and
  where Vade differs from OpenAI. The core gateway answers `/v1/embeddings`
  with HTTP 501 and a reason, and its errors carry `param` and `code`. Both
  gateways accept a chat history in which the openai-python stream helper
  sent `"role": null`. Host tests replay 15 client request shapes; 9 of them
  were captured from openai-python 3.41.0.
- **The agent reads PDF files.** When the agent reads a text-based PDF of up
  to 50 MiB in your project, it gets the text of each page under a
  `page N of M` line. It refuses encrypted files, and a scanned page gets a
  note that it has no text layer. You cannot attach a PDF to a chat message
  yet.
- **The agent reads your project instructions.** In direct and agent mode,
  at the start of a chat, it puts `AGENTS.md`, `CLAUDE.md` or
  `.vade/instructions.md` from each folder between the git root and the
  workspace into its system prompt, up to 32 KiB per file. `/instructions`
  lists the files.
- **FP8 is a supported route.** `vade up --fp8` starts the FP8 checkpoint
  on 2, 4 or 6 Sparks; no card has measured six. In the end-of-day cards,
  FP8 wrote prose at 36.6 on two Sparks and at 46.7 on four. FP8 is the one
  model in these cards for which four Sparks write faster than two. See
  [the FP8 install](docs/install-fp8.md).
- **Download a model with `vade pull`.** `vade pull <org>/<name>[@revision]`
  downloads a checkpoint from the Hugging Face Hub, continues an interrupted
  download and checks the size and hash of each file. The FP4-Local
  checkpoint is 32 files and 102.26 GB (`vade pull --dry-run`, 2026-10-29).
  `vade models` lists the local models. See
  [Download models](docs/pull-models.md).
- **Install and commands are easier to follow.** `sh scripts/install.sh`
  sets up the tools and the source build on one Spark. A plain-text error
  starts with `error:` and adds a `hint:` line when there is a next step.
  `vade doctor` runs every check and gives the fix under each failed row.
  `vade-harness setup claude` prints the settings for Claude Code.
- **A long chat rolls over instead of stopping.** When the next turn would
  not fit, or at 80% of the context window, the model writes one summary of
  the older turns. The last 2 turns stay word for word when they fit, else
  the last turn only. See
  [context and compaction](docs/context-and-compaction.md).
- **`min_p` sampling.** The engine applies `min_p` after top-k and top-p,
  on every request surface. See [sampling](docs/sampling.md).

### In progress for v0.2.0

This work was not merged when these notes were written. It is not part of
v0.2.0 until it merges.

- Structured output, phase 1: JSON mode.

The prefix cache on by default, `vade sessions`, MCP tool servers, shell
safety and document reading for office files (Word, Excel and PowerPoint)
and EPUB have merged since this list was written; see Unreleased and the
engine section.

### Known limits

- Four Sparks write one chat at about the speed of one Spark. With
  FP4-Local, they wrote prose at 42.8 in the end-of-day cards, against 42.0
  on one Spark and 53.8 on two. Chains are measured up to four Sparks.
- Text only. Vade does not read images.
- With FP4-Local, greedy decoding is the default, and the exactness checks
  cover only greedy requests. A request that sets a sampling field takes the
  checkpoint's defaults for the fields it leaves out, so its reply is
  sampled. An explicit `"temperature": 0` stays greedy. FP4-Local's
  penalties are neutral, so it gives the greedy token IDs. Another
  checkpoint can have penalties that change the answer.
- A sampled request on Flash-Next does not draft by default: it decodes
  plain. `typical_p`, `logit_bias` and `best_of` are ignored. See
  [sampling](docs/sampling.md).
- The engine ignores `response_format`. There is no structured output mode
  yet.
- With FP8, a request with no sampling field is sampled: each field it
  leaves out takes the checkpoint's value. Temperature 0 with neutral
  penalties is greedy. The gateways add a full greedy set to a request that
  sends none.
- With FP8, two Sparks and four Sparks give different token IDs: 7 of 66
  responses matched in the end-of-day cards. A fix is in screening.

### Engine

These entries were written as each change merged. Some name a `VADE_W2_*`
switch that a later entry removed: in v0.2.0 no `VADE_W2_*` switch has an
effect, and the engine chooses the path for each number of Sparks.

- Chats that arrive together can now start their prompts together. Before, the
  server could start the first chat's prompt while the request bytes of the
  other chats were still on the way, and those chats waited for that whole
  prompt pass. This split happened in 15 of 122 four- and eight-chat rounds
  of 21 two-Spark card runs of main `285fa8a` (the first
  `request_prefill_cohort` width of each chat in the first Spark's server
  log). Now the first prompt pass of an idle server waits for a burst when no
  chat decodes and every waiting chat is a new prompt that can join a shared
  prompt pass (on Flash-Next, up to 512 tokens). This includes a first chat
  that would start alone on the serial prompt path. A longer prompt is not
  held and starts as before. The hold ends at the latest 10 ms after the
  oldest chat connected. The waits that applied before still apply after it:
  up to 100 ms while requests are in preparation, one loop turn for unread
  request bytes, and admission. A lone request with nothing else arriving
  does not wait. `VADE_INTAKE_BURST_MS` sets the window (0 to 100 ms, 0 = no
  hold), and each such pass logs an `intake_burst` line. Token IDs do not
  change.
- `min_p` (0 to 1) keeps the tokens whose probability is at least `min_p`
  times the best token's. The engine applies it after top-k and top-p, as
  Hugging Face `transformers` does. Chat, completions, Responses and
  Messages read it, and replies echo it in `vade.sampling` when its value in
  effect is above 0. See [sampling](docs/sampling.md).
- A sampled request on Flash-Next decodes plain (no drafting) unless
  `flash-serve` runs with `VADE_MTP_SAMPLED=1` or `2`; with `2`, sampled
  rows also join a cohort's batched verify pass. Greedy requests still
  draft.
- Chat and completions refuse any `n` other than 1
  (`only n=1 is supported`).
  Responses and Messages ignore `n` and give one reply.
- Flash-Next on two Sparks admits more short chats at once. A chat now
  reserves only the KV pages its rows use. Before, every chat reserved the
  pages for the whole context (8,231,544,132 bytes at `--context 131072`,
  whatever the prompt length). The second Spark keeps the buffers of a chat
  that ends and gives them to the next chat in the slot. At
  `--context 131072`, a test sent 16 short chats (1,247 to 1,248 tokens
  each), one each second, and all 16 were active together, against 2 on main
  `54392da`. Their median wait in the
  queue was 0.16 s for chats 1 to 9 and 1.02 s for chats 10 to 16 (run
  `cap-c2b-x`; run `cap-r3-x` on the final patch gave 0.16 s and 1.01 s, see
  [benchmarks](docs/benchmarks.md#card-at-main-250fb49-after-the-change-to-memory-reservation)),
  against 44.8 s and 128.9 s. Two 100,378-token chats, sent 8 s apart, are
  still active together, but they reach the first token later, because the
  engine now creates their KV pages when they start: 76.9 s and 91.7 s,
  against 74.8 s and 88.9 s (+2.9 % and +3.1 %). Each figure comes from
  one server start per build, in the `cap-c2b` capacity test of 2026-10-28:
  the byte count
  from the main server's `request_admitted` lines (`reserved_bytes`), and
  the chat counts, queue waits and first-token times from the test client's
  log of each build. Token IDs do not change.

- The FP4-Local prefix cache is on by default on one Spark and on the two-Spark
  tensor split; FP4-Local on three or more Sparks (the layer chain, or the split in parts on four Sparks over RDMA) runs without it.
  `VADE_FLASHNEXT_PREFIX_CACHE=0` turns it off; `1` there is still
  refused. Only requests that send `cache:true` or a `prompt_cache_key` keep
  their prompt. At startup the cache takes a room of its own from the session
  pool: half of the bytes no full-context chat can use, 512 MiB to 4 GiB,
  never more than those bytes. The slot plan keeps at least 512 MiB beside
  one full-context chat, which near the memory limit gives fewer chat slots
  (8 instead of 16 at a context of 262,144 on one Spark). On two Sparks the
  room is at least one full-context chat, up to 4 GiB, so a 32K-token prompt
  is kept at a context of 32,768. Admission never uses the room and host
  memory pressure does not reclaim it. Before, a busy server under host
  memory pressure skipped every snapshot (`reason=no_room`). `VADE_FLASHNEXT_CACHE_ROOM=<bytes>` sets the
  room (`0`: none). The `flashnext ready` line shows `prefix_cache_room_bytes`.
  On one Spark a long prompt stays in place when its chat ends (no copy), and
  the next turn builds on it; `VADE_FLASHNEXT_ADOPT=0` keeps copies only. Two
  Sparks keep copies. Memory a kept prompt holds beyond the room yields to
  host memory pressure 30 seconds after the last live chat, when the engine
  gives spare memory back; before then it would stay with the engine anyway.
  `GET /health` has a `prefix_cache` object (hits, misses, reused tokens,
  kept snapshots, entries evicted for memory, expired entries and checkpoints, entries,
  retained bytes, room and snapshots the cache skipped by reason), and the
  log prints the same counts on a `prefix_cache_status` line.
  Token IDs do not change: the cache changes which prompt rows are computed
  again, not their values.

- With drafting on, prompts longer than 32K tokens on two Sparks reach the
  first token sooner. The first Spark now keeps up to 131,072 prompt rows for
  the drafter (KEEP_ROWS in engine/crates/engine/src/flash/next_pp.rs) and
  reads them while it waits for the second Spark's last tile, as it already
  did below 32K rows. Before, past 32,768 rows it read them between its own
  prompt tiles and slowed both Sparks. While a long drafting prompt runs, the
  first Spark holds its kept rows in host memory until the drafter reads them;
  the second Spark returns each tile's rows with its acknowledgment and does
  not hold them. A chain of three or more Sparks keeps the old 32,768-row
  limit (CHAIN_KEEP_ROWS): its middle rank returns all kept rows in one reply,
  and the reply length is a 32-bit byte count. Token IDs and the drafter's
  history do not change.

- Long prompts move fewer bytes in the hyper-connection mix. On fused prompt
  tiles, the norm no longer writes the BF16 hi/lo copy of the four residual
  streams (4 x 2560 features x 2 planes x 2 bytes = 40,960 bytes per token);
  the HC down projection forms the same words from the residual rows instead.
  Token IDs do not change.

- When enabled, the Flash-Next prefix cache can restore a prompt on the tensor
  split. Each rank keeps and restores its own matching state. The layer chain
  still refuses the cache. A second question on a 100K-token chat restores when
  the front snapshot fits the free memory; at `--context 131072` the 6.2 GB
  snapshot does not fit the 5.3 to 6.0 GB that stays free, so that turn reads
  the prompt again. On two Sparks at main `9d2ec6c`, with the cache on, the
  second question on a 32,103-token chat gave its first token in 0.75 to
  0.77 s, against 9.98 to 10.01 s with the cache off. On a 100,364-token
  chat at `--context 110592` it took 1.60 to 1.62 s, against 31.79 to
  31.86 s. The first question takes longer with the cache on, because the
  engine saves the prompt state: the medians are 11.07 s against 10.38 s
  at 32K, and 33.07 s against 31.81 s at 100K. Each range covers three
  server starts. Token IDs do not change
  ([benchmarks](docs/benchmarks.md#the-prefix-cache-on-two-sparks)).

- When enabled, the Flash-Next prefix cache can use free memory that an empty
  spare-buffer reserve has not allocated. Cache retention still yields to live
  requests and physical memory pressure.

- Two Sparks read a prompt of 28,672 tokens or more in wider passes. That is
  seven 4096-row workspaces (`WIDE_MIN_CHUNKS` in
  [`tp_switch.rs`](engine/crates/engine/src/flash/tp_switch.rs)). The middle
  of the prompt now runs in passes of the whole prompt workspace (4096 rows by
  default) instead of 2048 rows, so the experts run fewer, larger batches. The
  start and the end of the prompt keep 2048-row passes. Shorter prompts read as
  before. Token IDs do not change.

- Two Sparks read a cold prompt of 2K to 8K tokens faster. Each Spark reads
  half of the prompt, and the answer stays the same, token for token. In the
  final 2026-10-27 cards, a cold 8K prompt gave its first token in 2.19 s
  in the drafting ON card and 2.17 s in the drafting OFF card, against
  2.80 s and 2.77 s in that morning's cards. Every card sends this prompt
  with drafting OFF.
  Longer prompts read as before. In exchange, two Sparks hold one chat fewer at
  a 32K context: 8 instead of 9.

- Checkpoint indexes now reject fractional byte counts that rounded to integers.

- On two FP4-Local Sparks, a greedy decode step with drafting OFF now splits
  the output head by vocabulary halves across the two Sparks, as drafted
  verification already does. Each Spark computes half of the head. Token IDs
  do not change. Sampled steps and prompts still use the whole head.

- On two Sparks, a long prompt with drafting on no longer stops halfway. Past
  32768 prompt tokens, the first Spark used to wait for the second Spark to
  finish every queued part of the prompt. Now the drafter reads only the
  oldest returned rows that it needs, and the rest waits for the end of the
  prompt. The token IDs do not change.

- **A chat alone drafts the same depths every run.** The depth chooser no
  longer reads the time of single passes. On FP4-Local it prices verify rows
  with its cost prior and a draft price. Each chat measures the draft price
  over its first 16 cycles, rounds it to a power of two and keeps it for the
  rest of the chat. The same request on the same setup now drafts the same
  depth in each cycle every run, so its speed varies less from one server
  start to the next. Token IDs do not change, because verification selects
  every token. The `draft_cost` server log line now also shows `plan` (a hash
  of the rows of each cycle) and `draft_price`.

- A one-token decode pass packs the shared expert's input during routing.
  This removes one GPU launch per layer. The math keeps the same order.

- **The tensor split is the only two-Spark path, and the engine chooses each
  path itself.** No `VADE_W2_*` switch remains: `VADE_W2_PATH`, `VADE_W2_TP`,
  `VADE_W2_TP_WIRE`, `VADE_W2_TP_PROJ`, `VADE_W2_TP_HEAD`, `VADE_W2_PP`,
  `VADE_W2_PP_SPLIT`, `VADE_W2_PP_MTP`, `VADE_W2_PP_ASYNC`, `VADE_W2_PP_RDMA`,
  `VADE_W2_PP_STAGE_STATE`, `VADE_W2_PP_TIMEOUT_MS`, `VADE_W2_PP_C8`, the
  `VADE_W2_TP_DOORBELL*`, `VADE_W2_TP_TAKE_*` and `VADE_W2_TP_WIDE` wire
  settings, `VADE_W2_CHUNK` and `VADE_W2_BW` no longer have any effect. Each
  setting keeps the value that was its default, so token IDs do not change.
  FP4-Local on two Sparks runs the tensor split at its proven levels (the write
  wire, projection level 3, the head split by vocabulary halves, verify passes
  of up to 32 rows). On three or more Sparks it runs the layer chain; with
  `--tensor-map` on four Sparks it runs the older rank path. FP8 keeps its
  ring. The serving receipt names the path and why in `vade.numerics.path` and
  `vade.numerics.path_reason`. The split
  needs RDMA: `vade up` refuses `SEQPAR_RDMA=0` on two FP4-Local Sparks
  before it contacts a Spark. FP8 on two Sparks keeps its ring and may use
  TCP. The pipeline wire follows `SEQPAR_RDMA`. For every group of Sparks
  that serves one model, `vade status` shows the path and why in the
  `Multi-Spark path` row, and `vade status --json` shows them in
  `multi_spark_path` (before: `two_spark_path`, for two FP4-Local Sparks
  only). For the split, both also show its levels.

- **Plain decoding on two Sparks now runs as captured CUDA graphs.** With
  drafting OFF, the tensor split's one-token step now captures layers 1 to 47
  in the same segments that drafting's verify pass uses. The GPU runs one
  segment while the host captures the next, so fewer launch gaps stay in the
  step. Each segment keeps the same arithmetic in the same order, and the
  exchanges between the Sparks keep their order. Token IDs do not change.
  There is no setting. A first step of each shape, a segment that the graph
  budget cannot fund, or a step that a decode profile reads back from runs
  eagerly as before. The pass-count engine log line `flash_next segment_graph`
  (before: `flash_next mtp_graph`) now shows `verify=true` or `verify=false`.
  The deferred-free log line keeps the `flash_next mtp_graph` prefix.

- **Two Sparks run the tensor split by default.** `vade up --sparks 2` now
  runs the tensor split, with every proven level on: both Sparks work on every
  token. You set nothing. Before, two Sparks ran the layer pipeline. In the
  2026-10-26 cards with drafting ON, one chat wrote prose at 54.2 tokens per
  second against 42.2 on the pipeline, code at 100.9 against 80.3 and
  structured output at 130.4 against 100.8; eight chats reached 220.0 against
  201.7. With drafting OFF, eight chats are slower on the split: 110.6 against
  137.0. The split needs RDMA. The serving receipt records the path and the
  split's levels under `path` and `tensor_split`. See the [benchmarks
  page](docs/benchmarks.md#current-2026-10-26-main-262b0c3-home-dgx-sparks).

- On two Sparks, a long prompt from a chat that drafts no longer stalls
  halfway. The first Spark now reads the second Spark's replies for each
  prompt tile as they arrive. Before, it read them only late, so both Sparks
  waited on each other for seconds in a 32K prompt. The tokens do not change.

- Before the two-Spark tensor split serves requests, both ranks agree on the
  smaller session count that they can fund. Different free memory on the two
  Sparks no longer causes a startup refusal when both can fund a session.
  The count exchange uses a 600-second load deadline. A worker no longer
  fails when the front takes longer than the serving timeout to load.

- Flash-Next sizes its freed-memory reserve for the largest observed chat
  cohort, with a minimum of four conversations. During a session release, it
  checks memory pressure again after earlier buffers free device memory.
  Attention arenas can then return to the reserve when the memory margin
  permits it. Logs count new allocations, reuse, returns and direct frees.
  Under host memory pressure, a freed buffer also returns to the reserve
  while the buffers of its kind stay inside the admitted session plan. Eight
  chats then reuse their attention arenas without new allocations. Spares
  above the plan are still released. After 30 seconds without a request, a
  reserve releases its spares if it still reads pressure. A new request ends
  the idle period before its admission can allocate.

- With the two-Spark tensor split, the last rows of a long
  prompt now run on the layer pipeline like the prompt's earlier tiles, so the
  first token no longer waits for the two Sparks to swap the prompt's layer
  state. The swap now runs before the chat's second token. The tokens do not
  change.
- With the two-Spark tensor split, the swap of a long
  prompt's layer state between the two Sparks now goes GPU to GPU through the
  split's RDMA mailbox, in pieces of one mailbox slot, instead of through host
  memory and TCP. The state words are copied as they are, so the tokens do not
  change.

The cards of October 28, 2026 after #139 measured main `104205a` on one pair
of DGX Sparks, with the recipe of that morning. Rates below are tokens per
second.

- With drafting ON, eight chats reached 229.6, against 217.5 in the morning
  card (+12.0, +5.5 %). Eight chats now reuse their attention arenas. With
  drafting OFF, eight chats reached 125.4 against 124.3 on the same pair.
- With drafting ON, one chat wrote prose at 53.8, code at 100.2 and
  structured output at 128.7. With drafting OFF, prose was 35.0. The other
  rates are within 1 % of the morning cards. The 32K, 64K and 128K
  first-token cells and the one-Spark prefix-cache cell did not change.
- Token IDs matched in 66 of 66 responses between the ON and OFF cards and
  against the morning cards. A drafting-ON card at main `8b7b866` (#140)
  matched them too, and its 32K and 64K prompts reached their first token
  0.8 % to 1.5 % sooner. See the
  [benchmarks page](docs/benchmarks.md#current-2026-10-28-main-8b7b866-and-104205a-home-dgx-sparks).

The morning cards of October 28, 2026 measured main `bb24e80` (#136) with the
same recipe and the same two pairs of DGX Sparks as the final cards of
October 27. Rates below are tokens per second.

- With drafting ON, one chat wrote prose at 54.3, code at 100.7 and structured
  output at 129.3. Eight chats reached 217.5. A cold 8K prompt gave its first
  token in 2.19 s.
- With drafting OFF, one chat wrote prose at 35.1 and code at 35.3, and eight
  chats reached 124.3. A cold 8K prompt took 2.17 s.
- Every rate is within 0.9 % of the final October 27 cards. Token IDs matched
  in 66 of 66 responses between the ON and OFF cards, and against the final
  October 27 cards.
- With drafting ON, a 64,217-token prompt gave its first token in 20.04 s,
  against 21.06 s in the final October 27 run, and a 129,220-token prompt in
  42.47 s. A 32,099-token prompt took 10.03 s.
- On one Spark with context 65536 and the prefix cache on, the second question
  on a 32,103-token chat reused 32,064 tokens and gave its first token in
  0.62 s. With the cache off it took 16.29 s. See the
  [benchmarks page](docs/benchmarks.md#previous-2026-10-28-morning-main-bb24e80-home-dgx-sparks).

The final cards of October 27, 2026 measured main `09e1fc4` (#131) with the
FP4-Local checkpoint on two pairs of DGX Sparks, greedy decoding, thinking off
and no `VADE_W2_*` setting. Rates below are tokens per second.

- With drafting ON, one chat wrote prose at 54.1, code at 100.6 and structured
  output at 129.6. Eight chats reached 218.8. A cold 8K prompt gave its first
  token in 2.19 s.
- With drafting OFF, one chat wrote prose at 35.2 and code at 35.4, and eight
  chats reached 123.9. That morning, on the same pair, these were 29.7, 29.6
  and 110.8. A cold 8K prompt took 2.17 s.
- On a second start with context 69632 and drafting ON, a 32,099-token prompt
  gave its first token in 10.24 s and a 64,217-token prompt in 21.06 s.
- Token IDs matched in 66 of 66 responses between the ON and OFF cards, and
  against that morning's cards. See the
  [benchmarks page](docs/benchmarks.md#current-2026-10-27-final-main-09e1fc4-home-dgx-sparks).

Merges #105 through #110 were measured on October 26, 2026, with the FP4-Local
checkpoint on two DGX Sparks. These runs used greedy decoding, drafting on,
and thinking off. Rates below are tokens per second.

- **#110: Keep eight-chat verification split.** With the tensor split at
  `262b0c3`, verify steps of up to 32 rows split the routed experts between
  the Sparks. Other steps
  split experts only up to 16 rows. With projection level 3 and the split
  verify head enabled, eight code chats went from 173.127 to 219.097 in
  pooled measurements across two pairs, against `09a1184`. All 336 token-ID
  comparisons matched. A later RigMark card on the merged revision measured
  220.0 for eight chats. Both figures are aggregate throughput.
- **#109: Split more layer work between two Sparks.** At `09a1184`,
  projection level 2 splits output projections, attention query/gate
  projections, and the shared expert's down projection. Level 3 also divides
  linear-attention heads and their recurrent state on one-chat steps. A shorter first captured
  verify segment reduces the initial wait for GPU work. In two runs per
  revision on one pair, level 3 against level 1 on `dbc2385` changed prose
  from 48.104–48.244 to 51.079–51.243, code from 94.343–94.968 to
  97.711–97.775, and structured output from 121.116–121.417 to
  124.674–125.281. Both revisions kept the verify head unsplit. All 168
  token-ID comparisons matched.
- **#108: Share linear-attention input projections.** At `dbc2385`,
  projection level 1 divides the input projections of split layers between
  the Sparks when one chat runs. It exchanges both sets of output rows
  before waiting for either set. It also fixes weight-copy lookup and bounds
  the state transfer before allocation. In two runs per revision on one
  pair, against `41e3c22` with projection splitting off, prose changed from
  45.208–45.454 to 47.804–48.253, code from 91.190–91.347 to
  94.493–94.848, and structured output from 117.047–117.213 to
  119.726–121.516. Both revisions kept the verify head unsplit. All 168
  token-ID comparisons matched.
- **#107: Share token scoring between two Sparks.** At `41e3c22`, the
  split verify head scores half of the vocabulary on each Spark for one
  greedy chat. When bounds are available, each Spark screens its own half
  and fully scores the rows that can win. Sampled chats and drafts keep the
  whole head on the first Spark. Against `984147f` with the tensor split and
  RDMA write transfer,
  pooled measurements across two pairs changed prose from 45.617 to 47.091
  and structured output from 117.189 to 120.018. Code changed from 91.636
  to 92.053, a tie within the gate's tolerance. All 392 token-ID comparisons
  matched, including the extra eight-chat check.
- **#106: Let the transfer thread sleep.** At `984147f`, the RDMA write
  transfer thread sleeps when no exchange is in progress. During requests,
  it used 0.82 of a CPU core on the first Spark and 0.79–0.80 on the second,
  compared with 0.98 on each before. Throughput tied with `2ef7744`; all 48
  token-ID comparisons matched. A stop during startup also
  cancels the connection attempt if the other Spark does not answer.
- **#105: Check more draft tokens with the bounded head.** At `2ef7744`,
  the exact head that skips vocabulary rows that cannot win also covers
  verify steps of five to eight rows, up from two to four. Pooled
  measurements across two pairs, against `6f5388d` on the pipeline path,
  changed code from 78.689 to 80.375 and structured output from 98.215 to
  100.305. Prose changed from 42.008 to 42.118, a tie. All 336 token-ID
  comparisons and 79 exactness checks matched.
- On two Sparks with the FP8 checkpoint, the routed experts' decode tiles
  now keep two weight blocks in flight instead of one, so decoding is a little
  faster with the same tokens.
- On two Sparks with the FP8 checkpoint, one chat now picks how many tokens to
  draft on each step, up to seven, from how many drafts were accepted recently
  and how long each verify step took. Code and structured replies are much
  faster; prose stays as fast as before. The tokens do not change.
  `VADE_FLASH_MTP_DEPTH=1..7` sets the most it drafts; `VADE_MTP_ADAPT=0` or an
  explicit `--mtp` keeps the old fixed depth.
- On two Sparks with the FP8 checkpoint, drafts read a draft-only MXFP8 head
  copy, so each drafted token costs less: code and structured replies are
  about 9 percent faster, prose about 4 percent. Verification keeps the BF16
  head, so outputs are unchanged. Cost: rank 0 chat slots 19 -> 17, and
  +0.33 GB per rank. `VADE_FP8_DRAFT_MX8=0` turns it off.
- With the two-Spark tensor split, the write wire moves each exchange with
  one-sided RDMA writes straight from one GPU's mapped mailbox into the
  other's, with no host copies and no host work on the receiving side. It is
  now the split's only wire. The tokens are the same. Measured on two Sparks
  with a mailbox benchmark, one exchange: 17.6 to 10.3 microseconds at 10 KB,
  34.0 to 15.3 at 40 KB, 165.9 to 59.4 at 400 KB.
  One chat with drafting on, against the split on today's wire (main
  `62c9c83`): prose 44.76 to 45.68, code 87.14 to 91.48,
  structured 111.08 to 114.26 tokens per second, ids identical. The front
  Spark still waits on the other Spark's compute for most of each exchange.
- On one Spark, `VADE_MTP_COHORT=wide` lets five to eight chats draft three
  tokens each and check them in one 32-row step. Before, five chats proposed
  two tokens each and six to eight chats one each. Four or fewer chats keep
  the 16-row step. Measured on one Spark with eight code chats (`45b3950`
  against main `7e625e7`): 169.6 to 194.4 tokens per second, with the same
  tokens. Unset, nothing changes on one Spark. On two and four Sparks the
  layer pipeline now takes this step by default, with the cost controller
  opt-in (eight code chats on two Sparks, `277463d` against main `254fb86`:
  176.1 to 201.3 tokens per second, same tokens); `VADE_MTP_COHORT=off` on
  every Spark restores the previous step and controller.
- FP8 experts now launch fewer kernels per layer when a step has under 32
  tokens: the gate kernel also computes the activation and the quantized
  middle, and the routing kernel builds the tables for 2 to 31 tokens in one
  launch. The arithmetic order is unchanged, so the tokens are the same.
  Measured on two Sparks with the FP8 checkpoint against main: one chat 42.34
  to 42.77 tokens per second with drafting on and 25.78 to 25.79 with it off,
  eight chats 93.60 to 95.32. The expert weights already streamed near the
  memory limit, so the gain is small; `kernel-bench --fp8-moe` times each
  stage of the expert layer.
- Two Sparks can now split every decode step between them instead of passing
  it down the layer pipeline. This tensor split is now the only two-Spark path
  for FP4-Local. Both Sparks compute half of each routed expert
  and exchange the halves inside the GPU stream, so the tokens stay identical
  to one Spark. Long prompts run as pipeline tiles, and each chat then hands the
  other Spark the state it did not compute. Measured on two Sparks with drafting
  on, against the pipeline: prose 41.73 to 44.70, code 78.84 to 87.34,
  structured 97.99 to 111.41 tokens per second, one to four chats 4 to 9 %
  faster, 8 chats unchanged; ids identical on all 66 card requests. A cold 8K
  prompt takes 2.82 s instead of 2.79 s. Plain (drafting-off) decoding was about
  7 % slower in a smaller test: under the split it does not use pooled decode.
- On four Sparks, the layer pipeline now drafts for six to eight chats at
  once, under the same rules as two Sparks: each tick checks up to 16 drafted
  rows, and the engine switches a cohort to ordinary decoding when that is
  cheaper. Before, eight chats on four Sparks were always decoded plainly.
  Measured on four Sparks with eight code chats: 138.93 to 180.95 tokens per
  second, with the same tokens.
- On one Spark, when one chat runs alone, the drafting depth can now rise to 7
  instead of 5, as on the layer pipeline. Measured on one Spark: code 75.6 to
  81.6 tokens per second, structured output 88.3 to 99.7, prose unchanged, with
  the same tokens. Batches keep their depth. `VADE_FLASHNEXT_MTP_DEPTH=5`
  restores the previous ceiling.
- On the layer pipeline, the check of drafted tokens now replays each
  Spark's layers as captured GPU work, and when one chat runs alone the
  drafting depth can rise to 7 instead of 3 (the resident depth in the
  [glossary](docs/GLOSSARY.md)). Measured on two Sparks against depth 3:
  code +16 %, structured output +39 %, prose unchanged, with the same
  tokens. `VADE_FLASHNEXT_MTP_DEPTH=3` restores the previous ceiling, and
  `VADE_MTP_GRAPH=0` keeps every Spark's work eager. The 2026-10-25 two-
  and four-Spark cards on the [benchmarks page](docs/benchmarks.md#reading-these-cards)
  include it: with drafting on, one chat wrote code at 79.3 tokens per
  second on two Sparks, against 65.1 on the v0.1.0 pair.
- On the layer pipeline, the first Spark now projects the model's head for
  only the rows one greedy chat needs, as one Spark already does, instead
  of the full head. Measured on a pair, prose went from 41.26 to 42.60
  tokens per second with the same tokens. `VADE_HEAD_BOUNDS=0` turns it
  off.
- On the layer pipeline, each Spark now keeps chat state only for the layers
  it runs, which leaves room for more chats at once: 15 instead of 9 in our
  two-Spark test with a 109 GB memory ceiling (the number depends on the
  memory each Spark has free). In that test, sixteen chats sent at once
  finished 22 % faster with drafting on and 6 % faster with drafting off, with
  the same answers.

### CLI

- New `vade pull <org>/<name>[@revision]` downloads a model from the Hugging
  Face Hub into `~/models/<org>/<name>`. It takes the weights, configs,
  tokenizer files, chat template, README and license, continues interrupted
  downloads, and checks each file's size and hash. It writes
  `.vade-model.json` in the folder and a `<folder>.sha256` list beside it,
  which `vade launch` and `vade up` check before they serve an FP4-Local
  checkpoint. New `vade models` lists local models with their size. For a
  supported model it gives the command that can serve it; for another model
  it gives a recovery step or the reason. See
  [Download models](docs/pull-models.md).
- FP8 is a supported route, not a legacy one. `vade up --fp8` starts
  `Qwen/Qwen3.8-Flash-Next-FP8` on two Sparks, and `vade up --fp8 --sparks 4`
  on four. Six Sparks start too, but no card has measured six. The older flag
  `--fp8-legacy` still works and does the same thing.
  The CLI help and messages now name `--fp8`. The install guide moved to
  [docs/install-fp8.md](docs/install-fp8.md). It gives the current steps and
  the 2026-10-28 cards at main `76977df`: with drafting ON, prose reached
  36.6 tok/s on two Sparks and 46.7 tok/s on four. FP8 on two and on four
  Sparks gives different token IDs: 7 of 66 responses matched in those cards.
  The ranks add their parts in a different order on four Sparks, and a fix is
  in screening
  ([benchmarks](docs/benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).
  FP8 requests take each sampling field they leave out from the checkpoint,
  so temperature 0 with neutral penalties gives greedy replies. The engine
  does not change.
- From a controller, `vade doctor` checks more before a start. It shows a
  GPU use row with the rule that `vade up` applies (a listed compute
  process, or a power draw of 30 W or more). It checks the pinned Rust
  toolchains before a source build, as `vade install` now does too. It
  checks each Spark's staged engine, PTX and helper against the manifest,
  as `vade up` does.
- Controller calls to curl ignore your `~/.curlrc`, so its options cannot
  change the replies that `status`, `up` and `doctor` read.
- More messages name the cause: a negative number or `--flag=value` in an
  option, a controller config that is not UTF-8 text, and the file that a
  failed staging step copied. `vade install` says when `prebuilt_dir` serves
  no bundle, before it falls back to a source build.
- `vade doctor` now shows the binary version, four launch defaults, model file
  counts and sizes, and Spark listener ports before startup.
- `vade install` and `vade up` no longer refuse on a stale GPU utilization
  counter. A GPU counts as busy only when a compute app is listed or power
  draw is at least 30 W. A stale counter prints a notice and the command
  continues.
- `sh scripts/install.sh` sets up the pinned Rust tools and source build on
  one Spark, checks an existing checkpoint, and prints recovery steps on failure.
- On a Spark, `vade install` now compiles the GPU code into Vade's private
  startup cache before you run `vade up`. It leaves checkpoint files alone.
- `vade install` from a controller now builds every Spark at the same time
  instead of one after another. It still prints each machine's output in
  order, and if a machine fails it stops with that machine's error, as before.
- `vade launch` no longer opens a chat when its input isn't a terminal, for
  example under `script` or with input from `/dev/null`. It prints the ready
  line and the chat command, exits successfully and leaves the engine running,
  so you can connect later with `vade chat`.
- When you run Vade by a path, such as `./vade` from the launcher bundle
  folder, the next-step hints from `vade launch` and `vade status` repeat that
  path (for example `./vade chat`) instead of a bare `vade` that your shell
  may not find.
- A build supervisor that crashed, was killed or was lost to a reboot no
  longer blocks every later `vade install`, `launch`, `up`, `update` and
  `down` with "Build supervisor was lost". Once its processes are gone,
  recovery closes the record and prints one line naming the job folder.
  While anything may still run it keeps the record and the error as
  before, and it still sends no signals.
- `vade test` help lists world 1, which the determinism test accepts, and
  `vade help` puts a blank line between the evaluation commands and the
  gate usage.
- Every `vade` help page has the same layout: a summary, usage, an aligned
  options table, examples and notes. `vade --help bench` and `vade -h bench`
  now print the bench help, as `vade help bench` does.
- Errors print as `error:` with what happened, then `hint:` with the next
  step. Launch, install, up, down and chat print their steps in one aligned
  column, and `vade status`, the bare `vade` menu and the launch plan print
  aligned rows with the next commands. Colour appears only on a terminal;
  `NO_COLOR` or `TERM=dumb` turns it off, and JSON output and receipts never
  use it.
- `vade up --dry-run` and `vade launch --dry-run` name the context and output
  limits they would use, and whether each one is explicit, saved or the
  default, instead of printing values such as `explicit context: None`.
- `vade doctor` prints one row per check (`ok`, `FAIL`, `skip` or `plan`,
  the check, what it found), the fix under each failed row, then one PASS or
  FAIL line with the counts. A failed check no longer stops the rest, and a
  check that an earlier failure hid is `skip`, not a guess. From a
  controller with no engine running, the engine row is `skip` and doctor
  exits 0 when the machine checks pass.
- `vade config` (new) names the setup the lifecycle commands will use
  (local, controller, legacy or none) and what it holds. An owned engine
  whose local config is gone is still the local setup, reported as an error.
- `--json` now works on `doctor`, `config`, `detect` and `version` as well as
  `status`: one object with a `schema` field, the same checks and exit code as
  without it, and failures inside the object.
- Help pages start with examples, and `vade --help` groups the commands.
  `vade verify` argument errors use the `error:`/`hint:` form on stderr, and
  verify prints progress on stderr while a request is still running.

### Harness

- A long chat no longer stops at the end of the context window. When the next
  turn would not fit, or at 80% of the window, the model writes one state block
  for the older turns (plan, decisions, open tasks, files, facts), the last 2
  turns stay word for word when they fit (else the last turn only), and tool
  outputs stay as references (path and line ranges). The chat shows one line:
  `[context compacted: kept N turns, summary M tokens]`. `/compact` does the
  same on request, `/context` shows the token budget after each turn, and
  `vade-harness agent --no-compact` turns the automatic step off. The journal
  records each compaction, so a resumed chat continues from it. If even a
  compacted prompt does not fit, nothing is sent and the chat says so. The
  engine and its token IDs do not change; see
  [context and compaction](docs/context-and-compaction.md).
- The harness now reads your project rules when a chat starts. In direct and
  agent mode, it puts `AGENTS.md`, `CLAUDE.md` or `.vade/instructions.md` from
  each folder between the git root and the workspace, and
  `~/.vade/instructions.md`, into the system prompt. It reads them again
  before a turn when they change. `/instructions` lists the files, and
  `--no-instructions` turns them off. A link can point only down to a file inside
  the project folder that a workspace read would allow, and a cut file is
  always marked. See
  [project instructions](harness/docs/project-instructions.md).
- The agent can read text-based PDF files. `workspace_read` on a PDF of up to
  50 MiB returns the text of each page under a `page N of M` line, and its
  offset and limit page through that text. The harness parses the file itself
  with the Rust standard library: cross-reference tables and streams, object
  streams, Flate, LZW, ASCII85, ASCIIHex and run-length data, and fonts with
  ToUnicode maps, standard encodings or `/Differences`. Encrypted files are
  refused, and scanned pages get a note that there is no text layer. Each read
  has one work budget for bytes read and decoded, parsed objects, cached data,
  tables, font entries, work steps and nesting depth. Every parser and decoder
  charges it before the work, so a hostile file stops at a limit, and a note
  names the limit and the page. A cancelled tool call stops the read, also in
  a child agent.
- The agent harness now rejects replies with an empty finish reason.
- New `vade-harness quality-card` measures answer quality and the cost of
  exactness. It runs one fixed set of 12 prompts (code, structured output,
  prose) against a served model and prints two cards: quality scores from
  exact checks only (code answers run under unit tests), and strict
  drafting against plain decoding with token ids compared and the verify
  cost read from the server draft receipts. See the
  [quality card guide](harness/docs/quality-card.md).
- Chat has two voices: your turns sit behind a `▌` bar on every row and the
  answer has none. A thought is one dim row (time, tokens, seconds) that
  Ctrl+T expands. One status strip above the `›` prompt shows the model,
  context used of the window, tokens in and out and tokens per second as raw
  numbers; an unknown count is left out, never shown as 0. Text wraps
  between words.
- Claude Code works through both gateways again. Claude Code 2.1.310 puts
  its environment notes in every request as a `system` role entry inside
  the messages, which the engine refuses with HTTP 400, so `claude -p`
  failed outright. On `/v1/messages` and `count_tokens`, both gateways
  now rewrite just those role values to `user`, the recovery Claude Code
  documents; every other byte is forwarded unchanged.
- `vade-harness setup claude` prints the gateway settings for Claude Code:
  the API-root base URL, a dummy bearer, every model alias and the
  subagents pointed at the loaded model, and betas, tool search and
  nonessential traffic off. `--dedicated-home` also writes a private
  Claude Code home with `CLAUDE_CONFIG_DIR`, the verified context and
  output limits (Claude Code otherwise assumes 32000 output tokens), read
  and search-only permissions, the tool rule and a completed onboarding
  flag. Checked with Claude Code 2.1.310: a plain answer and a real Read
  tool round trip.
- `vade-harness setup aider`, `setup continue` and `setup open-webui`
  print each client's documented settings for the engine or the gateway:
  aider's `OPENAI_API_BASE` and `openai/` model prefix, a Continue
  `config.yaml` for chat, edit and apply (no autocomplete, because the
  engine ignores FIM suffixes), and Open WebUI's `OPENAI_API_BASE_URL`
  with the Docker host and `--allow-host` note. These print or `--write`
  only; there are no dedicated homes for them.
- The built-in gateway (`vade-harness gateway`) relays `GET /v1/models`
  and `/models` to the upstream's model list, keeping the query, so Open
  WebUI, Continue and the OpenAI SDKs can list models before they chat.
  It used to answer 404. In cluster mode the list comes from one verified
  endpoint. `POST` gets 405 and `/v1/models/{id}` 404.
- Both gateways' own refusals (401, 403, 404, 413, busy, upstream
  unreachable) now carry an error type in the route's dialect:
  `{"type":"error","error":{...}}` on the Messages routes, as Claude Code
  and the Anthropic SDK expect, and the OpenAI shape with a type such as
  `rate_limit_error`, `overloaded` or `server_error` elsewhere. Upstream
  error bodies are still passed through byte for byte.
- `vade-harness gateway` relays `POST /v1/kv/release` (it answered 404),
  so a native session behind it can confirm a checkpoint release. With
  several upstreams, cancel and release no longer answer 503: they go, in
  configuration order, to every endpoint whose contract discovery has
  verified, and you get the first reply that reports cancelled or
  released, otherwise the first reply, or 502 when none answers. An
  endpoint that never verified, or that answered as foreign or with a
  changed contract, does not see the key until the verified contract
  answers again. One gap remains: no handle-based cluster affinity for
  requests that name a checkpoint handle.
- In cluster mode, `vade-harness gateway` sends cancel and release to
  every verified endpoint at once, under one 1.5-second deadline (or the
  client request's own, if sooner), instead of one after another with a
  30-second idle budget each. One stalled endpoint no longer keeps the
  others from hearing a cancel. `vade-gateway` still fans out one
  endpoint at a time.
- `vade-gateway` keeps listening after a failed accept, such as an
  aborted handshake or a full descriptor table. It drops only that
  connection and logs accept errors with a short backoff.
  `vade-harness gateway` does the same.
- When `vade-gateway` cannot record a request or write its statistics, it
  says what happened to the request (forwarded but not recorded, served
  without its statistics line, or recording incomplete) and names the
  folder or file to check, instead of a bare "recorder start failed".
  `demo-compaction` with an argument says to run it on its own.
- On Linux, including the Spark itself, the coding agent is no longer
  offered `workspace_shell`, which has no sandbox there and only ever
  returned `unsupported_platform`. The agent is told it cannot build, test
  or run anything, re-reads its edits and names the command for you to
  run, and `/status` says there is no command tool. macOS is unchanged,
  and sessions saved by an earlier Linux build still resume.
- `--children --child-shell NAME` is refused while the arguments are
  parsed, with a message that names the cause, where the shell is not
  offered (Linux, or `VADE_HARNESS_SHELL=off`), instead of failing later
  with "invalid child tool scope".

### Documentation

- The README opens with a short summary and separate quick starts for one
  Spark and for two or more, and the benchmarks page adds the 2026-10-25 two-
  and four-Spark cards.
- The README and the [benchmarks page](docs/benchmarks.md) show measured
  v0.1.0 results, and the receipts behind them are published under
  [bench/receipts/2026-10-24](bench/receipts/2026-10-24/README.md).
- The 2026-10-25 cards ran on main `6cb46bf` and `80ac2d4`. Each row names
  its result folder; those folders are not in the repository yet.
- The [architecture page](docs/architecture.md) follows one request from
  the gateway through the scheduler to the model, and explains the storage
  formats, the kernels, drafting, the layer pipeline and the prefix cache.
- A new [examples folder](examples/README.md) has short scripts, each shown
  with the output it printed on a Spark: curl against the OpenAI and
  Anthropic endpoints, a standard-library Python client, Claude Code and the
  Codex CLI pointed at Vade, a `vade bench` run that reproduces the published
  one-Spark number, and a two-Spark launch from a controller that is a dry
  run by default.
- The results figure compares one Spark with two, with drafting on. In the
  2026-10-24 runs, a second Spark answered a long prompt sooner but didn't make
  a single chat write faster. Eight chats at once vary from run to run, and the
  figure shows the range we measured: 130.6 to 173.9 tokens per second on six
  single Sparks and 167.6 to 168.7 on three pairs, from the same-day cards with
  drafting on. The headline is checked against all nine cards.
- [INSTALL](INSTALL.md) covers more situations: an engine that turns away an
  extra chat with HTTP 503, checking each replica with `vade verify --url`,
  keeping the launcher bundle outside your checkout, and what to do when
  `vade install` refuses a checkout with local changes.
- The guides were rewritten in plainer language. Commands, numbers and links
  didn't change.
- Two facts were corrected against the source: rule files are capped at
  16 KiB each, and the automatic output reservation includes the
  16,384-token cap.
- README, [INSTALL](INSTALL.md) and the one-Spark quickstart say what chat,
  status, sampling and the bundle really do, checked against the source:
  controller `vade chat` always opens the agent, which refuses a context
  below 8192, and only local chat falls back to direct chat; local
  `vade up` steps a refused default or saved context down as `vade launch`
  does; controller `vade status` prints no URL, since the API is on the
  local port; naming any sampling field fills the rest from the checkpoint
  defaults; and the two legacy routes are named, with `vade run` needing a
  fresh `VADE_HOME` after `vade launch`.
- [INSTALL](INSTALL.md) and the CLI guide say that a controller install
  needs `vade-prebuilt-v1`, not the launcher bundle. INSTALL and the
  [benchmarks page](docs/benchmarks.md) say how a `vade bench` with no
  flags differs from a RigMark card.
- The [security policy](SECURITY.md) describes the gateways'
  `VADE_GATEWAY_TOKEN`. It applies to the gateways you run by hand; the
  replica gateway behind `vade up --replicas` never takes a token.
- The harness guide and the invisible-gateway page list the complete
  gateway sampling policy again: every shared default with its exact
  value, every sampling field, and the null rule.
- Engine docs: a new [unsafe-code page](engine/docs/unsafe.md) names the
  four boundaries where the engine uses `unsafe` (kernels, the CUDA
  driver, launches, RDMA), gives a per-package inventory and the exact
  count of `SAFETY` notes, lists the lints that make `unsafe` a compile
  error elsewhere, says how to write a `SAFETY` note, and states plainly
  that the ratchet does not count unsafe sites today, so review is the
  check. A new [module map](engine/docs/module-map.md) follows one request
  through `flash-serve` and names the main modules per package and where
  the tests are. Both are linked from the engine docs index.
- Engine docs corrections: messages are capped at 4,096, and eight spare
  connections are kept for `/health` and `/v1/cancel` once requests fill
  the 32 (there is no single control connection); Flash Next is not a
  separate research engine, the v0.1.0 model runs in `flash-serve`,
  started by `vade launch` and `vade up`; and the architecture page says
  the layer pipeline's drafting ceiling is 7.
- Smaller fixes: the harness guide puts the source-built harness on PATH,
  the network guide's RDMA GID example runs as written, the fresh-install
  guide's legacy controller examples match current output, the
  adding-a-model anchors point at the moved CLI module docs, and the
  benchmarks page and CLI guide are clearer about bench `ignore_eos`,
  `--checkpoint-hash` and the controller config.

### Tools

- The README states that CI runs locally. A new CI guide gives public host
  build and test commands and explains the private CI script's coverage.
- The security policy states which engine routes local gateway callers can
  reach. It lists access refusals and host tests for the gateway checks.
- The README states the engine's scope and limits. Contribution and security
  guides explain the proof and report steps. Issue forms ask for checkpoint
  details and token IDs. Agent rules now state the same proof and writing
  rules. The benchmark page adds one-Spark results and repeat commands.
- `vade-tools bench-docs <cards-dir> <repo-dir>` refreshes the marked benchmark
  tables from the newest measured main revision. It keeps the raw card values,
  measurement times and revision in both pages. It resolves drafting from the
  request and the recorded server default. An effective depth of 0 means OFF.
  It refuses an unknown drafting mode or depth.
- All repository tools now use Rust and the standard library. There are no
  Python files or Python setup requirements. The tools crate checks links and
  release bundles, tests shell examples, builds CI CUDA linkage stubs and draws
  the release figure. The model helper uses Rust validation and curl. API
  examples are available in shell and Rust. Interrupted model downloads release
  their lock after curl stops. Stalled transfers fail with a time limit. The
  shell API example reports stream errors and incomplete streams as failures.
- The chart tool `vade-tools benchmarks-chart` reads the release-card
  numbers from the committed receipts, reads and writes UTF-8, names a
  missing receipt folder, and stops if the cards stop supporting the
  figure's headline. The assets README lists both figures and gives the
  bar colours as hex values.
- `harness/scripts/route-a-check.sh` checks Route A on one Spark in one
  command: toolchain, CLI build, model, `vade launch`, one chat request,
  `vade verify` and `vade down`, each with PASS or FAIL and its seconds. It
  runs in a new private Vade home of its own, so it never touches `~/.vade`,
  and stops its engine also after a failed step or an interrupt. See
  [INSTALL](INSTALL.md#check-the-whole-route-in-one-go).

### Project

- A new agent task suite under
  [harness/tests/agent-suite](harness/tests/agent-suite/README.md) holds
  12 mixed tasks for the coding agent, each with a deterministic check.
  `cargo test` checks that every file a task names exists and is tracked;
  it does not run the tasks.
- Bug reports and feature requests now use issue forms. The bug form asks for
  the version, model, number of Sparks, how you run Vade, the exact command
  and the full error, plus `vade verify` output when answers are wrong.
- The security policy explains how to report a vulnerability privately, what
  it covers, and what Vade exposes on the network by default.
- [CONTRIBUTING](CONTRIBUTING.md) says how a pull request lands: CI, a
  maintainer's review (requested through CODEOWNERS), the device gate for
  engine changes, and the published update that carries it.
- Every public item in the CLI crate and the harness library has a doc
  comment (140 harness items had none), the CLI's `main.rs` maps its
  modules by area, and rustdoc builds without warnings.
- No CLI source file is over 1,200 lines any more, and every non-test
  harness source file is under 1,200 lines: tests moved beside their
  modules and the largest files were split along their seams, with no
  change in behaviour.
- New tests pin the CLI's `vade doctor`, `vade status`, `down`, `logs`,
  `detect` and `update` messages, its remaining argument errors and every
  state in which controller chat refuses to open; the engine's bounded
  JSON reader and admission refusal text, prompt partitions and loopback
  rank admission, the fabric peer layout and profiling wrapper, and the
  tokenizer against a real-format `tokenizer.json`; and the native
  `/status` tool list per platform. The harness tests no longer depend on
  an exported `VADE_GATEWAY_TOKEN`, the native tool tests run in both
  shell shapes, which fixes Linux CI, and the loopback admission tests
  retry a port lost to a parallel process.
- Clippy findings that pointed at real issues are fixed in the CLI and the
  harness, and the CLI has one SHA-256 implementation instead of two.

## v0.1.0 - 2026-10-24

The first public release. Vade runs a large language model on your own NVIDIA
DGX Spark and gives you three things: a coding agent in your terminal, a plain
chat, and an OpenAI-compatible API. It serves one model,
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (text only), on one Spark or on linked
Sparks. It is written in Rust using only the standard library, and it is built
so that the same request on the same setup returns the same token IDs on every
run; `vade verify` lets you check that on your own machine.

### Engine and serving

- An inference engine for the NVIDIA DGX Spark, written in Rust using only the
  standard library. The release path is the text-only Qwen Flash-Next
  FP4-Local model, on one Spark by default. Configured multi-Spark
  deployments run through the exact layer pipeline, which splits the model's
  layers across the Sparks.
- Drafting is on by default: an extra model head proposes tokens and the model
  verifies them. A request that sets no sampling field is decoded greedily, on
  any number of Sparks. A request that sets any sampling field starts from the
  checkpoint's own sampling defaults, and its values replace those defaults.
- Drafting on two Sparks. Both pipeline stages follow the same drafting
  policy for a cohort (a group of requests running at the same time), and the
  two-stage pipeline uses guarded cohort drafting, which switches a cohort
  back to plain decoding when drafting stops paying off.
- At startup, the pipeline refuses to run if its ranks (one process per
  Spark) have different artifacts, numerical settings or prompt options. It
  also checks the pipeline switches before it chooses a topology.
- `/health` and receipts report the drafting mode and depth the server
  actually ran, not the ones the request asked for.
- Release builds load verified precompiled kernels. Checkpoint hashing uses a
  safe streaming SHA-256 that does not fill the file cache.
- The startup memory check counts reclaimable page cache as free (Linux
  `MemAvailable`, capped by CUDA total), so it also works inside containers,
  where CUDA's own free-memory figure leaves the page cache out. A memory
  refusal names MemAvailable, the required bytes and the ceiling. Session
  pools hold back a 512 MiB reserve, so 8 and 16 concurrent chats at the
  default context have room to finish.
- The Flash-Next prefix cache is off by default. It stays off until tests show
  that serving with the cache is as fast as serving without it in every
  measured case. `VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on, on one Spark
  only.

### Model and installation

- A pinned, verified FP4-Local publisher revision with a committed SHA256
  manifest of the complete snapshot, plus a one-line fetch helper that checks
  the download against that revision.
- A replay-tested local install on one Spark. The launcher reads the Spark's
  identity and resources, finds the models you already have, generates local
  defaults, and runs install, up and chat in one terminal.
- A prebuilt launcher bundle for Linux aarch64. A release workflow builds it on
  version tags, checks every manifest hash, and attaches the bundle and its
  `SHA256SUMS` to the tag's release. If a bundle is rejected, the launcher
  falls back to building from source.
- Guides for first-time readers. README, INSTALL and the first-run guide
  start with the one-Spark launcher, keep the source route and the bundle
  route apart, and name the supported model explicitly.
- A two-Spark route in INSTALL, next to the one-Spark default.
- A supported-systems table in INSTALL. The engine runs on DGX Spark / GB10
  under Linux `aarch64`. For Route B, where you drive the Spark from a
  second computer (the controller), the controller can be Linux `aarch64` or
  `x86_64`, or macOS. For Route A, where the CLI, engine and chat all run on
  the Spark, a Windows computer is supported as the SSH client, but not as a
  controller.
- Source installs also work from a downloaded source archive with no Git
  metadata. The launcher records the archive's stamped commit as the build id,
  or builds as `unknown` when the archive carries none.

### CLI, chat and gateway

- Controller commands for install, launch, up, status, logs, verify and
  shutdown, with bounded JSON status, specific errors that say what to do next,
  and help that stays usable in pipes.
- `vade up` on the controller and `vade launch` on the Spark start the same
  FP4-Local defaults (context 32768, output 8192). If the engine refuses the
  default or saved context for memory, `vade launch` steps it down through
  16384, 8192 and 4096 and saves the result once a start succeeds; that is the
  only case where it steps down. An explicit `--context` that the engine
  refuses is an error.
- `--context 262144` keeps the single-chat long-context preset, and
  `vade status` prints the chats the engine admitted (how many it allows to
  run at once at this context). A controller bench measures concurrent chats
  instead of one queued chat.
- `vade up --replicas N` (2..16, FP4-Local) starts an independent one-Spark
  server on each of N Sparks behind one local `vade-gateway`, which sends each
  request to the least-busy replica. `vade-harness vade-gateway` runs the same
  gateway by hand, with `--upstream` repeated.
- A machine's `ssh` entry can be a `user@host` destination as well as an
  alias, host name or IPv4 address.
- `vade verify` checks that repeated greedy runs against a running server
  return the same tokens, and finds the endpoint of the server it owns
  automatically.
- Chat history lives under the Vade home (`~/.vade/projects`), never in the
  folder you chat from. Set `VADE_CHAT_IN_PROJECT=1` to keep it in a
  git-ignored `./.vade` instead.
- Chat and coding-agent workflows: steer a chat while it runs, keep replies
  that were interrupted, manage sessions, and use an OpenAI-compatible HTTP
  gateway with credential handling and recorded replay.
- Both gateways refuse browser-driven requests by default. They refuse a
  request whose Host header is anything other than loopback with the listener
  port, and any request that carries an `Origin` header. You can allow a
  browser client on purpose with exact allow-lists and an optional bearer
  token.

### Benchmarks

- `vade bench` runs with no flags. It finds the recorded endpoint, reads the
  model from `/health`, uses a bundled offline prompt and measures the
  server's own drafting default.
- Drafting-aware receipts (the saved result records) that keep the requested
  drafting mode apart from the mode and depth the server reports, and withhold
  rates when the two disagree.
- A benchmark page with methods, results and limitations, linking every quoted
  source folder to its published, sanitized receipt.
- A results chart in the README and on the benchmark page, drawn by a
  standard-library script from the committed receipts, so every label is a
  value that a receipt prints.

### Project and community

- Agent rules in AGENTS.md and the human contribution process in
  CONTRIBUTING.md, with a package map, architecture notes and decisions.
- A security policy, Code of Conduct, issue and pull request templates, NOTICE
  and the bundled third-party license texts.

For measurements, recipes and their limitations, see
[benchmark methods and results](docs/benchmarks.md). See [NOTICE](NOTICE) for
bundled third-party material and [LICENSE](LICENSE) for Vade's Apache-2.0
license.
