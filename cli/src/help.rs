//! The text of `vade --help` and `vade <command> --help`, in one layout:
//! summary, usage, options, examples and notes (see [`crate::style::Page`]).
use crate::style::{self, Page, Stream, Token};

const LIFECYCLE: &str = "Explicit VADE_CONFIG selects the controller; otherwise generated local setup wins. Keep the same VADE_CONFIG and VADE_HOME from start through chat, status, logs and down.";
const TRANSPORT: &str = "One Spark needs no fabric or RDMA; linked Sparks use RDMA by default. Set SEQPAR_RDMA=0 for TCP using the configured fabric addresses on any reachable interface; QSFP and RDMA memlock are unnecessary in TCP mode. Two FP4-Local Sparks serve only over RDMA; FP8 on two Sparks may use TCP.";
const GPU_BUSY: &str = "Only a running compute app or power draw of at least 30 W makes a GPU busy; a stale utilization reading is reported and never blocks.";
const DRY_RUN: (&str, &str) = ("--dry-run", "Show what would happen; change nothing");
const API_KEY: (&str, &str) = ("--api-key-file PATH", "Bearer credential file (or VADE_API_KEY_FILE)");
const CONTEXT: (&str, &str) = ("--context N", "Context window in tokens, 4096..262144 (default 32768)");
const OUTPUT: (&str, &str) = ("--output N", "Output limit in tokens, 1024..16384 (default 8192, or the context if smaller)");
const MTP: (&str, &str) = ("--mtp N", "Drafting depth 0..3; 0 turns drafting off (default: the engine chooses)");
/// The README's model command. It stays on one line, wider than [`style::WIDTH`], so it pastes as it is.
pub const PINNED_PULL: &str = "vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300";

/// The help page for `command`, or the top-level page; headings are styled
/// when standard output takes color.
pub fn help(command: Option<&str>) -> String {
    let on = style::color(Stream::Out);
    match command.and_then(page) {
        Some(page) => page.render(on),
        None if command == Some("verify") => crate::verify::help(),
        None if command == Some("read") => crate::read::help(),
        None => top(on),
    }
}

fn top(on: bool) -> String {
    let heading = |name: &str| style::style(Token::Heading, name, on);
    // Command groups share one name column; option groups size their own.
    let group = |name: &str, rows: &[(&str, &str)], width: usize| {
        let rows: Vec<(&str, String)> = rows.iter().map(|(name, text)| (*name, (*text).to_owned())).collect();
        format!("{}\n{}\n", heading(name), style::table_at(&rows, 2, width))
    };
    let mut out = format!("{} runs Qwen3.8 Flash-Next on your DGX Spark, as FP4-Local or FP8.\n\n{}\n  {PINNED_PULL}\n", style::style(Token::Title, "Vade", on), heading("Examples:"));
    out += &style::table(&[
        ("", "Download the model and check every file".into()),
        ("vade up", "Start the engine on this Spark".into()),
        ("vade chat", "Chat with the running engine".into()),
        ("vade status", "See what is running".into()),
        ("vade down", "Stop what vade started".into()),
    ], 2);
    out += &format!("\n{} vade <COMMAND> [OPTIONS]\n\n", heading("Usage:"));
    out += &group("Start and stop:", &[
        ("setup", "Ask which Sparks, checkpoint and model; install, start, check"),
        ("launch", "Install if needed, start on this Spark, then open chat"),
        ("up", "Start without chat (this Spark, or Sparks from a controller)"),
        ("down", "Stop what vade started"),
    ], 8);
    out += &group("Use:", &[
        ("chat", "Chat with the running engine"),
        ("sessions", "List, export or delete saved chat sessions"),
        ("read", "Print a file as Markdown (text, HTML, CSV, PDF, office, data, code)"),
        ("tools", "List the tools chat offers: native and MCP"),
        ("mcp", "Add, list, test, trust or remove MCP tool servers"),
    ], 8);
    out += &group("Models:", &[
        ("pull", "Download a model from Hugging Face, with resume and checks"),
        ("models", "List the models on this machine and what can serve them"),
    ], 8);
    out += &group("Look:", &[
        ("status", "Show what is running and whether it is healthy"),
        ("config", "Show which setup is selected and what it holds"),
        ("doctor", "Check the setup, one line per check, then PASS or FAIL"),
        ("detect", "Show this Spark's machine, memory, disk, driver and model"),
        ("logs", "Show recent engine logs"),
    ], 8);
    out += &group("Prove:", &[
        ("verify", "Check exact token IDs across runs, drafting ON and OFF"),
        ("bench", "Measure a running server from this client"),
        ("test", "Run the build, determinism and long-context gate checks"),
    ], 8);
    out += &group("Maintain:", &[
        ("install", "Save local defaults and install the engine and chat"),
        ("update", "Refresh the installed engine and chat"),
        ("run", "Run the separate legacy Qwen3-8B BF16 workflow"),
        ("version", "Print the CLI version and the commit of this build"),
    ], 8);
    out += &group("Options:", &[
        ("--json", "JSON for status, doctor, config, detect, models, version"),
        ("--dry-run", "Preview a lifecycle command; others run as usual"),
        ("--api-key-file PATH", "Credential for chat, doctor, verify, bench and local run"),
        ("-h, --help", "Show help"),
        ("-V, --version", "Print the CLI version and the commit of this build"),
    ], 0);
    out += &group("Environment:", &[
        ("VADE_HOME", "Local install root and saved defaults (default ~/.vade)"),
        ("VADE_CONFIG", "Controller config file; selects the controller (see Notes)"),
        ("VADE_GROUP", "A [group] in the controller config: machines, state, port"),
        ("VADE_API_KEY_FILE", "Credential file when --api-key-file is not given"),
        ("NO_COLOR", "Plain output without color"),
    ], 0);
    out += &format!("{}\n", heading("Notes:"));
    for note in [
        "Explicit VADE_CONFIG selects the controller for up, install, chat, status, logs, down and doctor, even after local setup; keep it set to the same file throughout that lifecycle. Without VADE_CONFIG, saved local setup wins over the user config directory.",
        "Put options after the command name. bench, verify and test do not take --dry-run or --json.",
    ] { out += &format!("  - {}\n", style::wrap(note, style::WIDTH, "    ")); }
    out += &format!("\nLearn more: {}\n", style::style(Token::Command, "vade <command> --help", on));
    out
}

fn page(command: &str) -> Option<Page<'static>> {
    Some(match command {
        "launch" => Page {
            summary: "Install if needed, start Vade on this Spark, then open chat.",
            usage: &["vade launch [OPTIONS]"],
            options: &[CONTEXT, OUTPUT, MTP, ("--memory-bytes N", "Memory budget for this Spark, in bytes"), ("--cache-bytes 0", "Turn the retained prompt cache off"), DRY_RUN],
            examples: &["vade launch", "vade launch --context 262144 --output 16384", "vade launch --dry-run"],
            notes: &[
                "Finds the FP4-Local checkpoint, saves defaults, installs if missing and starts the engine. It opens chat when standard input is a terminal. Run it again to attach.",
                "Local defaults match vade up from a controller: context 32768, output 8192 (or the context if smaller), drafting depth chosen by the engine.",
                "If the engine refuses the default or saved context for memory, launch tries smaller sizes from 32768, 16384, 8192 and 4096. It saves the first size that starts. A refused explicit --context is an error.",
                "The checkpoint comes from the saved config, then explicit VADE_MODELS, then the FP4-Local checkpoint under ~/models; launch asks if it is missing. Files install under ~/.vade (or VADE_HOME).",
                "Local launch does not take --chats or a positive --cache-bytes. See vade up --help for memory options. Unset VADE_CONFIG to use the local setup.",
                "Launch also takes --fp4-local, --sparks 1 and --fast-prompts from vade up; they change nothing here.",
            ],
        },
        "up" => Page {
            summary: "Start the engine without opening chat.",
            usage: &["vade up [OPTIONS]"],
            options: &[
                ("--fp4-local", "Qwen3.8 Flash-Next FP4-Local (the default; optional)"),
                ("--fp8", "Qwen3.8 Flash-Next FP8 on 2 or 4 Sparks; 6 Sparks start but are not covered (older spelling: --fp8-legacy): context 131072, output 65536 (or the context if smaller), drafting depth 2; output limit up to 262144"),
                ("--sparks a,b", "The Sparks as you type them after ssh, in any order. Vade reads the cables and sets them up"),
                ("--sparks N", "How many Sparks of your setup to start, from 1 to 24"),
                ("--link FROM:TO=ADDR", "With --sparks a,b: FROM's own address on its cable to TO, to override what Vade reads"),
                ("--hosts a,b", "Machines of your config, in cable order; this never changes the setup"),
                ("--replicas N", "2..16 independent one-Spark FP4-Local servers behind a local gateway"),
                ("--pairs", "With --replicas 2..12: N two-Spark tensor-split servers on 2N Sparks instead"),
                MTP, CONTEXT, OUTPUT,
                ("--chats N", "Concurrent FP8 chats, 1..32 (controller)"),
                ("--memory-bytes N", "Memory budget per rank, in bytes"),
                ("--cache-bytes N", "FP8 retained prompt cache, in bytes; 0 disables it. Local FP4 accepts only 0"),
                ("--fast-prompts", "Older option, kept for scripts; it has no effect"),
                DRY_RUN,
            ],
            examples: &["vade up", "vade up --sparks spark-a,spark-b,spark-c,spark-d", "vade up --context 262144 --output 16384", "VADE_CONFIG=cluster.conf vade up --sparks 2"],
            notes: &[
                "Default: Qwen3.8 Flash-Next FP4-Local on one Spark.",
                crate::config::FP4_SPARKS,
                crate::config::FP8_SPARKS,
                "--sparks a,b is all you need for 1 to 24 Sparks. Vade checks each Spark over SSH, reads the cables, prints the chain, copies itself to each Spark and starts them. Type the Sparks in any order. Each name is what you type after ssh, or user@address.",
                "A new list replaces the last setup: one line says so, the old file moves aside, and a running engine stops first. The same Sparks in the same order start with no new setup. After that, vade up alone starts the same Sparks, vade down stops them and vade chat opens a chat.",
                "Local up and launch start with context 32768 and output 8192 (or the context if smaller). They reuse saved limits. A memory refusal lowers a default or saved context to a smaller size from 32768, 16384, 8192 and 4096. The first size that starts is saved. A refused explicit --context is an error. Local --chats and positive --cache-bytes are refused; --cache-bytes 0 disables retention.",
                "Controller FP4 defaults match local ones on one Spark: context 32768, output 8192 (or the context if smaller). On 2 or more Sparks in one group, a chain or a pair, the default context is 131072; if it does not fit in memory, up falls to 32768 and says so. --context 262144 requests a larger context. The engine still sets the chat count from context and memory.",
                "Controller FP4 checks and passes on --chats and --cache-bytes, but its engine does not use them: it admits chats from memory and uses its own prefix cache setting.",
                "The engine chooses the drafting depth (MTP). On one FP4-Local Spark, concurrent chats can draft; the default request depth is 3 with strict verification. FP8 concurrent chats do not draft. A request with draft=plain turns drafting off; draft=strict asks for it when supported. --mtp sets the depth, and 0 turns drafting off.",
                "Two FP4-Local Sparks run the tensor split, the only two-Spark path for FP4-Local: both Sparks work on every token. It needs RDMA, so vade up refuses SEQPAR_RDMA=0 (TCP) for two FP4-Local Sparks. FP8 on two Sparks keeps its ring and may use TCP.",
                "FP8 needs an FP8 config such as cli/examples/vade.conf. See docs/benchmarks.md.",
                "--sparks N above 2 needs a setup with that many Sparks in cable order.",
                "VADE_GROUP=NAME selects a [group NAME] from the config: its hosts in rank order, its own ownership record and its own local_port. A group must be an arc of the cabling: each host is cabled to the next, in rank order; a closing cable is not needed, so any run of neighbors on a ring works. Set the same VADE_GROUP for status, chat, logs and down. Groups in one layout can run at the same time. See docs/eight-sparks.md.",
                "--replicas N starts one independent one-Spark server on each of the first N machines (or the N --hosts) and a local gateway on local_port; replica tunnels use the next N local ports. Each replica uses the FP4 defaults above. Use it when the model fits one Spark and many chats run at once; answers are the same on every replica.",
                "--replicas N --pairs makes each replica a two-Spark tensor split: machines 1 and 2 form the first pair, 3 and 4 the next, and each pair must share a fabric link. Only each pair's first machine gets a tunnel. The gateway keeps a chat's later turns on its pair while that pair has fewer than 8 requests in flight or is among the least busy, and gives each chat a stable prompt cache key. This is an option; the default four-Spark mode is unchanged. See docs/four-sparks.md.",
                "One Spark needs no fabric. Linked Sparks use RDMA by default. SEQPAR_RDMA=0 selects TCP, except for two FP4-Local Sparks. Keep VADE_CONFIG and VADE_HOME consistent through chat, status and down.",
                GPU_BUSY,
            ],
        },
        "down" => Page {
            summary: "Stop the engine and tunnel started by vade.",
            usage: &["vade down [--dry-run]"],
            options: &[DRY_RUN],
            examples: &["vade down", "vade down --dry-run"],
            notes: &[LIFECYCLE],
        },
        "status" => Page {
            summary: "Show the engine, machines, build, settings and health.",
            usage: &["vade status [--json] [--dry-run]"],
            options: &[
                ("--json", "Print one vade.status.v1 object; exit 0 for stopped or ready, 1 for errors or an unhealthy state"),
                ("--dry-run", "Read saved state only; unchecked values are null"),
            ],
            examples: &["vade status", "vade status --json"],
            notes: &["Checks share a 15-second deadline.", LIFECYCLE],
        },
        "logs" => Page {
            summary: "Show recent engine logs.",
            usage: &["vade logs [--dry-run]"],
            options: &[DRY_RUN],
            examples: &["vade logs", "vade logs | tail -20"],
            notes: &[LIFECYCLE],
        },
        "chat" => Page {
            summary: "Chat with the running engine.",
            usage: &["vade chat [--resume ID|NAME|last] [--name NAME] [--api-key-file PATH] [--dry-run]"],
            options: &[
                ("--resume ID|NAME|last", "Continue a saved session of this folder; last is the newest one with a turn"),
                ("--name NAME", "Give this session a name (renames a resumed one)"),
                API_KEY, DRY_RUN,
            ],
            examples: &["vade chat", "vade chat --name \"login bug\"", "vade chat --resume last", "VADE_CONFIG=cluster.conf vade chat"],
            notes: &[
                "Opens the Vade agent on the resident engine. Start the engine once with vade launch or vade up.",
                "Local contexts below 8192 tokens use direct text chat. On Linux the agent reads, searches and edits files but runs no shell commands or checks; a local launch prints a notice that says so.",
                "Typing during a reply interrupts it and keeps its partial answer in context. /queue on queues follow-ups; /queue off restores steering. /exit leaves the engine running.",
                "Chat history is kept in ~/.vade/projects (VADE_HOME changes the root), never in this folder; VADE_CHAT_IN_PROJECT=1 keeps it in a git-ignored ./.vade.",
                "Each agent chat is a saved session; direct text chat saves none. vade sessions lists them. A resume replays the saved history into the context; it reruns no tools and restores no files.",
                LIFECYCLE,
            ],
        },
        "sessions" => Page {
            summary: "List, export or delete saved chat sessions.",
            usage: &["vade sessions [list] [--all]", "vade sessions export ID|NAME|last [--markdown]", "vade sessions delete ID"],
            options: &[
                ("--all", "List the sessions of every project, not only this folder"),
                ("--markdown", "Export a readable Markdown transcript (the only format)"),
            ],
            examples: &["vade sessions", "vade sessions export last --markdown > chat.md", "vade sessions delete session-1793250720000-4242-0"],
            notes: &[
                "Rows show the ID, name, start, last turn, turn count, model and project, newest first. Times are UTC.",
                "Sessions belong to the folder where vade chat ran. Run export and delete from that folder.",
                "No engine is needed. delete refuses a session that a chat has open.",
                "Direct text chat (contexts below 8192 tokens) saves no transcript, so it has no sessions.",
            ],
        },
        "doctor" => Page {
            summary: "Check the setup: one line per check, then PASS or FAIL.",
            usage: &["vade doctor [--json] [--api-key-file PATH] [--dry-run]", "vade doctor --fabric [--ring] [--json] [--dry-run]"],
            options: &[("--fabric", "Check every configured cable, port, group arc and clock (controller)"), ("--ring", "With --fabric: expect one closed ring of every machine at one line rate"), ("--json", "Print one vade.doctor.v1 object with every check"), API_KEY, DRY_RUN],
            examples: &["vade doctor", "vade doctor --json", "VADE_CONFIG=cluster.conf vade doctor --dry-run", "VADE_CONFIG=cluster.conf vade doctor --fabric", "VADE_CONFIG=ring8.conf vade doctor --fabric --ring"],
            notes: &[
                "On this Spark: config, VADE_HOME writable, ownership, installed binaries, model files, nvidia-smi and driver, disk, free memory, port and engine health.",
                "From a controller: local tools, then SSH, nvidia-smi, GPU use, disk, free memory, model files and the staged build on each selected Spark, then the fabric, the port, the harness and engine health.",
                "Also shows the vade binary and version, context, output, drafting and prefix cache settings, model JSON files, safetensors count and bytes, and server ports on each Spark.",
                "Doctor checks the active Sparks, or the default vade up selection, through the same SSH options. It does not start an engine or change the setup.",
                "--fabric checks every machine in the controller config: the cabling shape, each [group] arc, each fabric port (address, 100000 Mb/s or more, MTU 9000, active RDMA device), each cable with both ends in the config and each of its two rails on its own (one subnet, speed, MTU, RDMA, 9000-byte pings both ways) and each clock (within 1000 ms, time sync on). See docs/eight-sparks.md.",
                "--ring also expects one closed ring of every machine: each Spark has exactly two neighbors in the config and the links close into one ring, and rail 0 and rail 1 of every cable link at one speed each.",
                "Each FAIL row gives the next step. Exit 0 means PASS, 1 means FAIL.",
                TRANSPORT, LIFECYCLE,
            ],
        },
        "setup" => Page {
            summary: "Ask a few questions, then install, start and check Vade.",
            usage: &["vade setup [OPTIONS]"],
            options: &[
                ("--yes", "Ask nothing; take the defaults and the flags below"),
                ("--hosts a,b", "The Sparks, as you type them after ssh, in any order. This is the same as vade up --sparks a,b"),
                ("--fp8", "The FP8 checkpoint on 2 or 4 Sparks (6 Sparks start but are not covered); --fp4-local is the default"),
                ("--replace-local", "Older option, kept for scripts; it has no effect"),
                ("--model PATH", "The model folder, absolute or under ~/ (on each Spark from a controller)"),
                ("--link FROM:TO=ADDR", "FROM's own address on its cable to TO, to override what Vade reads; once per cable end"),
                ("--dry-run", "Print the answers, the config and the steps; change nothing"),
            ],
            examples: &["vade setup", "vade setup --yes", "vade setup --yes --hosts spark-a,spark-b", "vade setup --yes --hosts spark-a,spark-b --dry-run"],
            notes: &[
                "Asks which Sparks to use (on a Spark the default is this Spark), FP4-Local or FP8 (only for 2, 4 or 6 Sparks; 6 Sparks are not covered yet), and where the model is. Each question shows its default in brackets; press Enter to take it.",
                "On this Spark, vade install saves the local defaults in ~/.vade/config. From another computer, setup writes a controller config at the path vade config shows (VADE_CONFIG, else the user config folder). It points at the installed bundle when vade runs from one, else at this checkout, and keeps Vade's private folders on this computer under ~/.vade. With no list of Sparks, setup uses the config that is there. A list moves that config aside.",
                "Then it runs vade install, vade up and vade doctor, and stops at the first step that fails. Doctor runs last, so it checks the Sparks that up started. Run vade setup again after a fix, without the answer options: it uses the setup it finds and repeats only the work that needs it. When a controller config already has an install, setup skips vade install; after a pull, run vade update.",
                "A list of Sparks, from --hosts or from the question, replaces the setup in place with no flag: a local setup moves to config.local, a controller config moves to config.before, and a running engine stops first. Setup deletes nothing. Press Enter at the question to keep the setup. It asks for a cable address only when it cannot read one.",
                "Without a terminal, setup asks nothing; add --yes. It does not download models: see vade pull.",
                LIFECYCLE,
            ],
        },
        "install" => Page {
            summary: "Install the engine, GPU kernels and chat for this setup.",
            usage: &["vade install [--dry-run]"],
            options: &[("--dry-run", "Print the plan: where, which files, model, state; change nothing")],
            examples: &["vade install", "vade install --dry-run", "VADE_CONFIG=cluster.conf vade install --dry-run"],
            notes: &[
                "Files come from the first that applies. On this Spark: the bundle in VADE_BUNDLE, a launcher bundle next to vade, then a source build of the checkout with the pinned Rust toolchains. From a controller: a prebuilt_dir bundle that matches the engine and CLI revisions, then a source build on each Spark.",
                "On this Spark, running it again is safe: files that still match their record are reused. From a controller, each install builds a new generation that a running engine does not use; while an engine runs, use vade update, which asks before a restart. Model downloads stay in your hands.",
                "On this Spark (saved local defaults, or no setup yet): find the model, save defaults once in ~/.vade/config, stage the engine, kernels and chat, and fill the GPU startup cache so the first vade up does not compile.",
                "From a controller (VADE_CONFIG, or the default controller config): check the tools, SSH, GPUs, disks and models, copy or build on each Spark with SHA-256 checks, then build the chat program here.",
                "On a computer that is not a Spark and has no controller config, run vade setup first.",
                "A generated run qwen3-8b installation uses run again; install requires an explicit VADE_CONFIG to select a controller.",
                GPU_BUSY,
                TRANSPORT, LIFECYCLE,
            ],
        },
        "update" => Page {
            summary: "Refresh the installed engine and chat.",
            usage: &["vade update [--dry-run]"],
            options: &[DRY_RUN],
            examples: &["vade update", "vade update --dry-run"],
            notes: &[
                "Locally, restage the current bundle or rebuild the current source, then report how to restart.",
                "With VADE_CONFIG, update source, rebuild and stage binaries, then offer a restart.",
                "For a newer Vade on this Spark, run the install command again: curl -fsSL https://vade.lol/install.sh | sh",
            ],
        },
        "detect" => Page {
            summary: "Show this Spark's machine, memory, disk, driver and model.",
            usage: &["vade detect [--json] [--dry-run]"],
            options: &[("--json", "Print one vade.detect.v1 object"), ("--dry-run", "Print the plan; read nothing")],
            examples: &["vade detect", "vade detect --json"],
            notes: &["Reads only; no files are changed and nothing is started."],
        },
        "version" => Page {
            summary: "Print the CLI version and the commit of this build.",
            usage: &["vade --version [--json]", "vade version [--json]", "vade -V [--json]"],
            options: &[("--json", "Print one vade.version.v1 object")],
            examples: &["vade --version", "vade version --json"],
            notes: &["A build with no commit says unknown commit. A build from a changed tree says modified.", "Loads no configuration and opens no connections."],
        },
        "run" => Page {
            summary: "Run the separate legacy Qwen3-8B BF16 workflow on one Spark.",
            usage: &["vade run qwen3-8b [OPTIONS]"],
            options: &[
                ("--port N", "Local server port, 1024..65535 (default 8080)"),
                ("--weights-source DIR", "Verify an existing official BF16 directory read-only, without weight downloads"),
                ("--release DIR", "Run a release folder"),
                ("--engine-source DIR", "Engine source checkout (with --harness-source, instead of --release)"),
                ("--harness-source DIR", "Harness source checkout"),
                API_KEY, DRY_RUN,
            ],
            examples: &["vade run qwen3-8b", "vade run qwen3-8b --dry-run"],
            notes: &[
                "Runs on one Linux aarch64 NVIDIA GB10. It checks prerequisites before any download, downloads the official pinned weights (resumable, hash-checked), starts a server, answers one prompt and runs doctor.",
                "Other --weights-source directory names are imported as Qwen3-8B inside VADE_HOME.",
                "VADE_HOME selects a private data folder for this workflow. Unset VADE_CONFIG. Use a fresh VADE_HOME if vade launch already saved defaults. Keep the same VADE_HOME for chat, status, logs and down. No system tools are installed.",
            ],
        },
        "config" => Page {
            summary: "Show which setup is selected and what it holds.",
            usage: &["vade config [--json]"],
            options: &[("--json", "Print one vade.config.v1 object")],
            examples: &["vade config", "vade config --json", "VADE_CONFIG=cluster.conf vade config"],
            notes: &[
                "The setup is local (saved defaults in VADE_HOME), controller (VADE_CONFIG or the user config file), legacy (vade run qwen3-8b) or none yet.",
                "Reads only. To change local limits, run vade down, then vade launch with the new flags. To change controller settings, stop the engine with its current config, then edit the file shown here.",
            ],
        },
        "pull" => Page {
            summary: "Download a model from the Hugging Face Hub.",
            usage: &["vade pull <org>/<name>[@revision] [OPTIONS]"],
            options: &[
                ("--include PATTERN", "Download only matching paths; repeat for more (* matches any text, / too)"),
                ("--to DIR", "Model folder (default ~/models/<org>/<name>)"),
                ("--dry-run", "Ask the Hub for the file list and sizes; download nothing"),
            ],
            examples: &[
                PINNED_PULL,
                "vade pull org/name --dry-run",
                "vade pull org/name --include '*.json' --to ~/models/org/name-metadata",
            ],
            notes: &[
                "Without --include, pull takes the weights (*.safetensors and the index), every JSON config, the tokenizer files, the chat template, the README and the license. The revision is a branch, tag or commit; the default is main. docs/pull-models.md gives the pinned command for each checkpoint Vade serves.",
                "Every file is checked after download: size, then the SHA-256 the Hub lists for large files or the Git blob ID for small ones. Run the same command again to continue an interrupted pull; partial files resume.",
                "Writes .vade-model.json in the folder (repository, commit, files, sizes, hashes, time, license and support state) and <folder>.sha256 beside it, which vade launch and vade up check before they serve FP4-Local. Ctrl-C stops the pull and writes no records.",
                "For a gated or private model, accept its terms on the model page, then set HF_TOKEN or save the token in ~/.vade/hf-token. The token is never printed or passed on the command line. HF_ENDPOINT names a mirror. Transfers use curl from PATH.",
            ],
        },
        "models" => Page {
            summary: "List the models on this machine and what can serve them.",
            usage: &["vade models [DIR] [--json]"],
            options: &[("--json", "Print one vade.models.v1 object")],
            examples: &["vade models", "vade models /data/models --json"],
            notes: &[
                "Looks three folders deep under DIR (default ~/models). Each row shows the size on disk, the repository and commit from .vade-model.json and the support state.",
                "The support state comes from the same metadata checks vade launch runs; no weights are read. A folder with partial files from vade pull shows as incomplete.",
            ],
        },
        "bench" => Page {
            summary: "Measure a running server from this client.",
            usage: &["vade bench [OPTIONS]", "vade bench --card --prompts FILE [--draft plain] [--receipt FILE]", "vade bench --card --compare CARD OTHER"],
            options: &[
                ("--url URL", "Server to measure (default: VADE_API_URL, then the selected installation's endpoint)"),
                ("--concurrency LIST", "Concurrent chats per cohort, such as 1,2,4,8 (default 1,2,4,8,16)"),
                ("--max-tokens N", "Tokens each chat must generate (default 256)"),
                ("--repeats N", "Repeats per cohort, none discarded (default 1)"),
                ("--draft plain|strict", "Override drafting on every request (default: server); with --card: plain or served"),
                ("--output FILE", "JSON receipt (default ./vade-bench.json) and a sibling .txt table; --out is an alias"),
                ("--api-key-file FILE", "Optional bearer credential; never stored in receipts"),
                ("--self-test", "Run against a tiny local mock server; no GPU"),
                ("--help-all", "List every option, including A/B runs and observation"),
                ("--card", "Send the 66 requests of the published card; save ./vade-card.json"),
                ("--prompts FILE", "With --card: RigMark prompts.json at revision 52c29c3"),
                ("--receipt FILE", "With --card: compare the token IDs with a receipt or card"),
                ("--compare CARD OTHER", "With --card: compare two saved files; send nothing"),
            ],
            examples: &["vade bench", "vade bench --concurrency 1,8 --max-tokens 128", "vade bench --self-test",
                "vade bench --card --prompts rigmark/prompts.json --receipt RECEIPT.json"],
            notes: &[
                "Wall time is measured on this client; token counts come from the server. Without --card, requests are greedy with cache and thinking off and a fixed length.",
                "Prints one row per cohort: tokens per second, time to first token and the bytes the server reported.",
                "vade bench --suite URL runs the development workloads instead; see engine/tools/bench/README.md.",
                "--card sends the card's requests in its order with thinking off, saves every token ID, and prints ids N/66 identical against --receipt. A receipt holds only token_ids_sha256, so it gives 66/66 or a difference; a full card gives the count. Without --draft plain, chats send no draft field, so the server's setting decides; the published drafting-ON cards were sent this way. See docs/results.md.",
            ],
        },
        "test" => Page {
            summary: "Run the build, determinism and long-context gate checks.",
            usage: &[
                "vade test build [OPTIONS]",
                "vade test determinism [OPTIONS] -- MODEL FIXTURES ADDR0 ADDR1 RANK [--world 1|2|4]",
                "vade test longctx prepare|run|compare [OPTIONS]",
                "vade test gdn-state REFERENCE CANDIDATE [SPLIT] [--identity] [--legacy-prefill-group N]",
                "vade test unix-flags-abi [--target TARGET]",
            ],
            options: &[],
            examples: &["vade test build --help", "vade test determinism --help", "vade test longctx --help"],
            notes: &[
                "build checks the current engine checkout and a private release snapshot; determinism drives the flash-bench byte comparisons and receipts.",
                "longctx freezes long-context prompts and compares teacher-forced logits and retrieval; gdn-state compares recurrent states; unix-flags-abi compiles Unix flag ABI assertions.",
                "Native Spark qualification remains separate from host checks. See cli/docs/test.md and cli/evals.md for every option, default, bound and receipt.",
            ],
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_page_fits_the_width_and_names_its_usage() {
        for command in [None, Some("launch"), Some("up"), Some("down"), Some("status"), Some("logs"), Some("chat"), Some("sessions"), Some("doctor"),
            Some("install"), Some("setup"), Some("update"), Some("detect"), Some("version"), Some("run"), Some("verify"), Some("read"), Some("config"), Some("bench"), Some("test"), Some("pull"), Some("models")] {
            let text = help(command);
            assert!(text.lines().all(|line| line.chars().count() <= style::WIDTH || line.trim_start() == PINNED_PULL), "{command:?}:\n{text}");
            assert!(text.contains(&format!("Usage: vade {}", command.unwrap_or("<COMMAND>").replace("version", "--version"))), "{command:?}:\n{text}");
            assert!(!text.contains('\x1b') && text.ends_with('\n') && !text.ends_with("\n\n"), "{command:?}");
            assert!(text.lines().nth(1) == Some("") && text.contains("\nExamples:\n"), "one-line summary, then examples: {command:?}:\n{text}");
        }
    }
    #[test]
    fn the_top_page_names_both_checkpoints_and_the_readme_commands() {
        let text = help(None);
        assert_eq!(text.lines().next(), Some("Vade runs Qwen3.8 Flash-Next on your DGX Spark, as FP4-Local or FP8."));
        let examples = text.split("\nUsage:").next().unwrap();
        let mut at = 0;
        for command in [PINNED_PULL, "vade up ", "vade chat ", "vade status ", "vade down "] {
            at += examples[at..].find(&format!("\n  {command}")).unwrap_or_else(|| panic!("{command}:\n{examples}"));
        }
        assert!(help(Some("pull")).contains(&format!("\n  {PINNED_PULL}\n")));
    }
    #[test]
    fn help_and_errors_give_the_same_spark_counts() {
        use crate::config::{FP4_SPARKS, FP8_SPARKS, MAX_FP4_SPARKS};
        // The wrapped help page, with its line breaks taken out.
        let up = help(Some("up")).split_whitespace().collect::<Vec<_>>().join(" ");
        for text in [FP4_SPARKS, FP8_SPARKS] { assert!(up.contains(text), "{text}:\n{up}"); }
        assert!(FP4_SPARKS.contains(&format!("3 to {MAX_FP4_SPARKS} Sparks")));
        assert!(up.contains(&format!("How many Sparks of your setup to start, from 1 to {MAX_FP4_SPARKS}")));
        let parse = |args: &[&str]| crate::cli::parse(&args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>());
        let beyond = (MAX_FP4_SPARKS + 1).to_string();
        let error = parse(&["up", "--sparks", &beyond]).unwrap_err();
        assert!(error.contains(&format!("from 1 to {MAX_FP4_SPARKS},")), "{error}");
        assert!(parse(&["up", "--sparks", &MAX_FP4_SPARKS.to_string()]).is_ok());
        for page in [Some("up"), Some("setup")] {
            assert!(!help(page).contains("1..24") && !help(page).contains("1 to 64"), "{page:?}");
        }
        // Six Sparks stay accepted for FP8, and every text marks them as not covered.
        let setup = help(Some("setup")).split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(setup.contains("(6 Sparks start but are not covered)") && setup.contains("6 Sparks are not covered yet"), "{setup}");
        assert!(up.contains("6 Sparks start but are not covered"), "{up}");
    }
}
