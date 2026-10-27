//! The text of `vade --help` and `vade <command> --help`, in one layout:
//! summary, usage, options, examples and notes (see [`crate::style::Page`]).
use crate::style::{self, Page, Stream, Token};

const LIFECYCLE: &str = "Explicit VADE_CONFIG selects the controller; otherwise generated local setup wins. Keep the same VADE_CONFIG and VADE_HOME from start through chat, status, logs and down.";
const TRANSPORT: &str = "One Spark needs no fabric or RDMA; linked Sparks use RDMA by default. Set SEQPAR_RDMA=0 for TCP using the configured fabric addresses on any reachable interface; QSFP and RDMA memlock are unnecessary in TCP mode.";
const DRY_RUN: (&str, &str) = ("--dry-run", "Show what would happen; change nothing");
const API_KEY: (&str, &str) = ("--api-key-file PATH", "Bearer credential file (or VADE_API_KEY_FILE)");
const CONTEXT: (&str, &str) = ("--context N", "Context window in tokens, 4096..262144 (default 32768)");
const OUTPUT: (&str, &str) = ("--output N", "Output limit in tokens, 1024..16384 (default min(8192, context))");
const MTP: (&str, &str) = ("--mtp N", "Drafting depth 0..3; 0 disables server drafting (default: engine-selected)");

/// The help page for `command`, or the top-level page; headings are styled
/// when standard output takes color.
pub fn help(command: Option<&str>) -> String {
    let on = style::color(Stream::Out);
    match command.and_then(page) {
        Some(page) => page.render(on),
        None if command == Some("verify") => crate::verify::help(),
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
    let mut out = format!("{} runs Qwen3.8 Flash-Next FP4-Local on your DGX Spark.\n\n{}\n", style::style(Token::Title, "Vade", on), heading("Examples:"));
    out += &style::table(&[
        ("vade launch", "Start on this Spark, then chat".into()),
        ("vade status", "See what is running".into()),
        ("vade doctor", "Check the setup, one line per check".into()),
        ("vade down", "Stop what vade started".into()),
    ], 2);
    out += &format!("\n{} vade <COMMAND> [OPTIONS]\n\n", heading("Usage:"));
    out += &group("Start and stop:", &[
        ("launch", "Install if missing, start on this Spark, then chat"),
        ("up", "Start without chat (this Spark, or Sparks from a controller)"),
        ("down", "Stop what vade started"),
    ], 7);
    out += &group("Use:", &[("chat", "Chat with the running engine")], 7);
    out += &group("Look:", &[
        ("status", "Show what is running and whether it is healthy"),
        ("config", "Show which setup is selected and what it holds"),
        ("doctor", "Check the setup: one line per check, then PASS or FAIL"),
        ("detect", "Show this Spark's machine, memory, disk, driver and model"),
        ("logs", "Show recent engine logs"),
    ], 7);
    out += &group("Prove:", &[
        ("verify", "Check exact token IDs across runs, drafting ON and OFF"),
        ("bench", "Measure a running server from this client"),
        ("test", "Run the build, determinism and long-context gate checks"),
    ], 7);
    out += &group("Maintain:", &[
        ("install", "Save local defaults and install the engine and chat"),
        ("update", "Refresh the installed engine and chat"),
        ("run", "Run the separate legacy Qwen3-8B BF16 workflow"),
        ("version", "Print the CLI version"),
    ], 7);
    out += &group("Options:", &[
        ("--json", "JSON output for status, doctor, config, detect, version"),
        ("--dry-run", "Preview a lifecycle command; nothing changes"),
        ("--api-key-file PATH", "Credential for chat, doctor and local run"),
        ("-h, --help", "Show help"),
        ("-V, --version", "Print the CLI version"),
    ], 0);
    out += &group("Environment:", &[
        ("VADE_HOME", "Local install root and saved defaults (default ~/.vade)"),
        ("VADE_CONFIG", "Controller config file; selects the controller (see Notes)"),
        ("VADE_API_KEY_FILE", "Credential file when --api-key-file is not given"),
        ("NO_COLOR", "Plain output without color"),
    ], 0);
    out += &format!("{}\n", heading("Notes:"));
    for note in [
        "Explicit VADE_CONFIG selects the controller for up, install, chat, status, logs, down and doctor, even after local setup; keep it set to the same file throughout that lifecycle. Without VADE_CONFIG, saved local setup wins over the user config directory.",
        "Global flags such as --dry-run before bench are unsupported.",
    ] { out += &format!("  - {}\n", style::wrap(note, style::WIDTH, "    ")); }
    out += &format!("\nLearn more: {}\n", style::style(Token::Command, "vade <command> --help", on));
    out
}

fn page(command: &str) -> Option<Page<'static>> {
    Some(match command {
        "launch" => Page {
            summary: "Start Vade on this Spark and open chat.",
            usage: &["vade launch [OPTIONS]"],
            options: &[CONTEXT, OUTPUT, MTP, ("--memory-bytes N", "Memory budget for this Spark, in bytes"), ("--cache-bytes 0", "Disable retained prompt cache, even with an inherited opt-in"), DRY_RUN],
            examples: &["vade launch", "vade launch --context 262144 --output 16384", "vade launch --dry-run"],
            notes: &[
                "Finds the FP4-Local checkpoint, saves defaults, installs if missing and starts the engine. It opens chat when standard input is a terminal. Run it again to attach.",
                "Local defaults match vade up from a controller: context 32768, output min(8192, context), engine-selected MTP.",
                "If the engine refuses the default or saved context for memory, launch tries smaller sizes from 32768, 16384, 8192 and 4096. It saves the first size that starts. A refused explicit --context is an error.",
                "The checkpoint comes from the saved config, then explicit VADE_MODELS, then the FP4-Local checkpoint under ~/models; launch asks if it is missing. Files install under ~/.vade (or VADE_HOME).",
                "Local --chats and positive --cache-bytes are unsupported. See vade up --help for memory options. Unset VADE_CONFIG for local lifecycle commands.",
            ],
        },
        "up" => Page {
            summary: "Start the engine without opening chat.",
            usage: &["vade up [OPTIONS]"],
            options: &[
                ("--fp4-local", "Qwen3.8 Flash-Next FP4-Local (the default; optional)"),
                ("--fp8-legacy", "Legacy FP8 on 2, 4 or 6 Sparks: context 131072, output min(65536, context), MTP 2; output ceiling 262144"),
                ("--sparks N", "Number of Sparks; FP4 supports 1..24"),
                ("--hosts a,b", "Machine names in cable order, separated by commas"),
                ("--replicas N", "2..16 independent one-Spark FP4-Local servers behind a local gateway"),
                MTP, CONTEXT, OUTPUT,
                ("--chats N", "Concurrent legacy FP8 chats, 1..32 (controller)"),
                ("--memory-bytes N", "Memory budget per rank, in bytes"),
                ("--cache-bytes N", "Legacy FP8 retained prompt cache, in bytes; 0 disables it. Local FP4 accepts only 0"),
                DRY_RUN,
            ],
            examples: &["vade up", "vade up --context 262144 --output 16384", "VADE_CONFIG=cluster.conf vade up --sparks 2"],
            notes: &[
                "Default: Qwen3.8 Flash-Next FP4-Local on one Spark.",
                "Local up and launch start with context 32768 and output min(8192, context). They reuse saved limits. A memory refusal lowers a default or saved context to a smaller size from 32768, 16384, 8192 and 4096. The first size that starts is saved. A refused explicit --context is an error. Local --chats and positive --cache-bytes are refused; --cache-bytes 0 disables retention.",
                "Controller FP4 defaults match local ones on any Spark count: context 32768, output min(8192, context). --context 262144 requests a larger context. The engine still sets the chat count from context and memory.",
                "Controller FP4 ignores --chats and --cache-bytes. Its engine admits chats from memory and uses its own prefix cache setting.",
                "MTP is engine-selected. FP4-Local one-Spark concurrent chats can draft; the default request depth is 3 with strict verification. Legacy FP8 concurrent chats decode plainly. Per-request draft=plain disables drafting; draft=strict requests it when supported. --mtp overrides MTP, and 0 disables server drafting.",
                "Spark counts above 2 require a current install and linked hosts in cable order.",
                "--replicas N starts one independent one-Spark server on each of the first N machines (or the N --hosts) and a local gateway on local_port; replica tunnels use the next N local ports. Each replica uses the FP4 defaults above. Use it when the model fits one Spark and many chats run at once; answers are the same on every replica.",
                "One Spark needs no fabric. Controller pairs and chains use RDMA; SEQPAR_RDMA=0 selects TCP. Keep VADE_CONFIG and VADE_HOME consistent through chat, status and down.",
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
            usage: &["vade chat [--api-key-file PATH] [--dry-run]"],
            options: &[API_KEY, DRY_RUN],
            examples: &["vade chat", "VADE_CONFIG=cluster.conf vade chat"],
            notes: &[
                "Opens the Vade agent on the resident engine. Start the engine once with vade launch or vade up.",
                "Local contexts below 8192 tokens use direct text chat. Local Linux agent mode disables automatic workspace checks with a notice.",
                "Typing during a reply interrupts it and keeps its partial answer in context. /queue on queues follow-ups; /queue off restores steering. /exit leaves the engine running.",
                "Chat history is kept in ~/.vade/projects (VADE_HOME changes the root), never in this folder; VADE_CHAT_IN_PROJECT=1 keeps it in a git-ignored ./.vade.",
                LIFECYCLE,
            ],
        },
        "doctor" => Page {
            summary: "Check the setup: one line per check, then PASS or FAIL.",
            usage: &["vade doctor [--json] [--api-key-file PATH] [--dry-run]"],
            options: &[("--json", "Print one vade.doctor.v1 object with every check"), API_KEY, DRY_RUN],
            examples: &["vade doctor", "vade doctor --json", "VADE_CONFIG=cluster.conf vade doctor --dry-run"],
            notes: &[
                "On this Spark: config, VADE_HOME writable, ownership, installed binaries, model files, nvidia-smi and driver, disk, free memory, port and engine health.",
                "From a controller: local tools, then SSH, nvidia-smi, disk, free memory and model files on each configured Spark, then the fabric and the harness.",
                "Each FAIL row gives the next step. Exit 0 means PASS, 1 means FAIL.",
                TRANSPORT, LIFECYCLE,
            ],
        },
        "install" => Page {
            summary: "Save local defaults and install the engine and chat.",
            usage: &["vade install [--dry-run]"],
            options: &[DRY_RUN],
            examples: &["vade install", "VADE_CONFIG=cluster.conf vade install --dry-run"],
            notes: &[
                "Model downloads stay in your hands.",
                "For a fresh local installation or saved owner defaults without VADE_CONFIG: detect existing weights, save defaults once in ~/.vade/config, and stage a local engine and chat program. No hand editing is needed.",
                "With VADE_CONFIG, or a default controller config before local setup: check prerequisites, copy an exact-source SHA256-verified prebuilt_dir bundle or build source, then check the harness. Model inventory checks overlap copy/build and are cached.",
                "A generated run qwen3-8b installation uses run again; install requires an explicit VADE_CONFIG to select a controller.",
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
            ],
        },
        "detect" => Page {
            summary: "Show this Spark's machine, memory, disk, driver and model.",
            usage: &["vade detect [--json]"],
            options: &[("--json", "Print one vade.detect.v1 object")],
            examples: &["vade detect", "vade detect --json"],
            notes: &["Reads only; no files are changed and nothing is started."],
        },
        "version" => Page {
            summary: "Print the CLI version.",
            usage: &["vade --version [--json]", "vade version [--json]", "vade -V"],
            options: &[("--json", "Print one vade.version.v1 object")],
            examples: &["vade --version", "vade version --json"],
            notes: &["Loads no configuration and opens no connections."],
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
                "On one Linux/aarch64 NVIDIA GB10: detect prerequisites before downloading, resume and hash official pinned weights, start an owned server, answer, run doctor.",
                "Other --weights-source directory names are imported as Qwen3-8B inside VADE_HOME.",
                "VADE_HOME selects a dedicated private data directory. Unset VADE_CONFIG; use a fresh VADE_HOME if local launch already has defaults. Keep the same selection for chat, status, logs and down. No system tool installation.",
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
        "bench" => Page {
            summary: "Measure a running server from this client.",
            usage: &["vade bench [OPTIONS]"],
            options: &[
                ("--url URL", "Server to measure (default: VADE_API_URL, then the selected installation's endpoint)"),
                ("--concurrency LIST", "Concurrent chats per cohort, such as 1,2,4,8 (default 1,2,4,8,16)"),
                ("--max-tokens N", "Tokens each chat must generate (default 256)"),
                ("--repeats N", "Repeats per cohort, none discarded (default 1)"),
                ("--draft plain|strict", "Override drafting on every request (default: server)"),
                ("--output FILE", "JSON receipt (default ./vade-bench.json) and a sibling .txt table"),
                ("--api-key-file FILE", "Optional bearer credential; never stored in receipts"),
                ("--self-test", "Run against a tiny local mock server; no GPU"),
                ("--help-all", "List every option, including A/B runs and observation"),
            ],
            examples: &["vade bench", "vade bench --concurrency 1,8 --max-tokens 128", "vade bench --self-test"],
            notes: &[
                "Wall time is measured on this client; token counts come from the server. Requests are greedy with cache and thinking off and a fixed length.",
                "Prints one row per cohort: tokens per second, time to first token and the bytes the server reported.",
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
        for command in [None, Some("launch"), Some("up"), Some("down"), Some("status"), Some("logs"), Some("chat"), Some("doctor"),
            Some("install"), Some("update"), Some("detect"), Some("version"), Some("run"), Some("verify"), Some("config"), Some("bench"), Some("test")] {
            let text = help(command);
            assert!(text.lines().all(|line| line.chars().count() <= style::WIDTH), "{command:?}:\n{text}");
            assert!(text.contains(&format!("Usage: vade {}", command.unwrap_or("<COMMAND>").replace("version", "--version"))), "{command:?}:\n{text}");
            assert!(!text.contains('\x1b') && text.ends_with('\n') && !text.ends_with("\n\n"), "{command:?}");
            assert!(text.lines().nth(1) == Some("") && text.contains("\nExamples:\n"), "one-line summary, then examples: {command:?}:\n{text}");
        }
    }
}
