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
/// when standard output takes colour.
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
    let mut out = format!("{} runs Qwen3.8 Flash-Next FP4-Local on your DGX Spark: start it, chat\nwith it, check it, stop it.\n\n{} vade <COMMAND> [OPTIONS]\n\n",
        style::style(Token::Title, "Vade", on), heading("Usage:"));
    out += &group("Get started:", &[
        ("launch", "Install if missing, start FP4-Local on this Spark, then chat"),
        ("chat", "Open chat here (direct text below 8192 local context tokens)"),
        ("status", "Show what is running (--json for scripts)"),
        ("down", "Stop what vade started"),
    ], 7);
    out += &group("Run:", &[
        ("up", "Start FP4-Local without chat (legacy FP8 needs --fp8-legacy)"),
        ("logs", "Show recent logs"),
        ("install", "Save local defaults and install; VADE_CONFIG uses the controller"),
        ("update", "Restage local binaries or update controller source"),
        ("run", "Run the separate official Qwen3-8B BF16 catalog"),
    ], 7);
    out += &group("Check:", &[
        ("doctor", "Check what needs attention"),
        ("detect", "Show this Spark and check available memory and disk"),
        ("verify", "Compare exact token IDs with drafting ON and OFF"),
        ("bench", "Measure a server from the client"),
        ("test", "Run build, determinism, long-context, GDN and ABI checks"),
        ("version", "Print the CLI version"),
    ], 7);
    out += &group("Options:", &[
        ("--dry-run", "Preview a lifecycle command; nothing changes"),
        ("--api-key-file PATH", "Credential for chat, doctor and local run"),
        ("-h, --help", "Show help"),
        ("-V, --version", "Print the CLI version"),
    ], 0);
    out += &group("Environment:", &[
        ("VADE_HOME", "Local install root and saved defaults (default ~/.vade)"),
        ("VADE_CONFIG", "Controller config file; selects the controller (see Notes)"),
        ("VADE_API_KEY_FILE", "Credential file when --api-key-file is not given"),
        ("NO_COLOR", "Plain output without colour"),
    ], 0);
    out += &format!("{}\n", heading("Notes:"));
    for note in [
        "Explicit VADE_CONFIG selects the controller for up, install, chat, status, logs, down and doctor, even after local setup; keep it set to the same file throughout that lifecycle. Without VADE_CONFIG, saved local setup wins over the user config directory.",
        "For bench, vade help bench, vade --help bench and vade -h bench also work; global flags such as --dry-run before bench are unsupported.",
    ] { out += &format!("  - {}\n", style::wrap(note, style::WIDTH, "    ")); }
    out += &format!("\nLearn more: {}\n", style::style(Token::Command, "vade <command> --help", on));
    out
}

fn page(command: &str) -> Option<Page<'static>> {
    Some(match command {
        "launch" => Page {
            summary: "Start Vade on this Spark and open chat. Finds the FP4-Local checkpoint, saves defaults, installs if missing, starts the engine, then opens chat. Run it again to attach.",
            usage: &["vade launch [OPTIONS]"],
            options: &[CONTEXT, OUTPUT, MTP, ("--cache-bytes 0", "Disable retained prompt cache, even with an inherited opt-in"), DRY_RUN],
            examples: &["vade launch", "vade launch --context 262144 --output 16384", "vade launch --dry-run"],
            notes: &[
                "Local defaults match vade up from a controller: context 32768, output min(8192, context), engine-selected MTP.",
                "If the engine refuses the default or saved context for memory, launch retries 16384, 8192 then 4096, saves the one that starts and says which; an explicit --context the engine refuses is an error.",
                "The checkpoint comes from the saved config, then explicit VADE_MODELS, then the FP4-Local checkpoint under ~/models; launch asks if it is missing. Files install under ~/.vade (or VADE_HOME).",
                "Local --chats and positive --cache-bytes are unsupported. See vade up --help for memory options. Unset VADE_CONFIG for local lifecycle commands.",
            ],
        },
        "up" => Page {
            summary: "Start the engine without opening chat. Default: Qwen3.8 Flash-Next FP4-Local on one Spark.",
            usage: &["vade up [OPTIONS]"],
            options: &[
                ("--fp4-local", "Qwen3.8 Flash-Next FP4-Local (the default; optional)"),
                ("--fp8-legacy", "Two-Spark legacy FP8: context 131072, output min(65536, context), MTP 2; output ceiling 262144"),
                ("--sparks N", "Number of Sparks; FP4 supports 1..24"),
                ("--hosts a,b", "Machine names in cable order, separated by commas"),
                ("--replicas N", "2..16 independent one-Spark FP4-Local servers behind a local gateway"),
                MTP, CONTEXT, OUTPUT,
                ("--chats N", "Concurrent chats, 1..32 (controller)"),
                ("--memory-bytes N", "Memory budget per rank, in bytes"),
                ("--cache-bytes N", "Retained prompt cache, in bytes; 0 disables retention"),
                DRY_RUN,
            ],
            examples: &["vade up", "vade up --context 262144 --output 16384", "VADE_CONFIG=cluster.conf vade up --sparks 2"],
            notes: &[
                "Local up and launch use context 32768 and output min(8192, context). The engine's memory admission decides: only a refused default or saved context is lowered (16384, 8192, 4096), and saved once it starts. Supported explicit limits are saved; an explicit --context the engine refuses is an error. Local --chats and positive --cache-bytes are refused; --cache-bytes 0 disables retention.",
                "Controller FP4 defaults match local ones on any Spark count: context 32768, output min(8192, context); the engine admits concurrent chats from context and memory. --context 262144 selects the single-chat long-context preset.",
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
            examples: &[],
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
            examples: &[],
            notes: &[LIFECYCLE],
        },
        "chat" => Page {
            summary: "Open the Vade agent on the resident engine. Start the engine once with vade launch or vade up.",
            usage: &["vade chat [--api-key-file PATH] [--dry-run]"],
            options: &[API_KEY, DRY_RUN],
            examples: &[],
            notes: &[
                "Local contexts below 8192 tokens use direct text chat. Local Linux agent mode disables automatic workspace checks with a notice.",
                "Typing during a reply interrupts it and keeps its partial answer in context. /queue on queues follow-ups; /queue off restores steering. /exit leaves the engine running.",
                "Chat history is kept in ~/.vade/projects (VADE_HOME changes the root), never in this folder; VADE_CHAT_IN_PROJECT=1 keeps it in a git-ignored ./.vade.",
                LIFECYCLE,
            ],
        },
        "doctor" => Page {
            summary: "Check what needs attention and explain next steps: the local driver, memory, disk, model, installed binaries and engine health, or the controller harness and configured machines.",
            usage: &["vade doctor [--api-key-file PATH] [--dry-run]"],
            options: &[API_KEY, DRY_RUN],
            examples: &[],
            notes: &[TRANSPORT, LIFECYCLE],
        },
        "install" => Page {
            summary: "Save local defaults and install the engine and chat. Model downloads stay in your hands.",
            usage: &["vade install [--dry-run]"],
            options: &[DRY_RUN],
            examples: &[],
            notes: &[
                "For a fresh local installation or saved owner defaults without VADE_CONFIG: detect existing weights, save defaults once in ~/.vade/config, and stage a local engine and chat program. No hand editing is needed.",
                "With VADE_CONFIG, or a default controller config before local setup: check prerequisites, copy an exact-source SHA256-verified prebuilt_dir bundle or build source, then run doctor. Model inventory checks overlap copy/build and are cached.",
                "A generated run qwen3-8b installation uses run again; install requires an explicit VADE_CONFIG to select a controller.",
                TRANSPORT, LIFECYCLE,
            ],
        },
        "update" => Page {
            summary: "Refresh the installed engine and chat.",
            usage: &["vade update [--dry-run]"],
            options: &[DRY_RUN],
            examples: &[],
            notes: &[
                "Locally, restage the current bundle or rebuild the current source, then report how to restart.",
                "With VADE_CONFIG, update source, rebuild and stage binaries, then offer a restart.",
            ],
        },
        "detect" => Page {
            summary: "Read this Spark's identity, available memory, disk, NVIDIA driver and existing model. No files are changed.",
            usage: &["vade detect"],
            options: &[],
            examples: &[],
            notes: &[],
        },
        "version" => Page {
            summary: "Print the CLI package version without loading configuration or opening connections.",
            usage: &["vade --version", "vade version", "vade -V"],
            options: &[],
            examples: &[],
            notes: &[],
        },
        "run" => Page {
            summary: "Run the separate official Qwen3-8B BF16 catalog locally on one Linux/aarch64 NVIDIA GB10: detect prerequisites before downloading, resume and hash official pinned weights, start an owned server, answer, run doctor.",
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
                "Other --weights-source directory names are imported as Qwen3-8B inside VADE_HOME.",
                "VADE_HOME selects a dedicated private data directory. Unset VADE_CONFIG; use a fresh VADE_HOME if local launch already has defaults. Keep the same selection for chat, status, logs and down. No system tool installation.",
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
            Some("install"), Some("update"), Some("detect"), Some("version"), Some("run"), Some("verify")] {
            let text = help(command);
            assert!(text.lines().all(|line| line.chars().count() <= style::WIDTH), "{command:?}:\n{text}");
            assert!(text.contains(&format!("Usage: vade {}", command.unwrap_or("<COMMAND>").replace("version", "--version"))), "{command:?}:\n{text}");
            assert!(!text.contains('\x1b') && text.ends_with('\n') && !text.ends_with("\n\n"), "{command:?}");
        }
    }
}
