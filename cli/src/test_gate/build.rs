//! The four build-gate steps, kept in the same order as the transition script.

use super::{TempDir, exec_logged};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const HELP: &str = "Usage: vade test build [--denylist FILE] [--features FEATURES] [--no-default-features] [--dry-run]\n\n\
FILE defaults to PRIVATE_DENYLIST. It must be a nonempty private denylist\n\
outside the source tree, as documented in docs/export.md. Never printed.\n\
Run from a committed, clean checkout for the release snapshot export.\n\
--features forwards the same Cargo features to source/export builds and check.\n\
The flash-routing-per-row feature is on by default; it selects only the gate's\n\
per-row capability (the poison cell and the 8-slot matrix). --no-default-features\n\
turns off that gate capability only: per-row rejection in the engine and server\n\
and the +4-byte gather word stay on. The aggregate comparison arm for timing is the\n\
main before isolation (7a26f6ae), never a --no-default-features build of this branch.\n\n\
Darwin: host check of the kernel builder (no NVPTX), workspace release build,\n\
exported workspace release build, cargo check. This is only a host gate.\n\
Linux aarch64: cargo kernels, workspace release build, exported kernels and\n\
workspace release build, cargo check. Requires the installed pinned NVPTX\n\
toolchain (or KERNEL_BIN); this command never installs it.\n\n\
All builds are offline. Export uses the existing Rust vade-export tool;\n\
no shell exporter exists. Snapshot, logs, and export build output are temporary\n\
and removed on exit. Source build output stays in the usual Cargo target dir.\n\
One PASS/FAIL line per attempted step; stop at the first failure.\n\
--dry-run prints commands only: no Cargo, export, or temporary files, no PASS.\n\
--help prints this help without running any step.\n";

#[derive(Clone, Copy)]
enum Host {
    Mac,
    Spark,
}

pub fn run(args: &[String]) -> i32 {
    let mut denylist = std::env::var_os("PRIVATE_DENYLIST").unwrap_or_default();
    let mut dry = false;
    let mut build_args = vec!["--offline".to_owned()];
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--help" | "-h" => {
                crate::public_output(HELP);
                return 0;
            }
            "--dry-run" => dry = true,
            "--no-default-features" => build_args.push(args[index].clone()),
            "--features" | "--denylist" => {
                let option = args[index].as_str();
                let Some(value) = args.get(index + 1).filter(|value| !value.is_empty()) else {
                    crate::output::stderr_line(format_args!(
                        "FAIL inputs: {option} needs {}",
                        if option == "--features" {
                            "FEATURES"
                        } else {
                            "FILE"
                        }
                    ));
                    return 2;
                };
                if option == "--features" {
                    build_args.extend([option.to_owned(), value.clone()]);
                } else {
                    denylist = OsString::from(value);
                }
                index += 1;
            }
            _ => {
                crate::output::stderr_line(format_args!("FAIL inputs: unknown argument (see --help)"));
                return 2;
            }
        }
        index += 1;
    }
    let host = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", _) => Host::Mac,
        ("linux", "aarch64") => Host::Spark,
        _ => {
            crate::output::stderr_line(format_args!("FAIL platform: expected Darwin or Linux aarch64"));
            return 2;
        }
    };
    if dry {
        dry_run(host, &build_args);
        return 0;
    }
    // Help and dry-run need no checkout, private config or denylist. A selected
    // root must never fall back to another checkout in the current directory.
    let engine = match super::engine_root() {
        Ok(engine) => engine.unwrap_or_else(|| PathBuf::from("engine")),
        Err(error) => {
            crate::output::stderr_line(format_args!("{error}"));
            return 2;
        }
    };
    let engine = engine.as_path();
    if !engine.join("tools/export/Cargo.toml").is_file() {
        crate::output::stderr_line(format_args!("FAIL inputs: run in the engine checkout or set VADE_ROOT to its repository"));
        return 2;
    }
    if !cargo_available(engine) {
        crate::output::stderr_line(format_args!("FAIL inputs: cargo is required"));
        return 2;
    }
    let denylist = match private_denylist(&denylist, engine) {
        Some(path) => path,
        None => {
            crate::output::stderr_line(format_args!(
                "FAIL inputs: set PRIVATE_DENYLIST or --denylist to an external nonempty file"
            ));
            return 2;
        }
    };
    let temporary = match TempDir::new("vade-build-gate") {
        Ok(temporary) => temporary,
        Err(error) => {
            crate::output::stderr_line(format_args!("FAIL temporary: {error}"));
            return 1;
        }
    };
    let snapshot = temporary.path.join("snapshot");
    let export_target = temporary.path.join("target");
    let log = temporary.path.join("step.log");
    let (kernel_label, kernel_args): (&str, &[&str]) = match host {
        Host::Mac => (
            "kernels-host (Mac; NVPTX unbuilt)",
            &["check", "--offline", "-p", "spark-build-kernels"],
        ),
        Host::Spark => ("kernels", &["kernels"]),
    };
    let mut kernels = cargo(engine);
    kernels.args(kernel_args);
    let status = step(kernel_label, &mut [kernels], &log);
    if status != 0 {
        return status;
    }
    let mut workspace = workspace_build(engine, &build_args);
    let status = step("workspace", std::slice::from_mut(&mut workspace), &log);
    if status != 0 {
        return status;
    }
    let mut export = cargo(engine);
    export.args([
        "run",
        "-p",
        "vade-export",
        "--offline",
        "--",
        "--source",
        ".",
        "--destination",
    ]);
    export.arg(&snapshot).arg("--denylist").arg(denylist);
    let mut export_commands = vec![export];
    if matches!(host, Host::Spark) {
        let mut kernels = cargo(&snapshot);
        kernels
            .arg("kernels")
            .env("CARGO_TARGET_DIR", &export_target);
        export_commands.push(kernels);
    }
    let mut exported_workspace = workspace_build(&snapshot, &build_args);
    exported_workspace.env("CARGO_TARGET_DIR", &export_target);
    export_commands.push(exported_workspace);
    let status = step("export", &mut export_commands, &log);
    if status != 0 {
        return status;
    }
    let mut check = cargo(engine);
    check.arg("check").args(&build_args);
    step("check", &mut [check], &log)
}

fn cargo(directory: &Path) -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(directory)
        .env("CARGO_NET_OFFLINE", "true");
    command
}

fn workspace_build(directory: &Path, args: &[String]) -> Command {
    let mut command = cargo(directory);
    command.args([
        "build",
        "--release",
        "--workspace",
        "--exclude",
        "spark-rust-kernels",
    ]);
    command.args(args);
    command
}

fn step(label: &str, commands: &mut [Command], log: &Path) -> i32 {
    if File::create(log).is_err() {
        crate::output::report(format_args!("FAIL {label} (exit 1)"), true);
        crate::output::stderr_line(format_args!("unable to create temporary build log"));
        return 1;
    }
    for command in commands {
        let status = match exec_logged(command, log, None) {
            Ok(status) => status,
            Err(error) => {
                crate::output::stderr_line(format_args!("{error}"));
                1
            }
        };
        if status != 0 {
            crate::output::report(format_args!("FAIL {label} (exit {status})"), true);
            if let Ok(bytes) = fs::read(log) {
                let _ = io::stderr().lock().write_all(&bytes);
            }
            return status;
        }
    }
    crate::output::report(format_args!("PASS {label}"), true);
    0
}

fn private_denylist(value: &OsString, engine: &Path) -> Option<PathBuf> {
    if value.is_empty() {
        return None;
    }
    let path = engine.join(value);
    let metadata = fs::metadata(&path).ok()?;
    if !metadata.is_file() || metadata.len() == 0 || File::open(&path).is_err() {
        return None;
    }
    // Like `cd dirname && pwd -P`, preserve the final component rather than
    // resolving a file symlink; the existing exporter checks its own inputs.
    Some(
        fs::canonicalize(path.parent()?)
            .ok()?
            .join(path.file_name()?),
    )
}

fn cargo_available(engine: &Path) -> bool {
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into())).any(
        |directory| {
            let path = engine.join(directory).join("cargo");
            let Ok(metadata) = fs::metadata(&path) else {
                return false;
            };
            if !metadata.is_file() {
                return false;
            }
            super::executable(&path)
        },
    )
}

fn dry_run(host: Host, args: &[String]) {
    match host {
        Host::Mac => crate::output::report(format_args!("DRY-RUN kernels-host: cargo check --offline -p spark-build-kernels"), true),
        Host::Spark => crate::output::report(format_args!("DRY-RUN kernels: cargo kernels"), true),
    }
    let args = args
        .iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ");
    crate::output::report(format_args!(
        "DRY-RUN workspace: cargo build --release --workspace --exclude spark-rust-kernels {args}"
    ), true);
    crate::output::report(format_args!(
        "DRY-RUN export: cargo run -p vade-export --offline -- --source . --destination \"$SNAPSHOT\" --denylist \"$PRIVATE_DENYLIST\""
    ), true);
    if matches!(host, Host::Spark) {
        crate::output::report(format_args!("DRY-RUN export: (cd \"$SNAPSHOT\" && cargo kernels)"), true);
    }
    crate::output::report(format_args!(
        "DRY-RUN export: (cd \"$SNAPSHOT\" && CARGO_TARGET_DIR=\"$EXPORT_TARGET\" cargo build --release --workspace --exclude spark-rust-kernels {args})"
    ), true);
    crate::output::report(format_args!("DRY-RUN check: cargo check {args}"), true);
}

// Match Bash's `%q` display while executing every argument directly, never
// through a shell. The displayed placeholders contain no private input path.
fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_owned();
    }
    let mut out = String::new();
    if value.chars().any(char::is_control) {
        out.push_str("$'");
        for character in value.chars() {
            match character {
                '\x07' => out.push_str("\\a"),
                '\x08' => out.push_str("\\b"),
                '\t' => out.push_str("\\t"),
                '\n' => out.push_str("\\n"),
                '\x0b' => out.push_str("\\v"),
                '\x0c' => out.push_str("\\f"),
                '\r' => out.push_str("\\r"),
                '\x1b' => out.push_str("\\E"),
                '\\' | '\'' => {
                    out.push('\\');
                    out.push(character);
                }
                character if character.is_control() => {
                    for byte in character.to_string().bytes() {
                        out.push_str(&format!("\\{byte:03o}"));
                    }
                }
                character => out.push(character),
            }
        }
        out.push('\'');
    } else {
        for (index, character) in value.chars().enumerate() {
            if " \"'\\`$&;|()<>*?[]{}!^,".contains(character) || (index == 0 && character == '#') {
                out.push('\\');
            }
            out.push(character);
        }
    }
    out
}
