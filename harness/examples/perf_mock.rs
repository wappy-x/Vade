//! Bounded macOS harness benchmark with first-party loopback fixtures only.
//! Run: cargo run --release --offline --example perf_mock -- target/release/vade-harness 3
//! Modes: all (default), pty, or workload NEW_JSON_FILE. No Engine or GPU.
#[cfg(target_os = "macos")]
#[path = "perf_mock/process.rs"]
mod process;
#[cfg(target_os = "macos")]
#[path = "perf_mock/stream.rs"]
mod stream;
#[cfg(target_os = "macos")]
#[path = "perf_mock/workload.rs"]
mod workload;

#[cfg(target_os = "macos")]
fn run() -> std::io::Result<()> {
    use std::{
        io,
        path::Path,
        time::{Duration, Instant},
    };
    let args: Vec<_> = std::env::args().collect();
    let binary = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("target/release/vade-harness"),
    )
    .canonicalize()?;
    let repeats = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(io::Error::other)?
        .unwrap_or(3);
    if !(1..=31).contains(&repeats) {
        return Err(io::Error::other("choose 1..31 repetitions"));
    }
    let mode = args.get(3).map(String::as_str).unwrap_or("all");
    let deadline = Instant::now() + Duration::from_secs(240);
    if mode == "workload" {
        return workload::run(
            &binary,
            args.get(4)
                .map(String::as_str)
                .unwrap_or("target/workload-mock.json"),
            deadline,
        );
    }
    if !matches!(mode, "all" | "pty") {
        return Err(io::Error::other(
            "choose all, pty, or workload NEW_JSON_FILE",
        ));
    }
    let workspace = process::Workspace::new()?;
    println!(
        "scope=host_only first_party_mock=true cpu_clock_ns_per_tick={:.9} deadline_seconds=240",
        process::tick_seconds(1)? * 1e9
    );
    for _ in 0..repeats {
        if mode == "all" {
            stream::startup(&binary, &workspace.0, deadline)?;
        }
        for raw in [false, true] {
            if mode == "pty" && !raw {
                continue;
            }
            for (mode, per_event) in [("direct", 1), ("agent", 8)] {
                stream::run(
                    &binary,
                    &workspace.0,
                    mode,
                    100_000,
                    per_event,
                    raw,
                    deadline,
                )?;
            }
        }
    }
    Ok(())
}
fn main() {
    #[cfg(target_os = "macos")]
    if let Err(error) = run() {
        eprintln!("mock benchmark failed: {error}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("this benchmark's process accounting and PTY ABI require macOS");
        std::process::exit(1);
    }
}
