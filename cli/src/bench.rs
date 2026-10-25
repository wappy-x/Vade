//! Benchmark receipts belong in the working directory, not the installation.

/// `vade bench`: run vade-bench, writing to `./vade-bench.json` unless
/// `--output` (or `--out`) is given.
pub fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32, String> {
    // Discovery and the development suite have their own argument handling.
    if arguments.first().is_some_and(|arg| arg == "--suite")
        || arguments.iter().any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        return vade_bench::run(arguments, interrupted);
    }
    let mut arguments: Vec<String> = arguments.iter()
        .map(|arg| if arg == "--out" { "--output".into() } else { arg.clone() }).collect();
    if arguments.windows(2).any(|pair| pair[0] == "--output" && pair[1].is_empty()) {
        return Err("missing value for --output".into());
    }
    if !arguments.iter().any(|arg| arg == "--output") {
        arguments.extend(["--output".into(), "./vade-bench.json".into()]);
    }
    vade_bench::run(&arguments, interrupted)
}
