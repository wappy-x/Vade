//! Compatibility launcher; all serving and request policy lives in flash-serve.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    if std::env::var("NEXT_TP_HTTP").as_deref() != Ok("1") {
        return Err(
            "next-http compatibility launcher requires NEXT_TP_HTTP=1; use flash-serve".into());
    }
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    spark_tp::next::RankArgs::parse(&a, true)?;
    if std::env::var("CONTEXT").is_ok_and(|v| v != "32768") {
        return Err("FP4-Local world 2/4 CONTEXT must be 32768".into());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?
        .with_file_name("flash-serve");
    let mut command = std::process::Command::new(executable);
    command
        .args(&a[..6])
        .arg("32768")
        .args(&a[7..])
        .arg("--tensor-map")
        .arg(&a[6])
        .args(["--mtp", "0"]);
    if let Ok(value) = std::env::var("OUTPUT") {
        command.env("FLASH_SERVE_OUTPUT_TOKENS", value);
    }
    Err(command.exec().to_string())
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run() {
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("Next HTTP requires Linux CUDA");
        std::process::exit(1);
    }
}
