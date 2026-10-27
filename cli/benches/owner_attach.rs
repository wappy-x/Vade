//! Optional Mac mock attach benchmark. Run the binary with --run on an idle host.
#![forbid(unsafe_code)]
use std::{io::Read,process::{Command,Stdio},time::{Duration,Instant}};
fn run() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    // cargo test --all-targets compiles and visits this target without measuring.
    if !args.iter().any(|a| a == "--run") { println!("Use cargo bench --bench owner_attach -- --run [--samples 20] [--target-seconds 1]."); return Ok(()); }
    if !cfg!(target_os = "macos") { return Err("this mock attach benchmark runs only on macOS".into()); }
    let (mut samples,mut target) = (20usize,1.0f64); let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--run"|"--bench" => (), "--samples" => { i += 1; samples = args.get(i).ok_or("missing sample count")?.parse()?; },
            "--target-seconds" => { i += 1; target = args.get(i).ok_or("missing target")?.parse()?; }, _ => return Err("unknown benchmark option".into())
        } i += 1;
    }
    if samples < 5 || !target.is_finite() || target <= 0. { return Err("use at least 5 samples and a positive finite target".into()); }
    let mut values = Vec::new();
    for sample in 0..=samples {
        let mut child = Command::new("cargo").args(["+1.99.0","test","--offline","--manifest-path"])
            .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .args(["--bin","vade","launch::tests::resident_attach_with_real_subprocesses_reuses_saved_state","--","--exact","--nocapture","--test-threads=1"])
            .env("RUSTUP_TOOLCHAIN","1.99.0").env("CARGO_NET_OFFLINE","true").stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        let mut stdout = child.stdout.take().unwrap(); let mut stderr = child.stderr.take().unwrap();
        let out = std::thread::spawn(move || { let mut s = String::new(); stdout.read_to_string(&mut s).map(|_|s) });
        let err = std::thread::spawn(move || { let mut s = String::new(); stderr.read_to_string(&mut s).map(|_|s) });
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? { break status; }
            if start.elapsed() > Duration::from_secs(180) { child.kill()?; child.wait()?; return Err("attach sample exceeded 180 seconds".into()); }
            std::thread::sleep(Duration::from_millis(10));
        };
        let stdout = out.join().map_err(|_|"stdout reader failed")??; let stderr = err.join().map_err(|_|"stderr reader failed")??;
        if !status.success() { return Err(format!("{stdout}{stderr}").into()); }
        let value: f64 = stdout.split_once("Mac resident attach with three real subprocesses: ").and_then(|(_,s)|s.split_once(" s")).ok_or("attach measurement missing")?.0.parse()?;
        if !value.is_finite() || value < 0. { return Err("invalid attach measurement".into()); } if sample > 0 { values.push(value); }
    }
    values.sort_by(f64::total_cmp); let n = values.len();
    let median = if n%2 == 0 {(values[n/2-1]+values[n/2])/2.} else {values[n/2]}; let p95 = values[(95*n).div_ceil(100)-1];
    println!("Mock attach: n={n}, median={median:.3}s, p95={p95:.3}s; target<{target:.3}s");
    if p95 < target { Ok(()) } else { Err("mock attach target was not met".into()) }
}
fn main() { if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); } }
