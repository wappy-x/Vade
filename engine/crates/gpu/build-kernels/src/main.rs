//! Build the NVPTX kernels with the pinned nightly, then stamp both PTX files.
//! Run from anywhere in the workspace with `cargo kernels`.
use std::{env, ffi::OsStr, fs, path::{Path, PathBuf}, process::{self, Command, Stdio}};
#[path = "../../ptx_stamp.rs"]
#[allow(dead_code)] // `verify` belongs to the host build scripts.
mod ptx_stamp;

const NIGHTLY: &str = "nightly-2026-10-01";
const TARGET: &str = "nvptx64-nvidia-cuda";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // No flag builds the two images every Linux engine build embeds: the production image, then the tensor-core
    // image. So `cargo kernels` stays the one kernel step of every build recipe (the installer runs only that).
    let variants: &[ptx_stamp::Variant] = match env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => &[ptx_stamp::Variant::Production, ptx_stamp::Variant::Tc],
        [flag] if flag == "--fp4" => &[ptx_stamp::Variant::Fp4],
        [flag] if flag == "--tc" => &[ptx_stamp::Variant::Tc],
        _ => return Err("usage: cargo kernels [--fp4 | --tc]".into()),
    };
    // The environment is read here, once, for every image of the run.
    // Paths and stamp names are relative to crates/gpu. Prefer the directory `cargo run`
    // passes: a copied or moved tree can reuse a binary that names the old checkout.
    let manifest = env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from).unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into());
    let bin = env::var_os("KERNEL_BIN").filter(|bin| !bin.is_empty());
    let search = env::var_os("PATH").unwrap_or_default();
    variants.iter().try_for_each(|&variant| build(variant, &manifest, bin.as_deref(), &search))
}

/// Build, copy and stamp one image.
fn build(variant: ptx_stamp::Variant, manifest: &Path, bin: Option<&OsStr>, search: &OsStr) -> Result<(), String> {
    let (fp4, tc) = (variant == ptx_stamp::Variant::Fp4, variant == ptx_stamp::Variant::Tc);
    let gpu = manifest.parent().unwrap();
    env::set_current_dir(gpu).map_err(|e| format!("cannot enter {}: {e}", gpu.display()))?;
    let mut cargo = match bin {
        Some(bin) => {
            let mut path = bin.to_os_string();
            path.push(":");
            path.push(search);
            let mut cargo = Command::new(Path::new(bin).join("cargo"));
            cargo.env("PATH", path);
            cargo
        }
        None => {
            let installed = Command::new("rustup").args(["target", "list", "--toolchain", NIGHTLY, "--installed"])
                .stderr(Stdio::null()).output()
                .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).lines().any(|line| line == TARGET));
            if !installed {
                return Err(format!("Missing nightly NVPTX toolchain/target. Run:\nrustup toolchain install {NIGHTLY} \
                    --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target {TARGET}"));
            }
            let mut cargo = Command::new("cargo");
            cargo.arg(format!("+{NIGHTLY}"));
            cargo
        }
    };
    // build-std fetches compiler dependencies on the first build. Keep each variant
    // at its workspace path, even with CARGO_TARGET_DIR; FP4 never touches production.
    // `cargo run` points CARGO at the host toolchain; the nightly build sets its own.
    // Block-scaled FP4 needs the architecture-specific target. Keep the ordinary
    // FP8/BF16 build unchanged; PTX 8.8 is supported by the installed CUDA driver.
    let mut build = vec!["build", "-Z", "build-std=core", "-p", variant.package()];
    if !variant.features().is_empty() { build.extend(["--features", variant.features()]); }
    build.extend(["--release", "--target", TARGET, "--target-dir", variant.target_dir()]);
    let status = cargo.args(build)
        .env("RUSTFLAGS", variant.flags()).env_remove("CARGO")
        .status().map_err(|e| format!("cannot run cargo: {e}"))?;
    if !status.success() { process::exit(status.code().unwrap_or(1)); }
    let (kernels, layer) = (variant.ptx_path(Path::new("."), false), variant.ptx_path(Path::new("."), true));
    if fp4 {
        fs::copy(kernels.with_file_name("spark_rust_kernels.ptx"), &kernels)
            .map_err(|e| format!("cannot copy FP4 PTX: {e}"))?;
    }
    fs::copy(&kernels, &layer).map_err(|e| format!("cannot copy {}: {e}", kernels.display()))?;
    for ptx in [kernels, layer] {
        ptx_stamp::stamp(Path::new("."), &ptx, variant).map_err(|e| format!("cannot stamp {}: {e}", ptx.display()))?;
    }
    if fp4 { eprintln!("FP4 PTX is isolated in target/fp4; host opt-in: SPARK_PTX_VARIANT=fp4 cargo build"); }
    if tc { register_check(&variant.ptx_path(Path::new("."), true), Path::new("kernels-tc/REGISTERS.txt"))?; }
    Ok(())
}

/// Kernel kit, part 2: the register check of the tensor-core image at build time. Compiles the stamped PTX with
/// ptxas (`sm_121a`, the release arguments) and holds every entry to its row of `kernels-tc/REGISTERS.txt`.
/// A machine that is not Linux and has no ptxas (the Mac) skips the check and says so. A Linux build with no ptxas fails.
fn register_check(ptx: &Path, table: &Path) -> Result<(), String> {
    let ptxas = match [PathBuf::from("ptxas"), PathBuf::from("/usr/local/cuda/bin/ptxas")].into_iter()
        .find(|bin| Command::new(bin).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())) {
        Some(bin) => bin,
        // A Linux build is a build for a Spark: an unchecked image must not be stamped there.
        None if cfg!(target_os = "linux") => return Err("TC REGISTER CHECK FAIL: no ptxas on this machine (PATH or /usr/local/cuda/bin); the tensor-core image is not stamped unchecked".into()),
        None => { eprintln!("TC REGISTER CHECK SKIPPED: no ptxas on this machine (not a Linux build); build on a Spark for the check"); return Ok(()); }
    };
    let out = Command::new(&ptxas).args(["--gpu-name=sm_121a", "-m64", "-O3", "-v", "-o"]).arg(ptx.with_extension("cubin")).arg(ptx)
        .output().map_err(|e| format!("cannot run ptxas: {e}"))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() { return Err(format!("TC REGISTER CHECK FAIL: ptxas rejected the image\n{text}")); }
    let number_before = |line: &str, tail: &str| -> Option<u64> {
        let end = line.find(tail)?;
        line[..end].trim_end().rsplit(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    // (entry, registers, shared bytes, stack + spill bytes)
    let mut measured: Vec<(String, u64, u64, u64)> = vec![];
    let (mut name, mut spill) = (None::<String>, 0u64);
    for line in text.lines() {
        if let Some(rest) = line.split("Compiling entry function '").nth(1) { name = rest.split('\'').next().map(str::to_owned); spill = 0; }
        else if line.contains("bytes spill stores") {
            spill = ["bytes stack frame", "bytes spill stores", "bytes spill loads"].iter().filter_map(|t| number_before(line, t)).sum();
        } else if let (Some(entry), Some(rest)) = (name.as_ref(), line.split("Used ").nth(1)) {
            let registers = rest.split(' ').next().and_then(|n| n.parse().ok()).ok_or("TC REGISTER CHECK FAIL: cannot read a ptxas register line")?;
            measured.push((entry.clone(), registers, number_before(line, "bytes smem").unwrap_or(0), spill));
            name = None;
        }
    }
    let rows = fs::read_to_string(table).map_err(|e| format!("cannot read {}: {e}", table.display()))?;
    let rows: Vec<(String, u64, u64, u64)> = rows.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        match f.as_slice() {
            [entry, threads, registers, shared] => Ok((entry.to_string(), threads.parse().map_err(|_| l)?, registers.parse().map_err(|_| l)?, shared.parse().map_err(|_| l)?)),
            _ => Err(l),
        }
    }).collect::<Result<_, &str>>().map_err(|l| format!("TC REGISTER CHECK FAIL: bad table row: {l}"))?;
    let mut failures = vec![];
    let mut report = String::new();
    for (entry, registers, shared, spill) in &measured {
        let Some((_, threads, max_registers, max_shared)) = rows.iter().find(|r| &r.0 == entry) else { failures.push(format!("{entry}: no row in the table")); continue; };
        // Blocks an SM: 65,536 registers and (design cap) 101,376 shared bytes; at most 1,536 threads.
        let by_registers = 65_536 / (registers * threads).max(1);
        let per_sm = by_registers.min(if *shared == 0 { u64::MAX } else { 101_376 / shared }).min(1_536 / threads);
        report += &format!("{entry} threads={threads} registers={registers} (max {max_registers}) shared={shared} (max {max_shared}) spill_bytes={spill} blocks_per_sm={per_sm}\n");
        if registers > max_registers { failures.push(format!("{entry}: {registers} registers, the table allows {max_registers}")); }
        if shared > max_shared { failures.push(format!("{entry}: {shared} shared bytes, the table allows {max_shared}")); }
        if *spill != 0 { failures.push(format!("{entry}: {spill} stack or spill bytes")); }
        if by_registers == 0 { failures.push(format!("{entry}: {registers} registers x {threads} threads cannot launch")); }
    }
    for (entry, ..) in &rows { if !measured.iter().any(|m| &m.0 == entry) { failures.push(format!("{entry}: in the table, not in the image")); } }
    fs::write(ptx.with_extension("registers.txt"), &report).map_err(|e| format!("cannot write the register receipt: {e}"))?;
    eprint!("{report}");
    if failures.is_empty() { eprintln!("TC REGISTER CHECK PASS entries={} ptxas={}", measured.len(), ptxas.display()); Ok(()) }
    else { Err(format!("TC REGISTER CHECK FAIL\n{}", failures.join("\n"))) }
}
