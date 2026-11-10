//! Build the NVPTX kernels with the pinned nightly, then stamp both PTX files.
//! Run from anywhere in the workspace with `cargo kernels`.
use std::{env, fs, path::{Path, PathBuf}, process::{self, Command, Stdio}};
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
    let fp4 = match env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => false,
        [flag] if flag == "--fp4" => true,
        _ => return Err("usage: cargo kernels [--fp4]".into()),
    };
    // Paths and stamp names are relative to crates/gpu. Prefer the directory `cargo run`
    // passes: a copied or moved tree can reuse a binary that names the old checkout.
    let manifest = env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from).unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into());
    let gpu = manifest.parent().unwrap();
    env::set_current_dir(gpu).map_err(|e| format!("cannot enter {}: {e}", gpu.display()))?;
    let mut cargo = match env::var_os("KERNEL_BIN").filter(|bin| !bin.is_empty()) {
        Some(bin) => {
            let mut path = bin.clone();
            path.push(":");
            path.push(env::var_os("PATH").unwrap_or_default());
            let mut cargo = Command::new(Path::new(&bin).join("cargo"));
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
    let variant = if fp4 { ptx_stamp::Variant::Fp4 } else { ptx_stamp::Variant::Production };
    let status = cargo.args(["build", "-Z", "build-std=core", "-p", "spark-rust-kernels", "--features", variant.features(),
            "--release", "--target", TARGET, "--target-dir", variant.target_dir()])
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
    Ok(())
}
