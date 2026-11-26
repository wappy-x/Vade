//! CHOICE: the Inkling image has its own stamp file and its own build command (`cargo kernels-inkling`), not a variant of `ptx_stamp.rs`.
//! REASON: `ptx_stamp.rs` and `build-kernels/src/main.rs` are in the general image's stamp list; so no file of that list changes (lead's ruling 2026-11-22 18:43:56).
//!
//! Bind the built Inkling PTX (crate `spark-inkling-kernels`, Ruling R6h) to its sources and build flags, as
//! `ptx_stamp.rs` does for the other images: the stamp text holds the identity and the SHA-256 of the sources and the
//! PTX; the host build (`build.rs`) refuses a stale image.
use std::{fs, io, path::{Path, PathBuf}};
#[allow(clippy::duplicate_mod)] // the shared file is included on purpose.
#[path = "../../tools/export/src/sha256.rs"]
mod sha256;

pub const PTX_VERSION: &str = "8.8";
pub const TARGET_CPU: &str = "sm_121a";
/// The flags of the other images (`ptx_stamp::Variant::flags`).
pub const FLAGS: &str = "-C target-cpu=sm_121a -C target-feature=+ptx88 -C overflow-checks=off";
pub const PACKAGE: &str = "spark-inkling-kernels";
pub const TARGET_DIR: &str = "../../target/inkling";
pub const BUILD_COMMAND: &str = "cargo kernels-inkling";
/// The crate's own output (`layer` false) or the stamped copy the host embeds (`layer` true).
pub fn ptx_path(root: &Path, layer: bool) -> PathBuf {
    root.join(TARGET_DIR).join("nvptx64-nvidia-cuda/release").join(if layer { "spark_inkling_layer.ptx" } else { "spark_inkling_kernels.ptx" })
}
fn identity() -> String { format!("image=inkling\ntarget-cpu={TARGET_CPU}\nptx-version={PTX_VERSION}\nrustflags={FLAGS}\n") }
/// The sources that make the image: its crate, the shared files it includes by path, and its build tools.
pub fn source_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = vec![root.join("kernels-inkling/Cargo.toml"), root.join("kernels-inkling/REGISTERS.txt"), root.join("kernels/src/ptx.rs"),
        root.join("src/fixed_math.rs"), root.join("src/block_ops.rs"), root.join("src/block_rows.rs"), root.join("src/block_attention.rs"),
        root.join("src/block_stream.rs"), root.join("src/block_experts.rs"), root.join("src/inkling_launch.rs"), root.join("src/staged_product.rs"), root.join("src/packed_product.rs"),
        root.join("src/few_product.rs"), root.join("src/bf16_panels.rs"), root.join("src/block_scores.rs"), root.join("src/block_values.rs"), root.join("src/staged_narrow.rs"), root.join("src/tc_rows.rs"), root.join("src/prefill_moe.rs"), root.join("src/prefill_gate.rs"), root.join("src/flash_window.rs"),
        root.join("src/fast_ops.rs"), root.join("build-kernels/src/bin/inkling.rs"), root.join("inkling_stamp.rs")];
    for entry in fs::read_dir(root.join("kernels-inkling/src"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "rs") { files.push(path); }
    }
    files.sort();
    Ok(files)
}
fn validate(bytes: &[u8]) -> io::Result<()> {
    let text = std::str::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    for (directive, expected) in [(".version ", PTX_VERSION), (".target ", TARGET_CPU)] {
        let actual = text.lines().find_map(|line| line.trim().strip_prefix(directive))
            .and_then(|value| value.split(|c: char| c.is_whitespace() || c == ',').next());
        if actual != Some(expected) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Inkling PTX mismatch: expected {directive}{expected}, got {actual:?}")));
        }
    }
    Ok(())
}
fn digest(root: &Path, files: &[PathBuf], ptx: &[u8]) -> io::Result<String> {
    let mut input = identity().into_bytes();
    for path in files {
        let name = path.strip_prefix(root).unwrap().to_string_lossy();
        let bytes = fs::read(path)?;
        input.extend_from_slice(&(name.len() as u64).to_le_bytes());
        input.extend_from_slice(name.as_bytes());
        input.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        input.extend_from_slice(&bytes);
    }
    input.extend_from_slice(&(ptx.len() as u64).to_le_bytes());
    input.extend_from_slice(ptx);
    Ok(sha256::digest(&input))
}
fn expected(root: &Path, bytes: &[u8]) -> io::Result<String> { Ok(format!("{}sha256={}\n", identity(), digest(root, &source_files(root)?, bytes)?)) }
/// Writes the stamp beside the PTX (`.ptx.sha256`). Used by `cargo kernels-inkling`.
#[allow(dead_code)]
pub fn stamp(root: &Path, ptx: &Path) -> io::Result<()> {
    let bytes = fs::read(ptx)?;
    if bytes.is_empty() { return Err(io::Error::new(io::ErrorKind::InvalidData, "empty PTX")); }
    validate(&bytes)?;
    fs::write(ptx.with_extension("ptx.sha256"), expected(root, &bytes)?)
}
/// Refuses a stale image: the stamp must name these sources and these bytes. Used by the host build script.
#[allow(dead_code)]
pub fn verify(root: &Path, ptx: &Path, bytes: &[u8]) -> io::Result<()> {
    validate(bytes)?;
    println!("cargo:rerun-if-changed={}", root.join("kernels-inkling/src").display());
    for file in source_files(root)? { println!("cargo:rerun-if-changed={}", file.display()); }
    let stamp = ptx.with_extension("ptx.sha256");
    println!("cargo:rerun-if-changed={}", stamp.display());
    if fs::read_to_string(stamp)? != expected(root, bytes)? {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "stale Inkling PTX or sources"));
    }
    Ok(())
}
