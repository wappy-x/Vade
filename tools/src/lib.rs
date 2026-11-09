#![forbid(unsafe_code)]
pub mod bundle;
pub mod docs;
pub mod json;
pub mod cuda;
pub mod chart;
pub mod fetch;
pub mod bench_docs;
pub mod rigmark_dashboard;
pub mod one_path;
pub mod unsafe_sites;
pub mod release_source;
// Share the existing safe, streaming implementation. No Cargo dependency is needed.
#[path = "../../engine/tools/export/src/sha256.rs"]
pub mod sha256;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Whether `s` is exactly `len` lowercase hex digits.
pub fn is_lower_hex(s: &str, len: usize) -> bool { s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
pub fn root() -> std::path::PathBuf { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().into() }
pub fn output(command: &mut std::process::Command) -> Result<Vec<u8>> {
    let result = command.output()?;
    if !result.status.success() { return Err(format!("command failed ({}): {}", result.status,
        String::from_utf8_lossy(&result.stderr)).into()); }
    Ok(result.stdout)
}
