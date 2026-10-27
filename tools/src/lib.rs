#![forbid(unsafe_code)]
pub mod bundle;
pub mod docs;
pub mod json;
pub mod cuda;
pub mod chart;
pub mod fetch;
// Share the existing safe, streaming implementation. No Cargo dependency is needed.
#[path = "../../engine/tools/export/src/sha256.rs"]
pub mod sha256;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn root() -> std::path::PathBuf { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().into() }
pub fn output(command: &mut std::process::Command) -> Result<Vec<u8>> {
    let result = command.output()?;
    if !result.status.success() { return Err(format!("command failed ({}): {}", result.status,
        String::from_utf8_lossy(&result.stderr)).into()); }
    Ok(result.stdout)
}
