//! Build identity of a local source tree. A Git checkout answers through Git.
//! An archive export carries `.vade-build-id`, which `git archive` (and so
//! GitHub's tarballs and zips) fills through the export-subst attribute.
//! Anything else builds as `unknown`. Std-only: cli/tests includes this file.
use std::fs;
use std::path::Path;

pub const FILE: &str = ".vade-build-id";
pub const UNKNOWN: &str = "unknown";
pub const NOTICE: &str = "This Vade tree came from an archive without a build id (or was edited after download), so it builds as 'unknown'; use git clone for a reproducible build id.";
// The same generated output the engine's archive provenance ignores.
const SKIP_DIRS: [&str; 5] = ["target", "artifacts", "provenance", "__pycache__", ".git"];

#[derive(Debug, PartialEq)]
pub enum Probe { Git, Archive(String), Unknown }

/// Only Git metadata at the tree root counts: an enclosing checkout's HEAD
/// never describes an extracted archive.
pub fn probe(root: &Path) -> Probe {
    if fs::symlink_metadata(root.join(".git")).is_ok() { return Probe::Git; }
    // A regular, small file only; never block on a FIFO or follow a symlink.
    let path = root.join(FILE);
    if !fs::symlink_metadata(&path).is_ok_and(|m| m.is_file() && m.len() <= 64) { return Probe::Unknown; }
    // An unsubstituted `$Format:%H$` is not hex and falls through to unknown.
    match fs::read_to_string(&path).map(|s| s.trim().to_owned()) {
        Ok(id) if id.len() == 40 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) => Probe::Archive(id),
        _ => Probe::Unknown,
    }
}

/// Without Git's index, the inventory is every regular file below `engine/`
/// and `harness/` plus `rust-toolchain.toml`, minus generated output. Sorted.
pub fn archive_files(root: &Path) -> Result<Vec<String>, String> {
    fn walk(root: &Path, dir: &str, files: &mut Vec<String>) -> Result<(), String> {
        let entries = fs::read_dir(root.join(dir)).map_err(|_| "A source folder is unreadable.")?;
        for entry in entries {
            let entry = entry.map_err(|_| "A source folder is unreadable.")?;
            let name = entry.file_name().into_string().map_err(|_| "The source inventory contains an invalid path.")?;
            let kind = entry.file_type().map_err(|_| "A source file is unreadable.")?;
            let path = format!("{dir}/{name}");
            if (kind.is_dir() || kind.is_symlink()) && SKIP_DIRS.contains(&name.as_str()) { continue; }
            if kind.is_dir() { walk(root, &path, files)?; continue; }
            if !kind.is_file() {
                return Err("A source file is not a regular file; restore the source tree before installing.".into());
            }
            if name.ends_with(".ptx") || name.ends_with(".pyc") || name == ".DS_Store" { continue; }
            files.push(path);
        }
        Ok(())
    }
    let mut files = Vec::new();
    for dir in ["engine", "harness"] { walk(root, dir, &mut files)?; }
    if fs::symlink_metadata(root.join("rust-toolchain.toml")).is_ok_and(|m| m.is_file()) {
        files.push("rust-toolchain.toml".into());
    }
    files.sort();
    Ok(files)
}

/// `git archive` stamps every entry with the commit time, so a listed file
/// newer than the build id file was changed after extraction.
pub fn edited(root: &Path, files: &[String]) -> bool {
    let modified = |name: &str| fs::symlink_metadata(root.join(name)).and_then(|m| m.modified());
    let Ok(stamp) = modified(FILE) else { return true; };
    files.iter().any(|name| modified(name).is_ok_and(|time| time > stamp))
}

/// The id a launch records for a tree without Git metadata, and whether the
/// archive notice applies.
pub fn archive_id(probe: Probe, root: &Path, files: &[String]) -> (String, bool) {
    match probe {
        Probe::Archive(id) if !edited(root, files) => (id, false),
        _ => (UNKNOWN.into(), true),
    }
}
