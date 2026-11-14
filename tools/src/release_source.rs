//! Prepare an archive for a release whose bytes do not depend on comments.
use std::{fs, path::Path};
use crate::Result;

fn ident(b: u8) -> bool { b.is_ascii_alphanumeric() || b == b'_' || b >= 128 }

/// Keep literals and Rust tokens. Give each token a fixed source line.
/// This is for executable builds. Run source checks and doc tests on the original.
pub fn canonical(source: &str) -> Result<String> {
    let b = source.as_bytes(); let mut out = String::new(); let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_whitespace() { i += 1; continue; }
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' { i += 1; } continue;
        }
        if b[i..].starts_with(b"/*") {
            i += 2; let mut depth = 1;
            while i < b.len() && depth != 0 {
                if b[i..].starts_with(b"/*") { depth += 1; i += 2; }
                else if b[i..].starts_with(b"*/") { depth -= 1; i += 2; }
                else { i += 1; }
            }
            if depth != 0 { return Err("unclosed release source comment".into()); }
            continue;
        }
        let start = i; let mut quote = i;
        if matches!(b[i], b'b' | b'c') { quote += 1; }
        let raw = b.get(quote) == Some(&b'r');
        if raw { quote += 1; while b.get(quote) == Some(&b'#') { quote += 1; } }
        if b.get(quote) == Some(&b'"') {
            let hashes = if raw { quote - start - 1 - usize::from(b[start] != b'r') } else { 0 };
            i = quote + 1; let mut closed = false;
            while i < b.len() {
                if b[i] == b'"' && (!raw || b.get(i + 1..i + 1 + hashes)
                    .is_some_and(|s| s.iter().all(|&c| c == b'#'))) {
                    i += 1 + hashes; closed = true; break;
                }
                if !raw && b[i] == b'\\' { i += 1; }
                i += 1;
            }
            if !closed { return Err("unclosed release source string".into()); }
        } else {
            let char_start = start + usize::from(b[start] == b'b');
            let mut character = false;
            if b.get(char_start) == Some(&b'\'') {
                let value = char_start + 1; let mut end = value;
                if b.get(value) == Some(&b'\\') {
                    end += 2;
                    if b.get(value + 1) == Some(&b'u') && b.get(end) == Some(&b'{') {
                        while end < b.len() && b[end] != b'}' { end += 1; } end += 1;
                    } else if b.get(value + 1) == Some(&b'x') { end += 2; }
                } else if let Some(ch) = source.get(value..).and_then(|s| s.chars().next()) { end += ch.len_utf8(); }
                if b.get(end) == Some(&b'\'') { i = end + 1; character = true; }
                else if char_start == start && b.get(value).is_some_and(|&c| ident(c)) {
                    i = value; while i < b.len() && ident(b[i]) { i += 1; } character = true;
                }
            }
            if !character {
                if b[start].is_ascii_digit() {
                    i += 1;
                    if b[start] == b'0' && matches!(b.get(i), Some(b'x' | b'o' | b'b')) {
                        while i < b.len() && ident(b[i]) { i += 1; }
                    } else {
                        while b.get(i).is_some_and(|c| c.is_ascii_digit() || *c == b'_') { i += 1; }
                        if b.get(i) == Some(&b'.') && b.get(i + 1) != Some(&b'.')
                            && !b.get(i + 1).is_some_and(|&c| ident(c) && !c.is_ascii_digit()) {
                            i += 1;
                            while b.get(i).is_some_and(|c| c.is_ascii_digit() || *c == b'_') { i += 1; }
                        }
                        if matches!(b.get(i), Some(b'e' | b'E')) {
                            i += 1; if matches!(b.get(i), Some(b'+' | b'-')) { i += 1; }
                            while b.get(i).is_some_and(|c| c.is_ascii_digit() || *c == b'_') { i += 1; }
                        }
                        while i < b.len() && ident(b[i]) { i += 1; }
                    }
                } else if ident(b[start]) {
                    i += 1;
                    if b[start] == b'r' && b.get(i) == Some(&b'#') { i += 1; }
                    while i < b.len() && ident(b[i]) { i += 1; }
                } else {
                    const JOINT: &[&[u8]] = &[b"<<=", b">>=", b"...", b"..=", b"::", b"->", b"=>",
                        b"==", b"!=", b"<=", b">=", b"&&", b"||", b"<<", b">>", b"+=", b"-=",
                        b"*=", b"/=", b"%=", b"^=", b"&=", b"|=", b".."];
                    i += JOINT.iter().find(|s| b[start..].starts_with(s)).map_or(1, |s| s.len());
                }
            }
        }
        out.push_str(&source[start..i]); out.push('\n');
    }
    Ok(out)
}

/// Copy an extracted archive. Never edit a checkout or overwrite an output.
pub fn prepare(source: &Path, destination: &Path) -> Result<()> {
    if source.join(".git").exists() { return Err("release source must be an extracted archive, not a checkout".into()); }
    let source = source.canonicalize()?;
    let parent = destination.parent().ok_or("release destination needs a parent folder")?.canonicalize()?;
    if parent.starts_with(&source) { return Err("release destination must be outside the source archive".into()); }
    if !source.join("engine/Cargo.toml").is_file() { return Err("release source needs engine/Cargo.toml".into()); }
    fs::create_dir(destination)?;
    fn copy(source: &Path, destination: &Path, engine: bool) -> Result<()> {
        let mut entries = fs::read_dir(source)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name();
            if matches!(name.to_str(), Some("target" | ".local" | ".git")) { continue; }
            let from = entry.path(); let to = destination.join(&name); let kind = entry.file_type()?;
            if kind.is_symlink() { return Err(format!("release source has a symlink: {}", from.display()).into()); }
            if kind.is_dir() { fs::create_dir(&to)?; copy(&from, &to, engine || name == "engine")?; }
            else if kind.is_file() {
                fs::copy(&from, &to)?;
                if engine && from.extension().is_some_and(|s| s == "rs") {
                    fs::write(&to, canonical(&fs::read_to_string(&from)?)?)?;
                }
            } else { return Err(format!("release source has a special file: {}", from.display()).into()); }
        }
        Ok(())
    }
    copy(&source, destination, false)?;
    // The original archive and the build receipt keep the revision. It is not
    // an input to rustc. An unknown revision also makes archive dirty state fixed.
    fs::write(destination.join("engine/REVISION"), "unknown\n")?;
    Ok(())
}
