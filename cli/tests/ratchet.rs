use std::{collections::BTreeMap, fs, path::Path, process::Command};

// Count comparisons (== != < > <= >=) between a named box count and an
// integer literal, in either order, including fields/methods and parentheses.
// Also count integer match arms on those names. No alias/dataflow analysis.
const BOX_NAMES: &[&str] = &[
    "world", "world_size", "boxes", "box_count", "num_boxes", "n_boxes",
    "ranks", "rank_count", "num_ranks", "n_ranks",
];
const OPERATORS: &[&str] = &["==", "!=", "<", ">", "<=", ">="];

// Small Rust lexer: ignore comments, strings (including raw strings), chars;
// keep lifetimes as identifiers. Parentheses do not interrupt comparisons.
fn tokens(source: &str) -> Vec<String> {
    let b = source.as_bytes();
    let (mut i, mut out) = (0, Vec::new());
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' { i += 1; }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < b.len() && depth > 0 {
                if b[i..].starts_with(b"/*") { depth += 1; i += 2; }
                else if b[i..].starts_with(b"*/") { depth -= 1; i += 2; }
                else { i += 1; }
            }
        } else {
            let mut quote = i;
            if b[i] == b'b' || b[i] == b'c' { quote += 1; }
            let raw = b.get(quote) == Some(&b'r');
            if raw {
                quote += 1;
                while b.get(quote) == Some(&b'#') { quote += 1; }
            }
            if b.get(quote) == Some(&b'"') {
                let hashes = if raw { quote - i - 1 - usize::from(b[i] != b'r') } else { 0 };
                i = quote + 1;
                while i < b.len() {
                    if b[i] == b'"' && (!raw || b.get(i + 1..i + 1 + hashes)
                        .is_some_and(|s| s.iter().all(|&c| c == b'#'))) {
                        i += 1 + hashes; break;
                    }
                    if !raw && b[i] == b'\\' { i += 1; }
                    i += 1;
                }
                out.push("<string>".into());
            } else if b[i] == b'\'' && (b.get(i + 2) == Some(&b'\'') || b.get(i + 1) == Some(&b'\\')) {
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' { i += 2; }
                    else if b[i] == b'\'' { i += 1; break; }
                    else { i += 1; }
                }
                out.push("<char>".into());
            } else if b[i].is_ascii_alphanumeric() || b[i] == b'_' {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
                out.push(source[start..i].into());
            } else if b[i].is_ascii_whitespace() || matches!(b[i], b'(' | b')') { i += 1; }
            else {
                let start = i;
                i += 1;
                if i < b.len() && matches!((b[start], b[i]),
                    (b'=' | b'!' | b'<' | b'>', b'=') | (b'=', b'>')) { i += 1; }
                out.push(String::from_utf8_lossy(&b[start..i]).into_owned());
            }
        }
    }
    out
}

fn integer(s: &str) -> bool {
    s.as_bytes().first().is_some_and(u8::is_ascii_digit)
}

fn branches(source: &str) -> usize {
    let t = tokens(source);
    let comparison = t.windows(3).filter(|w| OPERATORS.contains(&w[1].as_str())
        && ((BOX_NAMES.contains(&w[0].as_str()) && integer(&w[2]))
            || (integer(&w[0]) && BOX_NAMES.contains(&w[2].as_str())))).count();
    let (mut arms, mut depth, mut box_match) = (0, 0usize, None);
    for (i, token) in t.iter().enumerate() {
        if token == "match" {
            let expr: Vec<_> = t[i + 1..].iter().take_while(|s| *s != "{").collect();
            if expr.last().is_some_and(|s| BOX_NAMES.contains(&s.as_str())) {
                box_match = Some(depth + 1);
            }
        }
        if token == "{" { depth += 1; }
        if box_match == Some(depth) && integer(token)
            && t.get(i + 1).is_some_and(|s| s == "=>" || s == "|" || s == "if") { arms += 1; }
        if token == "}" {
            if box_match == Some(depth) { box_match = None; }
            depth = depth.saturating_sub(1);
        }
    }
    comparison + arms
}

// Cargo declarations: normal/dev/build/workspace/target dependency tables,
// inline and multiline entries, and dependency subtables. Only explicit paths
// to manifests inside this checkout count as first-party. Inherited workspace
// declarations fail closed until given explicit paths. Lockfile source entries
// count separately, so registry/git dependencies cannot hide in stale locks.
fn dependencies(root: &Path, manifest: &Path, source: &str) -> usize {
    fn external(root: &Path, manifest: &Path, declaration: &str) -> bool {
        let fields: Vec<_> = declaration.split(|c: char| !c.is_ascii_alphanumeric()
            && !matches!(c, '_' | '-' | '/' | '.' | '\\')).filter(|s| !s.is_empty()).collect();
        if fields.iter().any(|s| matches!(*s, "git" | "registry")) { return true; }
        fields.windows(2).find(|w| w[0] == "path").is_none_or(|w| {
            manifest.parent().unwrap().join(w[1]).join("Cargo.toml").canonicalize()
                .map_or(true, |p| !p.starts_with(root))
        })
    }
    let (mut count, mut section, mut entry) = (0, String::new(), String::new());
    for raw in source.lines().chain(std::iter::once("[end]")) {
        let line = raw.split('#').next().unwrap().trim();
        if line.starts_with('[') {
            if !entry.is_empty() { count += usize::from(external(root, manifest, &entry)); entry.clear(); }
            section = line.trim_matches(['[', ']']).replace(['\'', '"'], "");
            let parts: Vec<_> = section.split('.').collect();
            if let Some(index) = parts.iter().position(|p| matches!(*p,
                "dependencies" | "dev-dependencies" | "build-dependencies")) {
                if index + 1 < parts.len() { entry.push_str("{ "); }
            }
        } else if section.split('.').any(|p| matches!(p,
            "dependencies" | "dev-dependencies" | "build-dependencies")) && !line.is_empty() {
            entry.push_str(line);
            entry.push(' ');
            if !entry.starts_with("{ ") && (!entry.contains('{') || entry.contains('}')) {
                count += usize::from(external(root, manifest, &entry)); entry.clear();
            }
        }
    }
    count
}

#[test]
fn repository_ratchet() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().canonicalize().unwrap();
    // Git's inventory includes new unignored files, avoids build/private output,
    // and works in normal clones and linked worktrees; no recursive symlinks.
    let output = Command::new("git").current_dir(&root)
        .args(["ls-files", "--cached", "--others", "--exclude-standard", "-z"]).output().unwrap();
    assert!(output.status.success(), "git source inventory failed");
    let inventory = String::from_utf8(output.stdout).unwrap();
    let mut paths: Vec<_> = inventory.split('\0').filter(|s| !s.is_empty()).collect();
    paths.sort_unstable(); paths.dedup();
    let mut counts = BTreeMap::from([
        ("box_count_branches", 0usize), ("large_rs_files", 0), ("third_party_dependencies", 0),
    ]);
    for path in paths {
        let full = root.join(path);
        if !full.exists() { continue; } // A tracked file deleted in the working tree.
        let name = full.file_name().unwrap().to_str().unwrap();
        if full.extension().is_some_and(|s| s == "rs") {
            let source = fs::read_to_string(&full).unwrap();
            let n = branches(&source);
            *counts.get_mut("box_count_branches").unwrap() += n;
            if source.lines().count() > 1_500 {
                *counts.get_mut("large_rs_files").unwrap() += 1;
                println!("large Rust file: {path} ({} lines)", source.lines().count());
            }
            if n > 0 { println!("box-count comparisons/arms: {path}: {n}"); }
        } else if name == "Cargo.toml" || name == "Cargo.lock" {
            let source = fs::read_to_string(&full).unwrap();
            let n = if name == "Cargo.toml" { dependencies(&root, &full, &source) }
                else { source.lines().filter(|s| s.split_once('=')
                    .is_some_and(|(key, _)| key.trim() == "source")).count() };
            *counts.get_mut("third_party_dependencies").unwrap() += n;
            if n > 0 { println!("external dependency declarations/sources: {path}: {n}"); }
        }
    }
    let baseline = fs::read_to_string(root.join("cli/ratchet-baseline.txt")).unwrap();
    let limits: BTreeMap<_, usize> = baseline.lines().filter(|s| !s.starts_with('#') && !s.is_empty())
        .map(|s| { let (key, value) = s.split_once('=').expect("invalid baseline");
            (key, value.parse().expect("invalid baseline count")) }).collect();
    assert_eq!(limits.len(), counts.len(), "baseline keys must match counters");
    assert_eq!(limits.get("third_party_dependencies"), Some(&0), "dependency ceiling must stay zero");
    let mut increased = false;
    for (key, value) in &counts {
        let limit = limits.get(key).expect("missing baseline key");
        println!("{key}={value} (baseline {limit})");
        increased |= value > limit;
    }
    if counts.iter().any(|(key, value)| value < limits.get(key).unwrap()) {
        println!("Lower counts: replace baseline numbers with:");
        for (key, value) in &counts { println!("{key}={value}"); }
    }
    assert!(!increased, "repository ratchet increased; reduce the reported counts");
}
