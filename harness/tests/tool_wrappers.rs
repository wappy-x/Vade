//! Every `ToolExecutor` wrapper must forward every trait method that has a
//! default body. A wrapper that forgets one gets the default, which reports no
//! check and no missing deliverable, so `--check` was once skipped without a
//! word in command-line agent runs. This test reads the source: a new default
//! method or a new wrapper is covered without editing the test.
use std::path::{Path, PathBuf};

/// (wrapper, method, reason): a wrapper that leaves a default method on
/// purpose. Each entry must still match a real wrapper and method.
const ALLOWED: &[(&str, &str, &str)] = &[
    ("ActiveTools", "missing_deliverables", NOT_FORWARDED),
    ("ParentTools", "missing_deliverables", NOT_FORWARDED),
    ("ScopedTools", "missing_deliverables", NOT_FORWARDED),
];

/// Reason for the three entries above.
const NOT_FORWARDED: &str =
    "not forwarded on purpose: the detector is a heuristic that fails correct runs (\"Write a small HTTP server in Node.js.\", \"Add tests for the login function.\"); see the strict-detector follow-up";

/// Source with comments, strings and character literals blanked to spaces,
/// so braces inside them do not count. Lengths and offsets are kept.
fn code_only(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for byte in &mut out[from..to] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    while i < bytes.len() {
        let rest = &bytes[i..];
        if rest.starts_with(b"//") {
            let end = rest.iter().position(|&b| b == b'\n').map_or(bytes.len(), |n| i + n);
            blank(&mut out, i, end);
            i = end;
        } else if rest.starts_with(b"/*") {
            let mut depth = 0;
            let mut j = i;
            while j < bytes.len() {
                if bytes[j..].starts_with(b"/*") {
                    depth += 1;
                    j += 2;
                } else if bytes[j..].starts_with(b"*/") {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            blank(&mut out, i, j);
            i = j;
        } else if rest.starts_with(b"r#") || rest.starts_with(b"r\"") || rest.starts_with(b"br#") || rest.starts_with(b"br\"") {
            let start = i;
            let mut j = i + if rest[0] == b'b' { 2 } else { 1 };
            let hashes = bytes[j..].iter().take_while(|&&b| b == b'#').count();
            j += hashes;
            if bytes.get(j) != Some(&b'"') || (i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')) {
                i += 1;
                continue;
            }
            let close: Vec<u8> = std::iter::once(b'"').chain(std::iter::repeat_n(b'#', hashes)).collect();
            j += 1;
            while j < bytes.len() && !bytes[j..].starts_with(&close) {
                j += 1;
            }
            j = (j + close.len()).min(bytes.len());
            blank(&mut out, start, j);
            i = j;
        } else if rest[0] == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                j += if bytes[j] == b'\\' { 2 } else { 1 };
            }
            j = (j + 1).min(bytes.len());
            blank(&mut out, i, j);
            i = j;
        } else if rest[0] == b'\'' {
            // A character literal ('x', '\n', '\u{7f}', '{'); otherwise a lifetime.
            let end = if rest.get(1) == Some(&b'\\') {
                rest.iter().skip(2).position(|&b| b == b'\'') .map(|n| n + 3)
            } else {
                let width = std::str::from_utf8(&rest[1..rest.len().min(5)])
                    .ok()
                    .or_else(|| (2..5).rev().find_map(|n| std::str::from_utf8(&rest[1..rest.len().min(n)]).ok()))
                    .and_then(|s| s.chars().next())
                    .map_or(1, char::len_utf8);
                (rest.get(1 + width) == Some(&b'\'')) .then_some(width + 2)
            };
            match end {
                Some(n) => {
                    blank(&mut out, i, i + n);
                    i += n;
                }
                None => i += 1,
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(out).expect("blanking keeps UTF-8")
}

/// The body between the brace that opens at or after `from` and its match.
fn block(code: &str, from: usize) -> &str {
    let open = from + code[from..].find('{').expect("block opens");
    let mut depth = 0;
    for (n, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + n];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced block")
}

/// Names of the functions declared directly in a trait or impl body, each
/// with whether it has a body (`{`) or is only declared (`;`).
fn functions(body: &str) -> Vec<(String, bool)> {
    let mut found = Vec::new();
    let mut depth = 0;
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'f' if depth == 0
                && body[i..].starts_with("fn ")
                && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')) =>
            {
                let name: String = body[i + 3..]
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                let after = &body[i..];
                let has_body = match (after.find('{'), after.find(';')) {
                    (Some(open), Some(semi)) => open < semi,
                    (Some(_), None) => true,
                    _ => false,
                };
                found.push((name, has_body));
            }
            _ => {}
        }
        i += 1;
    }
    found
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn calls_self_field_execute(body: &str) -> bool {
    let mut at = 0;
    while let Some(n) = body[at..].find("self.") {
        let field = &body[at + n + "self.".len()..];
        let trimmed = field.trim_start();
        let ident_len = trimmed.bytes().take_while(|&b| is_ident(b)).count();
        if ident_len > 0 && trimmed[ident_len..].trim_start().starts_with(".execute(") {
            return true;
        }
        at += n + "self.".len();
    }
    false
}

fn struct_mentions_executor(code: &str, name: &str) -> bool {
    let needle = format!("struct {name}");
    let bytes = code.as_bytes();
    let mut at = 0;
    while let Some(n) = code[at..].find(&needle) {
        let pos = at + n;
        let end = pos + needle.len();
        at = end;
        if (pos > 0 && is_ident(bytes[pos - 1])) || bytes.get(end).is_some_and(|&b| is_ident(b)) {
            continue;
        }
        let after = &code[end..];
        let has_brace = match (after.find('{'), after.find(';')) {
            (Some(open), Some(semi)) => open < semi,
            (Some(_), None) => true,
            _ => false,
        };
        if has_brace {
            if block(code, end).contains("ToolExecutor") {
                return true;
            }
        } else if let Some(semi) = after.find(';')
            && after[..semi].contains("ToolExecutor")
        {
            return true;
        }
    }
    false
}

fn impl_header_start(code: &str, before: usize) -> usize {
    let bytes = code.as_bytes();
    let mut i = before;
    while i >= 4 {
        i -= 1;
        if code[i..].starts_with("impl")
            && (i == 0 || !is_ident(bytes[i - 1]))
            && !is_ident(bytes[i + 4])
        {
            return i;
        }
    }
    0
}

/// Every impl of `ToolExecutor` in `code` that wraps an inner executor
/// (via a `ToolExecutor` type bound, a `ToolExecutor` struct field, or a
/// `self.<field>.execute(` call): `(type name, method names, impl body)`.
fn wrappers_in_code(code: &str) -> Vec<(String, Vec<String>, String)> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(n) = code[at..].find("ToolExecutor for ") {
        let trait_pos = at + n;
        let start = trait_pos + "ToolExecutor for ".len();
        at = start;
        let name: String = code[start..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        let impl_start = impl_header_start(code, trait_pos);
        let open = start + code[start..].find('{').expect("impl block opens");
        let header = &code[impl_start..open];
        let body = block(code, start);
        let has_bound = header.matches("ToolExecutor").count() > 1;
        if !has_bound && !struct_mentions_executor(code, &name) && !calls_self_field_execute(body) {
            continue;
        }
        let defined = functions(body).into_iter().map(|(name, _)| name).collect();
        found.push((name, defined, body.to_owned()));
    }
    found
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Test fixtures may leave defaults; only shipped wrappers must forward.
fn test_only(path: &Path) -> bool {
    let name = path.file_name().unwrap().to_string_lossy();
    name == "tests.rs" || name.ends_with("_tests.rs") || path.components().any(|c| c.as_os_str() == "tests")
}

fn wrappers(src: &Path) -> Vec<(String, String, Vec<String>, String)> {
    let mut files = Vec::new();
    rust_files(src, &mut files);
    files.sort();
    let mut found = Vec::new();
    for path in files.iter().filter(|p| !test_only(p)) {
        let code = code_only(&std::fs::read_to_string(path).unwrap());
        let file = path.strip_prefix(src).unwrap().display().to_string();
        for (name, defined, body) in wrappers_in_code(&code) {
            found.push((file.clone(), name, defined, body));
        }
    }
    found
}

fn missing_forwards(
    defaults: &[String],
    wrappers: &[(String, String, Vec<String>, String)],
    allowed: &[(&str, &str, &str)],
) -> (Vec<String>, Vec<bool>) {
    let mut dropped = Vec::new();
    let mut used = vec![false; allowed.len()];
    for (file, wrapper, defined, body) in wrappers {
        for method in defaults {
            let forwards = defined.contains(method) && body.contains(&format!(".{method}("));
            if forwards {
                continue;
            }
            match allowed.iter().position(|(w, m, _)| w == wrapper && m == method) {
                Some(n) => used[n] = true,
                None => dropped.push(format!("{file}: {wrapper} does not forward {method}")),
            }
        }
    }
    (dropped, used)
}

#[test]
fn every_tool_executor_wrapper_forwards_every_default_method() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let agent = code_only(&std::fs::read_to_string(src.join("agent.rs")).unwrap());
    let trait_at = agent.find("pub trait ToolExecutor").expect("ToolExecutor trait in agent.rs");
    let defaults: Vec<String> = functions(block(&agent, trait_at))
        .into_iter()
        .filter_map(|(name, has_body)| has_body.then_some(name))
        .collect();
    // Guard the parser itself: these were the methods once dropped.
    for name in ["take_completion_check", "missing_deliverables", "take_pending_check", "check_enabled"] {
        assert!(defaults.iter().any(|d| d == name), "parser missed default {name}: {defaults:?}");
    }
    let wrappers = wrappers(&src);
    for name in ["ActiveTools", "ParentTools", "ScopedTools", "McpTools"] {
        assert!(wrappers.iter().any(|(_, w, _, _)| w == name), "walk missed wrapper {name}: {wrappers:?}");
    }
    let (dropped, used) = missing_forwards(&defaults, &wrappers, ALLOWED);
    assert!(
        dropped.is_empty(),
        "ToolExecutor wrappers fall back to default bodies (forward each to the inner executor):\n{}",
        dropped.join("\n")
    );
    for ((wrapper, method, _), used) in ALLOWED.iter().zip(used) {
        assert!(used, "stale ALLOWED entry: {wrapper} {method}");
    }
}

#[test]
fn source_reader_ignores_braces_in_strings_comments_and_chars() {
    let code = code_only("fn a() { let s = \"}\"; let r = r#\"{\"#; let c = '{'; // }\n /* { */ }\nfn b<'a>(x: &'a str) {}");
    let found = functions(&code);
    assert_eq!(found, [("a".to_owned(), true), ("b".to_owned(), true)]);
    assert_eq!(block(&code, 0).matches('}').count(), 0);
    let declared = functions("fn x(&self) -> bool; fn y(&self) { }");
    assert_eq!(declared, [("x".to_owned(), false), ("y".to_owned(), true)]);
}

#[test]
fn detector_catches_new_unforwarded_or_stubbed_default_method_and_skips_leaf_adapters() {
    let sample = code_only(
        r#"
        struct GenericWrap<T> { inner: T }
        impl<T: ToolExecutor> ToolExecutor for GenericWrap<T> {
            fn execute(&mut self) { self.inner.execute() }
            fn check_enabled(&self) -> bool { self.inner.check_enabled() }
        }
        struct DynWrap<'a> { inner: &'a mut dyn ToolExecutor }
        impl ToolExecutor for DynWrap<'_> {
            fn execute(&mut self) { self.inner.execute() }
            fn check_enabled(&self) -> bool { self.inner.check_enabled() }
            fn newly_added(&self) -> bool { false }
        }
        struct LeafAdapter;
        impl ToolExecutor for LeafAdapter {
            fn execute(&mut self) { let mut local = Helper; local.execute() }
        }
        "#,
    );
    let found: Vec<_> = wrappers_in_code(&sample)
        .into_iter()
        .map(|(w, d, b)| ("sample.rs".to_owned(), w, d, b))
        .collect();
    let names: Vec<_> = found.iter().map(|(_, w, _, _)| w.as_str()).collect();
    assert_eq!(names, ["GenericWrap", "DynWrap"]);
    let defaults = vec!["check_enabled".to_owned(), "newly_added".to_owned()];
    let (dropped, _) = missing_forwards(&defaults, &found, &[]);
    assert_eq!(
        dropped,
        [
            "sample.rs: GenericWrap does not forward newly_added",
            "sample.rs: DynWrap does not forward newly_added",
        ]
    );
}
