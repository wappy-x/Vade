//! Shared secret-related path names and path validation.
use std::io;
use std::path::{Component, Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SecretPathMatch { Exact, Prefix, Suffix }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SecretPathKind { Credential, Folder, History, Environment }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SecretPathRule {
    pub(crate) value: &'static str,
    pub(crate) match_kind: SecretPathMatch,
    kind: SecretPathKind,
}
const fn secret(value: &'static str, match_kind: SecretPathMatch, kind: SecretPathKind) -> SecretPathRule {
    SecretPathRule { value, match_kind, kind }
}
impl SecretPathRule {
    pub(crate) fn matches(self, lower: &str) -> bool {
        match self.match_kind {
            SecretPathMatch::Exact => lower == self.value,
            SecretPathMatch::Prefix => lower.starts_with(self.value),
            SecretPathMatch::Suffix => lower.ends_with(self.value),
        }
    }
    pub(crate) fn matches_attached(self, lower: &str) -> bool {
        match self.match_kind {
            SecretPathMatch::Exact | SecretPathMatch::Suffix => lower.ends_with(self.value),
            SecretPathMatch::Prefix => lower.contains(self.value),
        }
    }
    #[cfg(test)]
    pub(crate) fn is_folder(self) -> bool {
        self.kind == SecretPathKind::Folder
    }
}
use SecretPathKind::{Credential, Environment, Folder, History};
use SecretPathMatch::{Exact, Prefix, Suffix};

/// One source for direct read approval, child write scope, shell checks,
/// fixed-command checks, Git exclusions, and OS sandbox denial. Exact and
/// prefix rules match one path component. The shell boundary promises this:
/// protected file content needs the person's yes; entry names inside a
/// protected folder need the person's yes; the protected file or folder name
/// itself can appear when a command lists its ordinary parent or reports an
/// access error. Native workspace listing still hides these names as a user
/// convenience. That filter is not a secrecy boundary.
pub(crate) const SECRET_PATH_RULES: &[SecretPathRule] = &[
    secret("credential", Exact, Credential), secret("credentials", Exact, Credential),
    secret("credential.json", Exact, Credential), secret("credential.yaml", Exact, Credential),
    secret("credential.yml", Exact, Credential), secret("credential.toml", Exact, Credential),
    secret("credentials.json", Exact, Credential), secret("credentials.toml", Exact, Credential),
    secret("credentials.yaml", Exact, Credential), secret("credentials.yml", Exact, Credential),
    secret("secrets", Exact, Credential), secret("secrets.json", Exact, Credential),
    secret("secrets.yaml", Exact, Credential), secret("secrets.yml", Exact, Credential),
    secret("id_rsa", Exact, Credential), secret("id_dsa", Exact, Credential),
    secret("id_ecdsa", Exact, Credential), secret("id_ed25519", Exact, Credential),
    secret("authorized_keys", Exact, Credential), secret("known_hosts", Exact, Credential),
    secret("netrc", Exact, Credential), secret(".netrc", Exact, Credential),
    secret(".npmrc", Exact, Credential), secret(".pypirc", Exact, Credential),
    secret(".git-credentials", Exact, Credential), secret(".vault-token", Exact, Credential),
    secret("service-account.json", Exact, Credential), secret("service_account.json", Exact, Credential),
    secret("token", Exact, Credential), secret("token.txt", Exact, Credential),
    secret("token.json", Exact, Credential), secret("token.yaml", Exact, Credential),
    secret("token.yml", Exact, Credential), secret("token.toml", Exact, Credential),
    secret("tokens.json", Exact, Credential), secret("access-token", Exact, Credential),
    secret("access-token.txt", Exact, Credential), secret("access-token.json", Exact, Credential),
    secret("refresh-token", Exact, Credential), secret("refresh-token.txt", Exact, Credential),
    secret("refresh-token.json", Exact, Credential), secret("api-token", Exact, Credential),
    secret("api-token.txt", Exact, Credential), secret("api-token.json", Exact, Credential),
    secret("keys", Exact, Credential),
    secret(".pem", Suffix, Credential), secret(".key", Suffix, Credential),
    secret(".p12", Suffix, Credential), secret(".pfx", Suffix, Credential),
    secret(".keystore", Suffix, Credential), secret(".p8", Suffix, Credential),
    secret(".jks", Suffix, Credential), secret(".tfstate", Suffix, Credential),
    secret(".tfstate.backup", Suffix, Credential),
    secret(".ssh", Exact, Folder), secret(".gnupg", Exact, Folder), secret(".aws", Exact, Folder),
    secret(".azure", Exact, Folder), secret(".docker", Exact, Folder), secret(".kube", Exact, Folder),
    secret(".cargo", Exact, Folder), secret(".npm", Exact, Folder), secret(".gem", Exact, Folder),
    secret(".password-store", Exact, Folder), secret(".terraform.d", Exact, Folder),
    secret("keychains", Exact, Folder), secret("keyrings", Exact, Folder), secret("gcloud", Exact, Folder),
    secret(".gcloud", Exact, Folder),
    secret(".bash_history", Exact, History), secret(".zsh_history", Exact, History),
    secret(".sh_history", Exact, History), secret(".python_history", Exact, History),
    secret(".node_repl_history", Exact, History), secret("fish_history", Exact, History),
    secret(".env", Exact, Environment), secret(".env.", Prefix, Environment),
];

/// The kind of private data named by one path component.
pub(crate) fn secret_component_reason(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let rule = SECRET_PATH_RULES.iter().find(|rule| rule.matches(&lower))?;
    Some(match rule.kind {
        Environment => "an environment file can hold passwords or tokens",
        History => "shell history can hold pasted passwords, tokens or private commands",
        Folder => "this folder commonly holds keys or service credentials",
        Credential => "this filename commonly holds a key, token or credential",
    })
}

pub(crate) fn secret_path_reason(path: &str) -> Option<&'static str> {
    path.split('/').find_map(secret_component_reason)
}

pub(crate) fn allowed_component(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    name.is_ascii()
        && !name.starts_with('.')
        && !name.chars().any(char::is_control)
        && secret_component_reason(&lower).is_none()
}

pub(crate) fn safe_parts(path: &str) -> io::Result<Vec<&str>> {
    checked_parts(path, allowed_component)
}

pub(crate) fn approved_secret_read_component(name: &str) -> bool {
    name.is_ascii() && !name.chars().any(char::is_control)
}

pub(crate) fn approved_secret_read_parts(path: &str) -> io::Result<Vec<&str>> {
    checked_parts(path, approved_secret_read_component)
}

pub(crate) fn validate_approved_secret_path(path: &str) -> io::Result<()> {
    approved_secret_read_parts(path).map(|_| ())
}

fn checked_parts(path: &str, allowed: fn(&str) -> bool) -> io::Result<Vec<&str>> {
    if path.len() > 4096 || path.contains('\\') || path.chars().any(char::is_control) {
        return Err(io::Error::other("This is not a valid path inside the workspace. Use a relative path with forward slashes."));
    }
    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => (),
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| io::Error::other("This path is not valid UTF-8. Use a UTF-8 path."))?;
                if !allowed(value) {
                    return Err(io::Error::other("The agent does not open non-ASCII, hidden or credential-related paths."));
                }
                parts.push(value);
            }
            _ => return Err(io::Error::other("Use a path relative to the workspace, without .. or a leading /. The agent cannot reach files outside the workspace.")),
        }
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const LIST_ALLOW_MARKER: &str = "vade-secret-list-allow:";

    struct Literal { offset: usize, line: usize, value: String }

    fn decode_string(value: &str) -> Option<String> {
        let mut chars = value.chars().peekable();
        let mut decoded = String::new();
        while let Some(ch) = chars.next() {
            if ch != '\\' { decoded.push(ch); continue; }
            match chars.next()? {
                '\\' => decoded.push('\\'), '"' => decoded.push('"'), '\'' => decoded.push('\''),
                'n' => decoded.push('\n'), 'r' => decoded.push('\r'), 't' => decoded.push('\t'), '0' => decoded.push('\0'),
                'x' => {
                    let high = chars.next()?.to_digit(16)?;
                    let low = chars.next()?.to_digit(16)?;
                    decoded.push(char::from_u32(high * 16 + low)?);
                }
                'u' => {
                    if chars.next()? != '{' { return None; }
                    let mut number = String::new();
                    loop {
                        match chars.next()? {
                            '}' => break,
                            '_' => (),
                            digit if digit.is_ascii_hexdigit() => number.push(digit),
                            _ => return None,
                        }
                    }
                    decoded.push(char::from_u32(u32::from_str_radix(&number, 16).ok()?)?);
                }
                '\n' => while chars.peek().is_some_and(|next| next.is_whitespace()) { chars.next(); },
                '\r' => {
                    if chars.next()? != '\n' { return None; }
                    while chars.peek().is_some_and(|next| next.is_whitespace()) { chars.next(); }
                }
                _ => return None,
            }
        }
        Some(decoded)
    }

    fn lex_source(source: &str, label: &str) -> (Vec<u8>, Vec<Literal>, BTreeSet<usize>) {
        let bytes = source.as_bytes();
        let mut code = bytes.to_vec();
        let mut literals = Vec::new();
        let mut marked = BTreeSet::new();
        let mut i = 0;
        let mut line = 1;
        while i < bytes.len() {
            if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'/') {
                let start = i;
                let end = bytes[i..].iter().position(|byte| *byte == b'\n').map_or(bytes.len(), |at| i + at);
                let comment = &source[i + 2..end];
                if let Some(at) = comment.find(LIST_ALLOW_MARKER) {
                    let reason = comment[at + LIST_ALLOW_MARKER.len()..].trim();
                    assert!(!reason.is_empty(), "secret-list scan marker needs a reason at {label}:{line}");
                    let whole_line = source[..end].rsplit_once('\n').map_or(&source[..end], |(_, value)| value);
                    eprintln!("secret-list scan marker {label}:{line}: {}", whole_line.trim());
                    marked.insert(line);
                }
                code[start..end].fill(b' ');
                i = end;
                continue;
            }
            if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                let start = i;
                let mut depth = 1;
                i += 2;
                while i < bytes.len() && depth > 0 {
                    if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') { depth += 1; i += 2; }
                    else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') { depth -= 1; i += 2; }
                    else { if bytes[i] == b'\n' { line += 1; } i += 1; }
                }
                for (at, byte) in code[start..i].iter_mut().enumerate() {
                    if bytes[start + at] != b'\n' { *byte = b' '; }
                }
                continue;
            }
            let raw = if bytes[i] == b'r' { Some(i + 1) }
                else if bytes[i] == b'b' && bytes.get(i + 1) == Some(&b'r') { Some(i + 2) }
                else { None };
            if let Some(mut quote) = raw {
                let mut hashes = 0;
                while bytes.get(quote) == Some(&b'#') { hashes += 1; quote += 1; }
                if bytes.get(quote) == Some(&b'"') {
                    let start = i;
                    let value_start = quote + 1;
                    i = value_start;
                    let mut value_end = None;
                    while i < bytes.len() {
                        if bytes[i] == b'"' && bytes.get(i + 1..i + 1 + hashes).is_some_and(|tail| tail.iter().all(|byte| *byte == b'#')) {
                            value_end = Some(i);
                            i += hashes + 1;
                            break;
                        }
                        if bytes[i] == b'\n' { line += 1; }
                        i += 1;
                    }
                    if let Some(end) = value_end {
                        literals.push(Literal { offset: start, line: line - source[value_start..end].bytes().filter(|byte| *byte == b'\n').count(), value: source[value_start..end].to_owned() });
                    }
                    for (at, byte) in code[start..i].iter_mut().enumerate() {
                        if bytes[start + at] != b'\n' { *byte = b' '; }
                    }
                    continue;
                }
            }
            if bytes[i] == b'"' {
                let start = i;
                let start_line = line;
                i += 1;
                let value_start = i;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                        if i < bytes.len() && bytes[i] == b'\n' { line += 1; }
                    } else if bytes[i] == b'\n' { line += 1; }
                    i += 1;
                }
                let value_end = i;
                if i < bytes.len() { i += 1; }
                if value_end < bytes.len() {
                    if let Some(value) = decode_string(&source[value_start..value_end]) {
                        literals.push(Literal { offset: start, line: start_line, value });
                    }
                }
                for (at, byte) in code[start..i].iter_mut().enumerate() {
                    if bytes[start + at] != b'\n' { *byte = b' '; }
                }
                continue;
            }
            if bytes[i] == b'\'' {
                let start = i;
                let mut end = i + 1;
                if bytes.get(end) == Some(&b'\\') { end += 2; } else { end += 1; }
                if bytes.get(end) == Some(&b'\'') {
                    end += 1;
                    code[start..end].fill(b' ');
                    i = end;
                    continue;
                }
            }
            if bytes[i] == b'\n' { line += 1; }
            i += 1;
        }
        (code, literals, marked)
    }

    fn test_only_ranges(code: &[u8]) -> Vec<(usize, usize)> {
        const ATTRIBUTE: &[u8] = b"#[cfg(test)]";
        let mut ranges = Vec::new();
        let mut search = 0;
        while search + ATTRIBUTE.len() <= code.len() {
            let Some(relative) = code[search..].windows(ATTRIBUTE.len()).position(|window| window == ATTRIBUTE) else { break };
            let start = search + relative;
            let mut i = start + ATTRIBUTE.len();
            let (mut parentheses, mut brackets) = (0usize, 0usize);
            while i < code.len() {
                match code[i] {
                    b'(' => parentheses += 1,
                    b')' => parentheses = parentheses.saturating_sub(1),
                    b'[' => brackets += 1,
                    b']' => brackets = brackets.saturating_sub(1),
                    b';' if parentheses == 0 && brackets == 0 => { i += 1; break; }
                    b'{' if parentheses == 0 && brackets == 0 => {
                        let mut depth = 1;
                        i += 1;
                        while i < code.len() && depth > 0 {
                            if code[i] == b'{' { depth += 1; }
                            else if code[i] == b'}' { depth -= 1; }
                            i += 1;
                        }
                        break;
                    }
                    _ => (),
                }
                i += 1;
            }
            ranges.push((start, i));
            search = i.max(start + ATTRIBUTE.len());
        }
        ranges
    }

    fn copied_secret_names(source: &str, label: &str) -> BTreeSet<String> {
        let (code, literals, marked) = lex_source(source, label);
        let test_ranges = test_only_ranges(&code);
        literals.into_iter().filter(|literal| !marked.contains(&literal.line)
            && !test_ranges.iter().any(|(start, end)| literal.offset >= *start && literal.offset < *end))
            .filter(|literal| {
                // Only list entries count. Ordinary protocol lookups, such as
                // v.get("token") and v.get("keys"), do not make a name list.
                // An index such as protocol["token"] is a lookup too: its bracket follows a name,
                // a closing bracket or a closing parenthesis. Only a bracket that opens a list counts.
                // After a keyword that starts an expression (return ["x"], &mut ["x"]) the bracket opens a list.
                let (mut open, mut last, mut word, mut in_word) = (Vec::new(), b' ', Vec::<u8>::new(), false);
                for &byte in &code[..literal.offset] {
                    let name_byte = byte.is_ascii_alphanumeric() || byte == b'_';
                    match byte {
                        b'[' => {
                            let keyword = matches!(word.as_slice(), b"return" | b"break" | b"mut" | b"in" | b"else" | b"match" | b"if" | b"while" | b"yield" | b"move" | b"ref");
                            let after_name = last.is_ascii_alphanumeric() || last == b'_';
                            open.push(if after_name { keyword } else { !matches!(last, b')' | b']' | b'?') });
                        }
                        b']' => { open.pop(); }
                        _ => (),
                    }
                    if name_byte { if !in_word { word.clear(); } word.push(byte); }
                    in_word = name_byte;
                    if !byte.is_ascii_whitespace() { last = byte; if !name_byte { word.clear(); } }
                }
                open.contains(&true)
            })
            .filter_map(|literal| SECRET_PATH_RULES.iter().find(|rule| rule.value.eq_ignore_ascii_case(&literal.value)).map(|rule| rule.value.to_owned()))
            .collect()
    }

    fn test_fixture(path: &Path) -> bool {
        let name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default();
        name == "tests.rs" || name == "test_support.rs" || name == "test_workspace.rs"
            || name.ends_with("_tests.rs") || path.components().any(|part| matches!(part.as_os_str().to_str(), Some("tests" | "fixtures")))
    }

    #[test]
    fn data_driven_scan_rejects_both_reader_spellings() {
        let first = r#"if in_scope
            || ["token.txt", "gcloud"].contains(&part)
            || crate::secret_paths::secret_component_reason(part).is_some() {}"#;
        let second = r#"if crate::secret_paths::secret_component_reason(part).is_some()
|| ["token.txt", "gcloud"].contains(&part) {}"#;
        let expected = BTreeSet::from(["gcloud".to_owned(), "token.txt".to_owned()]);
        assert_eq!(copied_secret_names(first, "reader-one fixture"), expected);
        assert_eq!(copied_secret_names(second, "reader-two fixture"), expected);
        assert_eq!(copied_secret_names(r#"let names = ["token\x2etxt", "gcloud"];"#, "escaped fixture"), expected);
    }

    #[test]
    fn data_driven_scan_accepts_one_incidental_name() {
        let one_name = r#"let service = "gcloud";"#;
        assert!(copied_secret_names(one_name, "one-name fixture").is_empty());
        let marked = r#"let names = ["token.txt", "gcloud"]; // vade-secret-list-allow: required protocol names"#;
        assert!(copied_secret_names(marked, "marked fixture").is_empty());
    }

    #[test]
    fn scan_handles_renamed_raw_lists_comments_and_test_fixtures() {
        let renamed = r##"const PRIVATE_COMPONENTS: &[&str] = &[r#"token.txt"#, "gcloud"];"##;
        assert_eq!(copied_secret_names(renamed, "renamed fixture").len(), 2);
        let fixtures = r##"// "token.txt" and "gcloud"
/* outer /* "token.txt" */ "gcloud" */
#[cfg(test)] mod tests { let _ = ["token.txt", "gcloud"]; }
const ORDINARY: &str = "ordinary";"##;
        assert!(copied_secret_names(fixtures, "test and comment fixture").is_empty());
    }

    #[test]
    fn single_copied_name_is_rejected_but_protocol_lookups_are_allowed() {
        for name in ["credentials", ".pem", "token"] {
            let source = format!("const PRIVATE: &[&str] = &[{name:?}];");
            assert_eq!(copied_secret_names(&source, "single list entry"), BTreeSet::from([name.to_owned()]));
        }
        assert!(copied_secret_names(r#"let token = v.get("token"); let keys = v.get("keys");"#, "protocol lookups").is_empty());
        assert!(copied_secret_names(r#"let token = protocol["token"]; let keys = table.rows[0]["keys"];"#, "map lookups").is_empty());
        assert_eq!(copied_secret_names(r#"let names = vec!["token"]; check(&["keys"]);"#, "lists after a macro and a call").len(), 2);
        for list in [r#"fn names() -> [&'static str; 1] { return ["token"]; }"#, r#"let names = loop { break ["keys"]; };"#, r#"let names = &mut ["token"];"#] {
            assert_eq!(copied_secret_names(list, "list after a keyword").len(), 1, "{list}");
        }
    }

    #[test]
    fn production_source_has_no_old_secret_name_list_shape() {
        fn collect(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect(&path, files);
                } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                    files.push(path);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect(&root, &mut files);
        let old_names = [
            ["CREDENTIAL", "_NAMES"].concat(),
            ["CREDENTIAL", "_SUFFIXES"].concat(),
            ["SECRET_FOLDER", "_NAMES"].concat(),
            ["SHELL_HISTORY", "_NAMES"].concat(),
            ["SECRET", "_DIRS"].concat(),
        ];
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            for old in &old_names {
                assert!(!text.contains(old), "old secret-name list shape {old} in {}", path.display());
            }
        }
        let child = std::fs::read_to_string(root.join("child_agents/policy.rs")).unwrap();
        for old in [["\"creden", "tials\""].concat(), ["\".p", "em\""].concat()] {
            assert!(!child.contains(&old), "copied secret-name literal in child write policy: {old}");
        }
    }

    #[test]
    fn production_source_has_no_copied_secret_name_lists() {
        fn collect(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect(&path, files);
                } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                    files.push(path);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect(&root, &mut files);
        files.sort();
        let table = root.join("secret_paths.rs");
        let mut violations = Vec::new();
        for path in files {
            if path == table || test_fixture(&path) { continue; }
            let text = std::fs::read_to_string(&path).unwrap();
            // Retain the old identifier gate as well as the data-driven gate.
            let (mut code, _, _) = lex_source(&text, &path.display().to_string());
            for (start, end) in test_only_ranges(&code) { code[start..end].fill(b' '); }
            let production = String::from_utf8(code).unwrap();
            for old in [["CREDENTIAL", "_NAMES"].concat(), ["CREDENTIAL", "_SUFFIXES"].concat(),
                ["SECRET_FOLDER", "_NAMES"].concat(), ["SHELL_HISTORY", "_NAMES"].concat(), ["SECRET", "_DIRS"].concat()] {
                if production.contains(&old) { violations.push(format!("{}: old secret-name list shape {old}", path.display())); }
            }
            let names = copied_secret_names(&text, &path.display().to_string());
            if !names.is_empty() {
                violations.push(format!("{}: {}", path.display(), names.into_iter().collect::<Vec<_>>().join(", ")));
            }
        }
        // This is a literal scan. Names assembled from pieces stay outside it.
        // An intentional line can use LIST_ALLOW_MARKER with a reason. The scan
        // prints every such line so that a reviewer sees all exceptions.
        assert!(violations.is_empty(), "Rust source has a second protected-name list:\n{}", violations.join("\n"));
    }
}
