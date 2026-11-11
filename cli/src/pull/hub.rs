//! The Hugging Face Hub side of `vade pull`: names, URLs, the file listing and
//! the file selection. Nothing here touches the network or the disk.
use crate::evals::json::{self, Value};

/// The public Hub. `HF_ENDPOINT` names a mirror instead.
pub(super) const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The largest file a listing may name (16 TiB). With [`MAX_FILES`], every byte
/// total fits in an `i64`.
pub(super) const MAX_FILE: u64 = 1 << 44;
/// The most files a listing may name.
pub(super) const MAX_FILES: usize = 100_000;

/// One file in the repository listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Remote {
    pub path: String,
    pub size: u64,
    /// The Git blob ID (SHA-1 of `blob <size>\0` and the bytes).
    pub git: String,
    /// The SHA-256 the Hub stores for a large (LFS) file.
    pub sha256: Option<String>,
}

/// Check `org/name`: two parts of letters, digits, `.`, `_` or `-`.
pub(super) fn repo(value: &str) -> Result<(), String> {
    let part = |part: &str| (1..=96).contains(&part.len()) && !part.starts_with(['.', '-']) && !part.contains("..")
        && part.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    match value.split_once('/') {
        Some((org, name)) if part(org) && part(name) => Ok(()),
        _ => Err(format!("Give the model as org/name, such as nvidia/Qwen3.8-Flash-Next-FP4-Local, not {}.", crate::text::argument(value))),
    }
}

/// Check a branch, tag or commit name.
pub(super) fn revision(value: &str) -> Result<(), String> {
    if (1..=200).contains(&value.len()) && !value.starts_with(['/', '-', '.']) && !value.ends_with('/')
        && !value.contains("..") && !value.contains("//")
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/')) {
        Ok(())
    } else {
        Err(format!("Give the revision as a branch, tag or commit hash, not {}.", crate::text::argument(value)))
    }
}

/// Check `HF_ENDPOINT`: an HTTPS origin, or plain HTTP on this machine only.
pub(super) fn endpoint(value: &str) -> Result<String, String> {
    let value = value.trim_end_matches('/');
    let host = value.strip_prefix("https://").or_else(|| value.strip_prefix("http://")
        .filter(|rest| ["127.0.0.1", "localhost", "[::1]"].iter().any(|local| *rest == *local || rest.starts_with(&format!("{local}:")))));
    match host {
        Some(host) if !host.is_empty() && host.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':' | b'[' | b']')) => Ok(value.into()),
        _ => Err("Set HF_ENDPOINT to an https:// address with no path, or unset it to use huggingface.co.".into()),
    }
}

/// Percent-encode a URL path. `slash` keeps `/` as a separator.
pub(super) fn encode(value: &str, slash: bool) -> String {
    value.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) || (slash && b == b'/') {
        (b as char).to_string()
    } else {
        format!("%{b:02X}")
    }).collect()
}

pub(super) fn revision_url(endpoint: &str, repo: &str, revision: &str) -> String {
    format!("{endpoint}/api/models/{}/revision/{}", encode(repo, true), encode(revision, false))
}

pub(super) fn tree_url(endpoint: &str, repo: &str, commit: &str) -> String {
    format!("{endpoint}/api/models/{}/tree/{commit}?recursive=true", encode(repo, true))
}

pub(super) fn file_url(endpoint: &str, repo: &str, commit: &str, path: &str) -> String {
    format!("{endpoint}/{}/resolve/{commit}/{}", encode(repo, true), encode(path, true))
}

/// The commit a revision names, from the revision API reply.
pub(super) fn commit(reply: &str) -> Result<String, String> {
    let value = json::parse(reply).map_err(|_| "The Hub sent a revision reply that is not JSON.")?;
    match value["sha"].as_str() {
        Some(sha) if crate::text::is_lower_hex(sha, 40) => Ok(sha.into()),
        _ => Err("The Hub revision reply has no commit hash.".into()),
    }
}

/// A relative path with plain parts only, safe to join under the model folder.
pub(super) fn safe_path(path: &str) -> bool {
    !path.is_empty() && path.len() <= 1024 && !path.chars().any(char::is_control) && !path.contains('\\')
        && path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

/// The files on one page of the recursive tree listing.
pub(super) fn tree_page(reply: &str) -> Result<Vec<Remote>, String> {
    let bad = |what: &str| format!("The Hub file listing has {what}.");
    let value = json::parse(reply).map_err(|_| bad("data that is not JSON"))?;
    let mut files = Vec::new();
    for entry in value.as_array().ok_or_else(|| bad("no file list"))? {
        if entry["type"].as_str() != Some("file") { continue; }
        let path = entry["path"].as_str().filter(|path| safe_path(path)).ok_or_else(|| bad("an unsafe file name"))?;
        let size = entry["size"].as_i64().and_then(|size| u64::try_from(size).ok()).filter(|size| *size <= MAX_FILE).ok_or_else(|| bad("a file without a size, or one over 16 TiB"))?;
        let git = entry["oid"].as_str().filter(|oid| crate::text::is_lower_hex(oid, 40)).ok_or_else(|| bad("a file without a Git ID"))?;
        let sha256 = match &entry["lfs"] {
            Value::Null => None,
            lfs => {
                let sha = lfs["oid"].as_str().filter(|oid| crate::text::is_lower_hex(oid, 64)).ok_or_else(|| bad("a large file without a SHA-256"))?;
                if lfs["size"].as_i64() != i64::try_from(size).ok() { return Err(bad("a large file with two sizes")); }
                Some(sha.to_owned())
            }
        };
        files.push(Remote { path: path.into(), size, git: git.into(), sha256 });
    }
    Ok(files)
}

/// The next page from the `Link` response headers, or `None` on the last page.
/// A next page must stay on the same API; any other next link, or a `Link`
/// header that does not parse, is an error, so a listing never ends early in
/// silence.
pub(super) fn next_page(headers: &str, endpoint: &str) -> Result<Option<String>, String> {
    let bad = || "The Hub file listing has a next page that pull cannot follow (another address or a malformed link), so the list would be incomplete. Check HF_ENDPOINT, then run vade pull again.".to_owned();
    let mut next = None;
    for value in headers.lines().filter_map(|line| line.split_once(':').filter(|(name, _)| name.trim().eq_ignore_ascii_case("link")).map(|(_, value)| value)) {
        for (url, rel) in links(value).ok_or_else(bad)? {
            if !rel.split_ascii_whitespace().any(|kind| kind.eq_ignore_ascii_case("next")) { continue; }
            if next.is_some() || !url.starts_with(&format!("{endpoint}/api/models/")) || url.chars().any(|c| c.is_control() || c == ' ') { return Err(bad()); }
            next = Some(url);
        }
    }
    Ok(next)
}

/// The (target, rel) pairs of one `Link` header value (RFC 8288): `<url>`
/// then `; name=value` parameters, where a value is a token or a quoted
/// string; link values are separated by commas. `None` when it does not parse.
fn links(value: &str) -> Option<Vec<(String, String)>> {
    let mut rest = value.trim_start();
    let mut found = Vec::new();
    while !rest.is_empty() {
        let (url, tail) = rest.strip_prefix('<')?.split_once('>')?;
        let (mut tail, mut rel) = (tail.trim_start(), String::new());
        while let Some(param) = tail.strip_prefix(';') {
            let param = param.trim_start();
            let end = param.find(['=', ';', ',']).unwrap_or(param.len());
            let name = param[..end].trim();
            tail = param[end..].trim_start();
            let mut text = String::new();
            if let Some(after) = tail.strip_prefix('=') {
                let after = after.trim_start();
                if let Some(quoted) = after.strip_prefix('"') {
                    let mut chars = quoted.char_indices();
                    let close = loop {
                        match chars.next()? {
                            (_, '\\') => text.push(chars.next()?.1),
                            (index, '"') => break index,
                            (_, c) => text.push(c),
                        }
                    };
                    tail = quoted[close + 1..].trim_start();
                } else {
                    let end = after.find([';', ',']).unwrap_or(after.len());
                    text = after[..end].trim().to_owned();
                    tail = after[end..].trim_start();
                }
            }
            if name.eq_ignore_ascii_case("rel") && rel.is_empty() { rel = text; }
        }
        found.push((url.to_owned(), rel));
        rest = match tail.strip_prefix(',') { Some(more) => more.trim_start(), None if tail.is_empty() => "", None => return None };
    }
    Some(found)
}

/// The files Vade needs to serve a checkpoint: weights and their index, the
/// configs, tokenizer files, chat template, README and license.
pub(super) fn wanted(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    [".safetensors", ".json", ".jinja", ".tiktoken"].iter().any(|end| name.ends_with(end))
        || matches!(name.as_str(), "merges.txt" | "tokenizer.model" | "readme.md" | "license" | "license.md" | "license.txt" | "notice" | "notice.md")
}

/// The selected files: `--include` patterns when given, else [`wanted`]. No
/// pattern selects a path with a part that starts with a dot, so a listing
/// cannot reach the pull state or the manifest. Small files come first, so
/// the configs land before the weights.
pub(super) fn select(files: &[Remote], include: &[String]) -> Vec<Remote> {
    let mut chosen: Vec<Remote> = files.iter().filter(|file| !hidden(&file.path)).filter(|file| if include.is_empty() { wanted(&file.path) }
        else { include.iter().any(|pattern| glob(pattern, &file.path)) }).cloned().collect();
    chosen.sort_by(|a, b| a.path.ends_with(".safetensors").cmp(&b.path.ends_with(".safetensors")).then_with(|| a.path.cmp(&b.path)));
    chosen
}

/// Whether a part of `path` starts with a dot.
pub(super) fn hidden(path: &str) -> bool { path.split('/').any(|part| part.starts_with('.')) }

/// Shell-style match: `*` is any text, `/` included, and `?` is one character.
pub(super) fn glob(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut i, mut j, mut star, mut mark) = (0, 0, None, 0);
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) { i += 1; j += 1; }
        else if i < p.len() && p[i] == '*' { star = Some(i); mark = j; i += 1; }
        else if let Some(s) = star { i = s + 1; mark += 1; j = mark; }
        else { return false; }
    }
    p[i..].iter().all(|c| *c == '*')
}

/// The license in a model card's YAML front matter: `license_name` when
/// `license` is `other`, else `license`.
pub(super) fn license(readme: &str) -> Option<String> {
    let mut lines = readme.lines();
    if lines.next()?.trim_end() != "---" { return None; }
    let front: Vec<&str> = lines.take_while(|line| line.trim_end() != "---").collect();
    let field = |key: &str| front.iter().find_map(|line| line.strip_prefix(key)).map(|value| value.trim().trim_matches(['"', '\'']))
        .filter(|value| !value.is_empty() && value.len() <= 200 && !value.chars().any(char::is_control));
    field("license:").filter(|value| *value != "other").or_else(|| field("license_name:")).or_else(|| field("license:")).map(str::to_owned)
}
