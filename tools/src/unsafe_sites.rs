//! Source-site check for the unsafe rule. This does not expand macros or run GPU code.
#![forbid(unsafe_code)]
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path,process::Command};
use crate::Result;

#[derive(Debug)]
struct Token<'a> { text: &'a str, start: usize, literal: bool }
#[derive(Debug)]
struct Comment { start: usize, end: usize }
struct Lexed<'a> { tokens: Vec<Token<'a>>, comments: Vec<Comment>, pairs: Vec<usize> }
fn ident(b: u8) -> bool { b.is_ascii_alphanumeric() || b == b'_' || b >= 128 }
fn lex(source: &str) -> Result<Lexed<'_>> {
    let b = source.as_bytes(); let (mut i,mut tokens,mut comments) = (0,Vec::new(),Vec::new());
    while i < b.len() {
        if b[i].is_ascii_whitespace() { i += 1; continue; }
        let start = i;
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' { i += 1; }
            comments.push(Comment { start,end:i }); continue;
        }
        if b[i..].starts_with(b"/*") {
            i += 2; let mut depth = 1;
            while i < b.len() && depth > 0 {
                if b[i..].starts_with(b"/*") { depth += 1; i += 2; }
                else if b[i..].starts_with(b"*/") { depth -= 1; i += 2; }
                else { i += 1; }
            }
            if depth != 0 { return Err("unclosed source comment".into()); }
            comments.push(Comment { start,end:i }); continue;
        }
        let mut quote = i;
        if matches!(b[i],b'b'|b'c') { quote += 1; }
        let raw = b.get(quote) == Some(&b'r');
        if raw { quote += 1; while b.get(quote) == Some(&b'#') { quote += 1; } }
        let mut literal = false;
        if b.get(quote) == Some(&b'"') {
            literal = true;
            let hashes = if raw { quote-start-1-usize::from(b[start] != b'r') } else { 0 };
            i = quote+1; let mut closed = false;
            while i < b.len() {
                if b[i] == b'"' && (!raw || b.get(i+1..i+1+hashes).is_some_and(|s| s.iter().all(|&c| c == b'#'))) {
                    i += 1+hashes; closed = true; break;
                }
                if !raw && b[i] == b'\\' { i += 1; }
                i += 1;
            }
            if !closed { return Err("unclosed source string".into()); }
        } else {
            let char_start = i+usize::from(b[i] == b'b');
            if b.get(char_start) == Some(&b'\'') {
                let value = char_start+1;
                let mut end = value;
                if b.get(value) == Some(&b'\\') {
                    end += 2;
                    if b.get(value+1) == Some(&b'u') && b.get(end) == Some(&b'{') {
                        while end < b.len() && b[end] != b'}' { end += 1; } end += 1;
                    } else if b.get(value+1) == Some(&b'x') { end += 2; }
                } else if let Some(ch) = source.get(value..).and_then(|s| s.chars().next()) { end += ch.len_utf8(); }
                if b.get(end) == Some(&b'\'') { i = end+1; literal = true; }
            }
            if !literal {
                if b[i..].starts_with(b"r#") && b.get(i+2).is_some_and(|&c| ident(c)) { i += 2; }
                if b[i..].starts_with(b"->") { i += 2; }
                else if ident(b[i]) { while i < b.len() && ident(b[i]) { i += 1; } }
                else { i += 1; }
            }
        }
        tokens.push(Token { text:&source[start..i],start,literal });
    }
    let mut pairs = vec![usize::MAX;tokens.len()]; let mut stack = Vec::new();
    for (i,t) in tokens.iter().enumerate().filter(|(_,t)| !t.literal) {
        match t.text {
            "("|"["|"{" => stack.push(i),
            ")"|"]"|"}" => {
                let Some(open) = stack.pop() else { return Err("unmatched source delimiter".into()); };
                if !matches!((tokens[open].text,t.text),("(",")")|("[","]")|("{","}")) { return Err("mismatched source delimiter".into()); }
                pairs[open] = i; pairs[i] = open;
            },
            _ => ()
        }
    }
    if !stack.is_empty() { return Err("unclosed source delimiter".into()); }
    Ok(Lexed { tokens,comments,pairs })
}
struct Item { start: usize, header: usize, end: usize, name: String, function: bool }
fn word(t: &Token<'_>) -> bool { !t.literal && t.text.as_bytes().first().is_some_and(|&b| ident(b)) }
// Skip token groups in a signature. Braces in const generic arguments are not bodies.
fn header_end(l: &Lexed<'_>, mut i: usize) -> Option<usize> {
    let mut angle = 0usize;
    while i < l.tokens.len() {
        let t = &l.tokens[i];
        if !t.literal { match t.text {
            "<" => angle += 1,
            ">" if angle > 0 => angle -= 1,
            "{" if angle == 0 => return Some(i),
            ";" => return Some(i),
            "("|"["|"{" => { i = l.pairs[i]+1; continue; },
            "}" => return None,
            _ => ()
        } }
        i += 1;
    } None
}
// An impl name is its trait and type. Generic bounds are not its identity.
fn impl_name(tokens: &[Token<'_>]) -> String {
    let mut name = String::from("impl:"); let mut angle = 0usize;
    for t in &tokens[1..] {
        match t.text {
            "where" if angle == 0 => break,
            "<" => angle += 1,
            ">" if angle > 0 => angle -= 1,
            _ if angle == 0 => { name.push_str(t.text); name.push(' '); },
            _ => ()
        }
    }
    name.trim_end().into()
}
fn items(l: &Lexed<'_>) -> Vec<Item> {
    let t = &l.tokens; let mut out = Vec::new();
    for (i,token) in t.iter().enumerate().filter(|(_,t)| !t.literal) {
        let (name,function,from) = match token.text {
            "const" if t.get(i+1).is_some_and(|t| matches!(t.text,"fn"|"unsafe"|"async")) => continue,
            "static" if t.get(i+1).is_some_and(|t| t.text == "mut") && t.get(i+2).is_some_and(word) =>
                (format!("static:{}",t[i+2].text),false,i+3),
            "fn"|"mod"|"struct"|"enum"|"trait"|"union"|"type"|"const"|"static" if t.get(i+1).is_some_and(word) =>
                (format!("{}:{}",token.text,t[i+1].text),token.text == "fn",i+2),
            "fn" if t.get(i+1).is_some_and(|t| t.text == "$") && t.get(i+2).is_some_and(word) =>
                (format!("fn:${}",t[i+2].text),true,i+3),
            "impl" => (String::new(),false,i+1),
            "macro_rules" if t.get(i+1).is_some_and(|t| t.text == "!") && t.get(i+2).is_some_and(word) =>
                (format!("macro:{}",t[i+2].text),false,i+3),
            "extern" if t.get(i+1).is_some_and(|t| t.literal) && t.get(i+2).is_some_and(|t| t.text == "{") =>
                (format!("extern:{}",t[i+1].text),false,i+2),
            "extern" if t.get(i+1).is_some_and(|t| t.text == "{") => ("extern".into(),false,i+1),
            _ => continue
        };
        let Some(header) = header_end(l,from) else { continue; };
        let end = if t[header].text == "{" { l.pairs[header] } else { header };
        let name = if name.is_empty() { impl_name(&t[i..header]) } else { name };
        out.push(Item { start:i,header,end,name,function });
    }
    // `impl Trait` in a signature is a type, not an implementation item.
    let headers: Vec<_> = out.iter().map(|n| (n.start,n.header)).collect();
    out.retain(|n| !headers.iter().any(|&(start,end)| start < n.start && n.start < end));
    out
}
fn line_start(source: &str, at: usize) -> usize { source[..at].rfind('\n').map_or(0,|i| i+1) }
fn contract(source: &str, at: usize, comments: &[Comment]) -> Option<String> {
    let mut before = line_start(source,at); let mut text = String::new();
    // A contract can span adjacent comment lines. Blank lines and code break it.
    while let Some(c) = comments.iter().rev().find(|c| c.end <= before) {
        let gap = &source[c.end..before];
        if !gap.chars().all(char::is_whitespace) || gap.bytes().filter(|&b| b == b'\n').count() != 1
            || !source[line_start(source,c.start)..c.start].trim().is_empty() { break; }
        text = format!("{}\n{text}",&source[c.start..c.end]); before = line_start(source,c.start);
    }
    let valid = text.split_once("SAFETY:").is_some_and(|(_,s)| !s.trim_matches(|c:char| c.is_whitespace() || matches!(c,'/'|'*')).is_empty());
    valid.then_some(text)
}
#[derive(Debug)]
pub struct Site { pub key: String, path: String, pub item: String, kind: &'static str, line: usize, wrapper: Option<usize>, contract: bool, gpu: bool, kernel: bool }
// Include attributes and visibility before an entry function when finding its note.
fn entry_start(l: &Lexed<'_>, mut i: usize) -> usize {
    while i > 0 {
        match l.tokens[i-1].text {
            "pub"|"const"|"async" => i -= 1,
            ")" if l.pairs[i-1] > 0 && l.tokens[l.pairs[i-1]-1].text == "pub" => i = l.pairs[i-1]-1,
            "]" if l.pairs[i-1] > 0 && l.tokens[l.pairs[i-1]-1].text == "#" => i = l.pairs[i-1]-1,
            _ => break
        }
    } l.tokens[i].start
}
pub fn sites(crate_path: &str, path: &str, source: &str, gpu: bool) -> Result<Vec<Site>> {
    let l = lex(source)?; let named = items(&l); let mut out = Vec::new(); let mut counts = BTreeMap::new();
    for (i,t) in l.tokens.iter().enumerate().filter(|(_,t)| !t.literal && t.text == "unsafe") {
        let next = l.tokens.get(i+1).ok_or("unsafe has no following token")?;
        let mut own = i+1;
        let kind = match next.text {
            "{" => "block", "impl" => "impl", "trait" => "trait", "(" => "attribute",
            "fn" => "fn",
            "extern" => {
                own += 1;
                if l.tokens.get(own).is_some_and(|t| t.literal) { own += 1; }
                if l.tokens.get(own).is_some_and(|t| t.text == "fn") { "fn" } else { own = i+1; "extern" }
            },
            _ => return Err(format!("{path}: unsafe form is not supported").into())
        };
        if kind == "attribute" && i >= 2 && l.tokens[i-2].text == "#" && l.tokens[i-1].text == "[" {
            let after = l.pairs[i+1]+2;
            if let Some(n) = named.iter().find(|n| n.start >= after) {
                if !l.tokens[after..n.start].iter().any(|t| matches!(t.text,";"|"{"|"}")) { own = n.start; }
            }
        }
        let mut scope: Vec<_> = named.iter().filter(|n| n.start <= own && own <= n.end).collect();
        scope.sort_by_key(|n| (n.start,std::cmp::Reverse(n.end)));
        let item = if scope.is_empty() { "<file>".into() } else { scope.iter().map(|n| n.name.as_str()).collect::<Vec<_>>().join("::") };
        let wrapper = scope.iter().rev().find(|n| n.function && n.header < i).map(|n| n.start);
        let kernel = matches!(kind,"fn"|"attribute") && own >= 2 && l.tokens[own-1].text == "\"ptx-kernel\"" && l.tokens[own-2].text == "extern"
            && named.iter().any(|n| n.start == own && n.function && l.tokens[n.header].text == "{");
        let contract = contract(source,if kernel { entry_start(&l,if kind == "attribute" { i-2 } else { i }) } else { t.start },&l.comments);
        let ordinal = counts.entry((item.clone(),kind)).or_insert(0usize); *ordinal += 1;
        let file = path.strip_prefix(crate_path).and_then(|p| p.strip_prefix('/')).unwrap_or(path);
        let key = format!("{}\t{}\t{}\t{kind}:{}",crate_path.escape_default(),file.escape_default(),item.escape_default(),ordinal);
        out.push(Site { key,path:path.into(),item,kind,line:source[..t.start].bytes().filter(|&b| b == b'\n').count()+1,
            wrapper,contract:contract.is_some(),gpu,kernel });
    } Ok(out)
}
pub type Allowlist = BTreeMap<String,String>;
const FROZEN_RULING: &str = "frozen ruling 2026-11-02 11:54";
pub fn ruling(text: &str) -> Result<()> {
    let b = text.as_bytes();
    if b.len() != 23 || !b.starts_with(b"ruling ") || b[11] != b'-' || b[14] != b'-' || b[17] != b' ' || b[20] != b':'
        || b.iter().enumerate().any(|(i,c)| !matches!(i,0..=6|11|14|17|20) && !c.is_ascii_digit())
        || !(1..=12).contains(&text[12..14].parse::<u8>()?) || !(1..=31).contains(&text[15..17].parse::<u8>()?)
        || text[18..20].parse::<u8>()? > 23 || text[21..23].parse::<u8>()? > 59 {
        return Err("name the orchestrator ruling as: ruling YYYY-MM-DD HH:MM".into());
    } Ok(())
}
pub fn allowlist(text: &str) -> Result<Allowlist> {
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|s| !s.is_empty() && !s.starts_with('#')) {
        let fields: Vec<_> = line.split('\t').collect();
        if !matches!(fields.len(),4|5) || fields[..2].iter().any(|p| p.is_empty() || p.starts_with('/') || p.split('/').any(|s| matches!(s,".."|".")))
            || !fields[1].ends_with(".rs") || fields[2].is_empty() || !fields[3].split_once(':').is_some_and(|(kind,n)|
                matches!(kind,"block"|"fn"|"impl"|"extern"|"trait"|"attribute") && n.parse::<usize>().is_ok_and(|n| n > 0)) {
            return Err("invalid unsafe allowlist entry".into());
        }
        let approval = fields.get(4).copied().unwrap_or("");
        if approval.starts_with("frozen ") {
            if approval != FROZEN_RULING { return Err(format!("frozen unsafe allowlist entries must use: {FROZEN_RULING}").into()); }
        } else if !approval.is_empty() { ruling(approval)?; }
        if out.insert(fields[..4].join("\t"),approval.into()).is_some() { return Err("duplicate unsafe allowlist entry".into()); }
    } Ok(out)
}
pub fn violations(sites: &[Site], allowed: &Allowlist) -> Vec<String> {
    let mut remaining: BTreeSet<_> = allowed.keys().cloned().collect(); let mut issues = Vec::new(); let mut blocks = BTreeMap::new();
    for site in sites { remaining.remove(&site.key); }
    let block_key = |key: &str| key.rsplit_once('\t').is_some_and(|(_,kind)| kind.starts_with("block:"));
    let mut block_counts = BTreeMap::<&str,(usize,usize)>::new();
    for site in sites.iter().filter(|site| site.kind == "block") {
        block_counts.entry(site.key.rsplit_once('\t').unwrap().0).or_default().0 += 1;
    }
    for (key,approval) in allowed.iter().filter(|(key,_)| block_key(key)) {
        if approval == FROZEN_RULING { block_counts.entry(key.rsplit_once('\t').unwrap().0).or_default().1 += 1; }
    }
    // An inserted block can take a frozen ordinal. Check all block notes when an item grows.
    let grown_items: BTreeSet<_> = block_counts.into_iter().filter(|(_,counts)| counts.0 > counts.1).map(|(item,_)| item).collect();
    let stale_blocks = remaining.iter().filter(|key| block_key(key)).count();
    let new_blocks = sites.iter().filter(|site| site.kind == "block" && !allowed.contains_key(&site.key)).count();
    // Keys do not bind bodies. Equal missing and stale block counts can mean a move.
    // Keep the check failing until the allowlist diff has been reviewed.
    let moved = new_blocks > 0 && new_blocks == stale_blocks;
    if moved { issues.push("possible moved legacy block, refresh-unsafe-allowlist needs an orchestrator ruling".into()); }
    for site in sites {
        let approval = allowed.get(&site.key).map(String::as_str).unwrap_or("");
        if moved && site.kind == "block" { continue; }
        let frozen = approval == FROZEN_RULING;
        let here = format!("{}:{}: {}",site.path,site.line,site.item);
        if site.kind == "block" && !site.contract && (!frozen || grown_items.contains(site.key.rsplit_once('\t').unwrap().0)) {
            issues.push(format!("{here}: unsafe block needs a SAFETY: contract comment directly above it"));
        }
        if frozen { continue; }
        if site.kind == "fn" || (site.kind == "attribute" && site.kernel) {
            if site.gpu && site.kernel {
                if !site.contract { issues.push(format!("{here}: ptx-kernel entry needs a SAFETY: contract comment directly above it")); }
            } else if approval.is_empty() {
                let place = if site.gpu { "" } else { " outside the GPU crate" };
                issues.push(format!("{here}: new unsafe fn{place} needs an allowlist ruling line (ruling YYYY-MM-DD HH:MM)"));
            }
            continue;
        }
        if !site.gpu { issues.push(format!("{here}: new unsafe {} outside the GPU crate",site.kind)); }
        else if site.kind != "block" { issues.push(format!("{here}: only a new unsafe block is allowed in the GPU crate")); }
        if site.kind == "block" {
            if site.gpu {
                if let Some(wrapper) = site.wrapper {
                    let count = blocks.entry((&site.path,wrapper)).or_insert(0); *count += 1;
                    if *count > 1 { issues.push(format!("{here}: kernel wrapper has more than one new unsafe block")); }
                } else { issues.push(format!("{here}: new unsafe block needs a named kernel wrapper")); }
            }
        }
    }
    for key in remaining { if !moved || !block_key(&key) { issues.push(format!("stale unsafe allowlist entry: {key}")); } }
    issues
}
pub fn inventory(root: &Path) -> Result<Vec<Site>> {
    let mut git = Command::new("git"); git.current_dir(root).args(["ls-files","-z","--cached","--others","--exclude-standard","*.rs"]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { git.env_remove(key); } }
    let names = String::from_utf8(crate::output(&mut git)?)?;
    let paths: BTreeSet<_> = names.split('\0').filter(|s| !s.is_empty()).collect(); let mut out = Vec::new();
    for name in paths {
        let path = root.join(name); if !path.exists() { continue; }
        if fs::symlink_metadata(&path)?.file_type().is_symlink() { return Err(format!("{name}: Rust source must not be a symbolic link").into()); }
        let owner = path.ancestors().skip(1).take_while(|p| p.starts_with(root)).find(|p| p.join("Cargo.toml").is_file());
        let crate_path = owner.map(|p| p.strip_prefix(root).unwrap().to_string_lossy().replace('\\',"/")).unwrap_or_else(|| "<root>".into());
        let gpu = matches!(crate_path.as_str(),"engine/crates/gpu"|"engine/crates/gpu/kernels");
        out.extend(sites(&crate_path,name,&fs::read_to_string(&path)?,gpu).map_err(|e| format!("{name}: {e}"))?);
    } Ok(out)
}
const HEADER: &str = "# Unsafe source sites. Frozen at merge. Refresh only on an orchestrator ruling.\n# Crate path, source path in crate, item name, kind and ordinal, optional ruling. Fields use tabs.\n# Frozen rows record legacy sites. A new ordinary unsafe fn needs its own ruling.\n# Keys do not bind signatures, source tokens or safety contracts.\n";
pub fn render(sites: &[Site], allowed: &Allowlist) -> String {
    let keys: BTreeSet<_> = sites.iter().map(|s| s.key.as_str()).collect();
    let mut text = String::from(HEADER);
    for key in keys {
        text.push_str(key);
        if let Some(ruling) = allowed.get(key).filter(|r| !r.is_empty()) { text.push('\t'); text.push_str(ruling); }
        text.push('\n');
    }
    text
}
pub fn drift(sites: &[Site], allowed: &Allowlist) -> Vec<String> {
    let current: BTreeSet<_> = sites.iter().map(|s| s.key.clone()).collect(); let mut issues = Vec::new();
    for key in current.iter().filter(|key| !allowed.contains_key(*key)) { issues.push(format!("unsafe site missing from allowlist: {key}")); }
    for key in allowed.keys().filter(|key| !current.contains(*key)) { issues.push(format!("stale unsafe allowlist entry: {key}")); }
    issues
}
pub fn refresh(root: &Path, approval: &str) -> Result<()> {
    ruling(approval)?;
    let sites = inventory(root)?;
    let path = root.join("tools/unsafe-allowlist.txt");
    let allowed = read_allowlist(&path)?;
    fs::write(path,render(&sites,&allowed))?;
    println!("unsafe allowlist refreshed: {} sites; {approval}; review the diff",sites.len()); Ok(())
}
fn read_allowlist(path: &Path) -> Result<Allowlist> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}",path.display()))?;
    allowlist(&text).map_err(|e| format!("{}: {e}",path.display()).into())
}
pub fn check(root: &Path) -> Result<()> {
    let sites = inventory(root)?;
    let allowed = read_allowlist(&root.join("tools/unsafe-allowlist.txt"))?;
    let mut issues = violations(&sites,&allowed);
    if issues.is_empty() { issues = drift(&sites,&allowed); }
    if !issues.is_empty() { return Err(issues.join("\n").into()); }
    println!("unsafe source check: {} sites; {} allowlist entries; no violations",sites.len(),allowed.len()); Ok(())
}
