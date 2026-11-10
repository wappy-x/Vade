//! Check local documentation links without network access.
use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path, process::Command};
use crate::Result;
fn attributes(text: &str, keys: &[&str]) -> Vec<(usize,String)> {
    let mut out = Vec::new();
    for key in keys { for quote in ['"','\''] {
        let needle = format!("{key}={quote}"); let mut offset = 0;
        while let Some(i) = text[offset..].find(&needle) {
            let start = offset+i; let value = start+needle.len();
            let Some(end) = text[value..].find(quote) else { break; };
            out.push((start,text[value..value+end].into())); offset = value+end+1;
        }
    } } out
}
pub fn anchors(text: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<_> = attributes(text,&["id","name"]).into_iter().map(|(_,v)|v).collect();
    let mut seen = BTreeMap::new();
    for line in text.lines() {
        let hashes = line.bytes().take_while(|&b| b == b'#').count();
        if !(1..=6).contains(&hashes) || !line[hashes..].starts_with(char::is_whitespace) { continue; }
        let title = line[hashes..].trim().trim_end_matches('#').trim_end();
        let mut tag = false; let mut anchor = String::new();
        for ch in title.to_lowercase().chars() {
            if ch == '<' { tag = true; } else if ch == '>' && tag { tag = false; }
            else if !tag && (ch.is_alphanumeric() || matches!(ch,'_'|'-'|' ')) { anchor.push(if ch == ' ' { '-' } else { ch }); }
        }
        let count = seen.entry(anchor.clone()).or_insert(0);
        out.insert(if *count == 0 { anchor } else { format!("{anchor}-{count}") }); *count += 1;
    } out
}
fn targets(text: &str) -> Vec<(usize,String)> {
    let mut out = attributes(text,&["src","srcset","href"]); let mut offset = 0;
    while let Some(i) = text[offset..].find("](") {
        let start = offset+i+2;
        if let Some(end) = text[start..].find(')') {
            let value = &text[start..start+end]; if !value.contains('\n') { out.push((start,value.into())); }
        }
        offset = start;
    }
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let s = line.trim_start_matches([' ','\t']);
        if line.len()-s.len() <= 3 && s.starts_with('[')
            && let Some(i) = s.find("]:") && let Some(value) = s[i+2..].split_whitespace().next() { out.push((offset,value.into())); }
        offset += line.len();
    } out.sort_by_key(|x| x.0); out
}
fn unquote(s: &str) -> String {
    let mut out = Vec::new(); let b = s.as_bytes(); let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i+2 < b.len()
            && let (Some(a),Some(c)) = ((b[i+1] as char).to_digit(16),(b[i+2] as char).to_digit(16)) {
            out.push((a*16+c) as u8); i += 3; continue;
        } out.push(b[i]); i += 1;
    } String::from_utf8_lossy(&out).into_owned()
}
pub fn check(root: &Path, paths: &[String]) -> Result<Vec<String>> {
    let mut issues = Vec::new(); let mut cache = BTreeMap::new();
    for name in paths {
        let path = root.join(name); if !path.is_file() { continue; }
        let text = fs::read_to_string(&path)?;
        for (offset,target) in targets(&text) {
            let target = target.trim();
            let target = if let Some(s) = target.strip_prefix('<') { s.split('>').next().unwrap() }
                else { target.split(" \"").next().unwrap().split(" '").next().unwrap() };
            let scheme = target.split_once(':').is_some_and(|(s,_)| s.starts_with(|c:char| c.is_ascii_alphabetic())
                && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c,'+'|'.'|'-')));
            if scheme || target.starts_with("//") { continue; }
            let (url,fragment) = target.split_once('#').unwrap_or((target,""));
            let file = url.split('?').next().unwrap();
            let destination = if file.is_empty() { path.clone() } else { path.parent().unwrap().join(unquote(file)) };
            let line = text[..offset].bytes().filter(|&b| b == b'\n').count()+1;
            if !destination.exists() { issues.push(format!("{name}:{line}: missing path: {target}")); }
            else if !fragment.is_empty() && destination.extension().is_some_and(|s| s == "md") {
                if !cache.contains_key(&destination) { cache.insert(destination.clone(),anchors(&fs::read_to_string(&destination)?)); }
                if !cache[&destination].contains(&unquote(fragment)) { issues.push(format!("{name}:{line}: missing anchor: {target}")); }
            }
        }
    } Ok(issues)
}
pub fn run(root: &Path) -> Result<()> {
    let mut git = Command::new("git"); git.current_dir(root).args(["ls-files","-z","--cached","--others","--exclude-standard","*.md","*.txt"]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { git.env_remove(key); } }
    let names = String::from_utf8(crate::output(&mut git)?)?;
    let paths: Vec<_> = names.split('\0').filter(|s| !s.is_empty()).map(str::to_owned).collect::<BTreeSet<_>>().into_iter().collect();
    let issues = check(root,&paths)?;
    for issue in &issues { println!("{issue}"); }
    println!("{} documentation files; {} broken local references",paths.len(),issues.len());
    if issues.is_empty() { Ok(()) } else { Err("broken local documentation references".into()) }
}
