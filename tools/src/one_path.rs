//! Review new environment reads and topology path choices in engine source.
//! This token check is conservative. It does not resolve types or prove data flow.
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path,process::Command};
use crate::Result;

#[derive(Clone,Debug,Eq,Ord,PartialEq,PartialOrd)]
pub struct Key { pub rule: String, pub krate: String, pub item: String }
#[derive(Clone,Debug,Eq,PartialEq)]
pub struct Finding { pub key: Key, pub file: String, pub line: usize }
#[derive(Clone,Debug)]
struct Token { text: String, line: usize }
impl Token { fn is(&self, s: &str) -> bool { self.text == s } }

fn tokens(source: &str) -> Vec<Token> {
    let b=source.as_bytes(); let (mut i,mut line,mut out)=(0,1,Vec::new());
    while i<b.len() {
        let start=i; let at=line;
        if b[i..].starts_with(b"//") {
            while i<b.len() && b[i]!=b'\n' { i+=1; }
        } else if b[i..].starts_with(b"/*") {
            i+=2; let mut depth=1;
            while i<b.len() && depth>0 {
                if b[i..].starts_with(b"/*") { depth+=1; i+=2; }
                else if b[i..].starts_with(b"*/") { depth-=1; i+=2; }
                else { i+=1; }
            }
        } else if b[i].is_ascii_whitespace() { i+=1; }
        else {
            let mut q=i;
            if matches!(b[q],b'b'|b'c') { q+=1; }
            let raw=b.get(q)==Some(&b'r'); let mut hashes=0;
            if raw { q+=1; while b.get(q)==Some(&b'#') { hashes+=1; q+=1; } }
            if b.get(q)==Some(&b'"') {
                i=q+1;
                while i<b.len() {
                    if b[i]==b'"' && (!raw || b.get(i+1..i+1+hashes).is_some_and(|s|s.iter().all(|&v|v==b'#'))) {
                        i+=1+hashes; break;
                    }
                    if !raw && b[i]==b'\\' { i+=1; }
                    i=(i+1).min(b.len());
                }
                out.push(Token{text:"<literal>".into(),line:at});
            } else if b[i]==b'\'' || b[i..].starts_with(b"b'") {
                // A lifetime has no closing quote after its identifier.
                let quote=i+usize::from(b[i]==b'b'); let mut end=quote+1;
                if b.get(end)==Some(&b'\\') {
                    end+=1;
                    while end<b.len() && b[end]!=b'\'' && b[end]!=b'\n' { end+=1; }
                } else if end<b.len() {
                    end+=source[end..].chars().next().unwrap().len_utf8();
                }
                if b.get(end)==Some(&b'\'') {
                    i=end+1; out.push(Token{text:"<literal>".into(),line:at});
                } else { i+=1; out.push(Token{text:"'".into(),line:at}); }
            } else if b[i].is_ascii_alphanumeric() || b[i]==b'_' || b[i]>=128 {
                if b[i..].starts_with(b"r#") { i+=2; }
                let word=i; i+=1;
                while i<b.len() && (b[i].is_ascii_alphanumeric() || b[i]==b'_' || b[i]>=128) { i+=1; }
                out.push(Token{text:source[word..i].into(),line:at});
            } else {
                i+=1;
                if i<b.len() && matches!((b[start],b[i]),(b':',b':')|(b'='|b'-'|b'>',b'>')|(b'<',b'<')|(b'='|b'!'|b'<'|b'>',b'=')|(b'&',b'&')|(b'|',b'|')) { i+=1; }
                out.push(Token{text:source[start..i].into(),line:at});
            }
        }
        line+=b[start..i].iter().filter(|&&v|v==b'\n').count();
    }
    out
}
fn ident(s: &str) -> bool { s.as_bytes().first().is_some_and(|b|b.is_ascii_alphabetic() || *b==b'_' || *b>=128) }
fn close(t: &[Token], at: usize) -> usize {
    let end=match t[at].text.as_str() {"("=>")","["=>"]","{"=>"}",_=>return at};
    let mut i=at+1;
    while i<t.len() {
        if t[i].is(end) { return i; }
        if matches!(t[i].text.as_str(),"("|"["|"{") { i=close(t,i); }
        i+=1;
    }
    t.len().saturating_sub(1)
}
fn body(t: &[Token], mut i: usize, end: usize) -> Option<usize> {
    while i<end {
        if t[i].is("{") { return Some(i); }
        if t[i].is(";") { return None; }
        if t[i].is("(") || t[i].is("[") { i=close(t,i); }
        i+=1;
    }
    None
}
fn condition_end(t: &[Token], mut i: usize) -> Option<usize> {
    while i<t.len() {
        if t[i].is("{") || t[i].is("=>") {return Some(i);}
        if t[i].is(";") || t[i].is("}") {return None;}
        if t[i].is("(") || t[i].is("[") {i=close(t,i);}
        i+=1;
    }
    None
}
fn topology_patterns(t: &[Token], open: usize) -> bool {
    let end=close(t,open); let (mut i,mut start)=(open+1,open+1); let mut pattern=true;
    while i<end {
        if t[i].is("=>") {
            if t[start..i].iter().any(|v|v.is("Topology")) {return true;}
            pattern=false;
        } else if t[i].is(",") {start=i+1; pattern=true;}
        else if matches!(t[i].text.as_str(),"{"|"("|"[") {
            let last=close(t,i);
            if pattern && t[i..=last].iter().any(|v|v.is("Topology")) {return true;}
            i=last;
        }
        i+=1;
    }
    false
}
fn type_name(t: &[Token]) -> String {
    let (mut depth,mut out)=(0usize,String::new());
    for token in t {
        match token.text.as_str() {
            "<"=>depth+=1, ">"=>depth=depth.saturating_sub(1), ">>"=>depth=depth.saturating_sub(2), "where" if depth==0=>break,
            s if depth==0 && (ident(s) || s=="::")=>out.push_str(s), _=>(),
        }
    }
    out
}
fn owners(t: &[Token], start: usize, end: usize, prefix: &str, out: &mut [String]) {
    out[start..end].fill(prefix.into()); let mut i=start;
    while i<end {
        let kind=t[i].text.as_str();
        if kind=="impl" {
            if let Some(open)=body(t,i+1,end) {
                let head=&t[i+1..open]; let split=head.iter().position(|v|v.is("for"));
                let name=if let Some(at)=split {format!("{}@{}",type_name(&head[at+1..]),type_name(&head[..at]))} else {type_name(head)};
                let last=close(t,open).min(end-1);
                owners(t,open+1,last,&join(prefix,&name),out); i=last+1; continue;
            }
        } else if matches!(kind,"fn"|"mod"|"trait"|"const"|"static"|"macro_rules")
            && t.get(i+1).is_some_and(|v|(ident(&v.text) && !v.is("fn")) || v.is("!")) {
            let name_at=i+1+usize::from(kind=="macro_rules");
            if let Some(name)=t.get(name_at).filter(|v|ident(&v.text)) {
                let item=join(prefix,&name.text);
                if let Some(open)=body(t,name_at+1,end) {
                    let last=close(t,open).min(end-1); out[i..=open].fill(item.clone());
                    owners(t,open+1,last,&item,out); i=last+1; continue;
                } else if matches!(kind,"const"|"static") {
                    let last=(i..end).find(|&p|t[p].is(";")).unwrap_or(end-1);
                    out[i..=last].fill(item); i=last+1; continue;
                }
            }
        }
        i+=1;
    }
}
fn join(a: &str, b: &str) -> String { if a.is_empty() {b.into()} else {format!("{a}::{b}")} }

fn use_tree(t: &[Token], mut i: usize, end: usize, mut path: Vec<String>, out: &mut Vec<(String,Vec<String>)>) {
    while i<end {
        match t[i].text.as_str() {
            "::"=>i+=1,
            "{"=>{
                let last=close(t,i).min(end); let mut part=i+1; let mut p=part;
                while p<last {
                    if t[p].is("{") { p=close(t,p); }
                    else if t[p].is(",") {use_tree(t,part,p,path.clone(),out); part=p+1;}
                    p+=1;
                }
                use_tree(t,part,last,path,out); return;
            },
            "as"=>{if let Some(alias)=t.get(i+1) {out.push((alias.text.clone(),path));} return;},
            "*"=>{out.push(("*".into(),path)); return;},
            "self"=>i+=1,
            s if ident(s)=>{path.push(s.into()); i+=1;}, _=>return,
        }
    }
    if let Some(name)=path.last() {out.push((name.clone(),path));}
}
const READS: &[&str]=&["var","var_os","vars","vars_os"];
type Aliases=BTreeMap<String,BTreeSet<Vec<String>>>;
fn expand(path: &[String], aliases: &Aliases) -> BTreeSet<Vec<String>> {
    if let Some(head)=path.first().and_then(|s|aliases.get(s)) {
        head.iter().map(|head|head.iter().chain(path.iter().skip(1)).cloned().collect()).collect()
    } else {BTreeSet::from([path.to_vec()])}
}
fn imports(t: &[Token]) -> (Aliases,BTreeSet<usize>) {
    let mut declarations=Vec::new(); let mut ignored=BTreeSet::new();
    for i in 0..t.len() {
        if t[i].is("use") {
            let end=(i+1..t.len()).find(|&p|t[p].is(";")).unwrap_or(t.len());
            use_tree(t,i+1,end,Vec::new(),&mut declarations); ignored.extend(i..end);
        }
        if t[i].is("extern") && t.get(i+1).is_some_and(|v|v.is("crate"))
            && t.get(i+2).is_some_and(|v|v.is("std")) && t.get(i+3).is_some_and(|v|v.is("as")) {
            if let Some(alias)=t.get(i+4) {declarations.push((alias.text.clone(),vec!["std".into()]));}
        }
    }
    let mut aliases=Aliases::new();
    // Resolve chains of local imports. Keep the pass count bounded by the input.
    for _ in 0..=declarations.len() {
        let before=aliases.clone();
        for (name,path) in &declarations {
            for path in expand(path,&before) {
                // Keep all relevant aliases across scopes. Shadowing must not hide a read.
                let relevant=path==["std"] || path==["std","env"] || path.len()==3 && path[0]=="std" && path[1]=="env" && READS.contains(&path[2].as_str());
                if !relevant {continue;}
                if name=="*" {
                    if path==["std","env"] {for read in READS {aliases.entry((*read).into()).or_default().insert(vec!["std".into(),"env".into(),(*read).into()]);}}
                    else if path==["std"] {aliases.entry("env".into()).or_default().insert(vec!["std".into(),"env".into()]);}
                } else {aliases.entry(name.clone()).or_default().insert(path);}
            }
        }
        if aliases==before {break;}
    }
    (aliases,ignored)
}
fn topology(t: &[Token]) -> bool {
    t.iter().any(|v| {
        let s=v.text.to_ascii_lowercase();
        // These names describe tensor dimensions. Other tokens still count.
        if matches!(s.as_str(),"lora_rank"|"tensor_rank") || s.ends_with("_lora_rank") || s.ends_with("_tensor_rank") {return false;}
        matches!(s.as_str(),"world"|"world_size"|"boxes"|"box_count"|"num_boxes"|"n_boxes"|"ranks"|"rank_count"|"num_ranks"|"n_ranks"|"rank"|"rank_id"|"topology"|"rank_path"|"tensor_split"|"tp"|"pp"|"sparks"|"distributed")
            || ["_world","_world_size","_rank","_ranks","_topology"].iter().any(|end|s.ends_with(end))
    })
}
fn boundary(t: &Token) -> bool { matches!(t.text.as_str(),";"|"{"|"}"|"="|","|"&&"|"||"|"=>") }

/// Scan all source, including tests and inactive cfg blocks. Report relative paths.
pub fn scan(source: &str, krate: &str, module: &str, file: &str) -> Vec<Finding> {
    let t=tokens(source); let mut item=vec![module.into();t.len()]; owners(&t,0,t.len(),module,&mut item);
    let mut hits=BTreeSet::new(); let (aliases,ignored)=imports(&t);
    for i in 0..t.len() {
        if ignored.contains(&i) || !ident(&t[i].text) {continue;}
        if i>0 && (t[i-1].is(".") || t[i-1].is("::") && i>1 && ident(&t[i-2].text)) {continue;}
        let mut path=vec![t[i].text.clone()]; let mut p=i+1;
        while p+1<t.len() && t[p].is("::") && ident(&t[p+1].text) {path.push(t[p+1].text.clone()); p+=2;}
        if expand(&path,&aliases).iter().any(|resolved|resolved.len()==3 && resolved[0]=="std" && resolved[1]=="env" && READS.contains(&resolved[2].as_str())) {
            hits.insert((i,"env-read"));
        }
    }
    let mut covered=BTreeSet::new();
    for i in 0..t.len() {
        let kind=t[i].text.as_str();
        if matches!(kind,"if"|"while"|"match") {
            if let Some(end)=condition_end(&t,i+1) {
                if topology(&t[i+1..end]) || kind=="match" && t[end].is("{") && topology_patterns(&t,end) {
                    hits.insert((i,"topology-branch")); covered.extend(i..end);
                }
            }
        } else if !covered.contains(&i) && kind=="matches" && t.get(i+1).is_some_and(|v|v.is("!")) && t.get(i+2).is_some_and(|v|v.is("(")) {
            let end=close(&t,i+2);
            if topology(&t[i+3..end]) {hits.insert((i,"topology-branch")); covered.extend(i..=end);}
        }
    }
    let mut generics=ignored;
    for i in 0..t.len() {
        if t[i].is("fn") && t.get(i+1).is_some_and(|v|ident(&v.text)) {
            if let Some(end)=body(&t,i+1,t.len()) {generics.extend(i..end);}
            else if let Some(end)=(i+1..t.len()).find(|&p|t[p].is(";")) {generics.extend(i..end);}
        } else if matches!(t[i].text.as_str(),"struct"|"enum"|"trait"|"impl") {
            if let Some(end)=body(&t,i+1,t.len()) {
                let end=if t[i].is("struct") || t[i].is("enum") {close(&t,end)+1} else {end};
                generics.extend(i..end);
            } else if let Some(end)=(i+1..t.len()).find(|&p|t[p].is(";")) {generics.extend(i..end);}
        } else if t[i].is("type") {
            if let Some(end)=(i+1..t.len()).find(|&p|t[p].is(";")) {generics.extend(i..end);}
        }
    }
    for i in 1..t.len() {
        if t[i].is("<") && t[i-1].is("::") {
            let mut depth=1usize; let mut end=i+1;
            while end<t.len() && depth>0 {
                if t[end].is("<") {depth+=1;} else if t[end].is(">") {depth-=1;} else if t[end].is(">>") {depth=depth.saturating_sub(2);}
                end+=1;
            }
            generics.extend(i..end);
        }
    }
    for i in 0..t.len() {
        if covered.contains(&i) {continue;}
        if !generics.contains(&i) && matches!(t[i].text.as_str(),"=="|"!="|"<"|">"|"<="|">=") {
            let start=(0..i).rfind(|&p|boundary(&t[p])).map_or(0,|p|p+1);
            let end=(i+1..t.len()).find(|&p|boundary(&t[p])).unwrap_or(t.len());
            // Angle brackets in type parameters are not path choices.
            if topology(&t[start..end]) && !t[start..end].iter().any(|v|matches!(v.text.as_str(),"fn"|"impl"|"struct"|"enum"|"use"|"type")) {
                hits.insert((i,"topology-branch"));
            }
        }
    }
    hits.into_iter().map(|(i,rule)|Finding {key:Key {rule:rule.into(),krate:krate.into(),item:if item[i].is_empty(){"<module>".into()}else{item[i].clone()}},file:file.into(),line:t[i].line}).collect()
}

/// Each row has a rule, Cargo package, qualified item and exact site count.
pub fn allowlist(source: &str) -> Result<BTreeMap<Key,usize>> {
    let mut out=BTreeMap::new();
    for (line,text) in source.lines().enumerate() {
        let text=text.split('#').next().unwrap().trim(); if text.is_empty() {continue;}
        let fields:Vec<_>=text.split_whitespace().collect();
        if fields.len()!=4 || !matches!(fields[0],"env-read"|"topology-branch") {return Err(format!("invalid one-path allowlist row {}",line+1).into());}
        let count:usize=fields[3].parse()?; let key=Key {rule:fields[0].into(),krate:fields[1].into(),item:fields[2].into()};
        if count==0 || out.insert(key,count).is_some() {return Err(format!("zero or duplicate one-path allowlist row {}",line+1).into());}
    }
    Ok(out)
}
pub fn counts(findings: &[Finding]) -> BTreeMap<Key,usize> {
    let mut out=BTreeMap::new(); for f in findings {*out.entry(f.key.clone()).or_default()+=1;} out
}
pub fn check(findings: &[Finding], allowed: &BTreeMap<Key,usize>) -> Vec<String> {
    let actual=counts(findings); let mut errors=Vec::new();
    for (key,count) in &actual {
        let limit=allowed.get(key).copied().unwrap_or(0);
        if *count>limit {
            for f in findings.iter().filter(|f|&f.key==key) {errors.push(format!("{}:{}: {} {} {} (sites {count}, allowed {limit})",f.file,f.line,key.rule,key.krate,key.item));}
        }
    }
    for (key,limit) in allowed {
        let count=actual.get(key).copied().unwrap_or(0);
        if count<*limit {errors.push(format!("remove stale one-path allowance: {} {} {} (sites {count}, allowed {limit})",key.rule,key.krate,key.item));}
    }
    errors
}

fn package(manifest: &str) -> Option<String> {
    let mut in_package=false;
    for raw in manifest.lines() {
        let s=raw.split('#').next().unwrap().trim();
        if s.starts_with('[') {in_package=s=="[package]";}
        else if in_package {if let Some((key,value))=s.split_once('=') {if key.trim()=="name" {return Some(value.trim().trim_matches('"').into());}}}
    }
    None
}
fn module(path: &Path) -> String {
    let mut parts:Vec<_>=path.with_extension("").iter().map(|s|s.to_string_lossy().replace('-',"_")).collect();
    if parts.first().is_some_and(|s|s=="src") {parts.remove(0);}
    if parts.last().is_some_and(|s|s=="lib" || s=="mod") {parts.pop();}
    parts.join("::")
}
pub fn inventory(root: &Path) -> Result<Vec<Finding>> {
    // Use Git's source inventory, as the ratchet does. Never walk build output.
    let output=crate::output(Command::new("git").current_dir(root).args(["ls-files","--cached","--others","--exclude-standard","-z","--","engine"]))?;
    let inventory=String::from_utf8(output)?;
    let files:BTreeSet<_>=inventory.split('\0').filter(|s|!s.is_empty()).collect();
    let mut crates=BTreeMap::new();
    for file in &files {
        if file.ends_with("/Cargo.toml") && root.join(file).is_file() {
            if let Some(name)=package(&fs::read_to_string(root.join(file))?) {crates.insert(Path::new(file).parent().unwrap().to_path_buf(),name);}
        }
    }
    let mut out=Vec::new();
    for file in files {
        if !file.ends_with(".rs") || !root.join(file).exists() {continue;}
        let path=Path::new(file);
        let owner=crates.iter().filter(|(dir,_)|path.starts_with(dir)).max_by_key(|(dir,_)|dir.components().count());
        if fs::symlink_metadata(root.join(file))?.file_type().is_symlink() {return Err(format!("source link is not allowed: {file}").into());}
        let source=fs::read_to_string(root.join(file))?;
        if let Some((dir,krate))=owner {out.extend(scan(&source,krate,&module(path.strip_prefix(dir)?),file));}
        else {
            // Direct rustc tools have a crate name from the source file stem.
            let krate=path.file_stem().unwrap().to_string_lossy().replace('-',"_");
            out.extend(scan(&source,&krate,"",file));
        }
    }
    Ok(out)
}
pub fn run(root: &Path) -> Result<()> {
    let findings=inventory(root)?;
    for f in &findings {println!("FINDING {}:{} | {} {} {}",f.file,f.line,f.key.rule,f.key.krate,f.key.item);}
    let allowed=allowlist(&fs::read_to_string(root.join("tools/one-path-allowlist.txt"))?)?;
    let errors=check(&findings,&allowed);
    if !errors.is_empty() {
        for error in errors {eprintln!("{error}");}
        for (key,count) in counts(&findings) {println!("ALLOWLIST {} {} {} {count}",key.rule,key.krate,key.item);}
        return Err("one-path check failed; review each site against AGENTS.md".into());
    }
    println!("one-path check: PASS ({} allowed sites, {} item allowances)",findings.len(),allowed.len()); Ok(())
}

#[cfg(test)]
mod tests;
