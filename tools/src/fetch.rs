//! Public, pinned model downloads. curl supplies HTTPS; all validation is Rust.
use std::{collections::BTreeSet, fs::{self,File}, io::{Read,Write}, path::{Component,Path}, process::{Command,Stdio}};
use crate::{Result,json,is_lower_hex,sha256};
fn safe_path(name: &str) -> bool {
    !name.is_empty() && !name.chars().any(char::is_control) && !name.contains('\\') && !name.starts_with('/')
        && name.split('/').all(|s| !s.is_empty() && s != "." && s != "..")
}
fn no_links(path: &Path) -> Result<()> {
    let mut current = std::path::PathBuf::new();
    for part in path.components() {
        if matches!(part,Component::ParentDir) { return Err("parent paths are not allowed".into()); }
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => return Err("model paths must not contain links".into()),
            Ok(_) => (), Err(e) if e.kind() == std::io::ErrorKind::NotFound => (), Err(e) => return Err(e.into()) }
    } Ok(())
}
fn curl(lock: &File) -> Result<Command> {
    // The shell can trap signals without an unsafe handler. It keeps the same
    // open file description locked until curl stops, even if Rust is killed.
    let mut c = Command::new("sh"); c.args(["-c",include_str!("../../scripts/fetch-curl.sh"),"fetch-curl"])
        .arg(std::process::id().to_string()).stdin(Stdio::from(lock.try_clone()?));
    c.args(["--disable","--fail","--silent","--show-error","--location","--retry","3",
        "--connect-timeout","30","--speed-time","120","--speed-limit","1","--proto","=https","--proto-redir","=https"]); Ok(c)
}
fn encode(path: &str) -> String {
    path.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}
#[derive(Debug)]
pub struct Entry { pub name: String, pub size: u64, pub digest: String, pub lfs: bool }
pub fn entries(text: &str, revision: &str) -> Result<Vec<Entry>> {
    let value = json::parse(text)?;
    if value.get("sha").text()? != revision { return Err("model metadata revision differs".into()); }
    let mut entries = Vec::new(); let mut names = BTreeSet::new();
    for file in value.get("siblings").array()? {
        let name = file.get("rfilename").text()?.to_owned();
        if !safe_path(&name) || name == ".cache" || name.starts_with(".cache/") || !names.insert(name.clone()) { return Err("unsafe or duplicate model path".into()); }
        let lfs = file.get("lfs") != &json::Value::Null;
        let digest = if lfs { file.get("lfs").get("sha256") } else { file.get("blobId") }.text()?.to_owned();
        if !is_lower_hex(&digest,if lfs {64} else {40}) { return Err("invalid model file digest".into()); }
        entries.push(Entry { name, size: file.get("size").text()?.parse()?, digest, lfs });
    }
    if entries.is_empty() { return Err("model metadata has no files".into()); }
    entries.sort_by(|a,b| a.name.cmp(&b.name)); Ok(entries)
}
// Git stores the SHA-1 of "blob <length>\0" followed by the small file bytes.
fn git_digest(path: &Path, size: u64) -> Result<String> {
    let mut input = std::io::Cursor::new(format!("blob {size}\0").into_bytes()).chain(File::open(path)?);
    let mut state = [0x67452301u32,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0];
    let mut total = 0u64; let mut pending = Vec::new(); let mut buffer = [0;65536];
    loop {
        let n = input.read(&mut buffer)?; total += n as u64; pending.extend_from_slice(&buffer[..n]);
        if n == 0 { pending.push(0x80); while pending.len()%64 != 56 { pending.push(0); } pending.extend_from_slice(&total.wrapping_mul(8).to_be_bytes()); }
        let complete = pending.len()/64*64;
        for block in pending[..complete].as_chunks::<64>().0 {
            let mut w = [0u32;80];
            for (word,&bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) { *word = u32::from_be_bytes(bytes); }
            for i in 16..80 { w[i] = (w[i-3]^w[i-8]^w[i-14]^w[i-16]).rotate_left(1); }
            let [mut a,mut b,mut c,mut d,mut e] = state;
            for (i,&word) in w.iter().enumerate() {
                let (f,k) = match i { 0..=19 => ((b&c)|(!b&d),0x5a827999u32), 20..=39 => (b^c^d,0x6ed9eba1),
                    40..=59 => ((b&c)|(b&d)|(c&d),0x8f1bbcdc), _ => (b^c^d,0xca62c1d6) };
                let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(word);
                [e,d,c,b,a] = [d,c,b.rotate_left(30),a,t];
            }
            for (s,v) in state.iter_mut().zip([a,b,c,d,e]) { *s = s.wrapping_add(v); }
        }
        pending.drain(..complete); if n == 0 { break; }
    } Ok(state.iter().map(|n| format!("{n:08x}")).collect())
}
pub fn verify(path: &Path, entry: &Entry) -> Result<String> {
    no_links(path)?;
    if fs::metadata(path)?.len() != entry.size { return Err(format!("model file size differs: {}",entry.name).into()); }
    let sha = sha256::digest_reader(File::open(path)?)?;
    let digest = if entry.lfs { sha.clone() } else { git_digest(path,entry.size)? };
    if digest != entry.digest { return Err(format!("model file hash differs: {}",entry.name).into()); } Ok(sha)
}
fn inventory(path: &Path, base: &Path, names: &mut BTreeSet<String>) -> Result<()> {
    for entry in fs::read_dir(path)? { let entry = entry?; let relative = entry.path().strip_prefix(base)?.to_owned();
        if relative.starts_with(".cache/huggingface") { continue; }
        let kind = entry.file_type()?;
        if kind.is_dir() { inventory(&entry.path(),base,names)?; }
        else if kind.is_file() { names.insert(relative.to_str().ok_or("invalid model file name")?.into()); }
        else { return Err("model directory contains a link or special file".into()); }
    } Ok(())
}
pub fn run(repo: &str, revision: &str, model: &Path) -> Result<()> {
    if !safe_path(repo) || repo.split('/').count() != 2 || !is_lower_hex(revision,40) { return Err("supply a repository and full revision hash".into()); }
    no_links(model)?; fs::create_dir_all(model)?; let model = model.canonicalize()?;
    if model.parent().is_none() { return Err("model directory must not be the filesystem root".into()); }
    let state = model.with_file_name(format!("{}.fetch",model.file_name().unwrap().to_string_lossy())); no_links(&state)?;
    if !state.exists() {
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)] { use std::os::unix::fs::DirBuilderExt; builder.mode(0o700); }
        builder.create(&state)?;
    }
    let lock_path = state.join("lock"); no_links(&lock_path)?;
    // Never unlink this file: a waiter must lock the same inode. The OS releases
    // the lock after the last owner exits; no stale PID or age guess is needed.
    let lock = File::options().read(true).write(true).create(true).truncate(false).open(lock_path)?;
    lock.try_lock().map_err(|e| format!("another fetch may be running: {e}"))?;
    let metadata = crate::output(curl(&lock)?.args(["--max-time","120"])
        .arg(format!("https://huggingface.co/api/models/{}/revision/{revision}?blobs=true",encode(repo))))?;
    let entries = entries(std::str::from_utf8(&metadata)?,revision)?;
    let names: BTreeSet<_> = entries.iter().map(|e| e.name.clone()).collect(); let mut found = BTreeSet::new(); inventory(&model,&model,&mut found)?;
    if !found.is_subset(&names) { return Err("model directory has extra files".into()); }
    let mut manifest = String::new();
    for (i,entry) in entries.iter().enumerate() {
        let path = model.join(&entry.name); no_links(&path)?;
        if verify(&path,entry).is_err() {
            fs::create_dir_all(path.parent().unwrap())?;
            let part = state.join(format!("download-{i}.part")); no_links(&part)?;
            if part.exists() && (fs::metadata(&part)?.len() > entry.size || (fs::metadata(&part)?.len() == entry.size && verify(&part,entry).is_err())) {
                fs::rename(&part,state.join(format!("download-{i}.bad-{}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos())))?;
            }
            if verify(&part,entry).is_err() {
                crate::output(curl(&lock)?.args(["--continue-at","-"]).arg("--output").arg(&part)
                    .arg(format!("https://huggingface.co/{}/resolve/{revision}/{}",encode(repo),encode(&entry.name))))?;
            }
            verify(&part,entry)?; fs::rename(&part,&path)?;
        }
        manifest.push_str(&format!("{}  ./{}\n",verify(&path,entry)?,entry.name));
    }
    let mut found = BTreeSet::new(); inventory(&model,&model,&mut found)?;
    if found != names { return Err("model directory has missing or extra files".into()); }
    let pending = state.join("manifest"); no_links(&pending)?;
    let mut file = File::create(&pending)?; file.write_all(manifest.as_bytes())?; file.sync_all()?;
    let target = model.with_file_name(format!("{}.sha256",model.file_name().unwrap().to_string_lossy())); no_links(&target)?;
    fs::rename(pending,target)?; println!("verified {} files",entries.len()); Ok(())
}
