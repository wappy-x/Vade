//! CI-only CUDA linkage. Each symbol aborts. Never ship this library.
use std::{collections::BTreeSet, fs, path::Path, process::Command};
use crate::Result;
pub fn symbols(source: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let words: Vec<_> = source.split(|c: char| !c.is_ascii_alphanumeric() && c != '_').filter(|s| !s.is_empty()).collect();
    for pair in words.windows(2) { let name = pair[1];
        if pair[0] == "fn" && name.starts_with("cu") && name.as_bytes().get(2).is_some_and(u8::is_ascii_uppercase) { out.insert(name.into()); }
    } out
}
fn visit(path: &Path, names: &mut BTreeSet<String>) -> Result<()> {
    for entry in fs::read_dir(path)? { let entry = entry?; let kind = entry.file_type()?;
        if kind.is_dir() { visit(&entry.path(),names)?; }
        else if kind.is_file() && entry.path().extension().is_some_and(|s| s == "rs") { names.extend(symbols(&fs::read_to_string(entry.path())?)); }
    } Ok(())
}
pub fn source(names: &BTreeSet<String>) -> String {
    names.iter().map(|name| format!("#[unsafe(no_mangle)]\npub extern \"C\" fn {name}() -> i32 {{ panic!(\"device call in host CI: {name}\") }}\n")).collect()
}
pub fn run(root: &Path) -> Result<()> {
    if !cfg!(target_os = "linux") { return Err("ci-no-gpu runs only on Linux hosts without CUDA".into()); }
    let mut names = BTreeSet::new(); visit(&root.join("engine/crates"),&mut names)?;
    if names.is_empty() { return Err("no CUDA declarations found".into()); }
    let out = root.join("engine/target/ci-no-gpu"); fs::create_dir_all(&out)?;
    let file = out.join("cuda.rs"); fs::write(&file,source(&names))?;
    crate::output(Command::new("rustc").args(["--edition=2024","--crate-type=cdylib","-C","panic=abort"])
        .arg(file).arg("-o").arg(out.join("libcuda.so")))?;
    println!("CI-only CUDA linkage: {} symbols; device calls abort",names.len()); Ok(())
}
