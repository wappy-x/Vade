//! Build the Inkling kernel image (crate `spark-inkling-kernels`) with the pinned nightly, stamp it and hold every entry
//! to its row of `kernels-inkling/REGISTERS.txt`. Run with `cargo kernels-inkling`, after `cargo kernels`.
//! A separate command (lead's ruling 2026-11-22 18:43:56): the general build tool stays byte for byte main's.
//! No environment read (lead's ruling 2026-11-22 20:38): the crate folder is the compile-time `CARGO_MANIFEST_DIR`;
//! a toolchain folder or a ptxas path other than the defaults comes only as a command-line argument.
use std::{env, fs, path::{Path, PathBuf}, process::{self, Command}};
#[path = "../../../inkling_stamp.rs"]
#[allow(dead_code)] // `verify` and the build command name belong to the host build script.
mod inkling_stamp;

const NIGHTLY: &str = "nightly-2026-10-01";
const TARGET: &str = "nvptx64-nvidia-cuda";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    const USAGE: &str = "usage: cargo kernels-inkling [--allow-missing-ptxas] [--kernel-bin TOOLCHAIN_BIN_DIR] [--ptxas PTXAS]";
    let (mut allow_missing_ptxas, mut kernel_bin, mut ptxas) = (false, None::<PathBuf>, None::<PathBuf>);
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--allow-missing-ptxas") if !allow_missing_ptxas => allow_missing_ptxas = true,
            Some("--kernel-bin") if kernel_bin.is_none() => kernel_bin = Some(args.next().filter(|v| !v.is_empty()).ok_or(USAGE)?.into()),
            Some("--ptxas") if ptxas.is_none() => ptxas = Some(args.next().filter(|v| !v.is_empty()).ok_or(USAGE)?.into()),
            _ => return Err(USAGE.into()),
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gpu = manifest.parent().ok_or("no gpu folder")?;
    env::set_current_dir(gpu).map_err(|e| format!("cannot enter {}: {e}", gpu.display()))?;
    // A given toolchain folder: its own cargo, and its rustc named for cargo, so no search path is read or changed.
    let mut cargo = match &kernel_bin {
        Some(bin) => { let mut cargo = Command::new(bin.join("cargo")); cargo.env("RUSTC", bin.join("rustc")); cargo }
        None => { let mut cargo = Command::new("cargo"); cargo.arg(format!("+{NIGHTLY}")); cargo }
    };
    let status = cargo.args(["build", "-Z", "build-std=core", "-p", inkling_stamp::PACKAGE, "--release", "--target", TARGET, "--target-dir", inkling_stamp::TARGET_DIR])
        .env("RUSTFLAGS", inkling_stamp::FLAGS).env_remove("CARGO")
        .status().map_err(|e| format!("cannot run cargo: {e}"))?;
    if !status.success() { process::exit(status.code().unwrap_or(1)); }
    let (kernels, layer) = (inkling_stamp::ptx_path(Path::new("."), false), inkling_stamp::ptx_path(Path::new("."), true));
    fs::copy(&kernels, &layer).map_err(|e| format!("cannot copy {}: {e}", kernels.display()))?;
    cap_registers(&layer)?;
    register_check(&layer, Path::new("kernels-inkling/REGISTERS.txt"), ptxas, allow_missing_ptxas)?;
    inkling_stamp::stamp(Path::new("."), &layer).map_err(|e| format!("cannot stamp {}: {e}", layer.display()))
}

/// Entries with a register cap. Rust has no launch-bounds attribute, so the build writes `.maxnreg` after the entry's
/// parameter list, as `ptx_stamp.rs` does for the general image; the register check below then still refuses any spill.
/// resident_launch_plain_packed_tc_rows (256 threads, 128 x 128 tile, 64 accumulator registers a thread): 128 keeps two
/// blocks (16 warps) a multiprocessor; unbounded ptxas takes 138 (one block).
/// resident_launch_plain_staged_one (128 threads, a 12-tile load ring of 48 registers a thread): 96 keeps five blocks a
/// multiprocessor (one wave of a 228-block head part); unbounded ptxas takes 108 to 130 with the 2-to-8-row panel form.
const REGISTER_CAPS: [(&str, u32); 7] = [
    ("resident_launch_plain_packed_tc_rows", 128),
    ("resident_launch_prefill_packed_tc", 128),
    ("resident_launch_plain_staged_one", 80),
    ("resident_launch_flash_window", 168),
    ("resident_launch_plain_tc_big", 124),
    ("resident_launch_fast_stream_ops", 78),
    ("inkling_draft_head_fold", 64),
];
fn cap_registers(ptx: &Path) -> Result<(), String> {
    let mut text = fs::read_to_string(ptx).map_err(|e| format!("cannot read {}: {e}", ptx.display()))?;
    for (name, registers) in REGISTER_CAPS {
        let entry = text.find(&format!(".entry {name}(")).ok_or_else(|| format!("INKLING REGISTER CAP: missing entry {name}"))?;
        // Performance directives go between the parameter list and the body.
        let body = entry + text[entry..].find("\n{").ok_or_else(|| format!("INKLING REGISTER CAP: no body for {name}"))?;
        if !text[entry..body].contains(".maxnreg") { text.insert_str(body + 1, &format!(".maxnreg {registers}\n")); }
    }
    fs::write(ptx, text).map_err(|e| format!("cannot write {}: {e}", ptx.display()))
}

/// The register check of `cargo kernels` for the tensor-core image, with one more column: the stack bytes an entry may
/// use. ptxas (`sm_121a`, -O3) must report for every entry at most its row's registers, shared bytes and stack bytes,
/// no spill byte, a block that can launch; every entry has a row and every row an entry. ptxas is the `--ptxas`
/// argument, else `/usr/local/cuda/bin/ptxas`. No ptxas: Linux fails unless the caller allows it; other systems skip
/// and say so. The image is stamped only after the check.
fn register_check(ptx: &Path, table: &Path, given: Option<PathBuf>, allow_missing_ptxas: bool) -> Result<(), String> {
    if let Some(p) = given.as_ref().filter(|p| !p.exists()) { return Err(format!("INKLING REGISTER CHECK FAIL: no ptxas at {}", p.display())); }
    let ptxas = given.or_else(|| Some(PathBuf::from("/usr/local/cuda/bin/ptxas"))).filter(|p| p.exists());
    let Some(ptxas) = ptxas else {
        if allow_missing_ptxas || !cfg!(target_os = "linux") { eprintln!("INKLING REGISTER CHECK SKIPPED: no ptxas on this machine"); return Ok(()); }
        return Err("INKLING REGISTER CHECK FAIL: no ptxas on this machine; the Inkling image is not stamped unchecked".into());
    };
    let out = Command::new(&ptxas).args(["--gpu-name=sm_121a", "-m64", "-O3", "-v", "-o"]).arg(ptx.with_extension("cubin")).arg(ptx)
        .output().map_err(|e| format!("cannot run ptxas: {e}"))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() { return Err(format!("INKLING REGISTER CHECK FAIL: ptxas rejected the image\n{text}")); }
    let number_before = |line: &str, tail: &str| -> Option<u64> {
        let end = line.find(tail)?;
        line[..end].trim_end().rsplit(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    // (entry, registers, shared bytes, stack frame bytes, spill bytes)
    let mut measured: Vec<(String, u64, u64, u64, u64)> = vec![];
    let (mut name, mut stack, mut spill) = (None::<String>, 0u64, 0u64);
    for line in text.lines() {
        if let Some(rest) = line.split("Compiling entry function '").nth(1) { name = rest.split('\'').next().map(str::to_owned); stack = 0; spill = 0; }
        else if line.contains("bytes spill stores") {
            stack = number_before(line, "bytes stack frame").unwrap_or(0);
            spill = ["bytes spill stores", "bytes spill loads"].iter().filter_map(|t| number_before(line, t)).sum();
        } else if let (Some(entry), Some(rest)) = (name.as_ref(), line.split("Used ").nth(1)) {
            let registers = rest.split(' ').next().and_then(|n| n.parse().ok()).ok_or("INKLING REGISTER CHECK FAIL: cannot read a ptxas register line")?;
            measured.push((entry.clone(), registers, number_before(line, "bytes smem").unwrap_or(0), stack, spill));
            name = None;
        }
    }
    let rows: Vec<(String, [u64; 4])> = fs::read_to_string(table).map_err(|e| format!("cannot read {}: {e}", table.display()))?
        .lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let n = |s: &str| s.parse::<u64>().map_err(|_| l.to_string());
            match f.as_slice() { [entry, a, b, c, d] => Ok((entry.to_string(), [n(a)?, n(b)?, n(c)?, n(d)?])), _ => Err(l.to_string()) }
        }).collect::<Result<_, String>>().map_err(|l| format!("INKLING REGISTER CHECK FAIL: bad table row: {l}"))?;
    let (mut failures, mut report) = (vec![], String::new());
    for (entry, registers, shared, stack, spill) in &measured {
        let Some((_, [threads, max_registers, max_shared, max_stack])) = rows.iter().find(|r| &r.0 == entry) else { failures.push(format!("{entry}: no row in the table")); continue; };
        let by_registers = 65_536 / (registers * threads).max(1);
        report += &format!("{entry} threads={threads} registers={registers} (max {max_registers}) shared={shared} (max {max_shared}) stack_bytes={stack} (max {max_stack}) spill_bytes={spill} blocks_per_sm={}\n",
            by_registers.min(if *shared == 0 { u64::MAX } else { 101_376 / shared }).min(1_536 / threads));
        if registers > max_registers { failures.push(format!("{entry}: {registers} registers, the table allows {max_registers}")); }
        if shared > max_shared { failures.push(format!("{entry}: {shared} shared bytes, the table allows {max_shared}")); }
        if stack > max_stack { failures.push(format!("{entry}: {stack} stack bytes, the table allows {max_stack}")); }
        if *spill != 0 { failures.push(format!("{entry}: {spill} spill bytes")); }
        if by_registers == 0 { failures.push(format!("{entry}: {registers} registers x {threads} threads cannot launch")); }
    }
    for (entry, _) in &rows { if !measured.iter().any(|m| &m.0 == entry) { failures.push(format!("{entry}: in the table, not in the image")); } }
    fs::write(ptx.with_extension("registers.txt"), &report).map_err(|e| format!("cannot write the register receipt: {e}"))?;
    eprint!("{report}");
    if failures.is_empty() { eprintln!("INKLING REGISTER CHECK PASS entries={} ptxas={}", measured.len(), ptxas.display()); Ok(()) }
    else { Err(format!("INKLING REGISTER CHECK FAIL\n{}", failures.join("\n"))) }
}
#[cfg(test)]
mod tests {
    use super::inkling_stamp;
    use std::{fs, path::{Path, PathBuf}};
    /// Every file that a source of the Inkling crate includes by `#[path = "..."]` is in the stamp's source list, so an
    /// edit to it rebuilds the image (review REVSCORES: `block_scores.rs` and `staged_narrow.rs` were missing, and a
    /// stale PTX could run). The path is resolved from the folder of the file that holds the line.
    #[test]
    fn every_file_the_inkling_crate_includes_by_path_is_in_the_stamp() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let listed: Vec<PathBuf> = inkling_stamp::source_files(&root).unwrap().iter().map(|f| fs::canonicalize(f).unwrap()).collect();
        let (mut found, mut missing) = (Vec::new(), Vec::new());
        for entry in fs::read_dir(root.join("kernels-inkling/src")).unwrap() {
            let file = entry.unwrap().path();
            if file.extension().is_none_or(|e| e != "rs") { continue; }
            for line in fs::read_to_string(&file).unwrap().lines() {
                let Some(rest) = line.trim().strip_prefix("#[path") else { continue };
                let target = rest.split('"').nth(1).unwrap_or_else(|| panic!("{}: a #[path] line without a quoted path: {line}", file.display()));
                let path = fs::canonicalize(file.parent().unwrap().join(target)).unwrap_or_else(|e| panic!("{}: {target}: {e}", file.display()));
                if !listed.contains(&path) { missing.push(path.display().to_string()); }
                found.push(path);
            }
        }
        assert!(found.len() >= 12, "only {} #[path] lines found in kernels-inkling/src", found.len());
        for name in ["src/block_scores.rs", "src/staged_narrow.rs"] {
            assert!(found.contains(&fs::canonicalize(root.join(name)).unwrap()), "{name} is no longer included by path");
        }
        assert!(missing.is_empty(), "files included by #[path] but not in inkling_stamp::source_files: {missing:?}");
    }
}
