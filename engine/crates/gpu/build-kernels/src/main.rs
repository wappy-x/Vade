//! Build the NVPTX kernels with the pinned nightly, then stamp both PTX files.
//! Run from anywhere in the workspace with `cargo kernels`.
use std::{env, ffi::OsStr, fs, path::{Path, PathBuf}, process::{self, Command, Stdio}};
#[path = "../../ptx_stamp.rs"]
#[allow(dead_code)] // `verify` belongs to the host build scripts.
mod ptx_stamp;

const NIGHTLY: &str = "nightly-2026-10-01";
const TARGET: &str = "nvptx64-nvidia-cuda";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // No flag builds the two images every Linux engine build embeds: the production image, then the tensor-core
    // image. So `cargo kernels` stays the one kernel step of every build recipe (the installer runs only that).
    let (variants, allow_missing_ptxas): (&[ptx_stamp::Variant], bool) = match env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => (&[ptx_stamp::Variant::Production, ptx_stamp::Variant::Tc], false),
        [flag] if flag == "--fp4" => (&[ptx_stamp::Variant::Fp4], false),
        [flag] if flag == "--tc" => (&[ptx_stamp::Variant::Tc], false),
        [flag] if flag == "--allow-missing-ptxas" => (&[ptx_stamp::Variant::Production, ptx_stamp::Variant::Tc], true),
        _ => return Err("usage: cargo kernels [--fp4 | --tc | --allow-missing-ptxas]".into()),
    };
    // The environment is read here, once, for every image of the run.
    // Paths and stamp names are relative to crates/gpu. Prefer the directory `cargo run`
    // passes: a copied or moved tree can reuse a binary that names the old checkout.
    let manifest = env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from).unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into());
    let bin = env::var_os("KERNEL_BIN").filter(|bin| !bin.is_empty());
    let search = env::var_os("PATH").unwrap_or_default();
    variants.iter().try_for_each(|&variant| build(variant, &manifest, bin.as_deref(), &search, allow_missing_ptxas))
}

/// Build, copy and stamp one image.
fn build(variant: ptx_stamp::Variant, manifest: &Path, bin: Option<&OsStr>, search: &OsStr, allow_missing_ptxas: bool) -> Result<(), String> {
    let (fp4, tc) = (variant == ptx_stamp::Variant::Fp4, variant == ptx_stamp::Variant::Tc);
    let gpu = manifest.parent().unwrap();
    env::set_current_dir(gpu).map_err(|e| format!("cannot enter {}: {e}", gpu.display()))?;
    let mut cargo = match bin {
        Some(bin) => {
            let mut path = bin.to_os_string();
            path.push(":");
            path.push(search);
            let mut cargo = Command::new(Path::new(bin).join("cargo"));
            cargo.env("PATH", path);
            cargo
        }
        None => {
            let installed = Command::new("rustup").args(["target", "list", "--toolchain", NIGHTLY, "--installed"])
                .stderr(Stdio::null()).output()
                .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).lines().any(|line| line == TARGET));
            if !installed {
                return Err(format!("Missing nightly NVPTX toolchain/target. Run:\nrustup toolchain install {NIGHTLY} \
                    --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target {TARGET}"));
            }
            let mut cargo = Command::new("cargo");
            cargo.arg(format!("+{NIGHTLY}"));
            cargo
        }
    };
    // build-std fetches compiler dependencies on the first build. Keep each variant
    // at its workspace path, even with CARGO_TARGET_DIR; FP4 never touches production.
    // `cargo run` points CARGO at the host toolchain; the nightly build sets its own.
    // The block-scaled tensor-core steps (FP4, and MXFP8 since 2026-11-16) need the
    // architecture-specific target, so every variant has it (`ptx_stamp.rs`); PTX 8.8
    // is supported by the installed CUDA driver.
    let mut build = vec!["build", "-Z", "build-std=core", "-p", variant.package()];
    if !variant.features().is_empty() { build.extend(["--features", variant.features()]); }
    build.extend(["--release", "--target", TARGET, "--target-dir", variant.target_dir()]);
    let status = cargo.args(build)
        .env("RUSTFLAGS", variant.flags()).env_remove("CARGO")
        .status().map_err(|e| format!("cannot run cargo: {e}"))?;
    if !status.success() { process::exit(status.code().unwrap_or(1)); }
    let (kernels, layer) = (variant.ptx_path(Path::new("."), false), variant.ptx_path(Path::new("."), true));
    if fp4 {
        fs::copy(kernels.with_file_name("spark_rust_kernels.ptx"), &kernels)
            .map_err(|e| format!("cannot copy FP4 PTX: {e}"))?;
    }
    fs::copy(&kernels, &layer).map_err(|e| format!("cannot copy {}: {e}", kernels.display()))?;
    // The tensor-core stamp keeps the bytes, so the check runs on the exact image it would stamp.
    let check = |ptx: &Path| if tc { register_check(ptx, Path::new("kernels-tc/REGISTERS.txt"), search, allow_missing_ptxas) } else { Ok(()) };
    publish(&kernels, &layer, check, |ptx| ptx_stamp::stamp(Path::new("."), ptx, variant))?;
    if fp4 { eprintln!("FP4 PTX is isolated in target/fp4; host opt-in: SPARK_PTX_VARIANT=fp4 cargo build"); }
    Ok(())
}

/// Stamp both PTX files only after `check` passes on the layer copy. The old stamps go first: a failed check
/// leaves no stamp, so no host build can embed the image, not even one whose bytes match an earlier stamp.
fn publish(kernels: &Path, layer: &Path, check: impl FnOnce(&Path) -> Result<(), String>,
    stamp: impl Fn(&Path) -> std::io::Result<()>) -> Result<(), String> {
    for ptx in [kernels, layer] {
        let old = ptx.with_extension("ptx.sha256");
        match fs::remove_file(&old) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("cannot remove the old stamp {}: {e}", old.display())),
            _ => {}
        }
    }
    check(layer)?;
    let mut stamped = Vec::new();
    for ptx in [kernels, layer] {
        if let Err(e) = stamp(ptx) {
            for done in stamped { let _ = fs::remove_file(done); }
            return Err(format!("cannot stamp {}: {e}", ptx.display()));
        }
        stamped.push(ptx.with_extension("ptx.sha256"));
    }
    Ok(())
}

/// Kernel kit, part 2: the register check of the tensor-core image at build time. Compiles the stamped PTX with
/// ptxas (`sm_121a`, the release arguments) and holds every entry to its row of `kernels-tc/REGISTERS.txt`.
/// A machine that is not Linux and has no ptxas (the Mac) skips the check and says so. A Linux build with no ptxas fails
/// unless its caller explicitly permits an unavailable compiler, as the GitHub-hosted Linux job does.
fn register_check(ptx: &Path, table: &Path, search: &OsStr, allow_missing_ptxas: bool) -> Result<(), String> {
    let ptxas = match find_ptxas(&[PathBuf::from("ptxas"), PathBuf::from("/usr/local/cuda/bin/ptxas")], search)? {
        Some(bin) => bin,
        None if allow_missing_ptxas => {
            eprintln!("TC REGISTER CHECK SKIPPED: no ptxas on this machine (the caller allowed a missing compiler); build on a Spark for the check");
            return Ok(());
        }
        // A Linux build is a build for a Spark: an unchecked image must not be stamped there.
        None if cfg!(target_os = "linux") => return Err("TC REGISTER CHECK FAIL: no ptxas on this machine (PATH or /usr/local/cuda/bin); the tensor-core image is not stamped unchecked".into()),
        None => { eprintln!("TC REGISTER CHECK SKIPPED: no ptxas on this machine (not a Linux build); build on a Spark for the check"); return Ok(()); }
    };
    let out = Command::new(&ptxas).args(["--gpu-name=sm_121a", "-m64", "-O3", "-v", "-o"]).arg(ptx.with_extension("cubin")).arg(ptx)
        .output().map_err(|e| format!("cannot run ptxas: {e}"))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() { return Err(format!("TC REGISTER CHECK FAIL: ptxas rejected the image\n{text}")); }
    let number_before = |line: &str, tail: &str| -> Option<u64> {
        let end = line.find(tail)?;
        line[..end].trim_end().rsplit(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    // (entry, registers, shared bytes, stack + spill bytes)
    let mut measured: Vec<(String, u64, u64, u64)> = vec![];
    let (mut name, mut spill) = (None::<String>, 0u64);
    for line in text.lines() {
        if let Some(rest) = line.split("Compiling entry function '").nth(1) { name = rest.split('\'').next().map(str::to_owned); spill = 0; }
        else if line.contains("bytes spill stores") {
            spill = ["bytes stack frame", "bytes spill stores", "bytes spill loads"].iter().filter_map(|t| number_before(line, t)).sum();
        } else if let (Some(entry), Some(rest)) = (name.as_ref(), line.split("Used ").nth(1)) {
            let registers = rest.split(' ').next().and_then(|n| n.parse().ok()).ok_or("TC REGISTER CHECK FAIL: cannot read a ptxas register line")?;
            measured.push((entry.clone(), registers, number_before(line, "bytes smem").unwrap_or(0), spill));
            name = None;
        }
    }
    let rows = fs::read_to_string(table).map_err(|e| format!("cannot read {}: {e}", table.display()))?;
    let rows: Vec<(String, u64, u64, u64)> = rows.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        match f.as_slice() {
            [entry, threads, registers, shared] => Ok((entry.to_string(), threads.parse().map_err(|_| l)?, registers.parse().map_err(|_| l)?, shared.parse().map_err(|_| l)?)),
            _ => Err(l),
        }
    }).collect::<Result<_, &str>>().map_err(|l| format!("TC REGISTER CHECK FAIL: bad table row: {l}"))?;
    let mut failures = vec![];
    let mut report = String::new();
    for (entry, registers, shared, spill) in &measured {
        let Some((_, threads, max_registers, max_shared)) = rows.iter().find(|r| &r.0 == entry) else { failures.push(format!("{entry}: no row in the table")); continue; };
        // Blocks an SM: 65,536 registers and (design cap) 101,376 shared bytes; at most 1,536 threads.
        let by_registers = 65_536 / (registers * threads).max(1);
        let per_sm = by_registers.min(if *shared == 0 { u64::MAX } else { 101_376 / shared }).min(1_536 / threads);
        report += &format!("{entry} threads={threads} registers={registers} (max {max_registers}) shared={shared} (max {max_shared}) spill_bytes={spill} blocks_per_sm={per_sm}\n");
        if registers > max_registers { failures.push(format!("{entry}: {registers} registers, the table allows {max_registers}")); }
        if shared > max_shared { failures.push(format!("{entry}: {shared} shared bytes, the table allows {max_shared}")); }
        if *spill != 0 { failures.push(format!("{entry}: {spill} stack or spill bytes")); }
        if by_registers == 0 { failures.push(format!("{entry}: {registers} registers x {threads} threads cannot launch")); }
    }
    for (entry, ..) in &rows { if !measured.iter().any(|m| &m.0 == entry) { failures.push(format!("{entry}: in the table, not in the image")); } }
    fs::write(ptx.with_extension("registers.txt"), &report).map_err(|e| format!("cannot write the register receipt: {e}"))?;
    eprint!("{report}");
    if failures.is_empty() { eprintln!("TC REGISTER CHECK PASS entries={} ptxas={}", measured.len(), ptxas.display()); Ok(()) }
    else { Err(format!("TC REGISTER CHECK FAIL\n{}", failures.join("\n"))) }
}

/// Probe in order. Absence is decided by looking for the file, never by the start error (a script whose
/// interpreter is missing also fails to start with "not found"). A bare name is looked up in each `search` (PATH)
/// folder; a path is looked at directly. Neither follows links, so a dangling link counts as present. Only an
/// absent candidate permits trying the next one; a present one that cannot start or whose `--version` fails is an
/// error, even when a later candidate works.
fn find_ptxas(candidates: &[PathBuf], search: &OsStr) -> Result<Option<PathBuf>, String> {
    // Present only when the look succeeds and shows an entry. Any error from looking (not found, a folder that
    // cannot be searched, a PATH entry that is not a folder, and so on) means no entry here: try the next one.
    let present = |path: &Path| path.symlink_metadata().is_ok();
    for candidate in candidates {
        let found = if candidate.parent() == Some(Path::new("")) {
            // An empty PATH entry is the current folder, as for the shell.
            env::split_paths(search).map(|dir| if dir.as_os_str().is_empty() { PathBuf::from(".") } else { dir })
                .map(|dir| dir.join(candidate)).find(|path| present(path))
        } else {
            present(candidate).then(|| candidate.clone())
        };
        let Some(bin) = found else { continue };
        let reason = match Command::new(&bin).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status() {
            Ok(status) if status.success() => return Ok(Some(bin)),
            Ok(status) => status.to_string(),
            Err(error) => error.to_string(),
        };
        return Err(format!("TC REGISTER CHECK FAIL: ptxas at {} did not run ({reason}); the tensor-core image is not stamped unchecked", bin.display()));
    }
    Ok(None)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{os::unix::fs::{PermissionsExt, symlink}, sync::{Mutex, MutexGuard, atomic::{AtomicU64, Ordering}}};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    /// One test at a time writes and runs scripts: on Linux a script still open for writing in a child that another
    /// test thread forked fails to start ("text file busy"), which would read as a broken compiler.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    struct Folder(PathBuf, #[allow(dead_code)] MutexGuard<'static, ()>);
    impl Folder {
        fn new() -> Self {
            let lock = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
            let path = env::temp_dir().join(format!("ptxas-probe-{}-{}", process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir(&path).unwrap();
            Self(path, lock)
        }
        fn script(&self, name: &str, code: i32, mode: u32) -> PathBuf { self.file(name, &format!("#!/bin/sh\nexit {code}\n"), mode) }
        fn file(&self, name: &str, text: &str, mode: u32) -> PathBuf {
            let path = self.0.join(name);
            if let Some(dir) = path.parent() { fs::create_dir_all(dir).unwrap(); }
            fs::write(&path, text).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
            path
        }
        /// A PATH of these subfolders (created empty if missing).
        fn search(&self, dirs: &[&str]) -> std::ffi::OsString {
            env::join_paths(dirs.iter().map(|d| { let dir = self.0.join(d); fs::create_dir_all(&dir).unwrap(); dir })).unwrap()
        }
        fn broken(&self, path: &Path, reason: &str) -> String {
            format!("TC REGISTER CHECK FAIL: ptxas at {} did not run ({reason}); the tensor-core image is not stamped unchecked", path.display())
        }
    }
    impl Drop for Folder { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    #[test]
    fn failed_check_leaves_no_stamp() {
        // Regression: the image was stamped before the register check, so a failed check left a stamped image.
        let folder = Folder::new();
        let (kernels, layer) = (folder.file("k.ptx", "ptx", 0o600), folder.file("l.ptx", "ptx", 0o600));
        let old = [folder.file("k.ptx.sha256", "old", 0o600), folder.file("l.ptx.sha256", "old", 0o600)];
        let error = publish(&kernels, &layer, |_| Err("TC REGISTER CHECK FAIL".into()), |_| panic!("stamped")).unwrap_err();
        assert_eq!(error, "TC REGISTER CHECK FAIL");
        assert!(old.iter().all(|stamp| !stamp.exists()), "an old stamp is left");
    }
    #[test]
    fn passed_check_stamps_both_after_the_check() {
        let folder = Folder::new();
        let (kernels, layer) = (folder.file("k.ptx", "ptx", 0o600), folder.file("l.ptx", "ptx", 0o600));
        let order = std::cell::RefCell::new(vec![]);
        publish(&kernels, &layer, |ptx| { order.borrow_mut().push(format!("check {}", ptx.display())); Ok(()) },
            |ptx| { order.borrow_mut().push(format!("stamp {}", ptx.display())); Ok(()) }).unwrap();
        assert_eq!(order.into_inner(), [format!("check {}", layer.display()), format!("stamp {}", kernels.display()), format!("stamp {}", layer.display())]);
    }
    #[test]
    fn all_candidates_absent() {
        // Pins the existing missing-compiler behavior without depending on PATH.
        let folder = Folder::new();
        assert_eq!(find_ptxas(&[folder.0.join("one"), folder.0.join("two")], folder.0.as_os_str()).unwrap(), None);
    }
    #[test]
    fn failed_version_is_broken_even_with_a_working_fallback() {
        // Regression: the old probe skipped this broken compiler.
        let folder = Folder::new();
        let bad = folder.script("bad", 1, 0o700);
        let good = folder.script("good", 0, 0o700);
        let error = find_ptxas(&[bad.clone(), good], folder.0.as_os_str()).unwrap_err();
        assert_eq!(error, format!("TC REGISTER CHECK FAIL: ptxas at {} did not run (exit status: 1); the tensor-core image is not stamped unchecked", bad.display()));
    }
    #[test]
    fn non_executable_is_broken() {
        // Regression: the old probe treated permission denied as absence.
        let folder = Folder::new();
        let path = folder.script("blocked", 0, 0o600);
        let error = find_ptxas(std::slice::from_ref(&path), folder.0.as_os_str()).unwrap_err();
        assert!(error.contains(&format!("ptxas at {} did not run (", path.display())), "{error}");
        assert!(error.contains("Permission denied"), "{error}");
    }
    #[test]
    fn first_working_candidate_is_chosen_after_absence() {
        // Pins the existing ordered fallback and successful version behavior.
        let folder = Folder::new();
        let good = folder.script("good", 0, 0o700);
        let bad = folder.script("unused", 1, 0o700);
        assert_eq!(find_ptxas(&[folder.0.join("absent"), good.clone(), bad], folder.0.as_os_str()).unwrap(), Some(good));
    }
    #[test]
    fn missing_interpreter_is_broken_not_absent() {
        // Regression: starting this script fails with "not found", which the old probe read as no ptxas (SKIPPED).
        let folder = Folder::new();
        let search = folder.search(&["empty", "bin"]);
        let script = folder.file("bin/ptxas", "#!/nonexistent/interpreter\n", 0o700);
        let good = folder.script("good", 0, 0o700);
        let error = find_ptxas(&[PathBuf::from("ptxas"), good.clone()], &search).unwrap_err();
        assert!(error.starts_with(&format!("TC REGISTER CHECK FAIL: ptxas at {} did not run (", script.display())), "{error}");
        let error = find_ptxas(&[script.clone(), good], &search).unwrap_err();
        assert!(error.starts_with(&format!("TC REGISTER CHECK FAIL: ptxas at {} did not run (", script.display())), "{error}");
    }
    #[test]
    fn dangling_link_is_broken_not_absent() {
        // Regression: a link to nothing also fails to start with "not found"; the entry exists, so it is broken.
        let folder = Folder::new();
        let search = folder.search(&["bin"]);
        let link = folder.0.join("bin/ptxas");
        symlink(folder.0.join("nowhere"), &link).unwrap();
        let error = find_ptxas(&[PathBuf::from("ptxas")], &search).unwrap_err();
        assert!(error.starts_with(&format!("TC REGISTER CHECK FAIL: ptxas at {} did not run (", link.display())), "{error}");
        let error = find_ptxas(std::slice::from_ref(&link), &search).unwrap_err();
        assert!(error.starts_with(&format!("TC REGISTER CHECK FAIL: ptxas at {} did not run (", link.display())), "{error}");
    }
    #[test]
    fn bare_name_absent_from_every_path_folder() {
        // Pins absence through the PATH lookup: no folder has an entry named ptxas.
        let folder = Folder::new();
        let search = folder.search(&["one", "two"]);
        folder.script("one/not-ptxas", 0, 0o700);
        assert_eq!(find_ptxas(&[PathBuf::from("ptxas"), folder.0.join("absent")], &search).unwrap(), None);
    }
    #[test]
    fn bare_name_found_in_a_later_path_folder() {
        // Pins the PATH order: the first folder holding ptxas is used, and a working one is chosen.
        let folder = Folder::new();
        let search = folder.search(&["empty", "bin", "later"]);
        let good = folder.script("bin/ptxas", 0, 0o700);
        let bad = folder.script("later/ptxas", 1, 0o700);
        assert_eq!(find_ptxas(&[PathBuf::from("ptxas"), bad.clone()], &search).unwrap(), Some(good.clone()));
        let error = find_ptxas(&[PathBuf::from("ptxas")], &folder.search(&["later", "bin"])).unwrap_err();
        assert_eq!(error, folder.broken(&bad, "exit status: 1"));
    }
    /// Puts a folder's mode back on drop, so cleanup can remove it even when an assertion fails.
    struct Mode(PathBuf, u32);
    impl Drop for Mode { fn drop(&mut self) { let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(self.1)); } }
    fn locked(folder: &Folder, dir: &str) -> Mode {
        let path = folder.0.join(dir);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
        Mode(path, 0o700)
    }
    #[test]
    fn unsearchable_path_folder_is_skipped_for_a_later_working_one() {
        // Regression: a PATH folder without search permission read as a present, broken compiler.
        let folder = Folder::new();
        let search = folder.search(&["locked", "bin"]);
        let good = folder.script("bin/ptxas", 0, 0o700);
        let _mode = locked(&folder, "locked");
        assert_eq!(find_ptxas(&[PathBuf::from("ptxas")], &search).unwrap(), Some(good));
    }
    #[test]
    fn only_an_unsearchable_path_folder_is_absent() {
        // A PATH whose only folder cannot be searched has no ptxas.
        let folder = Folder::new();
        let search = folder.search(&["locked"]);
        let _mode = locked(&folder, "locked");
        assert_eq!(find_ptxas(&[PathBuf::from("ptxas"), folder.0.join("absent")], &search).unwrap(), None);
    }
    #[test]
    fn path_entry_that_is_a_file_is_skipped_for_a_later_working_one() {
        // A PATH entry that is a plain file, not a folder, holds no ptxas.
        let folder = Folder::new();
        let search = folder.search(&["bin"]);
        let file = folder.file("plain", "not a folder\n", 0o600);
        let good = folder.script("bin/ptxas", 0, 0o700);
        let search = env::join_paths([file].into_iter().chain(env::split_paths(&search))).unwrap();
        assert_eq!(find_ptxas(&[PathBuf::from("ptxas")], &search).unwrap(), Some(good));
    }
    #[test]
    fn failed_second_stamp_removes_first_stamp() {
        let folder = Folder::new();
        let kernels = folder.file("a.ptx", "k", 0o600);
        let layer = folder.file("b.ptx", "l", 0o600);
        let err = publish(&kernels, &layer, |_| Ok(()), |ptx| {
            if ptx == kernels { fs::write(ptx.with_extension("ptx.sha256"), "ok") }
            else { Err(std::io::Error::other("disk full")) }
        }).unwrap_err();
        assert!(err.contains("cannot stamp"), "{err}");
        assert!(!kernels.with_extension("ptx.sha256").exists());
        assert!(!layer.with_extension("ptx.sha256").exists());
    }
}
