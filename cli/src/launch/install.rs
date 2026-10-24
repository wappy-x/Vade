//! Local artifacts are published together after verification. The caller holds
//! the launcher lock; no model, system setting or external source tree is changed.
use super::build_id;
use crate::runner::{Exec, Runner};
use crate::text::{self, Fields};
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

const FILES: [(&str, &str); 3] = [
    ("engine", "flash-serve"),
    ("ptx", "flash.ptx"),
    ("harness", "vade-harness"),
];
const BUNDLE_MANIFEST: &str = "launcher-bundle.conf";

/// Metadata-only reuse keeps attaching independent of artifact size. The cold
/// start worker verifies engine/PTX hashes again before executing the engine.
pub(super) fn load(root: &Path) -> Result<Option<Fields>, String> {
    Ok(match inspect(root)? { Installed::Ready(fields) => Some(fields), _ => None })
}

#[derive(Debug, PartialEq)]
enum Installed { Missing, Changed(&'static str), Ready(Fields) }

fn inspect(root: &Path) -> Result<Installed, String> {
    let path = root.join("stage/install.conf");
    let Some(fields) = read_manifest(&path)? else { return Ok(Installed::Missing); };
    if fields.get("format").map(String::as_str) != Some("vade-local-install-v1")
        || fields.get("kind").map(String::as_str) != Some("flash")
    {
        return Err("Installed artifact record is invalid; keep it for inspection.".into());
    }
    let id = text::required(&fields, "generation")?;
    if !text::hex(id, 24) {
        return Err("Installed generation is invalid; keep it for inspection.".into());
    }
    let generation = root.join("stage/builds").join(id);
    for (key, name) in FILES {
        let path = generation.join(name);
        if fields.get(key).map(String::as_str) != path.to_str()
            || !fields.get(&format!("{key}_sha256")).is_some_and(|s| text::hex(s, 64))
        {
            return Err("Installed artifact paths or checksums are invalid; keep the record for inspection.".into());
        }
        text::reject_symlinks(&path)?;
        let Some(stamp) = fingerprint(&path)? else { return Ok(Installed::Changed(name)); };
        if fields.get(&format!("{key}_stamp")) != Some(&stamp) {
            return Ok(Installed::Changed(name));
        }
    }
    Ok(Installed::Ready(fields))
}

pub(super) fn ensure<R: Runner>(root: &Path, runner: &mut R) -> Result<Fields, String> {
    match inspect(root)? {
        Installed::Ready(installed) => {
            crate::output::line(format_args!("Install: using the saved local artifacts."))?;
            return Ok(installed);
        }
        Installed::Changed(name) => crate::output::line(format_args!(
            "Install: staged {name} is missing or changed since it was installed; replacing the staged files."))?,
        Installed::Missing => (),
    }
    refresh(root, runner)
}

/// Publish a fresh generation only after all replacement artifacts pass verification.
pub(super) fn refresh<R: Runner>(root: &Path, runner: &mut R) -> Result<Fields, String> {
    let bundle = bundle_path()?;
    let stage = root.join("stage");
    text::private_dir(&stage)?;
    if let Some(bundle) = bundle {
        crate::output::line(format_args!("Install: checking the supplied local bundle."))?;
        return publish(root, &bundle, Some(bundle_fields(&bundle)?));
    }
    let source = source_path()?;
    crate::output::line(format_args!("Install: building the local engine and chat program."))?;
    let snapshot = snapshot(&source, &stage, runner)?;
    let target = stage.join("build-target");
    text::private_dir(&target)?;
    // The kernel builder intentionally uses engine/target even when the host
    // target directory is overridden. Both live in the private install stage.
    build(root, runner, &snapshot.join("engine"), &target, "Build GPU kernels", &["kernels"])?;
    build(root, runner, &snapshot.join("engine"), &target, "Build engine", &[
        "build", "--release", "--offline", "-p", "spark-tp", "--bin", "flash-serve",
    ])?;
    build(root, runner, &snapshot.join("harness"), &target, "Build chat", &[
        "build", "--release", "--offline", "--bin", "vade-harness",
    ])?;
    let assembled = stage.join(format!("assembled-{}", text::random_hex(12)?));
    text::private_dir(&assembled)?;
    copy(&target.join("release/flash-serve"), &assembled.join("flash-serve"), true)?;
    copy(&target.join("release/vade-harness"), &assembled.join("vade-harness"), true)?;
    copy(&snapshot.join("engine/target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx"),
        &assembled.join("flash.ptx"), false)?;
    let outcome = publish(root, &assembled, None);
    let _ = fs::remove_dir_all(&assembled);
    outcome
}

fn bundle_path() -> Result<Option<PathBuf>, String> {
    if let Some(path) = std::env::var_os("VADE_BUNDLE") {
        let path = PathBuf::from(path);
        if !path.is_absolute() || !path.is_dir() {
            return Err("VADE_BUNDLE must name an existing absolute bundle folder.".into());
        }
        return Ok(Some(path));
    }
    let executable = std::env::current_exe().map_err(|_| "Cannot find the installed vade program.")?;
    Ok(executable.parent().filter(|dir| dir.join(BUNDLE_MANIFEST).is_file()).map(Path::to_owned))
}

fn bundle_fields(bundle: &Path) -> Result<Fields, String> {
    let fields = read_manifest(&bundle.join(BUNDLE_MANIFEST))?
        .ok_or("The local bundle is missing launcher-bundle.conf.")?;
    if fields.get("format").map(String::as_str) != Some("vade-launch-bundle-v1")
        || fields.get("target").map(String::as_str) != Some("aarch64-unknown-linux-gnu")
    {
        return Err("The local bundle needs the Linux ARM64 launcher format.".into());
    }
    for (key, _) in FILES {
        if !fields.get(&format!("{key}_sha256")).is_some_and(|s| text::hex(s, 64)) {
            return Err("The local bundle is missing a SHA256 checksum.".into());
        }
    }
    Ok(fields)
}

fn source_path() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("VADE_SOURCE") {
        let path = PathBuf::from(path);
        if !path.is_absolute() || !is_source(&path) {
            return Err("VADE_SOURCE must name an absolute Vade checkout with engine and harness folders.".into());
        }
        return Ok(path);
    }
    let executable = std::env::current_exe().map_err(|_| "Cannot locate the vade program.")?;
    let cwd = std::env::current_dir().map_err(|_| "Open an existing folder and retry vade install.")?;
    executable.ancestors().chain(cwd.ancestors()).find(|p| is_source(p))
        .map(Path::to_owned)
        .ok_or_else(|| "Vade source or a local bundle is missing; run vade install from your Vade checkout.".into())
}

fn is_source(path: &Path) -> bool {
    path.join("engine/Cargo.toml").is_file() && path.join("harness/Cargo.toml").is_file()
}

fn git<R: Runner>(runner: &mut R, source: &Path, args: &[&str]) -> Result<String, String> {
    let mut exec = Exec::new("inspect local source", "git", args.iter().map(|s| (*s).into()).collect());
    exec.cwd = Some(source.into());
    let reply = runner.run(&exec)?;
    if !reply.success { return Err("Cannot inspect this Vade checkout with Git. Install git or repair the checkout, or set VADE_BUNDLE to a local launcher bundle, then run vade install.".into()); }
    Ok(reply.stdout)
}

fn snapshot<R: Runner>(source: &Path, stage: &Path, runner: &mut R) -> Result<PathBuf, String> {
    let probe = build_id::probe(source);
    let git_tree = probe == build_id::Probe::Git;
    let (revision, files) = if git_tree {
        let revision = git(runner, source, &["rev-parse", "HEAD"])?.trim().to_owned();
        if !text::hex(&revision, 40) { return Err("Vade source has no complete Git revision.".into()); }
        clean_source(runner, source)?;
        let files = git(runner, source, &["ls-files", "--cached", "-z", "--", "engine", "harness", "rust-toolchain.toml"])?;
        (revision, files.split('\0').filter(|s| !s.is_empty()).map(str::to_owned).collect())
    } else {
        let files = build_id::archive_files(source)?;
        let (revision, notice) = build_id::archive_id(probe, source, &files);
        if notice { crate::output::line(format_args!("{}", build_id::NOTICE))?; }
        (revision, files)
    };
    if files.is_empty() { return Err("The checkout has no tracked engine or harness source.".into()); }
    let target = stage.join("source").join(&revision);
    // Only a full id names immutable content; never reuse an unknown tree.
    if revision == build_id::UNKNOWN && target.exists() {
        text::reject_symlinks(&target)?;
        fs::remove_dir_all(&target).map_err(|_| "Cannot clear the previous unknown source snapshot.")?;
    }
    text::private_dir(&target)?;
    for name in &files {
        let path = Path::new(name);
        if path.components().any(|c| !matches!(c, Component::Normal(_)))
            || !(name.starts_with("engine/") || name.starts_with("harness/") || name == "rust-toolchain.toml")
        {
            return Err("The source inventory contains an invalid path.".into());
        }
        let input = source.join(path);
        let output = target.join(path);
        let meta = fs::symlink_metadata(&input).map_err(|_| "A tracked source file is missing or unreadable.")?;
        if !meta.is_file() { return Err("A tracked source file is not a regular file; restore the checkout before installing.".into()); }
        text::private_dir(output.parent().ok_or("Invalid source path.")?)?;
        copy(&input, &output, meta.permissions().mode() & 0o100 != 0)?;
    }
    if git_tree {
        clean_source(runner, source)?;
        if git(runner, source, &["rev-parse", "HEAD"])?.trim() != revision {
            return Err("The Vade checkout changed during install; retry after the checkout finishes.".into());
        }
    }
    // Provenance reflects the saved source revision; builds read this snapshot,
    // never the original checkout or a parent checkout's Git metadata.
    text::atomic_write(&target.join("engine/REVISION"), &format!("{revision}\n"))?;
    Ok(target)
}

fn clean_source<R: Runner>(runner: &mut R, source: &Path) -> Result<(), String> {
    if !git(runner, source, &["diff", "--name-only", "HEAD", "--", "engine", "harness", "rust-toolchain.toml"])?.trim().is_empty() {
        return Err("Vade source has unsaved changes; commit or restore the engine and harness source before installing.".into());
    }
    Ok(())
}

fn build<R: Runner>(root: &Path, runner: &mut R, cwd: &Path, target: &Path, label: &str, args: &[&str]) -> Result<(), String> {
    let started = Instant::now();
    crate::output::line(format_args!("{label}: starting."))?;
    let mut exec = Exec::new(label, &crate::app::rust_tool("cargo"), args.iter().map(|s| (*s).into()).collect());
    exec.cwd = Some(cwd.into());
    exec.timeout = Duration::from_secs(1800);
    exec.unset = ["RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_TARGET", "RUSTC", "RUSTDOC", "CARGO", "KERNEL_BIN", "SPARK_PTX_VARIANT", "SPARK_LAYER_PTX", "SPARK_CUBIN"]
        .into_iter().map(str::to_owned).collect();
    exec.env.insert("CARGO_TARGET_DIR".into(), target.to_string_lossy().into_owned());
    exec.env.insert("CARGO_NET_OFFLINE".into(), "true".into());
    exec.env.insert("RUSTUP_AUTO_INSTALL".into(), "0".into());
    exec.env.insert("RUSTUP_TOOLCHAIN".into(), "1.99.0".into());
    if let Some(home) = std::env::var_os("HOME") {
        let bin = PathBuf::from(home).join(".cargo/bin");
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![bin];
        paths.extend(std::env::split_paths(&path));
        let path = std::env::join_paths(paths).map_err(|_| "Cannot construct the compiler search path.")?;
        exec.env.insert("PATH".into(), path.to_string_lossy().into_owned());
    }
    // Persist the complete command before starting a supervisor. Its lifetime
    // pipe and session anchor own every compiler descendant after CLI loss.
    crate::build_job::prepare(root, &mut exec)?;
    let result = runner.run(&exec).and_then(|output| {
        if output.success { Ok(()) } else {
            Err(format!("{label} failed; check the retained build log and the existing pinned Rust toolchains."))
        }
    });
    if result.is_err() { crate::output::operation_failed(); }
    let timing = crate::output::line(format_args!("{label}: {:.3} s.", started.elapsed().as_secs_f64()));
    result?;
    timing
}

fn publish(root: &Path, input: &Path, expected: Option<Fields>) -> Result<Fields, String> {
    let stage = root.join("stage/builds");
    text::private_dir(&stage)?;
    let id = text::random_hex(12)?;
    let temporary = stage.join(format!("{id}.part"));
    let destination = stage.join(&id);
    text::private_dir(&temporary)?;
    let mut fields = Fields::from([
        ("format".into(), "vade-local-install-v1".into()),
        ("generation".into(), id),
        ("kind".into(), "flash".into()),
    ]);
    let result = (|| {
        for (key, name) in FILES {
            let source = input.join(name);
            let digest = hash(&source)?;
            if expected.as_ref().is_some_and(|f| f.get(&format!("{key}_sha256")) != Some(&digest)) {
                return Err(format!("Local bundle {name} checksum does not match; nothing activated."));
            }
            if key != "ptx" { arm64_executable(&source)?; }
            let staged = temporary.join(name);
            copy(&source, &staged, key != "ptx")?;
            if hash(&staged)? != digest {
                return Err("A local artifact changed while copying; retry install after the copy finishes.".into());
            }
            fields.insert(key.into(), destination.join(name).to_string_lossy().into_owned());
            fields.insert(format!("{key}_sha256"), digest);
        }
        fs::rename(&temporary, &destination).map_err(|_| "Cannot publish the local artifacts; check free disk space.")?;
        File::open(&stage).and_then(|f| f.sync_all()).map_err(|_| "Cannot save the local artifact directory.")?;
        for (key, name) in FILES {
            let stamp = fingerprint(&destination.join(name))?.ok_or("Published artifact is missing.")?;
            fields.insert(format!("{key}_stamp"), stamp);
        }
        text::atomic_write(&root.join("stage/install.conf"), &text::encode(&fields))?;
        Ok(fields)
    })();
    if result.is_err() { let _ = fs::remove_dir_all(temporary); }
    result
}

#[cfg(test)]
pub(super) fn publish_test_bundle(root: &Path, input: &Path) -> Result<Fields, String> {
    publish(root, input, None)
}

fn read_manifest(path: &Path) -> Result<Option<Fields>, String> {
    text::reject_symlinks(path)?;
    let file = match super::open_regular(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("The local artifact record must be a readable regular file.".into()),
    };
    let mut value = String::new();
    file.take(16_385).read_to_string(&mut value).map_err(|_| "The local artifact record is not readable text.")?;
    if value.len() > 16_384 { return Err("The local artifact record is too large.".into()); }
    text::parse(&value).map(Some)
}

fn fingerprint(path: &Path) -> Result<Option<String>, String> {
    let meta = match fs::metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Cannot inspect installed artifacts; check their permissions.".into()),
    };
    if !meta.is_file() || meta.len() == 0 { return Ok(None); }
    Ok(Some(format!("{}:{}:{}:{}:{}:{}:{}:{}:{}", meta.dev(), meta.ino(), meta.len(), meta.mode(),
        meta.uid(), meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec())))
}

fn hash(path: &Path) -> Result<String, String> {
    let meta = fs::symlink_metadata(path).map_err(|_| "A local artifact is missing or unreadable.")?;
    if !meta.is_file() || meta.len() == 0 { return Err("A local artifact is empty or not a regular file.".into()); }
    crate::sha256::digest_reader(File::open(path).map_err(|_| "Cannot open a local artifact.")?)
        .map_err(|_| "Cannot verify a local artifact checksum.".into())
}

fn arm64_executable(path: &Path) -> Result<(), String> {
    let mut header = [0; 20];
    File::open(path).and_then(|mut f| f.read_exact(&mut header))
        .map_err(|_| "A local executable is truncated.")?;
    if &header[..6] != b"\x7fELF\x02\x01" || header[18..20] != [183, 0] {
        return Err("The local executables must be Linux ARM64 builds for this Spark.".into());
    }
    Ok(())
}

fn copy(input: &Path, output: &Path, executable: bool) -> Result<(), String> {
    text::reject_symlinks(output)?;
    fs::copy(input, output).map_err(|_| "Cannot stage local files; check available disk space and permissions.")?;
    fs::set_permissions(output, fs::Permissions::from_mode(if executable { 0o700 } else { 0o600 }))
        .map_err(|_| "Cannot make the staged artifact private.")?;
    File::open(output).and_then(|f| f.sync_all()).map_err(|_| "Cannot save a staged artifact.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PathBuf, PathBuf, Fields) {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-local-install-{}", text::random_hex(8).unwrap()));
        let bundle = root.join("bundle with spaces");
        text::private_dir(&bundle).unwrap();
        let mut expected = Fields::from([
            ("format".into(), "vade-launch-bundle-v1".into()),
            ("target".into(), "aarch64-unknown-linux-gnu".into()),
        ]);
        for (key, name) in FILES {
            let mut bytes = vec![0; 64];
            bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183;
            fs::write(bundle.join(name), bytes).unwrap();
            expected.insert(format!("{key}_sha256"), hash(&bundle.join(name)).unwrap());
        }
        text::atomic_write(&bundle.join(BUNDLE_MANIFEST), &text::encode(&expected)).unwrap();
        (root, bundle, expected)
    }

    // Answers a clean checkout's Git queries; `None` refuses every command.
    struct GitSource(Option<Vec<String>>);
    impl Runner for GitSource {
        fn run(&mut self, exec: &Exec) -> Result<crate::runner::Output, String> {
            let calls = self.0.as_mut().unwrap_or_else(|| panic!("Unexpected command {:?}", exec.args));
            assert_eq!(exec.program, "git");
            calls.push(exec.args[0].clone());
            let stdout = match exec.args[0].as_str() {
                "rev-parse" => format!("{}\n", "a".repeat(40)),
                "diff" => String::new(),
                "ls-files" => "engine/Cargo.toml\0harness/Cargo.toml\0".into(),
                _ => panic!("Unexpected Git command {:?}", exec.args),
            };
            Ok(crate::runner::Output { success: true, stdout })
        }
        fn interactive(&mut self, _: &Exec) -> Result<bool, String> { unreachable!() }
        fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
        fn pause(&mut self, _: Duration) { unreachable!() }
    }

    #[test]
    fn snapshot_build_id_comes_from_git_then_the_archive_id_then_unknown() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-build-id-{}", text::random_hex(8).unwrap()));
        let source = root.join("source");
        let stamp = std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let write = |name: &str, body: &str, time| {
            text::private_dir(source.join(name).parent().unwrap()).unwrap();
            fs::write(source.join(name), body).unwrap();
            File::options().write(true).open(source.join(name)).unwrap().set_modified(time).unwrap();
        };
        // Archive entries share the commit time; build output is newer.
        for (name, body) in [("engine/Cargo.toml", "engine"), ("harness/Cargo.toml", "harness"),
            ("rust-toolchain.toml", "toolchain"), (build_id::FILE, &format!("{}\n", "b".repeat(40)))] { write(name, body, stamp); }
        write("engine/target/kernels.ptx", "generated", stamp + Duration::from_secs(60));
        write("harness/target/release/vade-harness", "generated", stamp + Duration::from_secs(60));
        let run = |stage: &str, runner: &mut GitSource| {
            let writer = crate::output::testing::Capture::default();
            let bytes = writer.bytes.clone();
            let snapshot = crate::output::with_writer(writer, || snapshot(&source, &root.join(stage), runner)).unwrap();
            let revision = fs::read_to_string(snapshot.join("engine/REVISION")).unwrap();
            assert_eq!(snapshot, root.join(stage).join("source").join(revision.trim()));
            (revision.trim().to_owned(), String::from_utf8(bytes.take()).unwrap())
        };

        // Git present: Git answers and the build id file is never consulted.
        fs::create_dir(source.join(".git")).unwrap();
        let mut git = GitSource(Some(Vec::new()));
        assert_eq!(run("git", &mut git), ("a".repeat(40), String::new()));
        assert_eq!(git.0.unwrap(), ["rev-parse", "diff", "ls-files", "diff", "rev-parse"]);
        fs::remove_dir(source.join(".git")).unwrap();

        // Archive with a substituted id: no Git, no notice, generated output left behind.
        assert_eq!(run("archive", &mut GitSource(None)), ("b".repeat(40), String::new()));
        let copied = root.join("archive/source").join("b".repeat(40));
        assert_eq!(fs::read_to_string(copied.join("rust-toolchain.toml")).unwrap(), "toolchain");
        assert!(!copied.join("engine/target").exists() && !copied.join("harness/target").exists());

        // Archive without a substituted id, then an edited archive: unknown plus one notice line.
        let notice = format!("{}\n", build_id::NOTICE);
        write(build_id::FILE, "$Format:%H$\n", stamp);
        assert_eq!(run("raw", &mut GitSource(None)), (build_id::UNKNOWN.into(), notice.clone()));
        write(build_id::FILE, &format!("{}\n", "b".repeat(40)), stamp);
        write("harness/Cargo.toml", "edited", stamp + Duration::from_secs(1));
        assert_eq!(run("raw", &mut GitSource(None)), (build_id::UNKNOWN.into(), notice.clone()));
        // An unknown snapshot is rebuilt, never reused: a deleted file disappears.
        fs::remove_file(source.join("rust-toolchain.toml")).unwrap();
        fs::remove_file(source.join(build_id::FILE)).unwrap();
        assert_eq!(run("raw", &mut GitSource(None)), (build_id::UNKNOWN.into(), notice));
        let unknown = root.join("raw/source").join(build_id::UNKNOWN);
        assert_eq!(fs::read_to_string(unknown.join("harness/Cargo.toml")).unwrap(), "edited");
        assert!(!unknown.join("rust-toolchain.toml").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_publishes_atomically_and_reuse_detects_replaced_artifacts() {
        let (root, bundle, expected) = fixture();
        let home = root.join("home");
        assert!(load(&home).unwrap().is_none());
        let installed = publish(&home, &bundle, Some(expected)).unwrap();
        assert_eq!(load(&home).unwrap(), Some(installed.clone()));
        fs::write(&installed["harness"], b"changed").unwrap();
        assert!(load(&home).unwrap().is_none());
        assert_eq!(inspect(&home).unwrap(), Installed::Changed("vade-harness"));
        fs::remove_file(&installed["ptx"]).unwrap();
        assert_eq!(inspect(&home).unwrap(), Installed::Changed("flash.ptx"));
        assert_eq!(inspect(&root.join("fresh")).unwrap(), Installed::Missing);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_bundle_never_replaces_an_activation() {
        let (root, bundle, expected) = fixture();
        let home = root.join("home");
        let installed = publish(&home, &bundle, Some(expected.clone())).unwrap();
        fs::write(bundle.join("flash.ptx"), b"changed").unwrap();
        assert!(publish(&home, &bundle, Some(expected)).unwrap_err().contains("checksum"));
        assert_eq!(load(&home).unwrap(), Some(installed));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_refuses_wrong_target_and_wrong_executable() {
        let (root, bundle, mut expected) = fixture();
        expected.insert("target".into(), "x86_64-unknown-linux-gnu".into());
        text::atomic_write(&bundle.join(BUNDLE_MANIFEST), &text::encode(&expected)).unwrap();
        assert!(bundle_fields(&bundle).unwrap_err().contains("ARM64"));
        let mut bytes = fs::read(bundle.join("flash-serve")).unwrap(); bytes[18] = 62;
        fs::write(bundle.join("flash-serve"), bytes).unwrap();
        assert!(publish(&root.join("home"), &bundle, None).unwrap_err().contains("ARM64"));
        fs::remove_dir_all(root).unwrap();
    }

    struct ChildGuard(std::process::Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    fn child(mode: &str, root: &Path) -> std::process::Command {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "launch::install::tests::install_child", "--nocapture"])
            .env("VADE_TEST_INSTALL_MODE", mode)
            .env("VADE_TEST_INSTALL_ROOT", root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        command
    }
    fn bounded_status(child: &mut std::process::Child) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().unwrap() { return status; }
            assert!(Instant::now() < deadline, "install child did not finish before hang deadline");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    struct FailedBuild(bool);
    impl Runner for FailedBuild {
        fn run(&mut self, exec: &Exec) -> Result<crate::runner::Output, String> {
            assert!(exec.job.as_ref().unwrap().join("intent.conf").is_file());
            if self.0 { Err("Compiler could not start.".into()) }
            else { Ok(crate::runner::Output { success: false, stdout: String::new() }) }
        }
        fn interactive(&mut self, _: &Exec) -> Result<bool, String> { unreachable!() }
        fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
        fn pause(&mut self, _: Duration) { unreachable!() }
    }

    // This test is invoked in disposable subprocesses so FIFO regressions and
    // abrupt CLI death are bounded without leaving blocked test threads.
    #[test]
    fn install_child() {
        let Ok(mode) = std::env::var("VADE_TEST_INSTALL_MODE") else { return; };
        let root = PathBuf::from(std::env::var_os("VADE_TEST_INSTALL_ROOT").unwrap());
        match mode.as_str() {
            "build" => {
                let mut runner = crate::runner::SystemRunner { dry: false };
                build(&root, &mut runner, &root, &root.join("stage/build-target"), "Build engine", &["build"]).unwrap();
            }
            "output-failure" | "spawn-failure" => {
                let writer = crate::output::testing::Capture {
                    fail_on: Some((" s.", std::io::ErrorKind::BrokenPipe)),
                    ..Default::default()
                };
                crate::output::with_writer(writer, || {
                    let result = build(&root, &mut FailedBuild(mode == "spawn-failure"), &root,
                        &root.join("target"), "Build engine", &["build"]);
                    assert_eq!(crate::output::failure(), Some(std::io::ErrorKind::BrokenPipe));
                    // Mirror main's broken-pipe classification, then let the
                    // parent assert the observable nonzero process status.
                    std::process::exit(if result.is_err() && !crate::output::broken_pipe_success() { 1 } else { 0 });
                });
            }
            "bundle-fifo" => { assert!(bundle_fields(&root).unwrap_err().contains("record")); }
            "install-fifo" => { assert!(load(&root).unwrap_err().contains("record")); }
            _ => panic!("unexpected install child mode"),
        }
    }

    #[test]
    fn failed_build_keeps_nonzero_status_when_timing_stdout_breaks() {
        let _guard = crate::runner::process_test_lock();
        let (root, _, _) = fixture();
        for mode in ["output-failure", "spawn-failure"] {
            let mut process = ChildGuard(child(mode, &root).spawn().unwrap());
            assert_eq!(bounded_status(&mut process.0).code(), Some(1));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fifo_bundle_and_install_records_fail_within_bounded_subprocesses() {
        let _guard = crate::runner::process_test_lock();
        let (root, _, _) = fixture();
        text::private_dir(&root.join("stage")).unwrap();
        for (mode, path) in [("bundle-fifo", root.join(BUNDLE_MANIFEST)),
            ("install-fifo", root.join("stage/install.conf"))] {
            assert!(std::process::Command::new("mkfifo").arg(&path).status().unwrap().success());
            let mut process = ChildGuard(child(mode, &root).spawn().unwrap());
            assert!(bounded_status(&mut process.0).success(), "{mode} should reject the FIFO");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn owner_source_build_cli_loss_stops_compiler_descendants_before_recovery_returns() {
        let _guard = crate::runner::process_test_lock();
        let (root, _, _) = fixture();
        let target = root.join("stage/build-target");
        text::private_dir(&target).unwrap();
        let cargo = root.join(".cargo/bin/cargo");
        text::private_dir(cargo.parent().unwrap()).unwrap();
        fs::write(&cargo, "#!/bin/sh\nsleep 300 &\nprintf '%s\\n' \"$!\" > \"$CARGO_TARGET_DIR/descendant\"\nwait\n").unwrap();
        fs::set_permissions(&cargo, fs::Permissions::from_mode(0o700)).unwrap();
        let mut unrelated = ChildGuard(std::process::Command::new("sleep").arg("300").spawn().unwrap());
        let mut process = ChildGuard(child("build", &root).env("HOME", &root).spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(15);
        while !target.join("descendant").is_file() {
            assert!(process.0.try_wait().unwrap().is_none(), "build driver exited before spawning compiler descendant");
            assert!(Instant::now() < deadline, "compiler descendant did not start");
            std::thread::sleep(Duration::from_millis(20));
        }
        // The disposable launcher dies without running Drop or cleanup code.
        process.0.kill().unwrap();
        process.0.wait().unwrap();
        let mut runner = crate::runner::SystemRunner { dry: false };
        runner.recover_jobs(&root).unwrap();
        let jobs = fs::read_dir(root.join("build-jobs")).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(jobs.len(), 1);
        let job = jobs[0].path();
        let stopped = text::parse(&fs::read_to_string(job.join("stopped.conf")).unwrap()).unwrap();
        assert!(stopped["error"].contains("recorded group stopped"));
        let anchor = text::parse(&fs::read_to_string(job.join("anchor.conf")).unwrap()).unwrap();
        let output = std::process::Command::new("/bin/ps").args(["-axo", "pgid=,stat="]).output().unwrap();
        assert!(output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).lines().any(|line| {
            let mut fields = line.split_whitespace();
            fields.next() == Some(anchor["pgid"].as_str()) && fields.next().is_some_and(|state| !state.starts_with('Z'))
        }), "recorded compiler group must have no live descendants");
        assert!(unrelated.0.try_wait().unwrap().is_none(), "unrelated process must remain untouched");
        fs::remove_dir_all(root).unwrap();
    }

}
