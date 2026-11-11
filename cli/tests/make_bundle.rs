//! Run `scripts/make-bundle.sh --from-payload` on a fixture payload: two runs
//! must give the same bytes, and the archive must be one folder with MANIFEST.
use std::{fs, os::unix::fs::PermissionsExt, path::{Path, PathBuf}, process::Command, sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);
impl Temp {
    fn new(label: &str) -> Self {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap();
        loop {
            let path = root.join(format!("vade-{label}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test folder: {error}"),
            }
        }
    }
}
impl Drop for Temp { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

fn write(path: &Path, bytes: &[u8], mode: u32) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}
// ELF64, little endian, machine 183 (AArch64), with the strings the producer reads.
fn elf(strings: &str) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[..6].copy_from_slice(b"\x7fELF\x02\x01");
    bytes[18] = 183;
    bytes.extend_from_slice(strings.as_bytes());
    bytes
}
fn run(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(output.status.success(), "{command:?}: {}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}
fn sha256(path: &Path) -> String {
    let tool = ["/usr/bin/sha256sum", "/sbin/sha256sum"].into_iter().find(|p| Path::new(p).is_file());
    let mut command = match tool { Some(tool) => Command::new(tool), None => { let mut c = Command::new("shasum"); c.args(["-a", "256"]); c } };
    run(command.arg(path))[..64].to_owned()
}
fn set_times(path: &Path, seconds: u64) {
    if path.is_dir() {
        for entry in fs::read_dir(path).unwrap() { set_times(&entry.unwrap().path(), seconds); }
    }
    let time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds);
    fs::File::open(path).unwrap().set_times(fs::FileTimes::new().set_accessed(time).set_modified(time)).unwrap();
}
fn check_archive_times(path: &Path, epoch: u64) {
    let gzip = fs::read(path).unwrap();
    // No source name or current time may enter the gzip header.
    assert_eq!(&gzip[..8], b"\x1f\x8b\x08\x00\x00\x00\x00\x00");
    let tar = Command::new("gzip").args(["-dc"]).arg(path).output().unwrap();
    assert!(tar.status.success());
    let octal = |bytes: &[u8]| u64::from_str_radix(std::str::from_utf8(bytes).unwrap().trim_matches(['\0', ' ']), 8).unwrap();
    let mut at = 0;
    while tar.stdout[at..at + 512].iter().any(|&b| b != 0) {
        let header = &tar.stdout[at..at + 512];
        assert_eq!(&header[257..263], b"ustar\0");
        assert_eq!(octal(&header[136..148]), epoch, "member {:?}", String::from_utf8_lossy(&header[..100]));
        let size = usize::try_from(octal(&header[124..136])).unwrap();
        at += 512 + size.div_ceil(512) * 512;
    }
    assert!(at > 0 && tar.stdout[at..].iter().all(|&b| b == 0));
}

struct Fixture { temp: Temp, repo: PathBuf, payload: PathBuf, cli: PathBuf, home: PathBuf, revision: String, ptx: PathBuf }
// `stamp` is what the engine build leaves in flash-serve when it embeds a cubin.
fn fixture(stamp: &str) -> Fixture {
    let temp = Temp::new("make-bundle");
    let (repo, payload, cli, home) = (temp.0.join("repo"), temp.0.join("payload"), temp.0.join("cli"), temp.0.join("home"));
    fs::create_dir_all(&home).unwrap();
    write(&repo.join("scripts/make-bundle.sh"), &fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/make-bundle.sh")).unwrap(), 0o755);
    write(&repo.join("cli/Cargo.toml"), b"[package]\nname = \"vade-cli\"\nversion = \"0.1.0\"\n", 0o644);
    for name in ["LICENSE", "NOTICE", "engine/LICENSE", "engine/NOTICE", "engine/licenses/LICENSE-fixture",
        "engine/crates/tokenizer/UNICODE-NOTICE.txt", "engine/crates/tokenizer/QWEN-NOTICE.txt"] {
        write(&repo.join(name), format!("{name}\n").as_bytes(), 0o644);
    }
    let git = |args: &[&str]| run(Command::new("git").arg("-C").arg(&repo).args(args)
        .env("HOME", &home).env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "fixture").env("GIT_AUTHOR_EMAIL", "fixture")
        .env("GIT_COMMITTER_NAME", "fixture").env("GIT_COMMITTER_EMAIL", "fixture")
        .env("GIT_AUTHOR_DATE", "2026-10-31T12:00:00Z").env("GIT_COMMITTER_DATE", "2026-10-31T12:00:00Z"));
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&["-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture"]);
    let revision = git(&["rev-parse", "HEAD"]).trim().to_owned();
    fs::create_dir_all(&payload).unwrap();
    let tar = Command::new("git").arg("-C").arg(&repo).args(["archive", "--format=tar", "HEAD"]).env("HOME", &home).output().unwrap();
    assert!(tar.status.success());
    fs::write(payload.join("source.tar"), tar.stdout).unwrap();
    let engine = payload.join("artifacts/engine/target/release/flash-serve");
    let ptx = payload.join("artifacts/engine/target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx");
    write(&engine, &elf(&format!("libcuda.so.1\0GLIBC_2.17\0GLIBC_2.39\0{stamp}\0")), 0o755);
    write(&ptx, b"//\n.version 8.8\n.target sm_121\n.address_size 64\n", 0o644);
    write(&payload.join("build.json"), format!("{{\n  \"sha\": \"{revision}\",\n  \"binary_sha256\": \"{}\",\n  \"ptx_sha256\": \"{}\"\n}}\n",
        sha256(&engine), sha256(&ptx)).as_bytes(), 0o644);
    write(&cli.join("vade"), &elf("GLIBC_2.34\0"), 0o755);
    write(&cli.join("vade-harness"), &elf("GLIBC_2.17\0"), 0o755);
    write(&cli.join("REVISION"), format!("{revision}\n").as_bytes(), 0o644);
    Fixture { temp, repo, payload, cli, home, revision, ptx }
}
impl Fixture {
    fn command(&self, extra: &[&std::ffi::OsStr]) -> Command {
        let mut command = Command::new("/bin/sh");
        command.arg(self.repo.join("scripts/make-bundle.sh"))
            .args(["--from-payload".as_ref(), self.payload.as_os_str(), "--cli-dir".as_ref(), self.cli.as_os_str()]).args(extra)
            .env_clear().env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").env("HOME", &self.home).env("TMPDIR", &self.temp.0);
        command
    }
}

#[test]
fn from_payload_gives_the_same_bytes_twice_and_one_checked_folder() {
    let fixture = fixture("");
    let (temp, revision) = (&fixture.temp, &fixture.revision);
    let stem = format!("vade-0.1.0-{}-linux-aarch64-gb10", &revision[..7]);
    let epoch = run(Command::new("git").arg("-C").arg(&fixture.repo).args(["show", "-s", "--format=%ct", revision])).trim().parse().unwrap();
    let produce = |out: &str, seconds| {
        set_times(&fixture.payload, seconds);
        set_times(&fixture.cli, seconds);
        run(&mut fixture.command(&[temp.0.join(out).as_os_str()]));
        let output = temp.0.join(out);
        check_archive_times(&output.join(format!("{stem}.tar.gz")), epoch);
        (fs::read(output.join(format!("{stem}.tar.gz"))).unwrap(), fs::read_to_string(output.join("SHA256SUMS")).unwrap())
    };
    // Distinct input times and exact header checks do not depend on a clock tick.
    let first = produce("out-1", 1_000_000_000);
    let second = produce("out-2", 2_000_000_000);
    assert!(first == second, "two runs gave different bytes");
    assert_eq!(first.1, format!("{}  {stem}.tar.gz\n", sha256(&temp.0.join("out-1").join(format!("{stem}.tar.gz")))));
    // One top folder; MANIFEST names the commit, the oldest driver and glibc.
    let list = run(Command::new("tar").arg("-tzf").arg(temp.0.join("out-1").join(format!("{stem}.tar.gz"))));
    assert!(list.lines().all(|member| member == format!("{stem}/") || member.starts_with(&format!("{stem}/"))), "{list}");
    for file in ["vade", "flash-serve", "flash.ptx", "vade-harness", "launcher-bundle.conf", "MANIFEST", "README-BUNDLE", "LICENSE"] {
        assert!(list.lines().any(|member| member == format!("{stem}/{file}")), "{file}: {list}");
    }
    assert!(!list.contains("flash.cubin.conf"), "{list}");
    let manifest = run(Command::new("tar").arg("-xzOf").arg(temp.0.join("out-1").join(format!("{stem}.tar.gz"))).arg(format!("{stem}/MANIFEST")));
    assert!(!manifest.contains("cubin_"), "{manifest}");
    for line in [format!("sha={revision}"), "date=2026-10-31T12:00:00Z".into(), "ptx_isa=8.8".into(), "min_driver=580".into(), "glibc=2.39".into()] {
        assert!(manifest.lines().any(|l| l == line), "{line}: {manifest}");
    }
    // A taken output folder is refused before anything is written.
    let refused = fixture.command(&[temp.0.join("out-1").as_os_str()]).output().unwrap();
    assert!(!refused.status.success() && String::from_utf8_lossy(&refused.stderr).contains("output already exists"));
}

const PTXAS: &str = "ptxas_version=13.0.88\nptxas_sha256=a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487\nptxas_args=--gpu-name=sm_121 -m64 -O3\n";
// The three files make-cubin.sh writes, for the PTX at `ptx`.
fn cubin(folder: &Path, ptx: &Path, image: &[u8]) -> String {
    write(&folder.join("flash.cubin"), image, 0o644);
    fs::create_dir_all(folder).unwrap();
    fs::copy(ptx, folder.join("flash.ptx")).unwrap();
    let digest = sha256(&folder.join("flash.cubin"));
    write(&folder.join("flash.cubin.conf"), format!("format=vade-cubin-v1\ntarget=sm_121\nptx_sha256={}\ncubin_sha256={digest}\nmin_driver_api=13000\n{PTXAS}",
        sha256(ptx)).as_bytes(), 0o644);
    digest
}

#[test]
fn from_payload_lists_the_cubin_and_refuses_a_cubin_that_is_not_bound() {
    // The digest of b"compiled image", the bytes this test compiles below.
    let scratch = Temp::new("make-bundle-digest");
    write(&scratch.0.join("image"), b"compiled image", 0o644);
    let image = sha256(&scratch.0.join("image"));
    let fixture = fixture(&format!("cubin_sha256={image}"));
    let (temp, ptx, revision) = (&fixture.temp, &fixture.ptx, &fixture.revision);
    let folder = temp.0.join("cubin");
    assert_eq!(cubin(&folder, ptx, b"compiled image"), image);
    let flag = std::ffi::OsStr::new("--cubin-dir");
    let refused = |folder: &Path, out: &str, reason: &str| {
        let output = fixture.command(&[flag, folder.as_os_str(), temp.0.join(out).as_os_str()]).output().unwrap();
        assert!(!output.status.success() && String::from_utf8_lossy(&output.stderr).contains(reason), "{}", String::from_utf8_lossy(&output.stderr));
        assert!(!temp.0.join(out).exists());
    };

    run(&mut fixture.command(&[flag, folder.as_os_str(), temp.0.join("with").as_os_str()]));
    let stem = format!("vade-0.1.0-{}-linux-aarch64-gb10", &revision[..7]);
    let archive = temp.0.join("with").join(format!("{stem}.tar.gz"));
    let manifest = run(Command::new("tar").arg("-xzOf").arg(&archive).arg(format!("{stem}/MANIFEST")));
    let conf = fs::read_to_string(folder.join("flash.cubin.conf")).unwrap();
    assert_eq!(run(Command::new("tar").arg("-xzOf").arg(&archive).arg(format!("{stem}/flash.cubin.conf"))), conf);
    for line in [format!("cubin_sha256={image}"), format!("cubin_ptx_sha256={}", sha256(ptx)), "cubin_ptxas=13.0.88".into(),
        format!("file={}  flash.cubin.conf", sha256(&folder.join("flash.cubin.conf")))] {
        assert!(manifest.lines().any(|l| l == line), "{line}: {manifest}");
    }
    // The launcher manifest keeps its fields, so older CLIs read it unchanged.
    let launcher = run(Command::new("tar").arg("-xzOf").arg(&archive).arg(format!("{stem}/launcher-bundle.conf")));
    assert_eq!(launcher.lines().count(), 9, "{launcher}");
    assert!(run(Command::new("tar").arg("-xzOf").arg(&archive).arg(format!("{stem}/README-BUNDLE"))).contains("compiled for the GB10"));

    // A stamped engine needs its cubin folder, or the manifest would hide the image.
    let output = fixture.command(&[temp.0.join("none").as_os_str()]).output().unwrap();
    assert!(!output.status.success() && String::from_utf8_lossy(&output.stderr).contains("embeds a cubin"));
    // The compiled PTX differs by one byte: the hashes in the conf can not hide it.
    let other = temp.0.join("other-ptx");
    cubin(&other, ptx, b"compiled image");
    fs::write(other.join("flash.ptx"), [fs::read(ptx).unwrap(), b"\n".to_vec()].concat()).unwrap();
    refused(&other, "bad-ptx", "other PTX bytes");
    // The conf names other PTX bytes.
    let named = temp.0.join("named-ptx");
    cubin(&named, ptx, b"compiled image");
    let conf = fs::read_to_string(named.join("flash.cubin.conf")).unwrap();
    fs::write(named.join("flash.cubin.conf"), conf.replace(&sha256(ptx), &"0".repeat(64))).unwrap();
    refused(&named, "bad-conf", "names other PTX bytes");
    // The image differs from its conf.
    let changed = temp.0.join("changed-image");
    cubin(&changed, ptx, b"compiled image");
    fs::write(changed.join("flash.cubin"), b"another image").unwrap();
    refused(&changed, "bad-image", "does not match flash.cubin.conf");
    // A complete, correct folder for an image the engine does not hold.
    let absent = temp.0.join("absent-image");
    cubin(&absent, ptx, b"another image");
    refused(&absent, "bad-engine", "does not embed this cubin");
}

// make-cubin.sh with its two pins moved to a stand-in compiler in a saved archive.
struct Compiler { temp: Temp, archive: PathBuf, script: PathBuf, ptx: PathBuf }
fn compiler(version: &str) -> Compiler {
    let temp = Temp::new("make-cubin");
    let folder = "cuda_nvcc-linux-sbsa-13.0.88-archive";
    let ptxas = temp.0.join("toolkit").join(folder).join("bin/ptxas");
    // It records its arguments as the image, so the test can read the exact command.
    write(&ptxas, format!("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'Cuda compilation tools, release 13.0, V{version}'; exit 0; fi\n\
        printf '%s\\n' \"$*\" > flash.cubin\n").as_bytes(), 0o755);
    let archive = temp.0.join("saved.tar");
    run(Command::new("tar").arg("-cf").arg(&archive).arg("-C").arg(temp.0.join("toolkit")).arg(folder));
    let script = temp.0.join("make-cubin.sh");
    let source = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/make-cubin.sh")).unwrap()
        .replace("01b01e10aa2662ad1b3aeab3317151d7d6d4a650eeade55ded504f6b7fced18e", &sha256(&archive))
        .replace("a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487", &sha256(&ptxas));
    write(&script, source.as_bytes(), 0o755);
    let ptx = temp.0.join("in/spark_layer_kernels.ptx");
    write(&ptx, b"//\n.version 8.8\n.target sm_121\n.address_size 64\n", 0o644);
    Compiler { temp, archive, script, ptx }
}

#[test]
fn make_cubin_runs_the_pinned_command_on_exact_bytes_and_writes_the_manifest() {
    let c = compiler("13.0.88");
    let out = c.temp.0.join("out");
    let printed = run(Command::new("/bin/sh").arg(&c.script).arg(&c.ptx).arg(&out).arg(&c.archive));
    assert!(printed.contains("ptxas: 13.0.88"), "{printed}");
    assert_eq!(fs::read_to_string(out.join("flash.cubin")).unwrap(), "--gpu-name=sm_121 -m64 -O3 flash.ptx -o flash.cubin\n");
    assert_eq!(fs::read(out.join("flash.ptx")).unwrap(), fs::read(&c.ptx).unwrap());
    assert_eq!(fs::read_to_string(out.join("flash.cubin.conf")).unwrap(), format!(
        "format=vade-cubin-v1\ntarget=sm_121\nptx_sha256={}\ncubin_sha256={}\nmin_driver_api=13000\nptxas_version=13.0.88\nptxas_sha256={}\nptxas_args=--gpu-name=sm_121 -m64 -O3\n",
        sha256(&c.ptx), sha256(&out.join("flash.cubin")), sha256(&c.temp.0.join("toolkit/cuda_nvcc-linux-sbsa-13.0.88-archive/bin/ptxas"))));
    let names: Vec<_> = fs::read_dir(&c.temp.0).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    assert!(!names.iter().any(|name| name.starts_with("out.")), "{names:?}");
    // A second run does not replace the first.
    let again = Command::new("/bin/sh").arg(&c.script).arg(&c.ptx).arg(&out).arg(&c.archive).output().unwrap();
    assert!(!again.status.success() && String::from_utf8_lossy(&again.stderr).contains("output already exists"));
}

#[test]
fn make_cubin_refuses_another_compiler_or_target() {
    let refused = |c: &Compiler, archive: &Path, reason: &str| {
        let out = c.temp.0.join("refused");
        let output = Command::new("/bin/sh").arg(&c.script).arg(&c.ptx).arg(&out).arg(archive).output().unwrap();
        assert!(!output.status.success() && String::from_utf8_lossy(&output.stderr).contains(reason), "{}", String::from_utf8_lossy(&output.stderr));
        assert!(!out.exists());
    };
    let c = compiler("13.0.88");
    let wrong = c.temp.0.join("other.tar");
    fs::write(&wrong, b"not the pinned archive").unwrap();
    refused(&c, &wrong, "pinned SHA-256");
    write(&c.ptx, b"//\n.version 8.8\n.target sm_121a\n.address_size 64\n", 0o644);
    refused(&c, &c.archive, "does not target the GB10");
    let c = compiler("13.0.48");
    refused(&c, &c.archive, "does not report version 13.0.88");
}

// A release fails late if the producer and the engine build pin different compilers.
#[test]
fn make_cubin_and_the_engine_build_pin_the_same_compiler() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let script = fs::read_to_string(root.join("scripts/make-cubin.sh")).unwrap();
    let engine = fs::read_to_string(root.join("engine/crates/gpu/cubin.rs")).unwrap();
    for (key, constant) in [("version=", "PTXAS_VERSION"), ("ptxas_sha256=", "PTXAS_SHA256")] {
        let value = script.lines().find_map(|line| line.strip_prefix(key)).unwrap();
        assert!(engine.contains(&format!("const {constant}: &str = \"{value}\";")), "{constant}");
    }
    let arguments = script.lines().find_map(|line| line.strip_prefix("ptxas_args='")).unwrap().trim_end_matches('\'');
    assert!(engine.contains(&format!("const PTXAS_ARGS: &str = \"{arguments}\";")), "{arguments}");
}
