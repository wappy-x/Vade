//! Run `scripts/make-bundle.sh --from-payload` on a fixture payload: two runs
//! must give the same bytes, and the archive must be one folder with MANIFEST.
use std::{fs, os::unix::fs::PermissionsExt, path::{Path, PathBuf}, process::Command};

struct Temp(PathBuf);
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

#[test]
fn from_payload_gives_the_same_bytes_twice_and_one_checked_folder() {
    let temp = Temp(fs::canonicalize(std::env::temp_dir()).unwrap()
        .join(format!("vade-make-bundle-{}-{}", std::process::id(), std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())));
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
    write(&engine, &elf("libcuda.so.1\0GLIBC_2.17\0GLIBC_2.39\0"), 0o755);
    write(&ptx, b"//\n.version 8.8\n.target sm_121\n.address_size 64\n", 0o644);
    write(&payload.join("build.json"), format!("{{\n  \"sha\": \"{revision}\",\n  \"binary_sha256\": \"{}\",\n  \"ptx_sha256\": \"{}\"\n}}\n",
        sha256(&engine), sha256(&ptx)).as_bytes(), 0o644);
    write(&cli.join("vade"), &elf("GLIBC_2.34\0"), 0o755);
    write(&cli.join("vade-harness"), &elf("GLIBC_2.17\0"), 0o755);
    write(&cli.join("REVISION"), format!("{revision}\n").as_bytes(), 0o644);

    let stem = format!("vade-0.1.0-{}-linux-aarch64-gb10", &revision[..7]);
    let produce = |out: &str| {
        run(Command::new("/bin/sh").arg(repo.join("scripts/make-bundle.sh"))
            .args(["--from-payload".as_ref(), payload.as_os_str(), "--cli-dir".as_ref(), cli.as_os_str(), temp.0.join(out).as_os_str()])
            .env_clear().env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").env("HOME", &home).env("TMPDIR", &temp.0));
        let output = temp.0.join(out);
        (fs::read(output.join(format!("{stem}.tar.gz"))).unwrap(), fs::read_to_string(output.join("SHA256SUMS")).unwrap())
    };
    let first = produce("out-1");
    // Another second, so any file time or gzip time that leaks in changes the bytes.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let second = produce("out-2");
    assert!(first == second, "two runs gave different bytes");
    assert_eq!(first.1, format!("{}  {stem}.tar.gz\n", sha256(&temp.0.join("out-1").join(format!("{stem}.tar.gz")))));
    // One top folder; MANIFEST names the commit, the oldest driver and glibc.
    let list = run(Command::new("tar").arg("-tzf").arg(temp.0.join("out-1").join(format!("{stem}.tar.gz"))));
    assert!(list.lines().all(|member| member == format!("{stem}/") || member.starts_with(&format!("{stem}/"))), "{list}");
    for file in ["vade", "flash-serve", "flash.ptx", "vade-harness", "launcher-bundle.conf", "MANIFEST", "README-BUNDLE", "LICENSE"] {
        assert!(list.lines().any(|member| member == format!("{stem}/{file}")), "{file}: {list}");
    }
    let manifest = run(Command::new("tar").arg("-xzOf").arg(temp.0.join("out-1").join(format!("{stem}.tar.gz"))).arg(format!("{stem}/MANIFEST")));
    for line in [format!("sha={revision}"), "date=2026-10-31T12:00:00Z".into(), "ptx_isa=8.8".into(), "min_driver=580".into(), "glibc=2.39".into()] {
        assert!(manifest.lines().any(|l| l == line), "{line}: {manifest}");
    }
    // A taken output folder is refused before anything is written.
    let refused = Command::new("/bin/sh").arg(repo.join("scripts/make-bundle.sh"))
        .args(["--from-payload".as_ref(), payload.as_os_str(), "--cli-dir".as_ref(), cli.as_os_str(), temp.0.join("out-1").as_os_str()])
        .env_clear().env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").env("HOME", &home).output().unwrap();
    assert!(!refused.status.success() && String::from_utf8_lossy(&refused.stderr).contains("output already exists"));
}
