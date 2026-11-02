//! Exercise the actual source installer with fake host tools, never a GPU or network.
use std::{fs, os::unix::fs::PermissionsExt, path::{Path, PathBuf}, process::{Command, Output}, time::{SystemTime, UNIX_EPOCH}};

static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct Fixture { root: PathBuf }
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("vade source install {}-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        for dir in ["bin", "scripts", "engine", "harness", "cli/target/release", "home", "tmp"] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        for file in ["cli/Cargo.toml", "engine/Cargo.toml", "harness/Cargo.toml"] { fs::write(root.join(file), "").unwrap(); }
        fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/install.sh"), root.join("scripts/install.sh")).unwrap();
        let fixture = Self { root: fs::canonicalize(root).unwrap() };
        for tool in ["sh", "dirname", "mktemp", "rm", "mkdir", "mv", "cp", "chmod", "tail", "sed", "head", "ls"] {
            let path = ["/usr/bin", "/bin"].into_iter().map(|dir| Path::new(dir).join(tool))
                .find(|path| path.is_file()).unwrap();
            std::os::unix::fs::symlink(path, fixture.root.join("bin").join(tool)).unwrap();
        }
        fixture.tool("uname", "case $1 in -s) echo Linux ;; -m) echo aarch64 ;; esac");
        fixture.tool("cc", "exit 0");
        fixture.tool("nvidia-smi", r#"
[ "${INSTALL_DRIVER_FAIL-false}" = false ] || { echo 'NVIDIA-SMI has failed because it could not reach the driver.' >&2; exit 9; }
case "${1-}" in
    --query-gpu=name) printf '%s\n' "${INSTALL_GPU_NAME-NVIDIA GB10}" ;;
    --query-gpu=driver_version) printf '%s\n' "${INSTALL_DRIVER-580.95.05}" ;;
    --query-compute-apps=pid) [ "${INSTALL_GPU_BUSY-false}" = false ] || echo 1234 ;;
    '') if [ "${INSTALL_UMD-false}" = true ]; then
            printf '| NVIDIA-SMI 615.71.09   KMD Version: 615.71.09   CUDA UMD Version: %s |\n' "${INSTALL_CUDA-13.4}"
        else
            printf '| NVIDIA-SMI 580.95.05  Driver Version: 580.95.05  CUDA Version: %s |\n' "${INSTALL_CUDA-13.0}"
        fi ;;
esac
"#);
        fixture.tool("curl", "echo unexpected-download >> \"$INSTALL_CALLS\"; exit 91");
        fixture.tool("rustup", r#"
printf 'rustup %s\n' "$*" >> "$INSTALL_CALLS"
if [ "$1" = component ]; then
    [ "${INSTALL_MISSING_DEVICE-false}" = false ] || exit 1
    printf '%s\n' rust-src llvm-tools-aarch64-unknown-linux-gnu llvm-bitcode-linker-aarch64-unknown-linux-gnu rust-std-nvptx64-nvidia-cuda
elif [ "$1" = run ]; then
    [ "${INSTALL_MISSING_HOST-false}" = false ] || [ "$2" != 1.99.0 ] || exit 1
    [ "${4-}" != --print ] || [ "${INSTALL_NO_SYSROOT-false}" = true ] || printf '%s\n' "$INSTALL_ROOT/sysroot"
fi
"#);
        fixture.tool("cargo", r#"
printf 'cargo %s\n' "$*" >> "$INSTALL_CALLS"
if [ "$2" = fetch ]; then
    [ "$5" = "$INSTALL_ROOT/sysroot/lib/rustlib/src/rust/library/Cargo.toml" ] || exit 89
    [ "${INSTALL_FAIL_FETCH-false}" = false ] || { echo 'error: fixture fetch failure' >&2; exit 42; }
else
    [ "$CARGO_TARGET_DIR" = "$INSTALL_ROOT/cli/target" ] || exit 90
fi
"#);
        fixture.write("cli/target/release/vade", r#"
printf 'vade %s\n' "$*" >> "$INSTALL_CALLS"
[ "$VADE_SOURCE" = "$INSTALL_ROOT" ] || exit 90
# Like the real CLI's lock: a group or other bit on ~/.vade stops it.
[ ! -d "$HOME/.vade" ] || case $(ls -ld "$HOME/.vade") in d???------*) ;; *) echo 'Directory ~/.vade is shared.' >&2; exit 46 ;; esac
[ "${INSTALL_FAIL_VADE-}" != "$1" ] || { echo 'fixture diagnostic: fix this cause' >&2; exit 43; }
"#);
        fixture
    }
    fn write(&self, name: &str, script: &str) {
        let path = self.root.join(name);
        fs::write(&path, format!("#!/bin/sh\nset -eu\n{script}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    fn tool(&self, name: &str, script: &str) { self.write(&format!("bin/{name}"), script); }
    fn run(&self, env: &[(&str, &str)]) -> Output { self.run_args(env, &[]) }
    fn run_args(&self, env: &[(&str, &str)], args: &[&str]) -> Output { self.run_in("/bin/sh", env, args) }
    fn run_in(&self, shell: &str, env: &[(&str, &str)], args: &[&str]) -> Output {
        // Ubuntu's default umask 002: a folder the script creates is group-writable unless it says otherwise.
        Command::new(shell).args(["-c", "umask 002 && exec \"$0\" \"$@\""]).arg(shell).arg(self.root.join("scripts/install.sh"))
            .args(args)
            .current_dir(&self.root).env_clear()
            .env("PATH", self.root.join("bin")).env("TMPDIR", self.root.join("tmp"))
            .env("HOME", self.root.join("home")).env("CARGO_NET_OFFLINE", "true")
            .env("INSTALL_ROOT", &self.root).env("INSTALL_CALLS", self.root.join("calls"))
            .envs(env.iter().copied()).output().unwrap()
    }
    fn calls(&self) -> String { fs::read_to_string(self.root.join("calls")).unwrap_or_default() }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.root); } }

#[test]
fn cached_source_install_fetches_build_std_and_leaves_detect_and_kernels_to_vade_install() {
    let f = Fixture::new();
    let output = f.run(&[("CARGO_TARGET_DIR", "/unrelated/target")]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let calls = f.calls();
    assert!(!calls.contains("unexpected-download"), "{calls}");
    assert!(!calls.contains("toolchain install"), "{calls}");
    assert!(calls.contains("cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml"), "{calls}");
    // vade install detects this Spark and builds the kernels once (fresh install step 4).
    let fetch = format!("cargo +nightly-2026-10-01 fetch --locked --manifest-path {}/sysroot/lib/rustlib/src/rust/library/Cargo.toml\n", f.root.display());
    assert!(calls.ends_with(&format!("{fetch}vade __install-here\n")), "{calls}");
    assert!(!calls.contains("kernels") && !calls.contains("vade detect"), "{calls}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Install: fetching the Rust build-std files for the GPU kernels (none when cached).\n"), "{stdout}");
    // One next path: the model when it is missing, then up; no second install through vade setup.
    assert!(stdout.contains("1. If the model is not on this Spark yet, download and check it (102.26 GB):\n   vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300\n2. Start the engine:\n   vade up\n"), "{stdout}");
    assert!(stdout.contains("2. Start the engine:\n   vade up\n\n") && !stdout.contains("vade launch"), "{stdout}");
    // The last thing on screen is the one line about PATH, and it prints once.
    assert!(stdout.ends_with(PATH_LINE), "{stdout}");
    assert_eq!(stdout.matches("export PATH=").count(), 1, "{stdout}");
    assert!(!stdout.contains(".profile"), "{stdout}");
    assert!(!stdout.contains("setup"), "{stdout}");
}

#[test]
fn newer_drivers_print_cuda_umd_version_and_pass_the_same_check() {
    let f = Fixture::new();
    let output = f.run(&[("INSTALL_UMD", "true")]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(f.calls().ends_with("vade __install-here\n"), "{}", f.calls());
    let old = Fixture::new();
    let output = old.run(&[("INSTALL_UMD", "true"), ("INSTALL_CUDA", "12.8")]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && stderr.contains("must support CUDA 13.0 or newer") && stderr.contains("Next:"), "{stderr}");
    assert!(!old.calls().contains("rustup "), "{}", old.calls());
}

#[test]
fn launch_option_starts_only_after_a_successful_install() {
    let f = Fixture::new();
    let output = f.run_args(&[], &["--launch"]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(f.calls().ends_with("vade __install-here\nvade launch\n"), "{}", f.calls());
    fs::write(f.root.join("calls"), "").unwrap();
    let output = f.run_args(&[("INSTALL_FAIL_VADE", "__install-here")], &["--launch"]);
    assert!(!output.status.success());
    assert!(!f.calls().contains("vade launch"), "{}", f.calls());
}

#[test]
fn source_install_failures_stop_at_the_cause_and_print_a_next_step() {
    for (env, message, forbidden) in [
        (("VADE_CONFIG", "/controller.conf"), "unset VADE_CONFIG", "cargo "),
        (("VADE_BUNDLE", "/bundle"), "unset VADE_BUNDLE", "cargo "),
        (("INSTALL_GPU_BUSY", "true"), "stop that workload through the launcher", "rustup "),
        (("INSTALL_DRIVER_FAIL", "true"), "NVIDIA driver check failed (nvidia-smi error above)", "rustup "),
        (("INSTALL_GPU_NAME", "NVIDIA B200"), "needs an NVIDIA GB10 GPU (a DGX Spark); nvidia-smi reports: NVIDIA B200.", "rustup "),
        (("INSTALL_GPU_NAME", "NVIDIA GB100"), "needs an NVIDIA GB10 GPU", "rustup "),
        (("INSTALL_CUDA", "12.8"), "must support CUDA 13.0 or newer", "rustup "),
        (("INSTALL_CUDA", "N/A"), "must support CUDA 13.0 or newer", "rustup "),
        (("INSTALL_MISSING_HOST", "true"), "rerun this script online", "cargo "),
        (("INSTALL_MISSING_DEVICE", "true"), "rerun this script online", "cargo "),
        (("INSTALL_FAIL_FETCH", "true"), "Rust build-std fetch failed", "vade install"),
        (("INSTALL_NO_SYSROOT", "true"), "cannot find the nightly-2026-10-01 sysroot", "fetch"),
        (("INSTALL_FAIL_VADE", "__install-here"), "fixture diagnostic: fix this cause", "vade launch"),
    ] {
        let f = Fixture::new();
        let output = f.run(&[env]);
        assert!(!output.status.success(), "unexpected success: {env:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(message) && stderr.contains("Next:"), "{env:?}: {stderr}");
        assert!(!f.calls().contains(forbidden), "{env:?}: {}", f.calls());
        assert!(!f.calls().contains("unexpected-download"));
    }
}

#[test]
fn missing_compilers_are_installed_with_pins_without_changing_shell_profiles() {
    let f = Fixture::new();
    let output = f.run(&[("CARGO_NET_OFFLINE", "false"), ("INSTALL_MISSING_HOST", "true"),
        ("INSTALL_MISSING_DEVICE", "true")]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let calls = f.calls();
    assert!(calls.contains("rustup toolchain install 1.99.0 --profile minimal --no-self-update"), "{calls}");
    assert!(calls.contains("rustup toolchain install nightly-2026-10-01 --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda"), "{calls}");
    assert!(!f.root.join("home/.profile").exists());
    assert!(!f.root.join("home/.bashrc").exists());
}

#[test]
fn missing_rustup_downloads_the_official_installer_and_cleans_temporary_files() {
    for failure in ["", "download", "bootstrap"] {
        let f = Fixture::new();
        fs::rename(f.root.join("bin/rustup"), f.root.join("rustup-provider")).unwrap();
        f.write("bootstrap", r#"
printf 'bootstrap %s\n' "$*" >> "$INSTALL_CALLS"
[ "${INSTALL_FAIL_BOOTSTRAP-}" != bootstrap ] || exit 45
mkdir -p "$HOME/.cargo/bin"
cp "$INSTALL_ROOT/rustup-provider" "$HOME/.cargo/bin/rustup"
chmod 700 "$HOME/.cargo/bin/rustup"
"#);
        f.tool("curl", r#"
printf 'curl %s\n' "$*" >> "$INSTALL_CALLS"
[ "${INSTALL_FAIL_BOOTSTRAP-}" != download ] || exit 44
while [ "$#" -gt 0 ]; do
    if [ "$1" = --output ]; then shift; cp "$INSTALL_ROOT/bootstrap" "$1"; fi
    shift
done
"#);
        let output = f.run(&[("CARGO_NET_OFFLINE", "false"), ("INSTALL_FAIL_BOOTSTRAP", failure)]);
        let calls = f.calls();
        assert!(calls.contains("https://sh.rustup.rs"), "{calls}");
        assert!(fs::read_dir(f.root.join("tmp")).unwrap().next().is_none());
        assert!(!f.root.join("home/.profile").exists());
        if failure.is_empty() {
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            assert!(calls.contains("bootstrap -y --default-toolchain none --no-modify-path"), "{calls}");
            assert!(calls.ends_with("vade __install-here\n"), "{calls}");
        } else {
            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr).contains("Next:"));
            assert!(!calls.contains("cargo "), "{calls}");
        }
    }
}

// The prebuilt path: real tar, gzip, sha256sum and curl reading a file:/// copy
// of the download folder, so no network. Each bundle refusal must leave no
// bundle folder and no download folder, then build from source.
const SHA: &str = "1234567890abcdef1234567890abcdef12345678";
fn stem() -> String { format!("vade-0.1.0-{}-linux-aarch64-gb10", &SHA[..7]) }
fn system_tool(name: &str) -> PathBuf {
    ["/usr/bin", "/bin", "/usr/sbin", "/sbin"].into_iter().map(|dir| Path::new(dir).join(name))
        .find(|path| path.is_file()).unwrap_or_else(|| panic!("{name} is not installed"))
}
fn sha256(bytes: &[u8]) -> String {
    let mut child = Command::new(system_tool("sha256sum")).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped()).spawn().unwrap();
    std::io::Write::write_all(&mut child.stdin.take().unwrap(), bytes).unwrap();
    String::from_utf8(child.wait_with_output().unwrap().stdout).unwrap()[..64].to_owned()
}
// One ustar member: kind b'0' file, b'5' folder, b'2' symbolic link, b'3' character device.
fn member(tar: &mut Vec<u8>, name: &str, kind: u8, mode: u32, data: &[u8], link: &str) {
    let mut h = [0u8; 512];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h[100..108].copy_from_slice(format!("{mode:07o}\0").as_bytes());
    h[108..116].copy_from_slice(b"0000000\0");
    h[116..124].copy_from_slice(b"0000000\0");
    h[124..136].copy_from_slice(format!("{:011o}\0", data.len()).as_bytes());
    h[136..148].copy_from_slice(b"14000000000\0");
    h[156] = kind;
    h[157..157 + link.len()].copy_from_slice(link.as_bytes());
    h[257..265].copy_from_slice(b"ustar\x0000");
    h[329..337].copy_from_slice(b"0000000\0");
    h[337..345].copy_from_slice(b"0000000\0");
    h[148..156].fill(b' ');
    let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    tar.extend(h);
    tar.extend_from_slice(data);
    tar.resize(tar.len() + (512 - data.len() % 512) % 512, 0);
}
const BUNDLE_VADE: &str = "#!/bin/sh\nprintf 'bundle-vade %s\\n' \"$*\" >> \"$INSTALL_CALLS\"\n\
case $(ls -ld \"$HOME/.vade\") in d???------*) ;; *) echo 'Directory ~/.vade is shared.' >&2; exit 46 ;; esac\n\
[ \"${INSTALL_FAIL_BUNDLE-false}\" = false ] || { echo 'fixture bundle diagnostic' >&2; exit 44; }\n";
#[derive(Clone, Copy, PartialEq)]
enum Bad { None, Sha256, Manifest, Link, Device, Outside, DotDot }
fn bundle(bad: Bad) -> Vec<u8> {
    let stem = stem();
    let engine = b"fixture engine\n".as_slice();
    let mut manifest = format!("format=vade-release-v1\nname={stem}\nsha={SHA}\n");
    let engine_sum = if bad == Bad::Manifest { "0".repeat(64) } else { sha256(engine) };
    manifest += &format!("file={engine_sum}  flash-serve\nfile={}  vade\n", sha256(BUNDLE_VADE.as_bytes()));
    let mut tar = Vec::new();
    member(&mut tar, &format!("{stem}/"), b'5', 0o755, b"", "");
    member(&mut tar, &format!("{stem}/MANIFEST"), b'0', 0o644, manifest.as_bytes(), "");
    member(&mut tar, &format!("{stem}/flash-serve"), b'0', 0o755, engine, "");
    member(&mut tar, &format!("{stem}/vade"), b'0', 0o755, BUNDLE_VADE.as_bytes(), "");
    match bad {
        Bad::Link => member(&mut tar, &format!("{stem}/home"), b'2', 0o777, b"", "/home"),
        Bad::Device => member(&mut tar, &format!("{stem}/null"), b'3', 0o666, b"", ""),
        Bad::Outside => member(&mut tar, "outside.txt", b'0', 0o644, b"outside\n", ""),
        Bad::DotDot => member(&mut tar, &format!("{stem}/../escape.txt"), b'0', 0o644, b"escape\n", ""),
        _ => (),
    }
    tar.resize(tar.len() + 1024, 0);
    let mut gzip = Command::new(system_tool("gzip")).args(["-c", "-n"]).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped()).spawn().unwrap();
    let mut stdin = gzip.stdin.take().unwrap();
    let writer = std::thread::spawn(move || std::io::Write::write_all(&mut stdin, &tar).unwrap());
    let output = gzip.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(output.status.success());
    output.stdout
}
impl Fixture {
    /// Real archive tools, a fake Ubuntu 24.04 with glibc 2.39, and a download
    /// folder holding `archive` and its latest.json. Returns the run environment.
    fn prebuilt(&self, archive: &[u8], bad: Bad) -> Vec<(&'static str, String)> {
        for tool in ["tar", "gzip", "sha256sum", "grep", "tr", "cut", "sort", "mv", "curl"] {
            let _ = fs::remove_file(self.root.join("bin").join(tool));
            std::os::unix::fs::symlink(system_tool(tool), self.root.join("bin").join(tool)).unwrap();
        }
        self.tool("getconf", "[ \"$1\" = GNU_LIBC_VERSION ] && echo 'glibc 2.39'");
        fs::write(self.root.join("os-release"), "ID=ubuntu\nVERSION_ID=\"24.04\"\n").unwrap();
        let file = format!("{}.tar.gz", stem());
        fs::create_dir_all(self.root.join("dl/main")).unwrap();
        fs::write(self.root.join("dl/main").join(&file), archive).unwrap();
        let sum = if bad == Bad::Sha256 { "f".repeat(64) } else { sha256(archive) };
        fs::write(self.root.join("dl/main/latest.json"), format!("{{\n  \"sha\": \"{SHA}\",\n  \"file\": \"{file}\",\n  \
            \"sha256\": \"{sum}\",\n  \"date\": \"2026-10-31T00:00:00Z\",\n  \"min_driver\": \"580\",\n  \"glibc\": \"2.39\"\n}}\n")).unwrap();
        vec![("VADE_DL_BASE", format!("file://{}", self.root.join("dl").display()).replace(' ', "%20")),
            ("VADE_TEST_OS_RELEASE", self.root.join("os-release").display().to_string())]
    }
    fn run_prebuilt(&self, env: &[(&'static str, String)], args: &[&str]) -> Output {
        let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.run_args(&env, args)
    }
    fn bundles(&self) -> PathBuf { self.root.join("home/.vade/bundles") }
    /// `~/.vade` and its bundles folder, when the script made them, are private (the CLI refuses a shared ~/.vade).
    fn assert_private(&self, context: &str) {
        for dir in [self.root.join("home/.vade"), self.bundles()] {
            if let Ok(meta) = fs::metadata(&dir) {
                assert_eq!(meta.permissions().mode() & 0o077, 0, "{context}: {} is shared", dir.display());
            }
        }
    }
    fn left_downloads(&self) -> Vec<String> {
        fs::read_dir(self.bundles()).map(|dir| dir.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".download.")).collect()).unwrap_or_default()
    }
    fn tree(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let mut stack = vec![self.root.join("home")];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() { stack.push(path.clone()); }
                paths.push(path);
            }
        }
        paths.sort();
        paths
    }
}

#[test]
fn prebuilt_bundle_installs_from_its_folder_and_is_reused_without_a_second_download() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    let folder = f.bundles().join(stem());
    assert!(folder.join("MANIFEST").is_file() && folder.join("flash-serve").is_file(), "{stdout}");
    assert!(f.left_downloads().is_empty());
    f.assert_private("installed bundle");
    assert_eq!(f.calls(), "bundle-vade install\n");
    assert!(stdout.contains(&format!("Install: downloading the prebuilt bundle {}.tar.gz.\n", stem())), "{stdout}");
    assert!(stdout.contains(&format!("Install: prebuilt Vade {SHA} is in {}.\n", folder.display())), "{stdout}");
    assert!(stdout.contains(&format!("Install: installed prebuilt Vade {SHA}.\n")), "{stdout}");
    assert!(stdout.ends_with(PATH_LINE), "{stdout}");
    assert!(!stdout.contains("Prebuilt:"), "{stdout}");
    // A second run reuses the checked folder: latest.json only, no bundle download.
    fs::remove_file(f.root.join("dl/main").join(format!("{}.tar.gz", stem()))).unwrap();
    let again = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&again.stdout);
    assert!(again.status.success() && !stdout.contains("downloading"), "{stdout}");
    assert_eq!(f.calls(), "bundle-vade install\nbundle-vade install\n");
    let check = f.run_prebuilt(&env, &["--check"]);
    assert!(String::from_utf8_lossy(&check.stdout).contains(&format!("Check: would use {} (Vade {SHA}, already downloaded and checked)", folder.display())));
    assert_eq!(f.calls(), "bundle-vade install\nbundle-vade install\n");
}

#[test]
fn each_refused_bundle_leaves_nothing_and_builds_from_source() {
    let file = format!("{}.tar.gz", stem());
    for (bad, reason) in [
        (Bad::Sha256, format!("{file} does not match the sha256 in latest.json")),
        (Bad::Manifest, format!("{file} does not match its MANIFEST")),
        (Bad::Link, format!("{file} has members that are not files or folders")),
        (Bad::Device, format!("{file} has members that are not files or folders")),
        (Bad::Outside, format!("{file} has files outside {}/", stem())),
        (Bad::DotDot, format!("{file} has files outside {}/", stem())),
    ] {
        let f = Fixture::new();
        let env = f.prebuilt(&bundle(bad), bad);
        let output = f.run_prebuilt(&env, &[]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{reason}: {stdout}{}", String::from_utf8_lossy(&output.stderr));
        assert!(stdout.contains(&format!("Prebuilt: {reason}; building from source.\n")), "{reason}: {stdout}");
        assert!(!f.bundles().join(stem()).exists(), "{reason}");
        assert!(f.left_downloads().is_empty(), "{reason}: {:?}", f.left_downloads());
        f.assert_private(&reason);
        assert!(!f.root.join("home/escape.txt").exists() && !f.bundles().join("escape.txt").exists(), "{reason}");
        let calls = f.calls();
        assert!(calls.contains("cargo +1.99.0 build") && calls.ends_with("\nvade __install-here\n"), "{reason}: {calls}");
        assert!(!calls.contains("bundle-vade"), "{reason}: {calls}");
    }
}

#[test]
fn failed_bundle_install_and_unfit_systems_fall_back_to_the_source_build() {
    let f = Fixture::new();
    let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
    env.push(("INSTALL_FAIL_BUNDLE", "true".into()));
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    let vade = f.bundles().join(stem()).join("vade");
    assert!(stdout.contains(&format!("Prebuilt: \"{}\" install failed (diagnostic above); building from source.\n", vade.display())), "{stdout}");
    assert!(f.calls().starts_with("bundle-vade install\n") && f.calls().ends_with("\nvade __install-here\n"), "{}", f.calls());
    f.assert_private("failed bundle install");
    for (key, value, reason) in [
        ("INSTALL_DRIVER", "575.51.03", "the bundle needs NVIDIA driver 580 or newer and this Spark has 575.51.03"),
        ("VADE_CHANNEL", "../main", "VADE_CHANNEL may hold only letters, digits, dot, dash and underscore"),
        ("VADE_DL_BASE", "http://example.invalid/vade", "VADE_DL_BASE must start with https:// (or file:/// for a local copy)"),
    ] {
        let f = Fixture::new();
        let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
        env.retain(|(k, _)| *k != key);
        env.push((key, value.into()));
        let stdout = String::from_utf8_lossy(&f.run_prebuilt(&env, &[]).stdout).into_owned();
        assert!(stdout.contains(&format!("Prebuilt: {reason}; building from source.\n")), "{stdout}");
        assert!(!f.bundles().exists(), "{reason}");
    }
}

#[test]
fn check_writes_nothing_and_source_skips_the_bundle() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let before = f.tree();
    let output = f.run_prebuilt(&env, &["--check"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    let file = format!("{}.tar.gz", stem());
    assert!(stdout.contains(&format!("Check: would download {}/main/{file} (sha256 ", env[0].1)), "{stdout}");
    assert_eq!(f.tree(), before);
    assert_eq!(f.calls(), "");
    // --check on a system the bundle does not fit: the source plan, still nothing written.
    let mut unfit = env.clone();
    unfit.push(("INSTALL_DRIVER", "570.0".into()));
    let output = f.run_prebuilt(&unfit, &["--check"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success() && stdout.contains("Check: would install the pinned Rust toolchains"), "{stdout}");
    assert_eq!((f.tree(), f.calls()), (before, String::new()));
    // An unknown argument is an error whatever its place.
    for args in [["--check", "--bogus"], ["--bogus", "--check"]] {
        assert_eq!(f.run_prebuilt(&env, &args).status.code(), Some(1), "{args:?}");
    }
    // --source never reads latest.json; an inherited `download` folder is never removed.
    let keep = f.root.join("keep");
    fs::create_dir(&keep).unwrap();
    fs::write(keep.join("file"), "kept").unwrap();
    let mut source = env.clone();
    source.push(("download", keep.display().to_string()));
    let output = f.run_prebuilt(&source, &["--source"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("Prebuilt: --source was given; building from source.\n"), "{stdout}");
    assert!(!f.bundles().exists());
    assert!(f.calls().contains("cargo +1.99.0 build") && !f.calls().contains("bundle-vade"), "{}", f.calls());
    assert_eq!(fs::read_to_string(keep.join("file")).unwrap(), "kept");
}

// A piped install: `curl ... | sh`. $0 is "sh", the script is standard input,
// and the current folder says nothing about a checkout.
const CLONE_URL: &str = "https://github.com/wappy-x/vade.git";
const CURL_LINE: &str = "curl -fsSL https://vade.lol/install.sh | sh";
fn script() -> Vec<u8> { fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/install.sh")).unwrap() }
fn shells() -> Vec<&'static str> { ["/bin/sh", "/bin/dash"].into_iter().filter(|shell| Path::new(shell).is_file()).collect() }
impl Fixture {
    fn piped_with(&self, shell: &str, script: &[u8], cwd: &Path, env: &[(&str, &str)], args: &[&str]) -> Output {
        let mut child = Command::new(shell).args(["-c", "umask 002 && exec \"$0\" -s -- \"$@\""])
            .arg(if shell == "/bin/sh" { "sh" } else { shell }).args(args).current_dir(cwd).env_clear()
            .env("PATH", self.root.join("bin")).env("TMPDIR", self.root.join("tmp"))
            .env("HOME", self.root.join("home")).env("CARGO_NET_OFFLINE", "true")
            .env("INSTALL_ROOT", &self.root).env("INSTALL_CALLS", self.root.join("calls"))
            .envs(env.iter().copied()).stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
        // A shell that stops early closes the pipe; that is not a test failure.
        let _ = std::io::Write::write_all(&mut child.stdin.take().unwrap(), script);
        child.wait_with_output().unwrap()
    }
    /// The current folder is scripts/ of a checkout: the old script took its parent as the source.
    fn piped(&self, env: &[(&str, &str)], args: &[&str]) -> Output {
        self.piped_with("/bin/sh", &script(), &self.root.join("scripts"), env, args)
    }
    fn piped_prebuilt(&self, env: &[(&'static str, String)], args: &[&str]) -> Output {
        let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.piped(&env, args)
    }
    /// A fake `git clone` of public main: the checkout in public/, with its own
    /// vade program. Cargo says where it runs, so a test sees the build folder.
    fn public_main(&self) {
        for dir in ["scripts", "cli/target/release", "engine", "harness"] { fs::create_dir_all(self.root.join("public").join(dir)).unwrap(); }
        for file in ["cli/Cargo.toml", "engine/Cargo.toml", "harness/Cargo.toml"] { fs::write(self.root.join("public").join(file), "").unwrap(); }
        fs::write(self.root.join("public/scripts/install.sh"), script()).unwrap();
        self.write("public/cli/target/release/vade", "printf 'clone-vade %s source %s\\n' \"$*\" \"$VADE_SOURCE\" >> \"$INSTALL_CALLS\"");
        self.tool("git", r#"
printf 'git %s\n' "$*" >> "$INSTALL_CALLS"
[ "${INSTALL_FAIL_CLONE-false}" = false ] || { echo 'fatal: fixture clone failure' >&2; exit 128; }
[ "$GIT_TERMINAL_PROMPT" = 0 ] || exit 88
for target; do :; done
cp -R "$INSTALL_ROOT/public/." "$target/"
"#);
        self.tool("cargo", r#"
printf 'cargo %s in %s\n' "$2" "$PWD" >> "$INSTALL_CALLS"
[ "${INSTALL_FAIL_CARGO-false}" = false ] || { echo 'error: fixture build failure' >&2; exit 42; }
"#);
    }
    /// The folders a clone made in the Vade home.
    fn clones(&self) -> Vec<PathBuf> {
        let mut clones: Vec<PathBuf> = fs::read_dir(self.root.join("home/.vade")).map(|dir| dir.map(|e| e.unwrap().path())
            .filter(|path| path.file_name().unwrap().to_string_lossy().starts_with("source.")).collect()).unwrap_or_default();
        clones.sort();
        clones
    }
    fn tmp_is_empty(&self) -> bool { fs::read_dir(self.root.join("tmp")).unwrap().next().is_none() }
    fn run_file(&self, file: &Path, cwd: &Path, args: &[&str]) -> Output {
        Command::new("/bin/sh").arg(file).args(args).current_dir(cwd).env_clear()
            .env("PATH", self.root.join("bin")).env("TMPDIR", self.root.join("tmp"))
            .env("HOME", self.root.join("home")).env("CARGO_NET_OFFLINE", "true")
            .env("INSTALL_ROOT", &self.root).env("INSTALL_CALLS", self.root.join("calls")).output().unwrap()
    }
}

#[test]
fn the_installer_is_plain_sh_and_ends_in_main() {
    for shell in shells() {
        let check = Command::new(shell).arg("-n").arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/install.sh")).output().unwrap();
        assert!(check.status.success(), "{shell}: {}", String::from_utf8_lossy(&check.stderr));
    }
    let text = String::from_utf8(script()).unwrap();
    assert!(text.starts_with("#!/bin/sh\n"));
    assert_eq!(text.lines().last(), Some("main \"$@\""));
    assert!(text.ends_with("\n}\nmain \"$@\"\n"), "the last line needs a final newline");
    // Nothing runs before main: only comments are outside it.
    let (head, _) = text.split_once("\nmain() {\n").unwrap();
    assert!(head.lines().all(|line| line.starts_with('#')), "{head}");
    let repository = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/repository.rs")).unwrap();
    assert!(text.contains(&format!("clone_url={CLONE_URL}\n")) && repository.contains(&format!("\"{}\"", CLONE_URL.trim_end_matches(".git"))));
}

#[test]
fn a_cut_off_download_runs_nothing() {
    let whole = script();
    let text = String::from_utf8(whole.clone()).unwrap();
    let first = text.find("main() {\n").unwrap() + "main() {\n".len();
    let last = text.rfind("}\nmain \"$@\"\n").unwrap();
    let mut cuts: Vec<usize> = (first..last).step_by(211).collect();
    cuts.extend([first, last - 1, last]);
    assert!(cuts.len() > 50, "{}", cuts.len());
    let f = Fixture::new();
    f.public_main();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let clean = |context: String| {
        assert_eq!(f.calls(), "", "{context}");
        assert!(f.tree().is_empty() && f.tmp_is_empty(), "{context}: {:?}", f.tree());
    };
    for cut in cuts {
        for shell in shells() {
            let output = f.piped_with(shell, &whole[..cut], &f.root.join("scripts"), &env, &[]);
            assert!(!output.status.success(), "{shell} cut at {cut}");
            assert!(output.stdout.is_empty(), "{shell} cut at {cut}: {}", String::from_utf8_lossy(&output.stdout));
            clean(format!("{shell} cut at {cut}"));
        }
        // The same cut saved to a file in a checkout.
        fs::write(f.root.join("scripts/install.sh"), &whole[..cut]).unwrap();
        assert!(!f.run_args(&env, &[]).status.success(), "file cut at {cut}");
        clean(format!("file cut at {cut}"));
    }
    // A cut after the closing brace defines main and never calls it.
    let output = f.piped_with("/bin/sh", &whole[..last + 2], &f.root.join("scripts"), &env, &[]);
    assert!(output.status.success() && output.stdout.is_empty());
    clean("cut after main".into());
}

#[test]
fn piped_bundle_install_needs_no_checkout() {
    let f = Fixture::new();
    f.public_main();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let output = f.piped_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(f.calls(), "bundle-vade install\n");
    let folder = f.bundles().join(stem());
    assert!(stdout.contains(PATH_LINE) && !stdout.contains("Prebuilt:"), "{stdout}");
    assert_eq!(fs::read_to_string(f.script()).unwrap(), format!("#!/bin/sh\nexec '{}/vade' \"$@\"\n", folder.display()));
    assert!(f.clones().is_empty() && f.tmp_is_empty());
    // --launch opens chat on the terminal, never on the script; with no terminal it stops at the next steps.
    let output = f.piped_prebuilt(&env, &["--launch"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.ends_with("Run the commands above in a terminal.\n") != f.calls().ends_with("bundle-vade launch\n"), "{stdout}{}", f.calls());
}

#[test]
fn piped_source_build_clones_public_main_into_a_new_folder() {
    let f = Fixture::new();
    f.public_main();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let mut unfit = env.clone();
    unfit.push(("INSTALL_DRIVER", "575.51.03".into()));
    let output = f.piped_prebuilt(&unfit, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("Prebuilt: the bundle needs NVIDIA driver 580 or newer and this Spark has 575.51.03; building from source.\n"), "{stdout}");
    let clones = f.clones();
    assert_eq!(clones.len(), 1, "{stdout}");
    let clone = clones[0].display();
    let calls = f.calls();
    assert!(calls.starts_with(&format!("git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 clone --quiet --depth 1 {CLONE_URL} {clone}\n")), "{calls}");
    // The build runs in the clone, never in the folder above the current one.
    assert!(calls.contains(&format!("cargo build in {clone}\n")) && !calls.contains(&format!("in {}\n", f.root.display())), "{calls}");
    assert!(calls.ends_with(&format!("clone-vade __install-here source {clone}\n")), "{calls}");
    assert!(stdout.contains(PATH_LINE), "{stdout}");
    assert_eq!(fs::read_to_string(f.script()).unwrap(), format!("#!/bin/sh\nexec '{clone}/cli/target/release/vade' \"$@\"\n"));
    f.assert_private("clone");
    assert_eq!(fs::metadata(&clones[0]).unwrap().permissions().mode() & 0o077, 0);
    assert!(f.tmp_is_empty() && !f.root.join("target").exists());
    // --source skips latest.json; a second run takes a new folder and leaves the first.
    fs::remove_file(f.root.join("dl/main/latest.json")).unwrap();
    let output = f.piped_prebuilt(&env, &["--source"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success() && stdout.contains("Prebuilt: --source was given; building from source.\n"), "{stdout}");
    assert_eq!(f.clones().len(), 2);
    assert!(clones[0].join("scripts/install.sh").is_file() && !f.bundles().exists());
}

#[test]
fn piped_failures_leave_no_folder_and_name_the_one_line_retry() {
    for (env, args, message, retry, clones) in [
        (("INSTALL_FAIL_CLONE", "true"), &[][..], format!("cannot clone {CLONE_URL} (git error above)"), format!("Retry: {CURL_LINE}\n"), 0),
        (("INSTALL_FAIL_CLONE", "true"), &["--source"], format!("cannot clone {CLONE_URL} (git error above)"), format!("Retry: {CURL_LINE} -s -- --source\n"), 0),
        (("INSTALL_BAD_CLONE", "true"), &[], "is not a Vade checkout".into(), format!("Retry: {CURL_LINE}\n"), 0),
        // The clone is whole, so a build failure keeps it and retries in it.
        (("INSTALL_FAIL_CARGO", "true"), &[], "CLI build failed".into(), "/scripts/install.sh\" --source\n".into(), 1),
    ] {
        let f = Fixture::new();
        f.public_main();
        if env.0 == "INSTALL_BAD_CLONE" { fs::remove_file(f.root.join("public/harness/Cargo.toml")).unwrap(); }
        let output = f.piped(&[env], args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(stderr.contains(&*message) && stderr.contains("Next:") && stderr.ends_with(&*retry), "{stderr}");
        assert_eq!(f.clones().len(), clones, "{env:?}");
        assert!(f.tmp_is_empty());
        f.assert_private(env.0);
    }
    // No git: the script stops before it writes anything.
    let f = Fixture::new();
    let output = f.piped(&[], &[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && stderr.contains("missing git") && stderr.ends_with(&format!("Retry: {CURL_LINE}\n")), "{stderr}");
    assert!(f.tree().is_empty() && f.calls().is_empty(), "{:?}", f.tree());
}

#[test]
fn only_a_file_in_a_checkout_counts_as_a_checkout() {
    let f = Fixture::new();
    f.public_main();
    let plan = format!("Check: would clone {CLONE_URL} into a new folder in {}/.vade, then build from source there.\n", f.root.join("home").display());
    // Piped --check, and a saved copy outside a checkout: the clone plan, and nothing written.
    let output = f.piped(&[], &["--check"]);
    assert!(String::from_utf8_lossy(&output.stdout).ends_with(&plan), "{}", String::from_utf8_lossy(&output.stdout));
    fs::create_dir(f.root.join("tmp/away")).unwrap();
    fs::copy(f.root.join("scripts/install.sh"), f.root.join("tmp/away/install.sh")).unwrap();
    let output = f.run_file(&f.root.join("tmp/away/install.sh"), &f.root, &["--check"]);
    assert!(output.status.success() && String::from_utf8_lossy(&output.stdout).ends_with(&plan), "{}", String::from_utf8_lossy(&output.stdout));
    assert!(f.tree().is_empty() && f.calls().is_empty());
    // In a checkout, every way to name the file builds that checkout.
    let build = format!("build vade in {}/cli/target,", f.root.display());
    for (file, cwd) in [("scripts/install.sh", f.root.clone()), ("install.sh", f.root.join("scripts")), ("./install.sh", f.root.join("scripts"))] {
        let output = f.run_file(Path::new(file), &cwd, &["--source", "--check"]);
        assert!(String::from_utf8_lossy(&output.stdout).contains(&build), "{file}: {}", String::from_utf8_lossy(&output.stdout));
    }
    // A checkout without one of its parts is not one.
    fs::remove_file(f.root.join("harness/Cargo.toml")).unwrap();
    let output = f.run_file(&f.root.join("scripts/install.sh"), &f.root, &["--source", "--check"]);
    assert!(String::from_utf8_lossy(&output.stdout).ends_with(&plan), "{}", String::from_utf8_lossy(&output.stdout));
}

impl Fixture {
    /// The folder that stays on PATH, its script, and that script's text.
    fn script(&self) -> PathBuf { self.root.join("home/.vade/bin/vade") }
    /// Run the installed script as a new shell would, with the printed PATH.
    fn run_script(&self, args: &[&str]) -> Output {
        Command::new("/bin/sh").args(["-c", "export PATH=\"$HOME/.vade/bin:$PATH\" && exec vade \"$@\"", "sh"]).args(args).env_clear()
            .env("PATH", self.root.join("bin")).env("HOME", self.root.join("home"))
            .env("INSTALL_ROOT", &self.root).env("INSTALL_CALLS", self.root.join("calls")).env("VADE_SOURCE", &self.root)
            .output().unwrap()
    }
}

const PATH_LINE: &str = "Open a new terminal, or run: export PATH=\"$HOME/.vade/bin:$PATH\"\n";
const NO_SCRIPT_LINE: &str = "Run: export PATH=\"$HOME/.vade/bin:$PATH\"\n";

#[test]
fn one_fixed_folder_runs_the_current_install_on_every_path_and_shell() {
    for shell in ["/bin/sh", "/bin/dash"].into_iter().filter(|shell| Path::new(shell).is_file()) {
        let f = Fixture::new();
        let env = f.prebuilt(&bundle(Bad::None), Bad::None);
        let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
        // The source path first: this also proves the bundle path replaces the script.
        let source = f.run_in(shell, &env, &["--source"]);
        let source_out = String::from_utf8_lossy(&source.stdout).into_owned();
        assert!(source.status.success() && source_out.ends_with(PATH_LINE), "{shell}: {source_out}");
        assert_eq!(fs::read_to_string(f.script()).unwrap(), format!("#!/bin/sh\nexec '{}/cli/target/release/vade' \"$@\"\n", f.root.display()));
        assert!(f.run_script(&["from-source"]).status.success());
        assert!(f.calls().ends_with("vade from-source\n"), "{shell}: {}", f.calls());
        let installed = f.run_in(shell, &env, &[]);
        let installed_out = String::from_utf8_lossy(&installed.stdout).into_owned();
        assert!(installed.status.success() && installed_out.ends_with(PATH_LINE), "{shell}: {installed_out}");
        assert_eq!(fs::read_to_string(f.script()).unwrap(), format!("#!/bin/sh\nexec '{}/vade' \"$@\"\n", f.bundles().join(stem()).display()));
        assert!(f.run_script(&["from-bundle", "a b"]).status.success());
        assert!(f.calls().ends_with("bundle-vade from-bundle a b\n"), "{shell}: {}", f.calls());
        // Private, a plain file and never a link, no new file left, and no shell profile.
        assert_eq!(fs::metadata(f.root.join("home/.vade/bin")).unwrap().permissions().mode() & 0o077, 0);
        assert_eq!(fs::symlink_metadata(f.script()).unwrap().permissions().mode() & 0o777, 0o700);
        assert!(fs::symlink_metadata(f.script()).unwrap().is_file());
        assert!(!f.script().with_file_name("vade.new").exists());
        for profile in [".profile", ".bashrc"] { assert!(!f.root.join("home").join(profile).exists()); }
    }
}

#[test]
fn a_named_vade_home_gets_its_own_fixed_folder_and_a_quoted_script() {
    let f = Fixture::new();
    let home = f.root.join("home/it's here");
    // The fixture bundle checks HOME/.vade, as the real CLI checks its home.
    fs::create_dir(f.root.join("home/.vade")).unwrap();
    fs::set_permissions(f.root.join("home/.vade"), fs::Permissions::from_mode(0o700)).unwrap();
    let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
    env.push(("VADE_HOME", home.display().to_string()));
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    // No script in ~/.local/bin, so a new terminal would not find vade: the line says only what works.
    assert!(stdout.ends_with(&format!("Run: export PATH=\"{}/bin:$PATH\"\n", home.display())), "{stdout}");
    assert!(!f.path_script().exists() && !f.root.join("home/.local").exists(), "{stdout}");
    let run = Command::new(home.join("bin/vade")).arg("quoted").env_clear().env("PATH", f.root.join("bin"))
        .env("HOME", f.root.join("home")).env("INSTALL_CALLS", f.root.join("calls")).output().unwrap();
    assert!(run.status.success(), "{run:?}");
    assert!(f.calls().ends_with("bundle-vade quoted\n"), "{}", f.calls());
}

#[test]
fn a_folder_in_place_of_the_script_stops_with_a_next_step() {
    let f = Fixture::new();
    fs::create_dir_all(f.script()).unwrap();
    fs::set_permissions(f.root.join("home/.vade"), fs::Permissions::from_mode(0o700)).unwrap();
    let output = f.run(&[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && stderr.contains("/.vade/bin/vade. Next: check that path"), "{stderr}");
}

impl Fixture {
    /// The script for new shells, in the folder that the stock ~/.profile adds at login.
    fn path_script(&self) -> PathBuf { self.root.join("home/.local/bin/vade") }
    /// Run vade as a new login shell would: with ~/.local/bin on PATH, and no export line.
    fn run_login(&self, args: &[&str]) -> Output {
        Command::new("/bin/sh").args(["-c", "PATH=\"$HOME/.local/bin:$PATH\" && exec vade \"$@\"", "sh"]).args(args).env_clear()
            .env("PATH", self.root.join("bin")).env("HOME", self.root.join("home"))
            .env("INSTALL_ROOT", &self.root).env("INSTALL_CALLS", self.root.join("calls")).env("VADE_SOURCE", &self.root)
            .output().unwrap()
    }
    fn on_path(&self, folder: &str) -> String { format!("{}:{}", self.root.join(folder).display(), self.root.join("bin").display()) }
}

const PATH_SCRIPT: &str = "#!/bin/sh\n# Written by the Vade installer. vade-path-shim-v1\nexec \"$HOME/.vade/bin/vade\" \"$@\"\n";

#[test]
fn a_new_login_finds_vade_through_a_second_script_on_every_shell() {
    for shell in ["/bin/sh", "/bin/dash"].into_iter().filter(|shell| Path::new(shell).is_file()) {
        let f = Fixture::new();
        for run in ["first", "again"] {
            let output = f.run_in(shell, &[], &[]);
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            assert!(output.status.success() && stdout.ends_with(PATH_LINE), "{shell} {run}: {stdout}");
            assert_eq!(fs::read_to_string(f.path_script()).unwrap(), PATH_SCRIPT, "{shell} {run}");
            let meta = fs::symlink_metadata(f.path_script()).unwrap();
            assert!(meta.is_file() && !meta.file_type().is_symlink(), "{shell} {run}");
            assert_eq!(meta.permissions().mode() & 0o777, 0o700, "{shell} {run}");
            assert!(!f.path_script().with_file_name("vade.new").exists(), "{shell} {run}");
        }
        // A new login needs no export line: the stock ~/.profile adds ~/.local/bin.
        let login = f.run_login(&["from-login", "a b"]);
        assert!(login.status.success(), "{shell}: {login:?}");
        assert!(f.calls().ends_with("vade from-login a b\n"), "{shell}: {}", f.calls());
        for profile in [".profile", ".bashrc", ".bash_profile"] { assert!(!f.root.join("home").join(profile).exists(), "{shell}"); }
    }
}

/// A login shell is the only one that knows the profile. The fake one prints what a real one would.
#[test]
fn the_last_line_promises_a_new_terminal_only_when_a_login_shell_adds_the_folder() {
    let f = Fixture::new();
    let tool = |path: PathBuf, body: &str| { fs::write(&path, body).unwrap(); fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap(); path };
    tool(f.root.join("bin/timeout"), "#!/bin/sh\nshift\nexec \"$@\"\n");
    let shell = f.root.join("login-shell");
    let home = f.root.join("home").display().to_string();
    for (body, last) in [
        // A profile that does not add the folder: zsh, or a ~/.bash_profile that hides ~/.profile.
        ("#!/bin/sh\necho Welcome\necho vade-login-path=/usr/bin:/bin\n".to_owned(), "A new login here does not add ~/.local/bin to PATH, so a new terminal does not find vade.\nRun: export PATH=\"$HOME/.vade/bin:$PATH\"\n"),
        (format!("#!/bin/sh\necho vade-login-path={home}/.local/bin:/usr/bin\n"), PATH_LINE),
        ("#!/bin/sh\nexit 1\n".to_owned(), PATH_LINE),
    ] {
        tool(shell.clone(), &body);
        let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
        env.push(("SHELL", shell.display().to_string()));
        let output = f.run_prebuilt(&env, &[]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success() && stdout.ends_with(last), "{body}: {stdout}");
        assert_eq!(stdout.matches("export PATH=").count(), 1, "{stdout}");
    }
    for profile in [".profile", ".bashrc", ".bash_profile"] { assert!(!f.root.join("home").join(profile).exists()); }
}

#[test]
fn a_file_that_is_not_ours_in_local_bin_stays_as_it_was() {
    const FOREIGN: &str = "#!/bin/sh\nexec /opt/other/vade \"$@\"\n";
    for kind in ["file", "link", "folder"] {
        let f = Fixture::new();
        fs::create_dir_all(f.path_script().parent().unwrap()).unwrap();
        match kind {
            "file" => fs::write(f.path_script(), FOREIGN).unwrap(),
            "link" => std::os::unix::fs::symlink("/opt/other/vade", f.path_script()).unwrap(),
            _ => fs::create_dir(f.path_script()).unwrap(),
        }
        let output = f.run(&[]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{kind}: {stdout}");
        assert!(stdout.contains(&format!("Skipped {}: that file is not from Vade, so the installer left it alone.\n", f.path_script().display())), "{kind}: {stdout}");
        // With no script of ours, a new terminal would not find vade, so the line does not say so.
        assert!(stdout.ends_with(NO_SCRIPT_LINE), "{kind}: {stdout}");
        match kind {
            "file" => assert_eq!(fs::read_to_string(f.path_script()).unwrap(), FOREIGN),
            "link" => assert_eq!(fs::read_link(f.path_script()).unwrap(), Path::new("/opt/other/vade")),
            _ => assert!(f.path_script().is_dir()),
        }
        assert!(!f.path_script().with_file_name("vade.new").exists(), "{kind}");
    }
}

#[test]
fn a_local_folder_that_cannot_be_made_does_not_fail_the_install() {
    let f = Fixture::new();
    fs::write(f.root.join("home/.local"), "a file, not a folder").unwrap();
    let output = f.run(&[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains(&format!("Skipped {}: this account cannot write it.\n", f.path_script().display())), "{stdout}");
    assert!(stdout.ends_with(NO_SCRIPT_LINE) && f.script().is_file(), "{stdout}");
}

#[test]
fn a_shell_that_already_finds_vade_is_told_so() {
    for folder in ["home/.vade/bin", "home/.local/bin"] {
        let f = Fixture::new();
        let path = f.on_path(folder);
        let output = f.run(&[("PATH", &path)]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{folder}: {stdout}");
        assert!(stdout.ends_with("vade is on your PATH.\n") && !stdout.contains("export PATH="), "{folder}: {stdout}");
    }
    // Another vade first on PATH is not this install, so the export line stays.
    let f = Fixture::new();
    fs::create_dir(f.root.join("other")).unwrap();
    f.write("other/vade", "exit 0");
    let path = format!("{}:{}", f.on_path("other"), f.root.join("home/.vade/bin").display());
    let stdout = String::from_utf8_lossy(&f.run(&[("PATH", &path)]).stdout).into_owned();
    assert!(stdout.ends_with(PATH_LINE), "{stdout}");
}

#[test]
fn every_numbered_step_names_pull_or_up_and_launch_is_never_a_step() {
    let f = Fixture::new();
    let stdout = String::from_utf8_lossy(&f.run(&[]).stdout).into_owned();
    let steps: Vec<&str> = stdout.lines().skip_while(|line| *line != "Install complete. Next steps:").skip(1).collect();
    let commands: Vec<&str> = steps.iter().filter_map(|line| line.strip_prefix("   ")).collect();
    assert_eq!(commands.len(), 2, "{stdout}");
    assert!(commands[0].starts_with("vade pull ") && commands[1] == "vade up", "{stdout}");
    assert!(steps.iter().all(|line| !line.starts_with(|c: char| c.is_ascii_digit()) || !line.contains("launch")), "{stdout}");
    assert_eq!(stdout.matches("vade launch").count(), 0, "{stdout}");
}

#[test]
fn the_readme_the_installer_and_the_help_teach_the_same_three_commands() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let pull = "vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300";
    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(readme.contains(&format!("curl -fsSL https://vade.lol/install.sh | sh\n{pull}\nvade up\n")));
    let script = fs::read_to_string(root.join("scripts/install.sh")).unwrap();
    assert!(script.contains(&format!("'   {pull}'")) && script.contains("'   vade up'"));
    let help = Command::new(env!("CARGO_BIN_EXE_vade")).arg("--help").output().unwrap();
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains(&format!("\n  {pull}\n")) && help.contains("\n  vade up "), "{help}");
}

#[test]
fn a_bundle_install_needs_no_c_compiler() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    fs::remove_file(f.root.join("bin/cc")).unwrap();
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(f.calls(), "bundle-vade install\n");
    assert!(stdout.contains(&format!("Install: installed prebuilt Vade {SHA}.\n")) && !stdout.contains("Prebuilt:"), "{stdout}");
    // The same through the pipe, in a new home.
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    fs::remove_file(f.root.join("bin/cc")).unwrap();
    let output = f.piped_prebuilt(&env, &[]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(f.calls(), "bundle-vade install\n");
}

#[test]
fn a_source_build_with_no_c_compiler_stops_before_any_download_or_clone() {
    const STOP: &str = "missing cc, which a build from source needs. Next: ask your administrator to provide the prerequisites in docs/INSTALL.md#step-1-prerequisites, then retry.";
    // From a checkout, with --source.
    let f = Fixture::new();
    fs::remove_file(f.root.join("bin/cc")).unwrap();
    let output = f.run_args(&[], &["--source"]);
    let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{stdout}{stderr}");
    assert!(stdout.contains("Prebuilt: --source was given; building from source.\n"), "{stdout}");
    assert!(stderr.starts_with(&format!("install: {STOP}\nRetry: sh \"{}/scripts/install.sh\" --source\n", f.root.display())), "{stderr}");
    assert_eq!(f.calls(), "", "no Rust tool ran");
    // --check names the same stop and still exits 0.
    let output = f.run_args(&[], &["--source", "--check"]);
    assert!(output.status.success() && String::from_utf8_lossy(&output.stdout).contains(&format!("Check: the install would stop here: {STOP}\n")), "{output:?}");
    // Through the pipe, with a bundle that does not fit: no clone.
    let f = Fixture::new();
    let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
    env.push(("INSTALL_DRIVER", "575.51.03".into()));
    f.public_main();
    fs::remove_file(f.root.join("bin/cc")).unwrap();
    let output = f.piped_prebuilt(&env, &[]);
    let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{stdout}{stderr}");
    assert!(stdout.contains("Prebuilt: the bundle needs NVIDIA driver 580 or newer and this Spark has 575.51.03; building from source.\n"), "{stdout}");
    assert!(stderr.starts_with(&format!("install: {STOP}\nRetry: curl -fsSL https://vade.lol/install.sh | sh\n")), "{stderr}");
    assert!(f.clones().is_empty() && f.calls().is_empty(), "{}", f.calls());
}
