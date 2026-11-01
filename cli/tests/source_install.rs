//! Exercise the actual source installer with fake host tools, never a GPU or network.
use std::{fs, os::unix::fs::PermissionsExt, path::{Path, PathBuf}, process::{Command, Output}, time::{SystemTime, UNIX_EPOCH}};

static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct Fixture { root: PathBuf }
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("vade source install {}-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        for dir in ["bin", "scripts", "engine", "cli/target/release", "home", "tmp"] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/install.sh"), root.join("scripts/install.sh")).unwrap();
        let fixture = Self { root: fs::canonicalize(root).unwrap() };
        for tool in ["sh", "dirname", "mktemp", "rm", "mkdir", "cp", "chmod", "tail", "sed", "head", "ls"] {
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
    fn run_args(&self, env: &[(&str, &str)], args: &[&str]) -> Output {
        // Ubuntu's default umask 002: a folder the script creates is group-writable unless it says otherwise.
        Command::new("/bin/sh").args(["-c", "umask 002 && exec /bin/sh \"$0\" \"$@\""]).arg(self.root.join("scripts/install.sh"))
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
    assert!(calls.ends_with(&format!("{fetch}vade install\n")), "{calls}");
    assert!(!calls.contains("kernels") && !calls.contains("vade detect"), "{calls}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Install: fetching the Rust build-std files for the GPU kernels (none when cached).\n"), "{stdout}");
    // One next path: PATH, the model when it is missing, then launch; no second install through vade setup.
    assert!(stdout.contains(&format!("   export PATH=\"$HOME/.cargo/bin:{}/cli/target/release:$PATH\"\n", f.root.display())), "{stdout}");
    assert!(stdout.contains("   vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300\n") && stdout.ends_with("   vade launch\n"), "{stdout}");
    assert!(!stdout.contains("setup"), "{stdout}");
}

#[test]
fn newer_drivers_print_cuda_umd_version_and_pass_the_same_check() {
    let f = Fixture::new();
    let output = f.run(&[("INSTALL_UMD", "true")]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(f.calls().ends_with("vade install\n"), "{}", f.calls());
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
    assert!(f.calls().ends_with("vade install\nvade launch\n"), "{}", f.calls());
    fs::write(f.root.join("calls"), "").unwrap();
    let output = f.run_args(&[("INSTALL_FAIL_VADE", "install")], &["--launch"]);
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
        (("INSTALL_FAIL_VADE", "install"), "fixture diagnostic: fix this cause", "vade launch"),
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
            assert!(calls.ends_with("vade install\n"), "{calls}");
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
    assert!(stdout.contains(&format!("   export PATH=\"{}:$PATH\"\n", folder.display())), "{stdout}");
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
        assert!(calls.contains("cargo +1.99.0 build") && calls.ends_with("\nvade install\n"), "{reason}: {calls}");
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
    assert!(f.calls().starts_with("bundle-vade install\n") && f.calls().ends_with("\nvade install\n"), "{}", f.calls());
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
