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
        for tool in ["sh", "dirname", "mktemp", "rm", "mkdir", "cp", "chmod", "tail", "sed", "head"] {
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
        Command::new("/bin/sh").arg(self.root.join("scripts/install.sh"))
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
