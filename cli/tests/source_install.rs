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
        for tool in ["sh", "dirname", "mktemp", "rm", "mkdir", "cp", "chmod", "tail"] {
            let path = ["/usr/bin", "/bin"].into_iter().map(|dir| Path::new(dir).join(tool))
                .find(|path| path.is_file()).unwrap();
            std::os::unix::fs::symlink(path, fixture.root.join("bin").join(tool)).unwrap();
        }
        fixture.tool("uname", "case $1 in -s) echo Linux ;; -m) echo aarch64 ;; esac");
        fixture.tool("cc", "exit 0");
        fixture.tool("nvidia-smi", "[ \"${INSTALL_GPU_BUSY-false}\" = false ] || echo 1234");
        fixture.tool("curl", "echo unexpected-download >> \"$INSTALL_CALLS\"; exit 91");
        fixture.tool("rustup", r#"
printf 'rustup %s\n' "$*" >> "$INSTALL_CALLS"
if [ "$1" = component ]; then
    [ "${INSTALL_MISSING_DEVICE-false}" = false ] || exit 1
    printf '%s\n' rust-src llvm-tools-aarch64-unknown-linux-gnu llvm-bitcode-linker-aarch64-unknown-linux-gnu rust-std-nvptx64-nvidia-cuda
elif [ "$1" = run ]; then
    [ "${INSTALL_MISSING_HOST-false}" = false ] || [ "$2" != 1.99.0 ] || exit 1
fi
"#);
        fixture.tool("cargo", r#"
printf 'cargo %s\n' "$*" >> "$INSTALL_CALLS"
if [ "$2" = kernels ]; then
    printf 'warning: unused variable in a kernel\n' >&2
    [ "${INSTALL_FAIL_KERNEL-false}" = false ] || { echo 'error: fixture kernel failure' >&2; exit 42; }
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
fn cached_source_install_uses_pins_and_preflights_before_kernel_build() {
    let f = Fixture::new();
    let output = f.run(&[("CARGO_TARGET_DIR", "/unrelated/target")]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let calls = f.calls();
    assert!(!calls.contains("unexpected-download"), "{calls}");
    assert!(!calls.contains("toolchain install"), "{calls}");
    assert!(calls.contains("cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml"), "{calls}");
    assert!(calls.find("vade detect").unwrap() < calls.find("cargo +1.99.0 kernels").unwrap());
    assert!(calls.ends_with("vade install\n"), "{calls}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("\"{}/cli/target/release/vade\" launch", f.root.display())), "{stdout}");
    // Kernel compiler warnings go to the log, not to the screen (fresh install step 4).
    let log = f.root.join("engine/target/install-kernels.log");
    assert!(stdout.contains(&format!("Log: {}", log.display())), "{stdout}");
    assert!(fs::read_to_string(&log).unwrap().contains("warning: unused variable"));
    assert!(!stdout.contains("warning:") && !String::from_utf8_lossy(&output.stderr).contains("warning:"), "{stdout}");
}

#[test]
fn a_failed_kernel_build_shows_the_end_of_its_log() {
    let f = Fixture::new();
    let output = f.run(&[("INSTALL_FAIL_KERNEL", "true")]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error: fixture kernel failure") && stderr.contains("install-kernels.log"), "{stderr}");
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
        (("INSTALL_MISSING_HOST", "true"), "rerun this script online", "cargo "),
        (("INSTALL_MISSING_DEVICE", "true"), "rerun this script online", "cargo "),
        (("INSTALL_FAIL_KERNEL", "true"), "GPU kernel bootstrap failed", "vade install"),
        (("INSTALL_FAIL_VADE", "detect"), "fixture diagnostic: fix this cause", "cargo +1.99.0 kernels"),
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
