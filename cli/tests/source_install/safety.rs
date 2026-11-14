//! Removals never leave the Vade home, and --json always ends in one result line.
//! Every folder these tests make, link or remove is inside the fixture's own temp folder.
use super::*;
use std::time::Duration;

/// An unpacked older bundle named `vade-0.1.0-<sha>-linux-aarch64-gb10` in `dir`, `age` seconds after a fixed time.
fn old_bundle(dir: &Path, sha: &str, age: u64) -> PathBuf {
    let path = dir.join(format!("vade-0.1.0-{sha}-linux-aarch64-gb10"));
    fs::create_dir(&path).unwrap();
    fs::write(path.join("MANIFEST"), format!("format=vade-release-v1\nname={}\nsha=old\n", path.file_name().unwrap().to_str().unwrap())).unwrap();
    fs::File::open(&path).unwrap().set_modified(UNIX_EPOCH + Duration::from_secs(1_700_000_000 + age)).unwrap();
    path
}

fn installed(f: &Fixture, env: &[(&'static str, String)]) {
    let first = f.run_prebuilt(env, &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
}

#[test]
fn a_linked_bundles_folder_is_never_pruned() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    installed(&f, &env);
    let outside = f.root.join("outside bundles");
    fs::rename(f.bundles(), &outside).unwrap();
    std::os::unix::fs::symlink(&outside, f.bundles()).unwrap();
    let old = [old_bundle(&outside, "aaaaaaa", 1), old_bundle(&outside, "bbbbbbb", 2)];
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(old.iter().all(|path| path.is_dir()), "a bundle outside the home was removed: {stdout}");
    assert!(stdout.contains(&format!("Install: kept the older bundles: {}/home/.vade/bundles is a link", f.root.display())), "{stdout}");
}

#[test]
fn a_linked_home_is_refused_and_the_folder_it_names_keeps_every_bundle() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    installed(&f, &env);
    let outside = f.root.join("outside home");
    fs::rename(f.root.join("home/.vade"), &outside).unwrap();
    std::os::unix::fs::symlink(&outside, f.root.join("home/.vade")).unwrap();
    let old = [old_bundle(&outside.join("bundles"), "aaaaaaa", 1), old_bundle(&outside.join("bundles"), "bbbbbbb", 2)];
    let output = f.run_prebuilt(&env, &["--json"]);
    assert_eq!(output.status.code(), Some(1));
    let lines = json_lines(&output);
    assert!(lines.last().unwrap().contains("\"error\":{\"code\":\"bad_home\""), "{lines:?}");
    assert!(old.iter().all(|path| path.is_dir()) && outside.join("bundles").join(stem()).is_dir(), "{lines:?}");
    assert!(!outside.join(".install-lock").exists());
}

#[test]
fn a_bundle_name_with_a_space_and_a_star_is_pruned_as_one_name() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    installed(&f, &env);
    let odd = old_bundle(&f.bundles(), "a b*c", 1);
    let previous = old_bundle(&f.bundles(), "bbbbbbb", 2);
    // Names that a split list or a file name pattern would turn up instead.
    fs::write(f.root.join("vade-0.1.0-a"), "not a bundle").unwrap();
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(!odd.exists() && previous.is_dir() && f.bundles().join(stem()).is_dir(), "{stdout}");
    assert!(stdout.contains("Install: removed the older bundle vade-0.1.0-a b*c-linux-aarch64-gb10.\n"), "{stdout}");
    assert!(f.root.join("vade-0.1.0-a").is_file() && !f.bundles().join(".install-lock").exists());
}

#[test]
fn json_ends_in_one_result_line_when_a_signal_stops_the_install() {
    // The fake vade says it started, then waits for the test to send the signal.
    const WAITING_VADE: &str = r#"
printf 'vade %s\n' "$*" >> "$INSTALL_CALLS"
[ "$1" = __install-here ] || exit 0
: > "$INSTALL_ROOT/started"
waited=0
until [ -e "$INSTALL_ROOT/go" ]; do
    sleep 0.1; waited=$((waited + 1))
    [ "$waited" -lt 600 ] || exit 9
done
"#;
    for shell in shells() {
        for (signal, code) in [("HUP", 129), ("INT", 130), ("TERM", 143)] {
            let f = Fixture::new();
            std::os::unix::fs::symlink(system_tool("sleep"), f.root.join("bin/sleep")).unwrap();
            f.write("cli/target/release/vade", WAITING_VADE);
            let output = stopped_run(&f, shell, &[], &["--json"], signal);
            assert_eq!(output.status.code(), Some(code), "{shell} {signal}: {}", String::from_utf8_lossy(&output.stderr));
            let lines = json_lines(&output);
            let last = lines.last().unwrap();
            assert!(last.contains("\"ok\":false,\"error\":{\"code\":\"interrupted\",\"message\":\"the installer was stopped by SIG")
                && last.contains(&format!("SIG{signal}.\"")), "{shell} {signal}: {last}");
            assert!(!f.root.join("home/.vade/.install-lock").exists(), "{shell} {signal}: the lock stayed");
        }
    }
}

/// Start the installer with HUP, INT and TERM at their default action. A test run started in the
/// background of a script inherits INT ignored, and a shell cannot trap a signal ignored at its start:
/// the installer's INT trap would then do nothing and the test would fail for a reason outside the installer.
/// Perl resets the three signals and then replaces itself with the same command, so the pid stays the installer's.
fn with_default_signals(command: Command) -> Command {
    let mut wrapped = Command::new(system_tool("perl"));
    wrapped.args(["-e", r#"$SIG{$_} = "DEFAULT" for qw(HUP INT TERM); exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!\n""#])
        .arg(command.get_program()).args(command.get_args()).env_clear();
    for (name, value) in command.get_envs() {
        if let Some(value) = value { wrapped.env(name, value); }
    }
    if let Some(dir) = command.get_current_dir() { wrapped.current_dir(dir); }
    wrapped
}

/// Run the installer with `args`, wait until the fixture writes `started`, send `signal`, then let the fixture go on.
fn stopped_run(f: &Fixture, shell: &str, env: &[(&str, &str)], args: &[&str], signal: &str) -> Output {
    let child = with_default_signals(f.command(shell, env, args)).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped()).spawn().unwrap();
    let mut waited = 0;
    while !f.root.join("started").exists() {
        std::thread::sleep(Duration::from_millis(50));
        waited += 1;
        assert!(waited < 600, "{shell} {args:?}: the fixture never started");
    }
    let sent = Command::new(system_tool("kill")).arg(format!("-{signal}")).arg(child.id().to_string()).status().unwrap();
    assert!(sent.success());
    fs::write(f.root.join("go"), "").unwrap();
    child.wait_with_output().unwrap()
}

/// The one result line says the run was stopped, and the exit code is the signal's.
fn assert_interrupted(output: &Output, code: i32, context: &str) {
    assert_eq!(output.status.code(), Some(code), "{context}: {}", String::from_utf8_lossy(&output.stderr));
    let lines = json_lines(output);
    assert!(lines.last().unwrap().contains("\"ok\":false,\"error\":{\"code\":\"interrupted\""), "{context}: {lines:?}");
}

const WAIT_FOR_GO: &str = r#": > "$INSTALL_ROOT/started"
waited=0
until [ -e "$INSTALL_ROOT/go" ]; do
    sleep 0.1; waited=$((waited + 1))
    [ "$waited" -lt 600 ] || exit 9
done"#;

#[test]
fn a_signal_while_the_installed_vade_says_its_version_still_ends_in_one_result_line() {
    // Just before the install's result line.
    for shell in shells() {
        let f = Fixture::new();
        std::os::unix::fs::symlink(system_tool("sleep"), f.root.join("bin/sleep")).unwrap();
        f.write("cli/target/release/vade", &format!(r#"
case $1 in
    version) {WAIT_FOR_GO}
        echo '{{"sha":"1234567890abcdef1234567890abcdef12345678"}}' ;;
esac"#));
        let output = stopped_run(&f, shell, &[], &["--json"], "TERM");
        assert_interrupted(&output, 143, &format!("{shell} install"));
        assert!(f.root.join("home/.vade/bin/vade").is_file() && !f.root.join("home/.vade/.install-lock").exists());
    }
}

/// A run stopped while the clock is read for its result line: the third call of date.
fn stopped_at_the_last_clock(env: &[(&str, &str)], args: &[&str]) {
    for shell in shells() {
        let f = Fixture::new();
        std::os::unix::fs::symlink(system_tool("sleep"), f.root.join("bin/sleep")).unwrap();
        f.tool("date", &format!(r#"
calls=0
[ ! -f "$INSTALL_ROOT/date-calls" ] || read -r calls < "$INSTALL_ROOT/date-calls"
calls=$((calls + 1))
printf '%s\n' "$calls" > "$INSTALL_ROOT/date-calls"
if [ "$calls" = 3 ]; then
{WAIT_FOR_GO}
fi
exec {} "$@""#, system_tool("date").display()));
        let output = stopped_run(&f, shell, env, args, "TERM");
        assert_interrupted(&output, 143, &format!("{shell} {args:?}"));
    }
}

#[test]
fn a_signal_just_before_the_uninstall_result_still_ends_in_one_result_line() { stopped_at_the_last_clock(&[], &["--uninstall", "--json"]); }

#[test]
fn a_signal_just_before_the_check_result_still_ends_in_one_result_line() { stopped_at_the_last_clock(&[], &["--check", "--json"]); }

#[test]
fn a_signal_while_a_failure_line_is_built_still_ends_in_one_result_line() {
    // A busy GPU stops the install at the checks; the clock is read for that result line.
    stopped_at_the_last_clock(&[("INSTALL_GPU_BUSY", "true")], &["--json"]);
}

#[test]
fn a_failed_removal_in_the_exit_handler_never_leaves_the_lock_or_a_second_line() {
    let f = Fixture::new();
    f.public_main();
    // An rm that cannot remove the clone folder; every other removal works.
    fs::remove_file(f.root.join("bin/rm")).unwrap();
    f.tool("rm", &format!(r#"
for path; do case $path in */source.*) echo 'rm: fixture failure' >&2; exit 1 ;; esac; done
exec {} "$@""#, system_tool("rm").display()));
    let output = f.piped(&[("INSTALL_FAIL_CLONE", "true")], &["--json"]);
    assert_eq!(output.status.code(), Some(1));
    let lines = json_lines(&output);
    assert!(lines.last().unwrap().contains("\"code\":\"clone_failed\""), "{lines:?}");
    assert!(!f.root.join("home/.vade/.install-lock").exists(), "the lock survived the exit handler");
}

#[test]
fn the_download_never_writes_through_a_linked_bundles_folder() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    let outside = f.root.join("outside bundles");
    fs::create_dir(&outside).unwrap();
    fs::create_dir_all(f.root.join("home/.vade")).unwrap();
    fs::set_permissions(f.root.join("home/.vade"), fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(f.root.join("home/.vade/.vade-home"), "vade-home-v1\n").unwrap();
    std::os::unix::fs::symlink(&outside, f.bundles()).unwrap();
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("Prebuilt: {}/home/.vade/bundles is a link or not a plain folder; building from source.\n", f.root.display())), "{stdout}");
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0, "the download wrote through the link: {stdout}");
}

#[test]
fn an_install_stopped_half_way_leaves_a_home_that_installs_and_uninstalls() {
    let f = Fixture::new();
    std::os::unix::fs::symlink(system_tool("sleep"), f.root.join("bin/sleep")).unwrap();
    let mut env = f.prebuilt(&bundle(Bad::None), Bad::None);
    env.push(("INSTALL_BUNDLE_WAIT", "1".into()));
    let pairs: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    // Stopped while vade install runs: the bundle is unpacked, the script on PATH is not written yet.
    let output = stopped_run(&f, "/bin/sh", &pairs, &["--json"], "TERM");
    assert_interrupted(&output, 143, "stopped install");
    let home = f.root.join("home/.vade");
    assert!(home.join("bundles").join(stem()).is_dir() && !home.join("bin/vade").exists() && !home.join(".install-lock").exists());
    assert_eq!(fs::read_to_string(home.join(".vade-home")).unwrap(), "vade-home-v1\n");
    let again = f.run_prebuilt(&env, &[]);
    assert!(again.status.success(), "{}", String::from_utf8_lossy(&again.stderr));
    assert!(home.join("bin/vade").is_file());
    let removed = f.run_prebuilt(&env, &["--uninstall", "--all"]);
    assert!(removed.status.success(), "{}", String::from_utf8_lossy(&removed.stderr));
    assert!(!home.exists(), "{}", String::from_utf8_lossy(&removed.stdout));
}

/// A file outside every folder the installer may write, with text the tests check is unchanged.
fn outside_file(f: &Fixture) -> PathBuf {
    let path = f.root.join("outside.txt");
    fs::write(&path, "mine\n").unwrap();
    path
}

#[test]
fn the_local_bin_script_is_never_written_through_a_link() {
    // A link at the temporary name of the script.
    let f = Fixture::new();
    let outside = outside_file(&f);
    fs::create_dir_all(f.root.join("home/.local/bin")).unwrap();
    std::os::unix::fs::symlink(&outside, f.root.join("home/.local/bin/vade.new")).unwrap();
    let output = f.run(&[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(fs::read_to_string(&outside).unwrap(), "mine\n", "{stdout}");
    assert!(f.path_script().is_file() && !f.path_script().is_symlink(), "{stdout}");
    assert!(f.root.join("home/.local/bin/vade.new").is_symlink(), "{stdout}");
    // A link at ~/.local or at ~/.local/bin: nothing is written through it.
    for linked in ["home/.local", "home/.local/bin"] {
        let f = Fixture::new();
        let elsewhere = f.root.join("elsewhere");
        fs::create_dir_all(elsewhere.join("bin")).unwrap();
        if linked == "home/.local/bin" { fs::create_dir(f.root.join("home/.local")).unwrap(); }
        let target = if linked == "home/.local" { elsewhere.clone() } else { elsewhere.join("bin") };
        std::os::unix::fs::symlink(&target, f.root.join(linked)).unwrap();
        let output = f.run(&[]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{linked}: {stdout}");
        assert_eq!(fs::read_dir(elsewhere.join("bin")).unwrap().count(), 0, "{linked}: {stdout}");
        let home = f.root.join("home");
        assert!(stdout.contains(&format!("Skipped {}: {}/.local or {}/.local/bin is a link, and the installer never writes through one.\n",
            f.path_script().display(), home.display(), home.display())), "{linked}: {stdout}");
    }
}

#[test]
fn the_home_script_is_never_written_through_a_link() {
    // A link at the temporary name in the home's bin.
    let f = Fixture::new();
    assert!(f.run(&[]).status.success());
    let outside = outside_file(&f);
    std::os::unix::fs::symlink(&outside, f.root.join("home/.vade/bin/vade.new")).unwrap();
    let output = f.run(&[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(fs::read_to_string(&outside).unwrap(), "mine\n", "{stdout}");
    assert!(f.script().is_file() && !f.script().is_symlink(), "{stdout}");
    // A linked bin: the install stops, and nothing is written in the folder it names.
    let f = Fixture::new();
    assert!(f.run(&[]).status.success());
    let elsewhere = f.root.join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    fs::remove_dir_all(f.root.join("home/.vade/bin")).unwrap();
    std::os::unix::fs::symlink(&elsewhere, f.root.join("home/.vade/bin")).unwrap();
    let output = f.run(&[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("bin must be a folder, not a link"), "{stderr}");
    assert_eq!(fs::read_dir(&elsewhere).unwrap().count(), 0, "{stderr}");
}

#[test]
fn an_older_folder_with_a_bare_manifest_is_never_pruned() {
    let f = Fixture::new();
    let env = f.prebuilt(&bundle(Bad::None), Bad::None);
    installed(&f, &env);
    // Bundle names, but the MANIFEST does not start with the format line or names another folder.
    let bare = f.bundles().join("vade-0.1.0-fffffff-linux-aarch64-gb10");
    let other = f.bundles().join("vade-0.1.0-eeeeeee-linux-aarch64-gb10");
    for (dir, manifest) in [(&bare, "sha=old\n".to_owned()), (&other, "format=vade-release-v1\nname=vade-0.1.0-aaaaaaa-linux-aarch64-gb10\n".to_owned())] {
        fs::create_dir(dir).unwrap();
        fs::write(dir.join("MANIFEST"), manifest).unwrap();
        fs::write(dir.join("notes.txt"), "mine\n").unwrap();
        fs::File::open(dir).unwrap().set_modified(UNIX_EPOCH + Duration::from_secs(1_700_000_000)).unwrap();
    }
    let (older, previous) = (old_bundle(&f.bundles(), "aaaaaaa", 1), old_bundle(&f.bundles(), "bbbbbbb", 2));
    let output = f.run_prebuilt(&env, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(bare.join("notes.txt").is_file() && other.join("notes.txt").is_file(), "{stdout}");
    assert!(!older.exists() && previous.is_dir(), "{stdout}");
    assert!(!stdout.contains("fffffff") && !stdout.contains("eeeeeee"), "{stdout}");
}

/// --uninstall --json on `home` (VADE_HOME): exit 1, a bad_home line, and nothing removed.
fn refused_uninstall(f: &Fixture, env: &[(&str, &str)], context: &str) {
    let output = f.run_args(env, &["--uninstall", "--all", "--json"]);
    assert_eq!(output.status.code(), Some(1), "{context}: {}", String::from_utf8_lossy(&output.stderr));
    let lines = json_lines(&output);
    assert!(lines.last().unwrap().contains("\"error\":{\"code\":\"bad_home\""), "{context}: {lines:?}");
}

#[test]
fn an_empty_bundles_folder_or_cache_proves_no_vade_home() {
    let cache = format!(".partial-{}", "a".repeat(64));
    for layout in ["bundles", "bundles/.partial"] {
        let f = Fixture::new();
        let home = f.root.join("other");
        let dir = if layout == "bundles" { home.join("bundles") } else { home.join("bundles").join(&cache) };
        fs::create_dir_all(&dir).unwrap();
        refused_uninstall(&f, &[("VADE_HOME", home.to_str().unwrap())], layout);
        assert!(dir.is_dir() && !home.join(".vade-home").exists(), "{layout}");
        // The install does not take it over either.
        let output = f.run(&[("VADE_HOME", home.to_str().unwrap())]);
        assert_eq!(output.status.code(), Some(1), "{layout}: {}", String::from_utf8_lossy(&output.stdout));
        assert!(!home.join(".vade-home").exists(), "{layout}");
    }
}

#[test]
fn a_cache_lock_that_holds_a_file_proves_no_vade_home() {
    let f = Fixture::new();
    let home = f.root.join("other");
    let cache = home.join("bundles").join(format!(".partial-{}", "a".repeat(64)));
    fs::create_dir_all(cache.join("lock")).unwrap();
    fs::write(cache.join("bundle.part"), "part").unwrap();
    fs::write(cache.join("lock/notes.txt"), "mine\n").unwrap();
    refused_uninstall(&f, &[("VADE_HOME", home.to_str().unwrap())], "lock with a file");
    assert!(cache.join("lock/notes.txt").is_file());
}

#[test]
fn your_home_folder_reached_through_a_link_is_never_a_vade_home() {
    let f = Fixture::new();
    let real = f.root.join("deep/real-home");
    fs::create_dir_all(real.join("sessions")).unwrap();
    fs::write(real.join(".vade-home"), "vade-home-v1\n").unwrap();
    fs::write(real.join("sessions/chat"), "mine\n").unwrap();
    let link = f.root.join("link-home");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    for home in [real.clone(), f.root.join("deep")] {
        refused_uninstall(&f, &[("HOME", link.to_str().unwrap()), ("VADE_HOME", home.to_str().unwrap())], &home.display().to_string());
        assert!(real.join("sessions/chat").is_file());
    }
}

#[test]
fn the_config_folder_loses_only_the_files_vade_writes() {
    let f = Fixture::new();
    let config = f.root.join("home/.config/vade");
    fs::create_dir_all(&config).unwrap();
    for name in ["config", "config.before", "config.before-2", "notes.txt", "config.before-x"] { fs::write(config.join(name), "x\n").unwrap(); }
    let output = f.run_args(&[], &["--uninstall", "--all"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    let mut left: Vec<String> = fs::read_dir(&config).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    assert_eq!(left, ["config.before-x", "notes.txt"], "{stdout}");
    assert!(stdout.contains(&format!("Kept {} (it holds files that are not from Vade)\n", config.display())), "{stdout}");
    // Only Vade's files: they go, then the folder.
    fs::remove_file(config.join("notes.txt")).unwrap();
    fs::remove_file(config.join("config.before-x")).unwrap();
    fs::write(config.join("config"), "x\n").unwrap();
    let output = f.run_args(&[], &["--uninstall", "--all"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success() && !config.exists(), "{stdout}");
    assert!(stdout.contains(&format!("Removed {}\n", config.display())), "{stdout}");
}

#[test]
fn a_folder_where_vade_keeps_a_file_is_never_removed() {
    let f = Fixture::new();
    assert!(f.run(&[]).status.success());
    let token = f.root.join("home/.vade/hf-token");
    fs::create_dir(&token).unwrap();
    fs::write(token.join("notes.txt"), "mine\n").unwrap();
    let output = f.run_args(&[], &["--uninstall", "--all"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(token.join("notes.txt").is_file(), "{stdout}");
    assert!(stdout.contains(&format!("Kept {} (a folder where Vade keeps a file: the uninstall never removes one)\n", token.display())), "{stdout}");
}
