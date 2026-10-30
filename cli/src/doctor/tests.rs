use super::*;
use crate::runner::{Exec, Output};
use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt, path::PathBuf, sync::atomic::{AtomicU64, Ordering}, time::Duration};

static NEXT: AtomicU64 = AtomicU64::new(0);
const STATE: RemoteState = RemoteState { owned: false, memory: Some((120 << 30, 100 << 30)), split: false, cache_default: None };

struct FakeSsh { script: PathBuf, tools: PathBuf, destination: String, calls: usize, cache: String, chunk_error: bool }
impl Runner for FakeSsh {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        self.calls += 1;
        assert_eq!(exec.program, "ssh");
        let options = runner::ssh_options();
        assert_eq!(&exec.args[..options.len()], &options);
        assert_eq!(exec.args[options.len()], self.destination);
        assert_eq!(exec.args.len(), options.len() + 2);
        if self.chunk_error && exec.label.ends_with(" chunk") { return Err("The metadata chunk test could not read the file.".into()); }
        let mut fake = exec.clone();
        fake.program = self.script.to_string_lossy().into_owned();
        fake.env.insert("PATH".into(), format!("{}:{}", self.tools.display(), std::env::var("PATH").unwrap()));
        if self.cache == "unset" { fake.unset.push("VADE_FLASHNEXT_PREFIX_CACHE".into()); }
        else { fake.env.insert("VADE_FLASHNEXT_PREFIX_CACHE".into(), self.cache.clone()); }
        runner::SystemRunner { dry: false }.run(&fake)
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { panic!("Doctor must not start an interactive command.") }
    fn port_free(&mut self, _: u16) -> Result<(), String> { panic!("Remote ports must use SSH.") }
    fn pause(&mut self, _: Duration) { panic!("Doctor must not wait for a server.") }
}

fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn fixture(listeners: Option<&str>) -> (Machine, FakeSsh) {
    let folder = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-doctor-ssh-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let tools = folder.join("tools");
    let model = folder.join("model ' files");
    fs::create_dir_all(&tools).unwrap();
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("config.json"), "{\"hidden_size\":16}").unwrap();
    fs::write(model.join("tokenizer.json"), "{\"model\":{}}").unwrap();
    fs::write(model.join("part.safetensors"), [0u8; 32]).unwrap();
    fs::write(model.join("model.safetensors.index.json"), "{\"weight_map\":{\"a\":\"part.safetensors\"}}").unwrap();
    let script = folder.join("fake-ssh");
    executable(&script, "#!/bin/sh\nfor vade_arg; do vade_command=$vade_arg; done\nexec /bin/sh -c \"$vade_command\"\n");
    executable(&tools.join("ss"), &match listeners {
        Some(listeners) => format!("#!/bin/sh\nprintf '%s\\n' {}\n", runner::quote(listeners)),
        None => "#!/bin/sh\nexit 127\n".into(),
    });
    // Disk capacity must not depend on how much room the CI host has left.
    executable(&tools.join("df"), "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\nfixture 33554432 0 33554432 0%% /\\n'\n");
    let machine = Machine {
        name: "spark-a".into(), ssh: "user@spark-a".into(), verify_url: None, fabric: BTreeMap::new(),
        model_path: model.to_string_lossy().into_owned(), engine_source: "~/source/engine".into(), cli_source: "~/source/cli".into(),
        stage_dir: folder.join("install/not-created").to_string_lossy().into_owned(), state_dir: "~/state".into(),
    };
    let runner = FakeSsh { script, tools, destination: machine.ssh.clone(), calls: 0, cache: "0".into(), chunk_error: false };
    (machine, runner)
}

fn captured(run: impl FnOnce(&mut Report) -> Result<(), String>) -> (Result<(), String>, String) {
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || {
        let mut report = Report::new("controller", false);
        run(&mut report)?;
        report.finish()
    });
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    (result, printed)
}

#[test]
fn fake_ssh_checks_model_and_free_ports_on_the_configured_destination() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080), Port::new("0.0.0.0", 29960), Port::new("0.0.0.0", 29961)], false, STATE));
    result.unwrap();
    assert_eq!(runner.calls, 1);
    assert!(printed.contains("1 file, 32 bytes"), "{printed}");
    for port in [18080, 29960, 29961] { assert!(printed.contains(&format!("spark-a port {port}")), "{printed}"); }
    assert!(printed.contains("34359738368 bytes free"), "{printed}");
    assert!(printed.contains("checkpoint 32 bytes fits with working room before context"), "{printed}");
    assert!(printed.contains("PASS"), "{printed}");
}

#[test]
fn metadata_chunk_failure_keeps_disk_memory_cache_and_port_rows() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    let tokenizer = format!("{{\"model\":{{}}{}}}", " ".repeat(2 * 1024 * 1024));
    fs::write(Path::new(&machine_config.model_path).join("tokenizer.json"), tokenizer).unwrap();
    runner.chunk_error = true;
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
    assert!(result.is_err() && printed.contains("FAIL  spark-a tokenizer.json"), "{printed}");
    assert!(printed.contains("The metadata chunk test could not read the file."), "{printed}");
    for row in ["model disk", "install disk", "model memory", "default prefix cache", "port 18080"] {
        assert!(printed.contains(&format!("ok    spark-a {row}")), "{printed}");
    }
    assert!(!printed.contains("Run ssh"), "{printed}");
}

#[test]
fn fake_ssh_reports_busy_ipv4_and_ipv6_ports_with_a_fix() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some("LISTEN 0 128 0.0.0.0:18080 0.0.0.0:*\nLISTEN 0 128 [::]:29961 [::]:* v6only:0"));
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080), Port::new("0.0.0.0", 29960), Port::new("0.0.0.0", 29961)], false, STATE));
    assert!(result.is_err());
    assert_eq!(printed.matches("A program listens on this port.").count(), 2, "{printed}");
    assert!(printed.contains("Close that program or change the port in the config"), "{printed}");
    assert!(printed.contains("spark-a port 29960    free on the Spark"), "{printed}");
}

#[test]
fn fake_ssh_allows_listeners_on_other_interfaces_for_loopback_http() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some("LISTEN 0 128 127.0.0.2:18080 0.0.0.0:*\nLISTEN 0 128 [::1]:18080 [::]:*"));
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
    result.unwrap();
    assert!(printed.contains("spark-a port 18080    free on the Spark"), "{printed}");
}

#[test]
fn fake_ssh_reports_loopback_and_wildcard_http_conflicts() {
    let _guard = runner::process_test_lock();
    for endpoint in ["127.0.0.1", "0.0.0.0", "[::]", "*", "[::ffff:127.0.0.1]"] {
        let listeners = format!("LISTEN 0 128 {endpoint}:18080 *:* v6only:0");
        let (machine_config, mut runner) = fixture(Some(&listeners));
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
        assert!(result.is_err() && printed.contains("A program listens on this port."), "{printed}");
    }
}

#[test]
fn fake_ssh_checks_transport_bind_address() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some("LISTEN 0 128 127.0.0.2:29960 0.0.0.0:*"));
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::new("127.0.0.1", 29960)], false, STATE));
    result.unwrap();
    assert!(printed.contains("spark-a port 29960    free on the Spark"), "{printed}");
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::new("127.0.0.2", 29960)], false, STATE));
    assert!(result.is_err() && printed.contains("A program listens on this port."), "{printed}");
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::new("::ffff:127.0.0.2", 29960)], false, STATE));
    assert!(result.is_err() && printed.contains("A program listens on this port."), "{printed}");
}

#[test]
fn fake_ssh_distinguishes_ipv6_only_and_unknown_wildcard_listeners() {
    let _guard = runner::process_test_lock();
    for endpoint in ["[::]", "*"] {
        let listeners = format!("LISTEN 0 128 {endpoint}:18080 *:* v6only:1");
        let (machine_config, mut runner) = fixture(Some(&listeners));
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
        result.unwrap();
        assert!(printed.contains("spark-a port 18080    free on the Spark"), "{printed}");
        let listeners = format!("LISTEN 0 128 {endpoint}:18080 *:*");
        let (machine_config, mut runner) = fixture(Some(&listeners));
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
        assert!(result.is_err() && printed.contains("Cannot confirm whether the IPv6 listener also holds IPv4"), "{printed}");
        assert!(!printed.contains("Close that program"), "{printed}");
    }
}

#[test]
fn fake_ssh_reports_unavailable_ss_and_preserves_other_checks() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(None);
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
    assert!(result.is_err());
    assert!(printed.contains("Install the ss command on the Spark"), "{printed}");
    assert!(printed.contains("1 file, 32 bytes"), "{printed}");
}

struct Reply { result: Result<Output, String>, calls: usize }
impl Runner for Reply {
    fn run(&mut self, _: &Exec) -> Result<Output, String> { self.calls += 1; self.result.clone() }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { panic!("Unexpected interactive command.") }
    fn port_free(&mut self, _: u16) -> Result<(), String> { panic!("Unexpected local port check.") }
    fn pause(&mut self, _: Duration) { panic!("Unexpected wait.") }
}

#[test]
fn missing_or_malformed_remote_fields_do_not_pass() {
    let (machine_config, _) = fixture(Some(""));
    for stdout in ["", "not a record", "model_folder=present\nmodel_folder=present\n"] {
        let mut runner = Reply { result: Ok(Output { success: true, stdout: stdout.into() }), calls: 0 };
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
        assert!(result.is_err(), "{printed}");
        assert!(printed.contains("FAIL"), "{printed}");
        assert_eq!(runner.calls, 1);
    }
}

#[test]
fn dry_run_does_not_invoke_ssh() {
    let (machine_config, _) = fixture(Some(""));
    let mut runner = Reply { result: Err("SSH must not run".into()), calls: 0 };
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], true, STATE));
    result.unwrap();
    assert_eq!(runner.calls, 0);
    assert!(printed.contains("plan"), "{printed}");
    assert!(printed.contains("spark-a safetensors"), "{printed}");
}

#[test]
fn fake_ssh_reports_zero_and_insufficient_free_memory() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    for available in [0, 4 << 30] {
        let state = RemoteState { memory: Some((120 << 30, available)), ..STATE };
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
        assert!(result.is_err(), "{printed}");
        assert!(printed.contains("FAIL  spark-a model memory") && printed.contains("Not enough memory"), "{printed}");
        assert!(printed.contains("Dropping the page cache does not help"), "{printed}");
        assert!(printed.contains("1 file, 32 bytes"), "{printed}");
    }
}

#[test]
fn fake_ssh_reports_unknown_memory_and_skips_an_owned_engine() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    for owned in [false, true] {
        let state = RemoteState { owned, memory: None, ..STATE };
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
        assert!(result.is_err());
        assert!(printed.contains("free memory sample is unavailable") && printed.contains("Check /proc/meminfo"), "{printed}");
    }
    let state = RemoteState { owned: true, memory: Some((120 << 30, 0)), ..STATE };
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
    result.unwrap();
    assert!(printed.contains("skip  spark-a model memory") && printed.contains("a recorded engine can hold this memory"), "{printed}");
    assert!(!printed.contains("fits with working room"), "{printed}");
}

#[test]
fn fake_ssh_reports_split_memory_without_a_full_checkpoint_floor() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    let state = RemoteState { memory: Some((120 << 30, 4 << 30)), ..STATE };
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
    assert!(result.is_err() && printed.contains("Not enough memory"), "{printed}");
    let state = RemoteState { split: true, ..state };
    let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
    result.unwrap();
    assert!(printed.contains("4.0 GiB available") && printed.contains("startup checks the tensor share and context"), "{printed}");
    assert!(!printed.contains("fits with working room"), "{printed}");
    for memory in [None, Some((120 << 30, 0)), Some((0, 0)), Some((4 << 30, 8 << 30))] {
        let state = RemoteState { memory, ..state };
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
        assert!(result.is_err() && printed.contains("FAIL  spark-a model memory"), "{printed}");
    }
}

#[test]
fn fake_ssh_reads_the_cache_setting_from_the_spark_session() {
    let _guard = runner::process_test_lock();
    let (machine_config, mut runner) = fixture(Some(""));
    for (value, expected) in [("0", "off in the Spark SSH session"), ("1", "on in the Spark SSH session"), ("bad", "Spark prefix cache setting is invalid")] {
        runner.cache = value.into();
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, STATE));
        assert_eq!(result.is_ok(), value != "bad", "{printed}");
        assert!(printed.contains(expected), "{printed}");
    }
    // Unset follows the topology when the session names it.
    runner.cache = "unset".into();
    for (default, expected) in [(None, "unset in the Spark SSH session: on for one Spark and the tensor split, off on the layer chain"),
        (Some(true), "unset in the Spark SSH session: on for this topology"), (Some(false), "unset in the Spark SSH session: off on the layer chain")] {
        let state = RemoteState { cache_default: default, ..STATE };
        let (result, printed) = captured(|report| machine(report, &mut runner, &machine_config, &[Port::http(18080)], false, state));
        assert!(result.is_ok() && printed.contains(expected), "{printed}");
    }
}
