//! Offline launcher and recovery proofs; every command runs through FakeRunner.
#[path = "install_tests.rs"]
mod install_tests;
#[path = "tests/owner_dispatch.rs"]
mod owner_dispatch;
#[path = "tests/chat_guards.rs"]
mod chat_guards;
#[path = "tests/two_spark.rs"]
mod two_spark;
use super::*;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct FakeRunner { calls: Vec<Exec>, started: BTreeSet<String>, fail: Option<String>, foreign: Option<String>, bad_hash: bool, ports: Vec<u16>, dry: bool, doctor_bad: bool, tunnel_bad: bool, refused_start: Option<String>, inspect_error: Option<String>, mac_secret_seen: bool, alter_build_outputs: bool, local_state: PathBuf, exit_on_health: Option<String>, ready_model: Option<String>, health_extra: String, silent_probe: bool, gateway: Option<String>, gateway_start: Option<String>, gateway_exits: bool, gateway_stubborn: bool, signals: Vec<String>, probe: Option<(bool, &'static str)> }
impl Runner for FakeRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        self.calls.push(exec.clone());
        let request = if exec.program == "ssh" && exec.args.last().is_some_and(|a| a.contains("__worker")) { text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap() } else { Fields::new() };
        let name = exec.label.split_whitespace().next().unwrap_or("").to_string();
        let op = request.get("op").map(String::as_str).unwrap_or("");
        if op == "start" {
            let secret = &request["env.SEQPAR_CONTROL_SECRET"];
            if !self.dry { self.mac_secret_seen |= fs::read_dir(&self.local_state).unwrap().filter_map(Result::ok).any(|entry| entry.file_name().to_string_lossy().starts_with("env-") || fs::read_to_string(entry.path()).is_ok_and(|value| value.contains(secret))); }
            if self.refused_start.as_ref() == Some(&name) { return Ok(Output { success: false, stdout: "error_code=start_not_written\nphase=absent\nerror=This machine has another recorded owner. Stop it with its original launcher first.\n".into() }); }
            self.started.insert(name.clone());
        }
        if !self.dry && matches!(exec.label.as_str(), "harness build" | "controller CLI build") {
            let target = PathBuf::from(&exec.args[exec.args.iter().position(|a| a == "--target-dir").unwrap()+1]).join("release");
            let binary = &exec.args[exec.args.iter().position(|a| a == "--bin").unwrap()+1]; fs::create_dir_all(&target).unwrap(); fs::write(target.join(binary), b"Inert fake build artifact.\n").unwrap();
        }
        if op == "probe" && self.silent_probe { return Ok(Output { success: false, stdout: String::new() }); }
        if exec.label == "harness doctor" && self.doctor_bad { return Ok(Output { success: false, stdout: String::new() }); }
        if exec.label == "controller tunnel check" && self.tunnel_bad { return Ok(Output { success: false, stdout: String::new() }); }
        match exec.label.as_str() { "replica gateway" => { self.gateway = Some(exec.args.join(" ")); return Ok(Output { success: true, stdout: "4242\n".into() }) }
            "gateway identity" | "gateway command" => {
                assert_eq!(exec.env.get("LC_ALL").map(String::as_str), Some("C"), "ps columns are read in the C locale");
                let value = self.gateway.as_ref().map(|args| if exec.label == "gateway identity" { self.gateway_start.clone().unwrap_or_else(|| "Sat Oct 24  9:05:01 2026".into()) } else if self.gateway_exits { "<defunct>".into() } else { format!("/bin/vade-harness {args}") });
                return Ok(Output { success: value.is_some(), stdout: value.map(|v| format!("{v}\n")).unwrap_or_default() });
            }
            "replica gateway stop" => { assert_eq!(exec.args[1], "4242"); self.signals.push(exec.args[0].clone()); if exec.args[0] == "-KILL" || !self.gateway_stubborn { self.gateway = None; } } _ => {} }
        if self.fail.as_ref() == Some(&exec.label) { self.fail = None; return Err("Fake lost reply or interrupted cleanup.".into()); }
        if matches!(op, "inspect" | "stop" | "clean" | "gpu") && self.refused_start.as_ref() == Some(&name) { return Err("Foreign owner must never receive cleanup.".into()); }
        if op == "inspect" && self.foreign.as_ref() == Some(&name) { return Ok(Output { success: false, stdout: text::encode(&Fields::from([("error".into(), self.inspect_error.clone().unwrap_or_else(|| "The saved process number now belongs to a different process. Nothing was signaled. Leave these files in place; see Recovery in cli/README.md.".into()))])) }); }
        if op == "stop" { self.started.remove(&name); }
        let stdout = if exec.label == "fabric address resolution" { "127.0.0.1 STREAM\n".into() } else if matches!(exec.label.as_str(), "controller binary hashes" | "harness hash") {
            if self.alter_build_outputs && exec.label == "controller binary hashes" {
                for build in self.calls.iter().filter(|e| matches!(e.label.as_str(), "harness build" | "controller CLI build")) {
                    let target = PathBuf::from(&build.args[build.args.iter().position(|a| a == "--target-dir").unwrap()+1]);
                    let binary = &build.args[build.args.iter().position(|a| a == "--bin").unwrap()+1]; fs::write(target.join("release").join(binary), b"Changed after staging.\n").unwrap();
                }
            }
            exec.args[2..].iter().map(|path| {
                let good = fs::read(path).is_ok_and(|bytes| bytes == b"Inert fake build artifact.\n");
                let digit = if !good { "9" } else if Path::new(path).file_name().unwrap() == "vade-harness" { "4" } else { "5" };
                format!("{}  {path}\n", digit.repeat(64))
            }).collect::<String>()
        } else if exec.label.ends_with("tracked files") { "Cargo.toml\0src/main.rs\0src/flash-serve.rs\0".into()
        } else if exec.label.ends_with("remotes") { "origin\n".into()
        } else if exec.label.ends_with("artifact hashes") || exec.label.ends_with("build and stage") {
            format!("{}  binary\n{}  ptx\n{}  worker\n", if self.bad_hash { "9" } else { "1" }.repeat(64), "2".repeat(64), "3".repeat(64))
        } else if exec.label == "controller capabilities" { r#"{"object":"engine.capabilities","prefix_cache":false}"#.into()
        } else if exec.label == "controller events" {
            // A replica (read at 127.0.0.1:<its tunnel port>) reports its own counts.
            match exec.args.last().unwrap().strip_prefix("http://127.0.0.1:").and_then(|url| url.strip_suffix("/v1/events")) {
                Some(port) => format!(r#"{{"counts":{{"replica_{port}":1}},"recent":[]}}"#),
                None => r#"{"counts":{"prefix_cache_miss":2,"scheduler_pool":1},"recent":[{"unix_ms":0,"name":"scheduler_pool","line":"scheduler_pool active=8 connections=16"}]}"#.into(),
            }
        } else if exec.label.contains("health") {
            if let Some(name) = self.exit_on_health.take() { self.started.remove(&name); }
            format!("{{\"status\":\"ready\",\"model\":\"{}\"{}}}", self.ready_model.as_deref().unwrap_or("qwen3.8-flash-next-fp8"), self.health_extra)
        } else if op == "paths" { text::encode(&Fields::from([("ptx_path".into(), request["ptx_path"].replacen("~/", "/srv/", 1)), ("model_path".into(), request["model_path"].replacen("~/", "/srv/", 1))]))
        } else if op == "listener" { "listener=owned\n".into()
        } else if op == "inspect" { format!("alive={}\nphase=running\n", self.started.contains(&name))
        } else if op == "clean" { "log=Fake engine log.\n".into()
        } else if op == "logs" { text::encode(&Fields::from([("log".into(), "Fake engine log.\nflashnext ready world=1 context=32768 concurrency=rows active=8 connections=16\n".into())]))
        } else if !op.is_empty() { "phase=running\ngpu=empty\n".into()
        } else if let Some((success, stdout)) = self.probe.filter(|_| exec.label.ends_with("prerequisites")) { return Ok(Output { success, stdout: stdout.into() });
        } else if exec.label.ends_with("prerequisites") { "weights=present\nrust=ok\ndriver=ok\ndisk_kb=99999999\nmem_total_kb=127608480\nmem_available_kb=115343360\n".into()
        } else if exec.label.ends_with("doctor files") {
            let mut fields = Fields::from([
                ("model_folder".into(), "present".into()), ("model_config".into(), "present".into()),
                ("model_config_hex".into(), "7b7d".into()), ("model_tokenizer".into(), "present".into()),
                ("model_tokenizer_hex".into(), "7b7d".into()), ("model_index".into(), "present".into()),
                ("model_index_hex".into(), "7b227765696768745f6d6170223a7b2278223a226d6f64656c2e7361666574656e736f7273227d7d".into()),
                ("prefix_cache".into(), "0".into()),
                ("model_shards".into(), "1".into()), ("model_bytes".into(), "16".into()),
                ("model_complete".into(), "1".into()), ("model_shard.0".into(), "6d6f64656c2e7361666574656e736f7273:16".into()),
                ("model_disk_kb".into(), "99999999".into()), ("install_disk_kb".into(), "99999999".into()),
            ]);
            for port in [18080, 29960, 29961] { fields.insert(format!("port_{port}"), "free".into()); }
            text::encode(&fields)
        } else if exec.label.ends_with("revision") { match exec.label.as_str() { "CLI revision" => "b", "harness revision" => "c", _ => "a" }.repeat(40) } else { String::new() };
        Ok(Output { success: true, stdout })
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.calls.push(exec.clone()); Ok(true) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.ports.push(port); Ok(()) }
    fn pause(&mut self, _: Duration) {}
}
struct Fixture { dir: PathBuf, app: App<FakeRunner> }
impl Fixture {
    fn fp4(count: usize, dry: bool) -> Self {
        let mut fixture = Self::new(count, dry);
        fixture.app.config.model = "qwen3.8-flash-next-fp4-local".into();
        fixture.app.runner.ready_model = Some(fixture.app.config.model.clone());
        fixture
    }
    fn new(count: usize, dry: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = PathBuf::from("target").join(format!("vade-cli-t-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&dir).unwrap();
        let mut input = "[global]\nstate_dir = /state\nstage_dir = bin\nmodel = qwen3.8-flash-next-fp8\nmodel_repo = Qwen/Qwen3.8-Flash-Next-FP8\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n".to_string();
        for index in 0..count {
            let name = char::from(b'a' + index as u8);
            input.push_str(&format!("[machine {name}]\nssh = {name}-ssh\n"));
            for peer in [index.checked_sub(1), (index + 1 < count).then_some(index + 1)].into_iter().flatten() {
                let peer = char::from(b'a' + peer as u8);
                input.push_str(&format!("fabric.{peer} = fabric-{name}-to-{peer}\n"));
            }
        }
        let mut config = Config::parse(&input, &dir.join("config")).unwrap();
        // No real SSH socket is opened by FakeRunner. Keep its pathname short
        // without placing fixture files in the system temporary directory.
        config.state_dir = dir.join("state");
        config.activation = config.state_dir.join("build.conf");
        config.build_manifest = config.activation.clone();
        for source in [&config.engine_source, &config.harness_source, &config.cli_source] { fs::create_dir_all(source.join("src")).unwrap(); fs::write(source.join("Cargo.toml"), "# Fake source manifest.\n").unwrap(); fs::write(source.join("src/main.rs"), "fn main() {}\n").unwrap(); fs::write(source.join("src/flash-serve.rs"), "// FLASH_SERVE_OUTPUT_TOKENS\n").unwrap(); }
        let artifacts = config.machines.iter().map(|machine| (machine.name.clone(), Artifact {
            worker: format!("~/stage/{}/vade", machine.name), binary: format!("~/stage/{}/flash-serve", machine.name), ptx: "~/stage/flash.ptx".into(),
            binary_sha256: "1".repeat(64), ptx_sha256: "2".repeat(64), worker_sha256: "3".repeat(64),
        })).collect();
        text::private_dir(&config.stage_dir).unwrap();
        for binary in ["vade-harness", "vade"] { fs::write(config.stage_dir.join(binary), b"Inert fake build artifact.\n").unwrap(); }
        let manifest = Manifest { engine_sha: "a".repeat(40), cli_sha: "b".repeat(40), harness_sha256: Some("4".repeat(64)), supports_output: true, supports_fp4_chain: true, artifacts };
        text::atomic_write(&config.build_manifest, &manifest.encode()).unwrap();
        let runner = FakeRunner { dry, local_state: config.state_dir.clone(), ..FakeRunner::default() };
        Self { dir, app: App { config, runner, dry } }
    }
    fn recorded(&self) -> bool { self.app.config.state_dir.join("session.conf").exists() }
}
impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.dir).unwrap(); } }
fn inspections_precede_stops(calls: &[Exec], count: usize) {
    let stop = calls.iter().position(|e| e.label.ends_with(" stop")).unwrap();
    assert_eq!(calls.iter().filter(|e| e.label.ends_with(" inspect")).count(), count);
    assert!(calls.iter().enumerate().filter(|(_, e)| e.label.ends_with(" inspect")).all(|(i, _)| i < stop));
}
fn legacy_options() -> UpOptions {
    let Command::Up(options) = crate::cli::parse(&["up".into(), "--fp8-legacy".into()]).unwrap().command else { panic!() };
    options
}
fn fp4_options(extra: &[&str]) -> UpOptions {
    let args = ["up", "--fp4-local"].into_iter().chain(extra.iter().copied()).map(str::to_string).collect::<Vec<_>>();
    let Command::Up(options) = crate::cli::parse(&args).unwrap().command else { panic!() };
    options
}
#[test]
fn fp4_single_spark_launch_has_no_fabric_or_rdma_and_persists_the_profile() {
    let mut f = Fixture::fp4(1, false);
    let options = fp4_options(&[]);
    f.app.up(&options).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(session.fp4_local);
    // Controller defaults equal Route A; the engine's admitted count is recorded at ready.
    assert_eq!((session.context, session.output, session.chats, session.admitted_chats), (32768, 8192, None, Some(8)));
    assert_eq!(super::chats_label(&session), "8 (engine-admitted at this context)");
    assert_eq!(session.ranks.len(), 1);
    assert_eq!((&*session.ranks[0].listen, &*session.ranks[0].next), ("127.0.0.1:0", "127.0.0.1:0"));
    let args = session.engine_args(0);
    assert_eq!(args[0..4], ["0", "1", "127.0.0.1:0", "127.0.0.1:0"]);
    assert_eq!(args[6], "32768");
    assert!(args.iter().any(|s| s == "--fp4-local"));
    let start = f.app.runner.calls.iter().find(|call| call.label.ends_with(" start") && call.label != "controller tunnel start").unwrap();
    // One Spark keeps its checkpoint receipt private across starts.
    assert_eq!(text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap()["model_receipt"], "private");
    assert_eq!(args.iter().any(|s| s == "--mtp"), options.mtp_explicit);
    assert!(!f.app.runner.calls.iter().any(|c| c.label == "fabric address resolution"));
    let start = f.app.runner.calls.iter().find(|c| c.label.ends_with(" start")).unwrap();
    let request = text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap();
    assert_eq!(request["env.SEQPAR_RDMA"], "0");
    assert_eq!(request["env.TP_RDMA"], "0");
    assert_eq!(request["env_unset.MTP"], "1");
    let probe = f.app.runner.calls.iter().find(|c| c.label.ends_with(" probe")).unwrap();
    assert_eq!(text::parse(std::str::from_utf8(&probe.stdin).unwrap()).unwrap()["rdma"], "0");
    f.app.down().unwrap();
    assert!(!f.recorded());
}
#[test]
fn admitted_chats_come_from_the_last_engine_ready_line() {
    use crate::state::admitted_chats;
    let ready = |active: &str| format!("flashnext ready world=1 model=m context=32768 concurrency=rows active={active} connections=16");
    assert_eq!(admitted_chats(&format!("load\n{}\nrequest\n", ready("8"))), Some(8));
    assert_eq!(admitted_chats(&format!("{}\n{}\n", ready("1"), ready("8"))), Some(8));
    for log in ["", "Fake engine log.", "flash_serve ready slots=900", &ready("0"), &ready("33"), &ready("x")] {
        assert_eq!(admitted_chats(log), None, "{log}");
    }
    let mut f = Fixture::fp4(1, false);
    f.app.up(&fp4_options(&["--chats", "3"])).unwrap();
    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(super::chats_label(&session), "8 (engine-admitted at this context; requested 3)");
    session.admitted_chats = None;
    assert_eq!(super::chats_label(&session), "engine-admitted from context and memory (count not reported; requested 3)");
    session.chats = None;
    assert_eq!(super::chats_label(&session), "engine-admitted from context and memory (count not reported)");
}
#[test]
fn fp4_context_rungs_reach_engine_argv() {
    for (extra, sparks, context, output) in [
        (&[][..], 1usize, "32768", "8192"),
        (&["--chats", "1"][..], 1, "32768", "8192"),
        (&["--chats", "4"][..], 1, "32768", "8192"),
        (&["--chats", "8"][..], 1, "32768", "8192"),
        (&["--chats", "4", "--context", "8192"][..], 1, "8192", "8192"),
        (&["--context", "262144"][..], 1, "262144", "8192"),
        (&["--chats", "1", "--context", "262144", "--output", "16384"][..], 1, "262144", "16384"),
        (&["--sparks", "2"][..], 2, "32768", "8192"),
        (&["--sparks", "2", "--chats", "1"][..], 2, "32768", "8192"),
    ] {
        let mut f = Fixture::fp4(sparks, false);
        f.app.up(&fp4_options(extra)).unwrap();
        let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        assert_eq!(session.context.to_string(), context, "{extra:?}");
        assert_eq!(session.output.to_string(), output, "{extra:?}");
        assert_eq!(session.engine_args(0)[6], context, "{extra:?}");
    }
}
#[test]
fn fp4_pair_preserves_explicit_budgets_depth_and_rank_arguments() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2", "--mtp", "0", "--context", "8192", "--output", "4096"])).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(session.fp4_local && session.mtp_explicit);
    assert_eq!((session.context, session.output, session.mtp), (8192, 4096, 0));
    for i in 0..2 {
        let args = session.engine_args(i);
        assert_eq!(args[0], i.to_string());
        assert_eq!(args[1], "2");
        assert_eq!(&args[9..], &["--fp4-local", "--mtp", "0"]);
        assert!(!args.iter().any(|s| s == "--tensor-map"));
    }
    assert_eq!(session.ranks[0].next, session.ranks[1].listen);
    assert_eq!(session.ranks[1].next, session.ranks[0].listen);
}
#[test]
fn fp4_pair_reaches_user_at_host_ranks_without_ssh_aliases() {
    let mut f = Fixture::fp4(2, false);
    let hosts = ["user@127.0.0.1", "user@192.0.2.11"];
    for (machine, host) in f.app.config.machines.iter_mut().zip(hosts) { machine.ssh = host.into(); }
    f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(session.ranks.iter().map(|rank| rank.machine.ssh.as_str()).collect::<Vec<_>>(), hosts);
    for host in hosts { assert!(f.app.runner.calls.iter().any(|e| e.program == "ssh" && e.args.iter().any(|a| a == host)), "{host}"); }
    assert!(f.app.runner.calls.iter().filter(|e| e.program == "ssh").all(|e| e.args.iter().any(|a| hosts.contains(&a.as_str()))));
    f.app.down().unwrap();
    assert!(!f.recorded());
}
#[test]
fn unsupported_fp4_rank_counts_fail_before_any_remote_command() {
    let mut f = Fixture::fp4(4, false);
    for count in ["25", "48", "64"] {
        assert!(f.app.up(&fp4_options(&["--sparks", count])).unwrap_err().contains("1..24 Sparks"));
        assert!(f.app.runner.calls.is_empty());
        assert!(!f.recorded());
    }
}

#[test]
fn fp4_chains_cover_every_supported_size_and_preserve_owned_recovery() {
    for count in 1..=crate::config::MAX_FP4_SPARKS {
        let f = Fixture::fp4(count, false);
        let options = fp4_options(&["--sparks", &count.to_string()]);
        let session = Session::new(&f.app.config, &options, &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
        assert_eq!(session.ranks.len(), count);
        for index in 0..count {
            let args = session.engine_args(index);
            assert_eq!(args[0], index.to_string()); assert_eq!(args[1], count.to_string());
            assert!(!args.iter().any(|arg| arg == "--tensor-map" || arg == "--mtp"));
            if index + 1 < count { assert_eq!(session.ranks[index].next, session.ranks[index + 1].listen); }
        }
        if count != 2 {
            assert_eq!(session.ranks[0].listen, "127.0.0.1:0");
            assert_eq!(session.ranks[count - 1].next, "127.0.0.1:0");
        }
        session.save(&f.app.config.state_dir).unwrap();
        let restored = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        for index in 0..count { assert_eq!(restored.engine_args(index), session.engine_args(index)); }
    }
}

#[test]
fn fp4_chain_requires_build_capability_and_checks_links_before_remote_commands() {
    let mut f = Fixture::fp4(4, false);
    let current = f.app.manifest().unwrap();
    let old = current.encode().lines().filter(|line| !line.starts_with("supports_fp4_chain=")).collect::<Vec<_>>().join("\n");
    let old = Manifest::parse(&old).unwrap();
    assert!(!old.supports_fp4_chain);
    text::atomic_write(&f.app.config.build_manifest, &old.encode()).unwrap();
    let error = f.app.up(&fp4_options(&["--sparks", "4"])).unwrap_err();
    assert!(error.contains("Run vade install") && error.contains("Nothing started"), "{error}");
    assert!(f.app.runner.calls.is_empty() && !f.recorded());
    assert!(Manifest::parse(&current.encode().replace("supports_fp4_chain=true", "supports_fp4_chain=maybe")).is_err());
    text::atomic_write(&f.app.config.build_manifest, &current.encode()).unwrap();
    f.app.config.machines[2].fabric.remove("b");
    assert!(f.app.up(&fp4_options(&["--sparks", "4"])).unwrap_err().contains("Add fabric"));
    assert!(f.app.runner.calls.is_empty() && !f.recorded());
}

#[test]
fn fp4_chain_transport_launch_and_failed_start_cleanup_cover_every_rank() {
    const CASE: &str = "VADE_TEST_FP4_CHAIN_TRANSPORT";
    if let Ok(expected) = std::env::var(CASE) {
        for count in [3, 4, 6, 24] {
            let mut f = Fixture::fp4(count, false);
            f.app.up(&fp4_options(&["--sparks", &count.to_string(), "--mtp", "0"])).unwrap();
            let starts = f.app.runner.calls.iter().filter(|call| call.label.ends_with(" start") && call.label != "controller tunnel start");
            assert_eq!(starts.clone().count(), count);
            for start in starts {
                let request = text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap();
                assert!(request.keys().all(|key| !key.contains("VADE_W2")), "the engine chooses the chain itself");
                assert_eq!(request["model_receipt"], "pipeline");
                assert_eq!(request["env.SEQPAR_RDMA"], expected);
                assert_eq!(request["arg.1"], count.to_string());
                assert_eq!(request["arg.11"], "0");
            }
            f.app.runner.calls.clear(); f.app.down().unwrap();
            inspections_precede_stops(&f.app.runner.calls, count);
            assert!(!f.recorded() && f.app.runner.started.is_empty());
        }
        let mut f = Fixture::fp4(4, false);
        f.app.runner.refused_start = Some("c".into());
        assert!(f.app.up(&fp4_options(&["--sparks", "4"])).is_err());
        assert!(!f.recorded() && f.app.runner.started.is_empty());
        assert!(!f.app.runner.calls.iter().any(|call| call.label == "d start"));
        return;
    }
    for rdma in ["0", "1"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "app::tests::fp4_chain_transport_launch_and_failed_start_cleanup_cover_every_rank", "--nocapture"])
            .env(CASE, rdma).env("SEQPAR_RDMA", rdma).output().unwrap();
        assert!(output.status.success(), "{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
fn install_records_chain_support_from_the_built_source_and_old_sources_stay_off() {
    for supported in [false, true] {
        let mut f = Fixture::fp4(3, false);
        if supported {
            fs::write(f.app.config.engine_source.join("src/flash-serve.rs"),
                "// FLASH_SERVE_OUTPUT_TOKENS\n// next_service::run_pp(rank, world, listen, next, ...);\n").unwrap();
        }
        f.app.install().unwrap();
        let manifest = f.app.manifest().unwrap();
        assert_eq!(manifest.supports_fp4_chain, supported);
        assert_eq!(Manifest::parse(&manifest.encode()).unwrap().supports_fp4_chain, supported);
    }
}
#[test]
fn old_ownership_records_keep_the_legacy_launch_recipe() {
    let f = Fixture::new(2, false);
    let session = Session::new(&f.app.config, &legacy_options(), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    let encoded = session.encode().lines().filter(|line| !line.starts_with("fp4_local=") && !line.starts_with("mtp_explicit=")).collect::<Vec<_>>().join("\n");
    text::atomic_write(&f.app.config.state_dir.join("session.conf"), &encoded).unwrap();
    let restored = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(!restored.fp4_local && restored.mtp_explicit);
    assert_eq!(restored.engine_args(0), session.engine_args(0));
}
#[test]
fn credentials_precede_side_effects_and_open_health_ignores_them() {
    let _process_guard = crate::runner::process_test_lock();
    const CASE: &str = "VADE_AUTH_ENTRY_TEST";
    if let Ok(case) = std::env::var(CASE) {
        if case == "health" {
            crate::auth::configure(&mut Vec::new()).unwrap();
            let mut f = Fixture::new(2, false);
            f.app.up(&legacy_options()).unwrap();
            assert_eq!(Session::load(&f.app.config.state_dir).unwrap().unwrap().phase, "running");
            for command in [Command::Status, Command::Menu, Command::Down] { f.app.execute(command).unwrap(); }
        } else {
            let mut args = vec![case.clone()]; if case == "run" { args.push("qwen3-8b".into()); }
            crate::run_main(args);
            panic!("stale credential was accepted");
        }
        return;
    }
    let root = std::env::temp_dir().canonicalize().unwrap().join(format!("vade-auth-entry-{}", text::random_hex(8).unwrap()));
    let key = root.join("missing-key");
    for case in ["run", "chat", "doctor", "health"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "app::tests::credentials_precede_side_effects_and_open_health_ignores_them", "--nocapture"])
            .env(CASE, case).env("VADE_API_KEY_FILE", &key).env("VADE_HOME", &root).env_remove("VADE_CONFIG").output().unwrap();
        if case == "health" { assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr)); }
        else { assert_eq!(output.status.code(), Some(1)); let error = String::from_utf8_lossy(&output.stderr); assert!(error.contains("Cannot read --api-key-file") && error.contains("path not found") && error.contains("Check the path")); assert!(error.contains(&key.to_string_lossy().to_string())); }
        assert!(!root.exists(), "credential failure wrote state");
    }
}
#[test]
fn dead_tunnel_socket_does_not_block_owned_cleanup() {
    let mut f = Fixture::new(2, false); f.app.up(&legacy_options()).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap(); let socket = session.socket(&f.app.config.state_dir);
    drop(std::os::unix::net::UnixListener::bind(&socket).unwrap());
    assert!(socket.exists()); f.app.runner.tunnel_bad = true; f.app.runner.calls.clear();
    f.app.execute(Command::Down).unwrap(); inspections_precede_stops(&f.app.runner.calls, 2);
    assert!(!socket.exists() && !f.recorded() && f.app.runner.started.is_empty());
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "controller tunnel exit"));
}
#[test]
fn unverified_listening_or_foreign_socket_blocks_all_stops() {
    use std::os::unix::fs::symlink;
    let mut f = Fixture::new(2, false); f.app.up(&legacy_options()).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap(); let socket = session.socket(&f.app.config.state_dir);
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap(); f.app.runner.tunnel_bad = true; f.app.runner.calls.clear();
    assert!(f.app.execute(Command::Down).unwrap_err().contains("Recovery in cli/README.md"));
    assert!(socket.exists() && f.recorded());
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" stop") || e.label == "controller tunnel exit"));
    drop(listener); fs::remove_file(&socket).unwrap(); fs::write(&socket, "Foreign file.").unwrap(); f.app.runner.calls.clear();
    assert!(f.app.execute(Command::Down).is_err()); assert_eq!(fs::read_to_string(&socket).unwrap(), "Foreign file.");
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" stop") || e.label == "controller tunnel exit"));
    fs::remove_file(&socket).unwrap(); let foreign = f.dir.join("foreign"); fs::write(&foreign, "Keep.").unwrap(); symlink(&foreign, &socket).unwrap();
    assert!(f.app.execute(Command::Down).is_err()); assert_eq!(fs::read_to_string(&foreign).unwrap(), "Keep.");
    assert!(f.recorded() && f.app.runner.started.len() == 2);
}
#[test]
fn recovery_errors_keep_files_before_any_stop() {
    let mut f = Fixture::new(2, false); f.app.up(&legacy_options()).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap(); let socket = session.socket(&f.app.config.state_dir);
    drop(std::os::unix::net::UnixListener::bind(&socket).unwrap()); f.app.runner.tunnel_bad = true; f.app.runner.foreign = Some("b".into());
    for reason in ["This machine rebooted after launch.", "Start was interrupted before the process was saved."] {
        f.app.runner.inspect_error = Some(format!("{reason} Leave these files in place; see Recovery in cli/README.md.")); f.app.runner.calls.clear();
        let error = f.app.execute(Command::Down).unwrap_err(); assert!(error.contains(reason) && error.contains("Recovery in cli/README.md") && !error.contains("Preserve") && !error.contains("journal"));
        assert!(f.recorded() && socket.exists() && f.app.runner.started.len() == 2);
        assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" stop") || e.label.starts_with("controller tunnel")));
    }
    let recovery = include_str!("../../README.md").split("## Recovery").nth(1).unwrap();
    for step in ["session.conf", "active.conf", "journal.conf", "env.conf", "boot_id", "start_ticks", "exe_hex", "cmdline_hex", "permanent `.lock`"] { assert!(recovery.contains(step), "Missing recovery step: {step}"); }
}
#[test]
fn secret_never_reaches_mac_disk_even_when_start_and_cleanup_fail() {
    let mut f = Fixture::new(2, false); f.app.runner.fail = Some("b start".into()); f.app.runner.foreign = Some("b".into());
    assert!(f.app.up(&legacy_options()).is_err()); assert!(f.recorded());
    assert!(!f.app.runner.mac_secret_seen);
    assert!(!fs::read_dir(&f.app.config.state_dir).unwrap().any(|entry| entry.unwrap().file_name().to_string_lossy().starts_with("env-")));
    f.app.runner.foreign = None; f.app.execute(Command::Down).unwrap();
}
#[test]
fn foreign_start_with_nothing_written_rolls_back_only_owned_ranks() {
    let mut f = Fixture::new(2, false); f.app.runner.refused_start = Some("b".into());
    assert!(f.app.up(&legacy_options()).unwrap_err().contains("another recorded owner"));
    inspections_precede_stops(&f.app.runner.calls, 1);
    assert!(f.app.runner.started.is_empty() && !f.recorded());
    assert!(!f.app.runner.calls.iter().any(|e| ["b inspect", "b stop", "b gpu", "b clean"].contains(&e.label.as_str())));
}
#[test]
fn install_refuses_shared_directories_without_changing_permissions() {
    use std::os::unix::fs::PermissionsExt;
    for shared_manifest in [false, true] {
        let mut f = Fixture::new(2, false); let shared = f.dir.join("shared"); fs::create_dir(&shared).unwrap(); fs::set_permissions(&shared, fs::Permissions::from_mode(0o755)).unwrap(); fs::write(shared.join("keep"), "Other job.").unwrap();
        if shared_manifest { f.app.config.activation = shared.join("build.conf"); } else { f.app.config.stage_root = shared.clone(); }
        let error = f.app.execute(Command::Install).unwrap_err(); assert!(error.contains(&shared.display().to_string()) && error.contains(&format!("chmod 700 {}", quote(&shared.to_string_lossy()))));
        assert_eq!(fs::metadata(&shared).unwrap().permissions().mode() & 0o777, 0o755); assert_eq!(fs::read_to_string(shared.join("keep")).unwrap(), "Other job.");
        assert_eq!(fs::read_dir(&shared).unwrap().count(), 1);
    }
    let mut f = Fixture::new(2, false); f.app.runner.fail = Some("a build directories".into());
    assert!(f.app.execute(Command::Install).is_err());
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with("source copy") || e.label.ends_with("build and stage")));
}
#[test]
fn repeated_installs_reuse_machine_caches_and_keep_immutable_builds() {
    let mut f = Fixture::new(2, false); f.app.execute(Command::Install).unwrap(); f.app.execute(Command::Install).unwrap();
    for machine in &f.app.config.machines {
        let builds = f.app.runner.calls.iter().filter(|e| e.label == format!("{} build and stage", machine.name)).collect::<Vec<_>>();
        let helpers = f.app.runner.calls.iter().filter(|e| e.label == format!("{} helper build", machine.name)).collect::<Vec<_>>();
        assert_eq!((builds.len(), helpers.len()), (2, 2));
        for (build, helper) in builds.into_iter().zip(helpers) {
            let engine = build.args.last().unwrap(); let cli = helper.args.last().unwrap();
            assert!(engine.contains("vade_private_dir \"$destination\" existing") && cli.contains(" new\ncp \"$cli_cache/release/vade\""));
            assert!(engine.contains("export CARGO_TARGET_DIR=\"$engine_cache\"") && cli.contains("CARGO_TARGET_DIR=\"$cli_cache\" cargo build"));
            assert!(engine.contains("ln -s \"$engine_cache\" target") && cli.contains("ln -s \"$cli_cache\" target"));
            for script in [engine, cli] {
                assert!(script.contains("/revisions/") && script.contains("export RUSTUP_TOOLCHAIN=1.99.0"));
                assert!(!script.contains("rm -rf") && !script.contains("cargo clean"));
            }
            assert!(engine.contains(&format!("printf '%s\\n' {} > REVISION", quote(&"a".repeat(40)))));
            assert!(engine.contains("cp \"$engine_cache/release/flash-serve\""));
        }
    }
}
#[test]
fn staged_hashes_survive_changed_build_outputs_and_guard_every_harness_use() {
    let mut f = Fixture::new(2, false); f.app.runner.alter_build_outputs = true; f.app.execute(Command::Install).unwrap();
    let manifest = text::parse(&fs::read_to_string(&f.app.config.build_manifest).unwrap()).unwrap(); assert_eq!(manifest["harness_sha256"], "4".repeat(64)); assert_eq!(manifest["local_cli_sha256"], "5".repeat(64));
    let hash = f.app.runner.calls.iter().find(|e| e.label == "controller binary hashes").unwrap(); assert_eq!(hash.args[2], f.app.config.stage_dir.join("vade-harness").to_string_lossy()); assert_eq!(hash.args[3], f.app.config.stage_dir.join("vade").to_string_lossy());
    assert_eq!(fs::read(f.app.config.harness_source.join("target/release/vade-harness")).unwrap(), b"Changed after staging.\n");
    f.app.up(&legacy_options()).unwrap(); fs::write(f.app.config.stage_dir.join("vade-harness"), "Changed staged harness.").unwrap();
    for command in [Command::Chat, Command::Doctor] {
        // Chat stops at the hash; doctor reports it as its harness row's FAIL.
        f.app.runner.calls.clear(); let doctor = command == Command::Doctor; let error = f.app.execute(command).unwrap_err();
        assert!(if doctor { error.starts_with("1 doctor check(s) failed.") } else { error.contains("differs from its saved hash") }, "{error}");
        assert!(f.app.runner.calls.iter().any(|e| e.label == "harness hash")); assert!(!f.app.runner.calls.iter().any(|e| e.label == "Vade chat" || e.label.starts_with("harness doctor")));
    }
    f.app.execute(Command::Down).unwrap(); f.app.runner.calls.clear(); assert!(f.app.harness_doctor(true).unwrap_err().contains("differs from its saved hash"));
    assert!(!f.app.runner.calls.iter().any(|e| e.label.starts_with("harness doctor")));
    fs::write(f.app.config.stage_dir.join("vade-harness"), b"Inert fake build artifact.\n").unwrap(); let mut fields = manifest; fields.remove("harness_sha256"); text::atomic_write(&f.app.config.build_manifest, &text::encode(&fields)).unwrap();
    assert!(f.app.harness_doctor(true).unwrap_err().contains("no saved hash"));
}
#[test]
fn fake_full_lifecycle_and_recorded_targets() {
    let mut f = Fixture::new(2, false);
    f.app.execute(Command::Menu).unwrap(); assert!(f.app.runner.calls.is_empty());
    f.app.up(&UpOptions { mtp: 2, chats: Some(3), ..legacy_options() }).unwrap(); assert!(f.recorded());
    assert_eq!(Session::load(&f.app.config.state_dir).unwrap().unwrap().phase, "running");
    assert_eq!(f.app.runner.ports, vec![f.app.config.local_port]);
    let launch = f.app.runner.calls.iter().find(|e| e.label == "a start").unwrap();
    let rdma = if std::env::var("SEQPAR_RDMA").as_deref() == Ok("0") { "0" } else { "1" };
    for key in ["SEQPAR_RDMA", "TP_RDMA", "TP_RDMA_SMALL"] { assert_eq!(text::parse(&String::from_utf8_lossy(&launch.stdin)).unwrap()[&format!("env.{key}")], rdma); }
    assert!(f.app.runner.calls.iter().filter(|e| e.label.ends_with(" probe")).all(|e| text::parse(&String::from_utf8_lossy(&e.stdin)).unwrap()["rdma"] == rdma));
    let request = text::parse(&String::from_utf8_lossy(&launch.stdin)).unwrap(); assert_eq!(request["env.TP_TOPOLOGY"], "ring"); assert_eq!(request["env.SPARK_PROFILE"], "0"); for name in ["FLASH_SERVE_KV_BYTES", "SEQPAR_MEMORY_BYTES"] { match std::env::var(name) { Ok(value) => assert_eq!(request[&format!("env.{name}")], value.parse::<usize>().unwrap().to_string()), Err(_) => assert_eq!(request[&format!("env_unset.{name}")], "1") } } assert_eq!(request["env.FLASH_SERVE_MAX_ACTIVE"], "3"); assert_eq!(Session::load(&f.app.config.state_dir).unwrap().unwrap().chats, Some(3));
    assert_eq!(request["arg.4"], "/srv/stage/flash.ptx"); assert_eq!(request["arg.4"], Session::load(&f.app.config.state_dir).unwrap().unwrap().ranks[0].artifact.ptx);
    let first_start = f.app.runner.calls.iter().position(|e| e.label.ends_with(" start")).unwrap(); assert_eq!(f.app.runner.calls[..first_start].iter().filter(|e| e.label.ends_with(" paths")).count(), 2);
    assert_eq!(request["env_unset.FLASH_BF16_MMA"], "1"); assert_eq!(request["env.TP_WATCH_STDIN"], "0"); let secret = &request["env.SEQPAR_CONTROL_SECRET"]; assert!(!launch.args.iter().any(|a| a.contains(secret)) && !fs::read_to_string(f.app.config.state_dir.join("session.conf")).unwrap().contains(secret));
    assert!(!fs::read_dir(&f.app.config.state_dir).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().starts_with("env-")));
    f.app.execute(Command::Menu).unwrap(); f.app.execute(Command::Status).unwrap(); f.app.execute(Command::Chat).unwrap();
    let mut starting = Session::load(&f.app.config.state_dir).unwrap().unwrap(); starting.phase = "starting".into(); starting.save(&f.app.config.state_dir).unwrap(); f.app.runner.started.remove("b"); assert!(f.app.execute(Command::Status).unwrap_err().contains("stopped or missing ranks"));
    f.app.runner.started.insert("b".into()); starting.phase = "running".into(); starting.save(&f.app.config.state_dir).unwrap();
    let chat = f.app.runner.calls.iter().find(|e| e.label == "Vade chat").unwrap();
    assert_eq!(chat.args[0], "agent"); assert!(!chat.args.contains(&"--plain".into()) && chat.unset.contains(&"OPENAI_API_KEY".into()));
    assert_eq!(chat.args.contains(&"--no-check".into()), cfg!(target_os = "linux"));
    for machine in &mut f.app.config.machines { machine.ssh = "replacement-ssh".into(); machine.stage_dir = "~/replacement-stage".into(); }
    let start = f.app.runner.calls.len(); f.app.execute(Command::Down).unwrap();
    inspections_precede_stops(&f.app.runner.calls[start..], 2);
    let stop = f.app.runner.calls[start..].iter().find(|e| e.label == "a stop").unwrap();
    assert_eq!(stop.args[stop.args.len()-2], "a-ssh"); assert!(stop.args.last().unwrap().contains("stage/a/vade"));
    assert!(!f.recorded());
    for context in [4096, 131072] {
        let mut old = Fixture::new(2, false); let mut manifest = old.app.manifest().unwrap(); manifest.supports_output = false; text::atomic_write(&old.app.config.build_manifest, &manifest.encode()).unwrap();
        assert!(old.app.up(&UpOptions { context, output_explicit: true, ..legacy_options() }).unwrap_err().contains("fixed output limit")); old.app.up(&UpOptions { context, ..legacy_options() }).unwrap(); assert_eq!(Session::load(&old.app.config.state_dir).unwrap().unwrap().output, context.min(8192));
    }
}
#[test]
fn hash_failure_never_launches_and_lost_reply_rolls_back() {
    let mut f = Fixture::new(2, false); f.app.runner.bad_hash = true;
    assert!(f.app.up(&legacy_options()).unwrap_err().contains("hashes differ"));
    assert!(!f.recorded() && !f.app.runner.calls.iter().any(|e| e.label.ends_with(" start")));
    f.app.runner.bad_hash = false; f.app.runner.calls.clear(); f.app.runner.fail = Some("b start".into());
    assert!(f.app.up(&legacy_options()).is_err()); inspections_precede_stops(&f.app.runner.calls, 2);
    assert!(f.app.runner.started.is_empty()); assert!(!f.recorded());
    f.app.runner.calls.clear(); f.app.runner.exit_on_health = Some("b".into()); let error = f.app.up(&legacy_options()).unwrap_err(); assert!(error.contains("b (rank 1) exited") && error.contains("Fake engine log."));
    assert!(!f.recorded() && f.app.runner.started.is_empty() && !f.app.runner.calls.iter().any(|e| e.label == "controller tunnel start"));
}
#[test]
fn foreign_identity_blocks_every_stop_and_partial_cleanup_retries() {
    let mut f = Fixture::new(2, false); f.app.up(&legacy_options()).unwrap();
    f.app.runner.calls.clear(); f.app.runner.foreign = Some("b".into());
    assert!(f.app.execute(Command::Down).is_err()); assert!(f.recorded());
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" stop") || e.label == "controller tunnel exit"));
    f.app.runner.foreign = None; f.app.runner.calls.clear(); f.app.runner.fail = Some("b clean".into());
    assert!(f.app.execute(Command::Down).is_err()); assert!(f.recorded());
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap(); assert!(!session.ranks[0].attempted && session.ranks[1].attempted);
    f.app.runner.calls.clear(); f.app.execute(Command::Down).unwrap(); inspections_precede_stops(&f.app.runner.calls, 1);
    assert!(!f.recorded());
}
#[test]
fn unavailable_rank_cleanup_keeps_evidence_and_cleans_reachable_owner() {
    let mut f = Fixture::new(2, false); f.app.up(&legacy_options()).unwrap();
    f.app.runner.calls.clear(); f.app.runner.fail = Some("a inspect".into());
    assert!(f.app.execute(Command::Down).unwrap_err().contains("Transport:"));
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(session.ranks[0].attempted && !session.ranks[1].attempted && f.recorded());
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "a stop"));
    assert!(f.app.runner.calls.iter().any(|e| e.label == "b stop"));
    f.app.execute(Command::Down).unwrap(); assert!(!f.recorded());
}
#[test]
fn activation_failure_keeps_previous_generation() {
    let mut f = Fixture::new(2, false);
    f.app.execute(Command::Install).unwrap(); let active = fs::read(&f.app.config.activation).unwrap(); let old = f.app.config.stage_dir.clone();
    f.app.runner.fail = Some("controller binary hashes".into()); assert!(f.app.execute(Command::Install).is_err());
    assert_eq!(fs::read(&f.app.config.activation).unwrap(), active); f.app.config.resolve_generation().unwrap(); assert_eq!(f.app.config.stage_dir, old);
    f.app.up(&legacy_options()).unwrap();
    for call in f.app.runner.calls.iter().filter(|e| e.label.ends_with(" start") && !e.stdin.is_empty()) { let request = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap(); for key in ["arg.2", "arg.3"] { assert!(request[key].parse::<std::net::SocketAddr>().is_ok()); } }
    f.app.execute(Command::Down).unwrap();
}
#[test]
fn topology_addresses_and_six_ranks() {
    let mut f = Fixture::new(6, true); let manifest = f.app.manifest().unwrap();
    let ring = Session::new(&f.app.config, &legacy_options(), &manifest, "a".repeat(64)).unwrap();
    assert_eq!(ring.topology(), "ring"); assert_eq!(ring.ranks[0].next, ring.ranks[1].listen);
    f.app.config.machines[0].fabric.insert("b".into(), std::net::Ipv6Addr::LOCALHOST.to_string());
    let path = Session::new(&f.app.config, &UpOptions { sparks: Some(4), ..legacy_options() }, &manifest, "a".repeat(64)).unwrap();
    assert_eq!(path.topology(), "path"); assert!(path.ranks[0].listen.starts_with('['));
    assert_eq!(path.ranks[1].next, path.ranks[2].listen); assert!(path.ranks[1].next.starts_with("fabric-c-to-b:"));
    assert_eq!(path.ranks[3].next, path.ranks[0].listen);
    let six = Session::new(&f.app.config, &UpOptions { sparks: Some(6), ..legacy_options() }, &manifest, "a".repeat(64)).unwrap();
    assert_eq!(six.ranks[5].next, six.ranks[0].listen);
    let mut changed = manifest.clone(); changed.artifacts.get_mut("d").unwrap().ptx_sha256 = "9".repeat(64);
    assert!(Session::new(&f.app.config, &UpOptions { sparks: Some(4), ..legacy_options() }, &changed, "a".repeat(64)).unwrap_err().contains("different engine or PTX hashes"));
    let mut live = Fixture::new(4, false); live.app.up(&UpOptions { sparks: Some(4), ..legacy_options() }).unwrap();
    let starts = live.app.runner.calls.iter().filter(|e| e.label.ends_with(" start") && !e.stdin.is_empty()).collect::<Vec<_>>(); assert_eq!(starts.len(), 4);
    for (rank, call) in starts.iter().enumerate() {
        let request = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
        assert_eq!(request["env.TP_TOPOLOGY"], "path"); assert_eq!(request["arg.0"], rank.to_string()); assert_eq!(request["arg.1"], "4");
    }
    live.app.execute(Command::Down).unwrap(); assert!(live.app.runner.started.is_empty());
}
#[test]
fn dry_install_update_doctor_plan_builds_without_downloads() {
    let mut f = Fixture::new(2, true); let original = fs::read(&f.app.config.build_manifest).unwrap(); f.app.execute(Command::Install).unwrap(); f.app.execute(Command::Update).unwrap(); f.app.execute(Command::Doctor).unwrap();
    assert_eq!(fs::read(&f.app.config.build_manifest).unwrap(), original);
    assert!([&f.app.config.engine_source, &f.app.config.harness_source, &f.app.config.cli_source].iter().all(|p| !p.join("target").exists()));
    assert!(!f.recorded());
    assert_eq!(f.app.runner.calls.iter().filter(|e| e.label.ends_with(" update")).count(), 3);
    assert!(f.app.runner.calls.iter().filter(|e| e.label.ends_with(" update")).all(|e| e.args.ends_with(&["pull".into(), "--ff-only".into()])));
    assert!(f.app.runner.calls.iter().any(|e| e.label.ends_with("build and stage") && e.args.last().unwrap().contains("cargo kernels")));
    assert!(!f.app.runner.calls.iter().any(|e| e.program == "hf" || e.args.iter().any(|a| a.contains("hf download"))));
    let doctor = f.app.runner.calls.iter().find(|e| e.label == "harness doctor").unwrap();
    assert_eq!(doctor.timeout, Duration::from_secs(150)); assert_eq!(doctor.args.last().unwrap(), "120"); assert_eq!(doctor.args[0], "doctor"); assert_eq!(doctor.args[1], format!("http://localhost:{}", f.app.config.local_port)); assert!(doctor.unset.contains(&"OPENAI_BASE_URL".into()));
}
#[test]
fn fake_install_and_update_stage_fresh_builds_keep_a_running_engine() {
    let mut f = Fixture::new(2, false); f.app.execute(Command::Install).unwrap();
    let fields = text::parse(&fs::read_to_string(&f.app.config.build_manifest).unwrap()).unwrap();
    for (key, value) in [("engine_sha", "a".repeat(40)), ("cli_sha", "b".repeat(40)), ("harness_source_sha", "c".repeat(40)), ("harness_sha256", "4".repeat(64)), ("local_cli_sha256", "5".repeat(64))] { assert_eq!(fields[key], value); }
    for name in ["a", "b"] { for (key, digit) in [("binary_sha256", "1"), ("ptx_sha256", "2"), ("worker_sha256", "3")] { assert_eq!(fields[&format!("machine.{name}.{key}")], digit.repeat(64)); } }
    assert!(f.app.config.stage_dir.join("vade-harness").is_file() && f.app.config.stage_dir.join("vade").is_file());
    let first_worker = fields["machine.a.worker"].clone(); f.app.up(&legacy_options()).unwrap();
    let owned_record = fs::read(f.app.config.state_dir.join("session.conf")).unwrap(); f.app.runner.doctor_bad = true;
    assert!(f.app.harness_doctor(true).is_err()); f.app.runner.calls.clear();
    // Piped test stdin avoids any real restart choice; the running engine is retained.
    f.app.execute(Command::Update).unwrap(); assert_eq!(f.app.runner.started.len(), 2); assert_eq!(fs::read(f.app.config.state_dir.join("session.conf")).unwrap(), owned_record);
    let fields = text::parse(&fs::read_to_string(&f.app.config.build_manifest).unwrap()).unwrap(); assert_ne!(fields["machine.a.worker"], first_worker); assert_eq!(fields["supports_output"], "true");
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" stop") || e.label.ends_with(" start")));
    assert_eq!(f.app.runner.calls.iter().filter(|e| e.label.ends_with(" update")).count(), 3);
    let copies = f.app.runner.calls.iter().filter(|e| e.label.ends_with("source copy")).collect::<Vec<_>>(); assert_eq!(copies.len(), 4); assert!(copies.iter().all(|e| e.args.contains(&"--from0".into()) && e.args.contains(&"--files-from=-".into()) && e.stdin == b"Cargo.toml\0src/main.rs\0src/flash-serve.rs\0" && e.args.last().unwrap().ends_with(":./") && e.args.iter().any(|a| a.starts_with("--rsync-path=umask 077 && cd ") && a.contains("/revisions/") && a.ends_with(" && exec rsync"))));
    assert!(copies.iter().all(|e| e.args.contains(&"--chmod=go-rwx".into())));
    let models = f.app.runner.calls.iter().filter(|e| e.label.ends_with("complete model check")).collect::<Vec<_>>(); assert_eq!(models.len(), 2); assert!(models.iter().all(|e| { let q = text::parse(&String::from_utf8_lossy(&e.stdin)).unwrap(); q["op"] == "model" && !q.contains_key("token") && !q.contains_key("state_dir") }));
    for exec in f.app.runner.calls.iter().filter(|e| matches!(e.label.as_str(), "harness build" | "controller CLI build")) { assert!(exec.args.contains(&"--target-dir".into()) && exec.unset.contains(&"CARGO_TARGET_DIR".into())); }
    assert!(f.app.runner.calls.iter().filter(|e| e.label.ends_with("build and stage")).all(|e| e.args.last().unwrap().contains("export CARGO_TARGET_DIR=\"$engine_cache\"")));
}

#[test]
fn update_restart_hints_parse_back_to_the_running_or_configured_profile() {
    for (mut f, options) in [(Fixture::new(2, false), legacy_options()), (Fixture::fp4(1, false), fp4_options(&["--mtp", "0", "--context", "8192", "--output", "4096"]))] {
        f.app.up(&options).unwrap();
        let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        let capture = crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(), || f.app.update().unwrap());
        let output = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        let command = output.lines().find_map(|line| line.trim().strip_prefix("vade up ")).unwrap();
        assert!(command.contains(if session.fp4_local { "--fp4-local" } else { "--fp8-legacy" }));
        let args = std::iter::once("up").chain(command.split_whitespace()).map(str::to_string).collect::<Vec<_>>();
        let Command::Up(restart) = crate::cli::parse(&args).unwrap().command else { panic!() };
        assert_eq!((restart.fp4_local, restart.context, restart.output, restart.mtp, restart.mtp_explicit),
            (session.fp4_local, session.context, session.output, session.mtp, session.mtp_explicit));
        assert_eq!(restart.hosts, session.ranks.iter().map(|rank| rank.machine.name.clone()).collect::<Vec<_>>());
        assert_eq!((restart.chats, restart.memory_bytes, restart.cache_bytes), (session.chats, session.memory_bytes, session.cache_bytes));
        assert!(f.recorded(), "declining the restart must preserve the running engine");
        f.app.down().unwrap();
        let capture = crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(), || f.app.update().unwrap());
        let output = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        let command = output.lines().find_map(|line| line.strip_prefix("Start the new build: vade ")).unwrap();
        let args = command.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        let Command::Up(start) = crate::cli::parse(&args).unwrap().command else { panic!() };
        assert_eq!(start.fp4_local, session.fp4_local);
        f.app.config.selected(&start).unwrap();
    }
}

#[test]
fn every_tunnel_operation_uses_the_same_private_directory_and_short_name() {
    let f = Fixture::new(2, false);
    let manifest = Manifest::load(&f.app.config.build_manifest).unwrap();
    let session = Session::new(&f.app.config, &legacy_options(), &manifest, "a".repeat(64)).unwrap();
    for operation in [None, Some("check"), Some("exit")] {
        let exec = f.app.tunnel_at(&session, 0, operation);
        assert_eq!(exec.cwd.as_ref(), Some(&f.app.config.state_dir));
        let path = exec.args.windows(2).find(|pair| pair[0] == "-S").unwrap()[1].clone();
        assert_eq!(Path::new(&path).file_name(), session.socket(&f.app.config.state_dir).file_name());
        assert!(Path::new(&path).is_relative() && path.len() + 18 <= 104);
    }
}

#[test]
fn fp4_transport_selection_controls_both_pipeline_ranks_and_prerequisites() {
    const CASE: &str = "VADE_TEST_FP4_TRANSPORT";
    if let Ok(expected) = std::env::var(CASE) {
        let mut fixture = Fixture::fp4(2, false);
        fixture.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let starts = fixture.app.runner.calls.iter().filter(|c| c.label.ends_with(" start") && c.label != "controller tunnel start").collect::<Vec<_>>();
        assert_eq!(starts.len(), 2);
        for start in starts {
            let request = text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap();
            assert_eq!(request["env.SEQPAR_RDMA"], expected);
            assert_eq!(request["env.TP_RDMA"], expected);
            assert_eq!(request["model_receipt"], "pipeline");
        }
        for probe in fixture.app.runner.calls.iter().filter(|c| c.label.ends_with(" probe")) {
            assert_eq!(text::parse(std::str::from_utf8(&probe.stdin).unwrap()).unwrap()["rdma"], expected);
        }
        return;
    }
    // Two Sparks over TCP are refused (tests/two_spark.rs); chains cover TCP.
    for value in [None, Some("1")] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::fp4_transport_selection_controls_both_pipeline_ranks_and_prerequisites", "--nocapture"])
            .env(CASE, value.unwrap_or("1"));
        if let Some(value) = value { command.env("SEQPAR_RDMA", value); }
        else { command.env_remove("SEQPAR_RDMA"); }
        let result = command.output().unwrap();
        assert!(result.status.success(), "{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
    }
}

#[test]
fn json_status_reports_observed_health_and_keeps_offline_checks_unknown() {
    use crate::{evals::json::{self, Value}, status::Snapshot};
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
    let path = f.app.config.state_dir.join("session.conf");
    let before = fs::read(&path).unwrap();
    for case in ["ready", "unhealthy", "offline"] {
        f.app.dry = case == "offline";
        f.app.runner.foreign = (case == "unhealthy").then(|| "a".into());
        f.app.runner.tunnel_bad = case == "unhealthy";
        f.app.runner.calls.clear();
        let mut report = Snapshot::new("configured", f.app.dry);
        f.app.status_snapshot(&mut report).unwrap();
        let (encoded, code) = report.finish(Duration::ZERO);
        let value = json::parse(&encoded).unwrap();
        assert_eq!(value["schema"].as_str(), Some("vade.status.v1"));
        assert_eq!(value["settings"]["mtp"], Value::Null);
        assert_eq!(value["settings"]["mtp_source"].as_str(), Some("engine_default"));
        assert_eq!((&value["settings"]["context_limit"], &value["settings"]["admitted_chats"]), (&Value::from(32768usize), &Value::from(8usize)));
        assert_eq!(value["ranks"].as_array().unwrap().len(), 2);
        assert_eq!(code, i32::from(case == "unhealthy"));
        assert_eq!(value["ready"], if case == "offline" { Value::Null } else { Value::Bool(case == "ready") });
        if case == "offline" {
            assert!(f.app.runner.calls.is_empty(), "offline status must perform no probe");
            assert!(value["checks"].as_array().unwrap().iter().all(|check| check["ok"] == Value::Null));
        } else {
            assert_eq!(value["ranks"][1]["alive"], Value::Bool(true));
            assert!(f.app.runner.calls.iter().all(|call| !matches!(call.label.as_str(), "a stop" | "b stop" | "controller tunnel start")));
        }
        if case == "unhealthy" { assert_eq!(value["ranks"][0]["alive"], Value::Null); assert!(!value["errors"].as_array().unwrap().is_empty()); }
        assert!(!encoded.contains(&"0".repeat(64)), "ownership tokens are not status fields");
        assert_eq!(fs::read(&path).unwrap(), before, "status must preserve ownership evidence");
    }
}

#[test]
fn status_speed_aliases_reject_negative_and_unknown_values() {
    use crate::{evals::json::{self, Value}, status::Snapshot};
    let mut f = Fixture::fp4(1, false);
    f.app.up(&fp4_options(&[])).unwrap();
    for (extra, expected) in [
        (",\"tok_s\":12.5", Some(12.5)),
        (",\"tokens_per_second\":23.5", Some(23.5)),
        (",\"tokens_per_sec\":34.5", Some(34.5)),
        (",\"tok_s\":0", Some(0.0)),
        (",\"tok_s\":-1", None),
        (",\"tokens_per_second\":-1", None),
        (",\"tokens_per_sec\":-1", None),
        (",\"tok_s\":null", None),
        (",\"tokens_per_second\":\"unknown\"", None),
        ("", None),
    ] {
        f.app.runner.health_extra = extra.into();
        let mut report = Snapshot::new("configured", false);
        f.app.status_snapshot(&mut report).unwrap();
        let (encoded, code) = report.finish(Duration::ZERO);
        let value = json::parse(&encoded).unwrap();
        assert_eq!(code, 0);
        assert_eq!(value["tokens_per_second"], expected.map(Value::Float).unwrap_or(Value::Null), "{encoded}");
    }
}

#[test]
fn output_errors_before_launch_and_after_start_preserve_owned_cleanup() {
    use crate::output::{self, testing::Capture};
    use std::io::ErrorKind;
    for marker in ["Launch plan:", "ready on"] {
        for kind in [ErrorKind::BrokenPipe, ErrorKind::StorageFull] {
            let mut f = Fixture::fp4(2, false);
            output::with_writer(Capture { fail_on: Some((marker, kind)), ..Capture::default() }, || {
                assert!(f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err().contains("standard output"));
                assert_eq!(output::failure(), Some(kind));
            });
            assert!(f.app.runner.started.is_empty());
            assert!(!f.recorded());
            if marker == "Launch plan:" { assert!(f.app.runner.calls.is_empty()); }
            else { assert_eq!(f.app.runner.calls.iter().filter(|e| e.label.ends_with(" stop")).count(), 2); }
        }
    }
}

#[test]
fn cleanup_failure_after_closed_stdout_keeps_record_and_failing_exit() {
    use crate::output::{self, testing::Capture};
    let mut f = Fixture::fp4(1, false);
    f.app.runner.fail = Some("a stop".into());
    output::with_writer(Capture { fail_on: Some(("ready on", std::io::ErrorKind::BrokenPipe)), ..Capture::default() }, || {
        let error = f.app.up(&fp4_options(&[])).unwrap_err();
        assert!(error.contains("Cleanup could not finish"));
        assert!(f.recorded());
        assert!(output::failure().is_none());
        assert_eq!(output::exit_code(1), 1);
    });
}

#[test]
fn prerequisite_model_probe_accepts_indexed_nested_shards_without_claiming_verification() {
    let _lock = runner::process_test_lock();
    let mut f = Fixture::fp4(1, false);
    let model = f.dir.join("indexed-model");
    fs::create_dir_all(model.join("tensors")).unwrap();
    fs::write(model.join("config.json"), "{}").unwrap();
    fs::write(model.join("tokenizer.json"), "{}").unwrap();
    fs::write(model.join("model.safetensors.index.json"), r#"{"weight_map":{"w":"tensors/a.safetensors"}}"#).unwrap();
    fs::write(model.join("tensors/a.safetensors"), "placeholder").unwrap();
    let mut machine = f.app.config.machines[0].clone();
    machine.model_path = model.to_string_lossy().into_owned();
    f.app.machine_checks(&machine).unwrap();
    let command = f.app.runner.calls.last().unwrap().args.last().unwrap();
    let probe = &command[command.find("if test -d ").unwrap()..];
    let run = || {
        let output = std::process::Command::new("sh").args(["-c", probe]).output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    };
    assert!(run().contains("weights=present"), "indexed nested checkpoint was called missing");
    fs::remove_file(model.join("tokenizer.json")).unwrap();
    assert!(run().contains("weights=missing"));
    fs::write(model.join("tokenizer.json"), "{}").unwrap();
    fs::remove_file(model.join("model.safetensors.index.json")).unwrap();
    assert!(run().contains("weights=missing"), "unindexed nested shards are incomplete");
    fs::write(model.join("model.safetensors"), "placeholder").unwrap();
    assert!(run().contains("weights=present"), "single-file checkpoint regressed");
}

#[test]
fn startup_health_probes_are_transient_but_later_diagnostics_are_retained() {
    let mut f = Fixture::fp4(1, false);
    f.app.up(&fp4_options(&[])).unwrap();
    let probes: Vec<_> = f.app.runner.calls.iter().filter(|e| e.label == "engine health" || e.label == "controller health").collect();
    assert!(!probes.is_empty());
    assert!(probes.iter().all(|e| e.transient_probe && e.probe_log.is_some()));
    assert!(!f.app.remote_health_at(&Session::load(&f.app.config.state_dir).unwrap().unwrap(), 0).transient_probe);
}


// Regression failure-path proof. Existing FakeRunner performs every operation.
#[test]
fn observed_launcher_failure_exit_probe() {
    let Ok(case) = std::env::var("VADE_TEST_LAUNCHER_CASE") else { return; };
    let mut f = Fixture::fp4(1, false);
    let mut session = Session::new(&f.app.config, &fp4_options(&[]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    session.phase = "running".into();
    session.ranks[0].attempted = true;
    session.save(&f.app.config.state_dir).unwrap();
    f.app.runner.started.insert("a".into());
    let doctor = case.starts_with("doctor");
    f.app.runner.doctor_bad = doctor;
    f.app.runner.tunnel_bad = !doctor;
    let marker = if doctor { "Run this for" } else { "Controller connection:" };
    crate::output::with_writer(crate::output::testing::Capture {
        fail_on: case.ends_with("closed").then_some((marker, std::io::ErrorKind::BrokenPipe)),
        ..Default::default()
    }, || {
        let error = if doctor { f.app.harness_doctor(false) } else { f.app.status() }.unwrap_err();
        drop(f);
        crate::fail(&error);
    });
}
#[test]
fn observed_failed_launcher_survives_closed_stdout() {
    for case in ["doctor_open", "doctor_closed", "status_open", "status_closed"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "app::tests::observed_launcher_failure_exit_probe", "--nocapture"])
            .env("VADE_TEST_LAUNCHER_CASE", case).output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{case}: observed doctor/status failures must stay failures when stdout closes; stderr={}", String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
fn healthy_up_attaches_without_restarting_and_changed_settings_refuse() {
    let mut f = Fixture::fp4(1, false);
    let options = fp4_options(&[]);
    f.app.up(&options).unwrap();
    let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    f.app.runner.calls.clear();
    f.app.up(&options).unwrap();
    f.app.up(&UpOptions::default()).unwrap();
    assert_eq!(fs::read(f.app.config.state_dir.join("session.conf")).unwrap(), saved);
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" start") || e.label.ends_with(" stop") || e.label.contains("GPU")));
    assert!(f.app.up(&fp4_options(&["--context", "8192"])).unwrap_err().contains("different launch"));
    f.app.runner.tunnel_bad = true;
    assert!(f.app.up(&options).unwrap_err().contains("unhealthy"));
    assert!(f.app.runner.started.contains("a"));
}

#[test]
fn prebuilt_install_verifies_before_execution_and_reuses_warm_artifacts() {
    let mut f = Fixture::fp4(1, false);
    let root = f.dir.join("prebuilt");
    let dir = root.join(format!("{}-{}", "a".repeat(40), "b".repeat(40)));
    fs::create_dir_all(&dir).unwrap();
    let mut manifest = format!("format=vade-prebuilt-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={}\ncli_sha={}\n", "a".repeat(40), "b".repeat(40));
    for name in crate::prebuilt::FILES {
        let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183;
        fs::write(dir.join(name), &bytes).unwrap();
        manifest.push_str(&format!("{name}.sha256={}\n", crate::sha256::sha(&bytes)));
    }
    fs::write(dir.join("manifest.conf"), manifest).unwrap();
    f.app.config.prebuilt_dir = Some(root);
    f.app.execute(Command::Install).unwrap();
    let calls = &f.app.runner.calls;
    assert!(!calls.iter().any(|e| e.label.ends_with("source copy") || e.label.ends_with("helper build") || e.label.ends_with("build and stage")));
    let verified = calls.iter().position(|e| e.label == "a prebuilt helper hash").unwrap();
    let executed = calls.iter().position(|e| e.label == "a complete model check").unwrap();
    assert!(verified < executed);
    let prereqs = calls.iter().find(|e| e.label == "a prerequisites").unwrap().args.last().unwrap();
    assert!(!prereqs.contains("cargo --version") && !prereqs.contains("rustc --version") && !prereqs.contains("for tool in cargo"));
    f.app.runner.calls.clear();
    f.app.execute(Command::Install).unwrap();
    assert!(f.app.runner.calls.iter().any(|e| e.label == "a existing prebuilt hashes"));
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with("prebuilt copy")));
    // A bad remote helper must never be executed and never change activation.
    let before = fs::read(&f.app.config.activation).unwrap();
    f.app.runner.calls.clear(); f.app.runner.fail = Some("a existing prebuilt hashes".into());
    assert!(f.app.execute(Command::Install).is_err());
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with("complete model check")));
    assert_eq!(fs::read(&f.app.config.activation).unwrap(), before);
}

#[path = "tests/resident.rs"]
mod resident_tests;

#[test]
fn upgrading_an_unpinned_config_preserves_existing_session_inspection_and_shutdown() {
    for revision in ["", "main"] {
        let mut f = Fixture::fp4(1, false);
        f.app.up(&fp4_options(&[])).unwrap();
        let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
        let state = fs::canonicalize(&f.app.config.state_dir).unwrap();
        let stage = fs::canonicalize(&f.app.config.stage_root).unwrap();
        let revision_line = if revision.is_empty() { String::new() } else { format!("model_revision = {revision}\n") };
        let old_config = format!("[global]\nstate_dir = {}\nstage_dir = {}\nmodel = qwen3.8-flash-next-fp4-local\n{revision_line}[machine a]\nssh = a-ssh\n", state.display(), stage.display());
        f.app.config = Config::parse(&old_config, &f.dir.join("config")).unwrap();
        f.app.runner.calls.clear();
        assert!(f.app.up(&fp4_options(&[])).unwrap_err().contains("immutable"));
        assert!(f.app.install().unwrap_err().contains("immutable"));
        assert!(f.app.runner.calls.is_empty(), "unpinned starts/installations must stop before remote actions");
        f.app.execute(Command::Status).unwrap();
        f.app.execute(Command::Logs).unwrap();
        assert_eq!(fs::read(state.join("session.conf")).unwrap(), saved);
        f.app.runner.calls.clear();
        f.app.execute(Command::Down).unwrap();
        inspections_precede_stops(&f.app.runner.calls, 1);
        assert!(!f.recorded());
    }
}

#[test]
fn one_spark_prerequisites_use_the_launch_plan_transport_wording() {
    assert_eq!(transport_name(1, true), "local (one Spark; no fabric)");
    assert_eq!((transport_name(2, true), transport_name(2, false)), ("RDMA", "TCP"));
    let mut f = Fixture::fp4(1, false);
    let machine = f.app.config.machines[0].clone();
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    assert!(crate::output::with_writer(capture, || f.app.machine_checks_for_build(&machine, false)).unwrap());
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    assert!(printed.contains("transport: local (one Spark; no fabric); model files"), "{printed}");
    assert!(!printed.contains("TCP"), "{printed}");
}

#[test]
fn status_prints_the_engine_retention_state() {
    let mut f = Fixture::fp4(1, false);
    f.app.up(&fp4_options(&[])).unwrap();
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    assert!(printed.contains("\n  Retained prompt cache  off (engine report)\n  Event counts: prefix_cache_miss 2, scheduler_pool 1\n  Last 1 events (oldest first):\n    00:00:00Z ("), "{printed}");
    assert!(printed.contains(") scheduler_pool active=8 connections=16\n"), "{printed}");
    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(retention_label(&session, Some(true)), "on (engine report)");
    assert_eq!(retention_label(&session, None), "off by default; engine not checked");
    session.cache_bytes = Some(0);
    assert_eq!(retention_label(&session, None), "off (--cache-bytes 0)");
    session.fp4_local = false;
    session.cache_bytes = None;
    assert_eq!(retention_label(&session, None), "automatic");
    session.cache_bytes = Some(4096);
    assert_eq!(retention_label(&session, None), "4096 bytes/rank");
}

/// A user's ~/.curlrc (for example `--include` or `--output`) must not change any reply
/// the controller parses: every curl call, local or on a Spark, starts with -q.
#[test]
fn every_controller_curl_ignores_the_user_curl_config() {
    for replicas in [false, true] {
        let mut f = Fixture::fp4(2, false);
        f.app.up(&if replicas { fp4_options(&["--replicas", "2"]) } else { fp4_options(&[]) }).unwrap();
        crate::output::with_writer(crate::output::testing::Capture::default(), || f.app.execute(Command::Status)).unwrap();
        f.app.down().unwrap();
        let local = f.app.runner.calls.iter().filter(|e| e.program == "curl").collect::<Vec<_>>();
        let remote = f.app.runner.calls.iter().filter(|e| e.program == "ssh" && e.args.last().is_some_and(|a| a.contains("curl "))).collect::<Vec<_>>();
        assert!(local.iter().any(|e| e.label == "controller health") && local.iter().any(|e| e.label == "controller capabilities")
            && local.iter().any(|e| e.label == "controller events") && !remote.is_empty(), "replicas={replicas}");
        for exec in local { assert_eq!(exec.args.first().map(String::as_str), Some("-q"), "{}", exec.label); }
        for exec in remote { assert!(exec.args.last().unwrap().starts_with("curl -q "), "{}", exec.label); }
    }
}

#[test]
fn silent_pre_start_probe_claims_no_ownership_record() {
    let mut f = Fixture::fp4(1, false);
    f.app.runner.silent_probe = true;
    let error = f.app.up(&fp4_options(&[])).unwrap_err();
    assert!(error.starts_with("Transport: a did not reply to the pre-start check; nothing was started or recorded."), "{error}");
    assert!(Session::load(&f.app.config.state_dir).unwrap().is_none());
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" start")));
    f.app.runner.silent_probe = false;
    f.app.up(&fp4_options(&[])).unwrap();
}

#[test]
fn replicas_start_world_one_servers_with_own_tunnels_behind_an_owned_gateway() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    let s = Session::load(&f.app.config.state_dir).unwrap().unwrap(); let (port, serve) = (s.local_port, s.serve_port);
    assert!(s.replicas && s.gateway.as_ref().is_some_and(|g| g.0 == 4242 && g.1 == "Sat Oct 24 9:05:01 2026") && (s.context, s.engine_args(1)[0..4].join(" ").as_str()) == (32768, "0 1 127.0.0.1:0 127.0.0.1:0"));
    let starts: Vec<_> = f.app.runner.calls.iter().map(|e| text::parse(&String::from_utf8_lossy(&e.stdin)).unwrap_or_default()).filter(|r| r.get("op").is_some_and(|op| op == "start")).collect();
    assert!(starts.len() == 2 && starts.iter().all(|r| r["env.FLASH_SERVE_RANK"] == "0" && r["env.FLASH_SERVE_WORLD_SIZE"] == "1" && r["env.SEQPAR_RDMA"] == "0"));
    let tunnels: Vec<_> = f.app.runner.calls.iter().filter(|e| e.label == "controller tunnel start").map(|e| e.args[e.args.len() - 2..].join(" ")).collect();
    assert_eq!(tunnels, [format!("127.0.0.1:{}:127.0.0.1:{serve} a-ssh", port + 1), format!("127.0.0.1:{}:127.0.0.1:{serve} b-ssh", port + 2)]);
    assert_eq!(f.app.runner.gateway.clone().unwrap(), format!("vade-gateway --bind 127.0.0.1:{port} --upstream http://127.0.0.1:{} --upstream http://127.0.0.1:{}", port + 1, port + 2));
    assert_eq!(super::chats_label(&s), "8 per replica, 16 across 2 replicas (engine-admitted at this context)");
    let gateway = f.app.runner.calls.iter().find(|e| e.label == "replica gateway").unwrap();
    for key in ["VADE_GATEWAY_TOKEN", "VADE_GATEWAY_UPSTREAM", "VADE_API_KEY_FILE"] {
        assert!(gateway.unset.iter().any(|k| k == key) && !gateway.env.contains_key(key), "{key} never reaches the gateway");
    }
    f.app.status().unwrap(); f.app.down().unwrap();
    assert!(f.app.runner.gateway.is_none() && !f.recorded(), "down stops the identified gateway and clears the record");
    assert_eq!(f.app.runner.signals, ["-TERM"], "a gateway that stops on TERM never gets KILL");
}

#[test]
fn replica_status_reads_and_labels_each_server_events() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    let port = Session::load(&f.app.config.state_dir).unwrap().unwrap().local_port;
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    // Each replica is read at its own tunnel, not through the gateway, which picks one per request.
    assert!(printed.contains(&format!("\n  a:\n    Event counts: replica_{} 1\n    Last events: none\n  b:\n    Event counts: replica_{} 1\n", port + 1, port + 2)), "{printed}");
    f.app.down().unwrap();
}

#[test]
fn replicas_down_escalates_to_kill_only_when_the_gateway_ignores_term() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    f.app.runner.gateway_stubborn = true;
    f.app.down().unwrap();
    assert_eq!(f.app.runner.signals, ["-TERM", "-KILL"]);
    assert!(f.app.runner.gateway.is_none() && !f.recorded());
}

#[test]
fn replicas_down_never_signals_a_reused_gateway_pid() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    // The recorded PID now belongs to a process that started later (same command line).
    f.app.runner.gateway_start = Some("Sun Oct 25 10:00:00 2026".into());
    f.app.down().unwrap();
    assert!(f.app.runner.signals.is_empty() && f.app.runner.gateway.is_some(), "a reused PID is never signaled");
    assert!(f.app.runner.started.is_empty() && !f.recorded(), "down still stops the replicas and clears the record");
    // A reused PID running another program is never signaled either.
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    f.app.runner.gateway = Some("--other-tool".into());
    f.app.down().unwrap();
    assert!(f.app.runner.signals.is_empty() && !f.recorded());
}

#[test]
fn replicas_gateway_that_exits_at_startup_is_reported_and_never_signaled() {
    // An exited gateway stays a zombie while up runs: ps answers with `<defunct>`.
    let mut f = Fixture::fp4(2, false);
    f.app.runner.gateway_exits = true;
    let error = f.app.up(&fp4_options(&["--replicas", "2"])).unwrap_err();
    assert!(error.contains("The replica gateway exited at startup") && error.contains("-gateway.log"), "{error}");
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "controller health"), "up fails at once, not after the tunnel wait");
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "replica gateway stop"), "a defunct gateway is never signaled");
    assert!(f.app.runner.started.is_empty() && !f.recorded(), "the failed launch still stops both replicas and clears the record");
}

#[test]
fn replicas_gateway_identity_does_not_depend_on_the_locale_date() {
    // A Japanese-locale lstart has four fields; it must neither shift the command nor hide the gateway.
    let mut f = Fixture::fp4(2, false);
    f.app.runner.gateway_start = Some("\u{571f} 10/24 16:02:34 2026".into());
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    let s = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(s.gateway, Some((4242, "\u{571f} 10/24 16:02:34 2026".into())));
    assert_eq!(f.app.gateway_alive(&s).unwrap(), Some(true));
    f.app.down().unwrap();
    assert!(f.app.runner.gateway.is_none() && !f.recorded(), "down still finds and stops the gateway");
}

/// Doctor rows for checks an earlier failure stopped are `skip`, never a
/// claim about the files or tools the script did not reach.
#[test]
fn doctor_skips_checks_an_earlier_failure_hid() {
    let rows = |probe: Option<(bool, &'static str)>, machines: usize, transport: Result<bool, String>| {
        let mut f = Fixture::fp4(machines, false);
        if machines > 1 {
            let session = Session::new(&f.app.config, &fp4_options(&["--sparks", &machines.to_string()]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
            session.save(&f.app.config.state_dir).unwrap();
        }
        f.app.runner.probe = probe;
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || f.app.doctor_with(transport));
        (result, String::from_utf8(capture.bytes.borrow().clone()).unwrap())
    };
    let row = |printed: &str, name: &str| printed.lines().find(|line| line[6..].starts_with(name)).unwrap_or_else(|| panic!("{name}:\n{printed}")).to_owned();
    let (result, printed) = rows(Some((false, "gpu=NVIDIA GB10, 580.95\ndisk_kb=10\nmem_total_kb=127608480\nmem_available_kb=115343360\nmissing=disk\n")), 1, Ok(false));
    assert!(result.unwrap_err().starts_with("1 doctor check(s) failed."), "{printed}");
    assert!(row(&printed, "a disk").starts_with("FAIL"), "{printed}");
    assert_eq!(row(&printed, "a model files"), "skip  a model files         not checked; the script stopped at disk");
    let (_, printed) = rows(Some((false, "missing=cargo\n")), 1, Ok(false));
    assert_eq!(row(&printed, "a nvidia-smi"), "skip  a nvidia-smi          not checked; the script stopped at cargo");
    assert!(row(&printed, "a tools").starts_with("FAIL"), "{printed}");
    assert!(row(&printed, "a model files").starts_with("skip"), "{printed}");
    let (_, printed) = rows(Some((false, "missing=rsync\n")), 1, Ok(false));
    assert!(row(&printed, "a nvidia-smi").starts_with("ok"), "{printed}");
    let (_, printed) = rows(Some((false, "gpu=GB10\ndisk_kb=99999999\nmissing=rdma-device\n")), 2, Ok(true));
    assert!(row(&printed, "a rdma").starts_with("FAIL") && row(&printed, "a disk").starts_with("ok"), "{printed}");
    assert!(row(&printed, "a model files").starts_with("skip"), "{printed}");
    // SSH that never ran the script, a failed driver query, a failing tool
    // and an unnamed stop are four different rows.
    let (_, printed) = rows(Some((false, "")), 1, Ok(false));
    assert!(row(&printed, "a ssh").starts_with("FAIL") && row(&printed, "a nvidia-smi").starts_with("skip"), "{printed}");
    let (_, printed) = rows(Some((false, "reached=1\nfailed=driver\n")), 1, Ok(false));
    assert!(row(&printed, "a ssh").starts_with("ok") && row(&printed, "a nvidia-smi").contains("GPU query failed"), "{printed}");
    assert!(printed.contains("next: Run ssh a-ssh nvidia-smi to see the driver error"), "{printed}");
    let (_, printed) = rows(Some((false, "reached=1\nfailed=cargo\n")), 1, Ok(false));
    assert!(row(&printed, "a ssh").starts_with("ok") && row(&printed, "a tools").contains("cargo --version failed"), "{printed}");
    let (_, printed) = rows(Some((false, "reached=1\n")), 1, Ok(false));
    assert!(row(&printed, "a ssh").starts_with("ok") && row(&printed, "a check script").starts_with("FAIL"), "{printed}");
    assert!(row(&printed, "a nvidia-smi").starts_with("skip"), "{printed}");
    // A missing pinned toolchain keeps every earlier row, then fails its own row.
    let found = "reached=1\ngpu=GB10\ndisk_kb=99999999\nmem_total_kb=127608480\nmem_available_kb=115343360\nweights=present\n";
    let (result, printed) = rows(Some((false, Box::leak(format!("{found}failed=kernel-toolchain\n").into_boxed_str()))), 1, Ok(false));
    assert!(result.is_err() && row(&printed, "a nvidia-smi").contains("GB10") && row(&printed, "a disk").starts_with("ok"), "{printed}");
    assert!(row(&printed, "a model files").starts_with("ok") && row(&printed, "a toolchains").starts_with("FAIL"), "{printed}");
    assert!(printed.contains("rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda"), "{printed}");
    let (_, printed) = rows(Some((false, Box::leak(format!("{found}failed=toolchain\n").into_boxed_str()))), 1, Ok(false));
    assert!(row(&printed, "a toolchains").contains("no Rust 1.99.0 toolchain"), "{printed}");
    let (_, printed) = rows(Some((true, Box::leak(format!("{found}toolchains=ok\nrust=ok\ndriver=ok\n").into_boxed_str()))), 1, Ok(false));
    assert!(row(&printed, "a toolchains").starts_with("ok"), "{printed}");
    // GPU use follows the rule vade up applies, so doctor names what up would refuse.
    let gpu = |sample: &str, machines: usize| {
        let (result, printed) = rows(Some((true, Box::leak(format!("{found}{sample}rust=ok\ndriver=ok\n").into_boxed_str()))), machines, Ok(true));
        (result.is_ok(), row(&printed, "a GPU use"))
    };
    assert_eq!(gpu("gpu_apps=0\ngpu_watts=18.25\n", 1), (true, "ok    a GPU use             no compute processes; 18.2 W".into()));
    let (passed, line) = gpu("gpu_apps=2\ngpu_watts=41.5\n", 1);
    assert!(!passed && line.starts_with("FAIL") && line.contains("2 compute process(es) use the GPU."), "{line}");
    let (passed, line) = gpu("gpu_apps=0\ngpu_watts=30\n", 1);
    assert!(!passed && line.contains("draws 30.0 W with no compute process listed"), "{line}");
    let (passed, line) = gpu("gpu_apps=0\ngpu_watts=\n", 1);
    assert!(!passed && line.contains("did not report the GPU power draw"), "{line}");
    assert!(gpu("", 1).1.starts_with("skip"));
    // A recorded engine holds its own GPU.
    assert!(gpu("gpu_apps=1\ngpu_watts=60\n", 2).1.contains("a recorded engine can hold the GPU"));
    // The complete script still reports the files it found.
    let (result, printed) = rows(None, 1, Ok(false));
    result.unwrap();
    assert!(row(&printed, "a model files").starts_with("ok"), "{printed}");
    // An invalid SEQPAR_RDMA is one FAIL row; the report still finishes.
    let (result, printed) = rows(None, 2, Err("Set SEQPAR_RDMA to 0 or 1.".into()));
    assert!(result.is_err());
    assert!(row(&printed, "SEQPAR_RDMA").starts_with("FAIL") && row(&printed, "b ssh").starts_with("skip") && row(&printed, "fabric").starts_with("skip"), "{printed}");
    assert!(printed.lines().last().unwrap().starts_with("FAIL  "), "{printed}");
}

#[test]
fn doctor_requires_configured_wire_links_before_reporting_reachability() {
    let mut f = Fixture::new(2, false);
    for machine in &mut f.app.config.machines { machine.fabric.clear(); }
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true)));
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(result.is_err());
    assert!(printed.contains("FAIL  launch topology") && printed.contains("Add fabric.b under a and fabric.a under b"), "{printed}");
    assert!(!f.app.runner.calls.iter().any(|exec| exec.label.ends_with(" fabric")));
    assert!(f.app.runner.started.is_empty());
}

#[test]
fn doctor_uses_default_one_spark_selection_and_checks_the_full_memory_floor() {
    let mut f = Fixture::fp4(2, false);
    f.app.runner.probe = Some((true, "weights=present\nrust=ok\ndriver=ok\ndisk_kb=99999999\nmem_total_kb=125829120\nmem_available_kb=4194304\n"));
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true)));
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(result.is_err() && printed.contains("FAIL  a model memory") && printed.contains("Not enough memory"), "{printed}");
    assert!(printed.contains("one Spark; no fabric") && !printed.contains("b model memory"), "{printed}");
    assert!(!f.app.runner.calls.iter().any(|call| call.label.starts_with("b ") || call.label.ends_with(" fabric")));
    let probe = f.app.runner.calls.iter().find(|call| call.label == "a prerequisites").unwrap().args.last().unwrap();
    assert!(probe.contains("if test 0 = 1; then"), "{probe}");
    std::mem::forget(f);
}

#[test]
fn doctor_accepts_replica_configs_without_fabric_and_uses_the_saved_ranks() {
    let mut f = Fixture::fp4(3, true);
    for machine in &mut f.app.config.machines { machine.fabric.clear(); }
    let session = Session::new(&f.app.config, &fp4_options(&["--replicas", "2"]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    session.save(&f.app.config.state_dir).unwrap();
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.doctor_with(Err("Set SEQPAR_RDMA to 0 or 1.".into()))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("independent replicas; no fabric") && !printed.contains("FAIL"), "{printed}");
    assert!(printed.contains("a port 18080") && printed.contains("b port 18080"), "{printed}");
    assert!(!printed.contains("port 29960") && !printed.contains("c model files"), "{printed}");
    std::mem::forget(f);
}

#[test]
fn doctor_checks_http_only_on_the_serving_rank_of_an_active_chain() {
    for count in 2..=3 {
        let mut f = Fixture::fp4(count, true);
        let session = Session::new(&f.app.config, &fp4_options(&["--sparks", &count.to_string()]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
        session.save(&f.app.config.state_dir).unwrap();
        for machine in &mut f.app.config.machines { machine.fabric.clear(); }
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true)));
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert!(result.is_ok(), "{printed}");
        assert!(printed.contains("a port 18080") && !printed.contains("a port 29960"), "{printed}");
        for machine in session.ranks.iter().skip(1) {
            assert!(!printed.contains(&format!("{} port 18080", machine.machine.name)), "{printed}");
            assert!(printed.contains(&format!("{} port 29960", machine.machine.name)), "{printed}");
        }
        assert!(!printed.contains("port 29961"), "{printed}");
        assert_eq!(f.app.runner.calls.iter().filter(|call| call.label.ends_with(" fabric")).count(), count - 1);
        for link in session.ranks.windows(2) {
            let label = format!("{} to {} fabric", link[0].machine.name, link[1].machine.name);
            let call = f.app.runner.calls.iter().find(|call| call.label == label).unwrap();
            assert!(call.args.last().unwrap().contains(link[0].next.rsplit_once(':').unwrap().0));
        }
        std::mem::forget(f);
    }
}

#[test]
fn doctor_keeps_legacy_controller_limits() {
    let mut f = Fixture::new(2, true);
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("131072 tokens") && printed.contains("65536 tokens"), "{printed}");
}

/// A failed harness doctor names the command to rerun in its own error,
/// since doctor's report keeps only the error.
#[test]
fn failed_harness_doctor_error_names_its_retry_command() {
    let mut f = Fixture::fp4(1, false);
    let mut session = Session::new(&f.app.config, &fp4_options(&[]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    session.phase = "running".into();
    session.ranks[0].attempted = true;
    session.save(&f.app.config.state_dir).unwrap();
    f.app.runner.started.insert("a".into());
    f.app.runner.doctor_bad = true;
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || f.app.doctor());
    assert!(result.is_err());
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    let next = printed.lines().skip_while(|line| !line.contains("engine health")).nth(1).unwrap().to_owned();
    let harness = crate::runner::quote(&f.app.config.stage_dir.join("vade-harness").to_string_lossy());
    assert!(next.starts_with("      next: ") && next.contains(&format!("{harness} doctor http://")) && next.contains("--max-seconds 120"), "{printed}");
}

/// The real check script marks a failed driver query or tool version, so
/// doctor never mistakes either for an SSH failure.
#[test]
fn prerequisite_script_marks_failed_driver_queries_and_tool_versions() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = runner::process_test_lock();
    let mut f = Fixture::fp4(1, false);
    let machine = f.app.config.machines[0].clone();
    f.app.machine_checks_for_build(&machine, true).unwrap();
    let script = f.app.runner.calls.last().unwrap().args.last().unwrap().clone();
    let script = &script[script.find("set -eu").unwrap()..];
    let bin = f.dir.join("stub-bin");
    fs::create_dir_all(&bin).unwrap();
    let run = |failing: &str| {
        for tool in ["cargo", "rustc", "nvidia-smi", "sha256sum", "rsync", "curl"] {
            let path = bin.join(tool);
            fs::write(&path, if tool == failing { "#!/bin/sh\nexit 1\n" } else { "#!/bin/sh\nexit 0\n" }).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let output = std::process::Command::new("sh").args(["-c", script]).env("HOME", &f.dir)
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display())).output().unwrap();
        assert!(!output.status.success());
        String::from_utf8(output.stdout).unwrap()
    };
    assert_eq!(run("nvidia-smi"), "reached=1\nfailed=driver\n");
    assert_eq!(run("cargo"), "reached=1\nfailed=cargo\n");
    assert_eq!(run("rustc"), "reached=1\nfailed=rustc\n");
    // Source builds also need the pinned host and kernel toolchains. The stub df
    // reports ample room; the stub rustup answers its installed lists.
    fs::write(bin.join("df"), "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available\\nstub 1 1 99999999\\n'\n").unwrap();
    for tool in ["cargo", "rustc"] { fs::write(bin.join(tool), "#!/bin/sh\nexit 0\n").unwrap(); }
    fs::set_permissions(bin.join("df"), fs::Permissions::from_mode(0o700)).unwrap();
    let toolchains = |rustup: Option<(&str, &str)>, kernel_bin: &str| {
        let _ = fs::remove_file(bin.join("rustup"));
        if let Some((toolchains, components)) = rustup {
            fs::write(bin.join("rustup"), format!("#!/bin/sh\ntest \"$RUSTUP_AUTO_INSTALL\" = 0 || exit 9\ncase \"$1\" in\ntoolchain) printf '{toolchains}';;\n\
                component) test -n '{components}' || exit 1; printf '{components}';;\n*) exit 1;;\nesac\n")).unwrap();
            fs::set_permissions(bin.join("rustup"), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let output = std::process::Command::new("sh").args(["-c", script]).env("HOME", &f.dir).env("KERNEL_BIN", kernel_bin)
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display())).output().unwrap();
        (output.status.success(), String::from_utf8(output.stdout).unwrap())
    };
    let host = "stable-aarch64-unknown-linux-gnu (default)\\n1.99.0-aarch64-unknown-linux-gnu (active)\\n";
    let kernel = "cargo-aarch64-unknown-linux-gnu\\nllvm-bitcode-linker-aarch64-unknown-linux-gnu\\nllvm-tools-aarch64-unknown-linux-gnu\\nrust-src\\nrust-std-nvptx64-nvidia-cuda\\n";
    let (ok, out) = toolchains(None, "");
    assert!(!ok && out.ends_with("weights=missing\nfailed=rustup\n"), "{out}");
    let (ok, out) = toolchains(Some(("stable-aarch64-unknown-linux-gnu (default)\\n1.99.01-aarch64-unknown-linux-gnu\\n", kernel)), "");
    assert!(!ok && out.ends_with("failed=toolchain\n"), "{out}");
    let (ok, out) = toolchains(Some((host, "")), "");
    assert!(!ok && out.ends_with("failed=kernel-toolchain\n"), "{out}");
    let (ok, out) = toolchains(Some((host, &kernel.replace("rust-src\\n", ""))), "");
    assert!(!ok && out.ends_with("failed=kernel-toolchain\n"), "{out}");
    let (ok, out) = toolchains(Some((host, kernel)), "");
    assert!(ok && out.ends_with("toolchains=ok\nrust=ok\ndriver=ok\n"), "{out}");
    // KERNEL_BIN names the kernel compiler itself; rustup is then not required.
    let (ok, out) = toolchains(None, "/opt/kernel/bin");
    assert!(ok && !out.contains("toolchains="), "{out}");
    // The GPU use row reads the compute process count and the highest power draw.
    fs::write(bin.join("nvidia-smi"), "#!/bin/sh\ncase \"$1\" in\n--query-gpu=name,driver_version) echo 'NVIDIA GB10, 580.95';;\n--query-compute-apps=pid) printf '12\\n34\\n';;\n--query-gpu=power.draw) printf '19.5\\n41.25\\n';;\nesac\n").unwrap();
    let (ok, out) = toolchains(None, "/opt/kernel/bin");
    assert!(ok && out.contains("gpu=NVIDIA GB10, 580.95\ngpu_apps=2\ngpu_watts=41.25\n"), "{out}");
}

/// The toolchain pins doctor checks are the ones the build uses.
#[test]
fn doctor_toolchain_pins_match_the_repository() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let toolchain = fs::read_to_string(root.join("rust-toolchain.toml")).unwrap();
    assert!(toolchain.contains(&format!("channel = \"{}\"", diagnostics::HOST_TOOLCHAIN)), "{toolchain}");
    let kernels = fs::read_to_string(root.join("engine/crates/gpu/build-kernels/src/main.rs")).unwrap();
    assert!(kernels.contains(&format!("const NIGHTLY: &str = \"{}\";", diagnostics::KERNEL_TOOLCHAIN)));
    assert!(include_str!("deployment.rs").contains(&format!("export RUSTUP_TOOLCHAIN={}\n", diagnostics::HOST_TOOLCHAIN)));
}

/// Doctor hashes the staged files vade up would run, so a build that vanished
/// or changed on a Spark fails doctor, not the next start.
#[test]
fn doctor_checks_each_spark_staged_build_against_the_manifest() {
    let doctor = |f: &mut Fixture| {
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(false)));
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        let row = printed.lines().find(|line| line[6..].starts_with("a staged build")).unwrap_or_else(|| panic!("{printed}")).to_owned();
        (result.is_ok(), row)
    };
    let mut f = Fixture::fp4(1, false);
    let (passed, row) = doctor(&mut f);
    assert!(passed && row.ends_with("engine, PTX and helper match the build manifest"), "{row}");
    let hashed = f.app.runner.calls.iter().find(|e| e.label == "a artifact hashes").unwrap();
    assert_eq!(hashed.args.last().unwrap(), "sha256sum \"$HOME\"/stage/a/flash-serve \"$HOME\"/stage/flash.ptx \"$HOME\"/stage/a/vade");
    f.app.runner.bad_hash = true;
    let (passed, row) = doctor(&mut f);
    assert!(!passed && row.starts_with("FAIL") && row.contains("staged hashes differ from the manifest"), "{row}");
    // No manifest yet: nothing to hash, and doctor says what to run.
    let mut f = Fixture::fp4(1, false);
    fs::remove_file(&f.app.config.build_manifest).unwrap();
    assert!(doctor(&mut f).1.contains("skip  a staged build        no build manifest; run vade install"));
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "a artifact hashes"));
    // SSH that never ran the check script hides the hash check too.
    let mut f = Fixture::fp4(1, false);
    f.app.runner.probe = Some((false, ""));
    assert!(doctor(&mut f).1.contains("needs SSH"));
}
