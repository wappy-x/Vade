//! A controller that is one of its configured machines runs that machine's
//! commands as local child processes: the same command strings, the same
//! request bytes and the same rank order. The other machines keep SSH.
use super::*;
use crate::runner::{fake_host, host_names, Host};

/// A fake computer named `name`, run by `user`, with one own address.
fn host(name: &str) -> Host {
    Host { name: Some(name.into()), user: Some("vade-user".into()), addresses: vec!["192.0.2.10".parse().unwrap()], ..Host::default() }
}

/// `vade up --fp8` on two machines whose SSH destinations are `first` and
/// spark-b, on the computer `this`.
fn up_on(this: Host, first: &str) -> Fixture {
    let _host = fake_host(this);
    let mut f = Fixture::new(2, false);
    f.app.config.machines[0].ssh = first.into();
    f.app.config.machines[1].ssh = "spark-b".into();
    f.app.execute(Command::Up(fp8_options())).unwrap();
    f
}

/// The worker requests of one run, by label, in call order.
fn worker_calls(f: &Fixture) -> Vec<&Exec> {
    f.app.runner.calls.iter().filter(|c| c.args.last().is_some_and(|a| a.ends_with("__worker"))).collect()
}

#[test]
fn a_controller_that_is_rank_zero_starts_it_locally_with_the_same_request_bytes() {
    let remote = up_on(host("controller"), "spark-a");
    let local = up_on(host("spark-a"), "spark-a");
    let (remote_calls, local_calls) = (worker_calls(&remote), worker_calls(&local));
    assert_eq!(remote_calls.iter().map(|c| &c.label).collect::<Vec<_>>(), local_calls.iter().map(|c| &c.label).collect::<Vec<_>>(), "same worker operations in the same rank order");
    assert!(remote_calls.iter().any(|c| c.label == "a start") && remote_calls.iter().any(|c| c.label == "b start"));
    // Each launch has its own token, control secret and run id; map the local run's values to the remote run's.
    let parsed = |c: &Exec| text::parse(std::str::from_utf8(&c.stdin).unwrap()).unwrap();
    let start = |calls: &[&Exec], label: &str| parsed(calls.iter().find(|c| c.label == label).unwrap());
    let (remote_b, local_b) = (start(&remote_calls, "b start"), start(&local_calls, "b start"));
    let fresh = local_b.iter().filter(|(key, value)| remote_b[*key] != **value).map(|(key, value)| (value.clone(), remote_b[key].clone())).collect::<Vec<_>>();
    assert_eq!(fresh.iter().map(|(value, _)| local_b.iter().find(|(_, v)| *v == value).unwrap().0.as_str()).collect::<BTreeSet<_>>(),
        BTreeSet::from(["env.SEQPAR_CONTROL_SECRET", "env.TP_RUN_ID", "token"]), "only the per-launch random values differ");
    let same = |bytes: &[u8]| fresh.iter().fold(String::from_utf8(bytes.to_vec()).unwrap(), |text, (new, old)| text.replace(new.as_str(), old));
    for (r, l) in remote_calls.iter().zip(&local_calls) {
        let command = r.args.last().unwrap();
        assert_eq!(same(&l.stdin), String::from_utf8(r.stdin.clone()).unwrap(), "{}: request bytes", r.label);
        assert_eq!(r.timeout, l.timeout, "{}", r.label);
        assert_eq!(r.program, "ssh", "{}", r.label);
        if r.label.starts_with("a ") {
            // No SSH to itself: the same command string through sh -c.
            assert_eq!((l.program.as_str(), l.args.as_slice()), ("sh", &["-c".to_string(), command.clone()][..]), "{}", l.label);
            assert!(!l.unset.iter().any(|name| matches!(name.as_str(), "HOME" | "PATH")));
        } else {
            assert_eq!((&l.program, &l.args), (&r.program, &r.args), "{}: SSH exactly as before", r.label);
        }
    }
    // The engine slot, world size and arguments of rank zero are those of the remote run.
    let (remote_a, local_a) = (start(&remote_calls, "a start"), start(&local_calls, "a start"));
    assert_eq!((&local_a["env.FLASH_SERVE_RANK"], &local_a["env.FLASH_SERVE_WORLD_SIZE"]), (&"0".to_string(), &"2".to_string()));
    assert_eq!(local_a.iter().filter(|(k, _)| k.starts_with("arg.")).collect::<Vec<_>>(), remote_a.iter().filter(|(k, _)| k.starts_with("arg.")).collect::<Vec<_>>());
    // Nothing goes over SSH to spark-a, and rank zero serves the endpoint itself: no tunnel.
    assert!(!local.app.runner.calls.iter().any(|c| c.program == "ssh" && c.args.iter().any(|a| a == "spark-a")));
    assert!(!local.app.runner.calls.iter().any(|c| c.label.starts_with("controller tunnel")));
    assert!(remote.app.runner.calls.iter().any(|c| c.label == "controller tunnel start"));
    assert!(local.app.runner.calls.iter().any(|c| c.label == "a artifact hashes" && c.program == "sh"));
    assert!(local.app.runner.calls.iter().any(|c| c.label == "engine health" && c.program == "sh"));
}

#[test]
fn down_stops_a_local_rank_locally_and_opens_no_tunnel() {
    let _host = fake_host(host("spark-a"));
    let mut f = Fixture::new(2, false);
    f.app.config.machines[0].ssh = "localhost".into();
    f.app.config.machines[1].ssh = "spark-b".into();
    f.app.up(&fp8_options()).unwrap();
    f.app.runner.calls.clear();
    f.app.down().unwrap();
    assert!(!f.recorded());
    for op in ["inspect", "stop", "gpu", "clean"] {
        assert_eq!(f.app.runner.calls.iter().find(|c| c.label == format!("a {op}")).unwrap().program, "sh", "{op}");
        assert_eq!(f.app.runner.calls.iter().find(|c| c.label == format!("b {op}")).unwrap().program, "ssh", "{op}");
    }
    assert!(!f.app.runner.calls.iter().any(|c| c.label.starts_with("controller tunnel")));
}

#[test]
fn no_local_machine_keeps_every_ssh_argv_of_today() {
    let f = up_on(host("controller"), "spark-a");
    assert!(f.app.runner.calls.iter().filter(|c| !c.label.starts_with("SSH check of")).all(|c| !matches!(c.program.as_str(), "sh" | "/bin/sh")));
    let probe = f.app.runner.calls.iter().find(|c| c.label == "a probe").unwrap();
    let expected = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-o", "PermitLocalCommand=no", "-o", "ControlMaster=no", "-o", "ControlPath=none", "spark-a", "exec \"$HOME\"/stage/a/vade __worker"];
    assert_eq!((probe.program.as_str(), probe.args.as_slice()), ("ssh", &expected.map(String::from)[..]));
    assert!(probe.unset.is_empty() && probe.env.is_empty());
    // Copies keep today's rsync argv: options, the receiver's umask and folder, ssh, sources, host.
    let _host = fake_host(host("controller"));
    let copy = runner::copy_to("a prebuilt copy", "spark-a", "~/stage/x", vec!["--from0".into()], vec!["/src/one".into(), "/src/two".into()]);
    let ssh = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o PermitLocalCommand=no -o ControlMaster=no -o ControlPath=none";
    assert_eq!((copy.program.as_str(), copy.args.as_slice()), ("rsync", &["-a", "--chmod=go-rwx", "--from0", "--rsync-path=umask 077 && cd \"$HOME\"/stage/x && exec rsync",
        "-e", ssh, "/src/one", "/src/two", "spark-a:./"].map(String::from)[..]));
}

#[test]
fn a_copy_to_this_computer_is_a_local_rsync_in_the_same_folder() {
    let _host = fake_host(host("spark-a"));
    let copy = runner::copy_to("a prebuilt copy", "spark-a", "~/stage/x", vec!["--from0".into()], vec!["/src/one".into(), "relative/".into()]);
    let here = std::env::current_dir().unwrap().to_string_lossy().into_owned();
    assert_eq!(copy.program, "sh");
    assert_eq!(copy.args, ["-c", "umask 077 && cd \"$HOME\"/stage/x && exec rsync \"$@\"", "sh", "-a", "--chmod=go-rwx", "--from0", "/src/one", &format!("{here}/relative/"), "./"]);
}

#[test]
fn two_machines_on_this_computer_are_refused_before_anything_starts() {
    let _host = fake_host(host("spark-a"));
    let mut f = Fixture::new(2, false);
    f.app.config.machines[0].ssh = "spark-a".into();
    f.app.config.machines[1].ssh = "localhost".into();
    let error = f.app.execute(Command::Up(fp8_options())).unwrap_err();
    assert_eq!(error, "Machines a, b all name this computer, so nothing was started; give each machine the SSH destination of its own Spark.");
    assert!(f.app.runner.calls.is_empty() && f.app.runner.ports.is_empty() && !f.recorded());
    let error = f.app.execute(Command::Install).unwrap_err();
    assert!(error.starts_with("Machines a, b all name this computer"), "{error}");
    assert!(f.app.runner.calls.is_empty());
}

#[test]
fn a_local_serving_rank_needs_local_port_equal_to_serve_port() {
    let _host = fake_host(host("spark-a"));
    let mut f = Fixture::new(2, false);
    f.app.config.machines[0].ssh = "spark-a".into();
    f.app.config.local_port = 18090;
    let error = f.app.up(&fp8_options()).unwrap_err();
    assert_eq!(error, "a is this computer, so its engine is the endpoint and this launch needs local_port equal to serve_port, without --replicas; nothing was started.");
    assert!(f.app.runner.calls.is_empty() && !f.recorded());
    // A machine on this computer that does not serve HTTP needs no tunnel and no equal port.
    f.app.config.machines[0].ssh = "spark-c".into();
    f.app.config.machines[1].ssh = "spark-a".into();
    f.app.up(&fp8_options()).unwrap();
    assert!(f.app.runner.calls.iter().any(|c| c.label == "controller tunnel start"));
    assert_eq!(f.app.runner.calls.iter().find(|c| c.label == "b start").unwrap().program, "sh");
}

#[test]
fn destinations_that_name_this_computer() {
    let mut this = host("spark-a.example.test");
    this.aliases.insert("rank0".into(), ("192.0.2.10".into(), "vade-user".into()));
    this.aliases.insert("other-user".into(), ("spark-a".into(), "root".into()));
    this.aliases.insert("peer".into(), ("198.51.100.7".into(), "vade-user".into()));
    for (destination, local) in [
        ("spark-a", true), ("spark-a.example.test", true), ("SPARK-A", true), ("spark-a.other.test", false),
        ("localhost", true), ("127.0.0.1", true), ("127.0.1.1", true), ("::1", true), ("[::1]", true),
        ("vade-user@localhost", true), ("vade-user@spark-a", true), ("root@spark-a", false),
        ("192.0.2.10", true), ("vade-user@192.0.2.10", true), ("192.0.2.11", false), ("198.51.100.7", false),
        ("spark-b", false), ("spark-b.example.test", false), ("rank0", true), ("other-user", false), ("peer", false),
    ] {
        assert_eq!(host_names(destination, &this), local, "{destination}");
    }
    let short = host("spark-a");
    assert!(host_names("spark-a.example.test", &short) && !host_names("spark-ab", &short));
    assert!(!host_names("spark-a", &Host::default()) && host_names("localhost", &Host::default()));
}

#[test]
fn a_local_command_gets_its_stdin_and_home_like_an_ssh_session() {
    let _guard = runner::process_test_lock();
    let mut exec = runner::local("printf '%s|' \"$HOME\"; cat; printf '|%s' \"${CARGO_PKG_NAME:-unset}\"".into(), "local check");
    exec.stdin = b"op=probe\n".to_vec();
    let output = runner::SystemRunner { dry: false }.run(&exec).unwrap();
    assert!(output.success);
    // Cargo sets CARGO_PKG_NAME for this test process; an SSH session would not carry it.
    assert_eq!(output.stdout, format!("{}|op=probe\n|unset", std::env::var("HOME").unwrap()));
}
