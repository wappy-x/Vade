//! Invalid startup never creates output or sends an upstream request.
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{fs, net::TcpListener, process::Command};

#[test]
fn valid_gateway_announces_its_listener_without_contacting_the_upstream() {
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/v1", peer.local_addr().unwrap());
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
    command.args(["--upstream", &endpoint, "--bind", "127.0.0.1:0"])
        .env_remove("VADE_API_KEY_FILE").env_remove("VADE_GATEWAY_UPSTREAM");
    let mut process = cli::Cli::start(&mut command);
    process.until(true, "Ctrl-C stops the gateway.");
    assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn invalid_gateway_and_replay_startup_leave_output_paths_untouched() {
    let root = std::env::temp_dir().join(format!("vade-gateway-startup-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let empty = root.join("empty-recordings");
    fs::create_dir(&empty).unwrap();
    let empty = empty.to_str().unwrap();
    let missing = root.join("missing-recordings");
    let missing = missing.to_str().unwrap();
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let occupied = peer.local_addr().unwrap().to_string();
    let endpoint = format!("http://{occupied}/v1");
    let cases: Vec<(bool, Vec<&str>, bool)> = vec![
        (false, vec!["--unknown"], false),
        (false, vec!["--upstream"], false),
        (false, vec!["--upstream", ""], false),
        (false, vec!["--upstream", "invalid"], false),
        (false, vec![], false),
        (false, vec!["--upstream", &endpoint, "--bind", "0.0.0.0:8787"], false),
        (false, vec!["--upstream", &endpoint, "--bind", &occupied], false),
        (false, vec!["--upstream", &endpoint], true),
        (true, vec![empty, "--unknown"], false),
        (true, vec![empty, "--concurrency", "0"], false),
        (true, vec![empty, "--concurrency", "invalid"], false),
        (true, vec![empty], false),
        (true, vec![empty, "--upstream", "invalid"], false),
        (true, vec![missing, "--upstream", &endpoint], false),
        (true, vec![empty, "--upstream", &endpoint], false),
        (true, vec![empty, "--upstream", &endpoint], true),
    ];
    for (index, (replay, args, bad_key)) in cases.iter().enumerate() {
        let stats = root.join(format!("stats-{index}"));
        let record = root.join(format!("record-{index}"));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
        cmd.env_remove("VADE_GATEWAY_UPSTREAM").env_remove("VADE_API_KEY_FILE");
        if *replay {
            cmd.args(["replay", args[0]]).arg("--stats").arg(&stats).args(&args[1..]);
        } else {
            cmd.args(["--bind", "127.0.0.1:0", "--record"]).arg(&record)
                .arg("--stats").arg(&stats).args(args);
        }
        if *bad_key { cmd.env("VADE_API_KEY_FILE", root.join("missing-key")); }
        let output = cli::Cli::start(&mut cmd).finish();
        assert!(!output.status.success(), "case {index}: {output:?}");
        assert!(!output.stderr.is_empty(), "case {index}: {output:?}");
        assert!(output.stdout.is_empty(), "case {index}: {output:?}");
        assert!(!stats.exists() && !record.exists(), "case {index} created output");
        assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "case {index} contacted upstream");
    }
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}
