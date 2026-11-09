//! Real subprocesses prove long private paths need no filesystem redirects.
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    os::unix::net::UnixListener,
    path::PathBuf,
    process::{Command, Stdio},
};
const SOCKET: &str = "tunnel-0123456789abcdef.sock";

#[test]
fn long_paths_support_socket_probe_and_launcher_preview() {
    if std::env::var_os("VADE_TEST_SOCKET_LISTENER").is_some() {
        let listener = UnixListener::bind(SOCKET).unwrap();
        println!("ready");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_exact(&mut [0]).unwrap();
        drop(listener);
        return;
    }
    let root = std::env::temp_dir().canonicalize().unwrap()
        .join(format!("vade-control-{}", std::process::id())).join("x".repeat(180));
    // A failed run keeps its folder; a later run with the same process ID would find its socket.
    let _ = fs::remove_dir_all(root.parent().unwrap());
    fs::create_dir_all(&root).unwrap();
    assert!(root.as_os_str().len() > 104);
    let mut listener = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "long_paths_support_socket_probe_and_launcher_preview", "--nocapture"])
        .env("VADE_TEST_SOCKET_LISTENER", "1").current_dir(&root)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    let mut output = BufReader::new(listener.stdout.take().unwrap());
    loop {
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        if line.trim() == "ready" { break; }
    }
    let probe = |name: &str| Command::new(env!("CARGO_BIN_EXE_vade"))
        .args(["__socket-probe", name]).current_dir(&root).status().unwrap().success();
    assert!(!probe(SOCKET), "a listening socket cannot be removed");
    assert!(!probe("../tunnel-0123456789abcdef.sock"), "private entry accepts basenames only");
    listener.stdin.take().unwrap().write_all(b"x").unwrap();
    assert!(listener.wait().unwrap().success());
    assert!(root.join(SOCKET).exists());
    assert!(probe(SOCKET), "a closed socket in a long directory is recognized");
    fs::remove_file(root.join(SOCKET)).unwrap();

    let config = root.join("config");
    fs::write(&config, "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = state\nmodel = qwen3.8-flash-next-fp4-local\n[machine spark]\nssh = fixture-no-network\nmodel_path = /models/fixture\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vade"))
        .args(["up", "--fp4-local", "--dry-run"])
        .env("VADE_CONFIG", &config).env("VADE_HOME", root.join("home"))
        .env_remove("VADE_API_KEY_FILE").output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let preview = String::from_utf8_lossy(&output.stdout);
    assert!(preview.contains("-S ./tunnel-0000000000000000.sock"), "{preview}");
    assert!(preview.contains(&format!("# folder: {}", root.join("state").display())), "{preview}");
    assert!(!root.join("state").exists(), "preview must not write state");
    fs::remove_dir_all(PathBuf::from(root.parent().unwrap())).unwrap();
}
