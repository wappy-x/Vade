//! Entry-point initialization order and default state placement.
use super::*;

#[test]
fn chat_state_defaults_to_the_vade_home_and_never_the_workspace() {
    const CASE: &str = "VADE_STATE_DEFAULT_TEST";
    if let Ok(workspace) = std::env::var(CASE) {
        let mut args: Vec<String> = ["agent", "--plain", "--endpoint", "http://127.0.0.1:0", "--model", "mock", "--session", "private-chat", "--prompt", "private prompt", "--workspace"]
            .map(String::from).into();
        args.push(workspace);
        if cfg!(target_os = "linux") { args.push("--no-check".into()); }
        if std::env::var_os("VADE_STATE_OPT_IN").is_some() { args.push("--state-in-project".into()); }
        // Port zero fails after startup has opened the journal; parse errors exit 2.
        assert_eq!(run_main(args), ExitCode::FAILURE);
        return;
    }
    let root = std::env::temp_dir().join(format!("vade-state-default-{}", std::process::id()));
    let (workspace, home) = (root.join("project"), root.join("home"));
    std::fs::create_dir_all(&workspace).unwrap();
    for opt_in in [false, true] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "tests::chat_state_defaults_to_the_vade_home_and_never_the_workspace", "--nocapture"])
            .env(CASE, &workspace).env("VADE_HOME", &home).env_remove("VADE_API_KEY_FILE");
        if opt_in { command.env("VADE_STATE_OPT_IN", "1"); }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let state = home.join("projects").join(vade_harness::tools::project_key(&workspace.canonicalize().unwrap()));
        assert!(state.join("sessions/private-chat.journal").is_file());
        if !opt_in {
            assert_eq!(std::fs::read_dir(&workspace).unwrap().count(), 0, "default must not write into the workspace");
        } else {
            assert!(workspace.join(".vade/sessions/private-chat.journal").is_file());
            assert!(std::fs::read_to_string(workspace.join(".vade/.gitignore")).unwrap().lines().any(|line| line == "*"));
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn stale_key_fails_before_ui_or_gateway_startup() {
    const CASE: &str = "VADE_AUTH_ENTRY_TEST";
    if let Ok(mode) = std::env::var(CASE) {
        let address = std::env::var("VADE_AUTH_ENTRY_BIND").unwrap();
        let endpoint = format!("http://{address}/v1");
        let args = if mode == "gateway" {
            vec![
                mode,
                "--upstream".into(),
                endpoint,
                "--bind".into(),
                address,
            ]
        } else {
            vec![
                mode,
                "--endpoint".into(),
                endpoint,
                "--model".into(),
                "mock".into(),
                "--workspace".into(),
                std::env::var("VADE_AUTH_ENTRY_WORKSPACE").unwrap(),
            ]
        };
        assert_eq!(run_main(args), ExitCode::FAILURE);
        return;
    }
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let root = std::env::temp_dir().join(format!(
        "vade-auth-entry-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let key = root.join("missing-key");
    for mode in ["agent", "direct", "gateway"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::stale_key_fails_before_ui_or_gateway_startup",
                "--nocapture",
            ])
            .env(CASE, mode)
            .env("VADE_API_KEY_FILE", &key)
            .env("VADE_AUTH_ENTRY_WORKSPACE", &root)
            .env(
                "VADE_AUTH_ENTRY_BIND",
                listener.local_addr().unwrap().to_string(),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(&format!("{key:?}")), "{error}");
        assert!(error.contains("file not found. Check the selected path"), "{error}");
        assert_eq!(error.lines().count(), 1);
        assert!(!root.exists());
        assert!(
            !output.stdout.windows(4).any(|bytes| bytes == b"\x1b[?1"),
            "terminal entered alternate screen"
        );
    }
}
