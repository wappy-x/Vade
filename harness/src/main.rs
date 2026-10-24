mod cli;
mod delegation;
mod interaction;
mod native_children;

use std::io;
use std::process::ExitCode;
use vade_harness::{gateway, terminal};

use cli::{
    App,
    args::{Config, Mode, Parsed},
    events::{error_category, error_hint},
};
#[cfg(test)]
use cli::{
    approval::{ToolPreviews, request_tool_preview},
    args::{Permission, validate_url},
    delegated::executable_path,
    events::{
        Pending, TurnResult, WorkerEvent, WorkerResult, command_summary, patch_summary,
    },
    help::HELP,
    worker::emit_delta,
};
#[cfg(test)]
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
#[cfg(test)]
use vade_harness::{coding_tools, shell::ApprovalPolicy, terminal::Key};

fn main() -> ExitCode {
    run_main(std::env::args().skip(1).collect())
}

fn run_main(mut arguments: Vec<String>) -> ExitCode {
    // Utility commands own their flags and never create a workspace executor,
    // terminal agent UI, or delegated process.
    if let Err(error) = vade_harness::auth::configure(&mut arguments) {
        vade_harness::diagnostics::stderr_line(format_args!("vade: {error}"));
        return ExitCode::from(2);
    }
    let mut args = arguments.iter().cloned();
    match args.next().as_deref() {
        Some("workload") => return vade_harness::workload::cli(args),
        Some("setup") => return vade_harness::setup::cli(args),
        Some("doctor") => return vade_harness::doctor::cli(args),
        // The staged harness carries the standalone relay, e.g. for vade up --replicas.
        Some("vade-gateway") => return vade_harness::gateway::invisible::exit(args),
        Some("demo-compaction") => {
            if args.next().is_some() {
                vade_harness::diagnostics::stderr_line(format_args!("vade: demo-compaction takes no arguments"));
                return ExitCode::from(2);
            }
            return match vade_harness::context::offline_compaction_demo() {
                Ok(report) => vade_harness::diagnostics::print_stdout(&report),
                Err(error) => {
                    vade_harness::diagnostics::stderr_line(format_args!("vade: {error}"));
                    ExitCode::FAILURE
                }
            };
        }
        _ => {}
    }
    let parsed = match Config::parse(arguments) {
        Ok(parsed) => parsed,
        Err(error) => {
            vade_harness::diagnostics::stderr_line(format_args!("vade: {error}"));
            return ExitCode::from(2);
        }
    };
    let mut config = match parsed {
        Parsed::Help(mode) => {
            return vade_harness::diagnostics::print_stdout(mode.help());
        }
        Parsed::Version => {
            return vade_harness::diagnostics::print_stdout(&format!("vade-harness {}\n", env!("CARGO_PKG_VERSION")));
        }
        Parsed::Run(config) => config,
    };
    if matches!(config.mode, Mode::Agent | Mode::Direct | Mode::Gateway)
        && vade_harness::auth::preload().is_err()
    {
        // The loader prints the local path once, before any terminal or listener.
        return ExitCode::FAILURE;
    }
    if config.mode == Mode::Gateway {
        let result = if config.upstreams.len() == 1 {
            gateway::run_with_access(&config.bind, &config.upstreams[0], config.access)
        } else {
            gateway::Cluster::new(&config.upstreams, gateway::RoutingConfig::default())
                .and_then(|cluster| gateway::run_cluster_with_access(&config.bind, cluster, config.access))
        };
        match result {
            Ok(()) => return ExitCode::SUCCESS,
            Err(error) => {
                vade_harness::diagnostics::stderr_line(format_args!("vade: gateway stopped ({}).{}", error_category(&error), error_hint(&error)));
                return ExitCode::FAILURE;
            }
        }
    }
    if config.state.is_none() {
        // Chat state defaults to the Vade home so git never sees a conversation.
        match vade_harness::tools::StateLocation::from_env() {
            Ok(state) => config.state = Some(state),
            Err(error) => {
                vade_harness::diagnostics::stderr_line(format_args!("vade: {error}"));
                return ExitCode::from(2);
            }
        }
    }
    let result = App::new(config).and_then(|mut app| app.run());
    match result {
        Ok(code) => code,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(error) => {
            vade_harness::diagnostics::stderr_line(format_args!("vade: {}", terminal::sanitize(&error.to_string())));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_check_failure_only_changes_normal_run_exit() {
        assert_eq!([(true, true), (false, true), (true, false), (false, false)].map(|(s, f)| cli::ui::run_exit(s, f)), [3, 1, 0, 1].map(ExitCode::from));
    }
    fn parse(args: &[&str]) -> Result<Parsed, &'static str> {
        Config::parse(args.iter().map(|s| s.to_string())).map_err(|error| error.message)
    }
    #[test]
    fn chat_state_flags_select_the_location() {
        use vade_harness::tools::StateLocation;
        let state = |extra: &[&str]| {
            let mut args = vec!["direct", "--endpoint", "http://localhost:1", "--model", "m"];
            args.extend_from_slice(extra);
            match parse(&args) {
                Ok(Parsed::Run(config)) => Ok(config.state),
                Ok(_) => panic!(),
                Err(error) => Err(error),
            }
        };
        assert_eq!(state(&[]), Ok(None), "main applies the Vade-home default");
        assert_eq!(state(&["--vade-home", "/v"]), Ok(Some(StateLocation::Home("/v".into()))));
        assert_eq!(state(&["--state-in-project"]), Ok(Some(StateLocation::Project)));
        assert!(state(&["--vade-home", "/v", "--state-in-project"]).unwrap_err().contains("not both"));
        assert!(state(&["--vade-home", "relative"]).unwrap_err().contains("absolute"));
        assert!(state(&["--state-in-project", "--state-in-project"]).is_err());
        assert!(parse(&["gateway", "--upstream", "http://localhost:1", "--state-in-project"]).is_err());
    }
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
    #[test]
    fn completion_eviction_invalidates_recall_and_up_clamps_stale_indices() {
        let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
            panic!()
        };
        let mut app = App::new(config).unwrap();
        for i in 0..10 {
            app.complete_session(format!("prompt-{i}"), "answer".into())
                .unwrap();
        }
        app.input.set("unfinished draft".into());
        app.key(Key::Up).unwrap();
        assert_eq!(app.recall, Some(9));
        assert_eq!(app.input.text(), "prompt-9");
        assert_eq!(app.draft, "unfinished draft");

        // The worker completion path uses this same helper. One large answer
        // evicts all ten old turns while the user is browsing their recall.
        let prompt = "new prompt";
        app.complete_session(
            prompt.into(),
            "x".repeat(vade_harness::session::MAX_HISTORY_BYTES - prompt.len()),
        )
        .unwrap();
        assert_eq!(app.session.turns(), 1);
        assert_eq!(app.session.pruned_turns, 10);
        assert_eq!(app.recall, None);
        assert!(app.draft.is_empty());
        assert_eq!(
            app.input.text(),
            "prompt-9",
            "keep the active editor intact"
        );
        app.key(Key::Up).unwrap();
        assert_eq!(app.input.text(), prompt);
        assert_eq!(app.recall, Some(0));

        // Defense in depth if a future history mutation forgets invalidation.
        app.recall = Some(9);
        app.key(Key::Up).unwrap();
        assert_eq!(app.input.text(), prompt);
        assert_eq!(app.recall, Some(0));
        app.key(Key::Down).unwrap();
        assert_eq!(app.input.text(), "prompt-9");
        assert_eq!(app.recall, None);
    }
    #[test]
    fn bare_compact_uses_the_automatic_plan_and_refuses_an_empty_history() {
        let workspace =
            std::env::temp_dir().join(format!("vade-bare-compact-{}", std::process::id()));
        std::fs::create_dir(&workspace).unwrap();
        let Parsed::Run(config) = parse(&[
            "agent",
            "--plain",
            "--endpoint",
            "http://localhost:1",
            "--model",
            "mock",
            "--workspace",
            workspace.to_str().unwrap(),
        ])
        .unwrap() else {
            panic!()
        };
        let mut app = App::new(config).unwrap();
        let before = app.native.as_ref().unwrap().messages().len();
        app.line_input("/compact".into()).unwrap();
        app.line_input("/compact-preview".into()).unwrap();
        // Nothing closed is eligible yet: history and journal stay as they were.
        assert_eq!(app.native.as_ref().unwrap().messages().len(), before);
        drop(app);
        std::fs::remove_dir_all(workspace).unwrap();
    }
    #[test]
    fn single_key_approval_answers_only_on_an_empty_editor() {
        let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
            panic!()
        };
        let mut app = App::new(config).unwrap();
        let settled = Instant::now() - cli::ui::APPROVAL_SETTLE;
        // A draft keeps y/n as text; the approval stays pending.
        let (tx, rx) = mpsc::channel();
        app.approval_reply = Some(cli::approval::ApprovalReply::new(tx, None));
        app.approval_shown = Some(settled);
        app.input.set("not yet".into());
        app.key(Key::Text("y".into())).unwrap();
        assert_eq!(app.input.text(), "not yety");
        assert!(app.approval_reply.is_some());
        assert!(rx.try_recv().is_err());
        // An empty editor answers with one key.
        app.input.clear();
        app.key(Key::Text("n".into())).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(false));
        assert!(app.approval_reply.is_none());
        let (tx, rx) = mpsc::channel();
        app.approval_reply = Some(cli::approval::ApprovalReply::new(tx, None));
        app.approval_shown = Some(settled);
        app.key(Key::Text("Y".into())).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
        assert!(app.input.is_empty());
    }
    #[test]
    fn single_key_approval_ignores_multiline_input_and_type_ahead() {
        let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
            panic!()
        };
        let mut app = App::new(config).unwrap();
        // Multiline (/paste) collection: a leading y is part of the line.
        let (tx, rx) = mpsc::channel();
        app.approval_reply = Some(cli::approval::ApprovalReply::new(tx, None));
        app.approval_shown = Some(Instant::now() - cli::ui::APPROVAL_SETTLE);
        app.multiline = Some(String::new());
        app.key(Key::Text("y".into())).unwrap();
        assert_eq!(app.input.text(), "y");
        assert!(app.approval_reply.is_some());
        assert!(rx.try_recv().is_err());
        app.multiline = None;
        app.input.clear();
        // Type-ahead: a y within the settle time of the prompt appearing is
        // draft text, not an answer; after the settle time it answers.
        app.approval_shown = Some(Instant::now());
        app.key(Key::Text("y".into())).unwrap();
        assert_eq!(app.input.text(), "y");
        assert!(app.approval_reply.is_some());
        assert!(rx.try_recv().is_err());
        app.input.clear();
        app.approval_shown = Some(Instant::now() - cli::ui::APPROVAL_SETTLE);
        app.key(Key::Text("y".into())).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
        assert!(app.approval_reply.is_none());
    }
    #[test]
    fn ctrl_d_exits_through_cancellation_while_approval_is_pending() {
        let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
            panic!()
        };
        let mut app = App::new(config).unwrap();
        // Nonempty draft: Ctrl+D keeps the approval interaction as text.
        let (tx, rx) = mpsc::channel();
        app.approval_reply = Some(cli::approval::ApprovalReply::new(tx, None));
        app.approval_shown = Some(Instant::now() - cli::ui::APPROVAL_SETTLE);
        app.input.set("draft".into());
        app.key(Key::End).unwrap();
        assert!(app.approval_reply.is_some());
        // Empty editor: Ctrl+D is an exit, exactly like /exit during an
        // approval: quit, decline by closing the approval, and cancel.
        app.input.clear();
        app.key(Key::End).unwrap();
        assert!(app.approval_reply.is_none());
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_err(),
            "approval must be declined by closing, never approved");
    }
    #[test]
    fn safe_default_and_explicit_direct_configuration() {
        let Parsed::Run(c) = parse(&[]).unwrap() else {
            panic!()
        };
        assert_eq!(c.mode, Mode::Demo);
        assert!(c.endpoint.is_none());
        assert!(parse(&["direct"]).is_err());
        assert!(
            parse(&[
                "direct",
                "--endpoint",
                "http://127.0.0.1:9",
                "--model",
                "mock"
            ])
            .is_ok()
        );
        assert!(
            parse(&[
                "direct",
                "--endpoint",
                "https://example.invalid",
                "--model",
                "mock"
            ])
            .is_err()
        );
        assert!(parse(&["--endpoint", "http://example.invalid"]).is_err());
        assert!(
            parse(&[
                "agent",
                "--endpoint",
                "http://127.0.0.1:9",
                "--model",
                "mock"
            ])
            .is_err()
        );
        assert!(parse(&["agent", "--workspace", ".", "--model", "mock"]).is_err());
        assert!(
            parse(&[
                "agent",
                "--workspace",
                ".",
                "--endpoint",
                "http://127.0.0.1:9"
            ])
            .is_err()
        );
        for mode in [vec!["agent"], vec!["--mode", "agent"]] {
            let mut args = mode;
            args.extend([
                "--workspace",
                ".",
                "--endpoint",
                "http://127.0.0.1:9",
                "--model",
                "mock",
            ]);
            let Parsed::Run(config) = parse(&args).unwrap() else {
                panic!()
            };
            assert_eq!(config.mode, Mode::Agent);
        }
    }
    #[test]
    fn modes_reject_conflicts_and_irrelevant_options() {
        for args in [
            vec!["demo", "direct"],
            vec!["--mode", "bogus"],
            vec!["gateway"],
            vec!["--unknown"],
            vec!["--model"],
            vec!["--prompt", "a", "--prompt", "b"],
            vec![
                "gateway",
                "--upstream",
                "http://example.invalid",
                "--session",
                "x",
            ],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
        let Parsed::Run(c) = parse(&["gateway", "--upstream", "http://example.invalid"]).unwrap()
        else {
            panic!()
        };
        assert_eq!(c.bind, "127.0.0.1:8787");
    }
    #[test]
    fn children_require_explicit_native_opt_in() {
        for args in [
            vec!["--children"],
            vec!["direct", "--children"],
            vec!["codex", "--children"],
            vec!["hermes", "--children"],
            vec!["gateway", "--children"],
        ] {
            assert!(parse(&args).is_err());
        }
        let mut args = vec![
            "agent",
            "--workspace",
            ".",
            "--endpoint",
            "http://127.0.0.1:9",
            "--model",
            "mock",
        ];
        let Parsed::Run(config) = parse(&args).unwrap() else {
            panic!()
        };
        assert!(!config.children);
        args.push("--children");
        let Parsed::Run(config) = parse(&args).unwrap() else {
            panic!()
        };
        assert!(config.children);
        args.push("--children");
        assert!(parse(&args).is_err());
    }
    #[test]
    fn urls_refuse_embedded_credentials_and_control_characters() {
        for url in [
            "http://user:secret@example.invalid",
            "http://host/?token=x",
            "http://host/#x",
            "http://host/\n",
            "http://",
            "file:///x",
        ] {
            assert!(validate_url(url).is_err());
        }
        assert!(validate_url("http://[::1]:1234/v1").is_ok());
    }

    #[test]
    fn generation_flags_are_bounded_unique_and_native_or_direct_only() {
        let base = [
            "direct",
            "--endpoint",
            "http://example.invalid",
            "--model",
            "mock",
        ];
        for options in [
            vec!["--max-output-tokens", "0"],
            vec!["--max-output-tokens", "-1"],
            vec!["--max-output-tokens", "18446744073709551616"],
            vec!["--thinking", "auto"],
            vec!["--thinking", "on", "--thinking", "off"],
            vec!["--temperature", "NaN"],
            vec!["--temperature", "-0.1"],
            vec!["--seed", "-1"],
        ] {
            assert!(parse(&[base.as_slice(), &options].concat()).is_err());
        }
        let Parsed::Run(config) = parse(
            &[
                base.as_slice(),
                &["--max-output-tokens", "1", "--thinking", "off"],
            ]
            .concat(),
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(config.generation.max_output_tokens, Some(1));
        assert_eq!(config.generation.thinking, Some(false));
        assert!(parse(&["demo", "--thinking", "on"]).is_err());
        assert!(HELP.contains("--max-output-tokens N") && HELP.contains("--thinking on|off"));
    }

    #[test]
    fn repeated_upstreams_enable_bounded_cluster_configuration_only_in_gateway() {
        let Parsed::Run(config) = parse(&[
            "gateway",
            "--upstream",
            "http://127.0.0.1:18001",
            "--upstream",
            "http://127.0.0.1:18002",
            "--upstream",
            "http://127.0.0.1:18003",
        ])
        .unwrap() else {
            panic!()
        };
        assert_eq!(config.upstreams.len(), 3);
        assert_eq!(config.bind, "127.0.0.1:8787");
        assert!(
            parse(&[
                "demo",
                "--upstream",
                "http://127.0.0.1:18001",
                "--upstream",
                "http://127.0.0.1:18002"
            ])
            .is_err()
        );
        let mut args = vec!["gateway"];
        for _ in 0..65 {
            args.extend(["--upstream", "http://127.0.0.1:18001"]);
        }
        assert!(parse(&args).is_err());
    }
    #[test]
    fn browser_allow_lists_and_token_are_gateway_only_and_validated() {
        let base = ["gateway", "--upstream", "http://127.0.0.1:18001"];
        let with = |extra: &[&str]| parse(&[&base[..], extra].concat());
        assert!(with(&["--allow-origin", "https://app.example", "--allow-origin", "null", "--allow-host", "vade.lan", "--allow-host", "box:9", "--token", "t0k"]).is_ok());
        for extra in [
            &["--allow-origin", "*"][..],
            &["--allow-origin", "https://app.example/"],
            &["--allow-host", "a/b"],
            &["--token", "two words"],
            &["--token", "a", "--token", "b"],
            &["--token", ""],
            &["--token"],
        ] {
            assert!(with(extra).is_err(), "{extra:?}");
        }
        for args in [
            &["demo", "--token", "t"][..],
            &["direct", "--endpoint", "http://127.0.0.1:1/v1/chat/completions", "--model", "m", "--allow-origin", "null"],
            &["agent", "--endpoint", "http://127.0.0.1:1/v1/chat/completions", "--model", "m", "--workspace", ".", "--allow-host", "x"],
        ] {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
    #[test]
    fn codex_requires_explicit_workspace_and_permission() {
        for args in [
            vec!["codex"],
            vec!["codex", "--workspace", "."],
            vec!["codex", "--permission", "read-only"],
            vec!["codex", "--workspace", ".", "--permission", "unrestricted"],
            vec!["demo", "--permission", "workspace-write"],
            vec!["demo", "--thread", "thread-1"],
            vec!["demo", "--codex-bin", "mock"],
            vec![
                "codex",
                "--workspace",
                ".",
                "--permission",
                "read-only",
                "--endpoint",
                "http://example.invalid",
            ],
            vec![
                "codex",
                "--workspace",
                ".",
                "--permission",
                "read-only",
                "--thread",
                "bad\nthread",
            ],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
        let Parsed::Run(config) = parse(&[
            "codex",
            "--workspace",
            ".",
            "--permission",
            "read-only",
            "--codex-bin",
            "mock binary",
            "--thread",
            "thread-1",
        ])
        .unwrap() else {
            panic!()
        };
        assert_eq!(config.mode, Mode::Codex);
        assert_eq!(config.permission, Some(Permission::ReadOnly));
        assert_eq!(config.thread.as_deref(), Some("thread-1"));
        assert_eq!(config.codex_bin, PathBuf::from("mock binary"));
        assert!(config.model.is_none());
        let Parsed::Run(config) = parse(&[
            "--mode",
            "codex",
            "--workspace",
            ".",
            "--permission",
            "workspace-write",
        ])
        .unwrap() else {
            panic!()
        };
        assert_eq!(config.permission, Some(Permission::WorkspaceWrite));
        assert_eq!(
            executable_path(std::path::Path::new("codex")).unwrap(),
            PathBuf::from("codex")
        );
        assert_eq!(
            executable_path(std::path::Path::new("./mock-child")).unwrap(),
            std::env::current_dir().unwrap().join("./mock-child")
        );
    }

    #[test]
    fn native_policy_and_session_flags_are_explicit_and_mode_scoped() {
        let args = [
            "agent",
            "--endpoint",
            "http://127.0.0.1:1",
            "--model",
            "mock",
            "--workspace",
            ".",
        ];
        let Parsed::Run(config) = parse(&args).unwrap() else {
            panic!()
        };
        assert_eq!(config.approval, ApprovalPolicy::Ask);
        for tail in [
            vec!["--approval", "bad"],
            vec!["--context-tokens", "4096"],
            vec!["--resume", "../bad"],
            vec!["--session", "one", "--resume", "one"],
            vec!["--shell-exclude", "../escape"],
            vec!["--shell-exclude", "build|src"],
        ] {
            let mut all = args.to_vec();
            all.extend(tail);
            assert!(parse(&all).is_err());
        }
        for flag in [
            "--approval",
            "--resume",
            "--context-tokens",
            "--shell-exclude",
            "--shell-toolchain",
        ] {
            assert!(parse(&["demo", flag, "auto"]).is_err());
        }
        let mut all = args.to_vec();
        all.extend([
            "--approval",
            "never",
            "--resume",
            "saved",
            "--context-tokens",
            "16384",
            "--shell-exclude",
            "build",
            "--shell-exclude",
            "vendor",
            "--shell-toolchain",
            "installed-rust-distribution",
        ]);
        let Parsed::Run(config) = parse(&all).unwrap() else {
            panic!()
        };
        assert_eq!(config.approval, ApprovalPolicy::Never);
        assert_eq!(config.resume.as_deref(), Some("saved"));
        assert_eq!(config.context_tokens, 16384);
        assert_eq!(
            config.shell_toolchain,
            Some(PathBuf::from("installed-rust-distribution"))
        );
        assert_eq!(
            config.shell_exclusions.describe(),
            "target, node_modules, build, vendor"
        );
    }
    #[test]
    fn hermes_configuration_requires_selected_scope_and_denial_policy() {
        for args in [
            vec!["hermes"],
            vec!["hermes", "--workspace", "."],
            vec!["hermes", "--permission", "deny-callbacks"],
            vec!["hermes", "--workspace", ".", "--permission", "read-only"],
            vec![
                "hermes",
                "--workspace",
                ".",
                "--permission",
                "deny-callbacks",
                "--model",
                "override",
            ],
            vec![
                "hermes",
                "--workspace",
                ".",
                "--permission",
                "deny-callbacks",
                "--thread",
                "id",
            ],
            vec![
                "codex",
                "--workspace",
                ".",
                "--permission",
                "deny-callbacks",
            ],
            vec!["demo", "--hermes-bin", "mock"],
            vec!["demo", "--external-session", "id"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
        let Parsed::Run(config) = parse(&[
            "hermes",
            "--workspace",
            ".",
            "--permission",
            "deny-callbacks",
            "--hermes-bin",
            "mock binary",
            "--external-session",
            "saved-session",
        ])
        .unwrap() else {
            panic!()
        };
        assert_eq!(config.mode, Mode::Hermes);
        assert_eq!(config.permission, Some(Permission::DenyCallbacks));
        assert_eq!(config.hermes_bin, PathBuf::from("mock binary"));
        assert_eq!(config.external_session.as_deref(), Some("saved-session"));
        assert!(config.model.is_none());
    }
    #[test]
    fn pending_drop_disconnects_all_lanes_before_joining() {
        let (tx, rx) = interaction::lanes();
        let (controls, _) = mpsc::sync_channel(8);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let worker = std::thread::spawn(move || {
            for _ in 0..10000 {
                tx.render("fixture output");
            }
            tx.finish(WorkerResult {
                http: None,
                result: Ok(TurnResult::NotStarted),
                backend: None,
            });
            worker_finished.store(true, Ordering::Relaxed);
        });
        let pending = Pending {
            budget: None,
            native_progress: None,
            user: String::new(),
            started: Instant::now(),
            cancel: Arc::new(AtomicBool::new(false)),
            rx,
            controls,
            steering_pending: Vec::new(),
            render_skipped: false,
            render_skip: 0,
            round_displayed: false,
            received: false,
            canceled: false,
            displayed: 0,
            provisional: String::new(),
            edits: 0,
            worker: Some(worker),
            join_on_drop: true,
        };
        drop(pending);
        assert!(finished.load(Ordering::Relaxed));
    }
    #[test]
    fn output_binding_respects_cancellation_and_utf8_chunk_boundaries() {
        let (tx, rx) = interaction::lanes();
        let cancel = AtomicBool::new(false);
        let mut emitted = 0;
        let text = "é".repeat(3000);
        emit_delta(&tx, &cancel, &mut emitted, &text);
        let (joined, skipped) = rx.render();
        assert!(!skipped);
        assert_eq!(joined, text);
        cancel.store(true, Ordering::Relaxed);
        emit_delta(&tx, &cancel, &mut emitted, "must not render");
        assert!(rx.render().0.is_empty());
    }
    #[test]
    fn error_categories_do_not_leak_sensitive_details() {
        let error = io::Error::other("sensitive URL and response content");
        assert_eq!(error_category(&error), "transport error");
        assert!(error_hint(&error).contains("vade status"));
        assert!(!error_hint(&error).contains("sensitive"));
        let refused = io::Error::from(io::ErrorKind::ConnectionRefused);
        assert!(error_hint(&refused).contains("vade status"));
    }
    #[test]
    fn tool_result_rows_summarize_command_and_patch_receipts() {
        let ok = "status: Exited(Some(0))\nstdout:\nline one\n\nline two\nstderr:\n";
        assert_eq!(
            command_summary(ok),
            ("exit 0 · 2 lines".into(), false, None)
        );
        let failed = "status: Exited(Some(101))\nstdout:\n\nstderr:\nerror[E0425]: cannot find value\n  --> src/main.rs:1:1\n";
        assert_eq!(
            command_summary(failed),
            (
                "exit 101 · 2 stderr".into(),
                true,
                Some("error[E0425]: cannot find value".into())
            )
        );
        for detail in [
            "status: Timeout\nstdout:\n\nstderr:\n",
            "status: Timeout\nstdout:\nstderr:\n",
        ] {
            assert_eq!(command_summary(detail), ("timeout".into(), true, None));
        }
        assert_eq!(
            patch_summary("Applied 2 file operations."),
            ("applied 2 file operations".into(), false, None)
        );
        assert!(patch_summary("{\"ok\":false}").1);
        assert_eq!(
            patch_summary(r#"{"ok":false,"uncertain_writes":false}"#),
            (
                "patch refused · no files written · request a new preview".into(),
                true,
                None
            )
        );
        assert!(
            patch_summary(r#"{"ok":false,"uncertain_writes":true}"#)
                .0
                .contains("uncertain")
        );
    }
    #[test]
    fn native_previews_survive_render_overflow_and_wait_for_fact_capacity() {
        let (tx, rx) = interaction::lanes();
        let cancel = AtomicBool::new(false);
        assert!(tx.render(&"x".repeat(128 * 1024)));
        let preview = || coding_tools::Preview {
            title: "Command",
            detail: "exact operation",
            approval_required: false,
        };
        assert!(request_tool_preview(&tx, preview(), &cancel));
        assert!(
            matches!(rx.fact(), Ok(WorkerEvent::ToolPreview { detail, .. }) if detail == "exact operation")
        );
        for _ in 0..16 {
            tx.fact(WorkerEvent::Notice).unwrap();
        }
        let worker = std::thread::spawn(move || request_tool_preview(&tx, preview(), &cancel));
        assert!(matches!(rx.fact(), Ok(WorkerEvent::Notice)));
        assert!(worker.join().unwrap());
    }
}

#[cfg(test)]
#[path = "native_children_tests.rs"]
mod native_children_tests;
