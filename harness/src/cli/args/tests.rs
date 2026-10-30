//! Mode-scoped flags, defaults and argument validation.
use super::*;
use crate::cli::{delegated::executable_path, test_support::parse};

#[test]
fn no_web_is_a_launch_flag_for_native_chat() {
    let base = ["agent", "--endpoint", "http://example.org:80", "--model", "mock", "--workspace", ".", "--no-check"];
    let mut args = base.to_vec(); args.push("--no-web");
    let Parsed::Run(config) = parse(&args).unwrap() else { panic!() };
    assert!(!config.web);
    args.push("--no-web"); assert!(parse(&args).err().unwrap().contains("more than once"));
    assert!(parse(&["demo", "--no-web"]).is_err());
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
        vec!["--name", "-h"],
        vec!["--name", "last"],
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
