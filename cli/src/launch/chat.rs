//! Harness arguments for `vade chat` on a local launch: the native agent at
//! 8192 context tokens or more, direct text chat below, and where chat
//! journals are kept.
use std::{ffi::OsString, path::{Path, PathBuf}};

/// The native agent reserves at least this much context; direct text chat
/// negotiates its budget from the server and supports smaller launches.
pub(super) const AGENT_MIN_CONTEXT: u32 = 8192;

/// The line before chat opens on Linux: what the agent can and cannot do.
pub(super) const LINUX_NOTICE: &str = "On Linux, the agent reads, searches and edits files in this folder. It cannot run shell commands or checks; run builds and tests in your own terminal.";

/// The line before direct text chat opens, with the way back to the agent.
pub(super) fn low_context_notice(context: u32) -> String {
    format!("Context {context} is below the agent minimum of {AGENT_MIN_CONTEXT}, so chat opens without workspace tools and saves no session. For the agent, run vade down, then vade launch --context {AGENT_MIN_CONTEXT} or larger.")
}

pub(super) fn arguments(endpoint: &str, model: &str, workspace: &Path, context: u32, state: Vec<String>) -> Vec<String> {
    let agent = context >= AGENT_MIN_CONTEXT;
    let mut args = vec![
        if agent { "agent" } else { "direct" }.into(),
        "--endpoint".into(), endpoint.into(),
        "--model".into(), model.into(),
        "--workspace".into(), workspace.to_string_lossy().into_owned(),
    ];
    args.extend(state);
    if agent {
        args.extend(["--context-tokens".into(), context.to_string()]);
        if cfg!(target_os = "linux") { args.push("--no-check".into()); }
    }
    args
}

/// The harness arguments for `vade chat --resume` and `--name`. The harness
/// checks the values and resolves an ID, a name or `last`.
pub(crate) fn session_arguments(options: &crate::cli::ChatOptions) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(query) = &options.resume { args.extend(["--resume".into(), query.clone()]); }
    if let Some(name) = &options.name { args.extend(["--name".into(), name.clone()]); }
    args
}

pub(crate) const SESSIONS_FAILED: &str = "vade sessions did not finish; see the message above.";

/// `vade sessions`: the staged harness reads the saved sessions of the current
/// folder from the same state root that vade chat uses. No engine is needed.
pub(crate) fn sessions(harness: &str, action: &[String]) -> Result<crate::runner::Exec, String> {
    let workspace = std::env::current_dir().map_err(|_| "Open an existing project folder, then run vade sessions.")?;
    let mut args = vec!["sessions".to_string()];
    args.extend(action.iter().cloned());
    args.extend(["--workspace".into(), workspace.to_string_lossy().into_owned()]);
    args.extend(state_arguments()?);
    let mut exec = crate::runner::Exec::new("saved sessions", harness, args);
    exec.cwd = Some(workspace);
    Ok(exec)
}

/// Chat journals and input recall stay out of the project: the harness keeps
/// them in `<Vade home>/projects/<name>-<hash>`. VADE_CHAT_IN_PROJECT=1 opts into
/// the project's own git-ignored .vade folder instead.
/// The chat status strip's engine facts, which the harness cannot ask the
/// server for: `2 Sparks · drafting on`.
pub(crate) fn setup(sparks: usize, drafting: bool) -> String {
    format!("{sparks} Spark{}, drafting {}", if sparks == 1 { "" } else { "s" }, if drafting { "on" } else { "off" })
}

/// Where saved sessions are kept, for `vade sessions`.
pub(crate) fn state_arguments() -> Result<Vec<String>, String> {
    state_arguments_for(std::env::var_os("VADE_CHAT_IN_PROJECT"), super::root, false)
}

/// The state arguments for chat, which also reads the user MCP list.
pub(crate) fn chat_state_arguments() -> Result<Vec<String>, String> {
    state_arguments_for(std::env::var_os("VADE_CHAT_IN_PROJECT"), super::root, true)
}

fn state_arguments_for(opt_in: Option<OsString>, root: impl FnOnce() -> Result<PathBuf, String>, chat: bool) -> Result<Vec<String>, String> {
    match opt_in.as_ref().map(|value| value.to_str()) {
        None | Some(Some("0")) => Ok(vec!["--vade-home".into(), root()?.to_string_lossy().into_owned()]),
        // Chat reads the user MCP list from the Vade home; the launcher clears VADE_HOME.
        Some(Some("1")) => Ok(match root().ok().filter(|_| chat) {
            Some(root) => vec!["--state-in-project".into(), "--mcp-home".into(), root.to_string_lossy().into_owned()],
            None => vec!["--state-in-project".into()],
        }),
        _ => Err("Set VADE_CHAT_IN_PROJECT to 1 to keep chat history in the project's .vade folder, or unset it (or 0) for the Vade home.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{Exec, Runner, SystemRunner};
    use std::{fs, process::{Command, Output, Stdio}, time::{Duration, Instant}};

    fn run_harness(binary: &Path, workspace: &Path, args: &[String]) -> Output {
        let mut child = Command::new(binary).args(args).current_dir(workspace)
            .env_clear().env("HOME", workspace).env("PATH", std::env::var_os("PATH").unwrap())
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if child.try_wait().unwrap().is_some() { return child.wait_with_output().unwrap(); }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!("harness boundary check timed out: {}", String::from_utf8_lossy(&output.stderr));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn launcher_context_boundaries_pass_the_real_harness_parser() {
        let _lock = crate::runner::process_test_lock();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let target = root.join("harness/target");
        let mut build = Exec::new("launcher harness parser build", "cargo", vec!["+1.99.0".into(), "build".into(), "--offline".into(),
            "--manifest-path".into(), root.join("harness/Cargo.toml").to_string_lossy().into_owned(), "--bin".into(), "vade-harness".into()]);
        build.cwd = Some(root.join("engine"));
        build.timeout = Duration::from_secs(180);
        build.cancellable = false;
        build.env.extend([("CARGO_TARGET_DIR".into(), target.to_string_lossy().into_owned()),
            ("CARGO_NET_OFFLINE".into(), "true".into()), ("RUSTUP_AUTO_INSTALL".into(), "0".into())]);
        build.unset.push("CARGO_MAKEFLAGS".into());
        let output = SystemRunner { dry: false }.run(&build).unwrap();
        assert!(output.success, "{}", output.stdout);
        let scratch = root.join("engine/.local");
        crate::text::private_dir(&scratch).unwrap();
        let workspace = scratch.join(format!("launcher-parser-{}", crate::text::random_hex(12).unwrap()));
        crate::text::private_dir(&workspace).unwrap();
        let vade_home = workspace.with_extension("home");
        let home = vec!["--vade-home".to_string(), vade_home.to_string_lossy().into_owned()];
        crate::text::private_dir(&workspace.join(".vade")).unwrap();
        fs::write(workspace.join(".vade/check"), "cargo check\n").unwrap();
        let binary = target.join("debug/vade-harness");
        for context in [4096, 8191, 8192, 262144] {
            if context == AGENT_MIN_CONTEXT && cfg!(target_os = "macos") { fs::remove_file(workspace.join(".vade/check")).unwrap(); }
            let mut args = arguments("http://127.0.0.1:0", "parser-boundary", &workspace, context, home.clone());
            args.extend(["--prompt".into(), "hello".into()]);
            let output = run_harness(&binary, &workspace, &args);
            let diagnostics = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            // Port zero fails runtime client setup after argument parsing;
            // the agent reports the same invalid endpoint through its UI.
            assert_eq!(output.status.code(), Some(1), "context {context}: {diagnostics}");
            let error = if context < AGENT_MIN_CONTEXT { "endpoint port must be nonzero" } else { "Native setup failed (invalid upstream response); prompt not sent" };
            assert!(diagnostics.contains(error), "context {context}: {diagnostics}");
        }
        let mut invalid = arguments("http://127.0.0.1:0", "parser-boundary", &workspace, AGENT_MIN_CONTEXT, home.clone());
        let budget = invalid.iter().position(|arg| arg == "--context-tokens").unwrap() + 1;
        invalid[budget] = "4096".into();
        let output = run_harness(&binary, &workspace, &invalid);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("context token budget"));
        if cfg!(target_os = "linux") {
            let mut invalid = arguments("http://127.0.0.1:0", "parser-boundary", &workspace, AGENT_MIN_CONTEXT, home.clone());
            invalid.retain(|arg| arg != "--no-check");
            let output = run_harness(&binary, &workspace, &invalid);
            assert_eq!(output.status.code(), Some(2));
            assert!(String::from_utf8_lossy(&output.stderr).contains("macOS sandbox"));
        }
        // Agent journals land in the Vade home; the project keeps only its own .vade/check.
        assert_eq!(fs::read_dir(vade_home.join("projects")).unwrap().count(), 1);
        assert!(!workspace.join(".vade/sessions").exists() && !workspace.join(".vade/history").exists());
        // With history in the project, chat still reads the user MCP list from the Vade home.
        fs::write(vade_home.join("mcp.json"), r#"{"mcpServers":{"absent":{"command":"/nonexistent/mcp-server"}}}"#).unwrap();
        let state = state_arguments_for(Some("1".into()), || Ok(vade_home.clone()), true).unwrap();
        let mut args = arguments("http://127.0.0.1:0", "parser-boundary", &workspace, 262144, state);
        args.extend(["--prompt".into(), "hello".into()]);
        let output = run_harness(&binary, &workspace, &args);
        let diagnostics = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert_eq!(output.status.code(), Some(1), "{diagnostics}");
        assert!(diagnostics.contains("starting 1 MCP server"), "{diagnostics}");
        // vade sessions gets only the state arguments, which its parser takes.
        let mut args = vec!["sessions".to_string(), "--workspace".into(), workspace.to_string_lossy().into_owned()];
        args.extend(state_arguments_for(Some("1".into()), || Ok(vade_home.clone()), false).unwrap());
        let output = run_harness(&binary, &workspace, &args);
        let diagnostics = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert_eq!(output.status.code(), Some(0), "{diagnostics}");
        fs::remove_dir_all(workspace).unwrap();
        fs::remove_dir_all(vade_home).unwrap();
    }

    #[test]
    fn chat_state_defaults_to_the_vade_home_and_opts_into_the_project() {
        let root = || Ok(PathBuf::from("/home/owner/.vade"));
        for chat in [false, true] {
            assert_eq!(state_arguments_for(None, root, chat).unwrap(), ["--vade-home", "/home/owner/.vade"]);
            assert_eq!(state_arguments_for(Some("0".into()), root, chat).unwrap(), ["--vade-home", "/home/owner/.vade"]);
            assert_eq!(state_arguments_for(Some("1".into()), || Err("no home".into()), chat).unwrap(), ["--state-in-project"]);
            assert!(state_arguments_for(Some("yes".into()), root, chat).unwrap_err().contains("VADE_CHAT_IN_PROJECT"));
            assert!(state_arguments_for(None, || Err("Cannot find your home folder.".into()), chat).is_err());
        }
        // Only chat reads the MCP list; vade sessions takes no --mcp-home.
        assert_eq!(state_arguments_for(Some("1".into()), root, true).unwrap(), ["--state-in-project", "--mcp-home", "/home/owner/.vade"]);
        assert_eq!(state_arguments_for(Some("1".into()), root, false).unwrap(), ["--state-in-project"]);
        let args = arguments("http://127.0.0.1:18080", "m", Path::new("/w"), 4096, vec!["--state-in-project".into()]);
        assert_eq!(args.last().map(String::as_str), Some("--state-in-project"));
    }
}
