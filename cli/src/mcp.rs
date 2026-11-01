//! `vade tools` and `vade mcp`: the installed harness owns the MCP client, so
//! these commands run it with the same arguments.
use std::path::PathBuf;

/// The harness arguments for `args`, or `None` when `args` is another command.
/// `vade help mcp` becomes `mcp --help`.
pub fn arguments(args: &[String]) -> Option<Vec<String>> {
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["tools" | "mcp", ..] => Some(args.to_vec()),
        ["help" | "--help" | "-h", command @ ("tools" | "mcp")] => Some(vec![(*command).into(), "--help".into()]),
        _ => None,
    }
}

/// The installed `vade-harness`: the local install first (unless VADE_CONFIG
/// selects the controller), then the controller's staged copy.
fn harness() -> Result<PathBuf, String> {
    if std::env::var_os("VADE_CONFIG").is_none() && crate::launch::owner_configured()
        && let Some(path) = crate::launch::installed_harness()?
    {
        return Ok(path);
    }
    if crate::config::config_path().is_ok_and(|path| path.is_file()) {
        // The same generation and hash check as vade chat on a controller.
        let mut config = crate::config::Config::load()?;
        config.resolve_generation()?;
        let staged = config.stage_dir.join("vade-harness");
        if std::fs::symlink_metadata(&staged).is_ok_and(|meta| meta.is_file()) {
            crate::text::reject_symlinks(&staged)?;
            let expected = crate::state::Manifest::load(&config.build_manifest)?
                .harness_sha256
                .ok_or("The staged harness has no saved hash. Run vade install first.")?;
            let actual = std::fs::File::open(&staged)
                .and_then(crate::sha256::digest_reader)
                .map_err(|_| "Cannot read the staged harness. Run vade install again.")?;
            if actual != expected {
                return Err("The staged harness differs from its saved hash. Run vade install again.".into());
            }
            return Ok(staged);
        }
    }
    Err("The chat harness is not installed. Run vade install or vade launch, then try again.".into())
}

/// The help page that `args` asks for, by the rule the harness uses. Help needs
/// no install: these pages are compiled into `vade` too.
fn help(args: &[String]) -> Option<&'static str> {
    use vade_harness::mcp::cli::{MCP_HELP, TOOLS_HELP};
    let (command, rest) = args.split_first()?;
    let asked = |arg: &String| arg == "--help" || arg == "-h";
    match command.as_str() {
        "mcp" if rest.is_empty() || rest.iter().take_while(|arg| *arg != "--").any(asked) => Some(MCP_HELP),
        "tools" if rest.iter().any(asked) => Some(TOOLS_HELP),
        _ => None,
    }
}

/// Print the help page, or run the harness and return its exit code.
pub fn run(args: Vec<String>) -> Result<i32, String> {
    if let Some(page) = help(&args) {
        crate::output::text(format_args!("{page}"))?;
        return Ok(0);
    }
    let harness = harness()?;
    let status = std::process::Command::new(&harness)
        .args(args)
        .status()
        .map_err(|_| format!("Cannot run the installed harness {}. Run vade install again.", crate::text::quoted(&harness.to_string_lossy())))?;
    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_tools_and_mcp_are_forwarded() {
        let owned = |args: &[&str]| args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>();
        assert_eq!(super::arguments(&owned(&["mcp", "add", "x", "--", "srv"])), Some(owned(&["mcp", "add", "x", "--", "srv"])));
        assert_eq!(super::arguments(&owned(&["tools"])), Some(owned(&["tools"])));
        assert_eq!(super::arguments(&owned(&["help", "mcp"])), Some(owned(&["mcp", "--help"])));
        assert_eq!(super::arguments(&owned(&["--help", "tools"])), Some(owned(&["tools", "--help"])));
        for other in [&["chat"][..], &["help"], &["help", "chat"], &[]] {
            assert_eq!(super::arguments(&owned(other)), None);
        }
    }
    #[test]
    fn help_needs_no_harness_and_a_real_run_does() {
        use vade_harness::mcp::cli::{MCP_HELP, TOOLS_HELP};
        let help = |args: &[&str]| super::help(&args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>());
        for args in [&["mcp"][..], &["mcp", "--help"], &["mcp", "-h"], &["mcp", "add", "--help"]] {
            assert_eq!(help(args), Some(MCP_HELP), "{args:?}");
        }
        for args in [&["tools", "--help"][..], &["tools", "-h"]] {
            assert_eq!(help(args), Some(TOOLS_HELP), "{args:?}");
        }
        // After -- the word belongs to the server command, as in the harness.
        for args in [&["tools"][..], &["mcp", "list"], &["mcp", "add", "x", "--", "srv", "--help"]] {
            assert_eq!(help(args), None, "{args:?}");
        }
    }
}
