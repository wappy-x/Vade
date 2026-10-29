//! `vade mcp list|add|remove|test|trust` and `vade tools`.
use super::config::{self, ServerSpec, Source};
use super::{Hub, Server};
use crate::diagnostics::{print_stdout, stderr_line};
use crate::json::Value;
use crate::terminal::sanitize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

pub const MCP_HELP: &str = "\
Usage: vade mcp <command> [options]

Manage MCP (Model Context Protocol) servers for vade chat. Each server is a
program that speaks JSON-RPC on stdin and stdout. Its tools reach the model as
<server>__<tool>.

Commands:
  list                      Show the configured servers and where they come from.
  add NAME [options] -- COMMAND [ARG...]
                            Add a server to your user list.
      --project             Add it to .vade/mcp.json in this project instead.
      --env KEY=VALUE       Set an environment variable for the server (repeat).
      --cwd DIR             Run the server in DIR (relative to the project).
      --timeout-ms N        Time limit for one call, 1 to 600000 (default 60000).
      --disabled            Add the server but do not start it.
  remove NAME [--project]   Remove a server from your user list or the project list.
  test NAME                 Start the server, list its tools, resources and prompts, then stop it.
  trust NAME                Let a server from the project list start in chat.

Options:
  --workspace DIR           The project folder (default: the current folder).

Files: .vade/mcp.json in the project, and mcp.json in the Vade home
($VADE_HOME, else ~/.vade). A project entry overrides a user entry with the
same name. Set VADE_MCP=off to start no servers. See docs/mcp.md.
";

pub const TOOLS_HELP: &str = "\
Usage: vade tools [--workspace DIR]

List the tools a vade chat offers the model: the native workspace and memory
tools, then the tools of each configured MCP server. MCP servers start for the
listing and stop at the end. Set VADE_MCP=off to list native tools only, and
VADE_MEMORY=0 to leave out the memory tools.
";

fn fail(message: impl std::fmt::Display) -> ExitCode {
    stderr_line(format_args!("vade: {message}"));
    ExitCode::FAILURE
}
fn usage(message: impl std::fmt::Display) -> ExitCode {
    usage_for("mcp", message)
}
fn usage_for(command: &str, message: impl std::fmt::Display) -> ExitCode {
    stderr_line(format_args!("vade: {message}. Run vade {command} --help."));
    ExitCode::from(2)
}
/// The line before servers start, so that a slow server does not look like a hang.
pub fn starting(count: usize) {
    let plural = if count == 1 { "" } else { "s" };
    stderr_line(format_args!("vade: starting {count} MCP server{plural}"));
}

/// Split off `--workspace DIR` and resolve the project folder.
fn workspace(args: &mut Vec<String>) -> Result<PathBuf, String> {
    let mut root = None;
    // Words after `--` belong to the server command.
    while let Some(at) = args.iter().take_while(|arg| *arg != "--").position(|arg| arg == "--workspace") {
        if root.is_some() || at + 1 >= args.len() {
            return Err("give --workspace once, with a folder".into());
        }
        root = Some(PathBuf::from(args.remove(at + 1)));
        args.remove(at);
    }
    let root = match root {
        Some(root) => root,
        None => std::env::current_dir().map_err(|_| "the current folder is unavailable".to_owned())?,
    };
    root.canonicalize().map_err(|_| format!("cannot open the folder {}", root.display()))
}

pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let mut args: Vec<String> = args.into_iter().collect();
    if args.is_empty() || args.iter().take_while(|a| *a != "--").any(|a| a == "--help" || a == "-h") {
        return print_stdout(MCP_HELP);
    }
    let root = match workspace(&mut args) {
        Ok(root) => root,
        Err(error) => return usage(error),
    };
    if args.is_empty() {
        return usage("name an mcp command: list, add, remove, test or trust");
    }
    let command = args.remove(0);
    match (command.as_str(), args.as_slice()) {
        ("list", []) => list(&root),
        ("add", [_, ..]) => add(&root, &args),
        ("remove", [name]) => remove(&root, name, false),
        ("remove", [name, flag]) | ("remove", [flag, name]) if flag == "--project" => remove(&root, name, true),
        ("test", [name]) => test(&root, name),
        ("trust", [name]) => trust(&root, name),
        ("list" | "add" | "remove" | "test" | "trust", _) => usage(format!("wrong arguments for mcp {command}")),
        _ => usage(format!("unknown mcp command {:?}", sanitize(&command))),
    }
}

fn describe(spec: &ServerSpec) -> String {
    let state = if !spec.enabled {
        "disabled"
    } else if !spec.trusted {
        "not trusted"
    } else {
        "enabled"
    };
    let mut line = format!("{}  {}  {state}  {}", spec.name, spec.source.label(), sanitize(&spec.command));
    for arg in &spec.args {
        line.push(' ');
        line.push_str(&sanitize(arg));
    }
    // Values may be tokens; show the names only.
    if !spec.env.is_empty() {
        let names: Vec<_> = spec.env.iter().map(|(key, _)| sanitize(key)).collect();
        line.push_str(&format!("  env: {}", names.join(", ")));
    }
    if let Some(cwd) = &spec.cwd {
        line.push_str(&format!("  cwd: {}", sanitize(cwd)));
    }
    line
}

fn list(root: &Path) -> ExitCode {
    let loaded = config::load(root, None);
    for note in &loaded.notes {
        stderr_line(format_args!("vade: {note}"));
    }
    let user = config::user_path().map_or_else(|e| e.to_string(), |path| path.display().to_string());
    let mut out = format!("User list: {user}\nProject list: {}\n", config::project_path(root).display());
    if loaded.servers.is_empty() {
        out.push_str("No MCP servers. Add one with: vade mcp add NAME -- COMMAND [ARG...]\n");
    }
    for spec in &loaded.servers {
        out.push_str(&describe(spec));
        out.push('\n');
    }
    print_stdout(&out)
}

/// The file for `project` and its parsed contents. A broken file is never overwritten.
fn open(root: &Path, project: bool) -> Result<(PathBuf, Value), String> {
    let path = if project { config::project_path(root) } else { config::user_path().map_err(|e| e.to_string())? };
    let file = config::read_file(&path).map_err(|e| format!("{e}; fix or move the file first"))?;
    config::servers(&file).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((path, file))
}
/// The key that holds the server map in this file: `mcpServers` if the file uses it.
fn map_key(file: &Value) -> &'static str {
    if file.get("mcpServers").is_some() { "mcpServers" } else { "servers" }
}

fn add(root: &Path, rest: &[String]) -> ExitCode {
    let (mut project, mut env, mut cwd, mut timeout, mut enabled) = (false, Vec::new(), None, None, true);
    let mut name = None;
    let mut words = rest.iter();
    let mut command = Vec::new();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--project" => project = true,
            "--disabled" => enabled = false,
            "--env" | "--cwd" | "--timeout-ms" => {
                let Some(value) = words.next() else { return usage(format!("{word} needs a value")) };
                match word.as_str() {
                    "--env" => match value.split_once('=') {
                        Some((key, value)) if !key.is_empty() => env.push((key.to_owned(), Value::String(value.into()))),
                        _ => return usage("--env needs KEY=VALUE"),
                    },
                    "--cwd" => cwd = Some(value.clone()),
                    // Stored as a plain JSON number, so "0100" or "+100" cannot reach the file.
                    _ => match value.parse::<u64>() {
                        Ok(ms) if value.bytes().all(|b| b.is_ascii_digit()) => timeout = Some(ms.to_string()),
                        _ => return usage("--timeout-ms needs a whole number of milliseconds"),
                    },
                }
            }
            "--" => {
                command.extend(words.by_ref().cloned());
            }
            flag if flag.starts_with('-') => return usage(format!("unknown option {:?} for mcp add", sanitize(flag))),
            _ if name.is_none() => name = Some(word.as_str()),
            extra => return usage(format!("unexpected word {:?}; put the server command after --", sanitize(extra))),
        }
    }
    let Some(name) = name else { return usage("mcp add needs a server name") };
    let Some((program, arguments)) = command.split_first() else { return usage("mcp add needs a command after --") };
    let mut entry = vec![
        ("command".into(), Value::String(program.clone())),
        ("args".into(), Value::Array(arguments.iter().cloned().map(Value::String).collect())),
    ];
    if !env.is_empty() {
        entry.push(("env".into(), Value::Object(env)));
    }
    if let Some(cwd) = cwd {
        entry.push(("cwd".into(), Value::String(cwd)));
    }
    if let Some(timeout) = timeout {
        entry.push(("timeout_ms".into(), Value::Number(timeout)));
    }
    entry.push(("enabled".into(), Value::Bool(enabled)));
    let entry = Value::Object(entry);
    let source = if project { Source::Project } else { Source::User };
    if let Err(error) = config::parse_entry(name, &entry, source) {
        return usage(error);
    }
    // A project entry also writes a trust record; check that file first so a
    // broken user list stops the command before anything is written.
    if project && let Err(error) = open(root, false) {
        return fail(error);
    }
    let (path, mut file) = match open(root, project) {
        Ok(opened) => opened,
        Err(error) => return fail(error),
    };
    let key = map_key(&file);
    let mut map = config::servers(&file).map(<[_]>::to_vec).unwrap_or_default();
    if map.iter().any(|(old, _)| old == name) {
        return fail(format!("server {name} is already in {}; run vade mcp remove {name}{} first", path.display(), if project { " --project" } else { "" }));
    }
    map.push((name.into(), entry.clone()));
    config::set_member(&mut file, key, Value::Object(map));
    if let Err(error) = config::write_file(&path, &file) {
        return fail(format!("cannot write {}: {error}", path.display()));
    }
    // You added it yourself, so the project entry is trusted as written.
    if project && let Err(error) = record_trust(root, name, &entry) {
        return fail(error);
    }
    print_stdout(&format!("Added MCP server {name} to {}.\nCheck it with: vade mcp test {name}\n", path.display()))
}

fn remove(root: &Path, name: &str, project: bool) -> ExitCode {
    let (path, mut file) = match open(root, project) {
        Ok(opened) => opened,
        Err(error) => return fail(error),
    };
    let key = map_key(&file);
    let mut map = config::servers(&file).map(<[_]>::to_vec).unwrap_or_default();
    let before = map.len();
    map.retain(|(old, _)| old != name);
    if map.len() == before {
        let hint = if project { "" } else { "; for a project server add --project" };
        return fail(format!("server {:?} is not in {}{hint}", sanitize(name), path.display()));
    }
    config::set_member(&mut file, key, Value::Object(map));
    if let Err(error) = config::write_file(&path, &file) {
        return fail(format!("cannot write {}: {error}", path.display()));
    }
    if project && let Err(error) = edit_trust(root, name, None) {
        return fail(error);
    }
    print_stdout(&format!("Removed MCP server {name} from {}.\n", path.display()))
}

/// Replace this project's trust record for `name` with `entry`, or drop it.
fn edit_trust(root: &Path, name: &str, entry: Option<&Value>) -> Result<(), String> {
    let (path, mut user) = open(root, false)?;
    let project = root.to_string_lossy();
    let mut records = user.get("trusted").and_then(Value::as_array).map(<[_]>::to_vec).unwrap_or_default();
    let before = records.len();
    records.retain(|record| {
        record.get("project").and_then(Value::as_str) != Some(&project) || record.get("server").and_then(Value::as_str) != Some(name)
    });
    if entry.is_none() && records.len() == before {
        return Ok(());
    }
    records.extend(entry.map(|entry| Value::Object(vec![
        ("project".into(), Value::String(project.into_owned())),
        ("server".into(), Value::String(name.into())),
        ("entry".into(), entry.clone()),
    ])));
    config::set_member(&mut user, "trusted", Value::Array(records));
    config::write_file(&path, &user).map_err(|error| format!("cannot write {}: {error}", path.display()))
}
fn record_trust(root: &Path, name: &str, entry: &Value) -> Result<(), String> {
    edit_trust(root, name, Some(entry))
}

fn find(root: &Path, name: &str) -> Result<ServerSpec, ExitCode> {
    let loaded = config::load(root, None);
    for note in &loaded.notes {
        stderr_line(format_args!("vade: {note}"));
    }
    loaded.servers.into_iter().find(|spec| spec.name == name).ok_or_else(|| fail(format!("no MCP server named {:?}; run vade mcp list", sanitize(name))))
}

fn trust(root: &Path, name: &str) -> ExitCode {
    let spec = match find(root, name) {
        Ok(spec) => spec,
        Err(code) => return code,
    };
    if spec.source != Source::Project {
        return print_stdout(&format!("MCP server {name} is in your user list; it needs no trust.\n"));
    }
    match record_trust(root, name, &spec.entry) {
        Ok(()) => print_stdout(&format!(
            "Trusted project MCP server {name}:\n  {}\nIf .vade/mcp.json changes this entry, run vade mcp trust {name} again.\n",
            describe(&ServerSpec { trusted: true, ..spec })
        )),
        Err(error) => fail(error),
    }
}

fn names(items: &[Value]) -> String {
    items
        .iter()
        .map(|item| sanitize(item.get("uri").or_else(|| item.get("name")).and_then(Value::as_str).unwrap_or("?")))
        .collect::<Vec<_>>()
        .join(", ")
}

fn test(root: &Path, name: &str) -> ExitCode {
    let spec = match find(root, name) {
        Ok(spec) => spec,
        Err(code) => return code,
    };
    if !spec.trusted {
        stderr_line(format_args!("vade: project server {name} is not trusted; this test runs it once because you asked"));
    }
    let started = std::time::Instant::now();
    let hub = Hub::start(vec![ServerSpec { enabled: true, trusted: true, ..spec }], root);
    for note in &hub.notes {
        stderr_line(format_args!("vade: {note}"));
    }
    let Server { connection, tools, resources, prompts, error, .. } = &hub.servers[0];
    let Some(connection) = connection else {
        return fail(format!("MCP server {name} failed: {}", error.as_deref().unwrap_or("unknown error")));
    };
    let info = &connection.info;
    let mut out = format!(
        "MCP server {name}: started and listed in {} ms\n  Server: {} {}, protocol {}\n  Tools ({}):",
        started.elapsed().as_millis(),
        sanitize(&info.name),
        sanitize(&info.version),
        sanitize(&info.protocol),
        tools.len()
    );
    for tool in tools {
        out.push_str(&format!("\n    {}", tool.exposed));
    }
    out.push_str(&format!("\n  Resources ({}, listed only): {}\n  Prompts ({}, listed only): {}\n", resources.len(), names(resources), prompts.len(), names(prompts)));
    print_stdout(&out)
}

fn summary(description: &str) -> String {
    let text = sanitize(description).replace('\n', " ");
    let first = text.split_once(". ").map_or(text.as_str(), |(first, _)| first);
    let mut short: String = first.chars().take(100).collect();
    if short.len() < first.len() {
        short.push_str("...");
    }
    short
}

pub fn tools_cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let mut args: Vec<String> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return print_stdout(TOOLS_HELP);
    }
    let root = match workspace(&mut args) {
        Ok(root) => root,
        Err(error) => return usage_for("tools", error),
    };
    if let Some(extra) = args.first() {
        return usage_for("tools", format!("unknown argument {:?} for tools", sanitize(extra)));
    }
    let mut native = crate::coding_tools::definitions();
    // Agent chat adds the project memory tools unless VADE_MEMORY=0.
    if crate::memory::enabled(false, std::env::var("VADE_MEMORY").ok().as_deref()) {
        crate::memory::add_tools(&mut native);
    }
    let native = native.as_array().unwrap_or_default();
    let mut out = format!("Native tools ({}):\n", native.len());
    for tool in native {
        let function = tool.get("function");
        let field = |key| function.and_then(|f| f.get(key)).and_then(Value::as_str).unwrap_or_default();
        out.push_str(&format!("  {}  {}\n", field("name"), summary(field("description"))));
    }
    let hub = Hub::from_config(&root, None, starting);
    for note in &hub.notes {
        stderr_line(format_args!("vade: {note}"));
    }
    let count: usize = hub.servers.iter().map(|server| server.tools.len()).sum();
    out.push_str(&format!("MCP tools ({count}):\n"));
    for server in &hub.servers {
        for tool in &server.tools {
            out.push_str(&format!("  {}  {}\n", tool.exposed, summary(&tool.description)));
        }
    }
    if hub.servers.is_empty() {
        out.push_str("  No MCP servers are configured. See vade mcp --help.\n");
    }
    print_stdout(&out)
}
