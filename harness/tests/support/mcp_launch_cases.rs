// Included by the MCP test runner after its shared fixture helpers.
#[cfg(unix)]
fn trusted_server_reads_its_neighbor_in_place() {
    use std::os::unix::fs::PermissionsExt;
    let home = Temp::new("neighbor-home");
    let project = Temp::new("neighbor-project");
    let p = &project.0;
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::create_dir_all(p.join("server")).unwrap();
    std::fs::write(p.join("server/start.sh"), "#!/bin/sh\nexec /bin/sh \"$(dirname \"$0\")/neighbor.sh\"\n").unwrap();
    std::fs::set_permissions(p.join("server/start.sh"), std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(p.join("server/neighbor.sh"), format!("test \"$PWD\" = '{}' || exit 2\nexport FAKE_MCP=normal\nexec '{}'\n", p.join("server").display(), exe())).unwrap();
    std::fs::write(p.join(".vade/mcp.json"), r#"{"servers":{"neighbor":{"command":"./start.sh","cwd":"server"}}}"#).unwrap();
    assert!(harness(&home.0, p, &["mcp", "trust", "neighbor"], None).0);
    let (ok, out, err) = harness(&home.0, p, &["mcp", "test", "neighbor"], None);
    assert!(ok && out.contains("neighbor__echo"), "{out}{err}");
}

#[cfg(unix)]
fn trusted_interpreter_keeps_its_virtual_environment_path() {
    let home = Temp::new("venv-home");
    let project = Temp::new("venv-project");
    let p = &project.0;
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::create_dir_all(p.join("venv/bin")).unwrap();
    let interpreter = p.join("venv/bin/python");
    std::os::unix::fs::symlink(exe(), &interpreter).unwrap();
    std::fs::write(p.join("server.py"), "# fixture script\n").unwrap();
    std::fs::write(p.join(".vade/mcp.json"), format!(r#"{{"servers":{{"venv":{{"command":"venv/bin/python","args":["server.py"],"env":{{"FAKE_MCP":"venv-path","FAKE_ARGV0":{}}}}}}}}}"#, json::quote(&interpreter.to_string_lossy()))).unwrap();
    assert!(harness(&home.0, p, &["mcp", "trust", "venv"], None).0);
    let (ok, out, err) = harness(&home.0, p, &["mcp", "test", "venv"], None);
    assert!(ok && out.contains("venv__echo"), "{out}{err}");
}

#[cfg(unix)]
fn trusted_data_changes_and_writes_stay_in_place() {
    let home = Temp::new("data-home");
    let project = Temp::new("data-project");
    let p = &project.0;
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::write(p.join("server.sh"), format!("printf 'written' >> \"$1\"\nexport FAKE_MCP=normal\nexec '{}'\n", exe())).unwrap();
    std::fs::write(p.join("data.txt"), "old\n").unwrap();
    std::fs::write(p.join(".vade/mcp.json"), r#"{"servers":{"data":{"command":"/bin/sh","args":["server.sh","data.txt"]}}}"#).unwrap();
    // The script is listed; the data file is not checked, so it is not listed.
    let checked = format!("\n  program: /bin/sh\n  script: {}\n  {NOT_CHECKED}\n", p.join("server.sh").display());
    let (ok, out, err) = harness(&home.0, p, &["mcp", "trust", "data"], None);
    assert!(ok && out.contains(&checked) && !out.contains("data path"), "{out}{err}");
    let (ok, out, err) = harness(&home.0, p, &["mcp", "list"], None);
    assert!(ok && out.contains(&checked) && !out.contains("data path"), "{out}{err}");
    std::fs::write(p.join("data.txt"), "changed\n").unwrap();
    let (ok, out, err) = harness(&home.0, p, &["mcp", "test", "data"], None);
    assert!(ok && out.contains("data__echo"), "{out}{err}");
    assert_eq!(std::fs::read_to_string(p.join("data.txt")).unwrap(), "changed\nwritten");
    let (ok, out, err) = harness(&home.0, p, &["mcp", "test", "data"], None);
    assert!(ok, "{out}{err}");
    assert_eq!(std::fs::read_to_string(p.join("data.txt")).unwrap(), "changed\nwrittenwritten");
}

fn untrusted_mcp_test_gives_the_exact_trust_command() {
    let home = Temp::new("untrusted-test-home");
    let project = Temp::new("untrusted-test-project");
    let p = &project.0;
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::write(p.join(".vade/mcp.json"), format!(r#"{{"servers":{{"unchecked":{{"command":{},"env":{{"FAKE_MCP":"normal","FAKE_MCP_LOG":{}}}}}}}}}"#, json::quote(&exe()), json::quote(&p.join("started").to_string_lossy()))).unwrap();
    let (ok, out, err) = harness(&home.0, p, &["mcp", "test", "unchecked"], None);
    assert!(!ok && out.is_empty() && err.contains("is not trusted yet") && err.contains("vade mcp trust unchecked"), "{out}{err}");
    assert!(!err.contains("trust was not checked") && !p.join("started").exists(), "{err}");
}

/// The last line of every list of checked files, in `vade mcp trust` and `vade mcp list`.
#[cfg(unix)]
const NOT_CHECKED: &str = "Not checked: files and folders that other arguments name, and any file the program loads after it starts.";
/// The line in place of script lines for a command whose code is not bound.
#[cfg(unix)]
const CODE_NOT_CHECKED: &str = "code: not checked. This command loads its code when it starts. Your yes covers this exact command line and the program file only.";

/// A scratch home and project with fixture programs. Each name in `programs`
/// is a link in `<project>/bin` to this test program, which is the fake
/// server whatever its name and arguments are. No real interpreter runs.
#[cfg(unix)]
struct Launch {
    home: Temp,
    project: Temp,
}
#[cfg(unix)]
impl Launch {
    fn new(tag: &str, programs: &[&str]) -> Self {
        let (home, project) = (Temp::new(&format!("{tag}-home")), Temp::new(&format!("{tag}-project")));
        std::fs::create_dir_all(project.0.join(".vade")).unwrap();
        std::fs::create_dir_all(project.0.join("bin")).unwrap();
        for program in programs {
            std::os::unix::fs::symlink(exe(), project.0.join("bin").join(program)).unwrap();
        }
        Self { home, project }
    }
    fn log(&self) -> PathBuf {
        self.home.0.join("starts.log")
    }
    /// Write the project list with this one server, and return its entry.
    /// The entry's own PATH finds a bare command name in the fixture folder.
    fn serve(&self, name: &str, command: &str, args: &[&str]) -> String {
        let args: Vec<String> = args.iter().map(|arg| json::quote(arg)).collect();
        let entry = format!(
            r#"{{"command":{},"args":[{}],"env":{{"PATH":{},"FAKE_MCP":"normal","FAKE_MCP_LOG":{}}}}}"#,
            json::quote(command),
            args.join(","),
            json::quote(&self.project.0.join("bin").to_string_lossy()),
            json::quote(&self.log().to_string_lossy()),
        );
        std::fs::write(self.project.0.join(".vade/mcp.json"), format!(r#"{{"servers":{{"{name}":{entry}}}}}"#)).unwrap();
        entry
    }
    /// A yes as a released Vade recorded it: the project, the server and the entry only.
    fn old_record(&self, name: &str, entry: &str) {
        let project = json::quote(&self.project.0.to_string_lossy());
        std::fs::write(self.home.0.join("mcp.json"), format!(r#"{{"trusted":[{{"project":{project},"server":"{name}","entry":{entry}}}]}}"#)).unwrap();
    }
    fn run(&self, args: &[&str]) -> (bool, String, String) {
        harness(&self.home.0, &self.project.0, args, None)
    }
    /// How many times a server of this project started.
    fn starts(&self) -> usize {
        log_lines(&self.log()).iter().filter(|line| *line == "initialize").count()
    }
    /// Trust `name`, start it, change `code`, and start again: the second start is refused.
    fn changed_code_is_refused(&self, name: &str, code: &Path, case: &str) {
        let (ok, out, err) = self.run(&["mcp", "trust", name]);
        assert!(ok && out.contains(&format!("\n  script: {}\n  {NOT_CHECKED}\n", code.display())) && !out.contains(CODE_NOT_CHECKED), "{case}: {out}{err}");
        let (ok, out, err) = self.run(&["tools"]);
        assert!(ok && out.contains(&format!("{name}__echo")) && self.starts() == 1, "{case}: {out}{err}");
        std::fs::write(code, "print('changed after the yes')\n").unwrap();
        let (ok, out, err) = self.run(&["tools"]);
        assert!(ok && !out.contains(&format!("{name}__echo")) && self.starts() == 1, "{case}: the changed code ran: {out}{err}");
        assert!(err.contains(&format!("MCP: project server {name} did not start because a file selected by its command arguments changed. Project MCP programs run with your file and network access. Check the argument files, then run vade mcp trust {name}")), "{case}: {err}");
    }
}

/// `python3 server`: no extension and no `#!` line. Only the interpreter's
/// name says that the file is code.
#[cfg(unix)]
fn interpreter_code_without_a_script_name_is_bound() {
    for interpreter in ["python3", "python3.12", "node22"] {
        let launch = Launch::new(&format!("code-{interpreter}"), &[interpreter]);
        let code = launch.project.0.join("server");
        std::fs::write(&code, "print('first')\n").unwrap();
        launch.serve("code", interpreter, &["server"]);
        launch.changed_code_is_refused("code", &code, interpreter);
    }
}

/// `Server.PY`, and a link named `tool` to `impl.py`. `launch-server` is no
/// interpreter name, so there only the file's own name, or the name it
/// leads to, says that it is code.
#[cfg(unix)]
fn script_names_bind_code_in_any_letter_case_and_through_links() {
    for (tag, program, argument, code) in [
        ("upper", "python3", "Server.PY", "Server.PY"),
        ("link", "python3", "tool", "impl.py"),
        ("plain-upper", "launch-server", "Server.PY", "Server.PY"),
        ("plain-link", "launch-server", "tool", "impl.py"),
    ] {
        let launch = Launch::new(&format!("names-{tag}"), &[program]);
        let p = &launch.project.0;
        std::fs::write(p.join(code), "print('first')\n").unwrap();
        if argument != code {
            std::os::unix::fs::symlink(p.join(code), p.join(argument)).unwrap();
        }
        launch.serve("named", program, &[argument]);
        launch.changed_code_is_refused("named", &p.join(code), tag);
    }
}

#[cfg(unix)]
fn env_commands_are_refused_at_trust_time_and_at_start() {
    let reason = "its command is env, which starts another program that Vade cannot check. Name that program as the command, then run vade mcp";
    for (tag, command, args) in [("bare", "env", &["python3", "-m", "pkg"][..]), ("absolute", "/usr/bin/env", &["node", "x.js"])] {
        // The fixture `env`, `python3` and `node` are all the fake server, so a start of any of them reaches the log.
        let launch = Launch::new(&format!("env-{tag}"), &["env", "python3", "node"]);
        std::fs::write(launch.project.0.join("x.js"), "// server code\n").unwrap();
        let entry = launch.serve("wrapped", command, args);
        // At start, with no record.
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && !out.contains("wrapped__") && err.contains(&format!("vade: MCP: project server wrapped did not start because {reason} trust wrapped\n")), "{command}: {out}{err}");
        // At trust time. Nothing is written.
        let (ok, out, err) = launch.run(&["mcp", "trust", "wrapped"]);
        assert!(!ok && out.is_empty() && err.contains(&format!("vade: cannot trust project MCP server wrapped because {reason} trust wrapped\n")), "{command}: {out}{err}");
        assert!(!launch.home.0.join("mcp.json").exists(), "{command}");
        // A yes that an earlier build recorded for this entry starts nothing either.
        launch.old_record("wrapped", &entry);
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && !out.contains("wrapped__") && err.contains(&format!("vade: MCP: project server wrapped did not start because {reason} trust wrapped\n")), "{command}: {out}{err}");
        let (ok, out, err) = launch.run(&["mcp", "test", "wrapped"]);
        assert!(!ok && out.is_empty() && err.contains(&format!("vade: project server wrapped did not start because {reason} trust wrapped\n")), "{command}: {out}{err}");
        // Adding such an entry yourself is refused the same way.
        let (ok, out, err) = launch.run(&["mcp", "add", "--project", "mine", "--", command, "node", "x.js"]);
        assert!(!ok && out.is_empty() && err.contains(&format!("vade: cannot trust project MCP server mine because {reason} add --project again\n")), "{command}: {out}{err}");
        assert_eq!(launch.starts(), 0, "{command}: a program started");
    }
}

/// `npx -y pkg`, `python3 -m pkg`, `node .` and `sh -c "..."` name no code
/// file. A yes covers the command line and the program file, and says so.
#[cfg(unix)]
fn commands_that_load_their_code_are_trusted_and_say_what_is_not_checked() {
    let inline = format!("exec '{}'", exe());
    for (tag, command, args) in [
        ("npx", "npx", &["-y", "pkg"][..]),
        ("module", "python3", &["-m", "pkg"]),
        ("folder", "node", &["."]),
        // The real shell runs the inline command, which starts the fake server.
        ("inline", "/bin/sh", &["-c", &inline]),
    ] {
        let launch = Launch::new(&format!("unbound-{tag}"), &["npx", "python3", "node"]);
        launch.serve("loads", command, args);
        let said = format!("\n  {CODE_NOT_CHECKED}\n  {NOT_CHECKED}\n");
        let (ok, out, err) = launch.run(&["mcp", "trust", "loads"]);
        assert!(ok && err.is_empty() && out.contains("\n  program: ") && out.contains(&said) && !out.contains("script: "), "{tag}: {out}{err}");
        let (ok, out, err) = launch.run(&["mcp", "list"]);
        assert!(ok && out.contains("loads  project  enabled") && out.contains(&said), "{tag}: {out}{err}");
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && out.contains("loads__echo") && launch.starts() == 1, "{tag}: {out}{err}");
        // One changed character of the command line asks again.
        let mut changed: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
        let last = changed.last_mut().unwrap();
        let end = last.pop().unwrap();
        last.push(if end == 'x' { 'y' } else { 'x' });
        launch.serve("loads", command, &changed.iter().map(String::as_str).collect::<Vec<_>>());
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && !out.contains("loads__echo") && launch.starts() == 1, "{tag}: {out}{err}");
        assert!(err.contains("MCP: project server loads is not trusted because its configuration changed. Project MCP programs run with your file and network access. Check the command and program, then run vade mcp trust loads"), "{tag}: {err}");
    }
}

#[cfg(unix)]
fn argument_files_that_appear_or_change_after_trust_keep_trust() {
    let launch = Launch::new("later-files", &["launch-server", "python3"]);
    let p = &launch.project.0;
    let trust = |checked: &str| {
        let (ok, out, err) = launch.run(&["mcp", "trust", "later"]);
        assert!(ok && out.contains(checked) && !out.contains("data path"), "{checked}: {out}{err}");
    };
    let starts = |count: usize| {
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && out.contains("later__echo") && !err.contains("trust") && launch.starts() == count, "start {count}: {out}{err}");
    };
    // `state.db` does not exist at the yes; `notes.txt` is data.
    std::fs::write(p.join("notes.txt"), "old\n").unwrap();
    launch.serve("later", "launch-server", &["--state", "state.db", "notes.txt"]);
    trust(&format!("\n  program: {}\n  {NOT_CHECKED}\nYour yes covers this exact config entry, and the program's path and bytes.\n", p.join("bin/launch-server").display()));
    starts(1);
    std::fs::write(p.join("state.db"), "made by the server\n").unwrap();
    starts(2);
    std::fs::write(p.join("notes.txt"), "new\n").unwrap();
    std::fs::write(p.join("state.db"), "changed by the server\n").unwrap();
    starts(3);
    // With an interpreter, the first file argument is its code. A file that appears after it is not.
    std::fs::write(p.join("server.py"), "print('first')\n").unwrap();
    launch.serve("later", "python3", &["server.py", "out.log"]);
    trust(&format!("\n  script: {}\n  {NOT_CHECKED}\nYour yes covers this exact config entry, the program's path and bytes, and each script's path and bytes.\n", p.join("server.py").display()));
    starts(4);
    std::fs::write(p.join("out.log"), "made by the server\n").unwrap();
    starts(5);
    // Intended: a file that appears where the interpreter would take its code changes what is checked.
    launch.serve("later", "python3", &["-m", "pkg", "made-later"]);
    trust(&format!("\n  {CODE_NOT_CHECKED}\n"));
    starts(6);
    std::fs::write(p.join("made-later"), "print('new code')\n").unwrap();
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && !out.contains("later__echo") && launch.starts() == 6, "{out}{err}");
    assert!(err.contains("did not start because a file selected by its command arguments changed"), "{err}");
}

#[cfg(unix)]
fn trust_saves_the_program_stamp_and_a_start_does_not_write_the_record() {
    use std::os::unix::fs::MetadataExt;
    let launch = Launch::new("stamp", &["launch-server"]);
    launch.serve("stamped", "launch-server", &["stdio"]);
    assert!(launch.run(&["mcp", "trust", "stamped"]).0);
    let record = launch.home.0.join("mcp.json");
    let saved = std::fs::read_to_string(&record).unwrap();
    // The stamp is the program file's own: the file that the fixture link leads to.
    let program = std::fs::metadata(exe()).unwrap();
    let ns = |seconds: i64, nanos: i64| i128::from(seconds) * 1_000_000_000 + i128::from(nanos);
    for (key, number) in [
        ("device", program.dev().to_string()),
        ("inode", program.ino().to_string()),
        ("size", program.size().to_string()),
        ("modified_ns", ns(program.mtime(), program.mtime_nsec()).to_string()),
        ("changed_ns", ns(program.ctime(), program.ctime_nsec()).to_string()),
    ] {
        assert!(saved.contains(&format!("\"{key}\": {number}")), "{key} {number}: {saved}");
    }
    let written = std::fs::metadata(&record).unwrap();
    for start in 1..=2 {
        let (ok, out, err) = launch.run(&["tools"]);
        assert!(ok && out.contains("stamped__echo") && launch.starts() == start, "{out}{err}");
    }
    let after = std::fs::metadata(&record).unwrap();
    assert_eq!(std::fs::read_to_string(&record).unwrap(), saved);
    assert_eq!((after.ino(), after.mtime(), after.mtime_nsec()), (written.ino(), written.mtime(), written.mtime_nsec()), "a check wrote the trust file");
}

#[cfg(unix)]
fn a_record_from_an_older_vade_asks_for_a_new_yes_without_calling_it_changed() {
    let launch = Launch::new("old-record", &["launch-server"]);
    let entry = launch.serve("old", "launch-server", &["stdio"]);
    launch.old_record("old", &entry);
    let said = "project server old did not start because its trust record is from an older Vade and names no program. Check the command, then run vade mcp trust old\n";
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && !out.contains("old__echo") && err.contains(&format!("vade: MCP: {said}")) && !err.contains("changed"), "{out}{err}");
    let (ok, out, err) = launch.run(&["mcp", "test", "old"]);
    assert!(!ok && out.is_empty() && err.contains(&format!("vade: {said}")) && !err.contains("changed"), "{out}{err}");
    assert_eq!(launch.starts(), 0);
    let (ok, out, err) = launch.run(&["mcp", "trust", "old"]);
    assert!(ok, "{out}{err}");
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && out.contains("old__echo") && launch.starts() == 1 && !err.contains("trust"), "{out}{err}");
}

/// With an interpreter, a decoy file before the code (`-W ignore`) does not
/// leave the code unbound: every later file argument is bound too, except one
/// with a data extension.
#[cfg(unix)]
fn interpreter_binds_every_file_argument_except_later_data() {
    // The decoy `ignore` is there at the yes: `server` is still bound.
    let launch = Launch::new("decoy", &["python3"]);
    let p = &launch.project.0;
    std::fs::write(p.join("ignore"), "decoy\n").unwrap();
    std::fs::write(p.join("server"), "print('first')\n").unwrap();
    launch.serve("decoy", "python3", &["-W", "ignore", "server"]);
    let (ok, out, err) = launch.run(&["mcp", "trust", "decoy"]);
    assert!(ok && out.contains(&format!("\n  script: {}\n  script: {}\n", p.join("ignore").display(), p.join("server").display())), "{out}{err}");
    launch.changed_code_is_refused("decoy", &p.join("server"), "decoy");

    // A later data file may change; a later file without a data extension may not.
    let launch = Launch::new("later-data", &["python3"]);
    let p = &launch.project.0;
    std::fs::write(p.join("server.py"), "print('first')\n").unwrap();
    std::fs::write(p.join("data.sqlite"), "rows\n").unwrap();
    launch.serve("data", "python3", &["server.py", "--db", "data.sqlite"]);
    let (ok, out, err) = launch.run(&["mcp", "trust", "data"]);
    assert!(ok && out.contains(&format!("\n  script: {}\n  {NOT_CHECKED}\n", p.join("server.py").display())) && out.matches("script: ").count() == 1, "{out}{err}");
    std::fs::write(p.join("data.sqlite"), "changed rows\n").unwrap();
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && out.contains("data__echo") && launch.starts() == 1 && !err.contains("trust"), "{out}{err}");
    let launch = Launch::new("later-helper", &["python3"]);
    let p = &launch.project.0;
    std::fs::write(p.join("server.py"), "print('first')\n").unwrap();
    std::fs::write(p.join("helper"), "print('helper')\n").unwrap();
    launch.serve("helper", "python3", &["server.py", "helper"]);
    launch.changed_code_is_refused("helper", &p.join("helper"), "helper");

    // A decoy that appears after the yes changes which files are bound, so the start is refused.
    let launch = Launch::new("decoy-later", &["python3"]);
    let p = &launch.project.0;
    std::fs::write(p.join("server"), "print('first')\n").unwrap();
    launch.serve("decoy", "python3", &["-W", "ignore", "server"]);
    let (ok, out, err) = launch.run(&["mcp", "trust", "decoy"]);
    assert!(ok && out.contains(&format!("\n  program: {}\n  script: {}\n  {NOT_CHECKED}\n", p.join("bin/python3").display(), p.join("server").display())), "{out}{err}");
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && out.contains("decoy__echo") && launch.starts() == 1, "{out}{err}");
    std::fs::write(p.join("ignore"), "decoy\n").unwrap();
    let (ok, out, err) = launch.run(&["tools"]);
    assert!(ok && !out.contains("decoy__echo") && launch.starts() == 1, "{out}{err}");
    assert!(err.contains("MCP: project server decoy did not start because a file selected by its command arguments changed. Project MCP programs run with your file and network access. Check the argument files, then run vade mcp trust decoy"), "{err}");
}

/// `vade mcp add --project` and `vade mcp test` print the same checked lines
/// as `vade mcp trust`, before anything starts.
#[cfg(unix)]
fn add_and_test_say_what_is_checked() {
    let launch = Launch::new("say-checked", &["npx"]);
    let p = &launch.project.0;
    let said = format!("\n  program: {}\n  {CODE_NOT_CHECKED}\n  {NOT_CHECKED}\n", p.join("bin/npx").display());
    let path = format!("PATH={}", p.join("bin").display());
    let (ok, out, err) = launch.run(&["mcp", "add", "--project", "added", "--env", &path, "--", "npx", "-y", "pkg"]);
    assert!(ok && out.contains(&said) && out.contains("Trusted it as written. Checked before each start:\n"), "{out}{err}");
    assert_eq!(launch.starts(), 0);
    launch.serve("tested", "npx", &["-y", "pkg"]);
    assert!(launch.run(&["mcp", "trust", "tested"]).0);
    let (ok, out, err) = launch.run(&["mcp", "test", "tested"]);
    let (lines, started) = (out.find(&said), out.find("started and listed"));
    assert!(ok && out.starts_with("Project MCP server tested. Checked before each start:\n") && lines.is_some() && lines < started, "{out}{err}");
    assert_eq!(launch.starts(), 1);
}
