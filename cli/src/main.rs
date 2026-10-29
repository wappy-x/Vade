//! The `vade` command line: start, chat with, stop and check a Vade engine.
//!
//! `main` dispatches every command to one of three lifecycle owners, chosen by
//! what is configured (see `vade --help`):
//!
//! - [`launch`]: the owner flow on this Spark (`vade launch`, local defaults in
//!   `~/.vade` or `VADE_HOME`).
//! - [`app`]: the controller, which drives one or more Sparks over SSH from a
//!   config file (`VADE_CONFIG` or the user config directory).
//! - [`run`]: the separate legacy bootstrap for the official Qwen3-8B BF16
//!   catalog (`vade run qwen3-8b`).
//!
//! Module map:
//!
//! | Area | Modules |
//! | --- | --- |
//! | Arguments and help | [`cli`], [`help`] (help pages), [`auth`] (credential files), [`endpoint`] (which server a client command talks to) |
//! | Lifecycle owners | [`launch`], [`app`], [`run`], [`worker`] (the private helper that owns engine processes on a Spark) |
//! | Configuration and records | [`config`] (controller config), [`state`] (controller ownership records and locks), [`catalog`], [`repository`] |
//! | Machine and artifacts | [`detect`], [`prebuilt`], [`build_job`], [`sha256`] |
//! | Processes and files | [`runner`] (every subprocess), [`control_socket`], [`unix_file`], [`unix_flags`] |
//! | Output | [`output`] (fallible stdout), [`style`] (color, errors, tables and help layout), [`status`] (`status --json` schema), [`text`] (terminal-safe text and `key=value` records), [`json`] (health replies) |
//! | Client tools | [`bench`](mod@bench), [`verify`] (exact token IDs, drafting ON and OFF), [`pull`] (`vade pull` and `vade models`: Hugging Face downloads) |
//! | Local checks | [`test_gate`] (`vade test build` and `determinism`), [`evals`] (long-context, GDN and ABI checks) |
mod output;
mod prebuilt;
mod pull;
mod read;
mod repository;
mod sha256;
mod app;
mod auth;
mod bench;
mod build_job;
mod catalog;
mod checks;
mod cli;
mod config;
mod help;
mod detect;
mod doctor;
mod launch;
mod control_socket;
mod evals;
mod endpoint;
mod json;
mod run;
mod runner;
mod setup;
mod state;
mod status;
mod style;
mod test_gate;
mod text;
mod unix_file;
mod unix_flags;
mod worker;
mod verify;

fn main() {
    run_main(std::env::args().skip(1).collect());
}

fn run_main(mut args: Vec<String>) {
    if args.first().is_some_and(|arg| arg == "__socket-probe") {
        std::process::exit(if args.len() == 2 { control_socket::probe(&args[1]) } else { 1 });
    }
    // `vade test`, `vade bench` and their help pages share the CLI's layout;
    // `vade bench --help-all` lists every benchmark option.
    let topic = |name: &str| (args.len() == 2 && matches!(args[0].as_str(), "help" | "--help" | "-h") && args[1] == name)
        || (args.first().is_some_and(|arg| arg == name) && ((args.len() == 1 && name == "test")
            || (args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h"))));
    if topic("test") || topic("bench") {
        public_output(&cli::help(Some(if topic("test") { "test" } else { "bench" })));
        return;
    }
    if args.first().is_some_and(|arg| arg == "bench") && args[1..].iter().any(|arg| arg == "--help-all") {
        match bench::run(&["--help".into()], || false) {
            Ok(code) => std::process::exit(code),
            Err(error) => fail(&error),
        }
    }
    if args.first().is_some_and(|arg| arg == "test") {
        let code = if args
            .get(1)
            .is_some_and(|arg| matches!(arg.as_str(), "build" | "determinism"))
        {
            test_gate::run(&args[1..])
        } else {
            evals::run(&args[1..])
        };
        std::process::exit(output::exit_code(code));
    }
    if args.first().is_some_and(|arg| arg == "bench") {
        let _interruptions = if args.get(1).is_some_and(|arg| arg == "--suite") {
            None
        } else {
            Some(runner::Interruptions::install())
        };
        let arguments = endpoint::bench_arguments(&args[1..]).unwrap_or_else(|error| fail(&error));
        match bench::run(&arguments, runner::interrupted) {
            Ok(code) => std::process::exit(if runner::interrupted() { 130 } else { code }),
            Err(error) => fail(&error),
        }
    }
    if args
        .first()
        .is_some_and(|arg| matches!(arg.as_str(), "__build-supervisor" | "__build-anchor"))
    {
        if args.len() != 2 {
            fail_private("Invalid private build request.");
        }
        if let Err(error) = build_job::helper(&args[0], std::path::Path::new(&args[1])) {
            fail_private(&error);
        }
        return;
    }
    if args.first().is_some_and(|arg| arg == "__engine") {
        if args.len() != 1 {
            std::process::exit(2);
        }
        if let Err(error) = worker::engine() {
            fail_private(&error);
        }
        return;
    }
    if args.first().is_some_and(|arg| arg == "__worker") {
        if args.len() != 1 {
            fail("The private helper takes no arguments. Use vade --help.");
        }
        match worker::run() {
            Ok(true) => {}
            Ok(false) => std::process::exit(1),
            Err(error) => {
                output::report(format_args!(
                    "{}",
                    text::encode(&text::Fields::from([("error".into(), error)]))
                ), false);
                std::process::exit(1);
            }
        }
        return;
    }
    // vade read needs no engine and no key, so key options and
    // VADE_API_KEY_FILE do not apply to it; read::parse refuses them.
    if args.len() == 2 && matches!(args[0].as_str(), "help" | "--help" | "-h") && args[1] == "read" {
        public_output(&read::help());
        return;
    }
    if args.first().is_some_and(|arg| arg == "read") {
        if args[1..].iter().any(|arg| matches!(arg.as_str(), "--help" | "-h")) {
            public_output(&read::help());
            return;
        }
        std::process::exit(read::run(&args[1..]));
    }
    auth::configure(&mut args).unwrap_or_else(|error| fail(&error));
    if args.len() == 2 && matches!(args[0].as_str(), "help" | "--help" | "-h") && args[1] == "verify" {
        public_output(&verify::help());
        return;
    }
    if args.first().is_some_and(|arg| arg == "verify") {
        std::process::exit(verify::run(&args[1..]));
    }
    let parsed = cli::parse(&args).unwrap_or_else(|error| fail(&error));
    if let cli::Command::Help(command) = parsed.command {
        public_output(&cli::help(command.as_deref()));
        return;
    }
    // Downloads and the model list need no setup.
    if let cli::Command::Pull(arguments) | cli::Command::Models(arguments) = &parsed.command {
        let result = if matches!(parsed.command, cli::Command::Pull(_)) { pull::run(arguments, parsed.dry_run) } else { pull::models(arguments) };
        if let Err(error) = result { fail(&error); }
        return;
    }
    // From here every error of a --json command still ends in its one object
    // (see fail).
    checks::set_json(parsed.json.then(|| checks::Answer { schema: match parsed.command {
        cli::Command::Doctor => "vade.doctor.v1", cli::Command::Detect => "vade.detect.v1", cli::Command::Config => "vade.config.v1", _ => "vade.version.v1",
    }, mode: owner(&parsed.command), dry: parsed.dry_run }));
    if parsed.command == cli::Command::Version {
        if parsed.json {
            let value = evals::json::object(vec![("schema", "vade.version.v1".into()), ("version", env!("CARGO_PKG_VERSION").into())]);
            checks::print_json(&value).unwrap_or_else(|error| fail(&error));
        } else {
            public_output(concat!("vade ", env!("CARGO_PKG_VERSION"), "\n"));
        }
        return;
    }
    if parsed.command == cli::Command::Config {
        if let Err(error) = setup::show(parsed.json) { fail(&error); }
        return;
    }
    // Keep every JSON status outcome (including owner-config errors) inside
    // the versioned schema; status::run uses the same owner selector below.
    if parsed.command == cli::Command::StatusJson {
        let (output, code) = status::run(parsed.dry_run);
        output::report(format_args!("{output}"), true);
        std::process::exit(output::exit_code(code));
    }
    run::validate_start(&parsed.command).unwrap_or_else(|error| fail(&error));
    // Validate invocation credentials before any lifecycle dispatch, so an
    // unreadable key file fails identically on the controller, local-run and
    // owner-launch selections (dry runs never read the file).
    if matches!(
        parsed.command,
        cli::Command::Run(_) | cli::Command::Chat(_) | cli::Command::Doctor
    ) {
        auth::validate(parsed.dry_run).unwrap_or_else(|error| fail(&error));
    }
    if launch::handles(&parsed.command) {
        let _interruptions = if matches!(parsed.command, cli::Command::Launch(_) | cli::Command::Install | cli::Command::Update | cli::Command::Up(_) | cli::Command::Down) {
            Some(runner::Interruptions::install())
        } else { None };
        if let Err(error) = launch::run(parsed.command, parsed.dry_run) { fail(&error); }
        return;
    }
    let _interruptions = if matches!(
        parsed.command,
        cli::Command::Run(_) | cli::Command::Up(_) | cli::Command::Down
    ) {
        Some(runner::Interruptions::install())
    } else {
        None
    };
    let local_home = run::home().unwrap_or_else(|error| fail(&error));
    if run::handles(&parsed.command, &local_home) {
        let _lock = if !parsed.dry_run
            && matches!(parsed.command, cli::Command::Run(_) | cli::Command::Down)
        {
            Some(state::Lock::acquire(&local_home).unwrap_or_else(|error| fail(&error)))
        } else {
            None
        };
        let mut local = run::Local {
            root: local_home,
            runner: runner::SystemRunner {
                dry: parsed.dry_run,
            },
            dry: parsed.dry_run,
        };
        if let Err(error) = local.execute(parsed.command) {
            fail(&error);
        }
        return;
    }
    let config = match config::Config::load() {
        Ok(config) => config,
        // A fresh controller still gets one row per check and a FAIL line.
        Err(error) if parsed.command == cli::Command::Doctor => {
            let mut report = checks::Report::new(owner(&parsed.command), parsed.dry_run);
            let next = "On a Spark, run vade launch; for a controller, copy cli/examples/fp4-local.conf, edit it and set VADE_CONFIG (INSTALL.md).";
            let result = report.fail("config", &style::split(&error).0, next).and_then(|()| {
                if !parsed.dry_run {
                    doctor::binary(&mut report)?;
                    match doctor::start_options() {
                        Ok(options) => doctor::defaults(&mut report, options.context, options.output,
                            options.mtp_explicit.then_some(options.mtp), doctor::prefix_cache())?,
                        Err(error) => report.fail("launch defaults", &error, "Correct MTP, then run vade doctor.")?,
                    }
                    checks::local_tools(&mut report)?;
                }
                report.finish()
            });
            if let Err(error) = result { fail(&error); }
            return;
        }
        Err(error) if matches!(parsed.command, cli::Command::Menu) => {
            let error = text::terminal_text(&error);
            public_output(&format!("{} Vade needs setup.\n  {error}\n\n{}", style::paint(style::Token::Warn, "▲", style::Stream::Out),
                status::next_steps(&[("Start", "vade up"), ("Chat", "vade chat"), ("Stop", "vade down"), ("Check", "vade doctor")])));
            return;
        }
        Err(error) => fail(&error),
    };
    let _lock = if parsed.dry_run
        || !matches!(
            parsed.command,
            cli::Command::Up(_) | cli::Command::Down | cli::Command::Install | cli::Command::Update
        ) {
        None
    } else {
        Some(state::Lock::acquire(&config.state_dir).unwrap_or_else(|error| fail(&error)))
    };
    let mut app = app::App {
        config,
        runner: runner::SystemRunner {
            dry: parsed.dry_run,
        },
        dry: parsed.dry_run,
    };
    if let Err(error) = app.execute(parsed.command) {
        fail(&error);
    }
}

/// The setup `command` runs against, in dispatch order: `local`, `legacy`,
/// `controller`, or `none` before any setup.
fn owner(command: &cli::Command) -> &'static str {
    if launch::handles(command) { "local" }
    else if run::home().is_ok_and(|home| run::handles(command, &home)) { "legacy" }
    else if std::env::var_os("VADE_CONFIG").is_some() || config::config_path().is_ok_and(|path| path.is_file()) { "controller" }
    else { "none" }
}

fn public_output(message: &str) {
    if let Err(error) = output::text(format_args!("{message}")) { fail(&error); }
}

/// Print a command error as an `error:`/`hint:` block on stderr and exit 1.
fn fail(message: &str) -> ! {
    if message == checks::REPORTED { std::process::exit(output::exit_code(1)); }
    // A --json command answers with its object even when it fails early.
    if let Some(value) = checks::unanswered(message) {
        output::operation_failed();
        let _ = checks::print_json(&value);
        std::process::exit(output::exit_code(1));
    }
    if !output::broken_pipe_success() {
        let block = style::error_block(&text::terminal_text(message), style::color(style::Stream::Err));
        output::stderr_line(format_args!("{block}"));
    }
    std::process::exit(if output::broken_pipe_success() { 0 } else { 1 })
}

/// Private helpers report errors verbatim for the parent that started them.
fn fail_private(message: &str) -> ! {
    if !output::broken_pipe_success() {
        output::stderr_line(format_args!("{}", text::terminal_text(message)));
    }
    std::process::exit(if output::broken_pipe_success() { 0 } else { 1 })
}

#[cfg(test)]
#[path = "../tests/host_edges/mod.rs"]
mod host_edges;
