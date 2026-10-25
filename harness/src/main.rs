//! Command dispatch and startup ordering for the terminal harness.
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
                vade_harness::diagnostics::stderr_line(format_args!("vade: demo-compaction takes no arguments; run vade-harness demo-compaction on its own."));
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
#[path = "main/tests.rs"]
mod tests;


#[cfg(test)]
#[path = "native_children_tests.rs"]
mod native_children_tests;
