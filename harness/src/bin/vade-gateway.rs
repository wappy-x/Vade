fn main() -> std::process::ExitCode {
    vade_harness::diagnostics::install_panic_hook("vade-gateway");
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if let Err(error) = vade_harness::auth::configure(&mut args) {
        vade_harness::diagnostics::stderr_line(format_args!("vade-gateway: {}. Check --api-key-file PATH or VADE_API_KEY_FILE.",
            vade_harness::terminal::sanitize(&error.to_string())));
        return std::process::ExitCode::FAILURE;
    }
    vade_harness::gateway::invisible::exit(args.into_iter())
}
