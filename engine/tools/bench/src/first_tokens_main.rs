//! Prebuilt entry point for the shared chat proof client.
fn main() {
    match vade_bench::first_tokens::run(&std::env::args().skip(1).collect::<Vec<_>>(), || false) {
        Ok(code) => std::process::exit(code),
        Err(error) => { eprintln!("first-tokens: {error}"); std::process::exit(1); }
    }
}
