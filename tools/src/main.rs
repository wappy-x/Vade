#![forbid(unsafe_code)]
use std::io::{self,Read,Write};
use vade_tools::{Result,json};
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check-doc-links") if args.len() == 1 => vade_tools::docs::run(&vade_tools::root()),
        Some("check-bundle") if args.len() >= 2 => {
            let path = vade_tools::bundle::check(std::path::Path::new(&args[1]),&args[2..])?;
            println!("bundle manifest and archive hashes: PASS ({})",path.display()); Ok(())
        },
        Some("ci-no-gpu") if args.len() == 1 => vade_tools::cuda::run(&vade_tools::root()),
        Some("benchmarks-chart") if args.len() == 1 => vade_tools::chart::run(&vade_tools::root()),
        Some("bench-docs") if args.len() == 3 => vade_tools::bench_docs::run(std::path::Path::new(&args[1]),std::path::Path::new(&args[2])),
        Some("fetch-model") if args.len() == 4 => vade_tools::fetch::run(&args[1],&args[2],std::path::Path::new(&args[3])),
        Some("json-get"|"json-print"|"json-value"|"json-quote"|"json-stream-check") => {
            let mut input = String::new(); io::stdin().read_to_string(&mut input)?;
            if args[0] == "json-quote" { println!("{}",json::quote(&input)); return Ok(()); }
            let value = json::parse(&input)?; let mut part = &value;
            if args[0] == "json-stream-check" {
                if let json::Value::Object(fields) = &value { if let Some(error) = fields.get("error") { return Err(format!("stream error: {}",error.encode()).into()); } }
                value.get("choices").array()?; return Ok(());
            }
            for key in &args[1..] { part = part.get(key); }
            if args[0] == "json-value" { println!("{}",part.encode()); }
            else if *part != json::Value::Null {
                if args[0] == "json-print" { print!("{}",part.text()?); io::stdout().flush()?; }
                else { println!("{}",part.text()?); }
            } Ok(())
        },
        _ => Err("usage: vade-tools {check-doc-links|check-bundle DIR [FORBIDDEN ...]|ci-no-gpu|benchmarks-chart|bench-docs CARDS_DIR REPO_DIR|fetch-model REPO REV DIR|json-get [KEY ...]|json-print [KEY ...]|json-value [KEY ...]|json-quote|json-stream-check}".into())
    }
}
fn main() { if let Err(error) = run() { eprintln!("vade-tools: {error}"); std::process::exit(1); } }
