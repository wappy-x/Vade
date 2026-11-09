//! Build with rustc --edition=2024 -C panic=abort tools/draft-prompts.rs -L dependency=target/release/deps
//! --extern engine_tokenizer=target/release/libengine_tokenizer.rlib -o target/release/draft-prompts
use engine_tokenizer::{
    Tokenizer,
    chat::{self, Message, Options, Role},
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) || args.get(2).is_some_and(|v| v != "long-code") {
        return Err("usage: draft-prompts MODEL OUTPUT_DIR [long-code]".into());
    }
    let tokenizer = Tokenizer::from_directory(&args[0])?;
    std::fs::create_dir_all(&args[1])?;
    let cases = [
        (
            "code",
            "Change only the default timeout from 30 to 45 in this Rust code. Return the complete updated code without commentary.\n\n```rust\n#[derive(Clone, Debug)]\npub struct Config {\n    pub host: String,\n    pub port: u16,\n    pub timeout_seconds: u64,\n    pub retries: usize,\n}\n\nimpl Default for Config {\n    fn default() -> Self {\n        Self {\n            host: String::from(\"localhost\"),\n            port: 8080,\n            timeout_seconds: 30,\n            retries: 3,\n        }\n    }\n}\n\nimpl Config {\n    pub fn endpoint(&self) -> String {\n        format!(\"{}:{}\", self.host, self.port)\n    }\n}\n```",
        ),
        (
            "quotes",
            "Summarize the following incident in one sentence, then quote the two sentences beginning 'The deployment' and 'The service' exactly, each on its own line.\n\nThe deployment began at 09:15 after all integration checks passed. A configuration mismatch caused requests to queue behind a single worker. The service recovered at 09:42 after the team restored the previous connection settings. No customer records were lost, and queued requests completed before 10:00. The follow-up review recommended testing the connection limit under realistic traffic before future releases.",
        ),
        (
            "json",
            "Return this JSON exactly, changing only the status of job-003 from queued to complete. Do not add a code fence or commentary.\n{\"jobs\":[{\"id\":\"job-001\",\"status\":\"complete\",\"result\":{\"rows\":120,\"warnings\":0}},{\"id\":\"job-002\",\"status\":\"complete\",\"result\":{\"rows\":240,\"warnings\":1}},{\"id\":\"job-003\",\"status\":\"queued\",\"result\":{\"rows\":360,\"warnings\":0}},{\"id\":\"job-004\",\"status\":\"queued\",\"result\":{\"rows\":480,\"warnings\":2}}],\"summary\":{\"total\":4,\"region\":\"north\",\"version\":2}}",
        ),
        (
            "prose",
            "Write an original short story of about 150 words about a lighthouse keeper who hears music from an empty boat. Use vivid details and a surprising ending. Do not repeat this request.",
        ),
    ];
    let mut long = String::from(
        "Change only VERSION from 1 to 2. Return the complete Rust code unchanged otherwise, without commentary.\n\n```rust\npub const VERSION: u32 = 1;\n\n",
    );
    if args.len() == 3 {
        for i in 0..20 {
            long.push_str(&format!("pub fn record_{i:02}() -> (&'static str, u32) {{\n    (\"record-{i:02}\", {})\n}}\n\n", 100 + i));
        }
        long.push_str("```");
    }
    let selected = if args.len() == 3 {
        vec![("long-code", long.as_str())]
    } else {
        cases.to_vec()
    };
    for (name, content) in selected {
        let text = chat::format(
            &[Message::new(Role::User, content)],
            &Options {
                enable_thinking: false,
                ..Options::default()
            },
        )?;
        let ids = tokenizer.encode(&text)?;
        let path = std::path::Path::new(&args[1]);
        std::fs::write(path.join(format!("{name}.txt")), text)?;
        std::fs::write(
            path.join(format!("{name}.ids")),
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" "),
        )?;
        println!("case={name} prompt_tokens={}", ids.len());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}
