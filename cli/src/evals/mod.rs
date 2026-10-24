//! Local evaluation tools. No lifecycle configuration or GPU ownership is needed.
mod capture;
mod gdn;
mod hash;
mod help;
pub mod json;
mod longctx;
mod prepare;
mod unix_flags;
pub use hash::{Sha256, sha};
use std::path::PathBuf;

pub const CHECKPOINT: &str = "8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03";
// The frozen long-context suite is pinned to the FP8 checkpoint above.
pub const MODEL: &str = "qwen3.8-flash-next-fp8";
pub fn sampling() -> Vec<(&'static str, json::Value)> {
    vec![
        ("temperature", 0i64.into()),
        ("top_p", 1i64.into()),
        ("top_k", 0i64.into()),
        ("repetition_penalty", 1i64.into()),
        ("presence_penalty", 0i64.into()),
        ("frequency_penalty", 0i64.into()),
    ]
}
pub struct LongArgs {
    pub command: String,
    pub tokenizer_command: Vec<String>,
    pub frozen_bodies: PathBuf,
    pub ifeval_inputs: PathBuf,
    pub output: PathBuf,
    pub documents: usize,
    pub seed: String,
    pub keys_per_depth: usize,
    pub checkpoint_hash: String,
    pub max_tokens: usize,
    pub max_tokens_value: json::Value,
    pub manifest: PathBuf,
    pub top_k: usize,
    pub floor_factor: f64,
    pub report: Option<PathBuf>,
    pub a_url: String,
    pub b_url: String,
    pub c_url: Option<String>,
    pub timeout: f64,
    pub api_key_env: Option<String>,
    pub a_build: String,
    pub b_build: String,
    pub c_build: String,
    pub a_dumps: PathBuf,
    pub b_dumps: PathBuf,
    pub c_dumps: Option<PathBuf>,
}
fn decimal_integer(value: &str) -> Result<String, String> {
    json::integer_input(value)
}

impl LongArgs {
    fn parse(args: &[String]) -> Result<Self, String> {
        let command = args
            .first()
            .filter(|s| matches!(s.as_str(), "prepare" | "run" | "compare"))
            .ok_or("choose prepare, run or compare")?
            .clone();
        let mut value = Self {
            command,
            tokenizer_command: Vec::new(),
            frozen_bodies: PathBuf::new(),
            ifeval_inputs: PathBuf::new(),
            output: PathBuf::new(),
            documents: 4,
            seed: "1963583967".into(),
            keys_per_depth: 1,
            checkpoint_hash: CHECKPOINT.into(),
            max_tokens: 256,
            max_tokens_value: json::Value::Int(256),
            manifest: PathBuf::new(),
            top_k: 8,
            floor_factor: 1.5,
            report: None,
            a_url: String::new(),
            b_url: String::new(),
            c_url: None,
            timeout: 3600.,
            api_key_env: None,
            a_build: "7a4ad78a".into(),
            b_build: "caf433b".into(),
            c_build: "7a4ad78a+scoring+layer0-ulp".into(),
            a_dumps: PathBuf::new(),
            b_dumps: PathBuf::new(),
            c_dumps: None,
        };
        let allowed: &[&str] = match value.command.as_str() {
            "prepare" => &[
                "--tokenizer-command",
                "--frozen-bodies",
                "--ifeval-inputs",
                "--output",
                "--documents",
                "--seed",
                "--keys-per-depth",
                "--checkpoint-hash",
                "--max-tokens",
            ],
            "run" => &[
                "--manifest",
                "--top-k",
                "--floor-factor",
                "--report",
                "--a-url",
                "--b-url",
                "--c-url",
                "--output",
                "--timeout",
                "--api-key-env",
                "--a-build",
                "--b-build",
                "--c-build",
            ],
            _ => &[
                "--manifest",
                "--top-k",
                "--floor-factor",
                "--report",
                "--a-dumps",
                "--b-dumps",
                "--c-dumps",
            ],
        };
        let mut seen = std::collections::BTreeSet::new();
        let mut i = 1;
        while i < args.len() {
            let (mut flag, inline) = args[i]
                .split_once('=')
                .map_or((args[i].as_str(), None), |(a, b)| (a, Some(b)));
            if !allowed.contains(&flag) {
                let matches: Vec<_> = allowed
                    .iter()
                    .filter(|known| known.starts_with(flag))
                    .collect();
                if matches.len() == 1 {
                    flag = matches[0];
                } else {
                    return Err(format!("unrecognized arguments: {}", args[i]));
                }
            }
            seen.insert(flag.to_owned());
            i += 1;
            if flag == "--tokenizer-command" {
                value.tokenizer_command.clear();
                if let Some(s) = inline {
                    value.tokenizer_command.push(s.into());
                }
                while i < args.len() && !args[i].starts_with("--") {
                    value.tokenizer_command.push(args[i].clone());
                    i += 1;
                }
                if value.tokenizer_command.is_empty() {
                    return Err("--tokenizer-command requires at least one argument".into());
                }
                continue;
            }
            let s = match inline {
                Some(s) => s,
                None => {
                    let s = args
                        .get(i)
                        .ok_or_else(|| format!("{flag} requires a value"))?;
                    i += 1;
                    s.as_str()
                }
            };
            let integer = || {
                decimal_integer(s)
                    .and_then(|s| {
                        s.parse::<i64>()
                            .map_err(|_| "invalid int value".to_string())
                    })
                    .map_err(|_| format!("invalid int value: '{s}'"))
            };
            let unsigned = || {
                integer().and_then(|n| {
                    usize::try_from(n).map_err(|_| format!("invalid nonnegative value: '{s}'"))
                })
            };
            let real = || json::float_input(s).map_err(|_| format!("invalid float value: '{s}'"));
            match flag {
                "--frozen-bodies" => value.frozen_bodies = s.into(),
                "--ifeval-inputs" => value.ifeval_inputs = s.into(),
                "--output" => value.output = s.into(),
                "--documents" => value.documents = unsigned()?,
                "--seed" => value.seed = decimal_integer(s)?,
                "--keys-per-depth" => value.keys_per_depth = integer()?.max(0) as usize,
                "--checkpoint-hash" => value.checkpoint_hash = s.into(),
                "--max-tokens" => {
                    let number = decimal_integer(s)?;
                    value.max_tokens =
                        number
                            .parse::<usize>()
                            .unwrap_or(if number.starts_with('-') {
                                0
                            } else {
                                usize::MAX
                            });
                    value.max_tokens_value = json::parse(&number)?;
                }
                "--manifest" => value.manifest = s.into(),
                "--top-k" => value.top_k = unsigned()?,
                "--floor-factor" => value.floor_factor = real()?,
                "--report" => value.report = Some(s.into()),
                "--a-url" => value.a_url = s.into(),
                "--b-url" => value.b_url = s.into(),
                "--c-url" => value.c_url = Some(s.into()),
                "--timeout" => value.timeout = real()?,
                "--api-key-env" => value.api_key_env = Some(s.into()),
                "--a-build" => value.a_build = s.into(),
                "--b-build" => value.b_build = s.into(),
                "--c-build" => value.c_build = s.into(),
                "--a-dumps" => value.a_dumps = s.into(),
                "--b-dumps" => value.b_dumps = s.into(),
                "--c-dumps" => value.c_dumps = Some(s.into()),
                _ => unreachable!(),
            }
        }
        let required: &[&str] = match value.command.as_str() {
            "prepare" => &[
                "--tokenizer-command",
                "--frozen-bodies",
                "--ifeval-inputs",
                "--output",
            ],
            "run" => &["--manifest", "--a-url", "--b-url", "--output"],
            _ => &["--manifest", "--a-dumps", "--b-dumps"],
        };
        for flag in required {
            if !seen.contains(*flag) {
                return Err(format!("required argument: {flag}"));
            }
        }
        if value.command == "prepare" {
            if !(4..=8).contains(&value.documents) {
                return Err("--documents must be 4..8".into());
            }
            if !(1..=8).contains(&value.keys_per_depth)
                || value.max_tokens < 64 * value.keys_per_depth
            {
                return Err("K must be 1..8 with at least 64*K output tokens".into());
            }
        } else {
            if !(1..=32).contains(&value.top_k) {
                return Err("--top-k must be 1..32".into());
            }
            if !value.floor_factor.is_finite() || value.floor_factor < 1. {
                return Err("invalid floor factor".into());
            }
        }
        Ok(value)
    }
}
pub fn help() -> &'static str {
    "vade test longctx prepare|run|compare [OPTIONS]\nvade test gdn-state REFERENCE CANDIDATE [SPLIT] [--identity] [--legacy-prefill-group N]\nvade test unix-flags-abi [--target TARGET]\nFreeze long-context prompts, compare teacher-forced logits and retrieval or GDN states, or compile Unix flag ABI assertions.\nSee cli/evals.md for every option, default, bound and receipt.\n"
}
pub fn run(args: &[String]) -> i32 {
    if args.is_empty() || matches!(args[0].as_str(), "--help" | "-h") {
        crate::public_output(help());
        return 0;
    }
    if args[0] == "longctx" && args[1..].iter().any(|s| s == "--help" || s == "-h") {
        crate::public_output(help::longctx(args.get(1).map(String::as_str)));
        return 0;
    }
    let result = match args[0].as_str() {
        "gdn-state" => gdn::run(&args[1..]),
        "longctx" => run_longctx(&args[1..]),
        "unix-flags-abi" => unix_flags::run(&args[1..]),
        _ => Err("choose longctx, gdn-state or unix-flags-abi".into()),
    };
    match result {
        Ok(status) => status,
        Err(error) => {
            crate::output::stderr_line(format_args!("FAIL {error}"));
            2
        }
    }
}
fn run_longctx(args: &[String]) -> Result<i32, String> {
    let mut args = LongArgs::parse(args)?;
    if args.command == "prepare" {
        crate::output::report(format_args!("{}", json::packed(&prepare::run(&args)?)), true);
        return Ok(0);
    }
    let manifest = json::read(&args.manifest)?;
    if manifest["schema"].as_str() != Some("vade.longctx.v1")
        || manifest["checkpoint_sha256"].as_str() != Some(CHECKPOINT)
    {
        return Err("manifest is not the pinned evaluation suite".into());
    }
    if args.command == "run" {
        capture::run(&mut args, &manifest)?;
    }
    longctx::compare(&args, &manifest)
}
