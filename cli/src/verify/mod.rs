//! Compare complete generated token IDs, including across drafting modes.
//! `http` is the bounded HTTP/SSE client it uses.
mod http;

use crate::evals::json::{self, Value};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{mpsc, Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Progress while verify waits on its blocking requests: `write` gets the
/// current step and the time since start, first after `first`, then every
/// `every`, until the ticker is dropped. Nothing is written before the first
/// step is set.
struct Ticker {
    step: Arc<Mutex<String>>,
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Ticker {
    fn start(first: Duration, every: Duration, write: impl Fn(&str, Duration) + Send + 'static) -> Self {
        let step = Arc::new(Mutex::new(String::new()));
        let (stop, stopped) = mpsc::channel::<()>();
        let shared = Arc::clone(&step);
        let started = Instant::now();
        let thread = std::thread::spawn(move || {
            let mut wait = first;
            while let Err(mpsc::RecvTimeoutError::Timeout) = stopped.recv_timeout(wait) {
                let step = shared.lock().map(|step| step.clone()).unwrap_or_default();
                if !step.is_empty() { write(&step, started.elapsed()); }
                wait = every;
            }
        });
        Self { step, stop: Some(stop), thread: Some(thread) }
    }
    fn set(&self, step: String) {
        if let Ok(mut current) = self.step.lock() { *current = step; }
    }
}
impl Drop for Ticker {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

/// Usage for `vade verify`.
/// `vade verify --help`.
pub fn help() -> String {
    crate::style::Page {
        summary: "Check that every run returns the same token IDs, drafting OFF and ON.",
        usage: &["vade verify [OPTIONS]"],
        options: &[
            ("-n N", "Runs per mode, 1..10000 (default 20); --runs is an alias"),
            ("--prompt short|2k|8k", "Bundled offline prompt (default short); 2k/8k sizes are approximate and tokenizer-dependent"),
            ("--url URL", "Server to check (default VADE_API_URL, then the recorded endpoint)"),
            ("--machine NAME", "Add an independent running engine from that machine's verify_url in VADE_CONFIG"),
            ("--max-tokens N", "Tokens per run, 1..32768 (default 128)"),
            ("--timeout SECONDS", "Deadline for each request, 1..3600 (default 120)"),
            ("--receipt PATH", "Receipt file (default vade-verify.json in the current directory)"),
        ],
        examples: &["vade verify", "vade verify -n 3 --prompt 2k", "vade verify --url http://127.0.0.1:18080/v1"],
        notes: &[
            "Uses --url, then VADE_API_URL, then the selected installation's recorded endpoint. Keep the same VADE_CONFIG and VADE_HOME as up/run; explicit VADE_CONFIG selects the controller's local_port. Without a running installation, start vade launch or supply --url.",
            "Compares token IDs byte for byte across repeated greedy runs.",
            "Generates a fixed token count (ignores EOS) to check every ID, with cache and thinking off. ON requests strict drafting and must actually draft; unavailable drafting fails the check.",
            "Prints one result line; exit 0 means identical, 1 means different or incomplete. Run and token numbers start at 1.",
        ],
    }.render(crate::style::color(crate::style::Stream::Out))
}

struct Options {
    n: usize,
    prompt: String,
    url: Option<String>,
    machine: Option<String>,
    max_tokens: usize,
    timeout: Duration,
    receipt: PathBuf,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            n: 20,
            prompt: "short".into(),
            url: None,
            machine: None,
            max_tokens: 128,
            timeout: Duration::from_secs(120),
            receipt: "vade-verify.json".into(),
        }
    }
}
fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let key = if flag == "--runs" { "-n" } else { flag };
        if !matches!(
            key,
            "-n" | "--prompt" | "--url" | "--machine" | "--max-tokens" | "--timeout" | "--receipt"
        ) {
            return Err(format!(
                "Unknown option {}; run vade verify --help.",
                crate::text::argument(flag)
            ));
        }
        if !seen.insert(key) {
            return Err(format!("Use {key} only once."));
        }
        let value = args
            .get(index + 1)
            .filter(|v| !v.is_empty() && !v.starts_with('-'))
            .ok_or_else(|| format!("Give {key} a value."))?;
        let number = |maximum: usize| -> Result<usize, String> {
            value
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=maximum).contains(n))
                .ok_or_else(|| format!("Set {key} to a whole number from 1 to {maximum}."))
        };
        match key {
            "-n" => options.n = number(10000)?,
            "--prompt" => {
                options.prompt = value.to_ascii_lowercase();
                if !matches!(options.prompt.as_str(), "short" | "2k" | "8k") {
                    return Err("Choose --prompt short, 2k, or 8k.".into());
                }
            }
            "--url" => options.url = Some(value.clone()),
            "--machine" => options.machine = Some(value.clone()),
            "--max-tokens" => options.max_tokens = number(32768)?,
            "--timeout" => options.timeout = Duration::from_secs(number(3600)? as u64),
            "--receipt" => options.receipt = value.into(),
            _ => unreachable!(),
        }
        index += 2;
    }
    Ok(options)
}

fn prompt(name: &str) -> &'static str {
    match name {
        "2k" => include_str!("prompts/2k.txt"),
        "8k" => include_str!("prompts/8k.txt"),
        _ => include_str!("prompts/short.txt"),
    }
}

struct Target {
    name: String,
    url: String,
    endpoint: http::Endpoint,
    off_runs: usize,
    on_runs: usize,
    draft_runs: usize,
    identity: Value,
}
fn targets(options: &Options) -> Result<Vec<Target>, String> {
    let (url, config) = crate::endpoint::resolve(options.url.clone(), options.machine.is_some())?;
    let mut urls = vec![("primary".to_string(), url)];
    if let Some(name) = &options.machine {
        let machine = config
            .as_ref()
            .unwrap()
            .machines
            .iter()
            .find(|m| &m.name == name)
            .ok_or_else(|| {
                format!(
                    "Machine {} is not in your config.",
                    crate::text::argument(name)
                )
            })?;
        let url = machine.verify_url.as_ref().filter(|u| !u.is_empty())
            .ok_or("Set verify_url for the second machine to its independent running engine's HTTP URL.")?;
        if http::Endpoint::parse(url)?.same_address(&http::Endpoint::parse(&urls[0].1)?) {
            return Err("The second machine must have a distinct engine URL.".into());
        }
        urls.push((name.clone(), url.clone()));
    }
    urls.into_iter()
        .map(|(name, url)| {
            Ok(Target {
                endpoint: http::Endpoint::parse(&url)?,
                name,
                url,
                off_runs: 0,
                on_runs: 0,
                draft_runs: 0,
                identity: Value::Null,
            })
        })
        .collect()
}

fn id_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}
fn quoted(value: &str) -> String {
    crate::text::quoted(value)
}

struct Receipt {
    value: Value,
    started: Instant,
    runs_completed: usize,
    requests_completed: usize,
}
impl Receipt {
    fn new(options: &Options) -> Self {
        Self {
            value: json::object(vec![
                ("schema", "vade.verify.v1".into()),
                ("result", "error".into()),
                (
                    "started_unix_seconds",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64()
                        .into(),
                ),
                ("cli_version", env!("CARGO_PKG_VERSION").into()),
                ("prompt", options.prompt.clone().into()),
                (
                    "prompt_sha256",
                    crate::sha256::sha(prompt(&options.prompt).as_bytes()).into(),
                ),
                ("runs_requested", options.n.into()),
                ("runs_completed", 0usize.into()),
                ("requests_completed", 0usize.into()),
                ("machines", 0usize.into()),
                ("tokens_per_run", 0usize.into()),
                ("max_tokens", options.max_tokens.into()),
                ("sampling", "greedy".into()),
                (
                    "draft_modes",
                    Value::Array(vec!["plain".into(), "strict".into()]),
                ),
                ("cache", false.into()),
                ("thinking", false.into()),
                ("ignore_eos", true.into()),
                (
                    "comparison",
                    "all generated u32 token IDs, little-endian bytes, against run 1 primary OFF"
                        .into(),
                ),
                ("reference_ids_sha256", Value::Null),
                ("first_difference", Value::Null),
            ]),
            started: Instant::now(),
            runs_completed: 0,
            requests_completed: 0,
        }
    }
    fn record_targets(&mut self, targets: &[Target]) {
        self.value.insert("machines", targets.len().into());
        self.value.insert(
            "targets",
            Value::Array(
                targets
                    .iter()
                    .map(|target| {
                        json::object(vec![
                            ("name", target.name.clone().into()),
                            ("url", target.url.clone().into()),
                            ("off_runs", target.off_runs.into()),
                            ("on_runs", target.on_runs.into()),
                            ("draft_runs", target.draft_runs.into()),
                            ("identity", target.identity.clone()),
                        ])
                    })
                    .collect(),
            ),
        );
    }
}

fn verify(options: &Options, receipt: &mut Receipt) -> Result<(String, bool), String> {
    let mut targets = targets(options)?;
    receipt.record_targets(&targets);
    crate::auth::validate(false)?;
    let mut reference: Option<http::Reply> = None;
    let mut reference_bytes = Vec::new();
    // Progress goes to stderr (stdout keeps its one result line), also while
    // a request blocks: first after 2 s, then every 15 s.
    let ticker = Ticker::start(Duration::from_secs(2), Duration::from_secs(15), |step, elapsed| {
        crate::output::stderr_line(format_args!("{:>8} still {step} ({} s elapsed)", "Verify", elapsed.as_secs()));
    });
    let result = (|| {
        for run in 1..=options.n {
            for target in &mut targets {
                for draft_on in [false, true] {
                    let mode = if draft_on { "ON" } else { "OFF" };
                    ticker.set(format!("run {run}/{} {} drafting {mode}", options.n, quoted(&target.name)));
                    let reply = target
                        .endpoint
                        .completion(
                            prompt(&options.prompt),
                            draft_on,
                            options.max_tokens,
                            options.timeout,
                        )
                        .map_err(|error| {
                            format!("run {run} {} {mode}: {error}", quoted(&target.name))
                        })?;
                    receipt.requests_completed += 1;
                    target.identity = reply.identity.clone();
                    if draft_on {
                        target.on_runs += 1;
                    } else {
                        target.off_runs += 1;
                    }
                    if reply.draft_used || reply.model_draft_used {
                        target.draft_runs += 1;
                    }
                    if draft_on && !reply.draft_used && !reply.model_draft_used {
                        return Err(format!(
                            "run {run} {} ON: engine did not actually draft; exactness with drafting is unproven.",
                            quoted(&target.name)
                        ));
                    }
                    let bytes = id_bytes(&reply.token_ids);
                    if let Some(expected) = &reference {
                        if reply.model != expected.model
                            || reply.prompt_tokens != expected.prompt_tokens
                        {
                            return Err(format!(
                                "run {run} {} {mode}: model or prompt token count changed.",
                                quoted(&target.name)
                            ));
                        }
                        if bytes != reference_bytes {
                            let index = expected
                                .token_ids
                                .iter()
                                .zip(&reply.token_ids)
                                .position(|(a, b)| a != b)
                                .unwrap_or(expected.token_ids.len().min(reply.token_ids.len()));
                            let expected_id = expected.token_ids.get(index).copied();
                            let actual_id = reply.token_ids.get(index).copied();
                            let expected_text = expected
                                .token_texts
                                .get(index)
                                .map(String::as_str)
                                .unwrap_or("<end>");
                            let actual_text = reply
                                .token_texts
                                .get(index)
                                .map(String::as_str)
                                .unwrap_or("<end>");
                            receipt.value.insert("result", "different".into());
                            receipt.value.insert(
                                "first_difference",
                                json::object(vec![
                                    ("run", run.into()),
                                    ("token", (index + 1).into()),
                                    ("machine", target.name.clone().into()),
                                    ("draft", mode.into()),
                                    (
                                        "expected_id",
                                        expected_id
                                            .map(|id| (id as i64).into())
                                            .unwrap_or(Value::Null),
                                    ),
                                    (
                                        "actual_id",
                                        actual_id
                                            .map(|id| (id as i64).into())
                                            .unwrap_or(Value::Null),
                                    ),
                                    ("expected_text", expected_text.into()),
                                    ("actual_text", actual_text.into()),
                                ]),
                            );
                            let id = |id: Option<u32>| {
                                id.map(|id| id.to_string())
                                    .unwrap_or_else(|| "<end>".into())
                            };
                            return Ok((
                                format!(
                                    "DIFFERENT at run {run} token {}: primary OFF {} {} vs {} {mode} {} {}",
                                    index + 1,
                                    id(expected_id),
                                    quoted(expected_text),
                                    quoted(&target.name),
                                    id(actual_id),
                                    quoted(actual_text)
                                ),
                                false,
                            ));
                        }
                    } else {
                        receipt
                            .value
                            .insert("tokens_per_run", reply.token_ids.len().into());
                        receipt.value.insert("model", reply.model.clone().into());
                        receipt
                            .value
                            .insert("prompt_tokens", reply.prompt_tokens.into());
                        receipt
                            .value
                            .insert("reference_ids_sha256", crate::sha256::sha(&bytes).into());
                        receipt
                            .value
                            .insert("finish_reason", reply.finish_reason.clone().into());
                        receipt
                            .value
                            .insert("completion_tokens", reply.completion_tokens.into());
                        reference_bytes = bytes;
                        reference = Some(reply);
                    }
                }
            }
            receipt.runs_completed = run;
        }
        if targets.iter().any(|target| target.draft_runs == 0) {
            return Err("Drafting ON did not actually draft on every machine; exactness with drafting is unproven.".into());
        }
        receipt.value.insert("result", "identical".into());
        Ok((
            format!(
                "IDENTICAL: {}/{} runs, {} {}, {} tokens",
                options.n,
                options.n,
                targets.len(),
                if targets.len() == 1 {
                    "machine"
                } else {
                    "machines"
                },
                reference.as_ref().unwrap().token_ids.len()
            ),
            true,
        ))
    })();
    drop(ticker);
    receipt.record_targets(&targets);
    result
}

// Publish atomically without changing the caller's directory permissions.
fn write_receipt(path: &std::path::Path, receipt: &Value) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = std::path::absolute(path).map_err(|_| "Cannot resolve receipt path.".to_string())?;
    crate::text::reject_symlinks(&path)?;
    let parent = path
        .parent()
        .ok_or("Receipt path needs a parent directory.")?;
    let tmp = parent.join(format!(
        ".vade-verify-{}-{}",
        std::process::id(),
        crate::text::random_hex(8)?
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|error| crate::text::file_error("Cannot create receipt", &path, &error))?;
        writeln!(file, "{}", json::packed(receipt))
            .and_then(|_| file.sync_all())
            .map_err(|error| crate::text::file_error("Cannot write receipt", &path, &error))?;
        fs::rename(&tmp, &path)
            .map_err(|error| crate::text::file_error("Cannot publish receipt", &path, &error))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// `vade verify`: run each prompt with drafting ON and OFF, compare the token
/// IDs and write the receipt; returns the exit code.
pub fn run(args: &[String]) -> i32 {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        crate::public_output(&help());
        return 0;
    }
    let options = match parse(args) {
        Ok(options) => options,
        // A usage error is not a verification result: the standard error block.
        Err(error) => {
            let color = crate::style::color(crate::style::Stream::Err);
            crate::output::stderr_line(format_args!("{}", crate::style::error_block(&single_line(&error), color)));
            return 1;
        }
    };
    let mut receipt = Receipt::new(&options);
    receipt.value.insert("result", "running".into());
    if let Err(error) = write_receipt(&options.receipt, &receipt.value) {
        crate::output::report(format_args!("ERROR: {}", single_line(&error)), true);
        return 1;
    }
    receipt.value.insert("result", "error".into());
    let (mut line, mut identical) = match verify(&options, &mut receipt) {
        Ok(outcome) => outcome,
        Err(error) => {
            receipt.value.insert("error", error.clone().into());
            (format!("ERROR: {}", single_line(&error)), false)
        }
    };
    receipt
        .value
        .insert("runs_completed", receipt.runs_completed.into());
    receipt
        .value
        .insert("requests_completed", receipt.requests_completed.into());
    receipt.value.insert(
        "elapsed_ms",
        (receipt.started.elapsed().as_secs_f64() * 1000.0).into(),
    );
    if let Err(error) = write_receipt(&options.receipt, &receipt.value) {
        line = format!("ERROR: {}; check outcome: {}", single_line(&error), line);
        identical = false;
    }
    if crate::output::line(format_args!("{line}")).is_err() {
        return 1;
    }
    i32::from(!identical)
}

fn single_line(value: &str) -> String {
    crate::text::terminal_text(value)
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn progress_ticks_while_a_request_blocks_and_stops_when_dropped() {
        // Each progress line arrives on a channel; bounded waits replace sleeps.
        let (sender, lines) = mpsc::channel::<String>();
        let ticker = Ticker::start(Duration::from_millis(1), Duration::from_millis(1), move |step, _| { let _ = sender.send(step.to_owned()); });
        let next = || lines.recv_timeout(Duration::from_secs(30)).expect("a progress line while the request blocks");
        // Unset, the ticker writes nothing, so the first line is this step.
        let first = "run 1/1 \"primary\" drafting OFF";
        ticker.set(first.into());
        // One long "request" (this thread waiting): progress arrives during it.
        assert_eq!((next(), next()), (first.to_owned(), first.to_owned()));
        // A new step reaches a later line.
        let second = "run 1/1 \"primary\" drafting ON";
        ticker.set(second.into());
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut last = next();
        while last != second && Instant::now() < deadline { assert_eq!(last, first); last = next(); }
        assert_eq!(last, second);
        // Dropping joins the thread, which drops the sender: whatever was
        // queued drains, then the channel disconnects with nothing more.
        drop(ticker);
        let rest: Vec<String> = lines.iter().collect();
        assert!(rest.iter().all(|line| line == second), "{rest:?}");
        assert_eq!(lines.try_recv(), Err(mpsc::TryRecvError::Disconnected));
        // A verify that ends before the first interval prints nothing.
        let (sender, quiet) = mpsc::channel::<()>();
        let ticker = Ticker::start(Duration::from_secs(3600), Duration::from_secs(3600), move |_, _| { let _ = sender.send(()); });
        ticker.set("run 1/1 \"primary\" drafting OFF".into());
        drop(ticker);
        assert_eq!(quiet.try_recv(), Err(mpsc::TryRecvError::Disconnected));
    }
    #[test]
    fn options_are_bounded_and_unambiguous() {
        for args in [
            vec!["-n", "0"],
            vec!["-n", "10001"],
            vec!["--max-tokens", "0"],
            vec!["--timeout", "3601"],
            vec!["--prompt", "missing"],
            vec!["-n", "2", "--runs", "3"],
            vec!["--machine"],
            vec!["--dry-run"],
        ] {
            assert!(parse(&args.iter().map(|s| (*s).into()).collect::<Vec<_>>()).is_err());
        }
        let options = parse(&[]).unwrap();
        assert_eq!(options.n, 20);
        assert_eq!(options.max_tokens, 128);
    }
    #[test]
    fn ids_compare_all_four_bytes_and_length() {
        assert_ne!(id_bytes(&[1, 0x100]), id_bytes(&[1, 0]));
        assert_ne!(id_bytes(&[1, 2]), id_bytes(&[1]));
        assert_eq!(id_bytes(&[0x12345678]), [0x78, 0x56, 0x34, 0x12]);
    }
}
