//! Client-wall-clock scoreboard measurements, ported from the retained historical timing client.
mod drafting;
mod self_test;
pub(crate) mod stream;
pub(crate) mod support;
pub(crate) mod value;

use std::{fs, path::Path, sync::Arc, thread, time::Instant};
use stream::{request_body, stream_chat};
use support::*;
use value::*;

const METHOD: &str = "usage completion_tokens / (last [DONE] received - first request send); TCP connect, prefill, queueing and response transfer included";
const MARGIN: &str = "gain > abs(A - A_prime) + 0.02 * mean(A, A_prime); strict inequality; each run uses repeat medians; ABBA averages the two B runs";

#[derive(Clone)]
pub(crate) struct Config {
    pub url: Option<String>,
    pub model: String,
    pub model_explicit: bool,
    pub draft: Option<String>,
    pub prompts: Option<String>,
    pub bodies_dir: Option<String>,
    pub concurrency: Vec<usize>,
    pub max_tokens: usize,
    pub repeats: usize,
    pub prefill_body: Option<String>,
    pub prefill_body_sha256: Option<String>,
    pub checkpoint_hash: Option<String>,
    pub build_hash_field: Option<String>,
    pub checkpoint_hash_field: Option<String>,
    pub ab_urls: Vec<String>,
    pub aba_urls: Vec<String>,
    pub launch_commands: Vec<String>,
    pub rank0_log: Option<String>,
    pub ab_rank0_logs: Vec<String>,
    pub aba_rank0_logs: Vec<String>,
    pub observe_cmd: Vec<String>,
    pub observe_timeout: f64,
    pub output: String,
    pub timeout: f64,
    pub startup_timeout: f64,
    pub api_key: Option<String>,
    pub command_env: Vec<(String, String)>,
    pub interrupted: fn() -> bool,
    pub self_test: bool,
    pub order: String,
    pub clock: Instant,
}

const HELP: &str = "vade bench [OPTIONS]\nMeasure a running server: wall time on this client, token counts from the server.\nAll requests: greedy, cache off, thinking off, fixed length (ignore EOS).\nDrafting follows the server's default. The receipt keeps the mode you asked for,\nthe depth and build that /health reports, and what each response reported.\n\n  --url URL                        Recorded local endpoint by default; accepts /v1\n  --model ID                       Default /health model; fallback qwen3.8-flash-next-fp4-local\n  --draft plain|strict             Override drafting on every request (default: server)\n  --prompts FILE                   UTF-8 lines; default bundled offline writing prompt\n  --bodies-dir DIR                 Frozen request bodies with sibling capture directory\n  --concurrency 1,2,4,8,16         Distinct positive cohort widths\n  --max-tokens 256                 Required completion count per chat\n  --repeats 1                      Repeats per cohort, no discarded runs\n  --prefill-body FILE              Three cold exact-8192-token TTFT probes\n  --prefill-body-sha256 HASH       Check frozen probe file bytes\n  --checkpoint-hash HASH           Pin /health checkpoint (required for 8k/A/B)\n  --build-hash-field PATH          Override dotted /health identity field\n  --checkpoint-hash-field PATH     Override dotted /health checkpoint field\n  --ab-urls A B                    Run A,B,B,A; mean the two B run medians\n  --aba-urls A B A_PRIME           Run A,B,A'\n  --launch-commands CMD CMD [CMD]  Foreground shell commands, owned groups only\n  --rank0-log FILE                 Read only appended local rank-zero log bytes\n  --ab-rank0-logs A B              Logs for ABBA variants\n  --aba-rank0-logs A B A_PRIME     Logs for ABA variants\n  --observe-cmd CMD                Repeat per rank: GPU CSV rows, then uptime\n  --observe-timeout 30             Observation command deadline in seconds\n  --output FILE                    Default ./vade-bench.json; sibling .txt table\n  --timeout 1800                   HTTP I/O timeout seconds\n  --startup-timeout 600            Owned server readiness timeout seconds\n  --api-key-file FILE              Optional bearer file; never stored in receipts\n  --self-test                      Tiny local mock server; no GPU\n  --suite URL [OPTIONS]            Existing development suite, separate receipt format\n  --help                           Show this help\n\nHTTP only: use a local tunnel for HTTPS endpoints. VADE_API_KEY_FILE selects\nan optional credential file. URL discovery follows vade verify; use --url if no\nlocal installation is recorded. --out FILE is an alias for --output FILE.\nExisting output files are replaced.\nSee engine/tools/bench/README.md for request and receipt details.\n";

impl Config {
    pub(crate) fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut c = Self {
            url: None,
            model: "qwen3.8-flash-next-fp4-local".into(),
            model_explicit: false,
            draft: None,
            prompts: None,
            bodies_dir: None,
            concurrency: vec![1, 2, 4, 8, 16],
            max_tokens: 256,
            repeats: 1,
            prefill_body: None,
            prefill_body_sha256: None,
            checkpoint_hash: None,
            build_hash_field: None,
            checkpoint_hash_field: None,
            ab_urls: vec![],
            aba_urls: vec![],
            launch_commands: vec![],
            rank0_log: None,
            ab_rank0_logs: vec![],
            aba_rank0_logs: vec![],
            observe_cmd: vec![],
            observe_timeout: 30.,
            output: "./vade-bench.json".into(),
            timeout: 1800.,
            startup_timeout: 600.,
            api_key: None,
            command_env: vec![],
            interrupted: || false,
            self_test: false,
            order: "single".into(),
            clock: Instant::now(),
        };
        let mut key_file = std::env::var("VADE_API_KEY_FILE").ok();
        let mut i = 0;
        while i < arguments.len() {
            let flag = &arguments[i];
            i += 1;
            if flag == "--self-test" {
                c.self_test = true;
                continue;
            }
            let mut take = || -> Result<String, String> {
                let v = arguments
                    .get(i)
                    .filter(|v| !v.starts_with("--"))
                    .ok_or_else(|| format!("missing value for {flag}"))?
                    .clone();
                i += 1;
                Ok(v)
            };
            let value = take()?;
            match flag.as_str() {
                "--url" => c.url = Some(value),
                "--model" => { c.model = value; c.model_explicit = true; }
                "--draft" => {
                    if !matches!(value.as_str(), "plain" | "strict") {
                        return Err("--draft must be plain or strict".into());
                    }
                    c.draft = Some(value);
                }
                "--prompts" => c.prompts = Some(value),
                "--bodies-dir" => c.bodies_dir = Some(value),
                "--max-tokens" => {
                    c.max_tokens = value.parse().map_err(|_| "invalid --max-tokens")?
                }
                "--repeats" => c.repeats = value.parse().map_err(|_| "invalid --repeats")?,
                "--concurrency" => {
                    c.concurrency = value
                        .split(',')
                        .map(|v| {
                            v.trim()
                                .parse()
                                .map_err(|_| "invalid --concurrency".to_string())
                        })
                        .collect::<Result<_, _>>()?
                }
                "--prefill-body" => c.prefill_body = Some(value),
                "--prefill-body-sha256" => c.prefill_body_sha256 = Some(value),
                "--checkpoint-hash" => c.checkpoint_hash = Some(value),
                "--build-hash-field" => c.build_hash_field = Some(value),
                "--checkpoint-hash-field" => c.checkpoint_hash_field = Some(value),
                "--rank0-log" => c.rank0_log = Some(value),
                "--observe-cmd" => c.observe_cmd.push(value),
                "--observe-timeout" => {
                    c.observe_timeout = value.parse().map_err(|_| "invalid --observe-timeout")?
                }
                "--timeout" => c.timeout = value.parse().map_err(|_| "invalid --timeout")?,
                "--startup-timeout" => {
                    c.startup_timeout = value.parse().map_err(|_| "invalid --startup-timeout")?
                }
                "--output" => c.output = value,
                "--api-key-file" => key_file = Some(value),
                "--ab-urls" => c.ab_urls = vec![value, take()?],
                "--aba-urls" => c.aba_urls = vec![value, take()?, take()?],
                "--ab-rank0-logs" => c.ab_rank0_logs = vec![value, take()?],
                "--aba-rank0-logs" => c.aba_rank0_logs = vec![value, take()?, take()?],
                "--launch-commands" => {
                    c.launch_commands.push(value);
                    while arguments.get(i).is_some_and(|v| !v.starts_with("--")) {
                        c.launch_commands.push(arguments[i].clone());
                        i += 1;
                    }
                }
                _ => {
                    return Err(format!(
                        "unknown benchmark option {flag}; use vade bench --help"
                    ));
                }
            }
        }
        if c.concurrency.is_empty()
            || c.concurrency.contains(&0)
            || c.concurrency
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != c.concurrency.len()
        {
            return Err("--concurrency must contain distinct positive integers".into());
        }
        if c.max_tokens == 0
            || c.repeats == 0
            || [c.timeout, c.startup_timeout, c.observe_timeout]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("tokens, repeats and timeouts must be positive".into());
        }
        for (name, value) in [
            ("--timeout", c.timeout),
            ("--startup-timeout", c.startup_timeout),
            ("--observe-timeout", c.observe_timeout),
        ] {
            timeout_deadline(value, name)?;
        }
        let launches = c.launch_commands.len();
        if launches != 0 && launches != 2 && launches != 3 {
            return Err("--launch-commands needs two commands for ABBA or three for A/B/A'".into());
        }
        if !c.ab_urls.is_empty() && !c.aba_urls.is_empty() {
            return Err("choose --ab-urls or --aba-urls".into());
        }
        if (!c.ab_urls.is_empty() && launches == 3) || (!c.aba_urls.is_empty() && launches == 2) {
            return Err("URL count and launch command count must match".into());
        }
        c.order = if !c.aba_urls.is_empty() || launches == 3 {
            "ABA"
        } else if !c.ab_urls.is_empty() || launches == 2 {
            "ABBA"
        } else {
            "single"
        }
        .into();
        if c.order != "single" && c.prefill_body.is_none() {
            c.prefill_body = Some(
                Path::new(
                    c.bodies_dir
                        .as_ref()
                        .ok_or("A/B comparison requires --prefill-body FILE")?,
                )
                .join("pp8192.body")
                .to_string_lossy()
                .into(),
            );
        }
        if !c.ab_rank0_logs.is_empty() && c.order != "ABBA" {
            return Err("--ab-rank0-logs requires ABBA mode".into());
        }
        if !c.aba_rank0_logs.is_empty() && c.order != "ABA" {
            return Err("--aba-rank0-logs requires A/B/A' mode".into());
        }
        if c.rank0_log.is_some() && (!c.ab_rank0_logs.is_empty() || !c.aba_rank0_logs.is_empty()) {
            return Err("choose --rank0-log or per-variant rank-0 logs".into());
        }
        if !c.self_test {
            if c.url.is_none() && c.ab_urls.is_empty() && c.aba_urls.is_empty() {
                return Err("provide --url, --ab-urls or --aba-urls".into());
            }
            if c.prefill_body.is_some() && c.checkpoint_hash.is_none() {
                return Err("--prefill-body and A/B comparison require --checkpoint-hash".into());
            }
        }
        c.output = Path::new(&c.output)
            .with_extension("json")
            .to_string_lossy()
            .into();
        if key_file.is_none() {
            key_file = c
                .command_env
                .iter()
                .find(|(key, _)| key == "VADE_API_KEY_FILE")
                .map(|(_, value)| value.clone());
        }
        if !c.self_test
            && let Some(file) = key_file
        {
            c.api_key = Some(read_key(&file)?);
        }
        Ok(c)
    }
}

fn append(object: &mut Value, key: &str, values: impl IntoIterator<Item = Value>) {
    let mut list = object.get(key).arr().to_vec();
    list.extend(values);
    object.put(key, a(list));
}
fn err(object: &mut Value, error: impl Into<String>) {
    append(object, "errors", [s(error)]);
}
fn health(url: &str, args: &Config) -> Result<Value, String> {
    stream::get_health_with_cancel(url, args.timeout, args.api_key.as_deref(), args.interrupted)
}
fn nums(values: &[Value], key: &str) -> Vec<f64> {
    values.iter().filter_map(|v| v.get(key).num()).collect()
}
pub(crate) fn percentile(values: &[f64], fraction: f64) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * fraction;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    n(sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo as f64))
}
fn range(values: &[f64], eligible: bool) -> Value {
    if !eligible || values.is_empty() {
        Value::Null
    } else {
        a(vec![
            n(values.iter().copied().fold(f64::INFINITY, f64::min)),
            n(values.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        ])
    }
}
fn median(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    n(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.
    } else {
        sorted[middle]
    })
}
pub(crate) fn cohort_metrics(records: &[Value]) -> Value {
    let sends = nums(records, "send_clock_s");
    let completes = nums(records, "complete_clock_s");
    let elapsed = completes.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - sends.iter().copied().fold(f64::INFINITY, f64::min);
    let ok = records.iter().filter(|v| v.get("ok").yes()).count();
    let valid = ok == records.len();
    let total: u64 = records
        .iter()
        .filter_map(|v| v.get("usage").get("completion_tokens").uint())
        .sum();
    let ttfts = nums(records, "ttft_ms");
    o([
        ("ok", n(ok)),
        ("n", n(records.len())),
        ("wall_ms", n(elapsed * 1000.)),
        (
            "completion_tokens",
            if valid { n(total) } else { Value::Null },
        ),
        (
            "aggregate_tokens_per_s",
            if valid {
                n(total as f64 / elapsed)
            } else {
                Value::Null
            },
        ),
        (
            "ttft_p50_ms",
            if valid {
                percentile(&ttfts, 0.5)
            } else {
                Value::Null
            },
        ),
        (
            "ttft_p95_ms",
            if valid {
                percentile(&ttfts, 0.95)
            } else {
                Value::Null
            },
        ),
        (
            "send_spread_ms",
            n((sends.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - sends.iter().copied().fold(f64::INFINITY, f64::min))
                * 1000.),
        ),
    ])
}

fn finish_cell(
    cell: &mut Value,
    before: Vec<Value>,
    after: Vec<Value>,
    log: Option<&str>,
    mark: Value,
) {
    let flags = observation_flags(&before, &after, "start/end");
    for record in before.iter().chain(&after) {
        if let Some(error) = record.get("error").str() {
            err(
                cell,
                format!(
                    "observe rank={} {}: {error}",
                    encode(record.get("rank")),
                    record.get("phase").str().unwrap_or("unknown")
                ),
            );
        }
    }
    cell.put("observations", o([("start", a(before)), ("end", a(after))]));
    cell.put("flags", a(flags));
    let batch = read_batches(log, &mark);
    let mut bytes = cell.get("server_bytes").clone();
    merge_bytes(&mut bytes, batch.get("server_bytes"));
    cell.put("server_bytes", bytes);
    if let Some(error) = batch.get("error").str() {
        err(cell, error);
    }
    cell.put("batch_log", batch);
    let ok = cell
        .get("ok")
        .uint()
        .unwrap_or(u64::from(cell.get("ok").yes()));
    let eligible = cell.get("errors").arr().is_empty() && Some(ok) == cell.get("n").uint();
    cell.put("eligible", b(eligible));
    if !eligible {
        for key in ["aggregate_tokens_per_s", "ttft_p50_ms", "ttft_p95_ms"] {
            cell.put(key, Value::Null);
        }
        let mut requests = cell.get("requests").arr().to_vec();
        for record in &mut requests {
            record.put("tokens_per_s", Value::Null);
            record.put("writing_tokens_per_s", Value::Null);
            record.put("answer_content_tokens_per_s", Value::Null);
        }
        if matches!(cell.get("requests"), Value::Array(_)) {
            cell.put("requests", a(requests));
        }
        if !matches!(cell.get("tokens_per_s"), Value::Null) {
            cell.put("tokens_per_s", Value::Null);
        }
        cell.put("writing_tokens_per_s", Value::Null);
        cell.put("answer_content_tokens_per_s", Value::Null);
    }
}
fn merge_bytes(destination: &mut Value, source: &Value) {
    if let Value::Object(fields) = source {
        for (key, value) in fields {
            if value
                .integer_cmp(destination.get(key))
                .is_none_or(|order| order.is_ge())
            {
                destination.put(key, value.clone());
            }
        }
    }
}

pub(crate) fn run_cohort(
    url: &str,
    prompts: &[Value],
    count: usize,
    repeat: usize,
    args: &Config,
    log: Option<&str>,
    label: &str,
) -> Result<Value, String> {
    let before = health(url, args)?;
    let identity = validate_health(&before, args, false)?;
    let mark = log_mark(log);
    let start = observe(args, "start", label, count, repeat);
    let gate = Arc::new(stream::Gate::new(count + 1));
    let cohort = thread::scope(|scope| {
        let workers: Vec<_> = (0..count)
            .map(|i| {
                let gate = gate.clone();
                let body = request_body(&prompts[i % prompts.len()], args, args.max_tokens);
                scope.spawn(move || stream_chat(url, body, args, Some(gate)))
            })
            .collect();
        let release = gate.wait(args.timeout);
        let records = workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| "benchmark worker panicked".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        release?;
        Ok::<_, String>(records)
    });
    let end = observe(args, "end", label, count, repeat);
    let records = cohort?;
    let mut cell = cohort_metrics(&records);
    cell.put("concurrency", n(count));
    cell.put("repeat", n(repeat));
    cell.put("health_before", before.clone());
    cell.put("identity", identity.clone());
    cell.put("errors", a(vec![]));
    let drafting_check = drafting::validate_requests(identity.get("decoding"), &records);
    cell.put("requested_drafting_consistent", b(drafting_check.is_ok()));
    if let Err(error) = drafting_check { err(&mut cell, error); }
    match health(url, args).and_then(|after| {
        if validate_health(&after, args, false)? != identity {
            Err("server identity changed during cohort".into())
        } else {
            Ok(after)
        }
    }) {
        Ok(after) => cell.put("health_after", after),
        Err(error) => err(&mut cell, error),
    }
    for (i, record) in records.iter().enumerate() {
        if let Some(error) = record.get("error").str() {
            err(&mut cell, format!("chat {}: {error}", i + 1));
        }
    }
    let mut byte_objects = vec![&before, cell.get("health_after")];
    for record in &records {
        byte_objects.push(record.get("usage"));
        byte_objects.push(record.get("sampling_echo"));
    }
    cell.put("server_bytes", observed_bytes(&byte_objects));
    cell.put("draft_observations", drafting::observations(&records));
    cell.put("requests", a(records));
    finish_cell(&mut cell, start, end, log, mark);
    Ok(cell)
}

fn run_prefill(
    url: &str,
    template: &Value,
    source: &Value,
    args: &Config,
    log: Option<&str>,
    label: &str,
) -> Value {
    let mut probe = o([
        ("source", source.clone()),
        ("expected_prompt_tokens", n(8192)),
        ("cache", b(false)),
        ("runs", a(vec![])),
        ("errors", a(vec![])),
        ("flags", a(vec![])),
    ]);
    let mut identity = Value::Null;
    let mut records = vec![];
    for index in 1..=3 {
        let preparation = health(url, args).and_then(|before| {
            let current = validate_health(&before, args, true)?;
            if identity != Value::Null && identity != current {
                return Err("server identity changed between 8k probes".into());
            }
            identity = current;
            Ok(before)
        });
        let before = match preparation {
            Ok(v) => v,
            Err(error) => {
                err(&mut probe, format!("probe {index}: {error}"));
                break;
            }
        };
        let mark = log_mark(log);
        let start = observe(args, "start", label, 1, index);
        let mut record = stream_chat(
            url,
            request_body(
                template,
                args,
                template.get("max_tokens").uint().unwrap_or(16) as usize,
            ),
            args,
            None,
        );
        let end = observe(args, "end", label, 1, index);
        record.put("probe", n(index));
        record.put("health_before", before.clone());
        record.put("identity", identity.clone());
        record.put("n", n(1));
        record.put("errors", a(vec![]));
        if let Some(error) = record.get("error").str().map(str::to_owned) {
            err(&mut record, error);
        } else if record.get("usage").get("prompt_tokens").uint() != Some(8192) {
            record.put("ok", b(false));
            let count = encode(record.get("usage").get("prompt_tokens"));
            err(
                &mut record,
                format!("reports {count} prompt tokens, expected 8192"),
            );
        }
        match health(url, args) {
            Ok(after) => {
                let extra = observed_bytes(&[&before, &after]);
                let mut bytes = record.get("server_bytes").clone();
                // Python's prefill health update replaces equal-name counters at this boundary.
                if let Value::Object(fields) = extra {
                    for (key, value) in fields {
                        bytes.put(&key, value);
                    }
                }
                record.put("server_bytes", bytes);
                record.put("health_after", after.clone());
                match validate_health(&after, args, true) {
                    Ok(current) if current == identity => {}
                    Ok(_) => err(&mut record, "server identity changed during 8k probe"),
                    Err(error) => err(&mut record, error),
                }
            }
            Err(error) => err(&mut record, error),
        }
        if let Err(error) = drafting::validate_requests(identity.get("decoding"), std::slice::from_ref(&record)) {
            err(&mut record, error);
        }
        finish_cell(&mut record, start, end, log, mark);
        append(
            &mut probe,
            "errors",
            record.get("errors").arr().iter().map(|e| {
                s(format!(
                    "probe {index}: {}",
                    e.str().unwrap_or("unknown error")
                ))
            }),
        );
        append(&mut probe, "flags", record.get("flags").arr().to_vec());
        records.push(record);
    }
    let times = nums(&records, "ttft_ms");
    let eligible = times.len() == 3 && probe.get("errors").arr().is_empty();
    if !eligible {
        for record in &mut records {
            record.put("tokens_per_s", Value::Null);
            record.put("writing_tokens_per_s", Value::Null);
            record.put("answer_content_tokens_per_s", Value::Null);
        }
    }
    probe.put(
        "ok",
        n(records.iter().filter(|r| r.get("ok").yes()).count()),
    );
    probe.put("n", n(3));
    probe.put("identity", identity);
    probe.put("eligible", b(eligible));
    probe.put(
        "median_ms",
        if eligible {
            median(&times)
        } else {
            Value::Null
        },
    );
    probe.put("range_ms", range(&times, eligible));
    probe.put("runs", a(records));
    probe
}

fn run_arm(
    label: &str,
    url: &str,
    prompts: &[Value],
    prefill: Option<&(Value, Value)>,
    args: &Config,
    log: Option<&str>,
) -> Value {
    let mut arm = o([
        ("label", s(label)),
        ("url", s(url)),
        ("cells", a(vec![])),
        ("expected_repeats", n(args.repeats)),
        ("prefill_8k", Value::Null),
        ("errors", a(vec![])),
        ("flags", a(vec![])),
        (
            "mtp_policy",
            s("server defaults unless --draft overrides; requested mode, server-reported health fields and response echoes recorded separately"),
        ),
    ]);
    let result = (|| -> Result<(), String> {
        let initial = wait_for_checkpoint(url, args)?;
        let resolved = resolve_model(args, &initial);
        let args = &resolved;
        let identity = validate_health(&initial, args, prefill.is_some())?;
        arm.put("model", s(&args.model));
        arm.put("server_build_sha", identity.get("server_build_sha").clone());
        arm.put("decoding", identity.get("decoding").clone());
        arm.put("health_initial", initial);
        arm.put("identity", identity.clone());
        for count in &args.concurrency {
            for repeat in 1..=args.repeats {
                if (args.interrupted)() {
                    return Err("benchmark interrupted".into());
                }
                crate::stderr_line(format_args!("{label} concurrency={count} repeat={repeat}"));
                let cell = run_cohort(url, prompts, *count, repeat, args, log, label)?;
                append(&mut arm, "errors", cell.get("errors").arr().to_vec());
                append(&mut arm, "flags", cell.get("flags").arr().to_vec());
                append(&mut arm, "cells", [cell.clone()]);
                if !cell.get("requested_drafting_consistent").yes() {
                    return Err("request drafting provenance disagrees with receipt header; remaining cells refused".into());
                }
                if cell.get("identity") != &identity {
                    return Err("server identity changed between matrix cells".into());
                }
                if cell.get("requests").arr().iter().any(|r| {
                    r.get("error")
                        .str()
                        .is_some_and(|e| e.contains("draft") || e.contains("sampling echo"))
                }) {
                    return Err(
                        "server contradicts requested sampling/drafting; remaining cells refused".into(),
                    );
                }
            }
        }
        if let Some((template, source)) = prefill {
            crate::stderr_line(format_args!("{label} 8k TTFT: three uncached probes"));
            let probe = run_prefill(url, template, source, args, log, label);
            append(&mut arm, "errors", probe.get("errors").arr().to_vec());
            append(&mut arm, "flags", probe.get("flags").arr().to_vec());
            if probe.get("identity") != &identity {
                err(
                    &mut arm,
                    "server identity changed between matrix and 8k probes",
                );
            }
            arm.put("prefill_8k", probe);
        }
        Ok(())
    })();
    if let Err(error) = result {
        err(&mut arm, error);
    }
    let requests: Vec<_> = arm.get("cells").arr().iter()
        .flat_map(|cell| cell.get("requests").arr().to_vec())
        .chain(arm.get("prefill_8k").get("runs").arr().iter().cloned()).collect();
    arm.put("draft_observations", drafting::observations(&requests));
    if let Err(error) = drafting::validate_requests(arm.get("decoding"), &requests) {
        let mut decoding = arm.get("decoding").clone();
        decoding.put("drafting", s("unknown"));
        decoding.put("source", s("request drafting provenance disagrees with receipt header"));
        arm.put("decoding", decoding);
        err(&mut arm, error);
    }
    let eligible = arm.get("errors").arr().is_empty()
        && arm.get("cells").arr().len() == args.concurrency.len() * args.repeats;
    arm.put("eligible", b(eligible));
    arm
}

pub(crate) fn compare(
    a_value: &Value,
    b_value: &Value,
    a_prime: &Value,
    eligible: bool,
    lower: bool,
) -> Value {
    let mut row = o([
        ("A", a_value.clone()),
        ("B", b_value.clone()),
        ("A_prime", a_prime.clone()),
        ("baseline_mean", Value::Null),
        ("gain", Value::Null),
        ("noise", Value::Null),
        ("margin", Value::Null),
        ("verdict", s("INVALID")),
    ]);
    if let (Some(left), Some(candidate), Some(right)) =
        (a_value.num(), b_value.num(), a_prime.num())
    {
        let mean = (left + right) / 2.;
        let gain = if lower {
            mean - candidate
        } else {
            candidate - mean
        };
        let noise = (left - right).abs();
        let margin = 0.02 * mean;
        row.put("baseline_mean", n(mean));
        row.put("gain", n(gain));
        row.put("noise", n(noise));
        row.put("margin", n(margin));
        row.put(
            "verdict",
            s(if eligible {
                if gain > noise + margin {
                    "faster"
                } else {
                    "not faster"
                }
            } else {
                "INVALID"
            }),
        );
    }
    row
}

pub(crate) fn summarize(arm: &Value, count: usize) -> Value {
    let cells: Vec<_> = arm
        .get("cells")
        .arr()
        .iter()
        .filter(|c| c.get("concurrency").uint() == Some(count as u64))
        .cloned()
        .collect();
    let rates = nums(&cells, "aggregate_tokens_per_s");
    let requests: Vec<_> = cells
        .iter()
        .flat_map(|c| c.get("requests").arr().to_vec())
        .collect();
    let ttfts = nums(&requests, "ttft_ms");
    let expected = arm
        .get("expected_repeats")
        .uint()
        .unwrap_or(cells.len() as u64) as usize;
    let eligible = !cells.is_empty()
        && cells.len() == expected
        && rates.len() == cells.len()
        && cells.iter().all(|c| c.get("eligible").yes());
    o([
        ("concurrency", n(count)),
        (
            "median_tokens_per_s",
            if eligible {
                median(&rates)
            } else {
                Value::Null
            },
        ),
        ("range_tokens_per_s", range(&rates, eligible)),
        (
            "ttft_p50_ms",
            if eligible {
                percentile(&ttfts, 0.5)
            } else {
                Value::Null
            },
        ),
        (
            "ttft_p95_ms",
            if eligible {
                percentile(&ttfts, 0.95)
            } else {
                Value::Null
            },
        ),
        (
            "ok",
            n(cells.iter().filter_map(|c| c.get("ok").uint()).sum::<u64>()),
        ),
        ("n", n(count * expected)),
        (
            "flags",
            a(cells
                .iter()
                .flat_map(|c| c.get("flags").arr().to_vec())
                .collect()),
        ),
        ("eligible", b(eligible)),
        (
            "server_bytes",
            merge_flat_bytes(cells.iter().map(|c| c.get("server_bytes"))),
        ),
    ])
}
fn merge_flat_bytes<'a>(objects: impl IntoIterator<Item = &'a Value>) -> Value {
    let mut out = o([]);
    for obj in objects {
        merge_bytes(&mut out, obj);
    }
    out
}

fn baseline_flags(a_arm: &Value, a_prime: &mut Value) {
    let mut cells = a_prime.get("cells").arr().to_vec();
    let mut all = vec![];
    for cell in &mut cells {
        let left = a_arm.get("cells").arr().iter().find(|v| {
            v.get("concurrency") == cell.get("concurrency") && v.get("repeat") == cell.get("repeat")
        });
        if let Some(left) = left {
            let flags = between_flags(
                left,
                cell,
                &format!(
                    "c{} r{}",
                    encode(cell.get("concurrency")),
                    encode(cell.get("repeat"))
                ),
            );
            append(cell, "flags", flags.clone());
            all.extend(flags);
        }
    }
    a_prime.put("cells", a(cells));
    let mut probe = a_prime.get("prefill_8k").clone();
    let mut runs = probe.get("runs").arr().to_vec();
    for (left, right) in a_arm
        .get("prefill_8k")
        .get("runs")
        .arr()
        .iter()
        .zip(&mut runs)
    {
        let flags = between_flags(
            left,
            right,
            &format!("8k probe={}", encode(right.get("probe"))),
        );
        append(right, "flags", flags.clone());
        append(&mut probe, "flags", flags.clone());
        all.extend(flags);
    }
    if probe != Value::Null {
        probe.put("runs", a(runs));
        a_prime.put("prefill_8k", probe);
    }
    append(a_prime, "flags", all);
}
fn between_flags(left: &Value, right: &Value, context: &str) -> Vec<Value> {
    let mut flags = vec![];
    for (p, q) in [("start", "start"), ("end", "end"), ("end", "start")] {
        flags.extend(observation_flags(
            left.get("observations").get(p).arr(),
            right.get("observations").get(q).arr(),
            &format!("A/A_prime {context} {p}/{q}"),
        ));
    }
    flags
}
fn candidate_mean(summaries: &[Value], key: &str) -> Value {
    let eligible = summaries
        .iter()
        .all(|s| s.get("eligible").yes() && s.get(key).num().is_some());
    o([
        (
            key,
            if eligible {
                n(summaries
                    .iter()
                    .filter_map(|s| s.get(key).num())
                    .sum::<f64>()
                    / summaries.len() as f64)
            } else {
                Value::Null
            },
        ),
        ("eligible", b(eligible)),
        (
            "ok",
            n(summaries
                .iter()
                .filter_map(|s| s.get("ok").uint())
                .sum::<u64>()),
        ),
        (
            "n",
            n(summaries
                .iter()
                .filter_map(|s| s.get("n").uint())
                .sum::<u64>()),
        ),
        (
            "flags",
            a(summaries
                .iter()
                .flat_map(|s| s.get("flags").arr().to_vec())
                .collect()),
        ),
        (
            "server_bytes",
            observed_bytes(
                &summaries
                    .iter()
                    .map(|s| s.get("server_bytes"))
                    .collect::<Vec<_>>(),
            ),
        ),
        ("runs", a(summaries.to_vec())),
    ])
}
pub(crate) fn receipts(arms: &[Value], args: &Config) -> Vec<Value> {
    if arms.len() != 3 && arms.len() != 4 {
        return vec![];
    }
    let identity_ok = arms.iter().all(|a| a.get("eligible").yes())
        && arms.iter().all(|arm| arm.get("model") == arms[0].get("model"))
        && arms[0].get("identity") == arms[arms.len() - 1].get("identity")
        && (arms.len() != 4 || arms[1].get("identity") == arms[2].get("identity"));
    let mut rows = vec![];
    for count in &args.concurrency {
        let runs: Vec<_> = arms.iter().map(|a| summarize(a, *count)).collect();
        rows.push(receipt_row(
            &runs,
            "median_tokens_per_s",
            &format!("c{count}"),
            "tok/s",
            false,
            identity_ok,
        ));
    }
    if args.prefill_body.is_some() {
        let runs: Vec<_> = arms
            .iter()
            .map(|arm| {
                let p = arm.get("prefill_8k");
                o([
                    ("median_ms", p.get("median_ms").clone()),
                    ("eligible", b(p.get("eligible").yes())),
                    ("ok", p.get("ok").uint().map(n).unwrap_or(n(0))),
                    ("n", n(3)),
                    ("flags", a(p.get("flags").arr().to_vec())),
                    (
                        "server_bytes",
                        observed_bytes(
                            &p.get("runs")
                                .arr()
                                .iter()
                                .map(|r| r.get("server_bytes"))
                                .collect::<Vec<_>>(),
                        ),
                    ),
                ])
            })
            .collect();
        rows.push(receipt_row(
            &runs,
            "median_ms",
            "8k TTFT",
            "ms",
            true,
            identity_ok,
        ));
    }
    rows
}
fn receipt_row(
    runs: &[Value],
    key: &str,
    cell: &str,
    unit: &str,
    lower: bool,
    identity_ok: bool,
) -> Value {
    let summaries = vec![
        runs[0].clone(),
        candidate_mean(&runs[1..runs.len() - 1], key),
        runs[runs.len() - 1].clone(),
    ];
    let failed = !summaries.iter().all(|s| s.get("eligible").yes());
    let mut row = compare(
        if failed {
            &Value::Null
        } else {
            summaries[0].get(key)
        },
        if failed {
            &Value::Null
        } else {
            summaries[1].get(key)
        },
        if failed {
            &Value::Null
        } else {
            summaries[2].get(key)
        },
        identity_ok && summaries.iter().all(|s| s.get("flags").arr().is_empty()),
        lower,
    );
    row.put("cell", s(cell));
    row.put("unit", s(unit));
    row.put("summaries", a(summaries));
    row.put("failed", b(failed));
    row
}

struct ArmPlan {
    label: &'static str,
    url: String,
    log: Option<String>,
    command: Option<String>,
}
fn run_plan(args: &Config) -> Vec<ArmPlan> {
    let urls = if !args.ab_urls.is_empty() {
        args.ab_urls.clone()
    } else if !args.aba_urls.is_empty() {
        args.aba_urls.clone()
    } else {
        vec![args.url.clone().unwrap_or_default(); args.launch_commands.len().max(1)]
    };
    let logs = if !args.ab_rank0_logs.is_empty() {
        args.ab_rank0_logs.iter().cloned().map(Some).collect()
    } else if !args.aba_rank0_logs.is_empty() {
        args.aba_rank0_logs.iter().cloned().map(Some).collect()
    } else {
        vec![args.rank0_log.clone(); urls.len()]
    };
    let plan: &[(&str, usize)] = match args.order.as_str() {
        "ABBA" => &[("A", 0), ("B", 1), ("B_prime", 1), ("A_prime", 0)],
        "ABA" => &[("A", 0), ("B", 1), ("A_prime", 2)],
        _ => &[("single", 0)],
    };
    plan.iter()
        .map(|(label, index)| ArmPlan {
            label,
            url: urls[*index].clone(),
            log: logs[*index].clone(),
            command: args.launch_commands.get(*index).cloned(),
        })
        .collect()
}
pub(crate) fn execute(args: &Config) -> Value {
    let mut result = o([
        ("schema_version", n(4)),
        ("started_utc", s(utc_now())),
        ("method", s(METHOD)),
        ("margin_rule", s(MARGIN)),
        ("sampling", stream::sampling()),
        ("order", s(&args.order)),
        ("observation_ranks", n(args.observe_cmd.len())),
        ("concurrency", a(args.concurrency.iter().map(n).collect())),
        ("max_tokens", n(args.max_tokens)),
        ("repeats", n(args.repeats)),
        (
            "checkpoint_pin",
            args.checkpoint_hash.as_ref().map(s).unwrap_or(Value::Null),
        ),
        ("arms", a(vec![])),
        ("receipts", a(vec![])),
        ("errors", a(vec![])),
    ]);
    let outcome = (|| -> Result<(), String> {
        let (prompts, source) = load_prompts(args)?;
        result.put("prompt_source", source);
        result.put("prompt_count", n(prompts.len()));
        result.put(
            "prompt_assignment",
            s("same ordered prompts per repeat and arm, cyclic if concurrency exceeds corpus"),
        );
        let prefill = if let Some(file) = &args.prefill_body {
            let raw = fs::read(file).map_err(|e| e.to_string())?;
            let hash = digest(&raw);
            if args
                .prefill_body_sha256
                .as_ref()
                .is_some_and(|expected| expected.to_lowercase() != hash)
            {
                return Err("--prefill-body-sha256 does not match file".into());
            }
            Some((parse(&raw)?, o([("file", s(file)), ("sha256", s(hash))])))
        } else {
            None
        };
        let mut arms = vec![];
        for run in run_plan(args) {
            if (args.interrupted)() {
                err(&mut result, "benchmark interrupted");
                break;
            }
            let arm = (|| -> Result<Value, String> {
                let mut log = run.log.clone();
                let mut launched = None;
                if let Some(command) = &run.command {
                    if stream::responds(&run.url, args.timeout.min(2.))? {
                        return Err("launch URL already responds; stop that server first".into());
                    }
                    let launch_path =
                        Path::new(&args.output).with_extension(format!("{}.server.log", run.label));
                    let mut process =
                        Launched::start_with_env(command, &launch_path, &args.command_env)?;
                    process.wait_ready(&run.url, args)?;
                    if log.is_none() {
                        log = Some(launch_path.to_string_lossy().into());
                    }
                    launched = Some(process);
                }
                let mut arm = run_arm(
                    run.label,
                    &run.url,
                    &prompts,
                    prefill.as_ref(),
                    args,
                    log.as_deref(),
                );
                arm.put("launched_by_runner", b(run.command.is_some()));
                if let Some(process) = launched.as_mut() {
                    process.stop()?;
                }
                drop(launched);
                Ok(arm)
            })()
            .unwrap_or_else(|error| {
                o([
                    ("label", s(run.label)),
                    ("url", s(&run.url)),
                    ("cells", a(vec![])),
                    ("expected_repeats", n(args.repeats)),
                    ("errors", a(vec![s(error)])),
                    ("flags", a(vec![])),
                    ("eligible", b(false)),
                ])
            });
            arms.push(arm);
            if (args.interrupted)() {
                err(&mut result, "benchmark interrupted");
                break;
            }
        }
        if arms.len() == 3 || arms.len() == 4 {
            if arms.iter().any(|arm| arm.get("model") != arms[0].get("model")) {
                err(&mut result, "comparison server model IDs differ; all arms must serve the same model");
            }
            if arms[0].get("identity") != arms[arms.len() - 1].get("identity") {
                err(
                    &mut result,
                    "A and A_prime server build/checkpoint identities differ",
                );
            }
            if arms.len() == 4 && arms[1].get("identity") != arms[2].get("identity") {
                err(
                    &mut result,
                    "B and B_prime server build/checkpoint identities differ",
                );
            }
            let left = arms[0].clone();
            baseline_flags(&left, arms.last_mut().unwrap());
        }
        if arms.len() == 1 {
            result.put("model", arms[0].get("model").clone());
            result.put("server_build_sha", arms[0].get("server_build_sha").clone());
            result.put("decoding", arms[0].get("decoding").clone());
            result.put("draft_observations", arms[0].get("draft_observations").clone());
        }
        result.put("receipts", a(receipts(&arms, args)));
        result.put("arms", a(arms));
        Ok(())
    })();
    if let Err(error) = outcome {
        err(&mut result, error);
    }
    let ready = !result.get("arms").arr().is_empty()
        && result
            .get("arms")
            .arr()
            .iter()
            .all(|a| a.get("eligible").yes())
        && result.get("errors").arr().is_empty();
    let flagged = result
        .get("arms")
        .arr()
        .iter()
        .any(|a| !a.get("flags").arr().is_empty());
    result.put(
        "status",
        s(if !ready {
            "FAIL"
        } else if flagged {
            "PARTIAL"
        } else {
            "READY"
        }),
    );
    result
}

fn number(value: &Value) -> String {
    value
        .num()
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "unreported".into())
}
fn bytes_text(value: &Value) -> String {
    if let Value::Object(fields) = value {
        let mut fields = fields.clone();
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        if !fields.is_empty() {
            return fields
                .iter()
                .map(|(k, v)| format!("{k}={}", encode(v)))
                .collect::<Vec<_>>()
                .join(",");
        }
    }
    "unreported".into()
}
fn ok_text(value: &Value) -> String {
    format!(
        "ok={}/{}",
        value
            .get("ok")
            .uint()
            .unwrap_or(u64::from(value.get("ok").yes())),
        value.get("n").uint().unwrap_or(1)
    )
}
fn details(cell: &Value, lines: &mut Vec<String>) {
    for phase in ["start", "end"] {
        for record in cell.get("observations").get(phase).arr() {
            lines.push(format!(
                "  observe {phase} rank={}: {}",
                encode(record.get("rank")),
                encode(record)
            ));
        }
    }
    for flag in cell.get("flags").arr() {
        lines.push(format!("  FLAG {}", flag.str().unwrap_or("unknown")));
    }
    let log = cell.get("batch_log");
    if log != &Value::Null {
        lines.push(format!(
            "  flash_serve batch rows=N steps={} rows_per_step={} histogram={} error={}",
            encode(log.get("step_count")),
            encode(log.get("rows_per_step")),
            encode(log.get("row_histogram")),
            encode(log.get("error"))
        ));
    }
}
fn render(result: &Value, args: &Config) -> String {
    let mut lines=vec![result.get("status").str().unwrap_or("FAIL").into(),format!("CLIENT wall-clock: {METHOD}"),format!("Order: {}",args.order),format!("A/B rule: {MARGIN}"),"No warmups/discarded repeats; all requests cache=false, thinking=false, ignore_eos=true.".into(),"Bytes are health boundary observations / reported usage / supplied log counters; no inferred GPU peak.".into(),"FAILED cells publish no performance numbers; hardware flags invalidate comparison verdicts.".into(),format!("Observation ranks={} (zero means hardware unobserved).",args.observe_cmd.len())];
    for arm in result.get("arms").arr() {
        lines.push(format!("{} model={} server_build_sha={} decoding: {}", arm.get("label").str().unwrap_or("unknown"),
            encode(arm.get("model")), encode(arm.get("server_build_sha")), encode(arm.get("decoding"))));
        lines.push(format!("  server-reported request echoes: {}", encode(arm.get("draft_observations"))));
    }
    if !result.get("receipts").arr().is_empty() {
        lines.push("cell | unit | A | B (mean in ABBA) | A' | gain | noise | 2% margin | verdict | ok A / B / A' | observed server bytes A / B / A'".into());
        for row in result.get("receipts").arr() {
            let summaries = row.get("summaries").arr();
            let bytes = summaries
                .iter()
                .map(|s| bytes_text(s.get("server_bytes")))
                .collect::<Vec<_>>()
                .join(" / ");
            let counts = summaries
                .iter()
                .map(ok_text)
                .collect::<Vec<_>>()
                .join(" / ");
            if row.get("failed").yes() {
                lines.push(format!(
                    "{} FAILED | {counts} | bytes={bytes}",
                    row.get("cell").str().unwrap_or("unknown")
                ));
            } else {
                lines.push(format!(
                    "{} | {} | {} | {} | {counts} | {bytes}",
                    row.get("cell").str().unwrap_or("unknown"),
                    row.get("unit").str().unwrap_or("unknown"),
                    ["A", "B", "A_prime", "gain", "noise", "margin"]
                        .map(|k| number(row.get(k)))
                        .join(" | "),
                    row.get("verdict").str().unwrap_or("INVALID")
                ));
            }
        }
    } else {
        lines.push("arm | concurrency | ok | median tok/s | range tok/s | TTFT p50/p95 ms | observed server bytes".into());
        for arm in result.get("arms").arr() {
            for count in &args.concurrency {
                let summary = summarize(arm, *count);
                let label = arm.get("label").str().unwrap_or("unknown");
                if !summary.get("eligible").yes() {
                    lines.push(format!(
                        "{label} c{count} FAILED {} bytes={}",
                        ok_text(&summary),
                        bytes_text(summary.get("server_bytes"))
                    ));
                } else {
                    lines.push(format!(
                        "{label} | {count} | {} | {} | {} | {}/{} | {}",
                        ok_text(&summary),
                        number(summary.get("median_tokens_per_s")),
                        encode(summary.get("range_tokens_per_s")),
                        number(summary.get("ttft_p50_ms")),
                        number(summary.get("ttft_p95_ms")),
                        bytes_text(summary.get("server_bytes"))
                    ));
                }
            }
        }
    }
    for arm in result.get("arms").arr() {
        let label = arm.get("label").str().unwrap_or("unknown");
        lines.push(format!("{label} identity: {}", encode(arm.get("identity"))));
        for cell in arm.get("cells").arr() {
            let title = format!(
                "{label} c{} r{}",
                encode(cell.get("concurrency")),
                encode(cell.get("repeat"))
            );
            let bytes = bytes_text(cell.get("server_bytes"));
            if !cell.get("eligible").yes() {
                lines.push(format!("{title} FAILED {} bytes={bytes}", ok_text(cell)));
                for (i, record) in cell.get("requests").arr().iter().enumerate() {
                    lines.push(format!(
                        "  chat={} HTTP={} ok={}/1 error={}",
                        i + 1,
                        encode(record.get("http_status")),
                        u8::from(record.get("ok").yes()),
                        encode(record.get("error"))
                    ));
                }
            } else {
                lines.push(format!(
                    "{title} {}: tokens={} wall_ms={} tok/s={} bytes={bytes}",
                    ok_text(cell),
                    encode(cell.get("completion_tokens")),
                    number(cell.get("wall_ms")),
                    number(cell.get("aggregate_tokens_per_s"))
                ));
                for (i, record) in cell.get("requests").arr().iter().enumerate() {
                    let echo = record.get("sampling_echo");
                    let sampling = if matches!(echo,Value::Object(v) if !v.is_empty()) {
                        echo
                    } else {
                        record.get("request_sampling")
                    };
                    lines.push(format!("  chat={} ok=1/1 HTTP=200 tok/s={} writing_tok/s={} answer_content_tok/s={} TTFT_ms={} bytes={bytes} usage={} sampling_source={} sampling={}",i+1,number(record.get("tokens_per_s")),number(record.get("writing_tokens_per_s")),number(record.get("answer_content_tokens_per_s")),number(record.get("ttft_ms")),encode(record.get("usage")),record.get("sampling_source").str().unwrap_or("request"),encode(sampling)));
                }
            }
            details(cell, &mut lines);
        }
        let probe = arm.get("prefill_8k");
        if probe != &Value::Null {
            let bytes = observed_bytes(
                &probe
                    .get("runs")
                    .arr()
                    .iter()
                    .map(|r| r.get("server_bytes"))
                    .collect::<Vec<_>>(),
            );
            if probe.get("eligible").yes() {
                lines.push(format!(
                    "{label} 8k TTFT {} median_ms={} range_ms={} bytes={} identity={}",
                    ok_text(probe),
                    number(probe.get("median_ms")),
                    encode(probe.get("range_ms")),
                    bytes_text(&bytes),
                    encode(probe.get("identity"))
                ));
            } else {
                lines.push(format!(
                    "{label} 8k TTFT FAILED {} bytes={}",
                    ok_text(probe),
                    bytes_text(&bytes)
                ));
            }
            for record in probe.get("runs").arr() {
                if probe.get("eligible").yes() {
                    lines.push(format!(
                        "  probe={} {} TTFT_ms={} bytes={} usage={} sampling={}",
                        encode(record.get("probe")),
                        ok_text(record),
                        number(record.get("ttft_ms")),
                        bytes_text(record.get("server_bytes")),
                        encode(record.get("usage")),
                        encode(record.get("sampling_echo"))
                    ));
                } else {
                    lines.push(format!(
                        "  probe={} FAILED {} HTTP={} errors={}",
                        encode(record.get("probe")),
                        ok_text(record),
                        encode(record.get("http_status")),
                        encode(record.get("errors"))
                    ));
                }
                details(record, &mut lines);
            }
        }
        for error in arm.get("errors").arr() {
            lines.push(format!(
                "ERROR {label}: {}",
                error.str().unwrap_or("unknown")
            ));
        }
    }
    for error in result.get("errors").arr() {
        lines.push(format!("ERROR: {}", error.str().unwrap_or("unknown")));
    }
    lines.join("\n") + "\n"
}
pub(crate) fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32, String> {
    if arguments
        .iter()
        .any(|a| matches!(a.as_str(), "--help" | "-h"))
    {
        crate::write_help(HELP)?;
        return Ok(0);
    }
    let mut args = Config::parse(arguments)?;
    args.interrupted = interrupted;
    if args.self_test {
        self_test::self_test(&args)?;
        return Ok(0);
    }
    let result = execute(&args);
    let table = render(&result, &args);
    let output = Path::new(&args.output);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(output, encode(&result) + "\n").map_err(|e| e.to_string())?;
    let text = output.with_extension("txt");
    fs::write(&text, &table).map_err(|e| e.to_string())?;
    crate::write_output(format_args!(
        "{table}JSON: {}\nTABLE: {}\n",
        output.display(),
        text.display()
    ), false)?;
    Ok(if result.get("status").str() == Some("READY") {
        0
    } else {
        1
    })
}
