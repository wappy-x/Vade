//! Development workload suite behind `vade bench --suite`.
//!
//! `lib.rs` includes this file as `suite`; the standalone binary is `kernel-bench`.
//! This module coordinates workloads, `http` requests and `report` receipts.
//! The default benchmark protocol lives in the library's `timed` module.
pub(crate) mod http;
mod report;
use engine_tokenizer::{
    Tokenizer,
    chat::{self, Message, Options, Role},
};
use http::{Endpoint, Reply};
use report::{Record, Report};
use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

pub(crate) type Result<T> = std::result::Result<T, String>;
const PROSE: &str = "A friend is taking their first overnight train trip. Write a warm, practical letter about packing, finding the platform, sleeping on board, and enjoying the morning view. Use complete paragraphs and concrete details.";
const CODE: &str = "Edit this Rust function so it returns the first matching index without panicking on an empty slice. Explain the bug and show the corrected function and three examples. fn find(xs: &[i32], target: i32) -> usize { for i in 0..=xs.len() { if xs[i] == target { return i; } } 0 }";
const REASON: &str = "A library has three rooms. The blue room holds twice as many people as the green room. The red room holds twelve fewer than the blue room. Together they hold 108 people. Work through the equations, find each capacity, and check the answer.";
const PASSAGE: &str = "The town library opens beside the river each morning. Readers return books at the wooden desk, while volunteers arrange maps and repair worn covers. Upstairs, a quiet room looks across the bridge toward the station. The librarian keeps a notebook of questions that visitors ask, then uses those notes to choose new books for the shelves. Every Friday, neighbors meet to discuss what they have read and recommend stories to one another. ";
const LONG_END: &str =
    "\nSummarize the setting and the activities in these notes in a detailed paragraph.";
pub(crate) const HELP: &str = "vade bench --suite http://HOST:PORT [--tokenizer MODEL_DIR] [--output FILE] [--model ID] [--precision BF16] [--sparks 1] [--label before] [--timeout-secs 90] [--budget-secs 840] [--repeats 3] [--max-tokens 32] [--concurrency 1..32] [--server-pid PID] [--memory-file FILE] [--long-target N] [--long-output N] [--cache-target N] [--only all|decode|long|concurrency|cache|priority|followup|shared-system|long-output|request]";

pub struct Config {
    endpoint: Endpoint,
    model: String,
    precision: String,
    sparks: usize,
    label: String,
    output: String,
    tokenizer: Option<String>,
    timeout: Duration,
    budget: Duration,
    repeats: usize,
    max_tokens: usize,
    server_pid: Option<u32>,
    memory_file: Option<String>,
    only: String,
    long_target: Option<usize>,
    long_output: usize,
    cache_target: Option<usize>,
    concurrency: Option<usize>,
}
impl Config {
    fn parse(arguments: &[String]) -> Result<Self> {
        let mut args = arguments.iter().cloned();
        let url = args.next().ok_or(HELP)?;
        let mut c = Self {
            endpoint: Endpoint::parse(&url)?,
            model: "qwen3-8b".into(),
            precision: "BF16".into(),
            sparks: 1,
            label: "baseline".into(),
            output: "bench-results.json".into(),
            tokenizer: std::env::var("MODEL_DIR").ok(),
            timeout: Duration::from_secs(90),
            budget: Duration::from_secs(840),
            repeats: 3,
            max_tokens: 32,
            server_pid: None,
            memory_file: None,
            only: "all".into(),
            long_target: None,
            long_output: 2,
            cache_target: None,
            concurrency: None,
        };
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let number = || {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid {flag}"))
            };
            match flag.as_str() {
                "--model" => c.model = value,
                "--precision" => c.precision = value,
                "--sparks" => c.sparks = number()?,
                "--label" => c.label = value,
                "--output" => c.output = value,
                "--tokenizer" => c.tokenizer = Some(value),
                "--timeout-secs" => c.timeout = Duration::from_secs(number()? as u64),
                "--budget-secs" => c.budget = Duration::from_secs(number()? as u64),
                "--repeats" => c.repeats = number()?,
                "--max-tokens" => c.max_tokens = number()?,
                "--server-pid" => c.server_pid = Some(value.parse().map_err(|_| "invalid PID")?),
                "--memory-file" => c.memory_file = Some(value),
                "--only" => c.only = value,
                "--long-target" => c.long_target = Some(number()?),
                "--long-output" => c.long_output = number()?,
                "--cache-target" => c.cache_target = Some(number()?),
                "--concurrency" => c.concurrency = Some(number()?),
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        if c.sparks == 0
            || c.repeats == 0
            || c.repeats > 10
            || c.max_tokens < 2
            || c.max_tokens > 8192
            || !(2..=8192).contains(&c.long_output)
            || c.long_target == Some(0)
            || c.cache_target == Some(0)
            || c.concurrency.is_some_and(|n| !(1..=32).contains(&n))
            || c.timeout.is_zero()
            || c.budget.is_zero()
        {
            return Err("invalid zero/out-of-range setting".into());
        }
        if ![
            "all",
            "decode",
            "long",
            "concurrency",
            "cache",
            "priority",
            "long-output",
            "request",
            "followup",
            "shared-system",
        ]
        .contains(&c.only.as_str())
        {
            return Err("invalid --only case".into());
        }
        Ok(c)
    }
    fn wants(&self, group: &str) -> bool {
        self.only == "all" || self.only == group
    }
    fn body(
        &self,
        messages: &[Message],
        max_tokens: usize,
        priority: u8,
        cache: bool,
        session: Option<&str>,
    ) -> String {
        let messages = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::Developer => "developer",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                format!(
                    "{{\"role\":{},\"content\":{}}}",
                    chat::quote(role),
                    chat::quote(&m.content)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        if self.only == "followup" || self.only == "shared-system" {
            return format!(
                "{{\"model\":{},\"messages\":[{}],\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"max_tokens\":{}}}",
                chat::quote(&self.model), messages, max_tokens
            );
        }
        let session = session
            .map(|s| format!(",\"session\":{},\"fork\":true", chat::quote(s)))
            .unwrap_or_default();
        // GLM has no thinking-off form: ask low effort there, as the cards do.
        let thinking = if crate::timed::stream::glm_model(&self.model) { "\"reasoning_effort\":\"low\"" } else { "\"enable_thinking\":false" };
        format!(
            "{{\"model\":{},\"messages\":[{}],\"stream\":true,{thinking},\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"max_tokens\":{},\"priority\":{},\"cache\":{}{}}}",
            chat::quote(&self.model),
            messages,
            max_tokens,
            priority,
            cache,
            session
        )
    }
}
fn messages(text: &str) -> Vec<Message> {
    vec![Message::new(Role::User, text)]
}
fn repeat_prompt(prompt: &[Message], repeat: usize) -> Vec<Message> {
    let mut prompt = prompt.to_vec();
    if repeat > 0 {
        prompt.last_mut().unwrap().content.push_str(&format!("\nFor this version, include a concrete example numbered {}.", repeat + 1));
    }
    prompt
}
fn count(tokenizer: Option<&Tokenizer>, messages: &[Message]) -> Option<usize> {
    tokenizer
        .and_then(|t| {
            chat::encode(
                t,
                messages,
                &Options {
                    enable_thinking: false,
                    ..Default::default()
                },
            )
            .ok()
        })
        .map(|t| t.len())
}
fn long_prompt(target: usize, tokenizer: Option<&Tokenizer>) -> (Vec<Message>, Option<usize>) {
    // Repeated prose, truncated at word boundaries; actual chat-template count
    // comes from the checkpoint tokenizer or final server usage, never bytes/4.
    let text = PASSAGE.repeat(target.div_ceil(50) + 1);
    let words = text.split_whitespace().collect::<Vec<_>>();
    let mut low = 0;
    let mut high = words.len();
    if tokenizer.is_some() {
        while low + 1 < high {
            let mid = (low + high) / 2;
            let prompt = messages(&format!("{}{}", words[..mid].join(" "), LONG_END));
            if count(tokenizer, &prompt).unwrap() <= target {
                low = mid;
            } else {
                high = mid;
            }
        }
    } else {
        low = (target * 3 / 4).min(words.len());
    }
    let prompt = messages(&format!("{}{}", words[..low].join(" "), LONG_END));
    let actual = count(tokenizer, &prompt);
    (prompt, actual)
}
fn deadline(c: &Config, end: Instant) -> Instant {
    (Instant::now() + c.timeout).min(end)
}
/// The soft switch that turns thinking off in a portable prompt. GLM's template has none: there it would be prompt text.
fn no_think(model: &str) -> &'static str {
    if crate::timed::stream::glm_model(model) { "" } else { " /no_think" }
}

fn caching_prompt(target: usize, tokenizer: &Tokenizer, role: Role, model: &str) -> Result<Vec<Message>> {
    let text = PASSAGE.repeat(target.div_ceil(50) + 1);
    let words = text.split_whitespace().collect::<Vec<_>>();
    let suffix = if role == Role::System {
        "\nUse these library notes as context for every reply.".to_owned()
    } else {
        format!("\nSummarize these notes in a detailed paragraph.{}", no_think(model))
    };
    let size = |content: &str| -> Result<usize> {
        if role == Role::System {
            tokenizer.encode(content).map(|ids| ids.len()).map_err(|e| e.to_string())
        } else {
            count(Some(tokenizer), &[Message::new(role, content)])
                .ok_or_else(|| "cannot encode caching prompt".into())
        }
    };
    let (mut low, mut high) = (0, words.len());
    while low + 1 < high {
        let mid = (low + high) / 2;
        if size(&format!("{}{suffix}", words[..mid].join(" ")))? <= target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let mut content = words[..low].join(" ");
    let mut actual = size(&format!("{content}{suffix}"))?;
    while actual < target {
        let mut next = None;
        for pad in [" a", " x", ".", "!", "\n."] {
            let n = size(&format!("{content}{pad}{suffix}"))?;
            if n > actual && n <= target {
                next = Some((pad, n));
                break;
            }
        }
        let (pad, n) = next.ok_or_else(|| format!("cannot size caching prompt: target={target} actual={actual}"))?;
        content.push_str(pad);
        actual = n;
    }
    content.push_str(&suffix);
    if size(&content)? != target {
        return Err("cannot size caching prompt to exact token target".into());
    }
    Ok(vec![Message::new(role, content)])
}
fn call(
    c: &Config,
    prompt: &[Message],
    n: usize,
    priority: u8,
    cache: bool,
    session: Option<&str>,
    end: Instant,
) -> Reply {
    http::request(
        &c.endpoint,
        &c.body(prompt, n, priority, cache, session),
        deadline(c, end),
        Arc::new(AtomicBool::new(false)),
        None,
    )
}
fn serial(
    c: &Config,
    report: &mut Report,
    name: &str,
    prompt: &[Message],
    expected: Option<usize>,
    target: Option<usize>,
    output: usize,
    repeats: usize,
    end: Instant,
) -> Result<()> {
    for repeat in 0..repeats {
        if Instant::now() >= end {
            return report.skip(c, name, "suite budget exhausted");
        }
        let prompt = repeat_prompt(prompt, repeat);
        let expected = if repeat == 0 { expected } else { None };
        crate::stderr_line(format_args!(
            "running {name} repeat={}/{repeats} input_tokens={expected:?}",
            repeat + 1
        ));
        let start = Instant::now();
        let memory = report::Monitor::start(c);
        let reply = call(c, &prompt, output, 0, false, None, end);
        let mut record = Record::new(
            name,
            repeat,
            1,
            output,
            expected,
            target,
            vec![reply],
            start.elapsed(),
        );
        record.memory = memory.finish(c);
        let failed = record.replies[0].status != "ok";
        if failed {
            record.note = Some(format!(
                "exact request: {}",
                c.body(&prompt, output, 0, false, None)
            ));
        }
        report.push(c, record)?;
        if failed {
            break;
        }
    }
    Ok(())
}
pub(crate) fn run(arguments: &[String]) -> Result<()> {
    let c = Config::parse(arguments)?;
    let tokenizer = c
        .tokenizer
        .as_ref()
        .map(|path| Tokenizer::from_directory(path).map_err(|e| e.to_string()))
        .transpose()?;
    if c.only == "request" {
        let target = c
            .long_target
            .ok_or("--only request requires --long-target")?;
        let (prompt, _) = long_prompt(target, tokenizer.as_ref());
        return std::fs::write(&c.output, c.body(&prompt, c.long_output, 0, false, None))
            .map_err(|e| e.to_string());
    }
    let end = Instant::now() + c.budget;
    let mut report = Report::new();
    let prose = messages(PROSE);
    if c.wants("decode") {
        for (name, text) in [
            ("decode/prose", PROSE),
            ("decode/code", CODE),
            ("decode/reasoning", REASON),
        ] {
            let prompt = messages(text);
            serial(
                &c,
                &mut report,
                name,
                &prompt,
                count(tokenizer.as_ref(), &prompt),
                None,
                c.max_tokens,
                c.repeats,
                end,
            )?;
        }
    }
    if c.only == "followup" {
        let t = tokenizer.as_ref().ok_or("followup requires --tokenizer")?;
        let mut prompt = caching_prompt(28672, t, Role::User, &c.model)?;
        serial(&c, &mut report, "followup/turn1", &prompt, Some(28672), Some(28672), c.max_tokens, 1, end)?;
        let parent = report.records.last().and_then(|r| r.replies.first());
        if let Some(parent) = parent.filter(|r| r.status == "ok") {
            prompt.push(Message::new(Role::Assistant, &parent.text));
            prompt.push(Message::new(Role::User, &format!("Add one more useful tip.{}", no_think(&c.model))));
            serial(&c, &mut report, "followup/turn2", &prompt, count(Some(t), &prompt), None, c.max_tokens, 1, end)?;
        } else {
            report.skip(&c, "followup/turn2", "first turn did not complete")?;
        }
    }
    if c.wants("concurrency") || c.only == "shared-system" {
        let shared = if c.only == "shared-system" {
            Some(caching_prompt(8192, tokenizer.as_ref().ok_or("shared-system requires --tokenizer")?, Role::System, &c.model)?)
        } else {
            None
        };
        let sizes = if shared.is_some() { vec![16] } else { c.concurrency.map_or_else(|| vec![1, 4, 8, 16], |n| vec![n]) };
        for (concurrency, repeat) in sizes
            .into_iter().flat_map(|n| (0..c.repeats).map(move |r| (n, r))) {
            let name = if shared.is_some() { "shared-system/16".into() } else { format!("concurrency/{concurrency}") };
            if Instant::now() >= end {
                report.skip(&c, &name, "suite budget exhausted")?;
                continue;
            }
            crate::stderr_line(format_args!("running {name}"));
            let memory = report::Monitor::start(&c);
            let barrier = Arc::new(Barrier::new(concurrency + 1));
            // Distinct prompts across concurrent chats and across repeats.
            let bodies: Vec<_> = (0..concurrency).map(|i| {
                let prompt = if let Some(shared) = &shared {
                    let mut prompt = shared.clone();
                    prompt.extend(repeat_prompt(&messages(&format!(
                        "{}\nGive a separate response for reader {}.{}",
                        [PROSE, CODE, REASON][i % 3], i + 1, no_think(&c.model))), repeat));
                    prompt
                } else { repeat_prompt(&prose, i + 1) };
                c.body(&prompt, c.max_tokens, 0, false, None)
            }).collect();
            let request_deadline = deadline(&c, end);
            let replies = thread::scope(|scope| {
                let workers = bodies.iter()
                    .map(|body| {
                        let barrier = barrier.clone();
                        let endpoint = &c.endpoint;
                        scope.spawn(move || {
                            barrier.wait();
                            http::request(
                                endpoint,
                                body,
                                request_deadline,
                                Arc::new(AtomicBool::new(false)),
                                None,
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                let start = Instant::now();
                barrier.wait();
                let replies = workers
                    .into_iter()
                    .map(|w| w.join().expect("benchmark worker panicked"))
                    .collect();
                (replies, start.elapsed())
            });
            let mut record = Record::new(
                &name,
                repeat,
                concurrency,
                c.max_tokens,
                None,
                None,
                replies.0,
                replies.1,
            );
            record.memory = memory.finish(&c);
            if shared.is_some() {
                record.note = Some("16 synchronized chats; exactly 8192 system-content tokens; cold shared prefix, no priming request; only standard OpenAI fields".into());
            }
            report.push(&c, record)?;
        }
    }
    if c.wants("cache") && Instant::now() < end {
        let prose = c
            .cache_target
            .map(|n| long_prompt(n, tokenizer.as_ref()).0)
            .unwrap_or_else(|| prose.clone());
        crate::stderr_line(format_args!("running cache/cold"));
        let memory = report::Monitor::start(&c);
        let start = Instant::now();
        let cold = call(&c, &prose, c.max_tokens, 0, true, None, end);
        let mut record = Record::new(
            "cache/cold",
            0,
            1,
            c.max_tokens,
            count(tokenizer.as_ref(), &prose),
            None,
            vec![cold.clone()],
            start.elapsed(),
        );
        record.memory = memory.finish(&c);
        if record.replies[0].status != "ok" {
            record.note = Some(format!(
                "exact request: {}",
                c.body(&prose, c.max_tokens, 0, true, None)
            ));
        }
        report.push(&c, record)?;
        if let Some(session) = cold.session.as_ref().filter(|_| cold.status == "ok") {
            let mut followup = prose.clone();
            followup.push(Message::new(Role::Assistant, cold.text));
            followup.push(Message::new(Role::User, "Add one more useful tip."));
            for repeat in 0..c.repeats {
                if Instant::now() >= end {
                    break;
                }
                crate::stderr_line(format_args!("running cache/followup repeat={}", repeat + 1));
                let memory = report::Monitor::start(&c);
                let start = Instant::now();
                let reply = call(&c, &followup, c.max_tokens, 0, true, Some(session), end);
                let mut record = Record::new(
                    "cache/followup",
                    repeat,
                    1,
                    c.max_tokens,
                    count(tokenizer.as_ref(), &followup),
                    None,
                    vec![reply],
                    start.elapsed(),
                );
                record.memory = memory.finish(&c);
                if record.replies[0].status != "ok" {
                    record.note = Some(format!(
                        "request (session capability replaced with parent reference): {}",
                        c.body(&followup, c.max_tokens, 0, true, Some("$PARENT_SESSION"))
                    ));
                }
                report.push(&c, record)?;
            }
            serial(
                &c,
                &mut report,
                "cache/fresh-followup",
                &followup,
                count(tokenizer.as_ref(), &followup),
                None,
                c.max_tokens,
                1,
                end,
            )?;
        } else {
            report.skip(
                &c,
                "cache/followup",
                "cold request failed or server did not return a session capability",
            )?;
        }
    }
    if c.wants("priority") && Instant::now() < end {
        let (agent_prompt, agent_count) = long_prompt(8192, tokenizer.as_ref());
        let main_prompt = messages("Give three practical tips for reading more books.");
        serial(
            &c,
            &mut report,
            "priority/idle-main",
            &main_prompt,
            count(tokenizer.as_ref(), &main_prompt),
            None,
            4,
            1,
            end,
        )?;
        if Instant::now() < end {
            crate::stderr_line(format_args!("running priority/main-with-agent"));
            let memory = report::Monitor::start(&c);
            let start = Instant::now();
            let cancel = Arc::new(AtomicBool::new(false));
            let (tx, rx) = mpsc::channel();
            let agent_body = c.body(&agent_prompt, 2, 0, false, None);
            let agent_deadline = deadline(&c, end);
            let (main, agent, accepted) = thread::scope(|scope| {
                let handle = scope.spawn(|| {
                    http::request(
                        &c.endpoint,
                        &agent_body,
                        agent_deadline,
                        cancel.clone(),
                        Some(tx),
                    )
                });
                let accepted = rx
                    .recv_timeout(
                        agent_deadline
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_secs(10)),
                    )
                    .is_ok();
                if accepted {
                    thread::sleep(
                        Duration::from_millis(250)
                            .min(end.saturating_duration_since(Instant::now())),
                    );
                }
                let main = call(&c, &main_prompt, 4, 1, false, None, end);
                cancel.store(true, Ordering::Relaxed);
                (
                    main,
                    handle.join().expect("agent worker panicked"),
                    accepted,
                )
            });
            let mut record = Record::new(
                "priority/main-with-agent",
                0,
                1,
                4,
                count(tokenizer.as_ref(), &main_prompt),
                None,
                vec![main],
                start.elapsed(),
            );
            record.note = Some(format!(
                "main priority=1; agent priority=0; agent target=8192 actual={agent_count:?}; accepted={accepted}; agent_status={}; agent_ttft_ms={:?}; accepted-to-main delay=250ms; no agent content before cancellation={}",
                agent.status,
                agent.ttft_ms,
                agent.ttft_ms.is_none()
            ));
            record.background = Some(agent);
            record.memory = memory.finish(&c);
            report.push(&c, record)?;
        }
    }
    if c.wants("long") {
        for target in c
            .long_target
            .map(|n| vec![n])
            .unwrap_or_else(|| vec![1024, 8192, 28672])
        {
            let (prompt, actual) = long_prompt(target, tokenizer.as_ref());
            serial(
                &c,
                &mut report,
                &format!("ttft/{target}"),
                &prompt,
                actual,
                Some(target),
                c.long_output,
                1,
                end,
            )?;
        }
    }
    if c.only == "long-output" {
        let prompt = messages(
            "Write a detailed handbook for a town library. Include 100 numbered sections, each with at least one full paragraph of practical advice covering book selection, cataloguing, building maintenance, staffing, volunteer training, reader services, accessibility, community events, budgeting, and future plans. Start directly with section 1 and continue through section 100 without a preamble or conclusion. Do not abbreviate or summarize the sections.",
        );
        serial(
            &c,
            &mut report,
            "long-output/2048",
            &prompt,
            count(tokenizer.as_ref(), &prompt),
            None,
            2048,
            1,
            end,
        )?;
    }
    for group in ["cache", "priority"] {
        if c.wants(group) && !report.records.iter().any(|r| r.name.starts_with(group)) {
            report.skip(&c, group, "suite budget exhausted")?;
        }
    }
    report.print(&c)?;
    report.write(&c)?;
    crate::stderr_line(format_args!("saved {}", c.output));
    if report.cache_matches_fresh() == Some(false) {
        return Err("cached follow-up differs from fresh output; inspect JSON".into());
    }
    if report
        .records
        .iter()
        .any(|r| r.replies.iter().any(|r| r.status == "error"))
    {
        return Err(
            "one or more requests failed; inspect JSON (timeouts are reported separately)".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timed::value::{n, parse, s};

    #[test]
    fn suite_requests_are_greedy_without_seeds() {
        for mode in ["decode", "followup", "shared-system"] {
            let config = Config::parse(&[
                "http://127.0.0.1:8080".into(), "--only".into(), mode.into(),
            ]).unwrap();
            let body = parse(config.body(&messages("seed 42 is prompt text"), 32, 1, false, None).as_bytes()).unwrap();
            assert_eq!(body.get("temperature"), &n(0));
            let crate::timed::value::Value::Object(fields) = &body else { panic!("request object") };
            assert!(fields.iter().all(|(key, _)| key != "seed"), "{mode}: {body:?}");
            assert_eq!(body.get("messages").arr()[0].get("content"), &s("seed 42 is prompt text"));
        }
    }
    #[test]
    fn a_glm_model_gets_low_effort_and_no_soft_switch() {
        let config = |model: &str| Config::parse(&["http://127.0.0.1:8080".into(), "--model".into(), model.into(), "--only".into(), "decode".into()]).unwrap();
        let body = |model: &str| parse(config(model).body(&messages("hi"), 32, 1, false, None).as_bytes()).unwrap();
        // GLM has no thinking-off form and its server refuses the field: ask low effort.
        let glm = body("glm-5.3-flash-fp4-local");
        assert_eq!((glm.get("reasoning_effort"), glm.get("enable_thinking")), (&s("low"), &crate::timed::value::Value::Null));
        // Every other model keeps today's field.
        let qwen = body("qwen3.8-flash-next-fp8");
        assert_eq!((qwen.get("reasoning_effort"), qwen.get("enable_thinking")), (&crate::timed::value::Value::Null, &crate::timed::value::b(false)));
        assert_eq!((no_think("GLM-5.3-Flash"), no_think("qwen3.8-flash-next-fp8"), no_think("")), ("", " /no_think", " /no_think"));
    }
}
