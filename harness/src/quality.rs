//! Answer quality and exactness cost cards: a fixed prompt set run against a
//! served model through the chat API, scored with exact checks only.
//!
//! The quality card runs the set once with drafting as served (the request
//! omits `draft`, as the RigMark served card does) and prints one score per
//! class plus a total. Code answers are checked by unit tests: the scorer
//! appends fixed hidden tests and runs `rustc --test` on this machine.
//! Structured answers are parsed and compared field by field. Prose answers
//! are checked with a rubric of exact string, count and form rules.
//!
//! The exactness cost card runs the same set with `draft: "strict"` and with
//! `draft: "plain"` and compares the streamed token ids. The served
//! FlashNext path accepts only auto, plain or strict drafting
//! (`validate_request` in the engine's serve crate), so no relaxed verify
//! path exists to measure against. The card therefore reports the strict
//! and plain rates side by side, and the verify cost from the server draft
//! receipt lines (`draft id=... cycles=... accepted=... target_ms=...`)
//! when a server log file is supplied. Without a log it says so; the API
//! receipt carries no accepted-token or verify-time fields.
use crate::json::{self, Value};
use std::{
    fs::OpenOptions,
    io::{self, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    process::{Command, ExitCode, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

mod cases;
pub use cases::{CASES, Case, Rule, SET_VERSION, Scoring};

mod stream;
pub use stream::{Reply, apply_sse_data, send};

pub const HELP: &str = "Vade answer quality and exactness cost cards (HTTP chat API)

Usage: vade-harness quality-card --endpoint URL --model NAME --json FILE [OPTIONS]
       vade-harness quality-card --print-prompts

  --endpoint URL        Required http:// chat endpoint (base, /v1 or full path)
  --model NAME          Required explicit model; never substituted
  --json FILE           Required new JSON receipt file; never overwrite
  --card FILE           Optional new text file for the two printed cards
  --server-log FILE     Optional server log with draft receipt lines
  --api-key-file FILE   Global credential file (or VADE_API_KEY_FILE), sent as a Bearer key
  --only WHICH          quality, exactness or both (default both)
  --seed N              Request seed (default 20261027)
  --request-seconds N   Per-request deadline, 10..3600 (default 300)
  --rustc PATH          Rust compiler for code checks (default rustc)
  --no-code-run         Score code answers with text checks only
  --hardware TEXT       Optional hardware line printed on the cards
  --recipe TEXT         Optional recipe line printed on the cards
  --print-prompts       Print the fixed prompt set as JSON and exit
  -h, --help            Show this help

Requests are greedy (temperature 0), thinking off, cache off, one at a time.
The quality card omits the draft field (drafting as served). The exactness
card sends draft strict, then draft plain, and compares token ids.
Exit 0 needs every request complete and, for the exactness card, complete
token evidence on both arms (valid integer ids, the terminal event, counts
that match the streamed ids, the same finish and count on both arms, and
the returned draft modes strict then plain) with identical id lists.
A low quality score is a measurement, not a failure.
Code answers run under rustc --test on this machine unless --no-code-run is
given. No endpoint credentials are written to the receipt.
";

/// The three answer classes on the quality card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Code,
    Structured,
    Prose,
}
impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Structured => "structured",
            Self::Prose => "prose",
        }
    }
    fn card_label(self) -> &'static str {
        match self {
            Self::Code => "CODE",
            Self::Structured => "STRUCTURED",
            Self::Prose => "PROSE",
        }
    }
}
const CLASSES: [Class; 3] = [Class::Code, Class::Structured, Class::Prose];

/// Which cards one run builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Only {
    Quality,
    Exactness,
    Both,
}

/// One quality-card run: where to send the fixed set and where to write.
#[derive(Clone, Debug)]
pub struct Config {
    pub endpoint: String,
    pub model: String,
    pub json_path: PathBuf,
    pub card_path: Option<PathBuf>,
    pub server_log: Option<PathBuf>,
    pub only: Only,
    pub seed: u64,
    pub request_seconds: u64,
    pub rustc: String,
    pub code_run: bool,
    pub hardware: Option<String>,
    pub recipe: Option<String>,
    pub print_prompts: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            model: String::new(),
            json_path: PathBuf::new(),
            card_path: None,
            server_log: None,
            only: Only::Both,
            seed: 20261027,
            request_seconds: 300,
            rustc: "rustc".into(),
            code_run: true,
            hardware: None,
            recipe: None,
            print_prompts: false,
        }
    }
}
impl Config {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, &'static str> {
        Self::parse_options(args, &mut None)
    }
    fn parse_options(
        args: impl IntoIterator<Item = String>,
        option: &mut Option<String>,
    ) -> Result<Self, &'static str> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        let mut seen = Vec::new();
        while let Some(key) = args.next() {
            *option = crate::diagnostics::option_name(&key).map(str::to_owned);
            if seen.contains(&key) {
                return Err("duplicate quality-card option");
            }
            seen.push(key.clone());
            match key.as_str() {
                "--no-code-run" => {
                    config.code_run = false;
                    continue;
                }
                "--print-prompts" => {
                    config.print_prompts = true;
                    continue;
                }
                _ => {}
            }
            let value = args
                .next()
                .ok_or("quality-card option needs a value; see quality-card --help")?;
            match key.as_str() {
                "--endpoint" => config.endpoint = value,
                "--model" => config.model = value,
                "--json" => config.json_path = value.into(),
                "--card" => config.card_path = Some(value.into()),
                "--server-log" => config.server_log = Some(value.into()),
                "--only" => {
                    config.only = match value.as_str() {
                        "quality" => Only::Quality,
                        "exactness" => Only::Exactness,
                        "both" => Only::Both,
                        _ => return Err("--only must be quality, exactness or both"),
                    }
                }
                "--seed" => config.seed = value.parse().map_err(|_| "invalid seed")?,
                "--request-seconds" => {
                    config.request_seconds =
                        value.parse().map_err(|_| "invalid request deadline")?
                }
                "--rustc" => config.rustc = value,
                "--hardware" => config.hardware = Some(value),
                "--recipe" => config.recipe = Some(value),
                _ => return Err("unknown quality-card option; see quality-card --help"),
            }
        }
        *option = None;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<(), &'static str> {
        if self.print_prompts {
            return Ok(());
        }
        if self.endpoint.is_empty()
            || self.model.is_empty()
            || self.json_path.as_os_str().is_empty()
        {
            return Err("quality-card requires --endpoint, --model and --json");
        }
        if self.endpoint.contains('?') || self.endpoint.contains('#') {
            return Err("quality-card endpoint queries/fragments are unsupported");
        }
        if self.endpoint.starts_with("https://") {
            return Err("HTTPS is unsupported; use an HTTP endpoint");
        }
        parse_endpoint(&self.endpoint).map_err(|_| "invalid quality-card HTTP endpoint")?;
        if self.model.len() > 512 || self.model.chars().any(char::is_control) {
            return Err("invalid quality-card model name");
        }
        if !(10..=3600).contains(&self.request_seconds) {
            return Err("quality-card needs a 10..3600 second request deadline");
        }
        if self.rustc.is_empty() || self.rustc.chars().any(char::is_control) {
            return Err("invalid rustc path");
        }
        for text in [&self.hardware, &self.recipe].into_iter().flatten() {
            if text.len() > 512 || text.chars().any(|c| c.is_control() && c != ' ') {
                return Err("card label text is too long or holds control characters");
            }
        }
        Ok(())
    }
}

/// The request target split from the endpoint URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub path: String,
}
/// Accept a base URL, a `/v1` base or a full chat-completions URL.
pub fn parse_endpoint(endpoint: &str) -> io::Result<Endpoint> {
    let rest = endpoint
        .strip_prefix("http://")
        .ok_or_else(|| json::invalid("endpoint must start with http://"))?;
    let (authority, path) = match rest.find('/') {
        Some(split) => (&rest[..split], &rest[split..]),
        None => (rest, ""),
    };
    if authority.is_empty() || authority.contains('@') {
        return Err(json::invalid("endpoint authority is invalid"));
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            (host.to_owned(), port.parse().map_err(|_| json::invalid("bad port"))?)
        }
        _ => (authority.to_owned(), 80),
    };
    if host.is_empty() {
        return Err(json::invalid("endpoint host is empty"));
    }
    let path = path.trim_end_matches('/');
    let path = if path.ends_with("/chat/completions") {
        path.to_owned()
    } else if path.is_empty() {
        "/v1/chat/completions".to_owned()
    } else {
        format!("{path}/chat/completions")
    };
    Ok(Endpoint { host, port, path })
}

const SYSTEM: &str =
    "Work carefully and give a complete, technically precise answer. Follow the output form exactly.";

/// The full user prompt sent for one case.
pub fn full_prompt(case: &Case) -> String {
    match case.class {
        Class::Code => format!("{}{}", case.prompt, cases::CODE_TAIL),
        _ => case.prompt.to_owned(),
    }
}

fn obj(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn s(text: impl Into<String>) -> Value {
    Value::String(text.into())
}
fn n(value: impl ToString) -> Value {
    Value::Number(value.to_string())
}
fn opt_n(value: Option<u64>) -> Value {
    value.map(n).unwrap_or(Value::Null)
}
fn opt_f(value: Option<f64>) -> Value {
    value.filter(|v| v.is_finite()).map(n).unwrap_or(Value::Null)
}
fn opt_s(value: &Option<String>) -> Value {
    value.clone().map(s).unwrap_or(Value::Null)
}
fn opt_b(value: Option<bool>) -> Value {
    value.map(Value::Bool).unwrap_or(Value::Null)
}

/// Build one chat request body. `draft` is None for the served quality card.
/// Project memory is always off here. Use only the fixed card prompt.
pub fn request_body(
    config: &Config,
    case: &Case,
    draft: Option<&str>,
) -> io::Result<String> {
    let mut fields = vec![
        ("model", s(&config.model)),
        (
            "messages",
            Value::Array(vec![
                obj([("role", s("system")), ("content", s(SYSTEM))]),
                obj([("role", s("user")), ("content", s(full_prompt(case)))]),
            ]),
        ),
        ("temperature", n(0)),
        ("top_p", n(1)),
        ("top_k", n(0)),
        ("seed", n(config.seed)),
        ("max_tokens", n(case.max_tokens)),
        ("cache", Value::Bool(false)),
        (
            "chat_template_kwargs",
            obj([("enable_thinking", Value::Bool(false))]),
        ),
        ("include_usage_timing", Value::Bool(true)),
        ("stream", Value::Bool(true)),
        (
            "stream_options",
            obj([("include_usage", Value::Bool(true))]),
        ),
    ];
    if let Some(draft) = draft {
        fields.push(("draft", s(draft)));
    }
    json::stringify(&Value::Object(
        fields.into_iter().map(|(k, v)| (k.into(), v)).collect(),
    ))
}

/// One named check outcome. `run` is false when the check could not run at
/// all (for example code checks without a compiler); such checks stay out
/// of every denominator and the card labels them.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub run: bool,
    pub passed: bool,
    pub detail: String,
}
fn check(name: impl Into<String>, passed: bool, detail: impl Into<String>) -> Check {
    Check { name: name.into(), run: true, passed, detail: detail.into() }
}
fn not_run(name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check { name: name.into(), run: false, passed: false, detail: detail.into() }
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}
fn sentence_count(text: &str) -> usize {
    text.chars().filter(|c| matches!(c, '.' | '!' | '?')).count()
}
fn live_lines(text: &str) -> Vec<&str> {
    text.lines().filter(|line| !line.trim().is_empty()).collect()
}

/// Compare two parsed JSON values; numbers compare by value, not spelling.
pub fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.parse::<i64>(), b.parse::<i64>()) {
            (Ok(a), Ok(b)) => a == b,
            _ => match (a.parse::<f64>(), b.parse::<f64>()) {
                (Ok(a), Ok(b)) => a == b,
                _ => a == b,
            },
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.iter()
                        .find(|(other, _)| other == key)
                        .is_some_and(|(_, other)| values_equal(value, other))
                })
        }
        _ => left == right,
    }
}

fn rule_check(rule: Rule, answer: &str, parsed: Option<&Value>) -> Check {
    let name = rule.name();
    let trimmed = answer.trim();
    match rule {
        Rule::Contains(text) => check(name, trimmed.contains(text), ""),
        Rule::NotContains(text) => check(name, !trimmed.contains(text), ""),
        Rule::StartsWith(text) => check(name, trimmed.starts_with(text), ""),
        Rule::EndsWith(text) => check(name, trimmed.ends_with(text), ""),
        Rule::Words(low, high) => {
            let words = word_count(trimmed);
            check(name, (low..=high).contains(&words), format!("words {words}"))
        }
        Rule::Sentences(want) => {
            let got = sentence_count(trimmed);
            check(name, got == want, format!("sentences {got}"))
        }
        Rule::Lines(want) => {
            let got = live_lines(trimmed).len();
            check(name, got == want, format!("lines {got}"))
        }
        Rule::EveryLineStartsWith(prefix) => {
            let lines = live_lines(trimmed);
            check(
                name,
                !lines.is_empty() && lines.iter().all(|line| line.starts_with(prefix)),
                format!("lines {}", lines.len()),
            )
        }
        Rule::EqualsExact(text) => check(name, trimmed == text, ""),
        Rule::LineContains(index, text) => {
            let line = live_lines(trimmed).get(index).copied();
            check(
                name,
                line.is_some_and(|line| line.contains(text)),
                line.unwrap_or("line missing").to_owned(),
            )
        }
        Rule::EveryLineWordsAtMost(max) => {
            let lines = live_lines(trimmed);
            let worst = lines
                .iter()
                .map(|line| word_count(line.strip_prefix("- ").unwrap_or(line)))
                .max()
                .unwrap_or(0);
            check(
                name,
                !lines.is_empty() && worst <= max,
                format!("most words in one line {worst}"),
            )
        }
        Rule::JsonField(key, literal) => {
            let want = json::parse(literal).expect("fixed set literal parses");
            let got = parsed.and_then(|value| value.get(key));
            check(
                name,
                got.is_some_and(|value| values_equal(value, &want)),
                got.map_or_else(|| "field missing".into(), |v| format!("got {}", short(v))),
            )
        }
        Rule::JsonArrayLen(key, want) => {
            let got = parsed
                .and_then(|value| value.get(key))
                .and_then(Value::as_array)
                .map(<[Value]>::len);
            check(
                name,
                got == Some(want),
                got.map_or_else(|| "array missing".into(), |n| format!("items {n}")),
            )
        }
        Rule::JsonItemField(key, index, field, literal) => {
            let want = json::parse(literal).expect("fixed set literal parses");
            let got = parsed
                .and_then(|value| value.get(key))
                .and_then(Value::as_array)
                .and_then(|items| items.get(index))
                .and_then(|item| item.get(field));
            check(
                name,
                got.is_some_and(|value| values_equal(value, &want)),
                got.map_or_else(|| "item missing".into(), |v| format!("got {}", short(v))),
            )
        }
    }
}

fn short(value: &Value) -> String {
    let text = json::stringify(value).unwrap_or_else(|_| "?".into());
    if text.len() > 80 {
        format!("{}...", &text[..text.char_indices().nth(77).map_or(text.len(), |(i, _)| i)])
    } else {
        text
    }
}

/// Key-set checks for a JSON case, derived from its rules. Every fixed
/// JSON prompt names an exact field set, so the top-level keys must be
/// exactly the keys its rules read, and each object inside an item rule's
/// array must hold exactly the fields its rules read.
fn json_key_checks(rules: &[Rule], parsed: Option<&Value>) -> Vec<Check> {
    fn keys_of(value: &Value) -> Vec<&str> {
        match value {
            Value::Object(fields) => {
                let mut keys: Vec<&str> = fields.iter().map(|(key, _)| key.as_str()).collect();
                keys.sort_unstable();
                keys
            }
            _ => Vec::new(),
        }
    }
    fn exact(want: &mut Vec<&'static str>, got: &[&str]) -> bool {
        want.sort_unstable();
        want.dedup();
        want.as_slice() == got
    }
    let mut top: Vec<&'static str> = rules
        .iter()
        .filter_map(|rule| match rule {
            Rule::JsonField(key, _) | Rule::JsonArrayLen(key, _) => Some(*key),
            Rule::JsonItemField(key, ..) => Some(*key),
            _ => None,
        })
        .collect();
    let got = parsed.map(keys_of).unwrap_or_default();
    let detail = if got.is_empty() { "no object".into() } else { format!("keys {}", got.join(",")) };
    let mut checks = vec![check("exact json keys", parsed.is_some() && exact(&mut top, &got), detail)];
    let mut arrays: Vec<&'static str> = Vec::new();
    for rule in rules {
        let Rule::JsonItemField(key, ..) = rule else { continue };
        if !arrays.contains(key) {
            arrays.push(key);
        }
    }
    for key in arrays {
        let want: Vec<&'static str> = rules
            .iter()
            .filter_map(|rule| match rule {
                Rule::JsonItemField(array, _, field, _) if *array == key => Some(*field),
                _ => None,
            })
            .collect();
        let items = parsed
            .and_then(|value| value.get(key))
            .and_then(Value::as_array);
        let every = items.is_some_and(|items| {
            !items.is_empty()
                && items.iter().all(|item| {
                    let mut want = want.clone();
                    exact(&mut want, &keys_of(item))
                })
        });
        checks.push(check(
            format!("exact json {key} item keys"),
            every,
            items.map_or_else(|| "array missing".into(), |items| format!("items {}", items.len())),
        ));
    }
    checks
}

/// Score a rules or JSON case. Code cases go through [`score_code`].
/// A JSON answer must stand alone in the form its prompt asks for: the
/// trimmed answer itself must parse, with no surrounding fence removed.
pub fn score_rules(case: &Case, answer: &str) -> Vec<Check> {
    match case.scoring {
        Scoring::Rules(rules) => rules.iter().map(|rule| rule_check(*rule, answer, None)).collect(),
        Scoring::Json(rules) => match json::parse(answer.trim()) {
            Ok(value) => {
                let mut checks = vec![check("answer is one JSON document", true, "")];
                checks.extend(json_key_checks(rules, Some(&value)));
                checks.extend(rules.iter().map(|rule| rule_check(*rule, answer, Some(&value))));
                checks
            }
            Err(_) => {
                let mut checks = vec![check("answer is one JSON document", false, "parse failed")];
                checks.extend(json_key_checks(rules, None));
                checks.extend(rules.iter().map(|rule| rule_check(*rule, answer, None)));
                checks
            }
        },
        Scoring::Code { .. } => vec![not_run("code scoring", "use score_code")],
    }
}

/// The first fenced block in an answer, with its fence language.
pub fn first_fence(answer: &str) -> Option<(String, String)> {
    let start = answer.find("```")?;
    let rest = &answer[start + 3..];
    let (language, body) = rest.split_once('\n')?;
    let end = body.find("```")?;
    Some((language.trim().to_owned(), body[..end].to_owned()))
}

/// Stop the whole process group a [`run_bounded`] child leads. The child
/// starts as group leader, so the group id is the child pid. Group stop
/// needs the system `kill` tool; the standard library stops one process
/// only. A failure here falls back to stopping the direct child.
fn kill_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("/bin/kill")
            .arg("-KILL")
            .arg(format!("-{}", child.id()))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
}

/// Run a child to completion with a hard timeout; kill it past the deadline.
/// Both pipes drain on their own threads so a chatty child cannot deadlock.
/// Output stays inside the same deadline: a grandchild that inherits a pipe
/// cannot hold this call open after the direct child exits. The child leads
/// its own process group and the group is stopped on every exit, so work a
/// timed-out check started does not outlive the check.
fn run_bounded(mut command: Command, deadline: Duration) -> io::Result<(bool, String)> {
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn()?;
    let (sender, receiver) = mpsc::channel();
    fn drain(pipe: Option<impl Read + Send + 'static>, sender: mpsc::Sender<String>) {
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(pipe) = pipe {
                let _ = pipe.take(4 * 1024 * 1024).read_to_string(&mut text);
            }
            let _ = sender.send(text);
        });
    }
    drain(child.stdout.take(), sender.clone());
    drain(child.stderr.take(), sender);
    let started = Instant::now();
    let timed_out = || io::Error::new(io::ErrorKind::TimedOut, "code check deadline passed");
    let mut parts: Vec<String> = Vec::new();
    loop {
        while let Ok(text) = receiver.try_recv() {
            parts.push(text);
        }
        match child.try_wait()? {
            Some(status) => {
                while parts.len() < 2 {
                    let left = match deadline
                        .checked_sub(started.elapsed())
                        .filter(|left| !left.is_zero())
                    {
                        Some(left) => left,
                        None => {
                            kill_group(&mut child);
                            return Err(timed_out());
                        }
                    };
                    match receiver.recv_timeout(left) {
                        Ok(text) => parts.push(text),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            kill_group(&mut child);
                            return Err(timed_out());
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                kill_group(&mut child);
                return Ok((status.success(), parts.concat()));
            }
            None => {
                if started.elapsed() > deadline {
                    kill_group(&mut child);
                    let _ = child.wait();
                    return Err(timed_out());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// Hidden test names in one case, in source order.
fn hidden_test_names(hidden_tests: &str) -> Vec<String> {
    hidden_tests
        .match_indices("#[test] fn ")
        .filter_map(|(at, _)| {
            let rest = &hidden_tests[at + "#[test] fn ".len()..];
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
            Some(format!("quality_hidden::{}", &rest[..end]))
        })
        .collect()
}

/// Score one code answer: fence and signature text checks, then compile the
/// block with the case's hidden tests appended and run every test. The
/// answer's own tests must pass too; a failed own test fails its check even
/// when every hidden test passes. Model code runs on this machine with
/// unsafe code forbidden by the generated file header. Compile and test
/// share the one `deadline` for the case.
pub fn score_code(
    case: &Case,
    answer: &str,
    rustc: &str,
    run: bool,
    deadline: Duration,
) -> Vec<Check> {
    let Scoring::Code { signature, hidden_tests } = case.scoring else {
        return vec![not_run("code scoring", "not a code case")];
    };
    let names = hidden_test_names(hidden_tests);
    let fence = first_fence(answer);
    let mut checks = vec![check(
        "one fenced code block",
        fence.is_some(),
        fence.as_ref().map_or_else(|| "no fence".into(), |(lang, _)| format!("language {lang:?}")),
    )];
    let Some((_, code)) = fence else {
        checks.push(check("exact signature", false, "no code block"));
        checks.push(check("own unit tests present", false, "no code block"));
        checks.push(check("own unit tests pass", false, "no code block"));
        checks.extend(names.iter().map(|name| check(format!("hidden test {name}"), false, "no code block")));
        return checks;
    };
    checks.push(check("exact signature", code.contains(signature), ""));
    checks.push(check(
        "own unit tests present",
        code.matches("#[test]").count() >= 2,
        format!("test markers {}", code.matches("#[test]").count()),
    ));
    if !run {
        checks.push(not_run("own unit tests pass", "code run disabled"));
        checks.extend(
            names
                .iter()
                .map(|name| not_run(format!("hidden test {name}"), "code run disabled")),
        );
        return checks;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    static SCORE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SCORE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "vade-quality-{}-{stamp}-{seq}-{case_id}",
        std::process::id(),
        case_id = case.id
    ));
    let prepared = std::fs::create_dir_all(&dir).and_then(|()| {
        let source = format!("#![forbid(unsafe_code)]\n#![allow(dead_code)]\n{code}\n{hidden_tests}\n");
        std::fs::write(dir.join("answer.rs"), source)
    });
    if prepared.is_err() {
        checks.push(not_run("own unit tests pass", "temp file failed"));
        checks.extend(
            names
                .iter()
                .map(|name| not_run(format!("hidden test {name}"), "temp file failed")),
        );
        return checks;
    }
    let case_started = Instant::now();
    let binary = dir.join("answer-test");
    let mut compile = Command::new(rustc);
    compile
        .arg("--edition")
        .arg("2021")
        .arg("--test")
        .arg(dir.join("answer.rs"))
        .arg("-o")
        .arg(&binary);
    let compiled = run_bounded(compile, deadline);
    let (ok, output) = match compiled {
        Ok(pair) => pair,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&dir);
            let detail = if error.kind() == io::ErrorKind::NotFound {
                "rustc not found"
            } else if error.kind() == io::ErrorKind::TimedOut {
                "rustc timed out"
            } else {
                "rustc did not run"
            };
            checks.push(not_run("own unit tests pass", detail));
            checks.extend(
                names
                    .iter()
                    .map(|name| not_run(format!("hidden test {name}"), detail)),
            );
            return checks;
        }
    };
    checks.push(check(
        "code compiles under rustc --test",
        ok,
        if ok { String::new() } else { first_error_line(&output) },
    ));
    if !ok {
        let _ = std::fs::remove_dir_all(&dir);
        checks.push(check("own unit tests pass", false, "compile failed"));
        checks.extend(
            names
                .iter()
                .map(|name| check(format!("hidden test {name}"), false, "compile failed")),
        );
        return checks;
    }
    let left = deadline.checked_sub(case_started.elapsed()).unwrap_or_default();
    let mut test_run = Command::new(&binary);
    test_run.arg("--test-threads=1");
    let ran = if left.is_zero() {
        Err(io::Error::new(io::ErrorKind::TimedOut, "code check deadline passed"))
    } else {
        run_bounded(test_run, left)
    };
    let _ = std::fs::remove_dir_all(&dir);
    match ran {
        Ok((_, output)) => {
            let own_failed = output.lines().find(|line| {
                let line = line.trim();
                line.starts_with("test ")
                    && line.ends_with("FAILED")
                    && !names.iter().any(|name| line.contains(&format!("test {name}")))
            });
            checks.push(check(
                "own unit tests pass",
                own_failed.is_none(),
                own_failed.map_or_else(String::new, |line| line.trim().to_owned()),
            ));
            for name in &names {
                let passed = output
                    .lines()
                    .any(|line| line.contains(&format!("test {name}")) && line.trim_end().ends_with("ok"));
                checks.push(check(format!("hidden test {name}"), passed, ""));
            }
        }
        Err(_) => {
            checks.push(check("own unit tests pass", false, "test run failed or timed out"));
            checks.extend(
                names
                    .iter()
                    .map(|name| check(format!("hidden test {name}"), false, "test run failed or timed out")),
            );
        }
    }
    checks
}

fn first_error_line(output: &str) -> String {
    output
        .lines()
        .find(|line| line.starts_with("error"))
        .unwrap_or("compile failed")
        .chars()
        .take(120)
        .collect()
}

/// One server draft receipt line: `draft id=N cycles=.. rows=.. accepted=..
/// proposed=.. max_rows=.. target_ms=.. draft_ms=..`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftReceipt {
    pub id: u64,
    pub cycles: u64,
    pub rows: u64,
    pub accepted: u64,
    pub proposed: u64,
    pub target_ms: f64,
    pub draft_ms: f64,
}
impl DraftReceipt {
    /// Target (verify) milliseconds per verify step.
    pub fn verify_ms_per_step(&self) -> Option<f64> {
        (self.cycles > 0).then(|| self.target_ms / self.cycles as f64)
    }
    /// Accepted draft tokens per verify step.
    pub fn accepted_per_step(&self) -> Option<f64> {
        (self.cycles > 0).then(|| self.accepted as f64 / self.cycles as f64)
    }
}

/// Parse every draft receipt line in a server log.
pub fn parse_draft_receipts(log: &str) -> Vec<DraftReceipt> {
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix(key)?.strip_prefix('='))
    }
    log.lines()
        .filter_map(|line| line.find("draft id=").map(|at| &line[at..]))
        .filter_map(|line| {
            Some(DraftReceipt {
                id: field(line, "id")?.parse().ok()?,
                cycles: field(line, "cycles")?.parse().ok()?,
                rows: field(line, "rows")?.parse().ok()?,
                accepted: field(line, "accepted")?.parse().ok()?,
                proposed: field(line, "proposed")?.parse().ok()?,
                target_ms: field(line, "target_ms")?.parse().ok()?,
                draft_ms: field(line, "draft_ms")?.parse().ok()?,
            })
        })
        .collect()
}

/// Verify cost summed over a set of draft receipts.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifyCost {
    pub receipts: usize,
    pub cycles: u64,
    pub accepted: u64,
    pub proposed: u64,
    pub target_ms: f64,
    pub draft_ms: f64,
}
impl VerifyCost {
    fn add(&mut self, receipt: DraftReceipt) {
        self.receipts += 1;
        self.cycles += receipt.cycles;
        self.accepted += receipt.accepted;
        self.proposed += receipt.proposed;
        self.target_ms += receipt.target_ms;
        self.draft_ms += receipt.draft_ms;
    }
    pub fn verify_ms_per_step(&self) -> Option<f64> {
        (self.cycles > 0).then(|| self.target_ms / self.cycles as f64)
    }
    pub fn accepted_per_step(&self) -> Option<f64> {
        (self.cycles > 0).then(|| self.accepted as f64 / self.cycles as f64)
    }
}

/// One case on the quality card.
#[derive(Clone, Debug)]
pub struct QualityCase {
    pub case: &'static Case,
    pub reply: Reply,
    pub checks: Vec<Check>,
    pub error: Option<String>,
}
impl QualityCase {
    pub fn passed(&self) -> usize {
        self.checks.iter().filter(|c| c.run && c.passed).count()
    }
    pub fn run_checks(&self) -> usize {
        self.checks.iter().filter(|c| c.run).count()
    }
    pub fn full_pass(&self) -> bool {
        self.error.is_none() && self.run_checks() > 0 && self.passed() == self.run_checks()
    }
}

/// One case on the exactness card: strict against plain.
#[derive(Clone, Debug)]
pub struct ExactCase {
    pub case: &'static Case,
    pub strict: Option<Reply>,
    pub plain: Option<Reply>,
    pub error: Option<String>,
}
/// Why one reply is not complete token evidence, when it is not. This is
/// half of the one definition of complete evidence; [`evidence_failure`]
/// adds the pair conditions. A complete reply streams only exact integer
/// ids, ends with the terminal event, and reports a completion count and a
/// finish reason. The streamed ids must match the count. The engine counts
/// its end-of-sequence token without streaming its id on a stop finish
/// only, so one fewer id is complete for `stop` and for no other finish; a
/// length finish streams every counted token.
fn reply_evidence_failure(reply: &Reply) -> Option<String> {
    if reply.invalid_token_id {
        return Some("invalid token id".into());
    }
    if !reply.saw_done {
        return Some("no terminal event".into());
    }
    if reply.token_ids.is_empty() {
        return Some("no token ids".into());
    }
    let Some(counted) = reply.completion_tokens else {
        return Some("missing completion token count".into());
    };
    let Some(finish) = &reply.finish_reason else {
        return Some("missing finish reason".into());
    };
    let streamed = reply.token_ids.len() as u64;
    let matches = streamed == counted || (finish == "stop" && streamed + 1 == counted);
    if !matches {
        return Some(format!(
            "token id count mismatch: streamed {streamed} ids, completion_tokens {counted}, finish {finish}"
        ));
    }
    None
}
/// The one place that decides if a strict/plain pair carries complete
/// exactness evidence. Both replies must be complete on their own (see
/// `reply_evidence_failure`), the strict arm must report draft `strict`
/// and the plain arm draft `plain` (the engine downgrades strict to plain
/// when no draft depth is resident; plain against plain proves nothing),
/// and both arms must agree on the finish reason and the completion count.
/// Id lists are compared only after this check passes.
pub fn evidence_failure(strict: &Reply, plain: &Reply) -> Option<String> {
    if let Some(reason) = reply_evidence_failure(strict) {
        return Some(reason);
    }
    if let Some(reason) = reply_evidence_failure(plain) {
        return Some(reason);
    }
    if strict.draft.as_deref() != Some("strict") || plain.draft.as_deref() != Some("plain") {
        return Some(format!(
            "draft mode mismatch: strict arm reported {} (want strict), plain arm reported {} (want plain)",
            shown_mode(&strict.draft),
            shown_mode(&plain.draft)
        ));
    }
    if strict.finish_reason != plain.finish_reason {
        return Some(format!(
            "finish reason mismatch: strict arm {}, plain arm {}",
            shown_mode(&strict.finish_reason),
            shown_mode(&plain.finish_reason)
        ));
    }
    if strict.completion_tokens != plain.completion_tokens {
        return Some(format!(
            "completion token count mismatch: strict arm {}, plain arm {}",
            strict.completion_tokens.map_or_else(|| "missing".into(), |n| n.to_string()),
            plain.completion_tokens.map_or_else(|| "missing".into(), |n| n.to_string())
        ));
    }
    None
}
fn shown_mode(mode: &Option<String>) -> String {
    mode.clone().unwrap_or_else(|| "missing".into())
}
impl ExactCase {
    /// True when both arms streamed the same token ids and the same text.
    /// Missing ids are never identical: two empty lists prove nothing.
    pub fn ids_identical(&self) -> Option<bool> {
        let (strict, plain) = (self.strict.as_ref()?, self.plain.as_ref()?);
        if strict.token_ids.is_empty() || plain.token_ids.is_empty() {
            return Some(false);
        }
        Some(strict.token_ids == plain.token_ids && strict.text == plain.text)
    }
    /// True when the engine reports the requested modes back: `strict` on
    /// the strict arm and `plain` on the plain arm. The engine downgrades
    /// strict to plain when no draft depth is resident; that run measured
    /// plain against plain and must not pass as strict against plain.
    pub fn draft_modes_ok(&self) -> Option<bool> {
        let (strict, plain) = (self.strict.as_ref()?, self.plain.as_ref()?);
        Some(
            strict.draft.as_deref() == Some("strict")
                && plain.draft.as_deref() == Some("plain"),
        )
    }
    /// The first reason this case fails, or None when it passes. A case
    /// passes only with complete token evidence (see [`evidence_failure`])
    /// and identical ids and text. Id lists compare only when complete.
    pub fn failure_reason(&self) -> Option<String> {
        if let Some(error) = &self.error {
            return Some(error.clone());
        }
        let (Some(strict), Some(plain)) = (&self.strict, &self.plain) else {
            return Some("request failed".into());
        };
        if let Some(reason) = evidence_failure(strict, plain) {
            return Some(reason);
        }
        if self.ids_identical() != Some(true) {
            return Some(match self.first_diff() {
                Some(at) => format!("ids differ at token {at}"),
                None => "answer texts differ".into(),
            });
        }
        None
    }
    /// True when this case passes in full (see [`Self::failure_reason`]).
    pub fn passed(&self) -> bool {
        self.failure_reason().is_none()
    }
    /// First differing token position, for the receipt.
    pub fn first_diff(&self) -> Option<usize> {
        let (strict, plain) = (self.strict.as_ref()?, self.plain.as_ref()?);
        strict
            .token_ids
            .iter()
            .zip(&plain.token_ids)
            .position(|(a, b)| a != b)
            .or_else(|| (strict.token_ids.len() != plain.token_ids.len()).then(|| {
                strict.token_ids.len().min(plain.token_ids.len())
            }))
    }
}

/// A finished card run.
#[derive(Clone, Debug)]
pub struct Report {
    pub config: Config,
    pub quality: Vec<QualityCase>,
    pub exactness: Vec<ExactCase>,
    /// Draft receipts matched to strict-arm completion ids.
    pub verify: VerifyCost,
    /// The matched draft receipt lines, kept raw for the JSON receipt.
    pub verify_receipts: Vec<DraftReceipt>,
    /// How the verify cost was matched, or why it is unavailable.
    pub verify_basis: String,
    pub elapsed_s: f64,
}
impl Report {
    pub fn quality_score(&self, class: Option<Class>) -> (usize, usize) {
        let mut passed = 0;
        let mut total = 0;
        for item in self
            .quality
            .iter()
            .filter(|item| class.is_none_or(|want| item.case.class == want))
        {
            passed += item.passed();
            total += item.run_checks();
        }
        (passed, total)
    }
    /// True when at least one code run check in this report actually ran.
    /// The card and receipt derive their ran/not-run summary from the check
    /// results, never from the requested setting alone.
    pub fn code_checks_ran(&self) -> bool {
        self.quality.iter().any(|item| {
            item.checks.iter().any(|check| {
                check.run
                    && (check.name.starts_with("hidden test")
                        || check.name == "code compiles under rustc --test"
                        || check.name == "own unit tests pass")
            })
        })
    }
    /// Passed exactness cases over every exactness case. A pass needs token
    /// evidence, identical ids and returned draft modes (see
    /// `ExactCase::passed`); identical ids alone do not count, and a case
    /// with a missing reply stays in the denominator.
    pub fn exact_identical(&self) -> (usize, usize) {
        let same = self.exactness.iter().filter(|c| c.passed()).count();
        (same, self.exactness.len())
    }
    pub fn successful(&self) -> bool {
        let quality_ok = match self.config.only {
            Only::Exactness => true,
            _ => {
                self.quality.len() == CASES.len()
                    && self.quality.iter().all(|item| item.error.is_none())
            }
        };
        let exact_ok = match self.config.only {
            Only::Quality => true,
            _ => {
                self.exactness.len() == CASES.len()
                    && self.exactness.iter().all(|item| item.passed())
            }
        };
        quality_ok && exact_ok
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[middle])
    } else {
        Some((values[middle - 1] + values[middle]) / 2.0)
    }
}

/// The shared credential header, as every other harness command gets it.
/// `auth::configure` in `main` consumes `--api-key-file` (or
/// `VADE_API_KEY_FILE`) before dispatch, so this command reads the shared
/// store instead of parsing the flag itself.
fn configured_auth_header() -> io::Result<Option<String>> {
    Ok(crate::auth::ApiKey::configured()?.map(|key| key.bearer_header()))
}

/// Run the fixed set once per requested card, one request at a time.
/// A server log that cannot be read never discards the run: it is checked
/// before the first request and read again after the last one, and any
/// error is recorded as the verify basis on the finished report.
pub fn run(config: Config) -> io::Result<Report> {
    config.validate().map_err(io::Error::other)?;
    let endpoint = parse_endpoint(&config.endpoint)?;
    let auth = configured_auth_header()?;
    let log_probe = config.server_log.as_ref().map(|path| {
        std::fs::File::open(path)
            .map(|_| ())
            .map_err(|error| format!("server log unreadable before requests: {error}"))
    });
    let deadline = Duration::from_secs(config.request_seconds);
    let started = Instant::now();
    let mut quality = Vec::new();
    if config.only != Only::Exactness {
        for case in CASES {
            crate::diagnostics::stderr_line(format_args!(
                "quality-card: quality {} ({})",
                case.id,
                case.class.label()
            ));
            let item = match request_body(&config, case, None)
                .and_then(|body| send(&endpoint, &body, auth.as_deref(), deadline))
            {
                Ok(reply) => {
                    let checks = match case.scoring {
                        Scoring::Code { .. } => score_code(
                            case,
                            &reply.text,
                            &config.rustc,
                            config.code_run,
                            Duration::from_secs(60),
                        ),
                        _ => score_rules(case, &reply.text),
                    };
                    QualityCase { case, reply, checks, error: None }
                }
                Err(error) => QualityCase {
                    case,
                    reply: Reply::default(),
                    checks: Vec::new(),
                    error: Some(error.to_string()),
                },
            };
            quality.push(item);
        }
    }
    let mut exactness = Vec::new();
    if config.only != Only::Quality {
        for case in CASES {
            crate::diagnostics::stderr_line(format_args!(
                "quality-card: exactness {} strict then plain",
                case.id
            ));
            let strict = request_body(&config, case, Some("strict"))
                .and_then(|body| send(&endpoint, &body, auth.as_deref(), deadline));
            let plain = request_body(&config, case, Some("plain"))
                .and_then(|body| send(&endpoint, &body, auth.as_deref(), deadline));
            exactness.push(match (strict, plain) {
                (Ok(strict), Ok(plain)) => {
                    ExactCase { case, strict: Some(strict), plain: Some(plain), error: None }
                }
                (Ok(strict), Err(error)) => ExactCase {
                    case,
                    strict: Some(strict),
                    plain: None,
                    error: Some(error.to_string()),
                },
                (Err(error), Ok(plain)) => ExactCase {
                    case,
                    strict: None,
                    plain: Some(plain),
                    error: Some(error.to_string()),
                },
                (Err(error), Err(_)) => {
                    ExactCase { case, strict: None, plain: None, error: Some(error.to_string()) }
                }
            });
        }
    }
    let (verify, verify_receipts, verify_basis) = verify_cost(&config, &exactness, log_probe);
    Ok(Report {
        config,
        quality,
        exactness,
        verify,
        verify_receipts,
        verify_basis,
        elapsed_s: started.elapsed().as_secs_f64(),
    })
}

/// Match strict-arm completions to draft receipt lines. Completion ids are
/// reused across server runs, so the match must be exact: each distinct
/// strict completion id needs exactly one receipt line in the log. An
/// absent or repeated id makes the whole cost unavailable; unrelated log
/// lines are never summed as this run's cost.
fn verify_cost(
    config: &Config,
    exactness: &[ExactCase],
    log_probe: Option<Result<(), String>>,
) -> (VerifyCost, Vec<DraftReceipt>, String) {
    let none = |basis: String| (VerifyCost::default(), Vec::new(), basis);
    let Some(path) = &config.server_log else {
        return none("no server log supplied".into());
    };
    let log = match std::fs::read_to_string(path) {
        Ok(log) => log,
        Err(error) => {
            return none(match log_probe {
                Some(Err(probe)) => format!("{probe}; server log unreadable after requests: {error}"),
                _ => format!("server log unreadable after requests: {error}"),
            });
        }
    };
    let receipts = parse_draft_receipts(&log);
    if receipts.is_empty() {
        return none("server log holds no draft receipt lines".into());
    }
    let mut ids: Vec<u64> = Vec::new();
    for item in exactness {
        let Some(id) = item.strict.as_ref().and_then(|reply| reply.completion_id) else {
            continue;
        };
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    if ids.is_empty() {
        return none("verify cost unavailable: no strict completion ids to match".into());
    }
    let mut matched: Vec<DraftReceipt> = Vec::new();
    for id in &ids {
        let found: Vec<DraftReceipt> =
            receipts.iter().filter(|receipt| receipt.id == *id).copied().collect();
        if found.len() != 1 {
            return none(format!(
                "verify cost unavailable: completion id {id} matched {} draft receipt lines (need exactly one)",
                found.len()
            ));
        }
        matched.push(found[0]);
    }
    let mut cost = VerifyCost::default();
    for receipt in &matched {
        cost.add(*receipt);
    }
    (cost, matched, "draft receipts matched by completion id".into())
}

mod report;
pub use report::prompts_json;

/// `vade-harness quality-card`: run the fixed set, write the JSON receipt
/// and print both cards.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<_> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return crate::diagnostics::print_stdout(HELP);
    }
    let config = match crate::diagnostics::parse_command("quality-card", |option| {
        Config::parse_options(args, option)
    }) {
        Ok(config) => config,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: {error}"));
            return ExitCode::from(2);
        }
    };
    if config.print_prompts {
        return match prompts_json() {
            Ok(text) => crate::diagnostics::print_stdout(&format!("{text}\n")),
            Err(_) => ExitCode::FAILURE,
        };
    }
    let new_file = |path: &std::path::Path| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(path)
    };
    // Validate both destinations before connecting. create_new also refuses
    // symlinks and protects existing reports, as in the workload driver.
    let mut json_file = match new_file(&config.json_path) {
        Ok(file) => file,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: cannot create new quality-card JSON file. {}", crate::diagnostics::output_file_hint(&error)));
            return ExitCode::from(2);
        }
    };
    let mut card_file = match &config.card_path {
        Some(path) => match new_file(path) {
            Ok(file) => Some(file),
            Err(error) => {
                crate::diagnostics::stderr_line(format_args!("vade: cannot create new quality-card card file. {}", crate::diagnostics::output_file_hint(&error)));
                // Remove only the empty JSON file this call just created, so
                // the same --json path can be retried once fixed.
                drop(json_file);
                let _ = std::fs::remove_file(&config.json_path);
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    crate::diagnostics::stderr_line(format_args!(
        "quality-card: {} fixed prompts, one request at a time",
        CASES.len()
    ));
    let report = match run(config) {
        Ok(report) => report,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: quality-card setup failed ({error}); JSON is incomplete. Keep the partial file, check the endpoint and model, then retry with a new --json path."));
            return ExitCode::FAILURE;
        }
    };
    let saved = report.to_json().and_then(|body| {
        json_file.write_all(body.as_bytes())?;
        json_file.write_all(b"\n")?;
        json_file.sync_all()
    });
    if let Err(error) = saved {
        crate::diagnostics::stderr_line(format_args!("vade: quality-card JSON write failed. {}", crate::diagnostics::output_file_hint(&error)));
        return ExitCode::FAILURE;
    }
    let cards = report.cards();
    if let Some(file) = &mut card_file {
        if file.write_all(cards.as_bytes()).and_then(|()| file.sync_all()).is_err() {
            crate::diagnostics::stderr_line(format_args!("vade: quality-card card write failed."));
            return ExitCode::FAILURE;
        }
    }
    if crate::diagnostics::print_stdout(&cards) == ExitCode::FAILURE {
        return ExitCode::FAILURE;
    }
    crate::diagnostics::stderr_line(format_args!(
        "quality-card: receipt saved to the requested JSON file"
    ));
    if report.successful() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests;
