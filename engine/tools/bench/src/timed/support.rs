//! Frozen request inputs, boundary observations and exactly owned child processes.
use super::{
    Config, stream,
    value::{Value, a, digest, n, o, parse, s},
};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom},
    os::unix::{
        fs::MetadataExt,
        process::{CommandExt, ExitStatusExt},
    },
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BUILD_FIELDS: &[&str] = &[
    "vade.build.git_sha",
    "build_hash",
    "build_sha",
    "git_hash",
    "git_sha",
    "git_commit",
    "commit",
    "build.hash",
    "build.sha",
    "build.commit",
];
const CHECKPOINT_FIELDS: &[&str] = &[
    "vade.checkpoint.sha256",
    "checkpoint_hash",
    "checkpoint_sha256",
    "model_hash",
    "checkpoint.hash",
    "checkpoint.sha256",
    "identity.checkpoint_hash",
];

fn reported_hash(health: &Value, explicit: Option<&str>, candidates: &[&str]) -> Value {
    let fields = explicit
        .map(|x| vec![x])
        .unwrap_or_else(|| candidates.to_vec());
    fields
        .into_iter()
        .find_map(|key| {
            health
                .path(key)
                .str()
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(s)
        })
        .unwrap_or(Value::Null)
}

fn reported_build_sha(health: &Value) -> Value {
    BUILD_FIELDS.iter().find_map(|field| {
        let value = health.path(field);
        value.str().filter(|text| !text.trim().is_empty()).map(|_| value.clone())
    }).unwrap_or(Value::Null)
}

pub fn resolve_model(args: &Config, health: &Value) -> Config {
    let mut resolved = args.clone();
    if !resolved.model_explicit {
        if let Some(model) = health.get("model").str().filter(|id| !id.trim().is_empty()) {
            resolved.model = model.to_owned();
        }
    }
    resolved
}

pub fn validate_health(health: &Value, args: &Config, pinned: bool) -> Result<Value, String> {
    match health.get("model") {
        Value::Null => {}
        Value::String(model) if model.trim().is_empty() || model == &args.model => {}
        _ => return Err("/health model differs from --model".into()),
    }
    let decoding = super::drafting::metadata(health, args.draft.as_deref())?;
    if (pinned || args.checkpoint_hash.is_some())
        && matches!(health.path("vade.build.dirty"), Value::Bool(true))
    {
        return Err("pinned receipt requires a clean server build".into());
    }
    let identity = o([
        ("model", s(&args.model)),
        ("decoding", decoding),
        ("server_build_sha", reported_build_sha(health)),
        (
            "build_hash",
            reported_hash(health, args.build_hash_field.as_deref(), BUILD_FIELDS),
        ),
        (
            "checkpoint_hash",
            reported_hash(
                health,
                args.checkpoint_hash_field.as_deref(),
                CHECKPOINT_FIELDS,
            ),
        ),
        ("vade.ptx_sha256", health.path("vade.ptx_sha256").clone()),
        (
            "vade.layer_ptx_sha256",
            health.path("vade.layer_ptx_sha256").clone(),
        ),
        ("vade.numerics", health.path("vade.numerics").clone()),
    ]);
    if pinned && !identity.get("build_hash").yes() {
        return Err("8k receipt requires a server-reported /health build hash".into());
    }
    if let Some(pin) = &args.checkpoint_hash {
        let actual = identity
            .get("checkpoint_hash")
            .str()
            .ok_or("checkpoint pin cannot be verified: /health reports no checkpoint hash")?;
        if actual.to_lowercase() != pin.to_lowercase() {
            return Err("server checkpoint hash differs from --checkpoint-hash".into());
        }
    }
    Ok(identity)
}

/// A fresh launcher can serve before its background checkpoint hash completes.
/// Wait only for that transient state; wrong URLs and failed hashes fail directly.
pub fn wait_for_checkpoint(url: &str, args: &Config) -> Result<Value, String> {
    let end = timeout_deadline(args.startup_timeout, "--startup-timeout")?;
    loop {
        if (args.interrupted)() { return Err("benchmark interrupted".into()); }
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("/health checkpoint hash pending: --startup-timeout expired".into());
        }
        match stream::get_health_with_cancel(url, args.timeout.min(left.as_secs_f64()), args.interrupted) {
            Err(error) if error == "/health checkpoint hash pending" => {
                thread::sleep(Duration::from_millis(200).min(left));
            }
            result => return result,
        }
    }
}

// Python str.splitlines also recognizes these Unicode and control separators.
fn split_lines(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if matches!(
            ch,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            result.push(&text[start..index]);
            start = index + ch.len_utf8();
            if ch == '\r' && chars.peek().is_some_and(|(_, c)| *c == '\n') {
                start = chars.next().unwrap().0 + 1;
            }
        }
    }
    if start < text.len() {
        result.push(&text[start..]);
    }
    result
}

fn python_space(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}')
}
fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(value) => Some(f64::from(*value)),
        _ => value.num(),
    }
}

pub fn load_prompts(args: &Config) -> Result<(Vec<Value>, Value), String> {
    if let Some(path) = &args.prompts {
        let raw = fs::read(path).map_err(|e| e.to_string())?;
        let text = std::str::from_utf8(&raw).map_err(|e| e.to_string())?;
        let bodies: Vec<_> = split_lines(text)
            .into_iter()
            .filter(|line| !line.trim_matches(python_space).is_empty())
            .map(|line| {
                o([(
                    "messages",
                    a(vec![o([("role", s("user")), ("content", s(line))])]),
                )])
            })
            .collect();
        if bodies.is_empty() {
            return Err("--prompts has no nonempty lines".into());
        }
        return Ok((
            bodies,
            o([
                ("file", s(path)),
                ("sha256", s(digest(&raw))),
                ("format", s("one literal prompt per nonempty line")),
            ]),
        ));
    }
    let Some(root) = args.bodies_dir.as_deref() else {
        let text = include_str!("prompt.txt");
        return Ok((
            vec![o([("messages", a(vec![o([("role", s("user")), ("content", s(text))])]))])],
            o([("builtin", s("writing-v1")), ("sha256", s(digest(text.as_bytes())))]),
        ));
    };
    let root = Path::new(root);
    let context = root.join("pp448.body");
    let raw = fs::read(&context).map_err(|e| e.to_string())?;
    let body = parse(&raw)?;
    let content = body
        .get("messages")
        .arr()
        .first()
        .and_then(|x| x.get("content").str())
        .ok_or("pp448.body needs messages[0].content")?;
    let stem = content
        .rsplit_once("\nSummarize")
        .map(|(x, _)| x)
        .unwrap_or(content);
    let capture = root.parent().unwrap_or(Path::new(".")).join("capture");
    let mut paths: Vec<_> = fs::read_dir(capture)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|x| x.path())
        .filter(|x| x.extension().is_some_and(|ext| ext == "body"))
        .collect();
    paths.sort();
    let mut readers = BTreeMap::new();
    for path in paths {
        let data = fs::read(&path).map_err(|e| e.to_string())?;
        let body = parse(&data)?;
        let messages = body.get("messages").arr();
        if messages.len() != 1
            || body.get("max_tokens").num() != Some(128.0)
            || !matches!(body.get("cache"), Value::Bool(false))
        {
            continue;
        }
        let text = messages[0].get("content").str().unwrap_or("");
        let suffix = text.strip_suffix('\n').unwrap_or(text);
        let reader = suffix
            .rsplit_once("\nGive a separate response for reader ")
            .and_then(|(_, digits)| digits.strip_suffix('.'))
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|x| x.is_ascii_digit()))
            .and_then(|digits| digits.parse::<usize>().ok());
        if let Some(reader) = reader {
            if text.starts_with(stem)
                && [("temperature", 0.0), ("top_p", 1.0), ("top_k", 0.0)]
                    .into_iter()
                    .all(|(key, value)| numeric(body.get(key)) == Some(value))
            {
                readers.entry(reader).or_insert_with(|| {
                    (
                        body,
                        o([
                            ("file", s(path.to_string_lossy())),
                            ("sha256", s(digest(&data))),
                        ]),
                    )
                });
            }
        }
    }
    if (1..=16).any(|i| !readers.contains_key(&i)) {
        return Err("frozen request readers 1..16 missing; provide --prompts FILE".into());
    }
    let mut bodies = Vec::new();
    let mut sources = Vec::new();
    for i in 1..=16 {
        let (body, source) = readers.remove(&i).unwrap();
        bodies.push(body);
        sources.push(source);
    }
    Ok((
        bodies,
        o([
            ("context_body", s(context.to_string_lossy())),
            ("context_sha256", s(digest(&raw))),
            ("reader_bodies", a(sources)),
            (
                "note",
                s(
                    "frozen shared context in bodies/, complete requests in sibling capture/; no prompt text regenerated",
                ),
            ),
        ]),
    ))
}

pub fn log_mark(path: Option<&str>) -> Value {
    let Some(path) = path else {
        return Value::Null;
    };
    let metadata = fs::metadata(path).ok();
    o([
        ("offset", n(metadata.as_ref().map_or(0, |x| x.len()))),
        ("inode", metadata.map(|x| n(x.ino())).unwrap_or(Value::Null)),
    ])
}

fn word_char(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

pub fn read_batches(path: Option<&str>, mark: &Value) -> Value {
    let Some(path) = path else {
        return Value::Null;
    };
    let offset = mark.get("offset").uint().unwrap_or(0);
    let mut result = o([
        ("path", s(path)),
        ("start_offset", n(offset)),
        ("rows_per_step", a(vec![])),
        ("step_count", n(0)),
        ("row_histogram", o([])),
        ("server_bytes", o([])),
    ]);
    let read = (|| -> Result<(), String> {
        let metadata = fs::metadata(path).map_err(|_| "rank-0 log missing".to_string())?;
        if metadata.len() < offset
            || mark
                .get("inode")
                .uint()
                .is_some_and(|i| i != metadata.ino())
        {
            return Err("rank-0 log rotated/truncated during cell".into());
        }
        let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        file.take(metadata.len() - offset)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        result.put("end_offset", n(metadata.len()));
        let text = String::from_utf8_lossy(&data);
        let mut rows = Vec::new();
        let mut bytes = BTreeMap::<String, u64>::new();
        for line in split_lines(&text) {
            for (index, _) in line.match_indices("flash_serve batch rows=") {
                let tail = &line[index + "flash_serve batch rows=".len()..];
                let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
                if !digits.is_empty()
                    && tail[digits.len()..]
                        .chars()
                        .next()
                        .is_none_or(|c| !word_char(c))
                {
                    if let Ok(value) = digits.parse::<u64>() {
                        rows.push(n(value));
                        break;
                    }
                }
            }
            // Match the existing log counter grammar: lowercase/underscore names,
            // decimal nonnegative values, and word boundaries on each side.
            for (index, _) in line.match_indices("_bytes=") {
                let before = &line[..index];
                let start = before
                    .char_indices()
                    .rev()
                    .find(|(_, c)| !c.is_ascii_lowercase() && *c != '_')
                    .map_or(0, |(i, c)| i + c.len_utf8());
                if start == index || line[..start].chars().next_back().is_some_and(word_char) {
                    continue;
                }
                let tail = &line[index + "_bytes=".len()..];
                let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
                if digits.is_empty() || tail[digits.len()..].chars().next().is_some_and(word_char) {
                    continue;
                }
                if let Ok(value) = digits.parse::<u64>() {
                    let name = line[start..index + "_bytes".len()].to_string();
                    bytes
                        .entry(name)
                        .and_modify(|old| *old = (*old).max(value))
                        .or_insert(value);
                }
            }
        }
        let mut histogram = BTreeMap::<String, u64>::new();
        for row in &rows {
            *histogram
                .entry(row.uint().unwrap().to_string())
                .or_default() += 1;
        }
        result.put("step_count", n(rows.len()));
        result.put("rows_per_step", a(rows));
        result.put(
            "row_histogram",
            Value::Object(
                histogram
                    .into_iter()
                    .map(|(key, count)| (key, n(count)))
                    .collect(),
            ),
        );
        result.put(
            "server_bytes",
            Value::Object(
                bytes
                    .into_iter()
                    .map(|(key, count)| (key, n(count)))
                    .collect(),
            ),
        );
        Ok(())
    })();
    if let Err(error) = read {
        result.put("error", s(error));
    }
    result
}

pub fn observed_bytes(objects: &[&Value]) -> Value {
    let mut values = BTreeMap::<String, Value>::new();
    for object in objects {
        if let Value::Object(bytes) = stream::memory_bytes(object) {
            for (key, value) in bytes {
                let old = values.entry(key).or_insert_with(|| n(0));
                if value.integer_cmp(old).is_some_and(|order| order.is_ge()) {
                    *old = value;
                }
            }
        }
    }
    Value::Object(values.into_iter().collect())
}

fn csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut value = String::new();
    let mut quoted = false;
    let mut started = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if quoted {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    value.push('"');
                } else {
                    quoted = false;
                }
            } else {
                value.push(ch);
            }
        } else if ch == ',' {
            fields.push(std::mem::take(&mut value));
            started = false;
        } else if !started && ch == ' ' {
            continue;
        } else if !started && ch == '"' {
            quoted = true;
            started = true;
        } else {
            value.push(ch);
            started = true;
        }
    }
    fields.push(value);
    fields
}

pub fn parse_observation(stdout: &str) -> Result<Value, String> {
    let lines: Vec<_> = split_lines(stdout)
        .into_iter()
        .map(|x| x.trim_matches(python_space))
        .filter(|x| !x.is_empty())
        .collect();
    if lines.len() < 2 {
        return Err("observe stdout needs GPU CSV row(s), then /proc/uptime".into());
    }
    let uptime = lines
        .last()
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<f64>()
        .map_err(|_| "invalid host uptime")?;
    let mut gpus = Vec::new();
    for line in &lines[..lines.len() - 1] {
        let row = csv_row(line);
        if row.len() != 5 {
            return Err(
                "observe GPU CSV needs temperature, SM clock, memory clock, power, throttle mask"
                    .into(),
            );
        }
        let numbers: Vec<_> = row[..4]
            .iter()
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| "invalid observe GPU number")?;
        let mask = row[4].trim();
        let mask = if mask.to_ascii_lowercase().starts_with("0x") {
            u64::from_str_radix(&mask[2..], 16)
        } else {
            mask.parse::<u64>()
        }
        .map_err(|_| "invalid observe GPU number")?;
        if numbers.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("invalid observe GPU number".into());
        }
        gpus.push(o([
            ("temperature_c", n(numbers[0])),
            ("sm_clock_mhz", n(numbers[1])),
            ("memory_clock_mhz", n(numbers[2])),
            ("power_w", n(numbers[3])),
            ("throttle_reasons_active", n(mask)),
        ]));
    }
    if !uptime.is_finite() || uptime < 0.0 {
        return Err("invalid host uptime".into());
    }
    Ok(o([("gpus", a(gpus)), ("uptime_s", n(uptime))]))
}

pub fn timeout_duration(value: f64, name: &str) -> Result<Duration, String> {
    let duration = Duration::try_from_secs_f64(value)
        .map_err(|_| format!("{name} is outside the supported duration range"))?;
    if duration.is_zero() {
        return Err(format!("{name} must be at least one nanosecond"));
    }
    Ok(duration)
}

pub fn timeout_deadline(value: f64, name: &str) -> Result<Instant, String> {
    Instant::now()
        .checked_add(timeout_duration(value, name)?)
        .ok_or_else(|| format!("{name} is too large for a monotonic deadline"))
}
fn returncode(status: ExitStatus) -> Value {
    n(status
        .code()
        .unwrap_or_else(|| -status.signal().unwrap_or(0)))
}

/// A foreground shell and the process group created for this one invocation.
pub struct Launched {
    child: Child,
    group: u32,
    cleaned: bool,
}

impl Launched {
    fn spawn(
        command: &str,
        stdout: Stdio,
        stderr: Stdio,
        environment: &[(String, String)],
    ) -> Result<Self, String> {
        let child = Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .stdout(stdout)
            .stderr(stderr)
            .envs(environment.iter().map(|(k, v)| (k, v)))
            .process_group(0)
            .spawn()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            group: child.id(),
            child,
            cleaned: false,
        })
    }

    pub fn start_with_env(
        command: &str,
        log_path: &Path,
        environment: &[(String, String)],
    ) -> Result<Self, String> {
        if let Some(parent) = log_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .map_err(|e| e.to_string())?;
        let stderr = log.try_clone().map_err(|e| e.to_string())?;
        Self::spawn(command, log.into(), stderr.into(), environment)
    }

    pub fn wait_ready(&mut self, url: &str, args: &Config) -> Result<(), String> {
        let end = timeout_deadline(args.startup_timeout, "--startup-timeout")?;
        let mut error = "server did not start".to_string();
        while Instant::now() < end {
            if (args.interrupted)() {
                return Err("Interrupted".into());
            }
            if let Some(status) = self.child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!(
                    "launch command exited before readiness (code {})",
                    super::value::encode(&returncode(status))
                ));
            }
            match stream::get_health_with_cancel(url, args.timeout.min(2.0), args.interrupted) {
                Ok(health) => {
                    validate_health(&health, &resolve_model(args, &health), true)?;
                    return Ok(());
                }
                Err(e) => error = e,
            }
            thread::sleep(Duration::from_millis(250));
        }
        Err(format!("startup timeout: {error}"))
    }

    fn signal(&self, signal: &str) -> Result<(), String> {
        // /bin/kill receives only the exact process group created above. A group
        // that has already exited is normal; never discover or match other PIDs.
        let result = Command::new("/bin/kill")
            .args([signal, "--", &format!("-{}", self.group)])
            .env("LC_ALL", "C")
            .output()
            .map_err(|e| e.to_string())?;
        if result.status.success()
            || String::from_utf8_lossy(&result.stderr).contains("No such process")
        {
            Ok(())
        } else {
            Err(format!(
                "owned process group signal failed: {}",
                String::from_utf8_lossy(&result.stderr).trim()
            ))
        }
    }

    pub fn stop(&mut self) -> Result<(), String> {
        if self.cleaned {
            return Ok(());
        }
        self.signal("-TERM")?;
        let end = Instant::now() + Duration::from_secs(15);
        while self.child.try_wait().map_err(|e| e.to_string())?.is_none() && Instant::now() < end {
            thread::sleep(Duration::from_millis(25));
        }
        self.signal("-KILL")?;
        let end = Instant::now() + Duration::from_secs(5);
        while self.child.try_wait().map_err(|e| e.to_string())?.is_none() {
            if Instant::now() >= end {
                return Err("owned process did not exit after SIGKILL".into());
            }
            thread::sleep(Duration::from_millis(25));
        }
        self.cleaned = true;
        Ok(())
    }
}

impl Drop for Launched {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn capture(
    args: &Config,
    rank: usize,
    command: &str,
    phase: &str,
    label: &str,
    count: usize,
    repeat: usize,
) -> Value {
    let mut record = o([
        ("rank", n(rank)),
        ("phase", s(phase)),
        ("captured_utc", s(utc_now())),
    ]);
    let mut environment = args.command_env.clone();
    environment.extend(
        [
            ("VADE_RANK", rank.to_string()),
            ("VADE_PHASE", phase.to_string()),
            ("VADE_ARM", label.to_string()),
            ("VADE_CONCURRENCY", count.to_string()),
            ("VADE_REPEAT", repeat.to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value)),
    );
    let result = (|| -> Result<(), String> {
        if (args.interrupted)() {
            return Err("Interrupted".into());
        }
        let deadline = timeout_deadline(args.observe_timeout, "--observe-timeout")?;
        let mut launched = Launched::spawn(command, Stdio::piped(), Stdio::piped(), &environment)?;
        let (sender, receiver) = mpsc::channel();
        let readers: Vec<Box<dyn Read + Send>> = vec![
            Box::new(launched.child.stdout.take().unwrap()),
            Box::new(launched.child.stderr.take().unwrap()),
        ];
        for (index, mut reader) in readers.into_iter().enumerate() {
            let sender = sender.clone();
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let result = reader.read_to_end(&mut bytes).map(|_| bytes);
                let _ = sender.send((index, result));
            });
        }
        drop(sender);
        let mut outputs = [None, None];
        let mut status = None;
        let mut timed_out = false;
        let mut interrupted = false;
        loop {
            while let Ok((index, output)) = receiver.try_recv() {
                outputs[index] = Some(output.map_err(|e| e.to_string()));
            }
            if status.is_none() {
                status = launched.child.try_wait().map_err(|e| e.to_string())?;
            }
            if status.is_some() && outputs.iter().all(Option::is_some) {
                break;
            }
            if (args.interrupted)() {
                interrupted = true;
                break;
            }
            if Instant::now() >= deadline {
                timed_out = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let cleanup = launched.stop();
        let output_deadline = Instant::now() + Duration::from_secs(5);
        while outputs.iter().any(Option::is_none) {
            let remaining = output_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match receiver.recv_timeout(remaining) {
                Ok((index, output)) => outputs[index] = Some(output.map_err(|e| e.to_string())),
                Err(_) => break,
            }
        }
        let stdout = outputs[0].take().unwrap_or_else(|| {
            Err("observe stdout pipe remained open after owned-group cleanup".into())
        })?;
        let stderr = outputs[1].take().unwrap_or_else(|| {
            Err("observe stderr pipe remained open after owned-group cleanup".into())
        })?;
        let stdout = String::from_utf8_lossy(&stdout).into_owned();
        record.put("stdout", s(&stdout));
        record.put("stderr", s(String::from_utf8_lossy(&stderr)));
        let status = status
            .or(launched.child.try_wait().map_err(|e| e.to_string())?)
            .ok_or("observe command did not exit")?;
        record.put("returncode", returncode(status));
        if let Err(error) = cleanup {
            return Err(format!("observe cleanup: {error}"));
        }
        if interrupted {
            return Err("Interrupted".into());
        }
        if timed_out {
            return Err(format!(
                "observe timeout after {:?} s",
                args.observe_timeout
            ));
        }
        if !status.success() {
            return Err(format!(
                "observe command exited {}",
                super::value::encode(&returncode(status))
            ));
        }
        if let Value::Object(values) = parse_observation(&stdout)? {
            for (key, value) in values {
                record.put(&key, value);
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        record.put("error", s(error));
    }
    record
}

pub fn observe(args: &Config, phase: &str, label: &str, count: usize, repeat: usize) -> Vec<Value> {
    thread::scope(|scope| {
        let tasks: Vec<_> = args
            .observe_cmd
            .iter()
            .enumerate()
            .map(|(rank, command)| {
                scope.spawn(move || capture(args, rank, command, phase, label, count, repeat))
            })
            .collect();
        tasks
            .into_iter()
            .map(|task| {
                task.join()
                    .unwrap_or_else(|_| o([("error", s("observation worker panicked"))]))
            })
            .collect()
    })
}

pub fn observation_flags(before: &[Value], after: &[Value], context: &str) -> Vec<Value> {
    let mut flags = Vec::new();
    for (left, right) in before.iter().zip(after) {
        if left.get("error").yes() || right.get("error").yes() {
            continue;
        }
        let prefix = format!("rank={} {context}: ", left.get("rank").uint().unwrap_or(0));
        if right.get("uptime_s").num() < left.get("uptime_s").num() {
            flags.push(s(format!(
                "{prefix}RE-BASELINE uptime dropped (host reboot)"
            )));
        }
        if left.get("gpus").arr().len() != right.get("gpus").arr().len() {
            flags.push(s(format!("{prefix}GPU count changed")));
        }
        for (index, (x, y)) in left
            .get("gpus")
            .arr()
            .iter()
            .zip(right.get("gpus").arr())
            .enumerate()
        {
            let gpu = format!("{prefix}gpu={index} ");
            let sm_x = x.get("sm_clock_mhz").num().unwrap_or(0.0);
            let sm_y = y.get("sm_clock_mhz").num().unwrap_or(0.0);
            if sm_x != 0.0 && sm_y != 0.0 && (sm_x - sm_y).abs() / sm_x.max(sm_y) > 0.10 {
                flags.push(s(format!(
                    "{gpu}SM clock changed by more than 10% ({sm_x:?} -> {sm_y:?} MHz)"
                )));
            }
            let mem_x = x.get("memory_clock_mhz").num().unwrap_or(0.0);
            let mem_y = y.get("memory_clock_mhz").num().unwrap_or(0.0);
            if mem_x != mem_y {
                flags.push(s(format!(
                    "{gpu}memory clock changed ({mem_x:?} -> {mem_y:?} MHz)"
                )));
            }
            if x.get("throttle_reasons_active").uint() != y.get("throttle_reasons_active").uint() {
                flags.push(s(format!("{gpu}throttle reasons changed")));
            }
            if x.get("throttle_reasons_active").yes() || y.get("throttle_reasons_active").yes() {
                flags.push(s(format!("{gpu}RE-BASELINE active throttling")));
            }
        }
    }
    flags
}

fn utc_parts() -> (i64, i64, i64, u64, u64, u64, u32) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs();
    // Gregorian civil date from days since the Unix epoch, using 400-year eras.
    let days = (seconds / 86400) as i64 + 719468;
    let era = days.div_euclid(146097);
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (
        year,
        month,
        day,
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60,
        now.subsec_micros(),
    )
}

pub fn utc_now() -> String {
    let (year, month, day, hour, minute, second, micros) = utc_parts();
    let fraction = if micros == 0 {
        String::new()
    } else {
        format!(".{micros:06}")
    };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{fraction}+00:00")
}

#[cfg(test)]
mod checkpoint_wait_tests {
    use super::*;
    use std::{io::{BufRead, BufReader, Write}, net::TcpListener};

    fn health_server(states: &[&str]) -> (String, thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let states: Vec<String> = states.iter().map(|state| (*state).into()).collect();
        let server = thread::spawn(move || {
            for state in &states {
                let deadline = Instant::now() + Duration::from_secs(3);
                let mut socket = loop {
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "expected another /health request");
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) => panic!("mock health accept: {error}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with("GET /health HTTP/"));
                loop {
                    line.clear();
                    assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                    if line == "\r\n" { break; }
                }
                let body = format!(r#"{{"status":"ready","model":"fresh-launch-model","vade":{{"checkpoint":{{"hash_state":"{state}","sha256":"checkpoint-id"}}}}}}"#);
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            states.len()
        });
        (url, server)
    }

    fn config() -> Config {
        let mut args = Config::parse(&["--self-test".into()]).unwrap();
        args.startup_timeout = 2.;
        args.timeout = 1.;
        args
    }

    #[test]
    fn fresh_launch_waits_for_pending_checkpoint_then_returns_ready_health() {
        let (url, server) = health_server(&["pending", "ready"]);
        let result = wait_for_checkpoint(&url, &config()).unwrap();
        assert_eq!(result.get("model"), &s("fresh-launch-model"));
        assert_eq!(result.path("vade.checkpoint.hash_state"), &s("ready"));
        assert_eq!(result.path("vade.checkpoint.sha256"), &s("checkpoint-id"));
        assert_eq!(server.join().unwrap(), 2);
    }

    #[test]
    fn failed_checkpoint_is_returned_without_retrying() {
        for state in ["failed", "failed-stale", "stale"] {
            let (url, server) = health_server(&[state]);
            assert_eq!(wait_for_checkpoint(&url, &config()).unwrap_err(), format!("/health checkpoint hash {state}"));
            assert_eq!(server.join().unwrap(), 1);
        }
    }

    #[test]
    fn pending_checkpoint_honors_deadline_and_preconnection_cancellation() {
        let (url, server) = health_server(&["pending"]);
        let mut args = config();
        args.startup_timeout = 0.1;
        assert_eq!(wait_for_checkpoint(&url, &args).unwrap_err(), "/health checkpoint hash pending: --startup-timeout expired");
        assert_eq!(server.join().unwrap(), 1);
        args.interrupted = || true;
        assert_eq!(wait_for_checkpoint("invalid URL must not be used", &args).unwrap_err(), "benchmark interrupted");
    }
}
