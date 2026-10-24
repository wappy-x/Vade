use crate::{
    Config, Result,
    http::{Reply, get},
};
use engine_model::json::{self, Value};
use engine_tokenizer::chat::quote;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn num(v: Option<f64>) -> String {
    v.filter(|x| x.is_finite())
        .map(|x| format!("{x:.6}"))
        .unwrap_or("null".into())
}
fn uint(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or("null".into())
}
fn string(v: Option<&str>) -> String {
    v.map(quote).unwrap_or("null".into())
}
fn encode(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.clone(),
        Value::String(v) => quote(v),
        Value::Array(v) => format!("[{}]", v.iter().map(encode).collect::<Vec<_>>().join(",")),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!("{}:{}", quote(k), encode(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}
fn proc_bytes(pid: u32, field: &str) -> Option<u64> {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()?
        .lines()
        .find(|l| l.starts_with(field))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
#[derive(Default)]
pub struct Memory {
    rss_peak: Option<u64>,
    process_hwm: Option<u64>,
    client_hwm: Option<u64>,
    ledger: Option<Value>,
}
pub struct Monitor {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Option<u64>>>,
}
impl Monitor {
    pub fn start(c: &Config) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker = c.server_pid.map(|pid| {
            let stop = stop.clone();
            thread::spawn(move || {
                let mut peak = None;
                loop {
                    if let Some(rss) = proc_bytes(pid, "VmRSS:") {
                        peak = Some(peak.unwrap_or(0).max(rss));
                    }
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::park_timeout(Duration::from_millis(20));
                }
                peak
            })
        });
        Self { stop, worker }
    }
    pub fn finish(self, c: &Config) -> Memory {
        self.stop.store(true, Ordering::Relaxed);
        let rss_peak = self.worker.and_then(|w| {
            w.thread().unpark();
            w.join().ok().flatten()
        });
        let ledger = c
            .memory_file
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|bytes| json::parse(&bytes, json::Limits::default()).ok());
        Memory {
            rss_peak,
            process_hwm: c.server_pid.and_then(|pid| proc_bytes(pid, "VmHWM:")),
            client_hwm: proc_bytes(std::process::id(), "VmHWM:"),
            ledger,
        }
    }
}
impl Memory {
    fn json(&self) -> String {
        format!(
            "{{\"server_rss_sampled_peak_bytes\":{},\"server_lifetime_hwm_bytes\":{},\"client_lifetime_hwm_bytes\":{},\"ledger\":{},\"note\":\"RSS sampled every 20ms; HWM is process lifetime. RSS is not a CUDA allocation ledger; scopes must not be blindly summed. Missing measurements are null, never zero.\"}}",
            uint(self.rss_peak),
            uint(self.process_hwm),
            uint(self.client_hwm),
            self.ledger.as_ref().map(encode).unwrap_or("null".into())
        )
    }
    fn compact(&self) -> String {
        let gib = |n: Option<u64>| {
            n.map(|n| format!("{:.3}", n as f64 / 1073741824.))
                .unwrap_or("?".into())
        };
        let field = |name| {
            self.ledger
                .as_ref()
                .and_then(|v| get(v, name))
                .and_then(|v| v.u64().ok())
        };
        format!(
            "W{}/K{}/S{}/G{}/H{}",
            gib(field("weights_bytes")),
            gib(field("kv_peak_bytes")),
            gib(field("scratch_peak_bytes")),
            gib(field("device_peak_bytes")),
            gib(self.process_hwm)
        )
    }
}
fn reply_json(r: &Reply) -> String {
    let accepted = format!(
        "\"accepted_ms\":{},\"token_elapsed_ms\":{:?},",
        num(r.accepted_ms),
        r.token_elapsed_ms
    );
    format!(
        "{{{accepted}\"status\":{},\"error\":{},\"ttft_ms\":{},\"elapsed_ms\":{:.6},\"decode_tokens_per_sec\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"content_events\":{},\"token_ids\":{:?},\"content\":{},\"prefix_reused_tokens\":{},\"finish_reason\":{},\"response_model\":{},\"vade\":{}}}",
        quote(&r.status),
        string(r.error.as_deref()),
        num(r.ttft_ms),
        r.elapsed_ms,
        num(r.decode_tps),
        uint(r.prompt_tokens),
        uint(r.completion_tokens),
        r.content_events,
        r.token_ids,
        quote(&r.text),
        uint(r.reused_tokens),
        string(r.finish_reason.as_deref()),
        string(r.response_model.as_deref()),
        r.vade.as_ref().map(encode).unwrap_or_else(|| "null".into())
    )
}

pub struct Record {
    pub name: String,
    repeat: usize,
    concurrency: usize,
    max_tokens: usize,
    expected_prompt: Option<usize>,
    target_prompt: Option<usize>,
    pub replies: Vec<Reply>,
    elapsed: Duration,
    pub memory: Memory,
    pub note: Option<String>,
    pub background: Option<Reply>,
}
impl Record {
    pub fn new(
        name: &str,
        repeat: usize,
        concurrency: usize,
        max_tokens: usize,
        expected_prompt: Option<usize>,
        target_prompt: Option<usize>,
        replies: Vec<Reply>,
        elapsed: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            repeat,
            concurrency,
            max_tokens,
            expected_prompt,
            target_prompt,
            replies,
            elapsed,
            memory: Memory::default(),
            note: None,
            background: None,
        }
    }
    fn missing_sampling_echoes(&self) -> usize {
        self.replies
            .iter()
            .chain(self.background.iter())
            .filter(|r| r.vade.is_none())
            .count()
    }
    fn aggregate(&self) -> Option<f64> {
        if self.replies.is_empty()
            || self.replies.iter().any(|r| r.status != "ok")
            || self.elapsed.is_zero()
        {
            return None;
        }
        Some(
            self.replies
                .iter()
                .map(|r| r.completion_tokens.unwrap_or(0))
                .sum::<u64>() as f64
                / self.elapsed.as_secs_f64(),
        )
    }
    fn json(&self, c: &Config) -> String {
        format!(
            "{{{},\"case\":{},\"repeat\":{},\"concurrency\":{},\"max_tokens\":{},\"target_prompt_tokens\":{},\"tokenizer_prompt_tokens\":{},\"elapsed_ms\":{:.6},\"aggregate_tokens_per_sec\":{},\"memory\":{},\"note\":{},\"requests\":[{}],\"background\":{}}}",
            metadata(c),
            quote(&self.name),
            self.repeat + 1,
            self.concurrency,
            self.max_tokens,
            uint(self.target_prompt.map(|n| n as u64)),
            uint(self.expected_prompt.map(|n| n as u64)),
            self.elapsed.as_secs_f64() * 1000.,
            num(self.aggregate()),
            self.memory.json(),
            string(self.note.as_deref()),
            self.replies
                .iter()
                .map(reply_json)
                .collect::<Vec<_>>()
                .join(","),
            self.background
                .as_ref()
                .map(reply_json)
                .unwrap_or("null".into())
        )
    }
}
fn metadata(c: &Config) -> String {
    let sampling = if c.only == "followup" || c.only == "shared-system" {
        "{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"thinking\":\"disabled through messages and server template\"}"
    } else {
        "{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"enable_thinking\":false,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0}"
    };
    format!(
        "\"model\":{},\"precision\":{},\"sparks\":{},\"label\":{},\"sampling\":{sampling}",
        quote(&c.model),
        quote(&c.precision),
        c.sparks,
        quote(&c.label)
    )
}
pub fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some((v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.)
}
pub struct Report {
    pub records: Vec<Record>,
    started_unix: u64,
}
impl Report {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    pub fn skip(&mut self, c: &Config, name: &str, reason: &str) -> Result<()> {
        let mut record = Record::new(name, 0, 0, 0, None, None, Vec::new(), Duration::ZERO);
        record.note = Some(reason.into());
        self.push(c, record)
    }
    pub fn push(&mut self, c: &Config, mut record: Record) -> Result<()> {
        for r in &mut record.replies {
            if r.status == "ok"
                && record.expected_prompt.is_some()
                && r.prompt_tokens != record.expected_prompt.map(|n| n as u64)
            {
                r.status = "error".into();
                r.error = Some("server and local prompt token counts differ".into());
                r.decode_tps = None;
            }
            if r.status == "ok"
                && r.response_model
                    .as_deref()
                    .is_some_and(|model| model != c.model)
            {
                r.status = "error".into();
                r.error = Some("response model differs from requested model".into());
                r.decode_tps = None;
            }
        }
        for reply in &record.replies {
            crate::stderr_line(format_args!(
                "{} {} ttft_ms={} completion={} decode_tps={}{}",
                record.name,
                reply.status,
                num(reply.ttft_ms),
                uint(reply.completion_tokens),
                num(reply.decode_tps),
                reply
                    .error
                    .as_ref()
                    .map(|e| format!(" {e}"))
                    .unwrap_or_default()
            ));
        }
        let missing = record.missing_sampling_echoes();
        if missing > 0 {
            crate::stderr_line(format_args!("warning: {}: {missing} replies missing vade.sampling echo; greedy settings unverified (recorded as vade:null)", record.name));
        }
        self.records.push(record);
        self.write(c)
    }
    pub fn write(&self, c: &Config) -> Result<()> {
        let rows = self
            .records
            .iter()
            .map(|r| r.json(c))
            .collect::<Vec<_>>()
            .join(",\n");
        let cache_match = self
            .cache_matches_fresh()
            .map(|v| v.to_string())
            .unwrap_or("null".into());
        let missing_sampling_echoes = self.records.iter().map(Record::missing_sampling_echoes).sum::<usize>();
        let body = format!(
            "{{\"schema_version\":1,\"missing_sampling_echoes\":{missing_sampling_echoes},\"cache_matches_fresh\":{cache_match},\"started_unix\":{},\"method\":\"TTFT starts immediately before TCP connect and ends on first nonempty delta.content. Decode=(usage completion tokens minus generated tokens through first content)/(finish time minus first-content time); if token IDs are absent subtract one. EOS is counted by usage. Aggregate=sum completion tokens/group wall time including prefill and queueing, only when all requests succeed. Followup and shared-system send explicit greedy sampling (including non-standard top_k and repetition_penalty), /no_think messages and server template defaults; cache policy is the server default. Present sampling echoes must be greedy; absent echoes are null, warned on stderr and counted in missing_sampling_echoes (including background and unfinished replies). Other cases send cache=false except the explicit capability cache case. No chunk-count token estimates.\",\"timeout_secs\":{},\"budget_secs\":{},\"records\":[\n{}\n]}}\n",
            self.started_unix,
            c.timeout.as_secs(),
            c.budget.as_secs(),
            rows
        );
        let tmp = format!("{}.tmp", c.output);
        std::fs::write(&tmp, body).map_err(|e| e.to_string())?;
        std::fs::rename(tmp, &c.output).map_err(|e| e.to_string())
    }
    pub fn cache_matches_fresh(&self) -> Option<bool> {
        let fresh = self
            .records
            .iter()
            .find(|r| r.name == "cache/fresh-followup")?
            .replies
            .first()?;
        let cached = self
            .records
            .iter()
            .filter(|r| r.name == "cache/followup")
            .flat_map(|r| &r.replies)
            .collect::<Vec<_>>();
        if fresh.status != "ok" || cached.is_empty() || cached.iter().any(|r| r.status != "ok") {
            return None;
        }
        Some(
            cached
                .iter()
                .all(|r| r.token_ids == fresh.token_ids && r.text == fresh.text),
        )
    }
    pub fn print(&self, c: &Config) -> Result<()> {
        crate::write_output(format_args!(
            "case                         ok/n  in     out    first_TTFT_ms first_decode_t/s first_agg_t/s  memory_GiB(W/K/S/G/H)  model precision sparks sampling"
        ), true)?;
        let mut groups: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
        for r in &self.records {
            groups.entry(&r.name).or_default().push(r);
        }
        let show = |v: Option<f64>| v.map(|v| format!("{v:.2}")).unwrap_or("-".into());
        for (name, records) in groups {
            let replies = records
                .iter()
                .flat_map(|r| r.replies.iter())
                .collect::<Vec<_>>();
            let ok = replies.iter().filter(|r| r.status == "ok").count();
            let first = &records[0].replies;
            let ttft = median(
                first
                    .iter()
                    .filter(|r| r.status == "ok" || (c.only == "shared-system" && r.status == "timeout"))
                    .filter_map(|r| r.ttft_ms)
                    .collect(),
            );
            let decode = median(
                first
                    .iter()
                    .filter(|r| r.status == "ok")
                    .filter_map(|r| r.decode_tps)
                    .collect(),
            );
            let aggregate = records[0].aggregate();
            let input = replies
                .iter()
                .find_map(|r| r.prompt_tokens)
                .or(records[0].expected_prompt.map(|n| n as u64));
            let output = replies
                .iter()
                .filter_map(|r| r.completion_tokens)
                .sum::<u64>();
            let memory = records
                .iter()
                .max_by_key(|r| r.memory.process_hwm)
                .unwrap()
                .memory
                .compact();
            crate::write_output(format_args!(
                "{name:<28} {ok:>2}/{:<2} {:>6} {output:>6} {:>10} {:>9} {:>8}  {memory}  {} {} {} {}",
                replies.len(),
                input.map(|v| v.to_string()).unwrap_or("?".into()),
                show(ttft),
                show(decode),
                show(aggregate),
                c.model,
                c.precision,
                c.sparks,
                if c.only == "followup" || c.only == "shared-system" { "T0/p1/k0/s42/think0(messages/template)/rp1/pp0/fp0" } else { "T0/p1/k0/s42/think0/rp1/pp0/fp0" }
            ), true)?;
            if ok < replies.len() {
                crate::write_output(format_args!(
                    "  incomplete: {}",
                    replies
                        .iter()
                        .filter(|r| r.status != "ok")
                        .map(|r| format!(
                            "{} after {:.2}s{}",
                            r.status,
                            r.elapsed_ms / 1000.,
                            if r.ttft_ms.is_none() && r.accepted_ms.is_some() {
                                " without content after acceptance (TTFT exceeds this deadline)"
                            } else if r.ttft_ms.is_none() {
                                " without content; acceptance not observed"
                            } else {
                                ""
                            }
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                ), true)?;
            }
        }
        crate::write_output(format_args!(
            "Headline: first repeat only (median across concurrent clients); repeats use distinct deterministic prompts. Aggregate includes prefill. Out is total across repeats/clients. Memory: weights/KV peak/scratch peak/device peak/server lifetime RSS HWM; ? = unmeasured. See JSON for every request."
        ), true)?;
        if c.only == "shared-system" {
            crate::write_output(format_args!("TTFT includes observed first content from clients that later time out; aggregate requires all 16 final usage responses."), true)?;
        }
        if let Some(equal) = self.cache_matches_fresh() {
            crate::write_output(format_args!("Cached follow-ups match fresh token IDs and content: {equal}"), true)?;
        }
        Ok(())
    }
}
