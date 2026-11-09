//! Per-role summaries and the JSON and table renderings of a workload report.
use super::*;

#[derive(Debug)]
struct Summary {
    expected: usize,
    attempted: usize,
    completed: usize,
    errors: usize,
    cache_pairs: usize,
    cached_capped: u128,
    reusable: u128,
    admission_based: usize,
    output_samples: usize,
    output_tokens: u64,
    ttft_samples: usize,
    ttft_p50_ms: Option<f64>,
    ttft_p95_ms: Option<f64>,
    output_tok_s: Option<f64>,
    measured_output_tok_s: Option<f64>,
    wall_s: f64,
    requests: RequestCounts,
}
fn add_counts(to: &mut RequestCounts, counts: RequestCounts) {
    to.capabilities += counts.capabilities;
    to.admissions += counts.admissions;
    to.admitted += counts.admitted;
    to.refused += counts.refused;
    to.generations += counts.generations;
    to.http_429 += counts.http_429;
    to.http_503 += counts.http_503;
}
pub(super) fn percentile(sorted: &[f64], percent: usize) -> Option<f64> {
    if sorted.is_empty() {
        None
    } else {
        Some(sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)])
    }
}
impl Summary {
    fn new<'a>(samples: impl Iterator<Item = &'a Sample>, expected: usize) -> Self {
        let mut result = Self {
            expected,
            attempted: 0,
            completed: 0,
            errors: 0,
            cache_pairs: 0,
            cached_capped: 0,
            reusable: 0,
            admission_based: 0,
            output_samples: 0,
            output_tokens: 0,
            ttft_samples: 0,
            ttft_p50_ms: None,
            ttft_p95_ms: None,
            output_tok_s: None,
            measured_output_tok_s: None,
            wall_s: 0.0,
            requests: RequestCounts::default(),
        };
        let mut latencies = Vec::new();
        let (mut start, mut end) = (f64::INFINITY, 0.0f64);
        for sample in samples.filter(|s| s.turn >= 2) {
            result.attempted += 1;
            add_counts(&mut result.requests, sample.requests);
            start = start.min(sample.start_s);
            end = end.max(sample.end_s);
            if !sample.completed() {
                result.errors += 1;
                continue;
            }
            result.completed += 1;
            if let Some(t) = sample.ttft_ms {
                latencies.push(t);
            }
            if let Some(tokens) = sample.usage.completion_tokens {
                result.output_samples += 1;
                result.output_tokens = result.output_tokens.saturating_add(tokens);
            }
            if let Some((c, u)) = sample.usage.cached_tokens.zip(sample.u) {
                result.cache_pairs += 1;
                result.cached_capped += u128::from(c.min(u));
                result.reusable += u128::from(u);
                result.admission_based += usize::from(matches!(
                    sample.u_basis,
                    "exact_rendered_history_counts" | "admission_plus_reported_output"
                ));
            }
        }
        result.wall_s = if start.is_finite() {
            (end - start).max(0.0)
        } else {
            0.0
        };
        latencies.sort_by(f64::total_cmp);
        result.ttft_samples = latencies.len();
        result.ttft_p50_ms = percentile(&latencies, 50);
        result.ttft_p95_ms = percentile(&latencies, 95);
        if result.wall_s > 0.0 && result.output_samples > 0 {
            result.measured_output_tok_s = Some(result.output_tokens as f64 / result.wall_s);
            if result.output_samples == expected {
                result.output_tok_s = result.measured_output_tok_s;
            }
        }
        result
    }
    fn paired_ratio(&self) -> Option<f64> {
        (self.reusable > 0).then(|| self.cached_capped as f64 / self.reusable as f64)
    }
    fn ratio(&self) -> Option<f64> {
        if self.cache_pairs == self.expected && self.expected > 0 {
            self.paired_ratio()
        } else {
            None
        }
    }
    fn target(&self) -> &'static str {
        match self.ratio() {
            Some(_) if self.cached_capped * 100 >= self.reusable * 98 => "estimate_meets_target",
            Some(_) => "estimate_below_target",
            None => "insufficient_data",
        }
    }
    fn json(&self) -> Value {
        obj([
            ("expected_samples", n(self.expected)),
            ("attempted_samples", n(self.attempted)),
            ("completed_samples", n(self.completed)),
            ("errors", n(self.errors)),
            (
                "unattempted_samples",
                n(self.expected.saturating_sub(self.attempted)),
            ),
            ("cache_pairs", n(self.cache_pairs)),
            ("sum_min_c_u", n(self.cached_capped)),
            ("sum_u", n(self.reusable)),
            ("cache_ratio", opt_f(self.ratio())),
            ("paired_cache_ratio", opt_f(self.paired_ratio())),
            ("u_estimated", Value::Bool(true)),
            ("admission_based_samples", n(self.admission_based)),
            ("target", n(0.98)),
            ("target_status", s(self.target())),
            ("exact_target_verified", Value::Bool(false)),
            ("ttft_samples", n(self.ttft_samples)),
            ("ttft_p50_ms", opt_f(self.ttft_p50_ms)),
            ("ttft_p95_ms", opt_f(self.ttft_p95_ms)),
            ("output_samples", n(self.output_samples)),
            ("output_tokens", n(self.output_tokens)),
            ("output_tok_s", opt_f(self.output_tok_s)),
            ("measured_output_tok_s", opt_f(self.measured_output_tok_s)),
            ("wall_s", n(self.wall_s)),
            ("requests", counts_json(self.requests)),
        ])
    }
    fn throughput_json(&self) -> Value {
        // A combined cache target cannot establish either role's result.
        let Value::Object(mut fields) = self.json() else {
            unreachable!()
        };
        fields.retain(|(key, _)| {
            !matches!(
                key.as_str(),
                "cache_pairs"
                    | "sum_min_c_u"
                    | "sum_u"
                    | "cache_ratio"
                    | "paired_cache_ratio"
                    | "u_estimated"
                    | "admission_based_samples"
                    | "target"
                    | "target_status"
                    | "exact_target_verified"
            )
        });
        Value::Object(fields)
    }
}
fn counts_json(c: RequestCounts) -> Value {
    obj([
        ("capabilities", n(c.capabilities)),
        ("admissions", n(c.admissions)),
        ("admitted", n(c.admitted)),
        ("refused", n(c.refused)),
        ("generations", n(c.generations)),
        ("http_429", n(c.http_429)),
        ("http_503", n(c.http_503)),
    ])
}
fn input_json(input: &InputCount) -> Value {
    match *input {
        InputCount::Unknown => obj([("kind", s("unknown"))]),
        InputCount::Estimate(p) => obj([
            ("kind", s("wire_bytes_plus_256_estimate")),
            ("prompt_tokens", n(p)),
        ]),
        InputCount::Exact {
            prompt_tokens,
            output_tokens,
            context_limit,
            admitted,
        } => obj([
            ("kind", s("exact_admission")),
            ("prompt_tokens", n(prompt_tokens)),
            ("output_reservation", n(output_tokens)),
            ("context_limit", n(context_limit)),
            ("admitted", Value::Bool(admitted)),
        ]),
    }
}
impl Sample {
    fn json(&self) -> Value {
        obj([
            ("turn", n(self.turn)),
            ("measured", Value::Bool(self.turn >= 2)),
            ("start_s", n(self.start_s)),
            ("end_s", n(self.end_s)),
            ("elapsed_s", n(self.duration())),
            ("ttft_ms", opt_f(self.ttft_ms)),
            ("outcome", s(self.outcome)),
            ("error", self.error.map(s).unwrap_or(Value::Null)),
            ("input", input_json(&self.input)),
            ("reported_prompt_tokens", opt_n(self.usage.prompt_tokens)),
            (
                "reported_output_tokens",
                opt_n(self.usage.completion_tokens),
            ),
            ("reported_cached_tokens", opt_n(self.usage.cached_tokens)),
            ("invalid_usage", Value::Bool(self.invalid_usage)),
            ("wire_bytes", n(self.wire_bytes)),
            ("history_bytes", n(self.history_bytes)),
            ("requests", counts_json(self.requests)),
            ("u", opt_n(self.u)),
            ("u_basis", s(self.u_basis)),
            ("u_estimated", Value::Bool(self.u_estimated)),
            (
                "rendered_previous_tokens",
                opt_n(self.rendered_previous_tokens),
            ),
            (
                "prefix_count_requests",
                counts_json(self.prefix_count_requests),
            ),
            (
                "prefix_count_error",
                self.prefix_count_error.map(s).unwrap_or(Value::Null),
            ),
            (
                "prefix_count_elapsed_ms",
                opt_f(self.prefix_count_elapsed_ms),
            ),
            (
                "min_c_u",
                opt_n(self.u.zip(self.usage.cached_tokens).map(|(u, c)| u.min(c))),
            ),
            ("cache_domain_changes", n(self.cache_domain_changes)),
            ("prefix_preserved", Value::Bool(self.prefix_preserved)),
            (
                "last_http_status",
                opt_n(self.last_http_status.map(u64::from)),
            ),
            ("last_http_stage", s(self.last_http_stage)),
        ])
    }
}
impl Report {
    fn role_summary(&self, role: Role) -> Summary {
        let planned_sessions = match role {
            Role::Main => self.config.main_sessions,
            Role::Subordinate => self.config.sessions - self.config.main_sessions,
        };
        Summary::new(
            self.sessions
                .iter()
                .filter(|s| s.role == role)
                .flat_map(|s| s.samples.iter()),
            planned_sessions * (self.config.turns - 1),
        )
    }
    pub fn successful(&self) -> bool {
        !self.timed_out
            && self.sessions.len() == self.config.sessions
            && self
                .sessions
                .iter()
                .all(|s| s.stopped.is_none() && s.samples.len() == self.config.turns)
    }
    pub fn to_json(&self) -> io::Result<String> {
        let mut all_counts = RequestCounts::default();
        let mut prefix_counts = RequestCounts::default();
        for sample in self.sessions.iter().flat_map(|s| &s.samples) {
            add_counts(&mut all_counts, sample.requests);
            add_counts(&mut prefix_counts, sample.prefix_count_requests);
        }
        let c = &self.config;
        let value = obj([
            ("schema", s("vade.workload.v1")),
            (
                "method_source",
                s("https://github.com/andreasgru/ds41-flash-hybrid/blob/main/METHODS.md"),
            ),
            ("measurement_from_turn", n(2)),
            (
                "u_definition",
                s("token LCP(current prompt, previous prompt + previous generated output)"),
            ),
            (
                "u_limitation",
                s(
                    "Estimated LCP: prefer min(current exact prompt count, exact admission count of retained previous history including full assistant reply). Counts include renderer framing, not token IDs; templates, generation headers and retokenization can violate prefix continuity. Without exact admission use previous prompt + output counts.",
                ),
            ),
            (
                "ttft_definition",
                s(
                    "native request start to first generated SSE delta (content, reasoning or tool function data), including discovery/admission; unknown for JSON responses",
                ),
            ),
            (
                "throughput_definition",
                s(
                    "reported output tokens / warm wall window; includes discovery, admission, prefill and response validation; not decode-only speed",
                ),
            ),
            (
                "fallback_count_definition",
                s(
                    "server usage if present, else ceil(request wire bytes / 4) + 256; output fallback ceil(serialized assistant bytes / 4)",
                ),
            ),
            ("percentile_method", s("nearest rank")),
            (
                "config",
                obj([
                    ("sessions", n(c.sessions)),
                    ("main_sessions", n(c.main_sessions)),
                    ("turns", n(c.turns)),
                    ("main_system_bytes", n(c.main_system_bytes)),
                    ("sub_system_bytes", n(c.sub_system_bytes)),
                    ("main_user_bytes", n(c.main_user_bytes)),
                    ("sub_user_bytes", n(c.sub_user_bytes)),
                    ("main_tool_bytes", n(c.main_tool_bytes)),
                    ("sub_tool_bytes", n(c.sub_tool_bytes)),
                    ("output_tokens", n(c.output_tokens)),
                    ("seed", n(c.seed)),
                    ("max_seconds", n(c.max_seconds)),
                ]),
            ),
            ("elapsed_s", n(self.elapsed_s)),
            ("timed_out", Value::Bool(self.timed_out)),
            ("workload_completed", Value::Bool(self.successful())),
            ("all_turn_requests", counts_json(all_counts)),
            ("prefix_count_requests", counts_json(prefix_counts)),
            (
                "roles",
                obj([
                    ("main", self.role_summary(Role::Main).json()),
                    ("subordinate", self.role_summary(Role::Subordinate).json()),
                ]),
            ),
            (
                "aggregate",
                Summary::new(
                    self.sessions.iter().flat_map(|s| s.samples.iter()),
                    c.sessions * (c.turns - 1),
                )
                .throughput_json(),
            ),
            (
                "sessions",
                Value::Array(
                    self.sessions
                        .iter()
                        .map(|session| {
                            obj([
                                ("session", n(session.session)),
                                ("role", s(session.role.label())),
                                ("stopped", session.stopped.map(s).unwrap_or(Value::Null)),
                                (
                                    "summary",
                                    Summary::new(session.samples.iter(), c.turns - 1).json(),
                                ),
                                (
                                    "turns",
                                    Value::Array(
                                        session.samples.iter().map(Sample::json).collect(),
                                    ),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ]);
        json::stringify(&value)
    }
    pub fn table(&self) -> String {
        let mut text =
            String::from("Native workload · turns 2 onward · R target ≥ 0.98 per role\n\n");
        text.push_str("Scope             Done/plan    R est.   TTFT p50/p95 ms       out tok/s   Adm/deny  429/503 Err\n");
        text.push_str("────────────────────────────────────────────────────────────────────────────────────────────\n");
        let number = |n: Option<f64>| n.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
        let mut row = |name: &str, summary: Summary, show_ratio: bool| {
            use std::fmt::Write;
            let r = if show_ratio {
                summary
                    .ratio()
                    .map(|r| format!("~{r:.4}"))
                    .unwrap_or_else(|| "unknown".into())
            } else {
                "—".into()
            };
            let _ = writeln!(
                text,
                "{name:<17} {:>4}/{:<6} {r:>8} {:>9}/{:<9} {:>11} {:>5}/{:<4} {:>3}/{:<3} {:>3}",
                summary.completed,
                summary.expected,
                number(summary.ttft_p50_ms),
                number(summary.ttft_p95_ms),
                number(summary.output_tok_s),
                summary.requests.admissions,
                summary.requests.refused,
                summary.requests.http_429,
                summary.requests.http_503,
                summary.errors
            );
        };
        row("main total", self.role_summary(Role::Main), true);
        row(
            "subordinate total",
            self.role_summary(Role::Subordinate),
            true,
        );
        row(
            "all (throughput)",
            Summary::new(
                self.sessions.iter().flat_map(|s| s.samples.iter()),
                self.config.sessions * (self.config.turns - 1),
            ),
            false,
        );
        for session in &self.sessions {
            row(
                &format!(
                    "{}-{:02}",
                    if session.role == Role::Main {
                        "main"
                    } else {
                        "sub"
                    },
                    session.session
                ),
                Summary::new(session.samples.iter(), self.config.turns - 1),
                true,
            );
        }
        text.push_str(
            "\nPer-turn cache · C = reported cached tokens; U = estimated reusable prefix\n",
        );
        {
            use std::fmt::Write;
            let _ = writeln!(
                text,
                "{:<14} {:>4} {:>7} {:>7} {:>7} {:>9}  U basis / outcome",
                "Session", "Turn", "P", "C", "U", "C/U %"
            );
        }
        for session in &self.sessions {
            for sample in &session.samples {
                use std::fmt::Write;
                let token = |v: Option<u64>| v.map_or_else(|| "—".into(), |n| n.to_string());
                let p = match sample.input {
                    InputCount::Exact { prompt_tokens, .. } => Some(prompt_tokens),
                    _ => sample.usage.prompt_tokens,
                };
                let ratio = sample
                    .u
                    .zip(sample.usage.cached_tokens)
                    .filter(|(u, _)| *u > 0)
                    .map_or_else(
                        || "—".into(),
                        |(u, c)| format!("{:.2}", c.min(u) as f64 * 100.0 / u as f64),
                    );
                let _ = writeln!(
                    text,
                    "{:<14} {:>4} {:>7} {:>7} {:>7} {:>9}  {} / {}",
                    format!(
                        "{}-{:02}",
                        if session.role == Role::Main {
                            "main"
                        } else {
                            "sub"
                        },
                        session.session
                    ),
                    sample.turn,
                    token(p),
                    token(sample.usage.cached_tokens),
                    token(sample.u),
                    ratio,
                    sample.u_basis,
                    sample.error.unwrap_or(sample.outcome)
                );
            }
        }
        text.push_str("\n~ R and per-turn C/U assume rendered prefix stability; percentages cap C at U.\nExact retained-history admission counts are used when available.\n");
        for role in [Role::Main, Role::Subordinate] {
            let summary = self.role_summary(role);
            text.push_str(&format!(
                "{}: {}; cache pairs {}/{}, admission-based U {}, TTFT samples {}.\n",
                role.label(),
                summary.target(),
                summary.cache_pairs,
                summary.expected,
                summary.admission_based,
                summary.ttft_samples
            ));
        }
        text.push_str("TTFT includes admission; tok/s is end-to-end, aggregate uses an overlapping wall window.\n");
        let stopped = self.sessions.iter().filter(|s| s.stopped.is_some()).count();
        let all_errors = self
            .sessions
            .iter()
            .flat_map(|s| &s.samples)
            .filter(|s| s.error.is_some())
            .count();
        let mut counts = RequestCounts::default();
        let mut prefix_counts = RequestCounts::default();
        let mut prefix_failures = 0;
        for sample in self.sessions.iter().flat_map(|s| &s.samples) {
            add_counts(&mut counts, sample.requests);
            add_counts(&mut prefix_counts, sample.prefix_count_requests);
            prefix_failures += usize::from(sample.prefix_count_error.is_some());
        }
        text.push_str(&format!(
            "All turns: {} admission attempts / {} refused, HTTP 429: {}, HTTP 503: {}.\n",
            counts.admissions, counts.refused, counts.http_429, counts.http_503
        ));
        text.push_str(&format!(
            "Separate history counts: {} admission attempts / {} refused, HTTP 429: {}, HTTP 503: {}; {prefix_failures} failed observations (no inference).\n",
            prefix_counts.admissions, prefix_counts.refused, prefix_counts.http_429, prefix_counts.http_503
        ));
        text.push_str(&format!("All turns: {all_errors} errors, {stopped} stopped sessions; timed out: {}. JSON includes turn 1.\n", self.timed_out));
        text
    }
}
