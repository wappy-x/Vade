//! The engine card text (RigMark card style, then a markdown table) and the
//! `vade.engine-card.v1` JSON receipt.
use super::engine::*;
use super::engine_cases::*;
use super::report::{blank, body, fmt, reply_json, row, rule, section};
use super::*;
use std::fmt::Write as _;

/// FNV-1a 64 over a prompt, so the receipt can name a long generated prompt
/// without holding its text. The same seed and size regenerate it.
pub fn fnv64(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn pass_fail(score: (usize, usize)) -> &'static str {
    if score.1 > 0 && score.0 == score.1 { "PASS" } else { "FAIL" }
}
fn opt_reply(reply: Option<&Reply>) -> Value {
    reply.map(reply_json).unwrap_or(Value::Null)
}
fn score_value(score: (usize, usize)) -> Value {
    obj([("passed", n(score.0)), ("total", n(score.1)), ("result", s(pass_fail(score)))])
}
fn flag_text(flag: Option<bool>) -> String {
    flag.map_or_else(|| "unknown".into(), |f| f.to_string())
}
fn count_text(count: Option<u64>) -> String {
    count.map_or_else(|| "unknown".into(), |c| c.to_string())
}

/// The one line per check: name, what it proves, score.
pub const CHECK_NAMES: [(&str, &str); 5] = [
    ("Long-context recall", "a planted code is read back verbatim from 8K and 32K token prompts"),
    ("JSON mode validity", "response_format json_schema answers parse and pass their schema"),
    ("Tool-call correctness", "the model calls the named tool with exactly the expected arguments"),
    ("Cache invariance", "a prompt read cold and read from the prefix cache gives identical token ids"),
    ("Concurrency invariance", "8 simultaneous copies give the token ids of one request alone"),
];

impl EngineReport {
    fn scores(&self) -> [(usize, usize); 5] {
        [
            self.needle_score(),
            self.json_score(),
            self.tool_score(),
            self.cache_score(),
            self.concurrency_score(),
        ]
    }

    fn per_prompt_concurrency(&self) -> String {
        self.concurrency
            .iter()
            .map(|item| format!("{}/{}", item.identical(), CONCURRENT_COPIES))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn sha_text(&self) -> String {
        match (&self.server.git_sha, self.server.dirty) {
            (Some(sha), Some(true)) => format!("{sha} (dirty)"),
            (Some(sha), _) => sha.clone(),
            (None, _) => "unknown (the server reported no build sha)".into(),
        }
    }

    fn mtp_text(&self) -> String {
        match (self.server.mtp_depth, &self.server.draft) {
            (Some(0), draft) => format!("MTP off (draft {})", draft.as_deref().unwrap_or("unknown")),
            (Some(depth), draft) => format!("MTP on, depth {depth} (draft {})", draft.as_deref().unwrap_or("unknown")),
            (None, _) => "MTP unknown".into(),
        }
    }

    /// The boxed card, then the markdown table.
    pub fn card(&self) -> String {
        let mut out = String::new();
        let scores = self.scores();
        let grade = self.grade();
        let passing = scores.iter().filter(|score| pass_fail(**score) == "PASS").count();
        out.push_str(&rule('╭', '─', '╮'));
        blank(&mut out);
        row(&mut out, "E N G I N E   //   ENGINE BEHAVIOR CARD".into());
        row(&mut out, "FIVE CHECKS, EXACT TOKEN IDS, NO MODEL JUDGE".into());
        row(&mut out, format!("●  GRADE {}  ·  {passing}/5 CHECKS PASS", grade.label()));
        blank(&mut out);
        section(&mut out, "SERVER");
        body(&mut out, &format!("Model: {} (served as {})", self.config.model, self.server.served_model.as_deref().unwrap_or("unknown")));
        body(&mut out, &format!("Build: {}", self.sha_text()));
        row(
            &mut out,
            format!(
                "Sparks: {} (world_size)  ·  topology {}",
                count_text(self.server.world_size),
                self.server.topology.as_deref().unwrap_or("unknown")
            ),
        );
        row(&mut out, format!("NVFP4: {}  ·  {}", flag_text(self.server.nvfp4), self.mtp_text()));
        // The checkpoint directory is a host path: the receipt keeps it, the
        // shared card does not. The hash is server text, cut by characters.
        if let Some(sha) = &self.server.checkpoint_sha256 {
            body(&mut out, &format!("Checkpoint sha256 {}", sha.chars().take(16).collect::<String>()));
        }
        row(&mut out, format!("Set: {ENGINE_SET}  ·  seed {}", self.config.seed));
        blank(&mut out);
        section(&mut out, "CHECKS");
        row(&mut out, format!("{:<28} {:>6}   {:>8}", "CHECK", "RESULT", "NUMBER"));
        for (index, (score, (name, _))) in scores.iter().zip(CHECK_NAMES).enumerate() {
            row(
                &mut out,
                format!("{} {:<26} {:>6}   {:>8}", index + 1, name, pass_fail(*score), format!("{}/{}", score.0, score.1)),
            );
        }
        row(&mut out, format!("  concurrency per prompt: {}", self.per_prompt_concurrency()));
        blank(&mut out);
        section(&mut out, "DETAILS");
        self.details(&mut out);
        blank(&mut out);
        section(&mut out, "GRADE RULES");
        body(
            &mut out,
            "A+ = all five pass. A soft miss is exactly one item short in JSON mode or in tool calls. A = one soft miss, B = two. A missed or not-run needle, any cache or concurrency mismatch, or two items short in one check = C.",
        );
        blank(&mut out);
        section(&mut out, "RECEIPT");
        row(&mut out, "SOURCE  vade-harness quality-card --set engine-1".into());
        row(&mut out, format!("Schema vade.engine-card.v1  ·  {:.1} s wall", self.elapsed_s));
        if let Some(hardware) = &self.config.hardware {
            body(&mut out, &format!("Hardware: {hardware}"));
        }
        if let Some(recipe) = &self.config.recipe {
            body(&mut out, &format!("Recipe: {recipe}"));
        }
        body(&mut out, "SHARE THE CARD · LINK THE JSON RECEIPT");
        blank(&mut out);
        out.push_str(&rule('╰', '─', '╯'));
        out.push('\n');
        out.push_str(&self.markdown_table());
        out
    }

    fn details(&self, out: &mut String) {
        match &self.calibration {
            Ok(cal) => row(
                out,
                format!(
                    "Filler: {:.2} tokens per sentence (probes {} and {})",
                    cal.tokens_per_sentence, cal.probe_tokens[0], cal.probe_tokens[1]
                ),
            ),
            Err(error) => body(out, &format!("Filler calibration failed: {error}")),
        }
        for item in &self.needles {
            let tokens = item.reply.as_ref().and_then(|r| r.prompt_tokens);
            let mark = if item.passed() { "✓" } else { "✗" };
            if item.sizing.not_run.is_some() {
                row(out, format!("{mark} needle {:<4} NOT RUN (refused before sending)", item.needle.label));
            } else {
                row(
                    out,
                    format!(
                        "{mark} needle {:<4} prompt {} tokens  ·  first token {} s",
                        item.needle.label,
                        count_text(tokens),
                        fmt(item.reply.as_ref().and_then(|r| r.ttft_s), 2)
                    ),
                );
            }
            if let Some(reason) = item.failure_reason() {
                body(out, &reason);
            }
            if let Some(note) = &item.sizing.note {
                body(out, note);
            }
        }
        let failures = |out: &mut String, label: &str, reasons: Vec<(String, String)>| {
            for (id, reason) in reasons.iter().take(5) {
                body(out, &format!("✗ {label} {id}: {reason}"));
            }
            if reasons.len() > 5 {
                body(out, &format!("{} more {label} misses in the receipt", reasons.len() - 5));
            }
        };
        let split = |only: bool| {
            let items: Vec<&JsonItem> = self.json.iter().filter(|i| i.schema_only() == only).collect();
            format!("{}/{}", items.iter().filter(|i| i.passed()).count(), items.len())
        };
        row(out, format!("JSON: keys in the prompt {}  ·  keys only in the schema {}", split(false), split(true)));
        failures(out, "json", self.json.iter().filter_map(|i| i.failure_reason().map(|r| (i.id(), r))).collect());
        failures(out, "tool", self.tools.iter().filter_map(|i| i.failure_reason().map(|r| (i.case.id.to_owned(), r))).collect());
        let reused: Vec<u64> = self.cache.iter().filter_map(|i| i.warm.as_ref()?.cached_tokens).collect();
        row(
            out,
            format!(
                "Cache: cold runs send cache false; warm reused {}..{} tokens",
                reused.iter().min().copied().unwrap_or(0),
                reused.iter().max().copied().unwrap_or(0)
            ),
        );
        failures(out, "cache", self.cache.iter().filter_map(|i| i.failure_reason().map(|r| (format!("prompt-{}", i.index + 1), r))).collect());
        for item in &self.concurrency {
            let window = if item.overlapped() { "all copies in flight together" } else { "copies did not overlap" };
            row(out, format!("Concurrency {:<10} {}/{}  ·  {window}", item.id, item.identical(), CONCURRENT_COPIES));
            let mut modes: Vec<String> = item.copies.iter().filter_map(|c| c.reply.as_ref().ok()?.draft.clone()).collect();
            modes.sort();
            modes.dedup();
            let alone = item.alone.as_ref().ok().and_then(|r| r.draft.clone()).unwrap_or_else(|| "unknown".into());
            row(out, format!("  draft mode alone {alone}, copies {}", if modes.is_empty() { "unknown".into() } else { modes.join("+") }));
            let reasons: Vec<(String, String)> = (0..CONCURRENT_COPIES)
                .filter_map(|i| item.copy_failure(i).map(|r| (format!("{} copy {}", item.id, i + 1), r)))
                .collect();
            failures(out, "concurrency", reasons);
        }
    }

    /// The markdown table for status notes: one header row of run facts and
    /// one row per check.
    pub fn markdown_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "| Build sha | Sparks | Model | MTP | NVFP4 | Wall | Grade |");
        let _ = writeln!(out, "| --- | --- | --- | --- | --- | --- | --- |");
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {:.1} s | **{}** |",
            self.sha_text(),
            count_text(self.server.world_size),
            self.config.model,
            self.mtp_text(),
            flag_text(self.server.nvfp4),
            self.elapsed_s,
            self.grade().label()
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "| # | Check | Result | Number | What it proves |");
        let _ = writeln!(out, "| --- | --- | --- | --- | --- |");
        for (index, (score, (name, proves))) in self.scores().iter().zip(CHECK_NAMES).enumerate() {
            let extra = if index == 4 { format!(" ({})", self.per_prompt_concurrency()) } else { String::new() };
            let _ = writeln!(
                out,
                "| {} | {name} | {} | {}/{}{extra} | {proves} |",
                index + 1,
                pass_fail(*score),
                score.0,
                score.1
            );
        }
        out
    }

    /// The full JSON receipt: server facts, every reply with its token ids,
    /// every verdict, the grade.
    pub fn to_json(&self) -> io::Result<String> {
        let scores = self.scores();
        let server = &self.server;
        let needles = Value::Array(
            self.needles
                .iter()
                .map(|item| {
                    let prompt = (item.sentences > 0).then(|| needle_prompt(item.needle, item.sentences, true));
                    obj([
                        ("label", s(item.needle.label)),
                        ("target_tokens", n(item.needle.target_tokens)),
                        ("seed", n(item.needle.seed)),
                        ("filler_sentences", n(item.sentences)),
                        ("prompt_chars", n(item.prompt_chars)),
                        ("prompt_fnv64", prompt.as_deref().map(fnv64).map(s).unwrap_or(Value::Null)),
                        ("fact", s(item.needle.fact)),
                        ("answer", s(item.needle.answer)),
                        ("passed", Value::Bool(item.passed())),
                        ("failure_reason", opt_s(&item.failure_reason())),
                        ("not_run", opt_s(&item.sizing.not_run)),
                        ("sizing_note", opt_s(&item.sizing.note)),
                        (
                            "admission",
                            item.sizing.admission.as_ref().map_or(Value::Null, |a| {
                                obj([
                                    ("prompt_tokens", n(a.prompt_tokens)),
                                    ("context_limit", n(a.context_limit)),
                                    ("input_limit", n(a.input_limit)),
                                    ("admitted", Value::Bool(a.admitted)),
                                    ("reason", opt_s(&a.reason)),
                                ])
                            }),
                        ),
                        ("reply", opt_reply(item.reply.as_ref())),
                    ])
                })
                .collect(),
        );
        let json_items = Value::Array(
            self.json
                .iter()
                .map(|item| {
                    obj([
                        ("id", s(item.id())),
                        ("keys_in_prompt", Value::Bool(!item.schema_only())),
                        ("schema", json::parse(item.case.schema).unwrap_or(Value::Null)),
                        ("prompt", s(json_prompt(item.case, item.source))),
                        ("passed", Value::Bool(item.passed())),
                        ("failure_reason", opt_s(&item.failure_reason())),
                        ("reply", opt_reply(item.reply.as_ref())),
                    ])
                })
                .collect(),
        );
        let tool_items = Value::Array(
            self.tools
                .iter()
                .map(|item| {
                    obj([
                        ("id", s(item.case.id)),
                        ("prompt", s(item.case.prompt)),
                        ("want_name", s(item.case.name)),
                        ("want_arguments", json::parse(item.case.arguments).unwrap_or(Value::Null)),
                        ("passed", Value::Bool(item.passed())),
                        ("failure_reason", opt_s(&item.failure_reason())),
                        ("reply", opt_reply(item.reply.as_ref())),
                    ])
                })
                .collect(),
        );
        let cache_items = Value::Array(
            self.cache
                .iter()
                .map(|item| {
                    obj([
                        ("index", n(item.index)),
                        ("seed", n(4_000 + item.index as u64)),
                        ("filler_sentences", n(CACHE_SENTENCES)),
                        ("prompt_fnv64", s(fnv64(&cache_prompt(item.index)))),
                        ("passed", Value::Bool(item.passed())),
                        ("failure_reason", opt_s(&item.failure_reason())),
                        (
                            "prime_matches_cold",
                            match (&item.cold, &item.prime) {
                                (Some(cold), Some(prime)) => Value::Bool(pair_failure(cold, prime).is_none()),
                                _ => Value::Null,
                            },
                        ),
                        ("cold", opt_reply(item.cold.as_ref())),
                        ("prime", opt_reply(item.prime.as_ref())),
                        ("warm", opt_reply(item.warm.as_ref())),
                    ])
                })
                .collect(),
        );
        let concurrency_items = Value::Array(
            self.concurrency
                .iter()
                .map(|item| {
                    let prompt = CONCURRENCY_PROMPTS.iter().find(|(id, _)| *id == item.id).map_or("", |(_, p)| *p);
                    obj([
                        ("id", s(item.id)),
                        ("prompt", s(prompt)),
                        ("identical", n(item.identical())),
                        ("copies", n(CONCURRENT_COPIES)),
                        ("overlapped", Value::Bool(item.overlapped())),
                        (
                            "alone",
                            match &item.alone {
                                Ok(reply) => reply_json(reply),
                                Err(error) => obj([("error", s(error))]),
                            },
                        ),
                        (
                            "runs",
                            Value::Array(
                                item.copies
                                    .iter()
                                    .enumerate()
                                    .map(|(index, copy)| {
                                        obj([
                                            ("start_s", n(copy.start_s)),
                                            ("end_s", n(copy.end_s)),
                                            ("first_token_s", opt_f(copy.first_token_s())),
                                            ("failure_reason", opt_s(&item.copy_failure(index))),
                                            (
                                                "reply",
                                                match &copy.reply {
                                                    Ok(reply) => reply_json(reply),
                                                    Err(error) => obj([("error", s(error))]),
                                                },
                                            ),
                                        ])
                                    })
                                    .collect(),
                            ),
                        ),
                    ])
                })
                .collect(),
        );
        let check = |index: usize, items: Value| {
            obj([("name", s(CHECK_NAMES[index].0)), ("score", score_value(scores[index])), ("items", items)])
        };
        let value = obj([
            ("schema", s("vade.engine-card.v1")),
            ("set", s(ENGINE_SET)),
            ("model", s(&self.config.model)),
            ("hardware", opt_s(&self.config.hardware)),
            ("recipe", opt_s(&self.config.recipe)),
            (
                "server",
                obj([
                    ("git_sha", opt_s(&server.git_sha)),
                    ("dirty", opt_b(server.dirty)),
                    ("served_model", opt_s(&server.served_model)),
                    ("world_size", opt_n(server.world_size)),
                    ("topology", opt_s(&server.topology)),
                    ("nvfp4", opt_b(server.nvfp4)),
                    ("mtp_depth", opt_n(server.mtp_depth)),
                    ("draft", opt_s(&server.draft)),
                    ("checkpoint_directory", opt_s(&server.checkpoint_dir)),
                    ("checkpoint_sha256", opt_s(&server.checkpoint_sha256)),
                ]),
            ),
            (
                "request",
                obj([
                    ("temperature", n(0)),
                    ("seed", n(self.config.seed)),
                    ("thinking", Value::Bool(false)),
                    ("draft", s("omitted (as served)")),
                    ("cache", s("false on every request except the cache check's prime and warm runs, which send true (the served prefix cache is opt-in)")),
                    ("tool_choice", s("auto")),
                ]),
            ),
            ("grade", s(self.grade().label())),
            ("elapsed_s", n(self.elapsed_s)),
            (
                "calibration",
                match &self.calibration {
                    Ok(cal) => obj([
                        ("probe_sentences", Value::Array(PROBE_SENTENCES.iter().map(n).collect())),
                        ("probe_prompt_tokens", Value::Array(cal.probe_tokens.iter().map(n).collect())),
                        ("tokens_per_sentence", n(cal.tokens_per_sentence)),
                        ("frame_tokens", n(cal.frame_tokens)),
                    ]),
                    Err(error) => obj([("error", s(error))]),
                },
            ),
            (
                "checks",
                Value::Array(vec![
                    check(0, needles),
                    check(1, json_items),
                    check(2, tool_items),
                    check(3, cache_items),
                    check(4, concurrency_items),
                ]),
            ),
        ]);
        json::stringify(&value)
    }
}

/// `quality-card --set engine-1` after both output files exist: run the five
/// checks, save the receipt, print the card. Exit 0 only at grade A+.
pub(super) fn cli(config: Config, mut json_file: std::fs::File, mut card_file: Option<std::fs::File>) -> ExitCode {
    crate::diagnostics::stderr_line(format_args!(
        "quality-card: {ENGINE_SET}: needles, {} JSON, {} tool, {CACHE_PROMPTS} cache and {} concurrency prompts",
        JSON_CASES.len() * 5,
        TOOL_CASES.len(),
        CONCURRENCY_PROMPTS.len()
    ));
    let report = match run_engine(config) {
        Ok(report) => report,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: engine card setup failed ({error}); JSON is incomplete. Check the endpoint and model, then retry with a new --json path."));
            return ExitCode::FAILURE;
        }
    };
    let saved = report.to_json().and_then(|body| {
        json_file.write_all(body.as_bytes())?;
        json_file.write_all(b"\n")?;
        json_file.sync_all()
    });
    if let Err(error) = saved {
        crate::diagnostics::stderr_line(format_args!("vade: engine card JSON write failed. {}", crate::diagnostics::output_file_hint(&error)));
        return ExitCode::FAILURE;
    }
    let card = report.card();
    if let Some(file) = &mut card_file {
        if file.write_all(card.as_bytes()).and_then(|()| file.sync_all()).is_err() {
            crate::diagnostics::stderr_line(format_args!("vade: engine card file write failed."));
            return ExitCode::FAILURE;
        }
    }
    if crate::diagnostics::print_stdout(&card) == ExitCode::FAILURE {
        return ExitCode::FAILURE;
    }
    crate::diagnostics::stderr_line(format_args!("quality-card: receipt saved to the requested JSON file"));
    if report.grade() == Grade::APlus { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
