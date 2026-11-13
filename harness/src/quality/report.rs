//! JSON receipt and RigMark-style card text for a quality-card run.
use super::*;
use std::fmt::Write as _;

pub(super) fn reply_json(reply: &Reply) -> Value {
    obj([
        ("text", s(&reply.text)),
        ("reasoning", s(&reply.reasoning)),
        (
            "token_ids",
            Value::Array(reply.token_ids.iter().map(n).collect()),
        ),
        ("finish_reason", opt_s(&reply.finish_reason)),
        (
            "tool_calls",
            Value::Array(
                reply
                    .tool_calls
                    .iter()
                    .map(|call| {
                        obj([
                            ("index", n(call.index)),
                            ("id", opt_s(&call.id)),
                            ("name", s(&call.name)),
                            ("arguments", s(&call.arguments)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("saw_terminal_event", Value::Bool(reply.saw_done)),
        ("token_ids_valid", Value::Bool(!reply.invalid_token_id)),
        (
            "token_evidence_complete",
            Value::Bool(reply_evidence_failure(reply).is_none()),
        ),
        ("served_model", opt_s(&reply.served_model)),
        (
            "vade",
            reply
                .vade_receipt
                .as_deref()
                .and_then(|text| json::parse(text).ok())
                .unwrap_or(Value::Null),
        ),
        ("completion_id", opt_n(reply.completion_id)),
        ("prompt_tokens", opt_n(reply.prompt_tokens)),
        ("completion_tokens", opt_n(reply.completion_tokens)),
        ("cached_tokens", opt_n(reply.cached_tokens)),
        ("draft", opt_s(&reply.draft)),
        ("draft_used", opt_b(reply.draft_used)),
        ("model_draft_used", opt_b(reply.model_draft_used)),
        ("prefill_ms", opt_f(reply.prefill_ms)),
        ("decode_ms", opt_f(reply.decode_ms)),
        ("ttft_s", opt_f(reply.ttft_s)),
        ("total_s", n(reply.total_s)),
        ("end_to_end_tok_s", opt_f(reply.end_to_end_tok_s())),
        ("decode_tok_s_est", opt_f(reply.decode_tok_s())),
    ])
}

fn checks_json(checks: &[Check]) -> Value {
    Value::Array(
        checks
            .iter()
            .map(|check| {
                obj([
                    ("name", s(&check.name)),
                    ("run", Value::Bool(check.run)),
                    ("passed", Value::Bool(check.run && check.passed)),
                    ("detail", s(&check.detail)),
                ])
            })
            .collect(),
    )
}

fn score_json(passed: usize, total: usize) -> Value {
    obj([
        ("passed_checks", n(passed)),
        ("run_checks", n(total)),
        (
            "score",
            opt_f((total > 0).then(|| passed as f64 * 100.0 / total as f64)),
        ),
    ])
}

impl Report {
    /// The full JSON receipt: fixed prompts, raw answers, token ids, checks.
    pub fn to_json(&self) -> io::Result<String> {
        let config = &self.config;
        let (passed, total) = self.quality_score(None);
        let (same, compared) = self.exact_identical();
        let value = obj([
            ("schema", s("vade.quality-card.v1")),
            ("prompt_set", s(SET_VERSION)),
            ("model", s(&config.model)),
            ("hardware", opt_s(&config.hardware)),
            ("recipe", opt_s(&config.recipe)),
            (
                "request",
                obj([
                    ("temperature", n(0)),
                    ("seed", n(config.seed)),
                    ("thinking", Value::Bool(false)),
                    ("cache", Value::Bool(false)),
                    ("quality_draft", s("omitted (as served)")),
                    ("exactness_drafts", s("strict then plain")),
                ]),
            ),
            ("code_checks_run", Value::Bool(self.code_checks_ran())),
            ("elapsed_s", n(self.elapsed_s)),
            ("run_successful", Value::Bool(self.successful())),
            (
                "quality",
                obj([
                    ("total", score_json(passed, total)),
                    (
                        "classes",
                        Value::Object(
                            CLASSES
                                .iter()
                                .map(|class| {
                                    let (passed, total) = self.quality_score(Some(*class));
                                    (class.label().to_owned(), score_json(passed, total))
                                })
                                .collect(),
                        ),
                    ),
                    (
                        "cases",
                        Value::Array(
                            self.quality
                                .iter()
                                .map(|item| {
                                    obj([
                                        ("id", s(item.case.id)),
                                        ("class", s(item.case.class.label())),
                                        ("prompt", s(full_prompt(item.case))),
                                        ("error", opt_s(&item.error)),
                                        ("reply", reply_json(&item.reply)),
                                        ("checks", checks_json(&item.checks)),
                                        (
                                            "score",
                                            score_json(item.passed(), item.run_checks()),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ]),
            ),
            (
                "exactness",
                obj([
                    ("ids_identical", n(same)),
                    ("ids_compared", n(compared)),
                    (
                        "relaxed_path",
                        s("none on the served FlashNext path (draft fast is refused); strict is compared with plain"),
                    ),
                    (
                        "verify_cost",
                        obj([
                            ("basis", s(&self.verify_basis)),
                            ("receipts", n(self.verify.receipts)),
                            ("cycles", n(self.verify.cycles)),
                            ("accepted", n(self.verify.accepted)),
                            ("proposed", n(self.verify.proposed)),
                            ("target_ms", n(self.verify.target_ms)),
                            ("draft_ms", n(self.verify.draft_ms)),
                            ("verify_ms_per_step", opt_f(self.verify.verify_ms_per_step())),
                            ("accepted_per_step", opt_f(self.verify.accepted_per_step())),
                            (
                                "matched_receipts",
                                Value::Array(
                                    self.verify_receipts
                                        .iter()
                                        .map(|receipt| {
                                            obj([
                                                ("id", n(receipt.id)),
                                                ("cycles", n(receipt.cycles)),
                                                ("rows", n(receipt.rows)),
                                                ("accepted", n(receipt.accepted)),
                                                ("proposed", n(receipt.proposed)),
                                                ("target_ms", n(receipt.target_ms)),
                                                ("draft_ms", n(receipt.draft_ms)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ]),
                    ),
                    (
                        "cases",
                        Value::Array(
                            self.exactness
                                .iter()
                                .map(|item| {
                                    obj([
                                        ("id", s(item.case.id)),
                                        ("class", s(item.case.class.label())),
                                        ("prompt", s(full_prompt(item.case))),
                                        ("error", opt_s(&item.error)),
                                        (
                                            "ids_identical",
                                            opt_b(item.ids_identical()),
                                        ),
                                        (
                                            "draft_modes_ok",
                                            opt_b(item.draft_modes_ok()),
                                        ),
                                        (
                                            "strict_draft",
                                            opt_s(&item.strict.as_ref().and_then(|r| r.draft.clone())),
                                        ),
                                        (
                                            "plain_draft",
                                            opt_s(&item.plain.as_ref().and_then(|r| r.draft.clone())),
                                        ),
                                        ("passed", Value::Bool(item.passed())),
                                        (
                                            "failure_reason",
                                            opt_s(&item.failure_reason()),
                                        ),
                                        (
                                            "first_diff",
                                            item.first_diff().map(n).unwrap_or(Value::Null),
                                        ),
                                        (
                                            "strict",
                                            item.strict
                                                .as_ref()
                                                .map(reply_json)
                                                .unwrap_or(Value::Null),
                                        ),
                                        (
                                            "plain",
                                            item.plain
                                                .as_ref()
                                                .map(reply_json)
                                                .unwrap_or(Value::Null),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ]),
            ),
        ]);
        json::stringify(&value)
    }
}

pub(super) const WIDTH: usize = 72;

pub(super) fn rule(left: char, fill: char, right: char) -> String {
    format!("{left}{}{right}\n", fill.to_string().repeat(WIDTH))
}
pub(super) fn blank(out: &mut String) {
    let _ = writeln!(out, "│{}│", " ".repeat(WIDTH));
}
pub(super) fn section(out: &mut String, title: &str) {
    let head = format!("├─ {title} ");
    let fill = WIDTH + 1 - head.chars().count();
    let _ = writeln!(out, "{head}{}┤", "─".repeat(fill));
    blank(out);
}
/// Word-wrap one logical line into the card body (two-space indent).
pub(super) fn body(out: &mut String, text: &str) {
    let room = WIDTH - 4;
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > room {
            emit(out, &line);
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    emit(out, &line);
}
pub(super) fn emit(out: &mut String, line: &str) {
    let shown: String = line.chars().take(WIDTH - 4).collect();
    let pad = WIDTH - 2 - shown.chars().count();
    let _ = writeln!(out, "│  {shown}{}│", " ".repeat(pad));
}
/// One pre-formatted row; callers keep it inside the card width.
pub(super) fn row(out: &mut String, text: String) {
    emit(out, &text);
}
/// The complete-evidence facts for one passed exactness case.
fn evidence_text(item: &ExactCase) -> String {
    let arm = |reply: Option<&Reply>| match reply {
        Some(reply) => format!(
            "{} ids of {} counted (finish {})",
            reply.token_ids.len(),
            reply.completion_tokens.map_or_else(|| "missing".into(), |n| n.to_string()),
            shown_mode(&reply.finish_reason)
        ),
        None => "no reply".into(),
    };
    format!(
        "token evidence complete: strict {}, plain {}; terminal event on both arms",
        arm(item.strict.as_ref()),
        arm(item.plain.as_ref())
    )
}

pub(super) fn fmt(value: Option<f64>, digits: usize) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.digits$}"))
        .unwrap_or_else(|| "-".into())
}

impl Report {
    fn score_text(&self, class: Option<Class>) -> String {
        let (passed, total) = self.quality_score(class);
        if total == 0 {
            return "-".into();
        }
        format!("{:.1}", passed as f64 * 100.0 / total as f64)
    }
    fn class_median(&self, class: Class, arm: fn(&ExactCase) -> Option<&Reply>) -> Option<f64> {
        median(
            self.exactness
                .iter()
                .filter(|item| item.case.class == class)
                .filter_map(|item| arm(item).and_then(Reply::decode_tok_s))
                .collect(),
        )
    }

    /// Both cards in RigMark card style, quality first.
    pub fn cards(&self) -> String {
        let mut out = String::new();
        if self.config.only != Only::Exactness {
            self.quality_card(&mut out);
        }
        if self.config.only != Only::Quality {
            if !out.is_empty() {
                out.push('\n');
            }
            self.exactness_card(&mut out);
        }
        out
    }

    fn quality_card(&self, out: &mut String) {
        let (passed, total) = self.quality_score(None);
        out.push_str(&rule('╭', '─', '╮'));
        blank(out);
        row(out, "Q U A L I T Y   //   ANSWER QUALITY RECEIPT".into());
        row(out, "FIXED PROMPTS, EXACT CHECKS, NO MODEL JUDGE".into());
        row(
            out,
            format!(
                "●  {passed}/{total} CHECKS PASSED  ·  SCORE {} OF 100",
                self.score_text(None)
            ),
        );
        blank(out);
        section(out, "MODEL");
        body(out, &self.config.model);
        row(out, format!("Prompt set: {} ({} prompts)", SET_VERSION, CASES.len()));
        blank(out);
        section(out, "SCORES");
        row(
            out,
            format!(
                "{:<13} {:>6}   {:>9}   {:>7}   {:>10}",
                "CLASS", "Score", "Checks", "Cases", "Med tok/s"
            ),
        );
        for class in CLASSES {
            let (passed, total) = self.quality_score(Some(class));
            let full = self
                .quality
                .iter()
                .filter(|item| item.case.class == class && item.full_pass())
                .count();
            let cases = self
                .quality
                .iter()
                .filter(|item| item.case.class == class)
                .count();
            let rate = median(
                self.quality
                    .iter()
                    .filter(|item| item.case.class == class)
                    .filter_map(|item| item.reply.decode_tok_s())
                    .collect(),
            );
            row(
                out,
                format!(
                    "{:<13} {:>6}   {:>4}/{:<4}   {:>2}/{:<4}   {:>10}",
                    class.card_label(),
                    self.score_text(Some(class)),
                    passed,
                    total,
                    full,
                    cases,
                    fmt(rate, 1)
                ),
            );
        }
        row(
            out,
            format!(
                "{:<13} {:>6}   {:>4}/{:<4}",
                "TOTAL",
                self.score_text(None),
                passed,
                total
            ),
        );
        blank(out);
        body(
            out,
            "Score is 100 times passed checks over run checks. Med tok/s is a decode estimate from streamed token times.",
        );
        blank(out);
        section(out, "CASES");
        for item in &self.quality {
            if let Some(error) = &item.error {
                row(out, format!("✗ {:<16} request failed", item.case.id));
                body(out, error);
                continue;
            }
            let mark = if item.full_pass() { "✓" } else { "✗" };
            row(
                out,
                format!(
                    "{mark} {:<16} {:>2}/{:<2} checks   {} tok/s",
                    item.case.id,
                    item.passed(),
                    item.run_checks(),
                    fmt(item.reply.decode_tok_s(), 1)
                ),
            );
            if let Some(failed) = item.checks.iter().find(|c| c.run && !c.passed) {
                body(out, &format!("first failed check: {}", failed.name));
            }
            let skipped = item.checks.iter().filter(|c| !c.run).count();
            if skipped > 0 {
                body(out, &format!("{skipped} checks did not run (see receipt)"));
            }
        }
        blank(out);
        section(out, "SETTINGS");
        let drafted = self
            .quality
            .iter()
            .filter(|item| item.reply.draft_used == Some(true))
            .count();
        body(out, "Greedy (temperature 0), thinking off, cache off, one request at a time.");
        row(
            out,
            format!("Seed {} | draft field omitted (as served)", self.config.seed),
        );
        row(
            out,
            format!(
                "Drafting receipts: draft_used true on {drafted}/{} replies",
                self.quality.len()
            ),
        );
        row(
            out,
            format!(
                "Code unit tests: {}",
                if self.code_checks_ran() {
                    "run with rustc --test"
                } else if self.config.code_run {
                    "not run (see receipt)"
                } else {
                    "not run (--no-code-run)"
                }
            ),
        );
        blank(out);
        self.receipt(out);
        out.push_str(&rule('╰', '─', '╯'));
    }

    fn exactness_card(&self, out: &mut String) {
        let (same, compared) = self.exact_identical();
        out.push_str(&rule('╭', '─', '╮'));
        blank(out);
        row(out, "E X A C T N E S S   C O S T   //   STRICT AGAINST PLAIN".into());
        row(out, "SAME PROMPTS, TOKEN IDS COMPARED".into());
        row(
            out,
            format!(
                "{}  IDS IDENTICAL {same}/{compared}",
                if compared > 0 && same == compared { "●" } else { "✗" }
            ),
        );
        blank(out);
        section(out, "MODEL");
        body(out, &self.config.model);
        row(out, format!("Prompt set: {} ({} prompts)", SET_VERSION, CASES.len()));
        blank(out);
        section(out, "DECODE RATE (MEDIAN TOK/S, ESTIMATE)");
        row(
            out,
            format!("{:<13} {:>10}   {:>10}   {:>10}", "CLASS", "Strict", "Plain", "Gain x"),
        );
        for class in CLASSES {
            let strict = self.class_median(class, |item| item.strict.as_ref());
            let plain = self.class_median(class, |item| item.plain.as_ref());
            let gain = strict.zip(plain).and_then(|(a, b)| (b > 0.0).then(|| a / b));
            row(
                out,
                format!(
                    "{:<13} {:>10}   {:>10}   {:>10}",
                    class.card_label(),
                    fmt(strict, 1),
                    fmt(plain, 1),
                    fmt(gain, 2)
                ),
            );
        }
        blank(out);
        body(
            out,
            "Strict is the exact verify path with drafting. Plain is one row per token with no drafting. The gain is what drafting buys; the verify cost below is what it spends.",
        );
        blank(out);
        section(out, "VERIFY COST (SERVER DRAFT RECEIPTS)");
        if self.verify.receipts == 0 {
            body(
                out,
                &format!(
                    "Not measured: {}. The API receipt holds no accepted-token or verify-time fields.",
                    self.verify_basis
                ),
            );
        } else {
            row(out, format!("Basis: {}", self.verify_basis));
            row(
                out,
                format!(
                    "Receipts {}  ·  verify steps {}  ·  proposed {}",
                    self.verify.receipts, self.verify.cycles, self.verify.proposed
                ),
            );
            row(
                out,
                format!(
                    "Verify ms per step {}  ·  accepted per step {}",
                    fmt(self.verify.verify_ms_per_step(), 3),
                    fmt(self.verify.accepted_per_step(), 3)
                ),
            );
            row(
                out,
                format!(
                    "Target ms total {:.1}  ·  draft ms total {:.1}  ·  accepted {}",
                    self.verify.target_ms, self.verify.draft_ms, self.verify.accepted
                ),
            );
        }
        blank(out);
        section(out, "CASES");
        for item in &self.exactness {
            if item.passed() {
                row(
                    out,
                    format!(
                        "✓ {:<16} ids same ({} tokens)   {} vs {} tok/s",
                        item.case.id,
                        item.strict.as_ref().map_or(0, |r| r.token_ids.len()),
                        fmt(item.strict.as_ref().and_then(Reply::decode_tok_s), 1),
                        fmt(item.plain.as_ref().and_then(Reply::decode_tok_s), 1)
                    ),
                );
                body(
                    out,
                    &format!(
                        "draft modes: strict arm {}, plain arm {}",
                        shown_mode(&item.strict.as_ref().and_then(|r| r.draft.clone())),
                        shown_mode(&item.plain.as_ref().and_then(|r| r.draft.clone()))
                    ),
                );
                body(out, &evidence_text(item));
            } else {
                row(out, format!("✗ {:<16} failed", item.case.id));
                body(
                    out,
                    &item.failure_reason().unwrap_or_else(|| "request failed".into()),
                );
            }
        }
        blank(out);
        section(out, "NOTE");
        body(
            out,
            "No relaxed verify path exists on the served FlashNext path: draft fast is refused at request validation. This card therefore prices exactness as strict drafting against plain decoding, with ids compared on every prompt.",
        );
        blank(out);
        self.receipt(out);
        out.push_str(&rule('╰', '─', '╯'));
    }

    fn receipt(&self, out: &mut String) {
        section(out, "RECEIPT");
        row(out, "SOURCE  vade-harness quality-card".into());
        row(out, format!("Schema vade.quality-card.v1  ·  {:.1} s wall", self.elapsed_s));
        if let Some(hardware) = &self.config.hardware {
            body(out, &format!("Hardware: {hardware}"));
        }
        if let Some(recipe) = &self.config.recipe {
            body(out, &format!("Recipe: {recipe}"));
        }
        body(out, "SHARE THE CARD · LINK THE JSON RECEIPT");
        blank(out);
    }
}

/// The fixed prompt set as JSON, for `--print-prompts`.
pub fn prompts_json() -> io::Result<String> {
    let value = obj([
        ("prompt_set", s(SET_VERSION)),
        (
            "cases",
            Value::Array(
                CASES
                    .iter()
                    .map(|case| {
                        obj([
                            ("id", s(case.id)),
                            ("class", s(case.class.label())),
                            ("prompt", s(full_prompt(case))),
                            ("max_tokens", n(case.max_tokens)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    json::stringify(&value)
}
