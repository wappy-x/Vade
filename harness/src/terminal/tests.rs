//! Complete terminal screens, approval review and transcript/frame integration.
    use super::frame::{live_tail, window};
    #[test]
    fn inline_commits_once_and_resize_repaints_only_live_rows() {
        let mut t = Terminal::open(true);
        t.fullscreen = false;
        t.retain("paragraph longer than the dock\n", false);
        let lines = vec!["live".into()];
        let first = t.inline_bytes(&lines, 0, 1);
        assert!(first.contains("paragraph longer than the dock\r\n"));
        t.resized().unwrap();
        let second = t.inline_bytes(&lines, 0, 1);
        assert!(!second.contains("paragraph") && second.matches("live").count() == 1);
        assert!(second.contains("\x1b[J") && !second.contains("\x1b[2J"));
        assert_eq!(live_tail("a\nb\nc", 80, 2), "b\nc");
        assert!(live_tail(&"x".repeat(1024 * 1024), 80, 2).len() <= 4096);
    }
    use super::*;
    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "paced CPU/draw benchmark; run release serially with --nocapture"]
    fn snappy_draw() {
        unsafe extern "C" { fn clock_gettime(id: i32, time: *mut [i64; 2]) -> i32; }
        let cpu = || { let mut t = [0; 2]; assert_eq!(unsafe { clock_gettime(16, &mut t) }, 0); t[0] as f64 + t[1] as f64 / 1e9 };
        for run in 0..5 { for rate in [60, 200] { for fullscreen in [false, true] {
            let mut t = Terminal::open(true); t.raw = true; t.fullscreen = fullscreen;
            t.transcript.push_styled(&"old **answer** with code\n".repeat(3000), 109);
            t.answer_preview = "streaming answer text\n".repeat(3000); t.answer_time = "12:00".into();
            let period = Duration::from_secs_f64(1. / rate as f64);
            let (mut cost, mut bytes, mut dropped) = (0., 0, 0);
            for _ in 0..120 {
                let start = Instant::now(); let before = cpu(); t.preview_answer("word ");
                let (lines, row, col) = t.frame_lines("draft", 5, "› ", "status", "", "", "", 110, 24);
                bytes += if fullscreen { t.fullscreen_bytes(lines, row, col, 110, 24).len() } else { t.inline_bytes(&lines, row, col).len() }; cost += cpu() - before;
                dropped += usize::from(start.elapsed() > Duration::from_secs_f64(1. / 60.));
                std::thread::sleep(period.saturating_sub(start.elapsed()));
            }
            let start = Instant::now();
            std::hint::black_box(t.frame_lines("draft", 5, "› ", "status", "", "", "", 100, 30));
            println!("snappy run={run} rate={rate} fullscreen={fullscreen} draw_cpu_us_token={:.3} dropped_frames={dropped} bytes_token={:.3} resize_draw_ms={:.3}", cost * 1e6 / 120., bytes as f64 / 120., start.elapsed().as_secs_f64() * 1000.);
        }}}
    }
    #[test]
    fn tui_golden_screens() {
        let mut context = crate::contract::Context::default();
        context.turns.push_back(crate::contract::TurnUsage {
            usage: crate::contract::Usage {
                prompt_tokens: Some(4200),
                completion_tokens: Some(200),
                cached_tokens: Some(4032),
            },
            first_token_elapsed: Some(Duration::from_millis(1500)),
            decode_elapsed: Some(Duration::from_millis(8652)),
            ..Default::default()
        });
        let status = context.compact_status(32000);
        assert_eq!(
            status,
            "first token 1.5s · 23 tok/s · ctx 4.4k/32k · cached 4k/4.2k"
        );
        let sample = "# A cleaner terminal\n\n**Readable answers** with `src/main.rs`.\n- Short status\n* Clear commands\n1. Read the file\n2. Run the tests\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```";
        for width in [80, 160] {
            let mut snapshots = String::new();
            for scene in [
                "answer + status",
                "slash menu",
                "slash menu scrolled",
                "markdown",
                "tool approval",
                "wrapped input",
            ] {
                let mut terminal = Terminal::open(true);
                terminal.color = true;
                let text = match scene {
                    "answer + status" => format!(
                        "{}\n\n{}\n",
                        render::markdown(
                            "The change is **ready**. Run `cargo test` to check it.",
                            true
                        ),
                        render::style(&status, "2", true)
                    ),
                    "markdown" => format!(
                        "{}\n\n{}\n",
                        render::markdown(sample, true),
                        render::style(&status, "2", true)
                    ),
                    "tool approval" => format!(
                        "Read src/main.rs\nRun: cargo test\n{}\n",
                        render::approval(true)
                    ),
                    _ => "Vade\n\nHow can I help?\n".into(),
                };
                terminal.transcript.push_styled(&text, width - 1);
                let popup = if scene.starts_with("slash menu") {
                    render::menu(
                        &[
                            ("help", "Show commands and keyboard shortcuts."),
                            ("status", "Show session details."),
                            ("model", "Show the active model."),
                            ("new", "Start a fresh session."),
                            ("context", "Show context use."),
                            ("compact", "Compact older messages."),
                            ("resume", "Resume a saved session."),
                            ("retry", "Regenerate the last answer."),
                            ("quit", "Exit the chat."),
                        ],
                        if scene.ends_with("scrolled") { 8 } else { 1 },
                        width - 1,
                        10,
                        true,
                    )
                } else {
                    String::new()
                };
                let draft = if scene == "wrapped input" {
                    "Explain the changes to the terminal and keep this draft intact during resize. Include a short summary, the tests you ran, and any remaining issues. 🦀\nThen show the next step."
                } else if scene == "slash menu" {
                    "/"
                } else {
                    ""
                };
                let footer = if scene == "answer + status" {
                    "qwen3-8b · harness · ask · 4.4k/32k ctx · session-1791581595875-6512-0"
                } else {
                    ""
                };
                let (lines, row, col) = terminal.frame_lines(
                    draft,
                    draft.len(),
                    "› ",
                    footer,
                    "",
                    "",
                    &popup,
                    width,
                    24,
                );
                assert_eq!(lines.len(), 24);
                assert!(lines.iter().all(|line| render::width(line) < width));
                assert!(row < 24 && col <= width);
                snapshots.push_str(&format!(
                    "=== {scene} ({width} columns) ===\n{}\n",
                    lines.join("\n")
                ));
            }
            for (suffix, text) in [
                ("ansi", snapshots.clone()),
                ("txt", render::plain(&snapshots)),
            ] {
                let path = format!(
                    "{}/tests/goldens/tui-{width}.{suffix}",
                    env!("CARGO_MANIFEST_DIR")
                );
                if std::env::var_os("VADE_UPDATE_GOLDENS").is_some() {
                    std::fs::write(&path, &text).unwrap();
                }
                assert_eq!(std::fs::read_to_string(path).unwrap(), text);
            }
        }
        assert_eq!(render::markdown("`a*b` and 3*4", false), "a*b and 3*4");
        assert_eq!(render::markdown("~~~\n\n  x\n~~~", false), "~~~\n\n  x\n~~~");
        assert_eq!(
            render::markdown(
                "# Title\n## Sub\n---\n> quoted *text*\nsee [docs](https://example.invalid/a) now\n[open](no space) [x]",
                false
            ),
            "Title\nSub\n────────────────────────\n│ quoted text\nsee docs (https://example.invalid/a) now\n[open](no space) [x]"
        );
        assert_eq!(render::markdown("# Title", true), "\x1b[1;4mTitle\x1b[0m");
        assert_eq!(
            render::markdown("-- not a rule\n***", false),
            "-- not a rule\n────────────────────────"
        );
        let mut terminal = Terminal::open(true);
        for width in 1..=160 {
            for height in [1, 2, 4, 8, 49] {
                let (lines, row, col) = terminal.frame_lines(
                    "你好🦀\nlong draft",
                    10,
                    "› ",
                    "model · folder",
                    "⠦ thinking… · 3.0s",
                    "",
                    "",
                    width,
                    height,
                );
                assert_eq!(lines.len(), height);
                assert!(
                    lines
                        .iter()
                        .all(|line| render::width(line) <= width.saturating_sub(1).max(1)),
                    "{width}x{height}: {lines:?}"
                );
                assert!(row < height && col <= width);
            }
        }
    }
    #[test]
    fn block_goldens_label_you_thinking_answer_warning_and_footer() {
        // Clock times are masked; every other byte is compared.
        let mask = |text: &str| {
            let mut out = text.as_bytes().to_vec();
            for i in 0..out.len().saturating_sub(7) {
                let digits = [0, 1, 3, 4, 6, 7].iter().all(|j| out[i + j].is_ascii_digit());
                if digits && out[i + 2] == b':' && out[i + 5] == b':' {
                    out[i..i + 8].copy_from_slice(b"hh:mm:ss");
                }
            }
            String::from_utf8(out).unwrap()
        };
        for width in [120, 60] {
            let mut screens = Vec::new();
            for color in [true, false] {
                let mut t = Terminal::open(true);
                (t.raw, t.fullscreen, t.color, t.date) = (true, true, color, clock::now().1);
                t.user_message("Fix the parser").unwrap();
                t.note("[file: src/parser.rs, 2 lines]").unwrap();
                t.thinking_started = Some(Instant::now() - Duration::from_secs(14));
                t.preview_thinking("The parser drops the last token.", false);
                t.thinking("The parser drops the last token.", Some(1234), false).unwrap();
                if width == 120 { t.toggle_thinking(); }
                t.answer("I'll read the parser first.").unwrap();
                t.answer_status("╰ 14:02:40").unwrap();
                t.tool_call("Read src/parser.rs").unwrap();
                t.tool_result("120 lines · 0.1s", false).unwrap();
                t.answer("The test failed because the **last token** was dropped.").unwrap();
                let footer = "╰ 14:02:41 · 29.6s · 412 tokens out · 23 tok/s · prefill ≈884 tok/s · first token 3.2s · 3 model calls · ctx 44.1k/131k";
                t.note(&fit_status(footer, width - 1)).unwrap();
                t.warning(
                    "The next reply would not fit in the model's context.",
                    "Vade stopped and kept the history.",
                    "Use /compact to shrink older turns, or /new to start over.",
                )
                .unwrap();
                let status = "qwen3-8b · harness · approval: ask · ctx 44.1k/131k · session-1";
                let (lines, _, _) = t.frame_lines("", 0, "› ", status, "", "", "", width, 28);
                assert!(lines.iter().all(|line| render::width(line) < width));
                screens.push(mask(&lines.join("\n")));
            }
            // Labels alone separate the blocks when color is off.
            assert_eq!(render::plain(&screens[0]), screens[1]);
            // The plain golden drops trailing spaces (the empty `› ` prompt).
            let plain = screens[1].lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
            for (suffix, text) in [("ansi", &screens[0]), ("txt", &plain)] {
                let path = format!(
                    "{}/tests/goldens/blocks-{width}.{suffix}",
                    env!("CARGO_MANIFEST_DIR")
                );
                if std::env::var_os("VADE_UPDATE_GOLDENS").is_some() {
                    std::fs::write(&path, text).unwrap();
                }
                assert_eq!(&std::fs::read_to_string(path).unwrap(), text);
            }
        }
    }
    #[test]
    fn pager_search_reaches_third_match_and_jumps_prompt_headers() {
        let mut t = Terminal::open(true);
        t.raw = true;
        t.review_fullscreen = true;
        t.pager = Some(Pager::default());
        for i in 1..=3 {
            t.transcript.push(&format!("{}\nneedle {i}\n{}", render::user(&format!("turn {i}"), false), "gap\n".repeat(30)), 79);
        }
        let (w, h) = (t.width(), t.height());
        t.scroll = usize::MAX;
        t.pager.as_mut().unwrap().query = "needle".into();
        t.frame_lines("", 0, "", "", "", "", "", w, h);
        t.pager_key(&Key::Text("nnn".into())).unwrap();
        let (rows, _, _) = t.frame_lines("", 0, "", "", "", "", "", w, h);
        assert!(rows.iter().any(|r| r.contains("\x1b[7mneedle\x1b[27m 3")));
        assert_eq!(t.pager.as_ref().unwrap().hit, Some(2));
        for next in [false, false, true] {
            t.pager_jump(next, true);
            let (rows, _, _) = t.frame_lines("", 0, "", "", "", "", "", w, h);
            assert!(render::plain(&rows[1]).starts_with("▌ turn "));
        }
        assert_eq!(Decoder::default().feed(b"\x0f\x1b[111;5u"), [Key::CtrlO, Key::CtrlO]);
    }
    #[test]
    fn visible_window_spans_retained_rows_and_the_provisional_tail() {
        let rows: VecDeque<String> = (0..10).map(|i| format!("r{i}")).collect();
        let extra: Vec<String> = (0..3).map(|i| format!("p{i}")).collect();
        let text = |v: Vec<String>| v.join(" ");
        assert_eq!(
            text(window(&rows, &extra, 0, 13)),
            "r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 p0 p1 p2"
        );
        assert_eq!(text(window(&rows, &extra, 8, 13)), "r8 r9 p0 p1 p2");
        assert_eq!(text(window(&rows, &extra, 11, 13)), "p1 p2");
        assert_eq!(text(window(&rows, &extra, 2, 5)), "r2 r3 r4");
        assert_eq!(text(window(&rows, &[], 9, 10)), "r9");
        assert!(window(&rows, &extra, 13, 13).is_empty());
    }
    #[test]
    fn footer_keeps_its_first_item_and_trailing_hint_when_narrow() {
        let status = "qwen3-8b · harness · ask · 4.4k/32k ctx · session-1 · Esc cancels";
        assert_eq!(fit_status(status, 200), status);
        assert_eq!(
            fit_status(status, 40),
            "qwen3-8b · harness · ask · Esc cancels"
        );
        assert_eq!(fit_status(status, 24), "qwen3-8b · Esc cancels");
        assert_eq!(fit_status(status, 10), "qwen3-8b ·");
        assert_eq!(fit_status("one · two", 5), "one ·");
        assert_eq!(fit_status("", 5), "");
        let budgets = "round 7/60 · 4m/20m · ctx 18% · Esc cancels";
        assert_eq!(fit_status(budgets, 33), "round 7/60 · 4m/20m · Esc cancels");
        assert_eq!(fit_status(budgets, 24), "round 7/60 · Esc cancels");
        assert_eq!(render::budget_footer(budgets, false), budgets);
        assert!(render::budget_footer("round 48/60 · 16m/20m · ctx 80%", true).contains("\x1b[7mctx 80%"));
        assert!(render::budget_footer("round 4/60 · 1m/20m · ctx 26.3k/32.8k", true).contains("\x1b[7mctx 26.3k/32.8k"));
        assert!(!render::budget_footer("round 4/60 · 1m/20m · ctx 1.1k/32.8k", true).contains("\x1b[7mctx"));
        // The scroll hint is reserved whole, ahead of the facts.
        let mut terminal = Terminal::open(true);
        terminal.transcript.push(&"row\n".repeat(60), 39);
        terminal.scroll = 10;
        let (lines, _, _) = terminal.frame_lines(
            "",
            0,
            "› ",
            "qwen3-8b · harness · ask · session-1",
            "",
            "",
            "",
            40,
            12,
        );
        let footer = lines.iter().find(|l| l.contains("PgDn")).unwrap();
        assert_eq!(
            render::plain(footer),
            "qwen3-8b · session-1 · history · PgDn"
        );
    }
    #[test]
    fn approval_renders_all_5000_changed_lines_despite_history_trim_and_resize() {
        let f = crate::test_workspace::Fixture::new();
        f.write("file", &"a\n".repeat(2500));
        let patch = format!(
            "*** Begin Patch\n*** Update File: file\n@@\n{}{}*** End Patch",
            "-a\n".repeat(2500),
            "+b\n".repeat(2500)
        );
        let changed = |line: &str| {
            (line.starts_with('+') && !line.starts_with("+++ "))
                || (line.starts_with('-') && !line.starts_with("--- "))
        };
        let expected = patch.lines().filter(|line| changed(line)).count();
        let expected_lines: Vec<_> = patch.lines().filter(|line| changed(line)).collect();
        assert_eq!(expected, 5000);
        let prepared = crate::edit::prepare(&f.workspace(), &patch).unwrap();
        let preview = prepared.preview();
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        terminal.color = true;
        terminal
            .approval_preview("Patch preview", preview.diff())
            .unwrap();
        assert!(
            terminal
                .transcript
                .rows
                .iter()
                .filter(|line| changed(&render::plain(line)))
                .count()
                < expected
        );
        // Later notices can evict ordinary history, but never this approval.
        terminal.note(&"later\n".repeat(6000)).unwrap();
        for width in [80, 8, 160] {
            terminal.frame_lines("", 0, "› ", "Approval required", "", "", "", width, 24);
            let total = terminal.approval_view.as_ref().unwrap().rows.len();
            let mut pages = Vec::new();
            // One prompt row plus the status strip leaves 22 transcript rows.
            for end in (1..=total).rev().step_by(22) {
                terminal.scroll = total - end;
                let (lines, _, _) =
                    terminal.frame_lines("", 0, "› ", "Approval required", "", "", "", width, 24);
                pages.push(
                    lines[..end.min(22)]
                        .iter()
                        .map(|line| render::plain(line))
                        .filter(|line| changed(line))
                        .collect::<Vec<_>>(),
                );
            }
            let rendered: Vec<_> = pages.into_iter().rev().flatten().collect();
            assert_eq!(rendered, expected_lines, "changed content at width {width}");
        }
        terminal.approval_ready().unwrap();
        prepared
            .apply_with_receipts(
                &preview,
                &std::sync::atomic::AtomicBool::new(false),
                |_, _| {},
            )
            .unwrap();
        assert_eq!(f.read("file"), "b\n".repeat(2500));
    }
    #[test]
    fn small_approval_matches_existing_rendering_and_narrow_views_pause() {
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        terminal.color = true;
        terminal.note("prior").unwrap();
        terminal
            .approval_preview("Patch preview", "-old\n+new")
            .unwrap();
        assert_eq!(
            terminal.approval_view.as_ref().unwrap().rows,
            terminal.transcript.rows
        );
        terminal
            .note("Approval still pending; draft retained.")
            .unwrap();
        assert_eq!(
            terminal.approval_view.as_ref().unwrap().rows,
            terminal.transcript.rows
        );
        terminal.close_approval();
        terminal.approval_preview("Patch preview", "+🦀").unwrap();
        let (lines, _, _) =
            terminal.frame_lines("", 0, "› ", "Approval required · y / n", "", "", "", 2, 24);
        assert!(terminal.approval_view.as_ref().unwrap().incomplete);
        assert!(terminal.approval_ready_at(1, 24).is_err());
        assert!(!lines.iter().any(|line| line.contains("Approval required")));
        terminal.frame_lines("", 0, "› ", "Approval required", "", "", "", 80, 24);
        assert!(!terminal.approval_view.as_ref().unwrap().incomplete);
        assert!(terminal.approval_ready_at(79, 2).is_err());
        terminal.approval_ready_at(79, 24).unwrap();
        assert!(
            terminal
                .approval_view
                .as_ref()
                .unwrap()
                .rows
                .iter()
                .any(|line| render::plain(line) == "+🦀")
        );
    }
    #[test]
    fn review_appends_match_reflow_and_mirror_exhaustion_is_visible_once() {
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        terminal.color = true;
        terminal
            .approval_preview("Patch preview", &"-old\n+new\n".repeat(2500))
            .unwrap();
        for width in [79, 7, 159] {
            let view = terminal.approval_view.as_mut().unwrap();
            for text in [
                "partial",
                "\t🦀",
                " continuation\n",
                "\x1b[2mnotice\x1b[0m\n",
            ] {
                view.append_review(text, width);
                let mut full = view.clone();
                full.width = 0;
                full.reflow(width);
                assert_eq!(view.rows, full.rows);
                assert_eq!(view.incomplete, full.incomplete);
            }
        }
        terminal
            .note(&"x".repeat(MAX_REVIEW_HISTORY_BYTES))
            .unwrap();
        terminal.note("another hidden notice").unwrap();
        let view = terminal.approval_view.as_ref().unwrap();
        assert!(view.bytes <= MAX_REVIEW_HISTORY_BYTES);
        assert_eq!(
            view.lines
                .iter()
                .filter(|l| l.contains("[later output hidden"))
                .count(),
            1
        );
        assert_eq!(
            view.lines
                .iter()
                .filter(|l| render::plain(l).starts_with("+new"))
                .count(),
            2500
        );
    }
    #[test]
    fn approval_capacity_refuses_without_showing_a_partial_prompt() {
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        assert!(
            terminal
                .approval_preview("Patch preview", &"+x\n".repeat(MAX_APPROVAL_BYTES))
                .is_err()
        );
        assert!(terminal.approval_view.is_none());
        assert!(terminal.transcript.rows.is_empty());
        terminal
            .transcript
            .push(&format!("{}\n", "x".repeat(192 * 1024)), 79);
        terminal
            .approval_preview("Patch preview", "-a\n+b")
            .unwrap();
        terminal.approval_ready_at(1, 24).unwrap();
        terminal.close_approval();
        terminal
            .approval_preview("Patch preview", &format!("+{}", "x".repeat(192 * 1024)))
            .unwrap();
        assert!(terminal.approval_ready_at(1, 24).is_err());
        assert!(
            !terminal
                .approval_view
                .as_ref()
                .unwrap()
                .rows
                .iter()
                .any(|line| line.contains("Approval required"))
        );
        terminal.approval_ready_at(79, 24).unwrap();
    }
    #[test]
    fn approvals_refuse_unescaped_unicode_and_preserve_escaped_preview_once() {
        for raw in [false, true] {
            let mut terminal = Terminal::open(true);
            terminal.raw = raw;
            for c in [
                '\t',
                '\0',
                '\u{200b}',
                '\u{202e}',
                '\u{fe0f}',
                '\u{2800}',
                '\u{e0100}',
            ] {
                assert!(
                    terminal
                        .approval_preview("Patch preview", &format!("+left{c}right"))
                        .is_err()
                );
                assert!(
                    terminal
                        .approval_preview(&format!("Title{c}"), "+safe")
                        .is_err()
                );
                assert!(terminal.approval_view.is_none());
                assert!(terminal.transcript.rows.is_empty());
            }
        }
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        terminal.color = true;
        let raw = "+literal\\u{200b}\n+actual\u{200b}\n+tab\tend";
        let shown = escape_preview(raw, MAX_APPROVAL_BYTES).unwrap();
        terminal.approval_preview("Patch preview", &shown).unwrap();
        let view = terminal.approval_view.as_mut().unwrap();
        for width in [79, 1, 159] {
            view.reflow(width);
            let displayed: String = view.rows.iter().map(|line| render::plain(line)).collect();
            for line in shown.lines() {
                assert!(displayed.contains(line), "escaped line at width {width}");
            }
            assert!(!view.incomplete);
        }
    }
    #[test]
    fn short_inline_approval_preview_commits_the_exact_patch() {
        // A preview that fits inline never enters the review screen, so the
        // committed rows are what the user approves: no moved words, no
        // dropped break space, no repeated speaker bar.
        let mut terminal = Terminal::open(true);
        terminal.raw = true;
        let cells = terminal.width().saturating_sub(1).max(1);
        assert!(cells > 12);
        let moved = format!("+{} -removed", "a".repeat(cells - 9));
        let spaced = format!("+{} b", "a".repeat(cells - 1));
        let bar = format!("▌ {} d", "c".repeat(cells - 2));
        let detail = format!("{moved}\n{spaced}\n{bar}\n+tail  two  spaces");
        terminal.approval_preview("Patch preview", &detail).unwrap();
        assert!(!terminal.needs_review(cells + 1, 24));
        let frame = terminal.inline_bytes(&[], 0, 1);
        let committed = &frame[frame.find("\x1b[J").unwrap() + 3..frame.rfind("\r\n").unwrap()];
        let rows: Vec<String> = committed.split("\r\n").map(render::plain).collect();
        assert!(rows.iter().all(|row| render::width(row) <= cells));
        let start = rows.iter().position(|row| row == "Patch preview").unwrap() + 1;
        let end = rows.iter().position(|row| render::approval(false) == *row).unwrap();
        assert_eq!(rows[start..end].concat(), detail.replace('\n', ""));
        assert!(rows.contains(&" b".to_owned()) && rows.contains(&" d".to_owned()));
        assert!(!rows.iter().any(|row| row.starts_with("-removed") || row.starts_with("▌ d")));
    }
    #[test]
    fn exit_reply_tail_is_short_and_sanitized() {
        assert_eq!(clipboard::osc52("foobar", false).unwrap(), "\x1b]52;c;Zm9vYmFy\x07");
        assert_eq!(clipboard::last_code("```rs\nold\n```\n~~~sh\r\n  echo 🦀\r\n\r\n~~~\r\n"), Some("  echo 🦀\r\n\r\n"));
        assert_eq!(clipboard::osc52("foobar", true).unwrap(), "\x1bPtmux;\x1b\x1b]52;c;Zm9vYmFy\x07\x1b\\");
        assert!(clipboard::osc52(&"x".repeat(clipboard::MAX_COPY_BYTES + 1), false).unwrap_err().to_string().contains("64 KiB"));
        assert_eq!(
            reply_tail("old\nfirst\nsecond 🦀\nlast\n"),
            "first\nsecond 🦀\nlast"
        );
        assert_eq!(reply_tail(&"🦀".repeat(1000)).chars().count(), 601);
        assert_eq!(reply_tail("\x1b[2Jhello"), "[2Jhello");
    }
