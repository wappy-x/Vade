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
                let (lines, row, col) = t.frame_lines("draft", 5, "> ", "status", "", "", "", 110, 24);
                bytes += if fullscreen { t.fullscreen_bytes(lines, row, col, 110, 24).len() } else { t.inline_bytes(&lines, row, col).len() }; cost += cpu() - before;
                dropped += usize::from(start.elapsed() > Duration::from_secs_f64(1. / 60.));
                std::thread::sleep(period.saturating_sub(start.elapsed()));
            }
            let start = Instant::now();
            std::hint::black_box(t.frame_lines("draft", 5, "> ", "status", "", "", "", 100, 30));
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
            "first token 1.5s · 23 tok/s · ctx 4.4k/32k · 96% cached"
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
                    "> ",
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
                    "> ",
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
                        .all(|line| render::width(line) <= width.saturating_sub(1).max(1))
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
                let footer = "╰ 14:02:41 · 29.6s · 412 tokens written · write 23 tok/s · read ≈884 tok/s · first token 3.2s · 3 model calls · ctx 44.1k/131k";
                t.note(&fit_status(footer, width - 1)).unwrap();
                t.warning(
                    "The next reply would not fit in the model's context.",
                    "Vade stopped and kept the history.",
                    "Use /compact to shrink older turns, or /new to start over.",
                )
                .unwrap();
                let status = "qwen3-8b · harness · approval: ask · ctx 44.1k/131k · session-1";
                let (lines, _, _) = t.frame_lines("", 0, "> ", status, "", "", "", width, 28);
                assert!(lines.iter().all(|line| render::width(line) < width));
                screens.push(mask(&lines.join("\n")));
            }
            // Labels alone separate the blocks when colour is off.
            assert_eq!(render::plain(&screens[0]), screens[1]);
            for (suffix, text) in [("ansi", &screens[0]), ("txt", &screens[1])] {
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
            t.transcript.push(&format!("── You · turn {i} ──\nneedle {i}\n{}", "gap\n".repeat(30)), 79);
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
            assert!(render::plain(&rows[1]).starts_with("── You · "));
        }
        assert_eq!(Decoder::default().feed(b"\x0f\x1b[111;5u"), [Key::CtrlO, Key::CtrlO]);
    }
    #[test]
    fn transcript_reflows_unicode_and_bounds_logical_and_physical_history() {
        let mut t = Transcript::default();
        t.push("a你好\tb\nnext", 5);
        assert_eq!(
            t.rows.iter().cloned().collect::<Vec<_>>(),
            ["a你好", "    b", "next"]
        );
        t.reflow(3);
        assert!(
            t.rows
                .iter()
                .all(|r| r.chars().map(cell_width).sum::<usize>() <= 3)
        );
        assert_eq!(t.rows.iter().cloned().collect::<String>(), "a你好    bnext");
        t.push(&"x".repeat(1024 * 1024 + 4), 1);
        assert!(t.bytes <= 1024 * 1024 && t.rows.len() <= 16 * 1024);
        t.push("\nend", 1);
        assert_eq!(t.rows.back().map(String::as_str), Some("d"));
        t.push(&"line\n".repeat(5000), 10);
        assert!(t.lines.len() <= 4096);
    }
    #[test]
    fn transcript_trims_in_batches_and_keeps_utf8_safe_headroom() {
        let mut t = Transcript::default();
        t.push(&"line\n".repeat(4096), 80);
        assert_eq!(t.lines.len(), 4096 - 512);
        let bytes = t.bytes;
        for _ in 0..400 {
            t.push("next\n", 80);
        }
        assert_eq!(t.lines.len(), 4096 - 512 + 400);
        assert_eq!(t.bytes, bytes + 400 * 5);
        assert_eq!(t.lines.front().unwrap().text, "line");
        assert_eq!(t.rows.len(), t.lines.len());
        let mut t = Transcript::default();
        t.push(&"🦀".repeat(1024 * 1024 / 4 + 1), 80);
        assert!(t.bytes <= 1024 * 1024 - 128 * 1024);
        let bytes = t.bytes;
        t.push(&"🦀".repeat(1024), 80);
        assert_eq!(t.bytes, bytes + 4096);
        assert!(t.lines.front().unwrap().starts_with('🦀'));
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
        // The scroll hint is reserved whole, ahead of the facts.
        let mut terminal = Terminal::open(true);
        terminal.transcript.push(&"row\n".repeat(60), 39);
        terminal.scroll = 10;
        let (lines, _, _) = terminal.frame_lines(
            "",
            0,
            "> ",
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
    fn diff_lines_are_coloured_by_kind_and_stay_literal_without_colour() {
        let sample = "*** Update File: a.rs\n--- a.rs\n+++ a.rs\n@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n[context line truncated: 9 bytes]";
        assert_eq!(render::diff(sample, false), sample);
        let styled = render::diff(sample, true);
        assert!(styled.contains("\x1b[32m+new\x1b[0m"));
        assert!(styled.contains("\x1b[31m-old\x1b[0m"));
        assert!(styled.contains("\x1b[36m@@ -1,2 +1,2 @@\x1b[0m"));
        assert!(styled.contains("\x1b[1m+++ a.rs\x1b[0m"));
        assert!(styled.contains("\n ctx\n"));
        assert_eq!(render::plain(&styled), sample);
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
            terminal.frame_lines("", 0, "> ", "Approval required", "", "", "", width, 24);
            let total = terminal.approval_view.as_ref().unwrap().rows.len();
            let mut pages = Vec::new();
            // Empty boxed input plus footer leaves twenty transcript rows.
            for end in (1..=total).rev().step_by(20) {
                terminal.scroll = total - end;
                let (lines, _, _) =
                    terminal.frame_lines("", 0, "> ", "Approval required", "", "", "", width, 24);
                pages.push(
                    lines[..end.min(20)]
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
            terminal.frame_lines("", 0, "> ", "Approval required · y / n", "", "", "", 2, 24);
        assert!(terminal.approval_view.as_ref().unwrap().incomplete);
        assert!(terminal.approval_ready_at(1, 24).is_err());
        assert!(!lines.iter().any(|line| line.contains("Approval required")));
        terminal.frame_lines("", 0, "> ", "Approval required", "", "", "", 80, 24);
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
    fn preview_escapes_unicode_controls_and_literal_lookalikes() {
        let invisible = [
            '\0',
            '\u{001b}',
            '\u{0085}',
            '\r',
            '\t',
            '\u{00ad}',
            '\u{00a0}',
            '\u{0301}',
            '\u{034f}',
            '\u{0483}',
            '\u{05b0}',
            '\u{064b}',
            '\u{0890}',
            '\u{115f}',
            '\u{1160}',
            '\u{17b4}',
            '\u{180f}',
            '\u{200b}',
            '\u{200c}',
            '\u{200d}',
            '\u{200e}',
            '\u{2028}',
            '\u{202e}',
            '\u{2060}',
            '\u{2065}',
            '\u{2800}',
            '\u{3164}',
            '\u{fe0f}',
            '\u{feff}',
            '\u{ffa0}',
            '\u{fff0}',
            '\u{fff9}',
            '\u{13440}',
            '\u{1d173}',
            '\u{1d185}',
            '\u{e0000}',
            '\u{e0001}',
            '\u{e007f}',
            '\u{e0100}',
            '\u{e0fff}',
        ];
        for c in invisible {
            assert!(preview_char_needs_escape(c), "U+{:04X}", c as u32);
            let raw = format!("before{c}after");
            let shown = escape_preview(&raw, 64).unwrap();
            assert!(!shown.contains(c), "raw U+{:04X}", c as u32);
            assert!(!shown.chars().any(preview_char_needs_escape));
            assert_eq!(sanitize(&shown), shown);
            assert_eq!(render::plain(&render::diff(&shown, true)), shown);
            assert_ne!(shown, escape_preview(&shown, 128).unwrap());
        }
        assert_eq!(
            escape_preview("é 中 🦀 '\"\n", 64).unwrap(),
            "é 中 🦀 '\"\n"
        );
        assert_eq!(
            escape_preview("literal\\u{200b}\nactual\u{200b}\n", 64).unwrap(),
            "literal\\\\u{200b}\nactual\\u{200b}\n"
        );
    }
    #[test]
    fn preview_escape_limit_counts_expanded_bytes_and_utf8_boundaries() {
        let raw = "é\u{200b}\t\\\0";
        let expected = "é\\u{200b}\\t\\\\\\u{0}";
        assert_eq!(escape_preview(raw, expected.len()).unwrap(), expected);
        assert!(escape_preview(raw, expected.len() - 1).is_err());
        assert!(escape_preview("é", 1).is_err());
        assert_eq!(escape_preview("", 0).unwrap(), "");
        assert!(escape_preview(&"\0".repeat(MAX_APPROVAL_BYTES), MAX_APPROVAL_BYTES).is_err());
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
    fn untrusted_text_cannot_control_the_terminal() {
        assert_eq!(sanitize("x\x1b[2Jy\x07\r\u{202e}z\n\t"), "x[2Jyz\n\t");
    }
    #[test]
    fn fragmented_unicode_and_navigation() {
        let mut d = Decoder::default();
        assert!(d.feed(&[0xe2, 0x82]).is_empty());
        assert_eq!(d.feed(&[0xac]), [Key::Text("€".into())]);
        assert!(d.feed(b"\x1b[").is_empty());
        assert_eq!(d.feed(b"A"), [Key::Up]);
        assert_eq!(
            d.feed(b"\r\x03\x7f"),
            [Key::Enter, Key::Interrupt, Key::Backspace]
        );
    }
    #[test]
    fn fragmented_wheel_reports_cannot_recall_prompts_or_enter_pasted_text() {
        let mut d = Decoder::default();
        assert!(d.feed(b"\x1b[<64;12;").is_empty());
        assert_eq!(d.feed(b"5M\x1b[<81;12;5M"), [Key::WheelUp, Key::WheelDown]);
        assert!(d.feed(b"\x1b[<0;12;5M\x1b[<0;12;5m").is_empty());
        assert_eq!(d.feed(b"\x1b[A\x1b[B"), [Key::Up, Key::Down]);
        assert_eq!(
            d.feed(b"\x1b[200~\x1b[<64;12;5M\x1b[201~"),
            [Key::PasteStart, Key::PasteEnd]
        );
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
    #[test]
    fn bracketed_paste_does_not_submit_commands() {
        let mut d = Decoder::default();
        let keys = d.feed(b"\x1b[200~/quit\n\x03\x1b[201~");
        assert_eq!(keys.first(), Some(&Key::PasteStart));
        assert_eq!(keys.last(), Some(&Key::PasteEnd));
        assert!(!keys.contains(&Key::Enter));
        assert!(!keys.contains(&Key::Interrupt));
        assert!(keys.contains(&Key::Newline));
    }
    #[test]
    fn paste_normalizes_cr_lf_without_losing_blank_lines() {
        let mut decoder = Decoder::default();
        let keys = decoder.feed(b"\x1b[200~\r\nfirst\r\rsecond\n\x1b[201~");
        assert_eq!(keys.iter().filter(|key| **key == Key::Newline).count(), 4);
        assert!(!keys.contains(&Key::Enter));
        assert_eq!(
            decoder.feed(b"\x1b[13;2u\x1b[27;2;13~\x1b\r\n\x14"),
            [
                Key::Newline,
                Key::Newline,
                Key::Newline,
                Key::Newline,
                Key::CtrlT
            ]
        );
        assert!(decoder.feed(b"\x1b[13;2:3u\x1b[99;9u").is_empty());
        assert_eq!(
            decoder.feed(b"\x1b[99;5u\x1b[27;5;106~"),
            [Key::Interrupt, Key::Newline]
        );
    }
    #[test]
    fn cursor_view_survives_narrow_resize_without_changing_input() {
        let text = "start 🦀 café\nsecond line";
        let cursor = "start 🦀 ca".len();
        for width in [4, 8, 20, 80] {
            let view = render::editor(text, cursor, "> ", width, 8, true, true);
            assert!(view.lines.iter().all(|line| render::width(line) <= width));
            assert!(view.cursor_column < width);
            assert!(view.cursor_row < view.lines.len());
        }
        assert_eq!(text, "start 🦀 café\nsecond line");
    }

    #[test]
    fn meta_chords_and_modified_arrows_move_by_word_instead_of_clearing() {
        let mut d = Decoder::default();
        assert_eq!(
            d.feed(b"\x1bb\x1bf\x1bd\x1b\x7f\x1b[1;5C\x1b[1;5D\x1b[1;3C\x1b[1;3D\x0b\x0c\x1bOC"),
            [
                Key::WordLeft,
                Key::WordRight,
                Key::DeleteWordForward,
                Key::DeleteWord,
                Key::WordRight,
                Key::WordLeft,
                Key::WordRight,
                Key::WordLeft,
                Key::KillToEnd,
                Key::Redraw,
                Key::Right,
            ]
        );
        // An unknown chord is still a lone Escape; its letter never types.
        assert_eq!(d.feed(b"\x1bq"), [Key::Escape]);
        // Esc in one read and a letter in the next is Esc, then that letter.
        assert!(d.feed(b"\x1b").is_empty());
        assert_eq!(d.feed(b"b"), [Key::Escape, Key::Text("b".into())]);
        assert!(d.feed(b"\x1b").is_empty());
        assert_eq!(d.feed(b"\x7f"), [Key::Escape, Key::Backspace]);
        assert!(d.feed(b"\x1b").is_empty());
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(d.timeout(), Some(Key::Escape));
    }
    #[test]
    fn navigation_and_large_paste_decode_without_per_character_work_events() {
        let mut decoder = Decoder::default();
        assert_eq!(
            decoder.feed(b"\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~"),
            [Key::Right, Key::Left, Key::Home, Key::LineEnd, Key::Delete]
        );
        let mut bytes = b"\x1b[200~".to_vec();
        bytes.extend(vec![b'x'; MAX_INPUT_BYTES]);
        bytes.extend_from_slice(b"\x1b[201~");
        let keys = decoder.feed(&bytes);
        assert_eq!(keys.len(), 3);
        assert!(matches!(&keys[1], Key::Text(text) if text.len() == MAX_INPUT_BYTES));
        assert!(!keys.contains(&Key::Enter));
    }
