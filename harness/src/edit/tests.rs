use super::*;
use crate::test_workspace::Fixture;
fn apply(w: &Workspace, p: &str) -> io::Result<String> {
    prepare(w, p)?.apply(&AtomicBool::new(false))
}
#[test]
fn approval_hash_rejects_patch_preview_and_plan_mismatches_before_writes() {
    let f = Fixture::new();
    let w = f.workspace();
    let patch = "*** Begin Patch\n*** Add File: result\n+exact\n*** End Patch";
    for changed in 0..3 {
        let mut prepared = prepare(&w, patch).unwrap();
        let preview = prepared.preview();
        match changed {
            0 => prepared.patch.push('\n'), // Same parsed diff, different bytes.
            1 => prepared.diff.push_str("+hidden\n"),
            _ => prepared.changes[0].after = Some("hidden\n".into()),
        }
        let mut receipts = 0;
        let error = prepared
            .apply_with_receipts(&preview, &AtomicBool::new(false), |_, _| receipts += 1)
            .unwrap_err();
        assert!(error.to_string().contains("integrity mismatch"));
        assert!(!error.uncertain_writes);
        assert_eq!(receipts, 0);
        assert!(!f.0.join("result").exists());
    }
}
#[test]
fn previews_controls_and_invisible_unicode_then_applies_original_bytes() {
    let f = Fixture::new();
    let w = f.workspace();
    let characters = [
        ('\u{001b}', "\\u{1b}"),
        ('\u{0000}', "\\u{0}"),
        ('\u{0085}', "\\u{85}"),
        ('\u{202e}', "\\u{202e}"),
        ('\t', "\\t"),
        ('\r', "\\r"),
        ('\u{200b}', "\\u{200b}"),
        ('\u{200c}', "\\u{200c}"),
        ('\u{200d}', "\\u{200d}"),
        ('\u{2060}', "\\u{2060}"),
        ('\u{feff}', "\\u{feff}"),
        ('\u{0301}', "\\u{301}"),
        ('\u{00a0}', "\\u{a0}"),
        ('\u{2800}', "\\u{2800}"),
        ('\u{e0100}', "\\u{e0100}"),
    ];
    let payload: String = characters
        .iter()
        .map(|(c, _)| format!("left{c}right\n"))
        .collect();
    let patch = format!(
        "*** Begin Patch\n*** Add File: file\n{}*** End Patch",
        payload
            .split_inclusive('\n')
            .map(|line| format!("+{line}"))
            .collect::<String>()
    );
    assert_eq!(touched_paths(&patch).unwrap(), ["file"]);
    let prepared = prepare(&w, &patch).unwrap();
    let preview = prepared.preview();
    assert_eq!(preview.patch.as_bytes(), patch.as_bytes());
    for (_, escaped) in characters {
        assert!(preview.diff().contains(&format!("+left{escaped}right\n")));
    }
    assert!(
        !preview
            .diff()
            .chars()
            .any(crate::terminal::preview_char_needs_escape)
    );
    assert_eq!(crate::terminal::sanitize(preview.diff()), preview.diff());
    prepared
        .apply_with_receipts(&preview, &AtomicBool::new(false), |_, _| {})
        .unwrap();
    assert_eq!(f.read("file").as_bytes(), payload.as_bytes());

    // The same escape policy covers untouched context and removed lines.
    f.write("source", &format!("{payload}old\u{200b}\n"));
    let patch = "*** Begin Patch\n*** Update File: source\n@@\n-old\u{200b}\n+new\u{200d}\n*** End Patch";
    let prepared = prepare(&w, patch).unwrap();
    assert!(prepared.diff().contains(" left\\u{2800}right\n"));
    assert!(prepared.diff().contains(" left\\u{e0100}right\n"));
    assert!(prepared.diff().contains("-old\\u{200b}\n+new\\u{200d}\n"));
    prepared.apply(&AtomicBool::new(false)).unwrap();
    assert_eq!(f.read("source"), format!("{payload}new\u{200d}\n"));
}
#[test]
fn literal_escapes_and_non_ascii_content_are_distinct_in_preview() {
    let f = Fixture::new();
    let w = f.workspace();
    let path = "file";
    let payload = "literal\\u{200b}\nactual\u{200b}\n\\t\t\\r\r\\0\0\n";
    let patch = format!(
        "*** Begin Patch\n*** Add File: {path}\n{}*** End Patch",
        payload
            .split_inclusive('\n')
            .map(|line| format!("+{line}"))
            .collect::<String>()
    );
    let prepared = prepare(&w, &patch).unwrap();
    assert!(prepared.diff().contains("+++ file\n"));
    assert!(
        prepared
            .diff()
            .contains("+literal\\\\u{200b}\n+actual\\u{200b}\n+\\\\t\\t\\\\r\\r\\\\0\\u{0}\n")
    );
    prepared.apply(&AtomicBool::new(false)).unwrap();
    assert_eq!(f.read(path), payload);

    let target = "moved";
    let patch = format!(
        "*** Begin Patch\n*** Update File: {path}\n*** Move to: {target}\n*** End Patch"
    );
    assert_eq!(touched_paths(&patch).unwrap(), [path, target]);
    let prepared = prepare(&w, &patch).unwrap();
    assert!(
        prepared
            .diff()
            .contains("--- file\n+++ moved\n")
    );
    prepared.apply(&AtomicBool::new(false)).unwrap();
    assert!(!f.0.join(path).exists());
    assert_eq!(f.read(target), payload);
}
#[test]
fn invisible_byte_changes_after_approval_are_rejected_before_writing() {
    let f = Fixture::new();
    let w = f.workspace();
    let patch = "*** Begin Patch\n*** Add File: result\n+one\u{200b}two\n*** End Patch";
    for changed in 0..3 {
        let mut prepared = prepare(&w, patch).unwrap();
        let preview = prepared.preview();
        match changed {
            0 => prepared.patch = prepared.patch.replace('\u{200b}', "\u{200d}"),
            1 => prepared.diff = prepared.diff.replace("\\u{200b}", "\\u{200d}"),
            _ => prepared.changes[0].after = Some("one\u{200d}two\n".into()),
        }
        let error = prepared
            .apply_with_receipts(&preview, &AtomicBool::new(false), |_, _| {
                panic!("mismatched bytes must not commit")
            })
            .unwrap_err();
        assert!(error.to_string().contains("integrity mismatch"));
        assert!(!error.uncertain_writes);
        assert!(!f.0.join("result").exists());
    }
}
#[test]
fn escape_expansion_is_bounded_and_refuses_complete_patch_before_writes() {
    let f = Fixture::new();
    let w = f.workspace();
    let mut preview = Preview::default();
    preview.escaped(&"\0".repeat(MAX_PREVIEW));
    assert!(preview.overflow);
    assert!(preview.text.len() <= MAX_PREVIEW);
    assert!(preview.finish().is_err());
    let patch = format!(
        "*** Begin Patch\n*** Add File: file\n+{}\n*** End Patch",
        "\0".repeat(MAX_PATCH - 100)
    );
    assert!(patch.len() < MAX_PATCH);
    let error = prepare(&w, &patch).err().unwrap();
    assert!(error.to_string().contains("preview limit"));
    assert!(!f.0.join("file").exists());
}
#[test]
fn adds_updates_moves_and_deletes_exactly() {
    let f = Fixture::new();
    let w = f.workspace();
    let large = (0..1600).map(|i| format!("VALUE_{i:05} = {i}  # generated constant number {i} for the preview size test\n")).collect::<String>();
    assert!(large.len() > MAX_PATCH);
    f.write("large.py", &large);
    let prepared = prepare(&w, "*** Begin Patch\n*** Update File: large.py\n@@\n-VALUE_00800 = 800  # generated constant number 800 for the preview size test\n+VALUE_00800 = 800\n*** End Patch").unwrap();
    assert!(prepared.diff().contains("@@ -798,7 +798,7 @@\n"));
    assert!(!prepared.diff().contains("VALUE_00000"));
    assert!(prepared.diff().len() < 1024);
    prepared.apply(&AtomicBool::new(false)).unwrap();
    assert_eq!(
        f.read("large.py"),
        large.replace(
            "VALUE_00800 = 800  # generated constant number 800 for the preview size test",
            "VALUE_00800 = 800"
        )
    );
    apply(
        &w,
        "*** Begin Patch\n*** Add File: a\n+one\n+two\n*** End Patch\n",
    )
    .unwrap();
    assert_eq!(f.read("a"), "one\ntwo\n");
    let p = prepare(
        &w,
        "*** Begin Patch\n*** Update File: a\n@@\n one\n-two\n+three\n*** End Patch",
    )
    .unwrap();
    assert!(p.diff().contains("-two\n"));
    assert!(p.diff().contains("+three\n"));
    p.apply(&AtomicBool::new(false)).unwrap();
    apply(
        &w,
        "*** Begin Patch\n*** Update File: a\n*** Move to: b\n*** End Patch",
    )
    .unwrap();
    assert!(!f.0.join("a").exists());
    assert_eq!(f.read("b"), "one\nthree\n");
    apply(
        &w,
        "*** Begin Patch\n*** Delete File: b\n-one\n-three\n*** End Patch",
    )
    .unwrap();
    assert!(!f.0.join("b").exists());
}
#[test]
fn preview_merges_context_caps_context_lines_and_fails_closed() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write(
        "lines",
        &(0..30).map(|i| format!("line{i}\n")).collect::<String>(),
    );
    let p = prepare(&w, "*** Begin Patch\n*** Update File: lines\n@@\n-line5\n+first\n+extra\n line6\n-line7\n+second\n@@\n-line20\n*** End Patch").unwrap();
    assert!(p.diff().contains("@@ -3,9 +3,10 @@\n line2\n line3\n line4\n-line5\n+first\n+extra\n line6\n-line7\n+second\n line8\n line9\n line10\n"));
    assert!(p.diff().contains("@@ -18,7 +19,6 @@"));
    f.write("joined", "oldtail\n");
    assert!(prepare(&w, "*** Begin Patch\n*** Update File: joined\n@@\n-old\n\\ No newline at end of file\n tail\n*** End Patch").unwrap().diff().contains("-oldtail\n+tail\n"));
    f.write(
        "wide",
        &format!(
            "{}\nold\n{}\n",
            "x".repeat(MAX_PREVIEW),
            "y".repeat(MAX_PREVIEW)
        ),
    );
    let p = prepare(
        &w,
        "*** Begin Patch\n*** Update File: wide\n@@\n-old\n+new\n*** End Patch",
    )
    .unwrap();
    assert!(p.diff().len() < 4096);
    assert!(p.diff().contains(&format!(
        "@@ -1,3 +1,3 @@\n {}\n[context line truncated: {} bytes]\n-old\n+new\n {}\n[context line truncated: {} bytes]\n",
        "x".repeat(MAX_CONTEXT_LINE),
        MAX_PREVIEW,
        "y".repeat(MAX_CONTEXT_LINE),
        MAX_PREVIEW
    )));
    // Many hunks, each with wide context, cannot all be shown: fail closed.
    let wide = "z".repeat(MAX_CONTEXT_LINE * 4);
    let body: String = (0..400).map(|i| format!("{wide}\nk{i}\n")).collect();
    f.write("many", &format!("{body}{wide}\n"));
    let hunks: String = (0..400).map(|i| format!("@@\n-k{i}\n+v{i}\n")).collect();
    let e = prepare(
        &w,
        &format!("*** Begin Patch\n*** Update File: many\n{hunks}*** End Patch"),
    )
    .err()
    .unwrap();
    assert!(e.to_string().contains("preview limit"));
}
#[test]
fn conflict_after_preview_never_overwrites_external_edit() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "old\n");
    let p = prepare(
        &w,
        "*** Begin Patch\n*** Update File: a\n@@\n-old\n+new\n*** End Patch",
    )
    .unwrap();
    f.write("a", "concurrent\n");
    let preview = p.preview();
    let error = p
        .apply_with_receipts(&preview, &AtomicBool::new(false), |_, _| {
            panic!("preflight conflict must not commit")
        })
        .unwrap_err();
    assert!(!error.uncertain_writes);
    assert_eq!(f.read("a"), "concurrent\n");
}
#[test]
fn rejects_missing_ambiguous_and_out_of_order_old_text() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "x\nx\ny\n");
    for p in [
        "@@\n-x\n+z",
        "@@\n-missing\n+z",
        "@@\n-y\n+z\n@@\n x\n x\n+a",
    ] {
        assert!(
            prepare(
                &w,
                &format!("*** Begin Patch\n*** Update File: a\n{p}\n*** End Patch")
            )
            .err().is_some_and(|e| ["\"a\"", "workspace_read_lines", "Resend", "no files were written"].iter().all(|hint| e.to_string().contains(hint)))
        );
    }
    assert_eq!(f.read("a"), "x\nx\ny\n");
}
#[test]
fn accepts_format_slips_and_refuses_ambiguous_or_outside_forms() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let outside = Fixture::new();
    let w = f.workspace();
    // No markers, no '+' and a missing parent; then no End, bare empty line.
    let raw = "*** Add File: src/deep/new.rs\nfn main() {\n\n}";
    assert_eq!(touched_paths(raw).unwrap(), ["src/deep/new.rs"]);
    apply(&w, raw).unwrap();
    assert_eq!(f.read("src/deep/new.rs"), "fn main() {\n\n}\n");
    apply(&w, "*** Begin Patch\n*** Add File: src/deep/two\n+a\n\n+b").unwrap();
    assert_eq!(f.read("src/deep/two"), "a\n\nb\n");
    apply(&w, "*** Delete File: src/deep/two\n*** End Patch").unwrap();
    assert!(!f.0.join("src/deep/two").exists());
    // Spaces/tabs differ: unique match keeps the file's unchanged lines.
    f.write("ws", "fn a() {\n\tlet x = 1;  \n\treturn x;\n}\n");
    let summary = apply(
        &w,
        "*** Update File: ws\n@@\n fn a() {\n-    let x  = 1;\n+    let x = 2;\n     return x;",
    )
    .unwrap();
    assert!(summary.contains("whitespace-tolerant match used in \"ws\""));
    assert_eq!(f.read("ws"), "fn a() {\n    let x = 2;\n\treturn x;\n}\n");
    f.write("dup", "a  b\nx\na\tb\nold\r\n");
    let refused = |p: &str| prepare(&w, p).err().unwrap().to_string();
    assert!(refused("*** Update File: dup\n@@\n-a b\n+c").contains("matches more than once"));
    assert!(refused("*** Update File: dup\n@@\n-old\n+new").contains("old text missing"));
    assert!(
        refused("*** Add File: mixed\n+a\nb").contains("start every content line with '+'")
    );
    for fence in ["```", "```diff", "```patch"] { let patch = format!("{fence}\n*** Add File: fenced-{}\n+a\n```\n", fence.len()); assert_eq!(touched_paths(&patch).unwrap(), [format!("fenced-{}", fence.len())]); apply(&w, &patch).unwrap(); assert_eq!(f.read(&format!("fenced-{}", fence.len())), "a\n"); } assert!(refused("prose\n```diff\n*** Add File: no\n+x\n```").contains("Markdown fences")); assert!(refused("```diff\n*** Add File: no\n+x").contains("Markdown fences"));
    f.write("blank", "before\n\nold\nafter\n"); apply(&w, "*** Update File: blank\n@@ -99,4 +99,4 @@\n before\n\n-old\n+new\n after").unwrap(); assert_eq!(f.read("blank"), "before\n\nnew\nafter\n"); assert!(refused("*** Update File: dup\n@@ -1 +1 @@\n-a b\n+c").contains("matches more than once"));
    assert!(
        refused("*** Add File: n\n+x\n*** Add File: n/m\n+y")
            .contains("both a file and a folder")
    );
    // Parents are never created through links or outside the workspace,
    // including a link that appears after the preview.
    symlink(&outside.0, f.0.join("out")).unwrap();
    for path in [
        "out/new/file",
        "../new/file",
        "/tmp/new/file",
        "src/.hidden/file",
    ] {
        assert!(prepare(&w, &format!("*** Add File: {path}\n+x")).is_err());
    }
    let p = prepare(&w, "*** Add File: later/new/file\n+x").unwrap();
    symlink(&outside.0, f.0.join("later")).unwrap();
    assert!(p.apply(&AtomicBool::new(false)).is_err());
    assert!(!outside.0.join("new").exists());
    assert_eq!(f.read("dup"), "a  b\nx\na\tb\nold\r\n");
}
#[test]
fn preserves_no_final_newline_and_executable_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "old");
    std::fs::set_permissions(f.0.join("a"), std::fs::Permissions::from_mode(0o755)).unwrap();
    apply(&w, "*** Begin Patch\n*** Update File: a\n@@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n*** End Patch").unwrap();
    assert_eq!(f.read("a"), "new");
    assert_eq!(
        std::fs::metadata(f.0.join("a")).unwrap().mode() & 0o777,
        0o755
    );
}
#[test]
fn preflights_all_files_and_refuses_existing_move_destination() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "a\n");
    f.write("b", "b\n");
    assert!(
        prepare(
            &w,
            "*** Begin Patch\n*** Add File: c\n+new\n*** Delete File: a\n-wrong\n*** End Patch"
        )
        .is_err()
    );
    assert!(!f.0.join("c").exists());
    assert!(
        prepare(
            &w,
            "*** Begin Patch\n*** Update File: a\n*** Move to: b\n*** End Patch"
        )
        .is_err()
    );
    let p = prepare(&w, "*** Begin Patch\n*** Add File: c\n+new\n*** End Patch").unwrap();
    f.write("c", "external");
    assert!(p.apply(&AtomicBool::new(false)).is_err());
    assert_eq!(f.read("c"), "external");
}
#[test]
fn refuses_path_escapes_links_secrets_binary_and_large_files() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let outside = Fixture::new();
    let w = f.workspace();
    outside.write("a", "outside\n");
    symlink(&outside.0, f.0.join("link")).unwrap();
    std::fs::hard_link(outside.0.join("a"), f.0.join("hard")).unwrap();
    for path in [
        "../a",
        "/tmp/a",
        ".env",
        "credentials.json",
        "link/a",
        "hard",
    ] {
        assert!(prepare(&w, &format!("*** Begin Patch\n*** Update File: {path}\n@@\n-outside\n+changed\n*** End Patch")).is_err());
    }
    std::fs::write(f.0.join("binary"), [255]).unwrap();
    f.write("large", &"a".repeat(MAX_TEXT_BYTES + 1));
    for path in ["binary", "large"] {
        assert!(
            prepare(
                &w,
                &format!("*** Begin Patch\n*** Delete File: {path}\n*** End Patch")
            )
            .is_err()
        );
    }
    assert_eq!(outside.read("a"), "outside\n");
}
#[test]
fn cancellation_before_commit_has_no_side_effects() {
    let f = Fixture::new();
    let w = f.workspace();
    let p = prepare(&w, "*** Begin Patch\n*** Add File: a\n+new\n*** End Patch").unwrap();
    assert!(p.apply(&AtomicBool::new(true)).is_err());
    assert!(!f.0.join("a").exists());
}
#[test]
fn exact_line_boundaries_empty_files_and_crlf_are_preserved() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "prefix-old\n");
    assert!(
        prepare(
            &w,
            "*** Begin Patch\n*** Update File: a\n@@\n-old\n+new\n*** End Patch"
        )
        .is_err()
    );
    f.write("empty", "");
    apply(
        &w,
        "*** Begin Patch\n*** Update File: empty\n@@\n+new\n*** End Patch",
    )
    .unwrap();
    assert_eq!(f.read("empty"), "new\n");
    f.write("crlf", "old\r\n");
    apply(
        &w,
        "*** Begin Patch\n*** Update File: crlf\n@@\n-old\r\n+new\r\n*** End Patch",
    )
    .unwrap();
    assert_eq!(f.read("crlf"), "new\r\n");
}
#[test]
fn loose_match_never_moves_a_hunk_to_another_nesting_depth() {
    let f = Fixture::new();
    let w = f.workspace();
    // The closure (8 spaces) was already edited, so its exact text is gone.
    // The helper (4 spaces) differs only in indentation and must not match.
    let file = "fn outer() {\n    let f = || {\n        let x = compute();\n        let y = x + 2;\n        y\n    };\n}\nfn helper() {\n    let x = compute();\n    let y = x + 1;\n    y\n}\n";
    f.write("a.rs", file);
    let error = prepare(&w, "*** Update File: a.rs\n@@\n         let x = compute();\n-        let y = x + 1;\n+        let y = x + 3;").err().unwrap().to_string();
    assert!(error.contains("old text missing"), "{error}");
    assert_eq!(f.read("a.rs"), file);
    // Tabs against the same width in spaces still match, at 4 or 8 columns.
    for (tab, spaces) in [("\t", "    "), ("\t", "        ")] {
        f.write("t.go", &format!("func a() {{\n{tab}x  := 1\n}}\n"));
        apply(&w, &format!("*** Update File: t.go\n@@\n func a() {{\n-{spaces}x := 1\n+{spaces}x := 2")).unwrap();
        assert_eq!(f.read("t.go"), format!("func a() {{\n{spaces}x := 2\n}}\n"));
    }
}
#[test]
fn apply_patch_add_retry_refuses_to_strip_plus_from_mixed_bare_file() {
    let f = Fixture::new();
    let w = f.workspace();
    let patch = "*** Begin Patch\n*** Add File: fixture.txt\nbare_line\n+literal_plus_line\n*** End Patch";
    assert!(prepare(&w, patch).err().unwrap().to_string().contains("mixes '+'"));
    assert!(prepare_add_retry(&w, patch).is_err());
    f.write("fixture.txt", "old\n");
    assert!(prepare_add_retry(&w, patch).is_err());
    // A fully bare or fully prefixed body keeps its exact meaning.
    for (body, text) in [("bare\nalso", "bare\nalso\n"), ("+a\n++b", "a\n+b\n")] {
        f.write("fixture.txt", "old\n");
        let (header, retry) = prepare_add_retry(&w, &format!("*** Add File: fixture.txt\n{body}")).unwrap();
        assert_eq!(header, "*** Update File: fixture.txt");
        retry.apply(&AtomicBool::new(false)).unwrap();
        assert_eq!(f.read("fixture.txt"), text);
    }
}
/// A guard for the stamp check after preview. The ctime check inside one
/// `Snapshot::read` covers a race this test cannot time, so main passes too.
#[test]
fn same_length_change_after_preview_is_refused() {
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "one\n");
    let p = prepare(&w, "*** Update File: a\n@@\n-one\n+uno").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    f.write("a", "two\n");
    let error = p.apply(&AtomicBool::new(false)).unwrap_err().to_string();
    assert!(error.contains("source changed since preview"), "{error}");
    assert_eq!(f.read("a"), "two\n");
}
#[test]
fn multi_file_patch_rolls_back_or_finishes_instead_of_half_applying() {
    let patch = "*** Update File: a\n@@\n-a\n+A\n*** Add File: n\n+new\n*** Update File: b\n@@\n-b\n+B";
    // Another writer changes b after a and n commit: both are undone.
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "a\n");
    f.write("b", "b\n");
    let p = prepare(&w, patch).unwrap();
    let preview = p.preview();
    let mut receipts = Vec::new();
    let error = p.apply_with_receipts(&preview, &AtomicBool::new(false), |title, path| {
        receipts.push(format!("{title} {path}"));
        if path == "n" {
            std::fs::write(f.0.join("b"), "external\n").unwrap();
        }
    }).unwrap_err();
    assert!(error.uncertain_writes && error.to_string().contains("rolled back the 2 earlier"), "{error}");
    assert_eq!((f.read("a"), f.read("b")), ("a\n".into(), "external\n".into()));
    assert!(!f.0.join("n").exists());
    assert_eq!(receipts, ["File written a", "File written n", "File removed n", "File restored a"]);
    // Cancel after the first write lets the checked batch finish.
    let f = Fixture::new();
    let w = f.workspace();
    f.write("a", "a\n");
    f.write("b", "b\n");
    let p = prepare(&w, patch).unwrap();
    let preview = p.preview();
    let cancel = AtomicBool::new(false);
    p.apply_with_receipts(&preview, &cancel, |_, _| cancel.store(true, Ordering::Relaxed)).unwrap();
    assert_eq!((f.read("a"), f.read("n"), f.read("b")), ("A\n".into(), "new\n".into(), "B\n".into()));
}
#[test]
fn a_failed_write_after_the_first_file_rolls_back_every_undoable_file() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let w = f.workspace();
    std::fs::create_dir(f.0.join("sub")).unwrap();
    f.write("a", "a\n");
    f.write("m", "m\n");
    f.write("sub/b", "b\n");
    // Updates a, moves m (never undone), then fails to write into sub.
    let p = prepare(&w, "*** Update File: a\n@@\n-a\n+A\n*** Update File: m\n*** Move to: moved\n*** Update File: sub/b\n@@\n-b\n+B").unwrap();
    let preview = p.preview();
    let mut receipts = Vec::new();
    let error = p.apply_with_receipts(&preview, &AtomicBool::new(false), |title, path| {
        receipts.push(format!("{title} {path}"));
        if path == "m" {
            std::fs::set_permissions(f.0.join("sub"), std::fs::Permissions::from_mode(0o555)).unwrap();
        }
    }).unwrap_err();
    std::fs::set_permissions(f.0.join("sub"), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(error.uncertain_writes && error.to_string().contains("could not roll back all of the 2 earlier"), "{error}");
    // The move stays; a is still restored after the move's undo failed.
    assert_eq!((f.read("a"), f.read("moved"), f.read("sub/b")), ("a\n".into(), "m\n".into(), "b\n".into()));
    assert_eq!(receipts, ["File written a", "File written moved", "File removed m", "File restored a"]);
}
