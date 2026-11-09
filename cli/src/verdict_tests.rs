//! Inputs and expected reads from screen-verdict's selftest_1101.
//! The older-rule checks in that function do not apply to this fixed rule.
use super::*;
use crate::evals::json::object;

const FIXTURES: &str = include_str!("../tests/fixtures/verdict-1101.json");

fn strings(values: &[&str]) -> Vec<String> { values.iter().map(|v| (*v).into()).collect() }
fn options(values: &[&str]) -> Options { parse(&[strings(values), strings(&["cells.json"])].concat()).unwrap() }
fn screen(cells: &str) -> Value { json::parse(&format!(r#"{{"screen":"syn","cells":{cells}}}"#)).unwrap() }
fn evaluate(cells: &str, args: &[&str]) -> (String, i32) { judge(&[screen(cells)], &options(args)).unwrap() }

#[test]
fn every_1101_selftest_fixture_has_the_same_reads_verdict_text_and_exit_code() {
    let cases = json::parse(FIXTURES).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 12);
    for case in cases.as_array().unwrap() {
        let input = case["input"].clone();
        let o = options(&[]);
        let (text, code) = judge(std::slice::from_ref(&input), &o).unwrap();
        let reads: Vec<_> = input["cells"].as_array().unwrap().iter().map(|c| row(c, BASE, &o).unwrap().read).collect();
        let expected: Vec<_> = case["reads"].as_array().unwrap().iter().map(|r| r.as_str().unwrap()).collect();
        assert_eq!(reads, expected, "{}", case["case"].as_str().unwrap());
        let verdict = case["verdict"].as_str().unwrap();
        let tail = if verdict == "SECOND-START" { "; a cell is within the band of the line: run one more X start (A X X) and judge again" } else { "" };
        // Keep the expected rule text separate from the implementation constant.
        let expected = format!("{verdict} syn | rule 2026-11-01 band max(base, own A/A), cached < 1 s floor 10 ms, second X only near the line | {}{tail}", case["why"].as_str().unwrap());
        assert_eq!(text.lines().last(), Some(expected.as_str()), "{}", case["case"].as_str().unwrap());
        assert_eq!(code, match verdict { "SCREEN PASS" => 0, "SCREEN FAIL" => 1, "SECOND-START" => 3, _ => panic!("unknown fixture verdict") });
    }
    println!("verdict parity: {} saved cases matched reads, verdict text and exit codes", cases.as_array().unwrap().len());
}

#[test]
fn the_1101_report_selftest_names_the_floor_and_revision() {
    let fixtures = json::parse(FIXTURES).unwrap();
    let o = options(&["--sha", &"c".repeat(40)]);
    let (text, code) = judge(&[fixtures[0]["input"].clone()], &o).unwrap();
    assert_eq!(code, 0);
    assert!(text.lines().any(|l| l.starts_with("CELL OFF 8K/warm_replay") && l.contains("FLOOR; +2.5 ms <= 10 ms floor")), "{text}");
    assert!(text.lines().last().unwrap().starts_with(&format!("SCREEN PASS {} | syn | rule 2026-11-01", "c".repeat(40))));
}

#[test]
fn inherited_band_selftests_keep_their_verdicts() {
    // These inputs come from selftest_bands and selftest_r4. No history is needed.
    for (cells, args, read, verdict, code) in [
        (r#"[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]}]"#, &[][..], "FAIL", "RERUN", 4),
        (r#"[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]}]"#, &["--final"], "FAIL", "SCREEN FAIL", 1),
        (r#"[{"cell":"ON a","unit":"tok/s","a":[100,100.5],"x":[98,98.4]}]"#, &[], "FAIL", "SCREEN FAIL", 1),
        (r#"[{"cell":"ON a","unit":"tok/s","a":[100,102],"x":[99.5,99.7]}]"#, &[], "NOISE", "SCREEN PASS", 0),
        (r#"[{"cell":"ON a","unit":"tok/s","a":[100,100],"x":[99.4,99.4]}]"#, &[], "INSIDE", "SCREEN PASS", 0),
        (r#"[{"cell":"lat","unit":"s","a":[1,1],"x":[1.015,1.015]}]"#, &[], "FAIL", "SCREEN FAIL", 1),
        (r#"[{"cell":"ON C1","unit":"tok/s","a":[95,100],"x":[93,93]}]"#, &[], "NOISE", "SCREEN PASS", 0),
        (r#"[{"cell":"ON C1","unit":"tok/s","a":[100,96,100],"x":[95,95,95]}]"#, &["--final"], "NOISE", "SCREEN PASS", 0),
        (r#"[{"cell":"ON C1","unit":"tok/s","a":[96,100,100],"x":[95,95,95]}]"#, &["--final"], "FAIL", "SCREEN FAIL", 1),
        (r#"[{"cell":"ON C1","unit":"tok/s","a":[99],"x":[98.3]}]"#, &[], "INSIDE", "SCREEN PASS", 0),
        (r#"[{"cell":"ON C1","unit":"tok/s","a":[99],"x":[97.8]}]"#, &[], "SECOND-START", "SECOND-START", 3),
        (r#"[{"cell":"edge","unit":"tok/s","a":100,"x":99.2}]"#, &[], "INSIDE", "SCREEN PASS", 0),
    ] {
        let (text, actual_code) = evaluate(cells, args);
        assert_eq!(actual_code, code, "{text}");
        assert!(text.lines().next().unwrap().contains(&format!("| {read}")), "{text}");
        assert!(text.lines().last().unwrap().starts_with(verdict), "{text}");
    }
    let two = r#"[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]},{"cell":"ON b","unit":"tok/s","a":[100,101],"x":[99,99.4]}]"#;
    assert_eq!(evaluate(two, &[]).1, 1);
    let (text, _) = evaluate(r#"[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]}]"#, &[]);
    assert_eq!(text.lines().last().unwrap(), "RERUN ON a | syn | delta 1.294 % > band 1.000 % (own A/A (2 A starts)), inside 2 x own A/A 2.000 % | run a 3+3 cooled re-run of this cell (A X X A A X), then judge it with --final --only 'ON a'");
}

#[test]
fn single_x_uses_own_aa_for_the_second_start_limit() {
    for (x, read, code) in [(102.01, "SECOND-START", 3), (104.03, "SECOND-START", 3),
        (105.04, "SECOND-START", 3), (105.04001, "FAIL", 1)] {
        let cells = format!(r#"[{{"cell":"lat","unit":"s","a":[100,102],"x":[{x}]}}]"#);
        let (text, actual) = evaluate(&cells, &[]);
        assert_eq!(actual, code, "{text}");
        assert!(text.lines().next().unwrap().contains(&format!("| {read}; band 2.000 % (own A/A (2 A starts))")), "{text}");
    }
}

#[test]
fn fail_takes_precedence_over_second_start_and_blocks_rerun() {
    let second = r#"{"cell":"second","unit":"tok/s","a":[100],"x":[98.8]}"#;
    let fail = r#"{"cell":"fail","unit":"tok/s","a":[100,101],"x":[99,99.4]}"#;
    for cells in [format!("[{second},{fail}]"), format!("[{fail},{second}]")] {
        let (text, code) = evaluate(&cells, &[]);
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("| SECOND-START;") && text.contains("| FAIL;"), "{text}");
        assert!(text.lines().last().unwrap().starts_with("SCREEN FAIL"), "{text}");
    }
}

#[test]
fn pooled_values_use_recorded_aa_but_lists_use_their_own_aa() {
    let cells = r#"[{"cell":"ON C8","unit":"tok/s","a":98.0065,"x":97.141,"a_starts":4,"x_starts":3,"aa_pct":-1.424,"balanced_a":98.0065,"balanced_x":97.611}]"#;
    let (text, code) = evaluate(cells, &[]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("NOISE; band 1.424 % (own A/A (4 A starts))"), "{text}");
    let (text, code) = evaluate(r#"[{"cell":"ON C1","unit":"tok/s","a":[100,100],"x":[98,98],"aa_pct":20}]"#, &[]);
    assert_eq!(code, 1, "{text}");
}

#[test]
fn ids_values_match_the_script_and_override_all_other_decisions() {
    let s = screen(r#"[{"cell":"ON a","unit":"tok/s","a":[100,100],"x":[100,100]}]"#);
    for (ids, code) in [(Value::Int(0), 1), (Value::Null, 1), (Value::from("no"), 1), (Value::Bool(false), 1), (Value::Bool(true), 0)] {
        let mut s = s.clone(); s.insert("ids_equal", ids);
        assert_eq!(judge(&[s], &options(&[])).unwrap().1, code);
    }
    assert_eq!(judge(std::slice::from_ref(&s), &options(&[])).unwrap().1, 0);
    assert_eq!(judge(&[s], &options(&["--ids-ok", "no"])).unwrap().1, 1);
    let (text, code) = evaluate(r#"[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]}]"#, &["--ids-ok", "no"]);
    assert_eq!(code, 1);
    assert!(text.lines().last().unwrap().contains("IDS DIFFER (exactness gate)"));
}

#[test]
fn merged_screens_keep_their_bands_ids_and_names() {
    let mut a = screen(r#"[{"cell":"ON C1","unit":"tok/s","a":[100,100],"x":[99.05,99.05]}]"#);
    let b = a.clone(); a.insert("band_pct", Value::Float(1.0));
    let (text, code) = judge(&[a.clone(), b.clone()], &options(&[])).unwrap();
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("SCREEN FAIL syn + syn"));
    assert!(text.contains("ON C1 1.000 % base (base 1, own 0.000, hist n/a)"));
    assert_eq!(judge(&[a.clone(), b.clone()], &options(&["--band", "1"])).unwrap().1, 0);
    a.insert("ids_equal", Value::Int(0));
    assert_eq!(judge(&[a, b], &options(&["--band", "1"])).unwrap().1, 1);
}

#[test]
fn advisory_and_only_match_full_names_or_suffixes() {
    let cells = r#"[{"cell":"ON C8","unit":"tok/s","a":[100],"x":[98]},{"cell":"OFF C8","unit":"tok/s","a":[100],"x":[98]}]"#;
    assert_eq!(evaluate(cells, &["--advisory", "OFF C8"]).1, 1);
    assert!(judge(&[screen(cells)], &options(&["--advisory", "C8"])).is_err());
    assert!(judge(&[screen(cells)], &options(&["--only", "absent"])).is_err());
    let (text, code) = evaluate(cells, &["--only", "OFF C8"]);
    assert_eq!(code, 1); assert!(!text.contains("ON C8"));
    let cells = r#"[{"cell":"OFF 8K/warm_replay","unit":"tok/s","a":[100],"x":[98]}]"#;
    for suffix in ["warm_replay", "8K/warm_replay", "OFF 8K/warm_replay"] {
        assert!(judge(&[screen(cells)], &options(&["--advisory", suffix])).is_err());
    }
    assert_eq!(evaluate(cells, &["--advisory", "OFF 8K"]).1, 1);
    assert!(judge(&[screen(cells)], &options(&["--only", "warm_replay"])).is_err());
}

#[test]
fn advisory_suffixes_need_a_name_boundary() {
    for (cell, suffix, advisory) in [("OFF 8K/warm_replay", "warm_replay", true),
        ("OFF 8K/warm_replay", "8K/warm_replay", true), ("OFF C1", "OFF C1", true),
        ("OFF C1", "C1", true), ("OFF C21", "1", false), ("OFF C1", "1", false),
        ("ON prose", "e", false), ("ON code", "e", false), ("ON structured", "e", false)] {
        let c = object(vec![("cell", cell.into()), ("unit", "tok/s".into()), ("a", 100i64.into()), ("x", 98i64.into())]);
        assert_eq!(row(&c, BASE, &options(&["--advisory", suffix])).unwrap().advisory, advisory, "{cell}: {suffix}");
    }
}

#[test]
fn no_gating_cells_refuse_a_verdict_even_with_ids_differ() {
    let advisory = screen(r#"[{"cell":"OFF C1","unit":"tok/s","a":100,"x":98}]"#);
    let passing = screen(r#"[{"cell":"ON C1","unit":"tok/s","a":100,"x":100}]"#);
    for args in [&["--advisory", "C1"][..], &["--advisory", "C1", "--ids-ok", "no"],
        &["--advisory", "C1", "--only", "OFF C1"], &["--advisory", "C1", "--final"]] {
        assert!(judge(&[advisory.clone(), passing.clone()], &options(args)).is_err(), "{args:?}");
    }
    let (text, code) = judge(&[advisory, passing], &options(&["--advisory", "OFF C1"])).unwrap();
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("ADVISORY FAIL"), "{text}");
}

#[test]
fn duplicate_band_only_and_sha_flags_are_rejected() {
    let first_sha = "a".repeat(40);
    let second_sha = "b".repeat(40);
    for (flag, first, second) in [("--band", "1", "2"), ("--only", "rate", "other"),
        ("--sha", first_sha.as_str(), second_sha.as_str())] {
        for second in [first, second] {
            let args = strings(&[flag, first, "cells.json", flag, second]);
            assert_eq!(parse(&args).err().unwrap(), format!("Give {flag} only once."));
        }
    }
    assert_eq!(options(&["--advisory", "one", "--advisory", "two"]).advisory, strings(&["one", "two"]));
}

#[test]
fn malformed_aa_pct_is_rejected_for_every_arm_form() {
    for arm in [r#""a":[100]"#, r#""a":[100,100]"#, r#""a":100"#, r#""a":100,"a_starts":2"#] {
        let input = screen(&format!(r#"[{{"cell":"rate","unit":"tok/s",{arm},"x":[100,100]}}]"#));
        for bad in [Value::from("bad"), Value::Bool(false), Value::Array(vec![]),
            Value::Float(f64::INFINITY), Value::Float(f64::NAN)] {
            let mut c = input["cells"][0].clone(); c.insert("aa_pct", bad.clone());
            let mut s = input.clone(); s.insert("cells", Value::Array(vec![c]));
            for args in [&[][..], &["--band", "2"]] {
                assert!(judge(&[s.clone()], &options(args)).is_err(), "{arm}: {bad:?}: {args:?}");
            }
        }
    }
}

#[test]
fn malformed_band_on_clean_pct_is_rejected_without_pair_evidence() {
    let input = screen(r#"[{"cell":"rate","unit":"tok/s","a":[100,100],"x":[100,100]}]"#);
    for bad in [Value::from("bad"), Value::Bool(false), Value::Array(vec![]),
        Value::Float(-1.0), Value::Float(f64::INFINITY), Value::Float(f64::NAN)] {
        let mut c = input["cells"][0].clone(); c.insert("band_on_clean_pct", bad.clone());
        let mut s = input.clone(); s.insert("cells", Value::Array(vec![c]));
        for args in [&[][..], &["--band", "2"]] {
            assert!(judge(&[s.clone()], &options(args)).is_err(), "{bad:?}: {args:?}");
        }
    }
}

#[test]
fn pair_bands_need_every_on_cell_inside_even_when_advisory() {
    let off = r#"{"cell":"OFF C1","unit":"tok/s","a":[40,40],"x":[39.62,39.62],"band_on_clean_pct":1}"#;
    let inside = r#"{"cell":"ON C1","unit":"tok/s","a":[45,45],"x":[45,45]}"#;
    let floor = r#"{"cell":"ON 8K/warm_replay","unit":"s","a":[0.11,0.11],"x":[0.1122,0.1122]}"#;
    let (text, code) = evaluate(&format!("[{inside},{off}]"), &[]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("OFF C1 1.000 % base"), "{text}");
    let (text, code) = evaluate(&format!("[{inside},{floor},{off}]"), &["--advisory", "8K/warm_replay"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("OFF C1 0.800 % base"), "{text}");
    assert!(text.contains("ADVISORY FLOOR"), "{text}");
    for (cells, args) in [(format!("[{off}]"), &[][..]),
        (format!("[{inside},{off}]"), &["--only", "OFF C1"]), (format!("[{off},{off}]"), &[])] {
        let (text, code) = evaluate(&cells, args);
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("OFF C1 0.800 % base"), "{text}");
        assert!(!text.contains("OFF C1 1.000 % base"), "{text}");
    }
}

#[test]
fn cached_name_and_latency_limits_are_exact() {
    for (cell, unit, a, x, read) in [
        ("WARM REPLAY", "s", 0.5, 0.51, "FLOOR"),
        ("CACHE", "s", 0.5, 0.510001, "FAIL"),
        ("cache", "s", 1.0, 1.01, "SECOND-START"),
        ("warm_replay nocache", "s", 0.5, 0.505, "SECOND-START"),
        ("no-cache", "s", 0.5, 0.505, "SECOND-START"),
        ("uncached", "s", 0.5, 0.505, "SECOND-START"),
        ("warm_replay NO-CACHE", "s", 0.5, 0.51, "FAIL"),
        ("UNCACHED warm_replay", "s", 0.5, 0.51, "FAIL"),
        ("cache", "tok/s", 100.0, 98.8, "SECOND-START"),
        ("cache", "s", 0.5, 0.499, "INSIDE (X faster)"),
    ] {
        let c = object(vec![("cell", cell.into()), ("unit", unit.into()), ("a", Value::Array(vec![Value::Float(a)])), ("x", Value::Array(vec![Value::Float(x)]))]);
        assert_eq!(row(&c, BASE, &options(&[])).unwrap().read, read, "{cell}");
    }
}

#[test]
fn bad_input_never_becomes_a_pass() {
    for cells in ["[]", r#"[{"cell":"lat","a":[20,20],"x":[25,25]}]"#,
        r#"[{"cell":"lat","unit":"ms","a":20,"x":25}]"#,
        r#"[{"cell":"lat","unit":"s","a":[],"x":[1]}]"#,
        r#"[{"cell":"lat","unit":"s","a":[0],"x":[1]}]"#,
        r#"[{"cell":"lat","unit":"s","a":[-1,3],"x":[1]}]"#,
        r#"[{"cell":"lat","unit":"s","a":1,"x":1,"a_starts":0}]"#,
        r#"[{"cell":"lat","unit":"s","a":1,"x":1,"band_pct":-1}]"#,
        r#"[{"cell":"lat\nFAIL","unit":"s","a":1,"x":1}]"#,
        r#"[{"cell":"lat","unit":"s","a":1e-300,"x":1e300}]"#] {
        assert!(judge(&[screen(cells)], &options(&[])).is_err(), "{cells}");
    }
    assert!(judge(&[], &options(&[])).is_err());
    assert!(judge(&[screen("[]"), screen("[]")], &options(&[])).is_err());
    for args in [&[][..], &["--rule", "split", "file"], &["--band", "NaN", "file"], &["--band", "-1", "file"],
        &["--sha", "abc", "file"], &["--ids-ok", "maybe", "file"], &["--bogus", "file"], &["--only"], &["--only", "--final", "file"]] {
        assert!(parse(&strings(args)).is_err(), "{args:?}");
    }
}

#[test]
fn optional_null_bands_are_absent_and_overrides_still_validate_input() {
    let cell = r#"[{"cell":"rate","unit":"tok/s","a":100,"a_starts":2,"x":98.9,"x_starts":2}]"#;
    let mut inherited = screen(cell); inherited.insert("band_pct", Value::Float(1.2));
    let mut null_cell = inherited.clone();
    let mut c = null_cell["cells"][0].clone();
    for key in ["band_pct", "aa_pct", "band_on_clean_pct"] { c.insert(key, Value::Null); }
    null_cell.insert("cells", Value::Array(vec![c]));
    let absent = screen(cell);
    let mut null_screen = absent.clone(); null_screen.insert("band_pct", Value::Null);
    for args in [&[][..], &["--band", "1.4"]] {
        let o = options(args);
        assert_eq!(judge(&[inherited.clone()], &o).unwrap(), judge(&[null_cell.clone()], &o).unwrap());
        assert_eq!(judge(&[absent.clone()], &o).unwrap(), judge(&[null_screen.clone()], &o).unwrap());
        for bad in [Value::Float(-1.0), Value::from("bad"), Value::Bool(false)] {
            let mut s = absent.clone(); s.insert("band_pct", bad.clone());
            assert!(judge(&[s], &o).is_err(), "{args:?}: {bad:?}");
            let mut c = absent["cells"][0].clone(); c.insert("band_pct", bad.clone());
            let mut s = absent.clone(); s.insert("cells", Value::Array(vec![c]));
            assert!(judge(&[s], &o).is_err(), "{args:?}: {bad:?}");
        }
    }
}

#[test]
fn help_is_parsed_as_an_option_and_output_errors_return_usage_status() {
    assert!(parse(&strings(&["--help"])).unwrap().help);
    assert!(parse(&strings(&["-h"])).unwrap().help);
    let o = parse(&strings(&["--only", "-h", "--", "--help", "-h"])).unwrap();
    assert!(!o.help);
    assert_eq!(o.only.as_deref(), Some("-h"));
    assert_eq!(o.files, strings(&["--help", "-h"]));
    assert!(parse(&strings(&["--only", "--help"])).is_err());
    assert!(parse(&strings(&["--bogus", "--help"])).is_err());
    use crate::output::{self, testing::Capture};
    use std::io::ErrorKind;
    for (kind, code) in [(ErrorKind::BrokenPipe, 0), (ErrorKind::PermissionDenied, 2), (ErrorKind::StorageFull, 2)] {
        for flush in [false, true] {
            let writer = Capture { fail_on: (!flush).then_some(("Judge saved", kind)), flush_error: flush.then_some(kind), ..Capture::default() };
            output::with_writer(writer, || assert_eq!(output::exit_code(run(&strings(&["--help"]))), code));
        }
    }
}

#[test]
fn report_keeps_six_significant_digits_and_decode_means() {
    for (n, expected) in [(0.103941, "0.103941"), (5.953397, "5.9534"), (100.0, "100"), (0.00001, "1e-05"), (1000000.0, "1e+06")] {
        assert_eq!(short(n), expected);
    }
    let (text, code) = evaluate(r#"[{"cell":"ON prose","unit":"tok/s","a":100,"x":100},{"cell":"ON code","unit":"tok/s","a":100,"x":101},{"cell":"ON structured","unit":"tok/s","a":100,"x":102}]"#, &[]);
    assert_eq!(code, 0);
    assert!(text.contains("DECODE-MEAN ON | prose +0.000 %, code +1.000 %, structured +2.000 % -> mean +1.000 % | advisory, not gating"), "{text}");
}
