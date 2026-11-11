//! Run the saved-cell judge without an installation or a server.
use std::{fs, path::PathBuf, process::Command, sync::atomic::{AtomicUsize, Ordering}};
use vade_harness::json;

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("vade-verdict-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&root).unwrap(); Self(root)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command.args(["bench", "verdict"]).current_dir(&self.0).env_clear()
            .env("VADE_CONFIG", self.0.join("missing-config"))
            .env("VADE_HOME", self.0.join("missing-install"))
            .env("VADE_API_KEY_FILE", self.0.join("missing-credential"))
            .env("VADE_API_URL", "invalid://offline")
            .env("SCREEN_RULE", "split").env("SCREEN_ADVISORY_CELLS", "C1");
        command
    }
    fn check(&self, args: &[&str], code: i32) -> String {
        let output = self.command().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(code), "{args:?}: {output:?}");
        if code == 2 {
            assert!(output.stdout.is_empty() && !output.stderr.is_empty(), "{args:?}: {output:?}");
        } else { assert!(output.stderr.is_empty(), "{args:?}: {output:?}"); }
        String::from_utf8(output.stdout).unwrap()
    }
    fn refusal(&self, args: &[&str]) -> String {
        let output = self.command().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty() && !output.stderr.is_empty(), "{args:?}: {output:?}");
        String::from_utf8(output.stderr).unwrap()
    }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

#[test]
fn script_selftest_inputs_give_the_same_cli_verdicts() {
    let fixture = Fixture::new();
    let cases = json::parse(include_str!("fixtures/verdict-1101.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let path = fixture.0.join("cells.json");
        fs::write(&path, json::stringify(case.get("input").unwrap()).unwrap()).unwrap();
        for explicit in [false, true] {
            let mut command = fixture.command();
            if explicit { command.args(["--rule", "2026-11-01", "--no-record"]); }
            let output = command.arg(&path).output().unwrap();
            let text = String::from_utf8_lossy(&output.stdout);
            let verdict = case.get("verdict").unwrap().as_str().unwrap();
            assert_eq!(output.status.code(), Some(match verdict { "SCREEN PASS" => 0, "SCREEN FAIL" => 1, "SECOND-START" => 3, _ => panic!("unknown fixture verdict") }), "{case:?}: {text}");
            assert!(output.stderr.is_empty(), "{output:?}");
            assert!(text.lines().last().unwrap().starts_with(&format!("{verdict} syn | rule 2026-11-01")), "{text}");
        }
    }
    let entries = fs::read_dir(&fixture.0).unwrap().count();
    assert_eq!(entries, 1, "only the input file exists");
}

#[test]
fn verdict_help_names_the_rule_without_reading_setup() {
    let fixture = Fixture::new();
    for flag in ["--help", "-h"] {
        let output = fixture.command().arg(flag).output().unwrap();
        assert!(output.status.success() && output.stderr.is_empty(), "{output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        for expected in ["Usage: vade bench verdict", "Rule 2026-11-01", "0.8 %", "+10 ms", "SECOND-START", "No third X", "no saved station history",
            "2 bad input, no gating cells, or output error", "--band, --only and --sha each accept one value"] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
    }
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 0);
}

#[test]
fn bad_files_and_arguments_return_usage_status() {
    let fixture = Fixture::new();
    let path = fixture.0.join("cells.json");
    for input in ["not JSON", "[]", "{}", r#"{"cells":[{"cell":"lat","a":[20,20],"x":[25,25]}]}"#] {
        fs::write(&path, input).unwrap();
        let output = fixture.command().arg(&path).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        assert!(output.stdout.is_empty());
    }
    for args in [&[][..], &["missing.json"], &["--rule", "2026-10-31", "missing.json"], &["--band", "NaN", "missing.json"]] {
        let output = fixture.command().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn rerun_and_final_exit_codes_match_the_script() {
    let fixture = Fixture::new();
    let path = fixture.0.join("cells.json");
    fs::write(&path, r#"{"screen":"syn","cells":[{"cell":"ON a","unit":"tok/s","a":[100,101],"x":[99,99.4]}]}"#).unwrap();
    for (args, verdict, code) in [(&[][..], "RERUN", 4), (&["--final", "--only", "ON a"], "SCREEN FAIL", 1), (&["--ids-ok", "no"], "SCREEN FAIL", 1)] {
        let output = fixture.command().args(args).arg(&path).output().unwrap();
        assert_eq!(output.status.code(), Some(code), "{output:?}");
        assert!(String::from_utf8(output.stdout).unwrap().lines().last().unwrap().starts_with(verdict));
    }
    let output = fixture.command().args(["--only", "absent"]).arg(&path).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn help_values_and_dash_named_files_do_not_skip_the_judge() {
    let fixture = Fixture::new();
    let input = r#"{"screen":"dash","cells":[{"cell":"-h","unit":"tok/s","a":100,"x":98}]}"#;
    for file in ["-h", "--help", "-cells", "cells.json"] { fs::write(fixture.0.join(file), input).unwrap(); }
    for args in [&["--", "-h"][..], &["--", "--help"], &["--", "-cells"], &["./-h"], &["./--help"],
        &["--only", "-h", "cells.json"]] {
        let text = fixture.check(args, 1);
        assert!(text.lines().last().unwrap().starts_with("SCREEN FAIL dash"), "{text}");
        assert!(!text.contains("Usage:"), "{text}");
    }
    for args in [&["--only", "--help", "cells.json"][..], &["--band", "-h", "cells.json"],
        &["--bogus", "--help"], &["--", "-missing"],
        &["--advisory", "-h", "--only", "-h", "cells.json", "--ids-ok", "no"]] { fixture.check(args, 2); }
}

#[test]
fn multiple_files_keep_bands_names_and_ids_and_need_a_judged_cell() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("a.json"), r#"{"screen":"a","band_pct":1.2,"cells":[{"cell":"a rate","unit":"tok/s","a":[100,100],"x":[99,99]}]}"#).unwrap();
    fs::write(fixture.0.join("b.json"), r#"{"screen":"b","band_pct":null,"cells":[{"cell":"b rate","unit":"tok/s","a":[100,100],"x":[99,99]}]}"#).unwrap();
    let text = fixture.check(&["a.json", "b.json"], 1);
    assert!(text.contains("SCREEN FAIL a + b"), "{text}");
    assert!(text.contains("a rate 1.200 % base") && text.contains("b rate 0.800 % base"), "{text}");
    fixture.check(&["a.json", "b.json", "--band", "1.2"], 0);
    fs::write(fixture.0.join("ids.json"), r#"{"ids_equal":false,"cells":[]}"#).unwrap();
    let text = fixture.check(&["a.json", "ids.json"], 1);
    assert!(text.contains("IDS DIFFER"), "{text}");
    for file in ["empty-a.json", "empty-b.json"] {
        fs::write(fixture.0.join(file), r#"{"screen":"empty","cells":[]}"#).unwrap();
    }
    for args in [&["empty-a.json"][..], &["empty-a.json", "empty-b.json"],
        &["empty-a.json", "empty-b.json", "--only", "absent"], &["a.json", "b.json", "--only", "absent"]] {
        fixture.check(args, 2);
    }
    fixture.check(&["empty-a.json", "a.json"], 0);
}

#[test]
fn null_bands_are_absent_and_bad_bands_fail_even_with_an_override() {
    let fixture = Fixture::new();
    for (input, code) in [
        (r#"{"band_pct":null,"cells":[{"cell":"rate","unit":"tok/s","a":100,"x":100,"band_pct":null}]}"#, 0),
        (r#"{"band_pct":-1,"cells":[{"cell":"rate","unit":"tok/s","a":100,"x":100}]}"#, 2),
        (r#"{"cells":[{"cell":"rate","unit":"tok/s","a":100,"x":100,"band_pct":-1}]}"#, 2),
        (r#"{"band_pct":"bad","cells":[{"cell":"rate","unit":"tok/s","a":100,"x":100}]}"#, 2),
        (r#"{"cells":[{"cell":"rate","unit":"tok/s","a":100,"x":100,"band_pct":"bad"}]}"#, 2),
    ] {
        fs::write(fixture.0.join("cells.json"), input).unwrap();
        fixture.check(&["cells.json"], code);
        fixture.check(&["--band", "1", "cells.json"], code);
    }
}

#[test]
fn pair_widening_needs_a_judged_cell_without_widening() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("off.json"), r#"{"screen":"off","cells":[{"cell":"OFF rate","unit":"tok/s","a":[100,100],"x":[99,99],"band_on_clean_pct":1.2}]}"#).unwrap();
    fs::write(fixture.0.join("on.json"), r#"{"screen":"on","cells":[{"cell":"ON rate","unit":"tok/s","a":[100,100],"x":[100,100]}]}"#).unwrap();
    for args in [&["off.json"][..], &["on.json", "off.json", "--only", "OFF rate"]] {
        let text = fixture.check(args, 1);
        assert!(text.contains("OFF rate 0.800 % base"), "{text}");
    }
    let text = fixture.check(&["on.json", "off.json"], 0);
    assert!(text.contains("OFF rate 1.200 % base"), "{text}");
    let text = fixture.check(&["on.json", "off.json", "--band", "0.5"], 0);
    assert!(text.contains("OFF rate 1.200 % base") && text.contains("ON rate 0.500 % base"), "{text}");
}

#[test]
fn advisory_suffixes_and_cache_exclusions_change_the_cli_verdict() {
    let fixture = Fixture::new();
    for cell in ["no-cache", "uncached", "warm_replay no-cache", "UNCACHED warm_replay"] {
        let input = format!(r#"{{"cells":[{{"cell":"{cell}","unit":"s","a":[0.5],"x":[0.51]}}]}}"#);
        fs::write(fixture.0.join("cells.json"), input).unwrap();
        let text = fixture.check(&["cells.json"], 1);
        assert!(text.contains("| FAIL;") && !text.contains("FLOOR"), "{text}");
    }
    fs::write(fixture.0.join("cells.json"), r#"{"cells":[{"cell":"OFF 8K/warm_replay","unit":"s","a":[2],"x":[2.1]}]}"#).unwrap();
    fixture.check(&["cells.json"], 1);
    assert!(fixture.refusal(&["--advisory", "warm_replay", "cells.json"]).contains("No gating cells"));
    fs::write(fixture.0.join("gate.json"), r#"{"cells":[{"cell":"ON rate","unit":"tok/s","a":100,"x":100}]}"#).unwrap();
    let text = fixture.check(&["--advisory", "warm_replay", "cells.json", "gate.json"], 0);
    assert!(text.contains("ADVISORY FAIL"), "{text}");
}

#[test]
fn no_gating_cells_refuse_a_cli_verdict() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("cells.json"), r#"{"cells":[{"cell":"OFF 8K/warm_replay","unit":"tok/s","a":100,"x":98}]}"#).unwrap();
    fs::write(fixture.0.join("gate.json"), r#"{"cells":[{"cell":"ON rate","unit":"tok/s","a":100,"x":100}]}"#).unwrap();
    fs::write(fixture.0.join("empty.json"), r#"{"cells":[]}"#).unwrap();
    for args in [&["--advisory", "warm_replay", "cells.json"][..],
        &["--advisory", "warm_replay", "cells.json", "empty.json"],
        &["--advisory", "warm_replay", "--only", "OFF 8K/warm_replay", "cells.json", "gate.json"],
        &["--advisory", "warm_replay", "--advisory", "rate", "cells.json", "gate.json"],
        &["--advisory", "warm_replay", "--ids-ok", "no", "cells.json"]] {
        let error = fixture.refusal(args);
        assert!(error.contains("No gating cells"), "{args:?}: {error}");
    }
    for args in [&["empty.json"][..], &["--only", "absent", "cells.json", "gate.json"]] {
        assert!(fixture.refusal(args).contains("cell"), "{args:?}");
    }
}

#[test]
fn duplicate_band_only_and_sha_flags_refuse_a_cli_verdict() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("cells.json"), r#"{"cells":[{"cell":"rate","unit":"tok/s","a":100,"x":98}]}"#).unwrap();
    let first_sha = "a".repeat(40);
    let second_sha = "b".repeat(40);
    for (flag, first, second) in [("--band", "0.8", "3"), ("--only", "rate", "absent"),
        ("--sha", first_sha.as_str(), second_sha.as_str())] {
        for second in [first, second] {
            let error = fixture.refusal(&[flag, first, "cells.json", flag, second]);
            assert!(error.contains(&format!("Give {flag} only once.")), "{error}");
        }
    }
}

#[test]
fn malformed_aa_pct_refuses_a_cli_verdict_for_every_arm_form() {
    let fixture = Fixture::new();
    for arm in [r#""a":[100]"#, r#""a":[100,100]"#, r#""a":100"#, r#""a":100,"a_starts":2"#] {
        for bad in [r#""bad""#, "false", "[]", "{}"] {
            let input = format!(r#"{{"cells":[{{"cell":"rate","unit":"tok/s",{arm},"x":[100,100],"aa_pct":{bad}}}]}}"#);
            fs::write(fixture.0.join("cells.json"), input).unwrap();
            for args in [&["cells.json"][..], &["--band", "2", "cells.json"]] { fixture.refusal(args); }
        }
    }
}

#[test]
fn malformed_band_on_clean_pct_refuses_a_cli_verdict() {
    let fixture = Fixture::new();
    for bad in [r#""bad""#, "false", "[]", "{}", "-1"] {
        let input = format!(r#"{{"cells":[{{"cell":"rate","unit":"tok/s","a":100,"x":100,"band_on_clean_pct":{bad}}}]}}"#);
        fs::write(fixture.0.join("cells.json"), input).unwrap();
        for args in [&["cells.json"][..], &["--band", "2", "cells.json"]] { fixture.refusal(args); }
    }
}

#[test]
fn advisory_suffixes_keep_partial_names_in_the_cli_verdict() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("cells.json"), r#"{"cells":[{"cell":"ON prose","unit":"tok/s","a":100,"x":98},{"cell":"ON code","unit":"tok/s","a":100,"x":98},{"cell":"ON structured","unit":"tok/s","a":100,"x":98},{"cell":"OFF C1","unit":"tok/s","a":100,"x":98},{"cell":"OFF C21","unit":"tok/s","a":100,"x":98}]}"#).unwrap();
    let text = fixture.check(&["--advisory", "e", "--advisory", "1", "cells.json"], 1);
    assert!(!text.contains("ADVISORY"), "{text}");
}
