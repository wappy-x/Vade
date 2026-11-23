//! A checkpoint whose text service runs on one Spark starts on one Spark.
//! The plan in config.json is the signal. The folder name is not.
use super::*;
use std::fs;

const LOOP: &str = crate::text_service::LOOP_CONFIG;
const OTHER: &str = r#"{"model_type":"other","architectures":["Inkling2LoopForConditionalGeneration"]}"#;

fn with_config(count: usize, folder: &str, config: Option<&str>) -> Fixture {
    let mut f = Fixture::fp4(count, false);
    let root = f.dir.join(folder);
    fs::create_dir_all(&root).unwrap();
    if let Some(config) = config {
        fs::write(root.join("config.json"), config).unwrap();
    }
    let path = fs::canonicalize(&root).unwrap();
    for machine in &mut f.app.config.machines {
        machine.model_path = path.to_string_lossy().into_owned();
    }
    f
}

fn started(f: &Fixture) -> bool {
    f.app.runner.calls.iter().any(|call| call.label.ends_with(" start") && call.label != "controller tunnel start")
}

fn depth(args: &[String], value: &str) -> bool {
    args.windows(2).any(|pair| pair[0] == "--mtp" && pair[1] == value)
}

#[test]
fn a_loop_checkpoint_starts_on_one_spark() {
    let mut f = with_config(1, "checkpoint", Some(LOOP));
    let manifest = f.app.manifest().unwrap();
    let direct = Session::new(&f.app.config, &fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"]), &manifest, "a".repeat(64)).unwrap();
    assert_eq!((direct.group_len(), direct.topology(), direct.two_sparks()), (1, "path", false));
    assert_eq!((direct.mtp, direct.mtp_explicit, direct.context), (0, true, 32768));
    let args = direct.engine_args(0);
    assert_eq!(&args[..4], &[String::from("0"), String::from("1"), String::from("127.0.0.1:0"), String::from("127.0.0.1:0")]);
    assert!(args.iter().any(|arg| arg == "--diagnostic-unqualified-text") && depth(&args, "0"));
    assert!(!args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse"));
    assert!(!direct.ranks[0].machine.model_path.contains("inkling"));
    f.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    let saved_args = saved.engine_args(0);
    assert_eq!(&saved_args[..4], &[String::from("0"), String::from("1"), String::from("127.0.0.1:0"), String::from("127.0.0.1:0")]);
    assert!(saved_args.iter().any(|arg| arg == "--diagnostic-unqualified-text") && depth(&saved_args, "0"));
    assert!(!saved_args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse"));
    assert_eq!((saved.group_len(), saved.mtp, saved.topology()), (1, 0, "path"));
    assert!(!f.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
}

#[test]
fn a_loop_checkpoint_on_any_other_group_is_refused() {
    let cases: [(usize, &[&str]); 2] = [
        (2, &["--sparks", "2", "--diagnostic-unqualified-text"]),
        (4, &["--replicas", "2", "--pairs", "--diagnostic-unqualified-text"]),
    ];
    // A group that is neither one nor two Sparks reads no plan: main's refusal, also for a looped checkpoint.
    let mut four = with_config(4, "checkpoint", Some(LOOP));
    assert_eq!(four.app.up(&fp4_options(&["--sparks", "4", "--diagnostic-unqualified-text"])).unwrap_err(),
        "--diagnostic-unqualified-text runs on exactly two Sparks. Start it as: vade up --sparks 2 --diagnostic-unqualified-text.");
    assert!(!started(&four) && !four.recorded());
    for (count, args) in cases {
        let mut f = with_config(count, "checkpoint", Some(LOOP));
        let err = f.app.up(&fp4_options(args)).unwrap_err();
        assert!(err.contains("no transport between Sparks") && err.contains("RANK 0 of WORLD 1"), "{args:?} {err}");
        assert!(err.contains("vade up --sparks 1 --diagnostic-unqualified-text") && !err.contains("exactly two Sparks"), "{args:?} {err}");
        assert!(!started(&f) && !f.recorded(), "{args:?}");
    }
}

#[test]
fn loop_replicas_are_one_spark_each() {
    let mut f = with_config(2, "checkpoint", Some(LOOP));
    f.app.up(&fp4_options(&["--replicas", "2", "--diagnostic-unqualified-text"])).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.group_len(), session.ranks.len(), session.replica_sparks), (1, 2, 1));
    for index in 0..2 {
        let args = session.engine_args(index);
        assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
        assert_eq!((&*session.ranks[index].listen, &*session.ranks[index].next), ("127.0.0.1:0", "127.0.0.1:0"));
        assert!(depth(&args, "0") && !args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse"));
    }
}

#[test]
fn a_current_family_checkpoint_on_one_spark_stays_refused() {
    let mut remote = Fixture::fp4(1, false);
    let err = remote.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    // Main's refusal, byte for byte (REVSHARED finding 2).
    assert_eq!(err, "--diagnostic-unqualified-text runs on exactly two Sparks. Start it as: vade up --sparks 2 --diagnostic-unqualified-text.");
    assert!(!started(&remote) && !remote.recorded());
    assert!(remote.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let mut named = with_config(1, "inkling2-loop-checkpoint", Some(OTHER));
    let err = named.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("exactly two Sparks") && err.contains("vade up --sparks 2 --diagnostic-unqualified-text"), "{err}");
    assert!(!started(&named) && !named.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let mut pair = with_config(2, "inkling2-loop-checkpoint", Some(OTHER));
    pair.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    assert!(!pair.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let session = Session::load(&pair.app.config.state_dir).unwrap().unwrap();
    assert_eq!(session.mtp, crate::state::TEXT_DRAFT_DEPTH);
    assert!(depth(&session.engine_args(0), "3") && depth(&session.engine_args(1), "3"));
}

#[test]
fn a_named_draft_depth_is_refused_when_the_plan_has_no_draft() {
    for depth_arg in ["1", "3", "7"] {
        let mut f = with_config(1, "checkpoint", Some(LOOP));
        let err = f.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text", "--mtp", depth_arg])).unwrap_err();
        assert!(err.contains("Drafting is not built") && err.contains("--mtp 0") && err.contains("leave --mtp out"), "{err}");
        assert!(!started(&f) && !f.recorded());
    }
    let mut off = with_config(1, "checkpoint", Some(LOOP));
    off.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text", "--mtp", "0"])).unwrap();
    assert_eq!(Session::load(&off.app.config.state_dir).unwrap().unwrap().mtp, 0);
}

#[test]
fn a_looped_local_plan_on_two_sparks_does_not_start() {
    let mut f = Fixture::fp4(2, false);
    for (index, (folder, config)) in [("one", LOOP), ("two", OTHER)].into_iter().enumerate() {
        let root = f.dir.join(folder);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), config).unwrap();
        f.app.config.machines[index].model_path = fs::canonicalize(&root).unwrap().to_string_lossy().into_owned();
    }
    let err = f.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap_err();
    // Two Sparks read no plan, as on main, except a looped config on this computer, which refuses the pair.
    assert!(err.contains("no transport between Sparks") && !started(&f) && !f.recorded(), "{err}");
}

struct Reports {
    inner: FakeRunner,
    stdout: String,
    success: bool,
}
impl Runner for Reports {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = if matches!(exec.program.as_str(), "ssh" | "sh") && exec.args.last().is_some_and(|arg| arg.contains("__worker")) {
            text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default()
        } else {
            Fields::new()
        };
        if request.get("op").map(String::as_str) == Some("text-service") {
            self.inner.calls.push(exec.clone());
            return Ok(Output { success: self.success, stdout: self.stdout.clone() });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: std::time::Duration) { self.inner.pause(duration); }
}

fn take(f: &mut Fixture, stdout: &str, success: bool) -> App<Reports> {
    App { config: f.app.config.clone(), dry: false, runner: Reports { inner: std::mem::take(&mut f.app.runner), stdout: stdout.into(), success } }
}

#[test]
fn a_spark_report_sets_the_width_and_an_old_worker_stays_a_pair() {
    let alone = "text_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n";
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, alone, true);
    one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&one.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.mtp), (1, 0));
    assert!(depth(&saved.engine_args(0), "0"));
    let asked = one.runner.inner.calls.iter().find(|call| call.label.ends_with(" text service")).unwrap();
    let request = text::parse(&String::from_utf8_lossy(&asked.stdin)).unwrap();
    assert_eq!(request.get("op").map(String::as_str), Some("text-service"));
    assert!(request.get("model_path").is_some_and(|path| path.starts_with("~/")) && !request.contains_key("text_sparks"));
    let mut two_f = Fixture::fp4(2, false);
    let mut two = take(&mut two_f, alone, true);
    // Two Sparks take main's steps: the Spark is not asked; the engine refuses a looped checkpoint there.
    two.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    assert!(!two.runner.inner.calls.iter().any(|call| call.label.ends_with(" text service")));
    assert_eq!(Session::load(&two.config.state_dir).unwrap().unwrap().mtp, crate::state::TEXT_DRAFT_DEPTH);
    let mut old_f = Fixture::fp4(2, false);
    let mut old = take(&mut old_f, "phase=running\ngpu=empty\n", true);
    old.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    assert_eq!(Session::load(&old.config.state_dir).unwrap().unwrap().mtp, crate::state::TEXT_DRAFT_DEPTH);
    let mut legacy_f = Fixture::fp4(1, false);
    let mut legacy = take(&mut legacy_f, "error=Unknown machine operation. Reinstall vade.\n", false);
    let err = legacy.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("exactly two Sparks") && !legacy.runner.inner.calls.iter().any(|call| call.label.ends_with(" start")), "{err}");
    let mut broken_f = Fixture::fp4(1, false);
    let mut broken = take(&mut broken_f, "error=Cannot read the checkpoint config. missing\n", false);
    let err = broken.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("Cannot read the checkpoint config") && err.contains("Nothing started") && !broken.runner.inner.calls.iter().any(|call| call.label.ends_with(" start")), "{err}");
}

/// The helper's hash command is recorded before the text-service question.
fn verified_before_ask(calls: &[Exec]) {
    let mut asked = 0;
    for (index, call) in calls.iter().enumerate() {
        if !call.label.ends_with(" text service") {
            continue;
        }
        asked += 1;
        let name = call.label.split_whitespace().next().unwrap();
        let hash = format!("{name} artifact hashes");
        assert!(calls[..index].iter().any(|earlier| earlier.label == hash), "{hash} is missing before {}", call.label);
    }
    assert!(asked > 0, "the checkpoint was not asked");
}

fn start_args(calls: &[Exec]) -> Vec<String> {
    let call = calls.iter().find(|call| call.label.ends_with(" start") && call.label != "controller tunnel start").unwrap();
    let fields = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
    (0..).map(|index| fields.get(&format!("arg.{index}")).cloned()).take_while(Option::is_some).flatten().collect()
}

fn dry_reports(f: &mut Fixture, stdout: &str) -> App<Reports> {
    App { config: f.app.config.clone(), dry: true, runner: Reports { inner: std::mem::take(&mut f.app.runner), stdout: stdout.into(), success: true } }
}

#[test]
fn the_helper_is_verified_before_it_is_asked() {
    let mut broken = Fixture::fp4(1, false);
    broken.app.runner.bad_hash = true;
    let err = broken.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("staged hashes differ") && !err.contains("exactly two Sparks"), "{err}");
    assert!(broken.app.runner.calls.iter().any(|call| call.label.ends_with(" artifact hashes")));
    assert!(!broken.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")) && !started(&broken));
    let alone = "text_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n";
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, alone, true);
    one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    verified_before_ask(&one.runner.inner.calls);
}

/// `SystemRunner` in dry mode returns success and empty output. That reply is not a width.
#[test]
fn a_dry_run_with_an_empty_reply_does_not_guess_the_width() {
    let mut one_f = Fixture::fp4(1, true);
    let mut one = dry_reports(&mut one_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("does not know the text-service width") && printed.contains("not refused"), "{printed}");
    assert!(printed.contains("Start planned") && printed.contains("1 Spark"), "{printed}");
    assert!(!printed.contains("exactly two Sparks"), "{printed}");
    assert!(printed.contains("The drafting depth is not known in this dry run."), "{printed}");
    assert!(!printed.contains("engine default") && !printed.contains("MTP 5"), "{printed}");
    let args = start_args(&one.runner.inner.calls);
    assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
    assert!(!args.iter().any(|arg| arg == "--fp4-local"), "{args:?}");
    assert_eq!(one.runner.inner.calls.iter().filter(|call| call.label == "a artifact hashes").count(), 1);
    verified_before_ask(&one.runner.inner.calls);
    assert!(!one.config.state_dir.join("session.conf").exists());
    let mut two_f = Fixture::fp4(2, true);
    let mut two = dry_reports(&mut two_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || two.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    // Two Sparks: main's dry run, its lines and its depth; the Spark is not asked.
    assert!(!printed.contains("does not know the text-service width") && !printed.contains("The drafting depth is not known in this dry run."), "{printed}");
    assert!(printed.contains("2 Sparks") && printed.contains("MTP 3"), "{printed}");
    assert!(!two.runner.inner.calls.iter().any(|call| call.label.ends_with(" text service")));
    assert!(start_args(&two.runner.inner.calls).windows(2).any(|pair| pair == ["--mtp", "3"]));
    for name in ["a", "b"] {
        assert_eq!(two.runner.inner.calls.iter().filter(|call| call.label == format!("{name} artifact hashes")).count(), 1, "{name}");
    }
    let mut named_f = Fixture::fp4(1, true);
    let mut named = dry_reports(&mut named_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || named.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text", "--mtp", "5"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("The drafting depth is not known in this dry run."), "{printed}");
    assert!(!printed.contains("MTP 5"), "{printed}");
}

/// A real start whose helper exits 0 with an empty reply stays the two-Spark wire.
/// `remote_text_service` treats that reply as unread only when `self.dry` is set.
#[test]
fn a_non_dry_empty_reply_stays_the_two_spark_wire() {
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, "", true);
    let err = one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("exactly two Sparks"), "{err}");
    assert!(!one.runner.inner.calls.iter().any(|call| call.label.ends_with(" start") && call.label != "controller tunnel start"), "{err}");
}

#[test]
fn a_dry_run_of_a_local_loop_plan_still_reads_the_file() {
    let mut one = with_config(1, "checkpoint", Some(LOOP));
    one.app.dry = true;
    one.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    assert!(!one.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let args = start_args(&one.app.runner.calls);
    assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
    assert!(depth(&args, "0"), "{args:?}");
    let mut two = with_config(2, "checkpoint", Some(LOOP));
    two.app.dry = true;
    let err = two.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("no transport between Sparks"), "{err}");
}

#[test]
fn each_helper_hash_is_checked_once() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    for name in ["a", "b"] {
        let label = format!("{name} artifact hashes");
        assert_eq!(f.app.runner.calls.iter().filter(|call| call.label == label).count(), 1, "{label}");
    }
    // Two Sparks: no text-service question, as on main.
    assert!(!f.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
}

struct ByMachine { inner: FakeRunner }
impl Runner for ByMachine {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if exec.label.ends_with(" text service") {
            self.inner.calls.push(exec.clone());
            let stdout = if exec.label.starts_with("b ") {
                "text_sparks=2\ntext_draft_default=3\ntext_draft_max=7\n"
            } else {
                "text_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n"
            };
            return Ok(Output { success: true, stdout: stdout.into() });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: std::time::Duration) { self.inner.pause(duration); }
}

#[test]
fn each_machine_is_asked_and_a_disagreement_does_not_start() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: ByMachine { inner: std::mem::take(&mut f.app.runner) } };
    let err = app.up(&fp4_options(&["--replicas", "2", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("do not declare the same text service"), "{err}");
    assert!(!app.runner.inner.calls.iter().any(|call| call.label.ends_with(" start") && call.label != "controller tunnel start"), "{err}");
    verified_before_ask(&app.runner.inner.calls);
    assert!(app.runner.inner.calls.iter().any(|call| call.label == "a text service"));
    assert!(app.runner.inner.calls.iter().any(|call| call.label == "b text service"));
}

#[test]
fn a_loop_neighbor_log_names_one_spark() {
    // main prints `FAIL` plus the sentence from start.rs. The qualification sentence is not in that log.
    let neighbor = "FAIL this text service has no transport between Sparks: RANK 1 of WORLD 2 has 1 neighbor links; start it as RANK 0 of WORLD 1\n";
    let hint = health::text_refusal_hint(neighbor).unwrap();
    assert!(hint.contains("one Spark") && hint.contains("vade up --sparks 1 --diagnostic-unqualified-text") && !hint.contains("two Sparks"), "{hint}");
    let absent = "text service refused: this engine has not qualified this model family; it starts only as an unqualified diagnostic service on a loopback bind: give --diagnostic-unqualified-text\n";
    let hint = health::text_refusal_hint(absent).unwrap();
    assert!(hint.contains("two Sparks") && hint.contains("vade up --sparks 2 --diagnostic-unqualified-text") && !hint.contains("one Spark"), "{hint}");
    let glm = "text service refused: unsupported execution: complete resident decoder is absent.";
    let hint = health::text_refusal_hint(glm).unwrap();
    assert!(hint.contains("two Sparks") && hint.contains("vade up --sparks 2 --diagnostic-unqualified-text") && !hint.contains("one Spark"), "{hint}");
    assert!(health::text_refusal_hint("flashnext ready world=2").is_none());
}

#[test]
fn admitted_chats_reads_the_loop_ready_line() {
    // The ready line from loop_decoder/service/start.rs serve(). max_chats is LIVE_CHATS.
    let line = "text_service qualification=unqualified diagnostic_option=--diagnostic-unqualified-text ready bind=127.0.0.1:9 rank=0 world=1 path=loop_driver backend=device context=32768 output=8192 resident_bytes=4 session_bytes=5 session_pool_bytes=6 max_chats=1 drafting=off cache=false";
    assert_eq!(crate::state::admitted_chats(&format!("LOOP TEXT ADMISSION\nLOOP TEXT LOAD\n{line}\n")), Some(1));
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=4")), Some(4));
    assert_eq!(crate::state::admitted_chats("LOOP TEXT ADMISSION\nLOOP TEXT LOAD\n"), None);
    assert_eq!(crate::state::admitted_chats(&line.replace(" max_chats=1", "")), None);
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=0")), None);
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=33")), None);
    let flash = "flashnext ready world=1 model=m context=32768 concurrency=rows active=8 connections=16";
    assert_eq!(crate::state::admitted_chats(&format!("{line}\n{flash}\n")), Some(8));
    assert_eq!(crate::state::admitted_chats(&format!("{flash}\n{line}\n")), Some(1));
    assert_eq!(crate::state::admitted_chats("flash_serve ready slots=900"), None);
}

#[test]
fn admitted_chats_leaves_the_two_spark_text_ready_line_unread_as_main_does() {
    // The ready line from flash-serve/text.rs. max_chats=8 is written in that format string. Main records no
    // admitted count for it: session.conf has no admitted_chats and vade status --json shows null.
    let line = "text_service qualification=unqualified diagnostic_option=--diagnostic-unqualified-text ready bind=127.0.0.1:9 rank=0 world=2 path=tensor_split context=131072 resident_bytes=1 session_bytes=2 shared_workspace_bytes=3 session_pool_bytes=4 max_chats=8 default_draft=alone default_depth=3 predictor_depth=3 cache=true kept_cap_bytes=1 largest_kept_prompt_tokens=2";
    assert_eq!(crate::state::admitted_chats(line), None);
    assert_eq!(crate::state::admitted_chats(&format!("load\n{line}\n")), None);
}

#[test]
fn three_and_four_sparks_are_refused_with_main_text_also_in_a_dry_run() {
    // REVSHARED-2 item a: only a group of one Spark reads the plan; three or four Sparks with the typed option get
    // main's refusal, with no text-service question and no preview, dry or not.
    const MAIN: &str = "--diagnostic-unqualified-text runs on exactly two Sparks. Start it as: vade up --sparks 2 --diagnostic-unqualified-text.";
    for sparks in [3usize, 4] {
        let count = sparks.to_string();
        let args = ["--sparks", count.as_str(), "--diagnostic-unqualified-text"];
        let mut live = Fixture::fp4(sparks, false);
        assert_eq!(live.app.up(&fp4_options(&args)).unwrap_err(), MAIN, "{sparks} Sparks");
        assert!(!live.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")) && !started(&live) && !live.recorded());
        let mut dry_f = Fixture::fp4(sparks, true);
        let mut dry = dry_reports(&mut dry_f, "");
        let capture = crate::output::testing::Capture::default();
        let err = crate::output::with_writer(capture.clone(), || dry.up(&fp4_options(&args))).unwrap_err();
        assert_eq!(err, MAIN, "{sparks} Sparks, dry run");
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert!(!printed.contains("does not know the text-service width") && !printed.contains("Start planned"), "{printed}");
        assert!(!dry.runner.inner.calls.iter().any(|call| call.label.ends_with(" text service")));
    }
}
