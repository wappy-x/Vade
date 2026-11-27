//! Two FP4-Local Sparks run the tensor split, the only two-Spark path for FP4-Local: vade up sets
//! no level, carries no path switch and refuses TCP before any remote call; FP8 pairs keep their
//! ring and may use TCP. Status names the path and why for every group of Sparks on one model.
use super::*;

/// The worker start requests of one `vade up --sparks N`.
fn starts(f: &Fixture) -> Vec<Fields> {
    f.app.runner.calls.iter().filter(|c| c.label.ends_with(" start") && c.label != "controller tunnel start")
        .map(|start| text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap()).collect()
}

#[test]
fn up_with_two_sparks_runs_the_split_over_rdma_only() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_SPLIT";
    if let Ok(case) = std::env::var(CASE) {
        let mut f = Fixture::fp4(2, false);
        if case == "tcp" {
            // The refusal is local: no SSH, fabric, artifact, path or GPU check, no ownership record.
            for options in [fp4_options(&["--sparks", "2"]), fp4_options(&["--sparks", "2", "--mtp", "0"])] {
                let error = f.app.up(&options).unwrap_err();
                assert!(error.contains("needs RDMA") && error.contains("FP4-Local"), "{error}");
                assert!(f.app.runner.calls.is_empty(), "{:?}", f.app.runner.calls.iter().map(|c| &c.label).collect::<Vec<_>>());
                assert!(f.app.runner.ports.is_empty() && f.app.runner.started.is_empty() && !f.recorded());
            }
            // FP8 keeps its ring on two Sparks, and TCP serves it.
            let mut fp8 = Fixture::new(2, false);
            fp8.app.up(&fp8_options()).unwrap();
            let requests = starts(&fp8);
            assert_eq!(requests.len(), 2);
            assert!(requests.iter().all(|request| request["env.SEQPAR_RDMA"] == "0"), "{requests:?}");
            return;
        }
        f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let requests = starts(&f);
        assert_eq!(requests.len(), 2);
        for request in &requests {
            assert!(request.keys().all(|key| !key.contains("VADE_W2")), "{request:?}");
            assert_eq!(request["env.SEQPAR_RDMA"], "1");
        }
        // Four Sparks carry no path switch either: the engine chooses the split in parts over RDMA
        // or the chain.
        let mut chain = Fixture::fp4(4, false);
        chain.app.up(&fp4_options(&["--sparks", "4", "--mtp", "0"])).unwrap();
        assert!(starts(&chain).iter().all(|request| request.keys().all(|key| !key.contains("VADE_W2"))));
        return;
    }
    for (case, rdma) in [("rdma", None), ("rdma", Some("1")), ("tcp", Some("0"))] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::two_spark::up_with_two_sparks_runs_the_split_over_rdma_only", "--nocapture"]).env(CASE, case);
        match rdma { Some(value) => command.env("SEQPAR_RDMA", value), None => command.env_remove("SEQPAR_RDMA") };
        let result = command.output().unwrap();
        assert!(result.status.success(), "{case}: {}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"), "the child ran the case");
    }
}

const SPLIT: &str = r#","vade":{"numerics":{"world_size":2,"topology":"pipeline","path":"tensor_split","path_reason":"FP4-Local on two Sparks","tensor_split":{"experts":"output_columns","projection_level":3,"wire":"write","head":"vocabulary_halves"}}}"#;
const CHAIN: &str = r#","vade":{"numerics":{"world_size":4,"topology":"pipeline","path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take)"}}"#;
const RING: &str = r#","vade":{"numerics":{"world_size":2,"topology":"ring","path":"ring","path_reason":"FP8: the tensor split does not serve FP8"}}"#;

/// `vade status` text and `vade status --json` of a running launch.
fn status(f: &mut Fixture) -> (String, crate::evals::json::Value) {
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    let mut report = crate::status::Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    (printed, crate::evals::json::parse(&report.finish(Duration::ZERO).0).unwrap())
}

#[test]
fn status_names_every_multi_spark_path_and_its_reason() {
    use crate::evals::json::{self, Value};
    let split_levels = "projection level 3, write wire, head split by vocabulary halves";
    #[allow(clippy::type_complexity)] // test table of two-Spark fixture cases.
    let cases: [(&str, fn() -> Fixture, UpOptions, &str, String, &str); 4] = [
        ("split", || Fixture::fp4(2, false), fp4_options(&["--sparks", "2"]), SPLIT,
            format!("tensor split (FP4-Local on two Sparks; {split_levels})"),
            r#"{"path":"tensor_split","path_reason":"FP4-Local on two Sparks","projection_level":3.0,"wire":"write","head":"vocabulary_halves"}"#),
        ("chain", || Fixture::fp4(4, false), fp4_options(&["--sparks", "4"]), CHAIN,
            "layer chain (FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take))".into(),
            r#"{"path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take)","projection_level":null,"wire":null,"head":null}"#),
        ("FP8 ring", || Fixture::new(2, false), fp8_options(), RING,
            "ring (FP8: the tensor split does not serve FP8)".into(),
            r#"{"path":"ring","path_reason":"FP8: the tensor split does not serve FP8","projection_level":null,"wire":null,"head":null}"#),
        ("older engine", || Fixture::fp4(2, false), fp4_options(&["--sparks", "2"]), "", "not reported by this engine".into(), "null"),
    ];
    for (name, fixture, options, extra, row, expected) in cases {
        let mut f = fixture();
        f.app.runner.health_extra = extra.into();
        f.app.up(&options).unwrap();
        let (printed, value) = status(&mut f);
        assert!(printed.contains(&format!("\n  Multi-Spark path       {row}\n")), "{name}: {printed}");
        assert_eq!(value["ready"], Value::Bool(true), "{name}");
        assert_eq!(value["multi_spark_path"], json::parse(expected).unwrap(), "{name}");
    }
    // The split without a reason (an older receipt), and the rank path, which only --tensor-map selects.
    let split = health::MultiSparkPath::from_health(&format!("{{\"status\":\"ready\"{}}}", SPLIT.replace(r#""path_reason":"FP4-Local on two Sparks","#, ""))).unwrap();
    assert_eq!(split.label(), format!("tensor split ({split_levels})"));
    // The split over more ranks: its own reason, projection level 2 over the relay, the head in parts.
    let parts = health::MultiSparkPath::from_health(r#"{"vade":{"numerics":{"path":"tensor_split","path_reason":"FP4-Local on more than two Sparks whose widths cut into whole parts","tensor_split":{"experts":"output_columns","long_prompts":"sequence_split_or_chain_tiles","projection_level":2,"wire":"relay","head":"vocabulary_parts"}}}}"#).unwrap();
    assert_eq!(parts.label(), "tensor split (FP4-Local on more than two Sparks whose widths cut into whole parts; projection level 2, relay wire, head split by vocabulary parts)");
    let rank = health::MultiSparkPath::from_health(r#"{"vade":{"numerics":{"path":"rank_path","path_reason":"FP4-Local with --tensor-map"}}}"#).unwrap();
    assert_eq!(rank.label(), "rank path (FP4-Local with --tensor-map)");
    assert_eq!(health::MultiSparkPath::from_health(r#"{"status":"ready","vade":{"numerics":{"topology":"local"}}}"#), None);
    // One Spark and independent replicas serve no model together: no path row or field.
    for (count, options) in [(1, fp4_options(&[])), (2, fp4_options(&["--replicas", "2"]))] {
        let mut f = Fixture::fp4(count, false);
        f.app.runner.health_extra = SPLIT.into();
        f.app.up(&options).unwrap();
        let (printed, value) = status(&mut f);
        assert!(!printed.contains("Multi-Spark path"), "{printed}");
        assert_eq!(value.get("multi_spark_path"), None, "{value:?}");
        f.app.down().unwrap();
    }
}

/// `FakeRunner`, with one first start that the engine refuses for memory.
struct RefusesOnce { inner: FakeRunner, refused: bool, log: &'static str }
/// The FP4 engine's refusal of the long default context.
const FP4_MEMORY_REFUSAL: &str = "Requested size does not fit.\n  Asked: CONTEXT=131072 OUTPUT=8192\n  Available: 1 GiB\n";
/// The two-Spark text service's refusal of it, as rank 0 logs it (`flash-serve/text.rs`).
const TEXT_MEMORY_REFUSAL: &str = "STARTUP rank=0 failed in step memory_admission after 3.1 s (23.6 s since the process started): text service memory refusal: CONTEXT=131072; budget=111627461837; resident=92304140296; shared_workspace=13365415224; session_pool=5957906317; full_session=10529525760; reduce CONTEXT\nFAIL text service memory refusal: CONTEXT=131072; budget=111627461837; resident=92304140296; shared_workspace=13365415224; session_pool=5957906317; full_session=10529525760; reduce CONTEXT\n";
impl Runner for RefusesOnce {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default();
        let op = request.get("op").map(String::as_str);
        if op == Some("start") && !self.refused { self.inner.exit_on_health = Some("a".into()); }
        if op == Some("logs") && !self.refused {
            self.refused = true;
            return Ok(Output { success: true, stdout: text::encode(&Fields::from([("log".into(), self.log.into())])) });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

/// `FakeRunner`, with an engine that refuses a text checkpoint on every normal start.
struct RefusesText { inner: FakeRunner }
const TEXT_REFUSAL_LOG: &str = "text service refused: unsupported execution: complete resident decoder is absent.; source precision and zero-sign policy require owner approval\n";
impl Runner for RefusesText {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default();
        let op = request.get("op").map(String::as_str);
        if op == Some("start") { self.inner.exit_on_health = Some("a".into()); }
        if op == Some("logs") { return Ok(Output { success: true, stdout: text::encode(&Fields::from([("log".into(), TEXT_REFUSAL_LOG.into())])) }); }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

/// When an older engine refuses a text checkpoint, `up` prints its log line and a clear upgrade/flag hint.
#[test]
fn a_normal_start_that_meets_the_text_refusal_says_how_to_start() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesText { inner: std::mem::take(&mut f.app.runner) } };
    let error = app.up(&fp4_options(&["--sparks", "2", "--context", "32768"])).unwrap_err();
    assert!(error.contains("Last engine log lines:\ntext service refused:"), "{error}");
    assert!(error.contains("This older engine build requires --diagnostic-unqualified-text for GLM."), "{error}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    // The legacy flag still parses as a backward-compatible no-op.
    let Command::Up(options) = crate::cli::parse(&["up", "--sparks", "2", "--diagnostic-unqualified-text"].map(str::to_owned)).unwrap().command else { panic!() };
    assert!(options.diagnostic_unqualified_text);
    // Current flash-serve/text.rs serves GLM qualified without refusing on missing --diagnostic-unqualified-text.
    assert!(include_str!("../../../../engine/crates/tp/src/bin/flash-serve/text.rs").contains("text_service qualification=qualified ready"));
    // Any other stop keeps its own words and gets no hint.
    for log in ["", "Requested size does not fit.\n  Asked: CONTEXT=131072 OUTPUT=8192\n  Available: 1 GiB\n", "flashnext ready world=2"] { assert_eq!(health::text_refusal_hint(log), None, "{log}"); }
}

fn saved_context(f: &Fixture) -> u32 { Session::load(&f.app.config.state_dir).unwrap().unwrap().context }

#[test]
fn a_group_of_sparks_defaults_to_the_long_context_and_one_spark_does_not() {
    let context = |sparks: usize, options: &[&str]| {
        let mut f = Fixture::fp4(sparks, false);
        f.app.up(&fp4_options(options)).unwrap();
        saved_context(&f)
    };
    assert_eq!(context(1, &[]), 32768);
    assert_eq!(context(2, &["--sparks", "2"]), 131072);
    assert_eq!(context(4, &["--sparks", "4"]), 131072);
    assert_eq!(context(2, &["--hosts", "a,b"]), 131072);
    // A plain up after setup picks one Spark, and each plain replica is one Spark.
    assert_eq!(context(2, &[]), 32768);
    assert_eq!(context(2, &["--replicas", "2"]), 32768);
    // Each pair is two Sparks.
    assert_eq!(context(4, &["--replicas", "2", "--pairs"]), 131072);
    // A context the person names stays, on any count.
    assert_eq!(context(2, &["--sparks", "2", "--context", "32768"]), 32768);
    assert_eq!(context(2, &["--sparks", "2", "--context", "8192"]), 8192);
}

#[test]
fn a_running_engine_keeps_its_context_and_a_restart_keeps_it_too() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2", "--context", "32768"])).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    let recorded = crate::app::resident::recorded_options(&session);
    // A bare up, and the same up with no --context, attach with the saved size.
    let Command::Up(bare) = crate::cli::parse(&["up".to_string()]).unwrap().command else { panic!() };
    f.app.up(&bare).unwrap();
    f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
    assert_eq!(saved_context(&f), 32768);
    // vade update restarts from the saved options.
    f.app.down().unwrap();
    f.app.up(&recorded).unwrap();
    assert_eq!(saved_context(&f), 32768);
}

#[test]
fn a_long_default_that_does_not_fit_falls_to_the_short_one_and_says_so() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false, log: FP4_MEMORY_REFUSAL } };
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Context 131072 does not fit in memory on these Sparks. Starting with context 32768.\n"), "{printed}");
    assert!(printed.contains("context 32768 tokens"), "the ready line names the chosen size: {printed}");
    assert_eq!(Session::load(&app.config.state_dir).unwrap().unwrap().context, 32768);
    // A size that the person named stays their choice: the refusal is the answer.
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false, log: FP4_MEMORY_REFUSAL } };
    let error = app.up(&fp4_options(&["--sparks", "2", "--context", "131072"])).unwrap_err();
    assert!(error.contains("Requested size does not fit."), "{error}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
}

/// The text service words its refusal differently from the FP4 engine. A bare
/// start of it must fall to the short context too, not stop at the refusal.
#[test]
fn a_text_service_start_that_does_not_fit_the_long_default_falls_to_the_short_one() {
    let text = ["--sparks", "2", "--diagnostic-unqualified-text"];
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false, log: TEXT_MEMORY_REFUSAL } };
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.up(&fp4_options(&text))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Context 131072 does not fit in memory on these Sparks. Starting with context 32768.\n"), "{printed}");
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.context, session.resident_text, session.diagnostic_unqualified_text), (32768, true, false));
    assert_eq!(session.mtp, crate::state::TEXT_DRAFT_DEPTH, "the fall to the short context keeps the default drafting depth");
    // A size that the person named stays their choice: it never falls, and the refusal of it is the answer. (A
    // depth that nobody named falls to 0 first, because `--mtp 0` may fit that size: the next test group.)
    let (result, printed, starts, app) = text_up(&["--context", "131072"], &[TEXT_MEMORY_REFUSAL, TEXT_MEMORY_REFUSAL]);
    let error = result.unwrap_err();
    assert!(error.contains("text service memory refusal: CONTEXT=131072;"), "{error}");
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0")]));
    assert!(!printed.contains("Starting with context"), "{printed}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    let (result, _, starts, app) = text_up(&["--context", "131072", "--mtp", "0"], &[TEXT_MEMORY_REFUSAL]);
    assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=131072;"));
    assert_eq!(starts, both_ranks(&[("131072", "0")]));
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    // Another text service stop is not a context refusal: no second start.
    for log in ["text service needs 9 device bytes; budget=8; resident=1; session=1; reduce CONTEXT\n", "text service memory refusal: CONTEXT=32768; budget=1; resident=1; shared_workspace=1; session_pool=1; full_session=2; reduce CONTEXT\n"] {
        let mut f = Fixture::fp4(2, false);
        let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false, log } };
        let capture = crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(), || app.up(&fp4_options(&text))).unwrap_err();
        assert!(!String::from_utf8(capture.bytes.borrow().clone()).unwrap().contains("Starting with context"), "{log}");
    }
}

/// `FakeRunner`, with an engine whose next starts fail in turn, each with one of these rank 0 log lines. Then a start succeeds.
struct RefusesInTurn { inner: FakeRunner, logs: std::collections::VecDeque<&'static str>, now: Option<&'static str> }
/// The engine's stop when a start asks for a predictor and the checkpoint declares none, as rank 0 logs it
/// (`flash-serve/text.rs`, start step `predictor`).
const NO_DRAFT_LAYER_LOG: &str = "STARTUP rank=0 failed in step predictor after 0.0 s (19.8 s since the process started): checkpoint declares no predictor layer\n";
/// A start error that is neither fall's business.
const OTHER_START_LOG: &str = "STARTUP rank=0 failed in step wire_link after 5.0 s (24.8 s since the process started): wire link timed out\n";
impl Runner for RefusesInTurn {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default();
        let op = request.get("op").map(String::as_str);
        if op == Some("start") {
            // One start is rank a, then rank b: the next log line belongs to the whole start.
            if exec.label == "a start" { self.now = self.logs.pop_front(); }
            if self.now.is_some() { self.inner.exit_on_health = Some("a".into()); }
        }
        if let (Some("logs"), Some(log)) = (op, self.now) {
            return Ok(Output { success: true, stdout: text::encode(&Fields::from([("log".into(), log.into())])) });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}
const BARE_TEXT: [&str; 3] = ["--sparks", "2", "--diagnostic-unqualified-text"];
/// One `vade up` of the text service against an engine that fails its next starts with `logs`, in turn.
/// Returns the result, what was printed, every rank start as (label, context, depth) and the app. The app holds
/// the fixture, whose folder keeps the session record: it lives as long as the app does.
type TextStart = (Result<(), String>, String, Vec<(String, String, String)>, TextApp);
fn text_up(extra: &[&str], logs: &[&'static str]) -> TextStart {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesInTurn { inner: std::mem::take(&mut f.app.runner), logs: logs.iter().copied().collect(), now: None } };
    let capture = crate::output::testing::Capture::default();
    let options = fp4_options(&BARE_TEXT.iter().chain(extra).copied().collect::<Vec<_>>());
    let result = crate::output::with_writer(capture.clone(), || app.up(&options));
    let starts = rank_starts(&app.runner.inner.calls);
    (result, String::from_utf8(capture.bytes.borrow().clone()).unwrap(), starts, TextApp { app, _fixture: f })
}
struct TextApp { app: App<RefusesInTurn>, _fixture: Fixture }
impl std::ops::Deref for TextApp { type Target = App<RefusesInTurn>; fn deref(&self) -> &Self::Target { &self.app } }
impl std::ops::DerefMut for TextApp { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.app } }
fn rank_starts(calls: &[Exec]) -> Vec<(String, String, String)> {
    calls.iter().filter(|call| matches!(call.label.as_str(), "a start" | "b start")).map(|call| {
        let request = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
        let args: Vec<String> = (0..).map_while(|i| request.get(&format!("arg.{i}")).cloned()).collect();
        let depth = args.iter().position(|arg| arg == "--mtp").map(|at| args[at + 1].clone()).unwrap_or_default();
        (call.label.clone(), args[6].clone(), depth)
    }).collect()
}
/// What `starts` holds when each start is both ranks at this (context, depth), in this order.
fn both_ranks(starts: &[(&str, &str)]) -> Vec<(String, String, String)> {
    starts.iter().flat_map(|(context, depth)| ["a start", "b start"].map(|label| (label.to_string(), context.to_string(), depth.to_string()))).collect()
}
const NO_DRAFT_LINE: &str = "This model has no drafting layer. Starting without drafting.\n";
const SHORT_CONTEXT_LINE: &str = "Context 131072 does not fit in memory on these Sparks. Starting with context 32768.\n";

/// A GLM model folder with no drafting layer started yesterday, at the old default depth 0. The new default depth must
/// not fail that start: the engine's "no predictor layer" stop is answered by one second start at depth 0.
#[test]
fn a_default_text_start_on_a_model_with_no_drafting_layer_starts_without_drafting() {
    // The words this reads are the engine's own, and the engine loads the predictor before it admits memory.
    assert!(include_str!("../../../../engine/crates/tp/src/bounded_decoder/proposer.rs").contains(&format!("ok_or(\"{}\")", super::super::NO_DRAFT_LAYER)));
    let serve = include_str!("../../../../engine/crates/tp/src/bin/flash-serve/text.rs");
    assert!(serve.find("startup::step(\"predictor\")").unwrap() < serve.find("text service memory refusal").unwrap());
    let (result, printed, starts, mut app) = text_up(&[], &[NO_DRAFT_LAYER_LOG]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0")]), "one start at the default depth, one at depth 0");
    assert_eq!(printed.matches(NO_DRAFT_LINE).count(), 1, "{printed}");
    assert!(!printed.contains("Starting with context"), "{printed}");
    // The record: depth 0, passed to the engine cleanly without --diagnostic-unqualified-text, and marked as nobody's choice.
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.mtp, session.draft_fall, session.resident_text, session.diagnostic_unqualified_text, session.context), (0, Some(crate::cli::DraftFall::NoLayer), true, false, 131072));
    assert_eq!(session.engine_args(0)[9..], ["--mtp", "0"]);
    assert_eq!(session.mtp_label(), "0 (off: this model has no drafting layer)");
    assert!(printed.contains("drafting depth 0 (off: this model has no drafting layer)"), "{printed}");
    // A second start that names no depth is that same engine: it attaches. No refusal, no new start.
    crate::output::with_writer(crate::output::testing::Capture::default(), || app.up(&fp4_options(&BARE_TEXT))).unwrap();
    assert_eq!(rank_starts(&app.runner.inner.calls).len(), 4);
    // A start that names a depth asks for another engine: the record stays and the person is told to stop it first.
    let named = app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text", "--mtp", "3"])).unwrap_err();
    assert!(named.contains("Run vade down before changing it."), "{named}");
    // The restart command of `vade update` names no depth and no --diagnostic-unqualified-text: a restart tries the default again.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.update().unwrap());
    let hint = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    let line = hint.lines().find(|line| line.trim_start().starts_with("vade up ")).unwrap_or_else(|| panic!("{hint}"));
    assert!(!line.contains("--diagnostic-unqualified-text") && !line.contains("--mtp"), "{line}");
    // A start that meets no such stop keeps the default depth and carries no mark.
    let (result, printed, starts, app) = text_up(&[], &[]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3")]));
    assert!(!printed.contains(NO_DRAFT_LINE), "{printed}");
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.mtp, session.draft_fall), (crate::state::TEXT_DRAFT_DEPTH, None));
}

/// The fall is for the default only. A depth the person named, by `--mtp`, is their choice: the engine's stop is
/// the answer and nothing starts a second time. (`MTP` sets the same `mtp_explicit` as `--mtp`: `cli.rs`.)
#[test]
fn a_named_depth_on_a_model_with_no_drafting_layer_stays_the_engines_error() {
    for depth in ["1", "3", "7"] {
        let (result, printed, starts, app) = text_up(&["--mtp", depth], &[NO_DRAFT_LAYER_LOG]);
        let error = result.unwrap_err();
        assert!(error.contains("checkpoint declares no predictor layer"), "{error}");
        assert_eq!(starts, both_ranks(&[("131072", depth)]), "--mtp {depth}");
        assert!(!printed.contains(NO_DRAFT_LINE), "{printed}");
        assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    }
    // `--mtp 0` loads no predictor: a start that names it is a plain start with no mark.
    let (result, _, starts, app) = text_up(&["--mtp", "0"], &[]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "0")]));
    assert_eq!(Session::load(&app.config.state_dir).unwrap().unwrap().draft_fall, None);
    // The same engine answer to a start that named no depth is the fall, not an error.
    let (result, printed, starts, _) = text_up(&[], &[NO_DRAFT_LAYER_LOG]);
    result.unwrap();
    assert_eq!((starts.len(), printed.matches(NO_DRAFT_LINE).count()), (4, 1));
}

/// The two falls compose: the long default context to the short one, and the default depth to none. Each happens
/// at most once, in either order. A third stop of a kind that already fell is the answer.
#[test]
fn the_context_fall_and_the_drafting_fall_compose_once_each_in_either_order() {
    // The engine's own order: the predictor is loaded before memory is admitted.
    let (result, printed, starts, app) = text_up(&[], &[NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0"), ("32768", "0")]));
    assert_eq!((printed.matches(NO_DRAFT_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count()), (1, 1), "{printed}");
    assert!(printed.find(NO_DRAFT_LINE).unwrap() < printed.find(SHORT_CONTEXT_LINE).unwrap(), "{printed}");
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.context, session.mtp, session.draft_fall), (32768, 0, Some(crate::cli::DraftFall::NoLayer)));
    // The other order.
    let (result, printed, starts, app) = text_up(&[], &[TEXT_MEMORY_REFUSAL, NO_DRAFT_LAYER_LOG]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("32768", "3"), ("32768", "0")]));
    assert_eq!((printed.matches(NO_DRAFT_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count()), (1, 1), "{printed}");
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.context, session.mtp, session.draft_fall), (32768, 0, Some(crate::cli::DraftFall::NoLayer)));
    // At most one of each: after the context and the depth have fallen, a third stop is returned as it is, whatever
    // it says, and nothing is left recorded. Depth 0 is the depth's last fall, for either reason.
    for (logs, words) in [([NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL, NO_DRAFT_LAYER_LOG], "checkpoint declares no predictor layer"),
                          ([NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL, SHORT_MEMORY_REFUSAL], "text service memory refusal: CONTEXT=32768;"),
                          ([TEXT_MEMORY_REFUSAL, NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL], "text service memory refusal: CONTEXT=131072;"),
                          ([TEXT_MEMORY_REFUSAL, NO_DRAFT_LAYER_LOG, SHORT_MEMORY_REFUSAL], "text service memory refusal: CONTEXT=32768;"),
                          ([NO_DRAFT_LAYER_LOG, NO_DRAFT_LAYER_LOG, NO_DRAFT_LAYER_LOG], "checkpoint declares no predictor layer")] {
        let (result, printed, starts, app) = text_up(&[], &logs);
        let error = result.unwrap_err();
        assert!(error.contains(words), "{error}");
        assert!(printed.matches(NO_DRAFT_LINE).count() == 1 && printed.matches(SHORT_CONTEXT_LINE).count() <= 1 && !printed.contains(NO_MEMORY_LINE), "{printed}");
        assert!(starts.len() <= 6, "{starts:?}");
        assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    }
    // A context the person named does not fall, and the drafting fall still happens beside it.
    let (result, printed, starts, _) = text_up(&["--context", "131072"], &[NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL]);
    assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=131072;"));
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0")]));
    assert_eq!((printed.matches(NO_DRAFT_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count()), (1, 0), "{printed}");
}

/// The engine's refusal of the short context for memory, as rank 0 logs it. With the predictor a start needs more.
const SHORT_MEMORY_REFUSAL: &str = "STARTUP rank=0 failed in step memory_admission after 3.0 s (24.1 s since the process started): text service memory refusal: CONTEXT=32768; budget=106000000000; resident=94784987144; shared_workspace=8836481336; session_pool=2378531520; full_session=3852980224; reduce CONTEXT\n";
const NO_MEMORY_LINE: &str = "Not enough free memory for drafting. Starting without drafting.\n";

/// The default depth needs more memory than depth 0. A default start that does not fit must not fail where
/// `--mtp 0` starts: the context falls first, as before, and then the depth falls to 0, once.
#[test]
fn a_default_text_start_that_does_not_fit_with_drafting_starts_without_drafting() {
    // The long default context does not fit, then the short one does not fit with the predictor: three starts.
    let (result, printed, starts, mut app) = text_up(&[], &[TEXT_MEMORY_REFUSAL, SHORT_MEMORY_REFUSAL]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("32768", "3"), ("32768", "0")]), "the context falls first, then the depth");
    assert_eq!((printed.matches(SHORT_CONTEXT_LINE).count(), printed.matches(NO_MEMORY_LINE).count(), printed.matches(NO_DRAFT_LINE).count()), (1, 1, 0), "{printed}");
    assert!(printed.find(SHORT_CONTEXT_LINE).unwrap() < printed.find(NO_MEMORY_LINE).unwrap(), "{printed}");
    // The record: depth 0, passed to the engine as `--mtp 0` (no --diagnostic-unqualified-text), marked as a fall for memory and not as a choice.
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.context, session.mtp, session.draft_fall), (32768, 0, Some(crate::cli::DraftFall::Memory)));
    assert_eq!(session.engine_args(0)[9..], ["--mtp", "0"]);
    assert_eq!(session.mtp_label(), "0 (off: not enough free memory for drafting)");
    assert!(printed.contains("drafting depth 0 (off: not enough free memory for drafting)"), "{printed}");
    // While that engine runs, a second start that names no depth attaches to it: no refusal, no new start.
    crate::output::with_writer(crate::output::testing::Capture::default(), || app.up(&fp4_options(&BARE_TEXT))).unwrap();
    assert_eq!(rank_starts(&app.runner.inner.calls).len(), 6);
    // The restart command names no depth and no --diagnostic-unqualified-text: the next start tries drafting again.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.update().unwrap());
    let hint = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    let line = hint.lines().find(|line| line.trim_start().starts_with("vade up ")).unwrap_or_else(|| panic!("{hint}"));
    assert!(line.contains(" --context 32768") && !line.contains("--diagnostic-unqualified-text") && !line.contains("--mtp"), "{line}");
    app.down().unwrap();
    crate::output::with_writer(crate::output::testing::Capture::default(), || app.up(&fp4_options(&BARE_TEXT))).unwrap();
    let again = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((again.mtp, again.draft_fall), (crate::state::TEXT_DRAFT_DEPTH, None), "after vade down a bare start drafts again");
    // A context the person named cannot fall, so the depth falls at that context.
    let (result, printed, starts, app) = text_up(&["--context", "32768"], &[SHORT_MEMORY_REFUSAL]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("32768", "3"), ("32768", "0")]));
    assert_eq!((printed.matches(NO_MEMORY_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count()), (1, 0), "{printed}");
    assert_eq!(Session::load(&app.config.state_dir).unwrap().unwrap().draft_fall, Some(crate::cli::DraftFall::Memory));
    // The same at a long context the person named: where `--mtp 0` fits that size, the default starts there too.
    let (result, printed, starts, app) = text_up(&["--context", "131072"], &[TEXT_MEMORY_REFUSAL]);
    result.unwrap();
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0")]));
    assert_eq!((printed.matches(NO_MEMORY_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count()), (1, 0), "{printed}");
    let session = Session::load(&app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.context, session.mtp, session.draft_fall), (131072, 0, Some(crate::cli::DraftFall::Memory)));
    // A depth the person named is their choice: the refusal is the answer, at the short context too.
    for depth in ["1", "3", "7", "0"] {
        let (result, printed, starts, app) = text_up(&["--mtp", depth], &[TEXT_MEMORY_REFUSAL, SHORT_MEMORY_REFUSAL]);
        assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=32768;"), "--mtp {depth}");
        assert_eq!(starts, both_ranks(&[("131072", depth), ("32768", depth)]), "--mtp {depth}");
        assert!(!printed.contains(NO_MEMORY_LINE) && printed.matches(SHORT_CONTEXT_LINE).count() == 1, "{printed}");
        assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    }
    // A refusal that names another context than this start's is not this fall.
    let (result, printed, starts, _) = text_up(&["--context", "65536"], &[SHORT_MEMORY_REFUSAL]);
    assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=32768;"));
    assert_eq!(starts, both_ranks(&[("65536", "3")]));
    assert!(!printed.contains(NO_MEMORY_LINE), "{printed}");
}

/// The three falls compose: the context once, and the depth once, for the missing layer or for memory. A stop
/// after them is the answer, also when it is a memory refusal at depth 0 (then `--mtp 0` would not start either).
#[test]
fn the_three_falls_compose_and_each_happens_at_most_once() {
    // No drafting layer, and the long context does not fit: the depth falls for the layer, the context for memory.
    // A third stop, memory at the short context and depth 0, is what `--mtp 0` gets too: it is returned as it is.
    let (result, printed, starts, app) = text_up(&[], &[NO_DRAFT_LAYER_LOG, TEXT_MEMORY_REFUSAL, SHORT_MEMORY_REFUSAL]);
    assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=32768;"));
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0"), ("32768", "0")]));
    assert_eq!((printed.matches(NO_DRAFT_LINE).count(), printed.matches(SHORT_CONTEXT_LINE).count(), printed.matches(NO_MEMORY_LINE).count()), (1, 1, 0), "{printed}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    // Memory at both contexts: the context falls, then the depth. Every later stop is returned as it is: a second
    // memory refusal (depth 0 does not fit either), the layer error, or any other error. Never a fourth start.
    for (third, words) in [(SHORT_MEMORY_REFUSAL, "text service memory refusal: CONTEXT=32768;"), (NO_DRAFT_LAYER_LOG, "checkpoint declares no predictor layer"), (OTHER_START_LOG, "wire link timed out")] {
        let (result, printed, starts, app) = text_up(&[], &[TEXT_MEMORY_REFUSAL, SHORT_MEMORY_REFUSAL, third]);
        let error = result.unwrap_err();
        assert!(error.contains(words), "{error}");
        assert_eq!(starts, both_ranks(&[("131072", "3"), ("32768", "3"), ("32768", "0")]));
        assert_eq!((printed.matches(SHORT_CONTEXT_LINE).count(), printed.matches(NO_MEMORY_LINE).count(), printed.matches(NO_DRAFT_LINE).count()), (1, 1, 0), "{printed}");
        assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    }
    // The short context's refusal can only come after the context fell or was named: on a first bare start at the
    // long context it is not this start's context, so nothing falls.
    let (result, printed, starts, _) = text_up(&[], &[SHORT_MEMORY_REFUSAL]);
    assert!(result.unwrap_err().contains("text service memory refusal: CONTEXT=32768;"));
    assert_eq!(starts, both_ranks(&[("131072", "3")]));
    assert!(!printed.contains(NO_MEMORY_LINE) && !printed.contains("Starting with context"), "{printed}");
}

/// After the fall, any other start error is reported as it is: no third start, no record left, the words unchanged.
#[test]
fn a_different_start_error_after_the_drafting_fall_is_reported_as_it_is() {
    let (result, printed, starts, app) = text_up(&[], &[NO_DRAFT_LAYER_LOG, OTHER_START_LOG]);
    let error = result.unwrap_err();
    assert!(error.contains("wire link timed out") && !error.contains("checkpoint declares no predictor layer"), "{error}");
    assert_eq!(starts, both_ranks(&[("131072", "3"), ("131072", "0")]));
    assert_eq!(printed.matches(NO_DRAFT_LINE).count(), 1, "{printed}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    // The same error on the first start is not a fall at all.
    let (result, printed, starts, _) = text_up(&[], &[OTHER_START_LOG]);
    assert!(result.unwrap_err().contains("wire link timed out"));
    assert_eq!(starts, both_ranks(&[("131072", "3")]));
    assert!(!printed.contains(NO_DRAFT_LINE) && !printed.contains("Starting with context"), "{printed}");
}

/// What a bare `vade up --sparks 2 --diagnostic-unqualified-text` hands the engine on each rank: the long default
/// context first and, after the engine refuses that for memory, the short one. `--nocapture` prints every start.
#[test]
fn a_bare_text_service_start_passes_these_engine_arguments_on_both_ranks() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false, log: TEXT_MEMORY_REFUSAL } };
    app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    let mut starts = Vec::new();
    for call in app.runner.inner.calls.iter().filter(|call| matches!(call.label.as_str(), "a start" | "b start")) {
        let request = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
        let args: Vec<String> = (0..).map_while(|i| request.get(&format!("arg.{i}")).cloned()).collect();
        let env: Vec<String> = request.iter().filter(|(key, _)| (key.starts_with("env.") || key.starts_with("env_unset.")) && !key.ends_with("SEQPAR_CONTROL_SECRET") && !key.ends_with("TP_RUN_ID"))
            .map(|(key, value)| format!("{key}={value}")).collect();
        println!("ENGINE START {}: args: {}", call.label, args.join(" "));
        println!("ENGINE START {}: env: {}", call.label, env.join(" "));
        starts.push((call.label.clone(), args));
    }
    // Rank 0 and rank 1 at the long context, then both again at the short one: the same clean options each time.
    let options: Vec<String> = ["--mtp", "3"].map(String::from).into();
    assert_eq!(starts.iter().map(|(label, args)| (label.as_str(), args[0].as_str(), args[1].as_str(), args[6].as_str())).collect::<Vec<_>>(),
        [("a start", "0", "2", "131072"), ("b start", "1", "2", "131072"), ("a start", "0", "2", "32768"), ("b start", "1", "2", "32768")]);
    for (label, args) in &starts { assert_eq!(args[9..], options[..], "{label}: {args:?}"); }
}

/// The manifest of an older vade: one rank kept the folder of an earlier install.
#[test]
fn a_rank_in_an_older_install_folder_makes_up_install_again_and_start() {
    let mut f = Fixture::fp4(2, false);
    let path = f.app.config.build_manifest.clone();
    let mut manifest = Manifest::load(&path).unwrap();
    manifest.artifacts.get_mut("b").unwrap().ptx = "~/stage/older/flash.ptx".into();
    text::atomic_write(&path, &manifest.encode()).unwrap();
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("These Sparks hold Vade in different install folders, so vade up installs Vade on each Spark again."), "{printed}");
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(session.ranks[0].artifact.ptx, session.ranks[1].artifact.ptx);
    assert!(session.ranks[0].artifact.ptx.contains("/builds/"), "{}", session.ranks[0].artifact.ptx);
    assert_eq!(f.app.runner.started, ["a".to_owned(), "b".to_owned()].into());
}

/// A hand-written config can name two stage folders. No install fixes that, so the refusal stays.
#[test]
fn two_configured_stage_folders_keep_the_refusal_and_install_nothing() {
    let mut f = Fixture::fp4(2, false);
    f.app.config.machines[1].stage_dir = "~/other-stage".into();
    let path = f.app.config.build_manifest.clone();
    let mut manifest = Manifest::load(&path).unwrap();
    manifest.artifacts.get_mut("b").unwrap().ptx = "~/other-stage/flash.ptx".into();
    text::atomic_write(&path, &manifest.encode()).unwrap();
    let error = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
    assert!(error.contains("expanded absolute PTX paths differ"), "{error}");
    assert!(!f.app.runner.calls.iter().any(|exec| exec.label == "harness build"), "no install");
    assert!(!f.recorded());
}

/// A start that loses SSH shows ssh's own line and its fix, not a guess.
#[test]
fn the_check_before_a_start_shows_the_real_ssh_line() {
    let mut f = Fixture::fp4(2, false);
    f.app.runner.silent_probe = true;
    f.app.runner.ssh_says = Some("you@a-ssh: Permission denied (publickey).\n");
    let error = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
    assert!(error.starts_with("SSH checks failed. Vade never edits ~/.ssh."), "{error}");
    assert!(error.contains("ssh a-ssh failed: you@a-ssh: Permission denied (publickey)."), "{error}");
    assert_eq!(error.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{error}");
    assert!(!error.contains("did not reply"), "{error}");
    assert!(!f.recorded());
}

#[test]
fn the_ready_line_names_each_spark_and_the_context() {
    let mut f = Fixture::fp4(2, false);
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Vade is ready on 2 Sparks (a, b) at http://localhost:18080: context 131072 tokens, drafting depth engine default.\n"), "{printed}");
    assert!(printed.contains("Drafting request") && printed.contains("engine default"), "{printed}");
}

/// Destinations stay private, including after a config edit.
#[test]
fn up_status_and_doctor_hide_address_destinations_and_old_machine_names() {
    let mut f = Fixture::fp4(2, false);
    let hosts = ["user@192.0.2.10", "user@192.0.2.11"];
    for (machine, host) in f.app.config.machines.iter_mut().zip(hosts) { machine.ssh = host.into(); }
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.execute(Command::Up(fp4_options(&["--sparks", "2"])))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Machines") && printed.contains("Vade is ready on 2 Sparks (a, b)"), "{printed}");
    for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    for (index, rank) in session.ranks.iter_mut().enumerate() { rank.machine.name = format!("192-0-2-{}", index + 10); }
    session.save(&f.app.config.state_dir).unwrap();
    // The current config no longer has these destinations. The old record still does.
    for machine in &mut f.app.config.machines { machine.ssh = format!("{}-ssh", machine.name); }
    for command in [Command::Status, Command::Doctor] {
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || f.app.execute(command));
        let printed = format!("{}{}", String::from_utf8(capture.bytes.borrow().clone()).unwrap(), result.err().unwrap_or_default());
        assert!(printed.contains("spark-1") && printed.contains("spark-2"), "{printed}");
        for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    }
    let mut report = crate::status::Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    let printed = report.finish(Duration::ZERO).0;
    for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    crate::evals::json::parse(&printed).unwrap();
}

/// The text service keeps prompts by default. `--cache-bytes 0` is this CLI's "cache off" and status prints it so:
/// then every rank is started with the engine's off switch, once. Without it, or with a size above 0, no rank gets it.
#[test]
fn cache_bytes_zero_starts_every_text_rank_with_no_prompt_keep() {
    let f=Fixture::fp4(2,false);
    let text=["--sparks","2","--context","4096","--output","64","--diagnostic-unqualified-text"];
    for (extra,off) in [(&["--cache-bytes","0"][..],1),(&[],0)] {
        let options=fp4_options(&[&text[..],extra].concat());
        let session=f.app.requested_session(&options,None,true).unwrap();assert_eq!(session.ranks.len(),2);
        for at in 0..session.ranks.len() {
            let args=session.engine_args(at);assert_eq!(args.iter().filter(|s|s.as_str()=="--no-prompt-keep").count(),off,"{extra:?} rank {at}");
            assert!(!args.iter().any(|s|s=="--prompt-keep"),"{extra:?} rank {at}: the default needs no name");
        }
        if off==1 {assert_eq!(crate::app::retention_label(&session,None),"off (--cache-bytes 0)");}
    }
    // AN UPGRADE: a record that an older vade saved with cache_bytes=0 for this service. The record has no field of
    // its own for prompt keep, so what an older vade wrote is what this one writes. Loaded again, every rank is
    // started with the off switch: the user who turned the cache off keeps it off, and status still tells the truth.
    let saved=f.app.requested_session(&fp4_options(&[&text[..],&["--cache-bytes","0"]].concat()),None,true).unwrap();
    let record=saved.encode();assert!(record.contains("cache_bytes") && !record.contains("prompt_keep") && !record.contains("prompt-keep"),"{record}");
    saved.save(&f.app.config.state_dir).unwrap();
    let loaded=Session::load(&f.app.config.state_dir).unwrap().unwrap();assert_eq!((loaded.cache_bytes,loaded.resident_text,loaded.diagnostic_unqualified_text),(Some(0),true,false));
    for at in 0..loaded.ranks.len() {assert_eq!(loaded.engine_args(at),saved.engine_args(at));assert_eq!(loaded.engine_args(at).iter().filter(|s|s.as_str()=="--no-prompt-keep").count(),1,"rank {at} of the loaded record");}
    assert_eq!(crate::app::retention_label(&loaded,None),"off (--cache-bytes 0)");
    // A size above 0 never sets this cache: refused, with the way to turn it off.
    let refused=f.app.requested_session(&fp4_options(&[&text[..],&["--cache-bytes","1073741824"]].concat()),None,true).unwrap_err();
    assert!(refused.contains("--cache-bytes 0") && refused.contains("This text service sizes its prompt cache automatically"),"{refused}");
    // Only the text service has this switch: a normal start with --cache-bytes 0 never gets it.
    let plain=f.app.requested_session(&fp4_options(&["--cache-bytes","0"]),None,true).unwrap();
    for at in 0..plain.ranks.len() {assert!(!plain.engine_args(at).iter().any(|s|s=="--no-prompt-keep"));}
}
#[test]
fn the_restart_hint_of_an_update_omits_legacy_diagnostic_flag_for_qualified_text() {
    let hint=|options:&[&str]| {
        let mut f=Fixture::fp4(2,false);
        f.app.requested_session(&fp4_options(options),None,true).unwrap().save(&f.app.config.state_dir).unwrap();
        let capture=crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(),|| f.app.update().unwrap());
        String::from_utf8(capture.bytes.borrow().clone()).unwrap()
    };
    let text=hint(&["--sparks","2","--context","4096","--output","64","--diagnostic-unqualified-text"]);
    let line=text.lines().find(|line| line.trim_start().starts_with("vade up ")).unwrap_or_else(|| panic!("{text}"));
    assert!(!line.contains("--diagnostic-unqualified-text") && !line.contains("--fp4-local"),"{text}");
    assert!(line.contains(" --context 4096 --output 64"),"{text}");
    assert_eq!(line.matches(&format!(" --mtp {}",crate::state::TEXT_DRAFT_DEPTH)).count(),1,"{text}");
    let words:Vec<String>=line.split_whitespace().skip(1).map(str::to_owned).collect();
    let Command::Up(parsed)=crate::cli::parse(&words).unwrap().command else { panic!() };
    assert!(!parsed.diagnostic_unqualified_text && parsed.context==4096,"{line}");
    let plain=hint(&[]);
    assert!(plain.contains("vade up ") && !plain.contains("--diagnostic-unqualified-text"),"{plain}");
}

#[test]
fn resident_text_session_persists_and_bare_up_reuses_it() {
    let f=Fixture::fp4(2,false);
    let options=fp4_options(&["--sparks","2","--context","4096","--output","64","--diagnostic-unqualified-text"]);
    let resident=f.app.requested_session(&options,None,true).unwrap();
    assert!(resident.resident_text && !resident.diagnostic_unqualified_text);
    for at in 0..resident.ranks.len() {
        let args=resident.engine_args(at);
        assert!(!args.iter().any(|s|s=="--diagnostic-unqualified-text" || s=="--fp4-local" || s=="--decode-fuse"));
        let depth=args.iter().position(|s|s=="--mtp").unwrap();assert_eq!(args[depth+1],"3");
    }
    resident.save(&f.app.config.state_dir).unwrap();
    let saved=Session::load(&f.app.config.state_dir).unwrap().unwrap();assert!(saved.resident_text && !saved.diagnostic_unqualified_text);
    // Bare `vade up` / `vade chat` attaches cleanly to a running qualified GLM session.
    let attached=f.app.requested_session(&UpOptions::default(),Some(&saved),true).unwrap();
    assert!(resident::same_launch(&saved,&attached));
    let plain=f.app.requested_session(&UpOptions::default(),None,true).unwrap();
    assert!(!plain.resident_text && !plain.diagnostic_unqualified_text);assert!(!resident::same_launch(&saved,&plain));
    assert!(!plain.encode().contains("resident_text") && !plain.encode().contains("diagnostic_unqualified_text"));
}

#[test]
fn second_rail_follows_configured_fabric_aux_and_tp_dual_rail_opt_out() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_AUX";
    if let Ok(case) = std::env::var(CASE) {
        let lo = std::net::Ipv4Addr::LOCALHOST;
        if case == "invalid" {
            let mut f = Fixture::fp4(2, false);
            f.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
            f.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
            let err = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
            assert!(err.contains("TP_DUAL_RAIL"), "{err}");
            assert!(f.app.runner.calls.is_empty());
            return;
        }
        if case == "optout" {
            let mut f = Fixture::fp4(2, false);
            f.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
            f.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
            f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
            for req in starts(&f) {
                // The engines find a pair's second rail by themselves, so the opt-out must reach them: the start
                // names the second rail `off`, whatever fabric_aux says.
                assert_eq!(req.get("env.TP_AUX_LISTEN").map(String::as_str), Some("off"));
                assert_eq!(req.get("env.TP_AUX_NEXT").map(String::as_str), Some("off"));
                assert!(!req.contains_key("env_unset.TP_AUX_LISTEN") && !req.contains_key("env_unset.TP_AUX_NEXT"));
            }
            // The same with no fabric_aux in the config.
            let mut plain = Fixture::fp4(2, false);
            plain.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
            for req in starts(&plain) {
                assert_eq!(req.get("env.TP_AUX_LISTEN").map(String::as_str), Some("off"));
                assert_eq!(req.get("env.TP_AUX_NEXT").map(String::as_str), Some("off"));
            }
            return;
        }
        // Unconfigured two-Spark pair leaves TP_AUX_* unset exactly as on main.
        let mut unconf = Fixture::fp4(2, false);
        unconf.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        for req in starts(&unconf) {
            assert_eq!(req.get("env_unset.TP_AUX_LISTEN").map(String::as_str), Some("1"));
            assert_eq!(req.get("env_unset.TP_AUX_NEXT").map(String::as_str), Some("1"));
            assert!(!req.contains_key("env.TP_AUX_LISTEN") && !req.contains_key("env.TP_AUX_NEXT"));
        }
        assert!(!Session::load(&unconf.app.config.state_dir).unwrap().unwrap().encode().contains("aux_listen"));

        // Configured fabric_aux at both ends sets TP_AUX_LISTEN and TP_AUX_NEXT on tp_port + 1.
        let mut conf = Fixture::fp4(2, false);
        let aux_port = conf.app.config.tp_port + 1;
        conf.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
        conf.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
        conf.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let reqs = starts(&conf);
        assert_eq!(reqs.len(), 2);
        for req in &reqs {
            assert!(!req.contains_key("env_unset.TP_AUX_LISTEN") && !req.contains_key("env_unset.TP_AUX_NEXT"));
            assert_eq!(req.get("env.TP_AUX_LISTEN").map(String::as_str), Some(format!("{lo}:{aux_port}").as_str()));
            assert_eq!(req.get("env.TP_AUX_NEXT").map(String::as_str), Some(format!("{lo}:{aux_port}").as_str()));
        }
        // Changing fabric_aux refuses resident reuse without touching the running pair.
        conf.app.config.machines[0].fabric_aux.insert("b".into(), "another-aux-a".into());
        conf.app.runner.calls.clear();
        assert!(conf.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err().contains("different launch"));
        assert!(conf.app.runner.calls.is_empty());

        // One-sided fabric_aux is refused locally before any SSH call.
        let mut one_sided = Fixture::fp4(2, false);
        one_sided.app.config.machines[0].fabric_aux.insert("b".into(), "aux-only-a".into());
        assert!(one_sided.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err().contains("fabric_aux"));
        assert!(one_sided.app.runner.calls.is_empty());

        // Four-Spark FP4-Local chain keeps TP_AUX_* unset even when fabric_aux is present.
        let mut four = Fixture::fp4(4, false);
        for pair in [(0, 1, "a", "b"), (1, 2, "b", "c"), (2, 3, "c", "d")] {
            four.app.config.machines[pair.0].fabric_aux.insert(pair.3.into(), format!("aux-{}-{}", pair.2, pair.3));
            four.app.config.machines[pair.1].fabric_aux.insert(pair.2.into(), format!("aux-{}-{}", pair.3, pair.2));
        }
        four.app.up(&fp4_options(&["--sparks", "4", "--mtp", "0"])).unwrap();
        for req in starts(&four) {
            assert_eq!(req.get("env_unset.TP_AUX_LISTEN").map(String::as_str), Some("1"));
            assert!(!req.contains_key("env.TP_AUX_LISTEN"));
        }
        return;
    }
    for (case, dual) in [("default", None), ("default", Some("1")), ("optout", Some("0")), ("invalid", Some("maybe"))] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::two_spark::second_rail_follows_configured_fabric_aux_and_tp_dual_rail_opt_out", "--nocapture"]).env(CASE, case);
        match dual { Some(v) => command.env("TP_DUAL_RAIL", v), None => command.env_remove("TP_DUAL_RAIL") };
        command.env_remove("SEQPAR_RDMA");
        let result = command.output().unwrap();
        assert!(result.status.success(), "{case}: {}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }
}
