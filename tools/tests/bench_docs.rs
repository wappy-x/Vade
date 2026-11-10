use std::{fs,path::{Path,PathBuf},process::{Command,Output},sync::atomic::{AtomicUsize,Ordering},time::{SystemTime,UNIX_EPOCH}};
use vade_tools::json::{self,Value};
const START: &str = "<!-- bench-docs:start -->";
const END: &str = "<!-- bench-docs:end -->";
const CARD: &str = include_str!("fixtures/bench-docs/card.json");
const MTP_OFF: &str = include_str!("fixtures/bench-docs/mtp-off.card.json");
const SERVER_PLAIN: &str = include_str!("fixtures/bench-docs/server-plain.card.json");
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture { root: PathBuf, repo: PathBuf, cards: PathBuf, old: String, measured: String, other: String }
fn write(path: &Path, content: impl AsRef<[u8]>) { fs::create_dir_all(path.parent().unwrap()).unwrap(); fs::write(path,content).unwrap(); }
fn checked(command: &mut Command) -> String {
    let result = command.output().unwrap(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    String::from_utf8(result.stdout).unwrap().trim().into()
}
fn git(repo: &Path, args: &[&str]) -> String {
    checked(Command::new("git").arg("-C").arg(repo).args(["-c","core.hooksPath=/dev/null","-c","user.name=Test","-c","user.email=test@example.invalid","-c","commit.gpgsign=false"]).args(args))
}
fn set(value: &mut Value, path: &[&str], replacement: Value) {
    let Value::Object(fields) = value else { panic!("fixture field is not an object") };
    if path.len() == 1 { fields.insert(path[0].into(),replacement); }
    else { set(fields.get_mut(path[0]).expect("fixture field"),&path[1..],replacement); }
}
fn number(value: &str) -> Value { Value::Number(value.into()) }
fn text(value: &str) -> Value { Value::String(value.into()) }
impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = vade_tools::root().join(format!(".local/bench-doc-tests/{}-{nonce}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        let repo = root.join("repo"); let cards = root.join("cards"); fs::create_dir_all(&repo).unwrap(); fs::create_dir_all(&cards).unwrap();
        git(&repo,&["init","--initial-branch=main"]);
        let page = format!("# Keep this page\n\nvade up --sparks 2\nKeep the launch paragraph.\n\n{START}\nold table\n{END}\n\nKeep the end.\n");
        write(&repo.join("README.md"),&page); write(&repo.join("docs/benchmarks.md"),&page);
        git(&repo,&["add","."]); git(&repo,&["commit","-m","Initial docs"]); let old = git(&repo,&["rev-parse","HEAD"]);
        git(&repo,&["commit","--allow-empty","-m","Measured main"]); let measured = git(&repo,&["rev-parse","HEAD"]);
        git(&repo,&["switch","-c","other"]); git(&repo,&["commit","--allow-empty","-m","Unmerged change"]); let other = git(&repo,&["rev-parse","HEAD"]);
        git(&repo,&["switch","main"]); git(&repo,&["commit","--allow-empty","-m","Main without a card"]);
        Self {root,repo,cards,old,measured,other}
    }
    fn card(&self, name: &str, revision: &str, changes: &[(&[&str],Value)]) {
        self.card_from(CARD,name,revision,changes);
    }
    fn card_from(&self, source: &str, name: &str, revision: &str, changes: &[(&[&str],Value)]) {
        let mut value = json::parse(&source.replace("REV",revision)).unwrap();
        for (path,replacement) in changes { set(&mut value,path,replacement.clone()); }
        write(&self.cards.join(name),value.encode());
    }
    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_vade-tools")).args(["bench-docs"]).arg(&self.cards).arg(&self.repo).output().unwrap()
    }
    fn pages(&self) -> [Vec<u8>;2] { [fs::read(self.repo.join("README.md")).unwrap(),fs::read(self.repo.join("docs/benchmarks.md")).unwrap()] }
    fn times(&self) -> [SystemTime;2] { ["README.md","docs/benchmarks.md"].map(|p| fs::metadata(self.repo.join(p)).unwrap().modified().unwrap()) }
}
fn body(page: &str) -> &str { page.split_once(START).unwrap().1.split_once(END).unwrap().0 }
#[test] fn omitted_draft_with_zero_depth_keeps_on_and_off_cards_separate() {
    let f = Fixture::new(); f.card("on.card.json",&f.measured,&[]);
    f.card_from(MTP_OFF,"off.card.json",&f.measured,&[
        (&["run","finished_at"],text("2026-10-26T20:06:00+00:00")),
        (&["decode","prose","decode_tokens_per_second","median"],number("29.146")),
    ]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    for page in f.pages() {
        let page = String::from_utf8(page).unwrap();
        assert!(page.contains("| tensor split | ON | 54.153 |"));
        assert!(page.contains("| tensor split | OFF | 29.146 |"));
    }
}
#[test] fn omitted_request_mode_uses_plain_server_default() {
    let f = Fixture::new(); f.card_from(SERVER_PLAIN,"off.card.json",&f.measured,&[]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    for page in f.pages() {
        let page = String::from_utf8(page).unwrap();
        assert!(page.contains("| tensor split | OFF | 54.153 |")); assert!(!page.contains("| ON |"));
    }
}
#[test] fn positive_request_depth_overrides_plain_server_default() {
    for depth in ["1","3"] {
        let f = Fixture::new(); f.card_from(SERVER_PLAIN,"on.card.json",&f.measured,&[(&["settings","extra_body","mtp_depth"],number(depth))]);
        let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains("| tensor split | ON | 54.153 |"));
    }
}
#[test] fn plain_mode_or_zero_request_depth_disables_drafting() {
    for (mode,depth,server,expected) in [("plain","3","strict","OFF"),("strict","0","plain","OFF")] {
        let f = Fixture::new(); f.card("mode.card.json",&f.measured,&[
            (&["settings","extra_body","draft"],text(mode)),(&["settings","extra_body","mtp_depth"],number(depth)),
            (&["health","vade","numerics","draft"],text(server)),
        ]);
        let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains(&format!("| tensor split | {expected} | 54.153 |")));
    }
}
#[test] fn strict_with_zero_request_depth_keeps_on_and_off_cards_separate() {
    let f = Fixture::new(); f.card("on.card.json",&f.measured,&[]);
    f.card("off.card.json",&f.measured,&[
        (&["settings","extra_body","draft"],text("strict")),(&["settings","extra_body","mtp_depth"],number("0")),
        (&["run","finished_at"],text("2026-10-26T20:06:00+00:00")),
        (&["decode","prose","decode_tokens_per_second","median"],number("29.146")),
    ]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    for page in f.pages() {
        let page = String::from_utf8(page).unwrap();
        assert!(page.contains("| tensor split | ON | 54.153 |"));
        assert!(page.contains("| tensor split | OFF | 29.146 |"));
        assert!(!page.contains("| tensor split | ON | 29.146 |"));
    }
}
#[test] fn strict_with_inherited_zero_depth_disables_drafting() {
    for (explicit,depth,expected) in [(false,"0","OFF"),(true,"0","OFF"),(true,"3","ON")] {
        let f = Fixture::new();
        let mut changes = vec![(&["health","vade","numerics","mtp_depth"][..],number(depth))];
        if explicit { changes.push((&["settings","extra_body","draft"],text("strict"))); }
        f.card("off.card.json",&f.measured,&changes);
        let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        for page in f.pages() {
            let page = String::from_utf8(page).unwrap();
            assert!(page.contains(&format!("| tensor split | {expected} | 54.153 |")));
        }
    }
}
#[test] fn strict_with_positive_request_depth_overrides_plain_server_default() {
    for depth in ["1","2","3"] {
        let f = Fixture::new(); f.card_from(SERVER_PLAIN,"on.card.json",&f.measured,&[
            (&["settings","extra_body","draft"],text("strict")),(&["settings","extra_body","mtp_depth"],number(depth)),
        ]);
        let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains("| tensor split | ON | 54.153 |"));
    }
}
#[test] fn refuses_strict_with_uncertain_inherited_depth_before_writing_docs() {
    let no_depth = json::parse(r#"{"world_size":2,"topology":"pipeline","tensor_split":{},"draft":"strict"}"#).unwrap();
    let cases = [
        (vec!["health","vade","numerics"],no_depth),
        (vec!["health","vade","numerics","mtp_depth"],Value::Null),
        (vec!["health","vade","numerics","mtp_depth"],text("0")),
        (vec!["health","vade","numerics","mtp_depth"],number("-1")),
        (vec!["health","vade","numerics","mtp_depth"],number("1.5")),
        (vec!["health","vade","numerics","mtp_depth"],number("4")),
        (vec!["health","vade","numerics","draft"],text("plain")),
    ];
    for (path,value) in cases {
        let f = Fixture::new(); f.card("bad.card.json",&f.measured,&[
            (&["settings","extra_body","draft"],text("strict")),(&path,value),
        ]);
        let before = f.pages(); let times = f.times(); let result = f.run();
        assert!(!result.status.success()); assert_eq!(f.pages(),before); assert_eq!(f.times(),times);
        assert!(String::from_utf8_lossy(&result.stderr).contains("invalid card"));
    }
    let f = Fixture::new(); f.card_from(SERVER_PLAIN,"bad.card.json",&f.measured,&[(&["settings","extra_body","draft"],text("strict"))]);
    let before = f.pages(); let times = f.times(); let result = f.run();
    assert!(!result.status.success()); assert_eq!(f.pages(),before); assert_eq!(f.times(),times);
}
#[test] fn refuses_uncertain_or_invalid_drafting_before_writing_docs() {
    let no_mode = json::parse(r#"{"world_size":2,"topology":"pipeline","tensor_split":{},"mtp_depth":3}"#).unwrap();
    let cases = [
        (vec!["health","vade","numerics"],no_mode),
        (vec!["health","vade","numerics","draft"],Value::Null),
        (vec!["health","vade","numerics","draft"],text("auto")),
        (vec!["health","vade","numerics","draft"],text("unknown")),
        (vec!["settings","extra_body","draft"],Value::Null),
        (vec!["settings","extra_body","draft"],text("auto")),
        (vec!["settings","extra_body","mtp_depth"],Value::Null),
        (vec!["settings","extra_body","mtp_depth"],text("0")),
        (vec!["settings","extra_body","mtp_depth"],number("-1")),
        (vec!["settings","extra_body","mtp_depth"],number("1.5")),
        (vec!["settings","extra_body","mtp_depth"],number("4")),
    ];
    for (path,value) in cases {
        let f = Fixture::new(); f.card("bad.card.json",&f.measured,&[(&path,value)]);
        let before = f.pages(); let result = f.run(); assert!(!result.status.success()); assert_eq!(f.pages(),before);
        assert!(String::from_utf8_lossy(&result.stderr).contains("invalid card"));
    }
}
#[test] fn selects_measured_main_then_latest_card_per_setup_and_preserves_pages() {
    let f = Fixture::new();
    f.card("old.card.json",&f.old,&[(&["run","finished_at"],text("2026-10-29T20:05:00+00:00")),(&["decode","prose","decode_tokens_per_second","median"],number("901.001"))]);
    f.card("old-only-setup.card.json",&f.old,&[(&["health","vade","numerics","world_size"],number("1")),(&["decode","prose","decode_tokens_per_second","median"],number("905.005"))]);
    f.card("other.card.json",&f.other,&[(&["run","finished_at"],text("2026-10-30T20:05:00+00:00")),(&["decode","prose","decode_tokens_per_second","median"],number("902.002"))]);
    f.card("earlier.card.json",&f.measured,&[(&["run","finished_at"],text("2026-10-26T19:05:00+00:00")),(&["decode","prose","decode_tokens_per_second","median"],number("903.003"))]);
    f.card("nested/day/latest.card.json",&f.measured,&[]);
    f.card("nested/off.card.json",&f.measured,&[(&["settings","extra_body","draft"],text("plain")),(&["decode","prose","decode_tokens_per_second","median"],number("29.146"))]);
    f.card("pipeline.card.json",&f.measured,&[(&["health","vade","numerics","tensor_split"],Value::Null),(&["decode","prose","decode_tokens_per_second","median"],number("42.157"))]);
    write(&f.cards.join("not-a-card.json"),"not JSON");
    let before = f.pages(); let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let pages = f.pages(); let readme = String::from_utf8(pages[0].clone()).unwrap(); let benchmark = String::from_utf8(pages[1].clone()).unwrap();
    assert_ne!(body(&readme),body(&benchmark));
    assert!(body(&readme).contains("[benchmark history](docs/benchmarks.md)")); assert!(!body(&readme).contains("card SHA256")); assert!(!body(&readme).contains("bench-docs '<cards-dir>'"));
    assert!(body(&readme).contains("The last card finished on 2026-10-26. They do not measure later engine changes\non main."));
    for expected in ["2 Sparks, tensor split, drafting ON: finished","card SHA256","bench-docs '<cards-dir>'","These cards do not measure later engine changes on main."] { assert!(body(&benchmark).contains(expected),"missing {expected}"); }
    for page in [&readme,&benchmark] {
        for expected in ["54.153","100.924","130.372","75.389","149.833","176.303","2.788","29.146","42.157","example/model","2026-10-26",&f.measured] { assert!(page.contains(expected),"missing {expected} in {page}"); }
        for excluded in ["901.001","902.002","903.003","905.005","999.777"] { assert!(!page.contains(excluded),"included {excluded}"); }
    }
    for (old,new) in before.iter().zip(&pages) {
        let old = std::str::from_utf8(old).unwrap(); let new = std::str::from_utf8(new).unwrap();
        assert_eq!(old.split_once(START).unwrap().0,new.split_once(START).unwrap().0);
        assert_eq!(old.split_once(END).unwrap().1,new.split_once(END).unwrap().1);
    }
    let times = f.times(); let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    assert_eq!(f.pages(),pages); assert_eq!(f.times(),times); assert!(f.root.is_dir());
}
#[test] fn refuses_bad_markers_before_writing_either_page() {
    for invalid in ["No markers here.\n".to_string(),format!("{START}\na\n{START}\nb\n{END}\n"),format!("{END}\na\n{START}\n")] {
        let f = Fixture::new(); f.card("ok.card.json",&f.measured,&[]); write(&f.repo.join("docs/benchmarks.md"),invalid);
        let before = f.pages(); let result = f.run(); assert!(!result.status.success()); assert_eq!(f.pages(),before);
    }
}
#[test] fn refuses_invalid_selected_metrics_without_changing_docs() {
    for invalid in [Value::Null,text("12.0"),number("-1")] {
        let f = Fixture::new(); f.card("bad.card.json",&f.measured,&[(&["concurrency","8","aggregate_end_to_end_tokens_per_second","median"],invalid)]);
        let before = f.pages(); let result = f.run(); assert!(!result.status.success()); assert_eq!(f.pages(),before);
    }
}
#[test] fn ignores_incomplete_and_failed_cards_without_using_their_newer_times() {
    let f = Fixture::new(); f.card("good.card.json",&f.measured,&[]);
    for (name,field,value) in [("incomplete.card.json","complete",Value::Bool(false)),("failed.card.json","status",text("FAIL"))] {
        f.card(name,&f.measured,&[(&[field],value),(&["run","finished_at"],text("2026-10-27T20:05:00+00:00")),(&["decode","prose","decode_tokens_per_second","median"],number("904.004"))]);
    }
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains("54.153")); assert!(!page.contains("904.004"));
}
#[test] fn compares_rfc3339_offsets_and_subsecond_times() {
    let f = Fixture::new();
    for (name,time,prose) in [
        ("offset.card.json","2026-10-26T21:05:00.999+02:00","906.006"),
        ("fraction.card.json","2026-10-26T20:05:00.000999999Z","907.007"),
        ("latest.card.json","2026-10-26T15:05:00.001-05:00","54.153"),
    ] { f.card(name,&f.measured,&[(&["run","finished_at"],text(time)),(&["decode","prose","decode_tokens_per_second","median"],number(prose))]); }
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let page = String::from_utf8(f.pages()[0].clone()).unwrap();
    assert!(page.contains("54.153")); assert!(!page.contains("906.006")); assert!(!page.contains("907.007"));
}
#[test] fn equal_time_ties_use_card_content_hash_instead_of_directory_order() {
    let f = Fixture::new();
    f.card("a/one.card.json",&f.measured,&[(&["decode","prose","decode_tokens_per_second","median"],number("17.001"))]);
    f.card("z/two.card.json",&f.measured,&[(&["decode","prose","decode_tokens_per_second","median"],number("17.002"))]);
    let paths = [f.cards.join("a/one.card.json"),f.cards.join("z/two.card.json")];
    let sources = paths.each_ref().map(|p|fs::read(p).unwrap());
    let hashes = sources.each_ref().map(|s|vade_tools::sha256::digest(s));
    let expected = if hashes[0] < hashes[1] { "17.001" } else { "17.002" };
    let excluded = if hashes[0] < hashes[1] { "17.002" } else { "17.001" };
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let before = f.pages(); let page = String::from_utf8(before[0].clone()).unwrap(); assert!(page.contains(expected)); assert!(!page.contains(excluded));
    write(&paths[0],&sources[1]); write(&paths[1],&sources[0]);
    let times = f.times(); let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    assert_eq!(f.pages(),before); assert_eq!(f.times(),times);
}
#[test] fn no_matching_card_leaves_both_pages_unchanged() {
    let f = Fixture::new(); f.card("unmerged.card.json",&f.other,&[]);
    f.card("failed-main.card.json",&f.measured,&[(&["status"],text("FAIL"))]);
    let before = f.pages(); let result = f.run(); assert!(!result.status.success()); assert_eq!(f.pages(),before);
}
#[test] fn invalid_obsolete_cards_do_not_block_the_selected_card() {
    let f = Fixture::new();
    f.card("a-old-main.card.json",&f.old,&[
        (&["health","vade","numerics","world_size"],Value::Null),
        (&["decode","prose","decode_tokens_per_second","median"],number("-1")),
    ]);
    f.card("b-superseded.card.json",&f.measured,&[
        (&["run","finished_at"],text("2026-10-26T19:05:00+00:00")),
        (&["decode","prose","decode_tokens_per_second","median"],Value::Null),
    ]);
    f.card("z-selected.card.json",&f.measured,&[]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains("54.153")); assert!(page.contains(&f.measured));
}
#[test] fn id_coverage_counts_only_recorded_responses() {
    let f = Fixture::new();
    f.card("coverage.card.json",&f.measured,&[
        (&["decode","prose","runs"],json::parse("[{\"completion_tokens\":2}]").unwrap()),
        (&["token_ids"],json::parse("[1,2]").unwrap()),
    ]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let page = String::from_utf8(f.pages()[0].clone()).unwrap(); assert!(page.contains("6/7")); assert!(!page.contains("7/7"));
}
#[test] fn one_spark_card_uses_the_singular_noun() {
    let f = Fixture::new(); f.card("one.card.json",&f.measured,&[(&["health","vade","numerics","world_size"],number("1"))]);
    let result = f.run(); assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let page = String::from_utf8(f.pages()[1].clone()).unwrap();
    assert!(page.contains("\n1 Spark, tensor split, drafting ON: finished")); assert!(!page.contains("1 Sparks"));
}
