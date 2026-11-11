//! The card checklist: a complete, honest card folder passes every check; each broken receipt fails its own check.
//! Every test builds its own small git repository, so no test depends on the history depth of the checkout
//! (CI clones with depth 1).
use std::{fs, panic, path::{Path, PathBuf}, process::Command};
use vade_tools::card_qualify::{self, Options, CHECKS};

const GOLDENS: &str = env!("CARGO_BIN_EXE_check-goldens");
const TOOLS: &str = env!("CARGO_BIN_EXE_vade-tools");
const MODEL: &str = "glm-5.3-flash-fp4-local";
const KIT_SERVE: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const KIT_PTX: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const WEIGHT: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const FINGERPRINT: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const RECEIPT: &str = "{\"cell\":\"c15\",\"prompt_tokens\":15,\"output_ids\":[1,2,3],\"ok\":true}\n";
const SETTINGS: &str = r#""runs":5,"decode_tokens":8192,"temperature":0.0,"top_p":1.0,"seed":20260905"#;
const PLAIN: &str = r#"{"cache":false,"draft":"plain","mtp_depth":0,"reasoning_effort":"low"}"#;
const STRICT: &str = r#"{"cache":false,"draft":"strict","mtp_depth":2,"reasoning_effort":"low"}"#;

/// A throwaway repository on branch main: engine, CLI and harness source, docs embedded by the engine,
/// a saved bench receipt and the golden file.
struct Repo { dir: PathBuf }
impl Repo {
    fn new(name: &str, golden: &[u8]) -> Self {
        let dir = std::env::temp_dir().join(format!("vade-card-qualify-repo-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let r = Self { dir };
        r.git(&["init", "-q", "-b", "main"]);
        for (path, text) in [
            ("engine/src/lib.rs", "pub fn f() {}\n"),
            ("engine/src/chat.rs", "pub const T: &str = include_str!(\"../../docs/template.md\");\n"),
            ("engine/src/chat.jinja", "{{ messages }}\n"),
            ("engine/README.md", "# Engine\n"),
            ("engine/tools/bench/experiments/run-1/receipt.tsv", "a\t1\n"),
            ("docs/template.md", "template\n"),
            ("harness/src/lib.rs", "pub fn h() {}\n"),
            ("cli/src/main.rs", "fn main() {}\n"),
            ("tools/README.md", "tools\n"),
        ] { r.write(path, text.as_bytes()); }
        r.write(&format!("tools/goldens/{MODEL}.tsv"), golden);
        r.commit();
        r
    }
    fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git").arg("-C").arg(&self.dir)
            .args(["-c", "user.name=card-test", "-c", "user.email=card-test@example.invalid", "-c", "commit.gpgsign=false"])
            .args(args).output().unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }
    fn write(&self, path: &str, bytes: &[u8]) {
        let p = self.dir.join(path);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, bytes).unwrap();
    }
    /// Commit everything and return the new head.
    fn commit(&self) -> String { self.git(&["add", "-A"]); self.git(&["commit", "-q", "-m", "change"]); self.git(&["rev-parse", "HEAD"]) }
    fn change(&self, path: &str, text: &str) -> String { self.write(path, text.as_bytes()); self.commit() }
}
impl Drop for Repo { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.dir); } }

fn hash(i: usize) -> String { format!("{i:064x}") }
fn run(i: usize, rate: f64, finish: &str, out: &str) -> String {
    let head = format!(r#""run":{i},"finish_reason":"{finish}","stream_done_marker":true,"completion_validation":{{"valid":true}}"#);
    format!(r#"{{{head},"decode_tokens_per_second":{rate},"output_sha256":"{out}","reasoning_sha256":"{}","cached_prompt_tokens":0}}"#, hash(0))
}
/// One workload row from (run index, rate, answer hash) triples.
fn row(rows: &[(usize, f64, String)], finish: &str) -> String {
    let runs: Vec<String> = rows.iter().map(|(i, r, h)| run(*i, *r, finish, h)).collect();
    let mut s: Vec<f64> = rows.iter().map(|r| r.1).collect(); s.sort_by(f64::total_cmp);
    format!(r#"{{"runs":[{}],"completion_gate":{{"passed":5,"total":5}},"decode_tokens_per_second":{{"median":{}}}}}"#, runs.join(","), s[s.len() / 2])
}
fn workload(rates: [f64; 5], finish: &str, base: usize) -> String {
    row(&rates.iter().enumerate().map(|(i, r)| (i + 1, *r, hash(base + i))).collect::<Vec<_>>(), finish)
}
fn decode(finish: &str) -> String { scaled(finish, 0.) }
/// The same answers at another speed: `extra` tok/s added to every code and structured run.
fn scaled(finish: &str, extra: f64) -> String {
    let up = |r: [f64; 5]| r.map(|x| x + extra);
    format!(r#"{{"prose":{},"code":{},"structured":{}}}"#, workload([19.8, 19.9, 19.85, 19.86, 19.84], finish, 100),
        workload(up([19.2, 19.3, 19.25, 19.26, 19.1]), finish, 200), workload(up([20.1, 20.2, 20.15, 20.16, 20.0]), finish, 300))
}
fn prefill(cached: u32) -> String {
    let r = |i: usize, t: f64| format!(concat!(r#"{{"run":{},"ttft_seconds":{},"effective_prefill_tokens_per_second":{},"#,
        r#""cached_prompt_tokens":{},"requested_prompt_tokens":8192}}"#), i, t, 8192. / t, cached);
    let rates = [8192. / 69.3, 8192. / 69.2, 8192. / 69.4];
    let mut s = rates.to_vec(); s.sort_by(f64::total_cmp);
    format!(concat!(r#"{{"8192":{{"cold":{{"runs":[{},{},{}],"ttft_seconds":{{"median":69.3}},"#,
        r#""effective_prefill_tokens_per_second":{{"median":{:.3}}}}}}}}}"#), r(1, 69.3), r(2, 69.2), r(3, 69.4), s[1])
}
fn concurrency(rounds: usize) -> String {
    let cell = |c: usize| {
        let stream = r#"{"stream_done_marker":true,"finish_reason":"length"}"#;
        let rs: Vec<String> = (0..rounds).map(|i| format!(r#"{{"round":{},"streams":[{}],"aggregate_end_to_end_tokens_per_second":{}}}"#,
            i + 1, vec![stream; c].join(","), 18.0 + i as f64 / 100.)).collect();
        let mut v: Vec<f64> = (0..rounds).map(|i| 18.0 + i as f64 / 100.).collect(); v.sort_by(f64::total_cmp);
        let m = if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2. };
        format!(r#""{c}":{{"rounds":[{}],"aggregate_end_to_end_tokens_per_second":{{"median":{:.3}}}}}"#, rs.join(","), m)
    };
    format!("{{{},{},{},{}}}", cell(1), cell(2), cell(4), cell(8))
}
/// `json` with one more key added to its outer object.
fn plus(json: &str, extra: &str) -> String { format!("{},{extra}}}", &json[..json.len() - 1]) }

struct Card {
    engine: String, label: String, body: String, settings: String, drafter: String, revision: String,
    decode: String, prefill: String, concurrency: String,
}
impl Card {
    fn new(engine: &str) -> Self {
        Self { engine: engine.into(), label: "glm-card-2s".into(), body: PLAIN.into(), settings: SETTINGS.into(), drafter: "off".into(),
            revision: FINGERPRINT.into(), decode: decode("stop"), prefill: prefill(0), concurrency: concurrency(3) }
    }
    /// The drafting part: same prompts and settings, drafting on, faster code and structured rows.
    fn draft(engine: &str) -> Self {
        let mut d = Self::new(engine);
        d.body = STRICT.into(); d.drafter = "on".into(); d.decode = scaled("stop", 4.0);
        d.prefill = "{}".into(); d.concurrency = "{}".into();
        d
    }
    fn json(&self) -> String {
        let tool = r#"{"revision":"946394fa53db83a6865f6e3ae56594e1c299fb2a","dirty":false}"#;
        let appliance = format!(concat!(r#"{{"hardware":"2 NVIDIA DGX Spark GB10; a,b","serving_engine":"Vade {}","purpose":"benchmark","#,
            r#""topology":"ring","drafter":"{}","model_revision":"{}"}}"#), self.engine, self.drafter, self.revision);
        let ids = format!(r#""model":"{MODEL}","label":"{}","comparison_id":"rigmark-{}""#, self.label, self.engine);
        let run = format!(r#"{{{ids},"started_at":"2026-11-11T09:00:00+00:00","rigmark_release_check":{tool},"appliance":{appliance}}}"#);
        let settings = format!(r#"{{{},"extra_body":{}}}"#, self.settings, self.body);
        format!(r#"{{"schema":1,"protocol":"1.3.0","run":{run},"settings":{settings},"decode":{},"prefill":{},"concurrency":{}}}"#,
            self.decode, self.prefill, self.concurrency)
    }
}

struct Folder { dir: PathBuf, engine: String, repo: Repo }
impl Folder {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("vade-card-qualify-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("goldens")).unwrap();
        fs::write(dir.join("goldens/load1.jsonl"), RECEIPT).unwrap();
        let made = Command::new(GOLDENS).args(["--make", MODEL, "1"]).arg(dir.join("goldens/load1.jsonl")).output().unwrap();
        assert!(made.status.success(), "{}", String::from_utf8_lossy(&made.stderr));
        fs::write(dir.join("golden.tsv"), &made.stdout).unwrap();
        let repo = Repo::new(name, &made.stdout);
        let engine = repo.git(&["rev-parse", "HEAD"]);
        let f = Self { dir, engine, repo };
        f.put("card.json", &Card::new(&f.engine).json());
        f.put("card.card.txt", "| 15/15 BASIC OUTPUT GATES PASSED |\n");
        f.put("draft.json", &Card::draft(&f.engine).json());
        f.put("draft.card.txt", "| 15/15 BASIC OUTPUT GATES PASSED |\n");
        f.put("health.json", &f.health(r#""numerics":{"world_size":2}"#));
        f.put("SHA256SUMS", &format!("{KIT_SERVE}  flash-serve\n{KIT_PTX}  flash.ptx\n"));
        f.put("live-binary-a.txt", &format!("{KIT_SERVE}  /proc/1/exe\n"));
        f.put("live-binary-b.txt", &format!("{KIT_SERVE}  /proc/2/exe\n"));
        let weights = format!("{WEIGHT}  /m/model-00001-of-00001.safetensors\n{FINGERPRINT}  /m/config.json\n");
        f.put("checkpoint-sha256-a.txt", &weights);
        f.put("checkpoint-sha256-b.txt", &weights);
        f.put("quality-vs-maker.result", &f.quality("PASS (line written before data)", &f.engine));
        f
    }
    fn health(&self, numerics: &str) -> String {
        let identity = format!(r#""checkpoint_sha256":null,"checkpoint_fingerprint":"{FINGERPRINT}",{numerics}"#);
        format!(r#"{{"status":"ready","vade":{{"build":{{"git_sha":"{}","dirty":false}},"qualification":"unqualified",{identity}}}}}"#, self.engine)
    }
    fn quality(&self, verdict: &str, engine: &str) -> String {
        format!("RESULT-CARD BEGIN\nVERDICT: {verdict}\nENGINE: {engine}\nRESULT-CARD END\n")
    }
    fn put(&self, name: &str, text: &str) { fs::write(self.dir.join(name), text).unwrap(); }
    fn get(&self, name: &str) -> String { fs::read_to_string(self.dir.join(name)).unwrap() }
    fn edit(&self, name: &str, from: &str, to: &str) {
        let t = self.get(name);
        assert!(t.contains(from), "{name} has no {from}");
        self.put(name, &t.replace(from, to));
    }
    fn options(&self) -> Options {
        Options { dir: self.dir.clone(), repo: self.repo.dir.clone(), main_ref: "main".into(), goldens: Some(self.dir.join("golden.tsv")),
            goldens_bin: GOLDENS.into() }
    }
    fn failed(&self) -> Vec<&'static str> { failed_with(&self.options()) }
}
impl Drop for Folder { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.dir); } }
fn failed_with(options: &Options) -> Vec<&'static str> {
    card_qualify::check(options).unwrap().checks.iter().filter(|c| !c.pass).map(|c| c.id).collect()
}
fn card(f: &Folder, edit: impl FnOnce(&mut Card)) { let mut c = Card::new(&f.engine); edit(&mut c); f.put("card.json", &c.json()); }
fn draft(f: &Folder, edit: impl FnOnce(&mut Card)) { let mut c = Card::draft(&f.engine); edit(&mut c); f.put("draft.json", &c.json()); }
fn check_of(options: &Options, id: &str) -> card_qualify::Check {
    card_qualify::check(options).unwrap().checks.into_iter().find(|c| c.id == id).unwrap()
}

type Break = Box<dyn Fn(&Folder)>;
/// Each case breaks one fresh folder and names the checks that must fail. Every case runs; a panic counts as a miss.
fn expect(cases: Vec<(&str, &[&str], Break)>) {
    let mut misses = Vec::new();
    for (name, want, broke) in cases {
        let f = Folder::new(name);
        broke(&f);
        match panic::catch_unwind(panic::AssertUnwindSafe(|| f.failed())) {
            Ok(got) if got == want => (),
            Ok(got) => misses.push(format!("{name}: want {want:?}, got {got:?}")),
            Err(_) => misses.push(format!("{name}: want {want:?}, the checklist panicked")),
        }
    }
    assert!(misses.is_empty(), "{} case(s) missed:\n{}", misses.len(), misses.join("\n"));
}

#[test]
fn a_complete_card_passes_every_check() {
    let f = Folder::new("pass");
    let report = card_qualify::check(&f.options()).unwrap();
    let text = report.render();
    assert!(report.passed(), "{text}");
    assert_eq!(report.checks.len(), CHECKS.len());
    assert!(text.lines().last().unwrap().starts_with("CARD-QUALIFY PASS 12/12 engine="), "{text}");
    assert!(text.contains("read card.json sha256="), "every check names the receipts it read: {text}");
    assert!(text.contains("code 23.250 (+20.8% vs plain)"), "the drafting row prints its speed against plain: {text}");
}

#[test]
fn each_missing_or_wrong_receipt_fails_its_own_check() {
    expect(vec![
        ("no-draft", &["Q09"], Box::new(|f| fs::remove_file(f.dir.join("draft.json")).unwrap())),
        ("draft-differs", &["Q09"], Box::new(|f| f.edit("draft.json", &hash(101), &hash(999)))),
        ("quality-other", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result", &f.quality("PASS", &"a".repeat(40))))),
        ("quality-miss", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result", &f.quality("NOT MET", &f.engine)))),
        ("no-quality", &["Q11"], Box::new(|f| fs::remove_file(f.dir.join("quality-vs-maker.result")).unwrap())),
        ("live-differs", &["Q03"], Box::new(|f| f.put("live-binary-b.txt", &format!("{KIT_PTX}  /proc/2/exe\n")))),
        ("one-rank", &["Q03"], Box::new(|f| fs::remove_file(f.dir.join("live-binary-b.txt")).unwrap())),
        ("dirty", &["Q03"], Box::new(|f| f.edit("health.json", "\"dirty\":false", "\"dirty\":true"))),
        ("weights-differ", &["Q04"], Box::new(|f| f.put("checkpoint-sha256-b.txt", &format!("{KIT_PTX}  /m/model-00001-of-00001.safetensors\n")))),
        ("ids-differ", &["Q10"], Box::new(|f| f.put("goldens/load1.jsonl", &RECEIPT.replace("[1,2,3]", "[1,2,4]")))),
        ("no-gates", &["Q10"], Box::new(|f| fs::remove_file(f.dir.join("goldens/load1.jsonl")).unwrap())),
        ("cut-answers", &["Q05"], Box::new(|f| card(f, |c| c.decode = decode("length")))),
        ("cached-read", &["Q06"], Box::new(|f| card(f, |c| c.prefill = prefill(8192)))),
        ("two-rounds", &["Q07"], Box::new(|f| card(f, |c| c.concurrency = concurrency(2)))),
        ("max-effort", &["Q08", "Q09"], Box::new(|f| card(f, |c| c.body = r#"{"cache":false,"draft":"plain","mtp_depth":0}"#.into()))),
        ("fixture-label", &["Q12"], Box::new(|f| card(f, |c| c.label = "loopback fixture rates".into()))),
    ]);
}

/// Review finding 1: the drafting row counts only with real drafting (strict, MTP depth 1 or more, drafter on),
/// the plain settings, natural stops, all gates, real answer hashes and one run of each index.
#[test]
fn the_drafting_row_needs_real_drafting_real_hashes_and_distinct_runs() {
    let copies = |base: usize| row(&(0..5).map(|_| (1, 20.0, hash(base))).collect::<Vec<_>>(), "stop");
    expect(vec![
        ("plain-with-depth", &["Q08", "Q09"], Box::new(|f| draft(f, |d| d.body = d.body.replace("\"strict\"", "\"plain\"")))),
        ("strict-depth-0", &["Q08", "Q09"], Box::new(|f| draft(f, |d| d.body = d.body.replace("\"mtp_depth\":2", "\"mtp_depth\":0")))),
        ("drafter-off", &["Q08", "Q09"], Box::new(|f| draft(f, |d| d.drafter = "off".into()))),
        ("draft-cached", &["Q09"], Box::new(|f| draft(f, |d| d.body = d.body.replace("\"cache\":false", "\"cache\":true")))),
        ("draft-seed", &["Q09"], Box::new(|f| draft(f, |d| d.settings = d.settings.replace("20260905", "7")))),
        ("draft-warm", &["Q09"], Box::new(|f| draft(f, |d| d.settings = d.settings.replace("\"temperature\":0.0", "\"temperature\":0.7")))),
        ("draft-short-cap", &["Q09"], Box::new(|f| draft(f, |d| d.settings = d.settings.replace("8192", "1024")))),
        ("draft-cut", &["Q09"], Box::new(|f| draft(f, |d| d.decode = scaled("length", 4.0)))),
        ("draft-gates-text", &["Q09"], Box::new(|f| f.put("draft.card.txt", "| 14/15 BASIC OUTPUT GATES PASSED |\n"))),
        ("null-hashes", &["Q09"], Box::new(|f| for name in ["card.json", "draft.json"] {
            let mut t = f.get(name);
            for i in (100..105).chain(200..205).chain(300..305) { t = t.replace(&format!("\"{}\"", hash(i)), "null"); }
            f.put(name, &t);
        })),
        ("five-copies-of-run-1", &["Q09"], Box::new(move |f| draft(f, |d| {
            d.decode = format!(r#"{{"prose":{},"code":{},"structured":{}}}"#, copies(100), copies(200), copies(300));
        }))),
    ]);
}

/// Review findings 4 and 6: corrupt, truncated, empty or short receipts and missing fields fail; nothing panics.
#[test]
fn corrupt_empty_short_or_missing_receipts_fail_and_never_panic() {
    let odd = format!("a{}", "é".repeat(40));
    expect(vec![
        ("broken-json", &["Q01"], Box::new(|f| f.put("card-2.json", "{\"run\": {\"model\""))),
        ("not-utf8-json", &["Q01"], Box::new(|f| fs::write(f.dir.join("card-2.json"), b"{\"run\":\"\xff\"}").unwrap())),
        ("quality-truncated", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result", f.quality("PASS", &f.engine).trim_end_matches("RESULT-CARD END\n")))),
        ("quality-two-blocks", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result", &(f.quality("PASS", &f.engine) + &f.quality("PASS", &f.engine))))),
        ("quality-passable", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result", &f.quality("PASSABLE", &f.engine)))),
        ("quality-other-model", &["Q11"], Box::new(|f| f.put("quality-vs-maker.result",
            &f.quality("PASS", &f.engine).replace("RESULT-CARD END", "MODEL: other-model\nRESULT-CARD END")))),
        ("empty-draft-text", &["Q01", "Q09"], Box::new(|f| f.put("draft.card.txt", ""))),
        ("empty-gate-receipt", &["Q10"], Box::new(|f| f.put("goldens/load2.jsonl", ""))),
        ("no-world-size", &["Q12"], Box::new(|f| f.put("health.json", &f.health(r#""numerics":{}"#)))),
        ("no-requested-tokens", &["Q06"], Box::new(|f| card(f, |c| c.prefill = prefill(0).replace(",\"requested_prompt_tokens\":8192", "")))),
        ("no-seed", &["Q08", "Q09"], Box::new(|f| {
            card(f, |c| c.settings = c.settings.replace(",\"seed\":20260905", ""));
            draft(f, |d| d.settings = d.settings.replace(",\"seed\":20260905", ""));
        })),
        ("median-without-runs", &["Q12"], Box::new(|f| {
            let mut c = Card::new(&f.engine);
            c.decode = r#"{"prose":{"runs":[],"decode_tokens_per_second":{"median":29.0}}}"#.into();
            c.prefill = "{}".into(); c.concurrency = "{}".into();
            f.put("card-3.json", &c.json()); f.put("card-3.card.txt", "| 0/0 BASIC OUTPUT GATES PASSED |\n");
        })),
        ("prefill-32k-median", &["Q12"], Box::new(|f| card(f, |c| c.prefill = plus(&prefill(0), concat!(r#""32768":{"cold":{"runs":[{"run":1,"ttft_seconds":200.0,"#,
            r#""effective_prefill_tokens_per_second":163.84,"cached_prompt_tokens":0}],"ttft_seconds":{"median":150.0}}}"#))))),
        ("concurrency-16-median", &["Q12"], Box::new(|f| card(f, |c| c.concurrency = plus(&concurrency(3), concat!(r#""16":{"rounds":[{"round":1,"streams":[],"#,
            r#""aggregate_end_to_end_tokens_per_second":18.0}],"aggregate_end_to_end_tokens_per_second":{"median":30.0}}"#))))),
        ("live-hash-odd-bytes", &["Q03"], { let o = odd.clone(); Box::new(move |f| f.put("live-binary-b.txt", &format!("{o}  /proc/2/exe\n"))) }),
        ("kit-hash-short", &["Q03"], Box::new(|f| f.put("SHA256SUMS", &format!("abc  flash-serve\n{KIT_PTX}  flash.ptx\n")))),
        ("weights-odd-bytes", &["Q04"], { let o = odd.clone(); Box::new(move |f| for r in ["a", "b"] {
            f.put(&format!("checkpoint-sha256-{r}.txt"), &format!("{o}  m.safetensors\n{WEIGHT}  /m/model-00001-of-00001.safetensors\n"));
        }) }),
        ("weights-65-hex", &["Q04"], Box::new(|f| for r in ["a", "b"] {
            f.put(&format!("checkpoint-sha256-{r}.txt"), &format!("{WEIGHT}3  /m/model-00001-of-00001.safetensors\n"));
        })),
        ("weights-no-name", &["Q04"], Box::new(|f| for r in ["a", "b"] {
            f.put(&format!("checkpoint-sha256-{r}.txt"), &format!("{WEIGHT}  /m/model-00001-of-00001.safetensors\n{WEIGHT}\n"));
        })),
        ("weights-one-of-two-shards", &["Q04"], Box::new(|f| for r in ["a", "b"] {
            f.put(&format!("checkpoint-sha256-{r}.txt"), &format!("{WEIGHT}  /m/model-00001-of-00002.safetensors\n"));
        })),
        ("draft-revision", &["Q03"], Box::new(|f| draft(f, |d| d.revision = "5".repeat(64)))),
        ("gates-text-after-accent", &[], Box::new(|f| f.put("card.card.txt", "| é15/15 BASIC OUTPUT GATES PASSED |\n"))),
    ]);
}

/// Review finding 2: the digest binds the golden file and the main commit as well as the card folder.
#[test]
fn the_digest_binds_the_golden_file_and_the_main_commit() {
    let f = Folder::new("digest");
    let digest = |o: &Options| card_qualify::check(o).unwrap().digest;
    let first = digest(&f.options());
    assert_eq!(first, digest(&f.options()), "same inputs, same digest");
    let golden = f.get("golden.tsv");
    f.put("golden.tsv", &format!("{golden}\n"));
    assert_ne!(first, digest(&f.options()), "an edited --goldens file moves the digest");
    f.put("golden.tsv", &golden);
    assert_eq!(first, digest(&f.options()));
    f.repo.change("tools/README.md", "tools, changed\n");
    let moved = digest(&f.options());
    assert_ne!(first, moved, "a new main commit moves the digest");
    let mut git = f.options(); git.goldens = None;
    let from_git = digest(&git);
    f.repo.change(&format!("tools/goldens/{MODEL}.tsv"), &format!("{golden}\n"));
    assert_ne!(from_git, digest(&git), "a changed golden file on main moves the digest");
}

/// Review finding 3: Q11 binds on build inputs, and the test makes its own history (no HEAD~1 of the checkout).
#[test]
fn a_quality_receipt_binds_only_across_commits_with_the_same_build_inputs() {
    let cases: [(&str, &str, bool); 7] = [
        ("engine/README.md", "# Engine, reworded\n", true),
        ("tools/README.md", "tools, reworded\n", true),
        ("engine/tools/bench/experiments/run-1/receipt.tsv", "a\t2\n", true),
        ("engine/src/lib.rs", "pub fn f() { let _ = 1; }\n", false),
        ("engine/src/chat.jinja", "{{ messages | trim }}\n", false),
        ("docs/template.md", "template, embedded by the engine\n", false),
        ("harness/src/lib.rs", "pub fn h() { let _ = 2; }\n", false),
    ];
    let mut misses = Vec::new();
    for (i, (path, text, binds)) in cases.into_iter().enumerate() {
        let f = Folder::new(&format!("quality-tree-{i}"));
        let old = f.engine.clone();
        let new = f.repo.change(path, text);
        card(&f, |c| c.engine = new.clone());
        draft(&f, |d| d.engine = new.clone());
        let health = f.get("health.json").replace(&old, &new);
        f.put("health.json", &health);
        f.put("quality-vs-maker.result", &f.quality("PASS", &old));
        let got = f.failed();
        if got.is_empty() != binds { misses.push(format!("{path}: binds should be {binds}, failed {got:?}")); }
    }
    assert!(misses.is_empty(), "{}", misses.join("\n"));
}

#[test]
fn build_inputs_are_engine_cli_and_harness_files_not_docs_or_saved_receipts() {
    for path in ["engine/crates/tp/src/lib.rs", "engine/Cargo.lock", "cli/Cargo.toml", "engine/crates/gpu/kernels/x.ptx", "rust-toolchain.toml",
        "engine/.cargo/config.toml", "engine/crates/serve/src/tokenizer/glm.jinja", "harness/src/lib.rs", "engine/crates/model/data/table.json"] {
        assert!(card_qualify::build_input(path), "{path}");
    }
    for path in ["engine/README.md", "engine/tools/bench/experiments/head-split-3/SHA256SUMS",
        "engine/tools/bench/experiments/head-split-3/receipts/census-order.tsv", "engine/tools/bench/experiments/head-split-3/source.sha256",
        "cli/docs/usage.md", "tools/src/card_qualify.rs", "CHANGELOG.md"] {
        assert!(!card_qualify::build_input(path), "{path}");
    }
}

/// Review finding 7: one resolved main commit, no reading the git index, failure text from check-goldens, the card's model.
#[test]
fn the_golden_file_comes_from_the_resolved_main_commit() {
    let f = Folder::new("git-goldens");
    let mut options = f.options(); options.goldens = None;
    let q10 = check_of(&options, "Q10");
    assert!(q10.pass && q10.receipts.iter().any(|r| r.contains(&format!(":tools/goldens/{MODEL}.tsv sha256="))), "{q10:?}");
    options.main_ref = "no-such-ref".into();
    let q10 = check_of(&options, "Q10");
    assert!(!q10.pass && q10.detail.contains("no-such-ref"), "{q10:?}");
    options.main_ref = String::new();
    let (q02, q10) = (check_of(&options, "Q02"), check_of(&options, "Q10"));
    assert!(!q02.pass && !q10.pass, "an empty ref must not read the git index: {q10:?}");
    f.put("goldens/load1.jsonl", &RECEIPT.replace("[1,2,3]", "[1,2,4]"));
    let q10 = check_of(&f.options(), "Q10");
    assert!(!q10.pass && q10.detail.contains("first difference"), "the check-goldens failure line is shown: {q10:?}");
    f.put("goldens/load1.jsonl", RECEIPT);
    let other = Command::new(GOLDENS).args(["--make", "other-model", "1"]).arg(f.dir.join("goldens/load1.jsonl")).output().unwrap();
    f.put("golden.tsv", &String::from_utf8(other.stdout).unwrap());
    let q10 = check_of(&f.options(), "Q10");
    assert!(!q10.pass, "a golden file for another model does not pass: {q10:?}");
}

#[test]
fn a_relabeled_rate_fails_the_row_and_the_relabel_checks() {
    let f = Folder::new("relabel");
    f.edit("card.json", "\"decode_tokens_per_second\":{\"median\":19.85}", "\"decode_tokens_per_second\":{\"median\":29.85}");
    assert_eq!(f.failed(), vec!["Q05", "Q12"]);
}

#[test]
fn a_commit_off_main_fails() {
    let f = Folder::new("offmain");
    let other = "b".repeat(40);
    card(&f, |c| c.engine = other.clone());
    let failed = f.failed();
    assert!(failed.contains(&"Q02") && failed.contains(&"Q01"), "{failed:?}");
}

#[test]
fn the_command_prints_the_checklist_and_exits_one_unless_qualified() {
    let f = Folder::new("cli");
    let args = |f: &Folder| {
        let mut c = Command::new(TOOLS);
        c.arg("qualify-card").arg(&f.dir).args(["--main-ref", "main", "--goldens-bin", GOLDENS]).arg("--repo").arg(&f.repo.dir)
            .arg("--goldens").arg(f.dir.join("golden.tsv"));
        c
    };
    let ok = args(&f).output().unwrap();
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stdout));
    fs::remove_file(f.dir.join("quality-vs-maker.result")).unwrap();
    let bad = args(&f).output().unwrap();
    let out = String::from_utf8_lossy(&bad.stdout);
    assert_eq!(bad.status.code(), Some(1));
    assert!(out.contains("FAIL Q11 ") && out.contains("CARD-QUALIFY FAIL 11/12"), "{out}");
}

#[test]
fn card_parts_are_one_card() {
    // Section parts of one comparison id (decode, prefill, chats) qualify like one file.
    let f = Folder::new("parts");
    let mut d = Card::new(&f.engine); d.prefill = "{}".into(); d.concurrency = "{}".into();
    let mut p = Card::new(&f.engine); p.decode = "{}".into(); p.concurrency = "{}".into();
    let mut c = Card::new(&f.engine); c.decode = "{}".into(); c.prefill = "{}".into();
    fs::remove_file(f.dir.join("card.json")).unwrap();
    for (n, x) in [("part-decode", d), ("part-prefill8", p), ("part-concurrency", c)] {
        f.put(&format!("{n}.json"), &x.json());
        f.put(&format!("{n}.card.txt"), if n == "part-decode" { "15/15 BASIC OUTPUT GATES PASSED" } else { "0/3 BASIC OUTPUT GATES PASSED" });
    }
    assert_eq!(f.failed(), Vec::<&str>::new());
}

/// Review finding 8: the files of this tool keep every line at 200 characters or fewer.
#[test]
fn the_tool_files_keep_lines_under_200_characters() {
    let root = vade_tools::root();
    for file in ["tools/src/card_qualify.rs", "tools/tests/card_qualify.rs", "tools/src/main.rs", "tools/README.md"] {
        let text = fs::read_to_string(Path::new(&root).join(file)).unwrap();
        for (i, line) in text.lines().enumerate() {
            assert!(line.chars().count() <= 200, "{file}:{} has {} characters", i + 1, line.chars().count());
        }
    }
}
