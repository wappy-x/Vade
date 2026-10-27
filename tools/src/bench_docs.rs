//! Render the latest measured main revision from local RigMark cards.
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},process::Command};
use crate::{Result,json::{self,Value},sha256};

const START: &str = "<!-- bench-docs:start -->";
const END: &str = "<!-- bench-docs:end -->";
type Key = (u64,String,bool);
struct Card {
    key: Key, model: String, sha: String, finished: String, time: (i64,u32),
    metrics: Vec<String>, ids: u64, responses: u64, hash: String,
}
fn at<'a>(mut value: &'a Value, keys: &[&str]) -> &'a Value {
    for key in keys { value = value.get(key); } value
}
fn text(value: &Value, keys: &[&str]) -> Result<String> {
    match at(value,keys) { Value::String(s) if !s.is_empty() => Ok(s.clone()),
        _ => Err(format!("missing card text: {}",keys.join("." )).into()) }
}
fn count(value: &Value, keys: &[&str]) -> Result<u64> {
    match at(value,keys) { Value::Number(s) => Ok(s.parse()?),
        _ => Err(format!("missing card count: {}",keys.join(".")).into()) }
}
fn metric(value: &Value, keys: &[&str]) -> Result<String> {
    if let Value::Number(s) = at(value,keys) {
        let n: f64 = s.parse()?;
        if n.is_finite() && n >= 0. { return Ok(s.clone()); }
    }
    Err(format!("invalid card metric: {}",keys.join(".")).into())
}
fn stamp(s: &str) -> Result<(i64,u32)> {
    let invalid = || "invalid card run.finished_at; use RFC3339";
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return Err(invalid().into());
    }
    let number = |start: usize,end: usize| -> Result<i64> {
        let part = b.get(start..end).ok_or_else(invalid)?;
        if !part.iter().all(u8::is_ascii_digit) { return Err(invalid().into()); }
        Ok(std::str::from_utf8(part)?.parse()?)
    };
    let (year,month,day,hour,minute,second) = (number(0,4)?,number(5,7)?,number(8,10)?,number(11,13)?,number(14,16)?,number(17,19)?);
    let leap = year%4 == 0 && (year%100 != 0 || year%400 == 0);
    let months = [31,if leap {29} else {28},31,30,31,30,31,31,30,31,30,31];
    if year < 1 || !(1..=12).contains(&month) || day < 1 || day > months[(month-1) as usize]
        || hour > 23 || minute > 59 || second > 59 { return Err(invalid().into()); }
    let mut i = 19; let mut nanos = 0;
    if b.get(i) == Some(&b'.') {
        i += 1; let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) { i += 1; }
        let digits = i-start;
        if !(1..=9).contains(&digits) { return Err(invalid().into()); }
        nanos = number(start,i)? as u32 * 10u32.pow(9-digits as u32);
    }
    let offset = match b.get(i) {
        Some(b'Z') if i+1 == b.len() => 0,
        Some(sign @ (b'+'|b'-')) if i+6 == b.len() && b[i+3] == b':' => {
            let h = number(i+1,i+3)?; let m = number(i+4,i+6)?;
            if h > 23 || m > 59 { return Err(invalid().into()); }
            (h*3600+m*60) * if *sign == b'+' {1} else {-1}
        }, _ => return Err(invalid().into()),
    };
    let before = |y: i64| { let n = y-1; n*365+n/4-n/100+n/400 };
    let days = before(year)-before(1970)+months[..(month-1) as usize].iter().sum::<i64>()+day-1;
    Ok((days*86400+hour*3600+minute*60+second-offset,nanos))
}
fn ids_in_runs(runs: &Value) -> u64 {
    let Value::Array(runs) = runs else { return 0; };
    runs.iter().filter(|run| matches!(run.get("token_ids"),Value::Array(a) if !a.is_empty()
        && a.iter().all(|n| matches!(n,Value::Number(s) if s.parse::<u64>().is_ok())))).count() as u64
}
fn saved_ids(value: &Value) -> u64 {
    let mut ids = 0;
    if let Value::Object(workloads) = value.get("decode") {
        for workload in workloads.values() { ids += ids_in_runs(workload.get("runs")); }
    }
    if let Value::Object(cohorts) = value.get("concurrency") {
        for cohort in cohorts.values() { if let Value::Array(rounds) = cohort.get("rounds") {
            for round in rounds { ids += ids_in_runs(round.get("streams")); }
        } }
    }
    if let Value::Object(depths) = value.get("prefill") {
        for depth in depths.values() { for kind in ["cold","warm_replay"] { ids += ids_in_runs(depth.get(kind).get("runs")); } }
    }
    ids
}
fn draft_mode(value: &Value, field: &str) -> Result<bool> {
    match value {
        Value::String(s) if s == "plain" => Ok(false),
        Value::String(s) if s == "strict" => Ok(true),
        _ => Err(format!("unsupported card {field}; use plain or strict").into()),
    }
}
fn draft_depth(value: &Value, field: &str) -> Result<u64> {
    if let Value::Number(s) = value && let Ok(n) = s.parse::<u64>() && n <= 3 { return Ok(n); }
    Err(format!("invalid card {field}; use an integer in 0..=3").into())
}
fn drafting(value: &Value) -> Result<bool> {
    let numerics = at(value,&["health","vade","numerics"]);
    let (mut mode,mut depth) = (None,None);
    if let Value::Object(request) = at(value,&["settings","extra_body"]) {
        mode = request.get("draft").map(|m| draft_mode(m,"settings.extra_body.draft")).transpose()?;
        depth = request.get("mtp_depth").map(|d| draft_depth(d,"settings.extra_body.mtp_depth")).transpose()?;
    }
    // The parser resolves the mode first. Generation::enable_draft then forces Plain at depth 0.
    let enabled = match mode.or(depth.map(|n| n != 0)) {
        Some(enabled) => enabled,
        None => draft_mode(numerics.get("draft"),"health.vade.numerics.draft")?,
    };
    if !enabled { return Ok(false); }
    if let Some(depth) = depth { return Ok(depth != 0); }
    // Health masks depth to 0 for Plain. It cannot resolve an explicit Strict request's depth.
    if !draft_mode(numerics.get("draft"),"health.vade.numerics.draft")? {
        return Err("cannot resolve card drafting depth from a plain server default; record settings.extra_body.mtp_depth".into());
    }
    Ok(draft_depth(numerics.get("mtp_depth"),"health.vade.numerics.mtp_depth")? != 0)
}
fn card(value: &Value, source: &str, sha: String) -> Result<Card> {
    let numerics = at(value,&["health","vade","numerics"]);
    let sparks = count(numerics,&["world_size"])?;
    if sparks == 0 { return Err("card world_size must be positive".into()); }
    let path = if matches!(numerics.get("tensor_split"),Value::Object(_)) { "tensor split".into() }
        else { text(numerics,&["topology"])? };
    let drafting = drafting(value)?;
    let finished = text(value,&["run","finished_at"])?;
    Ok(Card { key:(sparks,path,drafting),model:text(value,&["run","model"])?,sha,
        time:stamp(&finished)?,finished,metrics:Vec::new(),ids:0,responses:0,hash:sha256::digest(source.as_bytes()) })
}
fn measurements(card: &mut Card, value: &Value) -> Result<()> {
    for workload in ["prose","code","structured"] {
        card.metrics.push(metric(value,&["decode",workload,"decode_tokens_per_second","median"])?);
    }
    for chats in ["1","4","8"] {
        card.metrics.push(metric(value,&["concurrency",chats,"aggregate_end_to_end_tokens_per_second","median"])?);
    }
    card.metrics.push(metric(value,&["prefill","8192","cold","ttft_seconds","median"])?);
    card.responses = count(value,&["response_audit","responses"])?; card.ids = saved_ids(value);
    if card.responses == 0 || card.ids > card.responses { return Err("invalid card response ID coverage".into()); }
    Ok(())
}
fn files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?; let kind = entry.file_type()?;
        if kind.is_dir() { files(&entry.path(),out)?; }
        else if kind.is_file() && entry.file_name().to_string_lossy().ends_with(".card.json") { out.push(entry.path()); }
    } Ok(())
}
fn select(cards: &Path, repo: &Path) -> Result<Vec<Card>> {
    let history = crate::output(Command::new("git").arg("-C").arg(repo).args(["log","--first-parent","--format=%H","main"]))?;
    let history = String::from_utf8(history)?;
    let revisions: BTreeMap<_,_> = history.lines().enumerate().map(|(i,s)| (s.to_owned(),i)).collect();
    let mut paths = Vec::new(); files(cards,&mut paths)?; paths.sort();
    let mut pending = Vec::new(); let mut newest = usize::MAX;
    for path in paths {
        let source = fs::read_to_string(&path)?;
        let value = json::parse(&source).map_err(|e| format!("invalid card {}: {e}",path.display()))?;
        if value.get("complete") != &Value::Bool(true) || value.get("status") != &Value::String("PASS".into()) { continue; }
        if at(&value,&["settings","extra_body","chat_template_kwargs","enable_thinking"]) != &Value::Bool(false) { continue; }
        let sha = text(&value,&["engine_sha"])?;
        let Some(&revision) = revisions.get(&sha) else { continue; };
        if revision > newest { continue; }
        if revision < newest { pending.clear(); newest = revision; }
        pending.push((value,source,sha,path));
    }
    let mut selected: BTreeMap<Key,(Card,Value,PathBuf)> = BTreeMap::new();
    for (value,source,sha,path) in pending {
        let next = card(&value,&source,sha).map_err(|e| format!("invalid card {}: {e}",path.display()))?;
        let replace = selected.get(&next.key).is_none_or(|(old,_,_)| next.time > old.time || (next.time == old.time && next.hash < old.hash));
        if replace { selected.insert(next.key.clone(),(next,value,path)); }
    }
    if selected.is_empty() { return Err("no complete, passing, thinking-OFF cards from main history".into()); }
    selected.into_values().map(|(mut card,value,path)| {
        measurements(&mut card,&value).map_err(|e| format!("invalid card {}: {e}",path.display()))?;
        Ok(card)
    }).collect()
}
fn cell(s: &str) -> String {
    s.replace('&',"&amp;").replace('<',"&lt;").replace('>',"&gt;").replace('|',"&#124;")
        .replace('`',"&#96;").replace(['\n','\r']," ")
}
fn render(cards: &[Card]) -> String {
    let mut out = String::from("\n\n### Latest measured main\n\n| Model | Sparks | Path | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) | IDs saved / responses |\n| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for c in cards {
        out.push_str(&format!("| {} | {} | {} | {} | {} | {}/{} |\n",cell(&c.model),c.key.0,cell(&c.key.1),
            if c.key.2 {"ON"} else {"OFF"},c.metrics.join(" | "),c.ids,c.responses));
    }
    out.push_str(&format!("\nMeasured main revision: `{}`.\n",cards[0].sha));
    for c in cards {
        out.push_str(&format!("\n{} Sparks, {}, drafting {}: finished `{}`; card SHA256 `{}`.\n",
            c.key.0,cell(&c.key.1),if c.key.2 {"ON"} else {"OFF"},c.finished,c.hash));
    }
    out.push_str(concat!("\nRates are card medians in tok/s. C1, C4 and C8 cover all chats together.\n",
        "Cold 8K is the median time to the first token. IDs saved counts responses\n",
        "with recorded token IDs. This count does not prove that the IDs match.\n\n",
        "Only the newest measured revision in main's first-parent history appears.\n",
        "Each row uses the last finished, complete, passing, thinking-OFF card for\n",
        "its Spark count, path and drafting setting. Older revisions do not fill gaps.\n\n",
        "From the repository root, refresh both tables with this command.\n",
        "Replace the quoted path with the folder that holds the cards.\n\n```sh\n",
        "cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- bench-docs '<cards-dir>' .\n```\n\n",
        "The sections below are historical results. Their dates and revision names\n",
        "apply only to those measurements.\n\n"));
    out
}
fn replace(source: &str, body: &str) -> Result<String> {
    if source.matches(START).count() != 1 || source.matches(END).count() != 1 {
        return Err("need one bench-docs start marker and one end marker".into());
    }
    let start = source.find(START).unwrap()+START.len(); let end = source.find(END).unwrap();
    if end < start { return Err("bench-docs end marker precedes start marker".into()); }
    Ok(format!("{}{body}{}",&source[..start],&source[end..]))
}
pub fn run(cards: &Path, repo: &Path) -> Result<()> {
    let body = render(&select(cards,repo)?); let mut updates = Vec::new();
    // Validate both documents before the first write.
    for name in ["README.md","docs/benchmarks.md"] {
        let path = repo.join(name);
        if !fs::symlink_metadata(&path)?.file_type().is_file() { return Err(format!("document must be a regular file: {name}").into()); }
        let source = fs::read_to_string(&path)?;
        let next = replace(&source,&body).map_err(|e| format!("{name}: {e}"))?;
        if source != next { updates.push((path,next)); }
    }
    for (path,next) in &updates { fs::write(path,next)?; }
    println!("bench-docs: updated {} document(s)",updates.len()); Ok(())
}
