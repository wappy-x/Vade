//! Add Spark columns and set labels to an existing internal dashboard.
use std::{collections::BTreeSet,fs,path::Path};
use crate::{Result,json::{self,Value}};

const RAW: &str = "const RAW = ";
const GENERATED: &str = "const GENERATED = ";
const DATA: &str = "/*__DATA__*/[]";
const BUILT: &str = "/*__GENERATED__*/\"\"";
const VERSION: &str = "<!-- rigmark-spark-columns:v1 -->";

fn replace(source: &str, from: &str, to: &str) -> Result<String> {
    if source.matches(from).count() != 1 { return Err("dashboard template has a missing or repeated anchor".into()); }
    Ok(source.replacen(from,to,1))
}
fn literal<'a>(page: &'a str, prefix: &str, end: &str) -> Result<&'a str> {
    if page.matches(prefix).count() != 1 { return Err("dashboard has a missing or repeated data anchor".into()); }
    page.split_once(prefix).and_then(|(_,s)| s.split_once(end)).map(|(s,_)| s)
        .ok_or_else(|| "dashboard has an unclosed data field".into())
}
fn members(value: &Value) -> Result<BTreeSet<String>> {
    let mut unique = BTreeSet::new();
    for member in value.array()? {
        let Value::String(member) = member else { return Err("set members must be text".into()); };
        if member.is_empty() || !member.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.') {
            return Err("set members must use letters, digits, dots, underscores or hyphens".into());
        }
        if !unique.insert(member.to_ascii_lowercase()) { return Err("set has a repeated member".into()); }
    }
    if unique.is_empty() { return Err("member list must not be empty".into()); }
    Ok(unique)
}
fn script(value: &Value) -> String { value.encode().replace('<',"\\u003c").replace('>',"\\u003e").replace('&',"\\u0026") }
fn roster(source: &str) -> Result<(String,String)> {
    let value = json::parse(source)?;
    let Value::Object(sets) = value.get("sets") else { return Err("set map needs a sets object of member arrays".into()); };
    if sets.is_empty() { return Err("set map must not be empty".into()); }
    let mut memberships = BTreeSet::new();
    for (name,names) in sets {
        if name.is_empty() || name.chars().any(char::is_control) { return Err("set name must be nonempty text without control characters".into()); }
        if !memberships.insert(members(names)?) { return Err("set map has repeated memberships".into()); }
    }
    let empty = Value::Object(Default::default());
    let card_members = match value.get("cards") { Value::Null => &empty, cards => cards };
    let Value::Object(cards) = card_members else { return Err("set map cards must be an object of saved member arrays".into()); };
    for (card,names) in cards {
        if card.is_empty() || card.chars().any(char::is_control) { return Err("card key must be nonempty text without control characters".into()); }
        members(names)?;
    }
    // The map is private input. Escape it for an inline script without logging its members.
    Ok((script(value.get("sets")),script(card_members)))
}

const STYLE: &str = r#"
.spark-columns { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:20px 0 28px; }
.spark-column { min-width:0; padding:18px; }
.spark-column h2 { margin:0 0 12px; font-size:18px; }
.spark-card { border-top:1px solid var(--border); padding:12px 0; overflow-wrap:anywhere; }
.spark-card a { color:var(--text); }
.spark-card p { margin:5px 0; color:var(--muted); font-size:12px; }
.spark-card strong { font-variant-numeric:tabular-nums; }
@media(max-width:1020px) { .spark-columns { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media(max-width:640px) { .spark-columns { grid-template-columns:minmax(0,1fr); } }
"#;
const DISPLAY: &str = r#"
function cardSet(r) {
  if (r.host === "rented") return "Unlisted";
  const saved = CARD_MEMBERS[`${r.folder}/${r.card}`];
  const members = saved ? saved.map(n => n.toLowerCase()).sort() : String(r.boxes || "").toLowerCase().split(/\s*(?:->|→|,|\+|>|\/|\s+)\s*/).filter(Boolean).sort();
  const matches = Object.entries(CARD_SETS).filter(([, names]) =>
    names.length === Number(r.sparks) && members.length === names.length &&
    names.map(n => n.toLowerCase()).sort().every((name, i) => name === members[i]));
  return matches.length === 1 ? matches[0][0] : "Unlisted";
}
function drawSparkColumns() {
  const metric = metricDef(state.metric), rows = filtered();
  $("spark-columns").innerHTML = [1, 2, 4, 8].map(count => {
    const latest = new Map();
    for (const r of [...rows].sort((a,b) => a.t - b.t)) {
      if (r.sparks !== count) continue;
      const key = JSON.stringify([seriesKey(r), r.set_name, r.thinking, r.drafting, r.status]);
      latest.set(key, r);
    }
    const cards = [...latest.values()].sort((a,b) => b.t - a.t).map(r =>
      `<article class="spark-card"><a href="${encodeURI(r.folder)}/${encodeURI(r.card)}">${esc(fmtWhen(r))}</a>` +
      `<p>${esc(r.model)} · ${esc(quantShown(r.quant))}</p><p>Set ${esc(r.set_name)} · ${esc(r.engine || "Unlisted")}</p>` +
      `<p>Thinking ${esc(r.thinking)} · drafting ${esc(r.drafting)} · ${esc(r.status)}</p>` +
      `<strong>${fmt(r[metric.key], metric.unit)} ${esc(metric.unit)}</strong><p>${esc(metric.label)}</p></article>`).join("");
    return `<section class="panel spark-column"><h2>${count} Spark${count === 1 ? "" : "s"}</h2>${cards || '<p class="meta">No card matches these filters.</p>'}</section>`;
  }).join("");
}
"#;

fn upgrade(template: &str, sets: &str, card_members: &str) -> Result<String> {
    if template.contains(VERSION) {
        let mut template = template.to_owned();
        for (prefix,value) in [("const CARD_SETS = ",sets),("const CARD_MEMBERS = ",card_members)] {
            let old = literal(&template,prefix,";\n")?;
            template = replace(&template,&format!("{prefix}{old};\n"),&format!("{prefix}{value};\n"))?;
        }
        return Ok(template);
    }
    let mut template = replace(template,"</style>",&format!("{STYLE}</style>"))?;
    let anchor = "  <section class=\"panel card\">\n    <div class=\"card-head\">\n      <div>\n        <h2>Over time</h2>";
    let section = format!("{VERSION}\n  <h2>Latest cards by Spark count</h2>\n  <p class=\"meta\">The filters apply. Each entry is the latest card for its setup, set, thinking, drafting and status. Set names need an exact member match. Unlisted means that the set map has no match.</p>\n  <div class=\"spark-columns\" id=\"spark-columns\" aria-label=\"Latest cards by Spark count\"></div>\n\n{anchor}");
    template = replace(&template,anchor,&section)?;
    template = replace(&template,"const DATA = RAW.map(r => {",&format!("const CARD_SETS = {sets};\nconst CARD_MEMBERS = {card_members};\n{DISPLAY}\nconst DATA = RAW.map(r => {{"))?;
    template = replace(&template,"  const o = { ...r };","  const o = { ...r, set_name: cardSet(r) };")?;
    template = replace(&template,"const COLS = [","const COLS = [\n  { key: \"set_name\", label: \"Set\", cls: \"l\", render: r => esc(r.set_name) },")?;
    let tip = "    <div class=\"t-row\"><span>Engine</span>";
    template = replace(&template,tip,&format!("    <div class=\"t-row\"><span>Set</span><b>${{esc(r.set_name)}}</b></div>\n{tip}"))?;
    template = replace(&template,"  drawLegend(); drawTime();","  drawSparkColumns(); drawLegend(); drawTime();")?;
    Ok(template)
}

/// Return a template and a page. Card JSON and the build label stay byte for byte the same.
pub fn render(template: &str, page: &str, sets: &str) -> Result<(String,String,usize)> {
    let (sets,card_members) = roster(sets)?;
    let payload = literal(page,RAW,";\nconst GENERATED = ")?;
    let cards = json::parse(payload)?;
    let count = cards.array()?.len();
    let built = literal(page,GENERATED,";\n")?;
    if !matches!(json::parse(built)?,Value::String(_)) { return Err("dashboard build label must be text".into()); }
    let template = upgrade(template,&sets,&card_members)?;
    // Resolve the same two markers as the existing builder. Do not parse or round measurements.
    let result = replace(&template,DATA,payload)?;
    let result = replace(&result,BUILT,built)?;
    Ok((template,result,count))
}

pub fn run(template: &Path, page: &Path, sets: &Path, out: &Path) -> Result<()> {
    let (template,page,count) = render(&fs::read_to_string(template)?,&fs::read_to_string(page)?,&fs::read_to_string(sets)?)?;
    // A fresh directory keeps this command from overwriting the active dashboard or a prior preview.
    fs::create_dir(out)?;
    fs::write(out.join("template.html"),template)?;
    fs::write(out.join("RigMark Dashboard.html"),page)?;
    println!("dashboard preview: PASS ({count} cards; embedded card data unchanged)");
    Ok(())
}
