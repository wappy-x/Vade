//! Host-only A/X judge. Rule 2026-11-01 uses the base or own A/A band,
//! the cached latency floor, and one second X start near the limit.
use crate::evals::json::{self, Value};
use std::{fmt::Write, path::Path};

const RULE: &str = "2026-11-01";
const BASE: f64 = 0.8;
const EPS: f64 = 1e-9;
const RULE_TEXT: &str = "rule 2026-11-01 band max(base, own A/A), cached < 1 s floor 10 ms, second X only near the line";

pub const HELP: &str = r#"Judge saved A/X cells on this host. Send no requests.

Usage: vade bench verdict FILE... [OPTIONS]

Rule 2026-11-01 (screen-verdict):
  Base band: 0.8 %. Cell band: max(base, own A/A).
  Own A/A: absolute percent change of the even A mean from the odd A mean.
  Cached seconds cells with A < 1 s also pass at up to +10 ms (FLOOR).
  Cached names contain warm_replay, warm replay or cache (case ignored).
  Names with nocache, no-cache or uncached do not use the floor.
  One X outside the base band needs a second X only within 2 x cell band.
  A second X outside the band fails. No third X is requested.
  With two starts per arm, a loss inside own A/A reads NOISE.
  One FAIL within 2 x own A/A requests RERUN unless --final is set.
  Token IDs must match. A present ids_equal must be JSON true.

Options:
  --rule 2026-11-01  Name the rule explicitly (also the default)
  --band PCT         Override each base band, except band_on_clean_pct
  --ids-ok yes|no    Result of the token ID check (default yes)
  --sha SHA40        Put the revision in the verdict line
  --advisory CELL    Exclude a full name or suffix; repeatable
  --only CELL        Judge only this full cell name
  --final            Judge a completed rerun; do not request another
  --no-record        Accepted for script callers; this command writes no files
  --                 Treat all remaining arguments as file names
  -h, --help         Show this page

--band, --only and --sha each accept one value. Do not repeat these flags.
An advisory suffix must start after a space or "/".

Input: a JSON object with screen, optional band_pct and ids_equal, and cells.
Each cell needs cell, unit (s or tok/s), a and x. Each arm is a list of
per-start values, or a pooled number with a_starts or x_starts.
A pooled A with at least two starts can give aa_pct. Each cell can set band_pct.
Optional band_pct, aa_pct and band_on_clean_pct can be null (absent).
Screen band_pct must be finite and nonnegative.
Selected cell band_pct and band_on_clean_pct must be finite and nonnegative.
These checks also apply with --band.
A selected cell aa_pct must be a finite number for every arm form.
A pooled A uses aa_pct only with at least two starts. It uses the absolute value.
Lists use their own A/A.
Cell band_on_clean_pct replaces the base when at least one cell without it
is judged and all such cells read INSIDE or INSIDE (X faster).
A null or zero value does not widen.
Example: {"screen":"check","ids_equal":true,"cells":[{"cell":"rate","unit":"tok/s","a":[100],"x":[99.3]}]}
Several files form one screen. Each file keeps its base band.
At least one selected cell must be gating (without --advisory).
The command refuses a verdict if no gating cells remain, including with --only.
This command uses no saved station history or environment rule switches.

Exit: 0 SCREEN PASS; 1 SCREEN FAIL; 2 bad input, no gating cells, or output error;
3 SECOND-START; 4 RERUN.
"#;

#[derive(Default)]
struct Options {
    files: Vec<String>, band: Option<f64>, advisory: Vec<String>, only: Option<String>,
    final_run: bool, ids_bad: bool, sha: Option<String>, help: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--" { o.files.extend_from_slice(&args[i + 1..]); break; }
        if matches!(flag, "--help" | "-h") { o.help = true; return Ok(o); }
        if flag == "--final" { o.final_run = true; i += 1; continue; }
        if flag == "--no-record" { i += 1; continue; }
        if !flag.starts_with('-') { o.files.push(flag.into()); i += 1; continue; }
        if !matches!(flag, "--rule" | "--band" | "--advisory" | "--only" | "--ids-ok" | "--sha") {
            return Err(format!("Unknown option {}. Run vade bench verdict --help.", crate::text::argument(flag)));
        }
        let value = args.get(i + 1).filter(|v| !v.is_empty() && !v.starts_with("--"))
            .ok_or_else(|| format!("Give {flag} a value. Run vade bench verdict --help."))?;
        match flag {
            "--rule" if value != RULE => return Err("Use rule 2026-11-01. Run vade bench verdict --help.".into()),
            "--band" if o.band.is_some() => return Err("Give --band only once.".into()),
            "--only" if o.only.is_some() => return Err("Give --only only once.".into()),
            "--sha" if o.sha.is_some() => return Err("Give --sha only once.".into()),
            "--band" => o.band = Some(nonnegative(value.parse::<f64>().map_err(|_| "Give --band a finite nonnegative number.")?)?),
            "--advisory" => o.advisory.push(value.clone()),
            "--only" => o.only = Some(value.clone()),
            "--ids-ok" => match value.as_str() {
                "yes" => o.ids_bad = false, "no" => o.ids_bad = true,
                _ => return Err("Give --ids-ok yes or no.".into()),
            },
            "--sha" => {
                if value.len() != 40 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("Give --sha a full hexadecimal revision.".into());
                }
                o.sha = Some(value.clone());
            }
            _ => {}
        }
        i += 2;
    }
    if o.files.is_empty() { return Err("Give at least one cells JSON file. Run vade bench verdict --help.".into()); }
    Ok(o)
}

/// Return the script's verdict exit code. Parse errors always return 2.
pub fn run(args: &[String]) -> i32 {
    let result = (|| {
        let options = parse(args)?;
        if options.help { return Ok((HELP.to_owned(), 0)); }
        let screens = options.files.iter().map(|file| {
            let input = std::fs::read_to_string(Path::new(file))
                .map_err(|e| crate::text::file_error("read", Path::new(file), &e))?;
            json::parse(&input).map_err(|e| format!("Invalid cells JSON: {e}"))
        }).collect::<Result<Vec<_>, String>>()?;
        judge(&screens, &options)
    })();
    match result {
        Ok((text, code)) => {
            if let Err(error) = crate::output::text(format_args!("{text}")) {
                if !crate::output::broken_pipe_success() {
                    crate::output::stderr_line(format_args!("bench verdict: {error}"));
                    return 2;
                }
            }
            code
        }
        Err(error) => { crate::output::stderr_line(format_args!("bench verdict: {}", crate::text::terminal_text(&error))); 2 }
    }
}

fn number(v: &Value) -> Result<f64, String> {
    v.as_f64().filter(|n| n.is_finite()).ok_or_else(|| "Give a finite number for each measurement and band.".into())
}

fn nonnegative(n: f64) -> Result<f64, String> {
    if n.is_finite() && n >= 0.0 { Ok(n) } else { Err("Give a finite nonnegative band.".into()) }
}

fn band(v: Option<&Value>, default: f64) -> Result<f64, String> {
    v.filter(|v| **v != Value::Null).map_or(Ok(default), |v| nonnegative(number(v)?))
}

fn name<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty() && !s.chars().any(char::is_control))
        .ok_or_else(|| format!("Give {key} a nonempty text value without control characters."))
}

struct Arm { mean: f64, starts: usize, values: Option<Vec<f64>> }

fn mean(values: &[f64]) -> Result<f64, String> {
    let m = values.iter().sum::<f64>() / values.len() as f64;
    if m.is_finite() && m > 0.0 { Ok(m) } else { Err("Give nonempty arms with finite positive measurements.".into()) }
}

fn arm(c: &Value, side: &str) -> Result<Arm, String> {
    let v = c.get(side).ok_or_else(|| format!("Give each cell an {side} measurement."))?;
    if let Some(values) = v.as_array() {
        let values = values.iter().map(number).collect::<Result<Vec<_>, _>>()?;
        if values.iter().any(|v| *v <= 0.0) { return Err("Give positive measurements.".into()); }
        return Ok(Arm { mean: mean(&values)?, starts: values.len(), values: Some(values) });
    }
    let n = number(v)?;
    if n <= 0.0 { return Err("Give positive measurements.".into()); }
    let starts = match c.get(&format!("{side}_starts")) {
        None => 1,
        Some(v) => v.as_i64().and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0)
            .ok_or("Give pooled measurements a positive integer start count.")?,
    };
    Ok(Arm { mean: n, starts, values: None })
}

struct Row {
    cell: String, a: f64, x: f64, na: usize, nx: usize, pct: f64, reg: f64,
    base: f64, band: f64, own: Option<f64>, source: String, read: &'static str, advisory: bool,
}

fn row(c: &Value, base: f64, o: &Options) -> Result<Row, String> {
    let cell = name(c, "cell")?.to_owned();
    let lower = match c.get("unit").and_then(Value::as_str) {
        Some("s") => true, Some("tok/s") => false,
        _ => return Err(format!("Cell {}: unit must be s or tok/s.", crate::text::argument(&cell))),
    };
    let a = arm(c, "a")?;
    let x = arm(c, "x")?;
    let recorded_own = c.get("aa_pct").filter(|v| **v != Value::Null).map(number).transpose()?.map(f64::abs);
    let own = if let Some(values) = &a.values {
        if values.len() < 2 { None } else {
            let odd: Vec<f64> = values.iter().step_by(2).copied().collect();
            let even: Vec<f64> = values.iter().skip(1).step_by(2).copied().collect();
            Some((100.0 * (mean(&even)? / mean(&odd)? - 1.0)).abs())
        }
    } else if a.starts >= 2 {
        recorded_own
    } else { None };
    let base = band(c.get("band_pct"), base)?;
    let base = o.band.unwrap_or(base);
    let band = base.max(own.unwrap_or(0.0));
    let pct = 100.0 * (x.mean / a.mean - 1.0);
    let reg = if lower { pct } else { -pct };
    if !pct.is_finite() || own.is_some_and(|s| !s.is_finite()) { return Err("Measurement ratio is too large.".into()); }
    let source = if own.is_some_and(|s| s > base) { format!("own A/A ({} A starts)", a.starts) } else { "base".into() };
    let mut read = if reg <= base + EPS { if reg >= 0.0 { "INSIDE" } else { "INSIDE (X faster)" } }
        else if a.starts.min(x.starts) < 2 { "SECOND-START" }
        else if reg <= band + EPS { "NOISE" } else { "FAIL" };
    let cached = cell.to_lowercase();
    let cached = !["nocache", "no-cache", "uncached"].iter().any(|s| cached.contains(s))
        && ["warm_replay", "warm replay", "cache"].iter().any(|s| cached.contains(s));
    if !matches!(read, "INSIDE" | "INSIDE (X faster)") {
        if lower && cached && a.mean < 1.0 && (x.mean - a.mean) * 1000.0 <= 10.0 + EPS { read = "FLOOR"; }
        else if read == "SECOND-START" && (x.starts >= 2 || reg > 2.0 * band + EPS) { read = "FAIL"; }
    }
    let advisory = o.advisory.iter().any(|s| cell.strip_suffix(s.as_str())
        .is_some_and(|prefix| prefix.is_empty() || prefix.ends_with(' ') || prefix.ends_with('/')));
    Ok(Row { cell, a: a.mean, x: x.mean, na: a.starts, nx: x.starts, pct, reg, base, band, own, source, read, advisory })
}

fn judge(screens: &[Value], o: &Options) -> Result<(String, i32), String> {
    let mut rows = Vec::new();
    let mut selected = Vec::new();
    let mut names = Vec::new();
    let mut ids_bad = o.ids_bad;
    for screen in screens {
        if !matches!(screen, Value::Object(_)) { return Err("Give a JSON screen object.".into()); }
        names.push(if screen.get("screen").is_some() { name(screen, "screen")? } else { "screen" });
        ids_bad |= screen.get("ids_equal").is_some_and(|v| *v != Value::Bool(true));
        let base = band(screen.get("band_pct"), BASE)?;
        let cells = screen.get("cells").and_then(Value::as_array).ok_or("Give the screen a cells array.")?;
        for c in cells {
            // The script checks units after --only has selected the cells.
            if o.only.as_ref().is_none_or(|only| c.get("cell").and_then(Value::as_str) == Some(only.as_str())) {
                rows.push(row(c, base, o)?);
                let wider = c.get("band_on_clean_pct").filter(|v| **v != Value::Null)
                    .map(|v| nonnegative(number(v)?)).transpose()?.filter(|n| *n != 0.0);
                selected.push((c, wider));
            }
        }
    }
    if rows.is_empty() {
        return Err(if o.only.is_some() { "No cell has the --only name." } else { "Give at least one cell to judge." }.into());
    }
    // Saved pair inputs may widen OFF's base only when every ON cell is inside.
    let mut narrow = rows.iter().zip(&selected).filter(|(_, (_, wider))| wider.is_none()).peekable();
    let clean = narrow.peek().is_some() && narrow.all(|(r, _)| matches!(r.read, "INSIDE" | "INSIDE (X faster)"));
    if clean {
        for (at, (c, wider)) in selected.iter().enumerate() {
            if let Some(wider) = wider {
                rows[at] = row(c, *wider, &Options { band: Some(*wider), advisory: o.advisory.clone(), ..Options::default() })?;
            }
        }
    }
    let live: Vec<_> = rows.iter().filter(|r| !r.advisory).collect();
    if live.is_empty() { return Err("No gating cells remain. Select at least one cell without --advisory.".into()); }
    let fails: Vec<_> = live.iter().copied().filter(|r| r.read == "FAIL").collect();
    let second = live.iter().any(|r| r.read == "SECOND-START");
    let rerun = if !ids_bad && !o.final_run && !second && fails.len() == 1
        && fails[0].own.is_some_and(|s| s > 0.0 && fails[0].reg <= 2.0 * s + EPS) { Some(fails[0]) } else { None };
    let (verdict, code) = if ids_bad { ("SCREEN FAIL", 1) } else if rerun.is_some() { ("RERUN", 4) }
        else if !fails.is_empty() { ("SCREEN FAIL", 1) } else if second { ("SECOND-START", 3) } else { ("SCREEN PASS", 0) };
    Ok((report(&rows, &names.join(" + "), o, verdict, ids_bad, rerun), code))
}

/// Six significant digits, as in the script's measurement columns.
fn short(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..6).contains(&exponent) {
        format!("{}e{exponent:+03}", mantissa.trim_end_matches('0').trim_end_matches('.'))
    } else {
        let fixed = format!("{:.*}", (5 - exponent).max(0) as usize, value);
        if fixed.contains('.') { fixed.trim_end_matches('0').trim_end_matches('.').into() } else { fixed }
    }
}

fn report(rows: &[Row], name: &str, o: &Options, verdict: &str, ids_bad: bool, rerun: Option<&Row>) -> String {
    let mut out = String::new();
    for r in rows {
        let advisory = if r.advisory { "ADVISORY " } else { "" };
        let _ = write!(out, "CELL {} | A {} ({} start{}) -> X {} ({}) | {:+.3} % | {advisory}{}",
            r.cell, short(r.a), r.na, if r.na == 1 { "" } else { "s" }, short(r.x), r.nx, r.pct, r.read);
        if r.read == "FLOOR" {
            let _ = write!(out, "; +{:.1} ms <= 10 ms floor (cached cell, A < 1 s; band {:.3} % vs delta {:.3} %)", (r.x - r.a) * 1000.0, r.band, r.reg);
        } else if !matches!(r.read, "INSIDE" | "INSIDE (X faster)") {
            let _ = write!(out, "; band {:.3} % ({}) vs delta {:.3} %", r.band, r.source, r.reg);
        }
        out.push('\n');
    }
    let categories = ["prose", "code", "structured"];
    let mut groups: Vec<(&str, [Option<f64>; 3])> = Vec::new();
    for r in rows {
        let (label, last) = r.cell.rsplit_once(' ').unwrap_or(("", &r.cell));
        if let Some(i) = categories.iter().position(|c| *c == last) {
            let at = groups.iter().position(|(l, _)| *l == label).unwrap_or_else(|| { groups.push((label, [None; 3])); groups.len() - 1 });
            groups[at].1[i] = Some(r.pct);
        }
    }
    for (label, values) in groups {
        if let [Some(prose), Some(code), Some(structured)] = values {
            let _ = writeln!(out, "DECODE-MEAN {} | prose {prose:+.3} %, code {code:+.3} %, structured {structured:+.3} % -> mean {:+.3} % | advisory, not gating",
                if label.is_empty() { "-" } else { label }, (prose + code + structured) / 3.0);
        }
    }
    let bands = rows.iter().map(|r| format!("{} {:.3} % {} (base {}, own {}, hist n/a)",
        r.cell, r.band, r.source, short(r.base), r.own.map_or("n/a".into(), |s| format!("{s:.3}")))).collect::<Vec<_>>();
    let _ = writeln!(out, "BANDS: {}", bands.join(" ; "));
    let head = o.sha.as_ref().map_or_else(|| name.to_owned(), |sha| format!("{sha} | {name}"));
    if let Some(r) = rerun {
        let _ = writeln!(out, "RERUN {} | {head} | delta {:.3} % > band {:.3} % ({}), inside 2 x own A/A {:.3} % | run a 3+3 cooled re-run of this cell (A X X A A X), then judge it with --final --only '{}'",
            r.cell, r.reg, r.band, r.source, 2.0 * r.own.unwrap(), r.cell);
        return out;
    }
    let mut why = Vec::new();
    if ids_bad { why.push("IDS DIFFER (exactness gate)".into()); }
    for r in rows.iter().filter(|r| !matches!(r.read, "INSIDE" | "INSIDE (X faster)")) {
        why.push(format!("{} {}{}", r.cell, if r.advisory { "ADVISORY " } else { "" }, r.read));
    }
    let tail = if verdict == "SECOND-START" { "; a cell is within the band of the line: run one more X start (A X X) and judge again" } else { "" };
    let _ = writeln!(out, "{verdict} {head} | {RULE_TEXT}{} | {}{tail}", if o.final_run { " | final 3+3" } else { "" },
        if why.is_empty() { "every cell inside or X faster".into() } else { why.join(", ") });
    out
}

#[cfg(test)]
#[path = "verdict_tests.rs"]
mod tests;
