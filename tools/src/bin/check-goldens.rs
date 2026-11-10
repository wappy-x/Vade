//! Compare served token IDs with the golden hashes tracked in `tools/goldens/`.
//! Host only: it reads receipt files. It starts no engine and needs no GPU.
use std::{collections::BTreeMap, fs, process::ExitCode};
use vade_tools::{json::{self, Value}, sha256};

const FORMAT: &str = "vade-goldens-v1";
/// The prefix fingerprint hashes every prefix again, so its cost grows with the square of the length.
const MAX_OUTPUT_TOKENS: usize = 4096;
const USAGE: &str = "usage: check-goldens MANIFEST RECEIPT.jsonl...\n       check-goldens --make MODEL EPOCH RECEIPT.jsonl...";

#[derive(Debug, Clone, PartialEq)]
struct Cell { prompt_tokens: usize, output_tokens: usize, sha256: String, prefixes: String }
#[derive(Debug, Default, PartialEq)]
struct Manifest { model: String, epoch: String, cells: BTreeMap<String, Cell> }
#[derive(Debug, PartialEq)]
struct Row { cell: String, prompt_tokens: Option<usize>, ids: Vec<u32>, ok: bool }

/// One decimal ID a line. This is the byte format of the lab's `*.golden-out.ids` files,
/// so `shasum -a 256` of such a file gives the manifest's sha256.
fn canonical(ids: &[u32]) -> Vec<u8> { ids.iter().flat_map(|id| format!("{id}\n").into_bytes()).collect() }

/// The sha256 of the whole sequence, and eight hex digits of the sha256 of each prefix.
/// The whole hash decides PASS or FAIL. The prefixes only find the first token that differs.
fn fingerprint(ids: &[u32]) -> (String, String) {
    let bytes = canonical(ids);
    let (mut prefixes, mut end) = (String::with_capacity(ids.len() * 8), 0);
    for id in ids {
        end += id.to_string().len() + 1;
        prefixes.push_str(&sha256::digest(&bytes[..end])[..8]);
    }
    (sha256::digest(&bytes), prefixes)
}

fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let (mut manifest, mut format) = (Manifest::default(), false);
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let bad = |why: &str| format!("manifest line {}: {why}", index + 1);
        match line.split('\t').collect::<Vec<_>>().as_slice() {
            ["format", value] if *value == FORMAT && !format => format = true,
            ["model", value] if manifest.model.is_empty() && !value.is_empty() => manifest.model = (*value).into(),
            ["epoch", value] if manifest.epoch.is_empty() && !value.is_empty() => manifest.epoch = (*value).into(),
            ["note", _, _] => (),
            ["cell", name, prompt, output, sha, prefixes] => {
                let prompt_tokens = prompt.parse::<usize>().map_err(|_| bad("prompt token count is not a number"))?;
                let output_tokens = output.parse::<usize>().map_err(|_| bad("output token count is not a number"))?;
                if output_tokens == 0 || output_tokens > MAX_OUTPUT_TOKENS { return Err(bad("output token count is out of range")); }
                if !vade_tools::is_lower_hex(sha, 64) { return Err(bad("sha256 needs 64 lowercase hex digits")); }
                if !vade_tools::is_lower_hex(prefixes, output_tokens * 8) { return Err(bad("prefix fingerprint does not fit the output token count")); }
                if prefixes[prefixes.len() - 8..] != sha[..8] { return Err(bad("last prefix fingerprint differs from the sha256")); }
                let cell = Cell { prompt_tokens, output_tokens, sha256: (*sha).into(), prefixes: (*prefixes).into() };
                if name.is_empty() || manifest.cells.insert((*name).into(), cell).is_some() { return Err(bad("empty or repeated cell name")); }
            }
            _ => return Err(bad("unknown, repeated or malformed row")),
        }
    }
    if !format || manifest.model.is_empty() || manifest.epoch.is_empty() || manifest.cells.is_empty() {
        return Err(format!("manifest needs a `format\t{FORMAT}` row, a model, an epoch and one cell or more"));
    }
    Ok(manifest)
}

fn parse_row(line: &str) -> Result<Row, String> {
    let value = json::parse(line)?;
    let Value::String(cell) = value.get("cell") else { return Err("receipt row has no cell name".into()); };
    let ids = value.get("output_ids").array().map_err(|_| "receipt row has no output_ids array")?.iter().map(|id| match id {
        Value::Number(number) => number.parse::<u32>().map_err(|_| "output_ids holds a number that is not a token ID".to_owned()),
        _ => Err("output_ids holds a value that is not a number".into()),
    }).collect::<Result<Vec<_>, _>>()?;
    let prompt_tokens = match value.get("prompt_tokens") { Value::Number(number) => number.parse().ok(), _ => None };
    Ok(Row { cell: cell.clone(), prompt_tokens, ids, ok: !matches!(value.get("ok"), Value::Bool(false)) })
}

/// Every nonempty line of every file, with its place for messages.
fn rows(files: &[(String, String)]) -> Result<Vec<(String, Row)>, String> {
    let mut out = Vec::new();
    for (name, text) in files {
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let place = format!("{name}:{}", index + 1);
            let row = parse_row(line).map_err(|error| format!("{place}: {error}"))?;
            if row.ids.is_empty() || row.ids.len() > MAX_OUTPUT_TOKENS { return Err(format!("{place}: output token count is out of range")); }
            out.push((place, row));
        }
    }
    if out.is_empty() { return Err("the receipt files hold no rows".into()); }
    Ok(out)
}

/// Nothing, or the first difference between one receipt row and its golden cell.
fn compare(cell: &Cell, row: &Row) -> Result<(), String> {
    if !row.ok { return Err("the receipt marks this row as not ok".into()); }
    if let Some(prompt) = row.prompt_tokens && prompt != cell.prompt_tokens {
        return Err(format!("the prompt has {prompt} tokens, the golden prompt has {}", cell.prompt_tokens));
    }
    let (sha, prefixes) = fingerprint(&row.ids);
    if sha == cell.sha256 { return Ok(()); }
    let shared = row.ids.len().min(cell.output_tokens);
    let hashes = format!("expected sha256 {}, got {sha}", cell.sha256);
    Err(match (0..shared).find(|&i| prefixes[i * 8..i * 8 + 8] != cell.prefixes[i * 8..i * 8 + 8]) {
        Some(i) => format!("token {} of {} differs (got ID {}); {hashes}", i + 1, cell.output_tokens, row.ids[i]),
        None => format!("the output has {} tokens, the golden output has {}, and the first {shared} agree; {hashes}", row.ids.len(), cell.output_tokens),
    })
}

/// The PASS line, or the FAIL line with the first difference in file and line order.
fn check(manifest: &Manifest, files: &[(String, String)]) -> Result<String, String> {
    let rows = rows(files)?;
    let mut seen = BTreeMap::new();
    for (place, row) in &rows {
        let cell = manifest.cells.get(&row.cell).ok_or_else(|| format!("first difference: {place} cell={}: the manifest has no such cell", row.cell))?;
        compare(cell, row).map_err(|difference| format!("first difference: {place} cell={}: {difference}", row.cell))?;
        *seen.entry(row.cell.as_str()).or_insert(0usize) += 1;
    }
    if let Some(missing) = manifest.cells.keys().find(|name| !seen.contains_key(name.as_str())) {
        return Err(format!("the receipts hold no row for cell {missing}"));
    }
    Ok(format!("GOLDENS PASS model={} epoch={} rows={} cells={}", manifest.model, manifest.epoch, rows.len(), seen.len()))
}

/// A manifest from receipts in which every row of a cell has the same IDs.
fn make(model: &str, epoch: &str, files: &[(String, String)]) -> Result<String, String> {
    if model.is_empty() || epoch.is_empty() || model.contains(['\t', '\n']) || epoch.contains(['\t', '\n']) { return Err("model and epoch need plain text".into()); }
    let mut cells: BTreeMap<String, (String, usize, Vec<u32>)> = BTreeMap::new();
    for (place, row) in rows(files)? {
        if !row.ok { return Err(format!("{place}: the receipt marks this row as not ok")); }
        let prompt = row.prompt_tokens.ok_or_else(|| format!("{place}: receipt row has no prompt_tokens"))?;
        match cells.get(&row.cell) {
            Some((first, first_prompt, ids)) if *first_prompt != prompt || *ids != row.ids => {
                return Err(format!("cell {} has two different answers: {first} and {place}. No golden was made.", row.cell));
            }
            Some(_) => (),
            None => { cells.insert(row.cell.clone(), (place, prompt, row.ids)); }
        }
    }
    let mut out = format!("format\t{FORMAT}\nmodel\t{model}\nepoch\t{epoch}\n");
    for (name, (_, prompt, ids)) in &cells {
        let (sha, prefixes) = fingerprint(ids);
        out.push_str(&format!("cell\t{name}\t{prompt}\t{}\t{sha}\t{prefixes}\n", ids.len()));
    }
    Ok(out)
}

fn read(paths: &[String]) -> Result<Vec<(String, String)>, String> {
    if paths.is_empty() { return Err(USAGE.into()); }
    paths.iter().map(|path| fs::read_to_string(path).map(|text| (path.clone(), text)).map_err(|error| format!("{path}: {error}"))).collect()
}

fn run(args: &[String]) -> Result<String, String> {
    match args {
        [flag, model, epoch, receipts @ ..] if flag == "--make" => make(model, epoch, &read(receipts)?),
        [manifest, receipts @ ..] if !manifest.starts_with("--") => {
            let text = fs::read_to_string(manifest).map_err(|error| format!("{manifest}: {error}"))?;
            check(&parse_manifest(&text).map_err(|error| format!("{manifest}: {error}"))?, &read(receipts)?)
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(text) => { println!("{}", text.trim_end()); ExitCode::SUCCESS }
        Err(error) => { eprintln!("GOLDENS FAIL {error}"); ExitCode::FAILURE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(cell: &str, prompt: usize, ids: &[u32]) -> String {
        format!("{{\"cell\":\"{cell}\",\"ok\":true,\"prompt_tokens\":{prompt},\"output_ids\":{ids:?}}}\n")
    }
    fn files(text: String) -> Vec<(String, String)> { vec![("run.jsonl".into(), text)] }

    #[test]
    fn the_hash_is_the_sha256_of_one_decimal_id_a_line() {
        // printf '785\n1196\n702\n' | shasum -a 256
        assert_eq!(canonical(&[785, 1196, 702]), b"785\n1196\n702\n");
        let (sha, prefixes) = fingerprint(&[785, 1196, 702]);
        assert_eq!(sha, sha256::digest(b"785\n1196\n702\n"));
        assert_eq!(prefixes.len(), 24);
        assert_eq!(prefixes[..8], sha256::digest(b"785\n")[..8]);
        assert_eq!(prefixes[16..], sha[..8]);
    }

    #[test]
    fn made_manifests_pass_their_own_receipts_and_name_the_first_changed_token() {
        let good = receipt("c15", 15, &[7, 8, 9, 10]) + &receipt("c127", 127, &[1, 2]) + &receipt("c15", 15, &[7, 8, 9, 10]);
        let manifest = parse_manifest(&make("demo", "rc.0", &files(good.clone())).unwrap()).unwrap();
        assert_eq!(check(&manifest, &files(good)).unwrap(), "GOLDENS PASS model=demo epoch=rc.0 rows=3 cells=2");

        let changed = receipt("c127", 127, &[1, 2]) + &receipt("c15", 15, &[7, 8, 99, 10]);
        let error = check(&manifest, &files(changed)).unwrap_err();
        assert!(error.starts_with("first difference: run.jsonl:2 cell=c15: token 3 of 4 differs (got ID 99)"), "{error}");

        let short = receipt("c127", 127, &[1, 2]) + &receipt("c15", 15, &[7, 8]);
        assert!(check(&manifest, &files(short)).unwrap_err().contains("the output has 2 tokens, the golden output has 4, and the first 2 agree"));
        let long = receipt("c127", 127, &[1, 2, 3]) + &receipt("c15", 15, &[7, 8, 9, 10]);
        assert!(check(&manifest, &files(long)).unwrap_err().contains("run.jsonl:1 cell=c127: the output has 3 tokens"));
    }

    #[test]
    fn a_gate_needs_every_cell_a_known_cell_an_ok_row_and_the_golden_prompt() {
        let manifest = parse_manifest(&make("demo", "rc.0", &files(receipt("a", 3, &[5]) + &receipt("b", 4, &[6]))).unwrap()).unwrap();
        assert_eq!(check(&manifest, &files(receipt("a", 3, &[5]))).unwrap_err(), "the receipts hold no row for cell b");
        assert!(check(&manifest, &files(receipt("z", 3, &[5]))).unwrap_err().contains("the manifest has no such cell"));
        assert!(check(&manifest, &files(receipt("a", 9, &[5]))).unwrap_err().contains("the prompt has 9 tokens, the golden prompt has 3"));
        let failed = "{\"cell\":\"a\",\"ok\":false,\"prompt_tokens\":3,\"output_ids\":[5]}\n".to_owned();
        assert!(check(&manifest, &files(failed)).unwrap_err().contains("not ok"));
        assert!(check(&manifest, &files(String::new())).unwrap_err().contains("no rows"));
        assert!(check(&manifest, &files("{\"cell\":\"a\",\"output_ids\":[1.5]}\n".into())).unwrap_err().contains("not a token ID"));
    }

    #[test]
    fn no_golden_is_made_from_two_answers() {
        let two = receipt("a", 3, &[5, 6]) + &receipt("a", 3, &[5, 7]);
        assert!(make("demo", "rc.0", &files(two)).unwrap_err().contains("cell a has two different answers: run.jsonl:1 and run.jsonl:2"));
        assert!(make("demo", "", &files(receipt("a", 3, &[5]))).is_err());
    }

    #[test]
    fn manifests_refuse_edited_rows() {
        let text = make("demo", "rc.0", &files(receipt("a", 3, &[5, 6]))).unwrap();
        assert!(parse_manifest(&text).is_ok());
        assert!(parse_manifest(&format!("{text}{}", text.lines().last().unwrap())).unwrap_err().contains("repeated cell"));
        assert!(parse_manifest(&text.replace("\t2\t", "\t3\t")).unwrap_err().contains("does not fit"));
        assert!(parse_manifest(&text.replace("format\t", "format\tv0-")).is_err());
        assert!(parse_manifest(&text.replace("model\tdemo\n", "")).is_err());
        let (sha, _) = fingerprint(&[5, 6]);
        let other = if sha.starts_with('0') { "1" } else { "0" };
        assert!(parse_manifest(&text.replace(&format!("\t{sha}\t"), &format!("\t{other}{}\t", &sha[1..]))).unwrap_err().contains("last prefix fingerprint"));
        assert!(parse_manifest(&format!("# a comment\nnote\tsource\tany text\n{text}")).is_ok());
    }

    /// The host half of the gate: every tracked manifest is well formed and names its model once.
    #[test]
    fn tracked_manifests_are_well_formed() {
        let folder = vade_tools::root().join("tools/goldens");
        let mut models = std::collections::BTreeSet::new();
        for entry in fs::read_dir(&folder).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "tsv") { continue; }
            let manifest = parse_manifest(&fs::read_to_string(&path).unwrap()).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(path.file_stem().unwrap().to_string_lossy(), manifest.model, "{}: the file name is the model name", path.display());
            assert!(models.insert(manifest.model), "{}: one manifest a model", path.display());
        }
        assert!(!models.is_empty(), "tools/goldens holds no manifest");
    }
}
