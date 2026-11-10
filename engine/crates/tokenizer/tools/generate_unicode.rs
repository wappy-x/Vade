//! Generate data tables from the official Unicode inputs in UNICODE-NOTICE.txt.
//! Compile with `rustc --edition=2024 -O crates/tokenizer/tools/generate_unicode.rs`.

#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn hex(value: &str) -> Result<u32, std::num::ParseIntError> {
    u32::from_str_radix(value, 16)
}

fn add_range(ranges: &mut Vec<(u32, u32)>, start: u32, end: u32) {
    if let Some(last) = ranges.last_mut().filter(|last| last.1 + 1 == start) {
        last.1 = end;
    } else {
        ranges.push((start, end));
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 4 {
        return Err("usage: generate_unicode UnicodeData-16.0.0.txt UnicodeData-12.1.0.txt CompositionExclusions-12.1.0.txt OUTPUT".into());
    }
    let sources = args[..3]
        .iter()
        .map(fs::read_to_string)
        .collect::<Result<Vec<_>, _>>()?;
    let mut letters = Vec::new();
    let mut numbers = Vec::new();
    let mut marks = Vec::new();
    let mut first = None;
    for line in sources[0].lines() {
        let row: Vec<_> = line.split(';').collect();
        if row.len() != 15 {
            return Err("invalid UnicodeData row".into());
        }
        let n = hex(row[0])?;
        if row[1].ends_with(", First>") {
            first = Some(n);
            continue;
        }
        let start = if row[1].ends_with(", Last>") {
            first.ok_or("Unicode range end without start")?
        } else {
            n
        };
        match row[2].as_bytes().first() {
            Some(b'L') => add_range(&mut letters, start, n),
            Some(b'N') => add_range(&mut numbers, start, n),
            Some(b'M') => add_range(&mut marks, start, n),
            _ => {}
        }
        first = None;
    }
    let mut ccc = BTreeMap::new();
    let mut decomp = Vec::new();
    for line in sources[1].lines() {
        let row: Vec<_> = line.split(';').collect();
        if row.len() != 15 {
            return Err("invalid UnicodeData row".into());
        }
        let n = hex(row[0])?;
        let class: u8 = row[3].parse()?;
        if class != 0 {
            ccc.insert(n, class);
        }
        if !row[5].is_empty() && !row[5].starts_with('<') {
            let values = row[5]
                .split_whitespace()
                .map(hex)
                .collect::<Result<Vec<_>, _>>()?;
            decomp.push((n, values));
        }
    }
    let excluded: BTreeSet<_> = sources[2]
        .lines()
        .map(|line| line.split('#').next().unwrap().trim())
        .filter(|line| !line.is_empty())
        .map(hex)
        .collect::<Result<_, _>>()?;
    let mut compose = Vec::new();
    for (n, values) in &decomp {
        if values.len() == 2 && !excluded.contains(n) && !ccc.contains_key(&values[0]) {
            compose.push((values[0], values[1], *n));
        }
    }
    compose.sort_unstable();

    let mut out = BufWriter::new(File::create(&args[3])?);
    writeln!(
        out,
        "// Unicode category data16.0.0; normalization data12.1.0."
    )?;
    writeln!(out, "// Generated data only; see UNICODE-NOTICE.txt.")?;
    for (path, source) in args.iter().zip(&sources) {
        let name = path
            .file_name()
            .ok_or("input filename missing")?
            .to_str()
            .ok_or("input filename must be UTF-8")?;
        writeln!(
            out,
            "// SHA256 {} {}",
            sha256::digest(source.as_bytes()),
            name
        )?;
    }
    for (name, ranges) in [("LETTERS", letters), ("NUMBERS", numbers), ("MARKS", marks)] {
        writeln!(out, "pub static {name}: &[(u32,u32)] = &[")?;
        for (start, end) in ranges {
            writeln!(out, "({start},{end}),")?;
        }
        writeln!(out, "];")?;
    }
    writeln!(out, "pub static CCC: &[(u32,u8)] = &[")?;
    for (n, class) in ccc {
        writeln!(out, "({n},{class}),")?;
    }
    writeln!(out, "];\npub static COMPOSE: &[(u32,u32,u32)] = &[")?;
    for (a, b, n) in compose {
        writeln!(out, "({a},{b},{n}),")?;
    }
    writeln!(out, "];\npub static DECOMP: &[(u32,&[u32])] = &[")?;
    for (n, values) in decomp {
        let values: Vec<_> = values.iter().map(u32::to_string).collect();
        writeln!(out, "({n},&[{}]),", values.join(","))?;
    }
    writeln!(out, "];")?;
    out.flush()?;
    Ok(())
}
