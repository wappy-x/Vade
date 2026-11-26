//! Host-only verdict for the ordered pair gate. No device call and no guessed timings.
#![forbid(unsafe_code)]
use crate::Result;
use std::{fs,path::Path};
fn field<'a>(line:&'a str,name:&str)->Result<&'a str> {line.split_whitespace().find_map(|s|s.strip_prefix(name)).ok_or_else(||format!("missing receipt field {name}"))}
fn integer(line:&str,name:&str)->Result<usize> {field(line,name)?.parse().map_err(|_|format!("invalid receipt field {name}"))}
fn validate(log:&str,boundaries:usize,candidate:bool)->Result<Vec<(usize,f64,usize)>> {
    for wanted in ["BOUNDARY PAIR PASS changing=1 drafting=off,on","BOUNDARY COHORT PASS depth=0 picked=0","BOUNDARY COHORT PASS depth=3 picked=0","BOUNDARY COHORT PASS depth=0 picked=1"] {
        if !log.lines().any(|line|line.starts_with(wanted)) {return Err(format!("missing pass line {wanted}"));}
    }
    let starts=log.lines().filter(|l|l.starts_with("BOUNDARY STARTUP BYTES PASS ")).collect::<Vec<_>>();
    let first=log.lines().filter(|l|l.starts_with("BOUNDARY FIRST STEP BYTES PASS ")).collect::<Vec<_>>();
    if starts.len()!=1 || first.len()!=8 {return Err("missing startup or first-step byte receipts".into());}
    let mut chats=std::collections::BTreeSet::new();
    for line in &first {chats.insert(integer(line,"chat=")?);}
    if chats!=(0..8).collect() {return Err("first-step chats are not complete".into());}
    for line in starts.into_iter().chain(first) {let budget=integer(line,"admission=")?;
        if integer(line,"live=")?>budget || integer(line,"peak=")?>budget {return Err("device bytes exceed admission".into());}}
    let mut times=Vec::new();let mut widths=std::collections::BTreeSet::new();
    for line in log.lines().filter(|l|l.starts_with("BOUNDARY STEP depth=0 picked=1 ")) {
        let rows=integer(line,"rows=")?;let nodes=integer(line,"boundary_launches=")?;
        let expected=boundaries.checked_mul(if candidate {rows.min(3)} else {rows}).ok_or("launch count overflow")?;
        let ms=field(line,"ms=")?.parse::<f64>().map_err(|_|"invalid step time")?;
        if !(1..=8).contains(&rows) || nodes!=expected || !ms.is_finite() || ms<=0.0 {return Err(format!("step time or launch count differs: {line}"));}
        widths.insert(rows);times.push((rows,ms,nodes));
    }
    if ![1,2,4,8].iter().all(|n|widths.contains(n)) {return Err("missing required timed cohort".into());}Ok(times)
}
pub(crate) fn run(args:&[String])->Result<()> {
    if args.len()!=8 {return Err("--boundary-receipts takes compare|memory|times BOUNDARIES M1 C1 M2 C2 M3 C3".into());}
    let mode=args[0].as_str();if !["compare","memory","times"].contains(&mode) {return Err("invalid boundary receipt mode".into());}
    let boundaries=args[1].parse::<usize>().map_err(|_|"invalid prepared boundary count")?;if boundaries==0 {return Err("empty boundary count".into());}
    let dirs=&args[2..];let reference=Path::new(&dirs[0]);
    for name in ["ALONE-IDS","ALONE-BITS"] {
        let want=fs::read(reference.join(name)).map_err(|e|e.to_string())?;
        if want.is_empty() {return Err(format!("empty {name}"));}
        for dir in dirs {let got=fs::read(Path::new(dir).join(name)).map_err(|e|e.to_string())?;if got!=want {return Err(format!("{name} differs in {dir}"));}}
    }
    for (load,dir) in dirs.iter().enumerate() {
        let log=fs::read_to_string(Path::new(dir).join("BOUNDARY-GATE")).map_err(|e|e.to_string())?;
        let times=validate(&log,boundaries,load%2==1)?;
        if mode=="memory" {for line in log.lines().filter(|l|l.starts_with("BOUNDARY STARTUP BYTES") || l.starts_with("BOUNDARY FIRST STEP BYTES")) {println!("MEASURED load={load} {line}");}}
        if mode=="times" {for (rows,ms,nodes) in times {if [4,8].contains(&rows) {println!("MEASURED load={load} rows={rows} step_ms={ms:.6} boundary_launches={nodes}");}}}
    }
    println!("BOUNDARY RECEIPTS PASS mode={mode} loads=6 ids=equal logit_bits=equal widths=1,2,4,8 counts=pass");Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    fn log()->String {
        let mut s="BOUNDARY PAIR PASS changing=1 drafting=off,on\nBOUNDARY COHORT PASS depth=0 picked=0\nBOUNDARY COHORT PASS depth=3 picked=0\nBOUNDARY COHORT PASS depth=0 picked=1\nBOUNDARY STARTUP BYTES PASS live=1 peak=2 admission=3\n".to_owned();
        for chat in 0..8 {s.push_str(&format!("BOUNDARY FIRST STEP BYTES PASS chat={chat} live=1 peak=2 admission=3\n"));}
        for rows in [1usize,2,4,8] {s.push_str(&format!("BOUNDARY STEP depth=0 picked=1 rows={rows} ms=1.25 boundary_launches={}\n",rows.min(3)*90));}s
    }
    #[test] fn receipt_checks_refuse_missing_widths_wrong_counts_and_bad_bytes() {
        let s=log();assert_eq!(validate(&s,90,true).unwrap().len(),4);
        for bad in [s.replace("rows=8","rows=7"),s.replace("boundary_launches=270","boundary_launches=720"),s.replace("peak=2","peak=4"),s.replace("chat=7","chat=6"),s.replace("ms=1.25","ms=NaN")] {assert!(validate(&bad,90,true).is_err());}
    }
}
