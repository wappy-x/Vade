//! Served MX path evidence: agreed prompt/end cumulative counter snapshots.
//! Compare each session within one fresh server window, separately per rank.
//! Layer scalar counts exclude the intentionally scalar output head.
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
type Result<T> = std::result::Result<T,String>;
const NAMES:[&str;5]=["mx_tc_step","mx_tc_rows_stage","mx_tc_rows","mx_tc_tile","mx_scalar_layer"];
fn fields(line:&str)->BTreeMap<&str,&str> {line.split_whitespace().filter_map(|w|w.split_once('=')).collect()}
fn count(f:&BTreeMap<&str,&str>,name:&str)->Result<u64> {f.get(name).ok_or_else(||format!("missing {name}"))?.parse().map_err(|_|format!("invalid {name}"))}
fn check(texts:&[String])->Result<[bool;3]> {
    let mut shots:BTreeMap<(u64,u64),[Option<[u64;5]>;2]>=BTreeMap::new();
    let(mut decode,mut verify)=(BTreeSet::new(),BTreeSet::new());
    for text in texts {for line in text.lines() {
        if line.starts_with("SERVED STEP rank=0 ") && line.contains("phase=decode ") {decode.insert(count(&fields(line),"session")?);}
        if line.starts_with("SERVED VERIFY rank=0 ") {verify.insert(count(&fields(line),"session")?);}
        if !line.starts_with("SERVED MX COUNTS ") {continue;}
        let f=fields(line);let rank=count(&f,"rank")?;let session=count(&f,"session")?;
        if rank>1 {return Err("unexpected rank".into());}
        let at=match f.get("at") {Some(&"prompt")=>0,Some(&"end")=>1,_=>return Err("bad snapshot phase".into())};
        let mut counts=[0;5];for (i,name) in NAMES.iter().enumerate() {counts[i]=count(&f,name)?;}
        if counts[4]!=0 {return Err(format!("rank{rank} session{session}: scalar layer fallback"));}
        let slot=&mut shots.entry((rank,session)).or_insert([None,None])[at];
        if slot.replace(counts).is_some() {return Err(format!("duplicate rank{rank} session{session} snapshot"));}
    }}
    if shots.is_empty() {return Err("no served MX snapshots".into());}
    let(mut covered,mut ranks)=([false;3],BTreeSet::new());
    for ((rank,session),[prompt,end]) in shots {
        let p=prompt.ok_or_else(||format!("rank{rank} session{session}: missing prompt"))?;
        let e=end.ok_or_else(||format!("rank{rank} session{session}: missing end"))?;
        if e.iter().zip(p).any(|(e,p)|*e<p) {return Err("counter decreased".into());}
        ranks.insert(rank);
        if p[3]==0 {return Err(format!("rank{rank} session{session}: no TC prompt tile"));} covered[0]=true;
        if decode.contains(&session) {
            if e[0]==p[0] {return Err(format!("rank{rank} session{session}: decode did not use TC step"));}
            covered[1]=true;
        }
        if verify.contains(&session) {
            if e[1]==p[1] && e[2]==p[2] {return Err(format!("rank{rank} session{session}: verify did not use TC rows"));}
            covered[2]=true;
        }
        println!("MX PATH rank={rank} session={session} prompt_tile={} step_delta={} stage_delta={} rows_delta={} scalar_layer=0",p[3],e[0]-p[0],e[1]-p[1],e[2]-p[2]);
    }
    if ranks!=BTreeSet::from([0,1]) {return Err("missing rank snapshots".into());}
    // At least one snapshot per declared served session must exist on both ranks.
    for session in decode.union(&verify) {
        for rank in 0..2 {
            let prefix=format!("SERVED MX COUNTS rank={rank} session={session} at=prompt ");
            if !texts.iter().any(|t|t.lines().any(|l|l.starts_with(&prefix))) {return Err(format!("missing rank{rank} session{session}"));}
        }
    }
    Ok(covered)
}
fn windows(path:&Path,out:&mut BTreeMap<String,Vec<String>>)->Result<()> {
    if path.is_dir() {for p in fs::read_dir(path).map_err(|e|e.to_string())? {windows(&p.map_err(|e|e.to_string())?.path(),out)?;}}
    else if path.file_name().and_then(|x|x.to_str()).is_some_and(|x|x.starts_with("rank")&&x.ends_with(".log")) {
        let window=path.parent().ok_or("missing parent")?.to_string_lossy().into_owned();
        out.entry(window).or_default().push(fs::read_to_string(path).map_err(|e|e.to_string())?);
    }Ok(())
}
fn run()->Result<()> {
    let args:Vec<_>=std::env::args().skip(1).collect();if args.is_empty() {return Err("usage: paths RUN_FOLDER...".into());}
    let plain=args.iter().any(|x|x=="--plain");
    let mut all=BTreeMap::new();for p in args.iter().filter(|x|x.as_str()!="--plain") {windows(Path::new(p),&mut all)?;}
    let mut coverage=[false;3];
    for (name,texts) in all {println!("WINDOW {name}");let c=check(&texts)?;for i in 0..3 {coverage[i]|=c[i];}}
    if !coverage[0] || !coverage[1] || (!plain && !coverage[2]) {return Err(format!("missing prompt/decode/verify coverage {coverage:?}"));}
    println!("SERVED MX PATHS PASS");Ok(())
}
fn main() {if let Err(e)=run() {eprintln!("SERVED MX PATHS FAIL: {e}");std::process::exit(1);}}
#[cfg(test)] mod tests {
    use super::*;
    fn fixture()->String {
        let mut s=String::from("SERVED STEP rank=0 session=1 phase=decode seconds=1\nSERVED VERIFY rank=0 session=1 rows=4\n");
        for r in 0..2 {for (at,n) in [("prompt",1),("end",2)] {s+=&format!("SERVED MX COUNTS rank={r} session=1 at={at} mx_tc_step={n} mx_tc_rows_stage={n} mx_tc_rows={n} mx_tc_tile=4 mx_scalar_layer=0\n");}}s
    }
    #[test] fn complete() {assert_eq!(check(&[fixture()]).unwrap(),[true;3]);}
    #[test] fn zero_decode_delta_rejected() {assert!(check(&[fixture().replace("mx_tc_step=2","mx_tc_step=1")]).is_err());}
    #[test] fn zero_verify_delta_rejected() {assert!(check(&[fixture().replace("mx_tc_rows_stage=2","mx_tc_rows_stage=1").replace("mx_tc_rows=2","mx_tc_rows=1")]).is_err());}
    #[test] fn zero_prompt_rejected() {assert!(check(&[fixture().replace("mx_tc_tile=4","mx_tc_tile=0")]).is_err());}
    #[test] fn absent_rank_rejected() {assert!(check(&[fixture().lines().filter(|x|!x.contains("rank=1")).map(|x|format!("{x}\n")).collect()]).is_err());}
    #[test] fn scalar_fallback_rejected() {assert!(check(&[fixture().replace("mx_scalar_layer=0","mx_scalar_layer=1")]).is_err());}
    #[test] fn missing_end_rejected() {assert!(check(&[fixture().lines().filter(|x|!x.contains("at=end")).map(|x|format!("{x}\n")).collect()]).is_err());}
    #[test] fn warm_prompt_cannot_prove_decode() {assert!(check(&[fixture().replace("mx_tc_step=2","mx_tc_step=1")]).is_err());}
    #[test] fn decreased_rejected() {assert!(check(&[fixture().replace("mx_tc_step=2","mx_tc_step=0")]).is_err());}
}
