//! BF16 latent prompt/end counters, separately for each served rank and session.
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
type Result<T> = std::result::Result<T,String>;
const NAMES:[&str;3]=["latent_absorption_tile","absorb_step","absorb_step_recorded"];
fn fields(line:&str)->BTreeMap<&str,&str> {line.split_whitespace().filter_map(|w|w.split_once('=')).collect()}
fn count(f:&BTreeMap<&str,&str>,name:&str)->Result<u64> {f.get(name).ok_or_else(||format!("missing {name}"))?.parse().map_err(|_|format!("invalid {name}"))}
fn check(texts:&[String])->Result<[bool;3]> {
    let mut shots:BTreeMap<(u64,u64),[Option<[u64;3]>;2]>=BTreeMap::new();
    let(mut decode,mut verify)=(BTreeSet::new(),BTreeSet::new());
    for text in texts {for line in text.lines() {
        if line.starts_with("SERVED STEP rank=0 ") && line.contains("phase=decode ") {decode.insert(count(&fields(line),"session")?);}
        if line.starts_with("SERVED VERIFY rank=0 ") {verify.insert(count(&fields(line),"session")?);}
        if !line.starts_with("SERVED BF COUNTS ") {continue;}
        let f=fields(line);let rank=count(&f,"rank")?;let session=count(&f,"session")?;
        if rank>1 {return Err("unexpected rank".into());}
        let at=match f.get("at") {Some(&"prompt")=>0,Some(&"end")=>1,_=>return Err("bad snapshot phase".into())};
        let mut c=[0;3];for (i,name) in NAMES.iter().enumerate() {c[i]=count(&f,name)?;}
        if shots.entry((rank,session)).or_insert([None,None])[at].replace(c).is_some() {return Err("duplicate snapshot".into());}
    }}
    if shots.is_empty() {return Err("no served BF snapshots".into());}
    let(mut coverage,mut ranks)=([false;3],BTreeSet::new());
    for (&(rank,session),[p,e]) in &shots {
        let p=p.ok_or_else(||format!("rank{rank} session{session}: missing prompt"))?;
        let e=e.ok_or_else(||format!("rank{rank} session{session}: missing end"))?;
        if e.iter().zip(p).any(|(e,p)|*e<p) {return Err("counter decreased".into());}
        if p[0]==0 {return Err("no prompt tile coverage in fresh window".into());} coverage[0]=true;
        let step=e[1]>p[1] || e[2]>p[2];
        if decode.contains(&session) {if !step {return Err(format!("rank{rank} session{session}: zero decode delta"));} coverage[1]=true;}
        if verify.contains(&session) {if !step && e[0]==p[0] {return Err(format!("rank{rank} session{session}: zero verification delta"));} coverage[2]=true;}
        ranks.insert(rank);
        println!("BF PATH rank={rank} session={session} prompt_tile={} tile_delta={} step_delta={} recorded_delta={}",p[0],e[0]-p[0],e[1]-p[1],e[2]-p[2]);
    }
    if ranks!=BTreeSet::from([0,1]) {return Err("missing rank".into());}
    for session in decode.union(&verify) {for rank in 0..2 {if !shots.contains_key(&(rank,*session)) {return Err(format!("missing rank{rank} session{session}"));}}}
    Ok(coverage)
}
fn windows(p:&Path,out:&mut BTreeMap<String,Vec<String>>)->Result<()> {
    if p.is_dir() {for child in fs::read_dir(p).map_err(|e|e.to_string())? {windows(&child.map_err(|e|e.to_string())?.path(),out)?;}}
    else if p.file_name().and_then(|x|x.to_str()).is_some_and(|x|x.starts_with("rank")&&x.ends_with(".log")) {
        out.entry(p.parent().ok_or("missing parent")?.to_string_lossy().into_owned()).or_default().push(fs::read_to_string(p).map_err(|e|e.to_string())?);
    }Ok(())
}
fn run()->Result<()> {
    let args:Vec<_>=std::env::args().skip(1).collect();if args.is_empty() {return Err("usage: bf-paths [--plain] RUN...".into());}
    let plain=args.iter().any(|x|x=="--plain");let mut all=BTreeMap::new();
    for p in args.iter().filter(|x|x.as_str()!="--plain") {windows(Path::new(p),&mut all)?;}
    let mut coverage=[false;3];for (name,texts) in all {println!("WINDOW {name}");let c=check(&texts)?;for i in 0..3 {coverage[i]|=c[i];}}
    if !coverage[0] || !coverage[1] || (!plain && !coverage[2]) {return Err(format!("missing prompt/decode/verify coverage {coverage:?}"));}
    println!("SERVED BF PATHS PASS");Ok(())
}
fn main() {if let Err(e)=run() {eprintln!("SERVED BF PATHS FAIL: {e}");std::process::exit(1);}}
#[cfg(test)] mod tests {
    use super::*;
    fn fixture()->String {
        let mut s=String::from("SERVED STEP rank=0 session=1 phase=decode seconds=1\nSERVED VERIFY rank=0 session=1 rows=4\n");
        for r in 0..2 {for (at,n) in [("prompt",1),("end",2)] {s+=&format!("SERVED BF COUNTS rank={r} session=1 at={at} latent_absorption_tile=4 absorb_step={n} absorb_step_recorded={n}\n");}}s
    }
    #[test] fn complete() {assert_eq!(check(&[fixture()]).unwrap(),[true;3]);}
    #[test] fn zero_delta_rejected() {assert!(check(&[fixture().replace("absorb_step=2","absorb_step=1").replace("absorb_step_recorded=2","absorb_step_recorded=1")]).is_err());}
    #[test] fn absent_rank_rejected() {assert!(check(&[fixture().lines().filter(|s|!s.contains("rank=1")).map(|s|format!("{s}\n")).collect()]).is_err());}
    #[test] fn missing_end_rejected() {assert!(check(&[fixture().lines().filter(|s|!s.contains("at=end")).map(|s|format!("{s}\n")).collect()]).is_err());}
    #[test] fn missing_field_rejected() {assert!(check(&[fixture().replace(" absorb_step_recorded=2","")]).is_err());}
    #[test] fn zero_prompt_rejected() {assert!(check(&[fixture().replace("latent_absorption_tile=4","latent_absorption_tile=0")]).is_err());}
    #[test] fn old_prompt_cannot_prove_decode() {assert!(check(&[fixture().replace("at=end latent_absorption_tile=4 absorb_step=2 absorb_step_recorded=2","at=end latent_absorption_tile=4 absorb_step=1 absorb_step_recorded=1")]).is_err());}
    #[test] fn decreased_rejected() {assert!(check(&[fixture().replace("absorb_step=2","absorb_step=0")]).is_err());}
}
