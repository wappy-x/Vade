//! Validate complete same-history quality evidence. Uses the repository JSON reader.
//! Usage: quality HISTORIES MAKER_REFERENCE MAIN_RUN [CANDIDATE_RUN]
#[allow(dead_code)]
#[path="../src/json.rs"] mod json;
use json::Value;
use std::{collections::BTreeMap,fs,path::Path};
type Result<T> = std::result::Result<T,String>;
fn read(p:&Path)->Result<Value> {json::parse(&fs::read_to_string(p).map_err(|e|format!("{}: {e}",p.display()))?)}
fn number(v:&Value)->Result<usize> {v.text()?.parse().map_err(|_|"invalid count".into())}
fn result_files(run:&Path)->Result<BTreeMap<String,Value>> {
    let mut found=BTreeMap::new();
    for p in fs::read_dir(run.join("QL/out-QL")).map_err(|e|e.to_string())? {
        let p=p.map_err(|e|e.to_string())?.path();
        if p.extension().is_none_or(|x|x!="json") {continue;}
        let v=read(&p)?;
        if v.get("top")==&Value::Null {continue;}
        let name=v.get("name").text()?.to_string();
        if found.insert(name.clone(),v).is_some() {return Err(format!("duplicate history {name}"));}
    }
    Ok(found)
}
fn validate(h:&Value,r:&Value,is_main:bool)->Result<(usize,usize)> {
    let history=h.get("history").array()?; let top=r.get("top").array()?;
    let name=h.get("name").text()?;
    if top.len()!=history.len() || number(r.get("done"))?!=history.len() {return Err(format!("{name}: incomplete history"));}
    if !r.get("errors").array()?.is_empty() {return Err(format!("{name}: request errors"));}
    let cold=match r.get("cold") {Value::Object(x)=>x,_=>return Err(format!("{name}: missing cold checks"))};
    for i in [0,history.len().checked_sub(1).ok_or("empty history")?] {
        let c=cold.get(&i.to_string()).ok_or_else(||format!("{name}: missing cold endpoint {i}"))?;
        if c.get("equal")!=&Value::Bool(true) || c.get("cold")!=&top[i] || c.get("kept")!=&top[i] || c.get("error")!=&Value::Null {
            return Err(format!("{name}: cold/cache mismatch at {i}"));
        }
    }
    if is_main && h.get("chat_request")!=&Value::Null && r.get("free8_equals_history")!=&Value::Bool(true) {return Err(format!("{name}: wrong chat prompt"));}
    let same=top.iter().zip(history).filter(|(a,b)|a==b).count();
    Ok((same,history.len()))
}
fn summarize(hist:&Value,reference:&Value,run:&Path,is_main:bool)->Result<(usize,usize,usize,usize)> {
    let results=result_files(run)?; let histories=hist.array()?;
    if results.len()!=histories.len() {return Err(format!("{}: missing or extra histories",run.display()));}
    let (mut same,mut total,mut maker,mut maker_total)=(0,0,0,0);
    for h in histories {
        let name=h.get("name").text()?;
        let r=results.get(name).ok_or_else(||format!("missing history {name}"))?;
        let (a,n)=validate(h,r,is_main)?;
        match h.get("measure").text()? {
            "main"=>{same+=a; total+=n; println!("MEASURED {} {name} main-history={a}/{n}",run.display());},
            "maker"=>{
                let target=reference.get(name).get("maker_top1").array()?;
                if target.len()!=n {return Err(format!("{name}: maker reference length"));}
                let a=r.get("top").array()?.iter().zip(target).filter(|(a,b)|a==b).count();
                maker+=a; maker_total+=n; println!("MEASURED {} {name} maker={a}/{n}",run.display());
            },_=>return Err(format!("unknown measure {name}"))
        }
    }
    if total!=2080 || maker_total!=208 {return Err(format!("unexpected evidence coverage main={total}, maker={maker_total}"));}
    if is_main && same!=total {return Err(format!("main self-quality differs {same}/{total}"));}
    if !is_main && same*100<total*99 {return Err(format!("candidate below 99 percent: {same}/{total}"));}
    println!("MEASURED {} main-history={same}/{total} maker={maker}/{maker_total}",run.display());
    Ok((same,total,maker,maker_total))
}
fn run()->Result<()> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {return Err("usage: quality HISTORIES MAKER_REFERENCE MAIN_RUN [CANDIDATE_RUN]".into());}
    let hist=read(Path::new(&args[0]))?; let reference=read(Path::new(&args[1]))?;
    let base=summarize(&hist,&reference,Path::new(&args[2]),true)?;
    if let Some(candidate)=args.get(3) {
        let new=summarize(&hist,&reference,Path::new(candidate),false)?;
        if new.2<base.2 || new.3!=base.3 {return Err(format!("maker quality dropped {} -> {} of {}",base.2,new.2,base.3));}
    }
    println!("QUALITY VALID: yes"); Ok(())
}
fn main() {if let Err(e)=run() {eprintln!("QUALITY FAIL: {e}");std::process::exit(1);}}
#[cfg(test)] mod tests {
    use super::*;
    fn fixture()->(Value,Value) {
        (json::parse(r#"{"name":"x","history":[1,2],"chat_request":{}}"#).unwrap(),json::parse(r#"{"top":[1,2],"done":2,"errors":[],"free8_equals_history":true,"cold":{"0":{"equal":true,"cold":1,"kept":1,"error":null},"1":{"equal":true,"cold":2,"kept":2,"error":null}}}"#).unwrap())
    }
    #[test] fn complete() {let(h,r)=fixture();assert_eq!(validate(&h,&r,true).unwrap(),(2,2));}
    #[test] fn missing_endpoint_rejected() {let(h,mut r)=fixture();if let Value::Object(ref mut x)=r {x.insert("cold".into(),Value::Object(BTreeMap::new()));}assert!(validate(&h,&r,true).is_err());}
    #[test] fn truncated_rejected() {let(h,mut r)=fixture();if let Value::Object(ref mut x)=r {x.insert("done".into(),Value::Number("1".into()));}assert!(validate(&h,&r,true).is_err());}
    #[test] fn wrong_prompt_rejected_on_main() {let(h,mut r)=fixture();if let Value::Object(ref mut x)=r {x.insert("free8_equals_history".into(),Value::Bool(false));}assert!(validate(&h,&r,true).is_err());assert!(validate(&h,&r,false).is_ok());}
    #[test] fn false_equal_flag_rejected() {let(h,mut r)=fixture();if let Value::Object(ref mut x)=r {x.insert("top".into(),json::parse("[9,2]").unwrap());}assert!(validate(&h,&r,false).is_err());}
}
