//! TP prefill/decode qualification against saved independent HF fixtures.
#[cfg(target_os="linux")]
fn run()->Result<(),String>{
    use engine_model::json::{self,Value,field};
    use std::{ffi::CString,path::{Path,PathBuf},time::Instant};
    use spark_tp::{model::Model,transport::Ring};
    fn get<'a>(v:&'a Value,k:&str)->Result<&'a Value,String>{field(v.object().map_err(|e|e.to_string())?,k).map_err(|e|e.to_string())}
    fn ids(v:&Value)->Result<Vec<u32>,String>{v.array().map_err(|e|e.to_string())?.iter().map(|v|v.u64().map(|n|n as u32).map_err(|e|e.to_string())).collect()}
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()<7{return Err("usage: tp-hf-check RANK WORLD LISTEN NEXT PTX MODEL FIXTURE.json...".into());}
    let rank=a[0].parse::<usize>().map_err(|e|e.to_string())?;let world=a[1].parse::<usize>().map_err(|e|e.to_string())?;
    let ptx=CString::new(std::fs::read(&a[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let root=if let Some(p)=a[5].strip_prefix("~/"){PathBuf::from(std::env::var("HOME").map_err(|e|e.to_string())?).join(p)}else{PathBuf::from(&a[5])};
    let ring=std::cell::RefCell::new(Ring::connect(rank,world,a[2].parse().map_err(|_|"listen")?,a[3].parse().map_err(|_|"next")?)?);
    spark_tp::runtime::verify_artifacts(&mut ring.borrow_mut(), &ptx, &root)?;
    let fixtures=a[6..].iter().map(|p|json::parse(&std::fs::read(p).map_err(|e|e.to_string())?,json::Limits::default()).map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
    spark_gpu_probe::driver::with_backend(0,&ptx,24*1024*1024*1024,|b|{
        let model=Model::load(b,&root,rank,world,2112)?;
        let mut failures=0;
        for(fixture,path)in fixtures.iter().zip(&a[6..]){
            let prompt=ids(get(fixture,"prompt_ids")?)?;let rows=get(fixture,"rows")?.array().map_err(|e|e.to_string())?;
            if prompt.is_empty() || rows.is_empty() {return Err(format!("HF fixture {path} is empty: prompt_ids={} rows={}",prompt.len(),rows.len()));}
            let mut s=model.session(&ring);s.configure_kv_bf16(std::env::var("SPARK_KV_BF16").as_deref()==Ok("1"))?;s.barrier()?;
            spark_tp::runtime::reset_heap_peak();let start=Instant::now();let mut logits=s.prefill(&prompt)?;let ttft=start.elapsed().as_secs_f64()*1000.;
            let(mut matches,mut full_max,mut top_max)=(0,0f32,0f64);
            for(i,row)in rows.iter().enumerate(){
                let token=get(row,"token")?.u64().map_err(|e|e.to_string())? as u32;
                if spark_engine::greedy_token(&logits)?==token{matches+=1;}
                let top=ids(get(row,"top_ids")?)?;let expected=get(row,"top_logits")?.array().map_err(|e|e.to_string())?;
                let expected=expected.iter().map(|e|e.f64().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
                top_max=top_max.max(top_error(&logits,&top,&expected).map_err(|e|format!("fixture {path} row {i}: {e}"))?);
                if let Some(file)=row.object().map_err(|e|e.to_string())?.get("full_logits"){
                    let file=Path::new(path).parent().unwrap().join(file.string().map_err(|e|e.to_string())?);
                    let bytes=std::fs::read(file).map_err(|e|e.to_string())?;
                    full_max=full_max.max(full_error(&logits,&bytes).map_err(|e|format!("fixture {path} row {i}: {e}"))?);
                }
                if i+1<rows.len(){logits=s.step(token)?;}
            }
            println!("HF rank={rank} world={world} fixture={} prompt={} top1={matches}/{} full_max_abs={full_max:.8} top20_max_abs={top_max:.8} ttft_ms={ttft:.3} kv_bytes={} peak_device_bytes={} pinned_peak_bytes={} host_heap_peak_bytes={} app_peak_bytes={}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,prompt.len(),rows.len(),s.cache_bytes(),s.device_peak_bytes(),s.pinned_peak_bytes(),spark_tp::runtime::heap_peak(),s.device_peak_bytes()+s.pinned_peak_bytes()+spark_tp::runtime::heap_peak());
            if matches!=rows.len()||full_max>0.01||top_max>0.01{failures+=1;}
        }
        if failures>0{return Err(format!("{failures} HF fixtures failed"));}
        Ok(())
    })
}
/// Largest |device - reference| over a row's top IDs. Empty or unequal ID and logit lists, an ID outside the row, or a
/// value that is not finite on either side fail: `f64::max` drops a NaN, so a NaN would otherwise pass.
#[cfg(any(target_os="linux",test))]
fn top_error(logits:&[f32],ids:&[u32],expected:&[f64])->Result<f64,String>{
    if ids.is_empty()||ids.len()!=expected.len(){return Err(format!("HF top_ids and top_logits must be equal and non-empty: top_ids={} top_logits={}",ids.len(),expected.len()));}
    let mut max=0f64;
    for(&id,&reference)in ids.iter().zip(expected){
        let &actual=logits.get(id as usize).ok_or_else(||format!("HF top id {id} is outside the row of {}",logits.len()))?;
        if !actual.is_finite()||!reference.is_finite(){return Err(format!("HF top id {id} is not finite: device={actual} reference={reference}"));}
        max=max.max((actual as f64-reference).abs());
    }
    Ok(max)
}
/// Largest |device - reference| over a full saved row of little-endian f32 values. An empty row, a width mismatch,
/// or a value that is not finite on either side fail.
#[cfg(any(target_os="linux",test))]
fn full_error(logits:&[f32],bytes:&[u8])->Result<f32,String>{
    if logits.is_empty()||bytes.len()!=logits.len()*4{return Err(format!("HF row width mismatch: device_logits={} reference_bytes={}",logits.len(),bytes.len()));}
    let mut max=0f32;
    for(at,(chunk,&actual))in bytes.as_chunks::<4>().0.iter().zip(logits).enumerate(){
        let reference=f32::from_le_bytes(*chunk);
        if !actual.is_finite()||!reference.is_finite(){return Err(format!("HF full logit {at} is not finite: device={actual} reference={reference}"));}
        max=max.max((actual-reference).abs());
    }
    Ok(max)
}
fn main(){
    #[cfg(target_os="linux")]
    if let Err(e)=run(){eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
#[global_allocator]
static ALLOCATOR:spark_tp::runtime::TrackedAllocator=spark_tp::runtime::TrackedAllocator;
#[cfg(test)]
mod tests{
    use super::{full_error,top_error};
    fn le(v:&[f32])->Vec<u8>{v.iter().flat_map(|v|v.to_le_bytes()).collect()}
    #[test]
    fn top_logits_fail_on_empty_unequal_outside_or_nonfinite_reference_data(){
        assert_eq!(top_error(&[1.0,2.0,3.0],&[2,0],&[3.0,0.5]).unwrap(),0.5);
        assert!(top_error(&[1.0,2.0],&[],&[]).is_err());
        assert!(top_error(&[1.0,2.0],&[0,1],&[1.0]).is_err(),"a short reference list");
        assert!(top_error(&[1.0,2.0],&[2],&[1.0]).is_err());
        assert!(top_error(&[1.0,2.0],&[0],&[f64::NAN]).is_err());
        assert!(top_error(&[f32::NAN,2.0],&[0],&[1.0]).is_err());
        assert!(top_error(&[1.0,2.0],&[1],&[f64::INFINITY]).is_err());
    }
    #[test]
    fn full_logits_fail_on_empty_short_or_nonfinite_reference_data(){
        assert_eq!(full_error(&[1.0,2.0],&le(&[1.0,2.25])).unwrap(),0.25);
        assert!(full_error(&[1.0,2.0],&[]).is_err());
        assert!(full_error(&[],&[]).is_err());
        assert!(full_error(&[1.0,2.0],&le(&[1.0])).is_err());
        assert_eq!(full_error(&[1.0,2.0],&le(&[1.0,f32::NAN])).unwrap_err(),"HF full logit 1 is not finite: device=2 reference=NaN");
        assert!(full_error(&[f32::NAN,2.0],&le(&[1.0,2.0])).is_err());
    }
}
