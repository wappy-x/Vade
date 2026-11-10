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
            let mut s=model.session(&ring);s.configure_kv_bf16(std::env::var("SPARK_KV_BF16").as_deref()==Ok("1"))?;s.barrier()?;
            spark_tp::runtime::reset_heap_peak();let start=Instant::now();let mut logits=s.prefill(&prompt)?;let ttft=start.elapsed().as_secs_f64()*1000.;
            let(mut matches,mut full_max,mut top_max)=(0,0f32,0f64);
            for(i,row)in rows.iter().enumerate(){
                let token=get(row,"token")?.u64().map_err(|e|e.to_string())? as u32;
                if spark_engine::greedy_token(&logits)?==token{matches+=1;}
                let top=ids(get(row,"top_ids")?)?;let expected=get(row,"top_logits")?.array().map_err(|e|e.to_string())?;
                for(&id,e)in top.iter().zip(expected){top_max=top_max.max((logits[id as usize] as f64-e.f64().map_err(|e|e.to_string())?).abs());}
                if let Some(file)=row.object().map_err(|e|e.to_string())?.get("full_logits"){
                    let file=Path::new(path).parent().unwrap().join(file.string().map_err(|e|e.to_string())?);
                    let bytes=std::fs::read(file).map_err(|e|e.to_string())?;
                    if bytes.len()!=logits.len()*4{return Err("HF row width mismatch".into());}
                    for(bytes,&actual)in bytes.as_chunks::<4>().0.iter().zip(&logits){full_max=full_max.max((actual-f32::from_le_bytes(*bytes)).abs());}
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
fn main(){
    #[cfg(target_os="linux")]
    if let Err(e)=run(){eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
#[global_allocator]
static ALLOCATOR:spark_tp::runtime::TrackedAllocator=spark_tp::runtime::TrackedAllocator;
