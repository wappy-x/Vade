//! TP throughput, strict verification and rollback using the real model.
#[cfg(target_os="linux")]
fn run()->spark_tp::Result<()> {
    use std::{ffi::CString,path::PathBuf,time::Instant};
    use spark_tp::{model::Model,transport::Ring};
    use spark_gpu_probe::driver::with_backend;
    use spark_engine::greedy_token;
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()!=10 {return Err("usage: tp-bench RANK WORLD LISTEN NEXT PTX MODEL MAX_PROMPT DECODE REPEATS check|bench".into());}
    let num=|i:usize|a[i].parse::<usize>().map_err(|e|e.to_string());
    let (rank,world,max,decode,repeats)=(num(0)?,num(1)?,num(6)?,num(7)?,num(8)?);
    if max<64 || max>8192 || decode==0 || repeats==0 {return Err("invalid benchmark dimensions".into());}
    let check=match a[9].as_str(){"check"=>true,"bench"=>false,_=>return Err("mode must be check or bench".into())};
    let bf16=std::env::var("SPARK_KV_BF16").as_deref()==Ok("1");
    let ptx=CString::new(std::fs::read(&a[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let root=if let Some(p)=a[5].strip_prefix("~/"){PathBuf::from(std::env::var("HOME").map_err(|e|e.to_string())?).join(p)}else{PathBuf::from(&a[5])};
    let ring=std::cell::RefCell::new(Ring::connect(rank,world,a[2].parse().map_err(|_|"listen")?,a[3].parse().map_err(|_|"next")?)?);
    spark_tp::runtime::verify_artifacts(&mut ring.borrow_mut(), &ptx, &root)?;
    let prompt=(1..=max as u32).collect::<Vec<_>>();
    let lengths=if max==64 {vec![64]} else {vec![64,max]};
    with_backend(0,&ptx,24*1024*1024*1024,|b|{
        let m=Model::load(b,&root,rank,world,max+decode+32)?;
        println!("LOAD rank={rank} world={world} resident_bytes={} load_peak_bytes={} host_heap_peak_bytes={}",m.resident_bytes,m.load_peak_bytes,spark_tp::runtime::heap_peak());
        ring.borrow_mut().barrier()?;
        for &n in &lengths {
            for repeat in 0..=repeats {
                spark_tp::runtime::reset_heap_peak();
                let mut s=m.session(&ring);s.configure_kv_bf16(bf16)?;s.barrier()?;
                let start=Instant::now();let logits=s.prefill(&prompt[..n])?;let ms=start.elapsed().as_secs_f64()*1000.;
                if repeat>0 {memory(rank,world,"prefill",n,repeat,&m,&s);println!("PREFILL rank={rank} world={world} tokens={n} repeat={repeat} bf16={bf16} ms={ms:.3} top1={}",greedy_token(&logits)?);}
            }
        }
        for repeat in 0..=repeats {
            let mut s=m.session(&ring);s.configure_kv_bf16(bf16)?;
            let mut token=greedy_token(&s.prefill(&prompt[..64])?)?;
            s.barrier()?;spark_tp::runtime::reset_heap_peak();
            let (calls_before,net_before)=s.collective_stats();
            let start=Instant::now();let mut ids=Vec::with_capacity(decode);
            for _ in 0..decode {token=s.step_greedy(token)?;ids.push(token);}
            let seconds=start.elapsed().as_secs_f64();
            if repeat>0 {
                let (calls,net)=s.collective_stats();let net_ms=(net-net_before).as_secs_f64()*1000./decode as f64;
                memory(rank,world,"decode",64,repeat,&m,&s);
                println!("DECODE rank={rank} world={world} repeat={repeat} context=64 count={decode} ms={:.3} tok_s={:.3} collective_ms={net_ms:.3} gathers_per_token={} tokens={ids:?}",seconds*1000./decode as f64,decode as f64/seconds,(calls-calls_before)/decode as u64);
            }
        }
        let mut s=m.session(&ring);s.configure_kv_bf16(bf16)?;
        s.prefill(&prompt[..64])?;let checkpoint=s.checkpoint();
        let verify=(65..81).collect::<Vec<u32>>();
        let reference=if check {Some(verify.iter().map(|&id|s.step(id)).collect::<Result<Vec<_>,_>>()?)}else{None};
        s.truncate(checkpoint)?;
        for repeat in 0..=repeats {
            s.truncate(checkpoint)?;s.barrier()?;spark_tp::runtime::reset_heap_peak();
            let start=Instant::now();let actual=s.verify_batch(&verify)?;let ms=start.elapsed().as_secs_f64()*1000.;
            if let Some(reference)=&reference {
                let mismatch=actual.iter().flatten().zip(reference.iter().flatten()).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
                println!("EXACT rank={rank} rows=16 bit_mismatches={mismatch}");
                if mismatch!=0 {return Err("verify differs from sequential decode".into());}
            }
            if repeat>0 {memory(rank,world,"verify",16,repeat,&m,&s);println!("VERIFY rank={rank} world={world} prefix=64 rows=16 repeat={repeat} ms={ms:.3}");}
        }
        if check {
            s.truncate(checkpoint+7)?;
            let actual=s.step(72)?;
            if let Some(reference)=&reference {if actual.iter().zip(&reference[7]).any(|(a,b)|a.to_bits()!=b.to_bits()){return Err("rollback/reappend changed logits".into());}}
            println!("ROLLBACK rank={rank} position={} exact=true",s.len());
        }
        Ok(())
    })
}
#[cfg(target_os="linux")]
fn memory(rank:usize,world:usize,phase:&str,n:usize,repeat:usize,m:&spark_tp::model::Model<'_,'_>,s:&spark_tp::model::Session<'_,'_,'_,'_>){
    let heap=spark_tp::runtime::heap_peak();let(rss,hwm)=spark_tp::runtime::host_memory();
    println!("MEMORY rank={rank} world={world} phase={phase} n={n} repeat={repeat} model_bytes={} kv_bytes={} device_peak_bytes={} pinned_peak_bytes={} host_heap_peak_bytes={heap} app_peak_bytes={} rss_bytes={rss} hwm_bytes={hwm}",m.resident_bytes,s.cache_bytes(),s.device_peak_bytes(),s.pinned_peak_bytes(),s.device_peak_bytes()+s.pinned_peak_bytes()+heap);
}
fn main(){
    #[cfg(target_os="linux")]
    if let Err(e)=run(){eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
#[global_allocator]
static ALLOCATOR:spark_tp::runtime::TrackedAllocator=spark_tp::runtime::TrackedAllocator;
