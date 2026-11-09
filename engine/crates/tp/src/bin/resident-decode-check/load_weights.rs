use super::*;
use spark_gpu_probe::driver::with_backend;
use std::{collections::BTreeMap,io::Write,sync::{Arc,Mutex,mpsc},time::Instant};

pub(super) fn check(root:&Path,rank:&str,parts:&str,out:&Path)->Result<()> {
    let topology=engine_model::plan::layer::Topology {rank:rank.parse::<u32>().map_err(|error|error.to_string())?,parts:parts.parse::<u32>().map_err(|error|error.to_string())?};
    topology.validate().map_err(|error|error.to_string())?;
    fs::create_dir(out).map_err(|error|error.to_string())?;let checkpoint=open_checkpoint(root)?;
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        let load=Instant::now();let weights=spark_tp::resident_weights::Weights::load_for_proof(&checkpoint,backend,topology)?;
        let load_seconds=load.elapsed().as_secs_f64();let hashing=Instant::now();
        let workers=std::thread::available_parallelism().map_or(1,|count|count.get().min(4));
        type Job=(String,String,usize,usize,Vec<u8>);
        let (send,receive)=mpsc::sync_channel::<Job>(workers);let receive=Arc::new(Mutex::new(receive));
        let (results,finished)=mpsc::channel::<(String,String,usize,usize,usize,String)>();
        let checked=std::thread::scope(|scope|->Result<BTreeMap<(String,String,usize),(usize,usize,String)>> {
            let mut threads=Vec::new();
            for worker in 0..workers {let receive=Arc::clone(&receive);let results=results.clone();
                let spawned=std::thread::Builder::new().name(format!("weight-proof-{worker}")).spawn_scoped(scope,move ||->Result<()> {
                    loop {let next=receive.lock().map_err(|_|"weight proof queue poisoned")?.recv();
                        let Ok((name,part,offset,total,bytes))=next else {break;};let count=bytes.len();let digest=sha256::digest(&bytes);
                        results.send((name,part,offset,total,count,digest)).map_err(|_|"weight proof collector stopped")?;
                    }Ok(())
                });
                match spawned {Ok(thread)=>threads.push(thread),Err(error)=>{drop(send);for thread in threads {let _=thread.join();}return Err(error.to_string());}}
            }
            drop(results);
            let copied=weights.visit_device_bytes(backend,|name,part,offset,total,bytes| {
                send.send((name.to_owned(),part.to_owned(),offset,total,bytes)).map_err(|_|"weight proof readers stopped".into())
            });drop(send);
            let mut joined=Ok(());for thread in threads {
                let result=thread.join().unwrap_or_else(|_|Err("weight proof reader panicked".into()));if joined.is_ok() {joined=result;}
            }
            copied?;joined?;let mut rows=BTreeMap::new();
            for (name,part,offset,total,count,digest) in finished {
                if rows.insert((name,part,offset),(total,count,digest)).is_some() {return Err("duplicate device weight chunk".into());}
            }Ok(rows)
        })?;
        let mut manifest=fs::File::create(out.join("DEVICE-BUFFER-SHA256")).map_err(|error|error.to_string())?;
        let mut total_bytes=0u128;let mut coverage:BTreeMap<(String,String),(usize,usize)>=BTreeMap::new();
        for ((name,part,offset),(total,count,digest)) in &checked {
            let row=coverage.entry((name.clone(),part.clone())).or_insert((*total,0));
            if row.0!=*total || row.1!=*offset || *count==0 || offset.checked_add(*count).is_none_or(|end|end>*total) {return Err("device weight chunk coverage differs".into());}
            row.1+=count;total_bytes+=*count as u128;
            writeln!(manifest,"{name}\t{part}\t{offset}\t{count}\t{total}\t{digest}").map_err(|error|error.to_string())?;
        }
        if coverage.values().any(|(total,done)|total!=done) {return Err("incomplete device weight buffer hash".into());}
        let root_hash=sha256::digest_reader(fs::File::open(out.join("DEVICE-BUFFER-SHA256")).map_err(|error|error.to_string())?).map_err(|error|error.to_string())?;
        let summary=format!("LOADER DEVICE HASH PASS rank={} parts={} buffers={} chunks={} bytes={total_bytes} load_seconds={load_seconds:.9} proof_seconds={:.9} manifest_sha256={root_hash}\n",topology.rank,topology.parts,coverage.len(),checked.len(),hashing.elapsed().as_secs_f64());
        write(&out.join("SUMMARY"),summary.as_bytes())?;write(&out.join("COMPLETE"),summary.as_bytes())?;print!("{summary}");Ok(())
    })
}
