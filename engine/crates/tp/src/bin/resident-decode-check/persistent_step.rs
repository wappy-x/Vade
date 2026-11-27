//! Two local ranks over the small saved-weight checkpoint. No model speed claim.
use std::{fs,net::TcpListener,path::Path,sync::Barrier,time::Instant};
use spark_tp::{Result,bounded_decoder::{Compute,Decoder,Device,HeadRows},transport::Ring};
use super::super::select_known::attention as checkpoint;
fn save(path:&Path,bytes:&[u8])->Result<()> {super::super::write(path,bytes)}
fn words(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_le_bytes()).collect()}
fn rank(rank:usize,addresses:[std::net::SocketAddr;2],out:&Path,barrier:&Barrier)->Result<()> {
    let c=checkpoint::Checkpoint::build_with(engine_model::safetensors::Dtype::Bf16,true,4200,false,|d| {
        if let engine_model::plan::description::TokenMixer::LatentAttention(a)=&mut d.layers[1].mixer {
            a.indexer.token_budget=2048;a.indexer.pool=4;
        }
    });
    let ring=Ring::connect_options(rank,2,addresses[rank],addresses[1-rank],true)?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,8usize<<30,|b| {
        b.configure_vector_pool(1<<30);
        // The cost setting is the same Device setting as the probe's --cost.
        // Local TCP exercises the split and the real decoder. Mailbox bits are
        // covered by exchange-stage; full model and RDMA costs need the steward.
        let d=Device::with_ring_profiled(b,Some(ring),true);
        let queue=b.resident_queue()?;b.enable_resident_launches()?;
        for context in [1040usize,4158] {
            let prompt=(0..context).map(|i|1+(i as u32*5)%15).collect::<Vec<_>>();
            for repeat in 1..=3 {
                let saved=out.join(format!("rank{rank}/c{context}/run{repeat}"));fs::create_dir_all(&saved).map_err(|e|e.to_string())?;
                let decoder=Decoder::new(&c.admission,&d,4200)?;let mut state=decoder.zero_state()?;let mut logits=Vec::new();
                for chunk in prompt.chunks(spark_tp::bounded_decoder::PROMPT_GROUP) {
                    let (mut rows,next)=decoder.prefill_rows(chunk,&state,HeadRows::Last)?;
                    state=next;logits=rows.pop().ok_or("small fixture has no prompt logits")?;
                }
                d.cost_report("fixture-prefill");
                save(&saved.join("INPUT_IDS"),prompt.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes())?;
                save(&saved.join("PROMPT_LOGITS"),&words(&logits))?;
                let mut ids=Vec::new();let before=b.stats();barrier.wait();let started=Instant::now();
                for at in 0..8 {
                    let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
                    if logits.iter().any(|value|!value.is_finite()) {return Err("small fixture has nonfinite logits".into());}
                    ids.push(best as u32);save(&saved.join(format!("{at}-logits.f32")),&words(&logits))?;
                    if at<7 {let (next,_,updated)=decoder.step(best as u32,&state,|_,_|Ok(()))?;logits=next;state=updated;}
                }
                let wall=started.elapsed().as_nanos();barrier.wait();let counts=b.stats().since(before);d.cost_report("fixture-decode");
                let timing=format!("fixture_only=true rank={rank} cell=c{context} repeat={repeat} decode_steps=7 whole_step_ns={} graph_replays={} host_launches={} stream_waits={} raw_logits=true transport=local_tcp cost=true\n",wall/7,counts.graph_launches,counts.queued_calls,counts.stream_wait_calls);
                print!("RESIDENT FIXTURE COST {timing}");save(&saved.join("TIMING"),timing.as_bytes())?;
                save(&saved.join("OUTPUT_IDS"),ids.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes())?;
            }
        }
        drop(queue);Ok(())
    })
}
pub fn check(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let listeners=[TcpListener::bind("127.0.0.1:0"),TcpListener::bind("127.0.0.1:0")];
    let a=listeners[0].as_ref().map_err(|e|e.to_string())?.local_addr().map_err(|e|e.to_string())?;
    let b=listeners[1].as_ref().map_err(|e|e.to_string())?.local_addr().map_err(|e|e.to_string())?;drop(listeners);
    let barrier=Barrier::new(2);
    std::thread::scope(|scope| {
        let first=scope.spawn(||rank(0,[a,b],out,&barrier));let second=scope.spawn(||rank(1,[a,b],out,&barrier));
        let first=first.join().map_err(|_|"small fixture rank zero panicked")?;
        let second=second.join().map_err(|_|"small fixture rank one panicked")?;first?;second
    })?;
    save(&out.join("COMPLETE"),b"PERSISTENT STEP FIXTURE PASS ranks=2 contexts=1040,4158 repeats=3 fixture_only=true\n")
}
