//! Greedy generation with the same device engine used by batched TP.
#[cfg(target_os = "linux")]
fn run() -> spark_tp::Result<()> {
    use std::{ffi::CString, io::Write, path::PathBuf, time::Instant};
    use spark_tp::{model::Model, transport::Ring};
    let mut a=std::env::args().skip(1).collect::<Vec<_>>();
    let ignore_eos=if let Some(at)=a.iter().position(|s|s=="--ignore-eos") {a.remove(at);true} else {false};
    let hybrid=if let Some(at)=a.iter().position(|s|s=="--seqpar") {a.remove(at);true} else {false};
    if hybrid && std::env::var("SEQPAR_SERIAL_COMPUTE").as_deref()==Ok("1") {
        return Err("hybrid hand-off requires concurrent ranks; unset SEQPAR_SERIAL_COMPUTE".into());
    }
    let usage="usage: tp-generate RANK WORLD LISTEN NEXT PTX MODEL GENERATED REPEATS [--seqpar] [--ignore-eos] (TOKEN... | --tokens-file FILE | --prompts-file FILE)";
    if a.len()<9 {return Err(usage.into());}
    let number=|i:usize|a[i].parse::<usize>().map_err(|e|e.to_string());
    let (rank,world,generated,repeats)=(number(0)?,number(1)?,number(6)?,number(7)?);
    if generated==0 || repeats==0 {return Err("generated/repeats must be nonzero".into());}
    let concurrent=a[8]=="--prompts-file";
    if concurrent&&hybrid{return Err("--prompts-file currently requires shard-only execution".into());}
    let from_file=a[8]=="--tokens-file"||concurrent;
    let mut tokens=if concurrent {
        if a.len()!=10{return Err(usage.into());}
        if rank==0 {
            let text=std::fs::read_to_string(&a[9]).map_err(|e|e.to_string())?;
            let prompts=text.lines().map(|line|line.split_whitespace().map(|s|s.parse::<u32>().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>()?;
            if prompts.is_empty()||prompts.len()>2048||prompts.iter().any(Vec::is_empty){return Err("provide 1..=2048 nonempty prompts, one per line".into());}
            let mut data=vec![prompts.len() as u32];
            for prompt in prompts {data.push(u32::try_from(prompt.len()).map_err(|_|"prompt too long")?);data.extend(prompt);}
            data
        }else{Vec::new()}
    }else if from_file {
        if a.len()!=10 {return Err(format!("--tokens-file requires exactly one file and no positional token IDs; {usage}"));}
        if rank==0 {
            std::fs::read_to_string(&a[9]).map_err(|e|format!("tokens file {}: {e}",a[9]))?
                .lines().enumerate().map(|(line,id)|id.trim().parse::<u32>()
                    .map_err(|e|format!("tokens file {} line {}: invalid token ID: {e}",a[9],line+1)))
                .collect::<Result<Vec<_>,_>>()?
        } else {Vec::new()}
    } else {
        a[8..].iter().map(|s|s.parse::<u32>().map_err(|e|format!("invalid prompt token ID {s:?}: {e}"))).collect::<Result<Vec<_>,_>>()?
    };
    if rank==0&&tokens.is_empty() {return Err("prompt must contain at least one token ID".into());}
    let ptx=CString::new(std::fs::read(&a[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let root=if let Some(p)=a[5].strip_prefix("~/") {PathBuf::from(std::env::var("HOME").map_err(|e|e.to_string())?).join(p)} else {PathBuf::from(&a[5])};
    let mut ring=Ring::connect(rank,world,a[2].parse().map_err(|_|"listen")?,a[3].parse().map_err(|_|"next")?)?;
    spark_tp::runtime::verify_artifacts(&mut ring, &ptx, &root)?;
    if from_file {
        // Only rank zero needs the file; every rank receives the same prompt.
        let mut count=(tokens.len() as u64).to_le_bytes();ring.command(&mut count)?;
        let count=usize::try_from(u64::from_le_bytes(count)).map_err(|_|"prompt length overflow")?;
        let mut bytes=if rank==0 {tokens.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>()}
            else {vec![0;count.checked_mul(4).ok_or("prompt byte count overflow")?]};
        ring.command_payload(&mut bytes)?;
        if rank!=0 {tokens=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();}
    }
    let prompts=if concurrent {
        let mut at=1;let mut prompts=Vec::with_capacity(tokens[0] as usize);
        for _ in 0..tokens[0] {let n=tokens[at] as usize;at+=1;prompts.push(tokens[at..at+n].to_vec());at+=n;}
        prompts
    }else{Vec::new()};
    let prompt_len=if concurrent{prompts.iter().map(Vec::len).max().unwrap()}else{tokens.len()};
    let capacity=prompt_len.checked_add(generated).ok_or("token count overflow")?;
    let partition=if hybrid {
        let weight=std::env::var("SEQPAR_ATTENTION_WEIGHT").map_or(Ok(spark_seqpar::Partition::prefill_weight(tokens.len())),|s|s.parse::<f64>().map_err(|e|e.to_string()))?;
        Some(spark_seqpar::Partition::prefill(tokens.len(),world,weight)?)
    } else {None};
    let mut sequence_ring=if hybrid && world>1 {
        let endpoint=|text:&str|->spark_tp::Result<std::net::SocketAddr>{
            let mut address=text.parse::<std::net::SocketAddr>().map_err(|e|e.to_string())?;
            address.set_port(address.port().checked_add(1024).ok_or("sequence port overflow")?);Ok(address)
        };
        let id=std::env::var("TP_RUN_ID").map_err(|_|"--seqpar needs TP_RUN_ID for multi-rank runs")?;
        let config=std::fs::read(root.join("config.json")).map_err(|e|format!("read model config: {e}"))?;
        let signature=tokens.iter().flat_map(|id|id.to_le_bytes())
            .chain(partition.as_ref().unwrap().offsets().iter().flat_map(|offset|(*offset as u64).to_le_bytes()))
            .chain(config)
            .fold(0xcbf29ce484222325u64,|hash,byte|(hash^u64::from(byte)).wrapping_mul(0x100000001b3));
        let id=format!("{id}:{signature:016x}");
        Some(if ring.topology()=="path" {spark_seqpar::Ring::connect_path(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?}
            else {spark_seqpar::Ring::connect(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?})
    } else {None};
    let ring=std::cell::RefCell::new(ring);
    let budget=if hybrid {spark_seqpar::memory_budget()?}else{
        let config=engine_model::qwen3::Config::open(&root.join("config.json")).map_err(|e|e.to_string())?;
        let base=Model::device_budget(&config,rank,world,capacity)?;
        let kv=2*config.layers as usize*spark_tp::rows(config.kv_heads as usize,rank,world).len()*config.head_dim as usize*capacity*4;
        base.checked_add(kv.checked_mul(prompts.len().saturating_sub(1)).ok_or("batch KV budget overflow")?).ok_or("batch device budget overflow")?
    };
    spark_gpu_probe::driver::with_backend(0,&ptx,budget,|b| {
        let full=if hybrid {
            let mut model=spark_engine::qwen::Model::load_with_kv_budget(b,&root,capacity,budget)?;
            model.prepare_batch_weights()?;Some(model)
        }else{None};
        let model=if let Some(full)=&full {Model::from_full(full,rank,world)?}else{Model::load(b,&root,rank,world,capacity)?};
        if concurrent {
            let ring=ring.into_inner();
            if rank!=0 {
                spark_tp::service::worker(&model,ring)?;
            }else{
                let engine=spark_tp::service::Engine::new(&model,ring)?;
                for repeat in 0..=repeats {
                    spark_tp::runtime::reset_heap_peak();
                    let start=Instant::now();
                    let mut sessions=prompts.iter().map(|_|engine.session()).collect::<Result<Vec<_>,_>>()?;
                    let mut rows=sessions.iter_mut().zip(&prompts).map(|(session,p)|spark_serve::BatchRows{session,tokens:p,output:spark_serve::BatchOutput::Last,greedy:false,top_k:0}).collect::<Vec<_>>();
                    let output=spark_serve::Engine::prefill_batch(&engine,&mut rows).map_err(|e|e.message)?;
                    let mut ids=output.into_iter().map(|p|match p{spark_serve::Predictions::Logits(v)=>spark_engine::greedy_token(&v[0]).map(|id|vec![id]),spark_serve::Predictions::Greedy(ids)=>Ok(ids),spark_serve::Predictions::TopK(_)=>Err("unexpected compact prompt output".into()),spark_serve::Predictions::Rejected(error)=>Err(error)}).collect::<Result<Vec<_>,_>>()?;
                    let prefill=start.elapsed().as_secs_f64();let decode=Instant::now();let mut steps=0;
                    for _ in 1..generated {
                        let active=ids.iter().enumerate().filter_map(|(i,row)|(ignore_eos||row.last().copied().unwrap() as u64!=model.config().eos_token).then_some(i)).collect::<Vec<_>>();
                        if active.is_empty(){break;}
                        let input=active.iter().map(|&i|*ids[i].last().unwrap()).collect::<Vec<_>>();
                        let mut rows=sessions.iter_mut().enumerate().filter_map(|(i,s)|active.binary_search(&i).is_ok().then_some(s)).collect::<Vec<_>>();
                        let output=engine.decode_batch(&mut rows,&input,true)?;steps+=output.len();
                        for (&i,row) in active.iter().zip(output){ids[i].push(row[0] as u32);}
                    }
                    let seconds=decode.elapsed().as_secs_f64();
                    if repeat>0 {
                        println!("batch world={world} repeat={repeat} chats={} prompt_tokens={} decode_steps={steps} prefill_s={prefill:.6} all_first_ms={:.3} decode_s={seconds:.6} aggregate_decode_tok_s={:.3} aggregate_total_tok_s={:.3}",prompts.len(),prompts.iter().map(Vec::len).sum::<usize>(),prefill*1000.,steps as f64/seconds,ids.iter().map(Vec::len).sum::<usize>() as f64/(prefill+seconds));
                        for (chat,ids) in ids.iter().enumerate(){println!("reply chat={chat} tokens={ids:?}");}
                    }
                }
                engine.shutdown()?;
            }
            println!("memory rank={rank} model_bytes={} device_bytes={} device_peak_bytes={} pinned_peak_bytes={} host_heap_peak_bytes={}",model.resident_bytes,b.allocated_bytes(),b.peak_allocated_bytes(),b.pinned_upload_bytes(),spark_tp::runtime::heap_peak());
            return Ok(());
        }
        for repeat in 0..=repeats {
            spark_tp::runtime::reset_heap_peak();
            let mut s=model.session(&ring);s.barrier()?;
            let start=Instant::now();
            let mut first=None;
            let mut handoff_after_first_ms=0.;
            let (mut token,prefill,handoff_bytes,handoff_ms)=if let Some(full)=&full {
                b.use_vec8(false)?;
                let mut source=full.session();source.configure_kv_bf16(false)?;
                let mut next=[0;4];
                let mut bytes=0;let mut ms=0.;let mut prefill=0.;
                spark_seqpar::prefill::prefill_with_finish(&mut source,&tokens,sequence_ring.as_mut(),partition.as_ref().unwrap(),
                    spark_seqpar::prefill::PrefillOptions{overlap:std::env::var("SEQPAR_OVERLAP").as_deref()!=Ok("0"),serialize_compute:false},|source,pending_layer| {
                        let handoff=Instant::now();
                        bytes=s.import_sequence_with_finish(source,partition.as_ref().unwrap(),|source| {
                            if let Some(layer)=pending_layer {source.sequence_finish_layer(layer)?;}
                            let logits=source.sequence_finish()?;
                            prefill=start.elapsed().as_secs_f64();
                            if rank+1==world {
                                let token=spark_engine::greedy_token(&logits)?;
                                next=token.to_le_bytes();
                                first=Some(Instant::now());
                                if repeat>0 {
                                    println!("first_token rank={rank} repeat={repeat} generated_index=0 token={token} ttft_ms={:.3}",first.unwrap().duration_since(start).as_secs_f64()*1000.);
                                    std::io::stdout().flush().map_err(|e|e.to_string())?;
                                }
                            }
                            Ok(())
                        })?;
                        s.barrier()?;
                        ms=handoff.elapsed().as_secs_f64()*1000.;
                        handoff_after_first_ms=first.map_or(0.,|t|t.elapsed().as_secs_f64()*1000.);
                        Ok(Vec::new())
                    })?;
                ring.borrow_mut().transfer_from_last(0,&mut next)?;
                b.use_vec8(true)?;
                (u32::from_le_bytes(next),prefill,bytes,ms)
            }else{
                let logits=s.prefill(&tokens)?;
                (spark_engine::greedy_token(&logits)?,start.elapsed().as_secs_f64(),0,0.)
            };
            let first=first.unwrap_or_else(Instant::now);
            let ttft=first.duration_since(start).as_secs_f64()*1000.;
            let mut ids=vec![token];let mut steady=None;let mut first_to_second_ms=0.;
            let reduction_before={let transport=ring.borrow();(transport.reduction_bytes,transport.reduction_elapsed)};
            for _ in 1..generated {
                if !ignore_eos && token as u64==model.config().eos_token {break;}
                token=s.step_greedy(token)?;ids.push(token);
                if ids.len()==2 {let second=Instant::now();first_to_second_ms=second.duration_since(first).as_secs_f64()*1000.;steady=Some(second);}
            }
            let seconds=steady.map_or(0.,|t|t.elapsed().as_secs_f64());
            let decode=if ids.len()>2 {(ids.len()-2) as f64/seconds}else{0.};
            if repeat>0 {
                let transport=ring.borrow();let reduction_bytes=transport.reduction_bytes-reduction_before.0;
                let reduction_ms=(transport.reduction_elapsed-reduction_before.1).as_secs_f64()*1000.;
                println!("reduction rank={rank} repeat={repeat} decode_steps={} bytes={reduction_bytes} ms={reduction_ms:.3} ms_per_step={:.3}",ids.len()-1,reduction_ms/ids.len().saturating_sub(1).max(1) as f64);
                println!("memory rank={rank} repeat={repeat} model_bytes={} kv_bytes={} device_bytes={} device_peak_bytes={} pinned_peak_bytes={} host_heap_peak_bytes={} app_peak_bytes={} process_hwm_bytes={}",model.resident_bytes,s.cache_bytes(),b.allocated_bytes(),b.peak_allocated_bytes(),s.pinned_peak_bytes(),spark_tp::runtime::heap_peak(),b.peak_allocated_bytes()+s.pinned_peak_bytes()+spark_tp::runtime::heap_peak(),spark_tp::runtime::host_memory().1);
                if rank==if hybrid {world-1}else{0} {println!("world={world} repeat={repeat} seqpar={hybrid} prompt={} generated={} prefill_s={prefill:.6} kv_redistribution_bytes={handoff_bytes} kv_redistribution_ms={handoff_ms:.3} kv_after_first_ms={handoff_after_first_ms:.3} ttft_ms={ttft:.3} first_to_second_ms={first_to_second_ms:.3} decode_tok_s={decode:.3} tokens={ids:?}",tokens.len(),ids.len());}
            }
        }
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e)=run() {eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os = "linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
#[global_allocator]
static ALLOCATOR:spark_tp::runtime::TrackedAllocator=spark_tp::runtime::TrackedAllocator;
