//! Compare batched prefill and teacher-forced decode with supplied HF logits.
#[cfg(target_os="linux")]
fn run()->Result<(),String>{
    use spark_gpu_probe::driver::KvFormat;
    use engine_model::json::{self,Value,field};
    use std::{ffi::CString,path::Path,time::Instant};
    fn get<'a>(v:&'a Value,k:&str)->Result<&'a Value,String>{field(v.object().map_err(|e|e.to_string())?,k).map_err(|e|e.to_string())}
    fn ids(v:&Value)->Result<Vec<u32>,String>{v.array().map_err(|e|e.to_string())?.iter().map(|v|{let n=v.u64().map_err(|e|e.to_string())?;u32::try_from(n).map_err(|_|"token ID exceeds u32".to_string())}).collect()}
    let mut args:Vec<_>=std::env::args().skip(1).collect();
    let fast_prefill=args.iter().any(|s|s=="--fast-prefill"); args.retain(|s|s!="--fast-prefill");
    let suite=args.iter().any(|s|s=="--suite"); args.retain(|s|s!="--suite");
    let bf16=suite||args.iter().any(|s|s=="--bf16-kv");
    let i16=args.iter().any(|s|s=="--i16-kv");
    if bf16&&i16{return Err("choose only one of --bf16-kv or --i16-kv".into());}
    let format=if i16{KvFormat::ScaledI16}else if bf16{KvFormat::Bf16}else{KvFormat::Fp32};
    args.retain(|s|s!="--bf16-kv"&&s!="--i16-kv");
    if args.len()<3{return Err("usage: hf-batch-check PTX MODEL FIXTURE.json... [--bf16-kv|--i16-kv] [--fast-prefill] [--suite]".into());}
    let ptx=CString::new(std::fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let fixtures=args[2..].iter().map(|p|{let data=std::fs::read(p).map_err(|e|e.to_string())?;json::parse(&data,json::Limits::default()).map_err(|e|e.to_string())}).collect::<Result<Vec<_>,_>>()?;
    // Derive capacity from the supplied histories, including teacher-forced decode.
    let mut context=1usize;
    for fixture in &fixtures {
        let prompt=ids(get(fixture,"prompt_ids")?)?;
        let rows=get(fixture,"rows")?.array().map_err(|e|e.to_string())?;
        if prompt.is_empty()||rows.is_empty(){return Err("empty HF prompt/rows".into());}
        if suite && (rows.len()!=64 || get(fixture,"device")?.string().map_err(|e|e.to_string())?!="cuda"
            || get(fixture,"dtype")?.string().map_err(|e|e.to_string())?!="bfloat16") {
            return Err("suite requires 64 CUDA BF16 reference positions per prompt".into());
        }
        context=context.max(prompt.len().checked_add(rows.len()-1).ok_or("HF context overflow")?);
    }
    let config=engine_model::qwen3::Checkpoint::inspect(Path::new(&args[1])).map_err(|e|e.to_string())?.config;
    let kv_budget=[config.layers,config.kv_heads,config.head_dim,context as u64,2,4].into_iter()
        .try_fold(1u64,|bytes,n|bytes.checked_mul(n).ok_or("HF KV budget overflow"))?;
    let kv_budget=usize::try_from(kv_budget).map_err(|_|"HF KV budget exceeds address space")?;
    spark_gpu_probe::driver::with_backend_mode(0,&ptx,100*1024*1024*1024,spark_engine::configured_matvec_mode()?,|b|{
        let mut model=spark_engine::qwen::Model::load_with_kv_budget(b,Path::new(&args[1]),context,kv_budget)?;
        model.prepare_batch_weights()?;
        let (mut failures,mut total_matches,mut total_rows,mut total_near,mut total_cpu_diffs,mut total_cpu_rows,mut peak)=(0,0,0,0,0,0,0usize);
        let total_start=Instant::now();
        for (fixture,path) in fixtures.iter().zip(&args[2..]){
            let noise=fixture.object().map_err(|e|e.to_string())?.get("fp32_comparison");
            let logit_tolerance=noise.map(|n|get(n,"full_logits_max_abs")?.f64().map(|v|2.0*v).map_err(|e|e.to_string())).transpose()?;
            let prompt=ids(get(fixture,"prompt_ids")?)?;
            let rows=get(fixture,"rows")?.array().map_err(|e|e.to_string())?;
            let mut session=model.session();
            session.configure_kv_format(format)?;
            session.configure_fast_prefill(fast_prefill)?;
            let start=Instant::now();
            let mut logits=session.prefill(&prompt)?;
            let ttft=start.elapsed().as_secs_f64()*1000.0;
            let (mut near,mut cpu_diffs,mut cpu_rows)=(0,0,0);
            let (mut matches,mut full_max,mut top_max)=(0,0.0f32,0.0f64);
            for (i,row) in rows.iter().enumerate(){
                if logits.iter().any(|x|!x.is_finite()){return Err(format!("non-finite logits in fixture {path}, row {i}, KV {format:?}"));}
                let token=u32::try_from(get(row,"token")?.u64().map_err(|e|e.to_string())?).map_err(|_|"reference token exceeds u32")?;
                let actual=spark_engine::greedy_token(&logits)?;
                if actual==token{matches+=1;}
                let top=ids(get(row,"top_ids")?)?;
                let expected=get(row,"top_logits")?.array().map_err(|e|e.to_string())?;
                if top.len()!=expected.len() || (suite && top.len()!=20) || top.iter().any(|&id|id as usize>=logits.len())
                    || token as usize>=logits.len() {return Err(format!("invalid top-20 row in {path}, position {i}"));}
                let values=expected.iter().map(|e|e.f64().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
                if suite && (values.windows(2).any(|w|w[0]<w[1])
                    || top.iter().enumerate().any(|(j,id)|top[..j].contains(id))
                    || !top.iter().zip(&values).any(|(&id,&value)|id==token && value==values[0])) {
                    return Err(format!("reference top-20 is not a greedy descending row in {path}, position {i}"));
                }
                let cpu=row.object().map_err(|e|e.to_string())?.get("cpu");
                let cpu_token=cpu.map(|c|get(c,"token")?.u64().map_err(|e|e.to_string())).transpose()?;
                if let Some(cpu_token)=cpu_token {
                    cpu_rows+=1;
                    if cpu_token!=token as u64 {cpu_diffs+=1;
                        let cpu_logits=get(cpu.unwrap(),"top_logits")?.array().map_err(|e|e.to_string())?;
                        if cpu_logits.len()<2{return Err("CPU reference needs its top-two logits".into());}
                        let cpu_margin=cpu_logits[0].f64().map_err(|e|e.to_string())?-cpu_logits[1].f64().map_err(|e|e.to_string())?;
                        println!("HF_CPU_CUDA fixture={} position={i} input_position={} cuda_token={token} cpu_token={cpu_token} cuda_margin={:.8} cpu_margin={cpu_margin:.8}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,prompt.len()-1+i,values[0]-values[1]);
                    }
                }
                if suite && actual==token {
                    println!("HF_POSITION fixture={} position={i} reference_token={token} vade_token={actual} match=true reference_margin={:.8} cpu_token={cpu_token:?}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,values[0]-values[1]);
                }
                if actual!=token && values.len()>=2 {
                    let margin=values[0]-values[1];
                    // BF16 spacing at the reference winning logit's magnitude; strict inequality.
                    let step=2.0f64.powi((values[0].abs().log2().floor() as i32).max(-126)-7);
                    let is_near=margin<step;
                    if is_near {near+=1;}
                    let actual_ref=top.iter().position(|&id|id==actual).map(|j|values[j]);
                    println!("HF_DISAGREEMENT fixture={} position={i} input_position={} reference_token={token} vade_token={actual} reference_margin={margin:.8} bf16_step={step:.8} near_tie={is_near} reference_actual_logit={actual_ref:?} cpu_token={cpu_token:?}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,prompt.len()-1+i);
                }
                for (&id,e) in top.iter().zip(expected){top_max=top_max.max((logits[id as usize] as f64-e.f64().map_err(|e|e.to_string())?).abs());}
                if let Some(file)=row.object().map_err(|e|e.to_string())?.get("full_logits"){
                    let file=Path::new(path).parent().unwrap().join(file.string().map_err(|e|e.to_string())?);
                    let bytes=std::fs::read(file).map_err(|e|e.to_string())?;
                    if bytes.len()!=logits.len()*4{return Err("HF row width mismatch".into());}
                    for (bytes,&actual) in bytes.as_chunks::<4>().0.iter().zip(&logits){full_max=full_max.max((actual-f32::from_le_bytes(*bytes)).abs());}
                }
                if i+1<rows.len(){logits=session.step(token)?;}
            }
            if suite {
                println!("HF fixture={} prompt={} kv_format={format:?} top1={matches}/{} full_logits=not_stored top20_max_abs={top_max:.8} ttft_ms={ttft:.3} kv_bytes={} peak_device_bytes={}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,prompt.len(),rows.len(),session.cache_bytes(),session.device_peak_bytes());
            } else {
                println!("HF fixture={} prompt={} bf16={bf16} kv_format={format:?} top1={matches}/{} full_max_abs={full_max:.8} top20_max_abs={top_max:.8} ttft_ms={ttft:.3} kv_bytes={} peak_device_bytes={}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,prompt.len(),rows.len(),session.cache_bytes(),session.device_peak_bytes());
            }
            total_matches+=matches; total_rows+=rows.len(); total_near+=near;
            total_cpu_diffs+=cpu_diffs; total_cpu_rows+=cpu_rows; peak=peak.max(session.device_peak_bytes());
            println!("HF_COUNTS fixture={} disagreements={} near_ties={near} cpu_cuda_disagreements={cpu_diffs}/{cpu_rows}",get(fixture,"name")?.string().map_err(|e|e.to_string())?,rows.len()-matches);
            // Owner gate: exact tokens and twice each case's BF16/FP32 logit difference.
            let tolerance=if suite{0.0}else{logit_tolerance.unwrap_or(if bf16{f64::INFINITY}else{0.01})};
            let logits_failed=if suite{false}else if logit_tolerance.is_some(){full_max as f64>tolerance||top_max>tolerance}else{(i16&&(full_max>=0.01||top_max>=0.01))||(!bf16&&!i16&&(full_max>0.01||top_max>0.01))};
            if matches!=rows.len()||logits_failed{failures+=1;}
            if !suite && model.config().num_experts!=0 {
                drop(session);
                let routing_path=Path::new(path).with_file_name(format!("{}-routing.json",Path::new(path).file_stem().unwrap().to_string_lossy()));
                let data=std::fs::read(&routing_path).map_err(|e|format!("{}: {e}",routing_path.display()))?;
                let routing=json::parse(&data,json::Limits::default()).map_err(|e|e.to_string())?;
                let positions=get(&routing,"positions")?.array().map_err(|e|e.to_string())?;
                let mut tokens=prompt.clone();
                for row in rows.iter().take(rows.len().saturating_sub(1)){tokens.push(get(row,"token")?.u64().map_err(|e|e.to_string())? as u32);}
                let mut expected=std::collections::BTreeMap::new();
                for position in positions {
                    let p=get(position,"input_position")?.u64().map_err(|e|e.to_string())? as usize;
                    let layers=get(position,"layers")?.array().map_err(|e|e.to_string())?;
                    if p>=tokens.len()||layers.len()!=model.config().layers as usize{return Err("invalid routing fixture position/layers".into());}
                    for (layer,row) in layers.iter().enumerate(){if expected.insert((layer,p),row).is_some(){return Err("duplicate routing position".into());}}
                }
                if expected.is_empty(){return Err("empty routing fixture".into());}
                let allowed_ids=noise.map(|n|get(n,"routing_ids_total")?.u64().map(|v|v*3/100).map_err(|e|e.to_string())).transpose()?.unwrap_or(0) as usize;
                let weight_tolerance=if noise.is_some(){f64::INFINITY}else{1e-5};
                let (mut matched,mut observed,mut agreed_ids,mut total_ids,mut max_weight_error)=(0,0,0,0,0.0f64);
                let mut session=model.session(); session.configure_kv_format(format)?; session.configure_fast_prefill(fast_prefill)?;
                session.prefill_with_routing_observer(&tokens,|layer,position,selected|{
                    if let Some(row)=expected.get(&(layer,position)) {
                        observed+=1;
                        let indices=ids(get(row,"indices")?)?;
                        let weights=get(row,"weights")?.array().map_err(|e|e.to_string())?;
                        if indices.len()!=selected.len()||weights.len()!=selected.len(){return Err("routing fixture width mismatch".into());}
                        total_ids+=selected.len();
                        let mut same=true;
                        for &(id,weight) in selected {
                            if !weight.is_finite(){return Err("non-finite routing weight".into());}
                            let Some(slot)=indices.iter().position(|&expected_id|id==expected_id) else {same=false; continue;};
                            agreed_ids+=1;
                            let expected_weight=&weights[slot];
                            let error=(weight as f64-expected_weight.f64().map_err(|e|e.to_string())?).abs();
                            max_weight_error=max_weight_error.max(error);
                        }
                        if same{matched+=1;}
                    } Ok(())
                })?;
                if let Some(noise)=noise&& get(noise,"routing_ids_total")?.u64().map_err(|e|e.to_string())? as usize!=total_ids{return Err("oracle routing count mismatch".into());}
                println!("ROUTING_PREFILL fixture={path} expert_ids={agreed_ids}/{total_ids} exact_id_rows={matched}/{} observed={observed} max_weight_abs={max_weight_error:.8} weight_tolerance={weight_tolerance:.8} allowed_changed_ids={allowed_ids} logit_tolerance={tolerance:.8} peak_device_bytes={}",expected.len(),session.device_peak_bytes());
                if observed!=expected.len()||total_ids-agreed_ids>allowed_ids||max_weight_error>weight_tolerance{failures+=1;}
            }
        }
        peak=peak.max(b.peak_allocated_bytes());
        println!("HF_TOTAL fixtures={} top1={total_matches}/{total_rows} disagreements={} near_ties={total_near} cpu_cuda_disagreements={total_cpu_diffs}/{total_cpu_rows} elapsed_ms={:.3} peak_device_bytes={peak}",fixtures.len(),total_rows-total_matches,total_start.elapsed().as_secs_f64()*1000.0);
        if failures>0{return Err(if suite{format!("{failures} HF comparisons failed (exact CUDA top-1; near-ties remain failures; top-20 error is descriptive)")}else{format!("{failures} HF comparisons failed (exact top-1; native BF16 oracle envelope when supplied; otherwise legacy tolerances)")});}
        Ok(())
    })
}
fn main(){
    #[cfg(target_os="linux")]
    if let Err(e)=run(){eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
