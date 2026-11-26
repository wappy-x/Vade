//! Native row exactness and TTFT/C1/C8 receipt. Each process is a fresh model load.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_engine::{flash::next_batched::{NextModel, NextSession}, greedy_token};
    use spark_gpu_probe::driver::with_backend;
    use std::{path::PathBuf, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help") {
        println!("flashnext_rows --model DIR [--check] [--measure] [--bounded-head-fallback] [--prefill 8192] [--decode 256] [--budget-gib 110]"); return Ok(());
    }
    let value = |key: &str| args.iter().position(|a| a == key).and_then(|i| args.get(i+1)).cloned();
    let number = |key: &str, default| value(key).map(|v| v.parse::<usize>().map_err(|_| format!("invalid {key}"))).unwrap_or(Ok(default));
    let root = PathBuf::from(value("--model").ok_or("--model required")?);
    let (prompt_len, decode) = (number("--prefill", 8192)?, number("--decode", 256)?);
    let budget = number("--budget-gib", 110)?.checked_mul(1<<30).ok_or("budget overflow")?;
    let capacity = prompt_len.checked_add(decode+1).ok_or("context overflow")?.max(1024);
    if prompt_len == 0 || decode == 0 || capacity > 262144 { return Err("invalid token counts".into()); }
    let fallback = args.iter().any(|a| a == "--bounded-head-fallback");
    let check = args.iter().any(|a| a == "--check");
    let measure = args.iter().any(|a| a == "--measure");
    if !check && !measure && !fallback { return Err("select --check and/or --measure".into()); }
    let bits = |row: Vec<f32>| row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let hash = |row: &[u32]| row.iter().fold(0xcbf29ce484222325u64, |h, v| (h ^ u64::from(*v)).wrapping_mul(0x100000001b3));
    with_backend(0, spark_gpu_probe::layer_ptx()?, budget, |b| {
        let load = Instant::now(); let model = if fallback {
            if !spark_engine::flash_head_options::load()? {return Err("fallback check requires VADE_HEAD_BOUNDS=1".into());}
            NextModel::load_with_mtp(b, &root, spark_engine::flash_next_options::Options::load()?, 5)?
        } else {NextModel::load(b, &root)?};
        println!("NEW FOLDER model_load_s={:.6} allocated_bytes={} sparse=true temperature=0 top_p=1 top_k=0 repetition_penalty=1 presence_penalty=0 frequency_penalty=0", load.elapsed().as_secs_f64(), b.allocated_bytes());
        if fallback {
            for count in 1..=spark_engine::flash_head_options::RETAINED_ROWS {
                let prompt:Vec<_>=(0..129).map(|i|100+(i*31%4096) as u32).collect();
                let input:Vec<_>=(0..count).map(|i|9707+i as u32).collect();
                let mut canonical=model.session(b,1024)?;
                let mut bounded=model.session(b,1024)?;
                // Lookup width admits verify rows past the resident MTP depth.
                if count>1 {canonical.enable_lookup(b,count-1)?;bounded.enable_lookup(b,count-1)?;}
                canonical.prefill_advance(b,&prompt)?;bounded.prefill_advance(b,&prompt)?;
                let ids=NextSession::verify_batch(b,&mut[&mut canonical],&[&input])?;
                let expected=canonical.decode_logits(b,count)?;
                let top=canonical.decode_top_k(b,count,8)?;
                let got=NextSession::verify_batch_greedy(b,&mut[&mut bounded],&[&input])?;
                if got!=ids {return Err("bounded greedy IDs disagree".into());}
                if !bounded.verify_left_sparse() {return Err(format!("verify of {count} rows took the whole head"));}
                // Another verify pass overwrites the shared workspace before asking for full rows.
                canonical.commit_mtp(b,canonical.position())?;NextSession::verify_batch(b,&mut[&mut canonical],&[&input])?;
                let actual=bounded.decode_logits(b,count)?;
                if actual.iter().flatten().map(|v|v.to_bits()).collect::<Vec<_>>()
                    !=expected.iter().flatten().map(|v|v.to_bits()).collect::<Vec<_>>() {
                    return Err("canonical full-logit fallback changed bits".into());
                }
                if bits(bounded.logits(b)?)!=bits(expected[count-1].clone()) {
                    return Err("canonical last-row fallback changed bits".into());
                }
                if bounded.decode_top_k(b,count,8)?!=top {return Err("canonical top-k fallback changed values".into());}
                if bounded.decode_logits(b,count)?!=expected {return Err("full rows unavailable after top-k fallback".into());}
                let forbidden=ids[0][0];
                let sample=|row:&[f32]| spark_engine::sample::Sampler::new(spark_engine::sample::Sampling::greedy())?
                    .sample(row,&prompt,prompt.len(),prompt.len()+1,|id|id!=forbidden);
                if sample(&actual[0])?!=sample(&expected[0])? {return Err("rejected-token fallback differs".into());}
                println!("CHECK bounded_head rows={count} IDs/full_logits/top_k/rejected_winner/overwritten_workspace PASS");
            }
            // The plain greedy step: full cohorts take the bounded head, a padded one (3) does not.
            for chats in [1usize,2,3,4,8] {
                let prompts:Vec<Vec<u32>>=(0..chats).map(|i|(0..129).map(|t|100+((i*113+t*31)%4096) as u32).collect()).collect();
                let inputs:Vec<_>=(0..chats).map(|i|9707+i as u32).collect();
                let mut canonical=(0..chats).map(|_|model.session(b,1024)).collect::<Result<Vec<_>,_>>()?;
                let mut greedy=(0..chats).map(|_|model.session(b,1024)).collect::<Result<Vec<_>,_>>()?;
                for (i,(c,g)) in canonical.iter_mut().zip(&mut greedy).enumerate() {
                    c.disable_mtp();g.disable_mtp();c.prefill_advance(b,&prompts[i])?;g.prefill_advance(b,&prompts[i])?;
                }
                let rows=NextSession::step_batch(b,&mut canonical.iter_mut().collect::<Vec<_>>(),&inputs)?;
                let ids=NextSession::step_batch_greedy(b,&mut greedy.iter_mut().collect::<Vec<_>>(),&inputs)?;
                if greedy.iter().any(|s|s.step_was_sparse()!=(chats!=3)) {return Err(format!("plain greedy head path at {chats} chats"));}
                if ids!=rows.iter().map(|r|greedy_token(r)).collect::<Result<Vec<_>,_>>()? {return Err("plain greedy IDs disagree".into());}
                let top=canonical.iter_mut().map(|s|s.decode_top_k(b,1,8)).collect::<Result<Vec<_>,_>>()?;
                // Same width: this step overwrites the workspace the greedy step used.
                NextSession::step_batch(b,&mut canonical.iter_mut().collect::<Vec<_>>(),&vec![198;chats])?;
                for (i,s) in greedy.iter_mut().enumerate() {
                    if bits(s.logits(b)?)!=bits(rows[i].clone()) {return Err("plain greedy full-logit fallback changed bits".into());}
                    if s.decode_top_k(b,1,8)?!=top[i] {return Err("plain greedy top-k fallback changed values".into());}
                }
                println!("CHECK bounded_plain chats={chats} IDs/full_logits/top_k/overwritten_workspace PASS");
            }
        }
        if check {
            let prompt: Vec<_> = (0..129).map(|i| 100+(i*31%4096) as u32).collect();
            let mut session = model.session(b, 1024)?; let mut reference = None;
            for chunk in [1, 2, 7, 16, 63, 64, 65, 127, 512] {
                session.reset(b)?; session.set_chunk_size(chunk)?;
                let first = bits(session.prefill(b, &prompt)?);
                session.step(b, 9707)?; let next = bits(session.logits(b)?);
                if let Some((a, c)) = &reference {
                    if &first != a || &next != c { return Err(format!("chunk invariance failed at {chunk}")); }
                } else { reference = Some((first.clone(), next)); }
                println!("CHECK chunk={chunk} rows=129 logits_fnv64={:016x} PASS", hash(&first));
            }
            drop(session);
            for width in 1..=16 {
                let prompts: Vec<Vec<u32>> = (0..width).map(|i| (0..(7+i)).map(|t| 100+(i*113+t*7) as u32).collect()).collect();
                let inputs: Vec<_> = (0..width).map(|i| 200+i as u32).collect();
                let mut sessions = (0..width).map(|_| model.session(b, 1024)).collect::<Result<Vec<_>, _>>()?;
                let mut expected = Vec::new();
                for (i, s) in sessions.iter_mut().enumerate() {
                    s.prefill_advance(b, &prompts[i])?; s.step(b, inputs[i])?; expected.push(bits(s.logits(b)?));
                    s.reset(b)?; s.prefill_advance(b, &prompts[i])?;
                }
                let mut refs: Vec<_> = sessions.iter_mut().collect();
                let got = NextSession::step_batch(b, &mut refs, &inputs)?;
                if got.into_iter().map(bits).collect::<Vec<_>>() != expected { return Err(format!("batch invariance failed at {width}")); }
                for (i, s) in sessions.iter_mut().enumerate() { s.reset(b)?; s.prefill_advance(b, &prompts[i])?; }
                let reversed: Vec<_> = inputs.iter().rev().copied().collect();
                let mut refs: Vec<_> = sessions.iter_mut().rev().collect();
                let got = NextSession::step_batch(b, &mut refs, &reversed)?;
                if got.into_iter().map(bits).collect::<Vec<_>>() != expected.into_iter().rev().collect::<Vec<_>>() {
                    return Err(format!("batch permutation failed at {width}"));
                }
                println!("CHECK chats={width} full_logits_and_permutation PASS");
            }
        }
        if measure {
            let prompt: Vec<_> = (0..prompt_len).map(|i| 100+(i*31%4096) as u32).collect();
            for chats in [1, 8, 16] {
                let started = Instant::now();
                let mut sessions = (0..chats).map(|_| model.session(b, capacity)).collect::<Result<Vec<_>, _>>()?;
                let mut next = Vec::new(); let mut ttft = 0.0;
                for (i, s) in sessions.iter_mut().enumerate() {
                    let row = s.prefill(b, &prompt)?; next.push(greedy_token(&row)?);
                    if i == 0 { ttft = started.elapsed().as_secs_f64(); }
                }
                b.synchronize()?;
                let started = Instant::now(); let mut checksum = 0u64;
                for _ in 0..decode {
                    let mut refs: Vec<_> = sessions.iter_mut().collect();
                    let rows = NextSession::step_batch(b, &mut refs, &next)?;
                    next = rows.iter().map(|r| greedy_token(r)).collect::<Result<Vec<_>, _>>()?;
                    checksum = checksum.rotate_left(7) ^ hash(&next);
                }
                let elapsed = started.elapsed().as_secs_f64();
                println!("MEASURE prompt_tokens={prompt_len} chunk=512 chats={chats} cold_ttft_s={ttft:.6} decode_tokens_per_chat={decode} decode_s={elapsed:.6} aggregate_tok_s={:.6} per_chat_tok_s={:.6} token_fnv64={checksum:016x} scope=native_engine_no_http", chats as f64*decode as f64/elapsed, decode as f64/elapsed);
            }
        }
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run() { eprintln!("FAIL {error}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("Flash-Next GPU receipt requires Linux/CUDA"); }
}
