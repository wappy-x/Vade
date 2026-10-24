//! Fresh-process, real-checkpoint prefill receipt. Raw layer bytes stay on the GPU host.
#[cfg(target_os="linux")]
fn run() -> Result<(),String> {
    use spark_engine::{flash::next_batched::{NextModel,NextSession},greedy_token};
    use spark_gpu_probe::driver::with_backend;
    use std::{path::PathBuf,time::Instant,io::Write};
    let args: Vec<_>=std::env::args().collect();
    if args.iter().any(|a| a=="--help") {
        println!("flashnext_prefill --model DIR --out NEW_DIR [--prefill 1024 --chats 1 --chunk 512 --decode 16] [--layers] [--solo] [--tokens IDS] [--force IDS]"); return Ok(());
    }
    let value=|key: &str| args.iter().position(|a| a==key).and_then(|i| args.get(i+1)).cloned();
    let number=|key: &str,n| value(key).map(|v| v.parse::<usize>().map_err(|_| format!("invalid {key}"))).unwrap_or(Ok(n));
    let (n,chats,chunk,decode)=(number("--prefill",1024)?,number("--chats",1)?,number("--chunk",512)?,number("--decode",16)?);
    if n==0 || chats==0 || chats>16 || n>131072 || decode==0 { return Err("invalid shape".into()); }
    let root=PathBuf::from(value("--model").ok_or("--model required")?);
    let out=PathBuf::from(value("--out").ok_or("--out required")?); std::fs::create_dir(&out).map_err(|e| e.to_string())?;
    let custom=value("--tokens").map(|v| v.split(',').map(|t| t.parse::<u32>().map_err(|e| e.to_string())).collect::<Result<Vec<_>,_>>()).transpose()?;
    let prompts: Vec<Vec<u32>>=(0..chats).map(|i| custom.clone().unwrap_or_else(|| (0..n).map(|t| 100+((t*31+i*113)%4096) as u32).collect())).collect();
    // --force: teacher forcing; decode step k feeds force[k] instead of the greedy token (logits still recorded).
    let force=value("--force").map(|v| v.split(',').map(|t| t.parse::<u32>().map_err(|e| e.to_string())).collect::<Result<Vec<_>,_>>()).transpose()?;
    if force.as_ref().is_some_and(|f| f.len()<decode) { return Err("--force needs at least --decode ids".into()); }
    let oracle=args.iter().any(|a| a=="--oracle");
    let reverse=args.iter().any(|a| a=="--reverse");
    let layers=oracle || args.iter().any(|a| a=="--layers"); let solo=args.iter().any(|a| a=="--solo");
    with_backend(0,spark_gpu_probe::layer_ptx()?,number("--budget-gib",110)?<<30,|b| {
        let load=Instant::now(); let model=NextModel::load(b,&root)?;
        println!("LOAD s={:.6} fast_prefill={} layers={layers}",load.elapsed().as_secs_f64(),std::env::var("VADE_FLASHNEXT_FAST_PREFILL").unwrap_or_default());
        let start=Instant::now();
        let mut sessions=(0..chats).map(|_| model.session(b,n.max(prompts[0].len())+decode+1)).collect::<Result<Vec<_>,_>>()?;
        for (i,s) in sessions.iter_mut().enumerate() { s.set_chunk_size(chunk)?; if oracle { s.enable_dump(); } if layers { s.check_layers(out.join(format!("chat{i}")))?; } }
        let mut input: Vec<_>=prompts.iter().map(Vec::as_slice).collect();
        if reverse { sessions.reverse(); input.reverse(); }
        b.synchronize()?;
        let rows=if solo { sessions.iter_mut().zip(&input).map(|(s,p)| s.prefill(b,p)).collect::<Result<Vec<_>,_>>()? }
            else { NextSession::prefill_batch(b,&mut sessions.iter_mut().collect::<Vec<_>>(),&input)? };
        b.synchronize()?; let prefill=start.elapsed().as_secs_f64();
        let mut next=rows.iter().map(|r| greedy_token(r)).collect::<Result<Vec<_>,_>>()?;
        if let Some(f)=&force { next=vec![f[0];chats]; }
        let mut ids=vec![Vec::new();chats];
        let mut logits=(0..chats).map(|i| std::fs::File::create(out.join(format!("chat{i}-logits.f32"))).map_err(|e| e.to_string())).collect::<Result<Vec<_>,_>>()?;
        let mut record=|rows: &[Vec<f32>]| -> Result<(),String> {
            for i in 0..chats { let row=&rows[if reverse {chats-1-i} else {i}]; if row.iter().any(|v| !v.is_finite()) { return Err("nonfinite logits".into()); }
                logits[i].write_all(&row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
            } Ok(())
        };
        record(&rows)?;
        // Stop the prompt-layer observer before decode; layer files cover exactly n prompt rows.
        if !oracle { for s in &mut sessions { s.stop_layer_check(); } }
        for step in 0..decode {
            for (i,&id) in next.iter().enumerate() { ids[if reverse {chats-1-i} else {i}].push(id); }
            if step+1<decode { let rows=NextSession::step_batch(b,&mut sessions.iter_mut().collect::<Vec<_>>(),&next)?;
                next=rows.iter().map(|r| greedy_token(r)).collect::<Result<Vec<_>,_>>()?; record(&rows)?;
                if let Some(f)=&force { next=vec![f[step+1];chats]; } }
        }
        if oracle { std::fs::write(out.join("routes.dump"),sessions[0].take_dump().ok_or("missing routes")?).map_err(|e| e.to_string())?; }
        std::fs::write(out.join("tokens.txt"),format!("{ids:?}\n")).map_err(|e| e.to_string())?;
        std::fs::write(out.join("timing.txt"),format!("{prefill:.9}\n")).map_err(|e| e.to_string())?;
        println!("MEASURE prompt_tokens={} chats={chats} chunk={chunk} prefill_s={prefill:.9} scope=native_cold_session_model_load_excluded",prompts[0].len());
        Ok(())
    })
}
fn main() {
    #[cfg(target_os="linux")]
    if let Err(e)=run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os="linux"))]
    { eprintln!("requires Linux/CUDA"); }
}
