#![cfg(target_os = "linux")]
//! Full-model GB10 gate for the FP4-Local GPU decode path (flash::next),
//! including the VADEDBG1 debug-diff against CPU goldens. Env:
//! VADE_FLASHNEXT_GPU=1 FLASH_NVFP4=1 VADE_FLASHNEXT_MODEL_DIR=<ckpt>
//! Optional: VADE_FLASHNEXT_PROMPT="1,2,3" VADE_FLASHNEXT_NEW=8
//! VADE_FLASHNEXT_CAPACITY=4096 VADE_FLASHNEXT_BUDGET_GIB=110
//! VADE_FLASHNEXT_DUMP=<path> VADE_FLASHNEXT_CPU_DUMP=<path> VADE_FLASHNEXT_TOL=0.05
use spark_engine::flash::next::NextModel;
use spark_gpu_probe::driver::with_backend;
use std::path::PathBuf;

struct Record { stage: String, len: usize, checksum: u64, first: Vec<f32> }
#[derive(Default)]
struct Dump { records: Vec<Record>, moe: Vec<(usize, Vec<usize>)>, tops: Vec<u32> }
fn bits(text: &str) -> Result<Vec<f32>, String> {
    if text.is_empty() { return Ok(Vec::new()); }
    text.split(',').map(|h| u32::from_str_radix(h, 16).map(f32::from_bits).map_err(|_| "bad bits".to_string())).collect()
}
fn parse_dump(text: &str) -> Result<Dump, String> {
    let (mut dump, mut header) = (Dump::default(), false);
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("VADEDBG1") => header = true,
            Some("token") => {}
            Some("rec") => {
                let (Some(stage), Some(len), Some(sum)) = (parts.next(), parts.next(), parts.next())
                    else { return Err("short rec line".into()) };
                dump.records.push(Record { stage: stage.into(),
                    len: len.parse().map_err(|_| "bad length".to_string())?,
                    checksum: u64::from_str_radix(sum, 16).map_err(|_| "bad checksum".to_string())?,
                    first: bits(parts.next().unwrap_or(""))? });
            }
            Some("moe") => {
                let (Some(layer), Some(ids)) = (parts.next(), parts.next())
                    else { return Err("short moe line".into()) };
                dump.moe.push((layer.parse().map_err(|_| "bad layer".to_string())?,
                    ids.split(',').filter(|s| !s.is_empty())
                        .map(|s| s.parse().map_err(|_| "bad expert id".to_string()))
                        .collect::<Result<_, _>>()?));
            }
            Some("top") => dump.tops.push(parts.next().ok_or("short top line")?
                .parse().map_err(|_| "bad argmax".to_string())?),
            _ => return Err("unknown dump line".into()),
        }
    }
    if header { Ok(dump) } else { Err("missing VADEDBG1 header".into()) }
}
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
fn flashnext_full_model_greedy_decode() -> Result<(), String> {
    if std::env::var_os("VADE_FLASHNEXT_GPU").is_none() {
        return Err("explicit GPU gate requires VADE_FLASHNEXT_GPU".into());
    }
    let root = PathBuf::from(std::env::var_os("VADE_FLASHNEXT_MODEL_DIR")
        .ok_or("VADE_FLASHNEXT_MODEL_DIR must name the FP4-Local checkpoint")?);
    let prompt: Vec<u32> = std::env::var("VADE_FLASHNEXT_PROMPT").unwrap_or_else(|_| "9707,25,264,3467".into())
        .split(',').map(|s| s.trim().parse().map_err(|_| "bad prompt token id".to_string()))
        .collect::<Result<_, _>>()?;
    let (new_tokens, capacity) = (env_usize("VADE_FLASHNEXT_NEW", 8), env_usize("VADE_FLASHNEXT_CAPACITY", 4096));
    let budget = env_usize("VADE_FLASHNEXT_BUDGET_GIB", 110) << 30;
    with_backend(0, spark_gpu_probe::layer_ptx()?, budget, |b| {
        let started = std::time::Instant::now();
        let model = NextModel::load(b, &root)?;
        let plan = model.memory_plan(capacity)?;
        println!("flashnext load_s={:.1} resident_weights={} expanded_bf16={} expanded_scales={} native_mxfp8={} kv={} recurrent={} plan_total={} allocated={}",
            started.elapsed().as_secs_f64(), plan.resident_weights, plan.expanded_bf16,
            plan.expanded_scales, plan.native_mxfp8, plan.kv, plan.recurrent, plan.total, b.allocated_bytes());
        let mut session = model.session(b, capacity)?;
        match std::env::var("VADE_FLASHNEXT_INDEXER").as_deref() {
            Ok("sparse") => session.set_indexer(Box::new(spark_engine::flash::next::SparseIndexer)),
            Ok("dense") | Err(std::env::VarError::NotPresent) => (),
            _ => return Err("VADE_FLASHNEXT_INDEXER must be sparse or dense".into()),
        }
        println!("flashnext allocated_after_session={}", b.allocated_bytes());
        if std::env::var_os("VADE_FLASHNEXT_BENCH").is_some() {
            if new_tokens == 0 { return Err("benchmark requires positive NEW".into()); }
            let mut token = 0; for &t in &prompt { token = session.step(b, t)?; }
            let started = std::time::Instant::now(); let mut ids = Vec::new();
            for _ in 0..new_tokens { token = session.step(b, token)?; ids.push(token); }
            b.synchronize()?;
            println!("flashnext bench switches={:?} tokens={} ms_per_token={:.6} ids={ids:?}",
                ["PROJ","FRONTEND"].map(|p| std::env::var(format!("VADE_FLASHNEXT_FAST_{p}")).unwrap_or_else(|_| "0".into())), new_tokens,
                started.elapsed().as_secs_f64()*1000.0/new_tokens as f64);
            return Ok(());
        }
        session.enable_dump();
        let started = std::time::Instant::now();
        let generated = match session.generate(b, &prompt, new_tokens) {
            Ok(ids) => ids,
            Err(error) => {
                if let (Some(path), Some(trace)) = (std::env::var_os("VADE_FLASHNEXT_DUMP"), session.take_dump()) {
                    std::fs::write(path, trace).map_err(|e| format!("{error}; partial dump: {e}"))?;
                }
                return Err(error);
            }
        };
        println!("flashnext prompt={prompt:?} generated={generated:?} steps={} s={:.2}",
            session.position(), started.elapsed().as_secs_f64());
        assert!(generated.iter().all(|&t| t < 248320));
        let dump_text = session.take_dump().ok_or("dump missing")?;
        let parsed = parse_dump(&dump_text)?;
        assert_eq!(parsed.tops.len(), session.position());
        assert_eq!(*parsed.tops.last().unwrap(), *generated.last().unwrap());
        if let Some(path) = std::env::var_os("VADE_FLASHNEXT_DUMP") {
            std::fs::write(&path, &dump_text).map_err(|e| e.to_string())?;
            println!("flashnext dump_written={}", PathBuf::from(path).display());
        }
        let logits = session.logits(b)?;
        if let Ok(expected) = std::env::var("VADE_FLASHNEXT_EXPECT") {
            let ids: Vec<u32> = expected.split(',').map(|v| v.trim().parse().map_err(|_| "bad golden token id".to_string())).collect::<Result<_, _>>()?;
            assert_eq!(generated, ids, "greedy CPU golden differs");
        }
        if let Some(path) = std::env::var_os("VADE_FLASHNEXT_LOGITS") {
            let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        }
        session.reset(b)?;
        assert_eq!(session.generate(b, &prompt, new_tokens)?, generated, "dump-free run diverged");
        assert!(logits.iter().zip(session.logits(b)?).all(|(a,b)| a.to_bits()==b.to_bits()), "dump changes full logit bytes");
        session.reset(b)?;
        session.enable_dump();
        assert_eq!(session.generate(b, &prompt, new_tokens)?, generated, "reset replay diverged");
        assert_eq!(session.take_dump().as_deref(), Some(dump_text.as_str()), "reset replay dump differs");
        println!("PASS flashnext determinism reset-replay");
        session.reset(b)?; session.enable_dump(); let mut stepped = Vec::new();
        for (i, &t) in prompt.iter().chain(generated.iter().take(new_tokens-1)).enumerate() {
            let next = session.step(b, t)?; if i+1 >= prompt.len() { stepped.push(next); }
        }
        assert_eq!(stepped, generated, "device feed differs from host-fed steps");
        assert_eq!(session.take_dump().as_deref(), Some(dump_text.as_str()));
        let forced: Vec<_> = prompt.iter().chain(generated.iter().take(generated.len()-1)).copied().collect();
        let mut expected=Vec::new(); session.reset(b)?;
        for &token in &forced { session.step(b,token)?; expected.push(session.logits(b)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>()); }
        for chunk in [1,4,16] { session.reset(b)?; let mut at=0;
            for part in forced.chunks(chunk) { for &token in part { session.step(b,token)?; assert_eq!(session.logits(b)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected[at],"full logits at {at}, chunk {chunk}"); at+=1; } }
        }
        if let Some(path)=std::env::var_os("VADE_FLASHNEXT_LOGITS") { std::fs::write(path,expected.iter().flatten().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string())?; }
        session.reset(b)?; let mut next=0; for &t in &prompt {next=session.step(b,t)?;} b.synchronize()?; let started=std::time::Instant::now();
        for _ in 0..new_tokens {next=session.step(b,next)?;} b.synchronize()?;
        println!("PASS flashnext full-logit bytes repeat/chunks=1,4,16; timed_decode_steps={new_tokens} ms_per_token={:.3}",started.elapsed().as_secs_f64()*1000.0/new_tokens as f64);
        session.reset(b)?; session.prepare(b)?;
        let (&last, prefix) = prompt.split_last().unwrap();
        for &token in prefix { session.step_prompt(b, token)?; }
        let logits = session.step_logits(b, last)?;
        assert_eq!(logits[generated[0] as usize], logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let mut replay = vec![generated[0]];
        while replay.len() < generated.len() { replay.push(session.step(b, *replay.last().unwrap())?); }
        assert_eq!(replay, generated, "prompt-only state differs");
        session.reset(b)?;
        for &token in prefix { session.step_prompt(b, token)?; }
        let top = session.step_top_k(b, last, 20)?;
        assert_eq!(top[0].0, generated[0]);
        assert!(top.iter().all(|&(id, value)| value.to_bits() == logits[id as usize].to_bits()));
        if let Some(cpu_path) = std::env::var_os("VADE_FLASHNEXT_CPU_DUMP") {
            let cpu = parse_dump(&std::fs::read_to_string(&cpu_path).map_err(|e| e.to_string())?)?;
            assert_eq!(cpu.records.len(), parsed.records.len(), "dump stage counts differ");
            let tol: f32 = std::env::var("VADE_FLASHNEXT_TOL").ok().map(|v| v.parse().map_err(|_| "bad VADE_FLASHNEXT_TOL".to_string())).transpose()?.unwrap_or(1.0/128.0);
            assert!(tol.is_finite() && tol > 0.0);
            let mut worst = ("", 0.0f32);
            for (a, g) in cpu.records.iter().zip(&parsed.records) {
                assert_eq!((&a.stage, a.len, a.first.len()), (&g.stage, g.len, g.first.len()), "dump stages misaligned");
                let max_abs = a.first.iter().zip(&g.first).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
                println!("diff {} checksum_equal={} max_abs_first16={max_abs:.6}", a.stage, a.checksum == g.checksum);
                let scale = a.first.iter().map(|x| x.abs()).fold(0.0, f32::max);
                let normwise = (max_abs - 1e-5).max(0.0) / scale.max(f32::MIN_POSITIVE);
                if normwise > worst.1 { worst = (&a.stage, normwise); }
            }
            let flips = cpu.moe.iter().zip(&parsed.moe).filter(|(a, g)| a.1 != g.1).count();
            println!("diff moe_route_flips={flips}/{} worst={} {:.6}", cpu.moe.len().min(parsed.moe.len()), worst.0, worst.1);
            assert!(worst.1 <= tol, "worst stage {} normwise {} exceeds tol {tol}", worst.0, worst.1);
            println!("PASS flashnext cpu-diff normwise_first16={:.6} tol={tol}", worst.1);
        }
        println!("PASS flashnext_full_model_greedy_decode");
        Ok(())
    })
}

#[test]
#[ignore = "skipped: no GPU; requires CUDA; no checkpoint"]
fn control_prefix_upload_preserves_device_predictions() -> Result<(), String> {
    with_backend(0, spark_gpu_probe::layer_ptx()?, 64 << 20, |b| {
        let mut ids = b.upload_vector(&[7.0; 36])?;
        assert!(b.write_vector(&mut ids, &[0.0; 3]).is_err());
        assert!(b.write_vector_bytes(&mut ids, &[0; 145]).is_err());
        b.write_vector_bytes(&mut ids, &[0; 12])?;
        let values = b.read_vector(&ids)?;
        assert_eq!(&values[..3], &[0.0; 3]);
        assert_eq!(&values[3..], &[7.0; 33]);
        Ok(())
    })
}
