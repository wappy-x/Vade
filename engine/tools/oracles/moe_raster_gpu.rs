//! Fresh-session microbench/byte helper. TSV: tag, chunk, decode, chats, profile, IDs.
//! Compile against each arm's own rlibs/PTX; drive it with an external measurement runner.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_engine::{
        flash::next_batched::{NextModel, NextSession},
        greedy_token,
    };
    use spark_gpu_probe::driver::with_backend;
    use std::{io::Write, path::PathBuf, time::Instant};
    #[link(name = "cuda")]
    unsafe extern "C" {
        fn cuProfilerStart() -> i32;
        fn cuProfilerStop() -> i32;
    }
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--help") {
        println!("moe_raster_gpu MODEL PLAN_TSV NEW_OUT");
        return Ok(());
    }
    if args.len() != 4 {
        return Err("MODEL PLAN_TSV NEW_OUT required".into());
    }
    let root = PathBuf::from(&args[1]);
    let plan = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir(&out).map_err(|e| e.to_string())?;
    with_backend(0, spark_gpu_probe::layer_ptx()?, 110usize << 30, |b| {
        let t = Instant::now();
        let model = NextModel::load(b, &root)?;
        println!("LOAD s={:.6}", t.elapsed().as_secs_f64());
        println!(
            "SAMPLING temperature=0 top_p=1 top_k=1 repetition_penalty=1 presence_penalty=0 frequency_penalty=0 mtp=0 native_greedy=true"
        );
        if model.vocabulary() != 248320 {
            return Err("unexpected vocabulary".into());
        }
        let mut capturing = false;
        for line in plan.lines() {
            let f: Vec<_> = line.split('\t').collect();
            if f.len() != 6 {
                return Err("bad plan".into());
            }
            let (tag, chunk, decode, chats, profile) = (
                f[0],
                f[1].parse::<usize>().map_err(|e| e.to_string())?,
                f[2].parse::<usize>().map_err(|e| e.to_string())?,
                f[3].parse::<usize>().map_err(|e| e.to_string())?,
                f[4] == "1",
            );
            let tokens = f[5]
                .split(',')
                .map(|v| v.parse::<u32>().map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            if tag.is_empty()
                || !tag
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
                || ![512, 2048].contains(&chunk)
                || ![1, 64].contains(&decode)
                || ![1, 8].contains(&chats)
                || !["0", "1"].contains(&f[4])
                || tokens.is_empty()
                || tokens.len() > 8192
                || tokens.iter().any(|&t| t as usize >= model.vocabulary())
            {
                return Err("invalid cell shape, tag or token IDs".into());
            }
            if profile && !capturing {
                let rc = unsafe { cuProfilerStart() };
                if rc != 0 {
                    return Err(format!("profiler start {rc}"));
                }
                capturing = true;
            }
            if !profile && capturing {
                let rc = unsafe { cuProfilerStop() };
                if rc != 0 {
                    return Err(format!("profiler stop {rc}"));
                }
                capturing = false;
            }
            let dest = out.join(tag);
            std::fs::create_dir(&dest).map_err(|e| e.to_string())?;
            let start = Instant::now();
            let mut sessions = (0..chats)
                .map(|_| model.session_with_chunk(b, tokens.len() + decode + 1, chunk))
                .collect::<Result<Vec<_>, _>>()?;
            for session in &mut sessions {
                session.set_chunk_size(chunk)?;
                if session.chunk_size() != chunk {
                    return Err("effective chunk differs from plan".into());
                }
            }
            let inputs = vec![tokens.as_slice(); chats];
            b.synchronize()?;
            let rows = if chats == 1 {
                vec![sessions[0].prefill(b, &tokens)?]
            } else {
                NextSession::prefill_batch(
                    b,
                    &mut sessions.iter_mut().collect::<Vec<_>>(),
                    &inputs,
                )?
            };
            b.synchronize()?;
            let prefill = start.elapsed().as_secs_f64();
            let mut next = rows
                .iter()
                .map(|r| greedy_token(r))
                .collect::<Result<Vec<_>, _>>()?;
            let mut ids = vec![Vec::new(); chats];
            let mut logits = (0..chats)
                .map(|i| {
                    std::fs::File::create(dest.join(format!("chat{i}-logits.f32")))
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut record = |rows: &[Vec<f32>]| -> Result<(), String> {
                for i in 0..chats {
                    if rows[i].iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite logits".into());
                    }
                    logits[i]
                        .write_all(
                            &rows[i]
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            record(&rows)?;
            for step in 0..decode {
                for (i, &id) in next.iter().enumerate() {
                    ids[i].push(id)
                }
                if step + 1 < decode {
                    let rows = NextSession::step_batch(
                        b,
                        &mut sessions.iter_mut().collect::<Vec<_>>(),
                        &next,
                    )?;
                    next = rows
                        .iter()
                        .map(|r| greedy_token(r))
                        .collect::<Result<Vec<_>, _>>()?;
                    record(&rows)?;
                }
            }
            std::fs::write(dest.join("tokens.txt"), format!("{ids:?}\n"))
                .map_err(|e| e.to_string())?;
            std::fs::write(dest.join("timing.txt"), format!("{prefill:.9}\n"))
                .map_err(|e| e.to_string())?;
            println!("CELL {tag} chunk={chunk} chats={chats} prefill_s={prefill:.9}");
            std::io::stdout().flush().map_err(|e| e.to_string())?;
        }
        if capturing {
            let rc = unsafe { cuProfilerStop() };
            if rc != 0 {
                return Err(format!("profiler stop {rc}"));
            }
        }
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1)
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("requires Linux/CUDA");
        std::process::exit(1);
    }
}
