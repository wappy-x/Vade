//! Native GPU rank service and qualification capture. All ranks execute the same
//! immutable case file: NAME PROMPT_CSV EXPECTED_CSV. MTP0, fixed token loop.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_tp::{
        next::{NextModel, RankTransport, SharedTransport},
        next_gpu::{Arena, Gpu},
    };
    use std::{
        ffi::CString,
        fs::{self, OpenOptions},
        io::Write,
        net::SocketAddr,
        path::Path,
        time::Instant,
    };
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    if a.len() != 9 {
        return Err("usage: next-tp RANK WORLD LISTEN NEXT PTX MODEL TENSOR_MAP CASES NEW_OUTPUT_DIR; NEXT_TP_SPLIT=ep|ep2tp2".into());
    }
    let args = spark_tp::next::RankArgs::parse(&a, false)?;
    let (rank, world, listen, next) = (args.rank, args.world, args.listen, args.next);
    let ep2 = spark_tp::next::ep2_enabled(world)?;
    let cohort = std::env::var("NEXT_TP_COHORT")
        .unwrap_or_else(|_| "1".into())
        .parse::<usize>()
        .map_err(|_| "cohort")?;
    if !matches!(cohort, 1 | 8) {
        return Err("NEXT_TP_COHORT must be 1 or 8".into());
    }
    let world_rows = matches!(world, 2 | 4) && cohort >= 2;
    let cpu_reference = match std::env::var("NEXT_TP_CPU_REFERENCE").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("NEXT_TP_CPU_REFERENCE must be 0 or 1".into()),
    };
    let hybrid = match std::env::var("NEXT_TP_SPLIT").as_deref() {
        Ok("ep2tp2") => true,
        Ok("ep") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("NEXT_TP_SPLIT must be ep or ep2tp2".into()),
    };
    spark_engine::flash::require_inference(engine_model::flash::WeightFormat::Nvfp4)?;
    let mut cp = engine_model::flash::shard::ShardCheckpoint::open_mode(
        Path::new(&a[5]),
        Path::new(&a[6]),
        rank,
        world,
        hybrid,
    )
    .map_err(|e| e.to_string())?;
    if ep2 {
        cp.plan = cp.plan.with_ep2().map_err(|e| e.to_string())?;
    }
    let model = NextModel::from_checkpoint(cp)?;
    let path = spark_engine::flash::path_topology(world)?;
    let cases = fs::read_to_string(&a[7]).map_err(|e| e.to_string())?;
    let ptx = fs::read(&a[4]).map_err(|e| e.to_string())?;
    for symbol in [
        "flash_next_routed",
        "flash_next_mxfp8_m1",
        "flash_next_route",
    ] {
        if !ptx.windows(symbol.len()).any(|b| b == symbol.as_bytes()) {
            return Err(format!("PTX lacks {symbol}"));
        }
    }
    let ptx = CString::new(ptx).map_err(|e| e.to_string())?;
    let mut commands = spark_tp::transport::Ring::connect_options(rank, world, listen, next, path)?;
    spark_tp::runtime::verify_artifacts(&mut commands, &ptx, Path::new(&a[5]))?;
    let mut agreement = vec![0u8; world * 16];
    let hash = cases.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    agreement[rank * 16..rank * 16 + 8].copy_from_slice(&hash.to_le_bytes());
    agreement[rank * 16 + 8..rank * 16 + 16].copy_from_slice(
        &((hybrid as u64)
            | ((ep2 as u64) << 1)
            | ((cpu_reference as u64) << 2)
            | ((world_rows as u64) << 3)
            | ((cohort as u64) << 4))
            .to_le_bytes(),
    );
    commands.all_gather(&mut agreement, 16)?;
    if agreement.chunks_exact(16).any(|v| v != &agreement[..16]) {
        return Err("rank case/split disagreement".into());
    }
    let addr = |mut a: SocketAddr| -> Result<SocketAddr, String> {
        a.set_port(a.port().checked_add(1).ok_or("data port overflow")?);
        Ok(a)
    };
    let run = std::env::var("TP_RUN_ID").map_err(|_| "fresh TP_RUN_ID required")?;
    let mut data = if path {
        spark_seqpar::Ring::connect_path(rank, world, addr(listen)?, addr(next)?, &run)?
    } else {
        spark_seqpar::Ring::connect(rank, world, addr(listen)?, addr(next)?, &run)?
    };
    data.enable_mailbox(1 << 20)?;
    data.seal_registrations()?;
    if world > 1 && !data.uses_rdma() {
        return Err("native GPU gate requires RDMA".into());
    }
    let memory = data.mailbox_memory();
    let mut transport = SharedTransport::new(data);
    // Measure warm dependent gather latency before weights, 8B/1/8-row packets.
    for row in [8, 10240, 81920] {
        let mut wire = vec![0u8; world * row];
        for _ in 0..8 {
            transport.bytes(&mut wire, row)?;
        }
        let mut ns = Vec::new();
        for _ in 0..64 {
            let now = Instant::now();
            transport.bytes(&mut wire, row)?;
            ns.push(now.elapsed().as_nanos());
        }
        ns.sort();
        println!(
            "alpha rank={rank} row_bytes={row} median_ns={} min_ns={} max_ns={} alpha_per_hop_ns={}",
            ns[32],
            ns[0],
            ns[63],
            ns[32] / world.saturating_sub(1).max(1) as u128
        );
    }
    let capacity = 4096;
    let budget = std::env::var("SEQPAR_MEMORY_BYTES")
        .unwrap_or_else(|_| "107374182400".into())
        .parse()
        .map_err(|_| "memory budget")?;
    fs::create_dir(&a[8]).map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend(0, &ptx, budget, |b| {
        let _registration = memory
            .map(|(base, bytes)| unsafe { b.register_host(base, bytes) })
            .transpose()?;
        let arena = Arena::load(b, &model.checkpoint)?;
        arena.smoke(b, &model.checkpoint)?;
        let (kv, state) = model
            .checkpoint
            .plan
            .cache_bytes(capacity as u64)
            .map_err(|e| e.to_string())?;
        println!(
            "arena rank={rank} world={world} expert_tp={} bytes={} kv_bytes={kv} recurrent_bytes={state}",
            model.checkpoint.plan.expert_tp, arena.bytes
        );
        for (index, line) in cases.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err("case needs name prompt expected".into());
            }
            let ids = |s: &str| {
                s.split(',')
                    .map(|v| v.parse::<u32>().map_err(|_| "bad case ID".to_string()))
                    .collect::<Result<Vec<_>, _>>()
            };
            let (prompt, expected) = (ids(fields[1])?, ids(fields[2])?);
            if prompt.is_empty()
                || expected.is_empty()
                || prompt.len() + expected.len() > capacity
                || !fields[0]
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            {
                return Err("invalid gate case".into());
            }
            let name = fields[0];
            let mut sessions = Vec::new();
            for c in 0..cohort {
                let stem = if cohort == 1 {
                    name.to_owned()
                } else {
                    format!("{name}-c{c}")
                };
                let create = |suffix: &str| {
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(Path::new(&a[8]).join(format!("{stem}.{suffix}")))
                        .map_err(|e| e.to_string())
                };
                let mut exchange = transport.clone();
                let mut gpu = Gpu::new(
                    b,
                    &arena,
                    &model.checkpoint,
                    capacity,
                    Box::new(move |v, row| exchange.bytes(v, row)),
                )?;
                gpu.capture(create("f32")?, create("routes")?);
                sessions.push(model.session(
                    transport.clone(),
                    gpu,
                    (index * cohort + c + 1) as u64,
                    capacity,
                )?);
            }
            let mut next = vec![0; cohort];
            let now = Instant::now();
            for t in prompt {
                for (s, n) in sessions.iter_mut().zip(&mut next) {
                    *n = s.step(t)?;
                }
            }
            let mut generated = next.iter().map(|&t| vec![t]).collect::<Vec<_>>();
            while generated[0].len() < expected.len() {
                next = if world_rows {
                    spark_tp::next::NextSession::step_rows(
                        &mut sessions.iter_mut().collect::<Vec<_>>(),
                        &next,
                    )?
                } else {
                    sessions
                        .iter_mut()
                        .zip(&next)
                        .map(|(s, &t)| s.step(t))
                        .collect::<Result<Vec<_>, _>>()?
                };
                for (ids, &t) in generated.iter_mut().zip(&next) {
                    ids.push(t);
                }
            }
            for (c, (session, ids)) in sessions.iter().zip(generated).enumerate() {
                let stem = if cohort == 1 {
                    name.to_owned()
                } else {
                    format!("{name}-c{c}")
                };
                let mut receipt = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(Path::new(&a[8]).join(format!("{stem}.receipt")))
                    .map_err(|e| e.to_string())?;
                writeln!(receipt,"rank={rank} world={world} world_rows={world_rows} ids={ids:?} steps={} elapsed_ns={} histogram={:?}",session.position(),now.elapsed().as_nanos(),session.kernels().histogram).map_err(|e|e.to_string())?;
                let matches = ids.iter().zip(&expected).filter(|(a, b)| a == b).count();
                writeln!(
                    receipt,
                    "reference={} agreement={matches}/{}",
                    if cpu_reference {
                        "one-spark-cpu"
                    } else {
                        "same-setup"
                    },
                    expected.len()
                )
                .map_err(|e| e.to_string())?;
                if !cpu_reference && ids != expected {
                    return Err(format!(
                        "{stem}: greedy IDs differ from same-setup goldens; full trace retained"
                    ));
                }
                println!(
                    "{} case={stem} rank={rank} tokens={} agreement={matches}/{}",
                    if cpu_reference { "COMPARE" } else { "PASS" },
                    ids.len(),
                    expected.len()
                );
            }
        }
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("next-tp requires Linux CUDA");
        std::process::exit(1);
    }
}
