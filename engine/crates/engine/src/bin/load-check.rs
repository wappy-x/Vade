//! Real-engine startup and same-context unload/reload check.
#[cfg(any(target_os = "linux", test))]
mod bulk {
    use spark_gpu_probe::kernel_params::flash_bf16_element;
    // A chunk contains complete permutation tiles and, for column slices,
    // complete source rows. Both host buffers remain bounded by capacity.
    pub fn chunk(remaining: usize, width: usize, stride: usize, tile: usize, capacity: usize)
        -> Result<(usize, usize), String> {
        if tile == 0 || width != 0 && (stride < width || tile % width != 0) {
            return Err("invalid comparison chunk layout".into());
        }
        let limit = if width == 0 { capacity } else { capacity / stride * width };
        let count = remaining.min(limit) / tile * tile;
        if count == 0 { return Err("comparison row exceeds bounded staging".into()); }
        let read = if width == 0 { count } else { (count / width - 1) * stride + width };
        Ok((count, read))
    }
    pub fn compare(name: &str, local: usize, source: &[u8], device: &[u8], cols: usize, rows: usize)
        -> Result<(), String> {
        if source.len() != device.len() { return Err("comparison chunk lengths differ".into()); }
        if cols != 0 && (local % (cols * 2) != 0 || source.len() % (cols * 2) != 0) {
            return Err("packed comparison requires complete rows".into());
        }
        let mismatch = |offset| Err(format!("raw tensor mismatch {name} local={}", local + offset));
        if cols == 0 {
            if let Some(i) = source.iter().zip(device).position(|(a, b)| a != b) { return mismatch(i); }
        } else {
            // Adjacent BF16 pairs stay contiguous in both packed layouts.
            // Visit logical coordinates in order to report the first bad byte.
            for row in 0..source.len() / (cols * 2) {
                for col in (0..cols).step_by(2) {
                    let logical = row * cols * 2 + col * 2;
                    let physical = flash_bf16_element(local / (cols * 2) + row, col, cols, rows) * 2 - local;
                    let (a, b) = (&source[logical..logical + 4], &device[physical..physical + 4]);
                    if a != b { return mismatch(logical + a.iter().zip(b).position(|(a, b)| a != b).unwrap()); }
                }
            }
        }
        Ok(())
    }
    #[test] fn chunks_and_tails() { assert_eq!(chunk(25, 0, 0, 8, 17), Ok((16, 16))); assert_eq!(chunk(8, 0, 0, 8, 17), Ok((8, 8))); assert_eq!(chunk(24, 4, 12, 8, 48), Ok((16, 40))); assert_eq!(chunk(8, 4, 12, 8, 48), Ok((8, 16))); assert!(chunk(8, 4, 12, 8, 12).is_err()); }
    #[test] fn packed_chunks_and_first_mismatch() {
        for (rows, cols, tile_rows) in [(12, 16, 4), (128, 16, 64), (128, 64, 64)] { let source = (0..rows * cols * 2).map(|i| (i % 251) as u8).collect::<Vec<_>>(); let mut device = vec![0; source.len()]; for i in 0..source.len() { let e = i / 2; device[flash_bf16_element(e / cols, e % cols, cols, rows) * 2 + i % 2] = source[i]; }
            let size = tile_rows * cols * 2; for at in (0..source.len()).step_by(size) { assert_eq!(compare("first", at, &source[at..at + size], &device[at..at + size], cols, rows), Ok(())); } let at = size; let bad = at + 3; let e = bad / 2; device[flash_bf16_element(e / cols, e % cols, cols, rows) * 2 + bad % 2] ^= 1; assert_eq!(compare("first", at, &source[at..at + size], &device[at..at + size], cols, rows), Err(format!("raw tensor mismatch first local={bad}"))); }
    }
    #[test] fn raw_first_tensor_and_offset() { assert_eq!([("first", vec![0, 2, 1]), ("second", vec![2, 1, 1])].iter().try_for_each(|(name, device)| compare(name, 8, &[0, 1, 1], device, 0, 0)), Err("raw tensor mismatch first local=9".into())); }
}
#[cfg(target_os = "linux")]
type Result<T, E = String> = std::result::Result<T, E>;
// Shared sequence-parallel sources are included whole; this binary uses a subset.
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/partition.rs"] mod partition;
#[cfg(target_os = "linux")] #[allow(dead_code)]
use engine_fabric::discovery;
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/control.rs"] mod control;
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/transport.rs"] mod transport;
#[cfg(target_os = "linux")] #[allow(dead_code)]
use engine_fabric::verbs;
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use std::{ffi::CString, io::{BufRead, Write}, path::Path, process::{Command, Stdio}, time::Instant};
    use spark_gpu_probe::driver::with_backend;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|a| a == "--flash-session") { return flash_session(&args[1..]); }
    if args.first().is_some_and(|a| matches!(a.as_str(), "--flash-front" | "--flash-qsa" | "--flash-moe")) { return flash_front(&args[1..], &args[0]); }
    if args.first().is_some_and(|a| a == "--flash") { return flash(&args[1..]); }
    if args.len() != 5 { return Err("usage: load-check PTX MODEL CAPACITY CYCLES NEW_TOKENS".into()); }
    if std::env::var_os("SPARK_LOAD_CHILD").is_none() {
        let start = Instant::now();
        let mut child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args(&args).env("SPARK_LOAD_CHILD", "1").stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        for line in std::io::BufReader::new(child.stdout.take().unwrap()).lines() {
            let line = line.map_err(|e| e.to_string())?;
            if line.starts_with("ready cycle=0 ") { println!("process_start_to_first_token_ms={:.3}", start.elapsed().as_secs_f64() * 1000.0); }
            println!("{line}");
        }
        if !child.wait().map_err(|e| e.to_string())?.success() { return Err("engine load child failed".into()); }
        return Ok(());
    }
    let capacity = args[2].parse::<usize>().map_err(|e| e.to_string())?;
    let cycles = args[3].parse::<usize>().map_err(|e| e.to_string())?;
    let count = args[4].parse::<usize>().map_err(|e| e.to_string())?;
    if cycles == 0 || count == 0 || count.checked_add(4).is_none_or(|n| n > capacity) { return Err("invalid cycle/token budget".into()); }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fn memory() -> String {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let bytes = [(&status, "VmRSS:"), (&status, "VmHWM:"), (&info, "MemAvailable:")].iter().map(|(text, key)| {
            let bytes = text.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok()).unwrap_or(0) * 1024;
            format!("{}={bytes}", key.trim_end_matches(':'))
        }).collect::<Vec<_>>().join(" ");
        let threads = status.lines().find(|l| l.starts_with("Threads:")).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("?");
        let fds = std::fs::read_dir("/proc/self/fd").map(|d| d.count()).unwrap_or(0);
        format!("{bytes} threads={threads} fds={fds}")
    }
    with_backend(0, &ptx, 100 * 1024 * 1024 * 1024, |backend| {
        let baseline = backend.allocated_bytes();
        println!("baseline gpu_bytes={baseline} {}", memory());
        let mut reference = Vec::new();
        for cycle in 0..cycles {
            let before = backend.stats();
            let load = Instant::now();
            let model = spark_engine::qwen::Model::load(backend, Path::new(&args[1]), capacity)?;
            let loaded = load.elapsed();
            let model_bytes = backend.allocated_bytes() - baseline;
            let stats = backend.stats().since(before);
            let mut session = model.session();
            let prefill = Instant::now();
            let logits = session.prefill(&[1, 42, 1024, 17])?;
            let mut token = spark_engine::greedy_token(&logits)?;
            let mut score = logits[token as usize];
            println!("ready cycle={cycle} token={token} load_ms={:.3} prefill_ms={:.3} loader_ms={:.3} allocation_wait_ms={:.3}",
                loaded.as_secs_f64() * 1000.0, prefill.elapsed().as_secs_f64() * 1000.0,
                stats.weight_read_upload_ns as f64 / 1e6, stats.allocation_ns as f64 / 1e6);
            std::io::stdout().flush().map_err(|e| e.to_string())?;
            let kv = session.cache_bytes();
            println!("ledger cycle={cycle} weights={} model_aux={} kv={kv} session_scratch={} pinned_host_bytes={} staging_device_peak=0 {}",
                stats.weight_bytes, model_bytes as u128 - stats.weight_bytes,
                backend.allocated_bytes() - baseline - model_bytes - kv, backend.pinned_host_bytes(), memory());
            let mut output = Vec::new();
            for i in 0..count {
                output.push((token, score.to_bits()));
                if i + 1 < count { (token, score) = session.step_greedy_with_score(token)?; }
            }
            if cycle == 0 { reference = output; }
            else if output != reference { return Err("reload changed generated tokens or logits".into()); }
            drop(session);
            drop(model);
            backend.synchronize()?;
            if backend.allocated_bytes() != baseline { return Err("model unload leaked device allocations".into()); }
            println!("unloaded cycle={cycle} gpu_bytes={} {}", backend.allocated_bytes(), memory());
        }
        println!("PASS cycles={cycles} exact_tokens_and_scores={count}");
        Ok(())
    })
}
#[cfg(target_os = "linux")]
fn flash_front(args: &[String], mode: &str) -> Result<()> {
    use engine_model::{flash::{Checkpoint, plan::RankPlan}, json::{self, field}, safetensors::{TensorFile, Limits}};
    use spark_engine::flash::{Weights, rows::{Rows, FrontScratch}, blocks::{Hc, HcScratch, Ple}, gdn::{Gdn, GdnScratch}};
    use std::{collections::BTreeMap, ffi::CString, net::SocketAddr, path::Path, time::Instant};
    if args.len() != 5 { return Err("usage: load-check --flash-front PTX MODEL FIXTURE.json ADDR0 ADDR1 (short-sublayers.safetensors beside fixture)".into()); }
    let root = Path::new(&args[1]);
    let fixture = json::parse(&std::fs::read(&args[2]).map_err(|e| e.to_string())?, json::Limits::default()).map_err(|e| e.to_string())?;
    let tokens = field(fixture.object().map_err(|e| e.to_string())?, "prompt_ids").and_then(|v| v.array()).map_err(|e| e.to_string())?.iter().map(|v| v.u64().map_err(|e| e.to_string()).and_then(|n| u32::try_from(n).map_err(|e| e.to_string()))).collect::<Result<Vec<_>>>()?;
    let file = TensorFile::open(&Path::new(&args[2]).with_file_name("short-sublayers.safetensors"), Limits::default()).map_err(|e| e.to_string())?; let mut snapshots = BTreeMap::new();
    for (name, tensor) in &file.header().tensors {
        if tensor.dtype != engine_model::safetensors::Dtype::F32 || tensor.shape.len() != 3 || tensor.shape[..2] != [1, tokens.len() as u64] { return Err(format!("invalid snapshot {name}")); }
        let mut bytes = vec![0; tensor.byte_len() as usize]; file.read_range(name, 0, &mut bytes).map_err(|e| e.to_string())?;
        snapshots.insert(name.clone(), bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>());
    }
    let row = |name: &str, position: usize, width: usize| snapshots.get(name).and_then(|v| v.get(position*width..(position+1)*width)).ok_or_else(|| format!("missing snapshot {name} position={position}"));
    let mut errors = BTreeMap::<String, f32>::new();
    let mut compare = |name: &str, actual: &[f32], expected: &[f32]| -> Result<()> {
        if actual.len() != expected.len() || actual.iter().chain(expected).any(|v| !v.is_finite()) { return Err(format!("invalid comparison {name}")); }
        let error = actual.iter().zip(expected).fold(0.0f32, |m, (a,b)| m.max((a-b).abs()));
        errors.entry(name.into()).and_modify(|m| *m = m.max(error)).or_insert(error); Ok(())
    };
    let addresses = [args[3].parse::<SocketAddr>().map_err(|e| e.to_string())?, args[4].parse::<SocketAddr>().map_err(|e| e.to_string())?];
    let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
    if mode == "--flash-moe" {
        use spark_engine::flash::moe::{Moe, MoeScratch};
        let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut rings = std::thread::scope(|s| { let jobs = (0..2).map(|rank| s.spawn(move || transport::Ring::connect(rank, 2, addresses[rank], addresses[1-rank], "flash-moe"))).collect::<Vec<_>>(); jobs.into_iter().map(|j| j.join().map_err(|_| "connect worker panicked")?).collect::<Result<Vec<_>>>() })?;
        spark_gpu_probe::driver::with_backend(0, &ptx, 8 * 1024 * 1024 * 1024, |backend| {
            let weights = (0..2).map(|r| Weights::load_prefix(backend, root, &checkpoint, RankPlan::new(r, 2).map_err(|e| e.to_string())?, "model.language_model.layers.0.mlp")).collect::<Result<Vec<_>>>()?;
            let quantized = Moe::activation_quantization()?; let layers = weights.iter().map(|w| Moe::load(w, 0, quantized)).collect::<Result<Vec<_>>>()?; let mut work = (0..2).map(|_| MoeScratch::new(backend)).collect::<Result<Vec<_>>>()?;
            let mut residual = (0..2).map(|_| backend.vector(10240)).collect::<Result<Vec<_>>>()?; let (mut ms, mut reduce_ms, mut host_scratch) = ([0.0; 2], 0.0, 0);
            for position in 0..tokens.len() { let mut packets = Vec::new();
                for r in 0..2 { backend.write_vector(&mut residual[r], row("layer0.expert_residual", position, 10240)?)?;
                    let start = Instant::now(); packets.push(layers[r].project(backend, &mut work[r], &residual[r])?); backend.synchronize()?; ms[r] += start.elapsed().as_secs_f64()*1000.0;
                    compare(&format!("layer0.routing_ids.rank{r}"), &packets[r].ids.map(|id| id as f32), row("layer0.routing_ids", position, 10)?)?;
                    compare(&format!("layer0.routing_weights.rank{r}"), &packets[r].weights, row("layer0.routing_weights", position, 10)?)?;
                } if !packets[0].same_routing(&packets[1]) { return Err("MLP ranks disagree on global routing".into()); } host_scratch = packets[0].values.len()*12 + 80;
                let start = Instant::now(); std::thread::scope(|s| {
                    let jobs = rings.iter_mut().zip(packets.iter_mut()).map(|(ring, packet)| s.spawn(move || { let mut scratch = vec![0.0; packet.values.len()*2]; ring.all_reduce_f32(&mut packet.values, &mut scratch) })).collect::<Vec<_>>();
                    for j in jobs { j.join().map_err(|_| "reduce worker panicked")??; } Ok::<(), String>(())
                })?; reduce_ms += start.elapsed().as_secs_f64()*1000.0;
                for r in 0..2 { let start = Instant::now(); layers[r].finish(backend, &mut work[r], &mut residual[r], &packets[r])?; backend.synchronize()?; ms[r] += start.elapsed().as_secs_f64()*1000.0;
                    compare(&format!("layer0.mlp_output.rank{r}"), &backend.read_vector(work[r].output())?, row("layer0.expert_output", position, 2560)?)?;
                    compare(&format!("layer0.output.rank{r}"), &backend.read_vector(&residual[r])?, row("layer0.output", position, 10240)?)?;
                } }
            for r in 0..2 { println!("flash_moe rank={r} positions={} prefill_ms={:.3} compute_ms={:.3} reduce_ms={reduce_ms:.3} expert_bytes={} weights_bytes={} device_scratch_bytes={} host_scratch_bytes={host_scratch} residual_bytes={} sent_bytes={} received_bytes={}", tokens.len(), ms[r]+reduce_ms, ms[r], layers[r].expert_bytes(), layers[r].weight_bytes(), work[r].device_bytes(), residual[r].len()*4, rings[r].stats.sent, rings[r].stats.received); }
            println!("flash_moe device_bytes={} snapshot_host_bytes={} both_layer_slices_resident=true input=official_attention_residual output=mlp_hc_injected", backend.allocated_bytes(), snapshots.values().map(|v| v.len()*4).sum::<usize>()); Ok(())
        })?;
        for (name, error) in &errors { println!("flash_compare component={name} positions={} max_abs_error={error:.9e}", tokens.len()); }
        if errors.iter().any(|(name, &e)| e > if name.contains("routing_ids") { 0.0 } else if name.contains("routing_weights") { 1e-5 } else { 1e-4 }) { return Err("MLP requires exact expert IDs, routing weights within 1e-5 and layer output within 1e-4".into()); }
        let quantized = spark_engine::flash::moe::Moe::activation_quantization()?;
        println!("COMPARED flash_moe ranks=2 precision={} other_activations=fp32 activation_quantization={quantized}", spark_engine::flash::moe::Moe::precision_name(quantized)); return Ok(());
    }
    if mode == "--flash-qsa" {
        use spark_engine::flash::qsa::{Qsa, QsaScratch, QsaGather};
        let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut rings = std::thread::scope(|s| { let jobs = (0..2).map(|rank| s.spawn(move || transport::Ring::connect(rank, 2, addresses[rank], addresses[1-rank], "flash-qsa"))).collect::<Vec<_>>(); jobs.into_iter().map(|j| j.join().map_err(|_| "connect worker panicked")?).collect::<Result<Vec<_>>>() })?;
        let gather = |rings: &mut [transport::Ring], batches: Vec<QsaGather>| std::thread::scope(|s| {
            let jobs = rings.iter_mut().zip(batches).map(|(ring, batch)| s.spawn(move || batch.gather(|wire, offsets| ring.all_gather_f32(wire, offsets)))).collect::<Vec<_>>();
            jobs.into_iter().map(|j| j.join().map_err(|_| "gather worker panicked")?).collect::<Result<Vec<_>>>()
        });
        spark_gpu_probe::driver::with_backend(0, &ptx, 1024 * 1024 * 1024, |backend| {
            let weights = (0..2).map(|r| Weights::load_prefix(backend, root, &checkpoint, RankPlan::new(r, 2).map_err(|e| e.to_string())?, "model.language_model.layers.3.self_attn.")).collect::<Result<Vec<_>>>()?;
            let norms = gather(&mut rings, weights.iter().map(|w| QsaGather::norms(w, 3)).collect::<Result<Vec<_>>>()?)?;
            let mut layers = weights.iter().map(|w| Qsa::load(backend, w, 3, tokens.len())).collect::<Result<Vec<_>>>()?;
            let mut work = weights.iter().map(|w| QsaScratch::new(backend, &w.plan, tokens.len(), tokens.len())).collect::<Result<Vec<_>>>()?;
            for (w, n) in work.iter_mut().zip(&norms) { w.set_norms(backend, n)?; }
            let mut input = backend.vector(tokens.len()*2560)?;
            for (offset, count, stage) in [(0, tokens.len(), "prefill"), (0, tokens.len()-1, "cached_prefix"), (tokens.len()-1, 1, "cached_last")] {
            if offset == 0 { for layer in &mut layers { layer.reset(); } }
            input.resize_within_capacity(count*2560)?; backend.write_vector(&mut input, &snapshots["layer3.attention_input"][offset*2560..(offset+count)*2560])?;
            let mut ms = [0.0; 2];
            for r in 0..2 { let start = Instant::now(); layers[r].project(backend, &mut work[r], &input, count)?; backend.synchronize()?; ms[r] += start.elapsed().as_secs_f64()*1000.0; }
            let gather_ms = 0.0;
            let mut outputs = Vec::new();
            for r in 0..2 { let start = Instant::now(); layers[r].attend(backend, &mut work[r])?; backend.synchronize()?; outputs.push(backend.read_vector(work[r].output())?); ms[r] += start.elapsed().as_secs_f64()*1000.0; }
            let start = Instant::now(); std::thread::scope(|s| {
                let jobs = rings.iter_mut().zip(outputs.iter_mut()).map(|(ring, output)| s.spawn(move || { let mut scratch = vec![0.0; output.len()*2]; ring.all_reduce_f32(output, &mut scratch) })).collect::<Vec<_>>();
                for j in jobs { j.join().map_err(|_| "reduce worker panicked")??; } Ok::<(), String>(())
            })?; for r in 0..2 { backend.write_vector(work[r].output_mut(), &outputs[r])?; } backend.synchronize()?; let reduce_ms = start.elapsed().as_secs_f64()*1000.0;
            for r in 0..2 {
                compare(&format!("layer3.qsa.{stage}.rank{r}"), &outputs[r], &snapshots["layer3.attention_output"][offset*2560..(offset+count)*2560])?;
                println!("flash_qsa rank={r} stage={stage} positions={count} elapsed_ms={:.3} compute_ms={:.3} gather_ms={gather_ms:.3} reduce_ms={reduce_ms:.3} weights_bytes={} index_keys_bytes={} kv_bytes={} scratch_bytes={} input_bytes={} sent_bytes={} received_bytes={}", ms[r]+gather_ms+reduce_ms, ms[r], layers[r].weight_bytes(), layers[r].index_bytes(), layers[r].kv_bytes(), work[r].device_bytes(), input.len()*4, rings[r].stats.sent, rings[r].stats.received);
            }
            }
            println!("flash_qsa device_bytes={} snapshot_host_bytes={} both_layer_slices_resident=true input=official_hc_mix output=reduced_before_hc", backend.allocated_bytes(), snapshots.values().map(|v| v.len()*4).sum::<usize>());
            Ok(())
        })?;
        for (name, error) in &errors { println!("flash_compare component={name} positions={} max_abs_error={error:.9e}", if name.contains("cached_last") { 1 } else if name.contains("cached_prefix") { tokens.len()-1 } else { tokens.len() }); }
        if errors.values().any(|&e| e > 1e-4) { return Err("indexed attention exceeds 1e-4 absolute error".into()); }
        println!("COMPARED flash_qsa ranks=2 precision=fp32_official_weights activation_quantization=false"); return Ok(());
    }
    let mut norm_bytes = [0u8; 256]; checkpoint.read_range(root, "model.language_model.layers.0.linear_attn.norm.weight", 0, &mut norm_bytes).map_err(|e| e.to_string())?;
    let norm = std::array::from_fn(|i| f32::from_bits((u16::from_le_bytes([norm_bytes[2*i], norm_bytes[2*i+1]]) as u32) << 16));
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend(0, &ptx, 100 * 1024 * 1024 * 1024, |backend| {
        let mut batches = Vec::new(); let mut partials = Vec::new(); let mut cached = Vec::new(); let mut resident = None;
        for rank in 0..2 {
            drop(resident.take()); let start = Instant::now();
            let weights = Weights::load(backend, root, &checkpoint, RankPlan::new(rank, 2).map_err(|e| e.to_string())?)?;
            let mut rows = Rows::new(&weights)?; batches.push(rows.capture(&weights, &tokens)?);
            println!("flash_front loaded rank={rank} resident_bytes={} load_seconds={:.3} gathered_host_bytes={} sequential_residency=true precision=fp32_official_weights activation_quantization=false", weights.resident_bytes(), start.elapsed().as_secs_f64(), batches.last_mut().unwrap().wire().len());
            {
                let mut gdn = Gdn::load(backend, &weights, 0)?; let mut work = GdnScratch::new(backend, &weights, tokens.len(), &norm)?;
                let mut input = backend.vector(tokens.len()*2560)?; backend.write_vector(&mut input, &snapshots["layer0.attention_input"])?;
                let start = Instant::now(); gdn.run(backend, &mut work, &input, tokens.len())?; backend.synchronize()?;
                println!("flash_gdn rank={rank} prefill_positions={} prefill_ms={:.3} weights_bytes={} recurrent_bytes={} state_bytes={} scratch_bytes={} input_bytes={}", tokens.len(), start.elapsed().as_secs_f64()*1000.0, weights.resident_bytes(), gdn.recurrent_bytes(), gdn.state_bytes(), work.device_bytes(), input.len()*4);
                partials.push(backend.read_vector(work.output())?);
                gdn.reset(backend)?; let mut prefix = backend.vector((tokens.len()-1)*2560)?;
                backend.write_vector(&mut prefix, &snapshots["layer0.attention_input"][..(tokens.len()-1)*2560])?; gdn.run(backend, &mut work, &prefix, tokens.len()-1)?;
                let mut last = backend.vector(2560)?; backend.write_vector(&mut last, row("layer0.attention_input", tokens.len()-1, 2560)?)?; gdn.run(backend, &mut work, &last, 1)?;
                cached.push(backend.read_vector(work.output())?);
            }
            resident = Some(weights);
        }
        std::thread::scope(|scope| {
            let jobs = batches.iter_mut().zip(partials.iter_mut()).zip(cached.iter_mut()).enumerate().map(|(rank, ((batch, output), last))| scope.spawn(move || {
                let offsets = batch.offsets().to_vec(); let mut ring = transport::Ring::connect(rank, 2, addresses[rank], addresses[1-rank], "flash-front")?;
                ring.all_gather_bytes(batch.wire(), &offsets)?; let mut scratch = vec![0.0; output.len()*2];
                ring.all_reduce_f32(output, &mut scratch)?; ring.all_reduce_f32(last, &mut scratch)?;
                println!("flash_front collective rank={rank} sent_bytes={} received_bytes={}", ring.stats.sent, ring.stats.received); Ok::<(), String>(())
            })).collect::<Vec<_>>();
            for job in jobs { job.join().map_err(|_| "row collective worker panicked")??; } Ok::<(), String>(())
        })?;
        for output in &partials { compare("layer0.gdn_prefill", output, &snapshots["layer0.attention_output"])?; }
        for output in &cached { compare("layer0.gdn_cached_last", output, row("layer0.attention_output", tokens.len()-1, 2560)?)?; }
        let baseline = backend.allocated_bytes(); let mut scratch = FrontScratch::new(backend)?; let mut hc_scratch = HcScratch::new(backend)?;
        let mut ple = Ple::load(backend, resident.as_ref().unwrap())?; let mut block = backend.vector(2560)?;
        for batch in &batches { ple.reset(backend)?; for position in 0..tokens.len() {
            scratch.expand(backend, &batch.token(position)?)?;
            compare("layer0.entry", &backend.read_vector(&scratch.residual)?, row("layer0.entry", position, 10240)?)?;
            compare("layer1.lexical", &backend.read_vector(&scratch.lexical)?, row("layer1.lexical", position, 2560)?)?;
            for layer in 0..2 {
                let p = format!("layer{layer}"); backend.write_vector(&mut scratch.residual, row(&format!("{p}.entry"), position, 10240)?)?;
                if layer == 1 { ple.project(backend, &mut scratch.residual, &scratch.lexical)?; compare("layer1.ple_residual", &backend.read_vector(&scratch.residual)?, row("layer1.attention_residual", position, 10240)?)?; }
                for (label, field, after) in [("attention", "attn", "expert_residual"), ("expert", "mlp", "output")] {
                    let hc = Hc::load(resident.as_ref().unwrap(), &format!("model.language_model.layers.{layer}.{field}_hyper_connection"))?;
                    hc.mix(backend, &mut hc_scratch, &scratch.residual)?; compare(&format!("{p}.{label}_mix"), &backend.read_vector(hc_scratch.mixed())?, row(&format!("{p}.{label}_input"), position, 2560)?)?;
                    backend.write_vector(&mut block, row(&format!("{p}.{label}_output"), position, 2560)?)?;
                    hc.combine(backend, &mut hc_scratch, &mut scratch.residual, &block)?; compare(&format!("{p}.{label}_injection"), &backend.read_vector(&scratch.residual)?, row(&format!("{p}.{after}"), position, 10240)?)?;
                }
            }
        }}
        println!("flash_front device_bytes_added={} snapshot_host_bytes={} rank_row_host_bytes={} rank_partial_host_bytes={} hf_expert_outputs_injected=true gdn_input=hf_hc_mix", backend.allocated_bytes()-baseline, snapshots.values().map(|v| v.len()*4).sum::<usize>(), batches[0].wire().len(), partials[0].len()*4);
        Ok(())
    })?;
    for (name, error) in &errors { println!("flash_compare component={name} positions={} max_abs_error={error:.9e}", if name.ends_with("cached_last") { 1 } else { tokens.len() }); }
    if errors["layer0.entry"] != 0.0 || errors["layer1.lexical"] != 0.0 { return Err("row conversion differs from official snapshots".into()); }
    println!("COMPARED flash_front ranks=2 precision=fp32_official_weights activation_quantization=false"); Ok(())
}
#[cfg(target_os = "linux")]
fn flash(args: &[String]) -> Result<(), String> {
    use engine_model::{flash::{Checkpoint, plan::{RankPlan, Selection, TWO_RANK_WEIGHT_BYTES}}, safetensors::Dtype};
    use spark_engine::flash::weights::Weights;
    use std::{ffi::CString, path::Path, time::Instant};
    let startup = Instant::now();
    if args.len() != 4 && !(args.len() == 5 && args[4] == "--all-bytes") {
        return Err("usage: load-check --flash PTX MODEL RANK RANKS [--all-bytes]".into()); }
    let rank = args[2].parse::<u64>().map_err(|e| e.to_string())?;
    let ranks = args[3].parse::<u64>().map_err(|e| e.to_string())?;
    if ranks != 2 || rank >= ranks { return Err("Flash ownership comparison requires rank 0 or 1 of 2".into()); }
    let root = Path::new(&args[1]);
    let start = Instant::now();
    let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
    let inspect_seconds = start.elapsed().as_secs_f64();
    let plan = RankPlan::new(rank, ranks).map_err(|e| e.to_string())?;
    let budget = plan.budget(&checkpoint).map_err(|e| e.to_string())?;
    if budget.weights != TWO_RANK_WEIGHT_BYTES { return Err(format!("unexpected plan bytes {}", budget.weights)); }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend(0, &ptx, 100 * 1024 * 1024 * 1024, |backend| {
        let start = Instant::now();
        let weights = Weights::load(backend, root, &checkpoint, plan)?;
        backend.synchronize()?;
        let load_seconds = start.elapsed().as_secs_f64();
        if weights.payload_bytes as u64 != budget.weights { return Err("loaded payload differs from plan".into()); }
        let status = std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
        let mem = |key: &str| status.lines().find(|s| s.starts_with(key)).and_then(|s| s.split_whitespace().nth(1)).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1024;
        println!("flash rank={rank}/{ranks} startup_ms={:.3} payload_bytes={} resident_bytes={} plan_bytes={} padding_bytes={} inspect_seconds={inspect_seconds:.3} load_seconds={load_seconds:.3} host_rss_bytes={} host_hwm_bytes={} peak_pinned_staging_bytes={} host_mirror_bytes=0 official_storage=true", startup.elapsed().as_secs_f64() * 1000.0, weights.payload_bytes, weights.resident_bytes(), budget.weights, weights.resident_bytes() - weights.payload_bytes, mem("VmRSS:"), mem("VmHWM:"), backend.peak_pinned_staging_bytes());
        let mut cases = Vec::new();
        let rows = |r| Selection::Rows(vec![r]);
        let mut add = |name: String, selection| cases.push((name, selection));
        for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] { add(name.into(), rows(rank*124160..(rank+1)*124160)); }
        for layer in [0, 46] {
            let p = format!("model.language_model.layers.{layer}.linear_attn");
            let k = rank*1024; let v = rank*3072;
            for n in ["in_proj_qkv.weight", "conv1d.weight"] { add(format!("{p}.{n}"), Selection::Rows(vec![k..k+1024, 2048+k..3072+k, 4096+v..7168+v])); }
            add(format!("{p}.in_proj_z.weight"), rows(v..v+3072));
            add(format!("{p}.out_proj.weight"), Selection::Columns(v..v+3072));
            for n in ["A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight"] { add(format!("{p}.{n}"), rows(rank*24..(rank+1)*24)); }
            add(format!("{p}.norm.weight"), rows(rank*64..(rank+1)*64));
        }
        for layer in [3, 47] {
            let p = format!("model.language_model.layers.{layer}.self_attn");
            add(format!("{p}.q_proj.weight"), rows(rank*6144..(rank+1)*6144));
            for n in ["k_proj.weight", "v_proj.weight"] { add(format!("{p}.{n}"), rows(rank*256..(rank+1)*256)); }
            add(format!("{p}.o_proj.weight"), Selection::Columns(rank*3072..(rank+1)*3072));
            for n in ["q_norm.weight", "k_norm.weight"] { add(format!("{p}.{n}"), rows(rank*128..(rank+1)*128)); }
            add(format!("{p}.indexer.index_qk_proj.weight"), rows(0..640));
            for n in ["q_layernorm.weight", "k_layernorm.weight"] { add(format!("{p}.indexer.{n}"), rows(rank*64..(rank+1)*64)); }
        }
        for layer in [0, 47] { for expert in [0, 1] {
            let split = if expert == 0 { 3 } else { 2 }; let range = if rank == 0 { 0..split } else { split..5 };
            for projection in ["gate_proj", "up_proj", "down_proj"] { for (suffix, unit) in [("weight", 128), ("weight_scale_inv", 1)] {
                let r = range.start*unit..range.end*unit;
                add(format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"), if projection == "down_proj" { Selection::Columns(r) } else { rows(r) });
            }}
        }}
        for projection in ["gate_proj", "up_proj", "down_proj"] { let r=rank*320..(rank+1)*320; add(format!("model.language_model.layers.0.mlp.shared_expert.{projection}.weight"), if projection == "down_proj" { Selection::Columns(r) } else { rows(r) }); }
        for name in ["mlp.gate.weight", "mlp.shared_expert_gate.weight", "attn_hyper_connection.hc_norm.weight", "attn_hyper_connection.input_mix_weight_down.weight", "mlp_hyper_connection.block_inject_weight.weight"] { add(format!("model.language_model.layers.0.{name}"), Selection::All); }
        for name in ["conv1d.weight", "key_proj.weight", "value_proj.weight", "norm_query.weight", "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_vocab_sizes", "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_embedding.weight_scale"] { add(format!("model.language_model.layers.1.ple.{name}"), Selection::All); }
        for part in [0, 2, 10, 63, 64, 65, 100, 127] { if part/64 == rank { add(format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{part}.weight"), rows(0..2500012)); } }
        let mut values = 0; let mut fp8_values = 0;
        for (name, expected_selection) in &cases {
            let slice = weights.tensor(name)?; let tensor = &slice.tensor; let scalar = tensor.dtype.bytes();
            if slice.selection != *expected_selection { return Err(format!("ownership differs from specification: {name}")); }
            let count = slice.bytes as u64 / scalar;
            let mut samples = vec![0, 1, 127, 128, count/2, count.saturating_sub(2), count-1];
            let stride = tensor.shape[1..].iter().product::<u64>();
            let width = if let Selection::Columns(r) = expected_selection { r.end-r.start } else { stride };
            samples.extend([width.saturating_sub(1), width, width+127, width+128]);
            if let Selection::Rows(ranges) = expected_selection { let mut edge=0; for r in ranges { edge += (r.end-r.start)*stride; samples.extend([edge.saturating_sub(1), edge]); } }
            samples.retain(|&i| i<count); samples.sort_unstable(); samples.dedup();
            for local in samples {
                let global = match expected_selection {
                    Selection::All => local,
                    Selection::Columns(r) => (local/(r.end-r.start))*tensor.shape[1] + r.start + local%(r.end-r.start),
                    Selection::Rows(ranges) => { let mut left=local; let mut found=None; for r in ranges { let n=(r.end-r.start)*stride; if left<n { found=Some(r.start*stride+left); break; } left-=n; } found.ok_or("sample outside expected rows")? },
                    Selection::Absent => return Err("absent sample".into()),
                };
                let mut source=[0u8;8]; let mut device=[0u8;8];
                checkpoint.read_range(root, name, global*scalar, &mut source[..scalar as usize]).map_err(|e| e.to_string())?;
                weights.read(name, (local*scalar) as usize, &mut device[..scalar as usize])?;
                if source != device { return Err(format!("raw slice mismatch {name} local={local} global={global}")); }
                if tensor.dtype == Dtype::Bf16 && weights.value_f32(name, local as usize)?.to_bits() != (u16::from_le_bytes(source[..2].try_into().unwrap()) as u32)<<16 { return Err(format!("BF16 promotion mismatch {name} local={local}")); }
                if tensor.dtype == Dtype::F8E4M3 {
                    let (scale_name, cell) = if let Some((p,_))=name.split_once(".shard_") { (format!("{p}.weight_scale"),0) } else { let cols=tensor.shape[1]; (format!("{}.weight_scale_inv", name.strip_suffix(".weight").unwrap()), global/cols/128*(cols/128)+(global%cols)/128) };
                    let mut scale=[0u8;2]; checkpoint.read_range(root, &scale_name, cell*2, &mut scale).map_err(|e| e.to_string())?;
                    let byte=source[0]; let exp=(byte>>3)&15; let mant=byte&7;
                    let magnitude=if exp==0 { mant as f64/512.0 } else if exp==15 && mant==7 { f64::NAN } else { (1.0+mant as f64/8.0)*2.0f64.powi(exp as i32-7) };
                    let decoded=(if byte&128 != 0 { -magnitude } else { magnitude }) as f32;
                    let expected=decoded*f32::from_bits((u16::from_le_bytes(scale) as u32)<<16); let bits=expected.to_bits(); let expected=if name.contains(".mlp.experts.") { f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000) } else { expected };
                    if weights.value_f32(name, local as usize)?.to_bits() != expected.to_bits() { return Err(format!("FP32 dequant mismatch {name} local={local} global={global}")); }
                    fp8_values+=1;
                }
                values+=1;
            }
        }
        if args.len() == 5 {
            unsafe extern "C" { fn cuMemcpyDtoH_v2(destination: *mut std::ffi::c_void, source: u64, bytes: usize) -> std::ffi::c_int; }
            let (mut source, mut device) = (vec![0u8; 4 << 20], vec![0u8; 4 << 20]);
            let mut compared = 0;
            backend.synchronize()?;
            for (name, slice) in &weights.tensors {
                let cols = if slice.tensor.dtype == Dtype::Bf16 && slice.tensor.shape.len() == 2
                    && name != "model.language_model.embed_tokens.weight" {
                    match &slice.selection { Selection::Columns(r) => (r.end - r.start) as usize,
                        _ => slice.tensor.shape[1] as usize }
                } else { 0 };
                let rows = if cols == 0 { 0 } else { slice.bytes / (cols * 2) };
                let packed = spark_gpu_probe::kernel_params::flash_bf16_packed(rows, cols);
                // Native tiles permute 64 complete rows; compact tiles permute 4.
                let tile = if packed { (if rows % 64 == 0 { 64 } else { 4 }) * cols * 2 } else { 1 };
                let address = weights.device_address(name)?;
                // Chunk boundaries are logical and physical tile boundaries;
                // no tensor mirror or device-side arithmetic is introduced.
                let spans = if let Selection::Columns(r) = &slice.selection {
                    vec![(r.start * slice.tensor.dtype.bytes(), slice.bytes as u64,
                        (r.end - r.start) as usize * slice.tensor.dtype.bytes() as usize,
                        slice.tensor.shape[1] as usize * slice.tensor.dtype.bytes() as usize)]
                } else { slice.selection.spans(&slice.tensor).map(|(at, n)| (at, n, 0, 0)).collect() };
                let mut local = 0usize;
                for (at, length, width, stride) in spans {
                    let mut done = 0;
                    while done < length as usize {
                        let (count, read) = bulk::chunk(length as usize - done, width, stride, tile.max(width), source.len())?;
                        let offset = at + if width == 0 { done as u64 } else { (done / width * stride) as u64 };
                        checkpoint.read_range(root, name, offset, &mut source[..read]).map_err(|e| e.to_string())?;
                        if width != 0 { for row in 1..count / width { source.copy_within(row * stride..row * stride + width, row * width); } }
                        if local.checked_add(count).is_none_or(|end| end > slice.bytes) { return Err("tensor read outside local slice".into()); }
                        // SAFETY: the immutable tensor owns this checked contiguous
                        // range; prior packing is synchronized and out has count bytes.
                        let code = unsafe { cuMemcpyDtoH_v2(device.as_mut_ptr().cast(), address + local as u64, count) };
                        if code != 0 { return Err(format!("CUDA bulk tensor read failed code={code}")); }
                        bulk::compare(name, local, &source[..count], &device[..count], if packed { cols } else { 0 }, rows)?;
                        done += count; local += count; compared += count;
                    }
                }
            }
            println!("PASS flash_all_bytes rank={rank} compared_bytes={compared} host_staging_bytes=8388608 max_error=0");
        }
        println!("PASS flash rank={rank} tensors={} exact_raw_values={values} exact_fp32_dequant_values={fp8_values} max_error=0", cases.len());
        Ok(())
    })
}
#[cfg(target_os = "linux")]
impl spark_engine::flash::session::Exchange for transport::Ring {
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<()> { self.all_gather_bytes(values, offsets) }
    fn gather_f32(&mut self, values: &mut [f32], offsets: &[usize]) -> Result<()> { self.all_gather_f32(values, offsets) }
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> { self.all_reduce_f32(values, scratch) }
    fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> { transport::Ring::mailbox_memory(self) }
    fn gather_slots_pipeline<T>(&mut self, offsets: &[Vec<usize>],
        fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        transport::Ring::gather_slots_pipeline(self, offsets, fill, receive, wait)
    }
    fn exchange_slots_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        transport::Ring::exchange_slots_pipeline(self, sizes, fill, receive, wait)
    }
}
#[cfg(target_os = "linux")]
fn flash_session(args: &[String]) -> Result<()> {
    use engine_model::{flash::Checkpoint, json::{self, field}}; use spark_engine::flash::session::Model; use std::{ffi::CString, net::SocketAddr, path::Path};
    if args.len() < 5 { return Err("usage: load-check --flash-session PTX MODEL FIXTURE.json ADDR0 ADDR1 [ADDR2 ...] [RANK] (omit RANK for two local ranks)".into()); }
    let rank = if args.len() == 5 { None } else { Some(args.last().unwrap().parse::<usize>().map_err(|e| e.to_string())?) };
    let world = args.len() - 3 - usize::from(rank.is_some());
    if world > 64 || rank.is_some_and(|r| r >= world) { return Err("Flash session rank must be less than world in 2..=64".into()); }
    let path = spark_engine::flash::path_topology(world)?;
    let activation_quantization = spark_engine::flash::moe::Moe::activation_quantization()?;
    let precision = spark_engine::flash::moe::Moe::precision_name(activation_quantization);
    let dump = std::env::var_os("FLASH_DUMP_LOGITS").map(std::path::PathBuf::from);
    if let Some(dir) = &dump { std::fs::create_dir_all(dir).map_err(|e| format!("create Flash logit directory: {e}"))?; }
    let root = Path::new(&args[1]); let fixture_path = Path::new(&args[2]);
    let fixture = json::parse(&std::fs::read(fixture_path).map_err(|e| e.to_string())?, json::Limits::default()).map_err(|e| e.to_string())?;
    let fixture = fixture.object().map_err(|e| e.to_string())?;
    let prompt = field(fixture, "prompt_ids").and_then(|v| v.array()).map_err(|e| e.to_string())?.iter().map(|v| v.u64().map_err(|e| e.to_string()).and_then(|n| u32::try_from(n).map_err(|e| e.to_string()))).collect::<Result<Vec<_>>>()?;
    let rows = field(fixture, "rows").and_then(|v| v.array()).map_err(|e| e.to_string())?;
    let addresses = args[3..3 + world].iter().map(|a| a.parse::<SocketAddr>().map_err(|e| e.to_string())).collect::<Result<Vec<_>>>()?;
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend(0, &ptx, 100 * 1024 * 1024 * 1024, |backend| {
        let capacity = (prompt.len()+rows.len()).max(64);
        let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
        let model = match rank { Some(rank) => Model::load_rank(backend, root, checkpoint, rank, world, capacity, 64)?, None => Model::load(backend, root, checkpoint)? };
        // Cold resident loading can outlast the ordinary collective timeout.
        let rings = std::thread::scope(|s| {
            let jobs = (0..world).filter(|&r| rank.is_none_or(|local| local == r)).map(|r| {
                let addresses = &addresses;
                s.spawn(move || transport::Ring::connect_topology(r, world, addresses[r], addresses[(r + 1) % world],
                    "flash-session", path, std::time::Duration::from_secs(600)))
            }).collect::<Vec<_>>();
            jobs.into_iter().map(|j| j.join().map_err(|_| "connect worker panicked")?).collect::<Result<Vec<_>>>()
        })?;
        let mut rings = rings;
        if let Some(r) = rank {
            // All ranks must agree on expert precision and logit dumps: a mismatch
            // would mix W8A8 and BF16 partials or desynchronize the collectives.
            let mut wire = vec![0u8; world * 2];
            wire[r * 2..r * 2 + 2].copy_from_slice(&[u8::from(activation_quantization), u8::from(dump.is_some())]);
            rings[0].all_gather_bytes(&mut wire, &(0..=world).map(|r| r * 2).collect::<Vec<_>>())?;
            if wire.chunks_exact(2).any(|flags| flags != &wire[..2]) { return Err("Flash ranks differ in FLASH_BF16_EXPERTS or FLASH_DUMP_LOGITS".into()); }
        }
        // Largest Flash collective: one 64-token MoE partial tile per rank (7.2 MB).
        if rank.is_some() { for ring in &mut rings { ring.enable_mailbox(8 << 20)?; } }
        for ring in &rings { println!("flash_session transport rank={} world={} topology={} rdma={}", ring.rank(), ring.world(), ring.topology(), ring.uses_rdma()); }
        let mut session = model.session(backend, rings, capacity, 64)?; let mut input = prompt.clone(); let (mut matches, mut logits_ok) = (true, true);
        for (position, row) in rows.iter().enumerate() {
            let row = row.object().map_err(|e| e.to_string())?; let expected = field(row, "token").and_then(|v| v.u64()).map_err(|e| e.to_string())?;
            let saved = row.get("full_logits").map(|v| v.string()).transpose().map_err(|e| e.to_string())?;
            let output = session.forward(backend, &input, saved.is_some() || dump.is_some())?; matches &= output.token as u64 == expected;
            println!("flash_session position={position} token={} expected={expected} stage={} input_tokens={} elapsed_ms={:.3} load_ms={:.3} forward_ms={:.3} ranks={world} sequential_two_rank_one_spark={}", output.token, if position==0 { "prefill" } else { "generated" }, input.len(), output.elapsed_ms, output.load_ms, output.elapsed_ms-output.load_ms, rank.is_none());
            if let Some(dir) = &dump {
                let logits = output.logits.as_ref().ok_or("missing Flash dump logits")?;
                if logits.iter().any(|v| !v.is_finite()) { return Err("nonfinite Flash dump logits".into()); }
                let bytes = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                // A rank suffix also makes a shared dump directory safe for two processes.
                let name = format!("position-{position:05}-rank-{}.f32", rank.unwrap_or(0));
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("write Flash logits: {e}"))?;
            }
            if let Some(name) = saved { let bytes = std::fs::read(fixture_path.with_file_name(name)).map_err(|e| e.to_string())?; let actual = output.logits.ok_or("missing requested logit row")?;
                if bytes.len() != actual.len()*4 || actual.iter().any(|v| !v.is_finite()) { return Err("invalid full logit row".into()); }
                let error = actual.iter().zip(bytes.chunks_exact(4)).try_fold(0.0f32, |max, (a,b)| { let expected=f32::from_le_bytes(b.try_into().unwrap()); if !expected.is_finite() { return Err("nonfinite reference logit"); } Ok(max.max((a-expected).abs())) })?;
                println!("flash_logits position={position} max_abs_error={error:.9e}"); logits_ok &= error<=1.33;
            } input.clear(); input.push(output.token);
        }
        session.report(); println!("flash_session tokens_match={} logits_within_1.33={logits_ok} precision={precision} other_activations=fp32 activation_quantization={activation_quantization}", if matches { "yes" } else { "no" });
        if !matches || !logits_ok { return Err("Flash complete-session reference mismatch".into()); } Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("load-check requires Linux CUDA"); std::process::exit(1); }
}
