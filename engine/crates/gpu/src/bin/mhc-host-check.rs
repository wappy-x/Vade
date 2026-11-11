//! L0 twin gate of the mHC host lane on one CUDA device. No model, no RDMA.
//! For every one-row shape of `mhc_params::split_cases` and two input sets (ordinary values,
//! special finite values) it runs a chain of three boundaries three ways and compares bits:
//! K = the kernels (mhc_boundary_rms, mhc_boundary_dots, mhc_boundary_finish),
//! H = the host math alone (mhc_host), F = the full host lane (put, service thread, take).
//! Planes: residual, mixing, coefficients, collapsed, output; then the final combine.
//! Then the same for row groups (the chats of a batched decode step): every shape at one row count
//! of 2 to 8, and the model's shapes (four streams, 4,095 to 4,097 wide) at every count. There H is
//! each row run alone as a one-row chain, so a row of a batch must have the bits of its own chat.
#[cfg(target_os = "linux")]
mod check {
    use std::{rc::Rc, sync::Arc};
    use spark_gpu_probe::{driver::{Backend, DeviceVector, MhcBoundaryPass, MhcHostLink}, mhc_host::{HostBoundary, Planes},
        mhc_host_service::{serve, Queue}, mhc_params::{split_cases, Shape}, norm_boundary::{NormShape, WeightType}};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }
    const SPECIAL: [u32; 20] = [0, 0x8000_0000, 0x0080_0000, 0x8080_0000, 1, 0x8000_0001, 0x007f_ffff, 0x0d80_0000, 0x8d80_0000,
        0x5e26_0000, 0xde26_0000, 0x3f80_0000, 0xbf80_0000, 0x3f80_8000, 0x3f80_7fff, 0x3f81_8000, 0x477f_e000, 0x3f7f_ffff, 0x4000_0001, 0xc0ff_ffff];
    /// 0 ordinary, 1 special finite values in two of three words, 2 the same plus NaN and infinities.
    fn input(rng: &mut Lcg, len: usize, set: usize) -> Vec<f32> {
        (0..len).map(|i| {
            if set == 2 && i % 97 == 5 { return f32::from_bits([0x7fc0_0000, 0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffff_ffff][(i / 97) % 5]); }
            if set >= 1 && i % 3 != 1 { f32::from_bits(SPECIAL[(i / 3 + i) % SPECIAL.len()]) } else { rng.next() * 4.0 }
        }).collect()
    }
    fn upload<'a>(b: &Backend<'a>, values: &[f32]) -> Result<Rc<DeviceVector<'a>>, String> {
        let mut v = b.vector(values.len())?; b.write_vector(&mut v, values)?; Ok(Rc::new(v))
    }
    fn bits(b: &Backend<'_>, v: &DeviceVector<'_>) -> Result<Vec<u32>, String> { Ok(b.read_vector_range(v, 0, v.len())?.into_iter().map(f32::to_bits).collect()) }
    fn host_bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
    /// The device planes of one pass, in the order of `NAMES`. A hosted pass must be delivered first.
    fn planes(b: &Backend<'_>, pass: &MhcBoundaryPass<'_, '_>, all: bool) -> Result<Vec<Vec<u32>>, String> {
        let mut out = vec![bits(b, pass.residual())?, bits(b, pass.coefficients())?, bits(b, &pass.normalized())?];
        if all { out.push(bits(b, pass.mixing_normalized())?); out.push(bits(b, pass.collapsed())?); }
        Ok(out)
    }
    const NAMES: [&str; 5] = ["residual", "coefficients", "output", "mixing", "collapsed"];
    /// The planes of the rows, each plane with its rows one after another, as the device holds them.
    fn host_planes(rows: &[Planes]) -> Vec<Vec<u32>> {
        let join = |plane: fn(&Planes) -> &Vec<f32>| rows.iter().flat_map(|p| host_bits(plane(p))).collect::<Vec<u32>>();
        vec![join(|p| &p.residual), join(|p| &p.coefficients), join(|p| &p.output), join(|p| &p.mixing), join(|p| &p.collapsed)]
    }
    #[derive(Default)]
    struct Tally { cases: u64, words: u64, nonfinite_cases: u64, nonfinite_differing_words: u64, nonfinite_place_differs: u64 }
    /// Finite sets: any differing bit is an error. Set 2: count differing words and check that the same words are nonfinite.
    fn compare(what: &str, shape: Shape, set: usize, site: usize, plane: &str, want: &[u32], got: &[u32], tally: &mut Tally) -> Result<(), String> {
        if want.len() != got.len() { return Err(format!("{what}: plane {plane} length differs")); }
        tally.words += want.len() as u64;
        let finite = |w: u32| w & 0x7f80_0000 != 0x7f80_0000;
        if set < 2 {
            if let Some(i) = (0..want.len()).find(|&i| want[i] != got[i]) {
                return Err(format!("{what}: rows={} streams={} hidden={} set={set} boundary={site} plane={plane} word={i} kernel={:#010x} other={:#010x}", shape.rows, shape.streams, shape.hidden, want[i], got[i]));
            }
        } else {
            for i in 0..want.len() {
                if want[i] != got[i] { tally.nonfinite_differing_words += 1; }
                if finite(want[i]) != finite(got[i]) || finite(want[i]) && want[i] != got[i] { tally.nonfinite_place_differs += 1; }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn case<'b, 'a>(b: &'b Backend<'a>, link: &MhcHostLink<'b, 'a>, queue: &Arc<Queue>, rows: u32, streams: u32, hidden: u32, set: usize, weight_type: WeightType, tally: &mut Tally) -> Result<(), String> {
        let one = Shape { rows: 1, streams, hidden, iterations: 20, rms_eps: 1e-6, eps: 1e-6 };
        let (shape, n) = (Shape { rows, ..one }, rows as usize);
        let (wide, count, h) = (one.wide(), one.coefficients(), hidden as usize);
        let mut rng = Lcg(u64::from(rows - 1) << 48 | u64::from(streams) << 32 | u64::from(hidden) << 8 | set as u64);
        // BF16 cases hold BF16 mHC weights, as the model does: the host keeps them as 16-bit lanes.
        let weights: Vec<f32> = (0..count * wide).map(|_| { let w = rng.next() * 0.05; if weight_type == WeightType::Bf16 { spark_gpu_probe::bf16::rounded(w) } else { w } }).collect();
        let base: Vec<f32> = (0..count).map(|_| rng.next()).collect();
        let scale = [0.01f32, 0.02, 0.03];
        let norm = NormShape::new(h, 1e-6, weight_type)?;
        let norm_weight: Vec<f32> = (0..h).map(|_| { let w = rng.next() + 1.5; if weight_type == WeightType::Bf16 { spark_gpu_probe::bf16::rounded(w) } else { w } }).collect();
        let first = input(&mut rng, n * wide, set);
        let blocks = [input(&mut rng, n * h, set), input(&mut rng, n * h, set), input(&mut rng, n * h, set)];
        // H: the host math alone, each row as its own one-row chain.
        let host = HostBoundary::new(one, norm, &weights, &base, &scale, &norm_weight, 3)?;
        let mut hp: [Vec<Planes>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for row in 0..n {
            let mut own = [host.planes(), host.planes(), host.planes()];
            host.run(None, &first[row * wide..(row + 1) * wide], &mut own[0]);
            { let (a, rest) = own.split_at_mut(1); host.run(Some((&a[0].residual, &a[0].coefficients)), &blocks[0][row * h..(row + 1) * h], &mut rest[0]); }
            { let (a, rest) = own.split_at_mut(2); host.run(Some((&a[1].residual, &a[1].coefficients)), &blocks[1][row * h..(row + 1) * h], &mut rest[0]); }
            for (site, planes) in own.into_iter().enumerate() { hp[site].push(planes); }
        }
        // K: the kernels. F: the full host lane. Same device inputs. Both are one-row boundaries asked for `rows` rows, as decode asks.
        let kernel = b.resident_mhc_boundary(one, norm, &weights, &base, &scale, &norm_weight)?;
        let hosted = b.resident_mhc_boundary_hosted(one, norm, &weights, &base, &scale, &norm_weight, link)?;
        let (source, block) = (upload(b, &first)?, [upload(b, &blocks[0])?, upload(b, &blocks[1])?, upload(b, &blocks[2])?]);
        let k1 = kernel.begin_rows(source.clone(), n)?;
        let f1 = hosted.begin_rows(source.clone(), n)?;
        if f1.hosted_take().is_none() { return Err(format!("an admitted boundary of {n} rows did not take the host lane")); }
        let k2 = kernel.begin_after_rows(k1, block[0].clone(), Rc::new(b.vector(n * wide)?), n)?;
        let f2 = hosted.begin_after_rows(f1, block[0].clone(), Rc::new(b.vector(n * wide)?), n)?;
        // Boundary 1 is the `previous` of boundary 2 until boundary 3 starts.
        // The service allows three deliveries after a run (four receive slots), as a decode step needs.
        // So: boundary 1 is delivered to its device planes here; boundary 3 is delivered by the final
        // combine below; boundary 2 is compared through its device output and the service planes.
        let (kp, fp) = (k2.previous().ok_or("kernel chain lost its first pass")?, f2.previous().ok_or("hosted chain lost its first pass")?);
        fp.deliver()?; b.synchronize()?;
        let kept1 = queue.planes_rows(fp.hosted_take().ok_or("first hosted pass has no job")?).ok_or("the service released planes that a live pass owns")?;
        let (want1, want2) = (planes(b, kp, true)?, planes(b, &k2, true)?);
        let (got1, got2) = (planes(b, fp, false)?, vec![Vec::new(), Vec::new(), bits(b, &f2.normalized())?]);
        let kept2 = queue.planes_rows(f2.hosted_take().ok_or("second hosted pass has no job")?).ok_or("the service lost the second pass")?;
        let k3 = kernel.begin_after_rows(k2, block[1].clone(), Rc::new(b.vector(n * wide)?), n)?;
        let f3 = hosted.begin_after_rows(f2, block[1].clone(), Rc::new(b.vector(n * wide)?), n)?;
        b.synchronize()?;
        let (want3, got3) = (planes(b, &k3, true)?, vec![Vec::new(), Vec::new(), bits(b, &f3.normalized())?]);
        let kept3 = queue.planes_rows(f3.hosted_take().ok_or("third hosted pass has no job")?).ok_or("the service lost the third pass")?;
        for (site, (want, got, kept)) in [(&want1, &got1, &kept1), (&want2, &got2, &kept2), (&want3, &got3, &kept3)].into_iter().enumerate() {
            let (alone, pooled) = (host_planes(&hp[site]), host_planes(kept));
            for (p, name) in NAMES.iter().enumerate() {
                compare("host math against kernels", shape, set, site, name, &want[p], &alone[p], tally)?;
                compare("service planes against kernels", shape, set, site, name, &want[p], &pooled[p], tally)?;
                if p < 3 && !got[p].is_empty() { compare("full host lane on the device against kernels", shape, set, site, name, &want[p], &got[p], tally)?; }
            }
        }
        // The real residual consumer: the final combine on the device, after a kernel chain and after a hosted chain.
        let (out_k, out_f) = (b.vector(n * wide)?, b.vector(n * wide)?);
        k3.materialize(&block[2], &out_k)?; f3.materialize(&block[2], &out_f)?; b.synchronize()?;
        compare("final combine after the host lane against kernels", shape, set, 3, "combined", &bits(b, &out_k)?, &bits(b, &out_f)?, tally)?;
        if link.queue().failed() { return Err("the host service failed".into()); }
        tally.cases += 1; if set == 2 { tally.nonfinite_cases += 1; }
        Ok(())
    }
    /// `time <boundaries> <sets> <helpers>`: GPU event time of one boundary on the model-like shape (4 streams, 4,096 wide),
    /// kernels against the host lane, over `sets` rotating parameter sets (1 = warm weights, 90 = cold as in a step).
    /// No CHAIN-SHADOW tail and no consumer here: this is the boundary alone.
    /// `rows` (1 to 8) times a row group: the kernels with that many rows against the host lane with one row a core.
    /// The mHC weights are BF16 values, as the model's are.
    /// `stand_in_mib` > 0 puts a device copy of that many MiB behind every boundary, as the stage a step runs between two
    /// boundaries (the host lane reads its next weights under it), and prints the boundary's time with the copies taken out.
    fn time(boundaries: usize, sets: usize, helpers: usize, shadow_blocks: u32, rows: usize, stand_in_mib: usize) -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let shape = Shape { rows: 1, streams: 4, hidden: 4096, iterations: 20, rms_eps: 1e-6, eps: 1e-6 };
            let (wide, count, h) = (shape.wide(), shape.coefficients(), 4096usize);
            let mailbox = b.tp_mailbox(rows * wide * 4)?;
            let mut host = mailbox.take_host().ok_or("host endpoint is taken")?; host.set_wide(true);
            let queue = Queue::new(); let link = MhcHostLink::new(&mailbox, queue.clone(), helpers + 1);
            let norm = NormShape::new(h, 1e-6, WeightType::Bf16)?; let mut rng = Lcg(0x71e5);
            let norm_weight: Vec<f32> = (0..h).map(|_| spark_gpu_probe::bf16::rounded(rng.next() + 1.5)).collect();
            let (mut kernel, mut hosted) = (Vec::new(), Vec::new());
            for _ in 0..sets {
                let weights: Vec<f32> = (0..count * wide).map(|_| spark_gpu_probe::bf16::rounded(rng.next() * 0.05)).collect(); let base: Vec<f32> = (0..count).map(|_| rng.next()).collect();
                kernel.push(b.resident_mhc_boundary(shape, norm, &weights, &base, &[0.01, 0.02, 0.03], &norm_weight)?);
                hosted.push(b.resident_mhc_boundary_hosted(shape, norm, &weights, &base, &[0.01, 0.02, 0.03], &norm_weight, &link)?);
            }
            let (source, block) = (upload(b, &input(&mut rng, rows * wide, 0))?, upload(b, &input(&mut rng, rows * h, 0))?);
            // CHAIN-SHADOW as in production: 7 MB under the dots launch and 4 MB under the finish launch, from 16 rotating
            // 11 MB stand-ins for the next stage's weights. `shadow_blocks` = 0 runs without a tail.
            let stand_ins: Vec<_> = (0..if shadow_blocks > 0 { 16 } else { 0 }).map(|_| b.vector((11 << 20) / 4)).collect::<Result<_, _>>()?;
            let stage_words = stand_in_mib << 18; let stage = if stage_words > 0 { Some((b.vector(stage_words)?, std::cell::RefCell::new(b.vector(stage_words)?))) } else { None };
            let stage_alone = match &stage { Some((from, to)) => { let mut best = u128::MAX;
                for _ in 0..3 { b.synchronize()?; let begin = b.gpu_mark()?; for _ in 1..boundaries { b.copy_vector_range(from, &mut to.borrow_mut(), 0, 0, stage_words)?; } let end = b.gpu_mark()?; b.synchronize()?; best = best.min(begin.ns_until(&end)?); }
                best }, None => 0 };
            std::thread::scope(|scope| {
                let service = scope.spawn(|| serve(host, &queue, helpers));
                // This thread queues the launches, as the decode thread does: it takes the first fast core past the host lane's five.
                spark_gpu_probe::driver::prefer_fast_core(5)?;
                let result = (|| -> Result<(), String> {
                    let mut last = [Vec::new(), Vec::new()];
                    for round in 0..3 { for (arm, (name, set)) in [("kernels", &kernel), ("host_lane", &hosted)].into_iter().enumerate() {
                        b.synchronize()?; let wall = std::time::Instant::now(); let begin = b.gpu_mark()?;
                        let mut pass = set[0].begin_rows(source.clone(), rows)?;
                        for i in 1..boundaries {
                            let shadow = if shadow_blocks > 0 { let weight = &stand_ins[i % stand_ins.len()];
                                Some(spark_gpu_probe::driver::Shadow { rms: None, dots: Some(b.shadow_vector_range(weight, 0, 7 << 20)?), finish: Some(b.shadow_vector_range(weight, 7 << 20, 4 << 20)?), blocks: shadow_blocks, evict_last: false }) } else { None };
                            pass = set[i % sets].begin_after_rows_shadowed(pass, block.clone(), Rc::new(b.vector(rows * wide)?), rows, shadow.as_ref())?;
                            if let Some((from, to)) = &stage { b.copy_vector_range(from, &mut to.borrow_mut(), 0, 0, stage_words)?; }
                        }
                        let queued = wall.elapsed(); let end = b.gpu_mark()?; b.synchronize()?;
                        let out = bits(b, &pass.normalized())?; drop(pass);
                        if stage.is_some() { println!("TIME round={round} arm={name} rows={rows} stand_in_mib={stand_in_mib} stand_in_us={:.3} spaced_gpu_us_per_boundary={:.3}",
                            stage_alone as f64 / 1000.0 / (boundaries - 1) as f64, (begin.ns_until(&end)? as f64 - stage_alone as f64) / 1000.0 / boundaries as f64); }
                        println!("TIME round={round} arm={name} rows={rows} boundaries={boundaries} sets={sets} helpers={helpers} shadow_blocks={shadow_blocks} gpu_us_per_boundary={:.3} host_queue_us_per_boundary={:.3}",
                            begin.ns_until(&end)? as f64 / 1000.0 / boundaries as f64, queued.as_nanos() as f64 / 1000.0 / boundaries as f64);
                        let stats = queue.take_stats();
                        if stats[0] > 0 { let per = |i: usize| stats[i] as f64 / 1000.0 / stats[0] as f64;
                            println!("TIME round={round} host_parts_us jobs={} wait_put={:.3} copy_in={:.3} residual={:.3} rms_chain={:.3} folds={:.3} finish={:.3} answer={:.3}", stats[0], per(1), per(2), per(3), per(4), per(5), per(6), per(7)); }
                        last[arm] = out;
                    } if last[0] != last[1] { return Err("the last output plane differs between the arms".into()); } }
                    println!("PASS the last output plane of the chain is bit-equal between the arms"); Ok(())
                })();
                queue.stop();
                let served = service.join().map_err(|_| "service thread panicked".to_string()).and_then(|r| r);
                result.and(served)
            })
        })
    }
    pub fn run() -> Result<(), String> {
        let args: Vec<String> = std::env::args().collect();
        if args.get(1).is_some_and(|a| a == "time") {
            let number = |i: usize, default: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
            return time(number(2, 2000).max(2), number(3, 90).max(1), number(4, 2), number(5, 0) as u32, number(6, 1).clamp(1, spark_gpu_probe::mhc_host_service::ROWS_MAX), number(7, 0));
        }
        let limit: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let shapes: Vec<(u32, u32)> = split_cases().into_iter().filter(|(_, _, rows)| rows.contains(&1)).map(|(n, h, _)| (n, h)).take(limit).collect();
            // Row groups: one count of 2 to 8 a shape, in rotation among the counts the shape admits; every count at the model's shapes.
            let groups: Vec<(u32, u32, u32)> = split_cases().into_iter().take(limit).enumerate().flat_map(|(i, (n, h, rows))| {
                let many: Vec<u32> = rows.into_iter().filter(|&rows| rows > 1).collect();
                let picked = if n == 4 && (4095..=4097).contains(&h) { many } else { many.get(i % many.len().max(1)).copied().into_iter().collect() };
                picked.into_iter().map(move |rows| (rows, n, h))
            }).collect();
            let widest = shapes.iter().map(|&(n, h)| n as usize * h as usize).max().ok_or("no shape")?;
            let mailbox = b.tp_mailbox(widest * 4 * spark_gpu_probe::mhc_host_service::ROWS_MAX)?;
            let mut host = mailbox.take_host().ok_or("host endpoint is taken")?; host.set_wide(true);
            let queue = Queue::new();
            queue.reserve_coefficients(shapes.iter().map(|&(streams, _)| streams as usize * (streams as usize + 2)).max().ok_or("no coefficient shape")?)?;
            let link = MhcHostLink::new(&mailbox, queue.clone(), 3);
            let mut tally = Tally::default();
            let result = std::thread::scope(|scope| {
                let service = scope.spawn(|| serve(host, &queue, 2));
                let result = (|| { for (i, &(streams, hidden)) in shapes.iter().enumerate() {
                    for set in 0..3 { case(b, &link, &queue, 1, streams, hidden, set, if i % 2 == 0 { WeightType::Bf16 } else { WeightType::F32 }, &mut tally)?; }
                } for (i, &(rows, streams, hidden)) in groups.iter().enumerate() {
                    for set in 0..3 { case(b, &link, &queue, rows, streams, hidden, set, if i % 2 == 0 { WeightType::Bf16 } else { WeightType::F32 }, &mut tally)?; }
                } b.synchronize() })();
                queue.stop();
                let served = service.join().map_err(|_| "service thread panicked".to_string()).and_then(|r| r);
                result.and(served)
            });
            println!("MHC HOST CHECK shapes={} row_groups={} cases={} compared_words={} nonfinite_cases={} nonfinite_differing_words={} nonfinite_place_or_finite_value_differs={}",
                shapes.len(), groups.len(), tally.cases, tally.words, tally.nonfinite_cases, tally.nonfinite_differing_words, tally.nonfinite_place_differs);
            result?;
            if mailbox.failed() { return Err("mailbox reported an error".into()); }
            if tally.nonfinite_place_differs != 0 { return Err("a nonfinite case has a finite word that differs, or a word that is finite on one side only".into()); }
            println!("PASS finite inputs: every plane of kernels, host math, service and full host lane is bit-equal; final combine bit-equal");
            println!("PASS row groups: every row of 2 to 8 rows has the bits of its own one-row chain, on the kernels and on the host lane");
            println!("PASS nonfinite inputs: the same words are nonfinite on both sides and every finite word is bit-equal (NaN payload bits may differ: nonfinite_differing_words above)");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("mhc-host-check requires Linux CUDA"); std::process::exit(1); }
}
