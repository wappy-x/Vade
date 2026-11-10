//! Time one mailbox round trip on one CUDA device: a put kernel publishes rows to the mapped
//! slot, a spinning host thread copies them to the receive slot and publishes, and a take kernel
//! copies them back to device memory. No model, no RDMA. The first AHEAD pairs are queued before
//! the host thread serves message one, so their period is device time, not host launch time.
#[cfg(target_os = "linux")]
mod check {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;
    use spark_gpu_probe::{driver::Backend, tp_mailbox_host::{ERROR, SEND_LEN, SEND_SEQ}};

    /// Pairs queued before the echo thread starts. Small, so no launch waits for stream space.
    const AHEAD: usize = 100;
    fn percentile(sorted: &[u64], q: f64) -> f64 { sorted[((sorted.len() - 1) as f64 * q) as usize] as f64 / 1000.0 }
    fn case(b: &Backend<'_>, words: usize, pairs: usize, pause: u16) -> Result<(), String> {
        let m = b.tp_mailbox(words * 4)?;
        let mut host = m.take_host().ok_or("echo host missing")?;
        host.set_take_pause(pause);
        let l = m.layout();
        let bits: Vec<u32> = (0..words).map(|i| [0, 0x8000_0000, 0x7fc0_0123, 0x3f80_0000][i % 4] ^ (i as u32).rotate_left(11)).collect();
        let mut source = b.vector(words)?;
        b.write_vector(&mut source, &bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
        let mut target = b.vector(words)?;
        b.write_vector(&mut target, &vec![f32::from_bits(0xdead_beef); words])?;
        b.synchronize()?;
        let (start, stalled) = (AtomicBool::new(false), AtomicBool::new(false));
        let (seen, served) = std::thread::scope(|scope| -> Result<(Vec<u64>, Vec<u64>), String> {
            let (host, start, stalled) = (&host, &start, &stalled);
            let echo = scope.spawn(move || {
                let mailbox = host.words();
                let (mut seen, mut served) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
                while !start.load(Ordering::Acquire) { std::hint::spin_loop(); }
                let origin = Instant::now();
                for seq in 1..=pairs as u64 {
                    let slot = seq as usize % l.slots;
                    let waiting = Instant::now(); let mut polls = 0u32;
                    while mailbox[SEND_SEQ + slot].load(Ordering::Acquire) != seq as u32 {
                        std::hint::spin_loop(); polls = polls.wrapping_add(1);
                        // A lost message must end the run, not hold the box.
                        if polls % (1 << 20) == 0 && waiting.elapsed().as_secs() >= 3 { mailbox[ERROR].store(1, Ordering::Release); stalled.store(true, Ordering::Release); return (seen, served); }
                    }
                    let at = origin.elapsed().as_nanos() as u64;
                    let count = mailbox[SEND_LEN + slot].load(Ordering::Relaxed) as usize / 4;
                    let (from, to) = (l.send_slot(slot) + l.frame, l.recv_slot(0, slot) + l.frame);
                    for i in 0..count.min(words) { mailbox[to + i].store(mailbox[from + i].load(Ordering::Relaxed), Ordering::Relaxed); }
                    mailbox[l.recv_seq(0, slot)].store(seq as u32, Ordering::Release);
                    seen.push(at); served.push(origin.elapsed().as_nanos() as u64 - at);
                }
                (seen, served)
            });
            // The first take waits for the echo thread, so the stream holds the first AHEAD pairs.
            let mut queued = Ok(());
            for seq in 1..=pairs as u64 {
                queued = b.tp_mailbox_put(&m, &source, 1, words, words, 0, seq)
                    .and_then(|()| b.tp_mailbox_take(&m, &mut target, 1, words, words, 0, seq, 2_000_000_000));
                if queued.is_err() || stalled.load(Ordering::Acquire) { break; }
                if seq as usize == AHEAD.min(pairs) { start.store(true, Ordering::Release); }
            }
            start.store(true, Ordering::Release);
            if queued.is_err() { m.fail(); }
            let drained = b.synchronize();
            let times = echo.join().map_err(|_| "echo thread failed".to_string())?;
            queued?; drained?;
            if stalled.load(Ordering::Acquire) { return Err("echo thread saw no put for three seconds".into()); }
            Ok(times)
        })?;
        if m.failed() { return Err("mailbox reported a copy error".into()); }
        let back: Vec<u32> = b.read_vector_range(&target, 0, words)?.into_iter().map(f32::to_bits).collect();
        if back != bits { return Err("echoed words differ".into()); }
        // Pairs 10..AHEAD ran with every launch already queued (device time). Later pairs include host launch pacing.
        let ahead = AHEAD.min(pairs);
        let mut period: Vec<u64> = seen[..ahead].windows(2).skip(10).map(|w| w[1] - w[0]).collect(); period.sort_unstable();
        let mut later: Vec<u64> = seen[ahead..].windows(2).map(|w| w[1] - w[0]).collect(); later.sort_unstable();
        let mut service: Vec<u64> = served[10..ahead].to_vec(); service.sort_unstable();
        println!("ECHO words={words} bytes={} pairs={pairs} take_pause_ns={} period_us min={:.3} median={:.3} p90={:.3} p99={:.3} | host_copy_us median={:.3} | device_part_us median={:.3} | bytes_equal=true",
            words * 4, if pause == 0 { 2048 } else { pause }, percentile(&period, 0.0), percentile(&period, 0.5), percentile(&period, 0.9), percentile(&period, 0.99),
            percentile(&service, 0.5), percentile(&period, 0.5) - percentile(&service, 0.5));
        if !later.is_empty() { println!("ECHO words={words} later_pairs={} period_us median={:.3} p90={:.3} (host launch pacing included)", later.len(), percentile(&later, 0.5), percentile(&later, 0.9)); }
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            for pause in [0u16, 256, 64] {
                for words in [24usize, 4096, 16384] { case(b, words, 2000, pause)?; }
            }
            println!("PASS mailbox echo round trips; every echoed vector is byte-equal");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("tp-mailbox-echo-check requires Linux CUDA"); std::process::exit(1); }
}
