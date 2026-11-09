//! Check packed mailbox kernels against separate exchanges on one CUDA device. No model or RDMA.
#[cfg(target_os = "linux")]
mod check {
    use std::sync::atomic::Ordering;
    use spark_gpu_probe::{driver::{Backend, DeviceVector}, tp_mailbox_host::{TpMailboxHost, SEND_SEQ}, tp_route::{Channels, Links}};

    fn sent(host: &TpMailboxHost<'_>, l: Channels, seq: u64) -> Result<Vec<u32>, String> {
        let words = host.words();
        let slot = seq as usize % l.slots;
        if words[SEND_SEQ + slot].load(Ordering::Acquire) != seq as u32 { return Err("put did not publish its sequence".into()); }
        let bytes = words[spark_gpu_probe::tp_mailbox_host::SEND_LEN + slot].load(Ordering::Relaxed) as usize;
        if bytes > (l.slot_words - l.frame) * 4 || bytes % 4 != 0 { return Err("put published a bad extent".into()); }
        let start = l.send_slot(slot) + l.frame;
        Ok(words[start..start + bytes / 4].iter().map(|w| w.load(Ordering::Relaxed)).collect())
    }
    fn deliver(host: &TpMailboxHost<'_>, l: Channels, channel: usize, seq: u64, bits: &[u32]) {
        let words = host.words();
        let slot = seq as usize % l.slots;
        let start = l.recv_slot(channel, slot) + l.frame;
        for (word, &bits) in words[start..start + bits.len()].iter().zip(bits) { word.store(bits, Ordering::Relaxed); }
        words[l.recv_seq(channel, slot)].store(seq as u32, Ordering::Release);
    }
    fn upload<'a>(b: &Backend<'a>, words: &[u32]) -> Result<DeviceVector<'a>, String> {
        let mut v = b.vector(words.len())?;
        b.write_vector(&mut v, &words.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
        Ok(v)
    }
    fn read(b: &Backend<'_>, v: &DeviceVector<'_>) -> Result<Vec<u32>, String> {
        Ok(b.read_vector_range(v, 0, v.len())?.into_iter().map(f32::to_bits).collect())
    }
    pub fn run() -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let mut cases = 0;
            for world in [2, 4] {
                let l = Channels::new(&Links::new(world, false)?, 32 * (10240 + 6144) / world)?;
                let packed = b.tp_mailbox_routed(l)?;
                let separate = b.tp_mailbox_routed(l)?;
                let mut ph = packed.take_host().ok_or("packed host missing")?;
                let mut sh = separate.take_host().ok_or("separate host missing")?;
                for wide in [false, true] {
                    ph.set_wide(wide); sh.set_wide(wide);
                    for rows in [1, 3, 16, 32] {
                        for widths in [[20, 12], [24, 8], [10240, 6144]] {
                            let columns = widths.map(|w| w / world);
                            let bits: [Vec<u32>; 2] = std::array::from_fn(|p| (0..rows * widths[p]).map(|i| {
                                [0, 0x8000_0000, 0x7fc0_0123, 0xffff_ffff][i % 4] ^ (i as u32).rotate_left(11) ^ (p as u32).rotate_left(23)
                            }).collect());
                            let sources = [upload(b, &bits[0])?, upload(b, &bits[1])?];
                            for rank in 0..world {
                                let own: [Vec<u32>; 2] = std::array::from_fn(|p| (0..bits[p].len()).map(|i| {
                                    if i % widths[p] / columns[p] == rank { bits[p][i] } else { 0xdead_beef }
                                }).collect());
                                let (mut px, mut py) = (upload(b, &own[0])?, upload(b, &own[1])?);
                                let (mut sx, mut sy) = (upload(b, &own[0])?, upload(b, &own[1])?);
                                // Reuse every slot and cross the low sequence word boundary.
                                for seq in [1, 2, 3, 4, 5, 6, 7, 8, u32::MAX as u64, u32::MAX as u64 + 1, u32::MAX as u64 + 2] {
                                    for (v, words) in [(&mut px, &own[0]), (&mut py, &own[1]), (&mut sx, &own[0]), (&mut sy, &own[1])] {
                                        b.write_vector(v, &words.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
                                    }
                                    for origin in (0..world).rev().filter(|&o| o != rank) {
                                        b.tp_mailbox_put_pair(&packed, [(&sources[0], widths[0], columns[0], origin * columns[0]),
                                            (&sources[1], widths[1], columns[1], origin * columns[1])], rows, seq)?;
                                        b.tp_mailbox_put(&separate, &sources[0], rows, widths[0], columns[0], origin * columns[0], 2 * seq)?;
                                        b.tp_mailbox_put(&separate, &sources[1], rows, widths[1], columns[1], origin * columns[1], 2 * seq + 1)?;
                                        b.synchronize()?;
                                        let joined = sent(&ph, l, seq)?;
                                        let (x, y) = (sent(&sh, l, 2 * seq)?, sent(&sh, l, 2 * seq + 1)?);
                                        if joined != x.iter().chain(&y).copied().collect::<Vec<_>>() { return Err("packed put bytes differ".into()); }
                                        let channel = l.channel(origin, rank);
                                        ph.words()[l.recv_seq(channel, seq as usize % l.slots)].store(!(seq as u32), Ordering::Release);
                                        // Queue the take before the publisher starts.
                                        b.tp_mailbox_take_pair_from(&packed, [(&mut px, widths[0], columns[0], origin * columns[0]),
                                            (&mut py, widths[1], columns[1], origin * columns[1])], rows, channel, seq, 1_000_000_000)?;
                                        std::thread::scope(|scope| -> Result<(), String> {
                                            let ph = &ph;
                                            let arrived = scope.spawn(move || {
                                                deliver(ph, l, channel, seq, &joined);
                                            });
                                            b.synchronize()?;
                                            arrived.join().map_err(|_| "delivery thread failed")?;
                                            Ok(())
                                        })?;
                                        deliver(&sh, l, channel, 2 * seq, &x);
                                        deliver(&sh, l, channel, 2 * seq + 1, &y);
                                        b.tp_mailbox_take_from(&separate, &mut sx, rows, widths[0], columns[0], origin * columns[0], channel, 2 * seq, 1_000_000_000)?;
                                        b.tp_mailbox_take_from(&separate, &mut sy, rows, widths[1], columns[1], origin * columns[1], channel, 2 * seq + 1, 1_000_000_000)?;
                                        b.synchronize()?;
                                        if read(b, &px)? != read(b, &sx)? || read(b, &py)? != read(b, &sy)? { return Err("packed take bytes differ".into()); }
                                        cases += 1;
                                    }
                                    if read(b, &px)? != bits[0] || read(b, &py)? != bits[1] { return Err("gather lost words".into()); }
                                }
                            }
                        }
                    }
                }
                if packed.failed() || separate.failed() { return Err("mailbox has a copy error".into()); }
            }
            for failed_peer in [false, true] {
                let m = b.tp_mailbox(64)?;
                let mut host = m.take_host().ok_or("error test host missing")?;
                let (mut x, mut y) = (upload(b, &[0xdead_beef; 8])?, upload(b, &[0x1234_5678; 8])?);
                if failed_peer { host.fail(); }
                b.tp_mailbox_take_pair_from(&m, [(&mut x, 8, 4, 0), (&mut y, 8, 4, 0)], 1, 0, 99, 1_000)?;
                b.synchronize()?;
                if !m.failed() || read(b, &x)? != vec![0xdead_beef; 8] || read(b, &y)? != vec![0x1234_5678; 8] {
                    return Err("failed pair take changed target words or lost its error".into());
                }
            }
            println!("PASS packed mailbox cases={cases}; separate bytes equal; worlds=2,4; scalar and quad; take queued before publication; slot reuse; sequence wrap");
            println!("PASS timeout and peer error; target words unchanged");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("tp-mailbox-pair-check requires Linux CUDA"); std::process::exit(1); }
}
