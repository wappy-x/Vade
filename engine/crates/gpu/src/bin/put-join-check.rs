//! The joined put against the launches it replaces, on one CUDA device. No model, no RDMA.
//! Head arm, as the stage gather and `gather_device` queue one message: BF16 round of each part
//! that asks, join of the parts, fresh zero fill of the gathered output, scatter copy of the own
//! rows, finite put. Joined arm: one launch.
//! BYTES: slot words, published length and sequence, and every output word are equal between the
//! arms and equal to the host's expectation, with the output and the slot poisoned first.
//! TIME: device time a message while a blocked take holds the stream, so every launch of the
//! burst is queued before the first one runs (no host pacing inside the span).
#[cfg(target_os = "linux")]
mod check {
    use std::sync::atomic::{AtomicU32, Ordering};
    use spark_gpu_probe::{driver::{Backend, DeviceVector, TpMailbox}, resident_launch::Kind, tp_mailbox_host::{SEND_LEN, SEND_SEQ}, tp_put_join::SHAPE};

    const POISON: u32 = 0xdead_beef;
    /// (own words a job, jobs, parts): the step's message shapes (pack-c1040.txt) and the three-word argmax message.
    const SHAPES: [(usize, usize, usize); 11] = [(3, 1, 1), (1168, 1, 1), (2048, 1, 1), (1024, 3, 3), (2048, 2, 2), (2048, 5, 1), (2112, 6, 6), (2048, 6, 1), (2048, 8, 1), (512, 22, 2), (4840, 16, 1)];
    fn bits(words: usize, salt: u32) -> Vec<u32> {
        // Finite words only: signed zero, subnormal, BF16 ties both ways, large and ordinary values.
        (0..words).map(|i| { let n = (i as u32).wrapping_mul(2654435761).wrapping_add(salt);
            match i % 7 { 0 => 0x8000_0000, 1 => n & 0x007f_ffff, 2 => 0x3f80_8000 | (n & 0x0001_0000), 3 => 0x7e80_0000 | (n & 0x00ff_ffff), 4 => 0xbf00_7fff, _ => (n & 0xbfff_ffff) % 0x7f00_0000 | (n & 0x8000_0000) } }).collect()
    }
    fn upload<'d>(b: &Backend<'d>, bits: &[u32]) -> Result<DeviceVector<'d>, String> {
        let mut v = b.vector(bits.len())?; b.write_vector_bytes(&mut v, &bits.iter().flat_map(|w| w.to_ne_bytes()).collect::<Vec<_>>())?; Ok(v)
    }
    fn read(b: &Backend<'_>, v: &DeviceVector<'_>) -> Result<Vec<u32>, String> { Ok(b.read_vector_range(v, 0, v.len())?.into_iter().map(f32::to_bits).collect()) }
    fn rows(count: usize, jobs: usize, rank: usize) -> Vec<[usize; 3]> { (0..jobs).map(|job| [job * count, job * 2 * count + rank * count, count]).collect() }
    /// The local parts a stage holds before its gather, each with its own round request.
    struct Local<'d> { parts: Vec<(DeviceVector<'d>, bool)>, words: usize, sent: Vec<u32>, source: Vec<Vec<u32>> }
    fn local<'d>(b: &Backend<'d>, words: usize, cut: usize, round: impl Fn(usize) -> bool, salt: u32) -> Result<Local<'d>, String> {
        let all = bits(words, salt); let (mut parts, mut sent, mut source) = (Vec::new(), Vec::new(), Vec::new());
        for part in 0..cut { let piece = &all[words * part / cut..words * (part + 1) / cut];
            sent.extend(piece.iter().map(|&w| if round(part) { spark_gpu_probe::bf16::rounded(f32::from_bits(w)).to_bits() } else { w }));
            parts.push((upload(b, piece)?, round(part))); source.push(piece.to_vec()); }
        Ok(Local { parts, words, sent, source })
    }
    /// The launches before the put, as the head queues them. Returns (rounded parts, joined parts, gathered output).
    fn pack<'d>(b: &Backend<'d>, local: &Local<'d>, rows: &[[usize; 3]]) -> Result<(Vec<Option<DeviceVector<'d>>>, Option<DeviceVector<'d>>, DeviceVector<'d>), String> {
        let rounded = local.parts.iter().map(|(part, round)| if *round { b.round_bf16_lanes(part).map(|v| Some(v.into_vector())) } else { Ok(None) }).collect::<Result<Vec<_>, _>>()?;
        let each: Vec<(&DeviceVector<'d>, usize)> = local.parts.iter().zip(&rounded).map(|((part, _), r)| { let v = r.as_ref().unwrap_or(part); (v, v.len()) }).collect();
        let packed = if each.len() > 1 { Some(b.resident_join(&each, &mut None, 0)?) } else { None };
        let mut joined = b.vector(local.words * 2)?; b.zero_vector(&mut joined)?;
        b.resident_scatter_copy(packed.as_ref().unwrap_or(each[0].0), &mut joined, rows)?;
        Ok((rounded, packed, joined))
    }
    fn put<'d>(b: &Backend<'d>, m: &TpMailbox<'d>, source: &DeviceVector<'d>, seq: u64) -> Result<(), String> { b.tp_mailbox_put_finite(m, source, 1, source.len(), source.len(), 0, seq) }
    fn head<'d>(b: &Backend<'d>, m: &TpMailbox<'d>, local: &Local<'d>, rows: &[[usize; 3]], seq: u64) -> Result<Vec<DeviceVector<'d>>, String> {
        let (rounded, packed, joined) = pack(b, local, rows)?;
        put(b, m, packed.as_ref().or(rounded[0].as_ref()).unwrap_or(&local.parts[0].0), seq)?;
        Ok(rounded.into_iter().flatten().chain(packed).chain([joined]).collect())
    }
    fn joined<'d>(b: &Backend<'d>, m: &TpMailbox<'d>, local: &Local<'d>, rows: &[[usize; 3]], seq: u64, poison: bool) -> Result<Vec<DeviceVector<'d>>, String> {
        let mut out = if poison { upload(b, &vec![POISON; local.words * 2])? } else { b.vector(local.words * 2)? };
        if !b.tp_mailbox_put_join(m, &local.parts.iter().map(|(part, round)| (part, *round)).collect::<Vec<_>>(), &mut out, rows, seq)? { return Err(format!("the joined put refused words={} jobs={}", local.words, rows.len())); }
        Ok(vec![out])
    }
    fn slot(m: &TpMailbox<'_>, all: &[AtomicU32], seq: u64, words: usize) -> (Vec<u32>, u32, u32) {
        let l = m.layout(); let slot = seq as usize % l.slots; let start = l.send_slot(slot) + l.frame;
        ((0..words).map(|i| all[start + i].load(Ordering::Relaxed)).collect(), all[SEND_LEN + slot].load(Ordering::Acquire), all[SEND_SEQ + slot].load(Ordering::Acquire))
    }
    /// Byte equality of one shape, in the mode the backend is in (eager or recorded). Returns compared words.
    fn bytes(b: &Backend<'_>, (count, jobs, cut): (usize, usize, usize), round: u8, ctas: u32, wide: bool, seq: &mut u64) -> Result<usize, String> {
        let words = count * jobs; let m = b.tp_mailbox(words * 4)?; let mut host = m.take_host().ok_or("host endpoint is taken")?;
        host.set_wide(wide); host.set_take_ctas(ctas); let all = host.words(); let l = m.layout(); let mut compared = 0;
        // round 0: no part; 1: every part; 2: odd parts and the last one.
        let local = local(b, words, cut, |part| round == 1 || round == 2 && (part % 2 == 1 || part + 1 == cut), count as u32 ^ jobs as u32)?;
        for rank in 0..2 {
            let rows = rows(count, jobs, rank); let mut expected = vec![0u32; words * 2];
            for &[from, to, n] in &rows { expected[to..to + n].copy_from_slice(&local.sent[from..from + n]); }
            for s in 0..l.slots { for i in 0..words { all[l.send_slot(s) + l.frame + i].store(POISON, Ordering::Relaxed); } }
            let (a, c) = (*seq + 1, *seq + 2); *seq += 2;
            let h = head(b, &m, &local, &rows, a)?; let j = joined(b, &m, &local, &rows, c, true)?; b.synchronize()?;
            let (h_out, j_out, h_slot, j_slot) = (read(b, h.last().ok_or("no head output")?)?, read(b, &j[0])?, slot(&m, all, a, words), slot(&m, all, c, words));
            let name = format!("count={count} jobs={jobs} parts={cut} round={round} rank={rank} ctas={ctas} four_word={wide}");
            if m.failed() { return Err(format!("mailbox error {name}")); }
            if h_slot != (local.sent.clone(), (words * 4) as u32, a as u32) { return Err(format!("head slot differs from the expectation {name}")); }
            if j_slot != (local.sent.clone(), (words * 4) as u32, c as u32) { return Err(format!("JOIN SLOT BYTE FAIL {name}")); }
            if h_out != expected { return Err(format!("head output differs from the expectation {name}")); }
            if j_out != expected { return Err(format!("JOIN OUTPUT BYTE FAIL {name} first={:?}", j_out.iter().zip(&expected).position(|(a, b)| a != b))); }
            for ((part, _), source) in local.parts.iter().zip(&local.source) { if &read(b, part)? != source { return Err(format!("a source part changed {name}")); } }
            compared += words * 3;
        }
        Ok(compared)
    }
    /// A nonfinite word, or one that rounds to infinity, sets the sticky error and nothing is published.
    fn nonfinite(b: &Backend<'_>) -> Result<usize, String> {
        let mut cases = 0;
        for (count, jobs, cut) in [(3usize, 1usize, 1usize), (2048, 1, 1), (2048, 6, 3), (2048, 8, 1)] { let words = count * jobs;
            for at in [0, 1, words / 2, words - 1] { for (value, round, refused) in [(0x7f80_0000u32, false, true), (0xffc0_0001, false, true), (0x7f7f_ffff, true, true), (0x7f7f_ffff, false, false), (0xff80_0000, true, true)] {
                let m = b.tp_mailbox(words * 4)?; let mut host = m.take_host().ok_or("host endpoint is taken")?; host.set_wide(true); host.set_take_ctas(6);
                let mut local = local(b, words, cut, |_| round, 9)?; let part = (0..cut).rfind(|&part| words * part / cut <= at).ok_or("no part")?;
                let mut piece = local.source[part].clone(); piece[at - words * part / cut] = value; local.parts[part].0 = upload(b, &piece)?;
                let _out = joined(b, &m, &local, &rows(count, jobs, 0), 1, true)?; b.synchronize()?;
                let published = host.words()[SEND_SEQ + 1 % m.layout().slots].load(Ordering::Acquire) == 1;
                if m.failed() != refused || published == refused { return Err(format!("JOIN NONFINITE FAIL words={words} parts={cut} at={at} value={value:#010x} round={round} failed={} published={published}", m.failed())); }
                cases += 1;
            } }
        }
        Ok(cases)
    }
    struct Gate<'m, 'd> { mailbox: &'m TpMailbox<'d>, words: &'m [AtomicU32], seq: u64, sink: DeviceVector<'d> }
    /// Device microseconds a message for `burst` messages queued behind a blocked take.
    fn burst<'d>(b: &Backend<'d>, gate: &mut Gate<'_, 'd>, burst: usize, seq: &mut u64, mut message: impl FnMut(u64) -> Result<Vec<DeviceVector<'d>>, String>) -> Result<f64, String> {
        gate.seq += 1; let g = gate.seq; let l = gate.mailbox.layout();
        b.tp_mailbox_take(gate.mailbox, &mut gate.sink, 1, 4, 4, 0, g, 5_000_000_000)?;
        let begin = b.gpu_mark()?; let mut keep = Vec::with_capacity(burst);
        for _ in 0..burst { *seq += 1; keep.push(message(*seq)?); }
        let end = b.gpu_mark()?;
        gate.words[l.recv_seq(0, g as usize % l.slots)].store(g as u32, Ordering::Release);
        b.synchronize()?; drop(keep);
        Ok(begin.ns_until(&end)? as f64 / 1000.0 / burst as f64)
    }
    fn median(mut v: Vec<f64>) -> f64 { v.sort_by(f64::total_cmp); v[v.len() / 2] }
    pub fn run() -> Result<(), String> {
        let args: Vec<String> = std::env::args().collect();
        let number = |i: usize, default: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
        let (messages, repeats, sweep) = (number(1, 60).max(1), number(2, 9).max(1), number(3, 0) != 0);
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let (mut seq, mut compared) = (0u64, 0usize);
            for &shape in &SHAPES { for round in 0..3 { for (ctas, wide) in [(6, true), (6, false), (1, true), (96, true), (5, false)] { compared += bytes(b, shape, round, ctas, wide, &mut seq)?; } } }
            for forced in [1 << 16 | 256, 7 << 16 | 256, 3 << 16 | 1024, 90 << 16 | 128] { SHAPE.store(forced, Ordering::Relaxed);
                for &shape in &SHAPES { for wide in [true, false] { compared += bytes(b, shape, 2, 6, wide, &mut seq)?; } } }
            SHAPE.store(0, Ordering::Relaxed);
            println!("JOIN BYTES PASS mode=eager shapes={} compared_words={compared} differing_words=0", SHAPES.len());
            println!("JOIN NONFINITE PASS cases={} publish_refused=true", nonfinite(b)?);
            let queue = b.resident_queue()?; b.enable_resident_launches()?;
            let before = b.stats().recorded_kinds; compared = 0;
            for &shape in &SHAPES { for round in 0..3 { for (ctas, wide) in [(6, true), (5, false)] { compared += bytes(b, shape, round, ctas, wide, &mut seq)?; } } }
            let kinds = b.stats().recorded_kinds; let recorded = |k: Kind| kinds[k as usize] - before[k as usize];
            if recorded(Kind::PutJoin) != (SHAPES.len() * 3 * 2 * 2) as u128 { return Err(format!("the joined put was not recorded: {}", recorded(Kind::PutJoin))); }
            println!("JOIN BYTES PASS mode=recorded shapes={} compared_words={compared} differing_words=0 recorded_put_join={} recorded_copy={} recorded_round={} recorded_put={} recorded_put_wide={}",
                SHAPES.len(), recorded(Kind::PutJoin), recorded(Kind::Copy), recorded(Kind::Round), recorded(Kind::Put), recorded(Kind::PutWide));
            // TIME, recorded mode (graph pieces, as a persistent decode step runs), production mailbox settings.
            let gate_mailbox = b.tp_mailbox(16)?; let gate_host = gate_mailbox.take_host().ok_or("gate endpoint is taken")?;
            let mut gate = Gate { mailbox: &gate_mailbox, words: gate_host.words(), seq: 0, sink: b.vector(4)? };
            for &(count, jobs, cut) in &SHAPES { for round in [false, true] {
                let words = count * jobs; let m = b.tp_mailbox(words * 4)?; let mut host = m.take_host().ok_or("host endpoint is taken")?; host.set_wide(true); host.set_take_ctas(6);
                let local = local(b, words, cut, |_| round, 5)?; let rows = rows(count, jobs, 0); b.synchronize()?; let mut seq = 0u64;
                let forced: Vec<u32> = if sweep { vec![1 << 16 | 1024, 4 << 16 | 1024, 8 << 16 | 1024, 16 << 16 | 1024, 32 << 16 | 1024, 16 << 16 | 512] } else { Vec::new() };
                let (mut h, mut j, mut p, mut k) = (Vec::new(), Vec::new(), Vec::new(), Vec::new()); let mut f = vec![Vec::new(); forced.len()];
                for _ in 0..repeats {
                    h.push(burst(b, &mut gate, messages, &mut seq, |s| head(b, &m, &local, &rows, s))?);
                    j.push(burst(b, &mut gate, messages, &mut seq, |s| joined(b, &m, &local, &rows, s, false))?);
                    p.push(burst(b, &mut gate, messages, &mut seq, |s| { put(b, &m, &local.parts[0].0, s)?; Ok(Vec::new()) })?);
                    k.push(burst(b, &mut gate, messages, &mut seq, |_| { let (rounded, packed, joined) = pack(b, &local, &rows)?; Ok(rounded.into_iter().flatten().chain(packed).chain([joined]).collect()) })?);
                    for (shape, times) in forced.iter().zip(&mut f) { SHAPE.store(*shape, Ordering::Relaxed);
                        times.push(burst(b, &mut gate, messages, &mut seq, |s| joined(b, &m, &local, &rows, s, false))?); SHAPE.store(0, Ordering::Relaxed); }
                }
                if m.failed() { return Err("mailbox error in a timed burst".into()); }
                let (h, j) = (median(h), median(j));
                println!("JOIN TIME count={count} jobs={jobs} parts={cut} words={words} round={round} head_us={h:.2} joined_us={j:.2} saved_us={:.2} first_part_put_only_us={:.2} pack_only_us={:.2}{} burst={messages} repeats={repeats}",
                    h - j, median(p), median(k), forced.iter().zip(f).map(|(shape, times)| format!(" joined_1+{}x{}_us={:.2}", shape >> 16, shape & 0xffff, median(times))).collect::<String>());
            } }
            drop(queue);
            println!("JOIN COMPLETE");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("put-join-check requires Linux CUDA"); std::process::exit(1); }
}
