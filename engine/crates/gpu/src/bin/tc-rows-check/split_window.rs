//! Device correctness, bit-exact M=2/4/8 == M=1, zero-copy ring delta, C8 == C1, and guard-poison
//! checks for `FLASH_WINDOW_SPLIT16`, `FLASH_WINDOW_SPLIT32`, and `FLASH_WINDOW_REDUCE`.
use spark_gpu_probe::driver::{Backend, DeviceVector};

const GUARD: usize = 64;
const POISON: u32 = 0x7fc0_dead;

fn weight(i: usize) -> u16 {
    let n = (i as u32).wrapping_mul(2_654_435_761) >> 16;
    match i % 11 {
        0 => 0x0000,
        1 => 0x8000,
        2 => 0x0001 | (n as u16 & 0x007f),
        3 => 0x4100 | (n as u16 & 0x007f),
        _ => (0x3800 + (n % 0x0800)) as u16 | ((n as u16 & 1) << 15),
    }
}
fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }
fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> {
    b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared()
}

pub fn run(b: &Backend<'_>) -> Result<usize, String> {
    let mut total_words = 0usize;
            // 5b2. FLASH_WINDOW_SPLIT (GQA-packed Split-KV decode/verify): accuracy vs CPU reference, 100% bit-exact
            // invariance M=1 vs M=2/4/8, zero-copy provisional ring delta vs committed ring, C1 vs C8 multi-chat planes,
            // sliding-window wrap at pos >= 1024, global 4K context, empty window, and guard poisons.
            {
                use spark_gpu_probe::{block_rows, causal_ops::{Op, Shape}, driver::OneClips, flash_window::{self, FW_HEAD_DIM, FW_REL_DIM}};
                let (heads, kv_heads, extent, scale, query_bound, moved_bound) = (24usize, 8usize, 1024usize, 0.088_388_35f32, 22.627_417f32, 8.0f32);
                let pack_u32 = |vals: &[u16]| -> Vec<u32> {
                    vals.as_chunks::<2>().0.iter().map(|[lo, hi]| u32::from(*lo) | (u32::from(*hi) << 16)).collect()
                };
                let pack_f32 = |vals: &[u32]| -> Vec<f32> { vals.iter().map(|&w| f32::from_bits(w)).collect() };
                let shared = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
                let q_scale: Vec<f32> = (0..heads * FW_HEAD_DIM).map(|i| 1.0 + input(i + 211) * 0.25).collect();
                let m_scale: Vec<f32> = (0..heads * FW_REL_DIM).map(|i| 1.0 + input(i + 223) * 0.25).collect();
                let table: Vec<f32> = (0..FW_REL_DIM * extent).map(|i| input(i + 227) * 0.25).collect();
                let (qs_dev, ms_dev, t_dev) = (shared(&q_scale)?, shared(&m_scale)?, shared(&table)?);
                let arm = |op: Op, heads: usize, dim: usize, positioned: bool, bound: f32| Shape {
                    op, heads: heads as u32, dim: dim as u32, taps: 1, rows: 0, cols: u32::from(positioned), epsilon: 0.0, bound,
                };

                // Test A: sliding window ring wrap (view=1024, pos=1020..1028 -> crosses 1024!) & global 4K (view=4128, pos=4092..4100).
                for &(is_ring, view, window, first_pos, positioned, label) in &[
                    (true, 1024usize, 1024usize, 1020usize, true, "ring_wrap_1024"),
                    (false, 4128usize, 4128usize, 4092usize, false, "global_4096"),
                ] {
                    let max_m = 16usize;
                    let q: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 307) * 3.0).collect();
                    let m: Vec<f32> = (0..max_m * heads * FW_REL_DIM).map(|i| input(i + 311) * 4.0).collect();
                    let sink: Vec<f32> = (0..max_m * heads).map(|i| input(i + 313) * 1.5).collect();
                    let gate: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 317) * 1.5).collect();
                    let pos_f: Vec<f32> = (0..max_m).map(|i| 1.15 + 0.05 * i as f32).collect();
                    // Base committed KV plane BEFORE first_pos (and full committed plane for M=1 reference).
                    let k_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 331)).collect();
                    let v_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 337)).collect();
                    let dk_u16: Vec<u16> = (0..max_m * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 900_001)).collect();
                    let dv_u16: Vec<u16> = (0..max_m * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 950_003)).collect();
                    let (mut k_base_u32, mut v_base_u32) = (pack_u32(&k_u16), pack_u32(&v_u16));
                    let (dk_u32, dv_u32) = (pack_u32(&dk_u16), pack_u32(&dv_u16));
                    let kv_row_words = kv_heads * (FW_HEAD_DIM / 2);
                    // For non-ring global plane, write all max_m rows in place; for ring, keep base plane untouched for delta test.
                    if !is_ring {
                        for r in 0..max_m {
                            let slot = first_pos + r;
                            k_base_u32[slot * kv_row_words..(slot + 1) * kv_row_words].copy_from_slice(&dk_u32[r * kv_row_words..(r + 1) * kv_row_words]);
                            v_base_u32[slot * kv_row_words..(slot + 1) * kv_row_words].copy_from_slice(&dv_u32[r * kv_row_words..(r + 1) * kv_row_words]);
                        }
                    }
                    let (q_dev, m_dev, s_dev, g_dev, p_dev) = (shared(&q)?, shared(&m)?, shared(&sink)?, shared(&gate)?, shared(&pos_f)?);
                    let (kb_dev, vb_dev) = (shared(&pack_f32(&k_base_u32))?, shared(&pack_f32(&v_base_u32))?);
                    let (dk_dev, dv_dev) = (shared(&pack_f32(&dk_u32))?, shared(&pack_f32(&dv_u32))?);

                    // Compute M=1 sequentially (updating ring slot per step just like plain decode!).
                    let mut k_seq = k_base_u32.clone();
                    let mut v_seq = v_base_u32.clone();
                    let mut seq_out = Vec::with_capacity(max_m * heads * FW_HEAD_DIM);
                    for r in 0..max_m {
                        let p = first_pos + r;
                        let live = (p + 1).min(window);
                        let slot = if is_ring { p % view } else { p };
                        if is_ring {
                            k_seq[slot * kv_row_words..(slot + 1) * kv_row_words].copy_from_slice(&dk_u32[r * kv_row_words..(r + 1) * kv_row_words]);
                            v_seq[slot * kv_row_words..(slot + 1) * kv_row_words].copy_from_slice(&dv_u32[r * kv_row_words..(r + 1) * kv_row_words]);
                        }
                        let ks_dev = shared(&pack_f32(&k_seq))?;
                        let vs_dev = shared(&pack_f32(&v_seq))?;
                        let st_dev = shared(&[live as f32, slot as f32])?;
                        let out1 = poisoned(b, heads * FW_HEAD_DIM)?;
                        let p_slice = if positioned { Some(p_dev.resident_range(r..r + 1)?) } else { None };
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: p_slice.as_ref(), query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_split_into(
                            1, heads, kv_heads, view, extent, scale, is_ring, p as u64, &[(p as u64, live)],
                            &q_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &ks_dev, &vs_dev, None,
                            &m_dev.resident_range(r * heads * FW_REL_DIM..(r + 1) * heads * FW_REL_DIM)?,
                            &t_dev, &s_dev.resident_range(r * heads..(r + 1) * heads)?,
                            &g_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &st_dev, clips, &[], &mut out1.resident_range(GUARD..GUARD + heads * FW_HEAD_DIM)?,
                        )?;
                        b.synchronize()?;
                        let got = b.read_vector_values(&out1)?;
                        seq_out.extend_from_slice(&got[GUARD..GUARD + heads * FW_HEAD_DIM]);
                        // Check row 0 accuracy against CPU reference!
                        if r == 0 {
                            let fw = if positioned { &pos_f[0..1] } else { &[] };
                            let qc = block_rows::reference(arm(Op::HeadClip, heads, FW_HEAD_DIM, positioned, query_bound), 1, 0, [&q[0..heads * FW_HEAD_DIM], fw, &[], &[]], &q_scale)?;
                            let mc = block_rows::reference(arm(Op::HeadClip, heads, FW_REL_DIM, positioned, moved_bound), 1, 0, [&m[0..heads * FW_REL_DIM], fw, &[], &[]], &m_scale)?;
                            let want = flash_window::flash_window_reference(
                                1, heads, kv_heads, view, extent, scale, &qc, &k_seq, &v_seq, &mc, &table,
                                &sink[0..heads], &gate[0..heads * FW_HEAD_DIM], &[live as f32, slot as f32],
                            );
                            let mut max_abs = 0.0f32;
                            for i in 0..want.len() {
                                let d = (got[GUARD + i] - want[i]).abs();
                                if d > max_abs { max_abs = d; }
                                if !got[GUARD + i].is_finite() || d > 0.01f32.max(want[i].abs() * 0.02) {
                                    return Err(format!("flash_window_split {label} M=1 accuracy fail at {i}: got={} want={}", got[GUARD + i], want[i]));
                                }
                            }
                            println!("FLASH WINDOW SPLIT {label} M=1 REF PASS max_abs={max_abs:.6}");
                        }
                    }

                    // Verify M=2, M=4 (SPLIT16), M=8, M=12, M=16 (SPLIT32, 1..2 Q CTAs) are 100% BIT-EXACT with sequential M=1 (and delta == committed ring)!
                    for m_rows in [2usize, 4, 8, 12, 16] {
                        let rpl: Vec<(u64, usize)> = (0..m_rows).map(|r| { let p = first_pos + r; (p as u64, (p + 1).min(window)) }).collect();
                        let st: Vec<f32> = rpl.iter().flat_map(|&(p, l)| [l as f32, (if is_ring { p as usize % view } else { p as usize }) as f32]).collect();
                        let st_dev = shared(&st)?;
                        let n = m_rows * heads * FW_HEAD_DIM;
                        let out = poisoned(b, n)?;
                        let p_slice = if positioned { Some(p_dev.resident_range(0..m_rows)?) } else { None };
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: p_slice.as_ref(), query_bound, moved_bound, written_keys: None, written_values: None };
                        let dk_slice = dk_dev.resident_range(0..m_rows * kv_row_words)?;
                        let dv_slice = dv_dev.resident_range(0..m_rows * kv_row_words)?;
                        let delta = if is_ring { Some((&dk_slice, &dv_slice)) } else { None };
                        b.flash_window_split_into(
                            m_rows, heads, kv_heads, view, extent, scale, is_ring, first_pos as u64, &rpl,
                            &q_dev.resident_range(0..n)?, &kb_dev, &vb_dev, delta,
                            &m_dev.resident_range(0..m_rows * heads * FW_REL_DIM)?, &t_dev,
                            &s_dev.resident_range(0..m_rows * heads)?, &g_dev.resident_range(0..n)?,
                            &st_dev, clips, &[], &mut out.resident_range(GUARD..GUARD + n)?,
                        )?;
                        b.synchronize()?;
                        let got = b.read_vector_values(&out)?;
                        for (side, slice) in [("before", &got[..GUARD]), ("after", &got[GUARD + n..])] {
                            if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                                return Err(format!("flash_window_split {label} M={m_rows}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                            }
                        }
                        if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != seq_out[i].to_bits()) {
                            return Err(format!("flash_window_split {label} M={m_rows} vs M=1 BIT-EXACT mismatch at {i}: got={:#x} ({}) seq={:#x} ({})",
                                got[GUARD + i].to_bits(), got[GUARD + i], seq_out[i].to_bits(), seq_out[i]));
                        }
                        println!("FLASH WINDOW SPLIT {label} M={m_rows} == M=1 BIT-EXACT PASS words={n} delta={is_ring}");
                        total_words += n;
                    }
                }

                // Test B: C8 multi-chat planes == C1 solo (mixed positions across 0..4097)!
                {
                    let (chats, view) = (8usize, 4128usize);
                    let chat_pos = [0u64, 17, 127, 256, 1023, 1024, 2500, 4097];
                    let q: Vec<f32> = (0..chats * heads * FW_HEAD_DIM).map(|i| input(i + 401) * 3.0).collect();
                    let m: Vec<f32> = (0..chats * heads * FW_REL_DIM).map(|i| input(i + 409) * 4.0).collect();
                    let sink: Vec<f32> = (0..chats * heads).map(|i| input(i + 419) * 1.5).collect();
                    let gate: Vec<f32> = (0..chats * heads * FW_HEAD_DIM).map(|i| input(i + 421) * 1.5).collect();
                    let k0 = shared(&pack_f32(&pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 431)).collect::<Vec<_>>())))?;
                    let v0 = shared(&pack_f32(&pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 433)).collect::<Vec<_>>())))?;
                    let (q_dev, m_dev, s_dev, g_dev) = (shared(&q)?, shared(&m)?, shared(&sink)?, shared(&gate)?);
                    let mut c1_out = Vec::with_capacity(chats * heads * FW_HEAD_DIM);
                    for (c, &p) in chat_pos.iter().enumerate() {
                        let (kp, vp) = if c % 2 == 0 { (&k0, &v0) } else { (&v0, &k0) };
                        let live = (p + 1) as usize;
                        let st_dev = shared(&[live as f32, p as f32])?;
                        let out = poisoned(b, heads * FW_HEAD_DIM)?;
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_split_into(
                            1, heads, kv_heads, view, extent, scale, false, p, &[(p, live)],
                            &q_dev.resident_range(c * heads * FW_HEAD_DIM..(c + 1) * heads * FW_HEAD_DIM)?,
                            kp, vp, None,
                            &m_dev.resident_range(c * heads * FW_REL_DIM..(c + 1) * heads * FW_REL_DIM)?,
                            &t_dev, &s_dev.resident_range(c * heads..(c + 1) * heads)?,
                            &g_dev.resident_range(c * heads * FW_HEAD_DIM..(c + 1) * heads * FW_HEAD_DIM)?,
                            &st_dev, clips, &[], &mut out.resident_range(GUARD..GUARD + heads * FW_HEAD_DIM)?,
                        )?;
                        b.synchronize()?;
                        c1_out.extend_from_slice(&b.read_vector_values(&out)?[GUARD..GUARD + heads * FW_HEAD_DIM]);
                    }
                    let rpl: Vec<(u64, usize)> = chat_pos.iter().map(|&p| (p, (p + 1) as usize)).collect();
                    let st: Vec<f32> = rpl.iter().flat_map(|&(p, l)| [l as f32, p as f32]).collect();
                    let st_dev = shared(&st)?;
                    let planes: Vec<(&_, &_)> = (0..chats).map(|c| if c % 2 == 0 { (&k0, &v0) } else { (&v0, &k0) }).collect();
                    let n = chats * heads * FW_HEAD_DIM;
                    let out = poisoned(b, n)?;
                    let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                    b.flash_window_split_into(
                        chats, heads, kv_heads, view, extent, scale, false, 0, &rpl,
                        &q_dev, &k0, &v0, None, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev, clips, &planes,
                        &mut out.resident_range(GUARD..GUARD + n)?,
                    )?;
                    b.synchronize()?;
                    let got = b.read_vector_values(&out)?;
                    if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != c1_out[i].to_bits()) {
                        return Err(format!("flash_window_split C8 vs C1 mismatch at {i}: got={:#x} c1={:#x}", got[GUARD + i].to_bits(), c1_out[i].to_bits()));
                    }
                    println!("FLASH WINDOW SPLIT C8 == C1 BIT-EXACT PASS words={n}");
                    total_words += n;
                }

                // Test C: C32 grouped multi-chat verify (32 chats x 1..=4 contiguous rows = 112 rows, both ring wrap & global 2K)
                // in ONE launch == sequential C1 M=1 BIT-EXACT + guard poisons!
                for &(is_ring, view, window, label) in &[
                    (true, 1024usize, 1024usize, "c32_grouped_ring"),
                    (false, 2080usize, 2080usize, "c32_grouped_global"),
                ] {
                    let chats = 32usize;
                    let counts: Vec<usize> = (0..chats).map(|c| if c < 24 { 4 } else { (c % 4) + 1 }).collect();
                    let total_rows: usize = counts.iter().sum();
                    let starts: Vec<u64> = (0..chats).map(|c| {
                        if is_ring {
                            match c % 5 { 0 => 15, 1 => 95, 2 => 510, 3 => 1022, _ => 1400 + (c as u64 * 17) }
                        } else {
                            match c % 5 { 0 => 0, 1 => 95, 2 => 255, 3 => 1023, _ => 2040 + (c as u64) }
                        }
                    }).collect();
                    let q: Vec<f32> = (0..total_rows * heads * FW_HEAD_DIM).map(|i| input(i + 503) * 3.0).collect();
                    let m: Vec<f32> = (0..total_rows * heads * FW_REL_DIM).map(|i| input(i + 509) * 4.0).collect();
                    let sink: Vec<f32> = (0..total_rows * heads).map(|i| input(i + 521) * 1.5).collect();
                    let gate: Vec<f32> = (0..total_rows * heads * FW_HEAD_DIM).map(|i| input(i + 523) * 1.5).collect();
                    let k0 = shared(&pack_f32(&pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 541)).collect::<Vec<_>>())))?;
                    let v0 = shared(&pack_f32(&pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 547)).collect::<Vec<_>>())))?;
                    let (q_dev, m_dev, s_dev, g_dev) = (shared(&q)?, shared(&m)?, shared(&sink)?, shared(&gate)?);
                    let mut c1_out = Vec::with_capacity(total_rows * heads * FW_HEAD_DIM);
                    let mut rpl = Vec::with_capacity(total_rows);
                    let mut st = Vec::with_capacity(2 * total_rows);
                    let mut planes: Vec<(&_, &_)> = Vec::with_capacity(total_rows);
                    let mut row_off = 0usize;
                    for c in 0..chats {
                        let (kp, vp) = if c % 2 == 0 { (&k0, &v0) } else { (&v0, &k0) };
                        for r in 0..counts[c] {
                            let p = starts[c] + r as u64;
                            let live = (p as usize + 1).min(window);
                            let slot = if is_ring { p as usize % view } else { p as usize };
                            rpl.push((p, live));
                            st.extend([live as f32, slot as f32]);
                            planes.push((kp, vp));
                            let st1 = shared(&[live as f32, slot as f32])?;
                            let out1 = poisoned(b, heads * FW_HEAD_DIM)?;
                            let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                            b.flash_window_split_into(
                                1, heads, kv_heads, view, extent, scale, is_ring, p, &[(p, live)],
                                &q_dev.resident_range(row_off * heads * FW_HEAD_DIM..(row_off + 1) * heads * FW_HEAD_DIM)?,
                                kp, vp, None,
                                &m_dev.resident_range(row_off * heads * FW_REL_DIM..(row_off + 1) * heads * FW_REL_DIM)?,
                                &t_dev, &s_dev.resident_range(row_off * heads..(row_off + 1) * heads)?,
                                &g_dev.resident_range(row_off * heads * FW_HEAD_DIM..(row_off + 1) * heads * FW_HEAD_DIM)?,
                                &st1, clips, &[], &mut out1.resident_range(GUARD..GUARD + heads * FW_HEAD_DIM)?,
                            )?;
                            b.synchronize()?;
                            c1_out.extend_from_slice(&b.read_vector_values(&out1)?[GUARD..GUARD + heads * FW_HEAD_DIM]);
                            row_off += 1;
                        }
                    }
                    let st_dev = shared(&st)?;
                    let n = total_rows * heads * FW_HEAD_DIM;
                    let out = poisoned(b, n)?;
                    let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                    b.flash_window_split_into(
                        total_rows, heads, kv_heads, view, extent, scale, is_ring, 0, &rpl,
                        &q_dev, &k0, &v0, None, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev, clips, &planes,
                        &mut out.resident_range(GUARD..GUARD + n)?,
                    )?;
                    b.synchronize()?;
                    let got = b.read_vector_values(&out)?;
                    for (side, slice) in [("before", &got[..GUARD]), ("after", &got[GUARD + n..])] {
                        if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                            return Err(format!("flash_window_split {label}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                        }
                    }
                    if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != c1_out[i].to_bits()) {
                        return Err(format!("flash_window_split {label} vs C1 mismatch at {i}: got={:#x} c1={:#x}", got[GUARD + i].to_bits(), c1_out[i].to_bits()));
                    }
                    println!("FLASH WINDOW SPLIT {label} rows={total_rows} == C1 BIT-EXACT PASS words={n}");
                    total_words += n;
                }

                // Test D: multi-split persistent CTA loop (`num_splits = 66`, `splits_per_cta = 2`) for both SPLIT16 (M=1,4)
                // and SPLIT32 (M=12): verifies persistent `s_local` loop, next-split prefetch, and guard poisons!
                {
                    let (view, first_pos, max_m) = (16_672usize, 16_636usize, 12usize);
                    let kv_row_words = kv_heads * (FW_HEAD_DIM / 2);
                    let q: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 701) * 3.0).collect();
                    let m: Vec<f32> = (0..max_m * heads * FW_REL_DIM).map(|i| input(i + 709) * 4.0).collect();
                    let sink: Vec<f32> = (0..max_m * heads).map(|i| input(i + 719) * 1.5).collect();
                    let gate: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 727) * 1.5).collect();
                    let k_u32 = pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 733)).collect::<Vec<_>>());
                    let v_u32 = pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 739)).collect::<Vec<_>>());
                    let (q_dev, m_dev, s_dev, g_dev) = (shared(&q)?, shared(&m)?, shared(&sink)?, shared(&gate)?);
                    let (k_dev, v_dev) = (shared(&pack_f32(&k_u32))?, shared(&pack_f32(&v_u32))?);
                    let _ = kv_row_words;
                    let mut seq_out = Vec::with_capacity(max_m * heads * FW_HEAD_DIM);
                    for r in 0..max_m {
                        let p = first_pos + r;
                        let live = p + 1;
                        let st1 = shared(&[live as f32, p as f32])?;
                        let out1 = poisoned(b, heads * FW_HEAD_DIM)?;
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_split_into(
                            1, heads, kv_heads, view, extent, scale, false, p as u64, &[(p as u64, live)],
                            &q_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &k_dev, &v_dev, None,
                            &m_dev.resident_range(r * heads * FW_REL_DIM..(r + 1) * heads * FW_REL_DIM)?,
                            &t_dev, &s_dev.resident_range(r * heads..(r + 1) * heads)?,
                            &g_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &st1, clips, &[], &mut out1.resident_range(GUARD..GUARD + heads * FW_HEAD_DIM)?,
                        )?;
                        b.synchronize()?;
                        seq_out.extend_from_slice(&b.read_vector_values(&out1)?[GUARD..GUARD + heads * FW_HEAD_DIM]);
                    }
                    for m_rows in [4usize, 12] {
                        let rpl: Vec<(u64, usize)> = (0..m_rows).map(|r| { let p = first_pos + r; (p as u64, p + 1) }).collect();
                        let st: Vec<f32> = rpl.iter().flat_map(|&(p, l)| [l as f32, p as f32]).collect();
                        let st_dev = shared(&st)?;
                        let n = m_rows * heads * FW_HEAD_DIM;
                        let out = poisoned(b, n)?;
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_split_into(
                            m_rows, heads, kv_heads, view, extent, scale, false, first_pos as u64, &rpl,
                            &q_dev.resident_range(0..n)?, &k_dev, &v_dev, None,
                            &m_dev.resident_range(0..m_rows * heads * FW_REL_DIM)?, &t_dev,
                            &s_dev.resident_range(0..m_rows * heads)?, &g_dev.resident_range(0..n)?,
                            &st_dev, clips, &[], &mut out.resident_range(GUARD..GUARD + n)?,
                        )?;
                        b.synchronize()?;
                        let got = b.read_vector_values(&out)?;
                        for (side, slice) in [("before", &got[..GUARD]), ("after", &got[GUARD + n..])] {
                            if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                                return Err(format!("flash_window_split global_16640 M={m_rows}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                            }
                        }
                        if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != seq_out[i].to_bits()) {
                            return Err(format!("flash_window_split global_16640 M={m_rows} vs M=1 mismatch at {i}: got={:#x} seq={:#x}", got[GUARD + i].to_bits(), seq_out[i].to_bits()));
                        }
                        println!("FLASH WINDOW SPLIT global_16640_spc2 M={m_rows} == M=1 BIT-EXACT PASS words={n}");
                        total_words += n;
                    }
                }

                // Test E: L2-resident wave-aligned banded split-reduce (`view = 49_200`, `M = 16`, `num_splits = 192` ->
                // `36.6 MiB > 16 MiB` -> 3 L2 bands of 72 + 72 + 48 splits, exercising first/middle/last state handoff!):
                {
                    let (view, first_pos, max_m) = (49_200usize, 49_136usize, 16usize);
                    let q: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 809) * 3.0).collect();
                    let m: Vec<f32> = (0..max_m * heads * FW_REL_DIM).map(|i| input(i + 811) * 4.0).collect();
                    let sink: Vec<f32> = (0..max_m * heads).map(|i| input(i + 821) * 1.5).collect();
                    let gate: Vec<f32> = (0..max_m * heads * FW_HEAD_DIM).map(|i| input(i + 823) * 1.5).collect();
                    let k_u32 = pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 827)).collect::<Vec<_>>());
                    let v_u32 = pack_u32(&(0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 829)).collect::<Vec<_>>());
                    let (q_dev, m_dev, s_dev, g_dev) = (shared(&q)?, shared(&m)?, shared(&sink)?, shared(&gate)?);
                    let (k_dev, v_dev) = (shared(&pack_f32(&k_u32))?, shared(&pack_f32(&v_u32))?);
                    let mut seq_out = Vec::with_capacity(max_m * heads * FW_HEAD_DIM);
                    for r in 0..max_m {
                        let p = first_pos + r;
                        let live = p + 1;
                        let st1 = shared(&[live as f32, p as f32])?;
                        let out1 = poisoned(b, heads * FW_HEAD_DIM)?;
                        let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_split_into(
                            1, heads, kv_heads, view, extent, scale, false, p as u64, &[(p as u64, live)],
                            &q_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &k_dev, &v_dev, None,
                            &m_dev.resident_range(r * heads * FW_REL_DIM..(r + 1) * heads * FW_REL_DIM)?,
                            &t_dev, &s_dev.resident_range(r * heads..(r + 1) * heads)?,
                            &g_dev.resident_range(r * heads * FW_HEAD_DIM..(r + 1) * heads * FW_HEAD_DIM)?,
                            &st1, clips, &[], &mut out1.resident_range(GUARD..GUARD + heads * FW_HEAD_DIM)?,
                        )?;
                        b.synchronize()?;
                        seq_out.extend_from_slice(&b.read_vector_values(&out1)?[GUARD..GUARD + heads * FW_HEAD_DIM]);
                    }
                    let rpl: Vec<(u64, usize)> = (0..max_m).map(|r| { let p = first_pos + r; (p as u64, p + 1) }).collect();
                    let st: Vec<f32> = rpl.iter().flat_map(|&(p, l)| [l as f32, p as f32]).collect();
                    let st_dev = shared(&st)?;
                    let n = max_m * heads * FW_HEAD_DIM;
                    let out = poisoned(b, n)?;
                    let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: None, query_bound, moved_bound, written_keys: None, written_values: None };
                    b.flash_window_split_into(
                        max_m, heads, kv_heads, view, extent, scale, false, first_pos as u64, &rpl,
                        &q_dev, &k_dev, &v_dev, None, &m_dev, &t_dev, &s_dev, &g_dev,
                        &st_dev, clips, &[], &mut out.resident_range(GUARD..GUARD + n)?,
                    )?;
                    b.synchronize()?;
                    let got = b.read_vector_values(&out)?;
                    for (side, slice) in [("before", &got[..GUARD]), ("after", &got[GUARD + n..])] {
                        if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                            return Err(format!("flash_window_split global_49152_banded: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                        }
                    }
                    if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != seq_out[i].to_bits()) {
                        return Err(format!("flash_window_split global_49152_banded M=16 vs M=1 mismatch at {i}: got={:#x} seq={:#x}", got[GUARD + i].to_bits(), seq_out[i].to_bits()));
                    }
                    println!("FLASH WINDOW SPLIT global_49152_banded M=16 == M=1 BIT-EXACT PASS words={n}");
                    total_words += n;
                }
            }
    Ok(total_words)
}
