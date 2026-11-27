//! Additional device bit-exact + guard-word checks for `few-check`:
//! - Strided M=2..=16 picked MoE (`Kind::PlainPackedRows`, NVFP4/MXFP8/unpacked)
//! - Fused `fast_softcap_argmax_into` (`STREAM_OP_SOFTCAP_ARGMAX`)
//! - Direct per-chat ring pointer `fast_short_conv_multi_rings_into`
//! - Multi-chat direct KV finish (`NORM_MODE_KV_DIRECT`) and direct ShortConv (`NORM_MODE_SHORT_CONV_DIRECT`)
use spark_gpu_probe::{driver::{Backend, DeviceVector}, resident_launch::Kind};

const GUARD: usize = 64;
const POISON: u32 = 0x7fc0_dead;

fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }

fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> {
    b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared()
}

pub fn run(b: &Backend<'_>, recorded: &dyn Fn(Kind) -> u128) -> Result<(), String> {
    // Strided M=2..=16 picked MoE (`Kind::PlainPackedRows` word 0 != 0): single-pass count=1..=16,
    // NVFP4 tagged panels (one-shot & multi-slab X), 3-stage MXFP8 (native & custom table), and
    // unpacked NVFP4 fallback, gated & ungated, bit-exact against unstrided `Kind::PlainPackedPicked`.
    {
        use spark_gpu_probe::packed_product::layout;
        let upload = |bytes: &[u8]| -> Result<DeviceVector<'_>, String> {
            let mut vec = b.vector(bytes.len() / 4)?;
            b.write_vector_bytes(&mut vec, bytes)?;
            vec.into_resident_shared()
        };
        let mut strided_cases = 0usize;
        for &(rows, cols, wide, tagged, custom_tbl, m_rows, repeat, gate) in &[
            (128usize, 256usize, false, true, false, 2usize, 8usize, true),
            (128, 1152, false, true, false, 8, 1, false),
            (128, 1152, false, true, false, 12, 1, false),
            (128, 2304, false, true, false, 16, 8, true),
            (70, 256, true, false, false, 2, 8, true),
            (128, 256, true, false, false, 12, 1, false),
            (64, 256, true, false, true, 4, 1, false),
            (70, 128, false, false, false, 10, 1, true),
        ] {
            let shape = layout(rows, cols, wide);
            let experts = 3usize;
            let mut raw_bytes = vec![0u8; experts * shape.bytes];
            let mut panel_bytes = Vec::with_capacity(if tagged { experts * shape.bytes } else { 0 });
            for (expert, unit) in raw_bytes.chunks_exact_mut(shape.bytes).enumerate() {
                for (i, v) in unit[..shape.scales].iter_mut().enumerate() {
                    *v = (i.wrapping_mul(37) + (i >> 3) + expert * 61) as u8;
                }
                for (i, v) in unit[shape.scales..shape.table].iter_mut().enumerate() {
                    let mut s = if wide {
                        (120 + (i + expert * 7) % 16) as u8
                    } else {
                        ((i.wrapping_mul(19) + expert * 47) & 0xff) as u8
                    };
                    if !wide && (s & 0x7f) == 0x7f { s ^= 1; }
                    *v = s;
                }
                if wide {
                    for (i, word) in unit[shape.table..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let val = if custom_tbl {
                            ((i as f32) - 128.0) / 64.0
                        } else if (i & 0x7f) == 0x7f {
                            0.0f32
                        } else {
                            let exp = (i >> 3) & 15;
                            let man = i & 7;
                            let mag = if exp == 0 {
                                (man as f32 / 8.0) * 2.0f32.powi(-6)
                            } else {
                                (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
                            };
                            if (i & 0x80) != 0 { -mag } else { mag }
                        };
                        word.copy_from_slice(&val.to_le_bytes());
                    }
                } else {
                    for (i, word) in unit[shape.table..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let v = spark_gpu_probe::bf16::rounded(input(i + expert * 23 + 11));
                        word.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                    }
                }
                if tagged {
                    panel_bytes.extend(spark_gpu_probe::decode_panels::pack_expert_tagged(unit, rows, cols));
                }
            }
            let stack = upload(if tagged { &panel_bytes } else { &raw_bytes })?;
            let picks_per_row = 8usize;
            let id_stride = 16usize;
            let picks = m_rows * picks_per_row;
            let mut flat_route = Vec::with_capacity(picks);
            let mut strided_route = vec![99.0f32; m_rows * id_stride];
            for r in 0..m_rows {
                for p in 0..picks_per_row {
                    let id = if p == 0 { 0.0f32 } else { ((r + p) % experts) as f32 };
                    flat_route.push(id);
                    strided_route[r * id_stride + p] = id;
                }
            }
            let flat_ids = b.upload_vector(&flat_route)?.into_resident_shared()?;
            let strided_ids = b.upload_vector(&strided_route)?.into_resident_shared()?;
            let in_rows = picks / repeat;
            let x_host: Vec<f32> = (0..in_rows * cols).map(|k| input(k + rows + m_rows * 13)).collect();
            let x_strided = b.upload_vector(&x_host)?.into_resident_shared()?;
            let mut x_expanded_host = Vec::with_capacity(picks * cols);
            for j in 0..picks {
                let r = j / repeat;
                x_expanded_host.extend_from_slice(&x_host[r * cols..(r + 1) * cols]);
            }
            let x_expanded = b.upload_vector(&x_expanded_host)?.into_resident_shared()?;
            let caps = gate.then_some([4.0, 25.0, 0.25, 0.04]);
            let writes = if gate { picks * (rows / 2) } else { picks * rows };

            let want_buf = poisoned(b, picks * rows)?;
            let mut want_slice = want_buf.resident_range(GUARD..GUARD + picks * rows)?;
            b.project_packed_picked_step_into(rows, cols, wide, gate, &x_expanded, false, &stack, experts, &flat_ids, &mut want_slice, caps, false, None)?;
            let want = b.read_vector_values(&want_buf)?;

            let got_buf = poisoned(b, picks * rows)?;
            let mut got_slice = got_buf.resident_range(GUARD..GUARD + picks * rows)?;
            let before = recorded(Kind::PlainPackedRows);
            b.project_packed_picked_strided_into(rows, cols, wide, gate, &x_strided, repeat, &stack, experts, &strided_ids, picks, picks_per_row, id_stride, &mut got_slice, caps, false, None)?;
            if recorded(Kind::PlainPackedRows) != before + 1 {
                return Err(format!("strided picked MoE did not route to PlainPackedRows (rows={rows} cols={cols} wide={wide} m={m_rows})"));
            }
            let got = b.read_vector_values(&got_buf)?;
            if got[..GUARD].iter().chain(&got[GUARD + writes..]).any(|w| w.to_bits() != POISON) {
                return Err(format!("strided picked MoE changed a guard word (rows={rows} cols={cols} wide={wide} m={m_rows})"));
            }
            if let Some(at) = want[GUARD..GUARD + writes].iter().zip(&got[GUARD..GUARD + writes]).position(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(format!("strided picked MoE mismatch at word {at} (rows={rows} cols={cols} wide={wide} tagged={tagged} m={m_rows} gate={gate})"));
            }
            strided_cases += 1;
        }
        println!("FEW CHECK STRIDED PICKED MOE cases={strided_cases} M=2..16 NVFP4+MXFP8+unpacked gated+ungated bit-exact and guards unchanged");
    }
    // Fused `fast_softcap_argmax_into` (STREAM_OP_SOFTCAP_ARGMAX) across single-stage (rows=64) and two-stage
    // (rows=3, 16) grids, odd/even unaligned column counts, ties, saturation ties, +/-Inf, and NaN rejection.
    {
        let bound = 30.0f32;
        let eps = (1.0f64 / f64::from(bound)) as f32;
        for &(rows, cols) in &[(3usize, 5003usize), (16, 10_002), (64, 4_099)] {
            let mut host = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for j in 0..cols {
                    host[r * cols + j] = (((r * 131 + j * 17) % 1009) as f32 / 64.0) - 8.0;
                }
                match r % 5 {
                    0 => { host[r * cols + 7] = 18.5; host[r * cols + cols - 3] = 18.5; }
                    1 => { host[r * cols + 11] = 350.0; host[r * cols + cols - 2] = 900.0; }
                    2 => { host[r * cols + 13] = f32::INFINITY; host[r * cols + 19] = 400.0; }
                    3 => { host[r * cols + 5] = f32::NEG_INFINITY; host[r * cols + cols - 1] = -200.0; }
                    _ => { host[r * cols + cols / 2] = 24.25; }
                }
            }
            let dev_in = b.upload_vector(&host)?;
            let out_buf = poisoned(b, rows * 3)?;
            let mut slice = out_buf.resident_range(GUARD..GUARD + rows * 3)?;
            b.fast_softcap_argmax_into(rows, cols, bound, eps, &dev_in, &mut slice)?;
            let back = b.read_vector_values(&out_buf)?;
            if back[..GUARD].iter().chain(&back[GUARD + rows * 3..]).any(|w| w.to_bits() != POISON) {
                return Err(format!("softcap_argmax rows={rows} cols={cols} changed a guard word"));
            }
            for r in 0..rows {
                let (mut wb, mut wid) = (f32::NEG_INFINITY, u32::MAX);
                for j in 0..cols {
                    let c = spark_gpu_probe::fixed_math::tanh(spark_gpu_probe::fixed_math::rn(host[r * cols + j]) * eps) * bound;
                    if c > wb || (c == wb && (j as u32) < wid) { wb = c; wid = j as u32; }
                }
                let got = &back[GUARD + r * 3..GUARD + r * 3 + 3];
                if got[0].to_bits() != wb.to_bits() || got[1] != wid as f32 || got[2] != 1.0 {
                    return Err(format!("softcap_argmax rows={rows} cols={cols} r={r} differs: got={got:?} want=({wb}, {wid}, 1.0)"));
                }
            }
        }
        for &(rows, cols) in &[(2usize, 8195usize), (48, 257)] {
            let mut host = vec![1.0f32; rows * cols];
            host[cols + cols / 2] = f32::NAN;
            let dev_in = b.upload_vector(&host)?;
            let mut out = b.vector(rows * 3)?;
            b.fast_softcap_argmax_into(rows, cols, bound, eps, &dev_in, &mut out)?;
            let got = b.read_vector_values(&out)?;
            if got[2] != 1.0 || got[5] != 0.0 {
                return Err(format!("softcap_argmax NaN flag mismatch rows={rows} cols={cols}: {:?}", &got[..6]));
            }
        }
        println!("FEW CHECK SOFTCAP ARGMAX cases=5 single/two-stage, saturation ties, +/-Inf, NaN exact and guards");
    }
    // Direct per-chat ring pointer `fast_short_conv_multi_rings_into` vs joined `fast_short_conv_multi_into`
    // covering `rows = 32` (8 chats), `rows = 128` (32 chats), and `rows = 256` (32 chats x 8 rows).
    {
        use spark_gpu_probe::fast_ops::ShortConvMulti;
        for &(chats, per_chat, taps, width) in &[(8usize, 4usize, 4usize, 256usize), (32, 4, 4, 256), (32, 8, 4, 256)] {
            let rows = chats * per_chat;
            let (mut mask, mut s_lo, mut s_hi) = ([0u64; 4], 0u64, 0u64);
            for c in 0..chats {
                let off = c * per_chat;
                mask[off >> 6] |= 1u64 << (off & 63);
                let slot = (c % taps) as u64;
                if c < 32 { s_lo |= slot << (2 * c); } else { s_hi |= slot << (2 * (c - 32)); }
            }
            let multi = ShortConvMulti::encode(rows, taps, mask, s_lo, s_hi).ok_or("multi encode failed")?;
            let inp_h: Vec<f32> = (0..rows * width).map(|i| input(i + 17)).collect();
            let add_h: Vec<f32> = (0..rows * width).map(|i| input(i + 53)).collect();
            let fix_h: Vec<f32> = (0..width * taps).map(|i| input(i + 97) * 0.25).collect();
            let mut rings_h = Vec::with_capacity(chats * taps * width);
            let mut ring_devs = Vec::with_capacity(chats);
            for c in 0..chats {
                let rv: Vec<f32> = (0..taps * width).map(|i| input(i + c * 313 + 7)).collect();
                rings_h.extend_from_slice(&rv);
                ring_devs.push(b.upload_vector(&rv)?.into_resident_shared()?);
            }
            let (inp_d, add_d, fix_d, joined_d) = (
                b.upload_vector(&inp_h)?, b.upload_vector(&add_h)?,
                b.upload_vector(&fix_h)?, b.upload_vector(&rings_h)?,
            );
            let (mut o_ref, mut e_ref) = (b.vector(rows * width)?, b.vector(rows * taps * width)?);
            b.fast_short_conv_multi_into(rows, 1, width, taps, 6, 0.4472136, &inp_d, None, Some(&add_d), &fix_d, &joined_d, &mut o_ref, &mut e_ref, multi)?;
            let (o_buf, e_buf) = (poisoned(b, rows * width)?, poisoned(b, rows * taps * width)?);
            let ring_refs: Vec<&DeviceVector<'_>> = ring_devs.iter().collect();
            b.fast_short_conv_multi_rings_into(
                rows, 1, width, taps, 6, 0.4472136, &inp_d, None, Some(&add_d), &fix_d, &ring_refs,
                &mut o_buf.resident_range(GUARD..GUARD + rows * width)?,
                &mut e_buf.resident_range(GUARD..GUARD + rows * taps * width)?,
                multi,
            )?;
            let (want_o, got_o) = (b.read_vector_values(&o_ref)?, b.read_vector_values(&o_buf)?);
            let (want_e, got_e) = (b.read_vector_values(&e_ref)?, b.read_vector_values(&e_buf)?);
            if got_o[..GUARD].iter().chain(&got_o[GUARD + rows * width..]).any(|w| w.to_bits() != POISON)
                || got_e[..GUARD].iter().chain(&got_e[GUARD + rows * taps * width..]).any(|w| w.to_bits() != POISON)
            {
                return Err(format!("short_conv_multi_rings chats={chats} changed a guard word"));
            }
            if got_o[GUARD..GUARD + rows * width].iter().zip(&want_o).any(|(a, b)| a.to_bits() != b.to_bits())
                || got_e[GUARD..GUARD + rows * taps * width].iter().zip(&want_e).any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("short_conv_multi_rings chats={chats} differs from joined multi"));
            }
        }
        println!("FEW CHECK SHORT CONV MULTI RINGS cases=3 chats=8,32 bit-exact and guards unchanged");
    }
    // Multi-chat direct KV finish (`NORM_MODE_KV_DIRECT`) vs `fast_kv_finish_pair_into`,
    // and direct ShortConv (`NORM_MODE_SHORT_CONV_DIRECT`) vs `fast_short_conv_multi_rings_into`.
    {
        use spark_gpu_probe::fast_ops::ShortConvMulti;
        for &(rows, kv_heads, taps, has_made) in &[(8usize, 8usize, 4usize, true), (16, 4, 3, false)] {
            let width = kv_heads * spark_gpu_probe::fast_ops::KV_FINISH_HEAD_DIM;
            let row_words = width / 2;
            let plane_rows = 32usize;
            let ki = b.upload_vector(&(0..rows * width).map(|i| input(i + 11)).collect::<Vec<_>>())?;
            let vi = b.upload_vector(&(0..rows * width).map(|i| input(i + 29)).collect::<Vec<_>>())?;
            let kf = b.upload_vector(&(0..width * taps).map(|i| input(i + 47) * 0.25).collect::<Vec<_>>())?;
            let vf = b.upload_vector(&(0..width * taps).map(|i| input(i + 61) * 0.25).collect::<Vec<_>>())?;
            let km = has_made.then(|| b.upload_vector(&(0..rows * kv_heads * taps).map(|i| input(i + 73) * 0.1).collect::<Vec<_>>())).transpose()?;
            let vm = has_made.then(|| b.upload_vector(&(0..rows * kv_heads * taps).map(|i| input(i + 89) * 0.1).collect::<Vec<_>>())).transpose()?;
            let mut k_rings = Vec::with_capacity(rows);
            let mut v_rings = Vec::with_capacity(rows);
            let mut k_planes = Vec::with_capacity(rows);
            let mut v_planes = Vec::with_capacity(rows);
            let mut slots = Vec::with_capacity(rows);
            for r in 0..rows {
                k_rings.push(b.upload_vector(&(0..taps * width).map(|i| input(i + r * 211 + 3)).collect::<Vec<_>>())?.into_resident_shared()?);
                v_rings.push(b.upload_vector(&(0..taps * width).map(|i| input(i + r * 223 + 7)).collect::<Vec<_>>())?.into_resident_shared()?);
                k_planes.push(b.upload_vector(&vec![f32::from_bits(POISON); plane_rows * row_words])?);
                v_planes.push(b.upload_vector(&vec![f32::from_bits(POISON); plane_rows * row_words])?);
                slots.push((r % taps, (r * 3 + 1) % plane_rows));
            }
            let (mut ref_kp, mut ref_vp) = (b.vector(rows * row_words)?, b.vector(rows * row_words)?);
            let (mut ref_ke, mut ref_ve) = (b.vector(rows * taps * width)?, b.vector(rows * taps * width)?);
            let pair_rings: Vec<_> = (0..rows).map(|r| (&k_rings[r], &v_rings[r], slots[r].0)).collect();
            b.fast_kv_finish_pair_into(
                rows, kv_heads, taps, slots[0].0, 1.5, 1e-6, 0.2, 0.2,
                &ki, &k_rings[0], &kf, km.as_ref(), &mut ref_kp, Some(&mut ref_ke),
                &vi, &v_rings[0], &vf, vm.as_ref(), &mut ref_vp, Some(&mut ref_ve),
                &pair_rings,
            )?;
            let (ke_buf, ve_buf) = (poisoned(b, rows * taps * width)?, poisoned(b, rows * taps * width)?);
            let chats: Vec<_> = (0..rows).map(|r| (&k_rings[r], &v_rings[r], slots[r].0, &k_planes[r], &v_planes[r], slots[r].1)).collect();
            b.fast_kv_finish_direct_into(
                rows, kv_heads, taps, plane_rows, 1.5, 1e-6, 0.2, 0.2,
                &ki, &kf, km.as_ref(), &mut ke_buf.resident_range(GUARD..GUARD + rows * taps * width)?,
                &vi, &vf, vm.as_ref(), &mut ve_buf.resident_range(GUARD..GUARD + rows * taps * width)?,
                &chats,
            )?;
            let (want_kp, want_vp) = (b.read_vector_values(&ref_kp)?, b.read_vector_values(&ref_vp)?);
            let (want_ke, want_ve) = (b.read_vector_values(&ref_ke)?, b.read_vector_values(&ref_ve)?);
            let (got_ke, got_ve) = (b.read_vector_values(&ke_buf)?, b.read_vector_values(&ve_buf)?);
            if got_ke[..GUARD].iter().chain(&got_ke[GUARD + want_ke.len()..]).any(|w| w.to_bits() != POISON)
                || got_ve[..GUARD].iter().chain(&got_ve[GUARD + want_ve.len()..]).any(|w| w.to_bits() != POISON)
                || got_ke[GUARD..GUARD + want_ke.len()].iter().zip(&want_ke).any(|(a, w)| a.to_bits() != w.to_bits())
                || got_ve[GUARD..GUARD + want_ve.len()].iter().zip(&want_ve).any(|(a, w)| a.to_bits() != w.to_bits())
            {
                return Err(format!("kv_finish_direct every mismatch rows={rows} taps={taps}"));
            }
            for r in 0..rows {
                let ps = slots[r].1;
                for (plane, want) in [(&k_planes[r], &want_kp), (&v_planes[r], &want_vp)] {
                    let back = b.read_vector_values(plane)?;
                    for p in 0..plane_rows {
                        let sl = &back[p * row_words..(p + 1) * row_words];
                        if p == ps {
                            if sl.iter().zip(&want[r * row_words..(r + 1) * row_words]).any(|(a, w)| a.to_bits() != w.to_bits()) {
                                return Err(format!("kv_finish_direct plane row mismatch rows={rows} r={r}"));
                            }
                        } else if sl.iter().any(|w| w.to_bits() != POISON) {
                            return Err(format!("kv_finish_direct overwrote untouched plane row rows={rows} r={r} p={p}"));
                        }
                    }
                }
            }
        }
        for &(rows, heads, dim, taps, form) in &[(8usize, 1usize, 2304usize, 4usize, 6u32), (12, 8, 128, 3, 1u32)] {
            let width = heads * dim;
            let (mut mask, mut s_lo) = ([0u64; 4], 0u64);
            let mut ring_devs = Vec::with_capacity(rows);
            let mut slots = Vec::with_capacity(rows);
            for r in 0..rows {
                mask[0] |= 1u64 << r;
                let sl = r % taps;
                s_lo |= (sl as u64) << (2 * r);
                slots.push(sl);
                ring_devs.push(b.upload_vector(&(0..taps * width).map(|i| input(i + r * 307 + 5)).collect::<Vec<_>>())?.into_resident_shared()?);
            }
            let multi = ShortConvMulti::encode(rows, taps, mask, s_lo, 0).ok_or("direct sc multi encode failed")?;
            let inp = b.upload_vector(&(0..rows * width).map(|i| input(i + 19)).collect::<Vec<_>>())?;
            let tok = ((form & 1) != 0).then(|| b.upload_vector(&(0..rows * heads * taps).map(|i| input(i + 41) * 0.2).collect::<Vec<_>>())).transpose()?;
            let add = ((form & 2) != 0).then(|| b.upload_vector(&(0..rows * width).map(|i| input(i + 67)).collect::<Vec<_>>())).transpose()?;
            let fix = b.upload_vector(&(0..width * taps).map(|i| input(i + 83) * 0.25).collect::<Vec<_>>())?;
            let (mut ref_o, mut ref_e) = (b.vector(rows * width)?, b.vector(rows * taps * width)?);
            let ring_refs: Vec<_> = ring_devs.iter().collect();
            b.fast_short_conv_multi_rings_into(rows, heads, dim, taps, form, 0.35, &inp, tok.as_ref(), add.as_ref(), &fix, &ring_refs, &mut ref_o, &mut ref_e, multi)?;
            let (o_buf, e_buf) = (poisoned(b, rows * width)?, poisoned(b, rows * taps * width)?);
            let direct_rings: Vec<_> = (0..rows).map(|r| (&ring_devs[r], slots[r])).collect();
            b.fast_short_conv_direct_into(
                rows, heads, dim, taps, form, 0.35, &inp, tok.as_ref(), add.as_ref(), &fix,
                &mut o_buf.resident_range(GUARD..GUARD + rows * width)?,
                &mut e_buf.resident_range(GUARD..GUARD + rows * taps * width)?,
                &direct_rings,
            )?;
            let (want_o, got_o) = (b.read_vector_values(&ref_o)?, b.read_vector_values(&o_buf)?);
            let (want_e, got_e) = (b.read_vector_values(&ref_e)?, b.read_vector_values(&e_buf)?);
            if got_o[..GUARD].iter().chain(&got_o[GUARD + want_o.len()..]).any(|w| w.to_bits() != POISON)
                || got_e[..GUARD].iter().chain(&got_e[GUARD + want_e.len()..]).any(|w| w.to_bits() != POISON)
                || got_o[GUARD..GUARD + want_o.len()].iter().zip(&want_o).any(|(a, w)| a.to_bits() != w.to_bits())
                || got_e[GUARD..GUARD + want_e.len()].iter().zip(&want_e).any(|(a, w)| a.to_bits() != w.to_bits())
            {
                return Err(format!("short_conv_direct mismatch rows={rows} taps={taps} form={form}"));
            }
        }
        println!("FEW CHECK KV_DIRECT + SHORT_CONV_DIRECT cases=4 bit-exact and guards unchanged");
    }
    // Vectorized Gate, RopeBf16, PackBf16, ScatterKvBatch, and FlashAttnBatch + FlashAttnMergeBatch bit-exactness.
    {
        use spark_gpu_probe::draft_ops::DraftOp::*;
        let rn = spark_gpu_probe::bf16::rounded;
        for n in [1usize, 3, 7, 16, 19, 128] {
            let g_vals = (0..n).map(|k| input(k + 11) * 2.0).collect::<Vec<_>>();
            let u_vals = (0..n).map(|k| input(k + 37) * 2.0).collect::<Vec<_>>();
            let (gd, ud) = (b.upload_vector(&g_vals)?, b.upload_vector(&u_vals)?);
            let got = b.read_vector_values(&b.draft_lanes(Gate(n), [Some(&gd), Some(&ud), None])?)?;
            for k in 0..n {
                let want = rn(rn(g_vals[k] * spark_gpu_probe::fixed_math::sigmoid(g_vals[k])) * u_vals[k]);
                if got[k].to_bits() != want.to_bits() {
                    return Err(format!("Gate vectorized n={n} k={k} differs"));
                }
            }
        }
        for (rows, heads, width) in [(1usize, 8usize, 128usize), (7, 8, 128), (19, 8, 128)] {
            let x_vals = (0..rows * heads * width).map(|k| rn(input(k + 53) * 2.0)).collect::<Vec<_>>();
            let t_vals = (0..rows * width).map(|k| input(k + 89)).collect::<Vec<_>>();
            let (xd, td) = (b.upload_vector(&x_vals)?, b.upload_vector(&t_vals)?);
            let rope_fp = b.draft_lanes(Rope { rows, heads, width }, [Some(&xd), Some(&td), None])?;
            let rope_bf = b.draft_lanes(RopeBf16 { rows, heads, width }, [Some(&xd), Some(&td), None])?;
            let packed_fp = b.draft_lanes(PackBf16(rows * heads * width), [Some(&rope_fp), None, None])?;
            let fp_vals = b.read_vector_values(&rope_fp)?;
            let bf_words: Vec<u32> = b.read_vector_values(&rope_bf)?.into_iter().map(f32::to_bits).collect();
            let pk_words: Vec<u32> = b.read_vector_values(&packed_fp)?.into_iter().map(f32::to_bits).collect();
            if bf_words != pk_words {
                return Err(format!("RopeBf16 != PackBf16(Rope) at rows={rows}"));
            }
            for p in 0..bf_words.len() {
                let want = (fp_vals[2 * p].to_bits() >> 16) | (fp_vals[2 * p + 1].to_bits() & 0xffff_0000);
                if bf_words[p] != want {
                    return Err(format!("RopeBf16 pair {p} differs at rows={rows}"));
                }
            }
        }
        for n in [6usize, 34] {
            let v = (0..n).map(|k| rn(input(k + 131))).collect::<Vec<_>>();
            let vd = b.upload_vector(&v)?;
            let got: Vec<u32> = b.read_vector_values(&b.draft_lanes(PackBf16(n), [Some(&vd), None, None])?)?.into_iter().map(f32::to_bits).collect();
            for p in 0..n / 2 {
                let want = (v[2 * p].to_bits() >> 16) | (v[2 * p + 1].to_bits() & 0xffff_0000);
                if got[p] != want {
                    return Err(format!("PackBf16 tail n={n} p={p} differs"));
                }
            }
        }
        for v_is_fp32 in [true, false] {
            let kv_u32_stride = if v_is_fp32 { 8 * 64 } else { 8 * 128 };
            let counts = [1usize, 3, 2, 5];
            let total_rows: usize = counts.iter().sum();
            let k_host = (0..total_rows * kv_u32_stride).map(|i| {
                if v_is_fp32 {
                    let lo = rn(input(2 * i + 7)).to_bits() >> 16;
                    let hi = rn(input(2 * i + 8)).to_bits() & 0xffff_0000;
                    f32::from_bits(lo | hi)
                } else {
                    rn(input(i + 7))
                }
            }).collect::<Vec<_>>();
            let v_host = (0..total_rows * kv_u32_stride * if v_is_fp32 { 2 } else { 1 }).map(|i| rn(input(i + 211))).collect::<Vec<_>>();
            let (ks, vs) = (b.upload_vector(&k_host)?, b.upload_vector(&v_host)?);
            let mut planes: Vec<(DeviceVector<'_>, DeviceVector<'_>, usize, usize, usize)> = Vec::new();
            let mut off = 0usize;
            for (bi, &cnt) in counts.iter().enumerate() {
                let dst_row = bi + 1;
                let cap_rows = dst_row + cnt + 2;
                let dk = b.upload_vector(&vec![f32::from_bits(POISON); cap_rows * kv_u32_stride])?;
                let dv = b.upload_vector(&vec![f32::from_bits(POISON); cap_rows * kv_u32_stride])?;
                planes.push((dk, dv, dst_row, off, cnt));
                off += cnt;
            }
            let mut refs: Vec<(&mut DeviceVector<'_>, &mut DeviceVector<'_>, usize, usize, usize)> =
                planes.iter_mut().map(|(dk, dv, dr, sr, c)| (dk, dv, *dr, *sr, *c)).collect();
            if !b.draft_scatter_kv_batch(&ks, &vs, total_rows, kv_u32_stride, v_is_fp32, &mut refs)? {
                return Err(format!("ScatterKvBatch returned false for writable planes v_is_fp32={v_is_fp32}"));
            }
            for (bi, (dk, dv, dst_row, src_row, cnt)) in planes.iter().enumerate() {
                let got_k: Vec<u32> = b.read_vector_values(dk)?.into_iter().map(f32::to_bits).collect();
                let got_v: Vec<u32> = b.read_vector_values(dv)?.into_iter().map(f32::to_bits).collect();
                let prefix = dst_row * kv_u32_stride;
                let span = cnt * kv_u32_stride;
                if got_k[..prefix].iter().chain(&got_k[prefix + span..]).any(|&w| w != POISON)
                    || got_v[..prefix].iter().chain(&got_v[prefix + span..]).any(|&w| w != POISON)
                {
                    return Err(format!("ScatterKvBatch overwrote guard row bi={bi} v_is_fp32={v_is_fp32}"));
                }
                for r in 0..*cnt {
                    for w in 0..kv_u32_stride {
                        let want_k = k_host[(src_row + r) * kv_u32_stride + w].to_bits();
                        let want_v = if v_is_fp32 {
                            let base = ((src_row + r) * kv_u32_stride + w) * 2;
                            (v_host[base].to_bits() >> 16) | (v_host[base + 1].to_bits() & 0xffff_0000)
                        } else {
                            v_host[(src_row + r) * kv_u32_stride + w].to_bits()
                        };
                        if got_k[(dst_row + r) * kv_u32_stride + w] != want_k || got_v[(dst_row + r) * kv_u32_stride + w] != want_v {
                            return Err(format!("ScatterKvBatch mismatch bi={bi} r={r} w={w} v_is_fp32={v_is_fp32}"));
                        }
                    }
                }
            }
        }
        let pack_bf16_host = |vals: &[f32]| -> Vec<f32> {
            vals.as_chunks::<2>().0.iter().map(|p| f32::from_bits((p[0].to_bits() >> 16) | (p[1].to_bits() & 0xffff_0000))).collect()
        };
        for (name, heads, kv, q_rows, new_rows, kv_bf16, k_new_bf16, causal_base, row_off, spec) in [
            ("gqa3_bf16_mixed", 24usize, 8usize, 7usize, 7usize, true, true, 0usize, 0usize, &[(0usize, 0usize, 1usize), (3, 31, 1), (5, 95, 8), (2, 260, 16)][..]),
            ("gqa3_fp32_s1", 24, 8, 3, 3, false, false, 0, 0, &[(0, 12, 1), (4, 20, 1)][..]),
            ("nongqa3_bf16", 16, 8, 4, 4, true, true, 0, 0, &[(1, 70, 8), (0, 260, 16)][..]),
            ("gqa3_bf16_causal", 24, 8, 15, 15, true, true, 135, 1, &[(2, 128, 8), (0, 128, 8)][..]),
        ] {
            let batch = spec.len();
            let q_host = (0..batch * q_rows * heads * 128).map(|i| rn(input(i + 17) * 0.5)).collect::<Vec<_>>();
            let kn_fp = (0..batch * new_rows * kv * 128).map(|i| rn(input(i + 41) * 0.5)).collect::<Vec<_>>();
            let vn_fp = (0..batch * new_rows * kv * 128).map(|i| rn(input(i + 67) * 0.5)).collect::<Vec<_>>();
            let q_dev = b.upload_vector(&q_host)?;
            let kn_dev = if k_new_bf16 { b.upload_vector(&pack_bf16_host(&kn_fp))? } else { b.upload_vector(&kn_fp)? };
            let vn_dev = b.upload_vector(&vn_fp)?;
            let mut cks = Vec::with_capacity(batch);
            let mut cvs = Vec::with_capacity(batch);
            let mut want_all = Vec::with_capacity(q_host.len());
            for (bi, &(rel_start, hist_keys, splits_b)) in spec.iter().enumerate() {
                let plane_rows = (rel_start + hist_keys).max(1);
                let hk_fp = (0..plane_rows * kv * 128).map(|i| rn(input(i + bi * 7919 + 101) * 0.5)).collect::<Vec<_>>();
                let hv_fp = (0..plane_rows * kv * 128).map(|i| rn(input(i + bi * 7927 + 149) * 0.5)).collect::<Vec<_>>();
                cks.push(if kv_bf16 { b.upload_vector(&pack_bf16_host(&hk_fp))? } else { b.upload_vector(&hk_fp)? });
                cvs.push(if kv_bf16 { b.upload_vector(&pack_bf16_host(&hv_fp))? } else { b.upload_vector(&hv_fp)? });
                let mut k_cat = hk_fp[rel_start * kv * 128..(rel_start + hist_keys) * kv * 128].to_vec();
                k_cat.extend_from_slice(&kn_fp[bi * new_rows * kv * 128..(bi + 1) * new_rows * kv * 128]);
                let mut v_cat = hv_fp[rel_start * kv * 128..(rel_start + hist_keys) * kv * 128].to_vec();
                v_cat.extend_from_slice(&vn_fp[bi * new_rows * kv * 128..(bi + 1) * new_rows * kv * 128]);
                let q_sub = b.upload_vector(&q_host[bi * q_rows * heads * 128..(bi + 1) * q_rows * heads * 128])?;
                let (kc, vc) = (b.upload_vector(&k_cat)?, b.upload_vector(&v_cat)?);
                let keys = hist_keys + new_rows;
                let part = b.draft_lanes(FlashAttn { rows: q_rows, heads, kv, width: 128, keys, splits: splits_b, causal_base, row_off }, [Some(&q_sub), Some(&kc), Some(&vc)])?;
                let out = if splits_b > 1 { b.draft_lanes(FlashAttnMerge { rows: q_rows, heads, width: 128, splits: splits_b }, [Some(&part), None, None])? } else { part };
                want_all.extend(b.read_vector_values(&out)?);
            }
            let chat_descs: Vec<_> = spec.iter().enumerate().map(|(bi, &(rs, hk, sp))| (&cks[bi], &cvs[bi], rs, hk, sp)).collect();
            let splits_vec: Vec<usize> = spec.iter().map(|&(_, _, sp)| sp).collect();
            let max_splits = splits_vec.iter().copied().max().unwrap();
            let part = b.draft_flash_attn_batch(&q_dev, &kn_dev, &vn_dev, &chat_descs, heads, kv, 128, q_rows, new_rows, kv_bf16, k_new_bf16, causal_base, row_off)?;
            let got_dev = if max_splits > 1 { b.draft_flash_attn_merge_batch(&part, &splits_vec, heads, q_rows, max_splits)? } else { part };
            let got_all = b.read_vector_values(&got_dev)?;
            if let Some(at) = got_all.iter().zip(&want_all).position(|(a, w)| a.to_bits() != w.to_bits()) {
                return Err(format!("FlashAttnBatch {name} bit mismatch at word {at}: got={} want={}", got_all[at], want_all[at]));
            }
        }
        println!("FEW CHECK DRAFT BATCH BF16 KV + FLASH_ATTN_BATCH + SCATTER_KV PASS");
    }
    // Pipelined wide `Panel` / `PanelBf16` across odd/even chunk counts (cols=64, 192, 256, 384) and multi-wave
    // large matrix (`rows=8256 > PANEL_SPLIT_ROWS_MAX`, `inputs=9, 32, 96, 224`) bit-exact vs `Panel` + `PackBf16`.
    {
        use spark_gpu_probe::{decode_panels, draft_ops::DraftOp::*, packed_product};
        let rn = spark_gpu_probe::bf16::rounded;
        for &(rows, cols, inputs) in &[
            (79usize, 64usize, 16usize),
            (79, 192, 32),
            (79, 256, 96),
            (8256, 192, 16),
            (8256, 256, 96),
            (8256, 256, 224),
        ] {
            let phys = rows.next_multiple_of(64);
            let layout = packed_product::layout(phys, cols, false);
            let mut bytes = vec![0u8; layout.bytes];
            for r in 0..rows {
                for c in (0..cols).step_by(2) {
                    let at = packed_product::code_copy(r, c / 64, 0, cols, false) + (c % 64) / 2;
                    let c0 = (r * 3 + c / 3 + c * r) % 8;
                    let c1 = (r * 3 + (c + 1) / 3 + (c + 1) * r) % 8;
                    bytes[at] = (c0 | (c1 << 4)) as u8;
                }
            }
            bytes[layout.scales..layout.table].fill(56);
            let panel = decode_panels::pack_draft(&bytes, phys, cols, 1.0);
            let mut weight = b.vector(panel.len() / 4)?;
            b.write_vector_bytes(&mut weight, &panel)?;
            let x = (0..inputs * cols).map(|at| [0.0, 1.5, -3.0, 4.0, 6.0][(at / cols + 3 * (at % cols)) % 5]).collect::<Vec<_>>();
            let x_dev = b.upload_vector(&x)?;
            let packed = b.draft_lanes(PackActivations(x.len()), [Some(&x_dev), None, None])?;
            let fp_out = b.draft_lanes(Panel { rows, columns: cols, inputs }, [Some(&weight), Some(&packed), None])?;
            let bf_out = b.draft_lanes(PanelBf16 { rows, columns: cols, inputs }, [Some(&weight), Some(&packed), None])?;
            let fp_vals = b.read_vector_values(&fp_out)?;
            let bf_words: Vec<u32> = b.read_vector_values(&bf_out)?.into_iter().map(f32::to_bits).collect();
            let row_words = rows.div_ceil(2);
            for t in 0..inputs {
                for r in (0..rows).step_by(2) {
                    let lo = rn(fp_vals[t * rows + r]).to_bits() >> 16;
                    let hi = if r + 1 < rows { rn(fp_vals[t * rows + r + 1]).to_bits() & 0xffff_0000 } else { 0 };
                    let got = bf_words[t * row_words + r / 2];
                    if got != (lo | hi) {
                        return Err(format!("PanelBf16 {rows}x{cols} inputs={inputs} t={t} r={r} mismatch: got={got:#x} want={:#x}", lo | hi));
                    }
                }
            }
        }
        println!("FEW CHECK PANEL_BF16 + PIPELINED WIDE PANEL PASS");
    }
    // `draft_fast_norm_into` both recorded and unrecorded (paused resident queue, exercising `pad_commands = 4`
    // around `warm_command!`) bit-exact vs `causal_operation(Op::RoundedNorm)` with guard words.
    {
        use spark_gpu_probe::causal_ops::{Op, Shape};
        for &(rows, dim, prefill) in &[(32usize, 2304usize, false), (128, 2304, true), (1024, 128, true)] {
            let x_host = (0..rows * dim).map(|k| input(k + rows + dim) * 2.0).collect::<Vec<_>>();
            let w_host = (0..dim).map(|k| 1.0 + input(k + 73) * 0.25).collect::<Vec<_>>();
            let (xd, wd) = (b.upload_vector(&x_host)?, b.upload_vector(&w_host)?);
            let shape = Shape {
                op: Op::RoundedNorm,
                heads: rows as u32,
                dim: dim as u32,
                taps: 1,
                rows: 0,
                cols: 0,
                epsilon: 1e-6,
                bound: 0.0,
            };
            let want = b.read_vector_values(&b.causal_operation(shape, [Some(&xd), None, None, None], Some(&wd))?)?;
            // Recorded path
            let rec_buf = poisoned(b, rows * dim)?;
            let mut rec_slice = rec_buf.resident_range(GUARD..GUARD + rows * dim)?;
            b.draft_fast_norm_into(rows, dim, 1e-6, prefill, &xd, &wd, &mut rec_slice)?;
            let rec = b.read_vector_values(&rec_buf)?;
            // Unrecorded path (pauses resident launches just like `rows_unrecorded(true)` in `mtp_batch.rs`)
            let unrec_buf = poisoned(b, rows * dim)?;
            let mut unrec_slice = unrec_buf.resident_range(GUARD..GUARD + rows * dim)?;
            let prev = b.pause_resident_launches(true)?;
            let res = b.draft_fast_norm_into(rows, dim, 1e-6, prefill, &xd, &wd, &mut unrec_slice);
            let unpause = b.pause_resident_launches(prev);
            res?;
            unpause?;
            let unrec = b.read_vector_values(&unrec_buf)?;
            for (tag, got) in [("recorded", &rec), ("unrecorded", &unrec)] {
                if got[..GUARD].iter().chain(&got[GUARD + rows * dim..]).any(|w| w.to_bits() != POISON) {
                    return Err(format!("draft_fast_norm_into ({tag}) changed a guard word rows={rows} dim={dim}"));
                }
                if let Some(at) = got[GUARD..GUARD + rows * dim].iter().zip(&want).position(|(a, w)| a.to_bits() != w.to_bits()) {
                    return Err(format!("draft_fast_norm_into ({tag}) mismatch at {at} rows={rows} dim={dim} prefill={prefill}"));
                }
            }
        }
        println!("FEW CHECK DRAFT_FAST_NORM RECORDED+UNRECORDED PASS");
    }
    Ok(())
}
