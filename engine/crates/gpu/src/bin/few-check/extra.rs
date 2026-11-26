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
    Ok(())
}
