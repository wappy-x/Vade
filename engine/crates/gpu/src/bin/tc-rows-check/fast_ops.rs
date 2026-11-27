//! Device bit-exact checks for `fast_mix_rows`, `fast_mix_rows_normed`, and `fast_gate_blend_norm`.
use spark_gpu_probe::driver::Backend;

fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }

pub fn run(b: &Backend<'_>) -> Result<usize, String> {
    let mut total_words = 0usize;
    for &(rows, asked, kept, dim, transpose, keep_normed) in &[
        (1usize, 1usize, 2usize, 2304usize, false, true),
        (1, 3, 5, 2304, false, true),
        (4, 1, 5, 2304, false, false),
        (4, 3, 4, 2304, true, true),
        (4, 48, 5, 2304, true, false),
        (8, 1, 8, 2560, false, true),
        (8, 3, 3, 1024, true, true),
        (16, 1, 4, 2304, false, false),
        (24, 3, 5, 2304, true, true),
        (32, 1, 4, 2304, false, true),
        (32, 3, 4, 2304, true, false),
        (64, 3, 4, 2304, true, true),
        (64, 3, 2, 2304, true, false),
        (64, 2, 5, 2304, true, true),
    ] {
        let scale = (1.0f64 / (dim as f64).sqrt()) as f32;
        let eps = 1.0e-6f32;
        let q_host: Vec<f32> = (0..asked * dim).map(|i| input(i + rows * 17 + asked * 31)).collect();
        let list_host: Vec<Vec<f32>> = (0..kept - 1)
            .map(|e| (0..rows * dim).map(|i| input(i + e * 733 + rows * 53)).collect())
            .collect();
        let stream_host: Vec<f32> = (0..rows * dim).map(|i| input(i + kept * 911 + dim)).collect();
        let q_dev = b.upload_vector(&q_host)?.into_resident_shared()?;
        let list_dev: Vec<_> = list_host.iter().map(|v| b.upload_vector(v).and_then(|d| d.into_resident_shared())).collect::<Result<_, _>>()?;
        let stream_dev = b.upload_vector(&stream_host)?.into_resident_shared()?;
        let refs: Vec<_> = list_dev.iter().collect();
        let (got_normed, got_out) = b.fast_mix_rows(rows, dim, scale, eps, transpose, keep_normed, &q_dev, &refs, &stream_dev)?;
        // Reference: separate fast_norm (PlainNorm) + joined stack + fast_row_dots + fast_mix_blend.
        let mut ref_normed = b.vector(rows * dim)?;
        b.fast_norm_into(rows, dim, false, eps, false, &stream_dev, None, &mut ref_normed)?;
        let mut ref_joined = b.vector(kept * rows * dim)?;
        for (e, v) in list_dev.iter().enumerate() { b.copy_range(v, &mut ref_joined, 0, e * rows * dim, rows * dim)?; }
        b.copy_range(&ref_normed, &mut ref_joined, 0, (kept - 1) * rows * dim, rows * dim)?;
        let mut ref_scores = b.vector(rows * asked * kept)?;
        b.fast_row_dots_into(rows, asked, kept, dim, &ref_joined, &q_dev, &mut ref_scores)?;
        let mut ref_out = b.vector(rows * asked * dim)?;
        b.fast_mix_blend_into(rows, asked, kept, dim, scale, transpose, &ref_scores, &ref_joined, &mut ref_out)?;
        let ref_out = ref_out.into_resident_shared()?;
        b.synchronize()?;
        let got_o = b.read_vector_values(&got_out)?;
        let want_o = b.read_vector_values(&ref_out)?;
        if let Some(i) = (0..want_o.len()).find(|&i| got_o[i].to_bits() != want_o[i].to_bits()) {
            return Err(format!("mix_one r={rows} q={asked} k={kept} d={dim} t={transpose}: out word {i} got {:#x} want {:#x}", got_o[i].to_bits(), want_o[i].to_bits()));
        }
        if let Some(gn) = got_normed {
            let got_n = b.read_vector_values(&gn)?;
            let want_n = b.read_vector_values(&ref_normed)?;
            if let Some(i) = (0..want_n.len()).find(|&i| got_n[i].to_bits() != want_n[i].to_bits()) {
                return Err(format!("mix_one r={rows} q={asked} k={kept} d={dim}: normed word {i} got {:#x} want {:#x}", got_n[i].to_bits(), want_n[i].to_bits()));
            }
        }
        // Also check fused post-blend RoundedNorm (unweighted keep_mixed=true, and weighted skip_mixed=true).
        let post_eps = 1.0e-5f32;
        let (_, got_m2, got_p_unw) = b.fast_mix_rows_normed(rows, dim, scale, eps, post_eps, transpose, false, true, &q_dev, &refs, &stream_dev, None, false)?;
        let mut ref_p_unw = b.vector(rows * asked * dim)?;
        b.fast_norm_into(rows * asked, dim, true, post_eps, false, &ref_out, None, &mut ref_p_unw)?;
        let (gm2, gpu, wpu) = (b.read_vector_values(got_m2.as_ref().unwrap())?, b.read_vector_values(&got_p_unw)?, b.read_vector_values(&ref_p_unw)?);
        if let Some(i) = (0..want_o.len()).find(|&i| gm2[i].to_bits() != want_o[i].to_bits()) {
            return Err(format!("mix_one_normed unw mixed r={rows} q={asked} d={dim}: word {i}"));
        }
        if let Some(i) = (0..wpu.len()).find(|&i| gpu[i].to_bits() != wpu[i].to_bits()) {
            return Err(format!("mix_one_normed unw post r={rows} q={asked} d={dim}: word {i} got {:#x} want {:#x}", gpu[i].to_bits(), wpu[i].to_bits()));
        }
        if asked == 3 && transpose {
            let pw_host: Vec<f32> = (0..3 * dim).map(|i| input(i + 419) + 1.25).collect();
            let pw_dev = b.upload_vector(&pw_host)?.into_resident_shared()?;
            let (gn3, gm3, got_p_w) = b.fast_mix_rows_normed(rows, dim, scale, eps, post_eps, true, keep_normed, false, &q_dev, &refs, &stream_dev, Some(&pw_dev), true)?;
            if gm3.is_some() { return Err("mix_one_normed skip_mixed returned mixed".into()); }
            if let Some(gn3) = gn3 {
                let got_n = b.read_vector_values(&gn3)?;
                let want_n = b.read_vector_values(&ref_normed)?;
                if let Some(i) = (0..want_n.len()).find(|&i| got_n[i].to_bits() != want_n[i].to_bits()) {
                    return Err(format!("mix_one_normed per_q normed r={rows} d={dim}: word {i}"));
                }
            }
            let got_pw = b.read_vector_values(&got_p_w)?;
            for q in 0..3 {
                let wq = b.upload_vector(&pw_host[q * dim..(q + 1) * dim])?;
                let mut rq = b.vector(rows * dim)?;
                b.fast_norm_into(rows, dim, true, post_eps, false, &ref_out.resident_range(q * rows * dim..(q + 1) * rows * dim)?, Some(&wq), &mut rq)?;
                let want_q = b.read_vector_values(&rq)?;
                let got_q = &got_pw[q * rows * dim..(q + 1) * rows * dim];
                if let Some(i) = (0..want_q.len()).find(|&i| got_q[i].to_bits() != want_q[i].to_bits()) {
                    return Err(format!("mix_one_normed per_q q={q} r={rows} d={dim}: word {i} got {:#x} want {:#x}", got_q[i].to_bits(), want_q[i].to_bits()));
                }
            }
            if rows >= 48 {
                let (_, _, got_half) = b.fast_mix_rows_normed_half(rows, dim, scale, eps, post_eps, true, false, false, &q_dev, &refs, &stream_dev, Some(&pw_dev), true)?;
                let got_h = b.read_vector_values(&got_half)?;
                let want_h: Vec<f32> = got_pw.as_chunks::<2>().0.iter().map(|[lo, hi]| {
                    f32::from_bits((lo.to_bits() >> 16) | (hi.to_bits() & 0xffff_0000))
                }).collect();
                if let Some(i) = (0..want_h.len()).find(|&i| got_h[i].to_bits() != want_h[i].to_bits()) {
                    return Err(format!("mix_one_normed_half fused_q r={rows} d={dim}: word {i} got {:#x} want {:#x}", got_h[i].to_bits(), want_h[i].to_bits()));
                }
            }
        } else if asked == 1 {
            let pw_host: Vec<f32> = (0..dim).map(|i| input(i + 523) + 1.5).collect();
            let pw_dev = b.upload_vector(&pw_host)?.into_resident_shared()?;
            let (_, _, got_p_w) = b.fast_mix_rows_normed(rows, dim, scale, eps, post_eps, transpose, false, false, &q_dev, &refs, &stream_dev, Some(&pw_dev), false)?;
            let mut ref_p_w = b.vector(rows * dim)?;
            b.fast_norm_into(rows, dim, true, post_eps, false, &ref_out, Some(&pw_dev), &mut ref_p_w)?;
            let (gpw, wpw) = (b.read_vector_values(&got_p_w)?, b.read_vector_values(&ref_p_w)?);
            if let Some(i) = (0..wpw.len()).find(|&i| gpw[i].to_bits() != wpw[i].to_bits()) {
                return Err(format!("mix_one_normed shared_w r={rows} d={dim}: word {i} got {:#x} want {:#x}", gpw[i].to_bits(), wpw[i].to_bits()));
            }
        }
        println!("MIX ONE+NORM r={rows} q={asked} k={kept} d={dim} t={transpose} keep={keep_normed} BIT-EXACT PASS words={}", want_o.len());
        total_words += want_o.len();
    }
    // Fused GateBlend + weighted RoundedNorm (1 gate and 3 batched gates).
    for &(rows, dim, count) in &[(4usize, 2304usize, 1usize), (4, 2304, 3), (8, 2560, 3)] {
        let eps = 1.0e-5f32;
        let mut gs = Vec::new();
        let mut ks = Vec::new();
        let mut ms = Vec::new();
        let mut ws = Vec::new();
        for idx in 0..count {
            gs.push(b.upload_vector(&(0..rows * 2 * dim).map(|i| input(i + 101 * (idx + 1))).collect::<Vec<_>>())?);
            ks.push(b.upload_vector(&(0..rows * dim).map(|i| input(i + 211 * (idx + 1))).collect::<Vec<_>>())?);
            ms.push(b.upload_vector(&(0..rows * dim).map(|i| input(i + 307 * (idx + 1))).collect::<Vec<_>>())?);
            ws.push(b.upload_vector(&(0..dim).map(|i| input(i + 409 * (idx + 1)) + 1.5).collect::<Vec<_>>())?);
        }
        let entries: Vec<_> = (0..count).map(|i| (&gs[i], &ks[i], &ms[i], Some(&ws[i]))).collect();
        let outs = b.fast_gate_blend_norm(rows, dim, eps, &entries)?;
        for idx in 0..count {
            let mut blended = b.vector(rows * dim)?;
            b.fast_gate_blend_into(rows, dim, &gs[idx], &ks[idx], &ms[idx], &mut blended)?;
            let mut want_v = b.vector(rows * dim)?;
            b.fast_norm_into(rows, dim, true, eps, false, &blended, Some(&ws[idx]), &mut want_v)?;
            let (got, want) = (b.read_vector_values(&outs[idx])?, b.read_vector_values(&want_v)?);
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("gate_blend_norm r={rows} d={dim} g={idx}/{count}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
            total_words += want.len();
        }
        println!("GATE BLEND NORM r={rows} d={dim} gates={count} BIT-EXACT PASS words={}", count * rows * dim);
    }
    // 5f. Strided RouteFold (`fast_route_fold_strided_into`) against standard `fast_route_fold_into`.
    for &(rows, picks, dim) in &[(4usize, 8usize, 2304usize), (8, 8, 2304)] {
        let downs_host: Vec<f32> = (0..rows * picks * dim).map(|i| input(i + rows * 41 + 3)).collect();
        let w_host: Vec<f32> = (0..rows * picks).map(|i| 0.1 + (input(i + 19).abs() * 0.25)).collect();
        let mut packed_rw = vec![0.0f32; rows * 2 * picks];
        for r in 0..rows {
            for p in 0..picks {
                packed_rw[r * 2 * picks + p] = p as f32;
                packed_rw[r * 2 * picks + picks + p] = w_host[r * picks + p];
            }
        }
        let downs_dev = b.upload_vector(&downs_host)?.into_resident_shared()?;
        let w_dev = b.upload_vector(&w_host)?.into_resident_shared()?;
        let rw_dev = b.upload_vector(&packed_rw)?.into_resident_shared()?;
        let (mut got_dev, mut ref_dev) = (b.vector(rows * dim)?, b.vector(rows * dim)?);
        b.fast_route_fold_strided_into(rows, picks, dim, 2 * picks, picks, &rw_dev, &downs_dev, &mut got_dev)?;
        b.fast_route_fold_into(rows, picks, dim, &w_dev, &downs_dev, &mut ref_dev)?;
        b.synchronize()?;
        let (got, want) = (b.read_vector_values(&got_dev)?, b.read_vector_values(&ref_dev)?);
        if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            return Err(format!("route_fold_strided r={rows} p={picks}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
        }
        println!("ROUTE FOLD STRIDED r={rows} p={picks} d={dim} BIT-EXACT PASS words={}", want.len());
        total_words += want.len();
    }
    // 5g. Zero-copy `short_conv_win_into` with `ring + every`, paired `fast_kv_finish_pair_into`, and
    // fused `fast_route_fold_conv_into` compared bit-for-bit against separate launches.
    for &(rows, taps, first_slot, made) in &[(4usize, 4usize, 1usize, false), (4, 4, 3, true), (8, 4, 2, true)] {
        let (kv_heads, head_dim) = (8usize, 128usize);
        let width = kv_heads * head_dim;
        let scale = (1.0f64 / ((if made { 2 * taps + 1 } else { taps + 1 }) as f64).sqrt()) as f32;
        let (k_bound, v_eps) = (8.0f32, 1.0e-6f32);
        let k_in_h: Vec<f32> = (0..rows * width).map(|i| input(i + rows * 13 + 7) * 2.5).collect();
        let k_rng_h: Vec<f32> = (0..taps * width).map(|i| input(i + first_slot * 31 + 11) * 2.5).collect();
        let k_fix_h: Vec<f32> = (0..width * taps).map(|i| input(i + 23) * 0.4).collect();
        let k_mad_h: Vec<f32> = (0..rows * kv_heads * taps).map(|i| input(i + 37) * 0.3).collect();
        let v_in_h: Vec<f32> = (0..rows * width).map(|i| input(i + rows * 19 + 43) * 2.5).collect();
        let v_rng_h: Vec<f32> = (0..taps * width).map(|i| input(i + first_slot * 47 + 53) * 2.5).collect();
        let v_fix_h: Vec<f32> = (0..width * taps).map(|i| input(i + 61) * 0.4).collect();
        let v_mad_h: Vec<f32> = (0..rows * kv_heads * taps).map(|i| input(i + 71) * 0.3).collect();
        let mk_win = |rng: &[f32], inp: &[f32]| -> Vec<f32> {
            let mut w = Vec::with_capacity((rows + taps - 1) * width);
            for j in 0..taps - 1 {
                let s = (first_slot + 1 + j) % taps;
                w.extend_from_slice(&rng[s * width..(s + 1) * width]);
            }
            w.extend_from_slice(inp);
            w
        };
        let (k_win_h, v_win_h) = (mk_win(&k_rng_h, &k_in_h), mk_win(&v_rng_h, &v_in_h));
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let (k_in_d, k_rng_d, k_fix_d, k_mad_d, k_win_d) = (up(&k_in_h)?, up(&k_rng_h)?, up(&k_fix_h)?, up(&k_mad_h)?, up(&k_win_h)?);
        let (v_in_d, v_rng_d, v_fix_d, v_mad_d, v_win_d) = (up(&v_in_h)?, up(&v_rng_h)?, up(&v_fix_h)?, up(&v_mad_h)?, up(&v_win_h)?);
        let form = if made { 1u32 } else { 0u32 };
        let (mut sc_o1, mut sc_e1, mut sc_o2, mut sc_e2) = (b.vector(rows * width)?, b.vector(rows * taps * width)?, b.vector(rows * width)?, b.vector(rows * taps * width)?);
        b.fast_short_conv_win_into(rows, kv_heads, head_dim, taps, form, scale, &k_in_d, made.then_some(&k_mad_d), None, &k_fix_d, Some((&k_rng_d, first_slot)), &mut sc_o1, Some((&mut sc_e1, first_slot)))?;
        b.fast_short_conv_win_into(rows, kv_heads, head_dim, taps, form, scale, &k_win_d, made.then_some(&k_mad_d), None, &k_fix_d, None, &mut sc_o2, Some((&mut sc_e2, first_slot)))?;
        let (mut v_sc_o, mut v_sc_e) = (b.vector(rows * width)?, b.vector(rows * taps * width)?);
        b.fast_short_conv_win_into(rows, kv_heads, head_dim, taps, form, scale, &v_win_d, made.then_some(&v_mad_d), None, &v_fix_d, None, &mut v_sc_o, Some((&mut v_sc_e, first_slot)))?;
        let mut k_clip = b.vector(rows * width)?;
        let hc = spark_gpu_probe::causal_ops::Shape { op: spark_gpu_probe::causal_ops::Op::HeadClip, heads: kv_heads as u32, dim: head_dim as u32, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: k_bound };
        b.fast_head_clip_into(hc, rows, &sc_o2, None, None, &mut k_clip)?;
        let mut v_norm = b.vector(rows * width)?;
        b.fast_norm_into(rows * kv_heads, head_dim, true, v_eps, false, &v_sc_o, None, &mut v_norm)?;
        let (mut kp_got, mut ke_got, mut vp_got, mut ve_got) = (b.vector(rows * width / 2)?, b.vector(rows * taps * width)?, b.vector(rows * width / 2)?, b.vector(rows * taps * width)?);
        b.fast_kv_finish_pair_into(rows, kv_heads, taps, first_slot, k_bound, v_eps, scale, scale,
            &k_in_d, &k_rng_d, &k_fix_d, made.then_some(&k_mad_d), &mut kp_got, Some(&mut ke_got),
            &v_in_d, &v_rng_d, &v_fix_d, made.then_some(&v_mad_d), &mut vp_got, Some(&mut ve_got), &[])?;
        b.synchronize()?;
        let bf16 = |x: f32| { let bits = x.to_bits(); bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16 };
        let pack_ref = |vals: &[f32]| -> Vec<f32> {
            (0..vals.len() / 2).map(|i| f32::from_bits(bf16(vals[2 * i]) | (bf16(vals[2 * i + 1]) << 16))).collect()
        };
        for (tag, got, want) in [
            ("sc_out", b.read_vector_values(&sc_o1)?, b.read_vector_values(&sc_o2)?),
            ("sc_every", b.read_vector_values(&sc_e1)?, b.read_vector_values(&sc_e2)?),
            ("kv_kp", b.read_vector_values(&kp_got)?, pack_ref(&b.read_vector_values(&k_clip)?)),
            ("kv_ke", b.read_vector_values(&ke_got)?, b.read_vector_values(&sc_e2)?),
            ("kv_vp", b.read_vector_values(&vp_got)?, pack_ref(&b.read_vector_values(&v_norm)?)),
            ("kv_ve", b.read_vector_values(&ve_got)?, b.read_vector_values(&v_sc_e)?),
        ] {
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("kv_finish_pair {tag} r={rows} made={made}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
        }
        let chat_rings = (0..rows).map(|r| if r % 2 == 0 {
            (&k_rng_d, &v_rng_d, r % taps)
        } else { (&v_rng_d, &k_rng_d, r % taps) }).collect::<Vec<_>>();
        b.fast_kv_finish_pair_into(rows, kv_heads, taps, first_slot, k_bound, v_eps, scale, scale,
            &k_in_d, &k_rng_d, &k_fix_d, made.then_some(&k_mad_d), &mut kp_got, Some(&mut ke_got),
            &v_in_d, &v_rng_d, &v_fix_d, made.then_some(&v_mad_d), &mut vp_got, Some(&mut ve_got), &chat_rings)?;
        b.synchronize()?;
        for (row, &(kr, vr, slot)) in chat_rings.iter().enumerate() {
            let ki = up(&k_in_h[row * width..(row + 1) * width])?;
            let vi = up(&v_in_h[row * width..(row + 1) * width])?;
            let n = kv_heads * taps;
            let km = up(&k_mad_h[row * n..(row + 1) * n])?;
            let vm = up(&v_mad_h[row * n..(row + 1) * n])?;
            let (mut kp, mut ke, mut vp, mut ve) = (b.vector(width/2)?, b.vector(taps*width)?, b.vector(width/2)?, b.vector(taps*width)?);
            b.fast_kv_finish_pair_into(1, kv_heads, taps, slot, k_bound, v_eps, scale, scale,
                &ki, kr, &k_fix_d, made.then_some(&km), &mut kp, Some(&mut ke),
                &vi, vr, &v_fix_d, made.then_some(&vm), &mut vp, Some(&mut ve), &[])?;
            b.synchronize()?;
            for (one, many) in [(&kp, &kp_got), (&ke, &ke_got), (&vp, &vp_got), (&ve, &ve_got)] {
                let reference = b.read_vector_values(one)?;
                let all = b.read_vector_values(many)?;
                let actual = &all[row*one.len()..(row+1)*one.len()];
                if reference.iter().zip(actual).any(|(a,z)| a.to_bits()!=z.to_bits()) {
                    return Err(format!("independent KV finish differs at row {row} made={made}"));
                }
                total_words += one.len();
            }
        }
        println!("KV FINISH PAIR r={rows} first={first_slot} made={made} BIT-EXACT PASS words={}", rows * width * (taps * 2 + 1));
        total_words += rows * width * (taps * 2 + 1);
    }
    for &(rows, picks, dim, taps, first_slot) in &[(4usize, 8usize, 2304usize, 4usize, 1usize), (8, 8, 2304, 4, 3)] {
        let bound = (1.0f64 / ((taps + 1) as f64).sqrt()) as f32;
        let downs_h: Vec<f32> = (0..rows * picks * dim).map(|i| input(i + rows * 29 + 5)).collect();
        let route_h: Vec<f32> = (0..rows * 2 * picks).map(|i| if i % (2 * picks) < picks { (i % picks) as f32 } else { 0.1 + input(i + 7).abs() * 0.2 }).collect();
        let res_h: Vec<f32> = (0..rows * dim).map(|i| input(i + 97) * 2.0).collect();
        let fix_h: Vec<f32> = (0..dim * taps).map(|i| input(i + 131) * 0.4).collect();
        let rng_h: Vec<f32> = (0..taps * dim).map(|i| input(i + 157) * 1.5).collect();
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let (downs_d, route_d, res_d, fix_d, rng_d) = (up(&downs_h)?, up(&route_h)?, up(&res_h)?, up(&fix_h)?, up(&rng_h)?);
        let (mut got_o, mut got_e) = (b.vector(rows * dim)?, b.vector(rows * taps * dim)?);
        b.fast_route_fold_conv_into(rows, picks, dim, 2 * picks, picks, taps, first_slot, bound, &route_d, &downs_d, &res_d, &fix_d, &rng_d, &mut got_o, Some(&mut got_e), &[])?;
        let mut ref_fold = b.vector(rows * dim)?;
        b.fast_route_fold_strided_into(rows, picks, dim, 2 * picks, picks, &route_d, &downs_d, &mut ref_fold)?;
        let (mut ref_o, mut ref_e) = (b.vector(rows * dim)?, b.vector(rows * taps * dim)?);
        b.fast_short_conv_win_into(rows, 1, dim, taps, 6, bound, &ref_fold, None, Some(&res_d), &fix_d, Some((&rng_d, first_slot)), &mut ref_o, Some((&mut ref_e, first_slot)))?;
        b.synchronize()?;
        for (tag, got, want) in [("out", b.read_vector_values(&got_o)?, b.read_vector_values(&ref_o)?), ("every", b.read_vector_values(&got_e)?, b.read_vector_values(&ref_e)?)] {
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("route_fold_conv {tag} r={rows}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
        }
        let chats=(0..rows).map(|row|(if row%2==0 {&rng_d} else {&fix_d},row%taps)).collect::<Vec<_>>();
        b.fast_route_fold_conv_into(rows,picks,dim,2*picks,picks,taps,first_slot,bound,
            &route_d,&downs_d,&res_d,&fix_d,&rng_d,&mut got_o,Some(&mut got_e),&chats)?;
        b.synchronize()?;
        let actual=[b.read_vector_values(&got_o)?,b.read_vector_values(&got_e)?];
        for (row,&(ring,slot)) in chats.iter().enumerate() {
            let fold=up(&b.read_vector_values(&ref_fold)?[row*dim..(row+1)*dim])?;
            let residual=res_d.resident_range(row*dim..(row+1)*dim)?;
            let (mut out,mut every)=(b.vector(dim)?,b.vector(taps*dim)?);
            b.fast_short_conv_win_into(1,1,dim,taps,6,bound,&fold,None,Some(&residual),&fix_d,
                Some((ring,slot)),&mut out,Some((&mut every,slot)))?;
            b.synchronize()?;
            for (all, one) in actual.iter().zip([&out,&every]) {
                let want=b.read_vector_values(one)?;
                if all[row*one.len()..(row+1)*one.len()].iter().zip(want).any(|(a,z)|a.to_bits()!=z.to_bits()) {
                    return Err(format!("independent expert close differs at row {row}"));
                }
                total_words+=one.len();
            }
        }
        println!("ROUTE FOLD CONV r={rows} p={picks} d={dim} first={first_slot} BIT-EXACT PASS words={}", rows * dim * (taps + 1));
        total_words += rows * dim * (taps + 1);
    }
    // 5h. Zero-LUT SM121a register NVFP4 decode (`mode == 3`, `NVFP4_FACTOR_MAGIC` in table row `0xff`)
    // covering ALL 254 finite E4M3 scale bytes x ALL 16 E2M1 codes (4,064 combinations) across ungated,
    // gated FP32, and coalesced gated HALF_OUT paths, bit-for-bit identical to widened BF16 reference.
    {
        use spark_gpu_probe::{packed_product, tc_rows};
        let e2m1 = |c: u8| -> f32 {
            const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let v = MAG[(c & 7) as usize];
            if (c & 8) != 0 { -v } else { v }
        };
        let e4m3 = |s: u8| -> f32 {
            let sign = if (s & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
            let exp = (s >> 3) & 0xf;
            let mant = s & 0x7;
            if exp == 0 { sign * (mant as f32) * (1.0 / 512.0) }
            else { sign * (1.0 + (mant as f32) * 0.125) * 2.0f32.powi(exp as i32 - 7) }
        };
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let (rows, picks, experts, p_rows, p_cols) = (17usize, 4usize, 4usize, 192usize, 256usize);
        let all = rows * picks;
        let parts = packed_product::layout(p_rows, p_cols, false);
        let factors = [0.00026041668f32, 0.001373291f32, 0.01953125f32, 0.000045776367f32];
        let mut raw_nv = vec![0u8; experts * parts.bytes];
        let mut wide_nv = Vec::with_capacity(experts);
        let finite_scales: Vec<u8> = (0u8..=255).filter(|&s| (s & 0x7f) != 0x7f).collect();
        for e in 0..experts {
            let factor = factors[e];
            let unit = &mut raw_nv[e * parts.bytes..(e + 1) * parts.bytes];
            let mut nv_table = [0u16; 256 * 16];
            for &sc in &finite_scales {
                for nib in 0..16u8 {
                    let idx = sc as usize * 16 + nib as usize;
                    let val = to_bf16(e2m1(nib) * e4m3(sc) * factor);
                    nv_table[idx] = val;
                    unit[parts.table + idx * 2..parts.table + idx * 2 + 2].copy_from_slice(&val.to_le_bytes());
                }
            }
            let tag = parts.table + 0xff * 32;
            unit[tag..tag + 4].copy_from_slice(&packed_product::NVFP4_FACTOR_MAGIC_LO.to_le_bytes());
            unit[tag + 4..tag + 8].copy_from_slice(&packed_product::NVFP4_FACTOR_MAGIC_HI.to_le_bytes());
            unit[tag + 8..tag + 12].copy_from_slice(&factor.to_bits().to_le_bytes());
            let mut wide_e = vec![0u16; p_rows * p_cols];
            for r in 0..p_rows {
                let out_r = packed_product::output_row(r, p_rows, true);
                for c in 0..p_cols {
                    let stage = c / packed_product::STAGE;
                    let lane = c % packed_product::STAGE;
                    let c_off = packed_product::code_copy(r, stage, 0, p_cols, false);
                    let s_off = packed_product::scale_copy(r, stage, p_cols, false);
                    // Cover every (finite scale, 4-bit code) combination across the 192x256 matrix.
                    let blk_idx = r * (p_cols / 16) + (c / 16);
                    let sc_b = finite_scales[(blk_idx + e * 61) % finite_scales.len()];
                    let nib = ((c + r + e) & 15) as u8;
                    if c % 2 == 0 {
                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0xf0) | nib;
                    } else {
                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0x0f) | (nib << 4);
                    }
                    unit[parts.scales + s_off + lane / 16] = sc_b;
                    wide_e[out_r * p_cols + c] = nv_table[sc_b as usize * 16 + nib as usize];
                }
            }
            wide_nv.push(wide_e);
        }
        let mut panel_nv = Vec::with_capacity(raw_nv.len());
        for unit in raw_nv.chunks_exact(parts.bytes) {
            panel_nv.extend(spark_gpu_probe::decode_panels::pack_expert_tagged(unit, p_rows, p_cols));
        }
        let panel_words: Vec<f32> = panel_nv.as_chunks::<4>().0.iter()
            .map(|&c| f32::from_bits(u32::from_le_bytes(c)))
            .collect();
        let panel_dev = b.upload_vector(&panel_words)?.into_resident_shared()?;
        let mut route_h = vec![0.0f32; rows * 2 * picks];
        for r in 0..rows {
            for s in 0..picks {
                route_h[r * 2 * picks + s] = ((r * 3 + s) % experts) as f32;
                route_h[r * 2 * picks + picks + s] = 0.25f32;
            }
        }
        let route_dev = b.upload_vector(&route_h)?.into_resident_shared()?;
        let mut w_out = b.vector(all)?;
        let plan = b.moe_group_route_into(rows, 2 * picks, picks, experts, &route_dev, &mut w_out)?;
        let zp_vals: Vec<f32> = (0..rows * p_cols).map(|k| input(k + 313)).collect();
        let zp_dev = b.upload_vector(&zp_vals)?.into_resident_shared()?;
        let mut p_out = b.vector(all * p_rows)?;
        b.project_tc_packed_moe_into(p_rows, p_cols, false, true, &panel_dev, &plan, &zp_dev, &mut p_out, false)?;
        let caps = [4.0f32, 25.0, 0.25, 0.04];
        let mut pg_out = b.vector(all * (p_rows / 2))?;
        b.project_tc_packed_gate_into(p_rows, p_cols, false, &panel_dev, &plan, &zp_dev, caps, &mut pg_out)?;
        let mut pgh_out = b.vector(all * (p_rows / 2) / 2)?;
        b.project_tc_packed_gate_half_into(p_rows, p_cols, false, &panel_dev, &plan, &zp_dev, caps, &mut pgh_out)?;
        b.synchronize()?;
        let mut grouped: Vec<usize> = (0..all).collect();
        grouped.sort_by_key(|&p| (((p / picks) * 3 + (p % picks)) % experts, p));
        let mut want_p = vec![0.0f32; all * p_rows];
        for (pos, &p) in grouped.iter().enumerate() {
            let (r, s) = (p / picks, p % picks);
            let e = (r * 3 + s) % experts;
            let xr = &zp_vals[r * p_cols..(r + 1) * p_cols];
            for n in 0..p_rows {
                want_p[pos * p_rows + n] = tc_rows::tc_word(&wide_nv[e][n * p_cols..(n + 1) * p_cols], xr);
            }
        }
        let got_p = b.read_vector_values(&p_out)?;
        if let Some(i) = (0..want_p.len()).find(|&i| got_p[i].to_bits() != want_p[i].to_bits()) {
            return Err(format!("nvfp4 reg mode3 ungated: word {i} got {:#x} want {:#x}", got_p[i].to_bits(), want_p[i].to_bits()));
        }
        let shape = spark_gpu_probe::causal_ops::Shape {
            op: spark_gpu_probe::causal_ops::Op::CappedGate,
            heads: 0, dim: (p_rows / 2) as u32, taps: 0, rows: all as u32, cols: 0, epsilon: caps[0], bound: caps[1],
        };
        let want_pg = spark_gpu_probe::causal_ops::reference(shape, [&want_p, &[], &[], &[]], &caps[2..])?;
        let got_pg = b.read_vector_values(&pg_out)?;
        if let Some(i) = (0..want_pg.len()).find(|&i| got_pg[i].to_bits() != want_pg[i].to_bits()) {
            return Err(format!("nvfp4 reg mode3 gated: word {i} got {:#x} want {:#x}", got_pg[i].to_bits(), want_pg[i].to_bits()));
        }
        let got_pgh = b.read_vector_values(&pgh_out)?;
        for i in 0..got_pgh.len() {
            let want_w = u32::from(to_bf16(want_pg[2 * i])) | (u32::from(to_bf16(want_pg[2 * i + 1])) << 16);
            if got_pgh[i].to_bits() != want_w {
                return Err(format!("nvfp4 reg mode3 gated half: word {i} got {:#x} want {:#x}", got_pgh[i].to_bits(), want_w));
            }
        }
        println!("NVFP4 REG MODE3 ALL-4064 PAIRS BIT-EXACT PASS words={}", want_p.len() + want_pg.len() + got_pgh.len());
        total_words += want_p.len() + want_pg.len() + got_pgh.len();
    }
    // 5i. Fused prefill `TC_MODE_GATE_BLEND_NORM` (1 gate & 3 batched gates, `dim = 2304`, `K = 64`, partial `BM = 8` rows)
    // compared bit-for-bit against separate `project_tc_rows_into` + `fast_gate_blend_norm`.
    for &(rows, dim, k, count) in &[(17usize, 2304usize, 64usize, 1usize), (25, 2304, 64, 3), (12, 1280, 64, 2)] {
        let eps = 1.0e-5f32;
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let mut ws = Vec::new();
        let mut inns = Vec::new();
        let mut ks = Vec::new();
        let mut ms = Vec::new();
        let mut nws = Vec::new();
        for g in 0..count {
            let w_bf16: Vec<u16> = (0..2 * dim * k).map(|i| to_bf16(input(i + g * 977 + 13) * 0.5)).collect();
            let w_bytes: Vec<u8> = w_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut w_dev = b.vector(dim * k)?;
            b.write_vector_bytes(&mut w_dev, &w_bytes)?;
            ws.push(w_dev.into_resident_shared()?);
            inns.push(b.upload_vector(&(0..rows * k).map(|i| input(i + g * 431 + 7) * 1.5).collect::<Vec<_>>())?.into_resident_shared()?);
            ks.push(b.upload_vector(&(0..rows * dim).map(|i| input(i + g * 557 + 19)).collect::<Vec<_>>())?.into_resident_shared()?);
            ms.push(b.upload_vector(&(0..rows * dim).map(|i| input(i + g * 683 + 29)).collect::<Vec<_>>())?.into_resident_shared()?);
            nws.push(b.upload_vector(&(0..dim).map(|i| input(i + g * 751 + 41) + 1.25).collect::<Vec<_>>())?.into_resident_shared()?);
        }
        let entries: Vec<_> = (0..count).map(|i| (&ws[i], &inns[i], &ks[i], &ms[i], Some(&nws[i]))).collect();
        let outs = b.project_tc_gate_blend_norm(rows, dim, k, eps, false, &entries)?;
        let outs_half = b.project_tc_gate_blend_norm(rows, dim, k, eps, true, &entries)?;
        for idx in 0..count {
            let mut gates = b.vector(rows * 2 * dim)?;
            b.project_tc_rows_into(2 * dim, k, rows, &inns[idx], &ws[idx], &mut gates)?;
            let want_v = b.fast_gate_blend_norm(rows, dim, eps, &[(&gates, &ks[idx], &ms[idx], Some(&nws[idx]))])?;
            b.synchronize()?;
            let (got, want) = (b.read_vector_values(&outs[idx])?, b.read_vector_values(&want_v[0])?);
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("tc_gate_blend_norm r={rows} d={dim} g={idx}/{count}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
            let got_h = b.read_vector_values(&outs_half[idx])?;
            for i in 0..got_h.len() {
                let want_w = u32::from(to_bf16(want[2 * i])) | (u32::from(to_bf16(want[2 * i + 1])) << 16);
                if got_h[i].to_bits() != want_w {
                    return Err(format!("tc_gate_blend_norm half r={rows} d={dim} g={idx}/{count}: word {i} got {:#x} want {:#x}", got_h[i].to_bits(), want_w));
                }
            }
            total_words += want.len() + got_h.len();
        }
        println!("TC GATE BLEND NORM r={rows} d={dim} k={k} gates={count} BIT-EXACT PASS words={}", count * rows * dim);
    }
    // 5j. 2D tiled prefill `ShortConv` (`rows = 135`, crosses 64-row tile boundary!) + fused prefill
    // `STREAM_OP_ROUTE_FOLD_CONV_PREFILL` (`FOLD_HALF` + `AfterExperts ShortConv` + `next_ring`) bit-exact vs reference.
    for &(rows, picks, dim, taps, first_slot) in &[
        (135usize, 8usize, 2304usize, 4usize, 2usize),
        (65, 8, 2304, 4, 3),
        (1029, 4, 512, 4, 1),
    ] {
        let bound = (1.0f64 / ((taps + 1) as f64).sqrt()) as f32;
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let downs_f: Vec<f32> = (0..rows * picks * dim).map(|i| input(i + rows * 37 + 11)).collect();
        let downs_half_words: Vec<f32> = (0..rows * picks * dim / 2)
            .map(|i| f32::from_bits(u32::from(to_bf16(downs_f[2 * i])) | (u32::from(to_bf16(downs_f[2 * i + 1])) << 16)))
            .collect();
        let w_h: Vec<f32> = (0..rows * picks).map(|i| 0.1 + input(i + 23).abs() * 0.2).collect();
        let res_h: Vec<f32> = (0..rows * dim).map(|i| input(i + 79) * 1.5).collect();
        let fix_h: Vec<f32> = (0..dim * taps).map(|i| input(i + 113) * 0.4).collect();
        let rng_h: Vec<f32> = (0..taps * dim).map(|i| input(i + 149) * 1.2).collect();
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let (dh_d, w_d, res_d, fix_d, rng_d) = (up(&downs_half_words)?, up(&w_h)?, up(&res_h)?, up(&fix_h)?, up(&rng_h)?);
        let (mut got_o, mut got_nr) = (b.vector(rows * dim)?, b.vector(taps * dim)?);
        b.fast_route_fold_conv_prefill_into(rows, picks, dim, taps, first_slot, bound, &w_d, &dh_d, &res_d, &fix_d, &rng_d, &mut got_o, &mut got_nr)?;
        // Reference: separate `fast_route_fold_half_into` + window-form `fast_short_conv_win_into` (non-tiled window path).
        let mut ref_fold = b.vector(rows * dim)?;
        b.fast_route_fold_half_into(rows, picks, dim, &w_d, &dh_d, &mut ref_fold)?;
        b.synchronize()?;
        let fold_h = b.read_vector_values(&ref_fold)?;
        let mut win_h = Vec::with_capacity((rows + taps - 1) * dim);
        for j in 0..taps - 1 {
            let s = (first_slot + 1 + j) % taps;
            win_h.extend_from_slice(&rng_h[s * dim..(s + 1) * dim]);
        }
        win_h.extend_from_slice(&fold_h);
        let win_d = up(&win_h)?;
        let mut ref_o_win = b.vector(rows * dim)?;
        b.fast_short_conv_win_into(rows, 1, dim, taps, 6, bound, &win_d, None, Some(&res_d), &fix_d, None, &mut ref_o_win, None)?;
        // Also check tiled prefill `fast_short_conv_win_into` (separate ring + inputs) against window reference!
        let mut got_o_tiled = b.vector(rows * dim)?;
        b.fast_short_conv_win_into(rows, 1, dim, taps, 6, bound, &ref_fold, None, Some(&res_d), &fix_d, Some((&rng_d, first_slot)), &mut got_o_tiled, None)?;
        b.synchronize()?;
        let want_o = b.read_vector_values(&ref_o_win)?;
        let got_o_v = b.read_vector_values(&got_o)?;
        let got_t_v = b.read_vector_values(&got_o_tiled)?;
        if let Some(i) = (0..want_o.len()).find(|&i| got_o_v[i].to_bits() != want_o[i].to_bits()) {
            return Err(format!("route_fold_conv_prefill out r={rows}: word {i} got {:#x} want {:#x}", got_o_v[i].to_bits(), want_o[i].to_bits()));
        }
        if let Some(i) = (0..want_o.len()).find(|&i| got_t_v[i].to_bits() != want_o[i].to_bits()) {
            return Err(format!("short_conv_tiled out r={rows}: word {i} got {:#x} want {:#x}", got_t_v[i].to_bits(), want_o[i].to_bits()));
        }
        let got_nr_v = b.read_vector_values(&got_nr)?;
        let last = (first_slot + rows - 1) % taps;
        for slot in 0..taps {
            let src_row = rows - 1 - (last + taps - slot) % taps;
            let want_slot = &fold_h[src_row * dim..(src_row + 1) * dim];
            let got_slot = &got_nr_v[slot * dim..(slot + 1) * dim];
            if let Some(i) = (0..dim).find(|&i| got_slot[i].to_bits() != want_slot[i].to_bits()) {
                return Err(format!("route_fold_conv_prefill next_ring r={rows} slot={slot}: word {i} got {:#x} want {:#x}", got_slot[i].to_bits(), want_slot[i].to_bits()));
            }
        }
        println!("ROUTE FOLD CONV PREFILL + TILED CONV r={rows} p={picks} d={dim} first={first_slot} BIT-EXACT PASS words={}", 2 * want_o.len() + taps * dim);
        total_words += 2 * want_o.len() + taps * dim;
    }
    // 5k. `FW_HALF_OUT` (`flash_window_attn_half_into`) bit-exact vs `flash_window_attn_into` + BF16 pack.
    {
        let (rows, heads, kv_heads, head_dim, rel_dim, view, extent) = (65usize, 24usize, 8usize, 128usize, 16usize, 128usize, 64usize);
        let scale = (1.0f64 / (head_dim as f64).sqrt()) as f32;
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let pack_h = |vals: &[f32]| -> Vec<f32> {
            (0..vals.len() / 2).map(|i| f32::from_bits(u32::from(to_bf16(vals[2 * i])) | (u32::from(to_bf16(vals[2 * i + 1])) << 16))).collect()
        };
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let q_d = up(&(0..rows * heads * head_dim).map(|i| input(i + 17) * 1.5).collect::<Vec<_>>())?;
        let k_d = up(&pack_h(&(0..view * kv_heads * head_dim).map(|i| input(i + 31)).collect::<Vec<_>>()))?;
        let v_d = up(&pack_h(&(0..view * kv_heads * head_dim).map(|i| input(i + 53)).collect::<Vec<_>>()))?;
        let m_d = up(&(0..rows * heads * rel_dim).map(|i| input(i + 71) * 1.2).collect::<Vec<_>>())?;
        let rt_d = up(&(0..rel_dim * extent).map(|i| input(i + 97) * 0.5).collect::<Vec<_>>())?;
        let sk_d = up(&(0..rows * heads).map(|i| input(i + 113) * 0.8).collect::<Vec<_>>())?;
        let gt_d = up(&(0..rows * heads * head_dim).map(|i| input(i + 131) * 1.3).collect::<Vec<_>>())?;
        let qs_d = up(&(0..heads * head_dim).map(|i| input(i + 157) + 1.1).collect::<Vec<_>>())?;
        let ms_d = up(&(0..heads * rel_dim).map(|i| input(i + 179) + 1.1).collect::<Vec<_>>())?;
        let pos_d = up(&(0..rows).map(|r| 1.0 + 0.01 * r as f32).collect::<Vec<_>>())?;
        let steps_h: Vec<f32> = (0..rows).flat_map(|r| [(r + 1) as f32, r as f32]).collect();
        let st_d = up(&steps_h)?;
        let clips = spark_gpu_probe::driver::OneClips {
            query_scale: Some(&qs_d), moved_scale: Some(&ms_d), position: Some(&pos_d),
            query_bound: 8.0, moved_bound: 8.0, written_keys: None, written_values: None,
        };
        let mut ref_o = b.vector(rows * heads * head_dim)?;
        let mut half_o = b.vector(rows * heads * head_dim / 2)?;
        b.flash_window_attn_into(rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, 0, &q_d, &k_d, &v_d, &m_d, &rt_d, &sk_d, &gt_d, &st_d, Some(clips), &mut ref_o)?;
        b.flash_window_attn_half_into(rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, 0, &q_d, &k_d, &v_d, &m_d, &rt_d, &sk_d, &gt_d, &st_d, Some(clips), &mut half_o)?;
        b.synchronize()?;
        let want_h = pack_h(&b.read_vector_values(&ref_o)?);
        let got_h = b.read_vector_values(&half_o)?;
        if let Some(i) = (0..want_h.len()).find(|&i| got_h[i].to_bits() != want_h[i].to_bits()) {
            return Err(format!("fw_half_out: word {i} got {:#x} want {:#x}", got_h[i].to_bits(), want_h[i].to_bits()));
        }
        println!("FLASH WINDOW HALF_OUT r={rows} h={heads}/{kv_heads} BIT-EXACT PASS words={}", want_h.len());
        total_words += want_h.len();
    }
    // 5l. `MIX_ONE_POST_HALF_OUT` and `MIX_LIST_NORM_HALF` bit-exact vs FP32 forms + BF16 pack.
    {
        let (rows, dim, kept) = (33usize, 2304usize, 5usize);
        let (scale, eps, post_eps) = ((1.0f64 / (dim as f64).sqrt()) as f32, 1.0e-6f32, 1.0e-5f32);
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let pack_h = |vals: &[f32]| -> Vec<f32> {
            (0..vals.len() / 2).map(|i| f32::from_bits(u32::from(to_bf16(vals[2 * i])) | (u32::from(to_bf16(vals[2 * i + 1])) << 16))).collect()
        };
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let list_d: Vec<_> = (0..kept - 1).map(|e| up(&(0..rows * dim).map(|i| input(i + e * 613 + 7)).collect::<Vec<_>>())).collect::<Result<_, _>>()?;
        let stream_d = up(&(0..rows * dim).map(|i| input(i + 911)).collect::<Vec<_>>())?;
        let refs: Vec<_> = list_d.iter().collect();
        // MIX_ONE_POST_HALF_OUT (asked = 3, per-query weights, transpose = true).
        let q3_d = up(&(0..3 * dim).map(|i| input(i + 29)).collect::<Vec<_>>())?;
        let pw3_d = up(&(0..3 * dim).map(|i| input(i + 419) + 1.25).collect::<Vec<_>>())?;
        let (_, _, ref_p3) = b.fast_mix_rows_normed(rows, dim, scale, eps, post_eps, true, false, false, &q3_d, &refs, &stream_d, Some(&pw3_d), true)?;
        let (_, _, got_p3) = b.fast_mix_rows_normed_half(rows, dim, scale, eps, post_eps, true, false, false, &q3_d, &refs, &stream_d, Some(&pw3_d), true)?;
        // MIX_LIST_NORM_HALF (asked = 48, 12 loop layers x 4 messages).
        let q48_d = up(&(0..48 * dim).map(|i| input(i + 577)).collect::<Vec<_>>())?;
        let mut all_refs = refs.clone();
        all_refs.push(&stream_d);
        let (ref_m48, ref_n48) = b.fast_mix_list_normed(rows, dim, scale, post_eps, &q48_d, &all_refs)?;
        let (got_m48, got_n48) = b.fast_mix_list_normed_half(rows, dim, scale, post_eps, &q48_d, &all_refs)?;
        b.synchronize()?;
        let want_p3 = pack_h(&b.read_vector_values(&ref_p3)?);
        let got_p3_v = b.read_vector_values(&got_p3)?;
        if let Some(i) = (0..want_p3.len()).find(|&i| got_p3_v[i].to_bits() != want_p3[i].to_bits()) {
            return Err(format!("mix_one_post_half: word {i} got {:#x} want {:#x}", got_p3_v[i].to_bits(), want_p3[i].to_bits()));
        }
        let (want_m48, got_m48_v) = (b.read_vector_values(&ref_m48)?, b.read_vector_values(&got_m48)?);
        if let Some(i) = (0..want_m48.len()).find(|&i| got_m48_v[i].to_bits() != want_m48[i].to_bits()) {
            return Err(format!("mix_list_norm_half msg: word {i}"));
        }
        let want_n48 = pack_h(&b.read_vector_values(&ref_n48)?);
        let got_n48_v = b.read_vector_values(&got_n48)?;
        if let Some(i) = (0..want_n48.len()).find(|&i| got_n48_v[i].to_bits() != want_n48[i].to_bits()) {
            return Err(format!("mix_list_norm_half normed: word {i} got {:#x} want {:#x}", got_n48_v[i].to_bits(), want_n48[i].to_bits()));
        }
        println!("MIX HALF_OUT (ONE + LIST_48) r={rows} d={dim} BIT-EXACT PASS words={}", want_p3.len() + want_n48.len());
        total_words += want_p3.len() + want_n48.len();
    }
    // 5m. `TC_MODE_NARROW_HALF` (`rows = 32, 128` single; `rows = 64` dual-half `GateDown`) + dense/split `HALF_IN`
    // bit-exact vs FP32 `project_tc_rows_into` / `project_tc_rows_pair_into` / `project_tc_rows_split_into`.
    {
        let (inputs, cols) = (529usize, 2304usize);
        let to_bf16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let pack_h = |vals: &[f32]| -> Vec<f32> {
            (0..vals.len() / 2).map(|i| f32::from_bits(u32::from(to_bf16(vals[2 * i])) | (u32::from(to_bf16(vals[2 * i + 1])) << 16))).collect()
        };
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let mk_w = |rows: usize, k: usize, seed: usize| -> Result<spark_gpu_probe::driver::DeviceVector<'_>, String> {
            let wb: Vec<u8> = (0..rows * k).flat_map(|i| to_bf16(input(i + seed) * 0.5).to_le_bytes()).collect();
            let mut d = b.vector(rows * k / 2)?;
            b.write_vector_bytes(&mut d, &wb)?;
            d.into_resident_shared()
        };
        let x1_h: Vec<f32> = (0..inputs * cols).map(|i| input(i + 107) * 1.5).collect();
        let x2_h: Vec<f32> = (0..inputs * cols).map(|i| input(i + 311) * 1.5).collect();
        let (x1_d, x2_d) = (up(&x1_h)?, up(&x2_h)?);
        let (x1_hd, x2_hd) = (up(&pack_h(&x1_h))?, up(&pack_h(&x2_h))?);
        for &rows in &[32usize, 128, 512] {
            let w_d = mk_w(rows, cols, rows * 13)?;
            let (mut ref_o, mut got_o) = (b.vector(inputs * rows)?, b.vector(inputs * rows)?);
            b.project_tc_rows_into(rows, cols, inputs, &x1_d, &w_d, &mut ref_o)?;
            b.project_tc_rows_half_in_into(rows, cols, inputs, &x1_hd, &w_d, &mut got_o)?;
            b.synchronize()?;
            let (want, got) = (b.read_vector_values(&ref_o)?, b.read_vector_values(&got_o)?);
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("tc_half_in rows={rows}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
            total_words += want.len();
        }
        // Dual-half `GateDown` (`rows = 64, cols = 4608, inputs = 529`) via `TC_MODE_NARROW_HALF`.
        let w_gd = mk_w(64, 2 * cols, 777)?;
        let (mut ref_gd, mut got_gd) = (b.vector(inputs * 64)?, b.vector(inputs * 64)?);
        b.project_tc_rows_pair_into(64, 2 * cols, inputs, &x1_d, &x2_d, &w_gd, &mut ref_gd)?;
        b.project_tc_rows_pair_half_in_into(64, 2 * cols, inputs, &x1_hd, &x2_hd, &w_gd, &mut got_gd)?;
        // Split multi-output `HALF_IN` (`parts = [256, 256]`, `rows = 512`).
        let w_sp = mk_w(512, cols, 991)?;
        let (mut rs0, mut rs1, mut gs0, mut gs1) = (b.vector(inputs * 256)?, b.vector(inputs * 256)?, b.vector(inputs * 256)?, b.vector(inputs * 256)?);
        b.project_tc_rows_split_into(512, cols, inputs, &x1_d, &w_sp, &mut [(&mut rs0, 256), (&mut rs1, 256)])?;
        b.project_tc_rows_split_half_in_into(512, cols, inputs, &x1_hd, &w_sp, &mut [(&mut gs0, 256), (&mut gs1, 256)])?;
        b.synchronize()?;
        for (tag, ref_v, got_v) in [("narrow_half_pair", &ref_gd, &got_gd), ("split_half_0", &rs0, &gs0), ("split_half_1", &rs1, &gs1)] {
            let (want, got) = (b.read_vector_values(ref_v)?, b.read_vector_values(got_v)?);
            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                return Err(format!("{tag}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
            }
            total_words += want.len();
        }
        println!("TC NARROW_HALF + DENSE/SPLIT HALF_IN inputs={inputs} BIT-EXACT PASS");
    }
    // 5n. 2D channel-tiled prefill `ShortConv` (`rows = 65` tile=64, `rows = 1029` tile=256) with fused `next_ring`,
    // bit-exact against untiled window `fast_short_conv_win_into`.
    for &(rows, dim, taps, first_slot, form) in &[
        (65usize, 2304usize, 4usize, 1usize, 6u32),
        (1029, 2304, 4, 3, 2),
        (67, 512, 3, 2, 0),
    ] {
        let bound = (1.0f64 / ((taps + 1) as f64).sqrt()) as f32;
        let inp_h: Vec<f32> = (0..rows * dim).map(|i| input(i + rows * 17 + 3) * 1.5).collect();
        let rng_h: Vec<f32> = (0..taps * dim).map(|i| input(i + first_slot * 41 + 9) * 1.5).collect();
        let add_h: Vec<f32> = (0..rows * dim).map(|i| input(i + rows * 7 + 19) * 1.2).collect();
        let fix_h: Vec<f32> = (0..dim * taps).map(|i| input(i + 83) * 0.4).collect();
        let mut win_h = Vec::with_capacity((rows + taps - 1) * dim);
        for j in 0..taps - 1 {
            let s = (first_slot + 1 + j) % taps;
            win_h.extend_from_slice(&rng_h[s * dim..(s + 1) * dim]);
        }
        win_h.extend_from_slice(&inp_h);
        let up = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
        let (inp_d, rng_d, add_d, fix_d, win_d) = (up(&inp_h)?, up(&rng_h)?, up(&add_h)?, up(&fix_h)?, up(&win_h)?);
        let has_add = (form & 2) != 0;
        let (mut got_o, mut got_nr, mut ref_o) = (b.vector(rows * dim)?, b.vector(taps * dim)?, b.vector(rows * dim)?);
        b.fast_short_conv_win_ring_into(rows, 1, dim, taps, form, bound, &inp_d, None, has_add.then_some(&add_d), &fix_d, Some((&rng_d, first_slot)), &mut got_o, Some(&mut got_nr))?;
        b.fast_short_conv_win_into(rows, 1, dim, taps, form, bound, &win_d, None, has_add.then_some(&add_d), &fix_d, None, &mut ref_o, None)?;
        b.synchronize()?;
        let (go, ro, gnr) = (b.read_vector_values(&got_o)?, b.read_vector_values(&ref_o)?, b.read_vector_values(&got_nr)?);
        if let Some(i) = (0..ro.len()).find(|&i| go[i].to_bits() != ro[i].to_bits()) {
            return Err(format!("short_conv_win_ring out r={rows}: word {i} got {:#x} want {:#x}", go[i].to_bits(), ro[i].to_bits()));
        }
        let last = (first_slot + rows - 1) % taps;
        for slot in 0..taps {
            let src_row = rows - 1 - (last + taps - slot) % taps;
            let want_r = &inp_h[src_row * dim..(src_row + 1) * dim];
            let got_r = &gnr[slot * dim..(slot + 1) * dim];
            if let Some(i) = (0..dim).find(|&i| got_r[i].to_bits() != want_r[i].to_bits()) {
                return Err(format!("short_conv_win_ring next_ring r={rows} slot={slot}: word {i} got {:#x} want {:#x}", got_r[i].to_bits(), want_r[i].to_bits()));
            }
        }
        println!("SHORT CONV WIN RING r={rows} d={dim} taps={taps} BIT-EXACT PASS words={}", rows * dim + taps * dim);
        total_words += rows * dim + taps * dim;
    }
    Ok(total_words)
}
