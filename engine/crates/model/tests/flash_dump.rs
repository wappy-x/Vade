//! VADEDBG1 dump format, MXFP8->BF16 expansion and traced CPU forward tests.
use engine_model::flash::{dump, mxfp8, nvfp4, refcpu::*};
use engine_model::Result;

#[test]
fn checksum_is_fnv1a_over_bit_patterns() {
    assert_eq!(dump::checksum(&[]), 0xcbf29ce484222325);
    let mut h = 0xcbf29ce484222325u64;
    for v in [1.5f32, -0.25, 0.0] {
        for b in v.to_bits().to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3); }
    }
    assert_eq!(dump::checksum(&[1.5, -0.25, 0.0]), h);
    assert_ne!(dump::checksum(&[0.0]), dump::checksum(&[-0.0]));
}

#[test]
fn dump_line_formats() {
    assert_eq!(dump::rec_line("embed", &[1.0, 2.0, 3.0]),
        format!("rec embed 3 {:016x} 3f800000,40000000,40400000", dump::checksum(&[1.0, 2.0, 3.0])));
    assert_eq!(dump::rec_line("x", &[]), format!("rec x 0 {:016x} ", dump::checksum(&[])));
    assert_eq!(dump::moe_line(1, &[(2, 0.25), (5, 0.75)]), "moe 1 2,5 3e800000,3f400000");
    let (top, argmax) = dump::top5(&[0.1, 2.0, -1.0, 2.0, 0.5, 0.4, 0.3]);
    assert_eq!(argmax, 1); // tie between ids 1 and 3: lowest wins
    assert_eq!(top[..2], [(1, 2.0), (3, 2.0)]);
    assert!(dump::top_line(&[0.1, 2.0, -1.0, 2.0, 0.5, 0.4, 0.3]).starts_with("top 1 1:40000000,3:40000000,"));
}

#[test]
fn mxfp8_expand_matches_fp32_oracle_rounded_once() -> Result<()> {
    let mut state = 0x243f6a8885a308d3u64;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let weights: Vec<u8> = (0..32 * 64).map(|_| {
        let b = (next() >> 32) as u8;
        if b & 127 == 127 { b & 0x80 } else { b } // keep E4M3 finite
    }).collect();
    let scales: Vec<u8> = (0..64).map(|_| 100 + (next() % 51) as u8).collect(); // finite FP32 range
    let mut fp32 = vec![0.0; weights.len()];
    mxfp8::dequantize(&weights, &scales, &mut fp32)?;
    let mut bf16 = vec![0u16; weights.len()];
    mxfp8::expand_to_bf16(&weights, &scales, &mut bf16)?;
    for (v, b) in fp32.iter().zip(&bf16) { assert_eq!(*b, nvfp4::f32_to_bf16(*v)); }
    assert!(mxfp8::expand_to_bf16(&[0x7f; 32], &[127], &mut [0u16; 32]).is_err());
    assert!(mxfp8::expand_to_bf16(&[0; 31], &[127], &mut [0u16; 31]).is_err());
    Ok(())
}

fn mat(r: usize, c: usize, d: Vec<f32>) -> M { M::new(r, c, d).unwrap() }
fn rm(seed: u64, r: usize, c: usize) -> M { mat(r, c, fill_random(seed, r * c, 0.08)) }
fn tiny_model() -> (ModelW, PleHash) {
    let d = Dims { hidden: 16, hc: 4, low: 4, qh: 2, kvh: 1, hd: 4, rot: 2, kh: 1, vh: 2,
        kd: 2, vd: 2, experts: 4, topk: 2, ew: 4, sw: 4, vocab: 32, eos: 31, eps: 1e-6, theta: 10000.0 };
    let wide = d.wide();
    let mut s = 2000u64;
    let hc = |sd: u64, inject: bool| HcW { norm: fill_random(sd, wide, 0.05),
        down: rm(sd + 1, d.low, wide), up: rm(sd + 2, wide, d.low),
        inject: inject.then(|| rm(sd + 3, d.hc, wide)) };
    let dex = |sd: u64, wdt: usize| DenseExpert { gate: rm(sd, wdt, d.hidden),
        up: rm(sd + 1, wdt, d.hidden), down: rm(sd + 2, d.hidden, wdt) };
    let mut layers = Vec::new();
    for li in 0..48 {
        s += 1;
        let b = s * 10;
        let attn = if layer_is_full(li) {
            AttnW::Full(FullW { q: rm(b, d.qh * 2 * d.hd, d.hidden), k: rm(b + 1, d.kvh * d.hd, d.hidden),
                v: rm(b + 2, d.kvh * d.hd, d.hidden), o: rm(b + 3, d.hidden, d.qh * d.hd),
                qn: vec![0.0; d.hd], kn: vec![0.0; d.hd] })
        } else {
            AttnW::Gdn(GdnW { qkv: rm(b, d.conv(), d.hidden), z: rm(b + 1, d.vh * d.vd, d.hidden),
                a: rm(b + 2, d.vh, d.hidden), b: rm(b + 3, d.vh, d.hidden),
                conv: fill_random(b + 4, d.conv() * 4, 0.2), alog: vec![0.1; d.vh],
                dt: vec![0.0; d.vh], norm: vec![1.0; d.vd], out: rm(b + 5, d.hidden, d.vh * d.vd) })
        };
        layers.push(LayerW { attn_hc: hc(b + 100, true), mlp_hc: hc(b + 200, true), attn,
            moe: MoeW { router: rm(b + 300, d.experts, d.hidden), shared: dex(b + 400, d.sw),
                shared_gate: rm(b + 500, 1, d.hidden) },
            experts: Box::new((0..d.experts).map(|e| dex(b + 600 + e as u64 * 7, d.ew)).collect::<Vec<_>>()),
            ple: (li == 1).then(|| PleW { key: rm(b + 700, wide, d.hidden),
                value: rm(b + 701, d.hidden, d.hidden), nk: vec![0.0; wide], nq: vec![0.0; wide],
                nc: vec![0.0; wide], conv: fill_random(b + 702, wide * 4, 0.1) }) });
    }
    let (sizes, offsets, _) = ple_vocab(100, 0, 4, 128);
    let hash = PleHash::new(ple_multipliers(32, 3, 0, 1234), sizes, offsets, 31);
    (ModelW { dims: d, embed: rm(42, 32, d.hidden), layers, mixer: hc(777, false),
        head: rm(888, 32, d.hidden) }, hash)
}

#[test]
fn traced_forward_matches_forward_and_emits_dump_stages() -> Result<()> {
    let (m, hash) = tiny_model();
    let lookup = |row: u64, out: &mut [f32]| -> Result<()> {
        for (j, o) in out.iter_mut().enumerate() { *o = (splitmix64(row ^ (j as u64 + 1)) % 97) as f32 * 0.002 - 0.1; }
        Ok(())
    };
    let fresh = || PleHash::new(hash.mult.clone(), hash.sizes.clone(), hash.offsets.clone(), 31);
    let tokens = [1u32, 5];
    let mut st = ModelState::new(&m, fresh());
    let plain = forward(&m, &mut st, &tokens, &DenseIndexer, &lookup)?;
    let mut lines = vec!["VADEDBG1 flash-cpu-test".to_string()];
    let mut st = ModelState::new(&m, fresh());
    for (pos, &tok) in tokens.iter().enumerate() {
        lines.push(format!("token {pos} {tok}"));
        forward_traced(&m, &mut st, &[tok], &DenseIndexer, &lookup, &mut |t| match t {
            Trace::Stage(name, values) => lines.push(dump::rec_line(&name, values)),
            Trace::Moe(layer, pairs) => lines.push(dump::moe_line(layer, pairs)),
            Trace::Logits(logits) => lines.push(dump::top_line(logits)),
        })?;
    }
    assert_eq!(st.pos, 2);
    // Per token: embed + ple_e/ple (layer 1) + 48 attn + 48 x + final records.
    let count = |prefix: &str| lines.iter().filter(|l| l.starts_with(prefix)).count();
    assert_eq!(count("rec "), 2 * (1 + 2 + 48 + 48 + 1));
    assert_eq!(count("moe "), 2 * 48);
    assert_eq!(count("top "), 2);
    assert!(lines[2].starts_with("rec embed "));
    assert!(lines[3].starts_with("rec layer0.attn "));
    assert!(lines.iter().any(|l| l.starts_with("rec layer1.ple_e ")));
    let last_top = lines.iter().rfind(|l| l.starts_with("top ")).unwrap();
    assert_eq!(last_top.split_whitespace().nth(1).unwrap().parse::<u32>().unwrap(), dump::top5(&plain).1);
    Ok(())
}
