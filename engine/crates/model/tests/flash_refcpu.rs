//! Golden tests for the FP32 CPU qwen4_exp reference. Expected values were
//! hand-computed from the reference formulas (transformers b0cdc61
//! modeling_qwen4_exp.py, cited as TF:line) with independent arithmetic.
use engine_model::flash::refcpu::*;
use engine_model::Result;

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) { assert!((x - y).abs() <= tol, "{x} vs {y}"); }
}
fn mat(r: usize, c: usize, d: Vec<f32>) -> M { M::new(r, c, d).unwrap() }
fn ident(n: usize) -> M {
    let mut d = vec![0.0; n * n];
    for i in 0..n { d[i * n + i] = 1.0; }
    mat(n, n, d)
}
fn tiny_dims() -> Dims {
    Dims { hidden: 16, hc: 4, low: 4, qh: 2, kvh: 1, hd: 4, rot: 2, kh: 1, vh: 2,
        kd: 2, vd: 2, experts: 4, topk: 2, ew: 4, sw: 4, vocab: 32, eos: 31,
        eps: 1e-6, theta: 10000.0 }
}

#[test]
fn real_dims_match_checkpoint() { // config.json text_config + SOURCES header
    let d = Dims::real();
    assert_eq!((d.hidden, d.hc, d.low, d.qh, d.kvh, d.hd, d.rot), (2560, 4, 320, 24, 2, 256, 64));
    assert_eq!((d.kh, d.vh, d.kd, d.vd, d.experts, d.topk, d.ew, d.sw), (16, 48, 128, 128, 512, 10, 640, 640));
    let full = (0..48).filter(|&l| layer_is_full(l)).count();
    assert_eq!((48 - full, full), (36, 12)); // full_attention_interval 4
    // 97 FP32 hyperconnection norms: 48 layers x2 + final mixer (MTP is BF16).
    let mut names = Vec::new();
    for l in 0..48 {
        for r in ["attn", "mlp"] {
            names.extend(tensor_names_hc(&format!("model.language_model.layers.{l}.{r}_hyper_connection"), true));
        }
    }
    names.extend(tensor_names_hc("model.language_model.hyper_connection_mixer", false));
    assert_eq!(names.iter().filter(|n| n.ends_with("hc_norm.weight")).count(), 97);
    assert_eq!(tensor_names_indexer("model.language_model.layers.3.self_attn")[0],
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight");
}

#[test]
fn hc_mix_and_combine_golden() { // TF:994-1022 GatedResidual, TF:1293-1300
    let d = Dims { hidden: 2, hc: 2, low: 2, ..tiny_dims() };
    let w = HcW { norm: vec![0.0, 0.5, -0.5, 0.0],
        down: mat(2, 4, vec![0.5, 0.0, -0.25, 0.0, 0.0, 1.0, 0.0, 0.5]),
        up: mat(4, 2, vec![1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.0]),
        inject: Some(mat(2, 4, vec![0.25, 0.0, 0.25, 0.0, 0.0, -0.5, 0.0, 1.0])) };
    let (u, r) = hc_mix(&d, &w, &[1.0, 2.0, 3.0, -1.0]).unwrap();
    close(&u, &[0.35475294, 0.22891767], 1e-6);
    close(&r, &[1.08127506, 0.66453453], 1e-6);
    let mut x = vec![1.0, 2.0, 3.0, -1.0];
    hc_combine(&d, &mut x, &r, &[0.3, -0.2]).unwrap();
    close(&x, &[1.32438252, 1.78374499, 3.19936036, -1.13290691], 1e-6);
    // Zero mix weights: gates are sigmoid(0)=1/2 and injection is exactly 1.
    let z = HcW { norm: vec![0.0; 4], down: M::zeros(2, 4), up: M::zeros(4, 2),
        inject: Some(M::zeros(2, 4)) };
    let (_, r) = hc_mix(&d, &z, &[3.0, 4.0, 3.0, 4.0]).unwrap();
    assert_eq!(r, [1.0, 1.0]);
}

#[test]
fn rope_partial_golden() { // TF:107-115 inv_freq, TF:626-661 rotate-half
    let (cos, sin) = rope_cos_sin(10000.0, 4, 2);
    let mut x = vec![1.0, 0.0, 0.0, 1.0, 9.0, 9.0];
    apply_rope(&mut x, &cos, &sin);
    close(&x, &[-0.41614684, -0.01999867, 0.90929743, 0.99980001, 9.0, 9.0], 1e-6);
    let (cos, sin) = rope_cos_sin(1e7, 64, 0); // position 0 is the identity
    let mut x: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let before = x.clone();
    apply_rope(&mut x, &cos, &sin);
    assert_eq!(x, before);
}

#[test]
fn full_attn_single_token_golden() { // TF:810-892; softmax of one key is 1
    let d = Dims { hidden: 2, qh: 1, kvh: 1, hd: 2, rot: 2, ..tiny_dims() };
    // q rows: query = I, gate = 0 -> sigmoid(gate) = 1/2 (TF:858-861,889).
    let w = FullW { q: mat(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        k: ident(2), v: ident(2), o: ident(2), qn: vec![0.0; 2], kn: vec![0.0; 2] };
    let mut cache = KvCache::default();
    let y = full_attn(&d, &w, &mut cache, &DenseIndexer, 3, 0, &[3.0, 4.0]).unwrap();
    close(&y, &[1.5, 2.0], 1e-6); // v * 1/2, o = I
}

#[test]
fn gated_norm_sigmoid_golden() { // TF:171-187 with sigmoid gate (not silu)
    let mut y = [0.0; 2];
    gated_norm(&[0.42426385, -0.21213192], &[2.0, 0.5], &[0.0, 3.0f32.ln()], 2, 1e-6, &mut y).unwrap();
    close(&y, &[1.26490544, -0.23716977], 1e-5);
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-7 && (softplus(0.0) - 2.0f32.ln()).abs() < 1e-6);
}

#[test]
fn gdn_two_steps_golden() { // TF:391-455 recurrence, TF:456-616 projections
    // hidden=6, qkv=I, conv keeps only tap 3, so c = silu(u); a=b=0 rows,
    // A_log=dt=0 -> beta=1/2, decay=exp(-ln2)=1/2 (TF:570-572).
    let d = Dims { hidden: 6, kh: 1, vh: 1, kd: 2, vd: 2, ..tiny_dims() };
    let mut conv = vec![0.0; 24];
    for ch in 0..6 { conv[ch * 4 + 3] = 1.0; }
    let mut outd = vec![0.0; 12];
    outd[0] = 1.0; outd[3] = 1.0; // out [6][2]: first two rows are I
    let w = GdnW { qkv: ident(6), z: M::zeros(2, 6), a: M::zeros(1, 6), b: M::zeros(1, 6),
        conv, alog: vec![0.0], dt: vec![0.0], norm: vec![1.0, 1.0], out: mat(6, 2, outd) };
    let mut st = GdnState::new(&d);
    let u = [3.0, 4.0, 1.0, 0.0, 2.0, -1.0];
    let y = gdn_token(&d, &w, &mut st, &u).unwrap();
    close(&y[..2], &[0.69900241, -0.10671624], 1e-5);
    close(&st.s, &[0.88079624, -0.13447059, 0.0, 0.0], 1e-5); // S = k (x) (v/2)
    let _ = gdn_token(&d, &w, &mut st, &u).unwrap();
    // Second step: S <- S/2 + k (x) (v - S^T k /2)/2 = 1.25 * S_first here.
    close(&st.s, &[1.10099530, -0.16808824, 0.0, 0.0], 1e-5);
}

#[test]
fn ple_multipliers_and_vocab_golden() { // TF:1032-1048, TF:1087-1104
    assert_eq!(ple_multipliers(248320, 3, 0, 1234),
        [23703573157769, 20109073645365, 8052911324071]);
    let (sizes, offsets, padded) = ple_vocab(20_000_000, 0, 16, 128);
    assert_eq!(&sizes[..4], &[20000003, 20000023, 20000033, 20000047]);
    assert_eq!(sizes[15], 20000171);
    assert_eq!(offsets[1], 20000003);
    assert_eq!(padded, 320_001_536); // == PLE_ROWS in the loader
    assert_eq!(padded, engine_model::flash::PLE_ROWS);
    let (s2, _, _) = ple_vocab(100, 0, 4, 128);
    assert_eq!(s2, [101, 103, 107, 109]);
}

#[test]
fn ple_row_ids_fixed_triples() { // TF:1150-1163: XOR mixes, euclid mod, offsets
    let (sizes, offsets, _) = ple_vocab(20_000_000, 0, 16, 128);
    let h = PleHash::new(ple_multipliers(248320, 3, 0, 1234), sizes, offsets, 248044);
    assert_eq!(h.rows_pure(5, 7, 11), [18991213, 39727615, 50096016, 68612008,
        93054523, 101202072, 123423469, 159718776, 164665585, 181064400,
        216863305, 224860832, 243060730, 261260670, 280660766, 317660843]);
    assert_eq!(h.rows_pure(0, 0, 0), h.offsets.iter().map(|&o| o as u64).collect::<Vec<_>>());
    assert_eq!(h.rows_pure(248319, 248319, 248319)[0], 3035822);
    // EOS-bounded streaming (TF:1106-1147): EOS sees its segment, next restarts.
    let (s, o, _) = ple_vocab(100, 0, 4, 128);
    let mut hs = PleHash::new(ple_multipliers(32, 3, 0, 1234), s, o, 31);
    let _ = hs.advance(5);
    let at_eos = hs.advance(31);
    assert_eq!(at_eos, hs.rows_pure(31, 5, 31));
    let after = hs.advance(7);
    assert_eq!(after, hs.rows_pure(7, 31, 31));
    let second = hs.advance(9);
    assert_eq!(second, hs.rows_pure(9, 7, 31));
    assert_eq!(ple_shard(2_500_012), 1);
}

#[test]
fn ple_token_golden() { // TF:1190-1245 gate + dilated conv (tap t only here)
    let d = Dims { hidden: 2, hc: 1, ..tiny_dims() };
    let mut conv = vec![0.0; 8];
    conv[3] = 1.0; conv[7] = 1.0;
    let w = PleW { key: ident(2), value: ident(2), nk: vec![0.0; 2], nq: vec![0.0; 2],
        nc: vec![0.0; 2], conv };
    let mut st = PleState::new(&d);
    let mut x = vec![1.0, 1.0];
    ple_token(&d, &w, &mut st, &mut x, &[1.0, 0.0]).unwrap();
    close(&x, &[2.86869093, 1.0], 1e-5);
}

#[test]
fn router_and_moe_golden() { // TF:951-991 softmax top-k renormalize + shared
    let (ids, w) = route(&[0.0, 1.0, 2.0, 3.0], 2);
    assert_eq!(ids, [3, 2]);
    close(&w, &[0.73105858, 0.26894142], 1e-6);
    let d = Dims { hidden: 2, experts: 4, topk: 2, ew: 2, sw: 2, ..tiny_dims() };
    let ex = || DenseExpert { gate: ident(2), up: ident(2), down: ident(2) };
    let experts: Vec<DenseExpert> = (0..4).map(|_| ex()).collect();
    let moe = MoeW { router: mat(4, 2, vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        shared: ex(), shared_gate: M::zeros(1, 2) };
    // expert(x)=silu(x)*x for x=[1,-1]; weights sum to 1; shared adds half.
    let y = moe_token(&d, &moe, &experts, &[1.0, -1.0]).unwrap();
    close(&y, &[1.09658787, 0.40341213], 1e-6);
}

#[test]
fn nvfp4_expert_matches_dense_oracle() {
    // Identity 16x16 in NVFP4: E2M1 code 2 = 1.0, E4M3 scale byte 56 = 1.0.
    let one = || {
        let mut packed = vec![0u8; 128];
        for i in 0..16 { packed[i * 8 + i / 2] |= if i % 2 == 0 { 2 } else { 2 << 4 }; }
        Nvfp4Expert { packed, scales: vec![56; 16], global: 1.0, r: 16, c: 16 }
    };
    let nv = Nvfp4Experts { gate: vec![one()], up: vec![one()], down: vec![one()] };
    let dense = vec![DenseExpert { gate: ident(16), up: ident(16), down: ident(16) }];
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 1.0).collect();
    let (mut a, mut b) = (vec![0.0; 16], vec![0.0; 16]);
    nv.expert(0, &x, &mut a).unwrap();
    dense.expert(0, &x, &mut b).unwrap();
    assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    // PLE row oracle: packed 0x21 -> [0.5, 1.0] with unit scales (nvfp4.rs).
    let mut out = [0.0; 16];
    ple_row_nvfp4(&[0x21; 8], &[56], 2.0, &mut out).unwrap();
    assert_eq!(&out[..2], &[1.0, 2.0]);
}

fn rm(seed: u64, r: usize, c: usize) -> M { mat(r, c, fill_random(seed, r * c, 0.08)) }
fn tiny_model() -> (ModelW, PleHash) {
    let d = tiny_dims();
    let wide = d.wide();
    let mut s = 1000u64;
    let mut next = move || { s += 1; s };
    let hc = |sd: u64, inject: bool| HcW { norm: fill_random(sd, wide, 0.05),
        down: rm(sd + 1, d.low, wide), up: rm(sd + 2, wide, d.low),
        inject: inject.then(|| rm(sd + 3, d.hc, wide)) };
    let dex = |sd: u64, w: usize| DenseExpert { gate: rm(sd, w, d.hidden),
        up: rm(sd + 1, w, d.hidden), down: rm(sd + 2, d.hidden, w) };
    let mut layers = Vec::new();
    for li in 0..48 {
        let b = next() * 10;
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
fn whole_model_driver_deterministic() -> Result<()> {
    let (m, hash) = tiny_model();
    assert_eq!(m.layers.iter().filter(|l| matches!(l.attn, AttnW::Full(_))).count(), 12);
    assert_eq!(m.layers.iter().filter(|l| l.ple.is_some()).count(), 1); // layer index 1
    let lookup = |row: u64, out: &mut [f32]| -> Result<()> {
        for (j, o) in out.iter_mut().enumerate() {
            *o = (splitmix64(row ^ (j as u64 + 1)) % 97) as f32 * 0.002 - 0.1;
        }
        Ok(())
    };
    let tokens = [1u32, 5, 31, 7, 9]; // includes the tiny EOS restart
    let run = |hash: PleHash| -> Result<Vec<u32>> {
        let mut st = ModelState::new(&m, hash);
        let logits = forward(&m, &mut st, &tokens, &DenseIndexer, &lookup)?;
        assert!(logits.iter().all(|v| v.is_finite()));
        Ok(logits.iter().map(|v| v.to_bits()).collect())
    };
    let a = run(PleHash::new(hash.mult.clone(), hash.sizes.clone(), hash.offsets.clone(), 31))?;
    let b = run(PleHash::new(hash.mult.clone(), hash.sizes.clone(), hash.offsets.clone(), 31))?;
    assert_eq!(a, b, "same bytes twice");
    // Chunked calls equal one call (state carries across forward calls).
    let mut st = ModelState::new(&m, PleHash::new(hash.mult, hash.sizes, hash.offsets, 31));
    forward(&m, &mut st, &tokens[..2], &DenseIndexer, &lookup)?;
    let tail = forward(&m, &mut st, &tokens[2..], &DenseIndexer, &lookup)?;
    assert_eq!(tail.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), a);
    Ok(())
}

