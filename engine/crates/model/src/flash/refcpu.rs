//! FP32 CPU reference for the qwen4_exp_text decoder (batch 1, token order).
//! Formulas follow the plain-PyTorch reference `modeling_qwen4_exp.py`
//! (transformers b0cdc61); comments cite it as TF:line. Weights are FP32
//! slices keyed by the loader's tensor names; quantization enters only
//! through the existing NVFP4 oracle (`super::nvfp4::dequantize`).
use super::nvfp4;
use crate::{Result, invalid};

fn need(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(invalid(format!("refcpu: {what}")))
    }
}
/// Row-major FP32 matrix `[r][c]`, applied as `y = W x` (nn.Linear layout).
#[derive(Debug, Clone, PartialEq)]
pub struct M {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f32>,
}
impl M {
    pub fn new(r: usize, c: usize, d: Vec<f32>) -> Result<Self> {
        need(
            d.len() == r * c && d.iter().all(|v| v.is_finite()),
            "matrix shape/finite",
        )?;
        Ok(Self { r, c, d })
    }
    pub fn zeros(r: usize, c: usize) -> Self {
        Self {
            r,
            c,
            d: vec![0.0; r * c],
        }
    }
    pub fn mv(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        need(x.len() == self.c && y.len() == self.r, "matvec shape")?;
        for (i, row) in self.d.chunks_exact(self.c).enumerate() {
            y[i] = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        Ok(())
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
/// TF:570-572 uses F.softplus; this stable form equals ln(1+e^x).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
pub fn softmax(x: &[f32], y: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for (o, &v) in y.iter_mut().zip(x) {
        *o = (v - m).exp();
        s += *o;
    }
    for o in y.iter_mut() {
        *o /= s;
    }
}
/// Grouped RMSNorm (TF:143-163): per `group`, `x*rsqrt(mean(x^2)+eps)`,
/// times `(1+w)` when `one_plus`, else plain `w` (gated norm, TF:171-187).
pub fn rms_group(
    x: &[f32],
    w: &[f32],
    group: usize,
    one_plus: bool,
    eps: f32,
    y: &mut [f32],
) -> Result<()> {
    need(
        group > 0 && x.len() == w.len() && x.len() == y.len() && x.len() % group == 0,
        "rms shape",
    )?;
    for (xc, (wc, yc)) in x
        .chunks_exact(group)
        .zip(w.chunks_exact(group).zip(y.chunks_exact_mut(group)))
    {
        let s = 1.0 / (xc.iter().map(|v| v * v).sum::<f32>() / group as f32 + eps).sqrt();
        for ((&v, &wi), o) in xc.iter().zip(wc).zip(yc.iter_mut()) {
            *o = v * s * if one_plus { 1.0 + wi } else { wi };
        }
    }
    Ok(())
}
/// Architecture dimensions; `real()` is the FP4-Local checkpoint contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dims {
    pub hidden: usize,
    pub hc: usize,
    pub low: usize,
    pub qh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub rot: usize,
    pub kh: usize,
    pub vh: usize,
    pub kd: usize,
    pub vd: usize,
    pub experts: usize,
    pub topk: usize,
    pub ew: usize,
    pub sw: usize,
    pub vocab: u64,
    pub eos: u64,
    pub eps: f32,
    pub theta: f32,
}
impl Dims {
    pub const fn real() -> Self {
        Self {
            hidden: 2560,
            hc: 4,
            low: 320,
            qh: 24,
            kvh: 2,
            hd: 256,
            rot: 64,
            kh: 16,
            vh: 48,
            kd: 128,
            vd: 128,
            experts: 512,
            topk: 10,
            ew: 640,
            sw: 640,
            vocab: 248320,
            eos: 248044,
            eps: 1e-6,
            theta: 1e7,
        }
    }
    pub fn validate(&self) -> Result<()> {
        need(
            self.hidden > 0
                && self.hc > 0
                && self.low > 0
                && self.qh % self.kvh.max(1) == 0
                && self.kvh > 0
                && self.rot <= self.hd
                && self.rot % 2 == 0
                && self.vh % self.kh.max(1) == 0
                && self.kh > 0
                && self.topk <= self.experts
                && self.eps > 0.0
                && self.theta > 1.0,
            "dims",
        )
    }
    pub fn wide(&self) -> usize {
        self.hc * self.hidden
    }
    pub fn conv(&self) -> usize {
        2 * self.kh * self.kd + self.vh * self.vd
    }
}
/// Real schedule: full attention every 4th layer (36 linear + 12 full of 48).
pub fn layer_is_full(layer: usize) -> bool {
    layer % 4 == 3
}
pub fn tensor_names_hc(prefix: &str, inject: bool) -> Vec<String> {
    let mut v: Vec<String> = [
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
    ]
    .iter()
    .map(|s| format!("{prefix}.{s}"))
    .collect();
    if inject {
        v.push(format!("{prefix}.block_inject_weight.weight"));
    }
    v
}

// ---- (1) Hyper-connections: GatedResidual, TF:994-1022 ----
/// Weights of one `{attn,mlp}_hyper_connection` (or the final mixer).
pub struct HcW {
    pub norm: Vec<f32>,
    pub down: M,
    pub up: M,
    pub inject: Option<M>,
}
/// Xn=RMSNorm_{1+w}(X) per stream; a=silu(down·Xn/hc); g=sigmoid(up·a);
/// u=mean_k(g_k ⊙ Xn_k); r=2·sigmoid(inj·Xn/hc). Returns (u, r).
pub fn hc_mix(d: &Dims, w: &HcW, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let wide = d.wide();
    need(
        x.len() == wide
            && w.norm.len() == wide
            && w.down.r == d.low
            && w.down.c == wide
            && w.up.r == wide
            && w.up.c == d.low,
        "hc shape",
    )?;
    let mut xn = vec![0.0; wide];
    rms_group(x, &w.norm, d.hidden, true, d.eps, &mut xn)?;
    let mut a = vec![0.0; d.low];
    w.down.mv(&xn, &mut a)?;
    for v in a.iter_mut() {
        *v = silu(*v / d.hc as f32);
    }
    let mut g = vec![0.0; wide];
    w.up.mv(&a, &mut g)?;
    let mut u = vec![0.0; d.hidden];
    for k in 0..d.hc {
        for i in 0..d.hidden {
            u[i] += sigmoid(g[k * d.hidden + i]) * xn[k * d.hidden + i];
        }
    }
    for v in u.iter_mut() {
        *v /= d.hc as f32;
    }
    let r = match &w.inject {
        None => Vec::new(),
        Some(m) => {
            need(m.r == d.hc && m.c == wide, "hc inject shape")?;
            let mut t = vec![0.0; d.hc];
            m.mv(&xn, &mut t)?;
            t.iter().map(|&v| 2.0 * sigmoid(v / d.hc as f32)).collect()
        }
    };
    Ok((u, r))
}
/// X_k += r_k · block_out, on the un-normalized residual (TF:1293-1300).
pub fn hc_combine(d: &Dims, x: &mut [f32], r: &[f32], block: &[f32]) -> Result<()> {
    need(
        x.len() == d.wide() && r.len() == d.hc && block.len() == d.hidden,
        "hc combine shape",
    )?;
    for k in 0..d.hc {
        for i in 0..d.hidden {
            x[k * d.hidden + i] += r[k] * block[i];
        }
    }
    Ok(())
}

// ---- (2) Output gates (output_gate_type = sigmoid) ----
/// GDN head gate (TF:171-187,490-492): (RMSNorm(o)·w) ⊙ sigmoid(z), per head.
pub fn gated_norm(
    o: &[f32],
    w: &[f32],
    z: &[f32],
    head_dim: usize,
    eps: f32,
    y: &mut [f32],
) -> Result<()> {
    need(
        o.len() == z.len() && o.len() == y.len() && w.len() == head_dim && o.len() % head_dim == 0,
        "gate shape",
    )?;
    let wr: Vec<f32> = w.iter().copied().cycle().take(o.len()).collect();
    rms_group(o, &wr, head_dim, false, eps, y)?;
    for (v, &zi) in y.iter_mut().zip(z) {
        *v *= sigmoid(zi);
    }
    Ok(())
}
/// Full-attention output gate (TF:858-861,889): attn ⊙ sigmoid(gate).
pub fn output_gate(attn: &mut [f32], gate: &[f32]) -> Result<()> {
    need(attn.len() == gate.len(), "output gate shape")?;
    for (v, &g) in attn.iter_mut().zip(gate) {
        *v *= sigmoid(g);
    }
    Ok(())
}

// ---- (3) Full attention: partial rotary + GQA + gate, TF:810-892 ----
/// inv_freq_i = theta^(-2i/rot), i<rot/2 (TF:107-115); cos/sin doubled
/// by cat(freqs, freqs) to length rot (TF:131-140, text positions equal).
pub fn rope_cos_sin(theta: f32, rot: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half: Vec<f32> = (0..rot / 2)
        .map(|i| pos as f32 / theta.powf(2.0 * i as f32 / rot as f32))
        .collect();
    let cos: Vec<f32> = half.iter().chain(&half).map(|f| f.cos()).collect();
    let sin: Vec<f32> = half.iter().chain(&half).map(|f| f.sin()).collect();
    (cos, sin)
}
/// Rotate-half on the first `rot` dims only; the rest pass through (TF:626-661).
pub fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let rot = cos.len();
    let h = rot / 2;
    let old: Vec<f32> = x[..rot].to_vec();
    for i in 0..rot {
        let rh = if i < h { -old[i + h] } else { old[i - h] };
        x[i] = old[i] * cos[i] + rh * sin[i];
    }
}
/// Sparse-attention indexer hook (lane flashnext-indexer owns `indexer_*`).
/// `None` means every causally visible position is attended (dense).
pub trait IndexerHook {
    fn select(&self, layer: usize, pos: usize, kv_len: usize) -> Option<Vec<usize>>;
}
pub struct DenseIndexer;
impl IndexerHook for DenseIndexer {
    fn select(&self, _l: usize, _p: usize, _n: usize) -> Option<Vec<usize>> {
        None
    }
}
pub fn tensor_names_indexer(prefix: &str) -> Vec<String> {
    [
        "indexer.index_qk_proj.weight",
        "indexer.q_layernorm.weight",
        "indexer.k_layernorm.weight",
    ]
    .iter()
    .map(|s| format!("{prefix}.{s}"))
    .collect()
}
pub struct FullW {
    pub q: M,
    pub k: M,
    pub v: M,
    pub o: M,
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
}
#[derive(Default)]
pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
} // per position: [kvh*hd]
/// One token: q_proj interleaved per head as (query|gate) 512-wide rows at
/// real dims (TF:858-861); q/k RMSNorm_{1+w} then partial RoPE; causal GQA
/// softmax at scale 1/sqrt(hd); output gate; o_proj.
pub fn full_attn(
    d: &Dims,
    w: &FullW,
    cache: &mut KvCache,
    idx: &dyn IndexerHook,
    layer: usize,
    pos: usize,
    u: &[f32],
) -> Result<Vec<f32>> {
    need(
        w.q.r == d.qh * 2 * d.hd
            && w.k.r == d.kvh * d.hd
            && w.o.r == d.hidden
            && w.qn.len() == d.hd
            && w.kn.len() == d.hd,
        "full-attn shape",
    )?;
    let (cos, sin) = rope_cos_sin(d.theta, d.rot, pos);
    let mut qg = vec![0.0; w.q.r];
    w.q.mv(u, &mut qg)?;
    let mut qs = vec![0.0; d.qh * d.hd];
    let mut gate = vec![0.0; d.qh * d.hd];
    for h in 0..d.qh {
        qs[h * d.hd..(h + 1) * d.hd].copy_from_slice(&qg[h * 2 * d.hd..h * 2 * d.hd + d.hd]);
        gate[h * d.hd..(h + 1) * d.hd]
            .copy_from_slice(&qg[h * 2 * d.hd + d.hd..(h + 1) * 2 * d.hd]);
    }
    let mut ks = vec![0.0; d.kvh * d.hd];
    let mut vs = vec![0.0; d.kvh * d.hd];
    w.k.mv(u, &mut ks)?;
    w.v.mv(u, &mut vs)?;
    for h in 0..d.qh {
        let (mut t, mut n) = (qs[h * d.hd..(h + 1) * d.hd].to_vec(), vec![0.0; d.hd]);
        rms_group(&t, &w.qn, d.hd, true, d.eps, &mut n)?;
        t.copy_from_slice(&n);
        apply_rope(&mut t, &cos, &sin);
        qs[h * d.hd..(h + 1) * d.hd].copy_from_slice(&t);
    }
    for h in 0..d.kvh {
        let (mut t, mut n) = (ks[h * d.hd..(h + 1) * d.hd].to_vec(), vec![0.0; d.hd]);
        rms_group(&t, &w.kn, d.hd, true, d.eps, &mut n)?;
        t.copy_from_slice(&n);
        apply_rope(&mut t, &cos, &sin);
        ks[h * d.hd..(h + 1) * d.hd].copy_from_slice(&t);
    }
    cache.k.push(ks);
    cache.v.push(vs);
    let kv_len = cache.k.len();
    let visible: Vec<usize> = idx
        .select(layer, pos, kv_len)
        .unwrap_or_else(|| (0..kv_len).collect());
    need(visible.iter().all(|&p| p < kv_len), "indexer range")?;
    let group = d.qh / d.kvh;
    let scale = 1.0 / (d.hd as f32).sqrt();
    let mut attn = vec![0.0; d.qh * d.hd];
    for h in 0..d.qh {
        let kv = h / group;
        let q = &qs[h * d.hd..(h + 1) * d.hd];
        let scores: Vec<f32> = visible
            .iter()
            .map(|&p| {
                q.iter()
                    .zip(&cache.k[p][kv * d.hd..(kv + 1) * d.hd])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let mut probs = vec![0.0; scores.len()];
        softmax(&scores, &mut probs);
        for (j, &p) in visible.iter().enumerate() {
            for i in 0..d.hd {
                attn[h * d.hd + i] += probs[j] * cache.v[p][kv * d.hd + i];
            }
        }
    }
    output_gate(&mut attn, &gate)?;
    let mut y = vec![0.0; d.hidden];
    w.o.mv(&attn, &mut y)?;
    Ok(y)
}

// ---- Gated DeltaNet recurrence, TF:391-455 + layer TF:456-616 ----
pub struct GdnW {
    pub qkv: M,
    pub z: M,
    pub a: M,
    pub b: M,
    pub conv: Vec<f32>, // [conv][4]
    pub alog: Vec<f32>,
    pub dt: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: M,
}
pub struct GdnState {
    pub s: Vec<f32>,
    pub hist: Vec<f32>,
} // s: [vh][kd][vd]; hist: last 3 raw qkv rows
impl GdnState {
    pub fn new(d: &Dims) -> Self {
        Self {
            s: vec![0.0; d.vh * d.kd * d.vd],
            hist: vec![0.0; 3 * d.conv()],
        }
    }
}
fn l2norm(v: &mut [f32]) {
    // TF:247-250: x * rsqrt(sum(x^2)+1e-6)
    let s = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}
/// Depthwise causal conv k=4 + SiLU over [hist(3), cur], then one recurrent
/// step per value head: S*=e^g; delta=(v-S^T k)*beta; S+=k⊗delta; o=S^T q.
pub fn gdn_token(d: &Dims, w: &GdnW, st: &mut GdnState, u: &[f32]) -> Result<Vec<f32>> {
    let conv = d.conv();
    need(
        w.qkv.r == conv
            && w.z.r == d.vh * d.vd
            && w.a.r == d.vh
            && w.conv.len() == conv * 4
            && w.alog.len() == d.vh
            && w.norm.len() == d.vd,
        "gdn shape",
    )?;
    let (mut raw, mut z, mut av, mut bv) = (
        vec![0.0; conv],
        vec![0.0; d.vh * d.vd],
        vec![0.0; d.vh],
        vec![0.0; d.vh],
    );
    w.qkv.mv(u, &mut raw)?;
    w.z.mv(u, &mut z)?;
    w.a.mv(u, &mut av)?;
    w.b.mv(u, &mut bv)?;
    let mut c = vec![0.0; conv];
    for ch in 0..conv {
        let mut acc = 0.0;
        for tap in 0..4 {
            // sample at t+tap-3, weight [ch][tap] (TF conv1d padding 3)
            let src = if tap < 3 {
                st.hist[tap * conv + ch]
            } else {
                raw[ch]
            };
            acc += src * w.conv[ch * 4 + tap];
        }
        c[ch] = silu(acc);
    }
    st.hist.copy_within(conv..3 * conv, 0);
    st.hist[2 * conv..].copy_from_slice(&raw);
    let kd_all = d.kh * d.kd;
    let repeat = d.vh / d.kh;
    let mut core = vec![0.0; d.vh * d.vd];
    for h in 0..d.vh {
        let kh = h / repeat;
        let mut q: Vec<f32> = c[kh * d.kd..(kh + 1) * d.kd].to_vec();
        let mut k: Vec<f32> = c[kd_all + kh * d.kd..kd_all + (kh + 1) * d.kd].to_vec();
        let v = &c[2 * kd_all + h * d.vd..2 * kd_all + (h + 1) * d.vd];
        l2norm(&mut q);
        l2norm(&mut k);
        for x in q.iter_mut() {
            *x /= (d.kd as f32).sqrt();
        }
        let beta = sigmoid(bv[h]);
        let g = -w.alog[h].exp() * softplus(av[h] + w.dt[h]);
        let s = &mut st.s[h * d.kd * d.vd..(h + 1) * d.kd * d.vd];
        let decay = g.exp();
        for x in s.iter_mut() {
            *x *= decay;
        }
        let mut delta = vec![0.0; d.vd];
        for j in 0..d.vd {
            let mem: f32 = (0..d.kd).map(|i| s[i * d.vd + j] * k[i]).sum();
            delta[j] = (v[j] - mem) * beta;
        }
        for i in 0..d.kd {
            for j in 0..d.vd {
                s[i * d.vd + j] += k[i] * delta[j];
            }
        }
        for j in 0..d.vd {
            core[h * d.vd + j] = (0..d.kd).map(|i| s[i * d.vd + j] * q[i]).sum();
        }
    }
    let mut gated = vec![0.0; d.vh * d.vd];
    gated_norm(&core, &w.norm, &z, d.vd, d.eps, &mut gated)?;
    let mut y = vec![0.0; d.hidden];
    w.out.mv(&gated, &mut y)?;
    Ok(y)
}

// ---- (4) PLE n-gram hashing, TF:1025-1171; layer TF:1174-1246 ----
pub fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}
/// m_j = 2*(splitmix64(seed0+GAMMA*(j+1)) mod half)+1 (TF:1032-1048).
pub fn ple_multipliers(vocab: u64, ngram: usize, ple_index: u64, seed: u64) -> Vec<i64> {
    let half = (((i64::MAX as u64) / vocab.max(1)) / 2).max(1);
    let base = seed.wrapping_add(10007u64.wrapping_mul(ple_index));
    (0..ngram)
        .map(|j| {
            let v = base.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(j as u64 + 1));
            (2 * (splitmix64(v) % half) + 1) as i64
        })
        .collect()
}
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}
/// Head h vocab = the (global_h+1)-th prime after base-1; cumulative
/// offsets; total padded up to `divisible` (TF:1087-1104).
pub fn ple_vocab(
    base: u64,
    ple_index: u64,
    heads: usize,
    divisible: u64,
) -> (Vec<i64>, Vec<i64>, u64) {
    let (mut sizes, mut offsets, mut total) = (Vec::new(), Vec::new(), 0u64);
    for h in 0..heads as u64 {
        let mut p = base - 1;
        for _ in 0..(ple_index * heads as u64 + h + 1) {
            p += 1;
            while !is_prime(p) {
                p += 1;
            }
        }
        sizes.push(p as i64);
        offsets.push(total as i64);
        total += p;
    }
    (
        sizes,
        offsets,
        total.div_ceil(divisible.max(1)) * divisible.max(1),
    )
}
/// EOS-bounded n-gram rows. `advance` mirrors TF:1106-1163 segment logic:
/// an EOS token still sees its segment; the next token restarts from EOS.
#[derive(Clone)]
pub struct PleHash {
    pub mult: Vec<i64>,
    pub sizes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub eos: i64,
    hist: [i64; 2],
}
impl PleHash {
    pub fn new(mult: Vec<i64>, sizes: Vec<i64>, offsets: Vec<i64>, eos: i64) -> Self {
        Self {
            mult,
            sizes,
            offsets,
            eos,
            hist: [eos; 2],
        }
    }
    pub fn reset(&mut self) {
        self.hist = [self.eos; 2];
    }
    /// Pure rows for (current, previous, two-back); callers pass EOS where
    /// a shift leaves the segment. Heads 0..8 bigrams, 8..16 trigrams at
    /// real dims (split = heads/2 in general here: (ngram-1) blocks).
    pub fn rows_pure(&self, cur: i64, p1: i64, p2: i64) -> Vec<u64> {
        let bigram = cur.wrapping_mul(self.mult[0]) ^ p1.wrapping_mul(self.mult[1]);
        let trigram = bigram ^ p2.wrapping_mul(self.mult[2]);
        let half = self.sizes.len() / 2;
        (0..self.sizes.len())
            .map(|h| {
                let hash = if h < half { bigram } else { trigram };
                (hash.rem_euclid(self.sizes[h]) + self.offsets[h]) as u64
            })
            .collect()
    }
    /// Advance canonical history without materializing rows on the host.
    pub fn advance_history(&mut self, token: u32) -> [i64; 2] {
        let prev = if self.hist[0] == self.eos {
            [self.eos; 2]
        } else {
            self.hist
        };
        self.hist = [token as i64, prev[0]];
        prev
    }
    pub fn advance(&mut self, token: u32) -> Vec<u64> {
        let prev = self.advance_history(token);
        self.rows_pure(token as i64, prev[0], prev[1])
    }
}
/// NVFP4 PLE row through the loader oracle: e2m1 nibbles (low = even) with
/// E4M3 scales per 16 and the shared FP32 global (SOURCES §6).
pub fn ple_row_nvfp4(packed: &[u8], scales: &[u8], global: f32, out: &mut [f32]) -> Result<()> {
    nvfp4::dequantize(packed, scales, global, out)
}
pub fn ple_shard(row: u64) -> u64 {
    row / super::PLE_PART_ROWS
}
pub struct PleW {
    pub key: M,
    pub value: M,
    pub nk: Vec<f32>,
    pub nq: Vec<f32>,
    pub nc: Vec<f32>,
    pub conv: Vec<f32>, // [hc*hidden][4], dilation = ngram_size = 3
}
pub struct PleState {
    pub hist: Vec<f32>,
} // 9 rows of normalized gated values
impl PleState {
    pub fn new(d: &Dims) -> Self {
        Self {
            hist: vec![0.0; 9 * d.wide()],
        }
    }
}
/// One token (TF:1190-1245): K=GNorm(key·e), V=value·e, Q=GNorm(X);
/// s_k=<K_k,Q_k>/sqrt(H) -> sign·sqrt(max(|s|,1e-6)); U_k=sigmoid(s_k)·V;
/// out = U + silu(depthwise conv k=4 dil=3 over GNorm(U)). Added to X.
pub fn ple_token(d: &Dims, w: &PleW, st: &mut PleState, x: &mut [f32], e: &[f32]) -> Result<()> {
    let wide = d.wide();
    need(
        x.len() == wide && w.key.r == wide && w.value.r == d.hidden && w.conv.len() == wide * 4,
        "ple shape",
    )?;
    let (mut kr, mut v, mut kn, mut qn) = (
        vec![0.0; wide],
        vec![0.0; d.hidden],
        vec![0.0; wide],
        vec![0.0; wide],
    );
    w.key.mv(e, &mut kr)?;
    w.value.mv(e, &mut v)?;
    rms_group(&kr, &w.nk, d.hidden, true, d.eps, &mut kn)?;
    rms_group(x, &w.nq, d.hidden, true, d.eps, &mut qn)?;
    let mut u = vec![0.0; wide];
    for k in 0..d.hc {
        let dot: f32 = (0..d.hidden)
            .map(|i| kn[k * d.hidden + i] * qn[k * d.hidden + i])
            .sum();
        let s = dot / (d.hidden as f32).sqrt();
        let s = s.signum() * s.abs().max(1e-6).sqrt();
        let g = sigmoid(s);
        for i in 0..d.hidden {
            u[k * d.hidden + i] = g * v[i];
        }
    }
    let mut un = vec![0.0; wide];
    rms_group(&u, &w.nc, d.hidden, true, d.eps, &mut un)?;
    for ch in 0..wide {
        // out[t] = silu(w0·in[t-9]+w1·in[t-6]+w2·in[t-3]+w3·in[t])
        let acc = w.conv[ch * 4] * st.hist[ch]
            + w.conv[ch * 4 + 1] * st.hist[3 * wide + ch]
            + w.conv[ch * 4 + 2] * st.hist[6 * wide + ch]
            + w.conv[ch * 4 + 3] * un[ch];
        x[ch] += u[ch] + silu(acc);
    }
    st.hist.copy_within(wide..9 * wide, 0);
    st.hist[8 * wide..].copy_from_slice(&un);
    Ok(())
}

// ---- (5) MoE: router TF:951-991; experts via the NVFP4 oracle ----
/// softmax in FP32, top-k by descending prob (ties: lower expert id first,
/// a documented choice where torch.topk is unspecified), renormalized.
pub fn route(logits: &[f32], topk: usize) -> (Vec<usize>, Vec<f32>) {
    let mut p = vec![0.0; logits.len()];
    softmax(logits, &mut p);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    order.truncate(topk);
    let s: f32 = order.iter().map(|&i| p[i]).sum();
    let w: Vec<f32> = order.iter().map(|&i| p[i] / s).collect();
    (order, w)
}
/// One routed expert: down(silu(gate·x) ⊙ up·x) (TF:951-991 expert MLP).
pub trait ExpertOracle {
    fn expert(&self, expert: usize, x: &[f32], y: &mut [f32]) -> Result<()>;
}
pub struct DenseExpert {
    pub gate: M,
    pub up: M,
    pub down: M,
}
pub fn dense_expert(e: &DenseExpert, x: &[f32], y: &mut [f32]) -> Result<()> {
    let (mut g, mut u) = (vec![0.0; e.gate.r], vec![0.0; e.up.r]);
    e.gate.mv(x, &mut g)?;
    e.up.mv(x, &mut u)?;
    for i in 0..g.len() {
        g[i] = silu(g[i]) * u[i];
    }
    e.down.mv(&g, y)
}
impl ExpertOracle for Vec<DenseExpert> {
    fn expert(&self, expert: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        dense_expert(
            self.get(expert)
                .ok_or_else(|| invalid("refcpu: expert range"))?,
            x,
            y,
        )
    }
}
/// Packed NVFP4 expert set: each projection dequantized row-block by
/// `nvfp4::dequantize` (the existing oracle entry point), then FP32 matvec.
/// Layout per expert/projection: packed `[r][c/2]`, scales `[r][c/16]`, global.
pub struct Nvfp4Expert {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub global: f32,
    pub r: usize,
    pub c: usize,
}
pub struct Nvfp4Experts {
    pub gate: Vec<Nvfp4Expert>,
    pub up: Vec<Nvfp4Expert>,
    pub down: Vec<Nvfp4Expert>,
}
fn nvfp4_mv(e: &Nvfp4Expert, x: &[f32], y: &mut [f32]) -> Result<()> {
    need(
        x.len() == e.c && y.len() == e.r && e.c % 16 == 0,
        "nvfp4 expert shape",
    )?;
    let mut w = vec![0.0; e.r * e.c];
    nvfp4::dequantize(&e.packed, &e.scales, e.global, &mut w)?;
    M {
        r: e.r,
        c: e.c,
        d: w,
    }
    .mv(x, y)
}
impl ExpertOracle for Nvfp4Experts {
    fn expert(&self, expert: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        need(
            expert < self.gate.len() && expert < self.up.len() && expert < self.down.len(),
            "expert range",
        )?;
        let (mut g, mut u) = (vec![0.0; self.gate[expert].r], vec![0.0; self.up[expert].r]);
        nvfp4_mv(&self.gate[expert], x, &mut g)?;
        nvfp4_mv(&self.up[expert], x, &mut u)?;
        for i in 0..g.len() {
            g[i] = silu(g[i]) * u[i];
        }
        nvfp4_mv(&self.down[expert], &g, y)
    }
}
pub struct MoeW {
    pub router: M,
    pub shared: DenseExpert,
    pub shared_gate: M,
}
pub fn moe_token(d: &Dims, w: &MoeW, experts: &dyn ExpertOracle, x: &[f32]) -> Result<Vec<f32>> {
    Ok(moe_token_routed(d, w, experts, x)?.0)
}
/// Weighted experts accumulated in ascending expert-id order (TF expert_hit
/// order), then shared·sigmoid(shared_gate·x) added (TF:987-989). Also
/// returns the ascending (expert, weight) route for debug dumps.
pub fn moe_token_routed(
    d: &Dims,
    w: &MoeW,
    experts: &dyn ExpertOracle,
    x: &[f32],
) -> Result<(Vec<f32>, Vec<(usize, f32)>)> {
    need(w.router.r == d.experts && w.shared_gate.r == 1, "moe shape")?;
    let mut logits = vec![0.0; d.experts];
    w.router.mv(x, &mut logits)?;
    let (ids, weights) = route(&logits, d.topk);
    let mut pairs: Vec<(usize, f32)> = ids.into_iter().zip(weights).collect();
    pairs.sort_by_key(|&(i, _)| i);
    let mut y = vec![0.0; d.hidden];
    let mut tmp = vec![0.0; d.hidden];
    for &(id, wt) in &pairs {
        experts.expert(id, x, &mut tmp)?;
        for i in 0..d.hidden {
            y[i] += wt * tmp[i];
        }
    }
    let mut sh = vec![0.0; d.hidden];
    dense_expert(&w.shared, x, &mut sh)?;
    let mut sg = [0.0; 1];
    w.shared_gate.mv(x, &mut sg)?;
    let s = sigmoid(sg[0]);
    for i in 0..d.hidden {
        y[i] += s * sh[i];
    }
    Ok((y, pairs))
}

// ---- (6) Whole-model driver, TF:1264-1301 layer order, TF:1471/1484 ----
pub enum AttnW {
    Gdn(GdnW),
    Full(FullW),
}
pub struct LayerW {
    pub attn_hc: HcW,
    pub mlp_hc: HcW,
    pub attn: AttnW,
    pub moe: MoeW,
    pub experts: Box<dyn ExpertOracle>,
    pub ple: Option<PleW>,
}
pub struct ModelW {
    pub dims: Dims,
    pub embed: M,
    pub layers: Vec<LayerW>,
    pub mixer: HcW,
    pub head: M,
}
pub struct ModelState {
    pub gdn: Vec<Option<GdnState>>,
    pub kv: Vec<Option<KvCache>>,
    pub ple: Option<PleState>,
    pub hash: PleHash,
    pub pos: usize,
}
impl ModelState {
    pub fn new(m: &ModelW, hash: PleHash) -> Self {
        Self {
            gdn: m
                .layers
                .iter()
                .map(|l| matches!(l.attn, AttnW::Gdn(_)).then(|| GdnState::new(&m.dims)))
                .collect(),
            kv: m
                .layers
                .iter()
                .map(|l| matches!(l.attn, AttnW::Full(_)).then(KvCache::default))
                .collect(),
            ple: m
                .layers
                .iter()
                .any(|l| l.ple.is_some())
                .then(|| PleState::new(&m.dims)),
            hash,
            pos: 0,
        }
    }
}
/// Deterministic tiny-weight filler (splitmix64 stream, values in (-s, s)).
pub fn fill_random(seed: u64, n: usize, s: f32) -> Vec<f32> {
    let mut v = seed;
    (0..n)
        .map(|_| {
            v = splitmix64(v);
            ((v >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s
        })
        .collect()
}
/// One debug-dump event from `forward_traced`; stage names match the GPU
/// writer (`embed`, `layer<L>.ple_e`, `layer<L>.ple`, `layer<L>.attn`,
/// `layer<L>.x`, `final`), with each layer's route and each token's logits.
pub enum Trace<'a> {
    Stage(String, &'a [f32]),
    Moe(usize, &'a [(usize, f32)]),
    Logits(&'a [f32]),
}
/// Tokens in order; PLE embeddings come from `ple_lookup(row, out[head_w])`
/// concatenated over hash heads. Returns last-token logits.
pub fn forward(
    m: &ModelW,
    st: &mut ModelState,
    tokens: &[u32],
    idx: &dyn IndexerHook,
    ple_lookup: &dyn Fn(u64, &mut [f32]) -> Result<()>,
) -> Result<Vec<f32>> {
    forward_traced(m, st, tokens, idx, ple_lookup, &mut |_| {})
}
pub fn forward_traced(
    m: &ModelW,
    st: &mut ModelState,
    tokens: &[u32],
    idx: &dyn IndexerHook,
    ple_lookup: &dyn Fn(u64, &mut [f32]) -> Result<()>,
    trace: &mut dyn FnMut(Trace),
) -> Result<Vec<f32>> {
    let d = &m.dims;
    d.validate()?;
    need(
        !tokens.is_empty() && tokens.iter().all(|&t| (t as u64) < d.vocab),
        "tokens",
    )?;
    let heads = st.hash.sizes.len();
    need(heads > 0 && d.hidden % heads == 0, "ple head width")?;
    let head_w = d.hidden / heads;
    let mut logits = Vec::new();
    for &tok in tokens {
        let rows = st.hash.advance(tok);
        let mut x: Vec<f32> = (0..d.hc)
            .flat_map(|_| m.embed.row(tok as usize).iter().copied())
            .collect();
        trace(Trace::Stage("embed".into(), &x));
        for (li, l) in m.layers.iter().enumerate() {
            if let Some(pw) = &l.ple {
                let mut e = vec![0.0; d.hidden];
                for (h, &r) in rows.iter().enumerate() {
                    ple_lookup(r, &mut e[h * head_w..(h + 1) * head_w])?;
                }
                trace(Trace::Stage(format!("layer{li}.ple_e"), &e));
                ple_token(
                    d,
                    pw,
                    st.ple
                        .as_mut()
                        .ok_or_else(|| invalid("refcpu: ple state"))?,
                    &mut x,
                    &e,
                )?;
                trace(Trace::Stage(format!("layer{li}.ple"), &x));
            }
            let (u, r) = hc_mix(d, &l.attn_hc, &x)?;
            let blk = match &l.attn {
                AttnW::Gdn(w) => gdn_token(
                    d,
                    w,
                    st.gdn[li]
                        .as_mut()
                        .ok_or_else(|| invalid("refcpu: gdn state"))?,
                    &u,
                )?,
                AttnW::Full(w) => full_attn(
                    d,
                    w,
                    st.kv[li]
                        .as_mut()
                        .ok_or_else(|| invalid("refcpu: kv state"))?,
                    idx,
                    li,
                    st.pos,
                    &u,
                )?,
            };
            trace(Trace::Stage(format!("layer{li}.attn"), &blk));
            hc_combine(d, &mut x, &r, &blk)?;
            let (u, r) = hc_mix(d, &l.mlp_hc, &x)?;
            let (blk, pairs) = moe_token_routed(d, &l.moe, &*l.experts, &u)?;
            trace(Trace::Moe(li, &pairs));
            hc_combine(d, &mut x, &r, &blk)?;
            trace(Trace::Stage(format!("layer{li}.x"), &x));
        }
        let (h, _) = hc_mix(d, &m.mixer, &x)?; // final mixer is the last norm; no separate final RMSNorm
        trace(Trace::Stage("final".into(), &h));
        logits = vec![0.0; m.head.r];
        m.head.mv(&h, &mut logits)?;
        trace(Trace::Logits(&logits));
        st.pos += 1;
    }
    Ok(logits)
}
