//! Qwen layer arithmetic shared by local execution and tensor-parallel ranks.
use spark_gpu_probe::driver::{Backend, DeviceVector, KvFormat, Matrix};

type Result<T> = std::result::Result<T, String>;

pub fn rope<'a>(b: &Backend<'a>, config: &engine_model::qwen3::Config, max_tokens: usize) -> Result<DeviceVector<'a>> {
    // FP32 inverse frequencies times FP32 positions, as in the reference model.
    let half = config.head_dim as usize / 2;
    let mut inverse_frequency: Vec<_> = (0..half).map(|i|
        1.0f32 / (config.rope_theta as f32).powf(i as f32 / half as f32)).collect();
    if let Some(r) = &config.llama_rope {
        for frequency in &mut inverse_frequency {
            let wavelength = 2.0 * std::f32::consts::PI / *frequency;
            if wavelength > r.original / r.low { *frequency /= r.factor; }
            else if wavelength >= r.original / r.high {
                let smooth = (r.original / wavelength - r.low) / (r.high - r.low);
                *frequency = (1.0 - smooth) * *frequency / r.factor + smooth * *frequency;
            }
        }
    }
    let mut attention_factor = 1.0f32;
    if config.yarn {
        // YaRN interpolates low frequencies and extrapolates high ones.
        // Official Qwen3 settings: factor=4, original=32768, beta_fast=32,
        // beta_slow=1, and the default 1 + 0.1*ln(factor) attention scale.
        let correction = |rotations: f64| config.head_dim as f64
            * (32768.0 / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * config.rope_theta.ln());
        let low = correction(32.0).floor().max(0.0) as f32;
        let high = correction(1.0).ceil().min((config.head_dim - 1) as f64) as f32;
        let span = if low == high { 0.001 } else { high - low };
        for (i, frequency) in inverse_frequency.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / span).clamp(0.0, 1.0);
            *frequency = (*frequency / 4.0) * ramp + *frequency * (1.0 - ramp);
        }
        attention_factor = (1.0 + 0.1 * 4.0f64.ln()) as f32;
    }
    let mut table = Vec::with_capacity(max_tokens * half * 2);
    for position in 0..max_tokens { for &frequency in &inverse_frequency {
        let (sin, cos) = (position as f32 * frequency).sin_cos();
        if config.yarn { table.extend([sin * attention_factor, cos * attention_factor]); }
        else { table.extend([sin, cos]); }
    }}
    b.upload_vector(&table)
}

pub struct Norms<'s, 'a> {
    pub input: &'s DeviceVector<'a>, pub post: &'s DeviceVector<'a>,
    pub q: &'s DeviceVector<'a>, pub k: &'s DeviceVector<'a>,
    pub next: Option<&'s DeviceVector<'a>>,
}

/// Empty projections belong to ranks with no heads or MLP rows.
pub struct Weights<'s, 'a> {
    pub q: Option<&'s Matrix<'a>>, pub k: Option<&'s Matrix<'a>>, pub v: Option<&'s Matrix<'a>>,
    pub o: Option<&'s Matrix<'a>>, pub gate: Option<&'s Matrix<'a>>,
    pub up: Option<&'s Matrix<'a>>, pub down: Option<&'s Matrix<'a>>,
}

pub struct Buffers<'s, 'a> {
    pub hidden: &'s mut DeviceVector<'a>, pub input: &'s mut DeviceVector<'a>,
    pub sums: &'s mut DeviceVector<'a>, pub q: &'s mut DeviceVector<'a>,
    pub k: &'s mut DeviceVector<'a>, pub v: &'s mut DeviceVector<'a>,
    pub attention: &'s mut DeviceVector<'a>, pub projected: &'s mut DeviceVector<'a>,
    pub gate: &'s mut DeviceVector<'a>, pub up: &'s mut DeviceVector<'a>,
    pub scores: &'s mut DeviceVector<'a>, pub parts: &'s mut DeviceVector<'a>,
    pub weight_pack: &'s mut DeviceVector<'a>, pub input_pack: &'s mut DeviceVector<'a>,
    pub packed_keys: &'s mut DeviceVector<'a>, pub packed_values: &'s mut DeviceVector<'a>,
}

#[derive(Clone, Copy)]
pub struct Run {
    pub position: usize, pub tokens: usize, pub first: bool, pub prompt_end: usize,
    pub heads: usize, pub kv_heads: usize, pub head_dim: usize,
    pub epsilon: f32, pub exact: bool, pub format: KvFormat, pub decode: bool,
    pub sums_ready: bool,
}

pub fn project(b: &Backend<'_>, weights: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, exact: bool, reuse: bool, decode: bool) -> Result<()> {
    if decode { b.matvec_device(weights, input, output) }
    else { b.matmul_reusing_input(weights, input, output, weight_pack, input_pack, tokens, exact, reuse) }
}

/// The hook completes each row-parallel O/down projection. True means the
/// residual was applied and either RMS partials or the supplied decode norm
/// output are ready; the latter uses the fourth (input) buffer.
pub fn forward(b: &Backend<'_>, weights: Weights<'_, '_>, norms: Norms<'_, '_>, rope: &DeviceVector<'_>,
    cache: Option<(&mut DeviceVector<'_>, &mut DeviceVector<'_>)>, s: Buffers<'_, '_>, r: Run,
    mut complete_projection: impl FnMut(&mut DeviceVector<'_>, &mut DeviceVector<'_>, &mut DeviceVector<'_>, &mut DeviceVector<'_>, Option<&DeviceVector<'_>>) -> Result<bool>) -> Result<()> {
    if r.decode && r.first { b.rms_device(s.hidden, norms.input, s.input, s.sums, r.epsilon)?; }
    if let Some((keys, values)) = cache {
        let (q, k, v) = (weights.q.ok_or("missing Q projection")?, weights.k.ok_or("missing K projection")?, weights.v.ok_or("missing V projection")?);
        if r.decode {
            b.qkv_device(q, k, v, s.input, s.q, s.k, s.v)?;
        } else if !r.exact && q.cols() % 128 == 0 && q.rows() % 128 == 0 && k.rows() % 128 == 0 {
            if !r.sums_ready { b.rms_partials_device(s.hidden, s.sums, norms.input.len())?; }
            b.rms_pack_apply_device(s.hidden, norms.input, s.input_pack, s.sums, r.epsilon, false)?;
            b.prefill_qkv_device(q, k, v, s.input_pack, s.q, s.k, s.v, r.tokens, false)?;
        } else {
            if !r.sums_ready { b.rms_partials_device(s.hidden, s.sums, norms.input.len())?; }
            b.rms_batch_apply_device(s.hidden, norms.input, s.input, s.sums, r.epsilon)?;
            for (weight, output, reuse) in [(q, &mut *s.q, false), (k, &mut *s.k, true), (v, &mut *s.v, true)] {
                project(b, weight, s.input, output, s.weight_pack, s.input_pack, r.tokens, r.exact, reuse, false)?;
            }
        }
        if r.decode && r.format == KvFormat::Fp32 {
            b.qk_rope_cache_device(s.q, s.k, s.v, norms.q, norms.k, rope, keys, values, r.position, r.epsilon)?;
            b.attention_device(s.q, keys, values, s.scores, s.parts, s.attention, r.heads, r.kv_heads, r.head_dim, r.position + 1)?;
        } else {
            b.qk_batch_device(s.q, s.k, s.v, norms.q, norms.k, rope, keys, values, r.position, r.epsilon, r.tokens, r.format)?;
            if !r.decode && !r.exact && r.head_dim == 128 {
                b.prefill_attention_device(s.q, keys, values, s.attention, s.packed_keys, s.packed_values,
                    r.heads, r.kv_heads, r.position, r.tokens, r.format, r.prompt_end, false)?;
            } else {
                b.attention_batch_device(s.q, keys, values, s.scores, s.parts, s.attention,
                    r.heads, r.kv_heads, r.head_dim, r.position, r.tokens, r.format)?;
            }
        }
    }
    if let Some(o) = weights.o { project(b, o, s.attention, s.projected, s.weight_pack, s.input_pack, r.tokens, r.exact, false, r.decode)?; }
    else { b.write_vector(s.projected, &vec![0.; s.projected.len()])?; }
    let residual_done=complete_projection(s.projected,s.hidden,s.sums,s.input,if r.decode{Some(norms.post)}else{None})?;
    if r.decode {
        if !residual_done { b.residual_rms_device(s.hidden, s.projected, norms.post, s.input, s.sums, r.epsilon)?; }
    }
    else if !residual_done { b.residual_device(s.hidden, s.projected)?; }
    if let (Some(gate), Some(up), Some(down)) = (weights.gate, weights.up, weights.down) {
        if r.decode {
            b.gate_up_swiglu_device(gate, up, s.input, s.gate, s.up)?;
            b.matvec_device(down, s.gate, s.projected)?;
        } else if !r.exact && gate.cols() % 128 == 0 && gate.rows() % 128 == 0 {
            if !residual_done { b.rms_partials_device(s.hidden, s.sums, norms.post.len())?; }
            b.rms_pack_apply_device(s.hidden, norms.post, s.input_pack, s.sums, r.epsilon, false)?;
            b.prefill_gate_up_packed_device(gate, up, s.input_pack, s.gate, r.tokens)?;
            b.matmul_packed_device(down, s.gate, s.projected, r.tokens)?;
        } else {
            if !residual_done { b.rms_partials_device(s.hidden, s.sums, norms.post.len())?; }
            b.rms_batch_apply_device(s.hidden, norms.post, s.input, s.sums, r.epsilon)?;
            project(b, gate, s.input, s.gate, s.weight_pack, s.input_pack, r.tokens, r.exact, false, false)?;
            project(b, up, s.input, s.up, s.weight_pack, s.input_pack, r.tokens, r.exact, true, false)?;
            b.swiglu_device(s.gate, s.up)?;
            project(b, down, s.gate, s.projected, s.weight_pack, s.input_pack, r.tokens, r.exact, false, false)?;
        }
    } else { b.write_vector(s.projected, &vec![0.; s.projected.len()])?; }
    if complete_projection(s.projected,s.hidden,s.sums,s.input,norms.next.filter(|_|r.decode))? {Ok(())}
    else if let Some(next) = norms.next.filter(|_| r.decode) {
        b.residual_rms_device(s.hidden, s.projected, next, s.input, s.sums, r.epsilon)
    } else { b.residual_device(s.hidden, s.projected) }
}

/// RoPE uses the full-head exponent even when only a proportional subset rotates.
pub fn model_rope<'a>(b: &Backend<'a>, dim: usize, theta: f64, fraction: f64,
    max_tokens: usize) -> Result<DeviceVector<'a>> {
    let half = dim / 2;
    let active = (half as f64 * fraction) as usize;
    let mut table = Vec::with_capacity(max_tokens * dim);
    let inverse: Vec<_> = (0..half).map(|i| if i < active {
        1.0f32 / (theta as f32).powf((2*i) as f32 / dim as f32)
    } else {0.0}).collect();
    let bf16 = |x: f32| f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000);
    for position in 0..max_tokens { for &frequency in &inverse {
        let (sin, cos) = (position as f32 * frequency).sin_cos();
        table.extend([bf16(sin), bf16(cos)]);
    }}
    b.upload_vector(&table)
}

pub struct BlockNorms<'s, 'a> {
    pub input: &'s DeviceVector<'a>, pub attention: &'s DeviceVector<'a>,
    pub pre_mlp: &'s DeviceVector<'a>, pub post_mlp: &'s DeviceVector<'a>,
    pub q: &'s DeviceVector<'a>, pub k: &'s DeviceVector<'a>, pub scalar: f32,
}
/// Gemma's post-branch norms precede BF16 residual additions. All projections
/// use the existing dense helpers, including the shared tied output matrix.
pub fn forward_normalized(b: &Backend<'_>, w: Weights<'_, '_>, n: BlockNorms<'_, '_>,
    rope: &DeviceVector<'_>, keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>,
    s: Buffers<'_, '_>, r: Run, window: usize, cache_capacity: usize) -> Result<()> {
    let q_norm = n.q; let k_norm = n.k;
    forward_normalized_attention(b, w, n, s, r, |q, k, v, scores, attention| {
        b.model_qkv_device(q, k, v, q_norm, k_norm, rope, keys, values,
            r.heads, r.kv_heads, r.head_dim, r.position, r.tokens, cache_capacity, r.epsilon)?;
        b.bounded_attention_device(q, keys, values, scores, attention,
            r.heads, r.kv_heads, r.head_dim, r.position, r.tokens, window, cache_capacity, 1.0)
    })
}

/// Independent sessions share all projections and norms; the caller binds
/// each attention row to its private absolute position and KV ring.
pub fn forward_normalized_attention(b: &Backend<'_>, w: Weights<'_, '_>, n: BlockNorms<'_, '_>,
    s: Buffers<'_, '_>, r: Run,
    mut attention: impl FnMut(&mut DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>,
        &mut DeviceVector<'_>, &mut DeviceVector<'_>) -> Result<()>) -> Result<()> {
    let h = n.input.len();
    b.model_rms_device(s.hidden, Some(n.input), s.input, s.sums, h, r.tokens, r.epsilon)?;
    b.matmul_bf16_device(w.q.ok_or("missing Q")?, s.input, s.q, s.weight_pack, s.input_pack, r.tokens, false)?;
    if let Some(k) = w.k {
        b.matmul_bf16_device(k, s.input, s.k, s.weight_pack, s.input_pack, r.tokens, true)?;
    }
    if let Some(v) = w.v {
        b.matmul_bf16_device(v, s.input, s.v, s.weight_pack, s.input_pack, r.tokens, true)?;
    }
    attention(s.q, s.k, if w.v.is_some() {s.v} else {s.k}, s.scores, s.attention)?;
    b.matmul_bf16_device(w.o.ok_or("missing O")?, s.attention, s.projected, s.weight_pack, s.input_pack, r.tokens, false)?;
    b.model_rms_device(s.projected, Some(n.attention), s.input, s.sums, h, r.tokens, r.epsilon)?;
    b.model_elementwise_device(s.hidden, Some(s.input), s.hidden, r.tokens*h, 1, 1.0)?;
    b.model_rms_device(s.hidden, Some(n.pre_mlp), s.input, s.sums, h, r.tokens, r.epsilon)?;
    let (gate, up, down) = (w.gate.ok_or("missing gate")?, w.up.ok_or("missing up")?, w.down.ok_or("missing down")?);
    for (weight, output, reuse) in [(gate, &mut *s.gate, false), (up, &mut *s.up, true)] {
        b.matmul_bf16_device(weight, s.input, output, s.weight_pack, s.input_pack, r.tokens, reuse)?;
    }
    b.model_elementwise_device(s.gate, Some(s.up), s.gate, r.tokens*gate.rows(), 2, 1.0)?;
    b.matmul_bf16_device(down, s.gate, s.projected, s.weight_pack, s.input_pack, r.tokens, false)?;
    b.model_rms_device(s.projected, Some(n.post_mlp), s.input, s.sums, h, r.tokens, r.epsilon)?;
    b.model_elementwise_device(s.hidden, Some(s.input), s.hidden, r.tokens*h, 1, 1.0)?;
    b.model_elementwise_device(s.hidden, None, s.hidden, r.tokens*h, 0, n.scalar)
}
