use super::*;

struct OwnedWeights {
    qa: Vec<f32>, qn: Vec<f32>, qb: Vec<f32>,
    kva: Vec<f32>, kvn: Vec<f32>, kvb: Vec<f32>,
}
impl OwnedWeights {
    fn view(&self) -> Weights<'_> {
        Weights { q_a_proj: &self.qa, q_a_layernorm: &self.qn, q_b_proj: &self.qb,
            kv_a_proj_with_mqa: &self.kva, kv_a_layernorm: &self.kvn, kv_b_proj: &self.kvb }
    }
    fn generated(c: Config) -> Self {
        let values = |len: usize, salt: usize| (0..len).map(|i| (((i * 17 + salt * 13) % 29) as i32 - 14) as f32 / 32.0).collect();
        Self { qa: values(c.q_lora_rank * c.hidden_size, 1),
            qn: (0..c.q_lora_rank).map(|i| 0.5 + i as f32 / 8.0).collect(),
            qb: values(c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * c.q_lora_rank, 3),
            kva: values((c.kv_lora_rank + c.qk_rope_head_dim) * c.hidden_size, 4),
            kvn: (0..c.kv_lora_rank).map(|i| 0.75 + i as f32 / 16.0).collect(),
            kvb: values(c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank, 6) }
    }
}
fn config(rope: usize) -> Config {
    Config { hidden_size: 5, q_lora_rank: 3, kv_lora_rank: 4, num_attention_heads: 3,
        qk_nope_head_dim: 2, qk_rope_head_dim: rope, v_head_dim: 3, rms_norm_eps: 0.00001 }
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn same_bits(actual: &[f32], expected: &[f32]) { assert_eq!(bits(actual), bits(expected)); }

// These scalar oracles do not call the reference math or expansion helpers.
fn matrix(w: &[f32], x: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in w.chunks_exact(x.len()) {
        let mut sum = 0.0_f32;
        for i in 0..x.len() { sum += row[i] * x[i]; }
        out.push(sum);
    }
    out
}
fn normalize(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0_f32;
    for &v in x { sum += v * v; }
    let inverse = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &weight)| (v * inverse) * weight).collect()
}
fn rotate(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    for i in 0..cos.len() {
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos[i] - b * sin[i];
        x[2 * i + 1] = b * cos[i] + a * sin[i];
    }
}
fn project_oracle(c: Config, w: &OwnedWeights, x: &[f32], cos: &[f32], sin: &[f32]) -> Projected {
    let low_query = normalize(&matrix(&w.qa, x), &w.qn, c.rms_norm_eps);
    let mut query = matrix(&w.qb, &low_query);
    let key_dim = c.qk_nope_head_dim + c.qk_rope_head_dim;
    for head in query.chunks_exact_mut(key_dim) { rotate(&mut head[c.qk_nope_head_dim..], cos, sin); }
    let low_kv = matrix(&w.kva, x);
    let latent = normalize(&low_kv[..c.kv_lora_rank], &w.kvn, c.rms_norm_eps);
    let mut rope_key = low_kv[c.kv_lora_rank..].to_vec();
    rotate(&mut rope_key, cos, sin);
    Projected { query, latent, rope_key }
}
fn expand_oracle(c: Config, w: &OwnedWeights, p: &Projected) -> (Vec<f32>, Vec<f32>) {
    let expanded = matrix(&w.kvb, &p.latent);
    let mut key = Vec::new();
    let mut value = Vec::new();
    for head in expanded.chunks_exact(c.qk_nope_head_dim + c.v_head_dim) {
        key.extend_from_slice(&head[..c.qk_nope_head_dim]);
        key.extend_from_slice(&p.rope_key);
        value.extend_from_slice(&head[c.qk_nope_head_dim..]);
    }
    (key, value)
}
fn full_attention(c: Config, query: &[f32], cache: &[(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
    let key_dim = c.qk_nope_head_dim + c.qk_rope_head_dim;
    let scale = 1.0_f32 / (key_dim as f32).sqrt();
    let mut out = vec![0.0_f32; c.num_attention_heads * c.v_head_dim];
    for h in 0..c.num_attention_heads {
        let mut scores = Vec::new();
        for (key, _) in cache {
            let mut score = 0.0_f32;
            for d in 0..key_dim { score += query[h * key_dim + d] * key[h * key_dim + d]; }
            scores.push(score * scale);
        }
        let mut maximum = f32::NEG_INFINITY;
        for &score in &scores { maximum = maximum.max(score); }
        let mut denominator = 0.0_f32;
        for score in &mut scores { *score = (*score - maximum).exp(); denominator += *score; }
        for score in &mut scores { *score /= denominator; }
        for d in 0..c.v_head_dim {
            let mut sum = 0.0_f32;
            for (t, (_, value)) in cache.iter().enumerate() { sum += scores[t] * value[h * c.v_head_dim + d]; }
            out[h * c.v_head_dim + d] = sum;
        }
    }
    out
}
fn compare_caches(c: Config) {
    let weights = OwnedWeights::generated(c);
    let reference = Reference::new(c, weights.view()).unwrap();
    for prompt in [1, 7, 64] {
        let mut latent = LatentCache::new(c).unwrap();
        let mut step_cache = LatentCache::new(c).unwrap();
        let mut full = Vec::new();
        assert!(latent.is_empty());
        for token in 0..prompt + 5 {
            let x: Vec<_> = (0..c.hidden_size).map(|i| (((token * 11 + i * 7) % 31) as i32 - 15) as f32 / 16.0).collect();
            let angles: Vec<_> = (0..c.qk_rope_head_dim / 2).map(|i| token as f32 / (i + 1) as f32).collect();
            let cos: Vec<_> = angles.iter().map(|a| a.cos()).collect();
            let sin: Vec<_> = angles.iter().map(|a| a.sin()).collect();
            let expected = project_oracle(c, &weights, &x, &cos, &sin);
            let projected = reference.project(&x, Rope { cos: &cos, sin: &sin }).unwrap();
            same_bits(&projected.query, &expected.query);
            same_bits(&projected.latent, &expected.latent);
            same_bits(&projected.rope_key, &expected.rope_key);
            full.push(expand_oracle(c, &weights, &expected));
            let expanded = reference.expand(&projected).unwrap();
            same_bits(&expanded.key, &full.last().unwrap().0);
            same_bits(&expanded.value, &full.last().unwrap().1);
            latent.push(&projected).unwrap();
            let actual = reference.attend(&projected.query, &latent).unwrap();
            let expected_output = full_attention(c, &expected.query, &full);
            assert_eq!(bits(&actual), bits(&expected_output), "prompt={prompt} token={token}");
            let stepped = reference.step(&mut step_cache, &x, Rope { cos: &cos, sin: &sin }).unwrap();
            same_bits(&stepped, &expected_output);
            assert_eq!(latent.len(), token + 1);
            assert_eq!(latent.payload_bytes(), (token + 1) * c.cache_bytes_per_token().unwrap().latent);
            assert_eq!(step_cache.payload_bytes(), latent.payload_bytes());
        }
    }
}

#[test]
fn latent_cache_matches_full_cache_without_rope() { compare_caches(config(0)); }

#[test]
fn latent_cache_matches_full_cache_with_rope() { compare_caches(config(4)); }

#[test]
fn latent_cache_matches_full_cache_with_only_rope_keys() {
    compare_caches(Config { qk_nope_head_dim: 0, ..config(4) });
}

#[test]
fn hand_vectors_check_norms_projections_rope_and_head_order() {
    let c = Config { hidden_size: 2, q_lora_rank: 2, kv_lora_rank: 2, num_attention_heads: 2,
        qk_nope_head_dim: 1, qk_rope_head_dim: 2, v_head_dim: 1, rms_norm_eps: 7.0 };
    let weights = OwnedWeights { qa: vec![1.0, 0.0, 0.0, 1.0], qn: vec![4.0, 8.0],
        qb: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, -1.0, 0.0, 0.0, 2.0],
        kva: vec![1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0], kvn: vec![8.0, 4.0],
        kvb: vec![1.0, 1.0, 2.0, 0.0, -1.0, 0.0, 0.0, -2.0] };
    let reference = Reference::new(c, weights.view()).unwrap();
    let p = reference.project(&[3.0, 3.0], Rope { cos: &[0.0], sin: &[1.0] }).unwrap();
    // Mean square is 9. Add epsilon 7, then take sqrt to get 4.
    // The normalized query is [3, 6]. The normalized latent is [6, -3].
    // A right-angle rotation maps (a, b) to (-b, a).
    same_bits(&p.query, &[3.0, -9.0, 6.0, 6.0, -12.0, -3.0]);
    same_bits(&p.latent, &[6.0, -3.0]);
    same_bits(&p.rope_key, &[0.0, 6.0]);
    let expanded = reference.expand(&p).unwrap();
    same_bits(&expanded.key, &[3.0, 0.0, 6.0, -6.0, 0.0, 6.0]);
    same_bits(&expanded.value, &[12.0, 6.0]);
    let mut cache = LatentCache::new(c).unwrap();
    cache.push(&p).unwrap();
    same_bits(&reference.attend(&p.query, &cache).unwrap(), &[12.0, 6.0]);
    let opposite = reference.project(&[-3.0, -3.0], Rope { cos: &[0.0], sin: &[1.0] }).unwrap();
    cache.push(&opposite).unwrap();
    // Zero scores give equal weights. Opposite values give zero output.
    same_bits(&reference.attend(&[0.0; 6], &cache).unwrap(), &[0.0, 0.0]);
}

#[test]
fn hand_vector_checks_nonuniform_attention_weights() {
    let c = Config { hidden_size: 1, q_lora_rank: 1, kv_lora_rank: 1, num_attention_heads: 1,
        qk_nope_head_dim: 1, qk_rope_head_dim: 0, v_head_dim: 1, rms_norm_eps: 1.0 };
    let weights = OwnedWeights { qa: vec![1.0], qn: vec![1.0], qb: vec![1.0],
        kva: vec![1.0], kvn: vec![1.0], kvb: vec![1.0, 3.0] };
    let reference = Reference::new(c, weights.view()).unwrap();
    let mut cache = LatentCache::new(c).unwrap();
    cache.push(&Projected { query: vec![1.0], latent: vec![1.0], rope_key: vec![] }).unwrap();
    same_bits(&reference.attend(&[1.0], &cache).unwrap(), &[3.0]);
    cache.push(&Projected { query: vec![1.0], latent: vec![2.0], rope_key: vec![] }).unwrap();
    // Scores [1, 2] and values [3, 6] give 3 + 3/(1 + exp(-1)).
    let actual = reference.attend(&[1.0], &cache).unwrap()[0];
    assert!((actual - 5.193176).abs() < 0.000001, "{actual}");
}

#[test]
fn public_config_cache_bytes_and_nested_config() {
    // Public config revision f5a0baba30120da613e27973ee3535a725d4ece3:
    // https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/f5a0baba30120da613e27973ee3535a725d4ece3/config.json
    let fields = r#"{"hidden_size":4096,"q_lora_rank":1536,"kv_lora_rank":512,
        "num_attention_heads":64,"qk_nope_head_dim":256,"qk_rope_head_dim":0,
        "v_head_dim":256,"rms_norm_eps":0.00001,"qk_head_dim":256}"#;
    let direct = Config::from_json(fields.as_bytes()).unwrap();
    let wrapped = Config::from_json(format!("{{\"text_config\":{fields}}}").as_bytes()).unwrap();
    assert_eq!(direct, wrapped);
    let bytes = direct.cache_bytes_per_token().unwrap();
    assert_eq!(bytes.latent, 512 * 4);
    assert_eq!(bytes.expanded, 64 * (256 + 256) * 4);
    assert_eq!(bytes.expanded / bytes.latent, 64);
    println!("MLA config receipt: hidden=4096 q_rank=1536 kv_rank=512 heads=64 nope=256 rope=0 value=256; f32 latent_bytes_per_token={} full_kv_bytes_per_token={}", bytes.latent, bytes.expanded);
}

#[test]
fn config_rejects_invalid_dimensions_and_overflow() {
    let c = config(2);
    for bad in [Config { hidden_size: 0, ..c }, Config { q_lora_rank: 0, ..c },
        Config { kv_lora_rank: 0, ..c }, Config { num_attention_heads: 0, ..c },
        Config { v_head_dim: 0, ..c }, Config { qk_rope_head_dim: 3, ..c },
        Config { qk_nope_head_dim: 0, qk_rope_head_dim: 0, ..c },
        Config { rms_norm_eps: 0.0, ..c }, Config { rms_norm_eps: f32::NAN, ..c },
        Config { rms_norm_eps: f32::INFINITY, ..c }, Config { hidden_size: usize::MAX, ..c },
        Config { qk_nope_head_dim: usize::MAX, ..c }, Config { num_attention_heads: usize::MAX, ..c }] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(LatentCache::new(bad).is_err());
    }
    assert!(Config::from_json(br#"{"hidden_size":5}"#).is_err());
    let fields = r#""hidden_size":5,"q_lora_rank":3,"kv_lora_rank":4,
        "num_attention_heads":3,"qk_nope_head_dim":2,"qk_rope_head_dim":2,
        "v_head_dim":3,"rms_norm_eps":0.00001"#;
    for extra in [r#""attention_bias":true"#, r#""qk_head_dim":5"#] {
        assert!(Config::from_json(format!("{{{fields},{extra}}}").as_bytes()).is_err());
    }
}

#[test]
fn invalid_inputs_leave_cache_unchanged() {
    let c = config(2);
    let weights = OwnedWeights::generated(c);
    let reference = Reference::new(c, weights.view()).unwrap();
    let mut cache = LatentCache::new(c).unwrap();
    assert!(reference.attend(&[0.0; 12], &cache).is_err());
    reference.step(&mut cache, &[1.0; 5], Rope { cos: &[1.0], sin: &[0.0] }).unwrap();
    let bytes = cache.payload_bytes();
    for input in [&[1.0; 4][..], &[f32::NAN; 5][..]] {
        assert!(reference.step(&mut cache, input, Rope { cos: &[1.0], sin: &[0.0] }).is_err());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.payload_bytes(), bytes);
    }
    assert!(reference.step(&mut cache, &[1.0; 5], Rope { cos: &[], sin: &[] }).is_err());
    assert_eq!(cache.len(), 1);
    let mut p = reference.project(&[1.0; 5], Rope { cos: &[1.0], sin: &[0.0] }).unwrap();
    p.latent.pop();
    assert!(cache.push(&p).is_err());
    assert_eq!(cache.len(), 1);
    assert!(reference.attend(&[0.0; 11], &cache).is_err());
}

#[test]
fn failed_attention_rolls_back_the_new_cache_row() {
    let c = Config { hidden_size: 1, q_lora_rank: 1, kv_lora_rank: 1, num_attention_heads: 1,
        qk_nope_head_dim: 1, qk_rope_head_dim: 0, v_head_dim: 1, rms_norm_eps: 1.0 };
    let weights = OwnedWeights { qa: vec![1.0], qn: vec![1.0], qb: vec![1.0],
        kva: vec![1.0], kvn: vec![4.0], kvb: vec![f32::MAX, 1.0] };
    let reference = Reference::new(c, weights.view()).unwrap();
    let mut cache = LatentCache::new(c).unwrap();
    same_bits(&reference.step(&mut cache, &[0.0], Rope { cos: &[], sin: &[] }).unwrap(), &[0.0]);
    let bytes = cache.payload_bytes();
    // The next projection is finite. Its expanded key overflows.
    let p = reference.project(&[1.0], Rope { cos: &[], sin: &[] }).unwrap();
    assert!(reference.expand(&p).is_err());
    assert!(reference.step(&mut cache, &[1.0], Rope { cos: &[], sin: &[] }).is_err());
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.payload_bytes(), bytes);
    same_bits(&reference.attend(&[0.0], &cache).unwrap(), &[0.0]);
}

#[test]
fn weights_require_exact_shapes_and_finite_values() {
    let c = config(2);
    let mut weights = OwnedWeights::generated(c);
    weights.qa.pop();
    assert!(Reference::new(c, weights.view()).is_err());
    weights = OwnedWeights::generated(c);
    weights.kvn[0] = f32::INFINITY;
    assert!(Reference::new(c, weights.view()).is_err());
    weights = OwnedWeights::generated(c);
    weights.kvb.push(0.0);
    assert!(Reference::new(c, weights.view()).is_err());
}
