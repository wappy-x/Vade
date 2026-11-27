use super::*;

struct OwnedWeights {
    wq_b: Vec<f32>, wk: Vec<f32>, weights_proj: Vec<f32>, k_norm_weight: Vec<f32>,
    k_norm_bias: Vec<f32>, ape: Vec<f32>, gate: Vec<f32>,
}
impl OwnedWeights {
    fn tiny(d: Config) -> Self {
        assert_eq!((d.hidden_size, d.q_lora_rank, d.index_n_heads, d.index_head_dim), (2, 2, 2, 2));
        Self { wq_b: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
            wk: vec![1.0, 0.0, 0.0, 1.0], weights_proj: vec![1.0, 2.0, -1.0, 1.0],
            k_norm_weight: vec![2.0, 3.0], k_norm_bias: vec![0.5, -0.25],
            ape: vec![0.0; d.index_kpool * d.index_head_dim], gate: vec![0.5, -0.25, -1.0, 2.0] }
    }
    fn borrow(&self) -> Weights<'_> {
        Weights { wq_b: &self.wq_b, wk: &self.wk, weights_proj: &self.weights_proj,
            k_norm_weight: &self.k_norm_weight, k_norm_bias: &self.k_norm_bias,
            index_kpool_compress_ape: &self.ape, index_kpool_compress_gate: &self.gate }
    }
}
fn config(pool: usize, topk: usize, tail: bool) -> Config {
    Config { hidden_size: 2, q_lora_rank: 2, index_n_heads: 2, index_head_dim: 2,
        index_topk: topk, index_kpool: pool, index_kpool_always_select_tail: tail }
}
fn token(key: [f32; 2]) -> Projected {
    Projected { query: vec![0.0; 4], key: key.to_vec(), head_weights: vec![0.0; 2], gate: vec![0.0; 2] }
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|x| x.to_bits()).collect() }
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= 2e-6, "{a} != {b}"); }
}
fn same_pools(a: &[Pool], b: &[Pool]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) { assert_eq!(a.indices, b.indices); assert_eq!(bits(&a.key), bits(&b.key)); }
}

#[test]
fn config_uses_direct_or_nested_data() {
    let text = r#"{"hidden_size":2,"q_lora_rank":2,"index_n_heads":2,"index_head_dim":2,"index_topk":3,"index_kpool":2,"index_kpool_always_select_tail":true,"index_kpool_compress":true,"mla_use_nope":true,"qk_rope_head_dim":0}"#;
    assert_eq!(Config::from_json(text.as_bytes()).unwrap(), config(2, 3, true));
    let nested = format!(r#"{{"text_config":{text}}}"#);
    assert_eq!(Config::from_json(nested.as_bytes()).unwrap(), config(2, 3, true));
    assert!(Config::from_json(text.replace("\"index_kpool_compress\":true", "\"index_kpool_compress\":false").as_bytes()).is_err());
    assert!(Config::from_json(text.replace("\"index_topk\":3", "\"index_topk\":0").as_bytes()).is_err());
}

#[test]
#[allow(clippy::neg_multiply)] // test arithmetic stays as written.
fn hand_projection_uses_layer_norm_gain_and_bias() {
    let d = config(2, 2, true);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let p = r.project(&[-1.0, 1.0], &[2.0, -1.0]).unwrap();
    assert_eq!(p.query, vec![2.0, -1.0, 1.0, 5.0]);
    let inv = 1.0 / (1.0f32 + 1e-6).sqrt();
    assert_eq!(bits(&p.key), bits(&[(-1.0 * inv) * 2.0 + 0.5, (1.0 * inv) * 3.0 - 0.25]));
    let scale = 1.0 / 2.0f32.sqrt();
    assert_eq!(bits(&p.head_weights), bits(&[1.0 * scale, 2.0 * scale]));
    assert_eq!(p.gate, vec![-0.75, 3.0]);
    let constant = r.project(&[3.0, 3.0], &[0.0, 0.0]).unwrap();
    assert_eq!(constant.key, w.k_norm_bias);
}

#[test]
fn unequal_dimensions_use_declared_matrix_strides() {
    let d = Config { hidden_size: 3, q_lora_rank: 2, index_n_heads: 1, index_head_dim: 4,
        index_topk: 3, index_kpool: 3, index_kpool_always_select_tail: true };
    let weights = Weights { wq_b: &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
        wk: &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        weights_proj: &[1.0, -1.0, 0.5], k_norm_weight: &[1.0, 2.0, 3.0, 4.0],
        k_norm_bias: &[0.5, -0.5, 1.0, -1.0], index_kpool_compress_ape: &[0.0; 12],
        index_kpool_compress_gate: &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0] };
    let r = Reference::new(d, weights).unwrap();
    let p = r.project(&[1.0, 2.0, 4.0], &[3.0, -1.0]).unwrap();
    assert_eq!(p.query, [3.0, -1.0, 2.0, 7.0]);
    assert_eq!(p.head_weights, [1.0]); assert_eq!(p.gate, [1.0, 4.0, -4.0, 3.0]);
    // K is [1,2,4,7]. Mean is 7/2. Population variance is 21/4.
    let inv = 1.0 / (5.25f32 + 1e-6).sqrt();
    assert_eq!(bits(&p.key), bits(&[(-2.5 * inv) + 0.5, (-1.5 * inv) * 2.0 - 0.5,
        (0.5 * inv) * 3.0 + 1.0, (3.5 * inv) * 4.0 - 1.0]));
    let hidden = [1.0, 2.0, 4.0].repeat(4);
    let residual = [3.0, -1.0].repeat(4);
    let mut cache = Cache::new(d).unwrap();
    assert_eq!(r.run_chunk(&mut cache, &hidden, &residual, &[true; 4]).unwrap(),
        [vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]]);
}

#[test]
fn hand_compression_has_per_channel_gate_and_position_softmax() {
    let d = config(2, 2, false);
    let mut w = OwnedWeights::tiny(d);
    let log_three = 3.0f32.ln();
    w.ape = vec![log_three, 0.0, 0.0, log_three];
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    let mut first = token([2.0, 8.0]); first.gate = vec![0.0, log_three];
    cache.push(&first, true).unwrap(); cache.push(&token([6.0, 4.0]), true).unwrap();
    let pools = r.pools(&cache).unwrap();
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].indices, vec![0, 1]);
    // Channel zero uses probabilities [3/4,1/4]. Channel one uses [1/2,1/2].
    close(&pools[0].key, &[3.0, 6.0]);
}

#[test]
fn relu_precedes_signed_head_weights() {
    let d = config(1, 1, false);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    cache.push(&token([1.0, 0.0]), true).unwrap(); cache.push(&token([-1.0, 0.0]), true).unwrap();
    let mut query = token([0.0, 0.0]);
    query.query = vec![2.0, 0.0, 1.0, 0.0]; query.head_weights = vec![-1.0, 1.0];
    // The first score is -1/sqrt(2). The second score is zero.
    assert_eq!(r.select(&query, &cache, 1).unwrap(), vec![1]);
}

#[test]
fn ties_keep_lower_raw_tokens_and_budget_counts_whole_pools() {
    for topk in [2, 3] {
        let d = config(2, topk, true);
        let w = OwnedWeights::tiny(d);
        let r = Reference::new(d, w.borrow()).unwrap();
        let mut cache = Cache::new(d).unwrap();
        for _ in 0..7 { cache.push(&token([1.0, 1.0]), true).unwrap(); }
        let q = token([0.0, 0.0]);
        assert_eq!(r.select(&q, &cache, 5).unwrap(), vec![0, 1]);
        assert_eq!(r.select(&q, &cache, 6).unwrap(), vec![0, 1, 6]);
    }
    let d = config(2, 1, true);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    for _ in 0..3 { cache.push(&token([1.0, 1.0]), true).unwrap(); }
    assert_eq!(r.select(&token([0.0, 0.0]), &cache, 2).unwrap(), vec![2]);
}

#[test]
fn pool_score_order_and_causal_visibility_are_explicit() {
    let d = config(2, 4, false);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    for value in [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 9.0, 9.0] { cache.push(&token([value, 0.0]), true).unwrap(); }
    let mut q = token([0.0, 0.0]); q.query = vec![1.0, 0.0, 0.0, 0.0]; q.head_weights = vec![1.0, 0.0];
    assert_eq!(r.select(&q, &cache, 2).unwrap(), vec![0, 1]);
    assert_eq!(r.select(&q, &cache, 5).unwrap(), vec![2, 3, 4, 5]);
    assert_eq!(r.select(&q, &cache, 7).unwrap(), vec![6, 7, 2, 3]);
}

#[test]
fn valid_masks_anchor_pools_without_compacting_holes() {
    let d = config(2, 4, true);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    for valid in [false, true, true, false, true, true, true, true] { cache.push(&token([1.0, 1.0]), valid).unwrap(); }
    let pools = r.pools(&cache).unwrap();
    assert_eq!(pools.iter().map(|p| p.indices.clone()).collect::<Vec<_>>(), vec![vec![1, 2], vec![5, 6]]);
    assert_eq!(r.select(&token([0.0, 0.0]), &cache, 7).unwrap(), vec![1, 2, 5, 6]);
    assert!(r.select(&token([0.0, 0.0]), &cache, 0).unwrap().is_empty());
    assert!(r.select(&token([0.0, 0.0]), &cache, 3).unwrap().is_empty());
}

#[test]
fn internal_hole_tail_follows_the_source_valid_count_rule() {
    let d = config(2, 4, true);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    for valid in [true, true, false, false, true, true, true] { cache.push(&token([1.0, 1.0]), valid).unwrap(); }
    // The source tail uses valid count, then raw indices. It can repeat a selected token.
    assert_eq!(r.select(&token([0.0, 0.0]), &cache, 6).unwrap(), vec![0, 1, 4, 5, 4]);
}

#[test]
fn prompt_and_chunk_runs_have_identical_selections_and_pool_bits() {
    for pool in [1, 2, 3, 4] {
        let d = config(pool, pool * 2 + 1, true);
        let mut w = OwnedWeights::tiny(d);
        for (i, v) in w.ape.iter_mut().enumerate() { *v = (i as f32 - 2.0) * 0.125; }
        let r = Reference::new(d, w.borrow()).unwrap();
        let hidden = (0..26).map(|i| ((i * 7 + 3) % 19) as f32 * 0.25 - 2.0).collect::<Vec<_>>();
        let residual = (0..26).map(|i| ((i * 11 + 1) % 23) as f32 * 0.125 - 1.0).collect::<Vec<_>>();
        for valid in [vec![true; 13], vec![false, false, true, true, true, true, true, true, true, true, true, true, true],
            vec![false, true, true, false, true, true, true, true, false, true, true, true, true]] {
            let mut whole = Cache::new(d).unwrap();
            let expected = r.run_chunk(&mut whole, &hidden, &residual, &valid).unwrap();
            let expected_pools = r.pools(&whole).unwrap();
            for chunk in [1, 2, 3, 4, 5, 7, 13] {
                let mut cache = Cache::new(d).unwrap();
                let mut actual = Vec::new();
                for start in (0..valid.len()).step_by(chunk) {
                    let end = (start + chunk).min(valid.len());
                    actual.extend(r.run_chunk(&mut cache, &hidden[start * 2..end * 2], &residual[start * 2..end * 2], &valid[start..end]).unwrap());
                }
                assert_eq!(cache.len(), whole.len());
                assert_eq!(actual, expected, "pool={pool} chunk={chunk}");
                same_pools(&r.pools(&cache).unwrap(), &expected_pools);
            }
        }
    }
}

#[test]
fn invalid_geometry_weights_and_projected_rows_return_errors() {
    let d = config(2, 2, true);
    for bad in [Config { hidden_size: 0, ..d }, Config { q_lora_rank: 0, ..d },
        Config { index_n_heads: 0, ..d }, Config { index_head_dim: 0, ..d },
        Config { index_topk: 0, ..d }, Config { index_kpool: 0, ..d },
        Config { hidden_size: usize::MAX, ..d }, Config { index_kpool: usize::MAX, ..d }] {
        assert!(bad.validate().is_err()); assert!(Cache::new(bad).is_err());
    }
    let mut w = OwnedWeights::tiny(d); w.k_norm_bias.pop();
    assert!(Reference::new(d, w.borrow()).is_err());
    let mut w = OwnedWeights::tiny(d); w.ape[0] = f32::NAN;
    assert!(Reference::new(d, w.borrow()).is_err());
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    assert!(r.project(&[1.0], &[1.0, 2.0]).is_err());
    assert!(r.project(&[1.0, 2.0], &[1.0]).is_err());
    assert!(r.project(&[f32::INFINITY, 1.0], &[1.0, 2.0]).is_err());
    let mut cache = Cache::new(d).unwrap();
    let mut bad = token([1.0, 2.0]); bad.key.pop();
    assert!(cache.push(&bad, true).is_err()); assert!(cache.is_empty());
    let mut bad = token([1.0, 2.0]); bad.gate[0] = f32::INFINITY;
    assert!(cache.push(&bad, false).is_err()); assert!(cache.is_empty());
    assert!(r.select(&token([0.0, 0.0]), &cache, 0).is_err());
    assert!(r.run_chunk(&mut cache, &[1.0], &[1.0, 2.0], &[true]).is_err());
    assert!(cache.is_empty());
}

#[test]
fn failed_chunk_restores_prior_cache_after_score_overflow() {
    let d = config(1, 1, false);
    let mut w = OwnedWeights::tiny(d);
    w.k_norm_bias = vec![1e20, 1e20];
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    r.run_chunk(&mut cache, &[1.0, 2.0], &[1.0, 1.0], &[true]).unwrap();
    let before = r.pools(&cache).unwrap();
    assert!(r.run_chunk(&mut cache, &[1.0, 2.0, 2.0, 3.0], &[1.0, 1.0, 1e20, 1e20], &[true, true]).is_err());
    assert_eq!(cache.len(), 1);
    same_pools(&r.pools(&cache).unwrap(), &before);
}

#[test]
fn signed_zero_scores_tie_by_lower_token_index() {
    let d = config(1, 2, false);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    for key in [[0.0, -0.0], [-0.0, 0.0], [0.0, 0.0], [-0.0, -0.0]] { cache.push(&token(key), true).unwrap(); }
    let mut q = token([0.0, 0.0]);
    q.query = vec![-0.0, 0.0, 0.0, -0.0]; q.head_weights = vec![-1.0, 1.0];
    assert_eq!(r.select(&q, &cache, 3).unwrap(), vec![0, 1]);
    q.query.fill(0.0); q.head_weights = vec![1.0, -1.0];
    assert_eq!(r.select(&q, &cache, 3).unwrap(), vec![0, 1]);
}

#[test]
fn empty_chunks_and_all_padding_have_no_selection() {
    let d = config(3, 2, true);
    let w = OwnedWeights::tiny(d);
    let r = Reference::new(d, w.borrow()).unwrap();
    let mut cache = Cache::new(d).unwrap();
    assert!(r.run_chunk(&mut cache, &[], &[], &[]).unwrap().is_empty());
    assert!(cache.is_empty());
    assert_eq!(r.run_chunk(&mut cache, &[1.0; 8], &[2.0; 8], &[false; 4]).unwrap(), vec![Vec::<usize>::new(); 4]);
    assert_eq!(cache.len(), 4);
    assert!(r.pools(&cache).unwrap().is_empty());
}
