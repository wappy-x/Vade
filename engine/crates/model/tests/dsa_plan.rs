//! Synthetic headers connect DSA plan data to the independent host reference.
use engine_model::{dsa as reference, plan::{ModelPlan, dsa::{DsaPlan, TieOrder, WeightRole}}, safetensors::{Dtype, Header, Limits, Tensor}};
use std::{collections::BTreeMap, io::{self, Cursor, Read, Seek, SeekFrom}};

const CONFIG: &str = r#"{"hidden_size":3,"q_lora_rank":2,"index_n_heads":1,"index_head_dim":4,"index_topk":3,"index_kpool":3,"index_kpool_always_select_tail":true,"index_kpool_compress":true}"#;
const PREFIX: &str = "model.language_model.layers.2.self_attn.indexer";
// These shapes and offsets are independent of the plan builder.
const HEADER: &str = r#"{
"model.language_model.layers.2.self_attn.indexer.wq_b.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},
"model.language_model.layers.2.self_attn.indexer.wk.weight":{"dtype":"F32","shape":[4,3],"data_offsets":[32,80]},
"model.language_model.layers.2.self_attn.indexer.weights_proj.weight":{"dtype":"F32","shape":[1,3],"data_offsets":[80,92]},
"model.language_model.layers.2.self_attn.indexer.k_norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[92,108]},
"model.language_model.layers.2.self_attn.indexer.k_norm.bias":{"dtype":"F32","shape":[4],"data_offsets":[108,124]},
"model.language_model.layers.2.self_attn.indexer.index_kpool_compress_ape":{"dtype":"F32","shape":[3,4],"data_offsets":[124,172]},
"model.language_model.layers.2.self_attn.indexer.index_kpool_compress_gate":{"dtype":"F32","shape":[4,3],"data_offsets":[172,220]}
}"#;

struct HeaderOnly { bytes: Cursor<Vec<u8>>, file_bytes: u64 }
impl Read for HeaderOnly {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.bytes.position() + out.len() as u64 > self.bytes.get_ref().len() as u64 {
            return Err(io::Error::other("test must not read payload"));
        }
        self.bytes.read(out)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        match position {
            SeekFrom::End(0) => { self.bytes.set_position(self.file_bytes); Ok(self.file_bytes) }
            other => self.bytes.seek(other),
        }
    }
}
fn headers() -> BTreeMap<String, Tensor> {
    let mut bytes = (HEADER.len() as u64).to_le_bytes().to_vec(); bytes.extend_from_slice(HEADER.as_bytes());
    let metadata_bytes = bytes.len() as u64;
    let mut reader = HeaderOnly { bytes: Cursor::new(bytes), file_bytes: metadata_bytes + 220 };
    let header = Header::read(&mut reader, Limits::default()).unwrap();
    assert_eq!(reader.bytes.position(), metadata_bytes);
    header.tensors
}
fn plan() -> DsaPlan { DsaPlan::from_config(CONFIG.as_bytes()).unwrap() }
fn config_from_plan(p: &DsaPlan) -> reference::Config {
    let g = p.geometry(); let s = p.selection(); let pool = p.pool();
    reference::Config { hidden_size: g.hidden as usize, q_lora_rank: g.query_rank as usize,
        index_n_heads: g.heads as usize, index_head_dim: g.head_dim as usize,
        index_topk: s.token_budget as usize, index_kpool: pool.tokens as usize,
        index_kpool_always_select_tail: s.tail_width != 0 }
}
fn hand_weights() -> BTreeMap<String, Vec<f32>> {
    [
        ("wq_b.weight", vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]),
        ("wk.weight", vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        ("weights_proj.weight", vec![1.0, -1.0, 0.5]),
        ("k_norm.weight", vec![1.0, 2.0, 3.0, 4.0]),
        ("k_norm.bias", vec![0.5, -0.5, 1.0, -1.0]),
        ("index_kpool_compress_ape", vec![0.0; 12]),
        ("index_kpool_compress_gate", vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0]),
    ].into_iter().map(|(name, values)| (format!("{PREFIX}.{name}"), values)).collect()
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }

#[test]
fn synthetic_header_runs_reference_shape_through_plan() {
    let p = plan(); let g = p.geometry(); let bound = p.with_headers(PREFIX, &headers()).unwrap();
    assert_eq!((g.hidden, g.query_rank, g.heads, g.head_dim), (3, 2, 1, 4));
    let values = hand_weights();
    let by_role = |role| -> &[f32] {
        let weight = bound.iter().find(|w| w.role == role).unwrap();
        let data = &values[&weight.name];
        assert_eq!(weight.tensor.dtype, Dtype::F32);
        assert_eq!(weight.tensor.shape.iter().product::<u64>(), data.len() as u64);
        data
    };
    let d = config_from_plan(&p);
    assert_eq!(d, reference::Config::from_json(CONFIG.as_bytes()).unwrap());
    let r = reference::Reference::new(d, reference::Weights {
        wq_b: by_role(WeightRole::Query), wk: by_role(WeightRole::Key),
        weights_proj: by_role(WeightRole::HeadWeight), k_norm_weight: by_role(WeightRole::KeyNormGain),
        k_norm_bias: by_role(WeightRole::KeyNormBias), index_kpool_compress_ape: by_role(WeightRole::PoolPosition),
        index_kpool_compress_gate: by_role(WeightRole::PoolGate),
    }).unwrap();
    let token = r.project(&[1.0, 2.0, 4.0], &[3.0, -1.0]).unwrap();
    assert_eq!(token.query, [3.0, -1.0, 2.0, 7.0]);
    assert_eq!(token.head_weights, [1.0]); assert_eq!(token.gate, [1.0, 4.0, -4.0, 3.0]);
    let inv = 1.0 / (5.25f32 + 1e-6).sqrt();
    assert_eq!(bits(&token.key), bits(&[(-2.5 * inv) + 0.5, (-1.5 * inv) * 2.0 - 0.5,
        (0.5 * inv) * 3.0 + 1.0, (3.5 * inv) * 4.0 - 1.0]));
    let hidden = [1.0, 2.0, 4.0].repeat(4); let residual = [3.0, -1.0].repeat(4);
    let mut cache = reference::Cache::new(d).unwrap(); assert!(cache.is_empty());
    let selected = r.run_chunk(&mut cache, &hidden, &residual, &[true; 4]).unwrap();
    assert_eq!(selected, [vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]]);
    for (position, actual) in selected.iter().enumerate() {
        let query = p.query(&[true; 4], position).unwrap();
        let mut expected: Vec<_> = query.candidate_pools.iter().take(query.select_pools).flat_map(|r| r.clone()).collect();
        expected.extend(query.tail_indices);
        assert_eq!(*actual, expected);
        assert!(actual.len() as u64 <= p.selection().output_width);
    }
    let pools = r.pools(&cache).unwrap(); assert_eq!(pools[0].indices, [0, 1, 2]);
    assert_eq!(pools[0].key.len() as u64, p.pool().channels);
    let mut chunked = reference::Cache::new(d).unwrap(); let mut by_chunk = Vec::new();
    for row in 0..4 {
        by_chunk.extend(r.run_chunk(&mut chunked, &hidden[row*3..row*3+3], &residual[row*2..row*2+2], &[true]).unwrap());
    }
    assert_eq!(selected, by_chunk); assert_eq!(cache.len(), chunked.len());
    assert_eq!(bits(&pools[0].key), bits(&r.pools(&chunked).unwrap()[0].key));
}

#[test]
fn logical_weights_and_pool_axes_follow_config() {
    let p = plan();
    let expected = [
        (WeightRole::Query, "wq_b.weight", vec![4, 2]),
        (WeightRole::Key, "wk.weight", vec![4, 3]),
        (WeightRole::HeadWeight, "weights_proj.weight", vec![1, 3]),
        (WeightRole::KeyNormGain, "k_norm.weight", vec![4]),
        (WeightRole::KeyNormBias, "k_norm.bias", vec![4]),
        (WeightRole::PoolPosition, "index_kpool_compress_ape", vec![3, 4]),
        (WeightRole::PoolGate, "index_kpool_compress_gate", vec![4, 3]),
    ];
    assert_eq!(p.weights().len(), expected.len());
    for (role, name, shape) in expected {
        let weight = p.weights().iter().find(|w| w.role == role).unwrap();
        assert_eq!((weight.name, &weight.shape), (name, &shape));
    }
    let pool = p.pool();
    assert_eq!((pool.tokens, pool.channels, pool.key_offset, pool.gate_offset, pool.valid_offset, pool.packed_width), (3, 4, 0, 4, 8, 9));
    let layout = p.layout(2, 7).unwrap();
    assert_eq!(layout.packed, [2, 7, 9]); assert_eq!(layout.keys, [2, 3, 4]);
    assert_eq!(layout.indices, [2, 3, 3]); assert_eq!(layout.valid, [2, 3]);
    let s = p.selection();
    assert_eq!((s.token_budget, s.pool_budget, s.tail_width, s.output_width, s.invalid_index), (3, 1, 2, 5, -1));
    assert_eq!(s.index_dtype, Dtype::I32); assert_eq!(s.tie_order, TieOrder::Unspecified);
    let changed = CONFIG.replace("\"index_n_heads\":1", "\"index_n_heads\":2").replace("\"index_head_dim\":4", "\"index_head_dim\":5");
    let changed = DsaPlan::from_config(changed.as_bytes()).unwrap();
    assert_eq!(changed.geometry().heads, 2);
    assert_eq!(changed.weights().iter().find(|w| w.role == WeightRole::Query).unwrap().shape, [10, 2]);
    assert_eq!(changed.weights().iter().find(|w| w.role == WeightRole::HeadWeight).unwrap().shape, [2, 3]);
    assert_eq!(changed.pool().packed_width, 11);
}

#[test]
fn direct_and_nested_config_need_no_model_name() {
    let p = plan(); let nested = format!(r#"{{"model_type":"unrelated","text_config":{CONFIG}}}"#);
    assert_eq!(p, DsaPlan::from_config(nested.as_bytes()).unwrap());
    assert_eq!(p, DsaPlan::from_config(CONFIG.replace(",\"index_kpool_compress\":true", "").as_bytes()).unwrap());
    let qwen = include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json");
    assert!(DsaPlan::from_config(qwen).is_err());
    assert!(ModelPlan::from_config(qwen).is_ok());
    let glm = format!(r#"{{"model_type":"glm5_next","text_config":{CONFIG}}}"#);
    assert!(ModelPlan::from_config(glm.as_bytes()).is_err());
}

#[test]
fn malformed_and_excess_config_values_fail() {
    for name in ["hidden_size", "q_lora_rank", "index_n_heads", "index_head_dim", "index_topk", "index_kpool"] {
        let original = match name { "hidden_size" | "index_topk" | "index_kpool" => "3", "q_lora_rank" => "2", "index_n_heads" => "1", _ => "4" };
        for bad in ["0", "-1", "true", "1.5", "18446744073709551615"] {
            let text = CONFIG.replace(&format!("\"{name}\":{original}"), &format!("\"{name}\":{bad}"));
            assert!(DsaPlan::from_config(text.as_bytes()).is_err(), "{name}={bad}");
        }
    }
    for text in ["{}".to_owned(), "[]".to_owned(), r#"{"text_config":false}"#.to_owned(),
        CONFIG.replace("\"index_kpool_compress\":true", "\"index_kpool_compress\":false"),
        CONFIG.replace("\"index_kpool_always_select_tail\":true", "\"index_kpool_always_select_tail\":1"),
        CONFIG.replace("\"q_lora_rank\":2,", "")] {
        assert!(DsaPlan::from_config(text.as_bytes()).is_err(), "{text}");
    }
    assert!(plan().layout(u64::MAX, 7).is_err());
    assert!(plan().layout(u64::MAX, 0).is_err());
    assert!(plan().layout(1, u64::MAX).is_err());
    assert!(plan().layout(0, 7).is_err());
    assert!(plan().query(&[], 0).is_err());
    assert!(plan().query(&[true], 1).is_err());
}

#[test]
fn header_binding_keeps_names_types_and_offsets() {
    let p = plan(); let mut input = headers();
    let outside = "model.language_model.layers.2.self_attn.indexer_extra.weight";
    input.insert(outside.into(), Tensor { dtype: Dtype::I8, shape: vec![1], start: 0, end: 1 });
    let bound = p.with_headers(PREFIX, &input).unwrap();
    assert_eq!(bound.len(), 7);
    for weight in &bound { assert_eq!(weight.tensor, input[&weight.name]); }
    for dtype in [Dtype::Bf16, Dtype::F16] {
        let mut input = headers();
        for tensor in input.values_mut() { tensor.dtype = dtype; tensor.end = tensor.start + tensor.shape.iter().product::<u64>() * dtype.bytes(); }
        assert!(p.with_headers(PREFIX, &input).is_ok());
    }
    let renamed: BTreeMap<_, _> = headers().into_iter().map(|(name, t)| (name.replace(PREFIX, "custom.indexer"), t)).collect();
    assert!(p.with_headers("custom.indexer", &renamed).is_ok());
}

#[test]
fn header_binding_rejects_missing_unknown_or_wrong_storage() {
    let p = plan(); let key = format!("{PREFIX}.wk.weight");
    for name in headers().keys() {
        let mut input = headers(); input.remove(name);
        assert!(p.with_headers(PREFIX, &input).is_err(), "missing {name}");
    }
    let mut input = headers(); input.get_mut(&key).unwrap().shape = vec![3, 4];
    assert!(p.with_headers(PREFIX, &input).is_err());
    for dtype in [Dtype::U8, Dtype::F8E4M3, Dtype::F64] {
        let mut input = headers(); input.get_mut(&key).unwrap().dtype = dtype;
        assert!(p.with_headers(PREFIX, &input).is_err());
    }
    for suffix in ["wk.weight_scale", "wk.bias", "unmapped.weight"] {
        let mut input = headers(); input.insert(format!("{PREFIX}.{suffix}"), Tensor { dtype: Dtype::F32, shape: vec![1], start: 0, end: 4 });
        assert!(p.with_headers(PREFIX, &input).is_err());
    }
    for (start, end) in [(80, 32), (32, 79), (32, 81), (u64::MAX, u64::MAX)] {
        let mut input = headers(); let tensor = input.get_mut(&key).unwrap(); tensor.start = start; tensor.end = end;
        assert!(p.with_headers(PREFIX, &input).is_err(), "offsets {start}..{end}");
    }
    for prefix in ["", "model.", ".model", "model..indexer"] {
        assert!(p.with_headers(prefix, &headers()).is_err(), "prefix {prefix}");
    }
}

#[test]
fn output_width_limit_includes_the_tail() {
    let max = i32::MAX as u64;
    let config = CONFIG.replace("\"index_topk\":3", &format!("\"index_topk\":{max}"));
    assert!(DsaPlan::from_config(config.as_bytes()).is_err());
    let no_tail = config.replace("\"index_kpool_always_select_tail\":true", "\"index_kpool_always_select_tail\":false");
    assert_eq!(DsaPlan::from_config(no_tail.as_bytes()).unwrap().selection().output_width, max);
    let boundary = CONFIG.replace("\"index_topk\":3", &format!("\"index_topk\":{}", max - 2));
    assert_eq!(DsaPlan::from_config(boundary.as_bytes()).unwrap().selection().output_width, max);
}

fn pool_two(topk: u64, tail: bool) -> DsaPlan {
    DsaPlan::from_config(format!(r#"{{"hidden_size":2,"q_lora_rank":2,"index_n_heads":2,"index_head_dim":2,"index_topk":{topk},"index_kpool":2,"index_kpool_always_select_tail":{tail}}}"#).as_bytes()).unwrap()
}
#[test]
fn per_query_pool_visibility_uses_raw_slots_and_complete_pools() {
    let p = pool_two(4, true); let valid = [false, true, true, false, true, true, true, true];
    let q = p.query(&valid, 7).unwrap();
    assert_eq!(q.candidate_pools, [1..3, 5..7]); assert_eq!(q.select_pools, 2); assert!(q.tail_indices.is_empty());
    let q = p.query(&valid, 2).unwrap(); assert_eq!(q.candidate_pools, [1..3]); assert_eq!(q.select_pools, 1);
    let q = p.query(&valid, 5).unwrap(); assert_eq!(q.candidate_pools, [1..3]);
    for position in [0, 3] {
        let q = p.query(&valid, position).unwrap();
        assert!(q.candidate_pools.is_empty()); assert_eq!(q.select_pools, 0); assert!(q.tail_indices.is_empty());
    }
    let q = p.query(&[false; 4], 3).unwrap();
    assert!(q.candidate_pools.is_empty()); assert!(q.tail_indices.is_empty());
}

#[test]
fn whole_pool_budget_and_source_tail_rule_are_explicit() {
    let p = pool_two(3, true); let q = p.query(&[true; 7], 6).unwrap();
    assert_eq!(q.candidate_pools, [0..2, 2..4, 4..6]); assert_eq!(q.select_pools, 1); assert_eq!(q.tail_indices, [6]);
    assert_eq!(p.selection().output_width, 4);
    let small = pool_two(1, true); let q = small.query(&[true; 3], 2).unwrap();
    assert_eq!(q.select_pools, 0); assert_eq!(q.tail_indices, [2]);
    let no_tail = pool_two(3, false); assert_eq!(no_tail.selection().output_width, 3);
    assert!(no_tail.query(&[true; 7], 6).unwrap().tail_indices.is_empty());
    let q = pool_two(4, true).query(&[true, true, false, false, true, true, true], 6).unwrap();
    assert_eq!(q.candidate_pools, [0..2, 4..6]); assert_eq!(q.tail_indices, [4]);
    // The source tail can repeat a pool token when the validity mask has holes.
    assert!(q.candidate_pools.iter().any(|range| range.contains(&q.tail_indices[0])));
}

#[test]
fn single_slot_pools_have_no_extra_tail() {
    let config = CONFIG.replace("\"index_kpool\":3", "\"index_kpool\":1");
    let tail = DsaPlan::from_config(config.as_bytes()).unwrap();
    let no_tail = DsaPlan::from_config(config.replace("\"index_kpool_always_select_tail\":true", "\"index_kpool_always_select_tail\":false").as_bytes()).unwrap();
    assert_eq!(tail.selection(), no_tail.selection());
    assert_eq!((tail.selection().pool_budget, tail.selection().tail_width, tail.selection().output_width), (3, 0, 3));
    let valid = [false, true, false, true, true];
    for position in 0..valid.len() { assert_eq!(tail.query(&valid, position).unwrap(), no_tail.query(&valid, position).unwrap()); }
    let query = tail.query(&valid, 4).unwrap();
    assert_eq!(query.candidate_pools, [1..2, 3..4, 4..5]); assert_eq!(query.select_pools, 3); assert!(query.tail_indices.is_empty());
}
