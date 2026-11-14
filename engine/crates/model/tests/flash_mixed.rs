use engine_model::{flash::{mixed::{Mixed, Plan, Storage, ScaleView}, mxfp8, plan::RankPlan}, json};
#[test]
fn mxfp8_values_groups_and_extremes() {
    for s in 0..255u8 { for w in 0..=255u8 {
        let e=(w>>3)&15; let m=w&7; let v=if e==0 { m as f64/512. } else { (1.+m as f64/8.)*2f64.powi(e as i32-7) };
        let want=((if w&128!=0 {-v} else {v})*2f64.powi(s as i32-127)) as f32; let got=mxfp8::dequant_value(w,s);
        if e==15 && m==7 { assert!(got.is_nan()); } else { assert_eq!(got.to_bits(),want.to_bits()); }
    } }
    assert!(mxfp8::dequant_value(0,255).is_nan()); let mut out=[0.;64];
    mxfp8::dequantize(&[0x38;64], &[127,128], &mut out).unwrap(); assert_eq!((&out[..32],&out[32..]),(&[1.;32][..],&[2.;32][..]));
    assert!(mxfp8::dequantize(&[0;31],&[0],&mut [0.;31]).is_err()); assert!(mxfp8::dequantize(&[0;32],&[],&mut [0.;32]).is_err());
}
#[test]
fn config_precedence_and_world_one() {
    let v=json::parse(br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":["a.keep*"],"quantized_layers":{"a":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf"},"a.inner":{"quant_algo":"MXFP8","group_size":32}}}"#,json::Limits::default()).unwrap();
    let q=Mixed::parse(v.object().unwrap()).unwrap(); assert_eq!(q.storage("a.inner.q"),Storage::Mxfp8); assert_eq!(q.storage("a.keep.q"),Storage::Plain); assert_eq!(q.storage("ab"),Storage::Plain); assert!(matches!(q.storage("a.experts.2.gate_proj"),Storage::Nvfp4 {csf:true,..}));
    assert!(RankPlan::new(0,2).unwrap().mixed_selection("x").is_err()); let p=RankPlan::new(0,1).unwrap(); assert_eq!(p.mixed_selection("model.visual.x").unwrap(),engine_model::flash::plan::Selection::Absent);
}

#[cfg(unix)]
#[test]
fn metadata_map_rejects_unsafe_paths_and_symlink_escape() {
    use std::fs;
    let root = std::env::temp_dir().join(format!("mixed-paths-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let source = include_str!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json");
    let (prefix, _) = source.rsplit_once("\"quantization_config\":").unwrap();
    let config = format!(r#"{prefix}"quantization_config":{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":[],"quantized_layers":{{"a":{{"quant_algo":"NVFP4","group_size":16}}}}}}}}"#);
    fs::write(root.join("config.json"), config).unwrap();
    for path in ["../a.safetensors", "/a.safetensors", "a//b.safetensors", "a/./b.safetensors", "a b.safetensors"] {
        fs::write(root.join("map.json"), format!(r#"{{"a":{{"file":{path:?},"dtype":"U8","shape":[1]}}}}"#)).unwrap();
        assert!(Plan::open(&root.join("config.json"), &root.join("map.json"), 1).err().unwrap().to_string().contains("unsafe relative"));
    }
    let outside = root.with_extension("outside"); fs::create_dir(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();
    fs::write(root.join("map.json"), br#"{"a":{"file":"escape/a.safetensors","dtype":"U8","shape":[1]}}"#).unwrap();
    assert!(Plan::open(&root.join("config.json"), &root.join("map.json"), 1).err().unwrap().to_string().contains("escapes model root"));
    fs::remove_dir_all(&root).unwrap(); fs::remove_dir(&outside).unwrap();
}
#[test]
#[ignore = "requires local config.json and tensor-map.json metadata, no weights"]
fn complete_checkpoint_metadata() {
    let root=std::path::PathBuf::from(std::env::var_os("VADE_FP4_METADATA").expect("metadata directory"));
    let p=Plan::open(&root.join("config.json"),&root.join("tensor-map.json"),1).unwrap(); assert_eq!((p.files.len(),p.model.shape.tensors.len()),(375487,375487));
    for (world,kv) in [(1,1811939328),(2,1006632960),(4,1006632960)] { for rank in 0..world { let s=engine_model::flash::shard::ShardPlan::new(&p.model.shape,&p.files,rank,world).unwrap(); assert!(s.budget.resident>0); assert_eq!(s.tensors.len(),375487); assert_eq!(s.cache_bytes(32768).unwrap().0,kv); } }
    assert!(Plan::open(&root.join("config.json"),&root.join("tensor-map.json"),2).is_err()); let c=engine_model::flash::Config::open(&root.join("config.json")).unwrap(); assert!(c.mixed.is_some());
    let n=p.nvfp4("model.language_model.layers.0.mlp.experts.0.gate_proj").unwrap(); assert_eq!((n.rows,n.columns),(640,2560)); assert!(matches!(n.scales,ScaleView::Csf {..})); assert!(n.input.is_some());
    let row=p.ple_row(127,2500011).unwrap(); assert_eq!((row.packed_bytes,row.scale_bytes),(80,10)); assert!(p.ple_row(128,0).is_err()); assert!(p.ple_row(0,2500012).is_err());
    assert_eq!(p.cache_bytes(32768,false).unwrap().0,1811939328); assert_eq!(p.cache_bytes(32768,true).unwrap().0,1962934272);
    let mut h=p.model.shape.tensors.iter().map(|t|(t.name.clone(),engine_model::safetensors::Tensor {dtype:t.dtype,shape:t.shape.clone(),start:0,end:0})).collect::<std::collections::BTreeMap<_,_>>();
    let scale="model.language_model.layers.0.linear_attn.in_proj_a.weight_scale"; h.get_mut(scale).unwrap().shape[1]+=1; assert!(p.model.clone().with_headers(&h).is_err()); h.remove(scale); assert!(p.model.clone().with_headers(&h).is_err());
}

#[test]
#[ignore = "requires the real checkpoint and tensor-map metadata"]
fn complete_checkpoint_loader() {
    use engine_model::flash::Checkpoint;
    let root = std::path::PathBuf::from(std::env::var_os("VADE_FP4_CHECKPOINT").expect("checkpoint directory"));
    let map = std::path::PathBuf::from(std::env::var_os("VADE_FP4_TENSOR_MAP").expect("tensor map"));
    let c = Checkpoint::inspect(&root).unwrap();
    c.prepare_load(&root).unwrap();
    let p = Plan::open(&root.join("config.json"), &map, 1).unwrap();
    let bound = c.mixed_plan(1).unwrap();
    assert_eq!((c.shards.len(), c.index.weight_map.len()), (21, 375487));
    assert_eq!(bound.files, p.files);
    let totals = |p: &Plan| p.classes.iter().map(|(c, t)| (*c, t.tensors, t.bytes)).collect::<Vec<_>>();
    assert_eq!(totals(&bound), totals(&p));
    println!("public_inspect=PASS prepare_load=PASS mixed_plan=PASS shards=21 tensors=375487");
}
