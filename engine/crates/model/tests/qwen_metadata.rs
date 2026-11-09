//! Official metadata only: compact fixtures expand independently of the planner.
use engine_model::{
    json::{self, Value},
    plan::{ModelPlan, PlanError},
    safetensors::{Dtype, Tensor},
    shapes::{Ffn, Mixer, Part, Place, Proj, Store},
    split::{Selection, SplitError, SplitPlan, SplitRequest},
};
use std::{collections::BTreeMap, fs, path::Path};

const FIXTURES: [(&[u8], &[u8]); 2] = [
    (include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json"),
     include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.headers.json")),
    (include_bytes!("fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json"),
     include_bytes!("fixtures/qwen/Qwen3.5-122B-A10B-FP8.headers.json")),
];
const GB: u64 = 1_000_000_000;
fn json(bytes: &[u8]) -> Value {
    json::parse(bytes, json::Limits { bytes: 64 << 20, nodes: 4_000_000, ..Default::default() }).unwrap()
}
fn tensor(value: &Value) -> Tensor {
    let o = value.object().unwrap();
    let shape: Vec<_> = o["shape"].array().unwrap().iter().map(|v| v.u64().unwrap()).collect();
    let dtype = Dtype::parse(o["dtype"].string().unwrap()).unwrap();
    let end = shape.iter().product::<u64>() * dtype.bytes();
    Tensor { shape, dtype, start: 0, end }
}
fn fixture(bytes: &[u8]) -> (Value, BTreeMap<String, Tensor>) {
    let doc = json(bytes);
    let mut headers = BTreeMap::new();
    for group in doc.object().unwrap()["groups"].array().unwrap() {
        let g = group.object().unwrap();
        let mut names = vec![g["name"].string().unwrap().to_owned()];
        for (axis, values) in g["axes"].array().unwrap().iter().enumerate() {
            let ids: Vec<_> = match values {
                Value::Number(_) => (0..values.u64().unwrap()).collect(),
                _ => values.array().unwrap().iter().map(|v| v.u64().unwrap()).collect(),
            };
            names = names.iter().flat_map(|n| ids.iter().map(|id|
                n.replace(&format!("{{{axis}}}"), &id.to_string()))).collect();
        }
        for name in names { assert!(headers.insert(name, tensor(group)).is_none()); }
    }
    let o = doc.object().unwrap();
    assert_eq!(headers.len() as u64, o["tensor_count"].u64().unwrap());
    assert_eq!(headers.values().map(Tensor::byte_len).sum::<u64>(), o["stored_bytes"].u64().unwrap());
    (doc, headers)
}
fn official(root: &Path, model: &str) -> BTreeMap<String, Tensor> {
    let index = json(&fs::read(root.join(format!("{model}.json"))).unwrap());
    let map = index.object().unwrap()["weight_map"].object().unwrap();
    let mut headers = BTreeMap::new();
    for file in fs::read_dir(root.join("headers").join(model)).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().to_str().unwrap().strip_suffix(".header.json").unwrap().to_owned();
        for (key, value) in json(&fs::read(file.path()).unwrap()).object().unwrap() {
            if key == "__metadata__" { continue; }
            assert_eq!(map[key].string().unwrap(), name);
            let t = tensor(value);
            let offsets = value.object().unwrap()["data_offsets"].array().unwrap();
            assert_eq!(offsets[1].u64().unwrap() - offsets[0].u64().unwrap(), t.byte_len());
            assert!(headers.insert(key.clone(), t).is_none());
        }
    }
    assert!(headers.keys().eq(map.keys()));
    headers
}
fn request(ranks: u32, context: u64, memory: u64) -> SplitRequest {
    SplitRequest { ranks, context, memory: vec![memory; ranks as usize], reserve: vec![16 * GB; ranks as usize],
        sequences: 1, cache_bytes: 2, state_bytes: 4, draft: true, expert_rows: false }
}

// Sparse payloads prove admission reads metadata only, even at official model sizes.
fn checkpoint(root: &Path, headers: &BTreeMap<String, Tensor>) {
    use std::io::Write;
    let mut offset = 0;
    let records = headers.iter().map(|(n,t)| {
        let start = offset; offset += t.byte_len();
        let dtype = match t.dtype { Dtype::Bf16 => "BF16", Dtype::F32 => "F32", Dtype::F8E4M3 => "F8_E4M3", _ => panic!() };
        format!(r#"{n:?}:{{"dtype":{dtype:?},"shape":{:?},"data_offsets":[{start},{offset}]}}"#, t.shape)
    }).collect::<Vec<_>>().join(",");
    let header = format!("{{{records}}}"); let mut f = fs::File::create(root.join("test.safetensors")).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap(); f.write_all(header.as_bytes()).unwrap(); f.set_len(8+header.len() as u64+offset).unwrap();
    let map = headers.keys().map(|n| format!(r#"{n:?}:"test.safetensors""#)).collect::<Vec<_>>().join(",");
    fs::write(root.join("model.safetensors.index.json"), format!(r#"{{"metadata":{{"total_size":{offset}}},"weight_map":{{{map}}}}}"#)).unwrap();
}
#[test]
fn fp8_moe_http_preflight_matches_cli_admission() {
    use engine_model::{cartridge::Cartridge, qwen3_5::plan};
    for (id, (config, bytes)) in FIXTURES.into_iter().enumerate() {
        let root = std::env::temp_dir().join(format!("http-fp8-{}-{id}", std::process::id())); fs::create_dir(&root).unwrap();
        let (_, mut headers) = fixture(bytes); fs::write(root.join("config.json"), config).unwrap(); checkpoint(&root, &headers);
        let (p,c) = plan::inspect(&root, false).unwrap(); let http = Cartridge::preflight(&root).unwrap();
        assert!(plan::uses_typed_loader(&p)); assert_eq!(http.precision, "FP8"); assert_eq!(http.weight_bytes, c.text_bytes);
        assert_eq!((http.state_bytes,http.kv_bytes_per_token,http.max_positions),(c.config.state_bytes(),c.config.kv_bytes_per_token(),c.config.max_positions));
        let (draft,_) = plan::inspect(&root, true).unwrap(); assert_eq!(draft,p);
        let bytes: u64=headers.iter().filter(|(n,_)| n.starts_with("mtp.")).map(|(_,t)|t.byte_len()).sum(); assert_eq!(plan::mtp_weight_bytes(&draft).unwrap(),bytes); assert!(draft.shape.tensors.iter().filter(|t| matches!(t.role.place,Place::Draft{..})).all(|t| t.dtype==headers[&t.name].dtype && t.shape==headers[&t.name].shape)); println!("MTP {id}: header_bytes={bytes}");
        assert!(c.config.mtp_weight_bytes().unwrap_err().to_string().contains("hybrid MoE MTP is not implemented"));
        let scale = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"; headers.remove(scale).unwrap(); checkpoint(&root, &headers);
        let error = Cartridge::preflight(&root).err().unwrap().to_string(); assert_eq!(error, format!("missing tensor {scale}")); println!("PREFLIGHT {id}: accepted; missing scales refused: {error}");
        let bad = std::str::from_utf8(config).unwrap().replace("\"activation_scheme\":\"dynamic\"", "\"activation_scheme\":\"static\""); fs::write(root.join("config.json"), bad).unwrap();
        let error = Cartridge::preflight(&root).err().unwrap().to_string(); assert!(error.contains("activation_scheme") && error.contains("static"));
        for name in ["config.json","model.safetensors.index.json","test.safetensors"] { fs::remove_file(root.join(name)).unwrap(); } fs::remove_dir(root).unwrap();
    }
}

#[test]
fn official_qwen_plans_and_resident_ledgers() {
    let weights = [
        vec![vec![36561646976], vec![18310968000; 2], vec![12284124868,12244085880,12192885636],
             vec![9191396384; 4], vec![6157663420,6157667356,6177578248,6198286216,6146715420,6114463836]],
        vec![vec![126249731584], vec![63182739200; 2], vec![42245495096,42222987580,42110433580],
             vec![31659467872; 4], vec![21179088148,21265286692,21251456068,21202610548,21101836440,21044026632]],
    ];
    let states = [
        vec![vec![66846720], vec![33423360;2], vec![23040000,23040000,20889600],
             vec![16711680;4], vec![12533760,12533760,10506240,10506240,10506240,10506240]],
        vec![vec![158072832], vec![79036416;2], vec![54411264,51978240,51978240],
             vec![39518208;4], vec![27205632,27353088,27353088,27205632,24772608,24772608]],
    ];
    for (model_id, (config, bytes)) in FIXTURES.into_iter().enumerate() {
        let (doc, headers) = fixture(bytes);
        let model = doc.object().unwrap()["model"].string().unwrap();
        if let Ok(root) = std::env::var("VADE_QWEN_METADATA") {
            let root = Path::new(&root);
            assert_eq!(json(config), json(&fs::read(root.join(format!("{model}.config.json"))).unwrap()));
            assert_eq!(headers, official(root, model));
        }
        let plan = ModelPlan::from_config(config).unwrap().with_headers(&headers).unwrap();
        let projections = plan.hybrid_projections().unwrap();
        for p in &projections { assert_eq!(p.row_bytes(0..p.rows).unwrap().end, p.rows*p.cols*p.weight.dtype.bytes()); assert!(p.row_bytes(p.rows..p.rows+1).is_err()); if let engine_model::plan::runtime::ProjectionStorage::Fp8 { scales, .. } = p.storage { assert_eq!(p.scale_index(p.rows-1,p.cols-1).unwrap(),scales.shape[0]*scales.shape[1]-1); assert!(p.scale_index(p.rows,0).is_err()); } }
        let mut overflow = projections[0]; overflow.rows=u64::MAX; overflow.cols=u64::MAX; assert!(matches!(overflow.row_bytes(0..u64::MAX),Err(PlanError::Overflow { .. })));
        let s = &plan.shape;
        assert_eq!((s.hidden, s.layers.len(), s.draft.len()), if model_id == 0 { (2048,40,1) } else { (3072,48,1) });
        assert_eq!(s.layers.iter().filter(|l| matches!(l.mixer, Mixer::Linear(_))).count(), if model_id == 0 {30} else {36});
        assert_eq!(s.tensors.iter().filter(|t| t.role.place == Place::Unused).count(),333);
        assert_eq!(s.tensors.iter().filter(|t| t.role.store == Store::BlockScale).count(),if model_id == 0 {31745} else {37939});
        assert_eq!(s.tensors.iter().filter(|t| t.dtype == Dtype::F32).count(),if model_id == 0 {0} else {72});
        assert!(s.layers.iter().all(|l| matches!(l.ffn, Ffn::Experts(e) if e.count == 256 && e.selected == 8)));
        let mut roles = BTreeMap::new();
        for t in &s.tensors {
            let part = match t.role.part { Part::Expert { proj, .. } => format!("Expert({proj:?})"), p => format!("{p:?}") };
            *roles.entry(format!("{part}/{:?}",t.role.store)).or_insert(0) += 1;
        }
        println!("PLAN {model}: roles={roles:?}");
        let text_bytes: u64 = headers.iter().filter(|(n,_)| !n.starts_with("model.visual."))
            .map(|(_,t)| t.byte_len()).sum();
        assert_eq!(SplitPlan::new(s,&request(1,131072,u64::MAX)).unwrap().ledgers[0].weights,text_bytes);
        for (case, ranks) in [1,2,3,4,6].into_iter().enumerate() {
            for context in [131072,262144] {
                let q = request(ranks, context, u64::MAX);
                let p = SplitPlan::new(s, &q).unwrap();
                assert_eq!(p, SplitPlan::new(s, &q).unwrap());
                assert_eq!(p.ledgers.iter().map(|l| l.weights).collect::<Vec<_>>(), weights[model_id][case]);
                assert_eq!(p.ledgers.iter().map(|l| l.state).collect::<Vec<_>>(), states[model_id][case]);
                for (rank, (l, o)) in p.ledgers.iter().zip(&p.owners).enumerate() {
                    assert_eq!(l.cache, context * 2 * (if model_id == 0 {11} else {13}) * 512 * (o.kv_heads.end-o.kv_heads.start));
                    let blocks: u64 = (0..256).map(|e| { let b=p.expert_blocks(rank as u32,e).unwrap(); b.end-b.start }).sum();
                    println!("RANK {model} N={ranks} ctx={context} r={rank}: weights={} cache={} state={} owners={o:?} expert_blocks={blocks}",l.weights,l.cache,l.state);
                }
                let fit = SplitPlan::new(s, &request(ranks,context,128*GB));
                if ![[true;5],[false,true,true,true,true]][model_id][case] {
                    let l=&p.ledgers[0];
                    assert_eq!(fit,Err(SplitError::NoFit { rank:0,needed:l.weights+l.cache+l.state+16*GB,available:128*GB }));
                } else { assert_eq!(fit.as_ref().unwrap(),&p); }
                println!("FIT {model} N={ranks} ctx={context}: {fit:?}",fit=fit.as_ref().map(|_| "yes"));
            }
        }
    }
}

#[test]
fn config_alternatives_and_scale_boundaries() {
    let config = std::str::from_utf8(FIXTURES[1].0).unwrap();
    let parse = |s: &str| ModelPlan::from_config(s.as_bytes());
    assert!(parse(config).is_ok());
    assert!(parse(config).unwrap().hybrid_projections().is_err());
    let conflict = config.replace("\"text_config\":{", "\"text_config\":{\"tie_word_embeddings\":true,");
    assert!(matches!(parse(&conflict),Err(PlanError::Unsupported { field:"tie_word_embeddings", .. })));
    for field in ["weight_per_tensor", "act_per_tensor"] {
        assert!(matches!(parse(&config.replace(&format!("\"{field}\":false"), &format!("\"{field}\":true"))),Err(PlanError::Unsupported { .. })));
    }
    assert!(matches!(parse(&config.replace("\"quant_method\":\"fp8\"", "\"fmt\":\"e5m2\",\"quant_method\":\"fp8\"")),Err(PlanError::Unsupported { field:"fmt", .. })));
    let (_, mut headers) = fixture(FIXTURES[0].1);
    let plan = ModelPlan::from_config(FIXTURES[0].0).unwrap().with_headers(&headers).unwrap();
    let mut bad=plan.clone(); bad.shape.residual=engine_model::shapes::Residual::Streams { count:2,rank:8 }; assert!(bad.hybrid_projections().is_err());
    let mut bad=plan.clone(); bad.shape.tensors.iter_mut().find(|t| t.role.store==Store::BlockScale).unwrap().dtype=Dtype::F32; assert!(bad.hybrid_projections().is_err());
    let mut bad=plan.clone(); if let Ffn::Experts(e)=&mut bad.shape.layers[0].ffn { e.storage=engine_model::shapes::ExpertStorage::Stacked; } assert!(bad.hybrid_projections().is_err());
    let p = SplitPlan::new(&plan.shape, &request(3,131072,u64::MAX)).unwrap();
    for (suffix, selections) in [
        ("gate_proj.weight_scale_inv", vec![Selection::Rows(vec![0..2]),Selection::Rows(vec![1..3]),Selection::Rows(vec![2..4])]),
        ("down_proj.weight_scale_inv",vec![Selection::Columns(0..2),Selection::Columns(1..3),Selection::Columns(2..4)]),
    ] {
        let name=format!("model.language_model.layers.0.mlp.shared_expert.{suffix}");
        let t=plan.shape.tensors.iter().find(|t| t.name==name).unwrap();
        for (rank,expected) in selections.into_iter().enumerate() { assert_eq!(p.selection(rank as u32,t).unwrap(),expected); }
    }
    let name="model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv";
    headers.remove(name);
    assert_eq!(
        ModelPlan::from_config(FIXTURES[0].0)
            .unwrap()
            .with_headers(&headers)
            .unwrap_err(),
        PlanError::MissingTensor { name: name.into() }
    );
    let mut scalar = plan
        .shape
        .tensors
        .iter()
        .find(|t| t.role.part == Part::Shared(Proj::Gate))
        .unwrap()
        .clone();
    scalar.role.store = Store::Scalar;
    assert!(matches!(
        p.selection(0, &scalar),
        Err(SplitError::Unsupported { .. })
    ));
    scalar.role.store = Store::BlockScale;
    scalar.shape = vec![1];
    assert!(matches!(
        p.selection(0, &scalar),
        Err(SplitError::Unsupported { .. })
    ));
    scalar.shape = vec![u64::MAX, 1];
    assert!(matches!(
        p.selection(0, &scalar),
        Err(SplitError::Overflow { .. })
    ));
}

#[test]
fn fp8_refuses_320001_tensors() {
    use engine_model::{
        safetensors::{Index, Limits},
        shapes::*,
    };
    for (config, bytes) in FIXTURES {
        let (_, mut headers) = fixture(bytes);
        let mut payload = 0;
        for t in headers.values_mut() {
            let bytes = t.byte_len();
            t.start = payload;
            payload += bytes;
            t.end = payload;
        }
        for id in headers.len()..320_001 {
            headers.insert(
                format!("model.visual.audit_{id}"),
                Tensor {
                    dtype: Dtype::U8,
                    shape: vec![0],
                    start: payload,
                    end: payload,
                },
            );
        }
        assert_eq!(headers.len(), 320_001);
        assert_eq!(headers.values().map(Tensor::byte_len).sum::<u64>(), payload);
        let error = ModelPlan::from_config(config)
            .unwrap()
            .with_headers(&headers)
            .unwrap_err();
        assert_eq!(
            error,
            PlanError::Invalid {
                field: "tensors",
                reason: "header count exceeds limit"
            }
        );
        let map = headers
            .keys()
            .map(|n| format!(r#"{n:?}:"test.safetensors""#))
            .collect::<Vec<_>>()
            .join(",");
        let index = format!(r#"{{"metadata":{{"total_size":{payload}}},"weight_map":{{{map}}}}}"#);
        assert!(index.len() < HEADER_BYTES);
        let limits = Limits {
            header_bytes: HEADER_BYTES,
            tensors: MAX_TENSORS,
            rank: MAX_TENSOR_RANK,
            shards: MAX_SHARDS,
        };
        assert!(
            Index::parse(index.as_bytes(), limits)
                .unwrap_err()
                .to_string()
                .contains("excessive index tensor count")
        );
    }
}
