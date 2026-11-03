//! A checkpoint-derived, typed model plan, independent of loading and serving.
mod config;
mod tensors;
mod error;
pub mod description;
pub mod glm;
pub mod glm_index;
pub mod storage;
pub mod layer;
pub use description::{ConfigPlan, TextDescription};
pub mod adapter;
pub use adapter::{DeferredPlan, MappedTensor};
#[cfg(test)]
mod mtp_tests;
pub mod dsa;
#[cfg(test)]
mod mixed_tests;
#[cfg(test)]
mod ffn_tests;
pub use crate::host_ops::FfnKind;
pub mod profile;
pub mod runtime;
pub use error::PlanError;
use crate::{safetensors::{self, Index, Limits, Tensor}, shapes::*};
use std::{collections::BTreeMap, path::Path};
type Result<T> = std::result::Result<T, PlanError>;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPlan {
    pub shape: ModelShape,
    pub rope: Rope,
    pub epsilon: f64,
    pub eos_token: u64,
    pub bos_token: u64,
    pub renormalize_top_k: bool,
    pub format: Format,
    schema: config::Schema,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    pub theta: f64,
    pub rotary_dim: u64,
    pub sections: Vec<u64>,
    pub interleaved: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Format {
    Bf16,
    Fp8 { block: [u64; 2], unquantized: Vec<String> },
    Mixed(crate::flash::mixed::Mixed),
}
impl ModelPlan {
    fn tensor_limit(&self) -> usize {
        if matches!(self.format, Format::Mixed(_)) {
            crate::flash::mixed::MAX_MIXED_TENSORS
        } else {
            MAX_TENSORS
        }
    }
    /// Parse bounded config metadata; tensor inventory, tying and lexical rows await headers.
    pub fn from_config(config_json: &[u8]) -> Result<Self> {
        adapter::ConfigPlan::from_config(config_json)?.into_model_plan()
    }
    /// Reconcile stored inventory; absent lm_head reuses embeddings; config contradictions refuse.
    pub fn with_headers(self, tensors: &BTreeMap<String, Tensor>) -> Result<Self> {
        tensors::reconcile(self, tensors)
    }
    /// Read bounded config, index and safetensors headers using the existing readers.
    pub fn inspect(root: &Path) -> Result<Self> {
        Self::inspect_parts(root).map(|(plan, _, _)| plan)
    }
    /// Reconciled plan and the exact inspected file metadata; never reads payloads.
    pub fn inspect_parts(root: &Path) -> Result<(Self, Index, BTreeMap<String, safetensors::Header>)> {
        let bytes = safetensors::read_bounded(&root.join("config.json"), CONFIG_BYTES)
            .map_err(|e| read_error("config.json", e))?;
        let plan = Self::from_config(&bytes)?;
        let limits = Limits {
            header_bytes: HEADER_BYTES,
            tensors: plan.tensor_limit(),
            rank: MAX_TENSOR_RANK,
            shards: MAX_SHARDS,
        };
        let path = root.join("model.safetensors.index.json");
        let index = if matches!(plan.format, Format::Mixed(_)) {
            safetensors::read_bounded(&path, limits.header_bytes)
                .and_then(|bytes| Index::parse_mixed(&bytes, limits))
        } else {
            Index::open(&path, limits)
        }
        .map_err(|e| read_error("model.safetensors.index.json", e))?;
        let shards = index
            .inspect_shards(root, limits)
            .map_err(|e| read_error("safetensors headers", e))?;
        let tensors = shards
            .values()
            .flat_map(|header| header.tensors.iter().map(|(n, t)| (n.clone(), t.clone())))
            .collect();
        Ok((plan.with_headers(&tensors)?, index, shards))
    }
}
fn read_error(file: &'static str, error: impl std::fmt::Display) -> PlanError {
    PlanError::Read { file, message: error.to_string() }
}
fn invalid(field: &'static str, reason: &'static str) -> PlanError {
    PlanError::Invalid { field, reason }
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(PlanError::Overflow { what: "dimension product" })
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(PlanError::Overflow { what: "dimension sum" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flash, qwen3_5, safetensors::Dtype};
    pub(super) fn synthetic(h: u64, layers: u64, q: u64, kv: u64, v: u64, k: u64, experts: u64, top: u64, width: u64) -> String {
        let interval = if layers == 2 { 2 } else { 4 }; let kinds = (0..layers).map(|i| if (i+1)%interval==0 { "\"full_attention\"" } else { "\"linear_attention\"" }).collect::<Vec<_>>().join(",");
        format!(r#"{{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForConditionalGeneration"],"tie_word_embeddings":false,"text_config":{{"model_type":"qwen3_5_moe_text","dtype":"bfloat16","hidden_act":"silu","output_gate_type":"swish","attn_output_gate":true,"attention_bias":false,"attention_dropout":0,"tie_word_embeddings":false,"use_cache":true,"mamba_ssm_dtype":"float32","hidden_size":{h},"vocab_size":248320,"max_position_embeddings":262144,"eos_token_id":248044,"rms_norm_eps":0.000001,"head_dim":256,"num_attention_heads":{q},"num_key_value_heads":{kv},"linear_num_key_heads":{k},"linear_num_value_heads":{v},"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_hidden_layers":{layers},"layer_types":[{kinds}],"full_attention_interval":{interval},"num_experts":{experts},"num_experts_per_tok":{top},"moe_intermediate_size":{width},"shared_expert_intermediate_size":{width},"mtp_num_hidden_layers":0,"rope_parameters":{{"rope_type":"default","rope_theta":10000000,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25}}}},"quantization_config":{{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128],"modules_to_not_convert":["lm_head"]}}}}"#)
    }
    pub(super) fn hybrid_headers(text: &str) -> BTreeMap<String, Tensor> {
        let c=qwen3_5::Config::parse(text.as_bytes()).unwrap(); let mut shapes=c.stored_shapes(); if c.mtp_layers!=0 { for (name,shape) in c.mtp_shapes().unwrap() { if c.quantized(&name,&shape) { shapes.insert(format!("{name}_scale_inv"),vec![shape[0].div_ceil(FP8_BLOCK),shape[1].div_ceil(FP8_BLOCK)]); } shapes.insert(name,shape); } }
        shapes.into_iter().map(|(n,shape)| { let dtype=if c.quantized(&n,&shape) { Dtype::F8E4M3 } else { Dtype::Bf16 }; (n,Tensor { dtype,shape,start:0,end:0 }) }).collect()
    }
    fn flash_formats() -> [String; 2] {
        let fp8 = flash_fixture();
        let prefix = fp8.rsplit_once("\"quantization_config\":").unwrap().0;
        let mixed = format!(r#"{prefix}"quantization_config":{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","ignore":[],"quantized_layers":{{"model.language_model.layers":{{"quant_algo":"NVFP4","group_size":16}},"lm_head":{{"quant_algo":"MXFP8","group_size":32}}}}}}}}"#);
        [fp8, mixed]
    }
    #[test]
    fn qwen_flash_next_plans_match_base_bytes() {
        let expected = [include_bytes!("../../tests/fixtures/ffn-plan/flash-next-fp8.plan.txt").as_slice(),
            include_bytes!("../../tests/fixtures/ffn-plan/flash-next-mixed.plan.txt").as_slice()];
        for (config, expected) in flash_formats().iter().zip(expected) {
            // Invalid deferred fields must never reach the deferred FFN parser.
            let unused = config.replace("\"hidden_size\":", "\"first_k_dense_replace\":true,\"mlp_layer_types\":false,\"hidden_size\":");
            for config in [config, &unused] {
                let plan = ModelPlan::from_config(config.as_bytes()).unwrap();
                assert_eq!(format!("{plan:?}").as_bytes(), expected);
                assert_eq!(ConfigPlan::from_config(config.as_bytes()).unwrap(), ConfigPlan::Supported(plan));
            }
        }
    }
    #[test]
    #[ignore = "skipped: no checkpoint; requires VADE_FLASH_CONFIG"]
    fn real_flash_config() {
        let path = std::env::var("VADE_FLASH_CONFIG").expect("explicit checkpoint test requires VADE_FLASH_CONFIG");
        let bytes = std::fs::read(path).unwrap(); let p = ModelPlan::from_config(&bytes).unwrap(); let c = flash::Config::parse(&bytes).unwrap();
        let s = &p.shape; let h = s.hidden; let Mixer::Linear(l) = s.layers.first().unwrap().mixer else { panic!() }; let Mixer::Attention(a) = s.layers.get(3).unwrap().mixer else { panic!() }; let Ffn::Experts(e) = s.layers.first().unwrap().ffn else { panic!() };
        let Residual::Streams { count: streams, rank } = s.residual else { panic!() }; let x = a.index.unwrap(); let lex = s.lexical.unwrap(); let ple = p.schema.ple.unwrap(); let Format::Fp8 { block, .. } = p.format else { panic!() };
        assert_eq!((c.hidden,c.vocab,c.max_positions,c.eos_token,c.epsilon,c.rope_theta,c.rotary_dim,&c.mrope_sections,c.renormalize_top_k,c.block,c.format),(h,s.vocab,s.max_positions,p.eos_token,p.epsilon,p.rope.theta,p.rope.rotary_dim,&p.rope.sections,p.renormalize_top_k,block,flash::WeightFormat::Fp8));
        assert_eq!((c.query_heads,c.kv_heads,c.head_dim,c.linear_key_heads,c.linear_value_heads,c.linear_key_dim,c.linear_value_dim,c.linear_conv),(a.query_heads,a.kv_heads,a.head_dim,l.key_heads,l.value_heads,l.key_dim,l.value_dim,l.conv));
        assert_eq!((c.experts,c.top_k,c.expert_width,c.shared_width,c.branches,c.mix_rank,c.index_heads,c.index_dim,c.index_budget,c.index_compression),(e.count,e.selected,e.width,e.shared_width,streams,rank,x.heads,x.dim,x.budget,x.compression));
        assert_eq!((&c.ple_layers,c.ngram_size,c.ngram_heads,c.ple_dim,c.ple_conv,c.ngram_parts,c.mtp_layers),(&vec![lex.layer],ple.size,ple.heads,ple.embed,ple.conv,lex.parts,s.draft.len() as u64)); assert_eq!(c.layers,s.layers.iter().map(|v| match v.mixer { Mixer::Linear(_) => flash::LayerKind::DeltaNet, _ => flash::LayerKind::SparseAttention }).collect::<Vec<_>>());
        assert!(s.tensors.is_empty()); assert_eq!(lex.part_rows,0); assert!(matches!(s.draft.first().unwrap().mixer,Mixer::Attention(AttentionShape { index: Some(_), .. })));
        let inv = tensors::expected(&p).unwrap(); let shape = |n: &str| inv.get(n).unwrap().shape.clone(); let base = "model.language_model.layers"; let ls = |n: &str| shape(&format!("{base}.0.{n}")); let ats = |n: &str| shape(&format!("{base}.3.self_attn.{n}"));
        let linear_ids = s.layers.iter().enumerate().filter_map(|(i,v)| matches!(v.mixer,Mixer::Linear(_)).then_some(i)).collect::<Vec<_>>();
        macro_rules! r { ($label:literal,$derived:expr,$literal:expr) => { ($label,format!("{:?}",$derived),format!("{:?}",$literal)) }; }
        let rows = [
            r!("model/src/flash/mod.rs:224",(h,s.layers.len()),(2560,48)), r!("model/src/flash/mod.rs:228",(a.query_heads,a.kv_heads,a.head_dim),(24,2,256)), r!("model/src/flash/mod.rs:238",(l.key_heads,l.value_heads,l.key_dim,l.value_dim,l.conv),(16,48,128,128,4)),
            r!("model/src/flash/mod.rs:247",(e.count,e.selected,e.width,e.shared_width),(512,10,640,640)), r!("model/src/flash/mod.rs:250",p.renormalize_top_k,true), r!("model/src/flash/mod.rs:252",(streams,rank,s.vocab),(4,320,248320)), r!("model/src/flash/mod.rs:256",(p.rope.rotary_dim,&p.rope.sections),(64,&vec![11,11,10])),
            r!("model/src/flash/mod.rs:266",(x.heads,x.dim,x.budget,x.compression,x.kv_heads),(4,128,2048,4,1)), r!("model/src/flash/mod.rs:270-277",(lex.layer,lex.parts,lex.row_dim),(1,128,160)), r!("model/src/flash/mod.rs:282",(s.draft.len(),matches!(s.draft.first().unwrap().mixer,Mixer::Attention(AttentionShape { index: Some(_), .. }))),(1,true)),
            r!("model/src/flash/mod.rs:294-300",(s.layers.len()-linear_ids.len(),linear_ids.len(),s.layers.iter().enumerate().all(|(i,v)| matches!(v.mixer,Mixer::Attention(_))==(i%4==3))),(12,36,true)), r!("model/src/flash/mod.rs:159,220",block,[128,128]),
            r!("model/src/flash/mod.rs:428-430",(ls("mlp.experts.0.gate_proj.weight"),ls("mlp.experts.0.up_proj.weight"),ls("mlp.experts.0.down_proj.weight")),(vec![640,2560],vec![640,2560],vec![2560,640])), r!("model/src/flash/mod.rs:443",(ls("mlp.experts.0.gate_proj.weight_scale_inv"),ls("mlp.experts.0.up_proj.weight_scale_inv"),ls("mlp.experts.0.down_proj.weight_scale_inv")),(vec![5,20],vec![5,20],vec![20,5])), r!("model/src/flash/mod.rs:454",ls("mlp.gate.weight"),vec![512,2560]),
            r!("model/src/flash/mod.rs:462-470",(ls("linear_attn.conv1d.weight"),ls("linear_attn.in_proj_qkv.weight"),ls("linear_attn.in_proj_z.weight"),ls("linear_attn.out_proj.weight"),ls("linear_attn.A_log"),ls("linear_attn.norm.weight")),(vec![10240,1,4],vec![10240,2560],vec![6144,2560],vec![2560,6144],vec![48],vec![128])),
            r!("model/src/flash/mod.rs:473-481",(ats("q_proj.weight"),ats("k_proj.weight"),ats("v_proj.weight"),ats("o_proj.weight"),ats("q_norm.weight"),ats("k_norm.weight"),ats("indexer.index_qk_proj.weight"),ats("indexer.q_layernorm.weight"),ats("indexer.k_layernorm.weight")),(vec![12288,2560],vec![512,2560],vec![512,2560],vec![2560,6144],vec![256],vec![256],vec![640,2560],vec![128],vec![128])),
            r!("model/src/flash/mod.rs:495-498",(shape("mtp.pre_fc_norm_hidden.weight"),shape("mtp.fc_embedding.weight"),shape("mtp.fc_hidden.weight")),(vec![10240],vec![2560,2560],vec![2560,2560])), r!("model/src/flash/mod.rs:549-559",(ls("attn_hyper_connection.hc_norm.weight"),ls("attn_hyper_connection.input_mix_weight_down.weight"),ls("attn_hyper_connection.input_mix_weight_up.weight"),ls("attn_hyper_connection.block_inject_weight.weight")),(vec![10240],vec![320,10240],vec![10240,320],vec![4,10240])),
            r!("model/src/flash/plan.rs:151-152",(a.query_heads,l.value_heads),(24,48)), r!("model/src/flash/plan.rs:160-161",(a.query_heads/a.kv_heads,l.value_heads/l.key_heads),(12,3)), r!("model/src/flash/plan.rs:166-167",(e.shared_width/BF16_GROUP,s.vocab),(80,248320)), r!("model/src/flash/plan.rs:176,209,257",(e.width/FP8_BLOCK,e.count,lex.parts),(5,512,128)),
            r!("model/src/flash/plan.rs:278-280",(2*a.head_dim,a.head_dim),(512,256)), r!("model/src/flash/plan.rs:285",(x.heads+x.kv_heads)*x.dim,640), r!("model/src/flash/plan.rs:293-294",(l.key_dim,l.value_dim),(128,128)), r!("model/src/flash/plan.rs:298-299",(l.key_heads*l.key_dim,2*l.key_heads*l.key_dim),(2048,4096)),
            r!("engine/src/flash/blocks.rs:10-12",(h,streams*h,rank),(2560,10240,320)), r!("engine/src/flash/gdn.rs:10-11",(h,l.key_dim),(2560,128)), r!("engine/src/flash/qsa.rs:9-11",(h,a.head_dim,(x.heads+x.kv_heads)*x.dim),(2560,256,640)), r!("engine/src/flash/moe.rs:11-13",(e.count,e.selected,e.width),(512,10,640)), r!("engine/src/flash/moe.rs:20-21",[h/FP8_BLOCK,e.width/FP8_BLOCK],[20,5]),
            r!("engine/src/flash/rows.rs:9-14",(h,streams,ple.heads*(ple.size-1),lex.row_dim,ple.heads*(ple.size-1)*lex.row_dim,p.eos_token,s.vocab),(2560,4,16,160,2560,248044,248320)), r!("engine/src/flash/candidate.rs:7",s.vocab,248320), r!("engine/src/flash/session.rs:509-510",(linear_ids.len(),l.key_dim,l.value_dim,s.layers.len()-linear_ids.len(),a.head_dim,x.dim),(36,128,128,12,256,128)), r!("engine/src/flash/session.rs:595",h/FP8_BLOCK,20),
            r!("gpu/src/fp8_abi.rs:89",(e.selected,h/2),(10,1280)), r!("gpu/src/fp8_abi.rs:111-124",(h/FP8_BLOCK,e.width/FP8_BLOCK),(20,5)), r!("gpu/kernels/src/flash_moe.rs:112,135,180-181",(e.selected,e.count,[e.selected+1,h]),(10,512,[11,2560])), r!("cli/src/test_gate/determinism.rs:529",s.vocab*4,993280), r!("cli/src/evals/gdn.rs:9",[l.key_dim,l.value_dim],[128,128]), r!("cli/src/evals/gdn.rs:11-14",linear_ids,(0..48).filter(|i| i%4!=3).collect::<Vec<_>>()),
        ];
        for (label,derived,literal) in rows { println!("ROW\t{label}\t{derived}\t{literal}\t{}",derived==literal); assert_eq!(derived,literal,"{label}"); }
    }
    #[test]
    fn hybrid_configs_and_typed_errors() {
        for [h,l,q,kv,v,k,e,top,w] in [[2048,40,16,2,32,16,256,8,512],[3072,48,32,2,64,16,256,8,1024]] {
            let p=ModelPlan::from_config(synthetic(h,l,q,kv,v,k,e,top,w).as_bytes()).unwrap(); assert_eq!((p.shape.hidden,p.shape.layers.len()),(h,l as usize));
            for (i,s) in p.shape.layers.iter().enumerate() { match s.mixer { Mixer::Linear(a) => { assert_ne!(i%4,3); assert_eq!((a.key_heads,a.value_heads),(k,v)); }, Mixer::Attention(a) => { assert_eq!(i%4,3); assert_eq!((a.query_heads,a.kv_heads,a.index),(q,kv,None)); } } assert_eq!(s.ffn,Ffn::Experts(ExpertShape { count:e,selected:top,width:w,shared_width:w,storage:ExpertStorage::Stacked })); }
        }
        let text=synthetic(128,2,4,2,4,2,4,2,128); let parse=|s: &str| ModelPlan::from_config(s.as_bytes());
        assert!(matches!(parse(&text.replace("\"hidden_size\":128,","")),Err(PlanError::Missing { field:"hidden_size" }))); assert!(matches!(parse(&text.replace("\"num_attention_heads\":4","\"num_attention_heads\":0")),Err(PlanError::Invalid { .. })));
        assert!(matches!(parse(&text.replace("\"num_key_value_heads\":2","\"num_key_value_heads\":3")),Err(PlanError::Invalid { .. }))); assert!(matches!(parse(&text.replace("full_attention\"","unknown_attention\"")),Err(PlanError::Unsupported { field:"layer_types", .. })));
        let p=parse(&text).unwrap(); let headers=hybrid_headers(&text); let reconciled=p.clone().with_headers(&headers).unwrap(); assert_eq!(reconciled.shape.tensors.len(),headers.len());
        assert_eq!((reconciled.profile_classes()[1], reconciled.profile_bytes(Place::Layer(0), 0)), ("gdn_recurrence", 263204));
        assert_eq!(reconciled.profile_state(0, 0, 1), (1, 274432));
        let omitted=text.replace("\"tie_word_embeddings\":false,",""); let tied=text.replace("\"tie_word_embeddings\":false","\"tie_word_embeddings\":true"); let mut shared=headers.clone(); shared.remove("lm_head.weight"); for (cfg,hs,want) in [(&omitted,&headers,false),(&omitted,&shared,true),(&tied,&shared,true)] { let plan=parse(cfg).unwrap().with_headers(hs).unwrap(); assert_eq!(plan.schema.tied,want); assert_eq!(plan.shape.tensors.len(),hs.len()); assert_eq!(plan.shape.tensors.iter().filter(|e| e.role.part==Part::Head).count(),usize::from(!want)); assert!(plan.shape.tensors.iter().any(|e| e.role.part==Part::Embedding)); assert_eq!(plan.clone().with_headers(hs).unwrap(),plan); }
        for (cfg,hs) in [(&text,&shared),(&tied,&headers)] { assert_eq!(parse(cfg).unwrap().with_headers(hs).unwrap_err(),PlanError::Invalid { field:"tie_word_embeddings",reason:"config contradicts checkpoint head storage" }); } assert!(!parse(&omitted).unwrap().with_headers(&shared).unwrap().with_headers(&headers).unwrap().schema.tied); assert_eq!(parse(&text.replace("\"tie_word_embeddings\":false","\"tie_word_embeddings\":null")).unwrap_err(),PlanError::WrongType { field:"tie_word_embeddings",expected:"boolean" });
        for suffix in ["A_log","dt_bias","norm.weight"] { let name=format!("model.language_model.layers.0.linear_attn.{suffix}"); let mut f32=headers.clone(); f32.get_mut(&name).unwrap().dtype=Dtype::F32; assert_eq!(p.clone().with_headers(&f32).unwrap().shape.tensors.iter().find(|e| e.name==name).unwrap().dtype,Dtype::F32); }
        let mut wrong_dtype=headers.clone(); wrong_dtype.get_mut("model.language_model.layers.0.linear_attn.in_proj_a.weight").unwrap().dtype=Dtype::F32; assert!(matches!(p.clone().with_headers(&wrong_dtype),Err(PlanError::Unsupported { field:"tensor dtype", .. })));
        let dense=text.replace("qwen3_5_moe","qwen3_5").replace("Qwen3_5Moe","Qwen3_5").replace("moe_intermediate_size","intermediate_size").replace("\"mtp_num_hidden_layers\":0","\"mtp_num_hidden_layers\":1"); let d=parse(&dense).unwrap(); assert_eq!(d.shape.draft.len(),1); d.with_headers(&hybrid_headers(&dense)).unwrap();
        assert!(matches!(parse(&text.replace("\"mtp_num_hidden_layers\":0,", "")),Err(PlanError::Missing { field:"mtp_num_hidden_layers" }))); assert_eq!(parse(&text.replace("\"mtp_num_hidden_layers\":0","\"mtp_num_hidden_layers\":1")).unwrap().shape.draft.len(),1);
        for model in ["qwen4_exp","qwen3_5_moe","qwen3_5"] { assert_eq!(parse(&format!(r#"{{"model_type":"{model}"}}"#)).unwrap_err(),PlanError::Missing { field:"text_config" }); }
        let scale=headers.keys().find(|n| n.ends_with("_scale_inv")).unwrap().clone(); let mut missing=headers.clone(); missing.remove(&scale); assert_eq!(p.clone().with_headers(&missing).unwrap_err(),PlanError::MissingTensor { name:scale.clone() });
        let mut wrong=headers.clone(); wrong.get_mut(&scale).unwrap().shape=vec![9,9]; assert!(matches!(p.clone().with_headers(&wrong),Err(PlanError::TensorMismatch { .. })));
        let mut vision=headers.clone(); vision.insert("model.visual.foo.mlp.experts.0.gate_proj.weight".into(),Tensor { dtype:crate::safetensors::Dtype::Bf16,shape:vec![1,1],start:0,end:0 }); let with_vision=p.clone().with_headers(&vision).unwrap(); assert_eq!(with_vision.shape.tensors.iter().find(|e| e.name.starts_with("model.visual.")).unwrap().role.place,Place::Unused);
        let mut per=p.clone(); for l in &mut per.shape.layers { if let Ffn::Experts(e)=&mut l.ffn { e.storage=ExpertStorage::PerExpert; } } let per_headers=tensors::expected(&per).unwrap().into_iter().map(|(n,e)| (n,Tensor { dtype:e.dtype,shape:e.shape,start:0,end:0 })).collect(); assert_eq!(p.clone().with_headers(&per_headers).unwrap().shape.layers,per.shape.layers);
        assert_eq!(p.clone().with_headers(&headers).unwrap().shape.tensors.iter().find(|e| e.name==scale).unwrap().role.store,Store::BlockScale); assert!(matches!(parse(&text.replace("\"shared_expert_intermediate_size\":128","\"shared_expert_intermediate_size\":0")),Err(PlanError::Invalid { .. })));
    }
    pub(super) fn flash_fixture() -> String {
        let text=synthetic(128,2,4,2,4,2,4,2,128).replace("qwen3_5_moe","qwen4_exp").replace("Qwen3_5Moe","Qwen4Exp").replace("swish","sigmoid").replace("\"fmt\":\"e4m3\"", "\"weight_per_tensor\":false,\"act_per_tensor\":false").replace("\"mtp_num_hidden_layers\":0",r#""hc_count":2,"hc_lowrank":8,"indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":64,"indexer_budget":16,"indexer_compress_ratio":2,"ple_layer_ids":[2],"heads_per_ngram":2,"ngram_size":3,"ple_embed_dim":128,"ple_conv_kernel_size":4,"split_ngram_parts":2,"mtp_use_dedicated_embeddings":false,"mtp_num_hidden_layers":1,"mtp":{"num_hidden_layers":1,"hybrid":true,"mtp_use_hidden_state_from_layer":null,"rope_theta":10000000,"layer_types":["full_attention"]}"#);
        text.replace("\"vocab_size\":248320","\"vocab_size\":256").replace("\"eos_token_id\":248044","\"eos_token_id\":255")
    }
    #[test]
    fn synthetic_flash_headers() {
        let text=flash_fixture(); let p=ModelPlan::from_config(text.as_bytes()).unwrap(); assert_eq!(p.shape.residual,Residual::Streams { count:2,rank:8 }); let Mixer::Attention(a)=p.shape.layers.get(1).unwrap().mixer else { panic!() }; assert_eq!(a.index,Some(IndexShape { heads:2,kv_heads:1,dim:64,budget:16,compression:2 })); assert_eq!(p.shape.draft.first().unwrap().mixer,Mixer::Attention(a));
        let mut filled=p.clone(); filled.shape.lexical.as_mut().unwrap().part_rows=7; let headers=tensors::expected(&filled).unwrap().into_iter().map(|(n,e)| (n,Tensor { dtype:e.dtype,shape:e.shape,start:0,end:0 })).collect::<BTreeMap<_,_>>(); let plan=p.clone().with_headers(&headers).unwrap(); assert_eq!(plan.shape.lexical,Some(LexicalShape { layer:1,parts:2,part_rows:7,row_dim:32 }));
        for (name,shape,part) in [("model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight",vec![192,128],Part::Index),("model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",vec![256],Part::Whole),("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",vec![7,32],Part::Lexical { part:1 }),("mtp.fc_hidden.weight",vec![128,128],Part::Whole)] { let e=plan.shape.tensors.iter().find(|e| e.name==name).unwrap(); assert_eq!((&e.shape,e.role.part),(&shape,part)); }
        assert!(matches!(ModelPlan::from_config(text.replace("\"indexer_n_heads\":2,","").as_bytes()),Err(PlanError::Missing { field:"indexer_n_heads" }))); let mut wrong=headers.clone(); wrong.get_mut("model.language_model.layers.0.linear_attn.A_log").unwrap().dtype=Dtype::F32; assert!(p.clone().with_headers(&wrong).is_err());
        let mut wrong=headers.clone(); wrong.get_mut("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight").unwrap().shape=vec![8,32]; assert!(matches!(p.clone().with_headers(&wrong),Err(PlanError::TensorMismatch { .. })));
        let root=std::env::temp_dir().join(format!("model-plan-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())); std::fs::create_dir(&root).unwrap(); let mut offset=0;
        let records=headers.iter().map(|(n,t)| { let start=offset; offset+=t.shape.iter().product::<u64>()*t.dtype.bytes(); let dtype=match t.dtype { Dtype::Bf16=>"BF16",Dtype::F8E4M3=>"F8_E4M3",Dtype::I64=>"I64",_=>panic!() }; format!(r#"{n:?}:{{"dtype":"{dtype}","shape":{:?},"data_offsets":[{start},{offset}]}}"#,t.shape) }).collect::<Vec<_>>().join(","); let header=format!("{{{records}}}");
        use std::io::Write; let mut file=std::fs::File::create(root.join("tiny.safetensors")).unwrap(); file.write_all(&(header.len() as u64).to_le_bytes()).unwrap(); file.write_all(header.as_bytes()).unwrap(); file.set_len(8+header.len() as u64+offset).unwrap(); drop(file);
        let map=headers.keys().map(|n| format!(r#"{n:?}:"tiny.safetensors""#)).collect::<Vec<_>>().join(","); std::fs::write(root.join("model.safetensors.index.json"),format!(r#"{{"metadata":{{"total_size":{offset}}},"weight_map":{{{map}}}}}"#)).unwrap(); std::fs::write(root.join("config.json"),text).unwrap(); assert_eq!(ModelPlan::inspect(&root).unwrap(),plan);
        for name in ["config.json","model.safetensors.index.json","tiny.safetensors"] { std::fs::remove_file(root.join(name)).unwrap(); } std::fs::remove_dir(root).unwrap();
    }
    #[test]
    #[ignore = "skipped: no checkpoint; requires VADE_FLASH_CHECKPOINT"]
    fn spark_inventory() {
        let path=std::env::var("VADE_FLASH_CHECKPOINT").expect("explicit checkpoint test requires VADE_FLASH_CHECKPOINT"); let root=Path::new(&path);
        let start=std::time::Instant::now(); let p=ModelPlan::inspect(root).unwrap(); let new_time=start.elapsed(); let start=std::time::Instant::now(); let old=flash::Checkpoint::inspect(root).unwrap(); let old_time=start.elapsed();
        let entries=p.shape.tensors.iter().filter(|e| e.role.place!=Place::Unused).map(|e| (e.name.as_str(),(e.dtype,&e.shape))).collect::<BTreeMap<_,_>>(); let old_entries=old.tensors().filter(|(n,_)| !n.starts_with("model.visual.")).map(|(n,t)| (n,(t.dtype,&t.shape))).collect::<BTreeMap<_,_>>(); assert_eq!(entries,old_entries);
        let l=p.shape.lexical.unwrap(); assert_eq!((l.parts*l.part_rows,l.part_rows,l.row_dim),(320_001_536,2_500_012,160)); println!("INSPECT new={new_time:?} old={old_time:?} nonvision={} total={} lexical={l:?}",entries.len(),p.shape.tensors.len());
    }
}
