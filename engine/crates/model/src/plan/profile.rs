//! Logical compulsory reads, not physical DRAM traffic or GEMM tile rereads.
use crate::{qwen3::Config, shapes::*, safetensors::Dtype};
use super::ModelPlan;
pub const CLASSES: [&str; 11] = ["dense_qkv_o_gemm", "dense_ffn", "full_attention", "moe_route",
    "moe_gather", "moe_experts", "moe_shared", "norms_residual", "lm_head", "transport_collectives", "other"];
const HYBRID_CLASSES: [&str; 11] = ["dense_projections_ffn", "gdn_recurrence", "full_attention", "moe_route",
    "moe_gather", "moe_experts", "moe_shared", "norms_residual", "lm_head", "transport_collectives", "other"];
impl ModelPlan {
    pub fn profile_classes(&self) -> &'static [&'static str] {
        if matches!(self.shape.residual, Residual::Streams { .. }) { &crate::flash::plan::PREFILL_PROFILE_CLASSES }
        else if self.shape.layers.iter().any(|l| matches!(l.mixer, Mixer::Linear(_))) { &HYBRID_CLASSES } else { &CLASSES }
    }
    pub fn profile_bytes(&self, place: Place, class: usize) -> usize {
        self.shape.tensors.iter().filter(|t| t.role.place == place).filter(|t| {
            let kind = match t.role.part {
                Part::Embedding | Part::Expert { .. } => return false,
                Part::Head => 8, Part::Shared(_) => 6, Part::LinearConv | Part::LinearNorm => 1,
                Part::LinearHeads if t.shape.len() == 1 => 1,
                Part::QueryNorm | Part::KeyNorm | Part::IndexNorm => 2,
                Part::Whole => if t.shape.len() == 2 { 3 } else { 7 }, _ => 0,
            }; kind == class
        }).map(|t| t.shape.iter().product::<u64>() as usize * t.dtype.bytes() as usize).sum()
    }
    pub fn profile_state(&self, layer: usize, past: usize, rows: usize) -> (usize, usize) {
        match self.shape.layers[layer].mixer {
            Mixer::Attention(a) => (2, 4 * (past + rows) * a.kv_heads as usize * a.head_dim as usize),
            Mixer::Linear(a) => (1, (rows.div_ceil(64) as u64 * a.value_heads * a.key_dim * a.value_dim * 4
                + (2 * a.key_heads * a.key_dim + a.value_heads * a.value_dim) * (a.conv - 1) * 4) as usize),
        }
    }
}
pub fn matrix_bytes(rows: usize, cols: usize, dtype: Dtype) -> usize {
    rows * cols * dtype.bytes() as usize + if dtype == Dtype::F8E4M3 { rows.div_ceil(128) * cols.div_ceil(128) * 2 } else { 0 }
}
/// The ordinary attention adapter's plan, derived from its validated config.
pub struct ProfilePlan { pub layer: LayerShape, pub hidden: usize, pub vocab: usize }
impl ProfilePlan {
    pub fn qwen(c: &Config) -> Self {
        Self { hidden: c.hidden_size as usize, vocab: c.vocab_size as usize, layer: LayerShape {
            mixer: Mixer::Attention(AttentionShape { query_heads: c.attention_heads, kv_heads: c.kv_heads,
                head_dim: c.head_dim, rotary_dim: c.head_dim, gated: false, index: None }),
            ffn: if c.num_experts == 0 { Ffn::Dense { width: c.intermediate_size } } else {
                Ffn::Experts(ExpertShape { count: c.num_experts, selected: c.num_experts_per_tok,
                    width: c.moe_intermediate_size, shared_width: 0, storage: ExpertStorage::PerExpert }) } } }
    }
    pub fn classes(&self) -> &'static [&'static str] { &CLASSES }
    pub fn layer_name(&self) -> &'static str { if matches!(self.layer.ffn, Ffn::Experts(_)) { "attention_moe" } else { "attention_dense" } }
    pub fn reads(&self, dtype: Dtype) -> [(usize, usize); 3] {
        let Mixer::Attention(a) = self.layer.mixer else { unreachable!() };
        let h = self.hidden; let d = a.head_dim as usize;
        let projection = |r, c| matrix_bytes(r, c, dtype);
        let (class, ffn) = match self.layer.ffn { Ffn::Dense { width } => (1, 3 * projection(width as usize, h)),
            Ffn::Experts(e) => (3, projection(e.count as usize, h)) };
        [(0, 2 * projection(a.query_heads as usize * d, h) + 2 * projection(a.kv_heads as usize * d, h)),
            (class, ffn), (7, (2 * h + 2 * d) * 4)]
    }
    pub fn kv_bytes(&self, length: usize, element_bytes: usize) -> usize {
        let Mixer::Attention(a) = self.layer.mixer else { unreachable!() };
        2 * length * a.kv_heads as usize * a.head_dim as usize * element_bytes
    }
}
#[cfg(test)]
#[test]
fn qwen_moe_plan_reads() {
    let c = Config::parse(br#"{"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","torch_dtype":"bfloat16","hidden_act":"silu","hidden_size":2048,"intermediate_size":6144,"moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,"decoder_sparse_step":1,"mlp_only_layers":[],"norm_topk_prob":true,"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,"vocab_size":151936,"max_position_embeddings":40960,"rms_norm_eps":0.000001,"rope_theta":1000000,"attention_bias":false,"attention_dropout":0,"use_sliding_window":false,"sliding_window":null,"rope_scaling":null,"use_cache":true,"tie_word_embeddings":false,"bos_token_id":151643,"eos_token_id":151645}"#).unwrap();
    let p = ProfilePlan::qwen(&c); let r = p.reads(Dtype::Bf16);
    assert_eq!(p.classes(), &["dense_qkv_o_gemm", "dense_ffn", "full_attention", "moe_route", "moe_gather", "moe_experts", "moe_shared", "norms_residual", "lm_head", "transport_collectives", "other"]);
    assert_eq!((r[0].1 * 48, r[1].1 * 48, p.kv_bytes(8222, 2) * 48), (1811939328, 25165824, 808255488));
    assert_eq!((3 * matrix_bytes(768, 2048, Dtype::Bf16) * 8 * 48, matrix_bytes(p.vocab, p.hidden, Dtype::Bf16)), (3623878656, 622329856));
    assert_eq!(matrix_bytes(768, 2048, Dtype::F8E4M3), 1573056);
    assert_eq!(r.iter().map(|r| r.1).sum::<usize>() * 48 + 3623878656 + 622329856 + p.hidden * 6, 6084161536);
}
