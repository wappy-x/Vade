//! Conditional BF16 boundary requirements for the deferred operation program.
//! This is not a device plan. See CAST-CONTRACT.md for source identities and gaps.
use super::*;
use crate::flash::nvfp4::{bf16_to_f32, f32_to_bf16};

pub const EXECUTION_SHA256: &str = "9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87";
/// Documented primitive revision and current-tree gaps. It is not an admission receipt.
pub const PRIMITIVES: &str = "engine/crates/model/CAST-CONTRACT.md";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype { Bf16, F32, U32, Unresolved }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane { pub buffer: Buffer, pub dtype: Dtype }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightPlane { pub weight: Weight, pub parameter: Dtype, pub operand: Dtype }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Boundary {
    pub id: String, pub kind: Kind, pub inputs: Vec<Plane>, pub outputs: Vec<Plane>, pub scratch: Vec<Plane>,
    pub weights: Vec<WeightPlane>,
    pub accumulator: Dtype,
    /// Line spans in the saved execution source with EXECUTION_SHA256.
    pub source: &'static str,
    pub rounding: &'static str,
    /// The fixed primitive order is distinct from a framework equivalence proof.
    pub primitive_order: &'static str,
    pub refused: &'static str,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract { pub inputs: Vec<Plane>, pub steps: Vec<Boundary>, pub returns: Vec<Plane> }

impl LayerPlan {
    /// Describe the source fallback with BF16 activations and ordinary weights.
    /// Headers, quantized Linear replacements, and cache backends are unverified.
    /// This method never changes the FP32 reference or returns an executable plan.
    pub fn bf16_cast_contract(&self) -> Result<Contract> {
        self.validate()?;
        let mut planes=BTreeMap::new();
        let mut inputs=Vec::new();
        for b in &self.inputs {
            let dtype=if b.element==Element::U32 { Dtype::U32 } else { match b.name.as_str() {
                "residual"=>Dtype::Bf16,"kda.state"=>Dtype::F32,_=>Dtype::Unresolved,
            } };
            let p=Plane { buffer:b.clone(),dtype }; planes.insert(b.name.clone(),p.clone()); inputs.push(p);
        }
        // Router projection is FP32 by its consumer operation, not a model name.
        let router_inputs=self.steps.iter().filter(|s| s.kind==Kind::Route).flat_map(|s| &s.inputs).collect::<std::collections::BTreeSet<_>>();
        let mut steps=Vec::new();
        for step in &self.steps {
            let incoming=step.inputs.iter().map(|n| planes[n].clone()).collect();
            let mut outputs=Vec::new();
            for (j,b) in step.outputs.iter().enumerate() {
                let dtype=if b.element==Element::U32 { Dtype::U32 } else { match step.kind {
                    Kind::MhcPrepare | Kind::KdaPrepare | Kind::ScorePools | Kind::Route => Dtype::F32,
                    Kind::KdaStep if j>0 => Dtype::F32,
                    Kind::Conv if j>0 => Dtype::Unresolved,
                    Kind::CacheAppend => Dtype::Unresolved,
                    Kind::Pool if j<2 => Dtype::Unresolved,
                    Kind::Project if router_inputs.contains(&b.name) => Dtype::F32,
                    _=>Dtype::Bf16,
                } };
                let p=Plane { buffer:b.clone(),dtype }; planes.insert(b.name.clone(),p.clone()); outputs.push(p);
            }
            let weights=step.weights.iter().map(|w| {
                let (parameter,operand)=if w.storage!=WeightStorage::Plain { (Dtype::Unresolved,Dtype::Unresolved) }
                    else { match step.kind {
                        Kind::Conv | Kind::KdaPrepare | Kind::Route => (Dtype::F32,Dtype::F32),
                        Kind::MhcPrepare | Kind::GatedNorm | Kind::Pool => (Dtype::Bf16,Dtype::F32),
                        Kind::Project if step.outputs.iter().any(|b| router_inputs.contains(&b.name)) => (Dtype::Bf16,Dtype::F32),
                        Kind::KeyNorm => (Dtype::Bf16,Dtype::Unresolved),
                        _=>(Dtype::Bf16,Dtype::Bf16),
                    } };
                WeightPlane {weight:w.clone(),parameter,operand}
            }).collect();
            let (accumulator,source,rounding,primitive_order,refused)=rules(step.kind);
            steps.push(Boundary { id:step.id.clone(),kind:step.kind,inputs:incoming,outputs,
                scratch:step.scratch.iter().map(|b| Plane {buffer:b.clone(),dtype:Dtype::F32}).collect(),
                weights,accumulator,source,rounding,primitive_order,refused });
        }
        let returns=self.returns.iter().map(|n| planes[n].clone()).collect();
        Ok(Contract { inputs,steps,returns })
    }
}
impl Contract {
    /// Source cast sites do not establish a complete arithmetic or cache contract.
    pub fn require_complete(&self) -> Result<()> {
        Err(invalid("BF16 boundary profile is conditional; headers, backends, and reduction equivalence remain unproved"))
    }
}

fn rules(kind: Kind) -> (Dtype,&'static str,&'static str,&'static str,&'static str) {
    use Dtype::*;
    match kind {
        Kind::MhcPrepare => (F32,"209-215,296-322",
            "Widen streams and fn to FP32. Coefficients and normalized scratch stay FP32.",
            "mhc_math: ascending flattened features; row sums then source-stream column sums; columns first, then rows/columns.",
            "Framework RMS, Linear, softmax, and sum trees are absent. Hub replacement is not pinned."),
        Kind::Collapse => (F32,"328",
            "Multiply FP32 pre by widened BF16 streams. Cast the stream sum to BF16.",
            "mhc_math::collapse: separate FP32 product/add in ascending source stream.",
            "Framework sum tree is absent. mhc_math has no final BF16 cast; candidate mhc_bf16_collapse adds it."),
        Kind::Norm => (F32,"76-81",
            "FP32 squares and normalization. Cast normalized value to BF16 BEFORE BF16 weight multiply. Round product to BF16.",
            "flash_norm: lane feature partials then fixed warp shuffle. See source record; not the scalar host fold.",
            "Framework mean tree and hub RMSNorm are not pinned. flash_norm has no cast before weight."),
        Kind::Project | Kind::ExpertProject | Kind::ExpertDown | Kind::ExpandKv => (Unresolved,"97-103,130-133,159,1150-1172,1187-1192",
            "Ordinary BF16 Linear returns BF16. Router widens input and weight to FP32. Quantized replacements remain refused.",
            "Projection order depends on storage: FP8 block and FP4 oracle ascend K with FMA; BF16 and MXFP8 use their fixed tile trees.",
            "Stored headers and quantized Linear implementation are absent. No source proves its accumulator, operand quantizer, or output cast."),
        Kind::Conv => (F32,"419-425,439-448,1378",
            "Fallback widens QKV/history to FP32 conv weight dtype. FP32 convolution and SiLU return to BF16. History copy uses its destination dtype.",
            "flash_gdn convolution visits oldest tap first. The selected logical target uses precision zero.",
            "Cache class and conv backend are absent. Existing history/conv weight ABI and precision-zero output do not implement these casts."),
        Kind::KdaPrepare => (F32,"358-364,475-486,733-734",
            "Convolved BF16 QKV widen to FP32. Forget projection widens before FP32 decay. Sigmoid beta rounds to BF16 BEFORE widening.",
            "flash_kda_prepare: four features per lane, then shuffle-down 16,8,4,2,1. FP32 decay/beta.",
            "Primitive beta lacks BF16 sigmoid rounding. Framework norm tree and FLA backend are not pinned."),
        Kind::KdaStep => (F32,"475-513,530-612,763",
            "Prepared values and recurrent state stay FP32. Round recurrent output to initial Q dtype, BF16. State stays FP32.",
            "flash_kda_step: causal token order; prediction and output visit key channels in ascending order, with separate product/add.",
            "Primitive output has no BF16 cast. Framework step and chunk reductions differ; FLA source is absent."),
        Kind::GatedNorm => (F32,"382-393,766-768",
            "Widen recurrent output, gate, and weight. Normalize, weight, then sigmoid gate in FP32. Cast final value to BF16.",
            "flash_gdn_gate: four feature partials per lane, then fixed warp shuffle.",
            "Framework mean tree and RMSNormGated hub replacement are not pinned."),
        Kind::CacheAppend | Kind::Gather => (Unresolved,"1190-1199,1258-1276",
            "Preserve latent dtype and bits. Gather only counted rows. No arithmetic cast. Unused compact rows are positive zero.",
            "Copy counted rows in ascending selected raw ID. Compact attention uses local slot IDs.",
            "Cache backend storage dtype is absent. Counted compact gather is a Vade binding requirement, not a framework cache implementation."),
        Kind::KeyNorm => (Unresolved,"835-836",
            "BF16 LayerNorm returns BF16 under this profile. Internal accumulator is not specified in the saved file.",
            "pooled_indexer_params::key_norm: ascending feature mean, then centered squares, then affine output.",
            "torch LayerNorm implementation and reduction tree are absent. Primitive output is FP32."),
        Kind::Pool => (Unresolved,"838-846,948-1005",
            "Retain BF16 keys/gates. Add gate and APE in FP32. Softmax in FP32; nan_to_num; cast probabilities to BF16. Round probability/key products and pool result to BF16.",
            "pooled_indexer_params::pool_channel: ascending pool positions for max, denominator, and weighted sum.",
            "Primitive omits probability and product casts. Framework sum tree and index cache backend are absent."),
        Kind::ScorePools => (F32,"862-867",
            "Widen query and pooled keys for FP32 scores. weights_proj returns its weight dtype BEFORE widening and head scaling.",
            "pooled_indexer_params::weighted_score: ascending feature dot, scale/ReLU, then ascending signed head sum.",
            "Projection weight dtype and matmul trees are absent. These are not all-BF16 scores."),
        Kind::Select => (U32,"875-912",
            "Signed integer indices end as int32; represent invalid -1 as u32::MAX. Counts exclude padding.",
            "pooled_select: descending score, lower pool ID on ties, then ascending raw token IDs.",
            "Framework topk tie rule is absent. Logical plan covers valid tokens only; padding policy needs a session binding."),
        Kind::Scores => (Unresolved,"1088-1092,1273-1275",
            "Eager BF16 QK matmul returns BF16 BEFORE scale. Round scale product and additive mask result to BF16.",
            "qsa_common::dot: ascending key features, separate FP32 product/add.",
            "Primitive scores stay FP32. Eager/SDPA backend choice and matmul tree are not pinned."),
        Kind::Softmax => (F32,"1094-1095",
            "Eager softmax uses FP32, then casts probabilities to query dtype, BF16. Inference dropout is zero.",
            "qsa FP32 softmax: ascending counted positions for max and denominator; padding is zero.",
            "Primitive does not cast probabilities. Framework softmax and SDPA implementation are absent."),
        Kind::Values => (Unresolved,"1096-1099",
            "Eager BF16 probability/value matmul returns BF16 before output projection.",
            "qsa FP32 values: ascending counted selected position, separate product/add per value feature.",
            "Primitive output stays FP32. Framework matmul and SDPA tree are absent."),
        Kind::Route => (F32,"157-182",
            "Router logits, sigmoid, bias, group scores, selected weights, normalization, and scaling stay FP32.",
            "corrected_sigmoid_route: top pair per group; selected-score order fold; lower ID ties.",
            "Framework unsorted topk order/ties and sum tree are absent. FP32 projection needs its own binding."),
        Kind::SwiGlu => (Bf16,"97-103,136-141",
            "Round gate/up projections to BF16. Clamp with BF16 scalar bound. Round SiLU to BF16, then round product to BF16.",
            "SwiGlu reference: pointwise; no reduction. Primitive exponential is fixed by its source.",
            "Framework SiLU implementation and quantized Linear outputs are absent. Host exponential is not a checkpoint proof."),
        Kind::ExpertSum => (Bf16,"120-134,199-206",
            "Multiply BF16 down output by FP32 route weight, cast contribution to BF16, round each expert add to BF16. Add shared BF16 row last and round.",
            "ungated_moe_finish: ascending expert ID, not selected slot order. shared is added once, last.",
            "Single-token fallback fold is described. Backend expert replacement and multi-token index_add order are not pinned."),
        Kind::Combine => (Unresolved,"1336-1338,1345-1347",
            "Cast post and comb to BF16. Round post/block product and residual matmul to BF16 separately. Round final add to BF16.",
            "mhc_math::combine: ascending source-stream residual dot, then post product plus residual.",
            "mhc_math uses uncast FP32 coefficients; candidate mhc_bf16_combine adds BF16 terms. Framework matmul tree is absent."),
    }
}

/// Finite-only RN-even conversion. Reject overflow to BF16 infinity as well.
/// This uses the existing Vade conversion, not an inferred framework rule.
pub fn round_bf16(value: f32) -> Result<f32> {
    need(value.is_finite(),"cast input is not finite")?;
    let out=bf16_to_f32(f32_to_bf16(value));
    need(out.is_finite(),"cast result is not finite in BF16")?; Ok(out)
}
fn exact_bf16(value: f32) -> Result<()> {
    need(round_bf16(value)?.to_bits()==value.to_bits(),"cast operand is not exact BF16")
}
/// Finish ordinary RMSNorm from a supplied normalized FP32 value, source 81.
/// The reduction and inverse root are outside this pointwise operation.
pub fn norm_weight(normalized: f32, weight: f32) -> Result<f32> {
    exact_bf16(weight)?; round_bf16(round_bf16(normalized)? * weight)
}
/// Single-token BF16 expert fold, source 120-134 and ungated_moe_finish.
/// IDs must be unique. Inputs are already BF16 Linear outputs. No allocation
/// or model state is retained. Any error leaves the caller's data unchanged.
pub fn expert_fold(ids: &[u32], weights: &[f32], down: &[f32], shared: &[f32]) -> Result<Vec<f32>> {
    let h=shared.len();
    need(h>0 && !ids.is_empty() && ids.len()==weights.len()
        && ids.len().checked_mul(h)==Some(down.len()),"invalid cast expert extents")?;
    need(weights.iter().all(|v| v.is_finite()) && !ids.contains(&u32::MAX),"invalid cast expert route")?;
    for &v in down.iter().chain(shared) { exact_bf16(v)?; }
    let mut order=(0..ids.len()).collect::<Vec<_>>(); order.sort_unstable_by_key(|&j| ids[j]);
    need(order.windows(2).all(|w| ids[w[0]]!=ids[w[1]]),"duplicate cast expert ID")?;
    let mut out=vec![0.0;h];
    for slot in order { for j in 0..h {
        let contribution=round_bf16(down[slot*h+j]*weights[slot])?;
        out[j]=round_bf16(out[j]+contribution)?;
    } }
    for j in 0..h { out[j]=round_bf16(out[j]+shared[j])?; } Ok(out)
}
/// Copy a counted BF16 latent selection. Ignore padding IDs after count.
/// This defines compact scratch only. It does not allocate or mutate a cache.
pub fn gather_bf16(cache: &[f32], width: usize, selected: &[u32], count: usize, position: usize) -> Result<Vec<f32>> {
    need(width>0 && cache.len()%width==0 && position<cache.len()/width && count<=selected.len(),"invalid cast gather extents")?;
    let length=selected.len().checked_mul(width).filter(|&n| n<=isize::MAX as usize/4)
        .ok_or_else(|| invalid("cast gather extent overflow"))?;
    let ids=&selected[..count];
    need(ids.iter().all(|&id| id as usize<=position) && ids.windows(2).all(|w| w[0]<w[1]),"invalid cast gather IDs")?;
    // Do not read unused cache rows or padding indices.
    for &id in ids { for &v in &cache[id as usize*width..(id as usize+1)*width] { exact_bf16(v)?; } }
    let mut out=vec![0.0;length];
    for (slot,&id) in ids.iter().enumerate() { out[slot*width..(slot+1)*width].copy_from_slice(&cache[id as usize*width..(id as usize+1)*width]); }
    Ok(out)
}
