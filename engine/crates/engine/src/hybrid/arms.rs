//! Typed-path capabilities. Common embedding/norm/projection kernels remain shared;
//! an inactive arm owns no reads, resolves no functions and evaluates no reserve.
use engine_model::{flash::LayerKind, plan::{Format, ModelPlan, Rope}, qwen3_5::Config, shapes::*};
type Result<T> = std::result::Result<T, String>;
/// Shared trunk dependencies: arm projections/norms reuse these exact entries.
pub const SHARED_KERNELS: &[&str] = &["batch_embedding","flash_norm_decode","flash_residual_norm_bf16","flash_residual_bf16","logits_bf16","logits_argmax","logits_argmax_finish","flash_bf16_pack","flash_hybrid_copy","flash_bf16_mma1","fp8_quantize_128","gemm_fp8_block128","gemm_fp8_block128_decode"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind { Gdn, Attention, MoeFp8, Mtp, Transport }
#[derive(Debug, Clone, Copy)]
pub struct Arm {
    pub kind: Kind, pub name: &'static str, pub on: bool,
    launches: &'static [&'static str],
    /// Existing allocation owners; these declarations do not add a second charge.
    pub memory: &'static [&'static str],
    /// Additional arm dependencies; all compute arms also use SHARED_KERNELS.
    pub requires: &'static [Kind],
}
impl Arm {
    pub fn kernels(&self) -> &'static [&'static str] { if self.on { self.launches } else { &[] } }
    pub fn reads(&self, t: &TensorEntry) -> bool {
        if !self.on { return false; }
        match (self.kind, t.role.place, t.role.part) {
            (Kind::Mtp, Place::Draft { .. }, _) => true,
            (_, Place::Draft { .. } | Place::Unused, _) => false,
            (Kind::Gdn, _, Part::LinearQkv | Part::LinearConv | Part::LinearZ | Part::LinearOut | Part::LinearHeads | Part::LinearNorm) => true,
            (Kind::Attention, _, Part::Query | Part::Key | Part::Value | Part::Output | Part::QueryNorm | Part::KeyNorm) => true,
            (Kind::MoeFp8, _, Part::Expert { .. } | Part::Shared(_)) => true,
            _ => false,
        }
    }
    pub fn weights<'a>(&'a self, p: &'a ModelPlan) -> impl Iterator<Item=&'a TensorEntry> { p.shape.tensors.iter().filter(|t| self.reads(t)) }
    pub fn weight_bytes(&self, p: &ModelPlan) -> Option<usize> {
        self.weights(p).try_fold(0usize, |sum,t| { let bytes=t.shape.iter().try_fold(t.dtype.bytes(),|n,d| n.checked_mul(*d))?; sum.checked_add(usize::try_from(bytes).ok()?) })
    }
    /// The arm's config fields are checked before any weight read or kernel setup.
    pub fn limits(&self, p: &ModelPlan, world: usize) -> Result<()> {
        if !self.on { return Ok(()); }
        for l in &p.shape.layers {
            match (self.kind,l.mixer,l.ffn) {
                (Kind::Gdn,Mixer::Linear(s),_) => Limits::gdn(s)?,
                (Kind::Attention,Mixer::Attention(s),_) => Limits::attention(s,&p.rope)?,
                (Kind::MoeFp8,_,Ffn::Experts(e)) => Limits::moe(e)?, _ => {}
            }
        }
        if self.kind==Kind::Transport { field("transport","world",world as u64,(MIN_SPARKS as usize..=MAX_SPARKS as usize).contains(&world),"SplitPlan rank capacity")?; }
        if self.kind==Kind::Mtp {
            field("mtp","mtp_num_hidden_layers",p.shape.draft.len() as u64,p.shape.draft.len()==Limits::DRAFT_LAYERS,"one draft block")?;
            for l in &p.shape.draft { if let Mixer::Attention(s)=l.mixer { Limits::attention(s,&p.rope)?; } }
        }
        Ok(())
    }
    /// Lazy funding is essential: even an overflowing/expensive off-arm formula is unused.
    pub fn reserve(&self, bytes: impl FnOnce() -> Option<usize>) -> Option<usize> {
        if self.on { bytes() } else { Some(0) }
    }
}
#[derive(Debug, Clone)]
pub struct Table { pub arms: [Arm; 5], world: usize }
impl Table {
    pub fn new(p: &ModelPlan, world: usize, mtp: bool) -> Self {
        let layers = || p.shape.layers.iter().chain(p.shape.draft.iter().filter(|_| mtp));
        Self::enabled([
            layers().any(|l| matches!(l.mixer,Mixer::Linear(_))),
            layers().any(|l| matches!(l.mixer,Mixer::Attention(_))),
            matches!(p.format,Format::Fp8 { .. }) && layers().any(|l| matches!(l.ffn,Ffn::Experts(_))),
            mtp && !p.shape.draft.is_empty(), world > Limits::LOCAL_WORLD,
        ],world)
    }
    /// Reservation callers already hold this geometry view of ModelPlan; never reparse config.
    pub fn for_config(c: &Config, world: usize, mtp: bool) -> Self {
        Self::enabled([c.layers.contains(&LayerKind::DeltaNet),c.layers.contains(&LayerKind::SparseAttention) || mtp && c.mtp_layers>0,c.fp8 && c.experts>0,mtp && c.mtp_layers>0,world>Limits::LOCAL_WORLD],world)
    }
    fn enabled(on: [bool;5], world: usize) -> Self {
        Self { world, arms: [
            Arm { kind:Kind::Gdn, name:"gdn", on:on[0],
                launches:&["flash_gdn_conv_prepare","flash_gdn_conv_prepare_rows","flash_gdn_conv_prepare_hist","flash_gdn_step_batch","flash_gdn_step_gate_batch","flash_gdn_history","flash_gdn_history_rows","flash_gdn_gate"], memory:&["GdnScratch","TokenStep","GdnState/conv history"], requires:&[] },
            Arm { kind:Kind::Attention, name:"attention", on:on[1],
                launches:&["flash_qsa_prepare","flash_dense_prepare_batch","flash_dense_attention_batch","flash_dense_scores","flash_dense_values"], memory:&["DenseScratch","DenseState BF16 KV","FP32 scores"], requires:&[] },
            Arm { kind:Kind::MoeFp8, name:"moe-fp8", on:on[2],
                launches:&["moe_route_rows","fp8_grouped_prefix","fp8_quantize_128","flash_silu","fp8_routed_clear","fp8_routed_count","fp8_routed_order","fp8_routed_gather","fp8_routed_scatter","fp8_routed_gate","fp8_routed_up","fp8_routed_down","ordered_moe_finish"], memory:&["forward::Moe vectors","expert descriptor tables","shared-expert workspace"], requires:&[] },
            Arm { kind:Kind::Mtp, name:"mtp", on:on[3],
                launches:&[], memory:&["mtp::fixed_reservation_bytes","draft KV","rollback/verifier state"], requires:&[Kind::Attention,Kind::MoeFp8] },
            Arm { kind:Kind::Transport, name:"transport", on:on[4],
                launches:&[], memory:&["tp::Binding rank-ordered host exchange buffers"], requires:&[] },
        ] }
    }
    pub fn arm(&self, kind: Kind) -> &Arm { self.arms.iter().find(|a| a.kind == kind).unwrap() }
    /// Norms/router and embedding/head belong to the shared trunk, not an optional arm.
    pub fn reads(&self, t: &TensorEntry) -> bool {
        matches!((t.role.place,t.role.part), (Place::Model,_) | (Place::Layer(_),Part::Whole))
            || self.arms.iter().any(|a| a.reads(t))
    }
    /// Host reduction retains [rows,selected+1,H] plus ascending-rank scratch.
    pub fn transport_bytes(&self, p: &ModelPlan, rows: usize) -> Option<usize> {
        self.arm(Kind::Transport).reserve(|| {
            let slots=p.shape.layers.iter().chain(p.shape.draft.iter().filter(|_| self.arm(Kind::Mtp).on)).try_fold(1u64,|n,l| Some(n.max(match l.ffn { Ffn::Experts(e)=>e.selected.checked_add(1)?,_=>1 })))?;
            rows.checked_mul(usize::try_from(p.shape.hidden.checked_mul(slots)?.max(p.shape.vocab)).ok()?)?.checked_mul(4)?.checked_mul(self.world.checked_add(1)?)
        })
    }
    pub fn setup(&self, mut require: impl FnMut(&Arm) -> Result<()>) -> Result<()> {
        for arm in self.arms.iter().filter(|a| a.on) { require(arm)?; }
        Ok(())
    }
    pub fn validate(&self, p: &ModelPlan) -> Result<()> {
        for a in self.arms.iter().filter(|a| a.on) { a.limits(p,self.world)?; }
        Ok(())
    }

}
fn field(arm: &str, name: &str, value: u64, valid: bool, supported: &str) -> Result<()> {
    if valid { Ok(()) } else { Err(format!("{arm} arm: {name}={value}; requires {supported}")) }
}
/// Kernel ABI capacities, not model identities. Actual dimensions always come from config.
pub struct Limits;
impl Limits {
    pub const GDN_DIM: u64 = 128;
    pub const CONV: u64 = 4;
    pub const HEAD_DIM: u64 = 256;
    pub const ROTARY: u64 = 64;
    pub const NORM_WIDTH: u64 = 5120;
    pub const FUSED_HEADS: u64 = 48;
    pub const LOCAL_WORLD: usize = 1;
    pub const DRAFT_LAYERS: usize = 1;
    pub fn gdn(s: LinearShape) -> Result<()> {
        for (name,value,want) in [("linear_key_head_dim",s.key_dim,Self::GDN_DIM),("linear_value_head_dim",s.value_dim,Self::GDN_DIM),("linear_conv_kernel_dim",s.conv,Self::CONV)] {
            field("gdn",name,value,value==want,&want.to_string())?;
        }
        let repeat=s.value_heads.checked_div(s.key_heads).unwrap_or(0);
        field("gdn","linear_num_value_heads",s.value_heads,s.key_heads>0 && s.value_heads.is_multiple_of(s.key_heads) && (1..=4).contains(&repeat),"1..=4 whole value groups per key head")
    }
    pub fn attention(s: AttentionShape, rope: &Rope) -> Result<()> {
        for (name,value,want) in [("head_dim",s.head_dim,Self::HEAD_DIM),("rotary_dim",s.rotary_dim,Self::ROTARY),("rope.rotary_dim",rope.rotary_dim,Self::ROTARY)] {
            field("attention",name,value,value==want,&want.to_string())?;
        }
        Ok(())
    }
    pub fn shared(c: &Config) -> Result<()> {
        Self::gdn(LinearShape { key_heads:c.linear_key_heads,value_heads:c.linear_value_heads,key_dim:c.linear_key_dim,value_dim:c.linear_value_dim,conv:c.linear_conv })?;
        for (name,value,want) in [("head_dim",c.head_dim,Self::HEAD_DIM),("rotary_dim",c.rotary_dim,Self::ROTARY)] { field("attention",name,value,value==want,&want.to_string())?; }
        Ok(())
    }
    pub fn moe(e: ExpertShape) -> Result<()> {
        let top=u32::try_from(e.selected).ok(); let width=u32::try_from(e.width).ok();
        field("moe-fp8","num_experts_per_tok",e.selected,top.is_some_and(|n| spark_gpu_probe::fp8_abi::RoutingLayout::checked(n,FP8_BLOCK as u32).is_some()),"routed kernel top-k capacity")?;
        field("moe-fp8","moe_intermediate_size",e.width,width.is_some_and(|n| spark_gpu_probe::fp8_abi::RoutingLayout::checked(top.unwrap(),n).is_some()),"positive block-aligned routed width")
    }
    pub fn fixed(c: &Config) -> Result<()> { Self::shared(c)?; Self::norm(c.hidden) }
    pub fn norm(width: u64) -> Result<()> { field("shared","hidden_size",width,width>0 && Self::norm_supported(width),&format!("1..={}",Self::NORM_WIDTH)) }
    pub fn norm_supported(width: u64) -> bool { width <= Self::NORM_WIDTH }
    pub fn fused_gates(heads: u64) -> bool { heads <= Self::FUSED_HEADS }
    pub fn world(world: usize) -> Result<()> { field("transport","world",world as u64,world==Self::LOCAL_WORLD,"world=1 (typed GPU execution)") }
    /// Preserve the existing automatic-draft policy; explicit MTP is geometry-checked.
    pub fn auto_mtp(c: &Config) -> bool {
        const AUTO_LAYERS: usize = 64;
        !c.fp8 && c.hidden==Self::NORM_WIDTH && c.layers.len()==AUTO_LAYERS && c.mtp_layers==Self::DRAFT_LAYERS as u64 && !c.tied_embeddings
    }
}
