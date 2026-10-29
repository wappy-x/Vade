//! Routed hybrid MLP over views into the single immutable expert arena.
use super::*;
use spark_gpu_probe::driver::MoeBatchScratch;

/// The hybrid forward pass supplies resident descriptors and allocated routing and work buffers.
/// # Safety
/// Retain all allocations and the matching descriptor table through stream completion.
#[allow(dead_code)] // Called by the hybrid forward pass.
pub(super) unsafe fn run_fp8(b: &Backend<'_>, p: spark_gpu_probe::fp8_abi::RoutedParams,
    experts: &[spark_gpu_probe::fp8_abi::ExpertProjection], global_intermediate: u64, work: [u64; 6],
    routing: Option<spark_gpu_probe::kernel_params::MoeRouteBatch>) -> Result<()> {
    unsafe { b.fp8_routed_ffn(p, experts, global_intermediate, work, routing) }
}

pub(super) struct Workspace<'a> {
    routed: MoeBatchScratch<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>, shared: DeviceVector<'a>,
}
impl<'a> Workspace<'a> {
    pub(super) fn reservation_bytes(c: &Config, tokens: usize) -> usize {
        if c.experts == 0 { return 0; }
        let (e, h, m, k) = (c.experts as usize, c.hidden as usize,
            c.intermediate as usize, c.selected_experts as usize);
        let alignment = if tokens * k >= 64 * e { 128 } else { 64 };
        let padded = (tokens * k + (alignment - 1) * e).div_ceil(alignment) * alignment;
        // The backend overlaps packed input with FP32 down projections.
        let metadata = 2 * tokens * e + 3 * tokens * k + 2 * e + 1;
        let workspace = padded * h + padded * m / 2;
        (metadata + workspace + tokens * (2 * c.shared_intermediate as usize + h)) * 4 + 1024
    }
    pub(super) fn new(model: &Model<'a, '_>, tokens: usize) -> Result<Option<Self>> {
        let c = &model.config;
        if c.experts == 0 || model.bound.is_some() { return Ok(None); }
        let b = model.backend;
        let mut v = b.vector_arena(&[tokens * c.shared_intermediate as usize,
            tokens * c.shared_intermediate as usize, tokens * c.hidden as usize])?.into_iter();
        let mut routed = b.moe_batch_scratch(&model.experts[0], c.selected_experts as usize, tokens)?;
        routed.bf16_boundaries();
        Ok(Some(Self { routed, gate: v.next().unwrap(), up: v.next().unwrap(), shared: v.next().unwrap() }))
    }
    /// After a forward's output read: a router row rejected for non-finite
    /// scores fails the forward; its uncommitted sessions stay invalid.
    pub(super) fn check(&self, b: &Backend<'_>, rows: usize) -> Result<()> {
        if b.moe_batch_rejected(&self.routed, rows)?.is_empty() { Ok(()) } else { Err(MOE_ROUTING_ERROR.into()) }
    }
    /// Batched forwards: which of `chats` own a rejected row; `owner` maps each
    /// of the forward's `rows` to its chat. Only those chats may fail.
    pub(super) fn rejected(&self, b: &Backend<'_>, rows: usize, chats: usize, owner: impl Fn(usize) -> usize) -> Result<Vec<bool>> {
        let mut rejected = vec![false; chats];
        for row in b.moe_batch_rejected(&self.routed, rows)? { rejected[owner(row)] = true; }
        Ok(rejected)
    }
    pub(super) fn run(&mut self, model: &Model<'a, '_>, layer: usize, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>,
        count: usize, _decode: bool, prompt_lengths: &[usize]) -> Result<()> {
        let (b, c) = (model.backend, &model.config);
        let p = format!("{PREFIX}.layers.{layer}.mlp");
        // Every layer is routed, so layer 0 starts each forward's row status.
        b.profile_phase(28, || b.moe_batch_device(model.matrix(&format!("{p}.gate.weight"))?, &model.experts[layer],
            input, output, &mut self.routed, weight_pack, input_pack, true, true, layer == 0, None))?;
        self.gate.resize_within_capacity(count * c.shared_intermediate as usize)?;
        let _class = b.prefill_class(6);
        self.up.resize_within_capacity(count * c.shared_intermediate as usize)?;
        self.shared.resize_within_capacity(count * c.hidden as usize)?;
        // Shared projections have no phase choice; typed FP8 uses fixed_project.
        let lengths = if prompt_lengths.is_empty() { std::slice::from_ref(&count) } else { prompt_lengths };
        let (h, m) = (c.hidden as usize, c.shared_intermediate as usize);
        let mut offset = 0;
        for &tokens in lengths {
            let input = input.slice(offset * h..(offset + tokens) * h)?;
            for (name, out, reuse) in [("gate_proj", &mut self.gate, false), ("up_proj", &mut self.up, true)] {
                let matrix = model.matrix(&format!("{p}.shared_expert.{name}.weight"))?;
                let mut out = out.slice(offset * m..(offset + tokens) * m)?;
                Session::project_reusing(b, matrix, &input, &mut out, weight_pack, input_pack, tokens, reuse)?;
            }
            offset += tokens;
        }
        launch_silu_precision(b, &self.gate, &self.up, 3)?;
        let down = model.matrix(&format!("{p}.shared_expert.down_proj.weight"))?;
        let mut offset = 0;
        for &tokens in lengths {
            let gate = self.gate.slice(offset * m..(offset + tokens) * m)?;
            let mut shared = self.shared.slice(offset * h..(offset + tokens) * h)?;
            Session::project(b, down, &gate, &mut shared, weight_pack, input_pack, tokens)?;
            offset += tokens;
        }
        let mut params = spark_gpu_probe::kernel_params::HybridSharedExpert {
            input: input.device_address(), gate: model.address(&format!("{p}.shared_expert_gate.weight"))?,
            shared: self.shared.device_address(), output: output.device_address(), hidden: c.hidden as u32,
            tokens: count as u32,
        };
        unsafe { b.launch_flash_shared("flash_moe_shared", count as u32, 128, 0,
            &mut [(&mut params as *mut spark_gpu_probe::kernel_params::HybridSharedExpert).cast()]) }
    }
}

/// Combine the FP32 routed expert rows with the reduced shared expert row.
/// # Safety
/// Retain valid disjoint input/output spans in this backend; IDs are unique, weights finite,
/// partials ID-ordered and rank-folded. Residual=0 lets the next norm add it once.
#[allow(dead_code)] // Called by the hybrid forward pass.
pub(super) unsafe fn finish_fp8(b: &Backend<'_>, p: spark_gpu_probe::fp8_abi::OrderedFinishParams) -> Result<()> {
    use spark_gpu_probe::fp8_abi::OrderedFinishParams;
    let shared=p.shared;
    let mut p = OrderedFinishParams::checked(p.finish, p.layout, p.tokens, p.residual)
        .ok_or("invalid ordered expert finish descriptor")?;
    if shared!=0 && (shared%4!=0 || shared.checked_add(p.tokens as u64*p.layout.hidden as u64*4).is_none()) { return Err("invalid shared expert span".into()); }
    p.shared=shared;
    unsafe { b.launch_flash_rows("ordered_moe_finish", p.finish.columns.div_ceil(256), p.tokens, 256, 0,
        &mut [(&mut p as *mut OrderedFinishParams).cast()]) }
}
