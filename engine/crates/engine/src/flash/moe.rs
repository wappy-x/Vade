//! Tensor-parallel routed/shared MLP and its four-stream HC injection.
//! FP8 W8A8 experts, or opt-in packed NVFP4 weights with BF16 operands.
//! Every expert stays rank-sharded; rank reduction precedes the weighted sum.
use super::{Weights, blocks::{invariant_pairs, invariant_rows, Hc, HcScratch, H, W},
    gdn::{bf16, launch, Arg::{Ptr, U32}}};
use engine_model::{flash::{plan::Selection, WeightFormat}, safetensors::Dtype};
use spark_gpu_probe::driver::{Backend, DeviceVector, Fp8GroupedScratch};
use std::marker::PhantomData;

type Result<T> = std::result::Result<T, String>;
const EXPERTS: usize = 512;
const TOP: usize = 10;
const CHANNELS: usize = 640;
pub(super) const PARTIALS: usize = (TOP + 1) * H;

#[derive(Clone, Copy)]
struct Fp8Matrix { weight: u64, scales: u64, rows: usize }
impl Fp8Matrix {
    #[allow(clippy::single_range_in_vec_init)] // a one-range Vec is intended.
    fn load(w: &Weights<'_>, prefix: &str, down: bool, channels: std::ops::Range<u64>) -> Result<Self> {
        let (shape, scale_shape) = if down { ([H as u64, 640], [20, 5]) }
            else { ([640, H as u64], [5, 20]) };
        let width = (channels.end - channels.start) as usize * 128;
        let output = w.plan.expert_output_rows.as_ref()
            .filter(|_| down && prefix.starts_with("model.language_model.layers."));
        let selection = if let Some(rows) = output { Selection::Rows(vec![rows.clone()]) }
            else if down { Selection::Columns(channels.start * 128..channels.end * 128) }
            else { Selection::Rows(vec![channels.start * 128..channels.end * 128]) };
        let scale_selection = if let Some(rows) = output { Selection::Rows(vec![rows.start / 128..rows.end / 128]) }
            else if down { Selection::Columns(channels.clone()) }
            else { Selection::Rows(vec![channels.clone()]) };
        let elements = output.map_or(H * width, |rows| (rows.end - rows.start) as usize * CHANNELS);
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale_inv");
        if elements == 0 { return Ok(Self { weight: 0, scales: 0, rows: if down { H } else { 0 } }); }
        let weight = w.tensor(&weight_name)?;
        let scale = w.tensor(&scale_name)?;
        if weight.tensor.dtype != Dtype::F8E4M3 || weight.tensor.shape != shape
            || scale.tensor.dtype != Dtype::Bf16 || scale.tensor.shape != scale_shape
            || weight.selection != selection || scale.selection != scale_selection
            || weight.bytes != elements || scale.bytes != 2 * elements / (128 * 128) {
            return Err(format!("{prefix}: FP8 projection and BF16 scale grid do not share the rank's block boundaries"));
        }
        let address = w.device_address(&weight_name)?;
        // W8A8 tiles copy weights with 16-byte cp.async.
        if address % 16 != 0 { return Err(format!("{prefix}: FP8 weights must start on 16 bytes")); }
        Ok(Self { weight: address, scales: w.device_address(&scale_name)?,
            rows: if down { H } else { width } })
    }
}

/// IDs/weights retain the router's descending-score order. Each output row is
/// independent until all-reduce: ten experts in ascending ID, then shared.
/// Transport must reduce values before passing this packet back to finish.
pub struct MoePartials {
    pub ids: [u32; TOP],
    pub weights: [f32; TOP],
    pub values: Vec<f32>,
}
impl MoePartials {
    pub const VALUE_BYTES: usize = PARTIALS * 4;
    /// FP32 values per token of the device partial rows.
    pub const VALUES: usize = PARTIALS;
    pub fn same_routing(&self, other: &Self) -> bool {
        self.ids == other.ids && self.weights.iter().zip(other.weights).all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

struct Pending { ids: [u32; TOP], weights: [f32; TOP], residual: u64 }

/// Reusable tile workspace bounds scratch independently of total prompt length.
/// Group selected experts once per tile; no KV state.
pub struct MoeScratch<'a> {
    hc: HcScratch<'a>,
    logits: DeviceVector<'a>,
    ids: DeviceVector<'a>,
    weights: DeviceVector<'a>,
    gate: DeviceVector<'a>,
    up: DeviceVector<'a>,
    shared_gate: DeviceVector<'a>,
    partials: DeviceVector<'a>,
    output: DeviceVector<'a>,
    input: DeviceVector<'a>,
    assignments: DeviceVector<'a>,
    descriptors: DeviceVector<'a>,
    descriptor_extra_bytes: usize,
    grouped: Option<Fp8GroupedScratch<'a>>,
    pending: Option<Pending>,
    bytes: usize,
}
impl<'a> MoeScratch<'a> {
    pub fn new(b: &Backend<'a>) -> Result<Self> { Self::with_capacity(b, 1) }
    pub(super) fn with_capacity(b: &Backend<'a>, capacity: usize) -> Result<Self> {
        let lengths = [EXPERTS * capacity, TOP * capacity, TOP * capacity,
            CHANNELS * capacity, CHANNELS * capacity, capacity, PARTIALS * capacity, H * capacity,
            H * capacity, TOP * 3 * capacity, EXPERTS * 14];
        let hc = HcScratch::with_tokens(b, capacity)?;
        let bytes = hc.device_bytes() + lengths.iter().map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>();
        let mut vectors = b.vector_arena(&lengths)?.into_iter();
        Ok(Self { hc, logits: vectors.next().unwrap(), ids: vectors.next().unwrap(),
            weights: vectors.next().unwrap(), gate: vectors.next().unwrap(), up: vectors.next().unwrap(),
            shared_gate: vectors.next().unwrap(), partials: vectors.next().unwrap(), output: vectors.next().unwrap(),
            input: vectors.next().unwrap(), assignments: vectors.next().unwrap(), descriptors: vectors.next().unwrap(),
            descriptor_extra_bytes: 0, grouped: None, pending: None, bytes })
    }
    pub(super) fn reserve_grouped(&mut self, b: &Backend<'a>, tokens: usize, activation_quantization: bool, compact: Option<usize>) -> Result<()> {
        b.reserve_fp8_grouped(&mut self.grouped, tokens, activation_quantization)?;
        if let Some(world) = compact { b.reserve_fp8_middle(self.grouped.as_mut().unwrap(), world)?; }
        Ok(())
    }
    pub(super) fn set_routing_flag(&mut self, flag: &spark_gpu_probe::driver::Fp8RoutingFlag<'a>) -> Result<()> {
        self.grouped.as_mut().ok_or("missing grouped routing")?.set_routing_flag(flag)
    }

    pub fn device_bytes(&self) -> usize { self.bytes + self.grouped.as_ref().map_or(0, |s| s.bytes()) }
    pub(super) fn prepared_tokens(&self) -> usize { self.grouped.as_ref().map_or(0, Fp8GroupedScratch::token_capacity) }
    /// A pair pass of up to four rows can swap its middle through mapped packets.
    pub(super) fn small_middle(&self, tokens: usize) -> bool {
        tokens <= 4 && (self.prepared_tokens() <= 4 || self.grouped.as_ref().is_some_and(Fp8GroupedScratch::small_middle_mapped))
    }
    pub(super) fn resize(&mut self, tokens: usize) -> Result<()> {
        self.logits.resize_within_capacity(tokens * EXPERTS)?;
        self.ids.resize_within_capacity(tokens * TOP)?;
        self.weights.resize_within_capacity(tokens * TOP)?;
        self.shared_gate.resize_within_capacity(tokens)?;
        self.partials.resize_within_capacity(tokens * PARTIALS)?;
        self.input.resize_within_capacity(tokens * H)?;
        self.assignments.resize_within_capacity(tokens * TOP * 3)
    }
    pub(super) fn routing_failed(&self, b: &Backend<'_>) -> Result<bool> {
        self.grouped.as_ref().ok_or("missing grouped routing")?.routing_failed(b)
    }
    pub fn output(&self) -> &DeviceVector<'a> { &self.output }
    pub fn mixed(&self) -> &DeviceVector<'a> { self.hc.mixed() }
    pub(super) fn partials(&self) -> &DeviceVector<'a> { &self.partials }
    pub(super) fn partials_mut(&mut self) -> &mut DeviceVector<'a> { &mut self.partials }
    pub(super) fn middle_packets(&self) -> Result<(DeviceVector<'a>, DeviceVector<'a>)> {
        self.grouped.as_ref().ok_or("missing grouped middle")?.middle_packets()
    }
    pub(super) fn middle_world(&self) -> Result<usize> {
        Ok(self.grouped.as_ref().ok_or("missing grouped middle")?.middle_world())
    }
    /// Drain the mapped producer once; transport callbacks only copy opaque
    /// bytes and never enqueue GPU consumers of the registered receive slots.
    pub(super) fn exchange_small_middle<E: engine_fabric::exchange::Exchange>(&self,
        b: &Backend<'_>, rank: &mut E, own: usize, tokens: usize, finished: bool) -> Result<()> {
        let grouped = self.grouped.as_ref().ok_or("missing grouped middle")?;
        let world = grouped.middle_world();
        if world == 2 {
            return unsafe { grouped.with_middle_host(b, finished, |send, receive| {
                rank.exchange_slots_pipeline(&[(send.len(), receive.len())],
                    |_, out| { out.copy_from_slice(send); Ok(()) },
                    |_, peer| { receive.copy_from_slice(peer); Ok(()) }, |()| Ok(()))
            }) };
        }
        if own >= world { return Err("compact gather rank".into()); }
        let widths = (0..world).map(|r| if finished {
            spark_gpu_probe::driver::fp8_abi::middle_columns(r, world).len()
        } else { spark_gpu_probe::driver::fp8_abi::middle_packet_words(world) }).collect::<Vec<_>>();
        unsafe { grouped.with_middle_rank_host(b, |send, receive, span| {
            let bytes = tokens * widths[own] * 4;
            receive[own * span..own * span + bytes].copy_from_slice(&send[..bytes]);
            let mut offsets = vec![0];
            for width in &widths { offsets.push(offsets.last().unwrap() + tokens * width * 4); }
            rank.gather_slots_pipeline(&[offsets],
                |_, at, slot| { slot.copy_from_slice(&send[at..at + slot.len()]); Ok(()) },
                |_, r, at, peer| { receive[r * span + at..r * span + at + peer.len()].copy_from_slice(peer); Ok(()) },
                |()| Ok(()))
        }) }
    }
}

pub struct Moe<'w> {
    hc: Hc<'w>,
    descriptors: Vec<u8>,
    router: u64,
    shared: [u64; 3],
    shared_width: usize,
    shared_gate: u64,
    expert_bytes: usize,
    weight_bytes: usize,
    activation_quantization: bool,
    nvfp4: bool,
    output_rank: Option<usize>,
    _weights: PhantomData<&'w ()>,
}
impl<'w> Moe<'w> {
    /// Legacy operands are diagnostic only; invalid settings never silently change math.
    pub fn activation_quantization() -> Result<bool> {
        match std::env::var("FLASH_BF16_EXPERTS") {
            Err(std::env::VarError::NotPresent) => Ok(true),
            Ok(value) if value == "0" => Ok(true),
            Ok(value) if value == "1" => Ok(false),
            _ => Err("FLASH_BF16_EXPERTS must be 0 or 1".into()),
        }
    }
    pub fn precision_name(activation_quantization: bool) -> &'static str {
        if activation_quantization { "w8a8_e4m3_k128_fp32_accumulation_bf16_expert_boundaries" }
            else { "bf16_expert_operands_fp32_accumulation" }
    }
    /// The expert precision setting is read once by the caller.
    pub fn load(w: &'w Weights<'_>, layer: usize, activation_quantization: bool) -> Result<Self> {
        if layer >= 48 { return Err("MLP layer exceeds Flash configuration".into()); }
        Self::load_prefix(w, &format!("model.language_model.layers.{layer}.mlp"), activation_quantization)
    }
    pub(super) fn load_prefix(w: &'w Weights<'_>, p: &str, activation_quantization: bool) -> Result<Self> {
        if p.starts_with("model.language_model.layers.") { super::require_inference(w.format)?; }
        let nvfp4 = w.format == WeightFormat::Nvfp4 && p.starts_with("model.language_model.layers.");
        let output_rank = w.plan.expert_output_rows.as_ref().filter(|_| p.starts_with("model.language_model.layers."))
            .map(|rows| -> Result<usize> {
                let columns = spark_gpu_probe::driver::fp8_abi::middle_columns(w.plan.rank as usize, w.plan.ranks as usize);
                if nvfp4 || !activation_quantization || w.plan.ranks > 65535
                    || *rows != (columns.start as u64..columns.end as u64) {
                    return Err("invalid output-owned Flash expert plan".into());
                }
                Ok(w.plan.rank as usize)
            }).transpose()?;
        let mut descriptors = Vec::with_capacity(EXPERTS * if nvfp4 { 80 } else { 56 });
        for id in 0..EXPERTS {
            let blocks = w.plan.expert_blocks(id as u64);
            let p = format!("{p}.experts.{id}");
            if nvfp4 {
                if blocks.is_empty() {
                    descriptors.resize(descriptors.len() + 80, 0);
                    continue;
                }
                let width = (blocks.end - blocks.start) * 128;
                if width > CHANNELS as u64 || !width.is_multiple_of(16) {
                    return Err(format!("{p}: invalid NVFP4 expert width {width}"));
                }
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    let matrix = super::nvfp4::Matrix::load(w, &format!("{p}.{projection}"))?.params;
                    let (rows, cols) = if projection == "down_proj" { (H as u32, width as u32) }
                        else { (width as u32, H as u32) };
                    if (matrix.rows, matrix.cols) != (rows, cols) {
                        return Err(format!("{p}: NVFP4 projection dimensions disagree with expert width"));
                    }
                    for value in [matrix.weight, matrix.block_scales, matrix.global_scale] {
                        descriptors.extend_from_slice(&value.to_ne_bytes());
                    }
                }
                descriptors.extend_from_slice(&(width as u32).to_ne_bytes());
                descriptors.extend_from_slice(&0u32.to_ne_bytes());
                continue;
            }
            let gate = Fp8Matrix::load(w, &format!("{p}.gate_proj"), false, blocks.clone())?;
            let up = Fp8Matrix::load(w, &format!("{p}.up_proj"), false, blocks.clone())?;
            let down = Fp8Matrix::load(w, &format!("{p}.down_proj"), true, blocks)?;
            for value in [gate.weight, gate.scales, up.weight, up.scales,
                down.weight, down.scales, gate.rows as u64] {
                descriptors.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let channels = &w.plan.shared_channels;
        if channels.start > channels.end || channels.end > CHANNELS as u64 || !channels.start.is_multiple_of(8) || !channels.end.is_multiple_of(8) {
            return Err("shared expert channels must use aligned eight-element ranges within 640".into());
        }
        let width = (channels.end - channels.start) as usize;
        let get = |suffix: &str, shape: &[u64], elements| bf16(w, &format!("{p}.{suffix}"), shape, elements);
        let expert_prefix = format!("{p}.experts.");
        Ok(Self { hc: Hc::load(w, &format!("{p}_hyper_connection"))?, descriptors,
            router: get("gate.weight", &[512, 2560], EXPERTS * H)?,
            shared: [get("shared_expert.gate_proj.weight", &[640, 2560], width * H)?,
                get("shared_expert.up_proj.weight", &[640, 2560], width * H)?,
                get("shared_expert.down_proj.weight", &[2560, 640], H * width)?],
            shared_width: width, shared_gate: get("shared_expert_gate.weight", &[1, 2560], H)?,
            expert_bytes: w.prefix_bytes(&expert_prefix), weight_bytes: w.prefix_bytes(p),
            activation_quantization, nvfp4, output_rank, _weights: PhantomData })
    }
    /// The 512 expert descriptors on the device, for a resident layer: uploaded
    /// once instead of before every routed step.
    pub(super) fn device_descriptors<'a>(&self, b: &Backend<'a>) -> Result<DeviceVector<'a>> {
        let mut descriptors = b.vector(self.descriptors.len() / 4)?;
        b.write_vector_bytes(&mut descriptors, &self.descriptors)?;
        Ok(descriptors)
    }
    /// Stored packed expert matrices and their scale grids, without padding.
    pub fn expert_bytes(&self) -> usize { self.expert_bytes }
    /// All MLP, router, shared and HC weights; no activation or host mirror.
    pub fn weight_bytes(&self) -> usize { self.weight_bytes }

    /// Mix the original four residual streams, route globally, and compute
    /// every selected local down partial. Readback synchronizes before transport.
    /// Each rank independently computes the same full512/top10 routing row.
    pub fn project<'a>(&self, b: &Backend<'a>, s: &mut MoeScratch<'a>, residual: &DeviceVector<'_>) -> Result<MoePartials> {
        let mut packet = self.project_device(b, s, residual)?;
        packet.values = b.read_vector(&s.partials)?;
        Ok(packet)
    }

    /// The host-packet API is retained for the two-local-rank block runner.
    pub(super) fn project_device<'a>(&self, b: &Backend<'a>, s: &mut MoeScratch<'a>, residual: &DeviceVector<'_>) -> Result<MoePartials> {
        if self.output_rank.is_some() { return Err("output-owned experts require the distributed middle exchange".into()); }
        if s.pending.is_some() || residual.len() != W { return Err("invalid MLP pending state".into()); }
        s.resize(1)?;
        self.hc.mix(b, &mut s.hc, residual)?;
        b.copy_prefix(s.hc.mixed(), &mut s.input, H)?;
        self.project_inputs(b, s, 1, None)?;
        if s.routing_failed(b)? { return Err("Flash MoE routing rejected non-finite or out-of-range selection".into()); }
        let ids = b.read_vector(&s.ids)?;
        let weights = b.read_vector(&s.weights)?;
        let ids = std::array::from_fn(|i| ids[i].to_bits());
        let weights = std::array::from_fn(|i| weights[i]);
        s.pending = Some(Pending { ids, weights, residual: residual.device_address() });
        Ok(MoePartials { ids, weights, values: Vec::new() })
    }

    /// Mix a complete tile before grouping; no per-token device readbacks.
    /// `resident` holds this layer's device descriptors (device_descriptors).
    pub(super) fn project_batch<'a>(&self, b: &Backend<'a>, s: &mut MoeScratch<'a>,
        hidden: &DeviceVector<'_>, tokens: usize, resident: Option<&DeviceVector<'_>>) -> Result<()> {
        s.resize(tokens)?;
        self.hc.mix_tiled(b, &mut s.hc, hidden, &s.input, tokens)?;
        self.project_inputs(b, s, tokens, resident)
    }

    fn project_inputs<'a>(&self, b: &Backend<'a>, s: &mut MoeScratch<'a>, tokens: usize,
        resident: Option<&DeviceVector<'_>>) -> Result<()> {
        if resident.is_none() {
            let words = self.descriptors.len() / 4;
            if s.descriptors.len() != words {
                let descriptors = b.vector(words)?;
                // The initial descriptor view still belongs to the shared arena.
                s.bytes = s.bytes - s.descriptor_extra_bytes + words * 4;
                s.descriptor_extra_bytes = words * 4;
                s.descriptors = descriptors;
            }
            b.write_vector_bytes(&mut s.descriptors, &self.descriptors)?;
        }
        let descriptors = resident.unwrap_or(&s.descriptors);
        let input = s.input.device_address();
        // Router logits, the shared-expert gate and the shared gate/up rows: one
        // row kernel at every decode, batch and verification width.
        invariant_pairs(b, input, tokens, [
            (self.router, s.logits.device_address(), EXPERTS),
            (self.shared_gate, s.shared_gate.device_address(), 1),
            (self.shared[0], s.gate.device_address(), self.shared_width),
            (self.shared[1], s.up.device_address(), self.shared_width),
        ])?;
        {
        let _class = b.prefill_class(3);
        b.prefill_read_bytes(3, || tokens * (EXPERTS + TOP * 2) * 4);
        let mut route = spark_gpu_probe::kernel_params::FlashMoeRoute {
            logits: s.logits.device_address(), ids: s.ids.device_address(), weights: s.weights.device_address(),
            tokens: tokens as u32, norm_topk: 1,
        };
        let entry = if b.has_flash("flash_moe_route_fast") { "flash_moe_route_fast" } else { "flash_moe_route" };
        unsafe { b.launch_flash(entry, tokens as u32, 128,
            &mut [(&mut route as *mut spark_gpu_probe::kernel_params::FlashMoeRoute).cast()])?; }
        launch(b, "flash_moe_assignments", tokens * TOP, &mut [Ptr(s.ids.device_address()),
            Ptr(s.weights.device_address()), Ptr(s.assignments.device_address()), U32(tokens as u32)])?;
        }
        unsafe {
            if self.nvfp4 {
                b.fp4_bf16_grouped_ffn(descriptors, &s.input, &s.assignments, &mut s.partials, &mut s.grouped)?;
            } else {
                b.fp8_grouped_ffn(descriptors, &s.input, &s.assignments, &mut s.partials, &mut s.grouped,
                    self.activation_quantization, self.output_rank.is_some())?;
            }
        }
        {
        let _class = b.prefill_class(6);
        let silu = if b.has_flash("flash_moe_silu_fast") { "flash_moe_silu_fast" } else { "flash_moe_silu" };
        launch(b, silu, self.shared_width * tokens, &mut [Ptr(s.gate.device_address()),
            Ptr(s.up.device_address()), U32((self.shared_width * tokens) as u32)])?;
        // Shared rows are strided among the ten expert partials.
        invariant_rows(b, self.shared[2], s.gate.device_address(), s.partials.device_address() + (TOP * H * 4) as u64,
            H, self.shared_width, tokens, PARTIALS)?;
        }
        if let Some(rank) = self.output_rank {
            b.pack_fp8_middle(s.grouped.as_ref().ok_or("missing grouped middle")?, rank)?;
        }
        Ok(())
    }

    pub(super) fn prepare_exchange(&self, s: &mut MoeScratch<'_>, tokens: usize,
        resident: Option<&DeviceVector<'_>>) -> Result<()> {
        let descriptors = resident.unwrap_or(&s.descriptors).device_address();
        unsafe { s.grouped.as_mut().ok_or("missing grouped middle")?
            .prepare_middle(descriptors, s.partials.device_address(), tokens) }
    }

    /// The output owner reproduces all original K-shard subtotals, then uses
    /// exactly the old per-expert rounding, ascending-ID sum and shared-last add.
    pub(super) fn complete_middle(&self, b: &Backend<'_>, s: &MoeScratch<'_>, tokens: usize) -> Result<()> {
        let rank = self.output_rank.ok_or("missing expert output ownership")?;
        b.finish_fp8_middle(s.grouped.as_ref().ok_or("missing grouped middle")?, rank)?;
        let (packet, _) = s.middle_packets()?;
        let columns = spark_gpu_probe::driver::fp8_abi::middle_columns(rank, s.middle_world()?);
        self.finish_columns(b, s, 0, tokens, columns.start, columns.len(), packet.device_address())
    }

    /// Finish and inject a whole tile; the HC still holds the tile's normalized streams.
    pub(super) fn finish_batch(&self, b: &Backend<'_>, s: &mut MoeScratch<'_>,
        hidden: &mut DeviceVector<'_>, tokens: usize) -> Result<()> {
        if let Some(rank) = self.output_rank {
            if s.middle_world()? == 2 {
                let (own, peer) = s.middle_packets()?;
                return self.hc.combine_columns(b, &mut s.hc, hidden, &own, &peer, rank, tokens);
            }
            b.restore_fp8_middle(s.grouped.as_ref().ok_or("missing grouped middle")?, &s.output)?;
        } else { self.finish_rows(b, s, 0, tokens)?; }
        self.inject_batch(b, s, hidden, tokens)
    }

    /// Inject a tile's finished output rows; the HC still holds its normalized streams.
    pub(super) fn inject_batch(&self, b: &Backend<'_>, s: &mut MoeScratch<'_>,
        hidden: &mut DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.hc.combine_tiled(b, &mut s.hc, hidden, &s.output, tokens)
    }

    /// Weighted expert and shared rows of tokens [start, start + tokens) into their output rows.
    pub(super) fn finish_rows(&self, b: &Backend<'_>, s: &MoeScratch<'_>, start: usize, tokens: usize) -> Result<()> {
        self.finish_columns(b, s, start, tokens, 0, H, s.output.device_address() + (start * H * 4) as u64)
    }
    fn finish_columns(&self, b: &Backend<'_>, s: &MoeScratch<'_>, start: usize, tokens: usize,
        column_start: usize, columns: usize, output: u64) -> Result<()> {
        let _class = b.prefill_class(4);
        if columns == 0 { return Ok(()); }
        if tokens == 0 { return Ok(()); }
        b.prefill_read_bytes(4, || tokens * ((TOP + 1) * columns + TOP * 2 + 1) * 4);
        let mut params = spark_gpu_probe::driver::fp8_abi::MoeFinishParams {
            error: s.grouped.as_ref().ok_or("missing grouped routing")?.routing_error_address() + (start * 4) as u64,
            ids: s.ids.device_address() + (start * TOP * 4) as u64, weights: s.weights.device_address() + (start * TOP * 4) as u64,
            partials: s.partials.device_address() + (start * PARTIALS * 4) as u64,
            shared_gate: s.shared_gate.device_address() + (start * 4) as u64,
            output, activation_quantization: u32::from(self.nvfp4 || self.activation_quantization),
            column_start: column_start as u32, columns: columns as u32,
        };
        unsafe { b.launch_flash_rows("flash_moe_finish", columns.div_ceil(256) as u32, tokens as u32, 256, 0,
            &mut [(&mut params as *mut spark_gpu_probe::driver::fp8_abi::MoeFinishParams).cast()]) }
    }

    fn finish_token(&self, b: &Backend<'_>, s: &mut MoeScratch<'_>, residual: &mut DeviceVector<'_>) -> Result<()> {
        self.finish_rows(b, s, 0, 1)?;
        self.hc.combine_tiled(b, &mut s.hc, residual, &s.output, 1)
    }

    /// Consume rank-reduced expert rows, weight/accumulate ascending expert ID,
    /// add the reduced shared expert once with its sigmoid gate, then inject HC.
    pub fn finish(&self, b: &Backend<'_>, s: &mut MoeScratch<'_>, residual: &mut DeviceVector<'_>, reduced: &MoePartials) -> Result<()> {
        if reduced.values.len() != PARTIALS { return Err("MLP reduction shape mismatch".into()); }
        b.write_vector(&mut s.partials, &reduced.values)?;
        self.finish_device(b, s, residual, reduced)
    }

    /// Consume partials already returned to the device by the collective.
    pub(super) fn finish_device(&self, b: &Backend<'_>, s: &mut MoeScratch<'_>, residual: &mut DeviceVector<'_>, reduced: &MoePartials) -> Result<()> {
        let pending = s.pending.as_ref().ok_or("MLP finish needs projected rank partials")?;
        if residual.device_address() != pending.residual
            || reduced.ids != pending.ids || reduced.weights.iter().zip(pending.weights).any(|(a,b)| a.to_bits() != b.to_bits()) {
            return Err("MLP reduction packet does not match the pending routing/residual".into());
        }
        self.finish_token(b, s, residual)?;
        s.pending = None;
        Ok(())
    }

    /// A physical rank can execute a full token using its transport callback.
    /// The two-local-rank runner uses project/finish to overlap the collectives.
    pub fn run<'a>(&self, b: &Backend<'a>, s: &mut MoeScratch<'a>, residual: &mut DeviceVector<'_>,
        reduce: impl FnOnce(&mut [f32]) -> Result<()>) -> Result<()> {
        let mut partials = self.project(b, s, residual)?;
        reduce(&mut partials.values)?;
        self.finish(b, s, residual, &partials)
    }
}
