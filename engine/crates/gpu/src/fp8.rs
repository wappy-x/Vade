//! Resident E4M3 projections use checkpoint-declared dynamic W8A8 math.
use super::*;

/// Sticky device status per row, shared by the workspaces in a forward pass.
/// The host reads mapped storage only after an existing output readback sync.
#[derive(Clone)]
pub struct Fp8RoutingFlag<'a>(Rc<Buffer<'a>>);
impl<'a> Fp8RoutingFlag<'a> {
    pub fn new(b: &Backend<'a>, rows: usize) -> Result<Self, String> {
        if rows == 0 { return Err("empty routing status".into()); }
        let flag = Self(Rc::new(Buffer::mapped(b.context, byte_count(rows, 4)?)?));
        flag.reset(b)?;
        Ok(flag)
    }
    /// Clear the full allocation, then optionally reject one active row.
    /// This runs outside graph capture, so every replay sees fresh status.
    pub fn clear(&self, b: &Backend<'_>, rows: usize, rank: usize) -> Result<(), String> {
        if rows == 0 || rows > self.0.bytes / 4 { return Err("routing status active row capacity".into()); }
        self.reset(b)?;
        if let Some(row) = crate::routing::forced_routing_rejection(rows, Some(rank)) {
            unsafe { check(cuMemsetD8Async(self.0.address + (row * 4) as u64, 1, 4, b.stream.raw), "routing row injection")?; }
        }
        Ok(())
    }
    /// Padded decode slots reuse their existing status words. Set them after
    /// the ordinary clear, outside capture, so replay excludes their experts.
    pub fn pad_rows(&self, b: &Backend<'_>, rows: usize, active: u16) -> Result<(), String> {
        if rows == 0 || rows > 16 || rows > self.0.bytes / 4
            || u32::from(active) >> rows != 0 || !ptr::eq(self.0._context, b.context) {
            return Err("routing survivor mask capacity or context".into());
        }
        b.context.activate()?;
        for row in 0..rows {
            if active & (1 << row) == 0 {
                unsafe { check(cuMemsetD8Async(self.0.address + (row * 4) as u64, 1, 4,
                    b.stream.raw), "routing padded row")?; }
            }
        }
        b.context.pending.set(true);
        Ok(())
    }
    fn reset(&self, b: &Backend<'_>) -> Result<(), String> {
        if !ptr::eq(self.0._context, b.context) { return Err("routing flag belongs to another context".into()); }
        b.context.activate()?;
        unsafe { check(cuMemsetD8Async(self.0.address, 0, self.0.bytes, b.stream.raw), "routing flag clear")?; }
        b.context.pending.set(true);
        Ok(())
    }
    /// Requires completion of the pass's GPU producers (output readback).
    pub fn failed_after_readback(&self) -> Result<bool, String> {
        for row in 0..self.0.bytes / 4 {
            if self.row_failed_after_readback(row)? { return Ok(true); }
        }
        Ok(false)
    }
    pub fn row_failed_after_readback(&self, row: usize) -> Result<bool, String> {
        if self.0._context.pending.get() || row >= self.0.bytes / 4 {
            return Err("routing status read before completion or outside row capacity".into());
        }
        Ok(unsafe { ptr::read_volatile(self.0.host.unwrap().cast::<u32>().add(row)) != 0 })
    }
    /// Stable device status address for the NVFP4 compact routing pipeline.
    pub fn device_address(&self) -> u64 { self.0.address }
    pub fn bytes(&self) -> usize { self.0.bytes }
}

/// Reusable storage for one Flash Next expert layer, allocated on its first call.
/// Quantized input is shared by all routed experts; down writes token partials.
pub struct Fp8GroupedScratch<'a> {
    assignments: Buffer<'a>, activations: Rc<Buffer<'a>>, tokens: usize, rows: usize, activation_quantization: bool,
    middle: Option<(fp8_abi::GroupedParams, usize)>,
    packets: Option<Rc<Buffer<'a>>>,
    middle_world: usize,
    routing: Option<Fp8RoutingFlag<'a>>,
}
impl<'a> Fp8GroupedScratch<'a> {
    pub fn set_routing_flag(&mut self, flag: &Fp8RoutingFlag<'a>) -> Result<(), String> {
        if !ptr::eq(self.assignments._context, flag.0._context) { return Err("routing flag belongs to another context".into()); }
        self.routing = Some(flag.clone());
        Ok(())
    }
    pub fn routing_error_address(&self) -> u64 {
        self.routing.as_ref().map_or(self.assignments.address, |f| f.0.address)
    }
    pub fn routing_failed(&self, b: &Backend<'_>) -> Result<bool, String> {
        if !ptr::eq(self.assignments._context, b.context) { return Err("routing scratch belongs to another context".into()); }
        b.synchronize()?;
        if let Some(flag) = &self.routing { return flag.failed_after_readback(); }
        let mut flags = vec![0u32; self.tokens];
        unsafe { b.context.check_execution(cuMemcpyDtoH_v2(flags.as_mut_ptr().cast(),
            self.routing_error_address(), flags.len() * 4), "MoE routing error readback")?; }
        Ok(flags.into_iter().any(|flag| flag != 0))
    }
    pub fn token_capacity(&self) -> usize { self.tokens }
    pub fn assignment_bytes(&self) -> usize { self.assignments.bytes }
    pub fn activation_bytes(&self) -> usize { self.activations.bytes }
    pub fn middle_world(&self) -> usize { self.middle_world }
    pub fn bytes(&self) -> usize { self.assignment_bytes() + self.activation_bytes() + self.packets.as_ref().map_or(0, |b| b.bytes) }
    /// TP2 prompts reclaim gate/up storage and small decode packets are
    /// mapped. General worlds own rank-major packets; views retain allocations.
    pub fn middle_packets(&self) -> Result<(DeviceVector<'a>, DeviceVector<'a>), String> {
        let (p, _) = self.middle.ok_or("missing grouped middle")?;
        let words = p.route.tokens as usize * fp8_abi::middle_packet_words(self.middle_world);
        let offset = grouped_layout(self.rows, self.tokens, true)?.0[1] as usize;
        let span = (words * 4).div_ceil(256) * 256;
        let (owner, offset) = match self.packets.as_ref().filter(|_| self.middle_world != 2 || p.route.tokens <= 4) {
            Some(packets) => (packets, 0), None => (&self.activations, offset),
        };
        let view = |offset| -> Result<DeviceVector<'a>, String> {
            Ok(DeviceVector { buffer: Buffer::view(owner, offset, words * 4)?, len: words, managed: false,
                #[cfg(debug_assertions)] projection_scratch: Default::default() })
        };
        let receive = if self.middle_world == 2 { view(offset + span)? } else {
            DeviceVector { buffer: Buffer::view(owner, offset + span, self.middle_world * span)?,
                len: self.middle_world * span / 4, managed: false,
                #[cfg(debug_assertions)] projection_scratch: Default::default() }
        };
        Ok((view(offset)?, receive))
    }
    /// Only the small compact packets are mapped, not the routed activations.
    /// Complete every prior GPU reader/writer before lending disjoint host slices.
    /// # Safety
    /// The callback must not enqueue GPU access to this workspace or retain its host slices.
    pub unsafe fn with_middle_host(&self, b: &Backend<'_>, finished: bool,
        exchange: impl FnOnce(&[u8], &mut [u8]) -> Result<(), String>) -> Result<(), String> {
        if !ptr::eq(self.assignments._context, b.context) { return Err("grouped middle belongs to another context".into()); }
        let (p, _) = self.middle.ok_or("missing grouped middle")?;
        let packets = self.packets.as_ref().filter(|_| p.route.tokens <= 4).ok_or("missing mapped middle packets")?;
        let span = (p.route.tokens as usize * fp8_abi::MIDDLE_PACKET_WORDS * 4).div_ceil(256) * 256;
        let bytes = p.route.tokens as usize * if finished { 1280 * 4 } else { fp8_abi::MIDDLE_PACKET_WORDS * 4 };
        let host = packets.host.ok_or("middle packets are not mapped")?.cast::<u8>();
        b.wait_moe_readiness()?;
        unsafe { exchange(std::slice::from_raw_parts(host, bytes), std::slice::from_raw_parts_mut(host.add(span), bytes)) }
    }
    /// Small general-world packets share their rank-major device layout with
    /// the host. One producer wait covers all neighbours; callbacks only copy
    /// bytes, so receive slots need no GPU consumer events before forwarding.
    /// # Safety
    /// The callback must not enqueue GPU work or retain either host slice.
    pub unsafe fn with_middle_rank_host(&self, b: &Backend<'_>,
        exchange: impl FnOnce(&[u8], &mut [u8], usize) -> Result<(), String>) -> Result<(), String> {
        if !ptr::eq(self.assignments._context, b.context) { return Err("grouped middle belongs to another context".into()); }
        let (p, _) = self.middle.ok_or("missing grouped middle")?;
        let packets = self.packets.as_ref().filter(|_| self.middle_world != 2 && self.tokens <= 4)
            .ok_or("missing mapped rank middle packets")?;
        let span = (p.route.tokens as usize * fp8_abi::middle_packet_words(self.middle_world) * 4).div_ceil(256) * 256;
        let host = packets.host.ok_or("rank middle packets are not mapped")?.cast::<u8>();
        b.wait_moe_readiness()?;
        unsafe { exchange(std::slice::from_raw_parts(host, span),
            std::slice::from_raw_parts_mut(host.add(span), self.middle_world * span), span) }
    }
    /// Graph replay does not execute its Rust capture closure. Refresh the
    /// host description for this layer/width before any collective uses it.
    /// # Safety
    /// Descriptors and output retain the same validity, context and
    /// capacities required by fp8_grouped_ffn through stream completion.
    pub unsafe fn prepare_middle(&mut self, experts: u64, output: u64, tokens: usize) -> Result<(), String> {
        if tokens == 0 || tokens > self.tokens { return Err("grouped middle token capacity".into()); }
        let (p, tasks) = self.middle.as_mut().ok_or("missing grouped middle")?;
        let pairs = tokens * 10;
        p.weights = experts;
        p.output = output;
        p.route.tokens = tokens as u32;
        p.route.pair_count = pairs as u32;
        p.route.row_capacity = (pairs + 15 * pairs.min(512)) as u32;
        *tasks = pairs.div_ceil(128) + pairs.min(512);
        Ok(())
    }
    fn middle_params(&self, rank: usize) -> Result<fp8_abi::MiddleParams, String> {
        if rank >= self.middle_world { return Err("compact expert exchange rank".into()); }
        let (p, _) = self.middle.ok_or("missing grouped middle")?;
        let offsets = grouped_layout(self.rows, self.tokens, true)?.0;
        let base = self.activations.address;
        let send = self.packets.as_ref().filter(|_| self.middle_world != 2 || p.route.tokens <= 4)
            .map_or(base + offsets[1], |packets| packets.address);
        let span = (p.route.tokens as usize * fp8_abi::middle_packet_words(self.middle_world) * 4).div_ceil(256) * 256;
        let codes = base + offsets[1] + if self.middle_world == 2 { (2 * span) as u64 } else { 0 };
        let scales = codes + p.route.row_capacity as u64 * 640;
        if scales + p.route.row_capacity as u64 * 20 > base + offsets[2] {
            return Err("compact expert packets exceed reclaimed middle workspace".into());
        }
        Ok(fp8_abi::MiddleParams { route: p.route, weights: p.weights,
            local_codes: base + offsets[2], local_scales: base + offsets[3],
            send, receive: send + span as u64, codes, scales, partials: p.output,
            rank: rank as u32 })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    quantize: Handle, gemm: Handle, block_decode: Handle, weights_bf16: Handle,
    grouped_clear: Handle, grouped_count: Handle, grouped_prefix: Handle, grouped_single: Handle,
    pub(super) grouped_gather: Handle, grouped_order: Handle, grouped_scatter: Handle,
    grouped_gate: Handle, grouped_down: Handle, grouped_gate_decode: Handle, grouped_decode: Handle,
    grouped_gate_prefill: Handle, grouped_down_prefill: Handle,
    middle_pack: Handle, middle_unpack: Handle,
    middle_pack_general: Handle, middle_unpack_general: Handle, middle_columns: Handle,
    grouped_down_general: Handle, grouped_down_general_prefill: Handle,
    prefill_gemm: Handle, prefill_qkv: Handle,
    routed: [Handle; 8], // clear/count/order/gather/scatter/gate/up/down
    routed_single: Option<[Handle; 3]>, // gate_up/down/quantize, resolved on first small-row routed use
    routed_quantize: Option<[Handle; 2]>, // raw route validation, fused top-k/input quantize
}
impl Kernels {
    pub(super) fn decode_expert(&self, function: Handle) -> bool {
        [self.grouped_clear, self.grouped_count, self.grouped_prefix, self.grouped_single,
            self.grouped_gather, self.grouped_order, self.grouped_scatter, self.grouped_gate,
            self.grouped_down, self.grouped_gate_decode, self.grouped_decode, self.grouped_gate_prefill,
            self.grouped_down_prefill, self.middle_pack, self.middle_unpack, self.middle_pack_general,
            self.middle_unpack_general, self.middle_columns, self.grouped_down_general,
            self.grouped_down_general_prefill].contains(&function) || self.routed.contains(&function) || self.routed_single.is_some_and(|f| f.contains(&function))
    }
    pub(super) fn kind(&self, function: Handle) -> Option<usize> {
        if self.routed_quantize.is_some_and(|f| f.contains(&function)) { return Some(10); }
        if self.routed_single.is_some_and(|f| f.contains(&function)) { return Some(0); }
        if let Some(i) = self.routed.iter().position(|&f| f == function) { return Some(if i < 4 { 10 } else if i == 4 { 7 } else { 0 }); }
        if [self.quantize, self.weights_bf16, self.grouped_clear, self.grouped_count,
            self.grouped_prefix, self.grouped_gather, self.grouped_order, self.grouped_single,
            self.middle_pack, self.middle_unpack, self.middle_pack_general, self.middle_unpack_general,
            self.middle_columns].contains(&function) { Some(10) }
        else if [self.gemm, self.block_decode, self.grouped_gate, self.grouped_down,
            self.grouped_gate_decode, self.grouped_decode, self.grouped_gate_prefill, self.grouped_down_prefill,
            self.prefill_gemm, self.grouped_down_general, self.grouped_down_general_prefill].contains(&function) { Some(0) }
        else if function == self.prefill_qkv { Some(14) }
        else if function == self.grouped_scatter { Some(7) } else { None }
    }
}

// W8A8 sub-buffers start on 256 bytes: the quantized operands feed 16-byte cp.async tiles.
// W8A8 needs no grouped down output; the output permutation writes token partials directly.
fn grouped_layout(rows: usize, tokens: usize, quantized: bool) -> Result<([u64; 6], usize), String> {
    const H: usize = 2560;
    const I: usize = 640;
    let sizes = [if quantized { 0 } else { byte_count(rows, H * 4)? }, byte_count(rows, I * 8)?, byte_count(rows, I)?,
        byte_count(rows, I / 32)?, byte_count(tokens, H)?, byte_count(tokens, H / 32)?];
    let (mut offsets, mut end) = ([0u64; 6], 0usize);
    for (offset, size) in offsets.iter_mut().zip(sizes) {
        let start = end.div_ceil(256) * 256;
        *offset = start as u64;
        end = start.checked_add(size).ok_or("grouped activation size overflow")?;
    }
    Ok((offsets, end))
}

impl<'a> Backend<'a> {
    /// Fund reusable quantization storage before invoking the fixed FP8 entry.
    pub fn fixed_fp8_scratch_words(dimensions: [usize; 2], tokens: usize) -> Result<usize, String> {
        super::batch::projection_params::FixedLinear::new(dimensions[0], dimensions[1], true)?
            .scratch(tokens).map(|(_, words)| words)
    }
    /// Fixed row-major dynamic W8A8 for prefill, decode, replay and verification.
    /// `round_output` names the global BF16 boundary; TP partials leave it false.
    pub fn fixed_fp8_projection(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, tokens: usize,
        round_output: bool) -> Result<(), String> {
        unsafe { self.fixed_fp8_projection_reusing_input(matrix, input, output, scratch, tokens, round_output, false) }
    }
    /// # Safety
    /// Reuse requires the same input bits, rows and K in the same scratch, with
    /// no intervening scratch write since its preceding fixed FP8 projection.
    pub unsafe fn fixed_fp8_projection_reusing_input(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, tokens: usize,
        round_output: bool, reuse: bool) -> Result<(), String> {
        let shape = super::batch::projection_params::FixedLinear::new(matrix.rows, matrix.cols, true)?;
        let (inputs, outputs) = shape.elements(tokens)?;
        let (scale_offset, words) = shape.scratch(tokens)?;
        let scales = matrix.fp8_scales.as_ref().ok_or("fixed FP8 requires BF16 block scales")?;
        if !ptr::eq(matrix.buffer._context, self.context) || !ptr::eq(scales._context, self.context)
            || matrix.packed || matrix.fp8_fragments || matrix.scalar_f32 || matrix.stride != matrix.cols
            || matrix.row_offset != 0 || matrix.column_offset != 0
            || matrix.buffer.address % 16 != 0 || scratch.buffer.address % 16 != 0 || scales.address % 2 != 0
            || matrix.buffer.bytes != byte_count(matrix.rows, matrix.cols)?
            || scales.bytes != byte_count(matrix.rows / 128 * (matrix.cols / 128), 2)? {
            return Err("fixed FP8 requires complete row-major weights and BF16 scales".into());
        }
        self.check_vector(input, inputs)?; self.check_vector(output, outputs)?;
        self.check_vector_prefix(scratch, words)?;
        if input.is_segmented() || output.is_segmented() || scratch.is_segmented() {
            return Err("fixed FP8 requires contiguous operands".into());
        }
        for (address, bytes) in [(input.buffer.address, inputs * 4), (output.buffer.address, outputs * 4),
            (matrix.buffer.address, matrix.buffer.bytes), (scales.address, scales.bytes)] {
            super::batch::projection_params::FixedLinear::disjoint(scratch.buffer.address, words * 4, address, bytes)?;
        }
        super::batch::projection_params::FixedLinear::disjoint(output.buffer.address, outputs * 4,
            matrix.buffer.address, matrix.buffer.bytes)?;
        super::batch::projection_params::FixedLinear::disjoint(output.buffer.address, outputs * 4, scales.address, scales.bytes)?;
        let kernels = self.fp8_kernels()?;
        // Both entries instantiate project_tile: identical K32 MMA chains,
        // separate scale multiply and ascending K128 FP32 additions.
        let narrow = tokens <= 32;
        let mut q = fp8_abi::QuantizeParams { input: input.buffer.address, output: scratch.buffer.address,
            scales: scratch.buffer.address + scale_offset as u64, rows: tokens as u32, cols: shape.cols,
            round_input: 1, packed_output: 0 };
        let mut p = fp8_abi::BlockParams { weights: matrix.buffer.address, weight_scales: scales.address,
            input: q.output, input_scales: q.scales, output: output.buffer.address, rows: q.rows,
            outputs: shape.outputs, cols: shape.cols, input_kind: 0, round_output: u32::from(round_output) };
        let [tm, tn, threads, shared] = if fp8_abi::register_decode(p.input_kind, p.rows, p.outputs) {
            fp8_abi::REGISTER_DECODE_TILE
        } else if tokens <= 16 { [16, 64, 128, 20608] }
        else if narrow { [32, 64, 128, 24832] } else { shape.tile() };
        self.decode_weight_read(|| matrix.buffer.bytes + scales.bytes);
        unsafe {
            if reuse {
                #[cfg(debug_assertions)] {
                    let (generation, packed) = scratch.projection_scratch.get();
                    debug_assert_eq!(packed, Some(generation), "FP8 reuse after scratch invalidation");
                }
            } else {
                scratch.invalidate_projection_scratch();
                self.enqueue_dims(kernels.quantize, (inputs / 128).div_ceil(8) as u32,
                    1, 1, 256, &mut [parameter(&mut q)])?;
                #[cfg(debug_assertions)] {
                    let (generation, _) = scratch.projection_scratch.get();
                    // Publish a reusable generation only after quantization was queued.
                    scratch.projection_scratch.set((generation, Some(generation)));
                }
            }
            self.enqueue_dims_shared(if narrow { kernels.block_decode } else { kernels.gemm }, p.rows.div_ceil(tm), p.outputs / tn,
                1, threads, shared, &mut [parameter(&mut p)])
        }
    }
    /// Reserve routing and activation buffers before a distributed inference step.
    pub fn reserve_fp8_grouped(&self, scratch: &mut Option<Fp8GroupedScratch<'a>>,
        tokens: usize, activation_quantization: bool) -> Result<(), String> {
        const E: usize = 512;
        const H: usize = 2560;
        const I: usize = 640;
        if tokens == 0 || tokens > 1 << 24 { return Err("invalid grouped workspace capacity".into()); }
        if scratch.as_ref().is_some_and(|s| !ptr::eq(s.assignments._context, self.context)) {
            return Err("grouped FP8 scratch belongs to another context".into());
        }
        let pairs = tokens * 10;
        let task_capacity = pairs.div_ceil(128) + pairs.min(E);
        let rows = pairs + 15 * pairs.min(E);
        if rows > u32::MAX as usize || task_capacity * 2 > 65535 {
            return Err("grouped FP8 routing exceeds the numeric ID or row ABI".into());
        }
        if scratch.as_ref().is_none_or(|s| s.tokens < tokens || s.activation_quantization != activation_quantization) {
            let words = tokens + E + E + 1 + 2 * pairs + tokens + 1 + 2 * task_capacity + 2 * rows;
            let timer = self.profiling.get().then(Instant::now);
            let routing = scratch.as_ref().and_then(|s| s.routing.clone());
            let middle_world = scratch.as_ref().map_or(2, |s| s.middle_world);
            *scratch = Some(Fp8GroupedScratch {
                assignments: Buffer::new(self.context, byte_count(words, 4)?)?,
                activations: Rc::new(Buffer::new(self.context, if activation_quantization {
                    grouped_layout(rows, tokens, true)?.1
                } else { byte_count(rows, H * 4 + I * 2)? })?),
                tokens, rows, activation_quantization, middle: None, packets: None, routing, middle_world,
            });
            self.stats.borrow_mut().allocation_ns += elapsed(timer);
        }
        Ok(())
    }

    /// Reserve before capture; TP2 prompts retain their reclaimed device packets.
    pub fn reserve_fp8_middle(&self, scratch: &mut Fp8GroupedScratch<'a>, world: usize) -> Result<(), String> {
        if !ptr::eq(scratch.assignments._context, self.context) || !scratch.activation_quantization {
            return Err("invalid compact middle workspace".into());
        }
        if world == 0 || world > 65535 || (scratch.packets.is_some() && scratch.middle_world != world) {
            return Err("compact middle world capacity".into());
        }
        scratch.middle_world = world;
        if world != 2 && scratch.packets.is_none() {
            let span = (scratch.tokens * fp8_abi::middle_packet_words(world) * 4).div_ceil(256) * 256;
            scratch.packets = Some(Rc::new(if scratch.tokens <= 4 {
                Buffer::mapped(self.context, (world + 1) * span)?
            } else { Buffer::new(self.context, (world + 1) * span)? }));
        }
        if scratch.tokens <= 4 && scratch.packets.is_none() {
            let span = (scratch.tokens * fp8_abi::MIDDLE_PACKET_WORDS * 4).div_ceil(256) * 256;
            scratch.packets = Some(Rc::new(Buffer::mapped(self.context, 2 * span)?));
        }
        Ok(())
    }

    /// Promote one projection into overwritten scratch, then consume it with
    /// the canonical BF16 GEMMs. The checkpoint stays resident only as FP8.
    pub fn prefill_weight_bf16<'w>(&self, matrix: &Matrix<'w>, scratch: &DeviceVector<'w>, offset: usize) -> Result<Matrix<'w>, String> {
        if !ptr::eq(matrix.buffer._context, self.context) || !(matrix.packed || matrix.is_fp8()) {
            return Err("prefill weights require local packed BF16 or resident FP8".into());
        }
        let buffer = if matrix.is_fp8() {
            let words = matrix.rows * matrix.cols / 2;
            self.check_vector(scratch, scratch.len)?;
            let view = scratch.slice(offset..offset.checked_add(words).ok_or("FP8 promotion capacity overflow")?)?;
            let mut params = crate::kernel_params::Fp8PrefillWeights {
                weights: matrix.buffer.address as *const u8,
                scales: matrix.fp8_scales.as_ref().unwrap().address as *const u16,
                output: view.buffer.address as *mut u32,
                rows: matrix.rows as u32, cols: matrix.cols as u32, experts: 1,
                weight_fragments: u32::from(matrix.fp8_fragments),
            };
            unsafe { self.enqueue_dims(self.fp8_kernels()?.weights_bf16,
                (matrix.rows * matrix.cols / 4).div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut params)])?; }
            view.buffer
        } else { matrix.weight_view(0)? };
        Ok(Matrix { buffer, rows: matrix.rows, cols: matrix.cols, packed: true, packed_blocked: !matrix.is_fp8() && matrix.packed_blocked,
            stride: matrix.stride, row_offset: matrix.row_offset, scalar_f32: false, fp8_scales: None, fp8_fragments: false, column_offset: 0 })
    }
    /// Long-prefill tiles read resident FP8 fragments directly, promoting each
    /// operand in registers exactly as prefill_weight_bf16 would; no scratch copy.
    pub fn prefill_fp8_direct(&self, matrices: &[&Matrix<'_>]) -> bool {
        matrices.iter().all(|m| m.is_fp8() && m.fp8_fragments && m.row_offset == 0 && m.stride == m.cols
            && m.rows % 128 == 0 && m.cols % 128 == 0 && m.buffer.address % 16 == 0
            && ptr::eq(m.buffer._context, self.context))
    }
    /// Projection plus residual and RMS partials, as matmul_packed_residual_bf16_device.
    pub fn matmul_packed_residual_fp8_device(&self, matrix: &Matrix<'_>, packed: &DeviceVector<'_>,
        state: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        if tokens == 0 || tokens % 256 != 0 || !self.prefill_fp8_direct(&[matrix]) { return Err("FP8 residual GEMM shape".into()); }
        self.check_vector(state, tokens * matrix.rows)?;
        self.check_vector(partials, partials.len)?;
        if partials.len < tokens * matrix.rows / 128 { return Err("FP8 residual GEMM capacity".into()); }
        self.prefill_fp8_gemm(self.fp8_kernels()?.prefill_gemm, &[matrix], packed, [state.buffer.address, 0, 0],
            partials.buffer.address, tokens, (tokens / 256) as u32, (matrix.rows / 128) as u32)
    }
    /// Q/K/V outputs rounded at the BF16 projection boundary, as prefill_qkv_packed_bf16_device.
    pub fn prefill_qkv_packed_fp8_device(&self, q: &Matrix<'_>, k: &Matrix<'_>, v: &Matrix<'_>, packed: &DeviceVector<'_>,
        out_q: &mut DeviceVector<'_>, out_k: &mut DeviceVector<'_>, out_v: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        if tokens == 0 || tokens % 256 != 0 || !self.prefill_fp8_direct(&[q, k, v]) || q.cols != k.cols || q.cols != v.cols || k.rows != v.rows {
            return Err("FP8 fused QKV shape".into());
        }
        self.check_vector(out_q, tokens * q.rows)?; self.check_vector(out_k, tokens * k.rows)?; self.check_vector(out_v, tokens * v.rows)?;
        self.prefill_fp8_gemm(self.fp8_kernels()?.prefill_qkv, &[q, k, v], packed,
            [out_q.buffer.address, out_k.buffer.address, out_v.buffer.address], 0, tokens,
            (tokens / 256) as u32, ((q.rows + k.rows + v.rows) / 128) as u32)
    }
    #[allow(clippy::too_many_arguments)]
    fn prefill_fp8_gemm(&self, function: Handle, matrices: &[&Matrix<'_>], packed: &DeviceVector<'_>, outputs: [u64; 3],
        partials: u64, tokens: usize, blocks: u32, rows: u32) -> Result<(), String> {
        use crate::kernel_params::{FP8_PREFILL_SHARED_BYTES, FP8_PREFILL_THREADS};
        let k = matrices[0].cols;
        self.check_vector(packed, packed.len)?;
        if packed.len < tokens.div_ceil(16) * 16 * k / 2 { return Err("FP8 prefill input capacity".into()); }
        let mut params = crate::kernel_params::Fp8PrefillGemm {
            weights: [std::ptr::null(); 3], scales: [std::ptr::null(); 3], input: packed.buffer.address as *const u32,
            outputs: outputs.map(|a| a as *mut f32), partials: partials as *mut f32, widths: [0; 3],
            k: k as u32, tokens: tokens as u32,
        };
        for (i, m) in matrices.iter().enumerate() {
            params.weights[i] = m.buffer.address as *const u8;
            params.scales[i] = m.fp8_scales.as_ref().ok_or("missing FP8 scales")?.address as *const u16;
            params.widths[i] = m.rows as u32;
        }
        unsafe { self.enqueue_dims_shared(function, blocks, rows, 1, FP8_PREFILL_THREADS, FP8_PREFILL_SHARED_BYTES, &mut [parameter(&mut params)]) }
    }
    /// Quantized chats retain the same W8A8 projection order as a single row.
    pub fn matmul_decode_bf16_reusing_input(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, packed: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<(), String> {
        if !matrix.is_fp8() { return self.matmul_decode_bf16_device(matrix, input, output, tokens); }
        self.fp8_dense_prefill(matrix, input, output, input, packed, tokens, reuse)
    }
    /// Checkpoint projections share one W8A8 kernel at every row count.
    pub(super) fn fp8_dense_decode(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.check_vector(input, tokens * matrix.cols)?;
        self.fp8_block_project(matrix, input, output, tokens, 0)
    }
    pub(super) fn fp8_dense_prefill(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, _weights: &DeviceVector<'_>, packed: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<(), String> {
        self.check_vector(input, tokens * matrix.cols)?;
        self.check_vector(packed, packed.len)?;
        let values = tokens.max(16) * matrix.cols;
        let scales = packed.buffer.address + values as u64;
        if packed.len * 4 < values + tokens * matrix.cols / 32 { return Err("W8A8 quantization scratch capacity".into()); }
        if !reuse {
            let mut p = fp8_abi::QuantizeParams { input: input.buffer.address, output: packed.buffer.address,
                scales, rows: tokens as u32, cols: matrix.cols as u32, round_input: 1, packed_output: u32::from(tokens <= 16) };
            unsafe { self.enqueue_dims(self.fp8_kernels()?.quantize, (tokens * matrix.cols / 128).div_ceil(8) as u32,
                1, 1, 256, &mut [parameter(&mut p)])?; }
        }
        self.fp8_block_project(matrix, packed, output, tokens, scales)
    }
    fn fp8_block_project(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, tokens: usize, scales: u64) -> Result<(), String> {
        if !ptr::eq(matrix.buffer._context, self.context) || tokens == 0
            || tokens > u32::MAX as usize || matrix.rows % 128 != 0 || matrix.cols % 128 != 0 {
            return Err("W8A8 projection context or shape".into());
        }
        self.check_vector(output, tokens * matrix.rows)?;
        if input.is_segmented() || output.is_segmented() { return Err("W8A8 projection requires contiguous operands".into()); }
        let mut p = fp8_abi::BlockParams { weights: matrix.buffer.address,
            weight_scales: matrix.fp8_scales.as_ref().ok_or("missing FP8 scales")?.address,
            input: input.buffer.address, input_scales: scales, output: output.buffer.address,
            rows: tokens as u32, outputs: matrix.rows as u32, cols: matrix.cols as u32,
            input_kind: if scales == 0 { 1 } else if tokens <= 16 { 2 } else { 0 }, round_output: 1 };
        let narrow = tokens <= 32 || scales == 0;
        self.decode_weight_read(|| matrix.buffer.bytes + matrix.fp8_scales.as_ref().unwrap().bytes);
        let [tm, tn, threads, shared] = if fp8_abi::register_decode(p.input_kind, p.rows, p.outputs) {
            fp8_abi::REGISTER_DECODE_TILE
        } else if p.input_kind == 2 { [16, 8, 128, 14336] }
        else if tokens > 16 && tokens <= 32 { [32, 64, 128, 24832] }
        else if narrow { [16, 64, 128, 20608] } else { [128, 128, 512, 66560] };
        unsafe { self.enqueue_dims_shared(if narrow { self.fp8_kernels()?.block_decode } else { self.fp8_kernels()?.gemm },
            p.rows.div_ceil(tm), p.outputs / tn, 1, threads, shared, &mut [parameter(&mut p)]) }
    }
    pub(super) fn fp8_kernels(&self) -> Result<Kernels, String> {
        if self.fp8.borrow().is_none() {
            let kernels = Kernels {
                quantize: self.module.function("fp8_quantize_128")?,
                gemm: self.module.function_with_shared("gemm_fp8_block128", 66560)?,
                block_decode: self.module.function_with_shared("gemm_fp8_block128_decode", 24832)?,
                weights_bf16: self.module.function("fp8_dequant_bf16_mma_weights")?,
                grouped_clear: self.module.function("fp8_grouped_clear")?,
                grouped_count: self.module.function("fp8_grouped_count")?,
                grouped_prefix: self.module.function("fp8_grouped_prefix")?,
                grouped_single: self.module.function("fp8_grouped_single")?,
                grouped_gather: self.module.function("fp8_grouped_gather")?,
                grouped_order: self.module.function("fp8_grouped_order")?,
                grouped_scatter: self.module.function("fp8_grouped_scatter")?,
                grouped_gate: self.module.function("fp8_grouped_gate_accurate")?,
                grouped_gate_decode: self.module.function("fp8_grouped_gate_decode_accurate")?,
                grouped_down: self.module.function("fp8_grouped_down")?,
                grouped_decode: self.module.function("fp8_grouped_down_decode")?,
                grouped_gate_prefill: self.module.function("fp8_grouped_gate_prefill")?,
                grouped_down_prefill: self.module.function("fp8_grouped_down_prefill")?,
                middle_pack: self.module.function("fp8_middle_pack")?,
                middle_unpack: self.module.function("fp8_middle_unpack")?,
                middle_pack_general: self.module.function("fp8_middle_pack_general")?,
                middle_unpack_general: self.module.function("fp8_middle_unpack_general")?,
                middle_columns: self.module.function("fp8_middle_columns")?,
                grouped_down_general: self.module.function("fp8_grouped_down_general")?,
                grouped_down_general_prefill: self.module.function("fp8_grouped_down_general_prefill")?,
                prefill_gemm: self.module.function_with_shared("gemm_fp8_t256n128", crate::kernel_params::FP8_PREFILL_SHARED_BYTES)?,
                prefill_qkv: self.module.function_with_shared("prefill_qkv_fp8_t256n128", crate::kernel_params::FP8_PREFILL_SHARED_BYTES)?,
                routed: [self.module.function("fp8_routed_clear")?, self.module.function("fp8_routed_count")?,
                    self.module.function("fp8_routed_order")?, self.module.function("fp8_routed_gather")?,
                    self.module.function("fp8_routed_scatter")?, self.module.function_with_shared("fp8_routed_gate", 66560)?,
                    self.module.function_with_shared("fp8_routed_up", 66560)?, self.module.function_with_shared("fp8_routed_down", 66560)?],
                routed_single: None,
                routed_quantize: None,
            };
            *self.fp8.borrow_mut() = Some(kernels);
        }
        let kernels = self.fp8.borrow();
        let kernels = kernels.as_ref().unwrap();
        Ok(*kernels)
    }

    /// Fixed W8A8 expert pipeline for every cohort/phase. No allocation or finish reduction.
    /// `p` names FP32 input and unweighted routed output; `work` holds input codes/scales,
    /// gate/up FP32, middle codes/scales, and reserved down staging, in that order.
    /// Down writes unweighted FP32 directly to the ascending-ID output slots.
    /// # Safety
    /// All addresses cover the checked spans in this context through stream completion.
    /// The immutable device table equals `descriptors` (optional shared-last at M<=4), with retained local weight/scale
    /// allocations. Route status is initialized by the forward owner (sticky across layers).
    /// Original block coordinates select local tensors before this call; no offset is reapplied.
    /// `routing`, when supplied, names disjoint live scores/IDs/weights for these rows;
    /// it replaces the preceding route launch without changing its scalar arithmetic.
    pub unsafe fn fp8_routed_ffn(&self, mut p: fp8_abi::RoutedParams,
        descriptors: &[fp8_abi::ExpertProjection], global_intermediate: u64, work: [u64; 6],
        routing: Option<crate::kernel_params::MoeRouteBatch>) -> Result<(), String> {
        if fp8_abi::RoutedParams::checked(p.grouped, p.layout).is_none() || p.grouped.output_shard != 0 {
            return Err("invalid fixed expert routing geometry".into());
        }
        let r = p.grouped.route;
        let (t, e, rows, pairs, h, i) = (r.tokens as usize, r.experts as usize,
            r.row_capacity as usize, r.pair_count as usize, p.grouped.cols as usize, p.layout.intermediate as usize);
        let mut shared = u32::from(t <= 4 && descriptors.len() == (e+1)*3);
        if descriptors.len() != (e+shared as usize)*3 { return Err("expert descriptor count".into()); }
        if let Some(route) = routing.as_ref() { if t > 4 || route.tokens != r.tokens || route.experts != r.experts
            || route.selected != p.layout.selected || route.status != r.error || route.scores == 0
            || route.indices == 0 || route.weights == 0 { return Err("fused routing geometry".into()); } }
        for triple in descriptors.chunks_exact(3) {
            let (g, u, d) = (triple[0], triple[1], triple[2]);
            for v in triple {
                if fp8_abi::ExpertProjection::checked(v.weights, v.scales, v.rows as u64, v.cols as u64,
                    v.first_n_block as u64, v.first_k_block as u64).is_none()
                    || v.weights % 16 != 0 || v.scales % 2 != 0 || v.scale_cols != v.cols / 128 || v.reserved != 0 {
                    return Err("invalid expert weight/scale descriptor".into());
                }
            }
            if g.cols as usize != h || u.cols != g.cols || u.rows != g.rows || g.rows as usize > i
                // Origins are global; routing scratch and descriptor rows are local.
                || global_intermediate % 128 != 0 || g.first_n_block as u64 + g.rows as u64 / 128 > global_intermediate / 128 || g.first_k_block != 0
                || d.rows as usize != h || d.cols != g.rows || g.first_n_block != u.first_n_block
                || g.first_k_block != u.first_k_block || d.first_k_block != g.first_n_block || d.first_n_block != g.first_k_block {
                return Err("inconsistent gate/up/down expert geometry".into());
            }
        }
        let tasks = pairs.div_ceil(128) + pairs.min(e);
        let quantize_blocks = u32::try_from((t*h/128).div_ceil(8)).map_err(|_| "expert quantizer grid overflow")?;
        if h.div_ceil(256) > 65535 { return Err("expert scatter grid overflow".into()); }
        let spans = [(r.error,t,4),(r.pairs,pairs,12),(r.counts,e,4),(r.starts,e+1,4),
            (r.pair_to_row,pairs,4),(r.token_pairs,pairs,4),(r.token_count,t,4),(r.tasks,1+2*tasks,4),
            (r.sorted_row,rows,4),(r.output_row,rows,4),(p.grouped.weights,descriptors.len(),40),
            (p.grouped.input,t*h,4),(p.grouped.output,t*p.layout.output_stride as usize*h,4),
            (work[0],t*h,1),(work[1],t*h/128,4),(work[2],rows*i,8),
            (work[3],rows*i,1),(work[4],rows*i/128,4),(work[5],rows*h,4)];
        for (n, &(address,count,width)) in spans.iter().enumerate() {
            let bytes = byte_count(count,width)? as u64;
            let alignment = if n == 10 { 8 } else if width == 1 { 16 } else { 4 };
            if address == 0 || address % alignment != 0 || address.checked_add(bytes).is_none() { return Err("expert scratch address/extent".into()); }
            for &(other,c,w) in &spans[..n] {
                super::batch::projection_params::FixedLinear::disjoint(address,bytes as usize,other,byte_count(c,w)?)?;
            }
        }
        let route_spans = routing.as_ref().map(|route| [(route.scores,t*e,4),(route.indices,pairs,4),(route.weights,pairs,4)]);
        if let Some(extra) = &route_spans { for (n,&(address,count,width)) in extra.iter().enumerate() {
            let bytes = byte_count(count,width)?;
            if address%4 != 0 || address.checked_add(bytes as u64).is_none() { return Err("fused routing span".into()); }
            for &(other,c,w) in spans.iter().chain(&extra[..n]) {
                super::batch::projection_params::FixedLinear::disjoint(address,bytes,other,byte_count(c,w)?)?;
            }
        } }
        for d in descriptors { for &(address,count,width) in spans.iter().chain(route_spans.iter().flatten()) {
            super::batch::projection_params::FixedLinear::disjoint(d.weights,byte_count(d.rows as usize,d.cols as usize)?,address,byte_count(count,width)?)?;
            super::batch::projection_params::FixedLinear::disjoint(d.scales,byte_count(d.rows as usize/128*d.scale_cols as usize,2)?,address,byte_count(count,width)?)?;
        } }
        let (input, output) = (p.grouped.input, p.grouped.output);
        let tile = fp8_abi::RoutedTile::new(r.tokens);
        let k = self.fp8_kernels()?;
        unsafe {
            if t <= 4 {
                let single = if let Some(single) = k.routed_single { single } else {
                    let single = [self.module.function_with_shared("fp8_routed_single_gate_up", 41216)?,
                        self.module.function_with_shared("fp8_routed_single_down", 20608)?,
                        self.module.function("fp8_quantize_activated")?];
                    self.fp8.borrow_mut().as_mut().unwrap().routed_single = Some(single);
                    single
                };
                let quantize = if let Some(quantize) = k.routed_quantize { quantize } else {
                    let quantize = [self.module.function("fp8_routed_small_quantize")?,
                        self.module.function("fp8_route_quantize")?];
                    self.fp8.borrow_mut().as_mut().unwrap().routed_quantize = Some(quantize);
                    quantize
                };
                let mut q = fp8_abi::QuantizeParams { input, output: work[0], scales: work[1],
                    rows: r.tokens, cols: p.grouped.cols, round_input: 1, packed_output: 0 };
                if let Some(mut route) = routing {
                    let mut pairs = r.pairs;
                    self.enqueue_dims(quantize[1], (t*h/128).div_ceil(7) as u32, 1, 1, 256,
                        &mut [parameter(&mut route),parameter(&mut pairs),parameter(&mut q)])?;
                } else { self.enqueue_dims(quantize[0], quantize_blocks, 1, 1, 256,
                    &mut [parameter(&mut p),parameter(&mut q)])?; }
                (p.grouped.input,p.grouped.input_scales,p.grouped.output) = (work[0],work[1],work[2]);
                let (mut middle, mut scales) = (work[3], work[4]);
                self.enqueue_dims_shared(single[0], (i / 64) as u32, p.layout.selected+shared, r.tokens, 256, 41216,
                    &mut [parameter(&mut p),parameter(&mut middle),parameter(&mut scales),parameter(&mut shared)])?;
                let (mut activated, mut qrows, mut qcols) = (work[2],
                    (t * (p.layout.selected + shared) as usize) as u32, i as u32);
                self.enqueue_dims(single[2], ((qrows as usize * (i / 128)).div_ceil(8)) as u32, 1, 1, 256,
                    &mut [parameter(&mut activated),parameter(&mut middle),parameter(&mut scales),
                        parameter(&mut qrows),parameter(&mut qcols)])?;
                (p.grouped.input,p.grouped.input_scales,p.grouped.output) = (work[3],work[4],output);
                return self.enqueue_dims_shared(single[1], (h / 64) as u32, p.layout.selected+shared, r.tokens, 128, 20608,
                    &mut [parameter(&mut p),parameter(&mut shared)]);
            }
            self.enqueue_dims(k.routed[0], e.max(t*32).div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut p)])?;
            self.enqueue_dims(k.routed[1], pairs.div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut p)])?;
            let mut route = r;
            self.enqueue_dims(k.grouped_prefix, 1, 1, 1, 32, &mut [parameter(&mut route)])?;
            self.enqueue_dims(k.routed[2], t.div_ceil(128) as u32, 1, 1, 128, &mut [parameter(&mut p)])?;
            let mut q = fp8_abi::QuantizeParams { input, output: work[0], scales: work[1],
                rows: r.tokens, cols: p.grouped.cols, round_input: 1, packed_output: 0 };
            self.enqueue_dims(k.quantize, quantize_blocks, 1, 1, 256, &mut [parameter(&mut q)])?;
            (p.grouped.input,p.grouped.input_scales,p.grouped.output) = (work[0],work[1],work[2]);
            for projection in 5..if tile.cols == 128 { 6 } else { 7 } {
                self.enqueue_dims_shared(k.routed[projection], tile.blocks(i, tile.cols == 128), tasks as u32,
                    1, tile.threads, tile.shared, &mut [parameter(&mut p)])?;
            }
            (p.grouped.input,p.grouped.input_scales,p.grouped.output) = (work[2],work[4],work[3]);
            self.enqueue_dims(k.routed[3], (i/8) as u32, tasks as u32, 1, 256, &mut [parameter(&mut p)])?;
            (p.grouped.input,p.grouped.input_scales,p.grouped.output) = (work[3],work[4],output);
            self.enqueue_dims_shared(k.routed[7], tile.blocks(h, false), tasks as u32, 1,
                tile.threads, tile.shared, &mut [parameter(&mut p)])
        }
    }

    /// Shared routing/allocation only: operand precision belongs to the caller.
    /// Assignments are token-major [tokens,10,3] (token, expert, weight).
    pub(super) unsafe fn prepare_grouped(&self, input: &DeviceVector<'_>,
        assignments: &DeviceVector<'_>, output: &DeviceVector<'_>,
        scratch: &mut Option<Fp8GroupedScratch<'a>>, activation_quantization: bool)
        -> Result<(fp8_abi::GroupedRouteParams, [u64; 6], usize), String> {
        let _class = self.prefill_class(3);
        const E: usize = 512;
        const H: usize = 2560;
        if input.len == 0 || input.len % H != 0 { return Err("invalid grouped input".into()); }
        let tokens = input.len / H;
        let pairs = tokens.checked_mul(10).ok_or("grouped assignment size overflow")?;
        self.prefill_read_bytes(3, || pairs * 3 * 4);
        self.check_vector(input, tokens * H)?;
        self.check_vector(output, tokens * 11 * H)?;
        self.check_vector(assignments, pairs.checked_mul(3).ok_or("grouped assignment size overflow")?)?;
        if input.is_segmented() || output.is_segmented() || assignments.is_segmented() {
            return Err("grouped FP8 FFN requires contiguous vectors".into());
        }
        self.reserve_fp8_grouped(scratch, tokens, activation_quantization)?;
        let task_capacity = pairs.div_ceil(128) + pairs.min(E);
        let rows = pairs.checked_add(15 * pairs.min(E)).ok_or("grouped row capacity overflow")?;
        let scratch = scratch.as_ref().unwrap();
        let capacity_pairs = scratch.tokens * 10;
        let capacity_tasks = capacity_pairs.div_ceil(128) + capacity_pairs.min(E);
        let error = scratch.routing.as_ref().map_or(scratch.assignments.address, |f| f.0.address);
        if scratch.routing.is_none() {
            // Standalone layer callers retain per-call row status.
            unsafe { check(cuMemsetD8Async(error, 0, scratch.tokens * 4, self.stream.raw), "routing flag clear")?; }
            self.context.pending.set(true);
        }
        if scratch.routing.as_ref().is_some_and(|f| f.bytes() < tokens * 4) {
            return Err("routing status row capacity".into());
        }
        let counts = scratch.assignments.address + (scratch.tokens * 4) as u64;
        let starts = counts + (E * 4) as u64;
        let pair_to_row = starts + ((E + 1) * 4) as u64;
        let token_pairs = pair_to_row + (capacity_pairs * 4) as u64;
        let token_count = token_pairs + (capacity_pairs * 4) as u64;
        let tasks = token_count + (scratch.tokens * 4) as u64;
        let sorted_row = tasks + 4 + (capacity_tasks * 8) as u64;
        let output_row = sorted_row + (scratch.rows * 4) as u64;
        let packed = scratch.activations.address;
        let offsets = grouped_layout(scratch.rows, scratch.tokens, activation_quantization)?.0;
        let mut route = fp8_abi::GroupedRouteParams {
            error, pairs: assignments.buffer.address, counts, starts, pair_to_row, token_pairs,
            token_count, tasks, sorted_row, output_row, experts: E as u32, tokens: tokens as u32,
            pair_count: pairs as u32, row_capacity: rows as u32,
        };
        let kernels = self.fp8_kernels()?;
        unsafe {
            if tokens == 1 {
                self.enqueue_dims(kernels.grouped_single, 1, 1, 1, 256, &mut [parameter(&mut route)])?;
            } else {
                self.enqueue_dims(kernels.grouped_clear, E.max(tokens * 32).div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut route)])?;
                self.enqueue_dims(kernels.grouped_count, pairs.div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut route)])?;
                self.enqueue_dims(kernels.grouped_prefix, 1, 1, 1, 32, &mut [parameter(&mut route)])?;
                self.enqueue_dims(kernels.grouped_order, tokens.div_ceil(128) as u32, 1, 1, 128, &mut [parameter(&mut route)])?;
            }
        }
        Ok((route, offsets.map(|offset| packed + offset), task_capacity))
    }

    /// Routed FFN over borrowed, rank-sharded resident FP8 tensors.
    /// `experts` holds 512 seven-u64 records: gate/scales, up/scales,
    /// down/scales, local intermediate width. Dynamic W8A8, or explicit BF16 operands.
    /// Output retains ten unweighted expert rows in ascending ID plus a reserved
    /// shared row per token, so rank reduction precedes the ordered weighted sum.
    /// # Safety
    /// Every descriptor address must remain valid through stream completion;
    /// widths are multiples of 128 up to 640 with matching scale grids (width 0: the
    /// rank owns no channels of that expert).
    /// FP8 weight addresses start on 16 bytes: W8A8 tiles copy them with cp.async.16.
    /// When deferring down, its descriptors instead hold aligned output rows over
    /// all 640 channels; the caller must exchange the middle before finishing.
    pub unsafe fn fp8_grouped_ffn(&self, experts: &DeviceVector<'_>,
        input: &DeviceVector<'_>, assignments: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        scratch: &mut Option<Fp8GroupedScratch<'a>>, activation_quantization: bool, defer_down: bool) -> Result<(), String> {
        const E: usize = 512;
        const H: usize = 2560;
        const I: usize = 640;
        if defer_down && !activation_quantization { return Err("compact expert exchange requires W8A8 operands".into()); }
        self.check_vector(experts, E * 14)?;
        let (route, addresses, task_capacity) = self.prepare_grouped(input, assignments, output, scratch, activation_quantization)?;
        if defer_down {
            let s = scratch.as_mut().unwrap();
            self.reserve_fp8_middle(s, s.middle_world)?;
        }
        let (tokens, rows) = (route.tokens as usize, route.row_capacity as usize);
        let (packed, middle) = (addresses[0], addresses[1]);
        let kernels = self.fp8_kernels()?;
        unsafe {
            let mut p = fp8_abi::GroupedParams {
                route, weights: experts.device_address(), input: input.buffer.address,
                input_scales: 0, output: packed, cols: H as u32,
                activation_quantization: u32::from(activation_quantization),
                output_shard: 0,
            };
            self.decode_expert_read(&p, false)?;
            if !defer_down { self.decode_expert_read(&p, true)?; }
            if activation_quantization {
                let [_, _, middle_q, middle_scales, input_q, input_scales] = addresses;
                // Original tokens are quantized once, before expert indirection.
                let mut q = fp8_abi::QuantizeParams {
                    input: input.buffer.address, output: input_q, scales: input_scales,
                    rows: tokens as u32, cols: H as u32, round_input: 1, packed_output: 0,
                };
                self.enqueue_dims(kernels.quantize, (tokens * H / 128).div_ceil(8) as u32,
                    1, 1, 256, &mut [parameter(&mut q)])?;
                (p.input, p.input_scales, p.output) = (input_q, input_scales, middle);
                // A routed expert has at most one row per input token.
                // Verification needs only its occupied sixteen-row planes.
                let planes = fp8_abi::grouped_planes(route.tokens, tokens >= 32);
                let (gate, down) = if tokens >= 32 {
                    (kernels.grouped_gate_prefill, kernels.grouped_down_prefill)
                } else { (kernels.grouped_gate, kernels.grouped_down) };
                self.enqueue_dims(gate, 20 * planes, task_capacity as u32, 1, 128, &mut [parameter(&mut p)])?;
                (p.input, p.input_scales, p.output) = (middle, middle_scales, middle_q);
                self.enqueue_dims(kernels.grouped_gather, 80, task_capacity as u32, 1, 256, &mut [parameter(&mut p)])?;
                (p.input, p.input_scales, p.output) = (middle_q, middle_scales, output.buffer.address);
                if defer_down {
                    scratch.as_mut().unwrap().middle = Some((p, task_capacity));
                    return Ok(());
                }
                self.enqueue_dims(down, (H / 64) as u32 * planes, task_capacity as u32, 1, 128, &mut [parameter(&mut p)])?;
                return Ok(());
            } else {
                self.enqueue_dims(kernels.grouped_gather, (rows * H / 8).div_ceil(256) as u32,
                    1, 1, 256, &mut [parameter(&mut p)])?;
                (p.input, p.output) = (packed, middle);
                let (function, blocks, columns, planes) = if tokens == 1 {
                    (kernels.grouped_gate_decode, (I / 8) as u32, 1, 10)
                } else { (kernels.grouped_gate, (I / 32) as u32, task_capacity as u32, 1) };
                self.enqueue_dims(function, blocks, columns, planes, 256, &mut [parameter(&mut p)])?;
                (p.input, p.output) = (middle, packed);
                let (function, blocks, columns, threads) = if tokens == 1 {
                    (kernels.grouped_decode, (H / 8) as u32, 1, 128)
                } else { (kernels.grouped_down, (H / 128) as u32, task_capacity as u32 * 2, 256) };
                self.enqueue_dims(function, blocks, columns, planes, threads, &mut [parameter(&mut p)])?;
            }
            (p.input, p.output) = (packed, output.buffer.address);
            self.enqueue_dims(kernels.grouped_scatter, tokens as u32, H.div_ceil(256) as u32,
                1, 256, &mut [parameter(&mut p)])
        }
    }

    pub fn pack_fp8_middle(&self, scratch: &Fp8GroupedScratch<'_>, rank: usize) -> Result<(), String> {
        if !ptr::eq(scratch.assignments._context, self.context) { return Err("grouped middle belongs to another context".into()); }
        let mut p = scratch.middle_params(rank)?;
        if scratch.middle_world != 2 {
            let mut g = fp8_abi::GeneralMiddleParams { middle: p, world: scratch.middle_world as u32, output: 0 };
            return unsafe { self.enqueue_dims(self.fp8_kernels()?.middle_pack_general,
                (p.route.tokens as usize * fp8_abi::middle_packet_words(scratch.middle_world)).div_ceil(256) as u32,
                1, 1, 256, &mut [parameter(&mut g)]) };
        }
        unsafe { self.enqueue_dims(self.fp8_kernels()?.middle_pack,
            (p.route.tokens as usize * fp8_abi::MIDDLE_PACKET_WORDS).div_ceil(256) as u32,
            1, 1, 256, &mut [parameter(&mut p)]) }
    }

    pub fn finish_fp8_middle(&self, scratch: &Fp8GroupedScratch<'_>, rank: usize) -> Result<(), String> {
        if !ptr::eq(scratch.assignments._context, self.context) { return Err("grouped middle belongs to another context".into()); }
        let mut middle = scratch.middle_params(rank)?;
        let (mut p, tasks) = scratch.middle.ok_or("missing grouped middle")?;
        let kernels = self.fp8_kernels()?;
        unsafe {
            if scratch.middle_world != 2 {
                let world = scratch.middle_world;
                let mut g = fp8_abi::GeneralMiddleParams { middle, world: world as u32, output: 0 };
                self.enqueue_dims(kernels.middle_unpack_general, (p.route.tokens as usize * 4210).div_ceil(256) as u32,
                    1, 1, 256, &mut [parameter(&mut g)])?;
                let columns = fp8_abi::middle_columns(rank, world);
                if columns.is_empty() { return Ok(()); }
                p.input = middle.codes;
                p.input_scales = middle.scales;
                p.output += (columns.start * 4) as u64;
                p.output_shard = ((world as u32) << 16) | columns.len() as u32;
                self.decode_expert_read(&p, true)?;
                return self.enqueue_dims(if p.route.tokens >= 32 { kernels.grouped_down_general_prefill } else { kernels.grouped_down_general },
                    (columns.len() / 64) as u32 * fp8_abi::grouped_planes(p.route.tokens, p.route.tokens >= 32), tasks as u32, 1, 128, &mut [parameter(&mut p)]);
            }
            self.enqueue_dims(kernels.middle_unpack, (p.route.tokens as usize * 2930).div_ceil(256) as u32,
                1, 1, 256, &mut [parameter(&mut middle)])?;
            p.input = middle.codes;
            p.input_scales = middle.scales;
            p.output += (rank * 1280 * 4) as u64;
            p.output_shard = 1;
            self.decode_expert_read(&p, true)?;
            self.enqueue_dims(if p.route.tokens >= 32 { kernels.grouped_down_prefill } else { kernels.grouped_down },
                20 * fp8_abi::grouped_planes(p.route.tokens, p.route.tokens >= 32), tasks as u32, 1, 128, &mut [parameter(&mut p)])
        }
    }

    // Tasks exclude rejected rows; each live M16 plane reuses one expert's weights.
    fn decode_expert_read(&self, p: &fp8_abi::GroupedParams, down: bool) -> Result<(), String> {
        self.decode_byte_probe(1, || {
            if p.activation_quantization != 1 { return Err("decode byte probe requires W8A8 experts".into()); }
            self.synchronize()?;
            let n = self.decode_words(p.route.tasks, 1)?[0] as usize;
            if n > p.route.pair_count as usize { return Err("decode byte probe: invalid expert tasks".into()); }
            let tasks = self.decode_words(p.route.tasks + 4, n * 2)?;
            let counts = self.decode_words(p.route.counts, p.route.experts as usize)?;
            let descriptors = self.decode_words(p.weights, p.route.experts as usize * 14)?;
            let (mut bytes, mut gathered) = (0, 0);
            for task in tasks.chunks_exact(2) {
                let expert = task[0] as usize;
                if expert >= counts.len() || task[1] >= counts[expert] { return Err("decode byte probe: invalid expert task".into()); }
                let width = descriptors[expert * 14 + 12] as usize;
                if width > 640 || width % 128 != 0 || descriptors[expert * 14 + 13] != 0 { return Err("decode byte probe: invalid expert width".into()); }
                let planes = (counts[expert] - task[1]).min(128).div_ceil(16) as usize;
                if !down { gathered += (counts[expert] - task[1]).min(128) as usize * width * 8; }
                let (rows, cols) = if !down { (width * 2, 2560) } else if p.output_shard == 0 { (2560, width) }
                    else { (if p.output_shard == 1 { 1280 } else { p.output_shard as usize & 65535 }, 640) };
                bytes += planes * decode_profile::fp8_read_bytes(rows, cols);
            }
            if !down { self.prefill_read_bytes(4, || gathered); }
            Ok(bytes)
        })
    }

    /// Restore only active rows; prompt tails and verification reuse a larger output allocation.
    pub fn restore_fp8_middle(&self, scratch: &Fp8GroupedScratch<'_>, output: &DeviceVector<'_>) -> Result<(), String> {
        if scratch.middle_world == 2 || !ptr::eq(scratch.assignments._context, self.context) {
            return Err("general middle restoration world or context".into());
        }
        let p = scratch.middle_params(0)?;
        self.check_vector_prefix(output, p.route.tokens as usize * 2560)?;
        let mut g = fp8_abi::GeneralMiddleParams { middle: p, world: scratch.middle_world as u32, output: output.device_address() };
        unsafe { self.enqueue_dims(self.fp8_kernels()?.middle_columns, (p.route.tokens as usize * 2560).div_ceil(256) as u32,
            1, 1, 256, &mut [parameter(&mut g)]) }
    }

}

#[cfg(test)]
mod register_decode_tests {
    use super::*;
    #[test]
    #[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
    fn block_project_m32_matches_m16_splits_bit_for_bit() -> Result<(), String> {
        if std::env::var_os("VADE_TYPED_35B_CHECKPOINT").is_none() {
            return Err("explicit GPU gate requires VADE_TYPED_35B_CHECKPOINT".into());
        }
        with_backend(0, crate::layer_ptx()?, 64*1024*1024, |b| {
            let n = 2048;
            for k in [640usize, 2048] {
                let mut weights = b.vector(n*k/4)?;
                b.write_vector_bytes(&mut weights, &(0..n*k).map(|v| (0x20+(v*13+v/k*7)%24) as u8
                    | if (v/k+v%k)%3==0 {0x80} else {0}).collect::<Vec<_>>())?;
                let mut scales = b.vector(n*k/16384/2)?;
                b.write_vector_bytes(&mut scales, &(0..n*k/16384).flat_map(|v|
                    (0x3c00u16+(v%5) as u16*32).to_le_bytes()).collect::<Vec<_>>())?;
                let matrix = Matrix { buffer: weights.buffer, rows:n, cols:k, packed:false, packed_blocked:false,
                    stride:k, row_offset:0, column_offset:0, scalar_f32:false, fp8_scales:Some(scales.buffer), fp8_fragments:false };
                let input = b.upload_vector(&(0..32*k).map(|v| ((v*17+v/k*3)%97) as f32/32.0-1.5).collect::<Vec<_>>())?;
                let (codes, xs) = (b.vector(32*k/4)?, b.vector(32*k/128)?);
                let mut q = fp8_abi::QuantizeParams { input:input.device_address(), output:codes.device_address(),
                    scales:xs.device_address(), rows:32, cols:k as u32, round_input:1, packed_output:0 };
                let kernels = b.fp8_kernels()?;
                unsafe { b.enqueue_dims(kernels.quantize, (32*k/128).div_ceil(8) as u32, 1,1,256,&mut [parameter(&mut q)])?; }
                for kind in [0,1] { for m in [17usize,31,32] {
                    let source = if kind==0 { &codes } else { &input };
                    let (mut actual, reference) = (b.vector(m*n)?, b.vector(m*n)?);
                    b.fp8_block_project(&matrix, source, &mut actual, m, if kind==0 {q.scales} else {0})?;
                    for start in [0usize,16] {
                        let mut p = fp8_abi::BlockParams { weights:matrix.device_address(), weight_scales:matrix.fp8_scales.as_ref().unwrap().address,
                            input:source.device_address()+(start*k*if kind==0 {1} else {4}) as u64,
                            input_scales:if kind==0 {q.scales+(start*k/128*4) as u64} else {0},
                            output:reference.device_address()+(start*n*4) as u64, rows:(m-start).min(16) as u32,
                            outputs:n as u32, cols:k as u32, input_kind:kind, round_output:1 };
                        unsafe { b.enqueue_dims_shared(kernels.block_decode,1,n as u32/64,1,128,20608,&mut [parameter(&mut p)])?; }
                    }
                    let a = b.read_vector_values(&actual)?; let r = b.read_vector_values(&reference)?;
                    assert!(a.iter().all(|v| v.is_finite()) && a.iter().any(|&v| v!=0.0));
                    assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "kind={kind} M={m} N={n} K={k}");
                } }
            }
            Ok(())
        })
    }
    #[test]
    #[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
    fn register_decode_matches_generic_bit_for_bit() -> Result<(), String> {
        if std::env::var_os("VADE_TYPED_35B_CHECKPOINT").is_none() {
            return Err("explicit GPU gate requires VADE_TYPED_35B_CHECKPOINT".into());
        }
        with_backend(0, crate::layer_ptx()?, 64*1024*1024, |b| {
            let kernels = b.fp8_kernels()?;
            for n in [128usize,256,512] { for k in [640usize,2048,3072] {
                let mut weights = b.vector(n*k/4)?;
                b.write_vector_bytes(&mut weights, &(0..n*k).map(|v|
                    (0x20+((v*13+v/k*7)%24) as u8) | if (v/k+v%k)%3==0 {0x80} else {0}).collect::<Vec<_>>())?;
                let mut scales = b.vector((n*k/16384).div_ceil(2))?;
                b.write_vector_bytes(&mut scales, &(0..n*k/16384).flat_map(|v|
                    (0x3c00u16+(v%5) as u16*32).to_le_bytes()).collect::<Vec<_>>())?;
                let input = b.upload_vector(&(0..16*k).map(|v| ((v*17+v/k*3)%97) as f32/32.0-1.5).collect::<Vec<_>>())?;
                let (codes, xs) = (b.vector(16*k/4)?, b.vector(16*k/128)?);
                let (mut small, mut generic) = (b.vector(16*n)?, b.vector(16*n)?);
                let mut q = fp8_abi::QuantizeParams { input: input.device_address(), output: codes.device_address(),
                    scales: xs.device_address(), rows:16, cols:k as u32, round_input:1, packed_output:0 };
                unsafe { b.enqueue_dims(kernels.quantize, (16*k/128).div_ceil(8) as u32,
                    1,1,256,&mut [parameter(&mut q)])?; }
                for m in 1..=16 { for rounded in [0,1] {
                    b.write_vector(&mut small, &vec![f32::NAN;16*n])?;
                    b.write_vector(&mut generic, &vec![f32::NAN;16*n])?;
                    let mut p = fp8_abi::BlockParams { weights:weights.device_address(), weight_scales:scales.device_address(),
                        input:q.output, input_scales:q.scales, output:small.device_address(), rows:m,
                        outputs:n as u32, cols:k as u32, input_kind:0, round_output:rounded };
                    let [tm,tn,threads,shared] = fp8_abi::REGISTER_DECODE_TILE;
                    unsafe { b.enqueue_dims_shared(kernels.block_decode,p.rows.div_ceil(tm),p.outputs/tn,
                        1,threads,shared,&mut [parameter(&mut p)])?; }
                    // The generic entry keeps the same K128 arithmetic on these operands.
                    p.rows=16; p.output=generic.device_address();
                    unsafe { b.enqueue_dims_shared(kernels.gemm,1,p.outputs/128,
                        1,512,66560,&mut [parameter(&mut p)])?; }
                    let a = b.read_vector_values(&small)?;
                    let g = b.read_vector_values(&generic)?;
                    let len = m as usize*n;
                    assert!(a[..len].iter().all(|v| v.is_finite()));
                    assert!(a[..len].iter().any(|&v| v!=0.0));
                    assert_eq!(a[..len].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        g[..len].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "M={m} N={n} K={k} round={rounded}");
                } }
            } }
            Ok(())
        })
    }
}
