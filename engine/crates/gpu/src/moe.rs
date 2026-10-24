//! Device-resident expert selection over the model's existing weight arena.
use super::*;

pub struct ExpertTable<'a> {
    // Keep the original views alive; pointer metadata never copies weights.
    _matrices: Vec<Matrix<'a>>, pointers: Buffer<'a>,
    hidden: usize, intermediate: usize, experts: usize,
}
impl ExpertTable<'_> {
    pub fn hidden(&self) -> usize { self.hidden }
    pub fn intermediate(&self) -> usize { self.intermediate }
    pub fn experts(&self) -> usize { self.experts }
    pub fn active_weight_bytes(&self, selected: usize) -> usize {
        (self.experts * self.hidden + selected * 3 * self.hidden * self.intermediate) * 2
    }
}

/// A routed row whose router scores were not all finite ran with zero expert
/// weight (kernel_params::finish_route). Only that row's request may fail.
pub const MOE_ROUTING_ERROR: &str = "MoE routing rejected non-finite router scores";

/// Test hook, off unless VADE_TEST_ROUTING_REJECT=N[:ROW[:ROWS[:RANK_MASK]]]
/// is set at startup (N=0 is off). Reject once at the N-th matching verdict.
/// N:ROW is the legacy form; ROW defaults to zero and is clamped.
/// ROWS=0 matches any width; N:3:8 selects row 3 in an eight-row forward.
/// Flash counts clears before capture/replay and filters global ranks by mask;
/// dense/hybrid count host verdicts without a distributed rank filter.
use crate::routing::{ROUTING_DEFERRED, forced_routing_rejection};

impl super::Backend<'_> {
    /// The native gate prepares no-hook references before the startup env hook
    /// counts its first verdict. No environment mutation or counter reset.
    pub fn defer_test_routing_rejection(&self, deferred: bool) {
        ROUTING_DEFERRED.store(deferred, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `status` holds one host-readable word per routed row (kernel_params::finish_route).
pub struct MoeScratch<'a> {
    scores: DeviceVector<'a>, indices: DeviceVector<'a>, weights: DeviceVector<'a>,
    activated: DeviceVector<'a>, projected: DeviceVector<'a>, completed: DeviceVector<'a>, status: DeviceVector<'a>, selected: usize,
}

pub struct MoeBatchScratch<'a> {
    scores: DeviceVector<'a>, indices: DeviceVector<'a>, weights: DeviceVector<'a>, status: DeviceVector<'a>,
    rows: DeviceVector<'a>, counts: DeviceVector<'a>, offsets: DeviceVector<'a>, inverse: DeviceVector<'a>,
    packed_input: DeviceVector<'a>, packed_activated: DeviceVector<'a>,
    projected: DeviceVector<'a>,
    tokens: usize, selected: usize, hidden: usize, intermediate: usize, experts: usize,
    bf16: bool,
}
impl MoeScratch<'_> {
    pub fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        for vector in [&self.scores, &self.indices, &self.weights, &self.activated, &self.projected, &self.completed, &self.status] {
            vector.allocations(out);
        }
    }
}
impl MoeBatchScratch<'_> {
    /// Hybrid checkpoints round router weights and expert accumulation to BF16.
    pub fn bf16_boundaries(&mut self) { self.bf16 = true; }
    pub fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        for vector in [&self.scores, &self.indices, &self.weights, &self.status, &self.rows, &self.counts, &self.offsets,
            &self.inverse, &self.packed_input, &self.packed_activated, &self.projected] {
            vector.allocations(out);
        }
    }
}

pub(super) struct Kernels {
    router: Handle, route: Handle, gate_up: Handle, down: Handle, merge: Handle,
    route_batch: Handle, route_group_decode: Handle, group: Handle, group_offsets: Handle, pack_grouped: Handle, pack_input: Handle,
    prefill_gate_up: Handle, prefill_gate_up_wide: Handle, decode_grouped_gate_up: Handle, decode_grouped_down: Handle, grouped_prefill_gate_up: Handle, grouped_prefill_down: Handle, prefill_down: Handle, merge_batch: Handle, merge_residual_rms: Handle,
}
impl Kernels {
    pub(super) fn kind(&self, function: Handle) -> Option<usize> {
        if [self.gate_up, self.prefill_gate_up, self.prefill_gate_up_wide, self.decode_grouped_gate_up, self.grouped_prefill_gate_up].contains(&function) { Some(21) }
        else if [self.down, self.prefill_down, self.decode_grouped_down, self.grouped_prefill_down].contains(&function) { Some(22) }
        else if [self.router, self.route, self.route_batch, self.route_group_decode, self.group, self.group_offsets].contains(&function) { Some(18) }
        else if [self.merge, self.merge_batch, self.merge_residual_rms, self.pack_grouped, self.pack_input].contains(&function) { Some(19) }
        else { None }
    }
}

impl<'a> Backend<'a> {
    /// Route prompt rows by warp and retain the canonical FFN pairs on-device.
    /// # Safety
    /// Payload spans must cover tokens rows; pairs holds tokens * selected * 3
    /// FP32 words. All spans are disjoint, on this context and live until completion.
    pub unsafe fn moe_route_rows(&self, route: &mut crate::kernel_params::MoeRouteBatch, mut pairs: u64) -> Result<(), String> {
        let threads = route.tokens * if route.tokens >= 256 { 32 } else { 1 };
        unsafe { self.launch_flash_shared("moe_route_rows", threads.div_ceil(128), 128, 0,
            &mut [parameter(route), parameter(&mut pairs)]) }
    }
    /// Pair unequal canonical BF16 matrices (router + shared-expert gate) in
    /// the one launch each would take alone at 1..=8 tokens: mma1_rows gives
    /// every eight-row tile to its original matrix, so the split K ranges,
    /// hi/lo merge and Fp32 (unrounded) outputs are unchanged per output.
    /// # Safety
    /// Both weights have the canonical layout/context of fixed_bf16_projection.
    pub unsafe fn fixed_bf16_pair_rows(&self, weights: [u64;2], dims: [usize;3], input: &DeviceVector<'_>,
        first: &mut DeviceVector<'_>, second: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, tokens: usize) -> Result<(),String> {
        use super::batch::projection_params::FixedLinear;
        let [n1,n2,k]=dims;
        if !(1..=8).contains(&tokens) || n1==0 || n2==0 || n1.checked_add(n2).is_none_or(|n| n>=1024)
            || k<=512 || k%16!=0 || input.buffer.address%8!=0 || weights.iter().any(|w| w%8!=0) {
            return Err("paired BF16 rows geometry/alignment".into()); }
        let (inputs,o1,o2)=(tokens*k,tokens*n1,tokens*n2);
        for (v,len) in [(input,inputs),(&*first,o1),(&*second,o2)] { self.check_vector(v,len)?;
            if v.is_segmented() { return Err("paired BF16 rows requires contiguous operands".into()); } }
        let spans=[(input.buffer.address,inputs*4),(first.buffer.address,o1*4),(second.buffer.address,o2*4)];
        for i in 0..3 { for j in i+1..3 { FixedLinear::disjoint(spans[i].0,spans[i].1,spans[j].0,spans[j].1)?; } }
        for (w,n) in weights.into_iter().zip([n1,n2]) { for &(out,bytes) in &spans[1..] { FixedLinear::disjoint(w,n*k*2,out,bytes)?; } }
        scratch.invalidate_projection_scratch();
        let mut p=crate::kernel_params::Bf16Rows { weight:weights[0],output:first.buffer.address,rows:n1 as u32,stride:n1 as u32,
            weight2:weights[1],output2:second.buffer.address,rows2:n2 as u32,stride2:n2 as u32,
            input:input.buffer.address,cols:k as u32,tokens:tokens as u32,input_bf16:0,hc_silu:0,input_plane_words:0 };
        unsafe { self.launch_flash_rows("flash_bf16_mma1_rows",(n1.div_ceil(8)+n2.div_ceil(8)) as u32,tokens.div_ceil(8) as u32,
            128,0,&mut [parameter(&mut p)]) }
    }

    pub fn expert_table(&self, gate: Vec<Matrix<'a>>, up: Vec<Matrix<'a>>, down: Vec<Matrix<'a>>) -> Result<ExpertTable<'a>, String> {
        let experts = gate.len();
        if experts == 0 || experts > u32::MAX as usize || up.len() != experts || down.len() != experts {
            return Err("expert table count mismatch".into());
        }
        let (hidden, intermediate) = (gate[0].cols, gate[0].rows);
        let mut addresses = Vec::with_capacity(experts * 24);
        for expert in 0..experts {
            for (matrix, rows, cols) in [(&gate[expert], intermediate, hidden),
                (&up[expert], intermediate, hidden), (&down[expert], hidden, intermediate)] {
                if matrix.rows != rows || matrix.cols != cols || !matrix.packed || cols % 16 != 0 || rows % 8 != 0
                    || !ptr::eq(matrix.buffer._context, self.context) {
                    return Err("expert matrix shape, packing or context mismatch".into());
                }
                addresses.extend_from_slice(&matrix.buffer.address.to_ne_bytes());
            }
        }
        let pointers = Buffer::new(self.context, addresses.len())?;
        self.write_buffer(&pointers, 0, &addresses)?;
        let _matrices = gate.into_iter().chain(up).chain(down).collect();
        Ok(ExpertTable { _matrices, pointers, hidden, intermediate, experts })
    }

    pub fn moe_scratch(&self, table: &ExpertTable<'_>, selected: usize) -> Result<MoeScratch<'a>, String> {
        if selected == 0 || selected > table.experts || selected > 65535 { return Err("invalid selected expert count".into()); }
        let mut vectors = self.vector_arena(&[table.experts, selected, selected,
            selected.checked_mul(table.intermediate).ok_or("expert activation overflow")?,
            selected.checked_mul(table.hidden).ok_or("expert projection overflow")?, 1])?.into_iter();
        let mut scratch = MoeScratch { scores: vectors.next().unwrap(), indices: vectors.next().unwrap(),
            weights: vectors.next().unwrap(), activated: vectors.next().unwrap(),
            projected: vectors.next().unwrap(), completed: vectors.next().unwrap(), status: self.shared_vector(1)?, selected };
        self.write_vector(&mut scratch.completed, &[0.0])?;
        Ok(scratch)
    }

    /// Grouped input and activations use the checkpoint's BF16 precision.
    /// The final FP32 projection reuses input storage after gate/up finishes.
    pub fn moe_batch_scratch(&self, table: &ExpertTable<'_>, selected: usize, tokens: usize) -> Result<MoeBatchScratch<'a>, String> {
        if tokens == 0 || selected == 0 || selected > table.experts || table.experts > 65535
            || table.hidden % 128 != 0 || table.intermediate % 128 != 0
            || !ptr::eq(table.pointers._context, self.context) {
            return Err("invalid grouped expert shape or context".into());
        }
        let tokens = tokens.min(16384);
        let (h, m, e) = (table.hidden, table.intermediate, table.experts);
        let product = |a: usize, b: usize| a.checked_mul(b).ok_or_else(|| "grouped expert scratch overflow".to_string());
        let routed = product(tokens, selected)?;
        let alignment = if routed >= 64 * e { 128 } else { 64 };
        let padded = routed.checked_add(product(alignment - 1, e)?).ok_or("grouped expert padding overflow")?.div_ceil(alignment) * alignment;
        let mut metadata = self.vector_arena(&[product(tokens, e)?, routed, routed,
            product(e, tokens)?, e, e + 1, routed])?.into_iter();
        let projected = product(padded, h)?;
        let packed_input = projected / 2;
        let activated = product(padded, m)? / 2;
        if projected > u32::MAX as usize || activated / 4 > u32::MAX as usize
            || product(h, m)? / 4 > u32::MAX as usize {
            return Err("grouped expert scratch or weights exceed 32-bit kernel indices".into());
        }
        let mut workspace = self.vector_workspace(&[(0, packed_input),
            (projected, activated), (0, projected)])?.into_iter();
        Ok(MoeBatchScratch {
            scores: metadata.next().unwrap(), indices: metadata.next().unwrap(), weights: metadata.next().unwrap(),
            status: self.shared_vector(tokens)?,
            rows: metadata.next().unwrap(), counts: metadata.next().unwrap(), offsets: metadata.next().unwrap(), inverse: metadata.next().unwrap(),
            packed_input: workspace.next().unwrap(), packed_activated: workspace.next().unwrap(),
            projected: workspace.next().unwrap(),
            tokens, selected, hidden: h, intermediate: m, experts: e, bf16: false,
        })
    }

    pub(super) fn require_moe_kernels(&self) -> Result<(), String> {
        if self.moe.borrow().is_none() {
            let kernels = Kernels { router: self.module.function("moe_router")?,
                route: self.module.function("moe_route")?,
                gate_up: self.module.function("moe_gate_up")?, down: self.module.function("moe_down")?,
                merge: self.module.function("moe_merge")?,
                route_batch: self.module.function("moe_route_batch")?,
                route_group_decode: self.module.function("moe_route_group_decode")?, group: self.module.function("moe_group")?,
                group_offsets: self.module.function("moe_group_offsets")?, pack_grouped: self.module.function("moe_pack_grouped")?,
                pack_input: self.module.function("moe_pack_input")?,
                prefill_gate_up: self.module.function("moe_prefill_gate_up")?,
                prefill_gate_up_wide: self.module.function_with_shared("moe_prefill_gate_up_wide", 98304)?,
                decode_grouped_gate_up: self.module.function("moe_decode_grouped_gate_up")?,
                decode_grouped_down: self.module.function("moe_decode_grouped_down")?,
                grouped_prefill_gate_up: self.module.function("moe_grouped_prefill_gate_up")?,
                grouped_prefill_down: self.module.function("moe_grouped_prefill_down")?,
                prefill_down: self.module.function("moe_prefill_down")?, merge_batch: self.module.function("moe_merge_batch")?,
                merge_residual_rms: self.module.function("moe_merge_residual_rms")? };
            *self.moe.borrow_mut() = Some(kernels);
        }
        Ok(())
    }

    /// One token's routed MLP: router, top-k, selected gate/up, selected down,
    /// weighted sum. Every launch uses the same stream, with no host downloads.
    /// `first_layer` starts a forward: it restarts the row's routing status.
    pub fn moe_device(&self, router: &Matrix<'_>, table: &ExpertTable<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, scratch: &mut MoeScratch<'_>, renormalize: bool, first_layer: bool) -> Result<(), String> {
        if router.rows != table.experts || router.cols != table.hidden
            || !ptr::eq(table.pointers._context, self.context)
            || !ptr::eq(router.buffer._context, self.context) {
            return Err("router/expert shape or context mismatch".into());
        }
        self.check_vector(input, table.hidden)?;
        if router.packed && input.buffer.address % 8 != 0 { return Err("packed router input requires 8-byte alignment".into()); }
        self.check_vector(output, table.hidden)?;
        self.check_vector(&scratch.scores, table.experts)?;
        self.check_vector(&scratch.indices, scratch.selected)?;
        self.check_vector(&scratch.weights, scratch.selected)?;
        self.check_vector(&scratch.activated, scratch.selected * table.intermediate)?;
        self.check_vector(&scratch.projected, scratch.selected * table.hidden)?;
        self.check_vector(&scratch.completed, 1)?;
        self.check_vector(&scratch.status, 1)?;
        self.require_moe_kernels()?;
        let flags = renormalize as u32 | if first_layer { crate::kernel_params::ROUTE_FIRST_LAYER } else { 0 };
        let previous = self.profile_kind.replace(Some(18));
        let fused = router.packed && router.rows % 8 == 0 && router.stride == router.cols && router.row_offset == 0 && router.column_offset == 0;
        let routed = if fused {
            let mut p = crate::kernel_params::MoeRouter {
                matrix: router.buffer.address, input: input.buffer.address, scores: scratch.scores.buffer.address,
                indices: scratch.indices.buffer.address, weights: scratch.weights.buffer.address,
                completed: scratch.completed.buffer.address, status: scratch.status.buffer.address,
                experts: router.rows as u32, width: router.cols as u32, selected: scratch.selected as u32, renormalize: flags,
            };
            unsafe { self.enqueue_dims_shared(self.moe.borrow().as_ref().unwrap().router,
                p.experts / 8, 1, 1, 128, 132, &mut [parameter(&mut p)]) }
        } else { self.matvec_device(router, input, &mut scratch.scores) };
        self.profile_kind.set(previous);
        routed?;
        let kernels = self.moe.borrow();
        let kernels = kernels.as_ref().unwrap();
        let mut gate = crate::kernel_params::MoeSingleProjection {
            tables: table.pointers.address, indices: scratch.indices.buffer.address,
            input: input.buffer.address, output: scratch.activated.buffer.address,
            width: table.intermediate as u32, input_width: table.hidden as u32,
        };
        let mut down = crate::kernel_params::MoeSingleProjection {
            tables: table.pointers.address, indices: scratch.indices.buffer.address,
            input: scratch.activated.buffer.address, output: scratch.projected.buffer.address,
            width: table.hidden as u32, input_width: table.intermediate as u32,
        };
        let mut merge = crate::kernel_params::MoeMerge {
            input: scratch.projected.buffer.address, weights: scratch.weights.buffer.address,
            output: output.buffer.address, hidden: table.hidden as u32, selected: scratch.selected as u32,
        };
        unsafe {
            if !fused {
                let mut route = crate::kernel_params::MoeRoute {
                    scores: scratch.scores.buffer.address, indices: scratch.indices.buffer.address,
                    weights: scratch.weights.buffer.address, status: scratch.status.buffer.address,
                    experts: table.experts as u32, selected: scratch.selected as u32, renormalize: flags,
                };
                self.enqueue_dims(kernels.route, 1, 1, 1, 32, &mut [parameter(&mut route)])?;
            }
            self.generic_byte_probe(5, || Ok(if self.decode_words(scratch.status.buffer.address, 1)?[0] != 0 { 0 } else {
                scratch.selected * 3 * table.hidden * table.intermediate * 2 }))?;
            self.generic_read_bytes(|| (4, scratch.selected * (table.hidden * 4 + 4)));
            self.enqueue_dims(kernels.gate_up,
                table.intermediate.div_ceil(8) as u32,
                scratch.selected as u32, 1, 128, &mut [parameter(&mut gate)])?;
            self.enqueue_dims(kernels.down,
                table.hidden.div_ceil(32) as u32,
                scratch.selected as u32, 1, 128, &mut [parameter(&mut down)])?;
            self.enqueue(kernels.merge, table.hidden, &mut [parameter(&mut merge)])
        }
    }

    /// Whether the scratch's last forward rejected its row. Read after that
    /// forward's output; its request must then fail.
    pub fn moe_rejected(&self, scratch: &MoeScratch<'_>) -> Result<bool, String> {
        Ok(!self.moe_rejected_rows(&scratch.status, 1)?.is_empty())
    }
    /// Rows of the scratch's last batched forward that the router rejected. A
    /// forward wider than the scratch (one long prompt) shares words per chunk,
    /// so every row that could own a set word is reported.
    pub fn moe_batch_rejected(&self, scratch: &MoeBatchScratch<'_>, rows: usize) -> Result<Vec<usize>, String> {
        self.moe_rejected_rows(&scratch.status, rows)
    }
    fn moe_rejected_rows(&self, status: &DeviceVector<'_>, rows: usize) -> Result<Vec<usize>, String> {
        self.check_vector(status, status.len)?;
        if rows == 0 { return Ok(Vec::new()); }
        if self.context.pending.get() { self.synchronize()?; }
        // Managed storage is host-readable once its stream has completed.
        let words = unsafe { std::slice::from_raw_parts(status.buffer.address as *const u32, rows.min(status.len)) };
        let mut rejected: Vec<usize> = (0..rows).filter(|&row| words[row % words.len()] != 0).collect();
        if let Some(row) = forced_routing_rejection(rows, None) {
            if !rejected.contains(&row) { rejected.push(row); }
        }
        Ok(rejected)
    }
    pub fn moe_routing(&self, scratch: &MoeScratch<'_>) -> Result<Vec<(u32, f32)>, String> {
        let indices = self.read_vector_values(&scratch.indices)?;
        let weights = self.read_vector(&scratch.weights)?;
        Ok(indices.into_iter().zip(weights).map(|(id, weight)| (id.to_bits(), weight)).collect())
    }

    /// Read token-major selections from the most recent batched expert call.
    /// Reject calls spanning multiple scratch chunks rather than returning only the last.
    pub fn moe_batch_routing(&self, scratch: &MoeBatchScratch<'_>, tokens: usize) -> Result<Vec<(u32, f32)>, String> {
        if tokens == 0 || tokens > scratch.tokens { return Err("routing observation exceeds expert scratch".into()); }
        let len = tokens * scratch.selected;
        let indices = self.read_vector_values(&scratch.indices.slice(0..len)?)?;
        let weights = self.read_vector(&scratch.weights.slice(0..len)?)?;
        Ok(indices.into_iter().zip(weights).map(|(id, weight)| (id.to_bits(), weight)).collect())
    }

    /// Group routed rows, run one tensor-core launch per projection, then
    /// scatter in ascending expert order. Weights remain in their original arena.
    /// With residual_partials, output is the residual state and receives RMS partials.
    /// `first_layer` starts a forward: it restarts every row's routing status.
    pub fn moe_batch_device(&self, router: &Matrix<'_>, table: &ExpertTable<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, scratch: &mut MoeBatchScratch<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, decode: bool, renormalize: bool,
        first_layer: bool, mut residual_partials: Option<&mut DeviceVector<'_>>) -> Result<(), String> {
        let h = table.hidden;
        if input.len == 0 || input.len % h != 0 { return Err("grouped expert input shape".into()); }
        self.check_vector(output, input.len)?;
        let tokens = input.len / h;
        if tokens <= scratch.tokens {
            return self.moe_batch_chunk(router, table, input, output, scratch, weight_pack, input_pack, decode, renormalize, first_layer, residual_partials);
        }
        // Attention can span thousands of rows while expert scratch stays bounded.
        for start in (0..tokens).step_by(scratch.tokens) {
            let end = (start + scratch.tokens).min(tokens);
            let input = input.slice(start * h..end * h)?;
            let mut output = output.slice(start * h..end * h)?;
            let mut partials = residual_partials.as_deref_mut().map(|p| p.slice(start * h.div_ceil(128)..end * h.div_ceil(128))).transpose()?;
            self.moe_batch_chunk(router, table, &input, &mut output, scratch, weight_pack, input_pack, decode, renormalize,
                first_layer && start == 0, partials.as_mut())?;
        }
        Ok(())
    }

    fn moe_batch_chunk(&self, router: &Matrix<'_>, table: &ExpertTable<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, scratch: &mut MoeBatchScratch<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, decode: bool, renormalize: bool,
        first_layer: bool, residual_partials: Option<&mut DeviceVector<'_>>) -> Result<(), String> {
        // The canonical grouped tile rounds gate/up and SiLU separately.
        // Hybrid BF16 checkpoints require those boundaries at every length.
        let decode = decode || scratch.bf16;
        let (t, h, m, e) = (input.len / table.hidden, table.hidden, table.intermediate, table.experts);
        if router.rows != e || router.cols != h || (scratch.hidden, scratch.intermediate, scratch.experts) != (h, m, e)
            || decode && (h % 256 != 0 || m % 256 != 0)
            || !ptr::eq(table.pointers._context, self.context) {
            return Err("grouped router/expert shape or context mismatch".into());
        }
        self.check_vector(input, t * h)?;
        self.check_vector(output, t * h)?;
        if let Some(p) = &residual_partials { self.check_vector(p, t * h.div_ceil(128))?; }
        scratch.scores.resize_within_capacity(t * e)?;
        for vector in [&scratch.scores, &scratch.indices, &scratch.weights, &scratch.status, &scratch.rows, &scratch.counts,
            &scratch.offsets, &scratch.inverse, &scratch.packed_input, &scratch.packed_activated,
            &scratch.projected] { self.check_vector(vector, vector.len)?; }
        self.require_moe_kernels()?;
        let previous = self.profile_kind.replace(Some(18));
        let routed = self.matmul_device(router, input, &mut scratch.scores, weight_pack, input_pack, t, true);
        self.profile_kind.set(previous);
        routed?;
        let kernels = self.moe.borrow();
        let kernels = kernels.as_ref().unwrap();
        let (scores, mut indices, mut weights) = (scratch.scores.buffer.address, scratch.indices.buffer.address, scratch.weights.buffer.address);
        let status = scratch.status.buffer.address;
        let (mut rows, mut counts, mut offsets, mut inverse) = (scratch.rows.buffer.address, scratch.counts.buffer.address,
            scratch.offsets.buffer.address, scratch.inverse.buffer.address);
        let (mut x, mut packed, mut activated) = (input.buffer.address, scratch.packed_input.buffer.address, scratch.packed_activated.buffer.address);
        let (mut projected, mut y) = (scratch.projected.buffer.address, output.buffer.address);
        let (mut pointers, mut tokens, mut hidden, mut intermediate, mut experts, mut selected, normalize) =
            (table.pointers.address, t as u32, h as u32, m as u32, e as u32, scratch.selected as u32,
                renormalize as u32 | ((scratch.bf16 as u32) << 1)
                    | if first_layer { crate::kernel_params::ROUTE_FIRST_LAYER } else { 0 });
        let wide = !decode && t * scratch.selected >= 64 * e;
        if wide && input_pack.len < t * h / 2 { return Err("expert input packing capacity".into()); }
        let grouped_prefill = decode && t >= 256;
        let alignment = if grouped_prefill { 64u32 } else if decode { 16u32 } else if wide { 128 } else { 64 };
        let padded = t * scratch.selected + (alignment as usize - 1) * e;
        let tile_rows = if decode { 16 } else { 32 };
        let tiles = padded.div_ceil(tile_rows) as u32;
        // Scores are dead after routing. Each compact tile contains at least
        // one assignment, so the score arena also bounds the device work list.
        let tile_experts = if decode { scores } else { 0 };
        let compact_tiles = (t * scratch.selected).min(e * t.div_ceil(16));
        unsafe {
            if wide {
                let mut destination = input_pack.buffer.address;
                let mut pairs = (t * h / 2) as u32;
                self.enqueue(kernels.pack_input, pairs as usize, &mut [parameter(&mut x), parameter(&mut destination), parameter(&mut pairs)])?;
                x = destination;
            }
            if decode && t <= 32 {
                let mut route = crate::kernel_params::MoeRouteGroup { scores, indices, weights, status,
                    rows, counts, offsets, inverse, experts, selected, tokens, renormalize: normalize };
                self.enqueue_dims(kernels.route_group_decode, 1, 1, 1, 1024,
                    &mut [parameter(&mut route)])?;
            } else {
                let mut route = crate::kernel_params::MoeRouteBatch { scores, indices, weights, status, experts, selected, tokens, renormalize: normalize };
                self.enqueue(kernels.route_batch, if t >= 256 { t * 32 } else { t }, &mut [parameter(&mut route)])?;
                self.enqueue(kernels.group, e * 32, &mut [parameter(&mut indices), parameter(&mut rows), parameter(&mut counts),
                    parameter(&mut experts), parameter(&mut selected), parameter(&mut tokens)])?;
                let mut group = crate::kernel_params::MoeGroupOffsets { rows, counts, offsets, inverse,
                    tile_experts, experts, tokens, alignment };
                self.enqueue_dims(kernels.group_offsets, 1, 1, 1, 256, &mut [parameter(&mut group)])?;
            }
            self.generic_byte_probe(5, || {
                let live = self.decode_words(counts, e)?;
                if live.iter().any(|&n| n as usize > t) { return Err("generic expert count outside plan".into()); }
                Ok(live.iter().filter(|&&n| n != 0).count() * 3 * h * m * 2)
            })?;
            self.generic_read_bytes(|| (4, t * scratch.selected * (h * 4 + 4)));
            let mut pack = crate::kernel_params::MoePack { input: x, rows, counts, offsets, tile_experts, packed,
                hidden, selected, tokens, experts, input_packed: wide as u32, tile_rows: tile_rows as u32 };
            let (pack_x, pack_y) = if decode { ((compact_tiles * h.div_ceil(128)).min(256) as u32, 1) }
                else { (tiles, h.div_ceil(128) as u32) };
            self.enqueue_dims(kernels.pack_grouped, pack_x, pack_y, 1, 128,
                &mut [parameter(&mut pack)])?;
            if decode {
                let mut projection = crate::kernel_params::MoeDecode { tables: pointers, counts, offsets,
                    tile_experts, packed, output: activated, width: intermediate, input_width: hidden, experts,
                    bf16: scratch.bf16 as u32 };
                self.enqueue_dims(if grouped_prefill { kernels.grouped_prefill_gate_up } else { kernels.decode_grouped_gate_up }, (compact_tiles * m.div_ceil(64)).min(384) as u32,
                    1, 1, 256, &mut [parameter(&mut projection)])?;
                projection.packed = activated;
                projection.output = projected;
                projection.width = hidden;
                projection.input_width = intermediate;
                self.enqueue_dims(if grouped_prefill { kernels.grouped_prefill_down } else { kernels.decode_grouped_down }, (compact_tiles * h.div_ceil(64)).min(384) as u32,
                    1, 1, 256, &mut [parameter(&mut projection)])?;
            } else {
                let (gate_up, gate_x, threads) = if wide {
                    (kernels.prefill_gate_up_wide, padded.div_ceil(128) as u32, 512)
                } else { (kernels.prefill_gate_up, tiles, 128) };
                self.enqueue_dims_shared(gate_up, gate_x, m.div_ceil(128) as u32, 1, threads, if wide { 98304 } else { 0 },
                    &mut [parameter(&mut pointers), parameter(&mut counts), parameter(&mut offsets), parameter(&mut packed),
                        parameter(&mut activated), parameter(&mut intermediate), parameter(&mut hidden), parameter(&mut experts)])?;
                let (down_x, down_y, shared) = if wide { (padded.div_ceil(128), h.div_ceil(128), 16384) }
                    else { (padded.div_ceil(64), h.div_ceil(256), 32768) };
                let mut wide = wide as u32;
                self.enqueue_dims_shared(kernels.prefill_down, down_x as u32, down_y as u32, 1, 256, shared,
                    &mut [parameter(&mut pointers), parameter(&mut counts), parameter(&mut offsets), parameter(&mut activated),
                        parameter(&mut projected), parameter(&mut hidden), parameter(&mut intermediate), parameter(&mut experts), parameter(&mut wide)])?;
            }
            if let Some(partials) = residual_partials {
                let mut partials = partials.buffer.address;
                self.enqueue_dims(kernels.merge_residual_rms, (h.div_ceil(128) * 32).div_ceil(512) as u32, tokens, 1, 512,
                    &mut [parameter(&mut projected), parameter(&mut weights), parameter(&mut inverse), parameter(&mut y),
                        parameter(&mut partials), parameter(&mut hidden), parameter(&mut selected), parameter(&mut tokens)])
            } else {
                let mut merge = crate::kernel_params::MoeMergeBatch { projected, weights, inverse,
                    output: y, hidden, selected, tokens, bf16: scratch.bf16 as u32 };
                self.enqueue(kernels.merge_batch, t * h, &mut [parameter(&mut merge)])
            }
        }
    }
}
