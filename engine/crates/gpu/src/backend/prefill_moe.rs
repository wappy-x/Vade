//! Safe host launch wrappers for the bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE kernel
//! (`PREFILL_PACKED_TC`, kind 121).
//!
//! Every extent, alignment, context ownership, and non-overlap invariant is verified via `crate::prefill_moe::check_*`
//! before recording any command into the resident launch queue.
use super::*;
use crate::fast_ops::CAPPED_GATE_TABLE_WORDS;
use crate::packed_product::{cols_word, layout};
use crate::prefill_moe::{
    self, PREFILL_HALF_IN, PREFILL_OP_FOLD, PREFILL_OP_GEMM, PREFILL_OP_PACK, PREFILL_OP_WORD,
    PREFILL_OUT_BF16, PREFILL_OUT_BF16_GATED, PREFILL_OUT_FP32, PREFILL_THREADS, bf16_row_bytes,
    fold_blocks, gemm_blocks, pack_blocks,
};
use crate::resident_launch::{Command, Kind};
use crate::tc_rows::{Span, moe_max_tiles, moe_tile_meta_words};

impl<'a> Backend<'a> {
    /// Vectorized FP32 -> packed BF16 activation packer (`PREFILL_OP_PACK`, bit-exact with `TC_MODE_PACK`).
    pub fn prefill_pack_bf16(&self, rows: usize, cols: usize, input: &DeviceVector<'_>) -> Result<DeviceVector<'a>, String> {
        if self.arm_capture() { return Err("prefill moe pack: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let elements = rows.checked_mul(cols).ok_or("prefill moe pack: element overflow")?;
        if !elements.is_multiple_of(8) { return Err("prefill moe pack: elements must be a multiple of 8".into()); }
        let mut packed = self.vector(elements / 2)?;
        self.skip_fill(&mut packed)?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        prefill_moe::check_pack(elements, span(input), span(&packed))?;
        self.check_vector(input, elements)?;
        if input.is_segmented() || packed.is_segmented() {
            return Err("prefill moe pack: operands must be contiguous".into());
        }
        self.require_arm_kernels(&[Kind::PrefillPackedTc.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PrefillPackedTc);
        command.word(0, input.device_address());
        command.word(1, packed.device_address());
        command.word(2, elements as u64);
        command.word(PREFILL_OP_WORD, PREFILL_OP_PACK);
        let grid = [pack_blocks(elements), 1, 1, PREFILL_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PrefillPackedTc, command, grid)? {
            return Err("prefill moe pack: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok(packed)
    }

    /// Vectorized BF16 `RouteFold` (`PREFILL_OP_FOLD`, bit-exact with `fast_route_fold_half_into`).
    pub fn prefill_route_fold_half_into(
        &self, rows: usize, picks: usize, dim: usize,
        weights: &DeviceVector<'_>, downs: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("prefill moe fold: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        prefill_moe::check_fold(rows, picks, dim, span(weights), span(downs), span(output))?;
        self.check_vector(weights, rows * picks)?;
        self.check_vector(downs, rows * picks * dim / 2)?;
        self.check_vector(output, rows * dim)?;
        if weights.is_segmented() || downs.is_segmented() || output.is_segmented() || !output.writable_from(0) {
            return Err("prefill moe fold: contiguous inputs and private contiguous output required".into());
        }
        self.require_arm_kernels(&[Kind::PrefillPackedTc.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PrefillPackedTc);
        command.word(0, weights.device_address());
        command.word(1, downs.device_address());
        command.word(2, output.device_address());
        command.word(3, rows as u64);
        command.word(4, picks as u64);
        command.word(5, dim as u64);
        command.word(PREFILL_OP_WORD, PREFILL_OP_FOLD);
        let grid = [fold_blocks(rows), 1, 1, PREFILL_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PrefillPackedTc, command, grid)? {
            return Err("prefill moe fold: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Standalone bit-exact W4A16/W8A16 prefill MoE GEMM (`PREFILL_OP_GEMM`).
    /// Packs FP32 `input` to BF16 via `PREFILL_OP_PACK` when `half_in` is false.
    #[allow(clippy::too_many_arguments, reason = "shape, format, route, caps and device operands")]
    pub fn project_prefill_packed_moe_into(
        &self, rows: usize, cols: usize, wide: bool, pairs: bool, format: u64,
        stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>, half_in: bool,
        output: &mut DeviceVector<'_>, down: bool, caps: Option<[f32; 4]>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("prefill moe gemm: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let [in_rows, out_slots, tile_meta] = route.route_tables();
        let all = route.picks();
        let experts = route.experts();
        let (in_table, out_table) = if down { (None, Some(out_slots)) } else { (Some(in_rows), None) };
        let in_count = if down {
            all
        } else if half_in {
            let row_words = cols / 2;
            if row_words == 0 || !input.len().is_multiple_of(row_words) {
                return Err("prefill moe gemm: bf16 input size mismatch".into());
            }
            input.len() / row_words
        } else {
            if !input.len().is_multiple_of(cols) { return Err("prefill moe gemm: fp32 input size mismatch".into()); }
            input.len() / cols
        };
        let out_count = all;

        let gate_words = 2 * CAPPED_GATE_TABLE_WORDS;
        let tables = caps.map(|_| self.vector(gate_words)).transpose()?;
        let encoded = if half_in { None } else { Some(self.prefill_pack_bf16(in_count, cols, input)?) };
        let act_in = encoded.as_ref().unwrap_or(input);

        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        prefill_moe::check_gemm(
            rows, cols, wide, pairs, format, all, experts,
            span(stack), span(tile_meta), span(act_in), in_count, in_table.map(span),
            span(output), out_count, out_table.map(span), tables.as_ref().map(span),
        )?;
        let unit_bytes = layout(rows, cols, wide).bytes;
        self.check_vector(stack, experts * (unit_bytes / 4))?;
        self.check_vector(tile_meta, moe_tile_meta_words(all, experts))?;
        let act_bytes = bf16_row_bytes(in_count, cols).ok_or("prefill moe gemm: act overflow")?;
        self.check_vector(act_in, act_bytes / 4)?;
        let width = if caps.is_some() { rows / 2 } else { rows };
        let out_words = match format {
            PREFILL_OUT_FP32 => out_count * width,
            PREFILL_OUT_BF16_GATED | PREFILL_OUT_BF16 => out_count * width / 2,
            _ => return Err("prefill moe gemm: unknown output format".into()),
        };
        self.check_vector(output, out_words)?;
        if let Some(t) = in_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("prefill moe gemm: in_table segmented".into()); } }
        if let Some(t) = out_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("prefill moe gemm: out_table segmented".into()); } }
        if stack.is_segmented() || tile_meta.is_segmented() || act_in.is_segmented() {
            return Err("prefill moe gemm: operand segmented".into());
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("prefill moe gemm: output requires private contiguous storage".into());
        }
        if let Some(t) = &tables { self.check_vector(t, gate_words)?; }
        self.require_arm_kernels(&[Kind::PrefillPackedTc.kernel()])?;
        self.resident_completion()?;
        if let (Some(c), Some(t)) = (caps, &tables) {
            self.record_capped_gate_tables(c, t)?;
        }
        let max_tiles = moe_max_tiles(all, experts);
        let mut command = Command::new(Kind::PrefillPackedTc);
        let payload = [
            stack.device_address(), act_in.device_address(), output.device_address(),
            rows as u64, u64::from(cols_word(cols, wide, pairs)) | (format << 32),
            all as u64, width as u64, unit_bytes as u64,
            in_table.map_or(0, DeviceVector::device_address), in_count as u64,
            out_table.map_or(0, DeviceVector::device_address), out_count as u64,
            tile_meta.device_address(), max_tiles as u64, experts as u64,
            tables.as_ref().map_or(0, DeviceVector::device_address),
            0, PREFILL_HALF_IN, PREFILL_OP_GEMM,
        ];
        for (i, w) in payload.into_iter().enumerate() { command.word(i, w); }
        let grid = [gemm_blocks(rows, max_tiles), 1, 1, PREFILL_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PrefillPackedTc, command, grid)? {
            return Err("prefill moe gemm: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Full bit-exact W4A16/W8A16 prefill expert layer (bit-identical to `PLAIN_PACKED_TC_ROWS` + `RouteFold`):
    /// 1. Optional `PREFILL_OP_PACK` on `input` (`[tokens, g_cols]` FP32 -> packed BF16) when `half_in` is false
    /// 2. `FAST_CAPPED_GATE` mode 2 exact coefficient table build + `PREFILL_OP_GEMM` on `GateUp` -> BF16 `active`
    /// 3. `PREFILL_OP_GEMM` on `Down` reading BF16 `active` -> BF16 `downs`
    /// 4. `PREFILL_OP_FOLD` (or `fast_route_fold_half_into` when `d_rows % 8 != 0`) -> `[tokens, d_rows]` FP32 `output`.
    #[allow(clippy::too_many_arguments, reason = "gate/down shapes, precision flags, route, caps, weights and operands")]
    pub fn project_prefill_packed_experts_into(
        &self,
        g_rows: usize, g_cols: usize, g_wide: bool, g_stack: &DeviceVector<'_>,
        d_rows: usize, d_cols: usize, d_wide: bool, d_stack: &DeviceVector<'_>,
        route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>, caps: [f32; 4],
        weights: &DeviceVector<'_>, picks: usize, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.project_prefill_packed_experts_form_into(
            g_rows, g_cols, g_wide, g_stack, d_rows, d_cols, d_wide, d_stack,
            route, input, false, caps, weights, picks, output,
        )
    }

    /// Form of `project_prefill_packed_experts_into` accepting already-packed BF16 input when `half_in` is true.
    #[allow(clippy::too_many_arguments, reason = "gate/down shapes, precision flags, route, input form, caps, weights and operands")]
    pub fn project_prefill_packed_experts_form_into(
        &self,
        g_rows: usize, g_cols: usize, g_wide: bool, g_stack: &DeviceVector<'_>,
        d_rows: usize, d_cols: usize, d_wide: bool, d_stack: &DeviceVector<'_>,
        route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>, half_in: bool, caps: [f32; 4],
        weights: &DeviceVector<'_>, picks: usize, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        let all = route.picks();
        if picks == 0 || !all.is_multiple_of(picks) { return Err("prefill moe experts: picks mismatch".into()); }
        let tokens = all / picks;
        let inner = g_rows / 2;
        if d_cols != inner || !inner.is_multiple_of(2) || !d_rows.is_multiple_of(2) {
            return Err("prefill moe experts: mismatched gate/down inner width or odd rows".into());
        }
        let mut active = self.vector(all * inner / 2)?;
        self.skip_fill(&mut active)?;
        self.project_prefill_packed_moe_into(
            g_rows, g_cols, g_wide, true, PREFILL_OUT_BF16_GATED,
            g_stack, route, input, half_in, &mut active, false, Some(caps),
        )?;
        let mut downs = self.vector(all * d_rows / 2)?;
        self.skip_fill(&mut downs)?;
        self.project_prefill_packed_moe_into(
            d_rows, d_cols, d_wide, false, PREFILL_OUT_BF16,
            d_stack, route, &active, true, &mut downs, true, None,
        )?;
        if d_rows.is_multiple_of(8) && ((downs.device_address() | output.device_address()) & 15) == 0 {
            self.prefill_route_fold_half_into(tokens, picks, d_rows, weights, &downs, output)
        } else {
            self.fast_route_fold_half_into(tokens, picks, d_rows, weights, &downs, output)
        }
    }
}
