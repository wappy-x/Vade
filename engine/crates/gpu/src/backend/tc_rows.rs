//! Host launch wrappers for `PLAIN_TC_ROWS` (kind 103) and `MOE_GROUP_ROUTE` (kind 104) in `kernels-inkling`.
//!
//! Safe code only: every extent, alignment, context ownership, and non-overlap invariant is verified
//! before recording any command into the resident launch queue.
use super::*;
use crate::resident_launch::{Command, Kind};
use crate::tc_rows::{self, Span, SplitOut};

/// Private routing tables produced together by the router. Its tiles partition the grouped rows
/// and its scatter table maps them to distinct pick slots.
pub struct MoeRoutePlan<'a> {
    rows: usize,
    all: usize,
    experts: usize,
    in_rows: DeviceVector<'a>,
    out_slots: DeviceVector<'a>,
    tile_meta: DeviceVector<'a>,
}

impl MoeRoutePlan<'_> {
    pub fn picks(&self) -> usize { self.all }
    pub fn experts(&self) -> usize { self.experts }
    /// The device tables of the plan: input row a place, pick a place, and the route tiles (`tile_meta`).
    pub fn route_tables(&self) -> [&DeviceVector<'_>; 3] { [&self.in_rows, &self.out_slots, &self.tile_meta] }

    fn tables(&self, down: bool) -> (usize, Option<&DeviceVector<'_>>, Option<&DeviceVector<'_>>) {
        if down { (self.all, None, Some(&self.out_slots)) } else { (self.rows, Some(&self.in_rows), None) }
    }
}

/// The clips of the exact one-row attention (`flash_window_one_into`): the scales of the query and of the relative
/// state (none: no scale), the position factor (none: no factor) and the two thresholds.
#[derive(Clone, Copy)]
pub struct OneClips<'v, 'd> {
    pub query_scale: Option<&'v DeviceVector<'d>>, pub moved_scale: Option<&'v DeviceVector<'d>>,
    pub position: Option<&'v DeviceVector<'d>>, pub query_bound: f32, pub moved_bound: f32,
    pub written_keys: Option<&'v DeviceVector<'d>>, pub written_values: Option<&'v DeviceVector<'d>>,
}

impl<'a> Backend<'a> {
    /// The caller has validated both input extents and alignment. Pack into private storage
    /// on the same recorded stream as the product. Buffer retirement follows that stream.
    fn pack_tc_input(&self, input: &DeviceVector<'_>, rows: usize, cols: usize) -> Result<DeviceVector<'a>, String> {
        let count = rows.checked_mul(cols).ok_or("tc pack: input extent overflow")?;
        let packed = self.vector(count / 2)?;
        let mut command = Command::new(Kind::PlainTcRows);
        command.word(1, input.device_address());
        command.word(2, packed.device_address());
        command.word(3, count as u64);
        command.word(7, tc_rows::TC_MODE_PACK);
        let blocks = (count / 8).div_ceil(tc_rows::TC_THREADS);
        if !self.record_resident_block(Kind::PlainTcRows, command, [blocks as u32, 1, 1, tc_rows::TC_THREADS as u32, 0])? {
            return Err("tc pack: launches are not recorded now".into());
        }
        Ok(packed)
    }
    /// The kind, mode, and grid of a dense product: direct-FP32 `PLAIN_TC_ROWS` (zero pack) for FP32 narrow
    /// (`rows <= 128`), short-K (`cols == 64`, `GateOuter`), and K/V prefill (`512 <= rows <= 1280`), 1-warp narrow
    /// packed-BF16 (`TC_FLAG_HALF_IN`) for `half_in` narrow `inputs <= 256`, `TC_MODE_NARROW_HALF` for `half_in`
    /// narrow `inputs >= 257`, `PLAIN_TC_BIG` for large prefill products, else packed `PLAIN_TC_ROWS` with
    /// wave-matched dynamic SMEM (sized for lossless Vade-BF11-BP `bf16_panels` when registered).
    fn tc_dense_launch(rows: usize, cols: usize, inputs: usize, split: bool, pair: bool, half_in: bool, panel_slices: Option<usize>) -> (Kind, u64, [u32; 5], bool, u64) {
        if !half_in && tc_rows::tc_fp32_direct_wanted(rows, cols, inputs, split, pair) {
            let (threads, blocks, smem) = tc_rows::tc_dense_spec(rows, cols, inputs, true);
            (Kind::PlainTcRows, tc_rows::TC_MODE_DENSE_FP32, [blocks, 1, 1, threads as u32, smem as u32], true, 0)
        } else if half_in && !split && tc_rows::tc_narrow_1w_wanted(rows, inputs) {
            let (threads, blocks, smem) = tc_rows::tc_dense_spec(rows, cols, inputs, true);
            (Kind::PlainTcRows, tc_rows::TC_MODE_DENSE_FP32, [blocks, 1, 1, threads as u32, smem as u32], false, tc_rows::TC_FLAG_HALF_IN)
        } else if half_in && !split && tc_rows::tc_narrow_wanted(rows, inputs) {
            (Kind::PlainTcBig, tc_rows::TC_MODE_NARROW_HALF, [tc_rows::tc_narrow_half_blocks(inputs), 1, 1, tc_rows::TC_THREADS as u32, 0], false, 0)
        } else if !pair && tc_rows::tc_big_wanted(rows, inputs) {
            (Kind::PlainTcBig, tc_rows::TC_MODE_DENSE, [tc_rows::tc_big_blocks(rows, inputs), 1, 1, tc_rows::TC_THREADS as u32, 0], false, 0)
        } else {
            let (threads, blocks, raw_smem) = tc_rows::tc_dense_spec(rows, cols, inputs, false);
            let smem = match panel_slices {
                Some(slices) => tc_rows::tc_dense_panel_smem(rows, cols, inputs, slices),
                None => raw_smem,
            };
            (Kind::PlainTcRows, tc_rows::TC_MODE_DENSE, [blocks, 1, 1, threads as u32, smem as u32], false, 0)
        }
    }
    /// Attach registered lossless Vade-BF11-BP `bf16_panels` `(base, first_band | (slices << 32))` to Mode 0 commands.
    fn attach_tc_panels(command: &mut Command, panels: Option<(u64, u64, usize)>, kind: Kind, mode: u64) {
        if kind == Kind::PlainTcRows && mode == tc_rows::TC_MODE_DENSE {
            if let Some((base, meta, _)) = panels {
                command.word(tc_rows::TC_PANEL_WORD, base);
                command.word(tc_rows::TC_PANEL_WORD + 1, meta);
            }
        }
    }
    fn tc_panels_for(&self, weight: &DeviceVector<'_>, rows: usize, cols: usize, inputs: usize) -> Option<(u64, u64, usize)> {
        // Small-row projections (`rows <= 1280`, e.g. K/V 1024 and K+KTap 1056) are L2-resident and faster on raw
        // `LDG.128`; larger matrices (`O_proj` 2304, `QKV` 6552/8600, `LM Head` 200k) stream Vade-BF11-BP panels.
        if rows <= tc_rows::TC_PANEL_ROWS_MIN || inputs > tc_rows::TC_SMALL_BM { return None; }
        let chunks = cols / crate::bf16_panels::CHUNK_COLS;
        let (base, first_band, slices) = self.panel_at(weight, cols)?;
        if base == 0 || (base & 15) != 0 || !matches!(slices, 1 | 4 | 8) || !chunks.is_multiple_of(slices) {
            return None;
        }
        if tc_rows::tc_coop_w_wanted(rows, cols, inputs) && (slices < 4 || !chunks.is_multiple_of(4)) {
            return None;
        }
        Some((base, (first_band as u64) | ((slices as u64) << 32), slices))
    }
    /// True when `project_tc_rows_into` can run this dense BF16 matrix with `inputs` rows now.
    pub fn tc_rows_ready(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        tc_rows::tc_shape_admitted(rows, cols, inputs) && !input.is_segmented() && !weight.is_segmented()
            && input.len() == inputs * cols && weight.len() == rows * cols / 2
            && (input.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }

    /// True when `project_tc_rows_half_in_into` / `split_half_in` can run on packed BF16 `input` (`inputs * cols / 2` words).
    pub fn tc_rows_half_ready(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        tc_rows::tc_shape_admitted(rows, cols, inputs) && !input.is_segmented() && !weight.is_segmented()
            && input.len() == inputs * cols / 2 && weight.len() == rows * cols / 2
            && (input.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }

    /// True when `project_tc_rows_pair_into` / `project_tc_rows_pair_form_into` can run over two `[inputs, cols/2]` inputs now.
    #[allow(clippy::too_many_arguments, reason = "shape and three operands")]
    pub fn tc_rows_pair_ready(&self, rows: usize, cols: usize, inputs: usize, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        tc_rows::tc_shape_admitted(rows, cols, inputs) && (cols / 2).is_multiple_of(tc_rows::TC_BK)
            && !input1.is_segmented() && !input2.is_segmented() && !weight.is_segmented()
            && input1.len() == inputs * (cols / 2) && input2.len() == inputs * (cols / 2) && weight.len() == rows * cols / 2
            && (input1.device_address() | input2.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }

    /// True when `project_tc_rows_pair_half_in_into` can run over two packed BF16 `[inputs, cols/2]` inputs (`inputs * cols / 4` words each).
    #[allow(clippy::too_many_arguments, reason = "shape and three operands")]
    pub fn tc_rows_pair_half_ready(&self, rows: usize, cols: usize, inputs: usize, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        tc_rows::tc_shape_admitted(rows, cols, inputs) && (cols / 2).is_multiple_of(tc_rows::TC_BK)
            && !input1.is_segmented() && !input2.is_segmented() && !weight.is_segmented()
            && input1.len() == inputs * (cols / 4) && input2.len() == inputs * (cols / 4) && weight.len() == rows * cols / 2
            && (input1.device_address() | input2.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }

    /// True when `project_tc_rows_group_pair_into` can run 1..=3 narrow dual-input matrices of one shape now.
    pub fn tc_rows_group_pair_ready(&self, rows: usize, cols: usize, inputs: usize, items: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>)]) -> bool {
        if items.is_empty() || items.len() > tc_rows::TC_GROUP_MAX || !tc_rows::tc_narrow_1w_wanted(rows, inputs) { return false; }
        items.iter().all(|&(w, x1, x2)| self.tc_rows_pair_ready(rows, cols, inputs, x1, x2, w))
    }

    /// Dual-input tensor-core BF16 product: multiplies `weight` (`rows x cols` BF16) with the row-wise
    /// concatenation `[input1[t], input2[t]]` (`inputs x cols/2` each) without copying or swapping inputs.
    #[allow(clippy::too_many_arguments, reason = "shape and four operands")]
    pub fn project_tc_rows_pair_into(&self, rows: usize, cols: usize, inputs: usize, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_pair_inner(rows, cols, inputs, false, false, input1, input2, weight, output)
    }

    /// Dual-input tensor-core BF16 product with optional fused BF16 (`rn`) output rounding.
    #[allow(clippy::too_many_arguments, reason = "shape, rounding flag and four operands")]
    pub fn project_tc_rows_pair_form_into(&self, rows: usize, cols: usize, inputs: usize, round_out: bool, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_pair_inner(rows, cols, inputs, false, round_out, input1, input2, weight, output)
    }

    /// Dual-input tensor-core BF16 product directly from two packed BF16 half-width inputs (`inputs * cols / 4` words each).
    #[allow(clippy::too_many_arguments, reason = "shape and four operands")]
    pub fn project_tc_rows_pair_half_in_into(&self, rows: usize, cols: usize, inputs: usize, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_pair_inner(rows, cols, inputs, true, false, input1, input2, weight, output)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, input/rounding flags and four operands")]
    fn project_tc_rows_pair_inner(&self, rows: usize, cols: usize, inputs: usize, half_in: bool, round_out: bool, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if self.arm_capture() { return Err("tc rows pair: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        tc_rows::check_tc_pair_form(rows, cols, inputs, half_in, span(weight), span(input1), span(input2), span(output))?;
        let half = cols / 2;
        let in_words = if half_in { inputs * (half / 2) } else { inputs * half };
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input1, in_words)?;
        self.check_vector(input2, in_words)?;
        self.check_vector(output, inputs * rows)?;
        if weight.is_segmented() || input1.is_segmented() || input2.is_segmented() { return Err("tc rows pair: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("tc rows pair: output requires private contiguous storage".into()); }
        let panels = self.tc_panels_for(weight, rows, cols, inputs);
        let (kind, mode, grid, direct, extra_flags) = Self::tc_dense_launch(rows, cols, inputs, false, true, half_in, panels.map(|p| p.2));
        if round_out && !direct { return Err("tc rows pair: fused rounding requires direct-FP32 narrow shape".into()); }
        self.require_arm_kernels(&[Kind::PlainTcRows.kernel(), kind.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(kind);
        let packed = if direct || half_in { None } else { Some((self.pack_tc_input(input1, inputs, half)?, self.pack_tc_input(input2, inputs, half)?)) };
        let (first, second) = packed.as_ref().map_or((input1.device_address(), input2.device_address()), |(p1, p2)| (p1.device_address(), p2.device_address()));
        let flags = extra_flags | if round_out { tc_rows::TC_FLAG_ROUND_OUT } else { 0 };
        let words = [
            weight.device_address(), first, output.device_address(),
            rows as u64, cols as u64, inputs as u64, rows as u64,
            mode, 0, inputs as u64, 0, inputs as u64, second, flags, 0, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        Self::attach_tc_panels(&mut command, panels, kind, mode);
        if !self.record_resident_block(kind, command, grid)? {
            return Err("tc rows pair: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Grouped dual-input narrow direct-FP32 product for 1..=3 matrices of one shape (`GateDown` Q/K/V)
    /// with optional fused BF16 (`rn`) output rounding, writing `[groups][inputs][rows]`.
    #[allow(clippy::too_many_arguments, reason = "shape, rounding flag, grouped operands and output")]
    pub fn project_tc_rows_group_pair_into(&self, rows: usize, cols: usize, inputs: usize, round_out: bool, items: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>)], output: &mut DeviceVector<'_>) -> Result<(), String> {
        if self.arm_capture() { return Err("tc rows group pair: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let spans: Vec<(Span, Span, Span)> = items.iter().map(|&(w, x1, x2)| (span(w), span(x1), span(x2))).collect();
        tc_rows::check_tc_group_pair(rows, cols, inputs, &spans, span(output))?;
        let half = cols / 2;
        self.check_vector(output, items.len() * inputs * rows)?;
        if output.is_segmented() || !output.writable_from(0) { return Err("tc rows group pair: output requires private contiguous storage".into()); }
        for &(w, x1, x2) in items {
            self.check_vector(w, rows * cols / 2)?;
            self.check_vector(x1, inputs * half)?;
            self.check_vector(x2, inputs * half)?;
            if w.is_segmented() || x1.is_segmented() || x2.is_segmented() {
                return Err("tc rows group pair: operand must be contiguous".into());
            }
        }
        let (threads, blocks, smem) = tc_rows::tc_dense_spec(rows, cols, inputs, true);
        self.require_arm_kernels(&[Kind::PlainTcRows.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PlainTcRows);
        let flags = tc_rows::TC_FLAG_GROUP_PAIR | if round_out { tc_rows::TC_FLAG_ROUND_OUT } else { 0 };
        let (w0, x1_0, x2_0) = items[0];
        let words = [
            w0.device_address(), x1_0.device_address(), output.device_address(),
            rows as u64, cols as u64, inputs as u64, rows as u64,
            tc_rows::TC_MODE_DENSE_FP32, 0, inputs as u64, 0, inputs as u64, x2_0.device_address(), flags, items.len() as u64, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        for (g, &(w, x1, x2)) in items.iter().enumerate().skip(1) {
            let base = tc_rows::TC_GROUP_WORD + (g - 1) * 3;
            command.word(base, w.device_address());
            command.word(base + 1, x1.device_address());
            command.word(base + 2, x2.device_address());
        }
        let grid = [blocks, items.len() as u32, 1, threads as u32, smem as u32];
        if !self.record_resident_block(Kind::PlainTcRows, command, grid)? {
            return Err("tc rows group pair: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Tensor-core BF16 product of `weight` (`rows x cols` BF16) with `input` (`inputs x cols` FP32)
    /// into contiguous `output` (`inputs x rows` FP32).
    pub fn project_tc_rows_into(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_strided_into(rows, cols, inputs, rows, input, weight, output)
    }

    /// Tensor-core BF16 product directly from packed BF16 `input` (`inputs * cols / 2` u32 words), skipping `TC_MODE_PACK`.
    pub fn project_tc_rows_half_in_into(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_form_into(rows, cols, inputs, rows, true, input, weight, output)
    }

    /// Strided tensor-core BF16 product: row `t` writes to `t * stride .. t * stride + rows`.
    #[allow(clippy::too_many_arguments, reason = "shape, stride and three operands")]
    pub fn project_tc_rows_strided_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_tc_rows_form_into(rows, cols, inputs, stride, false, input, weight, output)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, stride, input form and three operands")]
    fn project_tc_rows_form_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, half_in: bool, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if self.arm_capture() { return Err("tc rows: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        tc_rows::check_tc_dense_form(rows, cols, inputs, stride, half_in, span(weight), span(input), span(output))?;
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, if half_in { inputs * cols / 2 } else { inputs * cols })?;
        self.check_vector(output, (inputs - 1) * stride + rows)?;
        if weight.is_segmented() || input.is_segmented() { return Err("tc rows: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("tc rows: output requires private contiguous storage".into()); }
        let panels = self.tc_panels_for(weight, rows, cols, inputs);
        let (kind, mode, grid, direct, extra_flags) = Self::tc_dense_launch(rows, cols, inputs, false, false, half_in, panels.map(|p| p.2));
        self.require_arm_kernels(&[Kind::PlainTcRows.kernel(), kind.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(kind);
        let packed = if direct || half_in { None } else { Some(self.pack_tc_input(input, inputs, cols)?) };
        let words = [
            weight.device_address(), packed.as_ref().map_or(input.device_address(), |p| p.device_address()), output.device_address(),
            rows as u64, cols as u64, inputs as u64, stride as u64,
            mode, 0, inputs as u64, 0, inputs as u64, 0, extra_flags, 0, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        Self::attach_tc_panels(&mut command, panels, kind, mode);
        if !self.record_resident_block(kind, command, grid)? {
            return Err("tc rows: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Fused multi-output tensor-core BF16 product: one matrix (`rows x cols` BF16) times `input`
    /// (`inputs x cols` FP32), writing directly into 1..=8 contiguous even-width output buffers
    /// (`outputs[s]` holds `inputs * widths[s]` FP32 words) with zero intermediate buffer or copy.
    pub fn project_tc_rows_split_into(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)]) -> Result<(), String> {
        self.project_tc_rows_split_form_into(rows, cols, inputs, false, input, weight, outputs)
    }

    /// Fused multi-output tensor-core BF16 product directly from packed BF16 `input` (`inputs * cols / 2` u32 words),
    /// eliminating `TC_MODE_PACK`.
    pub fn project_tc_rows_split_half_in_into(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)]) -> Result<(), String> {
        self.project_tc_rows_split_form_into(rows, cols, inputs, true, input, weight, outputs)
    }

    fn project_tc_rows_split_form_into(&self, rows: usize, cols: usize, inputs: usize, half_in: bool, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)]) -> Result<(), String> {
        if self.arm_capture() { return Err("tc rows split: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let splits: Vec<SplitOut> = outputs.iter().map(|(v, w)| SplitOut { span: span(v), cols: *w }).collect();
        tc_rows::check_tc_split_form(rows, cols, inputs, half_in, span(weight), span(input), &splits)?;
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, if half_in { inputs * cols / 2 } else { inputs * cols })?;
        if weight.is_segmented() || input.is_segmented() { return Err("tc rows split: operand must be contiguous".into()); }
        for (v, w) in outputs.iter() {
            self.check_vector(v, inputs * *w)?;
            if v.is_segmented() || !v.writable_from(0) { return Err("tc rows split: output requires private contiguous storage".into()); }
        }
        let panels = self.tc_panels_for(weight, rows, cols, inputs);
        let (kind, mode, grid, direct, extra_flags) = Self::tc_dense_launch(rows, cols, inputs, true, false, half_in, panels.map(|p| p.2));
        self.require_arm_kernels(&[Kind::PlainTcRows.kernel(), kind.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(kind);
        let packed = if direct || half_in { None } else { Some(self.pack_tc_input(input, inputs, cols)?) };
        let head = [
            weight.device_address(), packed.as_ref().map_or(input.device_address(), DeviceVector::device_address), 0,
            rows as u64, cols as u64, inputs as u64, 0,
            mode, 0, inputs as u64, 0, inputs as u64, 0, extra_flags, 0, outputs.len() as u64,
        ];
        for (i, w) in head.into_iter().enumerate() { command.word(i, w); }
        Self::attach_tc_panels(&mut command, panels, kind, mode);
        let mut col_start = 0u64;
        for (s, (v, width)) in outputs.iter().enumerate() {
            command.word(16 + 2 * s, v.device_address());
            command.word(16 + 2 * s + 1, col_start | ((*width as u64) << 32));
            col_start += *width as u64;
        }
        if !self.record_resident_block(kind, command, grid)? {
            return Err("tc rows split: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Fused prefill `GateOuter (GateForget+GateGain) + GateBlend + RoundedNorm` (`TC_MODE_GATE_BLEND_NORM`)
    /// across 1..=3 gates in one `PLAIN_TC_BIG` launch, eliminating `c.round(&inner)`, `TC_MODE_PACK`, the
    /// `[rows, 2 * dim]` intermediate gate buffer, and the separate `STREAM_OP_GATE_BLEND_NORM` launch.
    pub fn project_tc_gate_blend_norm(
        &self, rows: usize, dim: usize, k: usize, eps: f32, half_out: bool,
        entries: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>, Option<&DeviceVector<'_>>)],
    ) -> Result<Vec<DeviceVector<'a>>, String> {
        if self.arm_capture() { return Err("tc gate blend norm: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        if !self.resident_launches_active() { return Err("tc gate blend norm: launches are not recorded now".into()); }
        let count = entries.len();
        if !tc_rows::gate_outer_admitted(rows, dim, k, count, tc_rows::TC_THREADS) {
            return Err("tc gate blend norm: shape outside envelope".into());
        }
        let w_words = dim.checked_mul(k).ok_or("tc gate blend norm: weight overflow")?;
        let in_words = rows.checked_mul(k).ok_or("tc gate blend norm: inner overflow")?;
        let v_words = rows.checked_mul(dim).ok_or("tc gate blend norm: row overflow")?;
        let out_words = if half_out { v_words / 2 } else { v_words };
        let mut outputs = Vec::with_capacity(count);
        for &(weights, inner, kept, msg, norm_w) in entries {
            self.check_vector(weights, w_words)?;
            self.check_vector(inner, in_words)?;
            self.check_vector(kept, v_words)?;
            self.check_vector(msg, v_words)?;
            if weights.is_segmented() || inner.is_segmented() || kept.is_segmented() || msg.is_segmented() {
                return Err("tc gate blend norm: operand must be contiguous".into());
            }
            if let Some(nw) = norm_w {
                self.check_vector(nw, dim)?;
                if nw.is_segmented() { return Err("tc gate blend norm: norm weight must be contiguous".into()); }
            }
            outputs.push(self.vector(out_words)?);
        }
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let spans: Vec<_> = entries.iter().zip(&outputs)
            .map(|(&(w, inn, kpt, msg, nw), o)| (span(w), span(inn), span(kpt), span(msg), nw.map(span), span(o)))
            .collect();
        tc_rows::check_tc_gate_blend_norm(rows, dim, k, eps, half_out, &spans)?;
        self.require_arm_kernels(&[Kind::PlainTcBig.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PlainTcBig);
        let head = [
            0u64, 0, 0, dim as u64, k as u64, rows as u64, count as u64,
            tc_rows::TC_MODE_GATE_BLEND_NORM,
            if half_out { tc_rows::GATE_OUTER_HALF_OUT } else { 0 },
            u64::from(eps.to_bits()),
        ];
        for (i, w) in head.into_iter().enumerate() { command.word(i, w); }
        for (idx, (&(w, inn, kpt, msg, nw), o)) in entries.iter().zip(&outputs).enumerate() {
            let base = tc_rows::GATE_OUTER_TABLE_AT + idx * 6;
            command.word(base, w.device_address());
            command.word(base + 1, inn.device_address());
            command.word(base + 2, kpt.device_address());
            command.word(base + 3, msg.device_address());
            command.word(base + 4, nw.map_or(0, DeviceVector::device_address));
            command.word(base + 5, o.device_address());
        }
        let grid = [count as u32, tc_rows::gate_outer_row_blocks(rows), 1, tc_rows::TC_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PlainTcBig, command, grid)? {
            return Err("tc gate blend norm: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok(outputs)
    }

    /// True when device-side MoE grouping + tensor-core MoE projection can run for this visit now.
    pub fn tc_moe_ready(&self, rows: usize, picks: usize, experts: usize, gate_rows: usize, hidden: usize, down_cols: usize, stack_gate: &DeviceVector<'_>, stack_down: &DeviceVector<'_>, input: &DeviceVector<'_>) -> bool {
        let Some(all) = rows.checked_mul(picks) else { return false };
        tc_rows::moe_route_admitted(rows, 2 * picks, picks, experts, tc_rows::moe_max_tiles(all, experts), tc_rows::MOE_ROUTE_THREADS)
            && gate_rows <= tc_rows::TC_ROWS_MAX && hidden <= tc_rows::TC_ROWS_MAX
            && tc_rows::tc_shape_admitted(gate_rows, hidden, 1)
            && tc_rows::tc_shape_admitted(hidden, down_cols, 1)
            && (gate_rows * hidden * 2).is_multiple_of(16) && (hidden * down_cols * 2).is_multiple_of(16)
            && !stack_gate.is_segmented() && !stack_down.is_segmented() && !input.is_segmented()
            && stack_gate.len() == experts * gate_rows * hidden / 2
            && stack_down.len() == experts * hidden * down_cols / 2
            && input.len() == rows * hidden
            && (stack_gate.device_address() | stack_down.device_address() | input.device_address()) & 15 == 0
            && self.resident_launches_active()
    }

    /// Device-side deterministic MoE router grouping (`MOE_GROUP_ROUTE`, 1 block of 256 threads).
    pub fn moe_group_route_into(&self, rows: usize, each: usize, picks: usize, experts: usize, route: &DeviceVector<'_>, weights_out: &mut DeviceVector<'_>) -> Result<MoeRoutePlan<'a>, String> {
        if self.arm_capture() { return Err("moe route: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let all = rows.checked_mul(picks).filter(|&n| n <= tc_rows::MOE_ALL_MAX).ok_or("moe route: too many picks")?;
        if experts > tc_rows::MOE_EXPERTS_MAX { return Err("moe route: too many experts".into()); }
        let max_tiles = tc_rows::moe_max_tiles(all, experts);
        if !tc_rows::moe_route_admitted(rows, each, picks, experts, max_tiles, tc_rows::MOE_ROUTE_THREADS) {
            return Err("moe route: shape outside envelope".into());
        }
        let meta_words = tc_rows::moe_tile_meta_words(all, experts);
        let (in_rows, out_slots, tile_meta) = (self.vector(all)?, self.vector(all)?, self.vector(meta_words)?);
        tc_rows::check_moe_group_route(rows, each, picks, experts, span(route), span(weights_out), span(&in_rows), span(&out_slots), span(&tile_meta))?;
        self.check_vector(route, rows * each)?;
        self.check_vector(weights_out, all)?;
        if route.is_segmented() { return Err("moe route: route input must be contiguous".into()); }
        for out in [&*weights_out, &in_rows, &out_slots, &tile_meta] {
            if out.is_segmented() || !out.writable_from(0) { return Err("moe route: output requires private contiguous storage".into()); }
        }
        self.require_arm_kernels(&[Kind::MoeGroupRoute.kernel()])?;
        self.resident_completion()?;
        let blocks = tc_rows::moe_route_blocks(rows);
        if blocks == 1 {
            let mut command = Command::new(Kind::MoeGroupRoute);
            let words = [
                route.device_address(), weights_out.device_address(), in_rows.device_address(),
                out_slots.device_address(), tile_meta.device_address(),
                rows as u64, each as u64, picks as u64, experts as u64, max_tiles as u64,
                tc_rows::MOE_ROUTE_PHASE_SINGLE, 0, 0, 1,
            ];
            for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
            let grid = [1, 1, 1, tc_rows::MOE_ROUTE_THREADS as u32, 0];
            if !self.record_resident_block(Kind::MoeGroupRoute, command, grid)? {
                return Err("moe route: launches are not recorded now".into());
            }
            self.resident_completion()?;
            return Ok(MoeRoutePlan { rows, all, experts, in_rows, out_slots, tile_meta });
        }
        let block_hist = self.vector(blocks * experts)?;
        let ids_buf = self.vector(all.div_ceil(2))?;
        if block_hist.is_segmented() || ids_buf.is_segmented() {
            return Err("moe route: scratch requires contiguous storage".into());
        }
        let base_words = [
            route.device_address(), weights_out.device_address(), in_rows.device_address(),
            out_slots.device_address(), tile_meta.device_address(),
            rows as u64, each as u64, picks as u64, experts as u64, max_tiles as u64,
            0, block_hist.device_address(), ids_buf.device_address(), blocks as u64,
        ];
        for (phase, gx) in [
            (tc_rows::MOE_ROUTE_PHASE_COUNT, blocks as u32),
            (tc_rows::MOE_ROUTE_PHASE_SCAN, 1u32),
            (tc_rows::MOE_ROUTE_PHASE_SCATTER, blocks as u32),
        ] {
            let mut command = Command::new(Kind::MoeGroupRoute);
            for (i, w) in base_words.into_iter().enumerate() { command.word(i, w); }
            command.word(10, phase);
            let grid = [gx, 1, 1, tc_rows::MOE_ROUTE_THREADS as u32, 0];
            if !self.record_resident_block(Kind::MoeGroupRoute, command, grid)? {
                return Err("moe route: launches are not recorded now".into());
            }
        }
        self.resident_completion()?;
        Ok(MoeRoutePlan { rows, all, experts, in_rows, out_slots, tile_meta })
    }

    /// Device-routed tensor-core MoE expert stack GEMM (`PLAIN_TC_ROWS` Mode 1, zero CPU sync).
    pub fn project_tc_moe_into(&self, rows: usize, cols: usize, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, down: bool) -> Result<(), String> {
        if self.arm_capture() { return Err("tc moe: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let (all, experts, tile_meta) = (route.all, route.experts, &route.tile_meta);
        let (in_count, in_table, out_table) = route.tables(down);
        let out_count = all;
        tc_rows::check_tc_moe(rows, cols, all, experts, span(stack), span(tile_meta), span(input), in_count, in_table.map(span), span(output), out_count, out_table.map(span))?;
        let meta_words = tc_rows::moe_tile_meta_words(all, experts);
        self.check_vector(stack, experts * rows * cols / 2)?;
        self.check_vector(tile_meta, meta_words)?;
        self.check_vector(input, in_count * cols)?;
        self.check_vector(output, out_count * rows)?;
        if let Some(t) = in_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("tc moe: in_table must be contiguous".into()); } }
        if let Some(t) = out_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("tc moe: out_table must be contiguous".into()); } }
        if stack.is_segmented() || tile_meta.is_segmented() || input.is_segmented() { return Err("tc moe: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("tc moe: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainTcRows.kernel()])?;
        self.resident_completion()?;
        let max_tiles = tc_rows::moe_max_tiles(all, experts);
        let mut command = Command::new(Kind::PlainTcRows);
        let packed = self.pack_tc_input(input, in_count, cols)?;
        let words = [
            stack.device_address(), packed.device_address(), output.device_address(),
            rows as u64, cols as u64, all as u64, rows as u64,
            tc_rows::TC_MODE_MOE,
            in_table.map_or(0, DeviceVector::device_address), in_count as u64,
            out_table.map_or(0, DeviceVector::device_address), out_count as u64,
            tile_meta.device_address(), max_tiles as u64, experts as u64, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = [(rows.div_ceil(tc_rows::TC_PLAIN_BN) * max_tiles) as u32, 1, 1, tc_rows::TC_THREADS as u32, tc_rows::TC_GEN_SMEM as u32];
        if !self.record_resident_block(Kind::PlainTcRows, command, grid)? {
            return Err("tc moe: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Device-routed tensor-core BF16 MoE expert stack GEMM directly from shipped tiled packed NVFP4/MXFP8 units
    /// (`PLAIN_PACKED_TC_ROWS`, kind 106), reading `tile_meta` produced by `moe_group_route_into`.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route and operands")]
    pub fn project_tc_packed_moe_into(
        &self, rows: usize, cols: usize, wide: bool, pairs: bool,
        stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, down: bool,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, pairs, stack, route, input, output, down, None, false, false)
    }

    /// The half `Down` form of `project_tc_packed_moe_into` (prefill, `packed_product::HALF_OUT`): no `PAIRS`, even
    /// `rows`, and `output` holds `picks x rows` BF16 values, two a word (`picks x rows / 2` words): each value is
    /// `rn` of the FP32 word of the other form. `fast_route_fold_half_into` reads them.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route and operands")]
    pub fn project_tc_packed_down_half_into(
        &self, rows: usize, cols: usize, wide: bool, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>,
        input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, false, stack, route, input, output, true, None, true, false)
    }

    /// `project_tc_packed_down_half_into` on a BF16 input (`packed_product::HALF_IN`): `input` holds `picks x cols`
    /// BF16 values, two a word (`picks x cols / 2` words), as `project_tc_packed_gate_half_into` writes them. No
    /// rounding launch runs before the product: the values are the ones it would make.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route and operands")]
    pub fn project_tc_packed_down_half_from_half_into(
        &self, rows: usize, cols: usize, wide: bool, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>,
        input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, false, stack, route, input, output, true, None, true, true)
    }

    /// The gated `GateUp` form of `project_tc_packed_moe_into` (prefill): mixed gate and up rows (`pairs`), and
    /// `output` holds `picks x rows / 2` words, each `CappedGate` of its gate and up word with `caps` = gate cap,
    /// up cap, reciprocal gate cap, reciprocal up cap. Records the build of the two exact `FAST_CAPPED_GATE`
    /// coefficient tables, then the product; the words equal the product followed by `fast_capped_gate_into`.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route, caps and operands")]
    pub fn project_tc_packed_gate_into(
        &self, rows: usize, cols: usize, wide: bool, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>,
        input: &DeviceVector<'_>, caps: [f32; 4], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, true, stack, route, input, output, false, Some(caps), false, false)
    }

    /// `project_tc_packed_gate_into` with BF16 outputs (`packed_product::HALF_OUT` on a gated launch): `output` holds
    /// `picks x rows / 2` BF16 values, two a word, each the value `cvt.rn.bf16x2.f32` makes of the gated FP32 word of
    /// the other form: the input of the `Down` product as that product rounds it.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route, caps and operands")]
    pub fn project_tc_packed_gate_half_into(
        &self, rows: usize, cols: usize, wide: bool, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>,
        input: &DeviceVector<'_>, caps: [f32; 4], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, true, stack, route, input, output, false, Some(caps), true, false)
    }

    /// `project_tc_packed_gate_half_into` directly from a packed BF16 input (`packed_product::HALF_IN`),
    /// eliminating the `TC_MODE_PACK` launch before `GateUp`.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route, caps and operands")]
    pub fn project_tc_packed_gate_half_from_half_into(
        &self, rows: usize, cols: usize, wide: bool, stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>,
        input: &DeviceVector<'_>, caps: [f32; 4], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.packed_moe_into(rows, cols, wide, true, stack, route, input, output, false, Some(caps), true, true)
    }

    /// The launch of every packed MoE form. The product reads BF16 input rows only (`packed_product::HALF_IN`): an
    /// FP32 `input` (`half_in` false) is rounded first by a `TC_MODE_PACK` launch into a private vector, with the
    /// rounding (`cvt.rn.bf16x2.f32`) the product made of each word before.
    #[allow(clippy::too_many_arguments, reason = "shape, packed form, route, caps, output and input forms and operands")]
    fn packed_moe_into(
        &self, rows: usize, cols: usize, wide: bool, pairs: bool,
        stack: &DeviceVector<'_>, route: &MoeRoutePlan<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, down: bool, caps: Option<[f32; 4]>, half: bool, half_in: bool,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("tc packed moe: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let (all, experts, tile_meta) = (route.all, route.experts, &route.tile_meta);
        let (in_count, in_table, out_table) = route.tables(down);
        let out_count = all;
        // Gated: a private vector for the two coefficient tables, built by a launch recorded just before the product.
        let gate_words = 2 * crate::fast_ops::CAPPED_GATE_TABLE_WORDS;
        let tables = caps.map(|_| self.vector(gate_words)).transpose()?;
        tc_rows::check_tc_packed_moe_input(
            rows, cols, wide, pairs, all, experts, span(stack), span(tile_meta),
            span(input), in_count, in_table.map(span),
            span(output), out_count, out_table.map(span), tables.as_ref().map(span), half, half_in,
        )?;
        let width = if caps.is_some() { rows / 2 } else { rows };
        // Half: two BF16 values a word (the size is checked above).
        let out_words = if half { out_count * width / 2 } else { out_count * width };
        let unit_bytes = crate::packed_product::layout(rows, cols, wide).bytes;
        let unit_words = unit_bytes / 4;
        self.check_vector(stack, experts * unit_words)?;
        self.check_vector(tile_meta, tc_rows::moe_tile_meta_words(all, experts))?;
        self.check_vector(input, if half_in { in_count * cols / 2 } else { in_count * cols })?;
        self.check_vector(output, out_words)?;
        if let Some(t) = in_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("tc packed moe: in_table segmented".into()); } }
        if let Some(t) = out_table { self.check_vector(t, all)?; if t.is_segmented() { return Err("tc packed moe: out_table segmented".into()); } }
        if stack.is_segmented() || tile_meta.is_segmented() || input.is_segmented() { return Err("tc packed moe: operand segmented".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("tc packed moe: output requires private contiguous storage".into()); }
        if let Some(t) = &tables { self.check_vector(t, gate_words)?; }
        self.require_arm_kernels(&[Kind::PlainPackedTcRows.kernel()])?;
        // The rounding of an FP32 input is a `PLAIN_TC_ROWS` launch (`TC_MODE_PACK`).
        if !half_in { self.require_arm_kernels(&[Kind::PlainTcRows.kernel()])?; }
        self.resident_completion()?;
        if let (Some(caps), Some(t)) = (caps, &tables) { self.record_capped_gate_tables(caps, t)?; }
        // FP32 input: the BF16 rows the product reads, rounded on the recorded stream before it.
        let rounded = if half_in { None } else { Some(self.pack_tc_input(input, in_count, cols)?) };
        let source = rounded.as_ref().map_or_else(|| input.device_address(), DeviceVector::device_address);
        let max_tiles = tc_rows::moe_max_tiles(all, experts);
        let mut command = Command::new(Kind::PlainPackedTcRows);
        let words = [
            stack.device_address(), source, output.device_address(),
            rows as u64, u64::from(crate::packed_product::cols_word(cols, wide, pairs)),
            all as u64, width as u64, unit_bytes as u64,
            in_table.map_or(0, DeviceVector::device_address), in_count as u64,
            out_table.map_or(0, DeviceVector::device_address), out_count as u64,
            tile_meta.device_address(), max_tiles as u64, experts as u64,
            tables.as_ref().map_or(0, DeviceVector::device_address),
            if half { crate::packed_product::HALF_OUT } else { 0 },
            crate::packed_product::HALF_IN,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = [tc_rows::tc_packed_moe_blocks(rows, max_tiles), 1, 1, tc_rows::TC_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PlainPackedTcRows, command, grid)? {
            return Err("tc packed moe: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// SM121a BF16 tensor-core fused causal/sliding-window attention (`FLASH_WINDOW_ATTN`, kind 105).
    /// Replaces `RelBias + WindowScores + SinkSoftmax + WindowValues` for contiguous chat spans with zero
    /// materialized `[rows, heads, view]` score or weight buffers.
    /// With `clips`, `query` and `moved` are the products before their clips and each block makes the clips of its rows
    /// (`flash_window::FW_CLIP_IN`): the words of the prefill `HeadClip` launches, then this form.
    #[allow(clippy::too_many_arguments, reason = "shape, scale and nine device operands")]
    pub fn flash_window_attn_into(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: Option<OneClips<'_, '_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.flash_window_attn_form_into(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, first_position, false,
            query, keys, values, moved, rel_table, sink, gate, steps, clips, output,
        )
    }

    /// `flash_window_attn_into` writing packed BF16 output (`FW_HALF_OUT`, `rows * heads * head_dim / 2` words),
    /// directly consumable by `project_tc_rows_half_in_into(Weight::Output)` without `TC_MODE_PACK`.
    #[allow(clippy::too_many_arguments, reason = "shape, scale and nine device operands")]
    pub fn flash_window_attn_half_into(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: Option<OneClips<'_, '_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.flash_window_attn_form_into(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, first_position, true,
            query, keys, values, moved, rel_table, sink, gate, steps, clips, output,
        )
    }

    #[allow(clippy::too_many_arguments, reason = "shape, scale, output form and nine device operands")]
    fn flash_window_attn_form_into(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, half_out: bool, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: Option<OneClips<'_, '_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        use crate::flash_window::{self, Span as FwSpan};
        if self.arm_capture() { return Err("flash window: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| FwSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let kv_words = view * kv_heads * (head_dim / 2);
        let prefix_span = |v: &DeviceVector<'_>, words: usize| FwSpan { address: v.device_address(), bytes: words as u64 * 4 };
        flash_window::check_flash_window_form(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, half_out,
            span(query), prefix_span(keys, kv_words), prefix_span(values, kv_words), span(moved), span(rel_table), span(sink), span(gate), span(steps), span(output),
        )?;
        let out_words = if half_out { rows * heads * head_dim / 2 } else { rows * heads * head_dim };
        self.check_vector(query, rows * heads * head_dim)?;
        self.check_vector_prefix(keys, kv_words)?;
        self.check_vector_prefix(values, kv_words)?;
        self.check_vector(moved, rows * heads * rel_dim)?;
        self.check_vector(rel_table, rel_dim * extent)?;
        self.check_vector(sink, rows * heads)?;
        self.check_vector(gate, rows * heads * head_dim)?;
        self.check_vector(steps, rows * 2)?;
        self.check_vector(output, out_words)?;
        for inp in [query, moved, rel_table, sink, gate, steps] {
            if inp.is_segmented() { return Err("flash window: input must be contiguous".into()); }
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("flash window: output requires private contiguous storage".into());
        }
        let meets = |a: FwSpan, b: FwSpan| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
        if meets(span(output), span(keys)) || meets(span(output), span(values)) {
            return Err("flash window: output overlaps KV plane".into());
        }
        if let Some(clips) = &clips {
            if clips.written_keys.is_some() || clips.written_values.is_some() {
                return Err("flash window: tensor-core form does not take written KV operands".into());
            }
            if !(clips.query_bound > 0.0 && clips.moved_bound > 0.0) { return Err("flash window: a clip threshold is not positive".into()); }
            for (v, n) in [(clips.query_scale, heads * head_dim), (clips.moved_scale, heads * rel_dim), (clips.position, rows)] {
                if let Some(v) = v {
                    self.check_vector(v, n)?;
                    if v.is_segmented() || output.device_address() < v.device_address() + 4 * n as u64 && v.device_address() < output.device_address() + 4 * out_words as u64 {
                        return Err("flash window: a clip operand is segmented or meets the output".into());
                    }
                }
            }
        }
        self.require_arm_kernels(&[Kind::FlashWindowAttn.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FlashWindowAttn);
        let words = [
            query.device_address(), keys.device_address(), values.device_address(),
            moved.device_address(), rel_table.device_address(), sink.device_address(),
            gate.device_address(), steps.device_address(), output.device_address(),
            rows as u64, heads as u64, kv_heads as u64, view as u64, extent as u64,
            u64::from(scale.to_bits()), first_position % flash_window::FW_BKV as u64,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let mut clip_flags = if half_out { flash_window::FW_HALF_OUT } else { 0 };
        if let Some(clips) = &clips {
            let address = |v: Option<&DeviceVector<'_>>| v.map_or(0, |v| v.device_address());
            command.word(17, address(clips.query_scale));
            command.word(18, address(clips.moved_scale));
            command.word(19, address(clips.position));
            command.word(20, u64::from(clips.query_bound.to_bits()));
            command.word(21, u64::from(clips.moved_bound.to_bits()));
            clip_flags |= flash_window::FW_CLIP_IN;
        }
        if clip_flags != 0 {
            command.word(flash_window::FW_CLIP_WORD, clip_flags);
        }
        let grid = [
            flash_window::flash_window_blocks_gqa(rows, heads, kv_heads), 1, 1,
            flash_window::flash_window_threads(heads, kv_heads) as u32, 0,
        ];
        if !self.record_resident_block(Kind::FlashWindowAttn, command, grid)? {
            return Err("flash window: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// The exact form of `FLASH_WINDOW_ATTN` (payload word `FW_MODE_WORD` = `FW_EXACT_ONE`): `rows` query rows (1 to
    /// `FW_ONE_ROWS_MAX`) of `heads` heads against one key view of at most `FW_ONE_VIEW_MAX` rows,
    /// `flash_window_one_split(rows)` blocks a query head of a row. Its words are those of the block arms `HeadClip` (query and relative state),
    /// `RelBias`, `WindowScores`, `SinkSoftmax` and `WindowValues` for each row; `steps` holds the live count and the
    /// slot of each row, `clips.position` one factor word a row. The operands are checked as for the tensor-core form
    /// with `rows` rows.
    #[allow(clippy::too_many_arguments, reason = "shape, scale and nine device operands")]
    pub fn flash_window_one_into(
        &self, rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, scale: f32,
        query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>, moved: &DeviceVector<'_>,
        rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>, gate: &DeviceVector<'_>, steps: &DeviceVector<'_>,
        clips: OneClips<'_, '_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.flash_window_planes_into(rows, heads, kv_heads, view, extent, scale, query, keys, values, moved,
            rel_table, sink, gate, steps, clips, &[], output)
    }

    /// Exact attention with a separate key and value view for each chat.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_window_planes_into(
        &self, rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, scale: f32,
        query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>, moved: &DeviceVector<'_>,
        rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>, gate: &DeviceVector<'_>, steps: &DeviceVector<'_>,
        clips: OneClips<'_, '_>, planes: &[(&DeviceVector<'_>, &DeviceVector<'_>)], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        use crate::flash_window::{self, Span as FwSpan};
        if self.arm_capture() { return Err("flash window one: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let (head_dim, rel_dim) = (flash_window::FW_HEAD_DIM, flash_window::FW_REL_DIM);
        let span = |v: &DeviceVector<'_>| FwSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let split = flash_window::flash_window_one_split(rows);
        let grid = rows.checked_mul(heads).and_then(|n| n.checked_mul(split)).ok_or("flash window one: grid overflow")?;
        if !flash_window::flash_window_one_admitted(rows, heads, kv_heads, view, extent, keys.device_address() as usize,
            values.device_address() as usize, flash_window::FW_THREADS, grid, split) {
            return Err("flash window one: shape outside the exact form".into());
        }
        let kv_words = view * kv_heads * (head_dim / 2);
        let prefix_span = |v: &DeviceVector<'_>, words: usize| FwSpan { address: v.device_address(), bytes: words as u64 * 4 };
        flash_window::check_flash_window(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale,
            span(query), prefix_span(keys, kv_words), prefix_span(values, kv_words), span(moved), span(rel_table), span(sink), span(gate), span(steps), span(output),
        )?;
        self.check_vector(query, rows * heads * head_dim)?;
        self.check_vector_prefix(keys, kv_words)?;
        self.check_vector_prefix(values, kv_words)?;
        self.check_vector(moved, rows * heads * rel_dim)?;
        self.check_vector(rel_table, rel_dim * extent)?;
        self.check_vector(sink, rows * heads)?;
        self.check_vector(gate, rows * heads * head_dim)?;
        self.check_vector(steps, 2 * rows)?;
        self.check_vector(output, rows * heads * head_dim)?;
        for inp in [query, moved, rel_table, sink, gate, steps] {
            if inp.is_segmented() { return Err("flash window one: input must be contiguous".into()); }
        }
        let meets = |a: FwSpan, b: FwSpan| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
        if meets(span(output), span(keys)) || meets(span(output), span(values)) {
            return Err("flash window one: output overlaps KV plane".into());
        }
        if !(clips.query_bound > 0.0 && clips.moved_bound > 0.0) { return Err("flash window one: a clip threshold is not positive".into()); }
        for (v, n) in [(clips.query_scale, heads * head_dim), (clips.moved_scale, heads * rel_dim), (clips.position, rows)] {
            if let Some(v) = v {
                self.check_vector(v, n)?;
                if v.is_segmented() || output.device_address() < v.device_address() + 4 * n as u64 && v.device_address() < output.device_address() + 4 * (rows * heads * head_dim) as u64 {
                    return Err("flash window one: a clip operand is segmented or meets the output".into());
                }
            }
        }
        if clips.written_keys.is_some() != clips.written_values.is_some() {
            return Err("flash window one: written_keys and written_values must both be present or both absent".into());
        }
        if clips.written_keys.is_some() && !planes.is_empty() {
            return Err("flash window one: written KV cannot be combined with per-row planes".into());
        }
        let written_words = rows * kv_heads * (head_dim / 2);
        for v in [clips.written_keys, clips.written_values].into_iter().flatten() {
            self.check_vector(v, written_words)?;
            if v.is_segmented() || !v.device_address().is_multiple_of(16)
                || output.device_address() < v.device_address() + 4 * written_words as u64 && v.device_address() < output.device_address() + 4 * (rows * heads * head_dim) as u64 {
                return Err("flash window one: written KV operand is segmented, unaligned, or meets output".into());
            }
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("flash window one: output requires private contiguous storage".into());
        }
        self.require_arm_kernels(&[Kind::FlashWindowAttn.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FlashWindowAttn);
        let words = [
            query.device_address(), keys.device_address(), values.device_address(),
            moved.device_address(), rel_table.device_address(), sink.device_address(),
            gate.device_address(), steps.device_address(), output.device_address(),
            rows as u64, heads as u64, kv_heads as u64, view as u64, extent as u64, u64::from(scale.to_bits()), 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(flash_window::FW_MODE_WORD, flash_window::FW_EXACT_ONE);
        let address = |v: Option<&DeviceVector<'_>>| v.map_or(0, |v| v.device_address());
        command.word(17, address(clips.query_scale));
        command.word(18, address(clips.moved_scale));
        command.word(19, address(clips.position));
        command.word(20, u64::from(clips.query_bound.to_bits()));
        command.word(21, u64::from(clips.moved_bound.to_bits()));
        command.word(flash_window::FW_SPLIT_WORD, split as u64);
        command.word(flash_window::FW_WRITTEN_KEYS, address(clips.written_keys));
        command.word(flash_window::FW_WRITTEN_VALUES, address(clips.written_values));
        if !planes.is_empty() && planes.len() != rows { return Err("flash window: one plane pair per row required".into()); }
        command.word(flash_window::FW_PLANE_COUNT, planes.len() as u64);
        for (row, &(key, value)) in planes.iter().enumerate() {
            for (side, plane) in [key, value].into_iter().enumerate() {
                self.check_vector_prefix(plane, kv_words)?;
                let p = span(plane);
                if p.address == 0 || p.address % 16 != 0 ||
                    p.address.checked_add(p.bytes).is_none_or(|end| span(output).address < end && p.address < span(output).address + span(output).bytes) {
                    return Err("flash window: invalid chat plane or overlap with output".into());
                }
                command.word(flash_window::FW_PLANES_AT + 2 * row + side, plane.device_address());
            }
        }
        let grid = [u32::try_from(grid).map_err(|_| "flash window one: grid overflow")?, 1, 1, flash_window::FW_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FlashWindowAttn, command, grid)? {
            return Err("flash window one: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// GQA-packed Split-KV SM121a BF16 tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16`/`32` +
    /// `FLASH_WINDOW_REDUCE`): supports `M = 1..=16` contiguous rows (with optional zero-copy provisional
    /// sliding-window `delta = Some((delta_keys, delta_values))`) and `C = 2..=16` multi-chat `planes`, with
    /// bit-identical per-token output across `M` and `C`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_window_split_into(
        &self, rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, scale: f32,
        is_ring: bool, first_pos: u64, row_pos_live: &[(u64, usize)],
        query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        delta: Option<(&DeviceVector<'_>, &DeviceVector<'_>)>,
        moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: OneClips<'_, '_>,
        planes: &[(&DeviceVector<'_>, &DeviceVector<'_>)], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        use crate::flash_window::{self, Span as FwSpan};
        if self.arm_capture() { return Err("flash window split: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let (head_dim, rel_dim) = (flash_window::FW_HEAD_DIM, flash_window::FW_REL_DIM);
        if row_pos_live.len() != rows { return Err("flash window split: row_pos_live length differs from rows".into()); }
        let multi_chat = !planes.is_empty();
        if multi_chat && (planes.len() != rows || delta.is_some()) {
            return Err("flash window split: invalid multi-chat planes or delta".into());
        }
        let split_kv = flash_window::flash_window_split_kv(is_ring);
        let (split_start, num_splits) = if multi_chat {
            let mut max_s = 1usize;
            for &(pos, live) in row_pos_live {
                if live > view || live as u64 > pos.saturating_add(1) || (!is_ring && live > 0 && pos >= view as u64) {
                    return Err("flash window split: multi-chat live or position out of bounds".into());
                }
                if live > 0 {
                    let klo = (pos + 1 - live as u64) as usize;
                    let khi = pos as usize;
                    let s = khi / split_kv - klo / split_kv + 1;
                    if s > max_s { max_s = s; }
                }
            }
            (0usize, max_s)
        } else {
            let (mut k_min, mut k_max, mut any) = (usize::MAX, 0usize, false);
            for (r, &(pos, live)) in row_pos_live.iter().enumerate() {
                if pos != first_pos + r as u64 || live > view || live as u64 > pos.saturating_add(1)
                    || (!is_ring && delta.is_none() && live > 0 && pos >= view as u64) {
                    return Err("flash window split: single-chat live or position out of bounds".into());
                }
                if live > 0 {
                    let klo = (pos + 1 - live as u64) as usize;
                    let khi = pos as usize;
                    if klo < k_min { k_min = klo; }
                    if khi > k_max { k_max = khi; }
                    any = true;
                }
            }
            if !any { (0, 1) } else {
                let s0 = k_min / split_kv;
                (s0, k_max / split_kv - s0 + 1)
            }
        };
        let (q_groups, rows_per_cta) = flash_window::flash_window_split_q_groups(rows, heads, kv_heads, multi_chat);
        let partial_words = rows.checked_mul(heads)
            .and_then(|n| n.checked_mul(num_splits))
            .and_then(|n| n.checked_mul(flash_window::FW_SPLIT_PARTIAL_STRIDE))
            .ok_or("flash window split: partial buffer overflow")?;
        let partials = self.vector(partial_words)?;
        if !flash_window::flash_window_split_admitted(
            rows, heads, kv_heads, view, extent, split_kv, num_splits, q_groups, rows_per_cta, planes.len(),
            query.device_address() as usize, keys.device_address() as usize, values.device_address() as usize,
            moved.device_address() as usize, partials.device_address() as usize, flash_window::FW_THREADS,
        ) {
            return Err("flash window split: shape outside split-KV envelope".into());
        }
        let span = |v: &DeviceVector<'_>| FwSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let row_words = kv_heads * (head_dim / 2);
        let kv_words = view * row_words;
        let prefix_span = |v: &DeviceVector<'_>, words: usize| FwSpan { address: v.device_address(), bytes: words as u64 * 4 };
        flash_window::check_flash_window(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale,
            span(query), prefix_span(keys, kv_words), prefix_span(values, kv_words), span(moved), span(rel_table), span(sink), span(gate), span(steps), span(output),
        )?;
        self.check_vector(query, rows * heads * head_dim)?;
        if multi_chat && !is_ring {
            let (pos0, live0) = row_pos_live[0];
            let need0 = if live0 > 0 { (pos0 as usize + 1) * row_words } else { row_words };
            self.check_vector_prefix(keys, need0)?;
            self.check_vector_prefix(values, need0)?;
        } else {
            self.check_vector_prefix(keys, kv_words)?;
            self.check_vector_prefix(values, kv_words)?;
        }
        self.check_vector(moved, rows * heads * rel_dim)?;
        self.check_vector(rel_table, rel_dim * extent)?;
        self.check_vector(sink, rows * heads)?;
        self.check_vector(gate, rows * heads * head_dim)?;
        self.check_vector(steps, 2 * rows)?;
        self.check_vector(output, rows * heads * head_dim)?;
        for inp in [query, moved, rel_table, sink, gate, steps] {
            if inp.is_segmented() { return Err("flash window split: input must be contiguous".into()); }
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("flash window split: output requires private contiguous storage".into());
        }
        if !(clips.query_bound > 0.0 && clips.moved_bound > 0.0) {
            return Err("flash window split: a clip threshold is not positive".into());
        }
        let meets = |a: FwSpan, b: FwSpan| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
        let out_sp = span(output);
        let part_sp = span(&partials);
        if meets(out_sp, part_sp) { return Err("flash window split: partials overlaps output".into()); }
        if meets(out_sp, span(keys)) || meets(out_sp, span(values)) || meets(part_sp, span(keys)) || meets(part_sp, span(values)) {
            return Err("flash window split: KV plane overlaps output or partials".into());
        }
        for (v, n) in [(clips.query_scale, heads * head_dim), (clips.moved_scale, heads * rel_dim), (clips.position, rows)] {
            if let Some(v) = v {
                self.check_vector(v, n)?;
                if v.is_segmented() || meets(out_sp, span(v)) || meets(part_sp, span(v)) {
                    return Err("flash window split: clip operand invalid or overlaps output".into());
                }
            }
        }
        if let Some((dk, dv)) = delta {
            let need_d = rows * kv_heads * (head_dim / 2);
            for d in [dk, dv] {
                self.check_vector(d, need_d)?;
                let dsp = span(d);
                if d.is_segmented() || dsp.address == 0 || dsp.address % 16 != 0 || meets(out_sp, dsp) || meets(part_sp, dsp) {
                    return Err("flash window split: delta KV operand invalid or overlaps output".into());
                }
            }
        }
        let split_kind = if q_groups == 1 { Kind::FlashWindowSplit16 } else { Kind::FlashWindowSplit32 };
        self.require_arm_kernels(&[split_kind.kernel(), Kind::FlashWindowReduce.kernel()])?;
        self.resident_completion()?;

        let mut cmd1 = Command::new(split_kind);
        let words1 = [
            query.device_address(), keys.device_address(), values.device_address(),
            moved.device_address(), rel_table.device_address(),
            delta.map_or(0, |(dk, _)| dk.device_address()), delta.map_or(0, |(_, dv)| dv.device_address()),
            steps.device_address(), partials.device_address(),
            rows as u64, heads as u64, kv_heads as u64, view as u64, extent as u64,
            u64::from(scale.to_bits()), first_pos,
        ];
        for (i, w) in words1.into_iter().enumerate() { cmd1.word(i, w); }
        cmd1.word(flash_window::FW_SP_FLAGS_WORD, flash_window::fw_split_pack_flags(split_kv, num_splits, is_ring));
        let addr = |v: Option<&DeviceVector<'_>>| v.map_or(0, |v| v.device_address());
        cmd1.word(17, addr(clips.query_scale));
        cmd1.word(18, addr(clips.moved_scale));
        cmd1.word(19, addr(clips.position));
        cmd1.word(20, u64::from(clips.query_bound.to_bits()));
        cmd1.word(21, u64::from(clips.moved_bound.to_bits()));
        cmd1.word(flash_window::FW_SP_SPLIT_START_WORD, split_start as u64);
        cmd1.word(flash_window::FW_SP_PLANE_COUNT_WORD, planes.len() as u64);
        cmd1.word(flash_window::FW_SP_ROWS_PER_CTA_WORD, rows_per_cta as u64);
        for (chat, &(key, value)) in planes.iter().enumerate() {
            let (pos, live) = row_pos_live[chat];
            let need_words = if is_ring {
                kv_words
            } else if live > 0 {
                (pos as usize + 1) * row_words
            } else {
                row_words
            };
            for (side, plane) in [key, value].into_iter().enumerate() {
                self.check_vector_prefix(plane, need_words)?;
                let p = span(plane);
                if p.address == 0 || p.address % 16 != 0 || meets(out_sp, p) || meets(part_sp, p) {
                    return Err("flash window split: invalid chat plane or overlap".into());
                }
                cmd1.word(flash_window::FW_SP_PLANES_AT + 2 * chat + side, plane.device_address());
            }
            cmd1.word(flash_window::FW_SP_POS_AT + chat, pos);
            cmd1.word(flash_window::FW_SP_LIVE_AT + chat, live as u64);
        }
        let q_ctas = rows.div_ceil(rows_per_cta);
        let gx1 = u32::try_from(q_ctas.checked_mul(kv_heads).ok_or("flash window split: grid x overflow")?)
            .map_err(|_| "flash window split: grid x overflow")?;
        let gy1 = u32::try_from(flash_window::flash_window_split_grid_y(num_splits)).map_err(|_| "flash window split: grid y overflow")?;
        if !self.record_resident_block(split_kind, cmd1, [gx1, gy1, 1, flash_window::FW_THREADS as u32, 0])? {
            return Err("flash window split: launches are not recorded now".into());
        }

        let mut cmd2 = Command::new(Kind::FlashWindowReduce);
        let words2 = [
            partials.device_address(), sink.device_address(), gate.device_address(), output.device_address(),
            rows as u64, heads as u64, num_splits as u64,
        ];
        for (i, w) in words2.into_iter().enumerate() { cmd2.word(i, w); }
        let gx2 = u32::try_from(rows.checked_mul(heads).ok_or("flash window reduce: grid overflow")?)
            .map_err(|_| "flash window reduce: grid overflow")?;
        if !self.record_resident_block(Kind::FlashWindowReduce, cmd2, [gx2, 1, 1, flash_window::FW_THREADS as u32, 0])? {
            return Err("flash window reduce: launches are not recorded now".into());
        }
        // `partials` drops here; inside the resident queue, deferred retirement keeps it alive until both launches finish.
        self.resident_completion()
    }

    /// The key or the value row of one token in one recorded arms launch: arm `ShortConv` (`shape`, no addend) with
    /// the token's `product` as the input of ring row `step` (the launch also writes it there), then `finish`
    /// (`HeadClip` with threshold `param`, or `RoundedNorm` with epsilon `param`, no scale or weight) on each head,
    /// and the finished words, packed two to a word as arm `PackBf16` packs them, into words `plane_word ..
    /// plane_word + width/2` of `plane`. `output` gets the finished words unpacked. The words are those of the four
    /// separate launches (ring fill, `ShortConv`, the clip or norm, `PackBf16` and the plane fill).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_finish_into(
        &self, shape: crate::causal_ops::Shape, finish: crate::causal_ops::Op, param: f32, ring: &mut DeviceVector<'_>,
        made: Option<&DeviceVector<'_>>, step: &DeviceVector<'_>, taps: &DeviceVector<'_>, product: &DeviceVector<'_>,
        plane: &mut DeviceVector<'_>, plane_word: usize, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        use crate::{block_ops::{self, device::Op as Arm}, block_stream::{CONV_ADDEND, CONV_TOKEN_TAPS}, causal_ops::Op, inkling_launch};
        self.require_resident_stream()?;
        let (heads, dim) = (shape.heads as usize, shape.dim as usize);
        let width = heads * dim;
        let arm = match finish { Op::HeadClip => Arm::HeadClip, Op::RoundedNorm => Arm::RoundedNorm, _ => return Err("conv finish: the finish is no clip or norm".into()) };
        if shape.op != Op::ShortConv || dim != block_ops::SUM_THREADS || heads == 0 || shape.cols & CONV_ADDEND != 0
            || shape.cols & !CONV_TOKEN_TAPS != 0 || (shape.cols & CONV_TOKEN_TAPS != 0) != made.is_some() || param.is_nan() || param <= 0.0 {
            return Err("conv finish: shape outside the one-launch form".into());
        }
        let operand = |v: &DeviceVector<'_>| crate::block_check::Operand { address: v.device_address(), words: v.len(), contiguous: !v.is_segmented(), own_context: self.check_vector(v, v.len()).is_ok() };
        let (_, _, units, words) = crate::block_check::operands(block_ops::checked_lengths(shape), [Some(operand(ring)), made.map(operand), None, Some(operand(step))],
            Some(operand(taps)), operand(output), self.arm_capture())?;
        if units * words != width || !width.is_multiple_of(2) { return Err("conv finish: the output differs from the width".into()); }
        let row_end = plane_word.checked_add(width / 2).ok_or("conv finish: plane row overflow")?;
        self.check_vector(product, width)?;
        self.check_vector(plane, plane.len())?;
        if product.is_segmented() || row_end > plane.len() || !plane.writable_from(plane_word) || !ring.writable_from(0) || !output.writable_from(0) {
            return Err("conv finish: product, plane row or ring is not private contiguous storage".into());
        }
        // Byte ranges: the ring and the plane row are written, so neither may meet any other operand or each other.
        let span = |v: &DeviceVector<'_>| (v.device_address(), v.device_address() + 4 * v.len() as u64);
        let row = (plane.device_address() + 4 * plane_word as u64, plane.device_address() + 4 * row_end as u64);
        let meet = |a: (u64, u64), b: (u64, u64)| a.0 < b.1 && b.0 < a.1;
        let read = [Some(span(step)), Some(span(taps)), Some(span(product)), made.map(span)];
        for written in [span(ring), row, span(output)] {
            if read.iter().flatten().any(|&r| meet(written, r)) { return Err("conv finish: a written range meets an operand".into()); }
        }
        if meet(span(ring), row) || meet(span(ring), span(output)) || meet(row, span(output)) { return Err("conv finish: written ranges meet".into()); }
        self.require_arm_kernels(&[Kind::BlockArms.kernel()])?;
        self.resident_completion()?;
        let args = crate::causal_ops::Arguments {
            input: [ring.device_address(), made.map_or(4, |v| v.device_address()), 4, step.device_address()],
            weight: taps.device_address(), output: output.device_address(), shape,
        };
        let mut command = crate::resident_launch_encode::causal(args);
        command.kind = Kind::BlockArms as u32;
        command.reserved = [arm as u32, param.to_bits(), inkling_launch::CONV_FINISH];
        command.word(inkling_launch::FINISH_PRODUCT_WORD, product.device_address());
        command.word(inkling_launch::FINISH_PLANE_WORD, row.0);
        let grid = [u32::try_from(heads).map_err(|_| "conv finish: grid overflow")?, 1, 1, block_ops::SUM_THREADS as u32, 0];
        if !self.record_resident_block(Kind::BlockArms, command, grid)? {
            return Err("conv finish: launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
