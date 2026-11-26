//! The small-batch BF16 product on the device (`few_product`; kernel body `kernels-inkling/src/plain_staged_few.rs`,
//! in the Inkling image; Ruling R6h, lead lift 2026-11-23 for this one entry, kind 101).
//!
//! Safe code only: the host check of every extent and every alignment, then the record of the launch. The body has
//! one entry, the recorded one (`Kind::PlainStagedFew`), so a call while launches are not recorded is refused; the
//! caller asks `staged_few_ready` first and keeps the rows tile or the row product for the other case.
use super::*;
use crate::few_product;
use crate::resident_launch::{Command, Kind};
use crate::staged_product::Span;

pub(super) struct Panel<'a> {
    // Retain the backing allocation so a raw address cannot resolve to stale
    // panels after a model or fixture drops its original view.
    source: DeviceVector<'a>,
    encoded: DeviceVector<'a>,
    columns: usize,
    tiles: usize,
    outliers: usize,
    slices: usize,
}

impl<'a> Backend<'a> {
    /// Encode a startup BF16 matrix into lossless Vade-BF11-BP panels.
    pub fn register_bf16_panels(
        &self, weight: &DeviceVector<'a>, rows: usize, cols: usize, bytes: &[u8],
    ) -> Result<usize, String> {
        use crate::bf16_panels;
        if self.resident_launches_active() || self.arm_capture() || weight.is_segmented() {
            return Err("BF16 panels require an idle resident stream and contiguous weights".into());
        }
        let raw = rows.checked_mul(cols).ok_or("BF16 panels: matrix size overflow")?;
        self.check_vector(weight, raw / 2)?;
        let address = weight.device_address();
        if self.inkling.panels.borrow().contains_key(&address) {
            return Err("BF16 panels: matrix already registered".into());
        }
        let source = weight.resident_range(0..raw / 2)?;
        let (encoded, tiles, outliers) = bf16_panels::encode_with_stats(rows, cols, bytes)?;
        let slices = bf16_panels::slice_count(rows, cols / bf16_panels::CHUNK_COLS);
        let panel = self.vector(encoded.len() / 4)?.into_resident_shared()?;
        // Keep the pinned staging bounded for the vocabulary head.
        let mut word = 0;
        for part in encoded.chunks(8 << 20) {
            let end = word + part.len() / 4;
            self.write_vector_bytes(&mut panel.resident_range(word..end)?, part)?;
            word = end;
        }
        self.inkling.panels.borrow_mut().insert(address, Panel { source, encoded: panel, columns: cols, tiles, outliers, slices });
        Ok(encoded.len())
    }

    /// Tile and raw-band counts of the registered model weights.
    pub fn bf16_panel_tiles(&self) -> (usize, usize) {
        self.inkling.panels.borrow().values().fold((0, 0), |(tiles, raw), matrix| {
            (tiles + matrix.tiles, raw + matrix.outliers)
        })
    }

    pub fn staged_few_has_panels(&self, weight: &DeviceVector<'_>, cols: usize) -> bool {
        self.few_panel(weight, cols, weight.len().saturating_mul(2) / cols.max(1)).is_some()
    }

    fn few_panel(&self, weight: &DeviceVector<'_>, cols: usize, stride: usize) -> Option<(u64, usize, usize)> {
        if crate::staged_narrow::shape_admitted(stride, cols) { return None; }
        self.panel_at(weight, cols)
    }

    /// The base device address, starting 16-row band, and encoded K-slice count (`panel.slices`) of the registered
    /// Vade-BF11-BP encoding covering `weight` (at a 16-row band boundary). Always launches `panel.slices` rather
    /// than recomputing from sub-view stride so sub-views inside stacked matrices never mismatch `slices_enc`.
    fn panel_at(&self, weight: &DeviceVector<'_>, cols: usize) -> Option<(u64, usize, usize)> {
        let address = weight.device_address();
        let panels = self.inkling.panels.borrow();
        let (&base, panel) = panels.range(..=address).next_back()?;
        let offset = usize::try_from(address - base).ok()?;
        if panel.columns != cols || !offset.is_multiple_of(cols * 2) || offset / 4 + weight.len() > panel.source.len() {
            return None;
        }
        let row = offset / (2 * cols);
        if !row.is_multiple_of(16) { return None; }
        Some((panel.encoded.device_address(), row / 16, panel.slices))
    }

    /// The Vade-BF11-BP panel form of `PlainStagedOne` for 1 to 8 input rows: returns
    /// `(panels_base, first_band, slices, part_chunks, warps)` using `panel.slices` directly.
    #[allow(clippy::type_complexity, reason = "panel base, first band, slices, staged chunks, warps per block")]
    fn one_panel_form(&self, rows: usize, cols: usize, inputs: usize, stride: usize, weight: &DeviceVector<'_>, force: Option<Option<usize>>) -> Result<Option<(u64, usize, usize, usize, usize)>, String> {
        let admitted = if inputs == 1 { few_product::one_panel_admitted(rows, cols) } else { few_product::one_panel_rows_admitted(rows, cols, inputs) };
        let sms = self.multiprocessor_count().max(1);
        let bands = rows.div_ceil(16);
        let plan = |(data, first_band, slices): (u64, usize, usize), given: Option<usize>| -> Option<(u64, usize, usize, usize, usize)> {
            let chunks = cols / crate::bf16_panels::CHUNK_COLS;
            if !matches!(slices, 1 | 4) || !chunks.is_multiple_of(slices) { return None; }
            let warps = if slices == 4 || force.is_some() {
                4
            } else if bands <= 3 * sms {
                1
            } else if bands <= 12 * sms {
                2
            } else {
                4
            };
            let blocks = if slices == 4 { bands } else { bands.div_ceil(warps) };
            let max_blk = match warps { 4 => 6, 2 => 12, _ => 16 };
            let eff_blocks = if (slices == 4 || warps == 1) && blocks <= 96 { blocks * 2 } else { blocks };
            let budget = few_product::one_panel_budget_capped(eff_blocks, sms, max_blk);
            let part_chunks = if let Some(part_tiles) = given {
                if part_tiles == 0 || !part_tiles.is_multiple_of(4) { return None; }
                let pc = part_tiles / 4;
                if slices == 4 {
                    let mut chosen = pc.max(4) / 4 * 4;
                    while chosen >= 4 {
                        if chunks.is_multiple_of(chosen)
                            && few_product::one_panel_shared_slices(inputs, chosen, 4) <= few_product::ONE_SHARED_MAX
                        {
                            break;
                        }
                        chosen -= 4;
                    }
                    (chosen >= 4).then_some(chosen)?
                } else {
                    (chunks.is_multiple_of(pc) && few_product::one_panel_shared_slices(inputs, pc, 1) <= few_product::ONE_SHARED_MAX).then_some(pc)?
                }
            } else {
                few_product::one_panel_part_chunks(inputs, chunks, budget, slices)?
            };
            Some((data, first_band, slices, part_chunks, warps))
        };
        match force {
            Some(given) => {
                let found = self.panel_at(weight, cols).filter(|_| admitted).ok_or("one-row panels: no registered panel form for these rows")?;
                Ok(Some(plan(found, given).ok_or("one-row panels: no staged part fits the shared budget")?))
            }
            None => {
                if !admitted || bands < sms { return Ok(None); }
                Ok(self.few_panel(weight, cols, stride).and_then(|found| plan(found, None)))
            }
        }
    }
    /// The words of a Vade-BF11-BP panel launch (`word 7` in `1..=4`) and its grid.
    fn one_panel_command(command: &mut Command, rows: usize, inputs: usize, (data, first_band, slices, part_chunks, warps): (u64, usize, usize, usize, usize)) -> [u32; 5] {
        let mode = match slices { 8 => 3, 6 => 4, 4 => 2, _ => 1 };
        command.word(7, mode);
        command.word(few_product::ONE_PANEL_WORD, data);
        command.word(few_product::ONE_PANEL_WORD + 1, first_band as u64);
        command.word(few_product::ONE_PART_WORD, part_chunks as u64);
        let blocks = if slices > 1 { rows.div_ceil(16) } else { rows.div_ceil(16 * warps) };
        let shared = few_product::one_panel_shared_slices(inputs, part_chunks, slices);
        [blocks as u32, 1, 1, (warps * 32) as u32, shared as u32]
    }
    fn few_panel_command(&self, _command: &mut Command, _weight: &DeviceVector<'_>, _cols: usize, _inputs: usize, _stride: usize) {
        // Untouched `plain_staged_few` uses the legacy 12-bpw tile format; all 1..=8-row Vade-BF11-BP panel launches
        // run through `Kind::PlainStagedOne` (`one_panel_form`).
    }
    /// True when a single Vade-BF11-BP panel launch handles the entire matrix (including the 200k-row vocabulary head)
    /// for `1..=8` input rows.
    pub fn staged_one_panel_ready(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        few_product::one_panel_rows_admitted(rows, cols, inputs) && rows >= 512
            && !input.is_segmented() && !weight.is_segmented()
            && input.len() == inputs * cols && weight.len() == rows * cols / 2
            && (input.device_address() | weight.device_address()) & 15 == 0
            && self.resident_launches_active()
            && self.few_panel(weight, cols, rows).is_some()
    }
    /// Direct register-ring panel form (`FEW_PANEL_RING`) for `9..=16` input rows: disabled while registered panels
    /// use the Vade-BF11-BP layout (all `1..=8`-row panel launches run through `Kind::PlainStagedOne` and `9..=16`-row
    /// dense products route through wave-matched `Kind::PlainTcRows`).
    fn few_panel_ring_form(&self, _rows: usize, _cols: usize, _inputs: usize, _stride: usize, _weight: &DeviceVector<'_>) -> Option<(u64, u64, usize, usize)> {
        None
    }
    fn few_panel_ring_command(command: &mut Command, rows: usize, cols: usize, inputs: usize, flags: u64, (data, meta, part, warps): (u64, u64, usize, usize)) -> [u32; 5] {
        command.word(7, part as u64);
        command.word(8, flags | few_product::FEW_PANEL_RING);
        let word = crate::bf16_panels::FEW_PANEL_WORD;
        command.word(word, data);
        command.word(word + 1, 0);
        command.word(word + 2, meta);
        [rows.div_ceil(16 * warps) as u32, 1, 1, (warps * 32) as u32, few_product::one_panel_shared_warps(inputs, cols, part, warps) as u32]
    }
    /// True when `project_staged_few_pair_into` runs this dual-input product now.
    #[allow(clippy::too_many_arguments, reason = "shape and three operands")]
    pub fn staged_few_pair_ready(&self, rows: usize, cols: usize, inputs: usize, input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        few_product::few_shape_admitted(rows, cols, inputs) && cols.is_multiple_of(2 * crate::staged_product::STAGE)
            && !input1.is_segmented() && !input2.is_segmented() && !weight.is_segmented()
            && input1.len() == inputs * (cols / 2) && input2.len() == inputs * (cols / 2) && weight.len() == rows * cols / 2
            && (input1.device_address() | input2.device_address() | weight.device_address()) & 15 == 0
            && self.resident_launches_active()
    }
    /// Dual-input small-batch BF16 product over `[input1[t], input2[t]]` with optional BF16 output rounding.
    #[allow(clippy::too_many_arguments, reason = "shape, rounding flag and four operands")]
    pub fn project_staged_few_pair_into(
        &self, rows: usize, cols: usize, inputs: usize, round_out: bool,
        input1: &DeviceVector<'_>, input2: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("few pair: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        few_product::few_check_pair(rows, cols, inputs, span(weight), span(input1), span(input2), span(output))?;
        let half = cols / 2;
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input1, inputs * half)?;
        self.check_vector(input2, inputs * half)?;
        self.check_vector(output, inputs * rows)?;
        if weight.is_segmented() || input1.is_segmented() || input2.is_segmented() { return Err("few pair: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("few pair: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainStagedFew.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "few pair: a count exceeds the ABI");
        let mut command = Command::new(Kind::PlainStagedFew);
        for (index, value) in [
            weight.device_address(), input1.device_address(), output.device_address(),
            word(rows)?, word(cols)?, word(inputs)?, word(rows)?,
            input2.device_address(), if round_out { few_product::FEW_ROUND_OUT } else { 0 }, 0,
        ].into_iter().enumerate() { command.word(index, value); }
        self.few_panel_command(&mut command, weight, cols, inputs, rows);
        let (blocks, threads) = few_product::few_geometry(rows, cols, rows);
        if !self.record_resident_block(Kind::PlainStagedFew, command, [blocks, 1, 1, threads, few_product::FEW_BLOCK_BYTES as u32])? {
            return Err("few pair: launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// True when `project_staged_few_group_pair_into` runs 1 to 3 grouped dual-input narrow matrices in one launch.
    pub fn staged_few_group_pair_ready(
        &self, rows: usize, cols: usize, inputs: usize,
        items: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>)],
    ) -> bool {
        (1..=3).contains(&items.len()) && crate::staged_narrow::shape_admitted(rows, cols)
            && items.iter().all(|&(w, x1, x2)| self.staged_few_pair_ready(rows, cols, inputs, x1, x2, w))
    }
    /// Grouped dual-input narrow `PLAIN_STAGED_FEW`: runs 1..=3 narrow matrices of one shape in one launch (`grid.y = items.len()`),
    /// writing `[groups][inputs][rows]` into `output` with optional fused BF16 output rounding.
    pub fn project_staged_few_group_pair_into(
        &self, rows: usize, cols: usize, inputs: usize, round_out: bool,
        items: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>)], output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("few group pair: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        let spans: Vec<(Span, Span, Span)> = items.iter().map(|&(w, a, b)| (span(w), span(a), span(b))).collect();
        few_product::few_check_group_pair(rows, cols, inputs, &spans, span(output))?;
        for &(w, x1, x2) in items {
            self.check_vector(w, rows * cols / 2)?;
            self.check_vector(x1, inputs * (cols / 2))?;
            self.check_vector(x2, inputs * (cols / 2))?;
            if w.is_segmented() || x1.is_segmented() || x2.is_segmented() { return Err("few group pair: operand must be contiguous".into()); }
        }
        self.check_vector(output, items.len() * inputs * rows)?;
        if output.is_segmented() || !output.writable_from(0) { return Err("few group pair: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainStagedFew.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "few group pair: a count exceeds the ABI");
        let mut command = Command::new(Kind::PlainStagedFew);
        let (w0, a0, b0) = items[0];
        let flags = few_product::FEW_GROUP_PAIR | if round_out { few_product::FEW_ROUND_OUT } else { 0 };
        for (index, value) in [
            w0.device_address(), a0.device_address(), output.device_address(),
            word(rows)?, word(cols)?, word(inputs)?, word(rows)?,
            b0.device_address(), flags, 0,
        ].into_iter().enumerate() { command.word(index, value); }
        for (g, &(w, a, b)) in items.iter().enumerate().skip(1) {
            let base = few_product::FEW_GROUP_WORD + (g - 1) * 3;
            command.word(base, w.device_address());
            command.word(base + 1, a.device_address());
            command.word(base + 2, b.device_address());
        }
        let (blocks, threads) = few_product::few_geometry(rows, cols, rows);
        if !self.record_resident_block(Kind::PlainStagedFew, command, [blocks, items.len() as u32, 1, threads, few_product::FEW_BLOCK_BYTES as u32])? {
            return Err("few group pair: launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// Split-output small-batch BF16 product writing directly into `1..=8` contiguous `[inputs][width_i]` outputs.
    pub fn project_staged_few_split_into(
        &self, rows: usize, cols: usize, inputs: usize,
        input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        self.few_split_into(rows, cols, inputs, input, weight, outputs, None)
    }
    /// The panel form of `project_staged_few_split_into` (1 to 8 input rows), whatever the rows (the device checks),
    /// staging `part` column tiles of input at once when given.
    #[allow(clippy::too_many_arguments, reason = "the shape, the staged part and the operands")]
    pub fn project_staged_panels_split_into(
        &self, rows: usize, cols: usize, inputs: usize, part: Option<usize>,
        input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        self.few_split_into(rows, cols, inputs, input, weight, outputs, Some(part))
    }
    #[allow(clippy::too_many_arguments, reason = "the shape, the operands and the panel form")]
    fn few_split_into(
        &self, rows: usize, cols: usize, inputs: usize,
        input: &DeviceVector<'_>, weight: &DeviceVector<'_>, outputs: &mut [(&mut DeviceVector<'_>, usize)], force_panel: Option<Option<usize>>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("few split: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        let splits: Vec<(Span, usize)> = outputs.iter().map(|(v, w)| (span(v), *w)).collect();
        few_product::few_check_split(rows, cols, inputs, span(weight), span(input), &splits)?;
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, inputs * cols)?;
        if weight.is_segmented() || input.is_segmented() { return Err("few split: operand must be contiguous".into()); }
        for (v, w) in outputs.iter() {
            self.check_vector(v, inputs * *w)?;
            if v.is_segmented() || !v.writable_from(0) { return Err("few split: output requires private contiguous storage".into()); }
        }
        // A panel form of 1 to 8 rows (`one_panel_form`) runs in the one-row entry: its flags in word 8 and the count in
        // word 9, the outputs from `ONE_SPLIT_WORD`. Every other split launch: the few-row entry.
        let panel = if inputs <= few_product::ONE_PANEL_INPUTS { self.one_panel_form(rows, cols, inputs, rows, weight, force_panel)? } else { None };
        if force_panel.is_some() && panel.is_none() { return Err("one-row panels: the panel form takes 1 to 8 input rows".into()); }
        let kind = if panel.is_some() { Kind::PlainStagedOne } else { Kind::PlainStagedFew };
        self.require_arm_kernels(&[kind.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "few split: a count exceeds the ABI");
        let mut command = Command::new(kind);
        let grid = if let Some(panel) = panel {
            for (index, value) in [weight.device_address(), input.device_address(), 0, word(rows)?, word(cols)?, word(inputs)?, word(rows)?]
                .into_iter().enumerate() { command.word(index, value); }
            command.word(few_product::ONE_FLAGS_WORD, few_product::FEW_SPLIT_OUT);
            command.word(few_product::ONE_FLAGS_WORD + 1, outputs.len() as u64);
            for (s_i, (v, w)) in outputs.iter().enumerate() {
                command.word(few_product::ONE_SPLIT_WORD + 2 * s_i, v.device_address());
                command.word(few_product::ONE_SPLIT_WORD + 2 * s_i + 1, word(*w)?);
            }
            Self::one_panel_command(&mut command, rows, inputs, panel)
        } else {
            for (index, value) in [
                weight.device_address(), input.device_address(), 0,
                word(rows)?, word(cols)?, word(inputs)?, word(rows)?,
                0, few_product::FEW_SPLIT_OUT, outputs.len() as u64,
            ].into_iter().enumerate() { command.word(index, value); }
            for (s_i, (v, w)) in outputs.iter().enumerate() {
                command.word(10 + 2 * s_i, v.device_address());
                command.word(10 + 2 * s_i + 1, word(*w)?);
            }
            if let Some(ring) = self.few_panel_ring_form(rows, cols, inputs, rows, weight) {
                Self::few_panel_ring_command(&mut command, rows, cols, inputs, few_product::FEW_SPLIT_OUT, ring)
            } else {
                self.few_panel_command(&mut command, weight, cols, inputs, rows);
                let (blocks, threads) = few_product::few_geometry(rows, cols, rows);
                [blocks, 1, 1, threads, few_product::FEW_BLOCK_BYTES as u32]
            }
        };
        if !self.record_resident_block(kind, command, grid)? {
            return Err("few split: launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// True when `project_staged_few_into` runs this product now: 1 to 16 input rows, the shape inside the envelope,
    /// the weights and the input contiguous, of their exact lengths and 16-byte aligned, and launches recorded.
    pub fn staged_few_ready(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        few_product::few_shape_admitted(rows, cols, inputs) && !input.is_segmented() && !weight.is_segmented()
            && input.len() == inputs * cols && weight.len() == rows * cols / 2
            && (input.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }
    /// The product of one BF16 matrix (`rows x cols` values, two a word, row-major) with `inputs` input rows
    /// (`[inputs][cols]`, 1 to 16 rows): input row `t` goes to the words `t * stride .. t * stride + rows` of
    /// `output`, which holds `(inputs - 1) * stride + rows` words. Use the full matrix row count as `stride` so
    /// all parts of a larger matrix use the same reduction order.
    #[allow(clippy::too_many_arguments, reason = "the shape, the stride and the three operands")]
    pub fn project_staged_few_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.few_into(rows, cols, inputs, stride, input, weight, output, None)
    }
    /// The one-row panel form of `project_staged_few_into`, whatever the rows (the device checks): `weight` must view
    /// rows of a matrix with registered panels from a 16-row band boundary, and the shape must take the panel form.
    pub fn project_staged_one_panels_into(&self, rows: usize, cols: usize, stride: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.few_into(rows, cols, 1, stride, input, weight, output, Some(None))
    }
    /// The panel form of `project_staged_few_into` for 1 to 8 input rows, whatever the rows (the device checks), staging
    /// `part` column tiles of input at once when given (a whole number of ring turns that divides the tiles).
    #[allow(clippy::too_many_arguments, reason = "the shape, the stride, the staged part and the three operands")]
    pub fn project_staged_panels_rows_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, part: Option<usize>, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.few_into(rows, cols, inputs, stride, input, weight, output, Some(part))
    }
    #[allow(clippy::too_many_arguments, reason = "the shape, the stride, the three operands and the panel form")]
    fn few_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, force_panel: Option<Option<usize>>) -> Result<(), String> {
        // A recorded command inside any stream capture would tie the graph to the recorder's mutable stage: refused,
        // as the other staged products are.
        if self.arm_capture() { return Err("few product: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        // Every extent, every alignment and the stride that the body relies on, before anything is recorded.
        if force_panel.is_some() {
            few_product::one_panel_check(rows, cols, inputs, stride, span(weight), span(input), span(output))?;
        } else {
            few_product::few_check(rows, cols, inputs, stride, span(weight), span(input), span(output))?;
        }
        // The owners: this context, the exact lengths, contiguous storage, and an output that is private.
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, inputs * cols)?;
        self.check_vector(output, (inputs - 1) * stride + rows)?;
        if weight.is_segmented() || input.is_segmented() { return Err("few product: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("few product: the output requires private contiguous storage".into()); }
        // One input row, or a panel form of 1 to 8 rows: the one-row entry (dynamic shared memory only). Other launches
        // of more rows: the few-row entry.
        let panel = if inputs <= few_product::ONE_PANEL_INPUTS { self.one_panel_form(rows, cols, inputs, stride, weight, force_panel)? } else { None };
        if force_panel.is_some() && panel.is_none() { return Err("one-row panels: the panel form takes 1 to 8 input rows".into()); }
        let kind = if inputs == 1 || panel.is_some() { Kind::PlainStagedOne } else { Kind::PlainStagedFew };
        self.require_arm_kernels(&[kind.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "few product: a count exceeds the ABI");
        let mut command = Command::new(kind);
        for (index, value) in [weight.device_address(), input.device_address(), output.device_address(), word(rows)?, word(cols)?, word(inputs)?, word(stride)?].into_iter().enumerate() { command.word(index, value); }
        let grid = if let Some(panel) = panel {
            // The panel form (`one_panel_form`): one 64-row group a block.
            Self::one_panel_command(&mut command, rows, inputs, panel)
        } else if inputs == 1 {
            let (blocks, threads) = few_product::few_geometry(rows, cols, stride);
            // A wide matrix of whole 256-column stages streams one 16-row MMA tile a block.
            let blocks = if threads == few_product::FEW_THREADS as u32 && cols.is_multiple_of(256) { rows.div_ceil(16) as u32 } else { blocks };
            [blocks, 1, 1, threads, few_product::ONE_BLOCK_BYTES as u32]
        } else if let Some(ring) = self.few_panel_ring_form(rows, cols, inputs, stride, weight) {
            Self::few_panel_ring_command(&mut command, rows, cols, inputs, 0, ring)
        } else {
            self.few_panel_command(&mut command, weight, cols, inputs, stride);
            let (blocks, threads) = few_product::few_geometry(rows, cols, stride);
            [blocks, 1, 1, threads, few_product::FEW_BLOCK_BYTES as u32]
        };
        if !self.record_resident_block(kind, command, grid)? {
            return Err("few product: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
