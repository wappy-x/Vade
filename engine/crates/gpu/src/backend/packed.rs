//! The packed row product on the device (`packed_product`; kernel body `kernels-inkling/src/plain_packed.rs` in the Inkling image; Ruling R6h).
//!
//! Safe code only: the host check of every extent and every alignment, then the record of the launch. The body has
//! one entry, the recorded one (`Kind::PlainPacked`), so a call while launches are not recorded is refused; the
//! caller asks `packed_product_ready` first and keeps the BF16 product for the other case.
use super::*;
use crate::packed_product::{self, Span};

impl<'a> Backend<'a> {
    /// True when `project_packed_into` runs a matrix of this shape now: the shape is inside the tile and launches
    /// are recorded.
    pub fn packed_product_ready(&self, rows: usize, cols: usize, wide: bool, pairs: bool) -> bool {
        packed_product::shape_admitted(rows, cols, wide, pairs) && self.resident_launches_active()
    }
    /// The product of one packed matrix with one input row. `unit` is the unit of the matrix
    /// (`packed_product::layout`: codes, block scales, weight table), `wide` says 8-bit codes, `pairs` says that
    /// gate rows and up rows are mixed in storage. `output` gets `rows` words: the words of
    /// `project_bf16_chain_into` on the matrix widened to BF16, gate rows first.
    #[allow(clippy::too_many_arguments, reason = "the five words of the launch and its three operands")]
    pub fn project_packed_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, input: &DeviceVector<'_>, unit: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        // A recorded command inside any stream capture would tie the graph to the recorder's mutable stage (review
        // REVPACK finding 1, as REVPROD finding 3 for the staged products): refused.
        if self.arm_capture() { return Err("packed product: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        // Every extent and every alignment that the body relies on, before anything is recorded.
        packed_product::check(rows, cols, wide, pairs, span(unit), span(input), span(output))?;
        // The owners: this context, the exact lengths, contiguous storage, and an output that is private.
        self.check_vector(unit, packed_product::layout(rows, cols, wide).bytes / 4)?;
        self.check_vector(input, cols)?;
        self.check_vector(output, rows)?;
        if unit.is_segmented() || input.is_segmented() { return Err("packed product: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("packed product: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[crate::resident_launch::Kind::PlainPacked.kernel()])?;
        self.resident_completion()?;
        let rows = u32::try_from(rows).map_err(|_| "packed product: the rows exceed the ABI")?;
        if !self.record_resident_packed(unit.device_address(), input.device_address(), output.device_address(), rows, packed_product::cols_word(cols, wide, pairs))? {
            return Err("packed product: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// The products of `units.len()` packed matrices of one shape in ONE launch: matrix `e` times its input (the one
    /// `input` for all when `shared`, else words `e * cols ..` of `input`) into words `e * rows ..` of `output`. Each
    /// block does what a block of `project_packed_into` does, so the words are the words of one call a matrix.
    #[allow(clippy::too_many_arguments, reason = "the shape, the operands and the input form")]
    pub fn project_packed_group_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, input: &DeviceVector<'_>, shared: bool, units: &[&DeviceVector<'_>], output: &mut DeviceVector<'_>) -> Result<(), String> {
        // Refused inside any stream capture, as `project_packed_into` (review REVPACK finding 1).
        if self.arm_capture() { return Err("packed group: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        let spans = units.iter().map(|u| span(u)).collect::<Vec<_>>();
        packed_product::check_group(rows, cols, wide, pairs, &spans, span(input), shared, span(output))?;
        let picks = units.len();
        for unit in units {
            self.check_vector(unit, packed_product::layout(rows, cols, wide).bytes / 4)?;
            if unit.is_segmented() { return Err("packed group: an operand must be contiguous".into()); }
        }
        self.check_vector(input, if shared { cols } else { cols * picks })?;
        self.check_vector(output, rows * picks)?;
        if input.is_segmented() { return Err("packed group: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("packed group: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[crate::resident_launch::Kind::PlainPackedGroup.kernel()])?;
        self.resident_completion()?;
        let rows = u32::try_from(rows).map_err(|_| "packed group: the rows exceed the ABI")?;
        let input_words = if shared { 0 } else { u32::try_from(cols).map_err(|_| "packed group: the columns exceed the ABI")? };
        let addresses = units.iter().map(|u| u.device_address()).collect::<Vec<_>>();
        if !self.record_resident_packed_group(&addresses, input.device_address(), output.device_address(), rows, packed_product::cols_word(cols, wide, pairs), input_words)? {
            return Err("packed group: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// True when `project_packed_picked_into` runs a matrix of this shape now: as `packed_product_ready`.
    pub fn packed_picked_ready(&self, rows: usize, cols: usize, wide: bool, pairs: bool) -> bool { self.packed_product_ready(rows, cols, wide, pairs) }
    /// The products of the experts that the route picked, in ONE launch, with the expert numbers read on the device:
    /// `ids` is `picks` words of the route output (FP32 whole numbers, in pick order), `stack` the units of all
    /// `experts` experts of one layer and projection back to back. Pick `e` multiplies the unit of expert `ids[e]` with
    /// its input (the one `input` for all when `shared`, else words `e * cols ..`) into words `e * rows ..` of
    /// `output`: the words of `project_packed_group_into` with those units. The host reads no expert number; a number
    /// that is not a whole number below `experts` stops the device (a trap).
    #[allow(clippy::too_many_arguments, reason = "the shape, the stack, the numbers and the operands")]
    pub fn project_packed_picked_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, input: &DeviceVector<'_>, shared: bool, stack: &DeviceVector<'_>, experts: usize, ids: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_packed_picked_gate_into(rows,cols,wide,pairs,input,shared,stack,experts,ids,output,None)
    }
    /// A fused gate writes the activated words contiguously in the first half of the checked output.
    #[allow(clippy::too_many_arguments)]
    pub fn project_packed_picked_gate_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, input: &DeviceVector<'_>, shared: bool, stack: &DeviceVector<'_>, experts: usize, ids: &DeviceVector<'_>, output: &mut DeviceVector<'_>,caps:Option<[f32;4]>) -> Result<(), String> {
        self.project_packed_picked_step_into(rows,cols,wide,pairs,input,shared,stack,experts,ids,output,caps,false,None)
    }

    /// One decoded token (`one_row`): the paired E4M3 conversion of the wide form, and for a narrow stack with a panel
    /// copy (`decode_panels::pack_expert`, same length) the coalesced one-row panel path.
    #[allow(clippy::too_many_arguments)]
    pub fn project_packed_picked_step_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, input: &DeviceVector<'_>, shared: bool, stack: &DeviceVector<'_>, experts: usize, ids: &DeviceVector<'_>, output: &mut DeviceVector<'_>,caps:Option<[f32;4]>,one_row:bool,panels:Option<&DeviceVector<'_>>) -> Result<(), String> {
        let picks = ids.len();
        self.project_packed_picked_strided_into(rows, cols, wide, pairs, input, if shared { 0 } else { 1 }, stack, experts, ids, picks, picks, picks, output, caps, one_row, panels)
    }
    /// Picked packed products with strided route rows (`id_stride >= picks_per_row`) and optional input row
    /// repeat (`repeat == 0` shares one row; `repeat >= 1` maps pick `j` to input row `j / repeat`).
    #[allow(clippy::too_many_arguments)]
    pub fn project_packed_picked_strided_into(
        &self, rows: usize, cols: usize, wide: bool, pairs: bool,
        input: &DeviceVector<'_>, repeat: usize,
        stack: &DeviceVector<'_>, experts: usize,
        ids: &DeviceVector<'_>, picks: usize, picks_per_row: usize, id_stride: usize,
        output: &mut DeviceVector<'_>, caps: Option<[f32; 4]>, one_row: bool, panels: Option<&DeviceVector<'_>>,
    ) -> Result<(), String> {
        self.project_packed_picked_strided_compact_into(rows, cols, wide, pairs, input, repeat, stack, experts, ids, picks, picks_per_row, id_stride, output, caps, one_row, panels, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn project_packed_picked_strided_compact_into(
        &self, rows: usize, cols: usize, wide: bool, pairs: bool,
        input: &DeviceVector<'_>, repeat: usize,
        stack: &DeviceVector<'_>, experts: usize,
        ids: &DeviceVector<'_>, picks: usize, picks_per_row: usize, id_stride: usize,
        output: &mut DeviceVector<'_>, caps: Option<[f32; 4]>, one_row: bool, panels: Option<&DeviceVector<'_>>,
        compact: Option<&DeviceVector<'_>>,
    ) -> Result<(), String> {
        if let Some(c) = caps { if !pairs || !c.iter().all(|v| v.is_finite() && *v > 0.0) { return Err("packed gate requires paired rows and positive finite caps".into()); } }
        if self.arm_capture() { return Err("packed picks: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        packed_product::check_picked_strided(rows, cols, wide, pairs, span(stack), experts, span(ids), picks, picks_per_row, id_stride, span(input), repeat, span(output))?;
        let unit_words = packed_product::layout(rows, cols, wide).bytes / 4;
        let input_rows = picks.checked_div(repeat).unwrap_or(1);
        self.check_vector(stack, experts * unit_words)?;
        self.check_vector(ids, (picks / picks_per_row) * id_stride)?;
        self.check_vector(input, input_rows * cols)?;
        self.check_vector(output, rows * picks)?;
        if stack.is_segmented() || ids.is_segmented() || input.is_segmented() { return Err("packed picks: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("packed picks: the output requires private contiguous storage".into()); }
        if let Some(cmp) = compact {
            self.check_vector(cmp, picks * crate::fast_ops::ROUTER_PICK_COMPACT_ENTRY_WORDS)?;
            let (c, o) = (span(cmp), span(output));
            if cmp.is_segmented() || c.address % 16 != 0
                || (c.address < o.address.saturating_add(o.bytes) && o.address < c.address.saturating_add(c.bytes))
            {
                return Err("packed picks: compact dispatch table requires a disjoint aligned extent".into());
            }
        }
        self.require_arm_kernels(&[crate::resident_launch::Kind::PlainPackedPicked.kernel()])?;
        self.resident_completion()?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "packed picks: the rows exceed the ABI")?;
        let experts_u32 = u32::try_from(experts).map_err(|_| "packed picks: the experts exceed the ABI")?;
        let input_words = if repeat == 0 { 0 } else { u32::try_from(cols).map_err(|_| "packed picks: the columns exceed the ABI")? };
        let (ppr, stride) = if id_stride == picks_per_row { (0, 0) } else { (picks_per_row, id_stride) };
        if let Some(panel) = panels {
            self.check_vector(panel, stack.len())?;
            let (p,o) = (span(panel),span(output));
            if !one_row || wide || !rows.is_multiple_of(64) || cols > packed_product::PANEL_INPUT_COLS || repeat > 1 ||
                panel.is_segmented() || p.address % 16 != 0 ||
                (p.address < o.address.saturating_add(o.bytes) && o.address < p.address.saturating_add(p.bytes)) {
                return Err("packed expert panels require a disjoint aligned single-row extent".into());
            }
        }
        if !self.record_resident_packed_picked(
            stack.device_address(), unit_words as u64 * 4, experts_u32, ids.device_address(),
            picks, input.device_address(), output.device_address(), rows_u32,
            packed_product::cols_word(cols, wide, pairs), input_words, caps,
            if repeat == 0 { 1 } else { repeat }, ppr, stride, one_row,
            panels.map_or(0, |p| p.device_address()),
            compact.map_or(0, |c| c.device_address()),
        )? {
            return Err("packed picks: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// True when `project_packed_rows_grouped_into` takes these runs now: every unit is the packed unit of a matrix of
    /// this shape inside the rows tile, contiguous and 16-byte aligned, the input contiguous and aligned, the staged
    /// tables fit the recorder's stage, and launches are recorded. With tables, two words a launch row are staged.
    #[allow(clippy::too_many_arguments, reason = "the shape, the form, the runs, the input and the table form")]
    pub fn packed_rows_grouped_ready(&self, rows: usize, cols: usize, wide: bool, pairs: bool, runs: &[(&DeviceVector<'_>, usize)], input: &DeviceVector<'_>, tables: bool) -> bool {
        use crate::staged_product as staged;
        let inputs = runs.iter().map(|run| run.1).sum::<usize>();
        let blocks = runs.iter().map(|run| rows.div_ceil(staged::ROWS_TILE) * run.1.div_ceil(staged::ROWS_TILE)).sum::<usize>();
        let bytes = runs.len() * 16 + blocks * 8 + if tables { inputs * 8 } else { 0 };
        let shaped = rows <= staged::ROWS_MATRIX_ROWS_MAX && cols <= staged::ROWS_COLS_MAX && cols > 0 && cols.is_multiple_of(packed_product::whole(wide)) && (!pairs || rows.is_multiple_of(2));
        shaped && !runs.is_empty() && runs.len() <= staged::ROWS_GROUP_RUNS_MAX && inputs <= staged::ROWS_GROUP_INPUTS_MAX
            && bytes <= crate::resident_launch::stage::BYTES / 4 && u32::try_from(blocks).is_ok()
            && runs.iter().all(|&(unit, count)| staged::rows_shape_admitted(rows, cols, count) && !unit.is_segmented() && unit.len() * 4 == packed_product::layout(rows, cols, wide).bytes && unit.device_address() & 15 == 0)
            && !input.is_segmented() && !input.is_empty() && input.len().is_multiple_of(cols.max(1)) && input.device_address() & 15 == 0 && self.resident_launches_active()
    }
    /// The expert runs of a visit in ONE launch of the packed rows tile (kind `PlainPackedRows`): run `r` is the packed
    /// unit `runs[r].0` of one matrix (`packed_product::layout`) with `runs[r].1` launch rows, the runs taking the launch
    /// rows in order. The tables name the input row and the output row of every launch row, as in
    /// `project_staged_rows_grouped_into`. The words are those of that call on the same matrices widened to BF16, gate
    /// rows first: the body widens each weight as the packed row product does and runs the chains of the rows tile.
    /// The recorder stages the run table, the block table and the two row tables with the command, all from host
    /// words this call checked (`packed_product::check_rows_grouped`).
    #[allow(clippy::too_many_arguments, reason = "the shape, the form, the stride, the runs, the operands and the two tables")]
    pub fn project_packed_rows_grouped_into(&self, rows: usize, cols: usize, wide: bool, pairs: bool, stride: usize, runs: &[(&DeviceVector<'_>, usize)], input: &DeviceVector<'_>, input_rows: Option<&[u32]>, output: &mut DeviceVector<'_>, output_rows: Option<&[u32]>) -> Result<(), String> {
        use crate::{resident_launch::Kind, staged_product as staged};
        // Refused inside any stream capture, as the other recorded products (review REVPACK finding 1).
        if self.arm_capture() { return Err("packed rows product: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        let inputs = runs.iter().map(|run| run.1).sum::<usize>();
        let input_count = if input_rows.is_some() && cols > 0 { input.len() / cols } else { inputs };
        let output_count = if output_rows.is_some() && stride > 0 && output.len() >= rows { (output.len() - rows) / stride + 1 } else { inputs };
        let spans = runs.iter().map(|&(unit, count)| (span(unit), count)).collect::<Vec<_>>();
        packed_product::check_rows_grouped(rows, cols, wide, pairs, stride, &spans, span(input), staged::Rows { count: input_count, table: input_rows }, span(output), staged::Rows { count: output_count, table: output_rows })?;
        let unit_words = packed_product::layout(rows, cols, wide).bytes / 4;
        for &(unit, _) in runs {
            self.check_vector(unit, unit_words)?;
            if unit.is_segmented() { return Err("packed rows product: a unit must be contiguous".into()); }
        }
        self.check_vector(input, input_count * cols)?;
        self.check_vector(output, (output_count - 1) * stride + rows)?;
        if input.is_segmented() { return Err("packed rows product: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("packed rows product: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainPackedRows.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "packed rows product: a count exceeds the ABI");
        // The staged bytes: the run table (16 bytes a run), the block table (8 bytes a block), then the row tables.
        let counts = runs.iter().map(|run| run.1).collect::<Vec<_>>();
        let blocks = staged::rows_group_blocks(rows, &counts);
        let grid_blocks = u32::try_from(blocks.len()).map_err(|_| "packed rows product: too many blocks")?;
        let mut bytes = Vec::with_capacity(runs.len() * 16 + blocks.len() * 8 + inputs * 8);
        let mut first = 0u32;
        for &(unit, count) in runs {
            bytes.extend_from_slice(&unit.device_address().to_ne_bytes());
            bytes.extend_from_slice(&first.to_ne_bytes());
            bytes.extend_from_slice(&(count as u32).to_ne_bytes());
            first += count as u32;
        }
        for block in &blocks { for half in block { bytes.extend_from_slice(&half.to_ne_bytes()); } }
        let tables_at = bytes.len() as u64;
        for row in input_rows.into_iter().chain(output_rows).flatten() { bytes.extend_from_slice(&row.to_ne_bytes()); }
        let input_bytes = input_rows.map_or(0, |table| table.len() as u64 * 4);
        let head = [input.device_address(), output.device_address(), word(rows)?, u64::from(packed_product::cols_word(cols, wide, pairs)), word(inputs)?, word(stride)?];
        let (run_count, block_count) = (runs.len() as u64, blocks.len() as u64);
        let recorded = self.record_resident_tabled(Kind::PlainPackedRows, [grid_blocks, 1, 1, staged::ROWS_THREADS as u32, 0], &bytes, |address| {
            let mut command = crate::resident_launch::Command::new(Kind::PlainPackedRows);
            let input_table = if input_rows.is_some() { address + tables_at } else { 0 };
            let output_table = if output_rows.is_some() { address + tables_at + input_bytes } else { 0 };
            // Word 0 stays 0: each run's unit comes from the run table.
            for (index, value) in [0].into_iter().chain(head).chain([input_table, input_count as u64, output_table, output_count as u64, address, run_count, address + run_count * 16, block_count]).enumerate() { command.word(index, value); }
            command
        })?;
        if !recorded {
            return Err("packed rows product: the grouped launch runs only as a recorded launch whose tables fit the stage, and that is not the case now".into());
        }
        self.resident_completion()
    }
}
#[cfg(test)]
mod tests {
    use crate::packed_product;
    use crate::resident_launch::Kind;
    const MARK: u32 = 0x7fc0_dead;
    /// A final partial warp still exchanges complete gate/up pairs and writes only the compact prefix.
    #[test]
    #[ignore = "requires CUDA and the Inkling image"]
    fn paired_gate_matches_separate_activation_for_partial_warps() {
        crate::driver::with_backend(0,crate::layer_ptx().unwrap(),usize::MAX,|b| {
            let (rows,cols)=(34,128);
            let _queue=b.resident_queue()?;
            b.enable_resident_launches()?;
            for wide in [false,true] {
                let layout=packed_product::layout(rows,cols,wide);
                let mut bytes=vec![0u8;2*layout.bytes];
                for (expert,unit) in bytes.chunks_exact_mut(layout.bytes).enumerate() {
                    for (i,byte) in unit[..layout.scales].iter_mut().enumerate() {*byte=(i*13+expert*37) as u8;}
                    unit[layout.scales..layout.table].fill(if wide {127} else {0});
                    if wide {
                        for (i,word) in unit[layout.table..layout.table+1024].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                            *word=((i as f32-128.0)/64.0).to_ne_bytes();
                        }
                    } else {
                        for (i,word) in unit[layout.table..layout.table+32].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                            *word=((((i as f32-8.0)/4.0).to_bits()>>16) as u16).to_ne_bytes();
                        }
                    }
                }
                let stack=bytes.as_chunks::<4>().0.iter().copied().map(f32::from_ne_bytes).collect::<Vec<_>>();
                let stack=b.upload_vector(&stack)?.into_resident_shared()?;
                for (picks,shared) in [(2,true),(packed_product::PICKS_MAX,false)] {
                    // Unequal groups exercise unused lanes and the last partial pair tile. Give
                    // each unshared input distinct words to detect writes into another pick.
                    let route=(0..picks).map(|e|if e%7==0 {0.0} else {1.0}).collect::<Vec<_>>();
                    let ids=b.upload_vector(&route)?.into_resident_shared()?;
                    let inputs=(0..cols*if shared {1} else {picks})
                        .map(|j|((j%251) as f32-67.0)/512.0).collect::<Vec<_>>();
                    let x=b.upload_vector(&inputs)?.into_resident_shared()?;
                    let mut raw=b.vector(rows*picks)?;
                    let mut fused=b.upload_vector(&vec![f32::from_bits(MARK);rows*picks])?;
                    b.project_packed_picked_into(rows,cols,wide,true,&x,shared,&stack,2,&ids,&mut raw)?;
                    let caps=[4.0,25.0,0.25,0.04];
                    b.project_packed_picked_gate_into(rows,cols,wide,true,&x,shared,&stack,2,&ids,&mut fused,Some(caps))?;
                    b.flush_resident_launches()?;
                    let raw=b.read_vector_values(&raw)?;
                    let out=b.read_vector_values(&fused)?;
                    let inner=rows/2;
                    for (e,pair) in raw.chunks_exact(rows).enumerate() {
                        for i in 0..inner {
                            use crate::fixed_math::{rn,sigmoid,tanh};
                            let (g,u)=(rn(pair[i]),rn(pair[inner+i]));
                            let want=rn(((tanh(g*caps[2])*caps[0])*sigmoid(g))*(tanh(u*caps[3])*caps[1]));
                            if want.to_bits()!=out[e*inner+i].to_bits() {return Err(format!("gate mismatch at pick {e}, row {i}"));}
                        }
                    }
                    if out[picks*inner..].iter().any(|v|v.to_bits()!=MARK) {return Err("gate wrote past active words".into());}
                }
            }
            Ok(())
        }).unwrap();
    }
    /// Review REVPACK finding 1 and REVIDS: a packed launch (kind 88), a grouped one (kind 95) and one with the expert
    /// numbers on the device (kind 96) inside a graph
    /// capture and inside a verify segment are refused before anything is recorded, with launch recording off and
    /// on; after them the mark is clear and the same calls run as recorded launches. Every graph capture
    /// (`capture_batch`, `capture_batch_bounded`, and `capture_decode`, which runs `capture_batch`) sets one mark,
    /// in `capture_with_allowance`, while its closure runs; the test sets that same mark, so it needs no unsafe block.
    #[test]
    #[ignore = "skipped: no GPU; requires CUDA and the Inkling image"]
    fn a_packed_product_inside_a_graph_capture_or_a_verify_segment_is_refused() {
        crate::driver::with_backend(0, crate::layer_ptx().unwrap(), usize::MAX, |b| {
            let (rows, cols) = (64usize, 64usize);
            let words = packed_product::layout(rows, cols, false).bytes / 4;
            let units = b.upload_vector(&vec![0.0f32; 2 * words])?.into_resident_shared()?;
            let (first, second) = (units.resident_range(0..words)?, units.resident_range(words..2 * words)?);
            let x = b.upload_vector(&vec![0.5f32; cols])?.into_resident_shared()?;
            let y = b.vector(2 * rows)?.into_resident_shared()?;
            let one = || b.project_packed_into(rows, cols, false, false, &x, &first, &mut y.resident_range(0..rows)?);
            let group = || b.project_packed_group_into(rows, cols, false, false, &x, true, &[&first, &second], &mut y.resident_range(0..2 * rows)?);
            let ids = b.upload_vector(&[1.0f32, 0.0])?.into_resident_shared()?;
            let picked = || b.project_packed_picked_into(rows, cols, false, false, &x, true, &units, 2, &ids, &mut y.resident_range(0..2 * rows)?);
            let refused = |result: Result<(), String>, name: &str| -> Result<(), String> {
                match result { Err(e) if e.contains("stream capture") => Ok(()), Err(e) => Err(format!("{name}: other error {e}")), Ok(()) => Err(format!("{name}: the packed product ran inside a capture")) }
            };
            let launches = || { let kinds = b.stats().recorded_kinds; kinds[Kind::PlainPacked as usize] + kinds[Kind::PlainPackedGroup as usize] + kinds[Kind::PlainPackedPicked as usize] };
            let calls: [(&dyn Fn() -> Result<(), String>, &str); 3] = [(&one, "packed product"), (&group, "packed group"), (&picked, "packed picks")];
            let mut queue = None;
            for recording in [false, true] {
                if recording { queue = Some(b.resident_queue()?); b.enable_resident_launches()?; }
                let before = launches();
                for (call, kind) in calls {
                    // The mark of a graph capture, as capture_with_allowance sets it around its closure.
                    let was = b.mark_graph_capture(true);
                    let result = call();
                    b.mark_graph_capture(was);
                    refused(result, &format!("{kind} graph capture, recording {recording}"))?;
                    // The verify segment: its capture is open between begin_segment and abort_segment.
                    if !b.begin_segment(1, 1)? { return Err("the verify segment did not open".into()); }
                    let result = call();
                    b.abort_segment();
                    refused(result, &format!("{kind} verify segment, recording {recording}"))?;
                }
                if launches() != before { return Err(format!("a refused call recorded {} launches", launches() - before)); }
                if b.arm_capture() { return Err("the capture mark stayed set".into()); }
            }
            one()?;
            group()?;
            picked()?;
            drop(queue);
            Ok(())
        }).unwrap();
    }
    /// Review REVPACK finding 4 and REVIDS: the device calls of kinds 88, 95 and 96 refuse a bad extent, a zero size
    /// and a buffer of another kind (a segmented vector, also as the output), each with an error and no launch
    /// recorded; a good call with the same operands runs.
    #[test]
    #[ignore = "skipped: no GPU; requires CUDA and the Inkling image"]
    fn the_packed_calls_refuse_a_bad_extent_a_zero_size_and_a_segmented_buffer_with_no_launch() {
        crate::driver::with_backend(0, crate::layer_ptx().unwrap(), usize::MAX, |b| {
            let (rows, cols) = (64usize, 64usize);
            let words = packed_product::layout(rows, cols, false).bytes / 4;
            let units = b.upload_vector(&vec![0.0f32; 2 * words + 4])?.into_resident_shared()?;
            let (first, second) = (units.resident_range(0..words)?, units.resident_range(words..2 * words)?);
            let short = units.resident_range(words..2 * words - 1)?;
            let x = b.upload_vector(&vec![0.5f32; cols])?.into_resident_shared()?;
            let y = b.upload_vector(&vec![f32::from_bits(MARK); 2 * rows])?.into_resident_shared()?;
            let segmented = b.segmented_vector(words, words)?;
            let (segmented_stack, segmented_output) = (b.segmented_vector(2 * words, 2 * words)?, b.segmented_vector(2 * rows, 2 * rows)?);
            let stack = units.resident_range(0..2 * words)?;
            let ids = b.upload_vector(&[1.0f32, 0.0])?.into_resident_shared()?;
            let _queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            let launches = || { let kinds = b.stats().recorded_kinds; kinds[Kind::PlainPacked as usize] + kinds[Kind::PlainPackedGroup as usize] + kinds[Kind::PlainPackedPicked as usize] };
            let before = launches();
            let bad: Vec<(&str, Result<(), String>)> = vec![
                ("one: a unit one word short", b.project_packed_into(rows, cols, false, false, &x, &short, &mut y.resident_range(0..rows)?)),
                ("one: an output one word short", b.project_packed_into(rows, cols, false, false, &x, &first, &mut y.resident_range(0..rows - 1)?)),
                ("one: no rows", b.project_packed_into(0, cols, false, false, &x, &first, &mut y.resident_range(0..rows)?)),
                ("one: no columns", b.project_packed_into(rows, 0, false, false, &x, &first, &mut y.resident_range(0..rows)?)),
                ("one: a segmented unit", b.project_packed_into(rows, cols, false, false, &x, &segmented, &mut y.resident_range(0..rows)?)),
                ("group: a unit one word short", b.project_packed_group_into(rows, cols, false, false, &x, true, &[&first, &short], &mut y.resident_range(0..2 * rows)?)),
                ("group: no rows", b.project_packed_group_into(0, cols, false, false, &x, true, &[&first, &second], &mut y.resident_range(0..2 * rows)?)),
                ("group: no matrices", b.project_packed_group_into(rows, cols, false, false, &x, true, &[], &mut y.resident_range(0..rows)?)),
                ("group: a segmented unit", b.project_packed_group_into(rows, cols, false, false, &x, true, &[&first, &segmented], &mut y.resident_range(0..2 * rows)?)),
                // The output in segments is the one output that may not be private (`writable_from`): refused as such.
                ("one: a segmented output", { let mut o = b.segmented_vector(rows, rows)?; b.project_packed_into(rows, cols, false, false, &x, &first, &mut o) }),
                ("group: a segmented output", { let mut o = b.segmented_vector(2 * rows, 2 * rows)?; b.project_packed_group_into(rows, cols, false, false, &x, true, &[&first, &second], &mut o) }),
                // Kind 96 (review REVIDS): a stack one unit short, no experts, no rows, a segmented stack, a segmented output.
                ("picked: a stack one unit short", b.project_packed_picked_into(rows, cols, false, false, &x, true, &first, 2, &ids, &mut y.resident_range(0..2 * rows)?)),
                ("picked: no experts", b.project_packed_picked_into(rows, cols, false, false, &x, true, &stack, 0, &ids, &mut y.resident_range(0..2 * rows)?)),
                ("picked: no rows", b.project_packed_picked_into(0, cols, false, false, &x, true, &stack, 2, &ids, &mut y.resident_range(0..2 * rows)?)),
                ("picked: an output one pick short", b.project_packed_picked_into(rows, cols, false, false, &x, true, &stack, 2, &ids, &mut y.resident_range(0..rows)?)),
                ("picked: a segmented stack", b.project_packed_picked_into(rows, cols, false, false, &x, true, &segmented_stack, 2, &ids, &mut y.resident_range(0..2 * rows)?)),
                ("picked: a segmented output", { let mut o = segmented_output; b.project_packed_picked_into(rows, cols, false, false, &x, true, &stack, 2, &ids, &mut o) }),
            ];
            b.flush_resident_launches()?;
            let taken = bad.iter().filter(|(_, r)| r.is_ok()).map(|(name, _)| *name).collect::<Vec<_>>();
            if !taken.is_empty() { return Err(format!("taken: {taken:?}")); }
            if launches() != before { return Err(format!("the refused calls recorded {} launches", launches() - before)); }
            if b.read_vector_values(&y)?.iter().any(|w| w.to_bits() != MARK) { return Err("a refused call wrote the output".into()); }
            b.project_packed_into(rows, cols, false, false, &x, &first, &mut y.resident_range(0..rows)?)?;
            b.project_packed_group_into(rows, cols, false, false, &x, true, &[&first, &second], &mut y.resident_range(0..2 * rows)?)?;
            b.project_packed_picked_into(rows, cols, false, false, &x, true, &stack, 2, &ids, &mut y.resident_range(0..2 * rows)?)?;
            if launches() != before + 3 { return Err("the good calls did not record one launch each".into()); }
            Ok(())
        }).unwrap();
    }
}
