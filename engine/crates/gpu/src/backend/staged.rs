//! The staged BF16 row product and its rows tile on the device (`staged_product`; kernel bodies
//! `kernels-inkling/src/plain_staged.rs` and `plain_staged_rows.rs`, in the Inkling image; Rulings R6g and R6h).
//!
//! Safe code only: the host check of every extent and every alignment, then the record of the launch. The body has
//! one entry, the recorded one (`Kind::PlainStaged`, `Kind::PlainStagedRows`), so a call while launches are not recorded is refused; the
//! caller asks `staged_product_ready` first and keeps `project_bf16_chain_into` for the other case.
use super::*;
use crate::resident_launch::{Command, Kind};
use crate::staged_product::{self, Span};

impl<'a> Backend<'a> {
    /// True when `project_staged_into` runs this product now: the shape is inside the tile, the weights and the
    /// input are contiguous and 16-byte aligned, and launches are recorded.
    pub fn staged_product_ready(&self, rows: usize, cols: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        staged_product::shape_admitted(rows, cols) && !input.is_segmented() && !weight.is_segmented()
            && (input.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }
    /// The product of one BF16 matrix (`rows x cols` values, two a word, row-major) with one input row of `cols`
    /// words. `output` gets `rows` words: the words of `project_bf16_chain_into`.
    pub fn project_staged_into(&self, rows: usize, cols: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        // A recorded command inside any stream capture would tie the graph to the recorder's mutable stage (review
        // REVPROD finding 3): refused, as the arms are.
        if self.arm_capture() { return Err("staged product: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        // Every extent and every alignment that the body relies on, before anything is recorded.
        staged_product::check(rows, cols, span(weight), span(input), span(output))?;
        // The owners: this context, the exact lengths, contiguous storage, and an output that is private.
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, cols)?;
        self.check_vector(output, rows)?;
        if weight.is_segmented() || input.is_segmented() { return Err("staged product: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("staged product: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainStaged.kernel()])?;
        self.resident_completion()?;
        let (rows, cols) = (u32::try_from(rows).map_err(|_| "staged product: the rows exceed the ABI")?, u32::try_from(cols).map_err(|_| "staged product: the columns exceed the ABI")?);
        let mut command = Command::new(Kind::PlainStaged);
        for (index, word) in [weight.device_address(), input.device_address(), output.device_address(), u64::from(rows), u64::from(cols)].into_iter().enumerate() { command.word(index, word); }
        let (blocks, threads) = staged_product::geometry(rows as usize);
        if !self.record_resident_block(Kind::PlainStaged, command, [blocks, 1, 1, threads, 0])? {
            return Err("staged product: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
    /// True when `project_staged_rows_into` runs this product now: the shape is inside the rows tile, the weights
    /// and the input are contiguous and 16-byte aligned, and launches are recorded.
    pub fn staged_rows_ready(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>) -> bool {
        staged_product::rows_shape_admitted(rows, cols, inputs) && !input.is_segmented() && !weight.is_segmented()
            && input.len() == inputs * cols && weight.len() == rows * cols / 2
            && (input.device_address() | weight.device_address()) & 15 == 0 && self.resident_launches_active()
    }
    /// The product of one BF16 matrix (`rows x cols` values, two a word, row-major) with `inputs` input rows
    /// (`[inputs][cols]`). `output` gets `[inputs][rows]`: each word is the word of `project_staged_into` on its
    /// input row.
    pub fn project_staged_rows_into(&self, rows: usize, cols: usize, inputs: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_staged_rows_strided_into(rows, cols, inputs, rows, input, weight, output)
    }
    /// `project_staged_rows_into` for one part of a larger matrix: input row `t` goes to the words
    /// `t * stride .. t * stride + rows` of `output`, which holds `(inputs - 1) * stride + rows` words. The parts of a
    /// matrix, each with its own range of the output, fill `[inputs][stride]` in place.
    #[allow(clippy::too_many_arguments, reason = "the shape, the stride and the three operands")]
    pub fn project_staged_rows_strided_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.project_staged_rows_at_into(rows, cols, inputs, stride, input, None, weight, output, None)
    }
    /// `project_staged_rows_strided_into` with index rows (the expert runs of a step): launch row `t` reads row
    /// `input_rows[t]` of `input` (which holds whole rows of `cols` words) and writes the words of row
    /// `output_rows[t]` of `output` (rows `stride` words apart). Each table holds `inputs` row numbers; None is the
    /// rows in order. The output rows are distinct. Each word is the word of `project_staged_into` on its input row.
    /// The caller gives host words only: this call checks them and the recorder stages exactly these words with the
    /// command (`record_resident_tabled`), so the replay reads the words the check saw and no caller can hand in a
    /// device table of its own (review REVPROD finding 1).
    #[allow(clippy::too_many_arguments, reason = "the shape, the stride, the three operands and the two tables")]
    pub fn project_staged_rows_at_into(&self, rows: usize, cols: usize, inputs: usize, stride: usize, input: &DeviceVector<'_>, input_rows: Option<&[u32]>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, output_rows: Option<&[u32]>) -> Result<(), String> {
        if self.arm_capture() { return Err("staged rows product: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        // The rows of each operand: from its length with a table, the launch rows without one.
        let input_count = if input_rows.is_some() && cols > 0 { input.len() / cols } else { inputs };
        let output_count = if output_rows.is_some() && stride > 0 && output.len() >= rows { (output.len() - rows) / stride + 1 } else { inputs };
        // Every extent, every alignment and every index that the body relies on, before anything is recorded.
        staged_product::check_rows_at(rows, cols, inputs, stride, span(weight), span(input), staged_product::Rows { count: input_count, table: input_rows }, span(output), staged_product::Rows { count: output_count, table: output_rows })?;
        // The owners: this context, the exact lengths, contiguous storage, and an output that is private.
        self.check_vector(weight, rows * cols / 2)?;
        self.check_vector(input, input_count * cols)?;
        self.check_vector(output, (output_count - 1) * stride + rows)?;
        if weight.is_segmented() || input.is_segmented() { return Err("staged rows product: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("staged rows product: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::PlainStagedRows.kernel()])?;
        self.resident_completion()?;
        let word = |value: usize| u32::try_from(value).map(u64::from).map_err(|_| "staged rows product: a count exceeds the ABI");
        let words = [weight.device_address(), input.device_address(), output.device_address(), word(rows)?, word(cols)?, word(inputs)?, word(stride)?];
        let grid = [staged_product::rows_blocks(rows, inputs), 1, 1, staged_product::ROWS_THREADS as u32, 0];
        let command = |table: Option<(u64, u64)>| {
            let mut command = Command::new(Kind::PlainStagedRows);
            let (input_table, output_table) = table.unwrap_or((0, 0));
            for (index, value) in words.into_iter().chain([input_table, input_count as u64, output_table, output_count as u64]).enumerate() { command.word(index, value); }
            command
        };
        let recorded = if input_rows.is_none() && output_rows.is_none() {
            self.record_resident_block(Kind::PlainStagedRows, command(None), grid)?
        } else {
            // One staged table: the input rows, then the output rows; a table that is None takes no bytes.
            let bytes = input_rows.into_iter().chain(output_rows).flatten().flat_map(|row| row.to_ne_bytes()).collect::<Vec<_>>();
            let input_bytes = input_rows.map_or(0, |table| table.len() as u64 * 4);
            self.record_resident_tabled(Kind::PlainStagedRows, grid, &bytes, |address| {
                command(Some((if input_rows.is_some() { address } else { 0 }, if output_rows.is_some() { address + input_bytes } else { 0 })))
            })?
        };
        if !recorded {
            return Err("staged rows product: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
#[cfg(test)]
mod tests {
    /// Review REVPROD finding 3 and REVPROD-2: a launch of kind 90 or 92 inside each graph capture and inside a verify
    /// segment is refused before anything is recorded, with launch recording off and on; after them the mark is clear
    /// and the same calls run as recorded launches.
    #[test]
    #[ignore = "skipped: no GPU; requires CUDA and the Inkling image"]
    fn a_staged_product_inside_each_graph_capture_is_refused() {
        crate::driver::with_backend(0, crate::layer_ptx().unwrap(), usize::MAX, |b| {
            let (rows, cols, inputs) = (256usize, 64usize, 2usize);
            let weight = b.upload_vector(&vec![0.25f32; rows * cols / 2])?.into_resident_shared()?;
            let x = b.upload_vector(&vec![0.5f32; inputs * cols])?.into_resident_shared()?;
            let position = b.upload_vector(&[0.0f32; 4])?;
            let y = b.vector(inputs * rows)?.into_resident_shared()?;
            let one = x.resident_range(0..cols)?;
            let row = || b.project_staged_into(rows, cols, &one, &weight, &mut y.resident_range(0..rows)?);
            let tile = || b.project_staged_rows_at_into(rows, cols, inputs, rows, &x, Some(&[1, 0]), &weight, &mut y.resident_range(0..inputs * rows)?, Some(&[1, 0]));
            let refused = |result: Result<(), String>, name: &str| -> Result<(), String> {
                match result { Err(e) if e.contains("stream capture") => Ok(()), Err(e) => Err(format!("{name}: other error {e}")), Ok(()) => Err(format!("{name}: the staged product was captured")) }
            };
            let calls: [(&dyn Fn() -> Result<(), String>, &str); 2] = [(&row, "row product"), (&tile, "rows tile")];
            let mut queue = None;
            for recording in [false, true] {
                if recording { queue = Some(b.resident_queue()?); b.enable_resident_launches()?; }
                for (call, kind) in calls {
                    // SAFETY: each capture records nothing (the staged product is refused before any launch, so the
                    // capture fails or is empty) and every buffer the closure names (weight, x, y, position) lives to
                    // the end of this test.
                    let graphs = unsafe { [b.capture_batch(1, 1, call), b.capture_batch_bounded(64, call), b.capture_decode(&position, 0, 1, 1, call)] };
                    for (graph, name) in graphs.into_iter().zip(["capture_batch", "capture_batch_bounded", "capture_decode"]) { refused(graph.map(drop), &format!("{kind} {name}, recording {recording}"))?; }
                    // The verify segment: its capture is open between begin_segment and abort_segment.
                    if !b.begin_segment(1, 1)? { return Err("the verify segment did not open".into()); }
                    let result = call();
                    b.abort_segment();
                    refused(result, &format!("{kind} verify segment, recording {recording}"))?;
                }
                if b.arm_capture() { return Err("the capture mark stayed set".into()); }
            }
            row()?;
            tile()?;
            drop(queue);
            Ok(())
        }).unwrap();
    }
}
