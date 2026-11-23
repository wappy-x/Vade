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
        if self.arm_capture() { return Err("packed picks: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |vector: &DeviceVector<'_>| Span { address: vector.device_address(), bytes: vector.len() as u64 * 4 };
        packed_product::check_picked(rows, cols, wide, pairs, span(stack), experts, span(ids), span(input), shared, span(output))?;
        let picks = ids.len();
        let unit_words = packed_product::layout(rows, cols, wide).bytes / 4;
        self.check_vector(stack, experts * unit_words)?;
        self.check_vector(ids, picks)?;
        self.check_vector(input, if shared { cols } else { cols * picks })?;
        self.check_vector(output, rows * picks)?;
        if stack.is_segmented() || ids.is_segmented() || input.is_segmented() { return Err("packed picks: an operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("packed picks: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[crate::resident_launch::Kind::PlainPackedPicked.kernel()])?;
        self.resident_completion()?;
        let rows = u32::try_from(rows).map_err(|_| "packed picks: the rows exceed the ABI")?;
        let experts = u32::try_from(experts).map_err(|_| "packed picks: the experts exceed the ABI")?;
        let input_words = if shared { 0 } else { u32::try_from(cols).map_err(|_| "packed picks: the columns exceed the ABI")? };
        if !self.record_resident_packed_picked(stack.device_address(), unit_words as u64 * 4, experts, ids.device_address(), picks, input.device_address(), output.device_address(), rows, packed_product::cols_word(cols, wide, pairs), input_words)? {
            return Err("packed picks: the launch runs only as a recorded launch, and launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
#[cfg(test)]
mod tests {
    use crate::packed_product;
    use crate::resident_launch::Kind;
    const MARK: u32 = 0x7fc0_dead;
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
