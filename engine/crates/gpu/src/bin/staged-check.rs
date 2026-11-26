//! The staged BF16 products (`staged_product`) on one CUDA device, small shapes. No model file.
//! ROW: the row product of a made-up BF16 matrix with one input row, into an output with guard words on both sides,
//! against the host chain (rising columns from +0.0, each product rounded before its add). The shapes walk the edges
//! of the launch geometry: 1 row, a warp less one, a whole warp, a warp and one, the last one-warp block, the first
//! four-warp block, and 1, 2, 9 and 18 stages of columns.
//! NARROW: the narrow product (`staged_narrow`, blocks of 4 rows) the same way: 1 row, part of a block, a block and
//! one, the few-row matrices of the model, 512 rows, 8,192 columns; with the leads of VIEWS.
//! ROWS: the rows tile of the same matrix with many input rows, the same way: one row, the narrow matrices of the
//! model (24 x 1152, 64 x 4608 at the widest columns, 128 x 2304: part of one 64-row tile or two whole tiles), a tile
//! and one row, and 1, 2, 63, 64, 65 and 130 input rows.
//! VIEWS: weights, input and output as ranges of larger vectors at 16-byte steps.
//! PARTS: one matrix as two parts that write their columns of one output in place (the output stride).
//! INDEX: index rows: launch rows read scattered rows of a larger input (some twice) and write a permutation of the
//! rows of a larger output, in one launch and as two launches over the two halves of the tables (the expert runs);
//! output rows that no launch row names keep the poison.
//! GROUPED: the runs of four matrices (9, 1, 70 and 64 launch rows) in ONE launch, scattered input rows, a permutation
//! of the output rows; every word against the chain of its run's matrix.
//! REFUSALS: a launch that the host check refuses records nothing and writes no word.
//! Small on purpose: it runs under the memory checker of the device tools
//! (`compute-sanitizer --tool memcheck staged-check`).
#[cfg(target_os = "linux")]
mod check {
    use spark_gpu_probe::{driver::{Backend, DeviceVector}, resident_launch::Kind};

    const GUARD: usize = 64;
    const POISON: u32 = 0x7fc0_dead;
    /// (matrix rows, columns) of the row product.
    const ROW_SHAPES: [(usize, usize); 14] = [(1, 64), (31, 64), (32, 128), (33, 64), (96, 576), (512, 64), (512, 576), (513, 64), (513, 128), (640, 576), (640, 1152), (24, 1152), (130, 1152), (2304, 64)];
    /// (matrix rows, columns) and the input rows of the rows tile.
    const ROWS_SHAPES: [(usize, usize); 9] = [(1, 64), (24, 1152), (64, 4608), (128, 2304), (193, 64), (256, 128), (257, 64), (257, 192), (384, 576)];
    const ROWS_INPUTS: [usize; 6] = [1, 2, 63, 64, 65, 130];
    /// (matrix rows, columns) of the narrow product: 1 row, part of a block, a block, a block and one, the few-row
    /// matrices of the model (Sink, GateDown, Router, Relative), the most rows, the most columns.
    const NARROW_SHAPES: [(usize, usize); 10] = [(1, 256), (3, 256), (4, 512), (5, 256), (24, 2304), (64, 4608), (128, 2304), (384, 2304), (512, 256), (7, 8192)];

    /// BF16 patterns, finite: both signs, both zeros, values below the normal range, values near 2^31, and ordinary
    /// values. No value is large enough for a sum to leave the finite range (a not-a-number has no one bit pattern).
    fn weight(i: usize) -> u16 {
        let n = (i as u32).wrapping_mul(2_654_435_761) >> 16;
        match i % 11 { 0 => 0x0000, 1 => 0x8000, 2 => 0x0001 | (n as u16 & 0x007f), 3 => 0x4f00 | (n as u16 & 0x007f), _ => (0x3800 + (n % 0x1000)) as u16 | ((n as u16 & 1) << 15) }
    }
    /// Input values with 24 bits in each value and both signs.
    fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }
    fn widen(bits: u16) -> f32 { f32::from_bits(u32::from(bits) << 16) }
    /// The matrix on the device: two BF16 values a word, the even column in the low half. `lead` words come first.
    fn matrix<'d>(b: &Backend<'d>, values: &[u16], lead: usize) -> Result<DeviceVector<'d>, String> {
        let mut vector = b.vector(lead + values.len() / 2)?;
        let bytes = vec![0u8; lead * 4].into_iter().chain(values.iter().flat_map(|v| v.to_le_bytes())).collect::<Vec<_>>();
        b.write_vector_bytes(&mut vector, &bytes)?;
        vector.into_resident_shared()
    }
    /// The words that one scalar launch must give: `[inputs][rows]`.
    fn chain(values: &[u16], x: &[f32], rows: usize, cols: usize, inputs: usize) -> Vec<f32> {
        (0..inputs).flat_map(|t| (0..rows).map(move |r| (0..cols).fold(0.0f32, |sum, c| sum + widen(values[r * cols + c]) * x[t * cols + c]))).collect()
    }
    /// The words that one Weights-as-A `m16n8k16` launch must give: `[inputs][rows]`.
    fn tc_chain(values: &[u16], x: &[f32], rows: usize, cols: usize, inputs: usize) -> Vec<f32> {
        (0..inputs).flat_map(|t| (0..rows).map(move |r| spark_gpu_probe::tc_rows::tc_word(&values[r * cols..(r + 1) * cols], &x[t * cols..(t + 1) * cols]))).collect()
    }
    struct Count { cases: usize, words: usize, different: usize, guards: usize, changed: usize }
    /// Compare the output between its guards with the host words, and the guards with the poison.
    fn compare(b: &Backend<'_>, whole: &DeviceVector<'_>, want: &[f32], count: &mut Count) -> Result<(), String> {
        let back = b.read_vector_values(whole)?;
        if back.len() != want.len() + 2 * GUARD { return Err(format!("{} words read back for {} words and the guards", back.len(), want.len())); }
        count.cases += 1; count.words += want.len(); count.guards += 2 * GUARD;
        count.different += back[GUARD..GUARD + want.len()].iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        count.changed += back[..GUARD].iter().chain(&back[GUARD + want.len()..]).filter(|word| word.to_bits() != POISON).count();
        Ok(())
    }
    fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> { b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared() }

    pub fn run() -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            let recorded = |kind: Kind| b.stats().recorded_kinds[kind as usize];
            let (row_before, rows_before) = (recorded(Kind::PlainStaged), recorded(Kind::PlainStagedRows));
            // ROW and VIEWS. `lead`: words in front of the matrix and of the input row in their vectors (16-byte steps).
            let mut row = Count { cases: 0, words: 0, different: 0, guards: 0, changed: 0 };
            for &(rows, cols) in &ROW_SHAPES { for lead in [0usize, 4, 12] {
                let values = (0..rows * cols).map(|i| weight(i + rows + lead)).collect::<Vec<_>>();
                let x = (0..cols).map(|k| input(k + 7 * rows)).collect::<Vec<_>>();
                let stored = matrix(b, &values, lead)?;
                let weights = stored.resident_range(lead..lead + rows * cols / 2)?;
                let source = b.upload_vector(&vec![0.0; lead].into_iter().chain(x.iter().copied()).collect::<Vec<_>>())?.into_resident_shared()?;
                let one = source.resident_range(lead..lead + cols)?;
                if !b.staged_product_ready(rows, cols, &one, &weights) { return Err(format!("the row product is not ready for {rows} x {cols}")); }
                let whole = poisoned(b, rows)?;
                b.project_staged_into(rows, cols, &one, &weights, &mut whole.resident_range(GUARD..GUARD + rows)?)?;
                compare(b, &whole, &tc_chain(&values, &x, rows, cols, 1), &mut row)?;
            } }
            println!("STAGED CHECK ROW cases={} rows={} different={} guard_words={} guard_words_changed={} recorded_launches={}", row.cases, row.words, row.different, row.guards, row.changed, recorded(Kind::PlainStaged) - row_before);
            // NARROW: the narrow product of the same kind of matrix, the same way (guard words, the leads of VIEWS).
            let narrow_before = recorded(Kind::PlainStagedNarrow);
            let mut narrow = Count { cases: 0, words: 0, different: 0, guards: 0, changed: 0 };
            for &(rows, cols) in &NARROW_SHAPES { for lead in [0usize, 4, 12] {
                let values = (0..rows * cols).map(|i| weight(i + 5 * rows + lead)).collect::<Vec<_>>();
                let x = (0..cols).map(|k| input(k + 3 * rows)).collect::<Vec<_>>();
                let stored = matrix(b, &values, lead)?;
                let weights = stored.resident_range(lead..lead + rows * cols / 2)?;
                let source = b.upload_vector(&vec![0.0; lead].into_iter().chain(x.iter().copied()).collect::<Vec<_>>())?.into_resident_shared()?;
                let one = source.resident_range(lead..lead + cols)?;
                if !b.staged_narrow_ready(rows, cols, &one, &weights) { return Err(format!("the narrow product is not ready for {rows} x {cols}")); }
                let whole = poisoned(b, rows)?;
                b.project_staged_narrow_into(rows, cols, &one, &weights, &mut whole.resident_range(GUARD..GUARD + rows)?)?;
                compare(b, &whole, &tc_chain(&values, &x, rows, cols, 1), &mut narrow)?;
            } }
            // The grouped form: 2 and 4 matrices of one shape, each with its own input, one launch; each part the chain
            // of its own matrix (the same compare: guard words around the whole `[groups][rows]` output).
            for &(rows, cols) in &NARROW_SHAPES { for groups in [2usize, 4] {
                let lead = 4 * (groups - 1);
                let (mut stores, mut sources, mut want) = (Vec::new(), Vec::new(), Vec::new());
                for g in 0..groups {
                    let values = (0..rows * cols).map(|i| weight(i + 9 * rows + 13 * g)).collect::<Vec<_>>();
                    let x = (0..cols).map(|k| input(k + 11 * rows + 17 * g)).collect::<Vec<_>>();
                    want.extend(tc_chain(&values, &x, rows, cols, 1));
                    stores.push(matrix(b, &values, lead)?);
                    sources.push(b.upload_vector(&vec![0.0; lead].into_iter().chain(x.iter().copied()).collect::<Vec<_>>())?.into_resident_shared()?);
                }
                let weights = stores.iter().map(|s| s.resident_range(lead..lead + rows * cols / 2)).collect::<Result<Vec<_>, _>>()?;
                let ones = sources.iter().map(|s| s.resident_range(lead..lead + cols)).collect::<Result<Vec<_>, _>>()?;
                let (weight_refs, one_refs) = (weights.iter().collect::<Vec<_>>(), ones.iter().collect::<Vec<_>>());
                if !b.staged_narrow_group_ready(rows, cols, &one_refs, &weight_refs) { return Err(format!("the grouped narrow product is not ready for {groups} x {rows} x {cols}")); }
                let whole = poisoned(b, groups * rows)?;
                b.project_staged_narrow_group_into(rows, cols, &one_refs, &weight_refs, &mut whole.resident_range(GUARD..GUARD + groups * rows)?)?;
                compare(b, &whole, &want, &mut narrow)?;
            } }
            println!("STAGED CHECK NARROW cases={} rows={} different={} guard_words={} guard_words_changed={} recorded_launches={}", narrow.cases, narrow.words, narrow.different, narrow.guards, narrow.changed, recorded(Kind::PlainStagedNarrow) - narrow_before);
            // ROWS.
            let mut tile = Count { cases: 0, words: 0, different: 0, guards: 0, changed: 0 };
            for &(rows, cols) in &ROWS_SHAPES { for inputs in ROWS_INPUTS { for lead in [0usize, 4] {
                let values = (0..rows * cols).map(|i| weight(i + 3 * rows + inputs)).collect::<Vec<_>>();
                let x = (0..inputs * cols).map(|k| input(k + 5 * inputs)).collect::<Vec<_>>();
                let stored = matrix(b, &values, lead)?;
                let weights = stored.resident_range(lead..lead + rows * cols / 2)?;
                let source = b.upload_vector(&vec![0.0; lead].into_iter().chain(x.iter().copied()).collect::<Vec<_>>())?.into_resident_shared()?;
                let all = source.resident_range(lead..lead + inputs * cols)?;
                if !b.staged_rows_ready(rows, cols, inputs, &all, &weights) { return Err(format!("the rows tile is not ready for {rows} x {cols}, {inputs} input rows")); }
                let whole = poisoned(b, inputs * rows)?;
                b.project_staged_rows_into(rows, cols, inputs, &all, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                compare(b, &whole, &chain(&values, &x, rows, cols, inputs), &mut tile)?;
            } } }
            // PARTS: a matrix of 512 rows as two parts of 256 rows that write their columns of one [inputs][512] output in
            // place (the stride), as the head runs above the rows of the tile.
            for inputs in [1usize, 65] {
                let (rows, cols, part) = (512usize, 128usize, 256usize);
                let values = (0..rows * cols).map(|i| weight(i + 11 * inputs)).collect::<Vec<_>>();
                let x = (0..inputs * cols).map(|k| input(k + 3 * inputs)).collect::<Vec<_>>();
                let stored = matrix(b, &values, 0)?;
                let all = b.upload_vector(&x)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                for first in [0usize, part] {
                    let weights = stored.resident_range(first * cols / 2..(first + part) * cols / 2)?;
                    b.project_staged_rows_strided_into(part, cols, inputs, rows, &all, &weights, &mut whole.resident_range(GUARD + first..GUARD + first + (inputs - 1) * rows + part)?)?;
                }
                compare(b, &whole, &chain(&values, &x, rows, cols, inputs), &mut tile)?;
            }
            // INDEX.
            let mut extra = 2usize;
            let mut grouped = 0usize;
            for (inputs, split) in [(1usize, false), (64, false), (65, true), (130, true)] {
                let (rows, cols) = (257usize, 192usize);
                let (sources, targets) = (inputs + 21, inputs + 9);
                let values = (0..rows * cols).map(|i| weight(i + 5 * inputs)).collect::<Vec<_>>();
                let x = (0..sources * cols).map(|k| input(k + 13 * inputs)).collect::<Vec<_>>();
                let ins = (0..inputs).map(|t| ((t * 37 + 5) % sources) as u32).collect::<Vec<_>>();
                let outs = (0..inputs).map(|t| ((t * 7 + 3) % targets) as u32).collect::<Vec<_>>();
                let mut seen = vec![false; targets];
                for &row in &outs { if std::mem::replace(&mut seen[row as usize], true) { return Err(format!("index case {inputs}: the output table names a row twice")); } }
                let weights = matrix(b, &values, 0)?;
                let all = b.upload_vector(&x)?.into_resident_shared()?;
                let whole = poisoned(b, targets * rows)?;
                let halves: Vec<std::ops::Range<usize>> = if split { vec![0..inputs / 2, inputs / 2..inputs] } else { std::iter::once(0..inputs).collect() };
                extra += halves.len() - 1;
                for half in halves {
                    let count = half.len();
                    b.project_staged_rows_at_into(rows, cols, count, rows, &all, Some(&ins[half.clone()]), &weights,
                        &mut whole.resident_range(GUARD..GUARD + targets * rows)?, Some(&outs[half]))?;
                }
                let gathered = ins.iter().flat_map(|&row| x[row as usize * cols..(row as usize + 1) * cols].iter().copied()).collect::<Vec<_>>();
                let words = chain(&values, &gathered, rows, cols, inputs);
                let mut want = vec![f32::from_bits(POISON); targets * rows];
                for (t, &row) in outs.iter().enumerate() { want[row as usize * rows..(row as usize + 1) * rows].copy_from_slice(&words[t * rows..(t + 1) * rows]); }
                compare(b, &whole, &want, &mut tile)?;
            }
            // GROUPED.
            for tables in [true, false] {
                let (rows, cols) = (257usize, 192usize);
                let counts = [9usize, 1, 70, 64];
                let n = counts.iter().sum::<usize>();
                let (sources, targets) = if tables { (37, n + 5) } else { (n, n) };
                let matrices = (0..counts.len()).map(|r| (0..rows * cols).map(|i| weight(i + 17 * r + 3)).collect::<Vec<_>>()).collect::<Vec<_>>();
                let stored = matrices.iter().map(|values| matrix(b, values, 0)).collect::<Result<Vec<_>, _>>()?;
                let x = (0..sources * cols).map(|k| input(k + 29)).collect::<Vec<_>>();
                let all = b.upload_vector(&x)?.into_resident_shared()?;
                let ins = (0..n).map(|t| if tables { ((t * 13 + 5) % sources) as u32 } else { t as u32 }).collect::<Vec<_>>();
                let outs = (0..n).map(|t| if tables { ((t * 7 + 3) % targets) as u32 } else { t as u32 }).collect::<Vec<_>>();
                let runs = stored.iter().zip(counts).collect::<Vec<_>>();
                if !b.staged_rows_grouped_ready(rows, cols, &runs, &all, tables) { return Err(format!("the grouped launch is not ready (tables {tables})")); }
                let whole = poisoned(b, targets * rows)?;
                b.project_staged_rows_grouped_into(rows, cols, rows, &runs, &all, tables.then_some(&ins[..]), &mut whole.resident_range(GUARD..GUARD + targets * rows)?, tables.then_some(&outs[..]))?;
                let run_of = counts.iter().enumerate().flat_map(|(r, &count)| std::iter::repeat_n(r, count)).collect::<Vec<_>>();
                let mut want = vec![f32::from_bits(POISON); targets * rows];
                for t in 0..n {
                    let row = ins[t] as usize;
                    let words = chain(&matrices[run_of[t]], &x[row * cols..(row + 1) * cols], rows, cols, 1);
                    want[outs[t] as usize * rows..(outs[t] as usize + 1) * rows].copy_from_slice(&words);
                }
                compare(b, &whole, &want, &mut tile)?;
                grouped += 1;
            }
            println!("STAGED CHECK GROUPED launches={grouped}");
            println!("STAGED CHECK ROWS cases={} words={} different={} guard_words={} guard_words_changed={} recorded_launches={}", tile.cases, tile.words, tile.different, tile.guards, tile.changed, recorded(Kind::PlainStagedRows) - rows_before);
            // REFUSALS: an input one word off the 16-byte step, an input one word short, an output one word short,
            // an output that is the input. Nothing is recorded and no word of the output changes.
            let (rows, cols, inputs) = (256usize, 128usize, 4usize);
            let values = (0..rows * cols).map(weight).collect::<Vec<_>>();
            let weights = matrix(b, &values, 0)?;
            let source = b.upload_vector(&(0..inputs * cols + 8).map(input).collect::<Vec<_>>())?.into_resident_shared()?;
            let whole = poisoned(b, inputs * rows)?;
            let (row_at, rows_at, narrow_at) = (recorded(Kind::PlainStaged), recorded(Kind::PlainStagedRows), recorded(Kind::PlainStagedNarrow));
            let mut refused = 0usize;
            let mut expect = |what: &str, result: Result<(), String>| -> Result<(), String> {
                match result { Err(error) if error.starts_with("staged") => { refused += 1; Ok(()) }, other => Err(format!("{what}: not refused by the host check: {other:?}")) }
            };
            expect("row, input off the 16-byte step", b.project_staged_into(rows, cols, &source.resident_range(1..1 + cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + rows)?))?;
            expect("row, input one word short", b.project_staged_into(rows, cols, &source.resident_range(0..cols - 1)?, &weights, &mut whole.resident_range(GUARD..GUARD + rows)?))?;
            expect("row, output one word short", b.project_staged_into(rows, cols, &source.resident_range(0..cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + rows - 1)?))?;
            expect("row, weights one word short", b.project_staged_into(rows, cols, &source.resident_range(0..cols)?, &weights.resident_range(0..rows * cols / 2 - 1)?, &mut whole.resident_range(GUARD..GUARD + rows)?))?;
            expect("row, the output is the input", b.project_staged_into(cols, cols, &source.resident_range(0..cols)?, &weights.resident_range(0..cols * cols / 2)?, &mut source.resident_range(0..cols)?))?;
            expect("rows, input off the 16-byte step", b.project_staged_rows_into(rows, cols, inputs, &source.resident_range(1..1 + inputs * cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?))?;
            expect("rows, input one row short", b.project_staged_rows_into(rows, cols, inputs, &source.resident_range(0..(inputs - 1) * cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?))?;
            expect("rows, output one word short", b.project_staged_rows_into(rows, cols, inputs, &source.resident_range(0..inputs * cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows - 1)?))?;
            expect("rows, stride below the rows", b.project_staged_rows_strided_into(rows, cols, inputs, rows - 1, &source.resident_range(0..inputs * cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + (inputs - 1) * (rows - 1) + rows)?))?;
            // Columns past the tile (4,672 = 73 whole stages, above the widest matrix of 4,608): an input row of its own.
            let wide = b.upload_vector(&vec![0.0f32; 4672])?.into_resident_shared()?;
            expect("rows, columns past the tile", b.project_staged_rows_into(7, 4672, 1, &wide, &weights.resident_range(0..7 * 4672 / 2)?, &mut whole.resident_range(GUARD..GUARD + 7)?))?;
            // Index rows: a table one word short, a row outside its operand, an output row named twice.
            // The tables are host words only, staged by the recorder: a device table that differs from the checked
            // words cannot be expressed (review REVPROD finding 1).
            let (zeros, outside, twice) = ([0u32; 4], [0u32, 1, 2, inputs as u32], [0u32, 1, 1, 2]);
            expect("index, a table one word short", b.project_staged_rows_at_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols)?, Some(&zeros[..3]), &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?, None))?;
            expect("index, an input row outside the input", b.project_staged_rows_at_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols)?, Some(&outside), &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?, None))?;
            expect("index, an output row named twice", b.project_staged_rows_at_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols)?, None, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?, Some(&twice)))?;
            expect("index, the two output rows of the review's case named twice", b.project_staged_rows_at_into(rows, cols, 2, rows, &source.resident_range(0..2 * cols)?, None, &weights, &mut whole.resident_range(GUARD..GUARD + 2 * rows)?, Some(&[0, 0])))?;
            // The narrow product (review REVNARROW): columns that are no whole stages of 256, an input off the 16-byte
            // step, more rows than the narrow envelope, no rows, weights one word short, an output one word short, an
            // output that is the input.
            expect("narrow, more rows than the envelope", b.project_staged_narrow_into(513, 256, &source.resident_range(0..256)?, &weights, &mut whole.resident_range(GUARD..GUARD + 16)?))?;
            expect("narrow, no rows", b.project_staged_narrow_into(0, 256, &source.resident_range(0..256)?, &weights.resident_range(0..16 * 128)?, &mut whole.resident_range(GUARD..GUARD + 16)?))?;
            expect("narrow, weights one word short", b.project_staged_narrow_into(16, 256, &source.resident_range(0..256)?, &weights.resident_range(0..16 * 128 - 1)?, &mut whole.resident_range(GUARD..GUARD + 16)?))?;
            expect("narrow, output one word short", b.project_staged_narrow_into(16, 256, &source.resident_range(0..256)?, &weights.resident_range(0..16 * 128)?, &mut whole.resident_range(GUARD..GUARD + 15)?))?;
            expect("narrow, the output is the input", b.project_staged_narrow_into(16, 256, &source.resident_range(0..256)?, &weights.resident_range(0..16 * 128)?, &mut source.resident_range(0..16)?))?;
            expect("narrow, columns not whole stages of 256", b.project_staged_narrow_into(rows, cols, &source.resident_range(0..cols)?, &weights, &mut whole.resident_range(GUARD..GUARD + rows)?))?;
            expect("narrow, input off the 16-byte step", b.project_staged_narrow_into(16, 256, &source.resident_range(1..257)?, &weights.resident_range(0..16 * 128)?, &mut whole.resident_range(GUARD..GUARD + 16)?))?;
            // The grouped narrow form: five matrices, two matrices and one input, an output one group short, an output
            // that is the input of the second matrix.
            let (w16, x256) = (weights.resident_range(0..16 * 128)?, source.resident_range(0..256)?);
            expect("narrow group, five matrices", b.project_staged_narrow_group_into(16, 256, &[&x256; 5], &[&w16; 5], &mut whole.resident_range(GUARD..GUARD + 80)?))?;
            expect("narrow group, an input short", b.project_staged_narrow_group_into(16, 256, &[&x256], &[&w16, &w16], &mut whole.resident_range(GUARD..GUARD + 32)?))?;
            expect("narrow group, output one group short", b.project_staged_narrow_group_into(16, 256, &[&x256, &x256], &[&w16, &w16], &mut whole.resident_range(GUARD..GUARD + 16)?))?;
            let x_late = source.resident_range(256..512)?;
            expect("narrow group, the output meets the second input", b.project_staged_narrow_group_into(16, 256, &[&x256, &x_late], &[&w16, &w16], &mut source.resident_range(240..272)?))?;
            let first = matrix(b, &values, 0)?;
            let two: [(&spark_gpu_probe::driver::DeviceVector<'_>, usize); 2] = [(&first, 2), (&weights, 2)];
            expect("grouped, an output row named twice", b.project_staged_rows_grouped_into(rows, cols, rows, &two, &source.resident_range(0..inputs * cols)?, None, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?, Some(&twice)))?;
            expect("grouped, a run of no launch row", b.project_staged_rows_grouped_into(rows, cols, rows, &[(&first, 0), (&weights, 4)], &source.resident_range(0..inputs * cols)?, None, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?, None))?;
            let untouched = b.read_vector_values(&whole)?.iter().filter(|word| word.to_bits() != POISON).count();
            let stray = (recorded(Kind::PlainStaged) - row_at) + (recorded(Kind::PlainStagedRows) - rows_at) + (recorded(Kind::PlainStagedNarrow) - narrow_at);
            println!("STAGED CHECK REFUSALS cases={refused} recorded_launches={stray} output_words_changed={untouched}");
            drop(queue);
            if row.different + row.changed + narrow.different + narrow.changed + tile.different + tile.changed + untouched > 0 || stray > 0 {
                return Err(format!("row: {} of {} rows differ, {} guard words changed; narrow: {} of {} rows differ, {} guard words changed; rows: {} of {} words differ, {} guard words changed; refusals: {stray} launches recorded, {untouched} output words changed",
                    row.different, row.words, row.changed, narrow.different, narrow.words, narrow.changed, tile.different, tile.words, tile.changed));
            }
            if recorded(Kind::PlainStagedNarrow) - narrow_before != narrow.cases as u128 { return Err("a narrow case did not record one launch of its entry".into()); }
            if recorded(Kind::PlainStaged) - row_before != row.cases as u128 || recorded(Kind::PlainStagedRows) - rows_before != (tile.cases + extra) as u128 { return Err("a case did not record one launch of its entry".into()); }
            println!("STAGED CHECK PASS");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("staged-check requires Linux CUDA"); std::process::exit(1); }
}
