//! Checkpoint shard loading, packed uploads and embedding row reads.
use super::*;

impl<'a> Backend<'a> {
    /// Row-major views retain the arena after its handle drops; no tensor allocation.
    pub fn weight_matrix(&self, arena: &WeightArena<'a>, offset: usize, rows: usize,
        cols: usize, dtype: crate::load::WeightDtype) -> Result<Matrix<'a>, String> {
        let width = dtype.bytes();
        if !ptr::eq(arena.buffer._context, self.context) || rows == 0 || cols == 0
            || rows > u32::MAX as usize || cols > u32::MAX as usize || offset % width != 0 {
            return Err("invalid typed weight view".into());
        }
        let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(width)).ok_or("weight view overflow")?;
        Ok(Matrix { buffer: Buffer::view(&arena.buffer, offset, bytes)?, rows, cols,
            packed: false, packed_blocked: false, stride: cols, row_offset: 0, column_offset: 0,
            scalar_f32: matches!(dtype, crate::load::WeightDtype::F32), fp8_scales: None, fp8_fragments: false })
    }
    /// Load selected BF16 rows into one packed arena per shard. The returned
    /// matrices follow request order and share arena ownership. Check tensor
    /// shapes before calling; file and allocation bounds are checked here.
    /// Four persistent readers serve the whole load; at most one allocation
    /// runs ahead of I/O. Dropping the last matrix releases its shard arena.
    pub fn load_bf16_shards(&self, shards: &[crate::load::WeightShard<'_>]) -> Result<Vec<Vec<Matrix<'a>>>, String> {
        self.load_weight_shards(shards, false, false)
    }
    /// All matrices share one allocation, including experts split across shards.
    pub fn load_bf16_shards_in_arena(&self, shards: &[crate::load::WeightShard<'_>]) -> Result<Vec<Vec<Matrix<'a>>>, String> {
        self.load_weight_shards(shards, true, false)
    }
    /// Exact text tensor reads in mixed text/vision shards, retaining FP8 bytes or packing BF16.
    pub fn load_text_shards_in_arena(&self, shards: &[crate::load::WeightShard<'_>]) -> Result<Vec<Vec<Matrix<'a>>>, String> {
        self.load_weight_shards(shards, true, true)
    }
    /// Small checkpoint parameters retain their declared FP32 storage.
    pub fn upload_f32_parameters(&self, values: &[f32]) -> Result<Matrix<'a>, String> {
        let mut vector = self.vector(values.len())?;
        self.write_vector(&mut vector, values)?;
        self.stats.borrow_mut().weight_bytes += std::mem::size_of_val(values) as u128;
        Ok(Matrix { buffer: vector.buffer, rows: 1, cols: values.len(), packed: false, packed_blocked: false,
            stride: values.len(), row_offset: 0, scalar_f32: true, fp8_scales: None, fp8_fragments: false, column_offset: 0 })
    }
    fn load_weight_shards(&self, shards: &[crate::load::WeightShard<'_>], single_arena: bool, exact: bool) -> Result<Vec<Vec<Matrix<'a>>>, String> {
        let plans = shards.iter().map(|shard| shard.matrices.iter().map(|m| {
            let tile = if !exact && m.columns % 64 == 0 { 64 } else { 8 };
            if m.packed && m.rows.len() % tile != 0 { return Err("packed weight selection splits a stored row tile".into()); }
            m.range()
        })
            .collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?;
        let sizes = plans.iter().map(|ranges| ranges.iter().try_fold(0usize,
            |sum, &(_, bytes)| sum.checked_add(bytes).ok_or("shard size overflow")))
            .collect::<Result<Vec<_>, _>>()?;
        let total = sizes.iter().try_fold(self.context.allocated.get(),
            |sum, &bytes| sum.checked_add(bytes).ok_or("weight size overflow"))?;
        if total.checked_add(self.graph_storage_bytes()).is_none_or(|n| n > self.context.budget.get()) { return Err(crate::admission::allocation(total as u128 + self.graph_storage_bytes() as u128, self.context.budget.get(), &format!("weights/rank total {total} bytes including live allocations; graphs {} bytes; sessions, CONTEXT and OUTPUT not supplied to weight loader", self.graph_storage_bytes()))); }
        if shards.is_empty() { return Ok(Vec::new()); }
        if sizes.contains(&0) { return Err("empty weight shard".into()); }
        let timer = Instant::now();
        self.require_batch_kernels()?;
        let uploader = FileUploader::new(self.context)?;
        let mut pending = Some(PendingBuffer::new(self.context,
            if single_arena { sizes.iter().sum() } else { sizes[0] })?);
        let mut shared_arena = None;
        let mut shared_offset = 0;
        let mut result = Vec::new();
        for (index, shard) in shards.iter().enumerate() {
            let wait = Instant::now();
            let arena = if let Some(arena) = &shared_arena { Rc::clone(arena) }
                else { Rc::new(pending.take().ok_or("missing weight allocation")?.wait()?) };
            self.stats.borrow_mut().allocation_ns += wait.elapsed().as_nanos();
            if single_arena { shared_arena = Some(Rc::clone(&arena)); }
            else if index + 1 < shards.len() { pending = Some(PendingBuffer::new(self.context, sizes[index + 1])?); }
            let mut ranges = Vec::new();
            let mut matrices = Vec::new();
            let mut destination = if single_arena { shared_offset } else { 0 };
            for (matrix, &(offset, bytes)) in shard.matrices.iter().zip(&plans[index]) {
                // W8A8 staging reads the checkpoint row-major bytes directly.
                let fp8_fragments = false;
                if fp8_fragments && (matrix.rows.len() % 128 != 0 || matrix.columns % 128 != 0) {
                    return Err("FP8 fragments require N/K divisible by 128".into());
                }
                matrices.push(Matrix { buffer: Buffer::view(&arena, destination, bytes)?,
                    rows: matrix.rows.len(), cols: matrix.columns, packed: matrix.packed, packed_blocked: !exact && matrix.columns % 64 == 0, stride: matrix.columns,
                    row_offset: 0, column_offset: 0, scalar_f32: false, fp8_scales: None, fp8_fragments });
                ranges.push((offset, bytes, &*arena, destination,
                    if matrix.packed || fp8_fragments { matrix.columns } else { 0 }, fp8_fragments));
                destination += bytes;
            }
            uploader.upload(shard.file, &ranges, self, exact)?;
            shared_offset = destination;
            result.push(matrices);
        }
        let mut stats = self.stats.borrow_mut();
        stats.weight_bytes += sizes.iter().sum::<usize>() as u128;
        stats.weight_read_upload_ns += timer.elapsed().as_nanos();
        Ok(result)
    }

    /// Read row-major shards directly into packed storage, including uneven
    /// vocabulary tails. Only a bounded host tile exists beside device weights.
    pub fn upload_packed_bf16(&self, rows: usize, cols: usize, chunk_bytes: usize,
        read: impl FnMut(usize, &mut [u8]) -> Result<(), String>) -> Result<Matrix<'a>, String> {
        self.require_batch_kernels()?;
        self.upload_matrix(rows, cols, chunk_bytes, false, read)
    }
    /// Native BF16 decode reads eight-row fragments, without blocked K tiles.
    /// Retain that layout for every width and pad only the final row tile.
    pub fn upload_native_bf16(&self, rows: usize, cols: usize, chunk_bytes: usize,
        read: impl FnMut(usize, &mut [u8]) -> Result<(), String>) -> Result<Matrix<'a>, String> {
        self.require_batch_kernels()?;
        self.upload_matrix(rows, cols, chunk_bytes, true, read)
    }
    fn upload_matrix(&self, rows: usize, cols: usize, chunk_bytes: usize, packed8: bool,
        mut read: impl FnMut(usize, &mut [u8]) -> Result<(), String>) -> Result<Matrix<'a>, String> {
        let timer = self.profiling.get().then(Instant::now);
        if !cfg!(target_endian = "little") { return Err("little-endian host required".into()); }
        if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err("matrix dimensions outside nonempty kernel ABI".into());
        }
        if chunk_bytes < 2 || chunk_bytes % 2 != 0 || chunk_bytes > 16 * 1024 * 1024 {
            return Err("upload chunk must be even and in 2..=16777216".into());
        }
        let row_bytes = cols.checked_mul(2).ok_or("row byte overflow")?;
        let tile_rows = if !packed8 && cols % 64 == 0 {64} else {8};
        let tile = row_bytes.checked_mul(tile_rows).ok_or("tile byte overflow")?;
        if cols % 16 != 0 || tile > 16 * 1024 * 1024 {
            return Err("packed weights require K divisible by 16 and a bounded row tile".into());
        }
        let stored_rows = rows.div_ceil(tile_rows) * tile_rows;
        let bytes = stored_rows.checked_mul(row_bytes).ok_or("matrix byte overflow")?;
        let payload = rows * row_bytes;
        let chunk_bytes = (chunk_bytes / tile).max(1) * tile;
        let buffer = Buffer::new(self.context, bytes)?;
        let mut chunk = vec![0u8; chunk_bytes.min(bytes)];
        let mut offset = 0usize;
        while offset < bytes {
            let count = chunk_bytes.min(bytes - offset);
            let valid = count.min(payload.saturating_sub(offset));
            read(offset, &mut chunk[..valid])?;
            chunk[valid..count].fill(0);
            crate::load::pack_rows_layout(&mut chunk[..count], cols, packed8);
            self.write_buffer(&buffer, offset, &chunk[..count])?;
            offset += count;
        }
        if self.profiling.get() {
            let mut stats = self.stats.borrow_mut();
            stats.weight_bytes += bytes as u128;
            stats.weight_read_upload_ns += elapsed(timer);
        }
        Ok(Matrix { buffer, rows, cols, packed: true, packed_blocked: !packed8 && cols % 64 == 0, stride: cols, row_offset: 0, column_offset: 0, scalar_f32: false, fp8_scales: None, fp8_fragments: false })
    }
    /// Read back one embedding row without a second resident table.
    pub fn shared_bf16_row(&self, matrix: &Matrix<'_>, row: usize) -> Result<Vec<f32>, String> { self.flush_resident_launches()?;
        if row >= matrix.rows || !std::ptr::eq(matrix.buffer._context, self.context) {
            return Err("invalid shared BF16 row".into());
        }
        if self.context.pending.get() { self.synchronize()?; }
        let tile_rows = if matrix.packed_blocked {64} else {8};
        let mut values = vec![0u16; matrix.cols * if matrix.packed { tile_rows } else { 1 }];
        let start = matrix.buffer.address + (if matrix.packed { row / tile_rows * matrix.cols * tile_rows * 2 }
            else { row * matrix.cols * 2 }) as u64;
        let _wait = self.context.progress.wait(WaitKind::Readback);
        self.context.activate()?;
        unsafe { check(cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), start, values.len() * 2), "embedding row read")?; }
        Ok((0..matrix.cols).map(|i| {
            let offset = if matrix.packed { (if tile_rows == 64 { (i / 64 * 32 + row % 64 / 8 * 4 + i % 64 / 16) * 32 }
                else { i / 16 * 32 } + row % 8 * 4 + i % 8 / 2) * 4
                + i % 2 + i % 16 / 8 * 2 } else { i };
            f32::from_bits((values[offset] as u32) << 16)
        }).collect())
    }
}

impl<'a> Matrix<'a> {
    /// Retain intersecting BF16 blocks for a selected original-coordinate FP8 slice.
    /// Non-block-aligned origins refuse existing fixed kernels via their offset guards.
    pub fn with_selected_fp8_scales(mut self, scales: Self, origin: [usize; 2]) -> Result<Self, String> {
        let [n, k] = origin.map(|v| v % 128);
        let rows = self.rows.checked_add(n).ok_or("selected scale rows overflow")?.div_ceil(128);
        let cols = self.cols.checked_add(k).ok_or("selected scale cols overflow")?.div_ceil(128);
        if self.packed || scales.packed || self.scalar_f32 || scales.scalar_f32 || self.is_fp8() || scales.is_fp8()
            || self.stride != self.cols || self.row_offset != 0 || self.column_offset != 0
            || scales.rows != rows || scales.cols != cols || self.buffer.bytes != self.rows * self.cols
            || scales.buffer.bytes != rows * cols * 2 || !ptr::eq(self.buffer._context, scales.buffer._context) {
            return Err("invalid selected FP8 weight/scale layout".into());
        }
        self.row_offset = n; self.column_offset = k; self.fp8_scales = Some(scales.buffer); Ok(self)
    }
}
