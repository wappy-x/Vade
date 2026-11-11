//! Device arithmetic and output reductions on the ordered stream.
use super::*;

impl<'a> Backend<'a> {
    /// Cast the final BF16 head's FP32 accumulator before sampling or export.
    pub fn round_logits_bf16(&self, logits: &mut DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(logits, logits.len)?;
        let (mut x, mut width) = (logits.buffer.address, logits.len as u32);
        unsafe { self.enqueue(self.layer_kernels()?.logits_bf16, logits.len,
            &mut [parameter(&mut x), parameter(&mut width)]) }
    }
    pub fn matvec_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if !std::ptr::eq(matrix.buffer._context, self.context) { return Err("matrix belongs to another context".into()); }
        self.check_vector(input, matrix.cols)?;
        self.check_vector(output, matrix.rows)?;
        if matrix.packed && !matrix.stride.is_multiple_of(64) && matrix.rows >= 128 && output.buffer.owner.as_ref().is_some_and(|owner|owner.host.is_some()) {
            let prefix=matrix.rows/128*64;
            for (first,count) in [(0,prefix),(prefix,matrix.rows-prefix)] {
                let (mut w,mut x,mut y)=(matrix.buffer.address+(first/8*matrix.stride*16) as u64,input.buffer.address,output.buffer.address+(first*4) as u64);
                let (mut rows,mut cols)=(count as u32,matrix.cols as u32);
                let mut args=[parameter(&mut w),parameter(&mut x),parameter(&mut y),parameter(&mut rows),parameter(&mut cols)];
                unsafe{self.packed_matvec_rows(matrix,&mut args,count)?;}
            }
            return Ok(());
        }
        let (mut w, mut x, mut y) = (matrix.buffer.address, input.buffer.address, output.buffer.address);
        let (mut rows, mut cols) = (matrix.rows as u32, matrix.cols as u32);
        let mut args = [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut rows), parameter(&mut cols)];
        if self.matrix_is_packed(matrix) {
            if x % 8 != 0 { return Err("packed decode input requires 8-byte alignment".into()); }
            return unsafe { self.packed_matvec(matrix, &mut args) };
        }
        if let Some(function) = self.vec8.get().filter(|_| self.use_vec8.get() && matrix.cols.is_multiple_of(8)) {
            return unsafe { self.enqueue_dims(function, matrix.rows.div_ceil(2) as u32, 1, 1, 64, &mut args) };
        }
        let work = matrix.rows * 32;
        unsafe { self.enqueue(self.matvec_function.get(), work, &mut args) }
    }
    pub fn embedding_device(&self, table: &Matrix<'_>, token: u32, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.generic_read_bytes(|| (10, table.cols * 2));
        if !std::ptr::eq(table.buffer._context, self.context) || token as usize >= table.rows { return Err("invalid device embedding".into()); }
        self.check_vector(output, table.cols)?;
        let mut packed = table.embedding_layout();
        let (mut table, mut output, mut token, mut width) = (table.buffer.address, output.buffer.address, token, table.cols as u32);
        let mut args = [parameter(&mut table), parameter(&mut output), parameter(&mut token), parameter(&mut width), parameter(&mut packed)];
        unsafe { self.enqueue(self.layer_kernels()?.embedding, width as usize, &mut args) }
    }
    pub fn embedding_selected_device(&self, table: &Matrix<'_>, selected: &DeviceVector<'_>, row: usize,
        output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.generic_read_bytes(|| (10, table.cols * 2));
        if !std::ptr::eq(table.buffer._context, self.context) || (row + 1) * 3 > selected.len {
            return Err("invalid selected embedding".into());
        }
        self.check_vector(selected, selected.len)?;
        self.check_vector(output, table.cols)?;
        let (mut w, mut y, mut s) = (table.buffer.address, output.buffer.address, selected.buffer.address + (row * 12) as u64);
        let (mut width, mut rows, mut packed) = (table.cols as u32, table.rows as u32, table.embedding_layout());
        unsafe { self.enqueue(self.layer_kernels()?.embedding_selected, table.cols,
            &mut [parameter(&mut w), parameter(&mut y), parameter(&mut s), parameter(&mut width), parameter(&mut rows), parameter(&mut packed)]) }
    }
    pub fn rms_device(&self, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, epsilon: f32) -> Result<(), String> {
        self.check_vector(input, weight.len)?; self.check_vector(weight, weight.len)?; self.check_vector(output, input.len)?;
        self.check_vector(partials, input.len.div_ceil(128))?;
        let (mut x, mut w, mut y, mut width, mut eps) = (input.buffer.address, weight.buffer.address, output.buffer.address, input.len as u32, epsilon);
        let mut p = partials.buffer.address;
        let kernels = self.layer_kernels()?;
        unsafe {
            if input.len <= 4096 {
                return self.enqueue_dims(kernels.rms_fused, 1, 1, 1, input.len.div_ceil(128) as u32 * 32,
                    &mut [parameter(&mut x), parameter(&mut w), parameter(&mut y), parameter(&mut width), parameter(&mut eps)]);
            }
            self.enqueue(kernels.rms, partials.len * 32, &mut [parameter(&mut x), parameter(&mut p), parameter(&mut width)])?;
            self.enqueue(kernels.rms_apply, input.len, &mut [parameter(&mut x), parameter(&mut w), parameter(&mut p), parameter(&mut y), parameter(&mut width), parameter(&mut eps)])
        }
    }
    pub fn residual_rms_device(&self, state: &mut DeviceVector<'_>, residual: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, epsilon: f32) -> Result<(), String> {
        if epsilon < 0.0 {
            self.residual_checkpoint_device(state, residual, true)?;
            return self.rms_device(state, weight, output, partials, epsilon);
        }
        for v in [&*state, residual, weight, &*output] { self.check_vector(v, state.len)?; }
        self.check_vector(partials, state.len.div_ceil(128))?;
        let (mut x, mut r, mut w, mut y, mut width, mut eps) = (state.buffer.address, residual.buffer.address, weight.buffer.address, output.buffer.address, state.len as u32, epsilon);
        let mut p = partials.buffer.address;
        let kernels = self.layer_kernels()?;
        unsafe {
            if state.len <= 4096 {
                return self.enqueue_dims(kernels.residual_rms_fused, 1, 1, 1, state.len.div_ceil(128) as u32 * 32,
                    &mut [parameter(&mut x), parameter(&mut r), parameter(&mut w), parameter(&mut y), parameter(&mut width), parameter(&mut eps)]);
            }
            self.enqueue(kernels.residual_rms, partials.len * 32, &mut [parameter(&mut x), parameter(&mut r), parameter(&mut p), parameter(&mut width)])?;
            self.enqueue(kernels.rms_apply, state.len, &mut [parameter(&mut x), parameter(&mut w), parameter(&mut p), parameter(&mut y), parameter(&mut width), parameter(&mut eps)])
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qk_rope_cache_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>,
        keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>, position: usize, epsilon: f32) -> Result<(), String> {
        let dim = q_norm.len;
        if dim == 0 || key.len == 0 || query.len == 0 || !dim.is_multiple_of(2) || !query.len.is_multiple_of(dim) || !key.len.is_multiple_of(dim) || position >= keys.len / key.len || position >= rope.len / dim {
            return Err("invalid QK/RoPE/cache dimensions".into());
        }
        for v in [&*query, key, value, q_norm, k_norm, rope, &*keys, &*values] { self.check_vector(v, v.len)?; }
        if value.len != key.len || values.len != keys.len || k_norm.len != dim { return Err("KV shape mismatch".into()); }
        let mut args = crate::kernel_params::QkRope {
            query: query.buffer.address, key: key.buffer.address, value: value.buffer.address,
            q_weight: q_norm.buffer.address, k_weight: k_norm.buffer.address, rope: rope.buffer.address,
            keys: keys.buffer.address, values: values.buffer.address,
            heads: (query.len / dim) as u32, kv_heads: (key.len / dim) as u32,
            dim: dim as u32, position: position as u32, epsilon, ..Default::default()
        };
        unsafe { self.enqueue(self.layer_kernels()?.qk_rope, (args.heads + args.kv_heads) as usize * 32, &mut args.cache_args()) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        scores: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>, heads: usize, kv_heads: usize, dim: usize, positions: usize) -> Result<(), String> {
        if dim == 128 && (heads == kv_heads * 4 || heads == kv_heads * 8) && positions != 0 {
            return self.attention_batch_device(query, keys, values, scores, partials, output, heads, kv_heads, dim, positions - 1, 1, false);
        }
        if heads == 0 || kv_heads == 0 || dim == 0 || !heads.is_multiple_of(kv_heads) || !keys.len.is_multiple_of(kv_heads * dim) {
            return Err("invalid attention dimensions".into());
        }
        let capacity = scores.len / heads;
        if positions == 0 || positions > capacity || positions > keys.len / (kv_heads * dim) { return Err("attention exceeds KV capacity".into()); }
        self.check_vector(query, heads * dim)?; self.check_vector(output, query.len)?;
        self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?; self.check_vector(scores, heads * capacity)?;
        self.check_vector(partials, heads * dim * capacity.div_ceil(64))?;
        let (mut q, mut k, mut v, mut s, mut o) = (query.buffer.address, keys.buffer.address, values.buffer.address, scores.buffer.address, output.buffer.address);
        let (mut h, mut kh, mut d, mut p, mut c) = (heads as u32, kv_heads as u32, dim as u32, positions as u32, capacity as u32);
        let kernels = self.layer_kernels()?;
        unsafe {
            let (score_kernel, groups) = if heads == kv_heads * 4 { (kernels.scores_gqa4, kh) } else { (kernels.scores, h) };
            self.enqueue_grid(score_kernel, positions * 32, groups, 1, &mut [parameter(&mut q), parameter(&mut k), parameter(&mut s),
                parameter(&mut h), parameter(&mut kh), parameter(&mut d), parameter(&mut p), parameter(&mut c)])?;
            self.enqueue(kernels.softmax, heads * 32, &mut [parameter(&mut s), parameter(&mut h), parameter(&mut p), parameter(&mut c)])?;
            if positions <= 64 {
                self.enqueue_grid(kernels.values, dim, h, 1, &mut [parameter(&mut s), parameter(&mut v), parameter(&mut o),
                    parameter(&mut h), parameter(&mut kh), parameter(&mut d), parameter(&mut p), parameter(&mut c)])
            } else {
                let (mut partial, mut width, mut chunks) = (partials.buffer.address, (heads * dim) as u32, positions.div_ceil(64) as u32);
                let (value_kernel, groups) = if heads == kv_heads * 4 { (kernels.value_gqa4, kh) } else { (kernels.value_split, h) };
                self.enqueue_grid(value_kernel, dim, groups, chunks, &mut [parameter(&mut s), parameter(&mut v), parameter(&mut partial),
                    parameter(&mut h), parameter(&mut kh), parameter(&mut d), parameter(&mut p), parameter(&mut c)])?;
                self.enqueue(kernels.value_merge, heads * dim, &mut [parameter(&mut partial), parameter(&mut o), parameter(&mut width), parameter(&mut chunks)])
            }
        }
    }
    pub fn swiglu_device(&self, gate: &mut DeviceVector<'_>, up: &DeviceVector<'_>) -> Result<(), String> {
        self.swiglu_precision_device(gate, up, false)
    }
    pub fn swiglu_precision_device(&self, gate: &mut DeviceVector<'_>, up: &DeviceVector<'_>, bf16: bool) -> Result<(), String> {
        self.swiglu_config_device(gate, up, bf16, None)
    }
    /// A checkpoint bound changes only the standalone epilogue. None preserves
    /// the existing arithmetic. BF16 mode retains each tensor dtype boundary.
    pub fn swiglu_config_device(&self, gate: &mut DeviceVector<'_>, up: &DeviceVector<'_>, bf16: bool,
        limit: Option<f32>) -> Result<(), String> {
        self.check_vector(gate, up.len)?; self.check_vector(up, up.len)?;
        if let Some(v) = limit {
            let b = crate::bf16::rounded(v);
            if !v.is_finite() || v <= 0.0 || !b.is_finite() || b <= 0.0 {
                return Err("swiglu_limit must be positive and finite in FP32 and BF16".into());
            }
        }
        if gate.len > u32::MAX as usize { return Err("SwiGLU width exceeds kernel range".into()); }
        let mut args = crate::kernel_params::SwiGlu { gate: gate.buffer.address,
            up: up.buffer.address, width: gate.len as u32, bf16: u32::from(bf16), limit: limit.unwrap_or(0.0) };
        unsafe { self.enqueue(self.layer_kernels()?.swiglu, gate.len, &mut [parameter(&mut args)]) }
    }
    /// Reuse the existing cast kernel, retaining no extra activation allocation.
    pub fn residual_checkpoint_device(&self, state: &mut DeviceVector<'_>, residual: &DeviceVector<'_>, bf16: bool) -> Result<(), String> {
        if bf16 {
            self.round_logits_bf16(&mut residual.slice(0..residual.len())?)?;
        }
        self.residual_device(state, residual)?;
        if bf16 { self.round_logits_bf16(state)?; }
        Ok(())
    }
    pub fn residual_device(&self, state: &mut DeviceVector<'_>, residual: &DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(state, residual.len)?; self.check_vector(residual, residual.len)?;
        let (mut x, mut r, mut width) = (state.buffer.address, residual.buffer.address, state.len as u32);
        unsafe { self.enqueue(self.layer_kernels()?.residual, state.len, &mut [parameter(&mut x), parameter(&mut r), parameter(&mut width)]) }
    }
    /// Add the dense projection and retain the next RMS reduction's partials.
    pub fn residual_rms_partials_device(&self, state: &mut DeviceVector<'_>, residual: &DeviceVector<'_>, partials: &mut DeviceVector<'_>, width: usize) -> Result<(), String> {
        if width == 0 || !state.len.is_multiple_of(width) { return Err("residual RMS shape".into()); }
        let rows = state.len / width;
        self.check_vector(state, state.len)?; self.check_vector(residual, state.len)?;
        self.check_vector(partials, rows * width.div_ceil(128))?;
        let (mut x, mut r, mut p, mut width) = (state.buffer.address, residual.buffer.address, partials.buffer.address, width as u32);
        unsafe { self.enqueue_grid(self.layer_kernels()?.residual_rms, (width as usize).div_ceil(128) * 32, rows as u32, 1,
            &mut [parameter(&mut x), parameter(&mut r), parameter(&mut p), parameter(&mut width)]) }
    }

    /// Greedy reduction with lowest-ID tie breaking and a nonfinite flag.
    pub fn argmax_device(&self, logits: &DeviceVector<'_>, partials: &mut DeviceVector<'_>, result: &mut DeviceVector<'_>) -> Result<(u32, f32), String> {
        self.enqueue_argmax_device(logits, partials, result, 0)?;
        self.read_argmax(result, 1, |values| Ok((values[1] as u32, values[0])))
    }
    pub fn enqueue_argmax_device(&self, logits: &DeviceVector<'_>, partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>, row: usize) -> Result<(), String> {
        let count = logits.len.div_ceil(1024);
        self.check_vector(logits, logits.len)?; self.check_vector(partials, count * 3)?; self.check_vector(result, result.len)?;
        if (row + 1) * 3 > result.len { return Err("argmax result row outside buffer".into()); }
        if logits.len > 1 << 24 { return Err("argmax ID exceeds exact FP32 integer range".into()); }
        let (mut x, mut p, mut y, mut width, mut count) = (logits.buffer.address, partials.buffer.address, result.buffer.address + (row * 12) as u64, logits.len as u32, count as u32);
        let kernels = self.layer_kernels()?;
        unsafe {
            self.enqueue(kernels.argmax, count as usize * 32, &mut [parameter(&mut x), parameter(&mut p), parameter(&mut width)])?;
            self.enqueue(kernels.argmax_finish, 32, &mut [parameter(&mut p), parameter(&mut y), parameter(&mut count)])?;
        }
        Ok(())
    }
    /// Queue selections and sticky routing status before one completion wait.
    pub fn read_argmax_status(&self, result: &DeviceVector<'_>, status: &DeviceVector<'_>, rows: usize)
        -> Result<(Vec<Result<(u32,f32),String>>, Vec<usize>),String> { self.flush_resident_launches()?;
        self.check_vector(result, rows.checked_mul(3).ok_or("selection extent")?)?;
        self.check_vector(status, rows)?;
        let bytes = rows.checked_mul(16).filter(|&n| n>0 && n<=self.uploads.borrow().capacity).ok_or("selection read extent")?;
        let host = self.uploads.borrow().buffer.address;
        self.context.activate()?;
        unsafe {
            check(cuMemcpyDtoHAsync_v2(host, result.buffer.address, rows*12, self.stream.raw), "selection read")?;
            self.context.pending.set(true);
            check(cuMemcpyDtoHAsync_v2(host.cast::<u8>().add(rows*12).cast(), status.buffer.address, rows*4, self.stream.raw), "routing read")?;
        }
        self.stats.borrow_mut().dtoh_calls+=2;
        self.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(host.cast::<f32>(), bytes/4) };
        let ids = values[..rows*3].as_chunks::<3>().0.iter().map(|r| {
            if r[2]!=1.0 || !r[0].is_finite() || !r[1].is_finite() { Err("nonfinite logits".into()) }
            else { Ok((r[1] as u32,r[0])) }
        }).collect();
        Ok((ids, crate::routing::rejected_rows(&values[rows*3..])))
    }
    pub fn read_argmax_results(&self, result: &DeviceVector<'_>, rows: usize) -> Result<Vec<(u32, f32)>, String> {
        self.read_argmax(result, rows, |values| Ok(values.as_chunks::<3>().0.iter().map(|row| (row[1] as u32, row[0])).collect()))
    }
    fn read_argmax<T>(&self, result: &DeviceVector<'_>, rows: usize,
        consume: impl FnOnce(&[f32]) -> Result<T, String>) -> Result<T, String> {
        self.read_argmax_raw(result, rows, |values| {
            if values.iter().any(|v| !v.is_finite()) { return Err("nonfinite device output".into()); }
            if values.as_chunks::<3>().0.iter().any(|row| row[2] != 1.0) { return Err("nonfinite logits".into()); }
            consume(values)
        })
    }
    /// Preserve row validity words for callers that gather per-request verdicts.
    /// The callback runs after GPU completion and must inspect every row's status.
    pub fn read_argmax_raw<T>(&self, result: &DeviceVector<'_>, rows: usize,
        consume: impl FnOnce(&[f32]) -> Result<T, String>) -> Result<T, String> { self.flush_resident_launches()?;
        self.check_vector(result, result.len)?;
        let bytes = rows.checked_mul(12).ok_or("argmax read size overflow")?;
        if rows == 0 || bytes > result.len * 4 || bytes > self.uploads.borrow().capacity {
            return Err("argmax read outside buffer".into());
        }
        // Reuse the staging prefix after its earlier reads on this stream finish.
        let host = self.uploads.borrow().buffer.address;
        self.context.activate()?;
        unsafe { check(cuMemcpyDtoHAsync_v2(host, result.buffer.address, bytes, self.stream.raw), "pinned argmax read")?; }
        self.context.pending.set(true);self.stats.borrow_mut().dtoh_calls+=1;
        self.synchronize()?;
        let download = self.profiling.get().then(Instant::now);
        let values = unsafe { std::slice::from_raw_parts(host.cast::<f32>(), rows * 3) };
        if self.profiling.get() { self.stats.borrow_mut().output_download_ns += elapsed(download); }
        consume(values)
    }
}

impl<'a> Backend<'a> {
    /// BF16 model RMS boundary, with optional unweighted normalization.
    pub fn model_rms_device(&self, input: &DeviceVector<'_>, weight: Option<&DeviceVector<'_>>,
        output: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, width: usize, rows: usize, epsilon: f32) -> Result<(), String> {
        if width == 0 || rows == 0 || !epsilon.is_finite() || epsilon <= 0.0 { return Err("invalid model RMS geometry".into()); }
        let elements = width.checked_mul(rows).ok_or("RMS size overflow")?;
        self.check_vector_prefix(input, elements)?; self.check_vector_prefix(output, elements)?;
        self.check_vector_prefix(partials, rows * width.div_ceil(128))?;
        if let Some(w) = weight { self.check_vector_prefix(w, width)?; }
        let mut a = crate::kernel_params::ModelRms { input: input.buffer.address,
            weight: weight.map_or(0, |w| w.buffer.address), partials: partials.buffer.address,
            output: output.buffer.address, width: width as u32, rows: rows as u32, epsilon };
        let k = self.layer_kernels()?;
        unsafe {
            self.enqueue_grid(k.model_rms, width.div_ceil(128) * 32, rows as u32, 1, &mut [parameter(&mut a)])?;
            self.enqueue_grid(k.model_rms_apply, width, rows as u32, 1, &mut [parameter(&mut a)])
        }
    }
    pub fn model_elementwise_device(&self, input: &DeviceVector<'_>, peer: Option<&DeviceVector<'_>>,
        output: &DeviceVector<'_>, elements: usize, operation: u32, scalar: f32) -> Result<(), String> {
        self.check_vector_prefix(input, elements)?; self.check_vector_prefix(output, elements)?;
        if operation > 4 || !scalar.is_finite() || operation == 3 && scalar <= 0.0 { return Err("invalid model elementwise operation".into()); }
        if matches!(operation,1|2|4) { self.check_vector_prefix(peer.ok_or("missing model peer")?, elements)?; }
        let mut a = crate::kernel_params::ModelElementwise { input: input.buffer.address,
            peer: peer.map_or(0, |p| p.buffer.address), output: output.buffer.address,
            elements: u32::try_from(elements).map_err(|_| "elementwise size overflow")?, operation, scalar };
        unsafe { self.enqueue(self.layer_kernels()?.model_elementwise, elements, &mut [parameter(&mut a)]) }
    }
    /// Distinct K/V norms operate on raw projections; K=V may alias the input.
    pub fn model_qkv_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>,
        keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, dim: usize, position: usize, rows: usize, cache_capacity: usize, epsilon: f32) -> Result<(), String> {
        if heads == 0 || kv_heads == 0 || dim == 0 || !dim.is_multiple_of(2) || rows == 0 || cache_capacity == 0
            || position.checked_add(rows).is_none_or(|end| end > rope.len / dim) { return Err("invalid model QKV geometry".into()); }
        for (v, n) in [(&*query, rows*heads*dim), (key, rows*kv_heads*dim), (value, rows*kv_heads*dim),
            (q_norm, dim), (k_norm, dim), (rope, (position+rows)*dim),
            (&*keys, (cache_capacity*kv_heads*dim).div_ceil(2)), (&*values, (cache_capacity*kv_heads*dim).div_ceil(2))] { self.check_vector_prefix(v, n)?; }
        let mut a = crate::kernel_params::ModelQkv { query: query.buffer.address, key: key.buffer.address, value: value.buffer.address,
            q_weight: q_norm.buffer.address, k_weight: k_norm.buffer.address, rope: rope.buffer.address,
            keys: keys.buffer.address, values: values.buffer.address, metadata: 0, heads: heads as u32, kv_heads: kv_heads as u32,
            dim: dim as u32, position: position as u32, rows: rows as u32, cache_capacity: cache_capacity as u32, epsilon };
        unsafe { self.enqueue_grid(self.layer_kernels()?.model_qkv, (heads+kv_heads)*32, rows as u32, 1, &mut [parameter(&mut a)]) }
    }
    /// Query-only Gemma assistant layers share immutable target KV. The existing
    /// QKV kernel launches only query warps, preserving its norm/RoPE arithmetic.
    pub fn model_query_device(&self, query: &mut DeviceVector<'_>, norm: &DeviceVector<'_>,
        rope: &DeviceVector<'_>, heads: usize, dim: usize, position: usize, epsilon: f32) -> Result<(), String> {
        if heads == 0 || !matches!(dim, 256 | 512) || !epsilon.is_finite() || epsilon <= 0.0
            || position.checked_add(1).is_none_or(|n| n > rope.len/dim) { return Err("invalid shared query geometry".into()); }
        self.check_vector_prefix(query, heads*dim)?; self.check_vector_prefix(norm, dim)?;
        self.check_vector_prefix(rope, (position+1)*dim)?;
        let q = query.buffer.address;
        let mut a = crate::kernel_params::ModelQkv { query: q, key: q, value: q,
            q_weight: norm.buffer.address, k_weight: norm.buffer.address, rope: rope.buffer.address,
            keys: q, values: q, metadata: 0, heads: heads as u32, kv_heads: 0, dim: dim as u32,
            position: position as u32, rows: 1, cache_capacity: 1, epsilon };
        unsafe { self.enqueue_grid(self.layer_kernels()?.model_qkv, heads*32, 1, 1, &mut [parameter(&mut a)]) }
    }
    /// Generic BF16 attention shared by prefill/decode; one scratch tile is reused.
    pub fn bounded_attention_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        scores: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>, heads: usize, kv_heads: usize, dim: usize,
        position: usize, rows: usize, window: usize, cache_capacity: usize, scale: f32) -> Result<(), String> {
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || dim == 0 || rows == 0 || cache_capacity == 0 || !scale.is_finite()
            || (window == 0 && position+rows > cache_capacity)
            || (window != 0 && cache_capacity < (window+rows-1).min(position+rows)) { return Err("invalid bounded attention geometry".into()); }
        self.check_vector_prefix(query, rows*heads*dim)?; self.check_vector_prefix(output, rows*heads*dim)?;
        self.check_vector_prefix(keys, (cache_capacity*kv_heads*dim).div_ceil(2))?; self.check_vector_prefix(values, keys.len)?;
        let capacity = if window == 0 {position+rows} else {window.min(position+rows)};
        let tile = rows.min(8);
        let mut a = crate::kernel_params::BoundedAttention { query: query.buffer.address, keys: keys.buffer.address,
            values: values.buffer.address, scores: scores.buffer.address, output: output.buffer.address, metadata: 0,
            heads: heads as u32, kv_heads: kv_heads as u32, dim: dim as u32, position: position as u32,
            rows: rows as u32, capacity: capacity as u32, window: window as u32, cache_capacity: cache_capacity as u32,
            scale, round_bf16: 1 };
        let k = self.layer_kernels()?;
        if rows >= 256 && matches!(dim, 256 | 512) {
            let tile = rows.min(256);
            let stride = if window == 0 {capacity.div_ceil(64)*64} else {capacity};
            self.check_vector_prefix(scores, tile*heads*stride)?;
            if window == 0 {
                a.capacity = stride as u32; a.round_bf16 = 2;
                a.metadata = scores.buffer.address+(tile*heads*stride*2) as u64;
            }
            for start in (0..rows).step_by(tile) {
                a.query = query.buffer.address+(start*heads*dim*4) as u64;
                a.output = output.buffer.address+(start*heads*dim*4) as u64;
                a.position = (position+start) as u32; a.rows = (rows-start).min(tile) as u32;
                self.prefill_attention_wide(a)?;
            }
            return Ok(());
        }
        if rows >= 256 && dim <= 512 {
            let tile = rows.min(32); let chunks = capacity.div_ceil(64);
            let score_words = tile*heads*capacity;
            self.check_vector_prefix(scores, score_words+tile*heads*dim*chunks)?;
            let partials = scores.buffer.address+(score_words*4) as u64;
            for start in (0..rows).step_by(tile) {
                let n = (rows-start).min(tile);
                a.query = query.buffer.address+(start*heads*dim*4) as u64;
                a.position = (position+start) as u32; a.rows = n as u32;
                a.capacity = if window == 0 {(position+start+n) as u32} else {capacity as u32};
                a.scores = scores.buffer.address; a.output = partials;
                let key_rows = if window == 0 {a.capacity as usize} else {capacity+7};
                unsafe {
                    self.enqueue_grid(k.bounded_scores_grouped, key_rows*32, heads as u32, n.div_ceil(8) as u32, &mut [parameter(&mut a)])?;
                    self.enqueue_grid(k.bounded_softmax, heads*32, n as u32, 1, &mut [parameter(&mut a)])?;
                    self.enqueue_dims(k.bounded_split, (n*heads) as u32, a.capacity.div_ceil(64), 1,
                        dim.max(128).next_power_of_two() as u32, &mut [parameter(&mut a)])?;
                    a.scores = partials; a.output = output.buffer.address+(start*heads*dim*4) as u64;
                    self.enqueue(k.bounded_merge, n*heads*dim, &mut [parameter(&mut a)])?;
                }
            }
            return Ok(());
        }
        if capacity <= 32768 && dim <= 512 {
            return unsafe { self.enqueue_dims_shared(k.bounded_flash, rows as u32, heads as u32, 1,
                dim.max(128).next_power_of_two() as u32, (capacity*2).div_ceil(256) as u32*256,
                &mut [parameter(&mut a)]) };
        }
        self.check_vector_prefix(scores, tile*heads*capacity)?;
        for start in (0..rows).step_by(tile) {
            let n = (rows-start).min(tile);
            a.query = query.buffer.address + (start*heads*dim*4) as u64;
            a.output = output.buffer.address + (start*heads*dim*4) as u64;
            a.position = (position+start) as u32; a.rows = n as u32;
            unsafe {
                self.enqueue_grid(k.bounded_scores, capacity*32, heads as u32, n as u32, &mut [parameter(&mut a)])?;
                self.enqueue_grid(k.bounded_softmax, heads*32, n as u32, 1, &mut [parameter(&mut a)])?;
                self.enqueue_grid(k.bounded_values, dim, heads as u32, n as u32, &mut [parameter(&mut a)])?;
            }
        }
        Ok(())
    }
}
