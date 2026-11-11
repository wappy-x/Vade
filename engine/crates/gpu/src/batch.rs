//! Batched forward launches; weight loading remains in backend.rs.
use super::*;
#[path = "projection_params.rs"]
#[allow(clippy::manual_is_multiple_of)] // Shared with the NVPTX crate; fix it with the kernels.
pub(super) mod projection_params;
use projection_params::{DecodeProjection, DecodeGateUp, PackedDecodeProjection, GdnProjection};

pub(super) struct BatchKernels {
    copy_strided: Handle,
    pack_weight: Handle, packed_decode: Handle, strided_decode: Handle,
    split_decode: Handle, split_merge: Handle,
    decode_qkv: Handle, decode_gate_up: Handle,
    decode_gate_up_bf16: Handle, native_decode: Handle, native_narrow: Handle, gemm_decode_native: Handle,
    embedding: Handle, qk: Handle, scores: Handle, softmax: Handle, values: Handle,
    scores_gqa: Handle, values_split: Handle, values_merge: Handle,
    values_full: Handle, softmax_staged: Handle,
    prefill_wide_scores: Handle, prefill_wide_values: Handle,
    verify_mma: Handle, verify_blocked: Handle, gemm_decode_n32: Handle,
    pack_one: Handle, gemm_one: Handle, gemm_large: Handle, gemm_one_small: Handle, gemm_one_small_native: Handle, gemm_decode: Handle,
    gemm_residual_wide: Handle, gemm_residual_o: Handle,
    gemm_residual: Handle, gemm_residual_large: Handle, tiled_residual: Handle, tiled_residual_large: Handle,
    gemm_wide_exact: Handle, gemm_wide: Handle, gemm_wide_packed: Handle, gate_up_wide: Handle, qkv_wide: Handle, qkv_wide_packed: Handle,
    tiled_gemm: Handle, tiled_large: Handle, tiled_gate_up_packed: Handle,
    tiled_qkv: Handle,
    prefill_bf16: Handle, prefill_gqa4: Handle, prefill_gqa8: Handle, prefill_pack: Handle,
    prefill_fragments: Handle, prefill_swizzled_packed: Handle, prefill_swizzle: Handle,
    prefill_fast: Handle, prefill_fast_pack: Handle,
    qkv: Handle, qkv_decode: Handle, gate_up_packed: Handle, gate_up_decode: Handle,
    decode_native_multi: Handle, decode_native_multi64: Handle, decode_native_multi32: Handle, decode_native_multi80: Handle, gemm_one_bf16: Handle, gemm_narrow_bf16: Handle, gemm_wide_bf16: Handle, gemm_residual_bf16: Handle, gate_up_wide_bf16: Handle, prefill_bf16_256: Handle, qkv_bf16: Handle, gdn_inputs: Handle, gate_up_bf16: Handle,
    rms_pack: Handle,
    decode_stream: Handle, decode_stream_gqa8: Handle, decode_gqa2: Handle, decode_verify: Handle, decode_merge: Handle,
    pub(super) graph: [(Handle, Handle); 8],
}
impl BatchKernels {
    pub(super) fn kind(&self, function: Handle) -> Option<usize> {
        if [self.rms_pack, self.pack_weight, self.pack_one].contains(&function) { Some(10) }
        else if [self.prefill_pack, self.prefill_swizzle, self.prefill_fast_pack].contains(&function) { Some(11) }
        else if function == self.copy_strided { Some(12) }
        else if [self.decode_qkv, self.qkv, self.qkv_decode, self.tiled_qkv, self.qkv_wide, self.qkv_wide_packed].contains(&function) { Some(14) }
        else if [self.decode_gate_up].contains(&function) { Some(16) }
        else if [self.gate_up_packed, self.gate_up_decode, self.tiled_gate_up_packed, self.gate_up_wide].contains(&function) { Some(20) }
        else if [self.gemm_residual_wide, self.gemm_residual_o, self.gemm_one, self.gemm_large, self.gemm_wide_exact, self.gemm_wide, self.gemm_wide_packed, self.gemm_residual, self.gemm_residual_large, self.tiled_residual, self.tiled_residual_large, self.gemm_one_small, self.gemm_decode, self.packed_decode, self.split_decode, self.split_merge, self.verify_mma, self.verify_blocked, self.gemm_decode_n32,
            self.tiled_gemm, self.tiled_large].contains(&function) { Some(0) }
        else if function == self.qk { Some(2) } else if [self.decode_stream, self.decode_stream_gqa8, self.decode_gqa2, self.decode_verify, self.decode_merge, self.prefill_fast, self.prefill_bf16, self.prefill_gqa4, self.prefill_fragments, self.prefill_gqa8, self.prefill_swizzled_packed, self.prefill_wide_scores, self.scores, self.scores_gqa].contains(&function) { Some(3) }
        else if [self.qkv_bf16].contains(&function) { Some(14) }
        else if [self.decode_gate_up_bf16].contains(&function) { Some(16) }
        else if [self.gate_up_bf16, self.gate_up_wide_bf16].contains(&function) { Some(20) }
        else if [self.gemm_narrow_bf16, self.gemm_wide_bf16, self.gemm_residual_bf16, self.native_decode, self.decode_native_multi, self.decode_native_multi64, self.decode_native_multi32, self.decode_native_multi80].contains(&function) { Some(0) }
        else if [self.prefill_bf16_256].contains(&function) { Some(3) }
        else if [self.softmax, self.softmax_staged].contains(&function) { Some(4) }
        else if [self.values, self.values_split, self.values_merge, self.values_full, self.prefill_wide_values].contains(&function) { Some(5) }
        else if function == self.embedding { Some(8) } else { None }
    }
}
impl<'a> Backend<'a> {
    /// One bounded canonical BF16 entry for every phase. Input/output are row-major FP32.
    /// # Safety
    /// Weights belong to this context and remain immutable through stream completion.
    /// They cover N*K BF16 values: canonical 64-row packing when N%64=0,
    /// compact four-row packing when N%4=0 otherwise, row-major when N%4!=0.
    /// Generic packed8 narrow matrices are incompatible; the loader must bind this layout.
    pub unsafe fn fixed_bf16_projection(&self, weights: u64, dimensions: [usize; 2],
        input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, tokens: usize,
        round_output: bool, reuse_weights: bool) -> Result<(), String> {
        let shape = projection_params::FixedLinear::new(dimensions[0], dimensions[1], false)?;
        let (inputs, outputs) = shape.elements(tokens)?;
        if weights == 0 || !weights.is_multiple_of(8) || !input.buffer.address.is_multiple_of(8) {
            return Err("fixed BF16 operand alignment".into());
        }
        weights.checked_add(byte_count(dimensions[0], dimensions[1] * 2)? as u64)
            .ok_or("fixed BF16 weight address overflow")?;
        self.check_vector(input, inputs)?; self.check_vector(output, outputs)?;
        if input.is_segmented() || output.is_segmented() { return Err("fixed BF16 requires contiguous operands".into()); }
        projection_params::FixedLinear::disjoint(input.buffer.address, inputs * 4, output.buffer.address, outputs * 4)?;
        projection_params::FixedLinear::disjoint(weights, dimensions[0] * dimensions[1] * 2, output.buffer.address, outputs * 4)?;
        scratch.invalidate_projection_scratch();
        // Reuse the caller's reserved input workspace; no prompt-only allocation.
        // Both entries keep separate hi/lo K16 chains and the four-range merge.
        let plane_words = tokens.div_ceil(16) * dimensions[1] / 16 * 128;
        let words = plane_words * crate::kernel_params::TERMS;
        if (tokens > 256 && shape.outputs >= 64 || tokens >= 8 && shape.outputs >= 1024)
            && shape.outputs % 64 == 0 && shape.cols % 64 == 0
            && words <= scratch.len() && scratch.buffer.address.is_multiple_of(16) && !scratch.is_segmented() {
            self.check_vector_prefix(scratch, words)?;
            for (address, bytes) in [(input.buffer.address, inputs * 4), (output.buffer.address, outputs * 4),
                (weights, dimensions[0] * dimensions[1] * 2)] {
                projection_params::FixedLinear::disjoint(scratch.buffer.address, words * 4, address, bytes)?;
            }
            let mut pack = crate::kernel_params::FlashBf16Input { input: input.buffer.address,
                output: scratch.buffer.address, tokens: tokens as u32, cols: shape.cols, hc_silu: 0, lanes: 0 };
            let mut p = crate::kernel_params::Bf16Rows { weight: weights, output: output.buffer.address,
                rows: shape.outputs, stride: shape.outputs, weight2: 0, output2: 0, rows2: 0, stride2: 0,
                input: scratch.buffer.address, cols: shape.cols, tokens: tokens as u32,
                input_bf16: 1, hc_silu: 0, input_plane_words: plane_words as u64 };
            unsafe {
                self.launch_flash_shared("flash_bf16_input", plane_words.div_ceil(4).div_ceil(128) as u32,
                    128, 0, &mut [parameter(&mut pack)])?;
                self.launch_flash_shared("flash_bf16_mma64", shape.outputs.div_ceil(64) * (tokens as u32).div_ceil(64),
                    256, crate::kernel_params::FLASH_BF16_MMA_SHARED as u32, &mut [parameter(&mut p)])?;
            }
            if round_output { self.round_logits_bf16(output)?; }
            return Ok(());
        }
        let [tm, tn, threads, shared] = shape.tile_reusing_weights(tokens, reuse_weights);
        let blocks = shape.outputs.div_ceil(tn);
        if tm > 8 {
        for first in (0..tokens).step_by(tm as usize) {
            let mut p = crate::kernel_params::Bf16Rows { weight: weights,
                output: output.buffer.address + (first * dimensions[0] * 4) as u64,
                rows: shape.outputs, stride: shape.outputs, weight2: 0, output2: 0, rows2: 0, stride2: 0,
                input: input.buffer.address + (first * dimensions[1] * 4) as u64, cols: shape.cols,
                tokens: (tokens - first).min(tm as usize) as u32, input_bf16: 0, hc_silu: 0, input_plane_words: 0 };
            unsafe { self.launch_flash_shared("flash_bf16_mma1_prompt", blocks, threads,
                shared, &mut [parameter(&mut p)])?; }
        }
        } else {
        // Grid Y covers the M8 token tiles that previously took one launch each.
        let mut p = crate::kernel_params::Bf16Rows { weight: weights, output: output.buffer.address,
            rows: shape.outputs, stride: shape.outputs, weight2: 0, output2: 0, rows2: 0, stride2: 0,
            input: input.buffer.address, cols: shape.cols,
            tokens: tokens as u32, input_bf16: 0, hc_silu: 0, input_plane_words: 0 };
        unsafe { self.launch_flash_rows("flash_bf16_mma1_rows", blocks, tokens.div_ceil(tm as usize) as u32, threads,
            shared, &mut [parameter(&mut p)])?; }
        }
        if round_output { self.round_logits_bf16(output)?; }
        Ok(())
    }
    /// Pair canonical narrow BF16 matrices without changing either K reduction.
    /// # Safety
    /// Both weights have the canonical layout/context required by fixed_bf16_projection.
    pub unsafe fn fixed_bf16_pair(&self, weights: [u64;2], dimensions: [usize;2], input: &DeviceVector<'_>,
        first: &mut DeviceVector<'_>, second: &mut DeviceVector<'_>, tokens: usize) -> Result<(),String> {
        let [n,k]=dimensions; let shape=projection_params::FixedLinear::new(n,k,false)?;
        let (inputs,outputs)=shape.elements(tokens)?;
        if n>=512 || n%4!=0 || k<=512 || !input.buffer.address.is_multiple_of(8) || weights.iter().any(|w|w%8!=0) { return Err("paired narrow BF16 geometry/alignment".into()); }
        for (v,len) in [(input,inputs),(&*first,outputs),(&*second,outputs)] { self.check_vector(v,len)?;
            if v.is_segmented() { return Err("paired BF16 requires contiguous operands".into()); } }
        let spans=[(input.buffer.address,inputs*4),(first.buffer.address,outputs*4),(second.buffer.address,outputs*4)];
        for i in 0..3 { for j in i+1..3 { projection_params::FixedLinear::disjoint(spans[i].0,spans[i].1,spans[j].0,spans[j].1)?; } }
        for w in weights { for &(out,bytes) in &spans[1..] { projection_params::FixedLinear::disjoint(w,n*k*2,out,bytes)?; } }
        let mut p=crate::kernel_params::Bf16Rows { weight:weights[0],output:first.buffer.address,rows:n as u32,stride:n as u32,
            weight2:weights[1],output2:second.buffer.address,rows2:n as u32,stride2:n as u32,
            input:input.buffer.address,cols:k as u32,tokens:tokens as u32,input_bf16:0,hc_silu:0,input_plane_words:0 };
        unsafe { self.launch_flash_rows("flash_bf16_mma1_rows",2*(n as u32).div_ceil(8),tokens.div_ceil(8) as u32,
            128,0,&mut [parameter(&mut p)])?; }
        self.round_logits_bf16(first)?; self.round_logits_bf16(second)
    }
    /// Read a contiguous FP32 range at a compute-stream boundary.
    pub fn read_vector_range(&self, vector: &DeviceVector<'_>, offset: usize, len: usize) -> Result<Vec<f32>, String> {
        self.check_vector(vector, vector.len)?;
        if offset.checked_add(len).is_none_or(|end| end > vector.len) { return Err("read range outside vector".into()); }
        let mut output = vec![0.0; len];
        self.read_vector_range_into(vector, offset, &mut output)?;
        Ok(output)
    }
    /// Copy raw FP32 bits directly into a reusable host exchange buffer.
    pub fn read_vector_range_into(&self, vector: &DeviceVector<'_>, offset: usize, output: &mut [f32]) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        if offset.checked_add(output.len()).is_none_or(|end| end > vector.len) { return Err("read range outside vector".into()); }
        self.synchronize()?;
        if !output.is_empty() {
            self.context.activate()?;
            unsafe { check(cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), vector.buffer.address + byte_count(offset, 4)? as u64,
                std::mem::size_of_val(output)), "range read")?; }
        }
        Ok(())
    }
    /// Queue a contiguous FP32 range upload on the compute stream.
    pub fn write_vector_range(&self, vector: &mut DeviceVector<'_>, offset: usize, values: &[f32]) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        if offset.checked_add(values.len()).is_none_or(|end| end > vector.len) { return Err("write range outside vector".into()); }
        self.write_buffer(&vector.buffer, byte_count(offset, 4)?, values)
    }
    /// Queue a contiguous device copy without staging through the host.
    pub fn copy_vector_range(&self, source: &DeviceVector<'_>, destination: &mut DeviceVector<'_>,
        source_offset: usize, destination_offset: usize, len: usize) -> Result<(), String> {
        self.copy_range(source, destination, source_offset, destination_offset, len)
    }
    pub fn require_batch_kernels(&self) -> Result<(), String> {
        if self.batch.borrow().is_none() {
            let layer = self.layer_module()?;
            *self.batch.borrow_mut() = Some(BatchKernels {
                graph: [
                    (layer.function("batch_qk_rope")?, layer.function("graph_batch_qk_rope")?),
                    (layer.function("batch_scores_gqa4")?, layer.function("graph_batch_scores_gqa4")?),
                    (layer.function("batch_softmax")?, layer.function("graph_batch_softmax")?),
                    (layer.function("batch_values_split_gqa4")?, layer.function("graph_batch_values_split_gqa4")?),
                    (self.module.function("decode_attention_stream")?, self.module.function("graph_decode_attention_stream")?),
                    (self.module.function("decode_attention_stream_gqa8")?, self.module.function("graph_decode_attention_stream_gqa8")?),
                    (self.module.function("decode_attention_gqa2")?, self.module.function("graph_decode_attention_gqa2")?),
                    (self.module.function("decode_attention_merge")?, self.module.function("graph_decode_attention_merge")?),
                ],
                copy_strided: layer.function("copy_strided")?,
                decode_stream: self.module.function("decode_attention_stream")?,
                decode_stream_gqa8: self.module.function("decode_attention_stream_gqa8")?,
                decode_gqa2: self.module.function("decode_attention_gqa2")?,
                decode_verify: self.module.function_with_shared("decode_attention_verify", 98304)?,
                decode_merge: self.module.function("decode_attention_merge")?,
                prefill_bf16: self.module.function_with_shared("prefill_attention_bf16", 65536)?,
                prefill_gqa4: self.module.function_with_shared("prefill_attention_gqa4_bf16", 32768)?,
                prefill_fragments: self.module.function_with_shared("prefill_attention_gqa4_fragments", 32768)?,
                gemm_residual_wide: self.module.function_with_shared("gemm_residual_rms_t256n128_tiled", 98304)?,
                gemm_residual_o: self.module.function_with_shared("gemm_residual_rms_o_t128n256", 98304)?,
                prefill_gqa8: self.module.function_with_shared("prefill_attention_swizzled_bf16", 32768)?,
                prefill_swizzled_packed: self.module.function_with_shared("prefill_attention_swizzled_packed_bf16", 32768)?,
                prefill_swizzle: self.module.function("prefill_pack_swizzled_bf16")?,
                prefill_bf16_256: self.module.function_with_shared("prefill_attention_256_bf16", 65536)?,
                prefill_fast: self.module.function_with_shared("prefill_attention_fast_adaptive", 65536)?,
                prefill_fast_pack: self.module.function("prefill_fast_pack")?,
                prefill_pack: self.module.function("prefill_pack_kv")?,
                gate_up_packed: self.module.function("prefill_gate_up_pair_packed")?,
                gate_up_wide: self.module.function_with_shared("prefill_gate_up_pair_t256n64_tiled", 98304)?,
                gemm_wide_exact: self.module.function_with_shared("gemm_bf16_mma_t256n128_exact", 49152)?,
                gemm_wide: self.module.function_with_shared("gemm_bf16_mma_t256n128_tiled", 98304)?,
                qkv_wide: self.module.function_with_shared("prefill_qkv_t256n128_tiled", 98304)?,
                gemm_wide_packed: self.module.function_with_shared("gemm_bf16_mma_t256n128_packed", 98304)?,
                qkv_wide_packed: self.module.function_with_shared("prefill_qkv_t256n128_packed", 98304)?,
                qkv: self.module.function("prefill_qkv")?,
                qkv_decode: self.module.function("gemm_bf16_decode_qkv")?,
                gate_up_decode: self.module.function("gemm_bf16_decode_gate_up")?,
                gate_up_bf16: self.module.function("prefill_gate_up_pair_bf16")?,
                gate_up_wide_bf16: self.module.function_with_shared("prefill_gate_up_pair_t256n64_bf16", 98304)?,
                gemm_wide_bf16: self.module.function_with_shared("gemm_bf16_mma_t256n128_bf16", 98304)?,
                gemm_residual_bf16: self.module.function_with_shared("gemm_residual_bf16_t256n128", 98304)?,
                qkv_bf16: self.module.function("prefill_qkv_bf16")?,
                gdn_inputs: self.module.function("prefill_gdn_bf16")?,
                rms_pack: self.module.function("rms_pack_bf16")?,
                pack_weight: self.module.function("pack_bf16_mma_weights")?,
                packed_decode: self.module.function("matvec_bf16_f32_packed8")?,
                strided_decode: self.module.function("matvec_bf16_f32_packed8_strided")?,
                split_decode: self.module.function("matvec_bf16_f32_split")?,
                split_merge: self.module.function("matvec_split_merge")?,
                decode_qkv: self.module.function("matvec_qkv_packed")?,
                decode_gate_up: self.module.function("gate_up_swiglu_packed")?,
                decode_gate_up_bf16: self.module.function("gate_up_swiglu_bf16_packed")?,
                gemm_decode_native: self.module.function("gemm_bf16_mma_t16n64_native")?,
                native_decode: self.module.function("matvec_bf16_native")?,
                native_narrow: self.module.function("matvec_bf16_native_narrow")?,
                embedding: layer.function("batch_embedding")?, qk: layer.function("batch_qk_rope")?,
                scores: layer.function("batch_scores")?, softmax: layer.function("batch_softmax")?,
                values: layer.function("batch_values")?,
                values_full: layer.function("batch_values_gqa8_full")?,
                softmax_staged: layer.function("batch_softmax_staged")?,
                scores_gqa: layer.function("batch_scores_gqa4")?, values_split: layer.function("batch_values_split_gqa4")?,
                values_merge: layer.function("batch_values_merge")?,
                prefill_wide_scores: self.module.function_with_shared("prefill_attention_wide_scores", 16384)?,
                prefill_wide_values: self.module.function_with_shared("prefill_attention_wide_values", 16384)?,
                verify_mma: self.module.function("verify_bf16_f32_mma")?,
                verify_blocked: self.module.function_with_shared("verify_bf16_blocked_mma", 98304)?,
                gemm_decode_n32: self.module.function("gemm_bf16_decode_n32")?,
                pack_one: self.module.function("pack_f32_bf16_mma_input")?,
                gemm_one: self.module.function("gemm_bf16_mma_pipe3_strided")?,
                gemm_large: self.module.function_with_shared("gemm_bf16_mma_m256_strided", 73728)?,
                gemm_residual: self.module.function("gemm_residual_rms_m128")?,
                gemm_residual_large: self.module.function_with_shared("gemm_residual_rms_m256", 73728)?,
                tiled_residual: self.module.function("gemm_residual_rms_m128_tiled")?,
                tiled_residual_large: self.module.function_with_shared("gemm_residual_rms_m256_tiled", 73728)?,
                gemm_one_small: self.module.function("gemm_bf16_mma_t32n48_strided")?,
                gemm_one_small_native: self.module.function("gemm_bf16_mma_t32n48_native")?,
                gemm_decode: self.module.function("gemm_bf16_decode")?,
                tiled_gemm: self.module.function("gemm_bf16_mma_pipe3_tiled")?,
                tiled_large: self.module.function_with_shared("gemm_bf16_mma_m256_tiled", 73728)?,
                tiled_gate_up_packed: self.module.function("prefill_gate_up_pair_tiled")?,
                tiled_qkv: self.module.function("prefill_qkv_tiled")?,
                decode_native_multi: self.module.function("decode_bf16_multi")?,
                decode_native_multi64: self.module.function_with_shared("decode_bf16_multi64", 98304)?,
                decode_native_multi32: self.module.function_with_shared("decode_bf16_multi32", 49152)?,
                decode_native_multi80: self.module.function_with_shared("decode_bf16_multi80", 98304)?,
                gemm_one_bf16: self.module.function("gemm_bf16_mma_pipe3_strided_bf16")?,
                gemm_narrow_bf16: self.module.function("gemm_bf16_mma_t16n48_bf16")?,
            });
        }
        Ok(())
    }
    pub fn fast_prefill_attention(&self, prompt_end: usize) -> bool {
        prompt_end >= 6144 && std::env::var("SPARK_ADAPTIVE_PREFILL").is_ok_and(|value| value == "1")
    }
    pub fn matrix_is_packed(&self, matrix: &Matrix<'_>) -> bool {
        matrix.packed
    }
    pub fn batch_input_components(&self) -> Result<usize,String> {
        self.require_batch_kernels()?;
        Ok(1)
    }

    pub(super) unsafe fn packed_matvec(&self, matrix: &Matrix<'_>, args: &mut [*mut c_void]) -> Result<(), String> {
        self.packed_matvec_rows(matrix,args,matrix.rows)
    }
    pub(super) unsafe fn packed_matvec_rows(&self, matrix: &Matrix<'_>, args: &mut [*mut c_void], rows: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        if !matrix.cols.is_multiple_of(16) || !matrix.buffer.address.is_multiple_of(16) { return Err("packed matvec requires aligned K divisible by 16".into()); }
        let previous = self.profile_kind.get();
        if self.profiling.get() && previous.is_none() { self.profile_kind.set(Some(if matrix.cols > matrix.rows {17} else {15})); }
        let kernels = self.batch.borrow();
        let kernels = kernels.as_ref().unwrap();
        if matrix.stride != matrix.cols || matrix.row_offset != 0 || matrix.column_offset != 0 {
            let (mut stride, mut offset, mut column) = (matrix.stride as u32, matrix.row_offset as u32, matrix.column_offset as u32);
            let mut view_args = [args[0], args[1], args[2], args[3], args[4], parameter(&mut stride), parameter(&mut offset), parameter(&mut column)];
            let result = self.enqueue_dims(kernels.strided_decode, (rows + matrix.row_offset % 8).div_ceil(16) as u32, 1, 1, 64, &mut view_args);
            self.profile_kind.set(previous);
            return result;
        }
        // The draft's O/down matrices have too few row tiles to hide memory
        // latency. Four column ranges reuse a single 32 KiB backend workspace.
        if matches!(matrix.cols, 2048 | 3072) && rows <= 2048 && rows.is_multiple_of(8) {
            let result = (|| {
                let mut scratch = self.decode_partials.borrow_mut();
                if scratch.is_none() { *scratch = Some(self.vector(4 * 2048)?); }
                let mut partials = scratch.as_ref().unwrap().buffer.address;
                let mut split_args = [args[0], args[1], parameter(&mut partials), args[3], args[4]];
                self.enqueue_dims(kernels.split_decode, rows.div_ceil(16) as u32, 4, 1, 64, &mut split_args)?;
                let mut a = crate::kernel_params::ProjectionMerge { input: partials,
                    output: *(args[2] as *const u64), elements: rows as u32, round_bf16: 0 };
                self.enqueue(kernels.split_merge, rows.div_ceil(4), &mut [parameter(&mut a)])
            })();
            self.profile_kind.set(previous);
            return result;
        }
        let result = self.enqueue_dims(kernels.packed_decode, rows.div_ceil(16) as u32, 1, 1, 64, args);
        self.profile_kind.set(previous);
        result
    }
    /// Input is already rounded at the BF16 model's activation boundaries.
    /// Projection rounds its FP32 accumulator to BF16 before the next operation.
    pub fn matvec_bf16_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if matrix.is_fp8() { return self.fp8_dense_decode(matrix, input, output, 1); }
        self.require_batch_kernels()?;
        if !std::ptr::eq(matrix.buffer._context, self.context) || !matrix.packed
            || matrix.stride != matrix.cols || matrix.row_offset != 0 || !matrix.cols.is_multiple_of(16) {
            return Err("native BF16 projection requires a complete packed matrix".into());
        }
        self.check_vector(input, matrix.cols)?; self.check_vector(output, matrix.rows)?;
        if !input.buffer.address.is_multiple_of(8) { return Err("native BF16 input must be 8-byte aligned".into()); }
        let (mut w, mut x, mut y) = (matrix.buffer.address, input.buffer.address, output.buffer.address);
        let (mut rows, mut width) = (matrix.rows as u32, matrix.cols as u32);
        if matrix.rows <= 64 {
            let mut params = DecodeProjection { weights: w, input: x, output: y, rows, width, tokens: 1 };
            return self.profile_phase(14, || unsafe {
                self.enqueue_dims_shared(self.batch.borrow().as_ref().unwrap().native_narrow, rows, 1, 1, 128, width * 4,
                    &mut [parameter(&mut params)]) });
        }
        let threads = if matches!(matrix.cols, 5120 | 6144) { 64 } else { 128 };
        self.profile_phase(if matrix.cols == 17408 {17} else if matrix.cols == 6144 {15} else {14}, || unsafe {
            self.enqueue_dims(self.batch.borrow().as_ref().unwrap().native_decode, matrix.rows.div_ceil(threads / 4) as u32, 1, 1, threads as u32,
                &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut rows), parameter(&mut width)]) })
    }
    /// Decode Q/K/V in one launch, retaining the separate packed weight storage.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_device(&self, query: &Matrix<'_>, key: &Matrix<'_>, value: &Matrix<'_>, input: &DeviceVector<'_>,
        q: &mut DeviceVector<'_>, k: &mut DeviceVector<'_>, v: &mut DeviceVector<'_>) -> Result<(), String> {
        if query.cols != key.cols || query.cols != value.cols || key.rows != value.rows { return Err("decode QKV shape mismatch".into()); }
        if !query.packed || !key.packed || !value.packed || !query.rows.is_multiple_of(8) || !key.rows.is_multiple_of(8)
            || [query, key, value].iter().any(|m| m.row_offset != 0 || m.stride != m.cols) {
            self.matvec_device(query, input, q)?;
            self.matvec_device(key, input, k)?;
            return self.matvec_device(value, input, v);
        }
        self.require_batch_kernels()?;
        for matrix in [query, key, value] {
            if !std::ptr::eq(matrix.buffer._context, self.context) { return Err("matrix belongs to another context".into()); }
            if matrix.cols % 16 != 0 || matrix.buffer.address % 16 != 0 { return Err("packed matvec requires aligned K divisible by 16".into()); }
        }
        self.check_vector(input, query.cols)?;
        if !input.buffer.address.is_multiple_of(8) { return Err("packed decode input requires 8-byte alignment".into()); }
        self.check_vector(q, query.rows)?; self.check_vector(k, key.rows)?; self.check_vector(v, value.rows)?;
        let (mut qw, mut kw, mut vw, mut x) = (query.buffer.address, key.buffer.address, value.buffer.address, input.buffer.address);
        let (mut qo, mut ko, mut vo) = (q.buffer.address, k.buffer.address, v.buffer.address);
        let (mut q_rows, mut kv_rows, mut width) = (query.rows as u32, key.rows as u32, query.cols as u32);
        unsafe { self.enqueue_dims(self.batch.borrow().as_ref().unwrap().decode_qkv, (query.rows + 2 * key.rows).div_ceil(16) as u32, 1, 1, 64,
            &mut [parameter(&mut qw), parameter(&mut kw), parameter(&mut vw), parameter(&mut x), parameter(&mut qo), parameter(&mut ko), parameter(&mut vo),
                parameter(&mut q_rows), parameter(&mut kv_rows), parameter(&mut width)]) }
    }
    /// Canonical packed decode arithmetic with SwiGLU applied before storing gate.
    pub fn gate_up_swiglu_device(&self, gate: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, up_scratch: &mut DeviceVector<'_>) -> Result<(), String> {
        self.gate_up_swiglu_precision_device(gate, up, input, output, up_scratch, false)
    }
    /// Native BF16 gate/up and activation rounding, fused into the projection.
    pub fn gate_up_swiglu_bf16_device(&self, gate: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, up_scratch: &mut DeviceVector<'_>) -> Result<(), String> {
        self.gate_up_swiglu_precision_device(gate, up, input, output, up_scratch, true)
    }
    fn gate_up_swiglu_precision_device(&self, gate: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, up_scratch: &mut DeviceVector<'_>, bf16: bool) -> Result<(), String> {
        if gate.rows != up.rows || gate.cols != up.cols { return Err("decode gate/up shape mismatch".into()); }
        if !gate.packed || !up.packed || [gate, up].iter().any(|m| m.row_offset != 0 || m.stride != m.cols) {
            self.matvec_device(gate, input, output)?;
            if up_scratch.len < up.rows {
                let mut scratch = self.vector(up.rows)?;
                self.matvec_device(up, input, &mut scratch)?;
                return self.swiglu_device(output, &scratch);
            }
            self.matvec_device(up, input, up_scratch)?;
            return self.swiglu_device(output, up_scratch);
        }
        self.require_batch_kernels()?;
        for matrix in [gate, up] {
            if !std::ptr::eq(matrix.buffer._context, self.context) { return Err("matrix belongs to another context".into()); }
            if matrix.cols % 16 != 0 || matrix.buffer.address % 16 != 0 { return Err("packed matvec requires aligned K divisible by 16".into()); }
        }
        self.check_vector(input, gate.cols)?; self.check_vector(output, gate.rows)?;
        if !input.buffer.address.is_multiple_of(8) { return Err("packed decode input requires 8-byte alignment".into()); }
        let (mut gw, mut uw, mut x, mut y) = (gate.buffer.address, up.buffer.address, input.buffer.address, output.buffer.address);
        let (mut rows, mut width) = (gate.rows as u32, gate.cols as u32);
        let kernels = self.batch.borrow();
        let kernels = kernels.as_ref().unwrap();
        let function = if bf16 { kernels.decode_gate_up_bf16 } else { kernels.decode_gate_up };
        unsafe { self.enqueue_dims(function, gate.rows.div_ceil(16) as u32, 1, 1, 64,
            &mut [parameter(&mut gw), parameter(&mut uw), parameter(&mut x), parameter(&mut y), parameter(&mut rows), parameter(&mut width)]) }
    }
    pub fn batch_embedding_device(&self, table: &Matrix<'_>, ids: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        self.generic_read_bytes(|| (10, ids.len * table.cols * 2));
        self.require_batch_kernels()?;
        if !std::ptr::eq(table.buffer._context, self.context) { return Err("embedding belongs to another context".into()); }
        if table.rows == 0 || table.rows > 1 << 24 || table.cols == 0 || ids.len == 0
            || table.row_offset != 0 || table.column_offset != 0 || table.stride != table.cols {
            return Err("invalid batch embedding shape".into());
        }
        self.check_vector(ids, ids.len)?;
        self.check_vector(output, ids.len.checked_mul(table.cols).ok_or("batch embedding size overflow")?)?;
        let mut args = crate::kernel_params::BatchEmbedding {
            table: table.buffer.address, ids: ids.buffer.address, output: output.buffer.address,
            width: u32::try_from(table.cols).map_err(|_| "embedding width exceeds kernel ABI")?,
            rows: table.rows as u32, packed: table.embedding_layout(),
        };
        let rows = u32::try_from(ids.len).map_err(|_| "embedding row count exceeds kernel ABI")?;
        unsafe { self.enqueue_grid(self.batch.borrow().as_ref().unwrap().embedding, table.cols, rows, 1,
            &mut [parameter(&mut args)]) }
    }
    /// Independent BF16 decode rows reuse each staged weight tile, retaining
    /// the scalar packed8 sum order. Input/output are promoted BF16 values.
    pub fn matmul_decode_bf16_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        if matrix.is_fp8() { return self.fp8_dense_decode(matrix, input, output, tokens); }
        self.require_batch_kernels()?;
        if !ptr::eq(matrix.buffer._context, self.context) || !matrix.packed
            || matrix.row_offset != 0 || matrix.stride != matrix.cols || !matrix.rows.is_multiple_of(16)
            || !matrix.cols.is_multiple_of(128) || !(1..=80).contains(&tokens) {
            return Err("shared BF16 decode requires complete packed weights, K%128, N%16 and 1..=80 rows".into());
        }
        self.check_vector(input, tokens * matrix.cols)?;
        self.check_vector(output, tokens * matrix.rows)?;
        let mut params = DecodeProjection { weights: matrix.buffer.address,
            input: input.buffer.address, output: output.buffer.address,
            rows: matrix.rows as u32, width: matrix.cols as u32, tokens: tokens as u32 };
        if matrix.rows <= 64 {
            return self.profile_phase(14, || unsafe {
                self.enqueue_dims_shared(self.batch.borrow().as_ref().unwrap().native_narrow,
                    params.rows, params.tokens, 1, 128, params.width * 4,
                    &mut [parameter(&mut params)]) });
        }
        let threads = if (17..=32).contains(&tokens) { 1024 } else if tokens <= 8 || tokens > 32 { 256 } else { 512 };
        let kernels = self.batch.borrow(); let kernels = kernels.as_ref().unwrap();
        let (function, tile, shared) = if (17..=32).contains(&tokens) {
            (kernels.decode_native_multi32, 24, 49152)
        } else if tokens > 32 {
            (kernels.decode_native_multi80, 16, 98304)
        } else if matrix.rows.is_multiple_of(64) && matrix.cols.is_multiple_of(256) {
            (kernels.decode_native_multi64, 64, 98304)
        } else { (kernels.decode_native_multi, 16, 0) };
        unsafe { self.enqueue_dims_shared(function,
            1, matrix.rows.div_ceil(tile as usize) as u32, 1, threads, shared,
            &mut [parameter(&mut params)]) }
    }
    /// One matrix-sized reusable weight pack, no retained duplicate model.
    pub fn matmul_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, exact: bool) -> Result<(), String> {
        self.matmul_reusing_input(matrix, input, output, weight_pack, input_pack, tokens, exact, false)
    }
    /// Reuse only after packing the same input/shape on this stream (QKV, gate/up).
    /// With reuse and non-exact projection, only input_pack is read.
    pub fn matmul_reusing_input(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, exact: bool, reuse: bool) -> Result<(), String> {
        self.matmul_prefill_device(matrix, input, output, weight_pack, input_pack, tokens, exact, reuse, false)
    }
    /// Hybrid narrow prefill uses packed8 weights without changing the sum order.
    pub fn matmul_prefill_native_short_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        packed: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        if !matrix.packed || !ptr::eq(matrix.buffer._context, self.context) || matrix.rows > 48
            || !matrix.cols.is_multiple_of(16) || matrix.row_offset != 0 || matrix.stride != matrix.cols || tokens == 0 {
            return Err("native narrow prefill shape".into());
        }
        self.check_vector(output, tokens * matrix.rows)?;
        self.check_vector(packed, packed.len)?;
        if packed.len < tokens.div_ceil(16) * 16 * matrix.cols / 2 { return Err("native narrow packing capacity".into()); }
        if !reuse { self.pack_prefill_bf16_device(input, packed, matrix.cols, tokens)?; }
        let mut params = DecodeProjection { weights: matrix.buffer.address, input: packed.buffer.address,
            output: output.buffer.address, rows: matrix.rows as u32, width: matrix.cols as u32, tokens: tokens as u32 };
        unsafe { self.enqueue_dims(self.batch.borrow().as_ref().unwrap().gemm_one_small_native,
            tokens.div_ceil(32) as u32, matrix.rows.div_ceil(96) as u32, 1, 64, &mut [parameter(&mut params)]) }
    }
    /// BF16 operands; decode/verify use four fixed K ranges and a balanced merge.
    pub fn matmul_prefill_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, exact: bool, reuse: bool, _fast: bool) -> Result<(), String> {
        self.matmul_prefill_precision_device(matrix, input, output, weight_pack, input_pack, tokens, exact, reuse, false, false)
    }
    /// Gemma's canonical four-range projection crosses an explicit BF16 boundary.
    pub fn matmul_bf16_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<(), String> {
        self.matmul_prefill_precision_device(matrix, input, output, weight_pack, input_pack, tokens, true, reuse, false, true)
    }
    /// Round full prefill tiles in their GEMM stores, with explicit bounded-tail rounding.
    pub fn matmul_prefill_bf16_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<(), String> {
        if matrix.is_fp8() { return self.fp8_dense_prefill(matrix, input, output, weight_pack, input_pack, tokens, reuse); }
        self.matmul_prefill_precision_device(matrix, input, output, weight_pack, input_pack, tokens, false, reuse, true, false)?;
        if matrix.row_offset == 0 && matrix.rows <= 48 && tokens >= 256 { return Ok(()); }
        let wide = matrix.row_offset == 0 && matrix.cols.is_multiple_of(64) && matrix.rows.is_multiple_of(128);
        self.round_prefill_tail(output, matrix.rows, tokens, wide)
    }
    fn round_prefill_tail(&self, output: &mut DeviceVector<'_>, width: usize, tokens: usize, wide: bool) -> Result<(), String> {
        let full = if wide { tokens / 256 * 256 } else { 0 };
        if full == tokens { return Ok(()); }
        self.round_logits_bf16(&mut output.slice(full * width..tokens * width)?)
    }
    fn matmul_prefill_precision_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, exact: bool, reuse: bool, bf16: bool, round_bf16: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        let decode_view = matrix.row_offset != 0 || matrix.column_offset != 0 || exact && matrix.stride != matrix.cols;
        if !std::ptr::eq(matrix.buffer._context, self.context) { return Err("matrix belongs to another context".into()); }
        self.check_vector(input, if reuse && !exact && !decode_view { input.len } else { tokens * matrix.cols })?;
        self.check_vector(output, tokens * matrix.rows)?;
        self.check_vector(weight_pack, weight_pack.len)?; self.check_vector(input_pack, input_pack.len)?;
        if !matrix.cols.is_multiple_of(16) || tokens == 0 { return Err("MMA requires K divisible by 16 and nonempty rows".into()); }
        let xf = tokens.div_ceil(16) * (matrix.cols / 16) * 32;
        if !matrix.packed { return Err("batch projections require packed weights".into()); }
        if !exact && input_pack.len < xf * 4 { return Err("MMA packing scratch too small".into()); }
        let (mut x, mut y) = (input.buffer.address, output.buffer.address);
        let (mut wp, mut xp) = (matrix.buffer.address, input_pack.buffer.address);
        if wp % 16 != 0 || xp % 16 != 0 || x % 4 != 0 || y % 4 != 0 { return Err("unaligned projection storage".into()); }
        let (mut n, mut k, mut t) = (matrix.rows as u32, matrix.cols as u32, tokens as u32);
        let (mut stride, mut offset, mut column) = (matrix.stride as u32, matrix.row_offset as u32, matrix.column_offset as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        let mut fused_boundary = false;
        let result = (|| unsafe {
            // Pack once for Q/K/V and gate/up, independent of cohort width.
            let exact_pack = exact && matrix.cols.is_multiple_of(128) && input_pack.len >= xf * 4;
            if exact_pack && !reuse {
                self.enqueue_dims(b.pack_one, xf.div_ceil(256) as u32, 1, 1, 256,
                    &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut k)])?;
            }
            // Matrix views use the same MMA order without a weight copy.
            if decode_view {
                if x % 8 != 0 { return Err("packed decode input requires 8-byte alignment".into()); }
                for _ in 0..tokens {
                    self.enqueue_dims(b.strided_decode,(matrix.rows + matrix.row_offset % 8).div_ceil(16) as u32,1,1,64,
                        &mut [parameter(&mut wp),parameter(&mut x),parameter(&mut y),parameter(&mut n),parameter(&mut k),parameter(&mut stride),parameter(&mut offset),parameter(&mut column)])?;
                    x += (matrix.cols*4) as u64; y += (matrix.rows*4) as u64;
                }
                return Ok(());
            }
            if exact {
                if round_bf16 && exact_pack && tokens.is_multiple_of(256) && matrix.rows.is_multiple_of(128) && matrix.cols.is_multiple_of(256) && matrix.stride.is_multiple_of(64) {
                    let mut a = crate::kernel_params::PrefillProjection { weights: wp, input: xp, output: y, round_bf16: u32::from(round_bf16),
                        width: n, inner: k, rows: t, stride };
                    fused_boundary = round_bf16;
                    return self.enqueue_dims_shared(b.gemm_wide_exact, tokens as u32 / 128, n / 64, 1, 256, 49152,
                        &mut [parameter(&mut a)]);
                }
                if exact_pack && matrix.rows.is_multiple_of(32) {
                    if tokens <= 64 && matrix.rows.is_multiple_of(64) && matrix.cols.is_multiple_of(256) {
                        let mut params = PackedDecodeProjection { weights: wp, input: xp, output: y,
                            rows: n, width: k, tokens: t, stride };
                        if tokens <= 16 && matrix.rows <= 8192 && matrix.cols >= 4096 {
                            return self.enqueue_dims(b.gemm_decode_n32, tokens.div_ceil(16) as u32, matrix.rows as u32/32, 1, 128,
                                &mut [parameter(&mut params)]);
                        }
                        return self.enqueue_dims(b.gemm_decode, tokens.div_ceil(16) as u32, matrix.rows as u32 / 64, 1, 256,
                            &mut [parameter(&mut params)]);
                    }
                    if matrix.rows >= 4096 && matrix.rows.is_multiple_of(64) && matrix.cols.is_multiple_of(256) {
                        return self.enqueue_dims_shared(b.verify_blocked, tokens.div_ceil(32) as u32, matrix.rows as u32 / 64, 1, 512, 98304,
                            &mut [parameter(&mut wp),parameter(&mut xp),parameter(&mut y),parameter(&mut n),parameter(&mut k),parameter(&mut t)]);
                    }
                    return self.enqueue_dims(b.verify_mma, tokens.div_ceil(32) as u32, matrix.rows as u32 / 32, 1, 256,
                        &mut [parameter(&mut wp),parameter(&mut xp),parameter(&mut y),parameter(&mut n),parameter(&mut k),parameter(&mut t)]);
                }
                // Small scratch and uneven shards use the same order row by row.
                if x % 8 != 0 { return Err("packed decode input requires 8-byte alignment".into()); }
                for _ in 0..tokens {
                    self.enqueue_dims(b.packed_decode,matrix.rows.div_ceil(16) as u32,1,1,64,
                        &mut [parameter(&mut wp),parameter(&mut x),parameter(&mut y),parameter(&mut n),parameter(&mut k)])?;
                    x += (matrix.cols*4) as u64; y += (matrix.rows*4) as u64;
                }
                return Ok(());
            }
            if !reuse { self.enqueue_dims(b.pack_one, xf.div_ceil(256) as u32, 1, 1, 256,
                &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut k)])?; }
            if bf16 {
            // Larger prefill uses the bounded K partitions; native chats keep
            // their existing reduction and the explicit BF16 store below.
            if bf16 && matrix.rows <= 48 && tokens >= 256 {
                return self.enqueue_dims(b.gemm_narrow_bf16, t.div_ceil(16), n.div_ceil(48), 1, 32,
                    &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
            }
            if tokens <= 16 && matrix.rows.is_multiple_of(64) && matrix.cols.is_multiple_of(64) {
                return self.enqueue_dims(b.gemm_decode_native, tokens.div_ceil(16) as u32, matrix.rows.div_ceil(64) as u32, 1, 128,
                    &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
            }
            if matrix.cols.is_multiple_of(64) && matrix.rows.is_multiple_of(128) {
                if bf16 && matrix.cols <= 8192 && tokens >= 256 {
                    return self.enqueue_dims(b.gemm_one_bf16, t.div_ceil(128), n / 128, 1, 256,
                        &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
                }
                if tokens >= 256 {
                    let mut full = t / 256 * 256;
                    self.enqueue_dims_shared(if bf16 { b.gemm_wide_bf16 } else { b.gemm_wide }, full / 256, n / 128, 1, 512, 98304,
                        &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut full), parameter(&mut stride)])?;
                    t -= full;
                    if t == 0 { return Ok(()); }
                    xp += full as u64 * k as u64 * 2;
                    y += full as u64 * n as u64 * 4;
                }
                return self.enqueue_dims(b.gemm_one, t.div_ceil(128), n / 128, 1, 256,
                    &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
            }
            }
            if self.decode_rows(tokens) && matrix.rows.is_multiple_of(64) && matrix.cols.is_multiple_of(256) {
                let mut params = PackedDecodeProjection { weights: wp, input: xp, output: y,
                    rows: n, width: k, tokens: t, stride };
                return self.enqueue_dims(b.gemm_decode, tokens.div_ceil(16) as u32, matrix.rows.div_ceil(64) as u32, 1, 256,
                    &mut [parameter(&mut params)]);
            }
            if matrix.cols.is_multiple_of(64) && matrix.rows.is_multiple_of(128) {
                if tokens.is_multiple_of(256) {
                    return self.enqueue_dims_shared(if matrix.stride.is_multiple_of(64) {b.gemm_wide} else {b.gemm_wide_packed}, tokens as u32 / 256, n / 128, 1, 512, 98304,
                        &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
                }
                if tokens >= 256 && matrix.cols >= 8192 {
                    return self.enqueue_dims_shared(if matrix.stride.is_multiple_of(64) {b.tiled_large} else {b.gemm_large}, tokens.div_ceil(256) as u32, matrix.rows as u32/128, 1, 256, 73728,
                        &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
                }
                return self.enqueue_dims(if matrix.stride.is_multiple_of(64) {b.tiled_gemm} else {b.gemm_one}, tokens.div_ceil(128) as u32, matrix.rows as u32/128, 1, 256,
                    &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]);
            }
            if bf16 {
                let mut params = DecodeProjection { weights: wp, input: xp, output: y, rows: n, width: k, tokens: t };
                return self.enqueue_dims(b.gemm_one_small_native, t.div_ceil(32), n.div_ceil(96), 1, 64, &mut [parameter(&mut params)]);
            }
            self.enqueue_dims(b.gemm_one_small, tokens.div_ceil(32) as u32, matrix.rows.div_ceil(96) as u32, 1, 64,
                &mut [parameter(&mut wp), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)])
        })();
        result?;
        if round_bf16 && !fused_boundary {
            self.model_elementwise_device(output, None, output, tokens*matrix.rows, 0, 1.0)?;
        }
        Ok(())
    }
    pub fn rms_partials_device(&self, input: &DeviceVector<'_>, partials: &mut DeviceVector<'_>, width: usize) -> Result<(), String> {
        if width == 0 || !input.len.is_multiple_of(width) { return Err("RMS partial shape".into()); }
        let rows = input.len / width;
        self.check_vector(input, input.len)?; self.check_vector(partials, rows * width.div_ceil(128))?;
        let (mut x, mut p, mut w) = (input.buffer.address, partials.buffer.address, width as u32);
        unsafe { self.enqueue_grid(self.layer_kernels()?.rms, width.div_ceil(128) * 32, rows as u32, 1,
            &mut [parameter(&mut x), parameter(&mut p), parameter(&mut w)]) }
    }
    pub fn rms_batch_device(&self, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, epsilon: f32) -> Result<(), String> {
        if weight.len > 0 && weight.len <= 4096 && input.len / weight.len <= 65535 {
            let rows = input.len / weight.len;
            self.check_vector(input, rows * weight.len)?; self.check_vector(weight, weight.len)?;
            self.check_vector(output, input.len)?;
            let (mut x, mut w, mut y, mut width, mut eps) = (input.buffer.address, weight.buffer.address,
                output.buffer.address, weight.len as u32, epsilon);
            return unsafe { self.enqueue_dims(self.layer_kernels()?.rms_fused, 1, rows as u32, 1,
                (weight.len.div_ceil(128) * 32) as u32,
                &mut [parameter(&mut x), parameter(&mut w), parameter(&mut y), parameter(&mut width), parameter(&mut eps)]) };
        }
        self.rms_partials_device(input, partials, weight.len)?;
        self.rms_batch_apply_device(input, weight, output, partials, epsilon)
    }

    /// Residual and normalization retain the exact two-stage sum in one CTA.
    pub fn residual_rms_batch_device(&self, state: &mut DeviceVector<'_>, residual: &DeviceVector<'_>,
        weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, epsilon: f32) -> Result<(), String> {
        if epsilon < 0.0 {
            self.residual_checkpoint_device(state, residual, true)?;
            return self.rms_batch_device(state, weight, output, partials, epsilon);
        }
        if weight.len > 0 && weight.len <= 4096 && state.len / weight.len <= 65535 {
            let rows = state.len / weight.len;
            self.check_vector(state, rows * weight.len)?; self.check_vector(residual, state.len)?;
            self.check_vector(weight, weight.len)?; self.check_vector(output, state.len)?;
            let (mut x, mut r, mut w, mut y, mut width, mut eps) = (state.buffer.address, residual.buffer.address,
                weight.buffer.address, output.buffer.address, weight.len as u32, epsilon);
            return unsafe { self.enqueue_dims(self.layer_kernels()?.residual_rms_fused, 1, rows as u32, 1,
                (weight.len.div_ceil(128) * 32) as u32,
                &mut [parameter(&mut x), parameter(&mut r), parameter(&mut w), parameter(&mut y), parameter(&mut width), parameter(&mut eps)]) };
        }
        self.residual_device(state, residual)?;
        self.rms_batch_device(state, weight, output, partials, epsilon)
    }
    /// Apply normalization using partials retained by the residual kernel.
    pub fn rms_batch_apply_device(&self, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, output: &mut DeviceVector<'_>, partials: &DeviceVector<'_>, epsilon: f32) -> Result<(), String> {
        if weight.len == 0 { return Err("empty RMS weight".into()); }
        let rows = input.len / weight.len;
        self.check_vector(input, rows * weight.len)?; self.check_vector(weight, weight.len)?; self.check_vector(output, input.len)?;
        self.check_vector(partials, rows * weight.len.div_ceil(128))?;
        let (mut x, mut w, mut y, mut p) = (input.buffer.address, weight.buffer.address, output.buffer.address, partials.buffer.address);
        let (mut width, mut eps) = (weight.len as u32, epsilon);
        let b = self.layer_kernels()?;
        unsafe {
            self.enqueue_grid(b.rms_apply, weight.len, rows as u32, 1,
                &mut [parameter(&mut x), parameter(&mut w), parameter(&mut p), parameter(&mut y), parameter(&mut width), parameter(&mut eps)])
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qk_batch_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>,
        keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>, position: usize, epsilon: f32, tokens: usize, format: impl Into<KvFormat>) -> Result<(), String> {
        self.qk_batch_range_device(query, key, value, q_norm, k_norm, rope, keys, values,
            position, epsilon, tokens, format.into(), None)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qk_batch_cached_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>, keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>,
        position: usize, epsilon: f32, tokens: usize, format: KvFormat, cache_base: usize) -> Result<(), String> {
        self.qk_batch_range_cached(query, key, value, q_norm, k_norm, rope, keys, values,
            position, epsilon, tokens, format, None, cache_base)
    }
    /// Sequence prefill computes K/V before exchange and Q after exchange.
    /// None projects both; Some(true) selects Q and Some(false) selects K/V.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_batch_range_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>,
        keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>, position: usize, epsilon: f32, tokens: usize, format: KvFormat,
        query_only: Option<bool>) -> Result<(), String> {
        self.qk_batch_range_cached(query, key, value, q_norm, k_norm, rope, keys, values,
            position, epsilon, tokens, format, query_only, 0)
    }
    #[allow(clippy::too_many_arguments)]
    fn qk_batch_range_cached(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>, keys: &mut DeviceVector<'_>, values: &mut DeviceVector<'_>,
        position: usize, epsilon: f32, tokens: usize, format: KvFormat, query_only: Option<bool>, cache_base: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        let dim = q_norm.len;
        if format == KvFormat::ScaledI16 && dim != 128 { return Err("scaled I16 KV requires head dimension 128".into()); }
        if tokens == 0 || dim == 0 { return Err("empty QK batch".into()); }
        let heads = query.len / tokens / dim;
        let kv_heads = key.len / tokens / dim;
        self.check_vector(q_norm, dim)?; self.check_vector(k_norm, dim)?;
        self.check_vector(rope, rope.len)?; self.check_vector(keys, keys.len)?;
        if kv_heads == 0 || heads == 0 || !dim.is_multiple_of(2) || position + tokens > rope.len / dim ||
            position < cache_base || (position + tokens - cache_base) * kv_heads * dim > format.element_capacity(keys.len) { return Err("batched QK exceeds capacity".into()); }
        self.check_vector(query, tokens * heads * dim)?; self.check_vector(key, tokens * kv_heads * dim)?;
        self.check_vector(value, key.len)?; self.check_vector(values, keys.len)?;
        let (p, replay_position) = self.decode_position(position);
        let first = if query_only == Some(false) { heads as u32 } else { 0 };
        let count = match query_only { Some(true) => heads, Some(false) => kv_heads, None => heads + kv_heads };
        let mut args = crate::kernel_params::QkRope {
            query: query.buffer.address, key: key.buffer.address, value: value.buffer.address,
            q_weight: q_norm.buffer.address, k_weight: k_norm.buffer.address, rope: rope.buffer.address,
            keys: keys.buffer.address, values: values.buffer.address,
            heads: heads as u32, kv_heads: kv_heads as u32, dim: dim as u32, position: p,
            epsilon, format: format as u32, first_head: first, last_head: first + count as u32,
            cache_base: cache_base as u32, replay_position, ..Default::default()
        };
        unsafe { self.enqueue_grid(self.batch.borrow().as_ref().unwrap().qk, count * 32, tokens as u32, 1, &mut args.batch_args()) }
    }
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // a zero divisor must still panic in %.
    pub fn attention_batch_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        scores: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>, heads: usize, kv_heads: usize, dim: usize,
        position: usize, tokens: usize, format: impl Into<KvFormat>) -> Result<(), String> {
        self.require_batch_kernels()?;
        let format = format.into();
        if format == KvFormat::ScaledI16 && dim != 128 { return Err("scaled I16 KV requires head dimension 128".into()); }
        if tokens == 0 || heads == 0 || kv_heads == 0 || dim == 0 { return Err("empty batch attention dimensions".into()); }
        if dim == 128 && heads == kv_heads * 2 && format == KvFormat::Bf16 && tokens == 1 && position < 4096 {
            self.check_vector(query, heads*dim)?; self.check_vector(output, heads*dim)?;
            if (position+1)*kv_heads*dim > format.element_capacity(keys.len) || values.len != keys.len {
                return Err("two-head attention exceeds KV capacity".into());
            }
            let (mut q, mut k, mut v, mut y) = (query.buffer.address, keys.buffer.address, values.buffer.address, output.buffer.address);
            let mut kv = kv_heads as u32;
            let (mut base, mut replay_position) = self.decode_position(position);
            let (mut parts, mut span, mut h) = ((position + 1).div_ceil(64) as u32, 64u32, heads as u32);
            if parts > 1 && partials.len < heads * 132 * parts as usize { return Err("two-head attention summary scratch too small".into()); }
            let mut summary = partials.buffer.address;
            let mut target = if parts == 1 { y } else { summary };
            let kernels = self.batch.borrow();
            let kernels = kernels.as_ref().unwrap();
            unsafe {
                self.enqueue_dims(kernels.decode_gqa2, parts, kv, 1, 128,
                    &mut [parameter(&mut q), parameter(&mut k), parameter(&mut v), parameter(&mut target), parameter(&mut kv), parameter(&mut base), parameter(&mut parts), parameter(&mut replay_position)])?;
                if parts > 1 {
                    self.enqueue_dims(kernels.decode_merge, h, 1, 1, 128,
                        &mut [parameter(&mut summary), parameter(&mut y), parameter(&mut h), parameter(&mut parts), parameter(&mut base), parameter(&mut span), parameter(&mut replay_position)])?;
                }
            }
            return Ok(());
        }
        if dim == 128 && (heads == kv_heads * 4 || heads == kv_heads * 8) {
            self.check_vector(query, tokens * heads * dim)?; self.check_vector(output, query.len)?;
            self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?;
            if (position + tokens) * kv_heads * dim > format.element_capacity(keys.len) {
                return Err("decode attention exceeds KV capacity".into());
            }
            let tile = tokens.min(16);
            let parts = self.decode_attention_parts(position + tokens);
            if parts > 1024 { return Err("decode attention exceeds summary capacity".into()); }
            self.check_vector(partials, partials.len)?;
            if parts > 1 && partials.len < tile * heads * 132 * parts { return Err("decode attention summary scratch too small".into()); }
            let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
            let (mut k, mut v, mut summary) = (keys.buffer.address, values.buffer.address, partials.buffer.address);
            let (mut kh, mut h, format) = (kv_heads as u32, heads as u32, format as u32);
            let mut start = 0;
            while start < tokens {
                let mut rows = (tokens - start).min(tile) as u32;
                if position + start < 4096 { rows = rows.min((4096 - position - start) as u32); }
                if position + start < 16384 { rows = rows.min((16384 - position - start) as u32); }
                let mut span = self.decode_attention_span(position + start + 1) as u32;
                let mut q = query.buffer.address + (start * heads * dim * 4) as u64;
                let mut y = output.buffer.address + (start * heads * dim * 4) as u64;
                let (mut base, mut replay_position) = self.decode_position(position + start);
                let mut parts = (position + start + rows as usize).div_ceil(span as usize) as u32;
                let mut target = if parts == 1 { y } else { summary };
                unsafe {
                    if heads == kv_heads * 4 && rows > 1 && base >= 16384 && format == KvFormat::Fp32 as u32 {
                        self.enqueue_dims_shared(b.decode_verify, parts, kh, 1, rows * 32, rows * 4096 + 32768,
                            &mut [parameter(&mut q), parameter(&mut k), parameter(&mut v), parameter(&mut target), parameter(&mut kh), parameter(&mut base), parameter(&mut parts), parameter(&mut span), parameter(&mut rows)])?;
                    } else {
                        let stream = if heads == kv_heads * 8 { b.decode_stream_gqa8 } else { b.decode_stream };
                        let groups = if format == KvFormat::Bf16 as u32 { kh } else { h / 4 };
                        let mut args = crate::kernel_params::DecodeStream { query: q, keys: k, values: v,
                            output: target, kv_heads: kh, base, parts, span, format };
                        let threads = if format == KvFormat::Bf16 as u32 { 64 } else { 128 };
                        self.enqueue_dims(stream, parts, groups, rows, threads,
                            &mut [parameter(&mut args), parameter(&mut replay_position)])?;
                    }
                    if parts > 1 {
                        self.enqueue_dims(b.decode_merge, h, rows, 1, 128,
                            &mut [parameter(&mut summary), parameter(&mut y), parameter(&mut h), parameter(&mut parts), parameter(&mut base), parameter(&mut span), parameter(&mut replay_position)])?;
                    }
                }
                start += rows as usize;
            }
            return Ok(());
        }
        let tile = tokens.min(64);
        let capacity = scores.len / heads / tile;
        if position + tokens > capacity || heads % kv_heads != 0 ||
            (position + tokens) * kv_heads * dim > format.element_capacity(keys.len) { return Err("batch attention dimensions".into()); }
        self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?;
        let full_values = heads == kv_heads * 8 && dim == 128 && matches!(format, KvFormat::Fp32 | KvFormat::Bf16);
        if full_values && partials.len < tile * heads { return Err("attention denominator scratch too small".into()); }
        if !full_values && partials.len < tile * heads * dim * (position + tokens).div_ceil(64) { return Err("attention partial scratch too small".into()); }
        self.check_vector(query, tokens * heads * dim)?; self.check_vector(output, query.len)?;
        // Split only launch rows; each row keeps its canonical softmax and
        // 64-key value partitions, including full native Llama contexts.
        let tile = if !full_values && matches!(heads / kv_heads, 4 | 5 | 8) {
            tile.min(65_535 / (position + tokens).div_ceil(64))
        } else { tile };
        if tile == 0 { return Err("attention value partitions exceed driver grid limit".into()); }
        let (k, mut v, mut s) = (keys.buffer.address, values.buffer.address, scores.buffer.address);
        let (mut h, mut kh, mut d, mut c, mut format) = (heads as u32, kv_heads as u32, dim as u32, capacity as u32, format as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        for start in (0..tokens).step_by(tile) {
        let mut rows = (tokens - start).min(tile) as u32;
        let q = query.buffer.address + (start * heads * dim * 4) as u64;
        let mut y = output.buffer.address + (start * heads * dim * 4) as u64;
        let (mut p, mut replay_position) = self.decode_position(position + start);
        let (mut parts, width, chunks) = (partials.buffer.address, (heads * dim) as u32, (position + start + rows as usize).div_ceil(64) as u32);
        let mut grouped_args = crate::kernel_params::GroupedAttention { query: q, keys: k, values: v,
            scores: s, partials: parts, output: y, replay_position, heads: h, kv_heads: kh, dim: d,
            position: p, capacity: c, format, rows, chunks, width };
        unsafe {
            let grouped = matches!(heads / kv_heads, 4 | 5 | 8);
            let (score_kernel, groups) = if grouped { (b.scores_gqa, kh) } else { (b.scores, h) };
            let score_args = &mut grouped_args.score_args();
            if heads == kv_heads * 8 && dim == 128 {
                self.enqueue_dims(score_kernel, (position + start + rows as usize).div_ceil(64) as u32, groups, rows.div_ceil(8), 256, score_args)?;
            } else {
                self.enqueue_grid(score_kernel, (position + start + rows as usize) * 32, groups, rows, score_args)?;
            }
            if full_values {
                self.enqueue_grid(b.softmax_staged, heads * 32, rows, 1,
                    &mut [parameter(&mut s), parameter(&mut parts), parameter(&mut h), parameter(&mut p), parameter(&mut c)])?;
                self.enqueue_dims(b.values_full, rows.div_ceil(4), kh, 1, 512,
                    &mut [parameter(&mut s), parameter(&mut v), parameter(&mut y), parameter(&mut h), parameter(&mut kh), parameter(&mut p), parameter(&mut c), parameter(&mut rows), parameter(&mut parts), parameter(&mut format)])?;
            } else {
                self.enqueue_grid(b.softmax, heads * 32, rows, 1,
                    &mut [parameter(&mut s), parameter(&mut h), parameter(&mut p), parameter(&mut c), parameter(&mut replay_position)])?;
                if grouped {
                    self.enqueue_grid(b.values_split, dim, kh, rows * chunks,
                        &mut grouped_args.value_args())?;
                    self.enqueue_grid(b.values_merge, heads * dim, rows, 1,
                        &mut grouped_args.merge_args())?;
                } else { self.enqueue_grid(b.values, dim, h, rows,
                    &mut [parameter(&mut s), parameter(&mut v), parameter(&mut y), parameter(&mut h), parameter(&mut kh), parameter(&mut d), parameter(&mut p), parameter(&mut c), parameter(&mut format)])?;
                }
            }
        }
        }
        Ok(())
    }
    pub fn copy_row(&self, source: &DeviceVector<'_>, destination: &mut DeviceVector<'_>, row: usize) -> Result<(), String> {
        let offset = row.checked_mul(destination.len).ok_or("row offset overflow")?;
        self.copy_strided(source, destination, offset, 0, destination.len, destination.len, destination.len, 1)
    }
    /// Queue rectangular subranges on the compute stream. Offsets, strides and
    /// width count FP32 elements; packed QKV/gate-up rows need no host splitting.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_strided(&self, source: &DeviceVector<'_>, destination: &mut DeviceVector<'_>,
        source_offset: usize, destination_offset: usize, source_stride: usize, destination_stride: usize,
        width: usize, rows: usize) -> Result<(), String> {
        self.check_vector(source, source.len)?; self.check_vector(destination, destination.len)?;
        if rows == 0 || width == 0 { return Ok(()); }
        let fits = |offset: usize, stride: usize, len: usize| {
            stride >= width && stride <= u32::MAX as usize && rows <= u32::MAX as usize &&
                (rows - 1).checked_mul(stride).and_then(|n| n.checked_add(offset))
                    .and_then(|n| n.checked_add(width)).is_some_and(|end| end <= len)
        };
        if !fits(source_offset, source_stride, source.len) || !fits(destination_offset, destination_stride, destination.len) {
            return Err("strided copy outside vector".into());
        }
        self.require_batch_kernels()?;
        let (mut src, mut dst) = (source.buffer.address + (source_offset * 4) as u64, destination.buffer.address + (destination_offset * 4) as u64);
        let (mut ss, mut ds, mut w, mut r) = (source_stride as u32, destination_stride as u32, width as u32, rows as u32);
        unsafe { self.enqueue(self.batch.borrow().as_ref().unwrap().copy_strided, width * rows,
            &mut [parameter(&mut src), parameter(&mut dst), parameter(&mut ss), parameter(&mut ds), parameter(&mut w), parameter(&mut r)]) }
    }
    /// Compact a head shard into its existing cache allocation. Each launch has
    /// disjoint source/destination ranges; ordered geometric spans need no scratch.
    pub fn compact_heads(&self, vector: &mut DeviceVector<'_>, offset: usize, stride: usize,
        width: usize, rows: usize) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        if rows == 0 || width == 0 { return Ok(()); }
        if width > stride / 2 || stride > u32::MAX as usize || rows > u32::MAX as usize
            || offset != 0 && offset < width
            || offset.checked_add(width).is_none_or(|end| end > stride)
            || (rows - 1).checked_mul(stride).and_then(|n| n.checked_add(offset))
                .and_then(|n| n.checked_add(width)).is_none_or(|end| end > vector.len)
            || !vector.writable_from(0) {
            return Err("head compaction requires an exclusive nonoverlapping shard".into());
        }
        self.require_batch_kernels()?;
        let copy = |first: usize, end: usize| -> Result<(), String> {
            let (mut src, mut dst) = (vector.buffer.address + ((first * stride + offset) * 4) as u64,
                vector.buffer.address + (first * width * 4) as u64);
            let (mut ss, mut ds, mut w, mut r) = (stride as u32, width as u32, width as u32, (end - first) as u32);
            unsafe { self.enqueue(self.batch.borrow().as_ref().unwrap().copy_strided, width * (end - first),
                &mut [parameter(&mut src), parameter(&mut dst), parameter(&mut ss), parameter(&mut ds), parameter(&mut w), parameter(&mut r)]) }
        };
        if offset != 0 { copy(0, 1)?; }
        let mut first = 1;
        while first < rows {
            let end = rows.min(first * stride / width);
            copy(first, end)?;
            first = end;
        }
        Ok(())
    }
}

impl Backend<'_> {

    pub(super) fn prefill_attention_grid(tokens: usize, heads: usize, kv_heads: usize, paired: bool) -> (u32, u32, u32, u32) {
        let group = heads / kv_heads;
        if group == 8 {
            (tokens.div_ceil(16) as u32, (kv_heads * 2) as u32, 128, 32768)
        } else if paired && group == 4 {
            (tokens.div_ceil(16) as u32, kv_heads as u32, 128, 32768)
        } else if group <= 8 && 8 % group == 0 {
            (tokens.div_ceil(128 / group) as u32, kv_heads as u32, 256, 65536)
        } else {
            (tokens.div_ceil(64) as u32, heads as u32, 128, 65536)
        }
    }

    /// Apply RMS normalization directly into MMA fragments, preserving the reduction tree.
    pub fn rms_pack_device(&self, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, packed: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, epsilon: f32, half: bool) -> Result<(), String> {
        self.rms_partials_device(input, partials, weight.len)?;
        self.rms_pack_apply_device(input, weight, packed, partials, epsilon, half)
    }
    /// Pack normalized fragments using the residual kernel's existing RMS partials.
    pub fn rms_pack_apply_device(&self, input: &DeviceVector<'_>, weight: &DeviceVector<'_>, packed: &mut DeviceVector<'_>, partials: &DeviceVector<'_>, epsilon: f32, _half: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        let width = weight.len;
        if width == 0 || !width.is_multiple_of(128) || input.len == 0 || !input.len.is_multiple_of(width) { return Err("packed RMS shape".into()); }
        let rows = input.len / width;
        self.check_vector(input, input.len)?; self.check_vector(weight, width)?;
        self.check_vector(partials, rows * width.div_ceil(128))?; self.check_vector(packed, packed.len)?;
        if packed.len < rows.div_ceil(16) * 16 * width / 2 { return Err("packed RMS output capacity".into()); }
        let (mut x, mut w, mut p, mut xp) = (input.buffer.address, weight.buffer.address, partials.buffer.address, packed.buffer.address);
        let (mut t, mut k, mut eps) = (rows as u32, width as u32, epsilon);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe {
            self.enqueue_dims(b.rms_pack, k / 128, t.div_ceil(16), 1, 256,
                &mut [parameter(&mut x), parameter(&mut w), parameter(&mut p), parameter(&mut xp), parameter(&mut t), parameter(&mut k), parameter(&mut eps)])
        }
    }
    /// Pack native BF16 activations into the reusable projection operand.
    pub fn pack_prefill_bf16_device(&self, input: &DeviceVector<'_>, packed: &mut DeviceVector<'_>, width: usize, tokens: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        if width == 0 || !width.is_multiple_of(16) || tokens == 0 { return Err("BF16 input packing shape".into()); }
        self.check_vector(input, tokens * width)?;
        self.check_vector(packed, packed.len)?;
        if packed.len < tokens.div_ceil(16) * 16 * width / 2 { return Err("BF16 input packing capacity".into()); }
        let (mut x, mut xp, mut t, mut k) = (input.buffer.address, packed.buffer.address, tokens as u32, width as u32);
        let fragments = tokens.div_ceil(16) * (width / 16) * 32;
        unsafe { self.enqueue_dims(self.batch.borrow().as_ref().unwrap().pack_one, fragments.div_ceil(256) as u32, 1, 1, 256,
            &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut k)]) }
    }
    /// Full BF16 tiles add the projection to state and retain only RMS partials.
    pub fn matmul_packed_residual_bf16_device(&self, matrix: &Matrix<'_>, packed: &DeviceVector<'_>, state: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        if !matrix.packed || !ptr::eq(matrix.buffer._context, self.context) || matrix.row_offset != 0
            || !matrix.cols.is_multiple_of(64) || !matrix.rows.is_multiple_of(128) || tokens == 0 || !tokens.is_multiple_of(256) {
            return Err("fused BF16 residual GEMM shape or context".into());
        }
        self.check_vector(packed, packed.len)?;
        self.check_vector(state, tokens * matrix.rows)?;
        self.check_vector(partials, partials.len)?;
        if packed.len < tokens * matrix.cols / 2 || partials.len < tokens * matrix.rows / 128 {
            return Err("fused BF16 residual GEMM capacity".into());
        }
        let (mut w, mut x, mut y, mut p) = (matrix.buffer.address, packed.buffer.address, state.buffer.address, partials.buffer.address);
        let (mut n, mut k, mut t, mut stride) = (matrix.rows as u32, matrix.cols as u32, tokens as u32, matrix.stride as u32);
        unsafe { self.enqueue_dims_shared(self.batch.borrow().as_ref().unwrap().gemm_residual_bf16, t / 256, n / 128, 1, 512, 98304,
            &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut p), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]) }
    }
    /// Consume the fused SwiGLU epilogue's BF16 fragments.
    pub fn matmul_packed_device(&self, matrix: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.matmul_packed_precision_device(matrix, packed, output, tokens, false)
    }
    /// Consume BF16 fragments and round the projection output at its store boundary.
    pub fn matmul_packed_bf16_device(&self, matrix: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.matmul_packed_precision_device(matrix, packed, output, tokens, true)?;
        self.round_prefill_tail(output, matrix.rows, tokens, true)
    }
    fn matmul_packed_precision_device(&self, matrix: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize, bf16: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        if !matrix.packed || !ptr::eq(matrix.buffer._context, self.context) || !matrix.cols.is_multiple_of(64) || !matrix.rows.is_multiple_of(128) || matrix.row_offset != 0 || matrix.column_offset != 0 || tokens == 0 {
            return Err("packed prefill GEMM shape or context".into());
        }
        self.check_vector(packed, packed.len)?; self.check_vector(output, tokens * matrix.rows)?;
        let required = tokens.div_ceil(16) * 16 * matrix.cols / 2;
        if packed.len < required { return Err("packed prefill GEMM input capacity".into()); }
        let (mut w, mut x, mut y) = (matrix.buffer.address, packed.buffer.address, output.buffer.address);
        let (mut n, mut k, mut t) = (matrix.rows as u32, matrix.cols as u32, tokens as u32);
        let mut stride = matrix.stride as u32;
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        if bf16 && tokens <= 16 && matrix.cols.is_multiple_of(64) {
            return unsafe { self.enqueue_dims(b.gemm_decode_native,t.div_ceil(16),n/64,1,128,
                &mut [parameter(&mut w),parameter(&mut x),parameter(&mut y),parameter(&mut n),parameter(&mut k),parameter(&mut t),parameter(&mut stride)]) };
        }
        if !bf16 && self.decode_rows(tokens) && matrix.cols.is_multiple_of(256) {
            let thin = matrix.rows <= 8192 && matrix.cols >= 4096;
            let mut params = PackedDecodeProjection { weights: w, input: x, output: y,
                rows: n, width: k, tokens: t, stride };
            return unsafe { self.enqueue_dims(if thin {b.gemm_decode_n32} else {b.gemm_decode}, tokens.div_ceil(16) as u32, n / if thin {32} else {64}, 1, if thin {128} else {256},
                &mut [parameter(&mut params)]) };
        }
        if bf16 {
        if tokens >= 256 {
            let mut full = t / 256 * 256;
            unsafe { self.enqueue_dims_shared(if bf16 { b.gemm_wide_bf16 } else { b.gemm_wide }, full / 256, n / 128, 1, 512, 98304,
                &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut full), parameter(&mut stride)])?; }
            t -= full;
            if t == 0 { return Ok(()); }
            x += full as u64 * k as u64 * 2;
            y += full as u64 * n as u64 * 4;
        }
        return unsafe { self.enqueue_dims(b.gemm_one, t.div_ceil(128), n / 128, 1, 256,
            &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]) };
        }
        if matrix.cols >= 128 && tokens.is_multiple_of(256) {
            return unsafe { self.enqueue_dims_shared(if matrix.stride.is_multiple_of(64) {b.gemm_wide} else {b.gemm_wide_packed}, tokens as u32 / 256, n / 128, 1, 512, 98304,
                &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]) };
        }
        if tokens >= 256 && matrix.cols >= 8192 {
            return unsafe { self.enqueue_dims_shared(if matrix.stride.is_multiple_of(64) {b.tiled_large} else {b.gemm_large}, tokens.div_ceil(256) as u32, n / 128, 1, 256, 73728,
                &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]) };
        }
        unsafe { self.enqueue_dims(if matrix.stride.is_multiple_of(64) {b.tiled_gemm} else {b.gemm_one}, tokens.div_ceil(128) as u32, n / 128, 1, 256,
            &mut [parameter(&mut w), parameter(&mut x), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)]) }
    }
    /// Project directly into the residual state and retain the next RMS partials.
    pub fn matmul_residual_rms_device(&self, matrix: &Matrix<'_>, input: &DeviceVector<'_>,
        state: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>,
        tokens: usize, packed: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        if !matrix.packed || !ptr::eq(matrix.buffer._context, self.context) || !matrix.cols.is_multiple_of(64)
            || !matrix.rows.is_multiple_of(128) || matrix.row_offset != 0 || matrix.column_offset != 0 || tokens == 0 {
            return Err("residual prefill GEMM shape or context".into());
        }
        let fragments = tokens.div_ceil(16) * (matrix.cols / 16) * 32;
        self.check_vector(input, input.len)?;
        self.check_vector(state, tokens * matrix.rows)?;
        self.check_vector(partials, partials.len)?;
        self.check_vector(input_pack, input_pack.len)?;
        if input.len < if packed { fragments * 4 } else { tokens * matrix.cols }
            || partials.len < tokens * (matrix.rows / 128) || (!packed && input_pack.len < fragments * 4) {
            return Err("residual prefill GEMM scratch capacity".into());
        }
        let (mut w, mut x, mut y, mut p) = (matrix.buffer.address, input.buffer.address, state.buffer.address, partials.buffer.address);
        let mut xp = if packed { x } else { input_pack.buffer.address };
        let (mut n, mut k, mut t, mut stride) = (matrix.rows as u32, matrix.cols as u32, tokens as u32, matrix.stride as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe {
            if !packed {
                self.enqueue_dims(b.pack_one, fragments.div_ceil(256) as u32, 1, 1, 256,
                    &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut k)])?;
            }
            // 8B and 32B O/down shapes on full 256-row chunks use the 512-thread tiles.
            if tokens.is_multiple_of(256) && matrix.stride.is_multiple_of(64)
                && ((n == 4096 && (k == 4096 || k == 12288 || k == 14336)) || (n == 5120 && (k == 8192 || k == 25600))) {
                let o = k == 4096 || k == 8192;
                let mut args = crate::kernel_params::PrefillResidual { weights: w, input: xp, output: y, partials: p,
                    width: n, inner: k, rows: t, stride };
                return self.enqueue_dims_shared(if o {b.gemm_residual_o} else {b.gemm_residual_wide},
                    tokens as u32 / if o {128} else {256}, n / if o {256} else {128}, 1, 512, 98304,
                    &mut [parameter(&mut args)]);
            }
            let large = tokens >= 256 && matrix.cols >= 8192;
            let function = match (matrix.stride.is_multiple_of(64), large) {
                (true, true) => b.tiled_residual_large, (true, false) => b.tiled_residual,
                (false, true) => b.gemm_residual_large, (false, false) => b.gemm_residual,
            };
            self.enqueue_dims_shared(function, tokens.div_ceil(if large {256} else {128}) as u32, n / 128, 1, 256,
                if large {73728} else {0}, &mut [parameter(&mut w), parameter(&mut xp), parameter(&mut y), parameter(&mut p),
                    parameter(&mut n), parameter(&mut k), parameter(&mut t), parameter(&mut stride)])
        }
    }

    /// Delta QKV/Z and split-reduction A/B share one packed-input launch. Its
    /// epilogue emits beta/decay directly, retaining the official BF16 boundaries.
    pub fn prefill_gdn_packed_bf16_device(&self, matrices: [&Matrix<'_>; 4], packed: &DeviceVector<'_>,
        outputs: [&DeviceVector<'_>; 5], alog: u64, dt: u64, tokens: usize, parameter_precision: u32) -> Result<(), String> {
        self.require_batch_kernels()?;
        let [qkv, z, a, gate] = matrices;
        let (width, heads) = (qkv.cols, a.rows);
        if tokens == 0 || width % 64 != 0 || qkv.rows % 128 != 0 || z.rows % 128 != 0
            || heads == 0 || heads > 48 || heads % 8 != 0 || heads != gate.rows {
            return Err("fused delta input shape".into());
        }
        for matrix in matrices {
            if !matrix.packed || matrix.row_offset != 0 || matrix.cols != width || matrix.stride != width
                || !ptr::eq(matrix.buffer._context, self.context) {
                return Err("fused delta inputs require local packed weights".into());
            }
        }
        self.check_vector(packed, packed.len)?;
        if packed.len < tokens.div_ceil(16) * 16 * width / 2 { return Err("fused delta input capacity".into()); }
        for (output, rows) in outputs.into_iter().zip([qkv.rows, z.rows, heads, heads, heads]) {
            self.check_vector(output, tokens * rows)?;
        }
        let mut p = GdnProjection { weights: matrices.map(|m| m.buffer.address), input: packed.buffer.address,
            outputs: outputs.map(|v| v.buffer.address), alog, dt, qkv_rows: qkv.rows as u32,
            z_rows: z.rows as u32, heads: heads as u32, width: width as u32, tokens: tokens as u32, parameter_precision };
        unsafe { self.enqueue_dims(self.batch.borrow().as_ref().unwrap().gdn_inputs, p.tokens.div_ceil(128),
            (p.qkv_rows + p.z_rows) / 128 + 1, 1, 256, &mut [parameter(&mut p)]) }
    }
    /// Pack once for a fused Q/K/V projection from BF16 model activations.
    pub fn prefill_qkv_bf16_device(&self, q: &Matrix<'_>, k: &Matrix<'_>, v: &Matrix<'_>, input: &DeviceVector<'_>,
        packed: &mut DeviceVector<'_>, out_q: &mut DeviceVector<'_>, out_k: &mut DeviceVector<'_>, out_v: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        self.check_vector(input, tokens * q.cols)?; self.check_vector(packed, packed.len)?;
        if tokens == 0 || !q.cols.is_multiple_of(64) { return Err("BF16 fused QKV input shape".into()); }
        let fragments = tokens.div_ceil(16) * (q.cols / 16) * 32;
        if packed.len < fragments * 4 { return Err("BF16 fused QKV packing scratch".into()); }
        let (mut x, mut xp) = (input.buffer.address, packed.buffer.address);
        let (mut t, mut width) = (tokens as u32, q.cols as u32);
        unsafe { self.enqueue_dims(self.batch.borrow().as_ref().unwrap().pack_one, fragments.div_ceil(256) as u32, 1, 1, 256,
            &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut width)])?; }
        self.prefill_qkv_packed_bf16_device(q, k, v, packed, out_q, out_k, out_v, tokens)
    }
    /// Consume normalized BF16 fragments without packing the same activation again.
    pub fn prefill_qkv_packed_bf16_device(&self, q: &Matrix<'_>, k: &Matrix<'_>, v: &Matrix<'_>, packed: &DeviceVector<'_>,
        out_q: &mut DeviceVector<'_>, out_k: &mut DeviceVector<'_>, out_v: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.prefill_qkv_precision_device(q, k, v, packed, out_q, out_k, out_v, tokens, true)?;
        self.round_prefill_tail(out_q, q.rows, tokens, true)?;
        self.round_prefill_tail(out_k, k.rows, tokens, true)?;
        self.round_prefill_tail(out_v, v.rows, tokens, true)
    }
    /// One logical concatenation of Q/K/V, with each output in its consumer's layout.
    pub fn prefill_qkv_device(&self, q: &Matrix<'_>, k: &Matrix<'_>, v: &Matrix<'_>, packed: &DeviceVector<'_>,
        out_q: &mut DeviceVector<'_>, out_k: &mut DeviceVector<'_>, out_v: &mut DeviceVector<'_>, tokens: usize, _half: bool) -> Result<(), String> {
        self.prefill_qkv_precision_device(q, k, v, packed, out_q, out_k, out_v, tokens, false)
    }
    fn prefill_qkv_precision_device(&self, q: &Matrix<'_>, k: &Matrix<'_>, v: &Matrix<'_>, packed: &DeviceVector<'_>,
        out_q: &mut DeviceVector<'_>, out_k: &mut DeviceVector<'_>, out_v: &mut DeviceVector<'_>, tokens: usize, bf16: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        if tokens == 0 || !q.rows.is_multiple_of(128) || !k.rows.is_multiple_of(128) || k.rows != v.rows || !q.cols.is_multiple_of(64) || q.cols != k.cols || q.cols != v.cols {
            return Err("fused QKV shape".into());
        }
        for matrix in [q, k, v] {
            if !matrix.packed || !ptr::eq(matrix.buffer._context, self.context) { return Err("fused QKV requires local packed weights".into()); }
        }
        self.check_vector(out_q, tokens * q.rows)?; self.check_vector(out_k, tokens * k.rows)?; self.check_vector(out_v, tokens * v.rows)?;
        let fragments = tokens.div_ceil(16) * (q.cols / 16) * 32;
        self.check_vector(packed, packed.len)?;
        if packed.len < fragments * 4 { return Err("fused QKV packing scratch".into()); }
        let (mut wq, mut wk, mut wv) = (q.buffer.address, k.buffer.address, v.buffer.address);
        let mut xp = packed.buffer.address;
        let (mut yq, mut yk, mut yv) = (out_q.buffer.address, out_k.buffer.address, out_v.buffer.address);
        let (mut nq, mut nkv, mut width, mut t) = (q.rows as u32, k.rows as u32, q.cols as u32, tokens as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe {
            if !bf16 && self.decode_rows(tokens) && q.cols.is_multiple_of(256) {
                return self.enqueue_dims(b.qkv_decode, tokens.div_ceil(16) as u32, (nq+2*nkv)/64, 1, 256,
                    &mut [parameter(&mut wq), parameter(&mut wk), parameter(&mut wv), parameter(&mut xp),
                        parameter(&mut yq), parameter(&mut yk), parameter(&mut yv), parameter(&mut nq), parameter(&mut nkv), parameter(&mut width), parameter(&mut t)]);
            }
            if bf16 {
            if bf16 && tokens >= 256 {
                return self.enqueue_dims(b.qkv_bf16, t.div_ceil(128), (nq + 2 * nkv) / 128, 1, 256,
                    &mut [parameter(&mut wq), parameter(&mut wk), parameter(&mut wv), parameter(&mut xp),
                        parameter(&mut yq), parameter(&mut yk), parameter(&mut yv), parameter(&mut nq), parameter(&mut nkv), parameter(&mut width), parameter(&mut t)]);
            }
            return self.enqueue_dims(b.qkv, t.div_ceil(128), (nq + 2 * nkv) / 128, 1, 256,
                    &mut [parameter(&mut wq), parameter(&mut wk), parameter(&mut wv), parameter(&mut xp), parameter(&mut yq), parameter(&mut yk), parameter(&mut yv), parameter(&mut nq), parameter(&mut nkv), parameter(&mut width), parameter(&mut t)]);
            }
            if tokens.is_multiple_of(256) && [q, k, v].iter().all(|matrix|
                matrix.stride == matrix.cols && matrix.row_offset == 0 && matrix.column_offset == 0) {
                return self.enqueue_dims_shared(if q.stride.is_multiple_of(64) {b.qkv_wide} else {b.qkv_wide_packed}, tokens as u32 / 256, (nq + 2 * nkv) / 128, 1, 512, 98304,
                    &mut [parameter(&mut wq), parameter(&mut wk), parameter(&mut wv), parameter(&mut xp),
                        parameter(&mut yq), parameter(&mut yk), parameter(&mut yv), parameter(&mut nq), parameter(&mut nkv), parameter(&mut width), parameter(&mut t)]);
            }
            self.enqueue_dims(if q.stride.is_multiple_of(64) {b.tiled_qkv} else {b.qkv}, tokens.div_ceil(128) as u32, (nq + 2 * nkv) / 128, 1, 256,
                &mut [parameter(&mut wq), parameter(&mut wk), parameter(&mut wv), parameter(&mut xp),
                    parameter(&mut yq), parameter(&mut yk), parameter(&mut yv), parameter(&mut nq), parameter(&mut nkv), parameter(&mut width), parameter(&mut t)])
        }
    }
    /// SwiGLU epilogue writes the next down GEMM's BF16 fragments directly.
    pub fn prefill_gate_up_packed_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.prefill_gate_up_packed_precision_device(weights, up, packed, output, tokens, false)
    }
    pub fn prefill_gate_up_packed_bf16_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.prefill_gate_up_packed_precision_device(weights, up, packed, output, tokens, true)
    }
    fn prefill_gate_up_packed_precision_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, packed: &DeviceVector<'_>, output: &mut DeviceVector<'_>, tokens: usize, bf16: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        let width = weights.cols; let rows = weights.rows;
        if !weights.packed || !up.packed || rows != up.rows || width != up.cols || tokens == 0 || !rows.is_multiple_of(64) || !width.is_multiple_of(64) { return Err("fused gate/up shape".into()); }
        if !ptr::eq(weights.buffer._context, self.context) || !ptr::eq(up.buffer._context, self.context) { return Err("fused weights context mismatch".into()); }
        self.check_vector(output, output.len)?;
        if output.len < tokens.div_ceil(16) * 16 * rows / 2 { return Err("fused gate/up output capacity".into()); }
        let fragments = tokens.div_ceil(16) * (width / 16) * 32;
        self.check_vector(packed, packed.len)?;
        if packed.len < fragments * 4 { return Err("fused gate/up packing scratch".into()); }
        let (mut w, mut y, mut xp) = (weights.buffer.address, output.buffer.address, packed.buffer.address);
        let mut u = up.buffer.address;
        let (mut n, mut k, mut t) = (rows as u32, width as u32, tokens as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe {
            if bf16 {
            if tokens >= 256 {
                let mut full = t / 256 * 256;
                self.enqueue_dims_shared(if bf16 { b.gate_up_wide_bf16 } else { b.gate_up_wide }, full / 256, n / 64, 1, 512, 98304,
                    &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut full)])?;
                t -= full;
                if t == 0 { return Ok(()); }
                xp += full as u64 * k as u64 * 2;
                y += full as u64 * n as u64 * 2;
            }
            return self.enqueue_dims(if bf16 { b.gate_up_bf16 } else { b.gate_up_packed }, t.div_ceil(128), n / 64, 1, 256,
                &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t)]);
            }
            if self.decode_rows(tokens) && width.is_multiple_of(256) {
                let mut params = DecodeGateUp { gate: w, up: u, input: xp, output: y,
                    rows: n, width: k, tokens: t };
                return self.enqueue_dims(b.gate_up_decode,tokens.div_ceil(16) as u32,rows as u32/32,1,128,
                    &mut [parameter(&mut params)]);
            }
            if weights.stride.is_multiple_of(64) && width >= 128 && tokens.is_multiple_of(256) {
                return self.enqueue_dims_shared(b.gate_up_wide, tokens as u32 / 256, rows as u32 / 64, 1, 512, 98304,
                    &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t)]);
            }
            self.enqueue_dims(if weights.stride.is_multiple_of(64) {b.tiled_gate_up_packed} else {b.gate_up_packed}, tokens.div_ceil(128) as u32, rows as u32 / 64, 1, 256,
                &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t)])
        }
    }
    /// Gate/up and down operand packing from unpacked FP32 storage.
    pub fn prefill_gate_up_fast_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, packed: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.prefill_gate_up_from_f32_device(weights, up, input, output, packed, tokens, false)
    }
    /// BF16 gate/up projections and SiLU; output is packed for the down GEMM.
    pub fn prefill_gate_up_bf16_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, packed: &mut DeviceVector<'_>, tokens: usize) -> Result<(), String> {
        self.prefill_gate_up_from_f32_device(weights, up, input, output, packed, tokens, true)
    }
    fn prefill_gate_up_from_f32_device(&self, weights: &Matrix<'_>, up: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, packed: &mut DeviceVector<'_>, tokens: usize, bf16: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        let width = weights.cols; let rows = weights.rows;
        if !weights.packed || !up.packed || rows != up.rows || width != up.cols || tokens == 0 || !rows.is_multiple_of(64) || !width.is_multiple_of(64) { return Err("fused gate/up shape".into()); }
        if !ptr::eq(weights.buffer._context, self.context) || !ptr::eq(up.buffer._context, self.context) { return Err("fused weights context mismatch".into()); }
        self.check_vector(input, tokens * width)?; self.check_vector(output, output.len)?; self.check_vector(packed, packed.len)?;
        let fragments = tokens.div_ceil(16) * (width / 16) * 32;
        if output.len < tokens.div_ceil(16) * 16 * rows / 2 || packed.len < fragments * 4 { return Err("fast gate/up packed capacity".into()); }
        let (mut w, mut u, mut x, mut y, mut xp) = (weights.buffer.address, up.buffer.address, input.buffer.address, output.buffer.address, packed.buffer.address);
        let (mut n, mut k, mut t) = (rows as u32, width as u32, tokens as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe {
            self.enqueue_dims(b.pack_one, fragments.div_ceil(256) as u32, 1, 1, 256,
                &mut [parameter(&mut x), parameter(&mut xp), parameter(&mut t), parameter(&mut k)])?;
            if bf16 {
            if tokens >= 256 {
                let mut full = t / 256 * 256;
                self.enqueue_dims_shared(if bf16 { b.gate_up_wide_bf16 } else { b.gate_up_wide }, full / 256, n / 64, 1, 512, 98304,
                    &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut full)])?;
                t -= full;
                if t == 0 { return Ok(()); }
                xp += full as u64 * k as u64 * 2;
                y += full as u64 * n as u64 * 2;
            }
            return self.enqueue_dims(b.gate_up_bf16, t.div_ceil(128), n / 64, 1, 256,
                &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t)]);
            }
            self.enqueue_dims(if weights.stride.is_multiple_of(64) {b.tiled_gate_up_packed} else {b.gate_up_packed}, tokens.div_ceil(128) as u32, rows as u32 / 64, 1, 256,
                &mut [parameter(&mut w), parameter(&mut u), parameter(&mut xp), parameter(&mut y), parameter(&mut n), parameter(&mut k), parameter(&mut t)])
        }
    }
    /// GQA4 attention emits the O projection's BF16 input fragments directly.
    pub fn prefill_attention_fragments_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, heads: usize, kv_heads: usize, position: usize, tokens: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        let capacity = KvFormat::Bf16.element_capacity(keys.len) / (kv_heads.max(1) * 128);
        if tokens == 0 || !tokens.is_multiple_of(16) || kv_heads == 0 || heads != kv_heads * 4 || position + tokens > capacity {
            return Err("attention fragment dimensions".into());
        }
        self.check_vector(query, tokens * heads * 128)?; self.check_vector(output, query.len)?;
        self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?;
        let mut args = crate::kernel_params::PrefillAttention { query: query.buffer.address,
            keys: keys.buffer.address, values: values.buffer.address, output: output.buffer.address,
            heads: heads as u32, kv_heads: kv_heads as u32, rows: tokens as u32,
            position: position as u32, capacity: capacity as u32, exp2: 0 };
        unsafe { self.enqueue_dims_shared(self.batch.borrow().as_ref().unwrap().prefill_fragments,
            tokens as u32 / 16, kv_heads as u32, 1, 128, 32768, &mut [parameter(&mut args)]) }


    }
    /// Causal attention against resident BF16 KV, with 128- or 256-wide heads.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention_bf16_dim_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, heads: usize, kv_heads: usize, dim: usize, position: usize, tokens: usize) -> Result<(), String> {
        self.require_batch_kernels()?;
        if ![128, 256].contains(&dim) || heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads)
            || heads > u32::MAX as usize || kv_heads > u32::MAX as usize || tokens == 0 || tokens > u32::MAX as usize {
            return Err("BF16 prefill attention dimensions".into());
        }
        let capacity = keys.len * 2 / (kv_heads * dim);
        if capacity > u32::MAX as usize || position.checked_add(tokens).is_none_or(|end| end > capacity) {
            return Err("BF16 prefill attention exceeds KV capacity".into());
        }
        let elements = tokens.checked_mul(heads).and_then(|n| n.checked_mul(dim)).ok_or("prefill query size overflow")?;
        self.check_vector(query, elements)?; self.check_vector(output, elements)?;
        self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?;
        if dim == 256 {
            let kv_elements = (position + tokens).checked_mul(kv_heads).and_then(|n| n.checked_mul(dim))
                .ok_or("hd256 attention KV size overflow")?;
            let query_elements = u32::try_from(elements).map_err(|_| "hd256 query exceeds validation ABI")?;
            let kv_elements = u32::try_from(kv_elements).map_err(|_| "hd256 KV exceeds validation ABI")?;
            // This path is prompt/verification only; callers keep it outside capture.
            // Four bytes of logical status (one aligned allocation), no input mirror.
            let mut error = self.vector(1)?;
            self.write_vector(&mut error, &[0.0])?;
            let mut p = crate::kernel_params::AttentionValidate { query: query.device_address(),
                keys: keys.device_address(), values: values.device_address(), error: error.device_address(),
                query_elements, kv_elements };
            unsafe { self.enqueue_dims(self.module.function("attention_validate_256")?,
                query_elements.max(kv_elements).div_ceil(256), 1, 1, 256, &mut [parameter(&mut p)])?; }
            if self.read_vector(&error)?[0].to_bits() != 0 {
                return Err("hd256 attention rejected non-finite or overflowing Q/K/V before launch".into());
            }
        }
        let mut args = crate::kernel_params::PrefillAttention { query: query.buffer.address,
            keys: keys.buffer.address, values: values.buffer.address, output: output.buffer.address,
            heads: heads as u32, kv_heads: kv_heads as u32, rows: tokens as u32,
            position: position as u32, capacity: capacity as u32, exp2: 0 };
        let group = heads / kv_heads;
        let (gx, gy, threads) = if dim == 256 && group == 6 {
            (tokens.div_ceil(16) as u32, kv_heads as u32, 192)
        } else if group <= 8 && 8 % group == 0 {
            (tokens.div_ceil(128 / group) as u32, kv_heads as u32, 256)
        } else { (tokens.div_ceil(64) as u32, heads as u32, 128) };
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        unsafe { self.enqueue_dims_shared(if dim == 256 { b.prefill_bf16_256 } else { b.prefill_bf16 },
            gx, gy, 1, threads, 65536, &mut [parameter(&mut args)]) }
    }

    /// Exact model-precision attention by default; adaptive long prompts require opt-in.
    pub fn prefill_attention_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, packed_keys: &mut DeviceVector<'_>, packed_values: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, position: usize, tokens: usize, format: impl Into<KvFormat>, prompt_end: usize, expert: bool) -> Result<(), String> {
        self.prefill_attention_output(query, keys, values, output, packed_keys, packed_values,
            heads, kv_heads, position, tokens, format.into(), prompt_end, expert, false)
    }
    /// Expert GQA8 prefill consumes this BF16 result directly in its O GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention_output_packed_device(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, packed_keys: &mut DeviceVector<'_>, packed_values: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, position: usize, tokens: usize) -> Result<(), String> {
        if tokens < 64 || heads != kv_heads * 8 { return Err("packed attention requires GQA8 prefill".into()); }
        self.prefill_attention_output(query, keys, values, output, packed_keys, packed_values,
            heads, kv_heads, position, tokens, KvFormat::Bf16, position + tokens, true, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn prefill_attention_output(&self, query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>, packed_keys: &mut DeviceVector<'_>, packed_values: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, position: usize, tokens: usize, format: KvFormat, prompt_end: usize, expert: bool, packed_output: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        let capacity = format.element_capacity(keys.len) / (kv_heads.max(1) * 128);
        if tokens == 0 || heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || position + tokens > capacity || query.len > u32::MAX as usize { return Err("prefill attention dimensions".into()); }
        self.check_vector(query, tokens * heads * 128)?; self.check_vector(output, query.len)?;
        self.check_vector(keys, keys.len)?; self.check_vector(values, keys.len)?;
        let (mut q, mut k, mut v, mut o) = (query.buffer.address, keys.buffer.address, values.buffer.address, output.buffer.address);
        let (mut h, mut kv, mut t, mut base, mut cap) = (heads as u32, kv_heads as u32, tokens as u32, position as u32, capacity as u32);
        let kernels = self.batch.borrow(); let b = kernels.as_ref().unwrap();
        if tokens >= if expert { 64 } else { 512 } && format == KvFormat::Bf16 && heads == kv_heads * 8 {
            let elements = (position + tokens).div_ceil(64) * 64 * kv_heads * 128;
            self.check_vector(packed_keys, packed_keys.len)?; self.check_vector(packed_values, packed_values.len)?;
            // Short shared-prompt batches reuse projection scratch, which may
            // not fit a swizzled prefix. Keep the direct-KV fallback.
            if elements <= u32::MAX as usize && packed_keys.len >= elements / 2 && packed_values.len >= elements / 2 {
                let (mut kp, mut vp, mut end) = (packed_keys.buffer.address, packed_values.buffer.address, (position + tokens) as u32);
                let (gx, gy, threads, shared) = Self::prefill_attention_grid(tokens, heads, kv_heads, false);
                unsafe {
                    self.enqueue_dims(b.prefill_swizzle, (elements / 2).div_ceil(256) as u32, 1, 1, 256,
                        &mut [parameter(&mut k), parameter(&mut v), parameter(&mut kp), parameter(&mut vp), parameter(&mut kv), parameter(&mut end)])?;
                    return self.enqueue_dims_shared(if packed_output { b.prefill_swizzled_packed } else { b.prefill_gqa8 }, gx, gy, 1, threads, shared,
                        &mut [parameter(&mut q), parameter(&mut kp), parameter(&mut vp), parameter(&mut o), parameter(&mut h), parameter(&mut kv), parameter(&mut t), parameter(&mut base), parameter(&mut cap)]);
                }
            }
        }
        if packed_output { return Err("swizzled prefill KV staging too small".into()); }
        if tokens >= 64 && format == KvFormat::Fp32 && heads == kv_heads * 4 && self.fast_prefill_attention(prompt_end) {
            let head_elements = (position + tokens).div_ceil(64) * 64 * 128;
            let packed_heads = kv_heads.min(packed_keys.len * 2 / head_elements).min(packed_values.len * 2 / head_elements);
            if packed_heads == 0 { return Err("fast prefill KV staging too small".into()); }
            self.check_vector(packed_keys, packed_keys.len)?; self.check_vector(packed_values, packed_values.len)?;
            let mut start = 0u32;
            let (mut kp, mut vp, mut end) = (packed_keys.buffer.address, packed_values.buffer.address, (position + tokens) as u32);
            for first in (0..kv_heads).step_by(packed_heads) {
                let mut first = first as u32;
                let mut count = (kv_heads - first as usize).min(packed_heads) as u32;
                let pairs = head_elements * count as usize / 2;
                unsafe {
                    self.enqueue_dims(b.prefill_fast_pack, pairs.div_ceil(256) as u32, 1, 1, 256,
                        &mut [parameter(&mut k), parameter(&mut v), parameter(&mut kp), parameter(&mut vp), parameter(&mut kv), parameter(&mut start), parameter(&mut end), parameter(&mut count), parameter(&mut first)])?;
                    cap = end;
                    self.enqueue_dims_shared(b.prefill_fast, tokens.div_ceil(32) as u32, count, 1, 256, 65536,
                        &mut [parameter(&mut q), parameter(&mut kp), parameter(&mut vp), parameter(&mut o), parameter(&mut h), parameter(&mut count), parameter(&mut t), parameter(&mut base), parameter(&mut cap), parameter(&mut first)])?;
                }
            }
            return Ok(());
        }
        if format != KvFormat::Bf16 {
            let elements = (position + tokens) * kv_heads * 128;
            if packed_keys.len < elements.div_ceil(2) || packed_values.len < elements.div_ceil(2) { return Err("prefill KV staging too small".into()); }
            self.check_vector(packed_keys, packed_keys.len)?; self.check_vector(packed_values, packed_values.len)?;
            let (mut kp, mut vp, mut n) = (packed_keys.buffer.address, packed_values.buffer.address, elements as u32);
            let mut format = format as u32;
            // Convert each KV operand once, with the same rounding as the raw
            // attention kernel, instead of repeating it for every query tile.
            unsafe { self.enqueue(b.prefill_pack, elements, &mut [parameter(&mut k), parameter(&mut v), parameter(&mut kp), parameter(&mut vp), parameter(&mut n), parameter(&mut format)])?; }
            k = kp; v = vp; cap = (position + tokens) as u32;
        }
        let (gx, gy, threads, shared) = Self::prefill_attention_grid(tokens, heads, kv_heads, true);
        if heads == kv_heads * 4 {
            unsafe { self.enqueue_dims_shared(b.prefill_gqa4, gx, gy, 1, threads, shared,
                &mut [parameter(&mut q), parameter(&mut k), parameter(&mut v), parameter(&mut o), parameter(&mut h), parameter(&mut kv), parameter(&mut t), parameter(&mut base), parameter(&mut cap)]) }
        } else {
            let mut args = crate::kernel_params::PrefillAttention { query: q, keys: k, values: v, output: o,
                heads: h, kv_heads: kv, rows: t, position: base, capacity: cap,
                // Same ex2 softmax as the swizzled GQA8 kernel, so a row's
                // bits do not depend on its chunk length (split/cached = cold).
                exp2: u32::from(heads == kv_heads * 8) };
            unsafe { self.enqueue_dims_shared(b.prefill_bf16, gx, gy, 1, threads, shared, &mut [parameter(&mut args)]) }
        }
    }
    /// Wide flash QK/PV with full-row model-precision softmax between passes.
    pub(super) fn prefill_attention_wide(&self, mut a: crate::kernel_params::BoundedAttention) -> Result<(), String> {
        self.require_batch_kernels()?;
        let batch = self.batch.borrow(); let b = batch.as_ref().unwrap();
        let grid = a.rows.div_ceil(64);
        unsafe {
            self.enqueue_dims_shared(b.prefill_wide_scores, grid, a.heads, 1, 128, 16384, &mut [parameter(&mut a)])?;
            self.enqueue_grid(self.layer_kernels()?.bounded_softmax, a.heads as usize*32, a.rows, 1, &mut [parameter(&mut a)])?;
            self.enqueue_dims_shared(b.prefill_wide_values, grid, a.heads, a.dim/128, 128, 16384, &mut [parameter(&mut a)])?;
        }
        Ok(())
    }

}
