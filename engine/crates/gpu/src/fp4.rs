//! Explicit W4A16 dispatch; the default FP8 path never selects these kernels.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Kernels { project: Handle, gate: Handle, down: Handle, scatter: Handle,
    exact_project: Handle, exact_swiglu: Handle, next_gate: Handle, next_down: Handle }

impl<'a> Backend<'a> {
    pub(super) fn fp4_kernels(&self) -> Result<Kernels, String> {
        if self.fp4.borrow().is_none() {
            *self.fp4.borrow_mut() = Some(Kernels {
                project: self.module.function("fp4_bf16_project")?,
                gate: self.module.function("fp4_grouped_gate")?,
                down: self.module.function("fp4_grouped_down")?,
                scatter: self.module.function("fp4_grouped_scatter")?,
                exact_project: self.module.function("fp4_exact_project")?,
                exact_swiglu: self.module.function("fp4_exact_swiglu")?,
                next_gate: self.module.function("fp4_next_grouped_gate")?, next_down: self.module.function("fp4_next_grouped_down")?,
            });
        }
        self.fp4.borrow().as_ref().copied().ok_or("FP4 kernels not prepared".into())
    }
    /// Project borrowed NVFP4 weights, rounding inputs and dequantized weights
    /// to BF16 and accumulating in the canonical four FP32 K ranges.
    /// # Safety
    /// Matrix addresses remain live through stream completion: N*K/2 packed
    /// bytes aligned to eight, N*K/16 E4M3 scale bytes, one aligned finite F32
    /// global scale. All operands are finite and output aliases neither input.
    pub unsafe fn fp4_bf16_project(&self, matrix: fp4_abi::Matrix, rows: usize,
        cols: usize, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if rows == 0 || rows % 8 != 0 || rows > u32::MAX as usize || cols == 0
            || cols % 16 != 0 || cols > u32::MAX as usize || input.len % cols != 0
            || matrix.weights == 0 || matrix.weights % 8 != 0 || matrix.scales == 0
            || matrix.global_scale == 0 || matrix.global_scale % 4 != 0 {
            return Err("invalid NVFP4 matrix layout".into());
        }
        let tokens = input.len / cols;
        if tokens == 0 || tokens.div_ceil(16) > 65535 {
            return Err("invalid NVFP4 token count".into());
        }
        self.check_vector(input, input.len)?;
        self.check_vector(output, tokens.checked_mul(rows).ok_or("NVFP4 output overflow")?)?;
        if input.is_segmented() || output.is_segmented() {
            return Err("NVFP4 projection requires contiguous vectors".into());
        }
        let packed = self.pack_flash_dense_input(input.device_address(), cols, tokens)?;
        let mut p = fp4_abi::ProjectParams { matrix, input: packed, output: output.device_address(),
            tokens: tokens as u32, rows: rows as u32, cols: cols as u32, reserved: 0 };
        self.profile_phase(0, || self.enqueue_dims(self.fp4_kernels()?.project,
            (rows / 8) as u32, tokens.div_ceil(16) as u32, 1, 128, &mut [parameter(&mut p)]))
    }

    /// Routed W4A16 FFN, with the FP8 routing/workspace and ordered output ABI.
    /// Scratch is reserved with reserve_fp8_grouped(..., false) before capture.
    /// Every token count uses identical BF16 operands and canonical K order.
    /// # Safety
    /// Experts contains 512 fp4_abi::Expert records, each with width 0..=640,
    /// width%16=0, and the three matrices specified by that ABI. Addresses obey
    /// fp4_bf16_project's contract and remain live until the stream completes.
    /// Assignments contains exactly ten distinct valid expert IDs per token,
    /// represented as FP32 (token, expert, weight). Output has eleven H rows per
    /// token, of which this writes ten unweighted rows in ascending expert ID.
    pub unsafe fn fp4_bf16_grouped_ffn(&self, experts: &DeviceVector<'_>,
        input: &DeviceVector<'_>, assignments: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        scratch: &mut Option<Fp8GroupedScratch<'a>>) -> Result<(), String> {
        unsafe { self.fp4_grouped(experts,input,assignments,output,scratch,false) }
    }
    /// Flash-Next FAST_MOE K16 chain/GU-down BF16 boundary with padded storage.
    /// # Safety
    /// As fp4_bf16_grouped_ffn; kept for the compact-layout byte/speed control.
    pub unsafe fn fp4_bf16_next_grouped_ffn(&self,experts: &DeviceVector<'_>,input: &DeviceVector<'_>,
        assignments: &DeviceVector<'_>,output: &mut DeviceVector<'_>,scratch: &mut Option<Fp8GroupedScratch<'a>>) -> Result<(),String> {
        unsafe { self.fp4_grouped(experts,input,assignments,output,scratch,true) }
    }
    unsafe fn fp4_grouped(&self,experts: &DeviceVector<'_>,input: &DeviceVector<'_>,assignments: &DeviceVector<'_>,
        output: &mut DeviceVector<'_>,scratch: &mut Option<Fp8GroupedScratch<'a>>,next: bool) -> Result<(),String> {
        const H: usize = 2560;
        self.check_vector(experts, 512 * std::mem::size_of::<fp4_abi::Expert>() / 4)?;
        if experts.is_segmented() { return Err("NVFP4 descriptors must be contiguous".into()); }
        let kernels = self.fp4_kernels()?;
        let (route, addresses, tasks) = self.prepare_grouped(input, assignments, output, scratch, false)?;
        let (packed, middle) = (addresses[0], addresses[1]);
        let mut p = fp8_abi::GroupedParams { route, weights: experts.device_address(),
            input: input.device_address(), input_scales: 0, output: packed,
            cols: H as u32, activation_quantization: 0, output_shard: 0 };
        self.enqueue_dims(self.fp8_kernels()?.grouped_gather,
            (route.row_capacity as usize * H / 8).div_ceil(256) as u32,
            1, 1, 256, &mut [parameter(&mut p)])?;
        let planes = route.tokens.div_ceil(16).min(8);
        self.profile_phase(0, || {
            (p.input, p.output) = (packed, middle);
            self.enqueue_dims(if next {kernels.next_gate}else{kernels.gate}, if next {20}else{80}, tasks as u32,
                planes, 256, &mut [parameter(&mut p)])?;
            (p.input, p.output) = (middle, packed);
            self.enqueue_dims(if next {kernels.next_down}else{kernels.down}, if next {80}else{320}, tasks as u32,
                planes, 128, &mut [parameter(&mut p)])
        })?;
        (p.input, p.output) = (packed, output.device_address());
        self.profile_phase(7, || self.enqueue_dims(kernels.scatter,
            route.tokens, H.div_ceil(256) as u32, 1, 256, &mut [parameter(&mut p)]))
    }

    /// Bit-exact W4A16 projection matching the model crate's project_ref
    /// oracle: BF16-pair input words, packed E2M1 weights and expanded E4M3
    /// scales as byte payloads, FP32 output [tokens, rows]. One ascending-K
    /// FMA sum per output at every token count (decode and prefill alike).
    /// # Safety
    /// Vectors stay live through stream completion and hold at least
    /// tokens*cols/2 input words, rows*cols/2 weight bytes, rows*cols/16
    /// scale bytes; output holds exactly tokens*rows FP32 values and aliases
    /// no input. Scales are codes 0..=126; all values finite.
    pub unsafe fn fp4_exact_project(&self, input: &DeviceVector<'_>, weights: &DeviceVector<'_>,
        scales: &DeviceVector<'_>, global_scale: f32, tokens: usize, rows: usize, cols: usize,
        output: &mut DeviceVector<'_>) -> Result<(), String> {
        let outputs = tokens.checked_mul(rows).filter(|&n| n > 0 && n.div_ceil(256) <= u32::MAX as usize
            && cols > 0 && cols % 16 == 0
            && tokens <= u32::MAX as usize && rows <= u32::MAX as usize && cols <= u32::MAX as usize
            && global_scale.is_finite() && global_scale > 0.0
            && input.len >= tokens * cols / 2 && weights.len * 4 >= rows * cols / 2
            && scales.len * 4 >= rows * cols / 16).ok_or("invalid exact NVFP4 projection")?;
        self.check_vector(output, outputs)?;
        if input.is_segmented() || weights.is_segmented() || scales.is_segmented() || output.is_segmented() {
            return Err("exact NVFP4 projection requires contiguous vectors".into());
        }
        let mut p = crate::kernel_params::Fp4ExactProject {
            input: input.device_address(), weights: weights.device_address(),
            scales: scales.device_address(), output: output.device_address(),
            global_scale, tokens: tokens as u32, rows: rows as u32, cols: cols as u32 };
        self.enqueue_dims(self.fp4_kernels()?.exact_project,
            outputs.div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut p)])
    }

    /// BF16(silu(gate) * up) packed two per word, matching swiglu_ref.
    /// # Safety
    /// gate/up hold at least 2*pairs finite FP32 values live through stream
    /// completion; output holds at least pairs words and aliases neither.
    pub unsafe fn fp4_exact_swiglu(&self, gate: &DeviceVector<'_>, up: &DeviceVector<'_>,
        pairs: usize, output: &mut DeviceVector<'_>) -> Result<(), String> {
        if pairs == 0 || pairs > u32::MAX as usize || gate.len < pairs * 2
            || up.len < pairs * 2 || output.len < pairs {
            return Err("invalid exact SwiGLU shape".into());
        }
        if gate.is_segmented() || up.is_segmented() || output.is_segmented() {
            return Err("exact SwiGLU requires contiguous vectors".into());
        }
        let mut p = crate::kernel_params::Fp4ExactSwiglu {
            gate: gate.device_address(), up: up.device_address(),
            output: output.device_address(), pairs: pairs as u32 };
        self.enqueue_dims(self.fp4_kernels()?.exact_swiglu,
            pairs.div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut p)])
    }

    /// fp4_exact_project with weight/scale storage addressed inside caller
    /// arenas (resident checkpoint weights and expanded scale arenas are not
    /// standalone vectors). Arithmetic and launch geometry are unchanged.
    /// # Safety
    /// As fp4_exact_project, with `weights`/`scales` naming live device
    /// storage of at least `weights_bytes`/`scales_bytes` bytes.
    pub unsafe fn fp4_exact_project_addr(&self, input: &DeviceVector<'_>,
        weights: u64, weights_bytes: usize, scales: u64, scales_bytes: usize,
        global_scale: f32, tokens: usize, rows: usize, cols: usize,
        output: &mut DeviceVector<'_>) -> Result<(), String> {
        let outputs = tokens.checked_mul(rows).filter(|&n| n > 0 && n.div_ceil(256) <= u32::MAX as usize
            && cols > 0 && cols % 16 == 0 && weights != 0 && scales != 0
            && tokens <= u32::MAX as usize && rows <= u32::MAX as usize && cols <= u32::MAX as usize
            && global_scale.is_finite() && global_scale > 0.0
            && input.len >= tokens * cols / 2 && weights_bytes >= rows * cols / 2
            && scales_bytes >= rows * cols / 16).ok_or("invalid exact NVFP4 projection")?;
        self.check_vector(output, outputs)?;
        if input.is_segmented() || output.is_segmented() { return Err("exact NVFP4 projection requires contiguous vectors".into()); }
        let mut p = crate::kernel_params::Fp4ExactProject { input: input.device_address(), weights, scales,
            output: output.device_address(), global_scale, tokens: tokens as u32, rows: rows as u32, cols: cols as u32 };
        self.enqueue_dims(self.fp4_kernels()?.exact_project,
            outputs.div_ceil(256) as u32, 1, 1, 256, &mut [parameter(&mut p)])
    }
}
