//! Shared host/NVPTX ABI for independent BF16 decode projections.
#[repr(C)]
pub struct DecodeProjection {
    pub weights: u64,
    pub input: u64,
    pub output: u64,
    pub rows: u32,
    pub width: u32,
    pub tokens: u32,
}

/// Canonical fused gate/up projection into BF16 fragments.
#[repr(C)]
pub struct DecodeGateUp {
    pub gate: u64,
    pub up: u64,
    pub input: u64,
    pub output: u64,
    pub rows: u32,
    pub width: u32,
    pub tokens: u32,
}

/// Packed decode operands with the checkpoint's physical weight stride.
#[repr(C)]
pub struct PackedDecodeProjection {
    pub weights: u64,
    pub input: u64,
    pub output: u64,
    pub rows: u32,
    pub width: u32,
    pub tokens: u32,
    pub stride: u32,
}

/// Dense GDN prefill inputs and gate epilogue, shared with the kernel.
#[repr(C)]
pub struct GdnProjection {
    pub weights: [u64; 4],
    pub input: u64,
    pub outputs: [u64; 5],
    pub alog: u64,
    pub dt: u64,
    pub qkv_rows: u32,
    pub z_rows: u32,
    pub heads: u32,
    pub width: u32,
    pub tokens: u32,
    pub parameter_precision: u32,
}

/// Host geometry for the fixed dense entries; phases and cohort sizes never select math.
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedLinear {
    pub outputs: u32,
    pub cols: u32,
    pub fp8: bool,
}
#[cfg(not(target_arch = "nvptx64"))]
impl FixedLinear {
    pub fn new(outputs: usize, cols: usize, fp8: bool) -> Result<Self, String> {
        let (outputs, cols) = (u32::try_from(outputs).map_err(|_| "linear N exceeds ABI")?,
            u32::try_from(cols).map_err(|_| "linear K exceeds ABI")?);
        if outputs == 0 || cols == 0 || cols % if fp8 { 128 } else { 16 } != 0
            || fp8 && outputs % 128 != 0 { return Err("fixed linear shape".into()); }
        (outputs as usize).checked_mul(cols as usize).and_then(|n| n.checked_mul(if fp8 { 1 } else { 2 }))
            .ok_or("fixed linear weight byte overflow")?;
        Ok(Self { outputs, cols, fp8 })
    }
    /// Logical FP32 extents; padding lives only in the kernel's masked tile.
    pub fn elements(self, tokens: usize) -> Result<(usize, usize), String> {
        if !(1..=1024).contains(&tokens) { return Err("fixed linear requires 1..=1024 rows".into()); }
        let extent = |width| tokens.checked_mul(width as usize).filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| "fixed linear extent exceeds ABI".to_string());
        Ok((extent(self.cols)?, extent(self.outputs)?))
    }
    /// Row-major E4M3 codes, aligned FP32 scales, total scratch in FP32 words.
    pub fn scratch(self, tokens: usize) -> Result<(usize, usize), String> {
        let (codes, _) = self.elements(tokens)?;
        if !self.fp8 { return Err("BF16 has no quantization scratch".into()); }
        let scales = codes.checked_add(255).ok_or("scratch alignment overflow")? / 256 * 256;
        let bytes = scales.checked_add(codes / 128 * 4).ok_or("scratch size overflow")?;
        Ok((scales, bytes.div_ceil(4)))
    }
    /// M/N tile, threads, dynamic shared bytes: pinned only by projection geometry.
    pub fn tile(self) -> [u32; 4] {
        if self.fp8 { [128, 128, 512, 66560] }
        else if self.outputs < 1024 && self.cols > 512 || self.cols == 320 && self.outputs >= 1024 { [8, 8, 128, 0] }
        else { [8, 16, 64, 0] }
    }
    /// Router/shared gate prompt CTAs stage each eight-output weight tile once.
    /// Four canonical K ranges and the M8 hi/lo operands remain unchanged.
    pub fn tile_reusing_weights(self, tokens: usize, reuse: bool) -> [u32; 4] {
        if reuse && !self.fp8 && tokens > 8 && self.outputs <= 512 && self.cols <= 4096 && self.cols % 64 == 0 {
            [256, 8, 256, 16 * self.cols]
        } else { self.tile() }
    }
    /// Validate address arithmetic and non-overlapping live device spans before enqueue.
    pub fn disjoint(a: u64, a_bytes: usize, b: u64, b_bytes: usize) -> Result<(), String> {
        let ae = a.checked_add(a_bytes as u64).ok_or("linear address overflow")?;
        let be = b.checked_add(b_bytes as u64).ok_or("linear address overflow")?;
        if a == 0 || b == 0 || a < be && b < ae { return Err("linear operands overlap or are null".into()); }
        Ok(())
    }
    /// Exact declared reduction schedule of the existing kernels, in original K coordinates.
    /// FP8: four ascending K32 MMAs per K128 block, then scaled ascending block sum.
    /// BF16: ascending K16 in four ranges, hi+lo per range, (P0+P1)+(P2+P3).
    #[allow(dead_code)] // Portable schedule receipt; CUDA consumes the unchanged kernel body.
    pub fn k_ranges(self) -> Vec<std::ops::Range<u32>> {
        if self.fp8 { (0..self.cols / 128).map(|b| b * 128..(b + 1) * 128).collect() }
        else { (0..4).map(|p| self.cols / 16 * p / 4 * 16..self.cols / 16 * (p + 1) / 4 * 16).collect() }
    }
}
