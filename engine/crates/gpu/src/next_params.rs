//! Native MXFP8 decode ABI. Row arithmetic is independent of launch geometry.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MxProjection {
    pub weights: u64,
    pub scales: u64,
    pub input: u64,
    pub output: u64,
    pub rows: u32,
    pub cols: u32,
    pub tokens: u32,
    /// Input is packed BF16 when 1, otherwise FP32 (rounded only for the GDN contract).
    pub input_bf16: u32,
    /// GDN projection boundaries round to BF16 represented in FP32.
    pub round: u32,
}
impl MxProjection {
    /// Decode a loaded FP32 operand. Only rounded GDN projections may discard
    /// its residual; ordinary projections consume the same FP32 input as PLAIN.
    #[inline(always)]
    pub fn f32_input(&self, bits: u32) -> f32 {
        let value = f32::from_bits(bits);
        if self.round != 0 {
            #[cfg(target_arch = "nvptx64")]
            {
                return crate::flash_gdn::rounded(value);
            }
            #[cfg(not(target_arch = "nvptx64"))]
            {
                return crate::bf16::rounded(value);
            }
        }
        value
    }
    pub fn kernel(&self) -> &'static str {
        if self.tokens == 1 {
            "flash_next_mxfp8_m1"
        } else if self.tokens <= 8 || self.round != 0 {
            "flash_next_mxfp8_pair"
        } else {
            "flash_next_mxfp8_bits"
        }
    }
    pub fn tile(&self) -> u32 {
        if self.round == 0 && self.tokens > 1 && self.tokens <= 8 {
            8 // port-mxfp8-gemm retained SIMT schedule, independent of N/K
        } else if self.tokens == 1 || self.rows < 128 {
            1
        } else if self.rows <= 640 || self.cols >= 4096 {
            if self.tokens <= 2 {
                2
            } else {
                4
            }
        } else {
            self.tokens
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MxLayer {
    pub projections: [MxProjection; 9],
    pub ends: [u32; 10],
    pub count: u32,
}
impl MxLayer {
    pub fn push(&mut self, p: MxProjection) -> Result<(), &'static str> {
        if self.count == 9
            || p.rows == 0
            || p.cols == 0
            || p.cols % 32 != 0
            || p.tokens == 0
            || p.tokens > 16
            || p.input_bf16 > 1
            || p.round > 1
            || [p.weights, p.scales, p.input, p.output].contains(&0)
            || p.weights % 16 != 0
            || p.input % 16 != 0
            || p.output % 4 != 0
        {
            return Err("invalid MXFP8 descriptor");
        }
        let blocks = p
            .rows
            .div_ceil(4)
            .checked_mul(p.tokens.div_ceil(p.tile()))
            .ok_or("MXFP8 grid overflow")?;
        let i = self.count as usize;
        self.ends[i + 1] = self.ends[i]
            .checked_add(blocks)
            .ok_or("MXFP8 grid overflow")?;
        self.projections[i] = p;
        self.count += 1;
        Ok(())
    }
}
