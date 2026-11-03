//! Logical FP32 mHC ABI. All counts come from operation data, not model names.
//! Buffers are row-major. Distinct fields must own disjoint device ranges.
//! Each caller must call `Shape::validate` before allocation or launch.
//! Kernel entries do not validate shapes or buffer extents.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub rows: u32, pub streams: u32, pub hidden: u32, pub iterations: u32,
    pub rms_eps: f32, pub eps: f32,
}
impl Shape {
    /// Bound work and address products before any allocation or launch.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.streams == 0 || self.hidden == 0 || self.iterations == 0
            || self.iterations > 1_048_576 { return Err("invalid mHC dimensions or iteration count"); }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 || !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("invalid mHC epsilon");
        }
        let n = u64::from(self.streams); let h = u64::from(self.hidden); let t = u64::from(self.rows);
        let w = n.checked_mul(h).ok_or("mHC width overflow")?;
        let p = n.checked_mul(n + 2).ok_or("mHC coefficient count overflow")?;
        for count in [w.checked_mul(p), t.checked_mul(w), t.checked_mul(p)] {
            if count.is_none_or(|v| v > u32::MAX as u64) { return Err("mHC buffer exceeds ABI limit"); }
        }
        Ok(())
    }
    pub fn wide(self) -> usize { self.streams as usize * self.hidden as usize }
    pub fn coefficients(self) -> usize { self.streams as usize * (self.streams as usize + 2) }
}
/// Input `[T,N,D]`, weights `[P,N*D]`, base `[P]`, scale `[3]`.
/// Scratch normalized `[T,N*D]`; output `[T,P]` = pre, post, `comb[src,dst]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Prepare {
    pub input: u64, pub weights: u64, pub base: u64, pub scale: u64,
    pub normalized: u64, pub coefficients: u64, pub shape: Shape,
}
/// Input `[T,N,D]`, coefficients `[T,P]`, output `[T,D]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Collapse {
    pub input: u64, pub coefficients: u64, pub output: u64,
    pub rows: u32, pub streams: u32, pub hidden: u32,
}
/// Raw input `[T,N,D]`, coefficients `[T,P]`, block `[T,D]`, output `[T,N,D]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Combine {
    pub input: u64, pub coefficients: u64, pub block: u64, pub output: u64,
    pub rows: u32, pub streams: u32, pub hidden: u32,
}
/// Raw input `[T,N,D]`, output `[T,D]`. Final RMS norm is a separate operation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Head { pub input: u64, pub output: u64, pub rows: u32, pub streams: u32, pub hidden: u32 }
