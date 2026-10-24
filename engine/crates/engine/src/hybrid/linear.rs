//! Fixed arithmetic bindings for the typed hybrid projection contract.
use crate::arms::Limits as arm_limits;
pub const ARITHMETIC_ID: &str = "token-step/fp8-k128/bf16-canonical/norm2-128/v1";
#[cfg(target_os = "linux")]
use engine_model::plan::runtime::{Projection, ProjectionStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputBoundary { Fp32, Bf16 }

/// Boundaries for kept BF16 a/b, router/shared gate and vocabulary projections.
pub fn bf16_boundary(role: engine_model::shapes::Role) -> Result<OutputBoundary, String> {
    use engine_model::shapes::{Part, Place, Store};
    match (role.place, role.part, role.store) {
        (Place::Draft { layer: None }, Part::Whole, Store::Weight) => Ok(OutputBoundary::Bf16),
        (Place::Layer(_) | Place::Draft { layer: Some(_) }, Part::LinearHeads, Store::Weight) => Ok(OutputBoundary::Bf16),
        (Place::Layer(_) | Place::Draft { layer: Some(_) }, Part::Whole, Store::Weight)
        | (Place::Model | Place::Draft { layer: None }, Part::Head, Store::Weight) => Ok(OutputBoundary::Fp32),
        _ => Err("BF16 boundary requires a/b, router/shared gate or head role".into()),
    }
}

/// One row CTA and one reduction graph, including residual and final/head norms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormBinding { pub width: usize, pub epsilon: f32 }
impl NormBinding {
    pub const ORDER: u32 = 2;
    pub const THREADS: u32 = 128;
    pub fn new(width: usize, epsilon: f32) -> Result<Self, String> {
        arm_limits::norm(width as u64)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("fixed norm dimensions or epsilon".into());
        }
        Ok(Self { width, epsilon })
    }
    pub fn elements(self, rows: usize) -> Result<usize, String> {
        if rows == 0 || rows > u32::MAX as usize { return Err("fixed norm row count".into()); }
        self.width.checked_mul(rows).ok_or("fixed norm extent overflow".into())
    }
}

/// Loader identity fixes typed norms; untyped reductions retain their legacy order/grid.
pub fn norm_launch(typed: bool, residual: bool, rows: usize, order: Option<u32>, four: bool) -> (u32, u32) {
    let threads = if !typed && residual && rows > 1 { 256 } else { NormBinding::THREADS };
    (threads, if typed { NormBinding::ORDER } else { order.unwrap_or(if four { 1 } else if rows == 1 { 2 } else { 0 }) })
}

/// The one-row shared gate is row-major; other weights require four-row packing.
pub fn bf16_rows_supported(rows: usize) -> bool {
    rows == 1 || rows != 0 && rows % 4 == 0
}

/// Pair only shapes that retain each projection's separate decode launch.
pub fn router_gate_pair_rows(n1: usize, n2: usize, k: usize, k2: usize, rows: usize) -> bool {
    k == k2 && (1..=8).contains(&rows) && n1.checked_add(n2).is_some_and(|n| n < 1024) && k > 512
}

/// Owns the once-prepared matrix; no raw view escapes to a legacy layout consumer.
#[cfg(target_os = "linux")]
pub struct CanonicalBf16<'a, 'b> {
    backend: &'b spark_gpu_probe::driver::Backend<'a>, matrix: spark_gpu_probe::driver::Matrix<'a>,
}
#[cfg(target_os = "linux")]
impl<'a, 'b> CanonicalBf16<'a, 'b> {
    pub fn dimensions(&self) -> [u64; 2] { [self.matrix.rows() as u64, self.matrix.cols() as u64] }
    /// GDN a/b share input and retain their separate BF16 output boundaries.
    pub(super) fn project_pair(&self, other: &Self, input: &spark_gpu_probe::driver::DeviceVector<'_>,
        first: &mut spark_gpu_probe::driver::DeviceVector<'_>, second: &mut spark_gpu_probe::driver::DeviceVector<'_>, rows: usize) -> Result<(),String> {
        if self.dimensions()!=other.dimensions() || !std::ptr::eq(self.backend,other.backend) { return Err("paired BF16 receipts differ".into()); }
        for start in (0..rows).step_by(1024) { let range=start..rows.min(start+1024); let [n,k]=self.dimensions().map(|n|n as usize);
            unsafe { self.backend.fixed_bf16_pair([self.matrix.device_address(),other.matrix.device_address()],[n,k],
                &input.slice(range.start*k..range.end*k)?,&mut first.slice(range.start*n..range.end*n)?,&mut second.slice(range.start*n..range.end*n)?,range.len())?; } }
        Ok(())
    }
    /// Fp32-boundary pair with unequal rows (router + shared gate); Ok(false)
    /// keeps the caller's separate projections wherever the paired launch
    /// would not be the very launch each matrix takes alone.
    pub(super) fn project_pair_rows(&self, other: &Self, input: &spark_gpu_probe::driver::DeviceVector<'_>,
        first: &mut spark_gpu_probe::driver::DeviceVector<'_>, second: &mut spark_gpu_probe::driver::DeviceVector<'_>,
        scratch: &mut spark_gpu_probe::driver::DeviceVector<'_>, rows: usize) -> Result<bool,String> {
        let ([n1,k],[n2,k2])=(self.dimensions(),other.dimensions());
        let (n1,n2,k)=(n1 as usize,n2 as usize,k as usize);
        if !std::ptr::eq(self.backend,other.backend) || !router_gate_pair_rows(n1,n2,k,k2 as usize,rows) { return Ok(false); }
        unsafe { self.backend.fixed_bf16_pair_rows([self.matrix.device_address(),other.matrix.device_address()],[n1,n2,k],
            &input.slice(0..rows*k)?,&mut first.slice(0..rows*n1)?,&mut second.slice(0..rows*n2)?,scratch,rows)?; }
        Ok(true)
    }
    /// Prepare with the existing in-place pack kernels and at most 64*K*2 scratch bytes.
    /// # Safety
    /// Matrix is genuine BF16 contiguous row-major storage in this backend's context,
    /// never previously packed; no aliases/readers may use its bytes during preparation.
    pub unsafe fn prepare(backend: &'b spark_gpu_probe::driver::Backend<'a>, matrix: spark_gpu_probe::driver::Matrix<'a>) -> Result<Self, String> {
        use spark_gpu_probe::kernel_params::{FlashBf16Pack, HybridCopy};
        let (n, k) = (matrix.rows(), matrix.cols());
        if backend.matrix_is_packed(&matrix) || matrix.is_fp8() || matrix.scalar_f32()
            || !bf16_rows_supported(n) || n > u32::MAX as usize || k == 0 || k % 16 != 0 || k > u32::MAX as usize / 128 {
            return Err("BF16 preparation requires unprepared row-major weights".into());
        }
        let bytes = n.checked_mul(k).and_then(|v| v.checked_mul(2)).ok_or("BF16 preparation extent overflow")?;
        matrix.device_address().checked_add(bytes as u64).ok_or("BF16 preparation address overflow")?;
        let mut p = FlashBf16Pack { source: 0, weight: matrix.device_address(), rows: n as u32, cols: k as u32 };
        if n % 64 == 0 {
            let stage = backend.vector(32 * k)?;
            for row in (0..n).step_by(64) {
                p.source = stage.device_address(); p.weight = matrix.device_address() + (row * k * 2) as u64; p.rows = 64;
                let mut copy = HybridCopy { source: p.weight as _, destination: p.source as _, words: (32*k) as u32, padding: 0, rows: 0 };
                unsafe {
                    backend.launch_flash_rows("flash_hybrid_copy", (32*k).div_ceil(128) as u32, 1, 128, 0, &mut [(&mut copy as *mut HybridCopy).cast()])?;
                    backend.launch_flash_shared("flash_bf16_pack", (k/8) as u32, 128, 0, &mut [(&mut p as *mut FlashBf16Pack).cast()])?;
                }
            }
            backend.synchronize()?;
        } else if n % 4 == 0 {
            unsafe { backend.launch_flash_shared("flash_bf16_pack", (n/4) as u32, 128, (8*k) as u32, &mut [(&mut p as *mut FlashBf16Pack).cast()])?; }
            backend.synchronize()?;
        }
        Ok(Self { backend, matrix })
    }
    /// a/b use Bf16; router, shared gate and head use Fp32 in every phase.
    pub fn project(&self, projection: &Projection<'_>, input: &spark_gpu_probe::driver::DeviceVector<'_>,
        output: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut spark_gpu_probe::driver::DeviceVector<'_>,
        rows: usize) -> Result<(), String> {
        let boundary = bf16_boundary(projection.weight.role)?;
        unsafe { project(self.backend, projection, &self.matrix, input, output, scratch, rows, boundary) }
    }
}

/// Bind by storage and declared boundary, never checkpoint name, phase or row count.
/// # Safety
/// BF16 weights carry a CanonicalBf16 receipt (or the same proven layout/context);
/// neither a raw row-major matrix nor a generic packed8 narrow matrix is sufficient.
#[cfg(target_os = "linux")]
pub unsafe fn project(b: &spark_gpu_probe::driver::Backend<'_>, projection: &Projection<'_>,
    weights: &spark_gpu_probe::driver::Matrix<'_>, input: &spark_gpu_probe::driver::DeviceVector<'_>,
    output: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut spark_gpu_probe::driver::DeviceVector<'_>,
    tokens: usize, boundary: OutputBoundary) -> Result<(), String> {
    unsafe { project_reusing(b, projection, weights, input, output, scratch, tokens, boundary, false) }
}
/// # Safety
/// The project contract applies; reuse additionally requires unchanged input,
/// row/K extents and quantization scratch since the preceding FP8 projection.
/// Any BF16 projection using that scratch invalidates FP8 reuse, including mma64.
#[cfg(target_os = "linux")]
pub unsafe fn project_reusing(b: &spark_gpu_probe::driver::Backend<'_>, projection: &Projection<'_>,
    weights: &spark_gpu_probe::driver::Matrix<'_>, input: &spark_gpu_probe::driver::DeviceVector<'_>,
    output: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut spark_gpu_probe::driver::DeviceVector<'_>,
    tokens: usize, boundary: OutputBoundary, reuse: bool) -> Result<(), String> {
    let fp8 = matches!(projection.storage, ProjectionStorage::Fp8 { .. });
    if projection.rows != weights.rows() as u64 || projection.cols != weights.cols() as u64
        || fp8 != weights.is_fp8() || weights.scalar_f32() {
        return Err("linear runtime weight disagrees with typed projection".into());
    }
    if fp8 { unsafe { b.fixed_fp8_projection_reusing_input(weights, input, output, scratch, tokens, boundary == OutputBoundary::Bf16, reuse) } }
    else { unsafe { b.fixed_bf16_projection(weights.device_address(), [weights.rows(), weights.cols()],
        input, output, scratch, tokens, boundary == OutputBoundary::Bf16,
        matches!(projection.weight.role, engine_model::shapes::Role { place: engine_model::shapes::Place::Layer(_)
            | engine_model::shapes::Place::Draft { layer: Some(_) }, part: engine_model::shapes::Part::Whole, .. })) } }
}
