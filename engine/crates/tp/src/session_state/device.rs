//! Safe copy bindings to the existing ordered CUDA backend. No new FFI boundary.
use super::{Store, Result, Pool, Layout, Budget, need, add};
use spark_gpu_probe::driver::{Backend, DeviceVector};

pub struct DeviceStore<'b,'d> { backend: &'b Backend<'d> }
impl<'b,'d> DeviceStore<'b,'d> {
    pub fn new(backend: &'b Backend<'d>) -> Self { Self {backend} }
    pub fn backend(&self) -> &'b Backend<'d> { self.backend }
    /// Check current resident charges as well as the prospective owner budget.
    /// Other users must not spend this pool's reserved capacity after admission.
    pub fn admit(self,layout: Layout,budget: Budget) -> Result<Pool<Self>> {
        need(self.backend.allocated_bytes()<=add(budget.weights,budget.runtime)?,"rank budget omits resident device allocations")?;
        need(budget.limit<=self.backend.allocation_budget(),"rank limit exceeds backend allocation budget")?;
        Pool::new(self,layout,budget)
    }
}
impl<'d> Store for DeviceStore<'_,'d> {
    type Buffer=DeviceVector<'d>;
    fn allocate(&self,bytes: usize) -> Result<Self::Buffer> {
        need(bytes>0 && bytes.is_multiple_of(64),"device session extent is not aligned")?; self.backend.vector(bytes/4)
    }
    fn size(&self,buffer: &Self::Buffer) -> usize { buffer.len()*4 }
    fn zero(&self,buffer: &mut Self::Buffer) -> Result<()> { self.backend.zero_vector(buffer) }
    fn copy(&self,source: &Self::Buffer,destination: &mut Self::Buffer) -> Result<()> {
        need(source.len()==destination.len(),"device session copy extent differs")?;
        self.backend.copy_prefix(source,destination,source.len())
    }
    fn write(&self,buffer: &mut Self::Buffer,bytes: &[u8]) -> Result<()> { self.backend.write_vector_bytes(buffer,bytes) }
    fn read(&self,buffer: &Self::Buffer) -> Result<Vec<u8>> {
        // No floating-point operation occurs. Preserve all bits, including NaNs.
        Ok(self.backend.read_vector_values(buffer)?.into_iter().flat_map(f32::to_le_bytes).collect())
    }
    fn wait(&self) -> Result<()> { self.backend.synchronize() }
}
